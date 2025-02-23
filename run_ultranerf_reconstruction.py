import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai.losses.ssim_loss import SSIMLoss
from monai.losses import LocalNormalizedCrossCorrelationLoss

from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from torch.nn import BCELoss
from load_us import load_us_data
from nerf_utils import create_nerf, img2mse, render_us, compute_loss, compute_regularization
from unerf_config import config_parser

torch.cuda.set_per_process_memory_fraction(0.95)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():

    parser = config_parser()
    args = parser.parse_args()

    if args.random_seed == 0:
        print("Setting deterministic behaviour")
        random_seed = 42
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    if args.dataset_type == "us":
        # IT CONVERTS THE POSE TRANSLATION FROM MM TO M ALREADY!!!
        images, poses, labels, poses_labels, i_test = load_us_data(
            args.datadir, confmap=args.confmap, pose_path=args.pose_path
        , reconstruction=True)

        if not isinstance(i_test, list):
            i_test = [i_test]

        i_val = i_test
        i_train = np.array(
            [
                i
                for i in np.arange(int(images.shape[0]))
                if (i not in i_test and i not in i_val)
            ]
        )

        print("Test {}, train {}".format(len(i_test), len(i_train)))

    else:
        print("Unknown dataset type", args.dataset_type, "exiting")
        return

    # Cast intrinsics to right types
    # The poses are not normalized. We scale down the space.
    # It is possible to normalize poses and remove scaling.
    scaling = 0.001
    near = 0
    probe_depth = args.probe_depth * scaling
    probe_width = args.probe_width * scaling
    far = probe_depth
    H, W = images.shape[1], images.shape[2]
    sy = probe_depth / float(H)
    sx = probe_width / float(W)
    sh = sy
    sw = sx

    basedir = args.basedir
    expname = args.expname

    # Create tensorboard writer
    if args.tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(basedir, "summaries", expname))

    # Create log dir and copy the config file
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, "config.txt")
        with open(f, "w") as file:
            file.write(open(args.config, "r").read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, optimizer, optimizer_rec = create_nerf(
        args, device=device
    )

    bds_dict = {
        "near": near,
        "far": far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    N_iters = args.n_iters
    print("Begin")
    print("TRAIN views are", i_train)
    print("TEST views are", i_test)
    print("VAL views are", i_val)

    # Losses
    ssim_weight = args.ssim_lambda
    l2_weight = 1.0 - ssim_weight
    ssim_loss = SSIMLoss(
        spatial_dims=2,
        data_range=1.0,
        kernel_type="gaussian",
        win_size=args.ssim_filter_size,
        k1=0.01,
        k2=0.1,
    )
    losses = {"l2": img2mse,
              "ssim": ssim_loss,
              "lncc": LocalNormalizedCrossCorrelationLoss(spatial_dims=2),
              'bce': BCELoss()}
    start = start + 1
    for i in trange(start, N_iters + 1):
        if args.rec_only_occ:
            render_kwargs_train['network_rec'].train()
            render_kwargs_train['network_fn'].eval()
            img_i = np.random.choice(
                i_train[0::args.rec_step]
            )  # Why? This does not guarantee that all images are used --> probably a weighted random would be better,
            # or removing from a temporary set as long as it's not empty

            target_rec = torch.Tensor(labels[img_i]).to(device).unsqueeze(0).unsqueeze(0)
            target = target_rec
            pts = torch.from_numpy(poses_labels[img_i]).to(device)
            render_kwargs_train["pts"] = pts
            #####  Core optimization loop  #####

            input_reconstruction = pts.squeeze()
            ret_reconstruction = render_kwargs_train["network_query_fn_rec"](input_reconstruction,
                                                                             render_kwargs_train["network_rec"])

            output = ret_reconstruction.permute(2, 1, 0)[None, ...]
            optimizer_rec.zero_grad()
            loss = dict()
            loss['bce'] = losses["bce"](output, target_rec)
            total_loss = loss["bce"]
            total_loss.backward()
            optimizer_rec.step()
            rendering_output = dict()
            rendering_output["rec"] = output
            time0 = time.time()
        else:
            if i < args.rec_iter:
                render_kwargs_train['network_rec'].eval()
                time0 = time.time()

                img_i = np.random.choice(
                    i_train
                )  # Why? This does not guarantee that all images are used --> probably a weighted random would be better,
                # or removing from a temporary set as long as it's not empty

                target = torch.Tensor(images[img_i]).to(device).unsqueeze(0).unsqueeze(0)
                # pose = torch.from_numpy(poses[img_i, :3, :4]).to(device).unsqueeze(0)
                pts = torch.from_numpy(poses_labels[img_i]).to(device)
                render_kwargs_train["pts"] = pts
                #####  Core optimization loop  #####
                rendering_output = render_us(
                    H, W, sw, sh, c2w=None, chunk=args.chunk, retraw=True, **render_kwargs_train
                )
                output_image = rendering_output["intensity_map"]

                optimizer.zero_grad()

                loss = compute_loss(output_image, target, args, losses, i)
                if args.reg and i > args.r_warm_up_it:
                    reg = compute_regularization(rendering_output, losses,
                                              weights=(args.r_lcc_penalty, args.r_tv_penalty, args.r_max_reflection))
                    loss = {**loss, **reg}

                total_loss = 0.0
                for loss_value in loss.values():
                    tmp = loss_value[0] * loss_value[1]
                    total_loss += tmp

                if type(total_loss) != torch.Tensor:
                    raise ValueError("Loss is not a tensor: Problem with loss calculation")

                total_loss.backward()
                optimizer.step()

            else:
                render_kwargs_train['network_rec'].train()
                render_kwargs_train['network_fn'].eval()
                img_i = np.random.choice(
                    i_train[::args.rec_step]
                )  # Why? This does not guarantee that all images are used --> probably a weighted random would be better,
                # or removing from a temporary set as long as it's not empty

                target_rec = torch.Tensor(labels[img_i]).to(device).unsqueeze(0).unsqueeze(0)
                pose = torch.from_numpy(poses[img_i, :3, :4]).to(device).unsqueeze(0)
                target = torch.Tensor(images[img_i]).to(device).unsqueeze(0).unsqueeze(0)
                pts = torch.from_numpy(poses_labels[img_i]).to(device)
                render_kwargs_train["pts"] = pts
                #####  Core optimization loop  #####
                rendering_output = render_us(
                    H, W, sw, sh, c2w=None, chunk=args.chunk, retraw=True, **render_kwargs_train
                )
                r = rendering_output["reflection_coeff"].detach().clone().permute(0, 3, 2, 1)
                a = rendering_output["attenuation_coeff"].detach().clone().permute(0, 3, 2, 1)
                s = rendering_output["scatter_amplitude"].detach().clone().permute(0, 3, 2, 1)

                theta = torch.concatenate([r, a, s], dim=-1)
                input_reconstruction = theta.squeeze()\
                    if args.rec_only_theta else torch.cat([pts.squeeze(), theta.squeeze()], dim=-1)
                ret_reconstruction = render_kwargs_train["network_query_fn_rec"](input_reconstruction,
                                                                                 render_kwargs_train["network_rec"])

                # rendering_output["confidence_maps"] *
                if args.confidence:
                    output = rendering_output["confidence_maps"] * ret_reconstruction.permute(2, 1, 0)[None, ...]
                else:
                    output = ret_reconstruction.permute(2, 1, 0)[None, ...]
                optimizer_rec.zero_grad()
                loss = dict()
                loss['bce'] = losses["bce"](output, target_rec)
                total_loss = loss["bce"]
                total_loss.backward()
                optimizer_rec.step()
                rendering_output["rec"] = output
                time0 = time.time()
            dt = time.time() - time0
            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (i / decay_steps))
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lrate
            ################################

        if args.tensorboard:
            writer.add_scalar("Loss/total_loss", total_loss.item(), i)
            for k, v in loss.items():
                writer.add_scalar(f"Loss/{k}", v[1].item(), i)
            writer.add_scalar("Learning rate", new_lrate, i)


        if (i + 1) % args.i_print == 0:

            rendering_path = os.path.join(basedir, expname, "train_rendering")
            os.makedirs(
                os.path.join(basedir, expname, "train_rendering"), exist_ok=True
            )

            # print(f"Step: {i+1}, Loss: {total_loss.item()}, Time: {dt}")  # type: ignore
            # detailed_loss_string = ", ".join(
            #     [f"{k}: {v[1].item()}" for k, v in loss.items()]
            # )
            # print(detailed_loss_string)

            plt.figure(figsize=(16, 8))
            for j, m in enumerate(rendering_output):

                plt.subplot(3, 5, j + 1)
                plt.title(m)
                plt.imshow(rendering_output[m].detach().cpu().numpy()[0, 0].T)

            plt.subplot(3, 5, 15)
            plt.title("Target")
            plt.imshow(target.detach().cpu().numpy()[0, 0].T)

            plt.savefig(
                os.path.join(rendering_path, "{:08d}.png".format(i + 1)),
                bbox_inches="tight",
                dpi=200,
            )

            if i > args.rec_iter:
                plt.subplot(3, 5, 14)
                plt.title("Target rec")
                plt.imshow(target_rec.detach().cpu().numpy()[0, 0].T)

                plt.savefig(
                    os.path.join(rendering_path, "{:08d}.png".format(i + 1)),
                    bbox_inches="tight",
                    dpi=200,
                )
                plt.close()
            plt.close()
        if (i + 1) % args.i_weights == 0:
            path = os.path.join(basedir, expname, "{:06d}.tar".format(i + 1))
            torch.save(
                {
                    "global_step": i,
                    "network_fn_state_dict": render_kwargs_train[
                        "network_fn"
                    ].state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    'network_rec_state_dict': render_kwargs_train['network_rec'].state_dict(),
                    "optimizer_rec_state_dict": optimizer_rec.state_dict(),
                },
                path,
            )
            # print('Saved checkpoints at', path)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cuda")
    train()
