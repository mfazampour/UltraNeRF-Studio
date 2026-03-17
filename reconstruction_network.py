import inspect
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
from load_us import load_us_data, load_rec_data
from nerf_utils import create_nerf, img2mse, render_us, compute_loss, compute_regularization, create_nets_for_reconstruction
from unerf_config import config_parser

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.95)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def norm_array(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
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
        labels, poses_labels = load_rec_data(
            args.datadir)

    # Cast intrinsics to right types
    # The poses are not normalized. We scale down the space.
    # It is possible to normalize poses and remove scaling.
    scaling = 0.001
    near = 0
    probe_depth = args.probe_depth * scaling
    probe_width = args.probe_width * scaling
    far = probe_depth
    H, W = labels.shape[1], labels.shape[2]
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
    render_kwargs_train, render_kwargs_test, start, optimizer, optimizer_rec = create_nets_for_reconstruction(
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
    # Losses
    ssim_weight = args.ssim_lambda
    l2_weight = 1.0 - ssim_weight
    ssim_kwargs = {
        "spatial_dims": 2,
        "win_size": args.ssim_filter_size,
        "k1": 0.01,
        "k2": 0.1,
    }
    ssim_signature = inspect.signature(SSIMLoss)
    if "data_range" in ssim_signature.parameters:
        ssim_kwargs["data_range"] = 1.0
    if "kernel_type" in ssim_signature.parameters:
        ssim_kwargs["kernel_type"] = "gaussian"
    ssim_loss = SSIMLoss(**ssim_kwargs)
    losses = {"l2": img2mse,
              "ssim": ssim_loss,
              "lncc": LocalNormalizedCrossCorrelationLoss(spatial_dims=2),
              'bce': BCELoss()}
    start = start + 1
    r_list = list()
    a_list = list()
    s_list = list()
    c_list = list()
    #get theta
    print("Eval NeRF")
    poses_labels = poses_labels[::2]
    labels = labels[::2]
    for i in poses_labels:
        render_kwargs_train['network_fn'].eval()
        pts = torch.from_numpy(i).to(device)
        render_kwargs_train["pts"] = pts
        #####  Core optimization loop  #####
        rendering_output = render_us(
            H, W, sw, sh, c2w=None, chunk=args.chunk, retraw=True, **render_kwargs_train
        )
        r = rendering_output["reflection_coeff"].detach().clone().permute(0, 3, 2, 1).cpu().numpy().squeeze(0)
        a = rendering_output["attenuation_coeff"].detach().clone().permute(0, 3, 2, 1).cpu().numpy().squeeze(0)
        s = rendering_output["scatter_amplitude"].detach().clone().permute(0, 3, 2, 1).cpu().numpy().squeeze(0)
        c = rendering_output["confidence_maps"].detach().clone().permute(0, 3, 2, 1).cpu().numpy().squeeze(0)
        r_list.append(r)
        a_list.append(a)
        s_list.append(s)
        c_list.append(c)

    r_n = norm_array(np.array(r_list))
    a_n = norm_array(np.array(a_list))
    s_n = norm_array(np.array(s_list))
    c_n = np.array(c_list)
    for i in trange(start, N_iters + 1):
        render_kwargs_train['network_rec'].train()
        render_kwargs_train['network_fn'].eval()
        img_i = np.random.choice(
            list(range(poses_labels.shape[0]))[::args.rec_step]
        )
        k = 4
        # Why? This does not guarantee that all images are used --> probably a weighted random would be better,
        # or removing from a temporary set as long as it's not empty
        target_rec = torch.Tensor(labels[img_i:img_i+4]).to(device).unsqueeze(0).permute(1, 2, 3, 0)
        # pts = torch.from_numpy(poses_labels[img_i:img_i+8]).to(device)
        r, a, s, c = torch.from_numpy(r_n[img_i:img_i+4]).to(device), torch.from_numpy(a_n[img_i:img_i+4]).to(device), \
        torch.from_numpy(s_n[img_i:img_i+4]).to(device), torch.from_numpy(c_n[img_i:img_i+4]).to(device)

        theta = torch.concatenate([r, a, s], dim=-1)
        input_reconstruction = theta.squeeze()
        ret_reconstruction = render_kwargs_train["network_query_fn_rec"](input_reconstruction,
                                                                         render_kwargs_train["network_rec"])
        # rendering_output["confidence_maps"] *
        if args.confidence:
            output = c * ret_reconstruction
        else:
            output = ret_reconstruction
        output = output.permute(0, 2, 1, 3)
        optimizer_rec.zero_grad()
        loss = dict()
        loss['bce'] = losses["bce"](output, target_rec)
        total_loss = loss["bce"]
        total_loss.backward()
        optimizer_rec.step()
        # print(output[0].shape)
        # print(rendering_output["reflection_coeff"].shape)
        rendering_output["rec"] = output[0].permute(2, 0, 1)[None,...]
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

            plt.savefig(
                os.path.join(rendering_path, "{:08d}.png".format(i + 1)),
                bbox_inches="tight",
                dpi=200,
            )

            if i > args.rec_iter:
                plt.subplot(3, 5, 14)
                plt.title("Target rec")
                plt.imshow(target_rec[0].permute(2, 0, 1)[None,...].detach().cpu().numpy()[0, 0].T)

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
    if hasattr(torch, "set_default_device") and torch.cuda.is_available():
        torch.set_default_device("cuda")
    train()
