"""Train the BARF-style PyTorch pose-refinement variant of UltraNeRF.

This entry point extends the baseline ultrasound renderer with BARF-style
coarse-to-fine positional encoding and learned per-frame pose refinement. It is
used when the input probe poses are noisy or when joint scene-and-pose
optimization is desired.

Outputs:
- checkpoints under ``logs/<expname>/``
- optional TensorBoard summaries
- periodic rendered visualizations during training
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if ROOT.name == "scripts":
    SRC = ROOT.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


import inspect
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai.losses.ssim_loss import SSIMLoss

from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from ultranerf.load_us import load_us_data
from ultranerf.nerf_utils import create_barf, img2mse, render_us

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.95)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def config_parser():

    import configargparse

    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--expname", type=str, help="experiment name")
    parser.add_argument(
        "--basedir", type=str, default="./logs/", help="where to store ckpts and logs"
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default="./data/synthetic_testing/l2",
        help="input data directory",
    )

    # training options
    parser.add_argument("--n_iters", type=int, default=100000)
    parser.add_argument("--ssim_filter_size", type=int, default=7)
    parser.add_argument("--ssim_lambda", type=float, default=0.75)
    parser.add_argument("--loss", type=str, default="l2")
    parser.add_argument("--probe_depth", type=int, default=140)
    parser.add_argument("--probe_width", type=int, default=80)
    parser.add_argument("--output_ch", type=int, default=5)

    parser.add_argument("--L", type=int, default=0)
    parser.add_argument("--c2f", type=tuple, default=None)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--confmap", type=bool, default=False)
    parser.add_argument("--pose_path", type=str, default=None)

    parser.add_argument("--pose_lr", type=float, default=1e-3)
    parser.add_argument("--pose_lr_end", type=float, default=1e-5)
    parser.add_argument("--warmup_pose", type=float, default=0)

    parser.add_argument(
        "--random_seed", type=int, default=-1
    )  # Set to 0 for deterministic behaviour

    parser.add_argument("--netdepth", type=int, default=8, help="layers in network")
    parser.add_argument("--netwidth", type=int, default=128, help="channels per layer")
    parser.add_argument(
        "--netdepth_fine", type=int, default=8, help="layers in fine network"
    )
    parser.add_argument(
        "--netwidth_fine",
        type=int,
        default=128,
        help="channels per layer in fine network",
    )
    parser.add_argument(
        "--N_rand",
        type=int,
        default=32 * 32 * 4,
        help="batch size (number of random rays per gradient step)",
    )
    parser.add_argument("--lrate", type=float, default=1e-4, help="learning rate")
    parser.add_argument(
        "--lrate_decay",
        type=int,
        default=250,
        help="exponential learning rate decay (in 1000 steps)",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=4096 * 16,
        help="number of rays processed in parallel, decrease if running out of memory",
    )
    parser.add_argument(
        "--netchunk",
        type=int,
        default=4096 * 16,
        help="number of pts sent through network in parallel, decrease if running out of memory",
    )

    parser.add_argument(
        "--ft_path",
        type=str,
        default=None,
        help="specific weights npy file to reload for coarse network",
    )

    # rendering options
    parser.add_argument(
        "--N_samples", type=int, default=64, help="number of coarse samples per ray"
    )
    parser.add_argument(
        "--i_embed",
        type=int,
        default=0,
        help="set 0 for default positional encoding, -1 for none",
    )
    parser.add_argument(
        "--i_embed_gauss",
        type=int,
        default=0,
        help="mapping size for Gaussian positional encoding, 0 for none",
    )

    parser.add_argument(
        "--multires",
        type=int,
        default=10,
        help="log2 of max freq for positional encoding (3D location)",
    )
    parser.add_argument(
        "--multires_views",
        type=int,
        default=4,
        help="log2 of max freq for positional encoding (2D direction)",
    )
    parser.add_argument(
        "--raw_noise_std",
        type=float,
        default=0.0,
        help="std dev of noise added to regularize sigma_a output, 1e0 recommended",
    )

    parser.add_argument(
        "--render_only",
        action="store_true",
        help="do not optimize, reload weights and render out render_poses path",
    )
    parser.add_argument(
        "--render_test",
        action="store_true",
        help="render the test set instead of render_poses path",
    )
    parser.add_argument(
        "--render_factor",
        type=int,
        default=0,
        help="downsampling factor to speed up rendering, set 4 or 8 for fast preview",
    )

    # training options

    # dataset options
    parser.add_argument("--dataset_type", type=str, default="us", help="options: us")
    parser.add_argument(
        "--testskip",
        type=int,
        default=8,
        help="will load 1/N images from test/val sets, useful for large datasets like deepvoxels",
    )

    # logging/saving options
    parser.add_argument(
        "--i_print",
        type=int,
        default=2000,
        help="frequency of console printout and metric loggin",
    )
    parser.add_argument(
        "--i_img", type=int, default=100, help="frequency of tensorboard image logging"
    )
    parser.add_argument(
        "--i_weights", type=int, default=10000, help="frequency of weight ckpt saving"
    )
    return parser


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
        images, poses, i_test = load_us_data(
            args.datadir, confmap=args.confmap, pose_path=args.pose_path
        )

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

    # Create barf model

    (
        render_kwargs_train,
        render_kwargs_test,
        start,
        optimizer,
        pose_optim,
        pose_sched,
    ) = create_barf(torch.tensor(poses[:, :3, :4]), args, device=device, mode="train")

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

    start = start + 1
    for i in trange(start, N_iters + 1):

        time0 = time.time()

        render_kwargs_train["network_fn"].progress.data.fill_(i / N_iters)

        img_i = np.random.choice(
            i_train
        )  # Why? This does not guarantee that all images are used

        if img_i < 4 or img_i > len(i_train + 1) - 4:  # Why?
            continue

        target = torch.Tensor(images[img_i]).to(device).unsqueeze(0).unsqueeze(0)
        pose = render_kwargs_train["pose_refine"].get_pose(int(img_i)).unsqueeze(0)

        #####  Core optimization loop  #####
        rendering_output = render_us(
            H, W, sw, sh, c2w=pose, chunk=args.chunk, retraw=True, **render_kwargs_train
        )
        output_image = rendering_output["intensity_map"]

        pose_optim.zero_grad()

        if args.warmup_pose:
            # simple linear warmup of pose learning rate
            pose_optim.param_groups[0]["lr_orig"] = pose_optim.param_groups[0][
                "lr"
            ]  # cache the original learning rate
            pose_optim.param_groups[0]["lr"] *= min(1, i / args.warmup_pose)

        optimizer.zero_grad()
        loss = {}

        if args.loss == "l2":
            l2_intensity_loss = img2mse(output_image, target)
            loss["l2"] = (1.0, l2_intensity_loss)
        elif args.loss == "ssim":
            ssim_intensity_loss = ssim_loss(output_image, target)
            loss["ssim"] = (ssim_weight, ssim_intensity_loss)
            l2_intensity_loss = img2mse(output_image, target)
            loss["l2"] = (l2_weight, l2_intensity_loss)

        total_loss = 0.0
        for loss_value in loss.values():
            tmp = loss_value[0] * loss_value[1]
            total_loss += tmp

        if type(total_loss) != torch.Tensor:
            raise ValueError("Loss is not a tensor: Problem with loss calculation")

        total_loss.backward()
        optimizer.step()
        pose_optim.step()
        pose_sched.step()

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

        dt = time.time() - time0
        if (i + 1) % args.i_print == 0:

            rendering_path = os.path.join(basedir, expname, "train_rendering")
            os.makedirs(
                os.path.join(basedir, expname, "train_rendering"), exist_ok=True
            )

            print(f"Step: {i+1}, Loss: {total_loss.item()}, Time: {dt}")  # type: ignore
            detailed_loss_string = ", ".join(
                [f"{k}: {v[1].item()}" for k, v in loss.items()]
            )
            print(detailed_loss_string)

            plt.figure(figsize=(16, 8))
            for j, m in enumerate(rendering_output):

                plt.subplot(3, 4, j + 1)
                plt.title(m)
                plt.imshow(rendering_output[m].detach().cpu().numpy()[0, 0].T)

            plt.subplot(3, 4, 12)
            plt.title("Target")
            plt.imshow(target.detach().cpu().numpy()[0, 0].T)

            plt.savefig(
                os.path.join(rendering_path, "{:08d}.png".format(i + 1)),
                bbox_inches="tight",
                dpi=200,
            )
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
                },
                path,
            )
            # print('Saved checkpoints at', path)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    if hasattr(torch, "set_default_device") and torch.cuda.is_available():
        torch.set_default_device("cuda")
    train()
