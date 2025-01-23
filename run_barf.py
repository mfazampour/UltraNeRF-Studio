import os
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from evident_border import get_borders
from load_us import load_us_data
from run_ultranerf_helpers import (BARF, PoseRefine, get_embedder,
                                   get_rays_us_linear, img2mse)

torch.cuda.set_per_process_memory_fraction(0.8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def gaussian_kernel(size: int, mean: float, std: float):
    delta_t = 1.0
    x_cos = torch.arange(-size, size + 1, dtype=torch.float32, device=device) * delta_t

    d1 = torch.distributions.Normal(mean, std * 2.0)
    d2 = torch.distributions.Normal(mean, std)

    vals_x = torch.exp(d1.log_prob(x_cos)).to(device)
    vals_y = torch.exp(d2.log_prob(x_cos)).to(device)

    gauss_kernel = torch.outer(vals_x, vals_y)
    gauss_kernel /= torch.sum(gauss_kernel)

    return gauss_kernel


g_size = 3
g_mean = 0.0
g_variance = 1.0
g_kernel = gaussian_kernel(g_size, g_mean, g_variance)
g_kernel = g_kernel.unsqueeze(0).unsqueeze(0)


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat(
            [fn(inputs[i : i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0
        )

    return ret


def run_network(inputs, fn, embed_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'."""
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(
        outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]
    )
    return outputs


def exclusive_cumprod(x):
    cumprod = torch.cumprod(x, dim=1)
    cumprod = torch.roll(cumprod, 1, dims=1)
    cumprod[:, 0] = 1.0
    return cumprod


def exclusive_cumsum(x):
    cumsum = torch.cumsum(x, dim=1)
    cumsum = torch.roll(cumsum, 1, dims=1)
    cumsum[:, 0] = 0.0
    return cumsum


def render_method_convolutional_ultrasound(raw, z_vals):
    # Compute distances between points
    dists = torch.abs(z_vals[..., :-1] - z_vals[..., 1:])
    dists = torch.cat([dists, dists[:, -1:]], dim=1)

    # Attenuation
    attenuation_coeff = torch.abs(raw[..., 0])
    log_attenuation = -attenuation_coeff * dists
    log_attenuation_transmission = exclusive_cumsum(log_attenuation)
    attenuation_transmission = torch.exp(log_attenuation_transmission)

    # Reflection
    prob_border = torch.sigmoid(raw[..., 2])
    border_distribution = torch.distributions.Bernoulli(probs=prob_border)
    border_indicator = border_distribution.sample().detach()

    reflection_coeff = torch.sigmoid(raw[..., 1])
    reflection_transmission = 1.0 - reflection_coeff  # PAPER CODE DIFFERENCE
    log_reflection_transmission = torch.log(reflection_transmission + 1e-8)
    log_reflection_transmission = exclusive_cumsum(log_reflection_transmission)
    reflection_transmission = torch.exp(log_reflection_transmission)

    # Border convolution

    border_indicator_conv_input = border_indicator.unsqueeze(0).unsqueeze(0)
    border_convolution = F.conv2d(border_indicator_conv_input, g_kernel, padding="same")
    border_convolution = border_convolution.squeeze(0).squeeze(0)

    # Backscattering
    density_coeff_value = torch.sigmoid(raw[..., 3])
    density_coeff = torch.ones_like(reflection_coeff) * density_coeff_value
    scatter_density_distribution = torch.distributions.Bernoulli(probs=density_coeff)
    scatterers_density = scatter_density_distribution.sample()

    amplitude = torch.sigmoid(raw[..., 4])
    scatterers_map = scatterers_density * amplitude
    scatterers_map_conv_input = scatterers_map.unsqueeze(0).unsqueeze(0)
    psf_scatter = F.conv2d(scatterers_map_conv_input, g_kernel, padding="same")
    psf_scatter = psf_scatter.squeeze(0).squeeze(0)

    # Compute remaining intensity
    transmission = attenuation_transmission * reflection_transmission

    # Final echo
    b = transmission * psf_scatter
    r = transmission * reflection_coeff * border_convolution
    intensity_map = b + r

    ret = {
        "intensity_map": intensity_map,
        "attenuation_coeff": attenuation_coeff,
        "reflection_coeff": reflection_coeff,
        "attenuation_transmission": attenuation_transmission,
        "reflection_transmission": reflection_transmission,
        "scatterers_density": scatterers_density,
        "scatterers_density_coeff": density_coeff,
        "scatter_amplitude": amplitude,
        "b": b,
        "r": r,
        "transmission": transmission,
        "border_convolution": border_convolution,
        "border_indicator": border_indicator,
    }
    return ret


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i : i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render_us(
    H,
    W,
    sw,
    sh,
    chunk=1024 * 32,
    rays=None,
    c2w=None,
    near=0.0,
    far=55.0 * 0.001,
    **kwargs,
):
    """Render rays"""
    if c2w is not None:
        rays_o, rays_d = get_rays_us_linear(H, W, sw, sh, c2w)
    else:
        if rays is None:
            raise ValueError("Must provide rays if c2w is not provided")
        rays_o, rays_d = rays

    sh = rays_d.shape  # [..., 3]

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(
        rays_d[..., :1]
    )
    near, far = near.to(device), far.to(device)

    rays = torch.cat([rays_o, rays_d, near, far], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)

    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    return all_ret


def create_barf(args):
    """Instantiate BARF's MLP model."""
    embed_fn, input_ch = get_embedder(
        args.multires, device, args.i_embed, args.i_embed_gauss
    )

    input_ch_views = 0
    embeddirs_fn = None

    output_ch = args.output_ch
    skips = [4]
    model = BARF(
        D=args.netdepth,
        W=args.netwidth,
        input_ch=input_ch,
        output_ch=output_ch,
        skips=skips,
        L=args.L,
    ).to(device)

    grad_vars = list(model.parameters())

    network_query_fn = lambda inputs, network_fn: run_network(
        inputs, network_fn, embed_fn=embed_fn, netchunk=args.netchunk
    )

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != "None":
        ckpts = [args.ft_path]
    else:
        ckpts = [
            os.path.join(basedir, expname, f)
            for f in sorted(os.listdir(os.path.join(basedir, expname)))
            if "tar" in f
        ]

    print("Found ckpts", ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print("Reloading from", ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt["global_step"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Load model
        model.load_state_dict(ckpt["network_fn_state_dict"])

    ##########################

    render_kwargs_train = {
        "network_query_fn": network_query_fn,
        "N_samples": args.N_samples,
        "network_fn": model,
        "ckpt": ckpt if len(ckpts) > 0 else None,
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def render_rays(
    ray_batch, network_fn, network_query_fn, N_samples, lindisp=False, **kwargs
):
    """Volumetric rendering."""

    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    t_vals = torch.linspace(0.0, 1.0, steps=N_samples, device=device)

    if not lindisp:
        z_vals = near * (1.0 - t_vals) + far * (t_vals)
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    origin = rays_o[..., None, :]
    step = rays_d[..., None, :] * z_vals[..., :, None]
    pts = origin + step

    raw = network_query_fn(pts, network_fn)
    ret = render_method_convolutional_ultrasound(raw, z_vals)

    return ret


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
    parser.add_argument("--border_lambda", type=float, default=0.0)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--confmap", type=bool, default=False)
    parser.add_argument("--pose_path", type=str, default=None)

    parser.add_argument("--pose_lr", type=float, default=1e-3)
    parser.add_argument("--pose_lr_end", type=float, default=1e-5)
    parser.add_argument("--warmup_pose", type=float, default=0)

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
        "--no_batching",
        action="store_true",
        help="only take random rays from 1 image at a time",
    )
    parser.add_argument(
        "--no_reload", action="store_true", help="do not reload weights from saved ckpt"
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
    parser.add_argument(
        "--precrop_iters",
        type=int,
        default=0,
        help="number of steps to train on central crops",
    )
    parser.add_argument(
        "--precrop_frac",
        type=float,
        default=0.5,
        help="fraction of img taken for central crops",
    )

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
        "--i_weights", type=int, default=50000, help="frequency of weight ckpt saving"
    )
    parser.add_argument(
        "--i_testset", type=int, default=5000000, help="frequency of testset saving"
    )
    parser.add_argument(
        "--i_video",
        type=int,
        default=5000000,
        help="frequency of render_poses video saving",
    )
    parser.add_argument(
        "--log_compression",
        action="store_true",
        help="use lossy compression for tensorboard logs",
    )

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None

    if args.dataset_type == "us":

        if args.pose_path:
            pose_path = args.pose_path
        else:
            pose_path = None

        images, poses, i_test = load_us_data(
            args.datadir, confmap=args.confmap, pose_path=pose_path
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
    # H, W = int(H), int(W)

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname

    if args.tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(basedir, "summaries", expname))

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
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_barf(
        args
    )

    # Create pose refiner
    pose_refine = PoseRefine(poses=torch.tensor(poses[:, :3, :4]), mode="train")
    pose_refine = pose_refine.to(device)

    pose_optim = torch.optim.Adam(pose_refine.parameters(), args.pose_lr)
    gamma = (args.pose_lr_end / args.pose_lr) ** (1.0 / args.n_iters)
    pose_sched = torch.optim.lr_scheduler.ExponentialLR(pose_optim, gamma)

    if render_kwargs_train["ckpt"] is not None:
        pose_refine.load_state_dict(
            render_kwargs_train["ckpt"]["pose_refine_state_dict"]
        )
        pose_optim.load_state_dict(render_kwargs_train["ckpt"]["pose_optim_state_dict"])
        pose_sched.load_state_dict(render_kwargs_train["ckpt"]["pose_sched_state_dict"])

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

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    start = start + 1
    for i in trange(start, N_iters + 1):
        time0 = time.time()

        render_kwargs_train["network_fn"].progress.data.fill_(i / N_iters)

        img_i = np.random.choice(i_train)
        target = torch.transpose(torch.tensor(images[img_i]), 0, 1).to(device)

        pose = pose_refine.get_pose(int(img_i))
        # pose = torch.from_numpy(poses[img_i, :3, :4]).to(device)

        ssim_weight = args.ssim_lambda
        l2_weight = 1.0 - ssim_weight

        rays_o, rays_d = get_rays_us_linear(H, W, sw, sh, pose)  # (H, W, 3), (H, W, 3)

        batch_rays = torch.stack([rays_o, rays_d], 0)  # (2, H*W, 3)

        loss = {}

        #####  Core optimization loop  #####

        rendering_output = render_us(
            H,
            W,
            sw,
            sh,
            chunk=args.chunk,
            c2w=pose,
            rays=batch_rays,
            **render_kwargs_train,
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

        if args.loss == "l2":
            l2_intensity_loss = img2mse(output_image, target)
            loss["l2"] = (1.0, l2_intensity_loss)
        elif args.loss == "ssim":
            raise

        # Calculate border loss
        if args.border_lambda > 0.0:
            evident_borders = (
                torch.tensor(
                    get_borders(
                        target.detach().cpu().numpy(),
                        niter=10,
                        kappa=0.1,
                        lambda_=0.1,
                        eps=1e-8,
                    )
                )
                .float()
                .to(device)
            )
            pred_border_indicator = rendering_output["border_indicator"]

            loss["border"] = (
                args.border_lambda,
                torch.mean(torch.abs(pred_border_indicator - evident_borders)),
            )

        total_loss = sum(loss[k][0] * loss[k][1] for k in loss)
        total_loss.backward()  # type: ignore

        optimizer.step()
        pose_optim.step()

        if args.warmup_pose:
            pose_optim.param_groups[0]["lr"] = pose_optim.param_groups[0][
                "lr_orig"
            ]  # reset learning rate

        pose_sched.step()

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
            writer.add_scalar("Pose learning rate", pose_optim.param_groups[0]["lr"], i)

        dt = time.time() - time0
        if (i + 1) % args.i_print == 0:

            rendering_path = os.path.join(basedir, expname, "train_rendering")
            os.makedirs(
                os.path.join(basedir, expname, "train_rendering"), exist_ok=True
            )

            print(f"Step: {(i+1)}, Loss: {total_loss.item()}, Time: {dt}")  # type: ignore
            detailed_loss_string = ", ".join(
                [f"{k}: {v[1].item()}" for k, v in loss.items()]
            )
            print(detailed_loss_string)

            plt.figure(figsize=(16, 10))
            for j, m in enumerate(rendering_output):

                plt.subplot(4, 4, j + 1)
                plt.title(m)
                plt.imshow(rendering_output[m].detach().cpu().numpy())

            plt.subplot(4, 4, 14)
            plt.title("Target")
            plt.imshow(target.detach().cpu().numpy())

            if args.border_lambda > 0.0:
                plt.subplot(4, 4, 15)
                plt.title("Evident borders")
                plt.imshow(evident_borders.detach().cpu().numpy())

            plt.savefig(
                os.path.join(rendering_path, "{:08d}.png".format((i + 1))),
                bbox_inches="tight",
                dpi=200,
            )
            plt.close()

        # Rest is logging
        if (i + 1) % args.i_weights == 0:
            path = os.path.join(basedir, expname, "{:06d}.tar".format((i + 1)))
            torch.save(
                {
                    "global_step": i,
                    "network_fn_state_dict": render_kwargs_train[
                        "network_fn"
                    ].state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "pose_refine_state_dict": pose_refine.state_dict(),
                    "pose_optim_state_dict": pose_optim.state_dict(),
                    "pose_sched_state_dict": pose_sched.state_dict(),
                },
                path,
            )
            # print('Saved checkpoints at', path)


if __name__ == "__main__":

    train()
