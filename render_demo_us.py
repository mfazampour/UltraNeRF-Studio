import argparse
import os
import pprint
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import run_ultranerf as run_nerf_ultrasound
from load_us import load_us_data
import open3d as o3d
import matplotlib.cm as cm

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser_console = argparse.ArgumentParser()
    parser_console.add_argument("--logdir", type=str, default="logs")
    parser_console.add_argument("--expname", type=str, default="test")
    parser_console.add_argument("--model_epoch", type=int, default=98000)
    args_console = parser_console.parse_args()

    config = os.path.join(args_console.logdir, args_console.expname, "config.txt")
    print("Args:")
    with open(config, "r") as f:
        print(f.read())

    parser = run_nerf_ultrasound.config_parser()
    model_no = f"{args_console.model_epoch:06d}"

    # Adjust here if your trained model checkpoint file differs in naming
    # For the above code snippet, checkpoints are saved as {step}.tar
    ft_path = os.path.join(args_console.logdir, args_console.expname, model_no + ".tar")
    args = parser.parse_args(["--config", config, "--ft_path", ft_path])

    print("Loaded args")

    model_name = os.path.basename(args.datadir)
    images, poses, i_test = load_us_data(args.datadir)

    H, W = images[0].shape
    H = int(H)
    W = int(W)

    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    print("Loaded image data")
    print(images.shape)
    print(poses.shape)

    near = 0.0
    far = args.probe_depth * 0.001

    # Create nerf model
    if args.reconstruction:
        _, render_kwargs_test, start, optimizer, optimizer_rec = run_nerf_ultrasound.create_nerf(
            args, device, mode="test"
        )
    else:
        _, render_kwargs_test, start, optimizer, _ = run_nerf_ultrasound.create_nerf(
            args, device, mode="test"
        )
    bds_dict = {
        "near": torch.tensor(near, dtype=torch.float32, device=device),
        "far": torch.tensor(far, dtype=torch.float32, device=device),
    }
    render_kwargs_test.update(bds_dict)

    print("Render kwargs:")
    pprint.pprint(render_kwargs_test)

    sw = args.probe_width * 0.001 / float(W)
    sh = args.probe_depth * 0.001 / float(H)

    # If you want a downsample factor, set here. For now, just keep as original.
    # down = 4
    render_kwargs_fast = {k: render_kwargs_test[k] for k in render_kwargs_test}

    frames = []
    impedance_map = []
    map_number = 0
    output_dir = os.path.join(
        args_console.logdir,
        args_console.expname,
        "output_maps_{}_{}_{}".format(model_name, model_no, map_number),
    )

    output_dir_params = os.path.join(output_dir, "params")
    output_dir_output = os.path.join(output_dir, "output")
    output_dir_compare = os.path.join(output_dir, "compare")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)
    os.makedirs(output_dir_params)
    os.makedirs(output_dir_output)
    os.makedirs(output_dir_compare)

    save_it = 300

    # We'll store the parameters from rendering in a dict of lists
    rendering_params_save = None

    # Convert poses and run rendering
    rec_list = None
    cmap = cm.plasma
    with (torch.no_grad()):
        for i, c2w in enumerate(poses[:600]):
            c2w_torch = torch.from_numpy(c2w[:3, :4]).to(device).unsqueeze(0)
            # render_us returns a dict of torch tensors
            rendering_params = run_nerf_ultrasound.render_us(
                H, W, sw, sh, c2w=c2w_torch, **render_kwargs_fast
            )

            pts = rendering_params["pts"].cpu().numpy() * 1000
            seg = rendering_params["reconstruction"].cpu().numpy().squeeze()
            seg[seg >= 0.5] = 255.
            seg[seg != 255.] = 0.
            seg = seg.transpose(1, 0)
            non_zero_mask = seg != 0
            seg = seg[non_zero_mask]
            pts = pts[non_zero_mask]
            # seg = seg.reshape(-1, 1)
            pts = pts.reshape(-1, 3)
            # rec = np.concatenate([pts, seg], axis=-1)
            rec = pts
            if rec_list is not None:
                rec_list = np.concatenate([rec_list, rec], axis=0)
            else:
                rec_list = rec

            # Convert intensity_map to uint8 and save.
            # Intensity maps are transposed here to match the saved image orientation.
    #         intensity_map = rendering_params["intensity_map"][0, 0]  # [H, W]
    #         intensity_map_transposed = intensity_map.T  # [W, H]
    #
    #         # Convert from [0,1] to uint8
    #         img_to_save = (
    #             (intensity_map_transposed * 255.0)
    #             .clamp(0, 255)
    #             .to(torch.uint8)
    #             .cpu()
    #             .numpy()
    #         )
    #
    #         real_image = (images[i] * 255).astype(np.uint8)
    #
    #         Image.fromarray(img_to_save).save(
    #             os.path.join(output_dir_output, f"Generated_{1000 + i}.png")
    #         )
    #
    #         plt.subplot(1, 2, 1)
    #         plt.imshow(img_to_save.T, cmap="gray")
    #         plt.title("Generated")
    #         plt.subplot(1, 2, 2)
    #         plt.imshow(real_image, cmap="gray")
    #         plt.title("Real")
    #         plt.savefig(os.path.join(output_dir_compare, f"Compare_{1000 + i}.png"))
    #
    #         # Save parameters for later
    #         if rendering_params_save is None:
    #             # Initialize the storage dict
    #             rendering_params_save = {}
    #             for key in rendering_params:
    #                 rendering_params_save[key] = []
    #
    #         for key, value in rendering_params.items():
    #             # Transpose values to match the saved image orientation.
    #             val_t = value[0, 0].T  # [W,H]
    #             rendering_params_save[key].append(val_t.cpu().numpy())
    #
    #         # Save intermediate results
    #         if i == save_it:
    #             # Save all parameters up to this point
    #             for key, value in rendering_params_save.items():
    #                 np_to_save = np.array(value)
    #                 np.save(f"{output_dir_params}/{key}.npy", np_to_save)
    #             rendering_params_save = None
    #
    #         elif i != save_it and i % save_it == 0 and i != 0:
    #             # Append results to existing files
    #             for key, value in rendering_params_save.items():
    #                 f_name = f"{output_dir_params}/{key}.npy"
    #                 np_to_save = np.array(value)
    #                 if os.path.exists(f_name):
    #                     np_existing = np.load(f_name)
    #                     new_to_save = np.concatenate((np_existing, np_to_save), axis=0)
    #                     np.save(f_name, new_to_save)
    #                 else:
    #                     np.save(f_name, np_to_save)
    #             rendering_params_save = None
    #
    # # Save any remaining parameters after loop ends
    # if rendering_params_save is not None:
    #     for key, value in rendering_params_save.items():
    #         f_name = f"{output_dir_params}/{key}.npy"
    #         np_to_save = np.array(value)
    #         if os.path.exists(f_name):
    #             np_existing = np.load(f_name)
    #             np_to_save = np.concatenate((np_existing, np_to_save), axis=0)
    #         np.save(f_name, np_to_save)

    # rec_list = np.array(rec_list)
    rec_list = rec_list.reshape(-1, 3)

    point_cloud = o3d.geometry.PointCloud()

    # Set the points (coordinates) of the point cloud
    point_cloud.points = o3d.utility.Vector3dVector(rec_list[..., :3])

    # Optionally, color the points by intensity (this requires RGB format)
    # colors = np.zeros((rec_list[..., :3].shape[0], 3))  # Initialize with black
    # colors[:, 0] = rec_list[..., 3]  # Set red channel to intensity values
    # point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])
