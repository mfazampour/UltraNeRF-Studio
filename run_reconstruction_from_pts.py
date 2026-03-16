"""Reconstruct geometry by querying the PyTorch reconstruction network on points.

This offline script loads a trained reconstruction-capable checkpoint, evaluates
the reconstruction network on a dense set of query points or a dense grid, and
then turns the predicted occupancy field into point clouds or meshes for
inspection.

Compared with ``run_reconstruction.py``, this path focuses more directly on
point-based or dense-grid reconstruction queries.
"""

import argparse
import os
import pprint
import shutil
import trimesh

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import run_ultranerf_reconstruction as run_nerf_ultrasound
from load_us import load_us_data
import open3d as o3d
import matplotlib.cm as cm
import open3d as o3d
import numpy as np
import torch
import mcubes  # For Marching Cubes


def denormalize_mesh(vertices, faces, min_vals, max_vals):
    """
    Denormalize a mesh's vertices back to the original scale.

    Args:
    - vertices (numpy array): The normalized vertices of the mesh (Nx3).
    - faces (numpy array): The faces of the mesh (Mx3) with indices referring to the vertices.
    - min_vals (tuple): The minimum values of the original mesh (min_x, min_y, min_z).
    - max_vals (tuple): The maximum values of the original mesh (max_x, max_y, max_z).

    Returns:
    - numpy array: The denormalized vertices (Nx3).
    - numpy array: The faces (Mx3) remain unchanged.
    """
    # Unpack the min and max values for x, y, z
    min_x, min_y, min_z = min_vals
    max_x, max_y, max_z = max_vals

    # Denormalize the vertices
    denormalized_vertices = np.zeros_like(vertices)
    denormalized_vertices[:, 0] = vertices[:, 0] * (max_x - min_x) + min_x  # x-axis
    denormalized_vertices[:, 1] = vertices[:, 1] * (max_y - min_y) + min_y  # y-axis
    denormalized_vertices[:, 2] = vertices[:, 2] * (max_z - min_z) + min_z  # z-axis

    # Faces remain unchanged, as they are just indices referring to the vertices
    denormalized_faces = faces

    return denormalized_vertices, denormalized_faces


def denormalize_point_cloud(normalized_points, min_vals, max_vals):
    """
    Denormalize a point cloud back to its original scale.

    Args:
    - normalized_points (numpy array): The normalized point cloud (Nx3).
    - min_vals (tuple): The minimum values of the original point cloud (min_x, min_y, min_z).
    - max_vals (tuple): The maximum values of the original point cloud (max_x, max_y, max_z).

    Returns:
    - numpy array: The denormalized point cloud (Nx3).
    """
    # Unpack the min and max values for x, y, z
    min_x, min_y, min_z = min_vals
    max_x, max_y, max_z = max_vals

    # Denormalize the points
    denormalized_points = np.zeros_like(normalized_points)
    denormalized_points[:, 0] = normalized_points[:, 0] * (max_x - min_x) + min_x  # x-axis
    denormalized_points[:, 1] = normalized_points[:, 1] * (max_y - min_y) + min_y  # y-axis
    denormalized_points[:, 2] = normalized_points[:, 2] * (max_z - min_z) + min_z  # z-axis

    return denormalized_points

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
    images, poses,  labels, poses_labels, i_test = load_us_data(args.datadir, reconstruction=True)

    H, W = images[0].shape
    H = int(H)
    W = int(W)

    images = images.astype(np.float32)
    poses = poses.astype(np.float32)
    poses_labels = poses_labels.astype(np.float32)

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
    weights_list = None
    cmap = cm.plasma

    import torch
    import numpy as np


    # Assume `occupancy_net` is your trained PyTorch model
    # The model should take (N, 3) tensor of (x, y, z) points and output (N,) occupancy values

    def mgrid(resolution=400, device='cuda'):
        # Generate a 3D grid of query points
        x = torch.linspace(0, 1, resolution, device=device)
        y = torch.linspace(0, 1, resolution, device=device)
        z = torch.linspace(0, 1, resolution, device=device)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')  # Shape: (res, res, res)

        # Flatten grid into (N, 3) shape
        points = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)  # (N, 3)

        return points  # Convert to NumPy for meshing


    # Example usage:
    # occupancy_net = YourOccupancyNetwork().to(device)  # Load your trained model
    # occupancy_grid = evaluate_occupancy(occupancy_net, resolution=128)

    pts = mgrid()
    print(pts.shape)
    poses_labels = pts.reshape(-1, 1000, 1000, 3)
    with ((torch.no_grad())):
        for i, pts in enumerate(poses_labels):
            # pts = torch.from_numpy(pts).to(device)
            # render_us returns a dict of torch tensors
            if args.rec_only_occ:
                print(i)
                render_kwargs_test["pts"] = pts

                input_reconstruction = torch.tensor(pts, device='cuda').squeeze()
                ret_reconstruction = render_kwargs_test["network_query_fn_rec"](input_reconstruction,
                                                                                render_kwargs_test["network_rec"])
            else:
                print(i)
                render_kwargs_test["pts"] = pts
                rendering_output = run_nerf_ultrasound.render_us(
                    H, W, sw, sh, c2w=None, chunk=args.chunk, retraw=True, **render_kwargs_test
                )
                r = rendering_output["reflection_coeff"].detach().clone().permute(0, 3, 2, 1)
                a = rendering_output["attenuation_coeff"].detach().clone().permute(0, 3, 2, 1)
                s = rendering_output["scatter_amplitude"].detach().clone().permute(0, 3, 2, 1)
                theta = torch.concatenate([r, a, s], dim=-1)
                input_reconstruction = theta.squeeze() \
                    if args.rec_only_theta else torch.cat([torch.tensor(pts, device='cuda').squeeze(), theta.squeeze()],
                                                          dim=-1)
                ret_reconstruction = render_kwargs_test["network_query_fn_rec"](input_reconstruction,
                                                                                render_kwargs_test["network_rec"])
                # rendering_output["confidence_maps"] *
                # if args.confidence:
                #     output = torch.sigmoid(rendering_output["confidence_maps"] * ret_reconstruction.permute(2, 1, 0)[None, ...])
                # else:
            output = ret_reconstruction.permute(2, 1, 0)[None, ...]
            seg = output.cpu().numpy().squeeze()
            weights = (torch.ones_like(output) / output.reshape(-1, 1).shape[0]).cpu().numpy().squeeze().squeeze() if args.rec_only_occ \
                else r.squeeze().cpu().numpy().squeeze().squeeze().transpose(1, 0)
            seg[seg >= 0.5] = 1.
            seg[seg != 1.] = 0.
            seg = seg.transpose(1, 0)
            seg = seg.reshape(-1, 1)
            pts = pts.reshape(-1, 3).cpu().numpy().squeeze().squeeze()
            rec = np.concatenate([pts, seg], axis=-1)
            if rec_list is not None:
                rec_list = np.concatenate([rec_list, rec], axis=0)
                weights_list = np.concatenate([weights_list, weights], axis=0)
            else:
                rec_list = rec
                weights_list = weights



    rec_list = rec_list.reshape(-1, 4)
    weights = weights_list.reshape(-1)
    occupied_points = rec_list[rec_list[..., 3] == 1.][..., :3]
    occupied_ind = np.where(rec_list[..., 3] == 1)
    weights = weights / np.sum(weights)
    # Define the number of points to sample (adjust as needed)
    num_samples = min(20000, rec_list.shape[0])  # Sample at most 50k points
    print(weights.shape)
    print(occupied_points.shape)
    # Sample points based on weights
    sample_indices = np.random.choice(rec_list.shape[0], size=num_samples, p=weights, replace=False)

    point_cloud = o3d.geometry.PointCloud()

    # Set the points (coordinates) of the point cloud
    point_cloud.points = o3d.utility.Vector3dVector(occupied_points)
    points = np.asarray(point_cloud.points)
    o3d.io.write_point_cloud("spine.ply", point_cloud)

    # downsampled_points = np.asarray(point_cloud.points)[sample_indices]
    # # Create new Open3D point cloud with downsampled data
    # downsampled_pcd = o3d.geometry.PointCloud()
    # downsampled_pcd.points = o3d.utility.Vector3dVector(downsampled_points)
    # Visualize the downsampled point cloud
    o3d.visualization.draw_geometries([point_cloud])
    # create mesh grid
    # # Extract the mesh using Marching Cubes
    # vertices, triangles = mcubes.marching_cubes(occupancy.astype(float), isovalue=0.5)
    #
    # # Convert to a Trimesh object and save
    # mesh = trimesh.Trimesh(vertices, triangles)
    # mesh.export("reconstructed_mesh.obj")

    # Visualize
    # mesh.show()
    # # Save and visualize
    # o3d.io.write_triangle_mesh("mesh_poisson.ply", mesh)
    # o3d.visualization.draw_geometries([mesh])
    occupancy_grid = rec_list[..., 3].reshape(400, 400, 400)

    vertices, triangles = mcubes.marching_cubes(occupancy_grid[40:360, 40:300, 40:300], isovalue=0.5)

    # Convert to a Trimesh object and save
    mesh = trimesh.Trimesh(vertices, triangles)
    mesh.export(f"reconstructed_mesh_{args.expname}.obj")

    # Visualize
    mesh.show()

    # min_values = [-48.145203, -45.86122,  -25.578945] sim
    # max_value = [47.307354, 39.781727, 26.907543]

    #22_51
    min_values = [-45.848377, -42.556973, -76.779076]
    max_values = [45.65656,  42.755325, 76.74994 ]

    denorm_mesh = denormalize_mesh(vertices, triangles, min_values, max_values)

    # denorm = denormalize_point_cloud(points, min_values, max_value)
    #
    # point_cloud = o3d.geometry.PointCloud()
    #
    # # Set the points (coordinates) of the point cloud
    # point_cloud.points = o3d.utility.Vector3dVector(denorm)
    #
    # # vertices, triangles = mcubes.marching_cubes(occupancy_grid, isovalue=0.5)
    #
    # # Convert to a Trimesh object and save
    # # mesh = trimesh.Trimesh(vertices, triangles)
    # # mesh.export(f"reconstructed_mesh_{args.expname}.obj")
    #
    # # Visualize
    # o3d.visualization.draw_geometries([point_cloud])

    mesh = trimesh.Trimesh(denorm_mesh[0], denorm_mesh[1])
    mesh.export(f"reconstructed_mesh_{args.expname}_denorm.obj")
    #
    # # Visualize
    # mesh.show()
