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
import open3d as o3d
import numpy as np
import torch
import mcubes  # For Marching Cubes


def extract_mesh_from_sdf(sdf, voxel_size=0.01):
    """
    Extracts a 3D mesh from an SDF using Marching Cubes.

    Args:
        sdf (torch.Tensor): Signed Distance Function (SDF) volume.
        voxel_size (float): Size of each voxel.

    Returns:
        o3d.geometry.TriangleMesh: Reconstructed 3D mesh.
    """
    sdf_np = sdf.cpu().numpy()  # Convert to NumPy
    verts, faces = mcubes.marching_cubes(sdf_np, isovalue=0)

    # Scale the mesh to correct size
    verts = verts * voxel_size

    # Convert to Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    return mesh
def compute_sdf(occupied_points, grid_res=100, truncation=0.05, device="cuda"):
    """
    Computes a Signed Distance Function (SDF) from an occupancy grid.

    Args:
        occupied_points (np.ndarray): Mx3 array of occupied points.
        grid_res (int): Resolution of the SDF grid.
        truncation (float): Truncation distance for TSDF.
        device (str): "cuda" for GPU acceleration.

    Returns:
        torch.Tensor: SDF volume (grid_res³).
    """
    occupied_points = torch.tensor(occupied_points, device=device, dtype=torch.float32)

    # Compute bounding box
    min_bound = occupied_points.min(dim=0)[0] - 0.1
    max_bound = occupied_points.max(dim=0)[0] + 0.1

    # Create 3D grid
    x = torch.linspace(min_bound[0], max_bound[0], grid_res, device=device)
    y = torch.linspace(min_bound[1], max_bound[1], grid_res, device=device)
    z = torch.linspace(min_bound[2], max_bound[2], grid_res, device=device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    grid_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)

    # Compute distance to nearest occupied point
    dists = torch.cdist(grid_points, occupied_points)
    min_distances, _ = torch.min(dists, dim=1)

    # Convert distances into SDF values
    sdf_values = torch.clamp(min_distances, -truncation, truncation) / truncation
    return sdf_values.view(grid_res, grid_res, grid_res)  # Reshape into 3D grid

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
    with (torch.no_grad()):
        for c2w, pts in zip(poses[:600], poses_labels[:600]):
            c2w_torch = torch.from_numpy(c2w[:3, :4]).to(device).unsqueeze(0)
            # render_us returns a dict of torch tensors
            rendering_output = run_nerf_ultrasound.render_us(
                H, W, sw, sh, c2w=c2w_torch, **render_kwargs_fast
            )

            r = rendering_output["reflection_coeff"].detach().clone().permute(0, 3, 2, 1)
            a = rendering_output["attenuation_coeff"].detach().clone().permute(0, 3, 2, 1)
            s = rendering_output["scatter_amplitude"].detach().clone().permute(0, 3, 2, 1)

            theta = torch.concatenate([r, a, s], dim=-1)

            input_reconstruction = torch.cat([torch.tensor(pts.squeeze(), device='cuda'), theta.squeeze()], dim=-1)
            ret_reconstruction = render_kwargs_test["network_query_fn_rec"](input_reconstruction,
                                                                            render_kwargs_test["network_rec"])
            # rendering_output["confidence_maps"] *
            output = rendering_output["confidence_maps"] * ret_reconstruction.permute(2, 1, 0)[None, ...]
            seg = output.cpu().numpy().squeeze()
            weights = rendering_output["reflection_coeff"].cpu().numpy().squeeze().squeeze().transpose(1, 0)

            seg[seg >= 0.5] = 1.
            seg[seg != 1.] = 0.
            seg = seg.transpose(1, 0)
            # non_zero_mask = seg != 0
            # seg = seg[non_zero_mask]
            # pts = pts[non_zero_mask]
            seg = seg.reshape(-1, 1)
            pts = pts.reshape(-1, 3)
            rec = np.concatenate([pts, seg], axis=-1)
            if rec_list is not None:
                rec_list = np.concatenate([rec_list, rec], axis=0)
                weights_list = np.concatenate([weights_list, weights], axis=0)
            else:
                rec_list = rec
                weights_list = weights

    rec_list = rec_list.reshape(-1, 4)
    weights = weights_list.reshape(-1)
    # Optionally, color the points by intensity (this requires RGB format)
    # colors = np.zeros((rec_list[..., :3].shape[0], 3))  # Initialize with black
    # colors[:, 0] = rec_list[..., 3]  # Set red channel to intensity values
    # point_cloud.colors = o3d.utility.Vector3dVector(colors)
    occupied_points = rec_list[rec_list[..., 3] == 1.][..., :3]
    occupied_ind = np.where(rec_list[..., 3] == 1)
    # Visualize the point cloud
    # Normalize weights to sum to 1 (for probability sampling)
    weights = weights[occupied_ind] / np.sum(weights[occupied_ind])

    # Define the number of points to sample (adjust as needed)
    num_samples = min(10000, rec_list.shape[0])  # Sample at most 50k points
    print(weights.shape)
    print(rec_list.shape)

    # Sample points based on weights
    sample_indices = np.random.choice(occupied_points.shape[0], size=num_samples, p=weights, replace=False)
    # sampled_points = rec_list[sample_indices]

    # sampled_indices = torch.randperm(rec_list.shape[0])[:2000]  # Use only 5,000 random points
    # sampled_points = rec_list[sampled_indices]
    # Compute distances only to sampled points

    point_cloud = o3d.geometry.PointCloud()

    # Set the points (coordinates) of the point cloud
    point_cloud.points = o3d.utility.Vector3dVector(occupied_points)
    points = np.asarray(point_cloud.points)

    # Compute min and max per axis
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)

    # Normalize using min-max scaling
    normalized_points = (points - min_vals) / (max_vals - min_vals)

    # Create new Open3D point cloud with normalized points
    normalized_pcd = o3d.geometry.PointCloud()
    normalized_pcd.points = o3d.utility.Vector3dVector(normalized_points)
    o3d.io.write_point_cloud("normalized_point_cloud.ply", normalized_pcd)
    normalized_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #
    # # Orient normals consistently
    # point_cloud.orient_normals_consistent_tangent_plane(k=10)

    # # Visualize point cloud with normals
    o3d.visualization.draw_geometries([normalized_pcd], point_show_normal=True)
    # o3d.visualization.draw_geometries([point_cloud])
    # occupied_points = sampled_points[sampled_points[..., 3] == 1.][..., :3]

    downsampled_points = np.asarray(normalized_pcd.points)[sample_indices]
    downsampled_normals = np.asarray(normalized_pcd.normals)[sample_indices]

    # Create new Open3D point cloud with downsampled data
    downsampled_pcd = o3d.geometry.PointCloud()
    downsampled_pcd.points = o3d.utility.Vector3dVector(downsampled_points)
    downsampled_pcd.normals = o3d.utility.Vector3dVector(downsampled_normals)

    # Visualize the downsampled point cloud
    o3d.visualization.draw_geometries([downsampled_pcd])

    # sdf = compute_sdf(occupied_points)
    # torch.save(sdf.cpu(), "sdf.pt")
    # print("SDF computed and saved.")

    # # Extract mesh
    # mesh = extract_mesh_from_sdf(sdf)
    # o3d.visualization.draw_geometries([mesh])

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(normalized_pcd, depth=9)

    # Save and visualize
    o3d.io.write_triangle_mesh("mesh_poisson.ply", mesh)
    o3d.visualization.draw_geometries([mesh])

    # # Save the mesh
    # o3d.io.write_triangle_mesh("reconstructed_mesh.ply", mesh)
    # print("Mesh extracted and saved.")
    #
    # mesh = o3d.io.read_triangle_mesh("reconstructed_mesh.ply")
    # o3d.visualization.draw_geometries([mesh])