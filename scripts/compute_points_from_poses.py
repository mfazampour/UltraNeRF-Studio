from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if ROOT.name == "scripts":
    SRC = ROOT.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import ultranerf.nerf_utils as nerf_utils
import numpy as np
import open3d as o3d


poses = np.load("./data/processed22_51_2002/poses.npy")
images = np.load("./data/processed22_51_2002/images.npy")[0]
H, W = images.shape[0], images.shape[1]
probe_depth, probe_width = 85, 51
sy = probe_depth / float(H)
sx = probe_width / float(W)
sh = sy
sw = sx
near = 0.
far = probe_depth
pts_s = None
print("New")
print(poses.shape)
print(images.shape)
min_p = None
max_p = None
list_p = list()
# Pre-calculate the total number of points (assuming len(poses) is known and each pts has the same size)
num_poses = poses.shape[0]

# Pre-allocate the array to store all points (reshape based on your data's shape)
pts_s = np.empty((num_poses*W*H, 3), dtype=np.float32).reshape(poses.shape[0], images.shape[1],
                                                                             images.shape[0], 3)

for i, c2w in enumerate(poses):
    print(i)
    p = torch.from_numpy(c2w).to('cuda')
    pts = nerf_utils.compute_pts_from_pose(H, W, sw, sh, p, near, far).cpu().numpy()

    # Fill the pre-allocated array directly
    pts_s[i] = pts

# Flatten and normalize the points as before
flattened_points = pts_s.reshape(-1, 3)

min_vals = flattened_points.min(axis=0)
max_vals = flattened_points.max(axis=0)
print(f"min: {min_vals}, max: {max_vals}")
normalized_points = np.empty_like(flattened_points)
normalized_points = (flattened_points - min_vals) / (max_vals - min_vals)
print("reshaping")
normalized_point_cloud = normalized_points.reshape(pts_s.shape)
# Save the results
print("saving")
np.save("./data/processed22_51_2002/poses_labels.npy", normalized_point_cloud)
print("saved")
# # Visualize and save the point cloud
# pc = o3d.geometry.PointCloud()
# pc.points = o3d.utility.Vector3dVector(flattened_points[::1000])  # downsampling for visualization
# o3d.io.write_point_cloud("processed22_51_2002.ply", pc)
# # for i, c2w in enumerate(poses):
#     print(i)
#     p = torch.from_numpy(c2w).to('cuda')
#     pts = nerf_utils.compute_pts_from_pose(H, W, sw, sh, p, near, far).cpu().numpy()[None, ...]
#     if pts_s is None:
#         pts_s = pts
#     else:
#         pts_s = np.concatenate([pts_s, pts], axis=0)
# # pts_s = np.array(list_p)
# flattened_points = pts_s.reshape(-1, 3)
#
# # Normalize to [0, 1] by subtracting the minimum and dividing by the range
# min_vals = flattened_points.min(axis=0)
# max_vals = flattened_points.max(axis=0)
#
# print(f"min: {min_vals}, max: {max_vals}")
#
# normalized_points = (flattened_points - min_vals) / (max_vals - min_vals)
# print("reshaping")
# normalized_point_cloud = normalized_points.reshape(pts_s.shape)
# print("saving")
#
# np.save("./data/processed22_51_2002/poses_labels.npy", normalized_point_cloud)
# # np.save("processed22_51_2002.npy", flattened_points)
# # # o3d.io.write_point_cloud(normalized_points.reshape(-1,3), "felix.ply")
# pc = o3d.geometry.PointCloud()
# pc.points = o3d.utility.Vector3dVector(flattened_points[::1000])
# o3d.io.write_point_cloud("processed22_51_2002.ply", pc)


def calculate_scale(point_cloud):
    # Convert point cloud to a numpy array if it's not already
    point_cloud = np.array(point_cloud)

    # Find the min and max for each axis (X, Y, Z)
    min_x, min_y, min_z = point_cloud.min(axis=0)
    max_x, max_y, max_z = point_cloud.max(axis=0)

    # Calculate the range (scale) for each axis
    scale_x = max_x - min_x
    scale_y = max_y - min_y
    scale_z = max_z - min_z

    return scale_x, scale_y, scale_z


# Example usage:
point_cloud = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                        [10, 11, 12]])

scale_x, scale_y, scale_z = calculate_scale(flattened_points)

print(f"Scale in X direction: {scale_x}")
print(f"Scale in Y direction: {scale_y}")
print(f"Scale in Z direction: {scale_z}")

