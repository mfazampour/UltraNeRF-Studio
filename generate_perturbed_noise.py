import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from camera import Lie, Pose
from test_utils import visualize_poses


def descale_poses(poses, H, W, probe_depth, probe_width):

    sh = probe_depth / H # real-world mm per pixel (y)
    sw = probe_width / W # real-world mm per pixel (x)

    poses[:, 0, 3] *= sw
    poses[:, 1, 3] *= sh

    return poses

def rescale_poses(poses, H, W, probe_depth, probe_width):

    sh = probe_depth / H # real-world mm per pixel (y)
    sw = probe_width / W # real-world mm per pixel (x)

    poses[:, 0, 3] /= sw
    poses[:, 1, 3] /= sh

    return poses

def normalize_translations(poses):
    
    min_translation = np.min(poses[:, :3, 3], axis=0)
    max_translation = np.max(poses[:, :3, 3], axis=0)

    translation_range = max_translation - min_translation

    poses[:, :3, 3] = (poses[:, :3, 3] - min_translation) / translation_range

    return poses, translation_range, min_translation


        
def denormalize_translations(poses, translation_range, min_translation):

    poses[:, :3, 3] = poses[:, :3, 3] * translation_range + min_translation

    return poses

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate perturbed noise")
    parser.add_argument(
        "--data_dir", type=str, default="data", help="UltraNeRF dataset directory"
    )
    parser.add_argument(
        "--probe_depth", type=int, default=100, help="Depth of the probe"
    )
    parser.add_argument(
        "--probe_width", type=int, default=37, help="Width of the probe"
    )

    args = parser.parse_args()

    lie = Lie()
    pose = Pose()

    data_dir = args.data_dir


    # Set noisy position config
    tss = [0.0, 0.01, 0.05]
    rss = [0.0, 0.1, 0.5]
    perturb_ratios = [0.1, 0.5]
    repeats = 1

    # Create configs for each combination except ts and rs both 0.0
    configs = []
    for ts in tss:
        for rs in rss:
            if ts == 0.0 and rs == 0.0:
                continue
            for pr in perturb_ratios:
                for _ in range(repeats):
                    configs.append((ts, rs, pr))

    

    for ts, rs, pr in configs:
        rotation_strength = rs
        translation_strength = ts

        print(f"Rotation Strength: {rotation_strength}, Translation Strength: {translation_strength}, Perturb Ratio: {pr}")

        for i in range(repeats):

            # Load poses
            poses_path = os.path.join(data_dir, "poses.npy")
            poses_np = np.load(poses_path)[:, :3, :4]

            # Load Images (for visualization only)
            image_path = os.path.join(data_dir, "images", "1.png")
            image = np.array(Image.open(image_path).convert("L"))
            H, W = image.shape
            
            # Descale poses
            poses_np = descale_poses(poses_np, H, W, args.probe_depth, args.probe_width)

            # Normalize translations
            poses_np, translation_range, min_translation = normalize_translations(poses_np)

            # Convert to tensor
            poses = torch.tensor(poses_np, dtype=torch.float32)

            num_poses = poses.shape[0]

            # Generate noise
            se3_noise = torch.randn(num_poses, 6)
            se3_noise[:, :3] *= rotation_strength
            se3_noise[:, 3:] *= translation_strength

            # Convert SE(3) noise and apply it
            SE3_noise = lie.se3_to_SE3(se3_noise)
            noisy_poses = pose.compose([SE3_noise, poses])

            # Denormalize translations
            noisy_poses = denormalize_translations(noisy_poses, translation_range, min_translation)
            poses = denormalize_translations(poses, translation_range, min_translation)

            # Convert to numpy for visualization
            org_poses = poses.numpy()
            noisy_poses = noisy_poses.numpy()

            # Create ID
            id_pose = f"{rotation_strength}_{translation_strength}_{pr}_{i}"

            # Visualize the original and perturbed poses
            visualize_poses(org_poses, noisy_poses, sample_ratio=0.1, arrow_length=5.0, title=id_pose)
            plt.show()

            # Rescale poses
            org_poses = rescale_poses(org_poses, H, W, args.probe_depth, args.probe_width)
            noisy_poses = rescale_poses(noisy_poses, H, W, args.probe_depth, args.probe_width)

            # Select slices to apply the noise
            num_slices = int(pr * poses.shape[0])
            selected = np.random.permutation(poses.shape[0])[:num_slices]

            org_poses[selected] = noisy_poses[selected]
            noisy_poses = org_poses

            # Convert back to 4x4 format

            noisy_poses = np.concatenate([noisy_poses, np.repeat(np.array([0, 0, 0, 1]).reshape(1, 1, 4), repeats=noisy_poses.shape[0], axis=0)], axis=1)

            # Save noisy poses
            # os.makedirs(os.path.join(data_dir, "noisy_poses"), exist_ok=True)
            # noisy_poses_path = os.path.join(data_dir, "noisy_poses", f"{id_pose}.npy")
            # np.save(noisy_poses_path, noisy_poses)