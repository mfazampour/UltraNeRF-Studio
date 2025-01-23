import argparse
import os

import nibabel as nib
import numpy as np
from PIL import Image

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert poses to ImFusion")
    parser.add_argument(
        "--data_dir", type=str, default="data", help="UltraNeRF dataset directory"
    )

    args = parser.parse_args()

    data_dir = args.data_dir

    # Load poses
    poses_path = os.path.join(data_dir, "poses.npy")
    poses_np = np.load(poses_path)

    poses_np[:, 2, 3] = 0

    poses_np = poses_np.transpose(0, 2, 1).reshape(-1, 16).astype(np.float32)

    # Save as CSV
    poses_csv = os.path.join(data_dir, "poses.csv")
    np.savetxt(poses_csv, poses_np, delimiter=",")

    # Load Images
    images_path = os.path.join(data_dir, "images")
    images = []
    for i in range(poses_np.shape[0]):
        image_path = os.path.join(images_path, f"{i}.png")
        image = Image.open(image_path).convert("L")
        images.append(np.array(image).astype(np.uint8))

    images = np.stack(images, axis=0)

    # Swap dimensions according to ultrasound standarts
    images = images.transpose(2, 1, 0)

    # Save as NIFTI
    images_nii_path = os.path.join(data_dir, "images.nii")
    images_nii = nib.Nifti1Image(images, np.eye(4))
    nib.save(images_nii, images_nii_path)
