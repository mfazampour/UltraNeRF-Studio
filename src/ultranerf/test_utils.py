import matplotlib.pyplot as plt
import numpy as np


def visualize_poses(
    original_poses, perturbed_poses, sample_ratio=1.0, arrow_length=0.5, title=None
):
    """
    Visualizes sampled original and perturbed camera poses in 3D space with reoriented axes and uniform axis scaling.
    In this orientation:
        - X is the width (image horizontal axis).
        - Y is the height (image vertical axis).
        - Z corresponds to the slices or depth.

    Parameters:
        original_poses (np.ndarray): Array of shape (N, 3, 4), original poses.
        perturbed_poses (np.ndarray): Array of shape (N, 3, 4), perturbed poses.
        sample_ratio (float): Ratio of poses to sample for visualization (0.0 < sample_ratio <= 1.0).
        arrow_length (float): Length of the arrows representing camera directions.
    """
    assert 0.0 < sample_ratio <= 1.0, "Sample ratio must be in the range (0.0, 1.0]."

    # Determine the number of samples
    num_samples = int(original_poses.shape[0] * sample_ratio)
    indices = np.linspace(0, original_poses.shape[0] - 1, num_samples, dtype=int)

    # Sample the poses
    sampled_original_poses = original_poses[indices]
    sampled_perturbed_poses = perturbed_poses[indices]

    # Extract translations and directions
    original_translations = sampled_original_poses[:, :, 3]
    perturbed_translations = sampled_perturbed_poses[:, :, 3]
    original_directions = sampled_original_poses[:, :, 2]  # Forward direction (z-axis)
    perturbed_directions = sampled_perturbed_poses[:, :, 2]

    # Reorient axes
    original_reoriented = original_translations[
        :, [0, 2, 1]
    ]  # Map X, Y, Z to width, slices, height
    perturbed_reoriented = perturbed_translations[:, [0, 2, 1]]
    original_directions_reoriented = original_directions[:, [0, 2, 1]]
    perturbed_directions_reoriented = perturbed_directions[:, [0, 2, 1]]

    # Create the plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot translations
    ax.scatter(
        original_reoriented[:, 0],
        original_reoriented[:, 1],
        original_reoriented[:, 2],
        color="blue",
        label="Original Translations",
        s=50,
        depthshade=True,
    )
    ax.scatter(
        perturbed_reoriented[:, 0],
        perturbed_reoriented[:, 1],
        perturbed_reoriented[:, 2],
        color="red",
        label="Perturbed Translations",
        s=50,
        depthshade=True,
    )

    # Plot directions as quivers
    for i in range(num_samples):
        ax.quiver(
            original_reoriented[i, 0],
            original_reoriented[i, 1],
            original_reoriented[i, 2],
            original_directions_reoriented[i, 0],
            original_directions_reoriented[i, 1],
            original_directions_reoriented[i, 2],
            color="blue",
            length=arrow_length,
            normalize=True,
            alpha=0.7,
        )
        ax.quiver(
            perturbed_reoriented[i, 0],
            perturbed_reoriented[i, 1],
            perturbed_reoriented[i, 2],
            perturbed_directions_reoriented[i, 0],
            perturbed_directions_reoriented[i, 1],
            perturbed_directions_reoriented[i, 2],
            color="red",
            length=arrow_length,
            normalize=True,
            alpha=0.7,
        )

    # Set labels and legend
    ax.set_xlabel("Width (X)")
    ax.set_ylabel("Slices (Z)")
    ax.set_zlabel("Height (Y)")
    # ax.set_title(f"Reoriented Visualization of Sampled Poses (Sample Ratio: {sample_ratio})")
    ax.set_title(title)

    ax.legend(loc="upper right")
    ax.grid(True)

    # Uniform aspect ratio for all axes
    all_points = np.concatenate([original_reoriented, perturbed_reoriented], axis=0)
    max_range = (all_points.max(axis=0) - all_points.min(axis=0)).max() / 2.0
    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) / 2.0
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) / 2.0
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) / 2.0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
