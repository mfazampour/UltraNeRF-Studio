import numpy as np


def process_pointcloud(points):
    """
    Process point cloud to remove duplicates, keeping points with value 1 when duplicates exist.

    Parameters:
    -----------
    points : numpy.ndarray
        Array of shape (n, 4) where each row is [x, y, z, value]
        value is either 0 or 1

    Returns:
    --------
    numpy.ndarray
        Processed point cloud with duplicates removed
    dict
        Statistics about the processing
    """
    # Create a structured array for unique identification of spatial coordinates
    coords = points[:, :3]
    values = points[:, 3]

    # Find unique coordinates and their indices
    _, unique_indices, inverse_indices = np.unique(coords, axis=0, return_index=True, return_inverse=True)

    # Count occurrences of each unique coordinate
    unique_counts = np.bincount(inverse_indices)

    # Find coordinates that appear multiple times
    duplicate_coords = unique_counts > 1
    duplicate_coord_indices = np.where(duplicate_coords)[0]

    # Initialize array to store final points
    processed_points = points[unique_indices].copy()

    # For each set of duplicate coordinates
    for dup_idx in duplicate_coord_indices:
        # Find all points with these coordinates
        dup_points_mask = (inverse_indices == dup_idx)
        dup_points = points[dup_points_mask]

        # If any of the duplicate points has value 1, use it
        if 1 in dup_points[:, 3]:
            processed_points[dup_idx, 3] = 1

    # Gather statistics
    stats = {
        "original_points": len(points),
        "processed_points": len(processed_points),
        "duplicate_sets": len(duplicate_coord_indices),
        "points_removed": len(points) - len(processed_points)
    }

    return processed_points, stats


# Example usage
if __name__ == "__main__":
    # Create sample point cloud with duplicates
    pts = np.load("./data/processed22_51_2002/poses_labels.npy").reshape(-1, 3)
    occ = np.load("./data/processed22_51_2002/labels.npy").reshape(-1, 1)
    occ[occ==255] = 1
    "loaded"
    sample_points = np.concatenate([pts, occ], axis=-1)
    # Process the point cloud
    "sampled"
    processed_cloud, stats = process_pointcloud(sample_points)
    "processed"
    np.save("processed22_51_occ.npy", processed_cloud)

    # Print results
    print("Original point cloud:")
    print(sample_points)
    print("\nProcessed point cloud:")
    print(processed_cloud)
    print("\nProcessing statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")