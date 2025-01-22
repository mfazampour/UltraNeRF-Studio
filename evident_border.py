import numpy as np
from scipy.ndimage import binary_erosion, convolve, gaussian_filter
from skimage.morphology import skeletonize


def speckle_reducing_anisotropic_diffusion(
    image: np.ndarray,
    niter: int = 20,
    kappa: float = 0.1,
    lambda_: float = 0.2,
    eps: float = 1e-9,
    **kwargs
) -> np.ndarray:
    """
    Apply a simplified Speckle-Reducing Anisotropic Diffusion to an ultrasound image.

    Parameters
    ----------
    image : np.ndarray (float32)
        Input ultrasound image with intensity values typically between 0 and 1.
    niter : int
        Number of diffusion iterations.
    kappa : float
        Edge-threshold parameter controlling diffusion. Smaller values focus more on preserving edges.
    lambda_ : float
        Diffusion step parameter, must be small for stability.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray (float32)
        The filtered image after speckle-reducing anisotropic diffusion.
    """

    # Ensure input is float32
    if image.dtype != np.float32:
        raise ValueError("Input image must be float32")

    # Ensure input is between 0 and 1
    if np.min(image) < 0 or np.max(image) > 1:
        raise ValueError("Input image must be between 0 and 1")

    # Ensure input has 2 dimensions
    if len(image.shape) != 2:
        raise ValueError("Input image must have 2 dimensions")

    # Safety check: avoid log of zero
    image = np.maximum(image, eps)

    # Log transform to convert multiplicative noise to additive noise
    log_img = np.log(image + eps)

    # Precompute image shape
    rows, cols = log_img.shape

    # For convenience, define shifts (N, S, E, W) for neighbors
    def shift_up(x):
        return np.vstack([x[0:1, :], x[0 : rows - 1, :]])

    def shift_down(x):
        return np.vstack([x[1:rows, :], x[rows - 1 : rows, :]])

    def shift_left(x):
        return np.hstack([x[:, 0:1], x[:, 0 : cols - 1]])

    def shift_right(x):
        return np.hstack([x[:, 1:cols], x[:, cols - 1 : cols]])

    # Iterative diffusion
    diff = log_img.copy()
    for _ in range(niter):
        # Compute gradients
        diffN = shift_up(diff) - diff
        diffS = shift_down(diff) - diff
        diffE = shift_right(diff) - diff
        diffW = shift_left(diff) - diff

        # Compute local statistics to mimic a SRAD-like approach:
        # Speckle-reducing anisotropic diffusion considers local intensity variance and mean.
        # Here, we use a simplified approach: local intensity and a coefficient that reduces diffusion in edges.

        # Local mean intensity (approx.)
        local_mean = (
            shift_up(diff)
            + shift_down(diff)
            + shift_left(diff)
            + shift_right(diff)
            + 4 * diff
        ) / 8.0

        # Local variance (approx.)
        local_var = (
            (shift_up(diff) - local_mean) ** 2
            + (shift_down(diff) - local_mean) ** 2
            + (shift_left(diff) - local_mean) ** 2
            + (shift_right(diff) - local_mean) ** 2
            + (diff - local_mean) ** 2
        ) / 5.0

        # Coefficient of variation
        # CV = sqrt(var)/mean. In log-space, mean should be close to local_mean.
        # Add eps for stability.
        cv = np.sqrt(local_var + eps) / (np.abs(local_mean) + eps)

        # Diffusion coefficient:
        # Lower diffusion where CV is large (edges or strong features).
        # This is a heuristic approach. For a proper SRAD, see the original formulation.
        c = np.exp(-(cv**2) / (kappa**2))

        # Update the image using diffusion
        # Weighted sum of neighbors controlled by the diffusion coefficient
        diff = diff + lambda_ * (c * diffN + c * diffS + c * diffE + c * diffW)

    # Exponentiate back to original domain
    filtered = np.exp(diff)

    # Normalize to the original intensity range approximately
    filtered = np.clip(filtered, 0, 1)

    return filtered.astype(np.float32)


def detect_edges(
    smoothed_image: np.ndarray, threshold_value: float = 0.1
) -> np.ndarray:
    """
    Apply Sobel edge detection and thresholding to a pre-smoothed ultrasound image.

    Parameters
    ----------
    smoothed_image : np.ndarray (float32)
        The pre-smoothed ultrasound image. Ideally scaled to [0,1] or a similar consistent range.
    threshold_value : float
        Threshold for binarizing the gradient magnitude. Adjust based on image intensity scaling.

    Returns
    -------
    np.ndarray (float32)
        Binary image representing detected edges.
    """
    # Ensure input is float32
    img = smoothed_image.astype(np.float32, copy=False)
    img = np.clip(img, 0, np.finfo(np.float32).max)

    # Define Sobel kernels for X and Y directions
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    # Compute gradients in X and Y directions
    Gx = convolve(img, sobel_x, mode="reflect")
    Gy = convolve(img, sobel_y, mode="reflect")

    # Compute gradient magnitude
    grad_mag = np.sqrt(Gx**2 + Gy**2)

    # Threshold the gradient magnitude to get binary edges
    edges_binary = (grad_mag > threshold_value).astype(np.float32)

    return edges_binary


def get_borders(image: np.ndarray, **kwargs) -> np.ndarray:

    filtered = speckle_reducing_anisotropic_diffusion(image, **kwargs)
    smoothed = gaussian_filter(filtered, sigma=2)
    borders = detect_edges(smoothed)
    eroded_borders = binary_erosion(borders, iterations=2)
    border_skeleton = skeletonize(eroded_borders)

    return border_skeleton


def main():

    import os

    import matplotlib.pyplot as plt
    from PIL import Image

    test_image = Image.open(
        os.path.join("data", "synthetic_testing", "l2", "images", "43.png")
    )
    test_image = np.array(test_image).astype(np.float32) / 255.0

    niter = 10
    kappa = 0.1
    lambda_ = 0.1
    eps = 1e-8

    # filtered = speckle_reducing_anisotropic_diffusion(
    #     test_image, niter, kappa, lambda_, eps
    # )
    # smoothed = gaussian_filter(filtered, sigma=2)
    # borders = detect_edges(smoothed)
    # eroded_borders = binary_erosion(borders, iterations=2)
    # border_skeleton = skeletonize(eroded_borders)

    # plt.figure(figsize=(20, 5))
    # plt.subplot(1, 6, 1)
    # plt.imshow(test_image)
    # plt.title("Original image")
    # plt.subplot(1, 6, 2)
    # plt.imshow(filtered)
    # plt.title(f"Speckle-reducing anisotropic diffusion")
    # plt.subplot(1, 6, 3)
    # plt.imshow(smoothed)
    # plt.title("Gaussian blur")
    # plt.subplot(1, 6, 4)
    # plt.imshow(borders)
    # plt.title("Detected edges")
    # plt.subplot(1, 6, 5)
    # plt.imshow(eroded_borders)
    # plt.title("Eroded edges")
    # plt.subplot(1, 6, 6)
    # plt.imshow(border_skeleton)
    # plt.title("Skeleton")

    borders = get_borders(
        test_image, niter=niter, kappa=kappa, lambda_=lambda_, eps=eps
    )

    plt.figure(figsize=(8, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(test_image)
    plt.title("Original image")
    plt.subplot(1, 2, 2)
    plt.imshow(borders)
    plt.title("Borders")

    plt.show()


if __name__ == "__main__":
    main()
