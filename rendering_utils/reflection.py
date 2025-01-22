import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def compute_orientation(image):
    """
    Compute the gradient orientation of the image using Sobel filters and arctan2.

    Args:
    - image (torch.Tensor): Input image of shape (1, 1, H, W)

    Returns:
    - orientation (torch.Tensor): Gradient orientation of the image of shape (1, 1, H, W)
    """

    # Define Sobel filters
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
    ).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
    ).reshape(1, 1, 3, 3)

    # Send filters to the same device as the input image
    sobel_x = sobel_x.to(image.device)
    sobel_y = sobel_y.to(image.device)

    # Compute gradients#
    padded_image = F.pad(image, pad=(1, 1, 1, 1), mode='reflect')
    G_x = F.conv2d(padded_image, sobel_x)
    G_y = F.conv2d(padded_image, sobel_y)

    # Compute gradient orientation for each pixel
    orientation = torch.atan2(G_x, G_y)

    return orientation


def calculate_reflection_coefficient(acoustic_impedance_map):

    # plt.imshow(acoustic_impedance_map[0, 0].cpu().numpy(), cmap="gray")
    # plt.show()

    # # calculate the orientation map
    # orientation_map = compute_orientation(acoustic_impedance_map)
    # assert torch.isnan(orientation_map).any() == False

    # plt.imshow(orientation_map[0, 0].cpu().numpy(), cmap="gray")
    # plt.show()
    # plt.imshow(torch.cos(orientation_map)[0, 0].cpu().numpy(), cmap="gray")
    # plt.show()

    shifted_acoustic_impedance_map = torch.roll(acoustic_impedance_map, shifts=-1, dims=2)
    acoustic_impedance_map_wo_last_row = acoustic_impedance_map[:, :, :-1, :]
    shifted_acoustic_impedance_map = shifted_acoustic_impedance_map[:, :, :-1, :]
    
    # Difference between the two maps
    acoustic_impedance_map_diff = (shifted_acoustic_impedance_map - acoustic_impedance_map_wo_last_row) ** 2 / (shifted_acoustic_impedance_map + acoustic_impedance_map_wo_last_row + 1e-12)

    assert torch.isnan(acoustic_impedance_map_diff).any() == False

    acoustic_impedance_map_diff = torch.nn.functional.pad(
        acoustic_impedance_map_diff, (0, 0, 0, 1), "constant", 0.0)
    
    # plt.imshow(acoustic_impedance_map_diff[0, 0].cpu().numpy(), cmap="gray")
    # plt.show()

    reflection_coefficient = acoustic_impedance_map_diff
    # reflection_coefficient = acoustic_impedance_map_diff * torch.cos(orientation_map) ** 2
    
        
    # plt.imshow(reflection_coefficient[0, 0].cpu().numpy(), cmap="gray")
    # plt.show()

    return reflection_coefficient



if __name__ == '__main__':
    
    # create image with 10 x 10 pixels
    # 1.0 is the acoustic impedance of water
    # 1.5 is the acoustic impedance of the object

    image_ones = torch.ones((1, 1, 10, 10))
    image_zeros = torch.zeros((1, 1, 10, 10))
    diagonal = torch.diag(torch.ones(10))[None, None, ...]
    vertical = torch.zeros((1, 1, 10, 10))
    vertical[:, :, :, 5:] = 1.0


    image = torch.cat([image_ones, image_zeros, diagonal, vertical], dim=2)

    # calculate reflection coefficient
    reflection_coefficient = calculate_reflection_coefficient(image)
