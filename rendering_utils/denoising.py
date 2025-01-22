import imageio
import matplotlib.pyplot as plt
import ptwt
import pywt
import torch


def soft_threshold(data, value):
    magnitude = torch.abs(data)

    thresholded = 1 - (value/(magnitude+1e-12))
    alpha = 1
    thresholded = torch.sigmoid(alpha * thresholded) * thresholded

    thresholded = data * thresholded
    return thresholded



def wavelet_decomposition(image, wavelet_name='db2', level=2):
    # Perform 2-level wavelet decomposition
    wavelet = pywt.Wavelet(wavelet_name)
    coeffs = ptwt.wavedec2(image, wavelet, level=level)
    new_coeffs = coeffs.copy()

    # Universal Threshold in wavelet domain
    # Estimate noise from the first level detail coefficients
    # Threshold detail coefficients to remove noise
    # noise_estimate = torch.median(torch.abs(cV1)) / 0.6745

    # image_size = torch.tensor(image.shape[2] * image.shape[3], dtype=torch.float32)
    # threshold = noise_estimate * torch.sqrt(2 * torch.log2(image_size))
    # cH1 = torch.threshold(cH1, threshold.item())
    # cV1 = torch.threshold(cV1, threshold)
    # cD1 = torch.threshold(cD1, threshold)

    level_coeffs = new_coeffs[-level]
    # Concatenate the absolute values of all coefficient tensors at this level
    abs_coeffs = torch.cat([torch.abs(c) for c in level_coeffs])

    # Calculate the median of these absolute values
    median_abs_coeff = torch.median(abs_coeffs)

    threshold = (median_abs_coeff / 0.6745) * (2 * torch.log(torch.tensor(image.numel(), dtype=torch.float32)))

    for i in range(1, len(coeffs)):
        new_coeffs[i] = [soft_threshold(coeff, value=threshold) for coeff in new_coeffs[i]]
    denoised_image = ptwt.waverec2(coeffs, wavelet)

    assert denoised_image.shape == image.shape


    return image-denoised_image



if __name__ == '__main__':
    
    # load image
    image_path = "data/synthetic_testing/r2/images/0.png"
    image = torch.tensor(imageio.imread(image_path), dtype=torch.float32)
    image = image[None, None, ...]
    noise = wavelet_decomposition(image)

    plt.imshow(image[0, 0].cpu().numpy(), cmap="gray")
    plt.show()
    plt.imshow(noise[0, 0].cpu().numpy(), cmap="gray")
    plt.show()

