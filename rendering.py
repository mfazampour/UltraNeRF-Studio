import matplotlib.pyplot as plt
import numpy as np
import ptwt
import pywt
import torch
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

from rendering_utils.denoising import wavelet_decomposition
from rendering_utils.reflection import calculate_reflection_coefficient
from nerf_utils import get_rays_us_linear, batchify_rays

def cumsum_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    r"""Mimick functionality of tf.math.cumsum(..., exclusive=True), as it isn't available in PyTorch.

    Args:
      tensor (torch.Tensor): Tensor whose cumsum (cumulative product, see `torch.cumsum`) along dim=-1
        is to be computed.

    Returns:
      cumsum (torch.Tensor): cumsum of Tensor along dim=-1, mimiciking the functionality of
        tf.math.cumsum(..., exclusive=True) (see `tf.math.cumsum` for details).
    """
    # TESTED
    # Only works for the last dimension (dim=-1) -> Why? 
    dim = -1
    # Compute regular cumsum first (this is equivalent to `tf.math.cumsum(..., exclusive=False)`).
    cumsum = torch.cumsum(tensor, dim)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumsum = torch.roll(cumsum, 1, dim)
    # Replace the first element by "0" as this is what tf.cumsum(..., exclusive=True) does.
    cumsum[..., 0] = 0.

    return cumsum


def gaussian_kernel(size: int, mean: float, std: float):
    delta_t = 1
    x_cos = np.array(list(range(-size, size + 1)), dtype=np.float32)
    x_cos *= delta_t

    d1 = torch.distributions.Normal(mean, std * 3)
    d2 = torch.distributions.Normal(mean, std)
    vals_x = d1.log_prob(
        torch.arange(-size, size + 1, dtype=torch.float32) * delta_t
    ).exp()
    vals_y = d2.log_prob(
        torch.arange(-size, size + 1, dtype=torch.float32) * delta_t
    ).exp()

    gauss_kernel = torch.einsum("i,j->ij", vals_x, vals_y)

    return gauss_kernel / torch.sum(gauss_kernel).reshape(1, 1)


def gaussian_kernel_3d(size: int, mean: float, std: float):
    delta_t = 1
    x_cos = np.array(list(range(-size, size + 1)), dtype=np.float32)
    x_cos *= delta_t

    d1 = torch.distributions.Normal(mean, std * 3)
    d2 = torch.distributions.Normal(mean, std)

    vals_x = d1.log_prob(
        torch.arange(-size, size + 1, dtype=torch.float32) * delta_t
    ).exp()
    vals_y = d2.log_prob(
        torch.arange(-size, size + 1, dtype=torch.float32) * delta_t
    ).exp()
    vals_z = d2.log_prob(
        torch.arange(-size, size + 1, dtype=torch.float32) * delta_t
    ).exp()

    # Create 3D Gaussian kernel by taking the outer product
    gauss_kernel_3d = torch.einsum("i,j,k->ijk", vals_x, vals_y, vals_z)

    # Normalize the kernel so that the sum of all elements equals 1
    gauss_kernel_3d = gauss_kernel_3d / torch.sum(gauss_kernel_3d)

    return gauss_kernel_3d


g_kernel = gaussian_kernel(3, 0.0, 1.0)
g_kernel = torch.tensor(g_kernel, dtype=torch.float32).to(device="cuda")

size = 3
mean = 0.0
std = 1.0
g_kernel3D = gaussian_kernel_3d(size, mean, std)
g_kernel3D = torch.tensor(g_kernel3D, dtype=torch.float32).to(device="cuda")

def rendering(raw_value, FREQUENCY = 8e6, log_compression_scale = 10):

    # return {"intensity_map": raw_value[None, None, ..., 0].permute(0, 1, 3, 2)}
    debug = True

    # adapt rendering to reflect the new input
    raw_value = raw_value[None, None, ...]
    raw_value = raw_value.permute(0, 1, 3, 2, 4)

    # ---------------------------  Split reflection and backscattering --------------------------- #
    acoustic_impedance_map = raw_value[..., 0]
    attenuation_map = raw_value[..., 1]

    backscattering_map = wavelet_decomposition(acoustic_impedance_map)

    # # differntiable thresholding
    # acoustic_impedance_map_diff = torch.abs(acoustic_impedance_map[..., -1:, :] - acoustic_impedance_map[..., :-1, :])
    # acoustic_impedance_map_diff = torch.cat([acoustic_impedance_map_diff, acoustic_impedance_map_diff[..., -1:, :]], dim=-2)

    # x = 0.1
    # backscattering_map = acoustic_impedance_map_diff * torch.sigmoid(x - torch.abs(acoustic_impedance_map_diff))

    

    # ----------------------------  Rendering ---------------------------- #

    batch_size, C,  W, H = attenuation_map.shape

    t_vals = torch.linspace(0.0, 1.0, H).to(device="cuda")
    z_vals = t_vals.expand(batch_size, W, -1)  # * 2
    # calculate the distance between the points we want to sample from
    dists = torch.abs(z_vals[..., :-1] - z_vals[..., 1:])  # dists.shape=(B, W, H-1, 1)
    dists = torch.cat([dists, dists[:, :, -1, None]], dim=-1)  # dists.shape=(B, W, H)


    # ---------------------------  Attenuation --------------------------- #

    # ATTENUATION
    # attenuation_coeff = torch.abs(attenuation_map)
    # If using a simgoid at the end of the network we do not need this
    attenuation_coeff = attenuation_map
    log_attenuation = -attenuation_coeff * dists
    log_attenuation_total = torch.cumsum(log_attenuation, dim=2)
    attenuation_total = torch.exp(log_attenuation_total)
    # DIFF: Changed cumprod with log cumsum to avoid numerical instability


    # ---------------------------  Reflection --------------------------- #
    #assert acoustic impedance map is of type float
    reflection_coeff = calculate_reflection_coefficient(acoustic_impedance_map)
    assert torch.isnan(reflection_coeff).any() == False
    assert reflection_coeff.dtype == torch.float32
    reflection_transmission = 1.0 - reflection_coeff
    log_reflection_total = torch.log(reflection_transmission + 1e-12)
    log_reflection_total = torch.cumsum(log_reflection_total, dim=2)
    reflection_total = torch.exp(log_reflection_total)
    # DIFF: Changed cumprod with log cumsum to avoid numerical instability

    # BACKSCATTERING
    psf_scatter = torch.nn.functional.conv2d(
        input=backscattering_map,
        weight=g_kernel[None, None, :, :],
        stride=1,
        padding="same",
    )

    # Compute remaining intensity at a point n
    remaining_intensity = attenuation_total * reflection_total

    # Compute backscattering part of the final echo
    b = remaining_intensity * psf_scatter

    # Compute reflection part of the final echo
    r = remaining_intensity * reflection_coeff

    # Compute the final echo
    # amplification_constant = log_compression_scale
    # alpha_amplification = lambda x: torch.log(
    #     torch.tensor(1.0) + amplification_constant * x
    # )
    intensity_map_pre_amp = (b + r)
    # intensity_map = alpha_amplification(intensity_map_pre_amp)

    ret = {
        "intensity_map_pre_amplified": intensity_map_pre_amp,
        "intensity_map": intensity_map_pre_amp,
        "confidence_maps": remaining_intensity,
        "acoustic_impedance_map": acoustic_impedance_map,
        "attenuation_map": attenuation_map,
        "attenuation_coeff": attenuation_coeff,
        "reflection_coeff": reflection_coeff,
        "attenuation_total": attenuation_total,
        "reflection_total": reflection_total,
        "backscattering_map": backscattering_map,
        "b": b,
        "r": r,
    }

    # Plotting each tensor in the output dictionary
    # for key, value in ret.items():
    #     if value is not None:  # Only plot if the value exists
    #         # Assuming that the first dimension is the batch size
    #         # and that you're interested in the first item in the batch
    #         plot_fig(key, value[0])

    return ret

def render_method_3(raw):
    def raw2attention(raw, dists):
        return torch.exp(-raw * dists)

    raw = raw[None, None, ...]
    raw = raw.permute(0, 1, 3, 2, 4)

    batch_size, C,  W, H, maps = raw.shape
    
    t_vals = torch.linspace(0.0, 1.0, H).to(device="cuda")
    z_vals = t_vals.expand(batch_size, W, -1)  # * 2
    # calculate the distance between the points we want to sample from

    # Compute 'distance' between each integration time along a ray.
    dists = torch.abs(z_vals[..., :-1, None] - z_vals[..., 1:, None])
    dists = torch.squeeze(dists)
    dists = torch.cat([dists, dists[:, -1, None]], dim=-1)

    # ATTENUATION
    attenuation_coeff = torch.abs(raw[..., 0])
    attenuation = raw2attention(attenuation_coeff, dists)
    attenuation = attenuation.permute(0, 1, 3, 2)
    log_attenuation = torch.log(attenuation + 1e-12)
    log_attenuation_total = cumsum_exclusive(log_attenuation)
    attenuation_total = torch.exp(log_attenuation_total)
    attenuation_total = attenuation_total.permute(0, 1, 3, 2)

    # REFLECTION
    # reflection_coeff = torch.zeros_like(raw[..., 0])

    reflection_coeff = torch.sigmoid(raw[..., 1])
    reflection_transmission = 1. - reflection_coeff
    # reflection_transmission = raw2reflection(reflection_coeff)
    reflection_transmission = reflection_transmission.permute(0, 1, 3, 2)
    log_reflection_transmission = torch.log(reflection_transmission + 1e-12)
    log_reflection_total = cumsum_exclusive(log_reflection_transmission)
    reflection_total = torch.exp(log_reflection_total)
    reflection_total = reflection_total.permute(0, 1, 3, 2)

    # BACKSCATTERING
    # density_coeff = torch.sigmoid(raw[..., 2])
    density_coeff = torch.ones_like(reflection_coeff) * 0.75
    scatter_density_distribution = RelaxedBernoulli(temperature=0.1, probs=density_coeff)
    scatterers_density = scatter_density_distribution.sample()
    amplitude = torch.sigmoid(raw[..., 2])
    scatterers_map = scatterers_density * amplitude

    psf_scatter = F.conv2d(scatterers_map, g_kernel[None, None, ...], stride=1, padding="same")
    # psf_scatter = torch.squeeze(psf_scatter)

    # Compute remaining intensity at a point n
    confidence_maps = attenuation_total * reflection_total

    # Compute backscattering and reflection parts of the final echo
    b = confidence_maps * psf_scatter
    r = confidence_maps * reflection_coeff

    # Compute the final echo
    amplification_constant = torch.tensor(3.14)
    alpha_amplification = lambda x: torch.log(torch.tensor(1.) + amplification_constant * x) * torch.log(torch.tensor(1.) + amplification_constant)
    r_amplified = alpha_amplification(r)
    intensity_map = b + r_amplified

    return {
        'intensity_map': intensity_map,
        'attenuation_coeff': attenuation_coeff,
        'reflection_coeff': reflection_coeff,
        'attenuation_total': attenuation_total,
        'reflection_total': reflection_total,
        'scatterers_density': scatterers_density,
        'scatterers_density_coeff': density_coeff,
        'scatter_amplitude': amplitude,
        'b': b,
        'r': r,
        'r_amplified': r_amplified,
        "confidence_maps": confidence_maps
    }


def render_method_ultra_nerf(raw):
    def raw2attention(raw, dists):
        return torch.exp(-raw * dists)


    raw = raw[None, None, ...]
    raw = raw.permute(0, 1, 3, 2, 4)

    batch_size, C, W, H, maps = raw.shape

    t_vals = torch.linspace(0.0, 1.0, H).to(device="cuda")
    z_vals = t_vals.expand(batch_size, W, -1)  # * 2
    # calculate the distance between the points we want to sample from

    # Compute 'distance' between each integration time along a ray.
    dists = torch.abs(z_vals[..., :-1, None] - z_vals[..., 1:, None])
    dists = torch.squeeze(dists)
    dists = torch.cat([dists, dists[:, -1, None]], dim=-1)

    # ATTENUATION
    attenuation_coeff = torch.abs(raw[..., 0])
    attenuation = raw2attention(attenuation_coeff, dists)
    attenuation = attenuation.permute(0, 1, 3, 2)
    log_attenuation = torch.log(attenuation + 1e-12)
    log_attenuation_total = cumsum_exclusive(log_attenuation)
    attenuation_total = torch.exp(log_attenuation_total)
    attenuation_total = attenuation_total.permute(0, 1, 3, 2)

    # REFLECTION
    # reflection_coeff = torch.zeros_like(raw[..., 0])
    prob_border = torch.sigmoid(raw[..., 2])
    b_prob_dist = RelaxedBernoulli(temperature=0.1, probs=prob_border)
    b_prob = b_prob_dist.sample()
    reflection_coeff = torch.sigmoid(raw[..., 1])
    reflection_transmission = 1. - reflection_coeff * b_prob
    reflection_transmission = reflection_transmission.permute(0, 1, 3, 2)
    log_reflection_transmission = torch.log(reflection_transmission + 1e-12)
    log_reflection_total = cumsum_exclusive(log_reflection_transmission)
    reflection_total = torch.exp(log_reflection_total)
    reflection_total = reflection_total.permute(0, 1, 3, 2)

    # BACKSCATTERING
    density_coeff = torch.sigmoid(raw[..., 3])
    # density_coeff = torch.ones_like(reflection_coeff) * 0.75
    scatter_density_distribution = RelaxedBernoulli(temperature=0.1, probs=density_coeff)
    scatterers_density = scatter_density_distribution.sample()
    amplitude = torch.sigmoid(raw[..., 4])
    scatterers_map = scatterers_density * amplitude

    psf_scatter = F.conv2d(scatterers_map, g_kernel[None, None, ...], stride=1, padding="same")
    # psf_scatter = torch.squeeze(psf_scatter)

    # Compute remaining intensity at a point n
    confidence_maps = attenuation_total * reflection_total

    # Compute backscattering and reflection parts of the final echo
    b = confidence_maps * psf_scatter
    r = confidence_maps * reflection_coeff

    intensity_map = b + r

    return {
        'intensity_map': intensity_map,
        'attenuation_coeff': attenuation_coeff,
        'reflection_coeff': reflection_coeff,
        'attenuation_total': attenuation_total,
        'reflection_total': reflection_total,
        'scatterers_density': scatterers_density,
        'scatterers_density_coeff': density_coeff,
        'scatter_amplitude': amplitude,
        'b': b,
        'r': r,
        "confidence_maps": confidence_maps
    }


def render_method_cos_theta(raw, d):
    def raw2attention(raw, dists):
        return torch.exp(-raw * dists)

    def raw2reflection(raw):
        return torch.exp(-raw)

    raw = raw[None, None, ...]
    raw = raw.permute(0, 1, 3, 2, 4)

    batch_size, C, W, H, maps = raw.shape

    t_vals = torch.linspace(0.0, 1.0, H).to(device="cuda")
    z_vals = t_vals.expand(batch_size, W, -1)  # * 2
    # calculate the distance between the points we want to sample from

    # Compute 'distance' between each integration time along a ray.
    dists = torch.abs(z_vals[..., :-1, None] - z_vals[..., 1:, None])
    dists = torch.squeeze(dists)
    dists = torch.cat([dists, dists[:, -1, None]], dim=-1)

    # ATTENUATION
    attenuation_coeff = torch.abs(raw[..., 0])
    attenuation = raw2attention(attenuation_coeff, dists)
    log_attenuation = torch.log(attenuation + 1e-12)
    log_attenuation_total = torch.cumsum(log_attenuation, dim=2)
    attenuation_total = torch.exp(log_attenuation_total)
    dx, dy, dz = torch.gradient(attenuation)
    normal_magnitude = torch.sqrt(dx**2 + dy**2 + dz**2)
    n = torch.stack([-dx, -dy, -dz], dim=-1)

    dot_product = n[..., 0] * d[0] + n[..., 1] * d[1] + n[..., 2] * d[2]
    # Calculate the magnitude of the light direction vector (|L|), which is 1 in this case
    d_magnitude = torch.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)

    # Cosine of the incidence angle
    cos_theta = dot_product / (normal_magnitude * d_magnitude)

    # REFLECTION
    # reflection_coeff = torch.zeros_like(raw[..., 0])

    reflection_coeff = torch.clamp_max(torch.sigmoid(raw[..., 1]), 0.34)
    # reflection_transmission = 1. - reflection_coeff
    reflection_transmission = raw2reflection(reflection_coeff)
    log_reflection_transmission = torch.log(reflection_transmission + 1e-12)
    log_reflection_total = torch.cumsum(log_reflection_transmission, dim=2)
    reflection_total = torch.exp(log_reflection_total)

    # BACKSCATTERING
    # density_coeff = torch.sigmoid(raw[..., 2])
    density_coeff = torch.ones_like(reflection_coeff) * 0.75
    scatter_density_distribution = RelaxedBernoulli(temperature=0.1, probs=density_coeff)
    scatterers_density = scatter_density_distribution.sample()
    amplitude = torch.sigmoid(raw[..., 2])
    scatterers_map = scatterers_density * amplitude

    psf_scatter = F.conv2d(scatterers_map, g_kernel[None, None, ...], stride=1, padding="same")
    # psf_scatter = torch.squeeze(psf_scatter)

    # Compute remaining intensity at a point n
    confidence_maps = attenuation_total * reflection_total

    # Compute backscattering and reflection parts of the final echo
    b = confidence_maps * psf_scatter
    r = confidence_maps * reflection_coeff

    # Compute the final echo
    amplification_constant = torch.tensor(np.pi)
    alpha_amplification = lambda x: torch.log(torch.tensor(1.) + amplification_constant * x) * torch.log(
        torch.tensor(1.) + amplification_constant)
    r_amplified = alpha_amplification(r)
    r_cos = cos_theta * r_amplified
    intensity_map = b + r_cos

    return {
        'intensity_map': intensity_map,
        'attenuation_coeff': attenuation_coeff,
        'reflection_coeff': reflection_coeff,
        'attenuation_total': attenuation_total,
        'reflection_total': reflection_total,
        'scatterers_density': scatterers_density,
        'scatterers_density_coeff': density_coeff,
        'scatter_amplitude': amplitude,
        'b': b,
        'r': r,
        'r_amplified': r_amplified,
        "confidence_maps": confidence_maps
    }

def render_method_3D(raw):
    def raw2attention(raw, dists):
        return torch.exp(-raw * dists)

    W, H, maps = raw.shape
    def raw2reflection(raw):
        return torch.exp(-raw)
    # print(raw.shape)
    # print(f"after reshape {raw.shape}")
    raw = raw[None, None,...]
    raw = raw.permute(0, 1, 3, 2, 4)

    batch_size, C, W, H, maps = raw.shape

    t_vals = torch.linspace(0.0, 1.0, H).to(device="cuda")
    z_vals = t_vals.expand(batch_size, W, -1)  # * 2
    # calculate the distance between the points we want to sample from

    # Compute 'distance' between each integration time along a ray.
    dists = torch.abs(z_vals[..., :-1, None] - z_vals[..., 1:, None])
    dists = torch.squeeze(dists)
    dists = torch.cat([dists, dists[:, -1, None]], dim=-1)

    # ATTENUATION
    attenuation_coeff = torch.abs(raw[..., 0])
    attenuation = raw2attention(attenuation_coeff, dists)
    attenuation_total = torch.cumprod(attenuation, dim=2)

    # REFLECTION
    # reflection_coeff = torch.zeros_like(raw[..., 0])

    reflection_coeff = torch.clamp_max(torch.sigmoid(raw[..., 1]), 0.34)
    reflection_transmission = raw2reflection(reflection_coeff)
    reflection_total = torch.cumprod(reflection_transmission, dim=2)

    # BACKSCATTERING
    density_coeff = torch.ones_like(reflection_coeff) * 0.75
    scatter_density_distribution = RelaxedBernoulli(temperature=0.1, probs=density_coeff)
    scatterers_density = scatter_density_distribution.sample()
    amplitude = torch.sigmoid(raw[..., 2])
    scatterers_map = scatterers_density * amplitude

    confidence_maps = attenuation_total * reflection_total

    # Compute backscattering and reflection parts of the final echo
    b = confidence_maps * scatterers_map
    r = confidence_maps * reflection_coeff

    # Compute the final echo
    amplification_constant = torch.tensor(3.14)
    alpha_amplification = lambda x: torch.log(torch.tensor(1.) + amplification_constant * x) * torch.log(
        torch.tensor(1.) + amplification_constant)
    r_amplified = alpha_amplification(r)


    intensity_map = b + r_amplified

    intensity_map = torch.reshape(intensity_map, (7, W, H // 7))[None, ...]
    intensity_map = F.conv3d(intensity_map, g_kernel3D[None, None, ...], stride=[7, 1, 1], padding=[0, 3, 3])

    return {
        'intensity_map': intensity_map,
        'attenuation_coeff': attenuation_coeff,
        'reflection_coeff': reflection_coeff,
        'attenuation_total': attenuation_total,
        'reflection_total': reflection_total,
        'scatterers_density': scatterers_density,
        'scatterers_density_coeff': density_coeff,
        'scatter_amplitude': amplitude,
        'b': b,
        'r': r,
        'r_amplified': r_amplified,
        "confidence_maps": confidence_maps
    }

# Note: 'g_kernel' needs to be defined or passed to the function as a PyTorch tensor.

def render_us(H, W, sw, sh, chunk=1024 * 32, rays=None, c2w=None, near=0., far=55. * 0.001, **kwargs):
    """Render rays."""

    # assert rays is not None or c2w is not None
    assert rays is not None or c2w is not None

    rays_o = None
    rays_d = None
    
    if c2w is not None:
        # Special case to render full image
        for c in c2w:
            if rays_o is None:
                rays_o, rays_d = get_rays_us_linear(H, W, sw, sh, c)
            else:
                o, d = get_rays_us_linear(H, W, sw, sh, c)
                rays_o = torch.concatenate((rays_o, o))
                rays_d = torch.concatenate((rays_d, d))
    else:
        # Use provided ray batch
        rays_o, rays_d = rays
    sh = rays_d.shape  # [..., 3]

    # Create ray batch
    rays_o = rays_o.reshape(-1, 3).float()
    rays_d = rays_d.reshape(-1, 3).float()
    near = near * torch.ones_like(rays_d[..., :1])
    far = far * torch.ones_like(rays_d[..., :1])

    # (ray origin, ray direction, min dist, max dist) for each ray


    rays = torch.cat([rays_o, rays_d, near, far], dim=-1)
    # Render and reshape
    all_ret = batchify_rays(rays, chunk=chunk, **kwargs)
    # for k in all_ret:
    #     k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
    #     all_ret[k] = all_ret[k].reshape(k_sh)
    return all_ret
