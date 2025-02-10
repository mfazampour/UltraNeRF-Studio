import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

from rendering_utils.denoising import wavelet_decomposition
from rendering_utils.reflection import calculate_reflection_coefficient


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
    cumsum[..., 0] = 0.0

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


g_kernel = gaussian_kernel(3, 0.0, 1.0).float().to(device="cuda")

size = 3
mean = 0.0
std = 1.0
g_kernel3D = gaussian_kernel_3d(size, mean, std).float().to(device="cuda")


def rendering(raw_value, FREQUENCY=8e6, log_compression_scale=10):

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

    batch_size, C, W, H = attenuation_map.shape

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
    # assert acoustic impedance map is of type float
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
    intensity_map_pre_amp = b + r
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

    reflection_coeff = torch.sigmoid(raw[..., 1])
    reflection_transmission = 1.0 - reflection_coeff
    # reflection_transmission = raw2reflection(reflection_coeff)
    reflection_transmission = reflection_transmission.permute(0, 1, 3, 2)
    log_reflection_transmission = torch.log(reflection_transmission + 1e-12)
    log_reflection_total = cumsum_exclusive(log_reflection_transmission)
    reflection_total = torch.exp(log_reflection_total)
    reflection_total = reflection_total.permute(0, 1, 3, 2)

    # BACKSCATTERING
    # density_coeff = torch.sigmoid(raw[..., 2])
    density_coeff = torch.ones_like(reflection_coeff) * 0.75
    scatter_density_distribution = RelaxedBernoulli(
        temperature=0.1, probs=density_coeff
    )
    scatterers_density = scatter_density_distribution.sample()
    amplitude = torch.sigmoid(raw[..., 2])
    scatterers_map = scatterers_density * amplitude

    psf_scatter = F.conv2d(
        scatterers_map, g_kernel[None, None, ...], stride=1, padding="same"
    )
    # psf_scatter = torch.squeeze(psf_scatter)

    # Compute remaining intensity at a point n
    confidence_maps = attenuation_total * reflection_total

    # Compute backscattering and reflection parts of the final echo
    b = confidence_maps * psf_scatter
    r = confidence_maps * reflection_coeff

    # Compute the final echo
    amplification_constant = torch.tensor(np.pi)
    alpha_amplification = lambda x: torch.log(
        torch.tensor(1.0) + amplification_constant * x
    ) * torch.log(torch.tensor(1.0) + amplification_constant)
    r_amplified = alpha_amplification(r)
    intensity_map = b + r_amplified

    return {
        "intensity_map": intensity_map,
        "attenuation_coeff": attenuation_coeff,
        "reflection_coeff": reflection_coeff,
        "attenuation_total": attenuation_total,
        "reflection_total": reflection_total,
        "scatterers_density": scatterers_density,
        "scatterers_density_coeff": density_coeff,
        "scatter_amplitude": amplitude,
        "b": b,
        "r": r,
        "r_amplified": r_amplified,
        "confidence_maps": confidence_maps,
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
    reflection_transmission = 1.0 - reflection_coeff * b_prob
    reflection_transmission = reflection_transmission.permute(0, 1, 3, 2)
    log_reflection_transmission = torch.log(reflection_transmission + 1e-12)
    log_reflection_total = cumsum_exclusive(log_reflection_transmission)
    reflection_total = torch.exp(log_reflection_total)
    reflection_total = reflection_total.permute(0, 1, 3, 2)

    # BACKSCATTERING
    density_coeff = torch.sigmoid(raw[..., 3])
    # density_coeff = torch.ones_like(reflection_coeff) * 0.75
    scatter_density_distribution = RelaxedBernoulli(
        temperature=0.1, probs=density_coeff
    )
    scatterers_density = scatter_density_distribution.sample()
    amplitude = torch.sigmoid(raw[..., 4])
    scatterers_map = scatterers_density * amplitude

    psf_scatter = F.conv2d(
        scatterers_map, g_kernel[None, None, ...], stride=1, padding="same"
    )
    # psf_scatter = torch.squeeze(psf_scatter)

    # Compute remaining intensity at a point n
    confidence_maps = attenuation_total * reflection_total

    # Compute backscattering and reflection parts of the final echo
    b = confidence_maps * psf_scatter
    r = confidence_maps * reflection_coeff

    intensity_map = b + r

    return {
        "intensity_map": intensity_map,
        "attenuation_coeff": attenuation_coeff,
        "reflection_coeff": reflection_coeff,
        "attenuation_total": attenuation_total,
        "reflection_total": reflection_total,
        "scatterers_density": scatterers_density,
        "scatterers_density_coeff": density_coeff,
        "scatter_amplitude": amplitude,
        "b": b,
        "r": r,
        "confidence_maps": confidence_maps,
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
    scatter_density_distribution = RelaxedBernoulli(
        temperature=0.1, probs=density_coeff
    )
    scatterers_density = scatter_density_distribution.sample()
    amplitude = torch.sigmoid(raw[..., 2])
    scatterers_map = scatterers_density * amplitude

    psf_scatter = F.conv2d(
        scatterers_map, g_kernel[None, None, ...], stride=1, padding="same"
    )
    # psf_scatter = torch.squeeze(psf_scatter)

    # Compute remaining intensity at a point n
    confidence_maps = attenuation_total * reflection_total

    # Compute backscattering and reflection parts of the final echo
    b = confidence_maps * psf_scatter
    r = confidence_maps * reflection_coeff

    # Compute the final echo
    amplification_constant = torch.tensor(np.pi)
    alpha_amplification = lambda x: torch.log(
        torch.tensor(1.0) + amplification_constant * x
    ) * torch.log(torch.tensor(1.0) + amplification_constant)
    r_amplified = alpha_amplification(r)
    r_cos = cos_theta * r_amplified
    intensity_map = b + r_cos

    return {
        "intensity_map": intensity_map,
        "attenuation_coeff": attenuation_coeff,
        "reflection_coeff": reflection_coeff,
        "attenuation_total": attenuation_total,
        "reflection_total": reflection_total,
        "scatterers_density": scatterers_density,
        "scatterers_density_coeff": density_coeff,
        "scatter_amplitude": amplitude,
        "b": b,
        "r": r,
        "r_amplified": r_amplified,
        "confidence_maps": confidence_maps,
    }


def render_method_3D(raw):
    def raw2attention(raw, dists):
        return torch.exp(-raw * dists)

    W, H, maps = raw.shape

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
    attenuation_total = torch.cumprod(attenuation, dim=2)

    # REFLECTION
    # reflection_coeff = torch.zeros_like(raw[..., 0])

    reflection_coeff = torch.clamp_max(torch.sigmoid(raw[..., 1]), 0.34)
    reflection_transmission = raw2reflection(reflection_coeff)
    reflection_total = torch.cumprod(reflection_transmission, dim=2)

    # BACKSCATTERING
    density_coeff = torch.ones_like(reflection_coeff) * 0.75
    scatter_density_distribution = RelaxedBernoulli(
        temperature=0.1, probs=density_coeff
    )
    scatterers_density = scatter_density_distribution.sample()
    amplitude = torch.sigmoid(raw[..., 2])
    scatterers_map = scatterers_density * amplitude

    confidence_maps = attenuation_total * reflection_total

    # Compute backscattering and reflection parts of the final echo
    b = confidence_maps * scatterers_map
    r = confidence_maps * reflection_coeff

    # Compute the final echo
    amplification_constant = torch.tensor(3.14)
    alpha_amplification = lambda x: torch.log(
        torch.tensor(1.0) + amplification_constant * x
    ) * torch.log(torch.tensor(1.0) + amplification_constant)
    r_amplified = alpha_amplification(r)

    intensity_map = b + r_amplified

    intensity_map = torch.reshape(intensity_map, (7, W, H // 7))[None, ...]
    intensity_map = F.conv3d(
        intensity_map, g_kernel3D[None, None, ...], stride=[7, 1, 1], padding=[0, 3, 3]
    )

    return {
        "intensity_map": intensity_map,
        "attenuation_coeff": attenuation_coeff,
        "reflection_coeff": reflection_coeff,
        "attenuation_total": attenuation_total,
        "reflection_total": reflection_total,
        "scatterers_density": scatterers_density,
        "scatterers_density_coeff": density_coeff,
        "scatter_amplitude": amplitude,
        "b": b,
        "r": r,
        "r_amplified": r_amplified,
        "confidence_maps": confidence_maps,
    }


def render_rays_us(
    ray_batch,
    network_fn,
    network_query_fn,
    N_samples,
    lindisp=False,
    **kwargs,
):
    """Volumetric rendering.

    Args:
    ray_batch: Tensor of shape [batch_size, ...]. We define rays and do not sample.

    Returns:
    Rendered outputs.
    """



def render_rays_us_with_reconstruction(
        ray_batch,
        network_fn,
        network_rec,
        network_query_fn,
        network_query_fn_rec,
        N_samples,
        lindisp=False,
        **kwargs,
):
    """Volumetric rendering.

    Args:
    ray_batch: Tensor of shape [batch_size, ...]. We define rays and do not sample.

    Returns:
    Rendered outputs.
    """

    def raw2outputs(raw, z_vals):
        """Transforms model's predictions to semantically meaningful values."""
        # TODO: add args controlling the rendering method
        ret = render_method_3(
            raw
        )  # Assuming render_method_3 is defined elsewhere
        # ret = rendering(raw, z_vals)
        return ret

    ###############################
    # Batch size
    N_rays = ray_batch.shape[0]

    # Extract ray origin, direction
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each

    # Extract lower, upper bound for ray distance
    bounds = ray_batch[..., 6:8].reshape(-1, 1, 2)
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    # Decide where to sample along each ray
    t_vals = torch.linspace(0.0, 1.0, N_samples).to(ray_batch.device)
    if not lindisp:
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    z_vals = z_vals.expand(N_rays, N_samples)

    # Points in space to evaluate model at
    origin = rays_o.unsqueeze(-2)
    step = rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)

    pts = step + origin

    # Evaluate model at each point
    raw = network_query_fn(pts, network_fn)  # [N_rays, N_samples , 5]
    ret = raw2outputs(raw, z_vals)

    # input_reconstruction = torch.cat([pts.detach().clone(), raw.detach().clone()], dim=-1)
    #
    # ret_reconstruction = network_query_fn_rec(input_reconstruction, network_rec)
    #
    # ret["reconstruction"] = ret_reconstruction.permute(2, 1, 0)[None, ...]
    # ret['pts'] = pts

    # if retraw:
    #     ret['raw'] = raw

    return ret