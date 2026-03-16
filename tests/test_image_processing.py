import numpy as np
import torch

from evident_border import detect_edges, get_borders, speckle_reducing_anisotropic_diffusion
from rendering_utils.reflection import calculate_reflection_coefficient


def test_srad_rejects_non_float32_inputs():
    image = np.ones((8, 8), dtype=np.float64)

    try:
        speckle_reducing_anisotropic_diffusion(image)
    except ValueError as exc:
        assert "float32" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-float32 input")


def test_srad_preserves_shape_and_range_for_valid_input():
    image = np.linspace(0.1, 0.9, 64, dtype=np.float32).reshape(8, 8)

    filtered = speckle_reducing_anisotropic_diffusion(image, niter=2, kappa=0.2, lambda_=0.1)

    assert filtered.shape == image.shape
    assert filtered.dtype == np.float32
    assert filtered.min() >= 0.0
    assert filtered.max() <= 1.0


def test_detect_edges_finds_step_boundary():
    image = np.zeros((12, 12), dtype=np.float32)
    image[:, 6:] = 1.0

    edges = detect_edges(image, threshold_value=0.1)

    assert edges.shape == image.shape
    assert edges.sum() > 0
    assert edges[:, 5:7].sum() > 0


def test_get_borders_returns_sparse_binary_like_map():
    image = np.zeros((32, 32), dtype=np.float32)
    image[8:24, 16:] = 1.0

    borders = get_borders(image, niter=2, kappa=0.2, lambda_=0.1)

    assert borders.shape == image.shape
    assert borders.dtype in (np.bool_, np.float32, np.float64)
    assert borders.sum() > 0


def test_reflection_coefficient_is_zero_for_constant_map_and_positive_at_boundary():
    constant = torch.ones((1, 1, 6, 6), dtype=torch.float32)
    stepped = torch.ones((1, 1, 6, 6), dtype=torch.float32)
    stepped[:, :, 3:, :] = 2.0

    constant_reflection = calculate_reflection_coefficient(constant)
    stepped_reflection = calculate_reflection_coefficient(stepped)

    assert torch.allclose(constant_reflection, torch.zeros_like(constant_reflection))
    assert stepped_reflection.shape == stepped.shape
    assert torch.max(stepped_reflection).item() > 0.0
