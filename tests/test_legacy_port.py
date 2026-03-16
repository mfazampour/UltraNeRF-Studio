import torch

from model import BARF, NeRF, Reconstruction


def test_pytorch_nerf_port_produces_expected_output_shape():
    model = NeRF(D=4, W=32, input_ch=3, output_ch=5, skips=[2])
    points = torch.randn(11, 3)

    outputs = model(points)

    assert outputs.shape == (11, 5)
    assert torch.isfinite(outputs).all()


def test_pytorch_barf_port_supports_coarse_to_fine_encoding():
    model = BARF(D=4, W=32, input_ch=3, output_ch=5, skips=[2], L=3, c2f=(0.1, 0.5))
    model.progress.data.fill_(0.25)
    points = torch.randn(9, 3)

    outputs = model(points)

    assert outputs.shape == (9, 5)
    assert torch.isfinite(outputs).all()


def test_pytorch_reconstruction_head_returns_probabilities():
    model = Reconstruction(D=4, W=32, input_ch=6, skips=[2])
    features = torch.randn(7, 6)

    outputs = model(features)

    assert outputs.shape == (7, 1)
    assert torch.all(outputs >= 0.0)
    assert torch.all(outputs <= 1.0)
