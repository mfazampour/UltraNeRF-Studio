import numpy as np
import pytest
import torch

from ultranerf.camera import Lie, Pose
from ultranerf.model import BARF, NeRF, PoseRefine, Reconstruction


def test_pose_compose_and_invert_are_consistent():
    pose_ops = Pose()
    first = pose_ops(t=torch.tensor([1.0, 0.0, 0.0]))
    second = pose_ops(t=torch.tensor([0.0, 2.0, 0.0]))

    composed = pose_ops.compose([first, second])
    inverted = pose_ops.invert(composed)
    identity = pose_ops.compose([composed, inverted])

    assert torch.allclose(composed[:, 3], torch.tensor([1.0, 2.0, 0.0]))
    assert torch.allclose(identity[:, :3], torch.eye(3), atol=1e-5)
    assert torch.allclose(identity[:, 3], torch.zeros(3), atol=1e-5)


def test_lie_zero_twist_returns_identity_transform():
    lie = Lie()
    transform = lie.se3_to_SE3(torch.zeros(1, 6))

    expected = torch.cat([torch.eye(3), torch.zeros(3, 1)], dim=1).unsqueeze(0)
    assert torch.allclose(transform, expected, atol=1e-6)


def test_pose_refine_returns_original_pose_for_zero_offsets():
    poses = torch.cat([torch.eye(3), torch.zeros(3, 1)], dim=1).unsqueeze(0)
    refiner = PoseRefine(poses=poses, mode="train")

    refined = refiner.get_pose(0)

    assert torch.allclose(refined, poses, atol=1e-6)


def test_pose_refine_scales_translation_offsets_from_mm_to_m():
    poses = torch.cat([torch.eye(3), torch.zeros(3, 1)], dim=1).unsqueeze(0)
    refiner = PoseRefine(poses=poses, mode="train")
    refiner.se3_refine.weight.data[0] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    refined = refiner.get_pose(0)

    assert refined[0, 3].item() == pytest.approx(0.001, rel=0.0, abs=1e-6)


def test_nerf_and_barf_forward_shapes_are_stable():
    points = torch.randn(7, 3)

    nerf = NeRF(D=4, W=16, input_ch=3, output_ch=5, skips=[2])
    barf = BARF(D=4, W=16, input_ch=3, output_ch=5, skips=[2], L=2, c2f=(0.1, 0.5))
    barf.progress.data.fill_(0.3)

    nerf_out = nerf(points)
    barf_out = barf(points)

    assert nerf_out.shape == (7, 5)
    assert barf_out.shape == (7, 5)
    assert torch.isfinite(barf_out).all()


def test_reconstruction_output_is_probability_like():
    model = Reconstruction(D=3, W=8, input_ch=6, skips=[1])
    inputs = torch.randn(10, 6)

    outputs = model(inputs)

    assert outputs.shape == (10, 1)
    assert torch.all(outputs >= 0.0)
    assert torch.all(outputs <= 1.0)
