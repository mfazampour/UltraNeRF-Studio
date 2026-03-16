import importlib
import sys

import torch


def import_with_cpu_safe_cuda_patch(monkeypatch, module_name: str):
    original_to = torch.Tensor.to

    def patched_to(self, *args, **kwargs):
        args = list(args)
        if args and args[0] == "cuda":
            args[0] = "cpu"
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", patched_to)
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_rendering_cumsum_exclusive_matches_expected_result(monkeypatch):
    rendering = import_with_cpu_safe_cuda_patch(monkeypatch, "rendering")
    values = torch.tensor([[1.0, 2.0, 3.0]])

    result = rendering.cumsum_exclusive(values)

    assert torch.allclose(result, torch.tensor([[0.0, 1.0, 3.0]]))


def test_nerf_utils_get_rays_us_linear_returns_expected_geometry(monkeypatch):
    import_with_cpu_safe_cuda_patch(monkeypatch, "rendering")
    sys.modules.pop("nerf_utils", None)
    nerf_utils = importlib.import_module("nerf_utils")

    pose = torch.cat([torch.eye(3), torch.zeros(3, 1)], dim=1)
    rays_o, rays_d = nerf_utils.get_rays_us_linear(H=4, W=4, sw=0.5, sh=1.0, c2w=pose)

    expected_origins = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ]
    )
    expected_dirs = torch.tensor([[0.0, 1.0, 0.0]]).expand_as(expected_origins)

    assert torch.allclose(rays_o, expected_origins)
    assert torch.allclose(rays_d, expected_dirs)
