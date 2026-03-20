from types import SimpleNamespace

import numpy as np
import torch

from ultranerf.visualization.nerf_session import NerfRuntime, NerfSession, pose_mm_to_model_pose_m


class FakeParser:
    def parse_args(self, args=None):
        return SimpleNamespace(chunk=123, config_args=list(args or []))


def make_runtime(call_log):
    def config_parser():
        return FakeParser()

    def create_nerf(args, device, mode="test"):
        call_log["create_nerf"] = {
            "args": args,
            "device": str(device),
            "mode": mode,
        }
        return None, {"network_fn": "fake-network"}, 0, None, None

    def render_us(H, W, sw, sh, c2w=None, chunk=None, **kwargs):
        call_log["render_us"] = {
            "H": H,
            "W": W,
            "sw": sw,
            "sh": sh,
            "c2w": c2w.detach().cpu().numpy(),
            "chunk": chunk,
            "kwargs": kwargs,
        }
        return {"intensity_map": torch.ones((1, 1, H, W), dtype=torch.float32)}

    return NerfRuntime(
        torch=torch,
        config_parser=config_parser,
        create_nerf=create_nerf,
        render_us=render_us,
    )


def test_pose_mm_to_model_pose_m_scales_translation_only():
    pose_mm = np.eye(4, dtype=np.float32)
    pose_mm[:3, 3] = np.array([10.0, 20.0, 30.0], dtype=np.float32)

    pose_m = pose_mm_to_model_pose_m(pose_mm)

    assert np.allclose(pose_m[:3, :3], np.eye(3, dtype=np.float32))
    assert np.allclose(pose_m[:3, 3], np.array([0.01, 0.02, 0.03], dtype=np.float32))


def test_nerf_session_from_checkpoint_computes_meter_scale_probe_spacing():
    call_log = {}
    runtime = make_runtime(call_log)

    session = NerfSession.from_checkpoint(
        config_path="configs/config_base_nerf.txt",
        checkpoint_path="logs/example/001000.tar",
        image_shape=(140, 80),
        probe_width_mm=80.0,
        probe_depth_mm=140.0,
        device="cpu",
        runtime=runtime,
    )

    assert session.sw_m == 0.001
    assert session.sh_m == 0.001
    assert session.near_m == 0.0
    assert session.far_m == 0.14
    assert call_log["create_nerf"]["mode"] == "test"


def test_nerf_session_render_pose_calls_runtime_with_converted_pose():
    call_log = {}
    runtime = make_runtime(call_log)
    session = NerfSession.from_checkpoint(
        config_path="configs/config_base_nerf.txt",
        checkpoint_path="logs/example/001000.tar",
        image_shape=(4, 5),
        probe_width_mm=50.0,
        probe_depth_mm=40.0,
        device="cpu",
        runtime=runtime,
    )
    pose_mm = np.eye(4, dtype=np.float32)
    pose_mm[:3, 3] = np.array([5.0, 6.0, 7.0], dtype=np.float32)

    output = session.render_pose(pose_mm)

    render_call = call_log["render_us"]
    assert render_call["H"] == 4
    assert render_call["W"] == 5
    assert render_call["chunk"] == 123
    assert np.allclose(render_call["c2w"][0, :, 3], np.array([0.005, 0.006, 0.007], dtype=np.float32))
    assert render_call["kwargs"]["near"] == 0.0
    assert render_call["kwargs"]["far"] == 0.04
    assert "intensity_map" in output
