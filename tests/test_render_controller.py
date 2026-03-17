import numpy as np

from visualization.render_controller import RenderController


class FakeSession:
    def __init__(self):
        self.calls = []

    def render_pose(self, pose_probe_to_world_mm, **kwargs):
        pose = np.asarray(pose_probe_to_world_mm, dtype=np.float32)
        self.calls.append({"pose": pose.copy(), "kwargs": dict(kwargs)})
        return {"intensity_map": pose.copy()}


def test_manual_render_controller_updates_pose_without_rendering_automatically():
    session = FakeSession()
    controller = RenderController(nerf_session=session, trigger_mode="manual")
    pose = np.eye(4, dtype=np.float32)

    controller.initialize(pose)
    result = controller.set_probe_pose(pose)

    assert result is None
    assert controller.state is not None
    assert controller.state.dirty
    assert len(session.calls) == 0


def test_manual_render_controller_renders_on_request():
    session = FakeSession()
    controller = RenderController(nerf_session=session, trigger_mode="manual")
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    controller.initialize(pose)
    output = controller.render_current_pose()

    assert len(session.calls) == 1
    assert np.allclose(session.calls[0]["pose"], pose)
    assert "intensity_map" in output
    assert not controller.state.dirty


def test_on_pose_change_controller_renders_when_pose_changes():
    session = FakeSession()
    controller = RenderController(nerf_session=session, trigger_mode="on_pose_change", render_overrides={"retraw": True})
    pose = np.eye(4, dtype=np.float32)
    updated_pose = np.eye(4, dtype=np.float32)
    updated_pose[:3, 3] = np.array([5.0, 0.0, 0.0], dtype=np.float32)

    controller.initialize(pose)
    output = controller.set_probe_pose(updated_pose)

    assert len(session.calls) == 2
    assert np.allclose(session.calls[-1]["pose"], updated_pose)
    assert session.calls[-1]["kwargs"] == {"retraw": True}
    assert np.allclose(output["intensity_map"], updated_pose)
    assert np.allclose(controller.state.last_render_pose_mm, updated_pose)


def test_throttled_controller_defers_renders_until_interval_passes():
    session = FakeSession()
    current_time = {"value": 0.0}
    controller = RenderController(
        nerf_session=session,
        trigger_mode="on_pose_change_throttled",
        min_render_interval_s=0.5,
        time_fn=lambda: current_time["value"],
    )
    pose = np.eye(4, dtype=np.float32)
    updated_pose = np.eye(4, dtype=np.float32)
    updated_pose[:3, 3] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    controller.initialize(pose)
    assert len(session.calls) == 1

    current_time["value"] = 0.1
    result = controller.set_probe_pose(updated_pose)
    assert result is None
    assert len(session.calls) == 1
    assert controller.state.dirty

    current_time["value"] = 0.6
    flushed = controller.flush_pending_render()
    assert flushed is not None
    assert len(session.calls) == 2
    assert np.allclose(session.calls[-1]["pose"], updated_pose)
    assert not controller.state.dirty


def test_render_now_force_path_preserves_last_good_output_when_next_render_fails():
    class FlakySession:
        def __init__(self):
            self.calls = 0

        def render_pose(self, pose_probe_to_world_mm, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"intensity_map": np.asarray(pose_probe_to_world_mm, dtype=np.float32)}
            raise RuntimeError("synthetic failure")

    session = FlakySession()
    controller = RenderController(nerf_session=session, trigger_mode="manual")
    pose = np.eye(4, dtype=np.float32)

    controller.initialize(pose)
    first_output = controller.render_current_pose()

    try:
        controller.render_current_pose()
    except RuntimeError as exc:
        assert str(exc) == "synthetic failure"
    else:
        raise AssertionError("Expected render failure")

    assert np.allclose(controller.state.last_render_output["intensity_map"], first_output["intensity_map"])
    assert controller.state.last_error == "synthetic failure"
