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
