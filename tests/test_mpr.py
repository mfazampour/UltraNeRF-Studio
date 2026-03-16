import numpy as np

from visualization.mpr import orthogonal_slice_indices, selection_from_world_point, update_selection_for_view_click
from visualization.transforms import VolumeGeometry


def make_geometry() -> VolumeGeometry:
    return VolumeGeometry(origin_mm=np.array([10.0, 20.0, 30.0]), spacing_mm=np.array([1.0, 2.0, 5.0]))


def test_selection_from_world_point_maps_to_clamped_voxel_indices():
    geometry = make_geometry()
    selection = selection_from_world_point(
        world_point_mm=np.array([12.2, 24.4, 42.7], dtype=np.float32),
        geometry=geometry,
        volume_shape=(5, 5, 5),
    )

    assert np.allclose(selection.voxel_point, np.array([2.2, 2.2, 2.54], dtype=np.float32))
    assert selection.voxel_indices == (2, 2, 3)
    assert np.allclose(selection.world_point_mm, np.array([12.0, 24.0, 45.0], dtype=np.float32))


def test_selection_clamps_when_world_point_is_outside_volume():
    geometry = make_geometry()
    selection = selection_from_world_point(
        world_point_mm=np.array([100.0, -50.0, 999.0], dtype=np.float32),
        geometry=geometry,
        volume_shape=(5, 5, 5),
    )

    assert selection.voxel_indices == (4, 0, 4)
    assert np.allclose(selection.world_point_mm, np.array([14.0, 20.0, 50.0], dtype=np.float32))


def test_update_selection_for_axial_click_updates_x_and_y_only():
    geometry = make_geometry()
    selection = selection_from_world_point(
        world_point_mm=np.array([11.0, 22.0, 35.0], dtype=np.float32),
        geometry=geometry,
        volume_shape=(5, 5, 5),
    )

    updated = update_selection_for_view_click(
        selection,
        view="axial",
        first_axis_value=3.0,
        second_axis_value=4.0,
        geometry=geometry,
        volume_shape=(5, 5, 5),
    )

    assert updated.voxel_indices == (3, 4, 1)
    assert np.allclose(updated.world_point_mm, np.array([13.0, 28.0, 35.0], dtype=np.float32))


def test_orthogonal_slice_indices_expose_active_plane_for_each_view():
    geometry = make_geometry()
    selection = selection_from_world_point(
        world_point_mm=np.array([13.0, 26.0, 40.0], dtype=np.float32),
        geometry=geometry,
        volume_shape=(5, 5, 5),
    )

    indices = orthogonal_slice_indices(selection)

    assert indices == {"sagittal": 3, "coronal": 3, "axial": 2}
