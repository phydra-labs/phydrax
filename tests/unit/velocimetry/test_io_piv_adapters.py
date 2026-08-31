#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from phydrax.velocimetry.imaging import DenseDisplacementField2D
from phydrax.velocimetry.io import (
    AdapterError,
    AdapterStatus,
    piv_to_observation_sequence,
    piv_to_tensor_grid,
    read_openpiv_text,
    read_pivlab,
    require_pivpy,
    require_xarray,
    write_openpiv_text,
    write_pivlab,
)
from phydrax.velocimetry.piv import PhysicalPIVResult2D


def _pixel_field() -> DenseDisplacementField2D:
    row, column = np.meshgrid(
        np.array([1.0, 3.0]), np.array([2.0, 4.0, 6.0]), indexing="ij"
    )
    positions = np.stack((row, column), axis=-1)
    displacement = np.array(
        [
            [[0.0, 0.0], [1.0, 2.0], [2.0, -1.0]],
            [[9.0, 9.0], [-2.0, 3.0], [4.0, 5.0]],
        ]
    )
    valid = np.array([[True, True, True], [False, True, True]])
    return DenseDisplacementField2D(
        positions,
        displacement,
        valid,
        geometry_id="image-geometry",
        field_id="pixel-field",
        provenance=("raw-piv",),
    )


def _physical_field() -> PhysicalPIVResult2D:
    x = np.broadcast_to(np.array([10.0, 12.0, 14.0])[None, :], (2, 3))
    y = np.broadcast_to(np.array([4.0, 2.0])[:, None], (2, 3))
    positions = np.stack((x, y), axis=-1)
    velocity = np.array(
        [
            [[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]],
            [[8.0, 9.0], [-1.0, 1.0], [2.0, -2.0]],
        ]
    )
    valid = np.array([[True, True, True], [False, True, True]])
    return PhysicalPIVResult2D(
        positions,
        velocity * 0.5,
        velocity,
        valid,
        "pixel-field",
        "calibration",
        "m",
        "s",
    )


def test_openpiv_pixel_round_trip_preserves_orientation_zero_and_invalidity(tmp_path):
    path = tmp_path / "field.txt"
    source = _pixel_field()
    write_report = write_openpiv_text(
        path,
        source,
        coordinate_convention="physical",
        delta_t=2.0,
    )
    restored, read_report = read_openpiv_text(
        path,
        value_kind="pixel-displacement",
        geometry_id="image-geometry",
        coordinate_convention="physical",
        delta_t=2.0,
    )

    np.testing.assert_allclose(restored.positions_rc, source.positions_rc)
    np.testing.assert_allclose(
        np.asarray(restored.displacement_rc)[np.asarray(source.valid)],
        np.asarray(source.displacement_rc)[np.asarray(source.valid)],
    )
    assert bool(np.asarray(restored.valid)[0, 0])
    np.testing.assert_array_equal(np.asarray(restored.displacement_rc)[0, 0], [0.0, 0.0])
    assert not bool(np.asarray(restored.valid)[1, 0])
    assert write_report.status == AdapterStatus.DECLARED_LOSS
    assert read_report.status == AdapterStatus.DECLARED_LOSS


def test_openpiv_physical_velocity_targets_right_handed_physical_result(tmp_path):
    path = tmp_path / "physical.txt"
    path.write_text(
        "# x y u v flags mask\n"
        "10 2 1 -3 0 0\n"
        "12 2 0 0 0 0\n"
        "10 4 2 5 1 0\n"
        "12 4 -1 1 0 0\n",
        encoding="utf-8",
    )

    field, report = read_openpiv_text(
        path,
        value_kind="physical-velocity",
        spatial_unit="m",
        time_unit="s",
        delta_t=0.25,
    )

    assert isinstance(field, PhysicalPIVResult2D)
    np.testing.assert_array_equal(np.asarray(field.positions_xy)[0, 0], [10.0, 2.0])
    np.testing.assert_array_equal(np.asarray(field.velocity_xy)[0, 0], [1.0, -3.0])
    np.testing.assert_array_equal(np.asarray(field.displacement_xy)[0, 0], [0.25, -0.75])
    assert not bool(np.asarray(field.valid)[1, 0])
    assert "OpenPIV y -> physical y" in report.coordinate_mapping


def test_openpiv_rejects_duplicate_or_incomplete_grid(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text(
        "# x y u v flags mask\n0 0 1 1 0 0\n0 0 2 2 0 0\n",
        encoding="utf-8",
    )
    with pytest.raises(AdapterError, match="rectilinear|duplicate"):
        read_openpiv_text(
            path,
            value_kind="pixel-displacement",
            geometry_id="geometry",
            coordinate_convention="image",
        )


@pytest.mark.parametrize("suffix", [".mat", ".h5"])
def test_pivlab_supported_mat_and_hdf5_layouts_round_trip_pixel_field(tmp_path, suffix):
    path = tmp_path / f"field{suffix}"
    source = _pixel_field()
    write_report = write_pivlab(path, source, y_axis="down")
    fields, read_report = read_pivlab(
        path,
        geometry_id="image-geometry",
        y_axis="down",
        stage="original",
    )

    assert len(fields) == 1
    restored = fields[0]
    assert isinstance(restored, DenseDisplacementField2D)
    np.testing.assert_allclose(restored.positions_rc, source.positions_rc)
    np.testing.assert_allclose(
        np.asarray(restored.displacement_rc)[np.asarray(source.valid)],
        np.asarray(source.displacement_rc)[np.asarray(source.valid)],
    )
    np.testing.assert_array_equal(restored.valid, source.valid)
    assert write_report.losses
    assert read_report.losses


def test_pivlab_physical_layout_returns_physical_result(tmp_path):
    path = tmp_path / "physical.mat"
    source = _physical_field()
    write_pivlab(path, source, y_axis="up")
    fields, _ = read_pivlab(path, y_axis="up", delta_t=0.5)

    restored = fields[0]
    assert isinstance(restored, PhysicalPIVResult2D)
    np.testing.assert_allclose(restored.positions_xy, source.positions_xy)
    np.testing.assert_allclose(
        np.asarray(restored.velocity_xy)[np.asarray(source.valid)],
        np.asarray(source.velocity_xy)[np.asarray(source.valid)],
    )
    np.testing.assert_array_equal(restored.valid, source.valid)


def test_physical_piv_grid_and_observation_adapters_flip_y_without_component_swap():
    source = _physical_field()
    grid, space, values, valid, report = piv_to_tensor_grid(source)

    np.testing.assert_array_equal(grid.axes[0].nodes, [2.0, 4.0])
    np.testing.assert_array_equal(grid.axes[1].nodes, [10.0, 12.0, 14.0])
    np.testing.assert_array_equal(values[1, 1], [0.0, 0.0])
    assert bool(valid[1, 1])
    assert not bool(valid[0, 0])
    assert space.layout.value_shape == (2, 3, 2)
    assert report.status == AdapterStatus.LOSSLESS

    sequence, sequence_grid, _ = piv_to_observation_sequence(
        (source, source),
        np.array([0.0, 1.0]),
    )
    mask = np.asarray(sequence.observation_mask)
    assert mask.shape == (2, 2, 3, 2)
    assert mask[0, 1, 1].all()
    assert not mask[0, 0, 0].any()
    np.testing.assert_array_equal(np.asarray(sequence.values)[0, 1, 1], [0.0, 0.0])
    assert sequence_grid.prepared_id == grid.prepared_id


def test_optional_labeled_dependencies_fail_at_call_boundary(monkeypatch):
    import phydrax.velocimetry.io._xarray as adapter

    original = importlib.util.find_spec
    monkeypatch.setattr(
        adapter.importlib.util,
        "find_spec",
        lambda name: None if name in ("xarray", "pivpy") else original(name),
    )

    with pytest.raises(AdapterError) as xarray_error:
        require_xarray()
    with pytest.raises(AdapterError) as pivpy_error:
        require_pivpy()
    assert xarray_error.value.status == AdapterStatus.OPTIONAL_DEPENDENCY_UNAVAILABLE
    assert pivpy_error.value.status == AdapterStatus.OPTIONAL_DEPENDENCY_UNAVAILABLE
