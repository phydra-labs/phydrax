"""Authored coordinates and unnormalized carrier gauges of the wedge fixture."""

import numpy as np
import pytest
from numpy.typing import NDArray

from phydrax.geometry.brep._patches import PlanePatch
from tests._support.brep_feature_sources import unnormalized_wedge


def _require_authored_wedge(
    points: NDArray[np.float64], first_axes: NDArray[np.float64]
) -> None:
    expected_points = np.asarray(
        [
            (x, y, z)
            for z in (0.0, 4.0)
            for x, y in ((0.0, 0.0), (0.0, -5.0), (4.0, -5.0), (4.0, -3.0))
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(points, expected_points)
    np.testing.assert_array_equal(
        first_axes,
        (
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, -5.0, 0.0),
            (4.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),
            (-4.0, 3.0, 0.0),
        ),
    )


def _source_arrays() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    source = unnormalized_wedge()
    points = np.asarray(source.require_geometry().vertex_points, dtype=np.float64)
    first_axes = np.asarray(
        [
            np.asarray(patch.first_axis)
            for patch in source.patches
            if isinstance(patch, PlanePatch)
        ],
        dtype=np.float64,
    )
    return points, first_axes


def test_unnormalized_wedge_preserves_integer_source_and_carrier_gauges() -> None:
    _require_authored_wedge(*_source_arrays())


def test_unnormalized_wedge_oracle_rejects_vertex_corruption() -> None:
    points, axes = _source_arrays()
    _require_authored_wedge(points, axes)
    points = points.copy()
    points[0, 0] += 1.0
    with pytest.raises(AssertionError):
        _require_authored_wedge(points, axes)


def test_unnormalized_wedge_oracle_rejects_normalized_carrier() -> None:
    points, axes = _source_arrays()
    _require_authored_wedge(points, axes)
    axes = axes.copy()
    axes[2] /= 5.0
    with pytest.raises(AssertionError):
        _require_authored_wedge(points, axes)
