# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Stable row identities and independent analytic values of shear fixtures."""

import numpy as np
import pytest
from numpy.typing import NDArray

from tests._support.geometry_transfer_cases import (
    apply_shear,
    shear_term,
    transfer_triangles,
)


def test_transfer_triangles_preserve_xy_vertex_and_cell_identities() -> None:
    mesh = transfer_triangles(2)
    expected_points = np.asarray(
        (
            (0, 0),
            (0.5, 0),
            (1, 0),
            (0, 0.5),
            (0.5, 0.5),
            (1, 0.5),
            (0, 1),
            (0.5, 1),
            (1, 1),
        ),
        dtype=np.float64,
    )
    expected_cells = np.asarray(
        (
            (0, 1, 4),
            (0, 4, 3),
            (1, 2, 5),
            (1, 5, 4),
            (3, 4, 7),
            (3, 7, 6),
            (4, 5, 8),
            (4, 8, 7),
        ),
        dtype=np.int32,
    )
    np.testing.assert_array_equal(mesh.coordinates, expected_points)
    np.testing.assert_array_equal(mesh.blocks[0].vertices, expected_cells)
    np.testing.assert_array_equal(mesh.vertex_global_ids, np.arange(9, dtype=np.int64))
    np.testing.assert_array_equal(mesh.blocks[0].global_ids, np.arange(8, dtype=np.int64))


@pytest.mark.parametrize(
    ("degree", "points", "expected"),
    (
        (2, np.asarray(((0, 2),), dtype=np.float64), 0.4),
        (3, np.asarray(((0, 2),), dtype=np.float64), 0.8),
        (2, np.asarray(((0, 2, 2),), dtype=np.float64), 0.4),
        (3, np.asarray(((0, 2, 2),), dtype=np.float64), 0.8),
    ),
    ids=("quadratic-plane", "cubic-plane", "quadratic-volume", "cubic-volume"),
)
def test_shear_fixture_has_analytic_values_without_mutating_inputs(
    degree: int,
    points: NDArray[np.float64],
    expected: float,
) -> None:
    before = points.copy()
    moved = apply_shear(points, shear_term(degree, points.shape[1]), 1.0)
    assert moved[0, 0] == expected
    np.testing.assert_array_equal(moved[:, 1:], before[:, 1:])
    np.testing.assert_array_equal(points, before)
