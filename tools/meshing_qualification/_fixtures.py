# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from typing import Any

import numpy as np

import phydrax as phx


_HEXAHEDRON_CORNERS = np.asarray(
    (
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    )
)


_KUHN_TETRAHEDRA = np.asarray(
    ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))
)


def _kuhn_tetrahedra(cells_per_axis: int, /) -> tuple[np.ndarray, np.ndarray]:
    """Positively oriented Kuhn tetrahedra of a unit-cube lattice."""
    count = cells_per_axis
    axis = np.linspace(0.0, 1.0, count + 1)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        (-1, 3)
    )
    index = np.arange((count + 1) ** 3, dtype=np.int32).reshape((count + 1,) * 3)
    corners = np.stack(
        [
            index[a : a + count, b : b + count, c : c + count].reshape((-1,))
            for a, b, c in _HEXAHEDRON_CORNERS
        ],
        axis=1,
    )
    tetrahedra = corners[:, _KUHN_TETRAHEDRA].reshape((-1, 4))
    first, second, third, fourth = (points[tetrahedra[:, i]] for i in range(4))
    volume = np.sum(np.cross(second - first, third - first) * (fourth - first), axis=1)
    tetrahedra[volume < 0.0] = tetrahedra[volume < 0.0][:, [1, 0, 2, 3]]
    return points, tetrahedra


def _planar_triangles(cells_per_axis: int, /) -> np.ndarray:
    """Counterclockwise triangles of a square lattice disk."""
    index = np.arange((cells_per_axis + 1) ** 2, dtype=np.int32).reshape(
        (cells_per_axis + 1, cells_per_axis + 1)
    )
    lower = index[:-1, :-1].reshape((-1,))
    right = index[1:, :-1].reshape((-1,))
    upper = index[1:, 1:].reshape((-1,))
    left = index[:-1, 1:].reshape((-1,))
    return np.concatenate(
        (np.stack((lower, right, upper), axis=1), np.stack((lower, upper, left), axis=1))
    )


def _jittered_unit_square_triangles(cells_per_axis: int, seed: int, /) -> Any:
    """Counterclockwise lattice triangles with interior vertices jittered in place."""
    axis = np.linspace(0.0, 1.0, cells_per_axis + 1)
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape((-1, 2))
    interior = np.all((points > 0.0) & (points < 1.0), axis=1)
    jitter = np.random.default_rng(seed).uniform(-0.2, 0.2, points.shape)
    points[interior] += jitter[interior] / cells_per_axis
    return phx.discretization.CellMesh.from_triangles(
        points, _planar_triangles(cells_per_axis)
    )


def _remap_field(points: np.ndarray, /) -> np.ndarray:
    x, y = points[..., 0], points[..., 1]
    return np.sin(2.0 * np.pi * x) * np.cos(np.pi * y) + x * x
