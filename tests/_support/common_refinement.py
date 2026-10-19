"""Ordered unit-domain meshes and moment assertions for common refinement."""

from __future__ import annotations

import numpy as np
import pytest

from phydrax.discretization import CellBlock, CellMesh
from phydrax.geometry import CommonRefinementStatus, PreparedCommonRefinement


# Locked empirical tolerances from tests/unit/geometry/test_supermesh.py at
# d2dfcb941, for its bounded unit-square/unit-cube fixtures. Their values are
# preserved when splitting the scenarios into separately collected contracts.
CELL_MEASURE_RTOL = 1e-14
CELL_CENTROID_RTOL = 1e-13
COMPLETE_MEASURE_RTOL = 1e-13
COMPLETE_MOMENT_RTOL = 1e-13
COMPLETE_SECOND_MOMENT_RTOL = 1e-12
COVERAGE_DEFECT_LIMIT = 1e-12
PARTITION_MEASURE_ATOL = 1e-16
SOURCE_COVERAGE_RTOL = 1e-12
SINGLE_CELL_RTOL = 1e-15


def square_grid(cells: int, /) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, 1.0, cells + 1, dtype=np.float64)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.column_stack((x.ravel(), y.ravel()))
    index = np.arange((cells + 1) ** 2, dtype=np.int64).reshape(cells + 1, cells + 1)
    quads = np.stack(
        (index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]), axis=-1
    ).reshape((-1, 4))
    return points, quads


def quad_mesh(cells: int, /) -> CellMesh:
    points, quads = square_grid(cells)
    return CellMesh(points, (CellBlock("quads", "quadrilateral", quads),))


def perturbed_triangles(cells: int, seed: int, /) -> CellMesh:
    points, quads = square_grid(cells)
    rng = np.random.default_rng(seed)
    interior = np.all((points > 0.0) & (points < 1.0), axis=1)
    points[interior] += rng.uniform(-0.2, 0.2, (int(np.sum(interior)), 2)) / cells
    return CellMesh.from_triangles(
        points, np.concatenate((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]))
    )


def unit_cube_moments(dimension: int, /) -> tuple[np.ndarray, np.ndarray]:
    first = np.full((dimension,), 0.5, dtype=np.float64)
    second = (
        np.full((dimension, dimension), 0.25, dtype=np.float64)
        + np.eye(dimension, dtype=np.float64) / 12.0
    )
    return first, second


def cube_lattice(cells: int, /) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, 1.0, cells + 1, dtype=np.float64)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    index = np.arange((cells + 1) ** 3, dtype=np.int64).reshape((cells + 1,) * 3)
    corners = np.stack(
        [
            index[i : cells + i, j : cells + j, k : cells + k].ravel()
            for k in (0, 1)
            for j in (0, 1)
            for i in (0, 1)
        ],
        axis=1,
    )
    return points, corners


def kuhn_tetrahedra(cells: int, /) -> CellMesh:
    points, corners = cube_lattice(cells)
    paths = ((1, 3, 7), (1, 5, 7), (2, 3, 7), (2, 6, 7), (4, 5, 7), (4, 6, 7))
    tetrahedra = np.concatenate([corners[:, [0, a, b, c]] for a, b, c in paths])
    edges = points[tetrahedra[:, 1:]] - points[tetrahedra[:, :1]]
    negative = np.sum(edges[:, 0] * np.cross(edges[:, 1], edges[:, 2]), axis=1) < 0.0
    tetrahedra[negative] = tetrahedra[negative][:, [0, 2, 1, 3]]
    return CellMesh.from_tetrahedra(points, tetrahedra)


def hexahedra(cells: int, /) -> CellMesh:
    points, corners = cube_lattice(cells)
    return CellMesh(
        points,
        (CellBlock("hexahedra", "hexahedron", corners[:, [0, 1, 3, 2, 4, 5, 7, 6]]),),
    )


def assert_complete_moments(
    refinement: PreparedCommonRefinement, dimension: int, /
) -> None:
    first, second = unit_cube_moments(dimension)
    assert refinement.status is CommonRefinementStatus.SUCCESS
    assert refinement.succeeded
    assert np.sum(np.asarray(refinement.volumes)) == pytest.approx(
        1.0, rel=COMPLETE_MEASURE_RTOL
    )
    np.testing.assert_allclose(
        np.sum(np.asarray(refinement.first_moments), axis=0),
        first,
        rtol=COMPLETE_MOMENT_RTOL,
    )
    np.testing.assert_allclose(
        np.sum(np.asarray(refinement.second_moments), axis=0),
        second,
        rtol=COMPLETE_SECOND_MOMENT_RTOL,
    )
    evidence = refinement.evidence
    assert evidence.maximum_relative_source_defect < COVERAGE_DEFECT_LIMIT
    assert evidence.maximum_relative_target_defect < COVERAGE_DEFECT_LIMIT
    assert evidence.source_gap_count == evidence.source_double_count == 0
