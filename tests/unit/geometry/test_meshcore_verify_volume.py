"""The exact native tetrahedral certificates on Delaunay tetrahedralizations of convex sets."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from phydrax._meshcore import (
    delaunay_3d,
    tetrahedra_validity,
    VerdictStatus,
    verify_tetrahedra,
    verify_volume,
)


def _convex_case(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Points on and inside the unit sphere are in general position, so the
    # convex hull's triangulation is unique and must equal the Delaunay boundary.
    generator = np.random.default_rng(seed)
    directions = generator.normal(size=(200, 3))
    shell = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    inner = generator.normal(size=(300, 3))
    inner *= (0.9 * generator.uniform(size=(300, 1)) ** (1.0 / 3.0)) / np.linalg.norm(
        inner, axis=1, keepdims=True
    )
    points = np.concatenate((shell, inner))
    cells, _ = delaunay_3d(points)
    hull = ConvexHull(points)
    surface = hull.simplices.astype(np.int32)
    centroid = points.mean(axis=0)
    a, b, c = (points[surface[:, k]] for k in range(3))
    outward = np.einsum("ij,ij->i", np.cross(b - a, c - a), a - centroid) > 0.0
    surface[~outward] = surface[~outward][:, (0, 2, 1)]
    return points, np.asarray(cells, dtype=np.int32), surface


def _union(
    *parts: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points, cells, surfaces = [], [], []
    offset = 0
    for part_points, part_cells, part_surface in parts:
        points.append(part_points)
        cells.append(part_cells + offset)
        surfaces.append(part_surface + offset)
        offset += part_points.shape[0]
    return (
        np.concatenate(points),
        np.concatenate(cells).astype(np.int32),
        np.concatenate(surfaces).astype(np.int32),
    )


def _moved(
    case: tuple[np.ndarray, np.ndarray, np.ndarray], scale: float, shift: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return scale * case[0] + shift, case[1], case[2]


def test_delaunay_tetrahedralization_of_a_convex_set_is_certified() -> None:
    points, cells, surface = _convex_case(11)
    verdict = verify_volume(points, cells, surface)
    assert verdict.certified
    assert verdict.boundary_faces == surface.shape[0]
    embedding = verify_tetrahedra(points, cells)
    assert embedding.status is VerdictStatus.CERTIFIED
    assert embedding.boundary_faces == surface.shape[0]
    assert embedding.boundary_shells == 1


def test_an_inverted_cell_is_reported() -> None:
    points, cells, surface = _convex_case(12)
    cells[7, [0, 1]] = cells[7, [1, 0]]
    verdict = verify_volume(points, cells, surface)
    assert not verdict.certified
    assert verdict.nonpositive_cells == 1
    assert verdict.first_nonpositive_cell == 7
    embedding = verify_tetrahedra(points, cells)
    assert embedding.status is VerdictStatus.VIOLATED
    assert embedding.first_nonpositive_cell == 7


def test_a_removed_cell_leaves_unmatched_faces() -> None:
    points, cells, surface = _convex_case(13)
    verdict = verify_volume(points, np.delete(cells, 3, axis=0), surface)
    assert not verdict.certified
    assert verdict.unmatched_boundary_faces > 0


def test_a_wrong_region_label_is_reported() -> None:
    points, cells, surface = _convex_case(15)
    regions = np.zeros((cells.shape[0],), dtype=np.int32)
    sides = np.tile(np.asarray([[0, -1]], dtype=np.int32), (surface.shape[0], 1))
    assert verify_volume(
        points, cells, surface, cell_regions=regions, surface_regions=sides
    ).certified
    regions[5] = 1
    verdict = verify_volume(
        points, cells, surface, cell_regions=regions, surface_regions=sides
    )
    assert not verdict.certified
    assert verdict.region_mismatches > 0


def test_separate_solids_are_embedded_and_nested_solids_are_not() -> None:
    first = _convex_case(16)
    second = _convex_case(17)
    apart = _union(first, _moved(second, 1.0, 3.0))
    verdict = verify_tetrahedra(apart[0], apart[1])
    assert verdict.status is VerdictStatus.CERTIFIED
    assert verdict.boundary_shells == 2
    nested = _union(first, _moved(second, 0.25, 0.0))
    verdict = verify_tetrahedra(nested[0], nested[1])
    assert verdict.status is VerdictStatus.VIOLATED
    assert verdict.misnested_shells == 1
    assert verdict.boundary_contacts == 0
    crossing = _union(first, _moved(second, 1.0, 0.5))
    verdict = verify_tetrahedra(crossing[0], crossing[1])
    assert verdict.status is VerdictStatus.VIOLATED
    assert verdict.boundary_contacts > 0


def test_cell_indices_out_of_range_are_refused() -> None:
    points, cells, surface = _convex_case(14)
    cells[0, 0] = points.shape[0]
    with pytest.raises(ValueError):
        verify_volume(points, cells, surface)
    with pytest.raises(ValueError):
        verify_tetrahedra(points, cells)


def _exact_determinant(points: np.ndarray, cell: np.ndarray) -> Fraction:
    a, b, c, d = ([Fraction(float(value)) for value in points[row]] for row in cell)
    e = [[q[axis] - a[axis] for axis in range(3)] for q in (b, c, d)]
    return (
        e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1])
        - e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0])
        + e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0])
    )


def test_affine_validity_matches_exact_rational_determinants() -> None:
    points, cells, _ = _convex_case(18)
    sliver = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    points = np.concatenate((points, sliver, [[0.25, 0.25, 2.0**-30]]))
    base = points.shape[0] - 4
    cells = np.concatenate(
        (cells, [[base + 2, base, base + 1, base + 3], [base, base + 1, base + 2, 0]])
    ).astype(np.int32)
    floor = 2.0**-20
    valid, lower, upper = tetrahedra_validity(points, cells, floor)
    for row in (*range(0, cells.shape[0] - 2, 37), cells.shape[0] - 2):
        determinant = _exact_determinant(points, cells[row])
        assert Fraction(float(lower[row])) <= determinant <= Fraction(float(upper[row]))
        a = points[cells[row, 0]]
        scale = Fraction(floor) ** 2
        for k in range(1, 4):
            edge = [
                Fraction(float(points[cells[row, k], axis])) - Fraction(float(a[axis]))
                for axis in range(3)
            ]
            scale *= sum(value * value for value in edge)
        assert bool(valid[row]) == (determinant > 0 and determinant**2 > scale)
    assert not valid[-2]
    with pytest.raises(ValueError):
        tetrahedra_validity(np.full_like(points, 2.0**200), cells, floor)
