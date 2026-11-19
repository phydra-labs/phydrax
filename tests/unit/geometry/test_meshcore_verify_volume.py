"""The exact volume certificate on Delaunay tetrahedralizations of convex point sets."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from phydrax._meshcore import delaunay_3d, verify_volume


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


def test_delaunay_tetrahedralization_of_a_convex_set_is_certified() -> None:
    points, cells, surface = _convex_case(11)
    verdict = verify_volume(points, cells, surface)
    assert verdict.certified
    assert verdict.boundary_faces == surface.shape[0]


def test_an_inverted_cell_is_reported() -> None:
    points, cells, surface = _convex_case(12)
    cells[7, [0, 1]] = cells[7, [1, 0]]
    verdict = verify_volume(points, cells, surface)
    assert not verdict.certified
    assert verdict.nonpositive_cells == 1
    assert verdict.first_nonpositive_cell == 7


def test_a_removed_cell_leaves_unmatched_faces() -> None:
    points, cells, surface = _convex_case(13)
    verdict = verify_volume(points, np.delete(cells, 3, axis=0), surface)
    assert not verdict.certified
    assert verdict.unmatched_boundary_faces > 0


def test_cell_indices_out_of_range_are_refused() -> None:
    points, cells, surface = _convex_case(14)
    cells[0, 0] = points.shape[0]
    with pytest.raises(Exception):
        verify_volume(points, cells, surface)
