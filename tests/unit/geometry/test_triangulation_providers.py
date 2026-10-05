#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import numpy as np
import pytest

from phydrax.geometry import DelaunayTriangulation


@pytest.mark.parametrize("dimension", [2, 3])
def test_qhull_delaunay_covers_the_hull_with_oriented_canonical_simplices(
    dimension: int,
) -> None:
    rng = np.random.default_rng(5)
    corners = np.stack(
        np.meshgrid(*([0.0, 1.0],) * dimension, indexing="ij"), axis=-1
    ).reshape(-1, dimension)
    points = np.concatenate((corners, rng.uniform(0.05, 0.95, (40, dimension))))
    triangulation = DelaunayTriangulation(points, provider="qhull")
    simplices = np.asarray(triangulation.simplices)
    vertices = points[simplices]
    determinants = np.linalg.det(np.swapaxes(vertices[:, 1:] - vertices[:, :1], 1, 2))
    evidence = triangulation.evidence

    assert np.all(determinants > 0.0)
    # Unit-cube convex hull volume, independent of the triangulation.
    np.testing.assert_allclose(
        np.sum(determinants) / np.prod(np.arange(1, dimension + 1)), 1.0
    )
    np.testing.assert_array_equal(np.unique(simplices), np.arange(points.shape[0]))
    np.testing.assert_array_equal(simplices, simplices[np.lexsort(simplices.T[::-1])])
    assert evidence.provider == "qhull"
    assert evidence.predicate_mode == "filtered"
    assert evidence.duplicate_count == 0 and evidence.redundant_count == 0
    repeated = DelaunayTriangulation(points, provider="qhull")
    np.testing.assert_array_equal(repeated.simplices, simplices)
    assert repeated.evidence.evidence_id == evidence.evidence_id


def test_qhull_delaunay_maps_duplicates_and_refuses_invalid_selectors() -> None:
    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)))
    triangulation = DelaunayTriangulation(points, provider="qhull")

    assert int(triangulation.vertex_map[4]) == 1
    assert triangulation.evidence.duplicate_count == 1
    assert 4 not in np.asarray(triangulation.simplices)
    with pytest.raises(ValueError, match="provider"):
        DelaunayTriangulation(points, provider="exact")  # ty: ignore[invalid-argument-type]
