#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from math import factorial, sqrt

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from phydrax.geometry import DelaunayTriangulation, SimplexQualitySubcomplex


def _measures(points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    corners = points[simplices]
    edges = np.swapaxes(corners[:, 1:] - corners[:, :1], 1, 2)
    return np.abs(np.linalg.det(edges)) / factorial(points.shape[1])


def _facet_component_count(simplices: np.ndarray) -> int:
    facets = {}
    pairs = []
    for index, simplex in enumerate(simplices):
        for drop in range(simplices.shape[1]):
            key = tuple(sorted(np.delete(simplex, drop)))
            if key in facets:
                pairs.append((facets[key], index))
            facets[key] = index
    indices = np.asarray(pairs, dtype=np.intp).reshape(-1, 2)
    rows, columns = indices[:, 0], indices[:, 1]
    graph = coo_matrix(
        (np.ones(rows.size, dtype=np.float64), (rows, columns)),
        shape=(len(simplices), len(simplices)),
    )
    return connected_components(graph.tocsr(), directed=False)[0]


def test_quality_is_one_for_regular_and_vanishes_for_flat_simplices() -> None:
    regular = np.asarray(
        ((1.0, 1.0, 1.0), (1.0, -1.0, -1.0), (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0))
    )
    # A sliver: four corners of a unit square, two lifted by 0.01.
    sliver = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.01), (1.0, 1.0, 0.0), (0.0, 1.0, 0.01))
    )
    points = np.concatenate((regular, sliver))
    screened = SimplexQualitySubcomplex(
        points, np.asarray(((0, 1, 2, 3), (4, 5, 6, 7))), minimum_quality=0.0
    )
    # Volume over the regular tetrahedron of the RMS edge (l^3 / (6 sqrt 2)).
    edges = sliver[[0, 0, 0, 1, 1, 2]] - sliver[[1, 2, 3, 2, 3, 3]]
    rms = sqrt(np.mean(np.sum(edges**2, axis=1)))
    expected = _measures(sliver, np.arange(4)[None])[0] / (rms**3 / (6.0 * sqrt(2.0)))

    np.testing.assert_allclose(screened.quality, (1.0, expected), rtol=1e-12)
    assert expected < 0.05
    flat = SimplexQualitySubcomplex(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 1.0))),
        np.asarray(((0, 1, 2), (0, 1, 3))),
        minimum_quality=0.0,
    )
    assert flat.evidence.degenerate_count == 1
    np.testing.assert_array_equal(flat.simplices, ((0, 1, 3),))


def test_slivers_of_a_jittered_lattice_are_excluded_with_measure_evidence() -> None:
    side = 6
    grid = np.stack(np.meshgrid(*(np.linspace(0.0, 1.0, side),) * 3, indexing="ij"), -1)
    points = grid.reshape(-1, 3)
    interior = ~np.any(np.isclose(points, 0.0) | np.isclose(points, 1.0), axis=1)
    points[interior] += np.random.default_rng(3).uniform(-0.03, 0.03, (interior.sum(), 3))
    simplices = np.asarray(DelaunayTriangulation(points, provider="qhull").simplices)
    screened = SimplexQualitySubcomplex(points, simplices, minimum_quality=0.1)
    evidence = screened.evidence
    measures = _measures(points, simplices)
    usable = screened.quality > 1e-12
    excluded = usable & ~screened.retained

    assert evidence.excluded_count == np.count_nonzero(excluded) > 0
    assert evidence.simplex_count == simplices.shape[0]
    np.testing.assert_array_equal(screened.simplices, simplices[screened.retained])
    np.testing.assert_allclose(
        evidence.excluded_measure_fraction,
        measures[excluded].sum() / measures[usable].sum(),
    )
    # Every point stays a vertex and the retained simplices stay facet-connected.
    np.testing.assert_array_equal(
        np.unique(screened.simplices), np.arange(points.shape[0])
    )
    assert _facet_component_count(np.asarray(screened.simplices)) == 1
    restored = screened.retained & (screened.quality < 0.1)
    assert evidence.restored_count == np.count_nonzero(restored)
    assert np.all(screened.quality[screened.retained & ~restored] >= 0.1)


def test_needed_low_quality_simplices_are_restored() -> None:
    points = np.asarray(
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (0.52, 0.52),  # nearly on the facet (1, 2): thin bridge
            (1.3, 1.3),
            (0.5, -0.02),  # only reached through a thin triangle
        )
    )
    simplices = np.asarray(((0, 1, 2), (1, 3, 2), (2, 3, 4), (0, 5, 1)))
    screened = SimplexQualitySubcomplex(points, simplices, minimum_quality=0.2)

    assert screened.quality[1] < 0.2 and screened.quality[3] < 0.2
    # The bridge keeps (0, 1, 2) and (2, 3, 4) facet-connected; the cap is the
    # only simplex covering point 5.
    np.testing.assert_array_equal(screened.retained, (True, True, True, True))
    assert screened.evidence.restored_count == 2
    assert screened.evidence.excluded_count == 0
    with pytest.raises(ValueError, match="minimum_quality"):
        SimplexQualitySubcomplex(points, simplices, minimum_quality=1.0)
