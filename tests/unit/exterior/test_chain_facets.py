#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._cell_geometry_validity import _simplex_child_maps
from phydrax.discretization._simplicial_locator import PreparedSimplicialCellLocator


def _locator(
    coordinates: tuple[tuple[float, float], ...],
    cells: tuple[tuple[int, int, int], ...],
) -> PreparedSimplicialCellLocator:
    mesh = phx.discretization.CellMesh(
        jnp.asarray(coordinates, dtype=jnp.float64),
        (
            phx.discretization.CellBlock(
                "triangles", "triangle", jnp.asarray(cells, dtype=jnp.int32)
            ),
        ),
    )
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
    ).prepare()
    return PreparedSimplicialCellLocator(
        phx.discretization.fem.prepare_finite_element_cell_map(discretization, 0),
        discretization.default_runtime.coordinates,
        phx.discretization.SimplicialLocationPolicy(len(cells), 8, 1),
    )


def _square() -> PreparedSimplicialCellLocator:
    return _locator(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)),
        ((0, 1, 2), (1, 3, 2)),
    )


def test_affine_facet_crossing_is_exact_under_jit() -> None:
    locator = _square()
    start = jnp.asarray(((0.1, 0.1),), dtype=jnp.float64)
    end = jnp.asarray(((0.9, 0.9),), dtype=jnp.float64)
    result = jax.jit(lambda a, b: locator.locate_segment(a, b, maximum_segments=2))(
        start, end
    )
    np.testing.assert_array_equal(result.cell_ids, ((0, 1),))
    np.testing.assert_allclose(result.intervals, (((0.0, 0.5), (0.5, 1.0)),), atol=1e-14)
    assert bool(result.successful[0])
    assert bool(result.crossed[0])
    assert not bool(result.overflow[0])
    assert not bool(result.exited[0])


def test_facet_capacity_reports_truncated_connected_path() -> None:
    result = _square().locate_segment(
        jnp.asarray(((0.1, 0.1),)),
        jnp.asarray(((0.9, 0.9),)),
        maximum_segments=1,
    )
    np.testing.assert_array_equal(result.counts, (1,))
    np.testing.assert_allclose(result.intervals[0, 0], (0.0, 0.5), atol=1e-14)
    assert bool(result.overflow[0])
    assert not bool(result.successful[0])
    assert not bool(result.exited[0])


def test_domain_exit_retains_exact_last_facet_parameter() -> None:
    locator = _locator(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), ((0, 1, 2),))
    result = locator.locate_segment(
        jnp.asarray(((0.2, 0.2),)),
        jnp.asarray(((1.2, 0.2),)),
        maximum_segments=2,
    )
    np.testing.assert_allclose(result.intervals[0, 0], (0.0, 0.6), atol=1e-14)
    np.testing.assert_array_equal(result.valid, ((True, False),))
    assert bool(result.exited[0])
    assert bool(result.successful[0])
    assert not bool(result.overflow[0])


def test_vertex_tie_enters_non_face_neighbor_without_jitter() -> None:
    locator = _locator(
        ((0.0, 0.0), (-1.0, 0.0), (0.0, -1.0), (1.0, 0.0), (0.0, 1.0)),
        ((0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1)),
    )
    result = locator.locate_segment(
        jnp.asarray(((-0.4, -0.2),)),
        jnp.asarray(((0.4, 0.2),)),
        maximum_segments=2,
    )
    np.testing.assert_array_equal(result.cell_ids, ((0, 2),))
    np.testing.assert_allclose(result.intervals, (((0.0, 0.5), (0.5, 1.0)),), atol=1e-14)
    assert bool(result.tied[0, 0])
    assert bool(result.successful[0])


def test_facet_aligned_path_uses_lowest_cell_deterministically() -> None:
    result = _square().locate_segment(
        jnp.asarray(((0.8, 0.2),)),
        jnp.asarray(((0.2, 0.8),)),
        maximum_segments=2,
    )
    np.testing.assert_array_equal(result.cell_ids, ((0, -1),))
    np.testing.assert_array_equal(result.counts, (1,))
    np.testing.assert_allclose(result.intervals[0, 0], (0.0, 1.0), atol=1e-14)
    assert bool(result.tied[0, 0])
    assert bool(result.successful[0])


def test_zero_segment_is_one_valid_stationary_interval() -> None:
    points = jnp.asarray(((0.2, 0.3),), dtype=jnp.float64)
    result = _square().locate_segment(points, points, maximum_segments=2)
    np.testing.assert_array_equal(result.cell_ids, ((0, -1),))
    np.testing.assert_allclose(result.intervals[0, 0], (0.0, 1.0), atol=1e-14)
    assert bool(result.successful[0])
    assert not bool(result.crossed[0])


def test_outside_start_does_not_reenter_disconnected_path() -> None:
    result = _square().locate_segment(
        jnp.asarray(((-0.2, 0.2),)),
        jnp.asarray(((0.2, 0.2),)),
        maximum_segments=2,
    )
    np.testing.assert_array_equal(result.counts, (0,))
    assert not bool(result.successful[0])
    assert not bool(result.exited[0])
    assert not bool(result.overflow[0])


@pytest.mark.parametrize("dimension", (1, 4, 5))
def test_generic_simplex_subdivision_partitions_reference_volume(dimension: int) -> None:
    origins, matrices = _simplex_child_maps(dimension)
    np.testing.assert_allclose(
        np.sum(np.abs(np.linalg.det(matrices))), 1.0, rtol=0.0, atol=1e-14
    )
    barycentric = np.random.default_rng(73).dirichlet(
        np.ones((dimension + 1,), dtype=np.float64), size=100
    )
    points = barycentric[:, 1:]
    membership = []
    for origin, matrix in zip(origins, matrices, strict=True):
        local = np.linalg.solve(matrix, (points - origin).T).T
        membership.append(
            np.all(local >= -1e-13, axis=1) & (np.sum(local, axis=1) <= 1 + 1e-13)
        )
    np.testing.assert_array_equal(
        np.sum(np.stack(membership, axis=1), axis=1),
        np.ones((points.shape[0],), dtype=np.int64),
    )


@pytest.mark.parametrize("dimension", (4, 5))
def test_generic_dimension_locator_walks_shared_simplex_facet(dimension: int) -> None:
    coordinates = np.concatenate(
        (
            np.zeros((1, dimension), dtype=np.float64),
            np.eye(dimension, dtype=np.float64),
            np.ones((1, dimension), dtype=np.float64),
        ),
        axis=0,
    )
    cells = np.asarray(
        (
            tuple(range(dimension + 1)),
            (1, dimension + 1, *range(2, dimension + 1)),
        ),
        dtype=np.int32,
    )
    mesh = phx.discretization.CellMesh.from_simplices(
        coordinates, cells, dimension=dimension
    )
    prepared = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u",
            phx.discretization.fem.form_element(
                f"simplex:{dimension}",
                0,
                1,
                family="trimmed",
                twist="untwisted",
                proxy="scalar",
            ),
        ),
    ).prepare()
    locator = PreparedSimplicialCellLocator(
        phx.discretization.fem.prepare_finite_element_cell_map(prepared, 0),
        prepared.default_runtime.coordinates,
        phx.discretization.SimplicialLocationPolicy(2, 8, 1),
    )
    first = jnp.full((1, dimension), 0.5 / dimension, dtype=jnp.float64)
    last = jnp.full((1, dimension), 1.5 / dimension, dtype=jnp.float64)
    result = jax.jit(lambda a, b: locator.locate_segment(a, b, maximum_segments=2))(
        first, last
    )
    np.testing.assert_array_equal(result.cell_ids, ((0, 1),))
    np.testing.assert_allclose(
        result.intervals, (((0.0, 0.5), (0.5, 1.0)),), rtol=0.0, atol=2e-13
    )
    assert bool(result.successful[0])
    assert not bool(result.overflow[0])
