# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Compiled local preparation, fixed-support refresh and coordinate derivatives.

Oracles are independent of the stencil kernel: exact monomial derivatives,
analytic derivatives of a nonpolynomial field, central coordinate
differences, and the neighborhood's own selection-gap trust margin.
"""

from __future__ import annotations

import functools
import itertools
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.discretization.meshfree import (
    fit_chart_stencils,
    LocalStencilPolicy,
    LocalStencilRefreshStatus,
    MeshfreeFunctional,
    MeshfreeNeighborhoodPlan,
    MeshfreeOperator,
    MeshfreeRowStatus,
    prepare_chart_stencils,
    prepare_local_stencils,
    PreparedLocalStencils,
    PreparedMeshfreeNeighborhood,
    refresh_chart_stencils,
    refresh_local_stencils,
)
from phydrax.discretization.meshfree._types import MeshfreeApproximation


def _jittered_grid(dimension: int, per_axis: int, seed: int) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, per_axis)
    grid = np.asarray(list(itertools.product(axis, repeat=dimension)), dtype=np.float64)
    spacing = 2.0 / (per_axis - 1)
    jitter = np.random.default_rng(seed).uniform(-0.2, 0.2, grid.shape) * spacing
    return grid + jitter


def _interior_targets(dimension: int) -> np.ndarray:
    return np.asarray(
        [[0.11, -0.07, 0.05][:dimension], [-0.13, 0.09, -0.04][:dimension]],
        dtype=np.float64,
    )


def _policy(
    approximation: MeshfreeApproximation, degree: int, chunk_rows: int = 128
) -> LocalStencilPolicy:
    return LocalStencilPolicy(
        approximation=approximation, polynomial_degree=degree, chunk_rows=chunk_rows
    )


def _neighbors(dimension: int, degree: int) -> int:
    return 2 * math.comb(dimension + degree, degree)


def test_policy_parameters_belong_to_their_method() -> None:
    gmls = LocalStencilPolicy()
    phs = LocalStencilPolicy(approximation="phs-rbf-fd", polynomial_degree=2)
    assert gmls.weight_kernel == "inverse-square" and gmls.phs_power is None
    assert phs.phs_power == 3 and phs.weight_kernel is None
    with pytest.raises(ValueError, match="no weight kernel"):
        LocalStencilPolicy(approximation="phs-rbf-fd", weight_kernel="wendland-c2")
    with pytest.raises(ValueError, match="no radial power"):
        LocalStencilPolicy(phs_power=5)
    with pytest.raises(ValueError, match="odd"):
        LocalStencilPolicy(approximation="phs-rbf-fd", phs_power=4)


@pytest.mark.parametrize("approximation", ["gmls", "phs-rbf-fd"])
@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_monomial_derivatives_are_reproduced(
    approximation: MeshfreeApproximation, dimension: int, degree: int
) -> None:
    per_axis = {1: 24, 2: 9, 3: 6}[dimension]
    sources = _jittered_grid(dimension, per_axis, seed=dimension + degree)
    targets = _interior_targets(dimension)
    neighborhood = MeshfreeNeighborhoodPlan(
        sources, _neighbors(dimension, degree), targets=targets
    ).prepare()
    first = tuple(int(axis == 0) for axis in range(dimension))
    second = tuple(2 * int(axis == dimension - 1) for axis in range(dimension))
    stencils = prepare_local_stencils(
        neighborhood,
        sources,
        targets,
        (
            MeshfreeFunctional((first,), (1.0,), name="first"),
            MeshfreeFunctional((second,), (1.0,), name="second"),
        ),
        _policy(approximation, degree),
    )
    # u = x0^degree + x_last^degree, differentiated exactly by hand.
    values = sources[:, 0] ** degree + sources[:, -1] ** degree
    first_exact = degree * targets[:, 0] ** (degree - 1) + (
        degree * targets[:, -1] ** (degree - 1) if dimension == 1 else 0.0
    )
    second_exact = degree * (degree - 1) * targets[:, -1] ** (degree - 2) + (
        degree * (degree - 1) * targets[:, 0] ** (degree - 2) if dimension == 1 else 0.0
    )
    np.testing.assert_allclose(
        MeshfreeOperator(stencils, 0).apply(values), first_exact, atol=1e-7
    )
    np.testing.assert_allclose(
        MeshfreeOperator(stencils, 1).apply(values), second_exact, atol=1e-6
    )


_REFINEMENT = {1: (21, 41, 81, 161), 2: (9, 17, 33, 65), 3: (5, 9, 17, 33)}


def _smooth_field(points: np.ndarray) -> np.ndarray:
    phase = np.sum(points[:, 1:], axis=1) + 0.3 * points[:, 0]
    return np.exp(0.7 * points[:, 0]) * np.cos(phase)


def _smooth_first_derivative(points: np.ndarray) -> np.ndarray:
    phase = np.sum(points[:, 1:], axis=1) + 0.3 * points[:, 0]
    return np.exp(0.7 * points[:, 0]) * (0.7 * np.cos(phase) - 0.3 * np.sin(phase))


@functools.cache
def _refinement_ladder(
    dimension: int, degree: int
) -> tuple[
    np.ndarray, tuple[tuple[float, np.ndarray, PreparedMeshfreeNeighborhood], ...]
]:
    """Interior targets and, per resolution, measured fill distance and support."""
    rng = np.random.default_rng(dimension)
    targets = rng.uniform(-0.5, 0.5, (24, dimension))
    probes = rng.uniform(-0.5, 0.5, (400, dimension))
    ladder = []
    for per_axis in _REFINEMENT[dimension]:
        sources = _jittered_grid(dimension, per_axis, seed=per_axis)
        # Fill distance over the interior region containing every target.
        fill = float(
            jnp.max(
                MeshfreeNeighborhoodPlan(sources, 1, targets=probes).prepare().distances
            )
        )
        neighborhood = MeshfreeNeighborhoodPlan(
            sources, _neighbors(dimension, degree), targets=targets
        ).prepare()
        ladder.append((fill, sources, neighborhood))
    return targets, tuple(ladder)


@pytest.mark.parametrize("approximation", ["gmls", "phs-rbf-fd"])
@pytest.mark.parametrize("degree", [2, 3, 4])
@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_interior_first_derivative_converges_at_polynomial_order(
    approximation: MeshfreeApproximation, dimension: int, degree: int
) -> None:
    # A degree-p local fit differentiates smooth fields with O(h^p) first
    # derivative error; the slope is fitted against measured fill distance
    # over four genuine halvings on interior rows (no boundary-layer rows).
    targets, ladder = _refinement_ladder(dimension, degree)
    direction = tuple(int(axis == 0) for axis in range(dimension))
    fills, errors = [], []
    for fill, sources, neighborhood in ladder:
        stencils = prepare_local_stencils(
            neighborhood,
            sources,
            targets,
            (MeshfreeFunctional((direction,), (1.0,)),),
            _policy(approximation, degree),
        )
        approximation_ = np.asarray(
            MeshfreeOperator(stencils).apply(_smooth_field(sources))
        )
        error = approximation_ - _smooth_first_derivative(targets)
        fills.append(fill)
        errors.append(float(np.max(np.abs(error))))
    ratios = np.asarray(fills[:-1]) / np.asarray(fills[1:])
    assert np.all(ratios > 1.5), "refinement must genuinely halve the fill distance"
    slope = np.polyfit(np.log(fills), np.log(errors), 1)[0]
    assert slope >= degree - 0.5


@pytest.mark.parametrize("approximation", ["gmls", "phs-rbf-fd"])
@pytest.mark.parametrize("chunk_rows", [1, 7, 64])
def test_ragged_chunk_capacity_preserves_weights_and_evidence(
    approximation: MeshfreeApproximation, chunk_rows: int
) -> None:
    points = _jittered_grid(2, 7, seed=3)
    neighborhood = MeshfreeNeighborhoodPlan(points, 12).prepare()
    functionals = (
        MeshfreeFunctional(((1, 0),), (1.0,)),
        MeshfreeFunctional(((2, 0), (0, 2)), (1.0, 1.0)),
    )
    reference = prepare_local_stencils(
        neighborhood, points, points, functionals, _policy(approximation, 2, 49)
    )
    chunked = prepare_local_stencils(
        neighborhood, points, points, functionals, _policy(approximation, 2, chunk_rows)
    )
    for left, right in zip(reference.weights, chunked.weights, strict=True):
        # Batched SVD rounding may differ only near roundoff times conditioning.
        scale = float(jnp.max(jnp.abs(left)))
        np.testing.assert_allclose(left, right, rtol=0.0, atol=1e-10 * scale)
    np.testing.assert_array_equal(reference.evidence.status, chunked.evidence.status)
    np.testing.assert_allclose(
        reference.evidence.condition, chunked.evidence.condition, rtol=1e-9
    )
    assert reference.report.refused_rows == chunked.report.refused_rows == 0


@pytest.mark.parametrize("approximation", ["gmls", "phs-rbf-fd"])
def test_partial_rows_fit_only_their_valid_support(
    approximation: MeshfreeApproximation,
) -> None:
    points = _jittered_grid(2, 6, seed=9)
    neighborhood = MeshfreeNeighborhoodPlan(points, 12).prepare()
    valid = np.asarray(neighborhood.relation.valid).copy()
    valid[:, -3:] = False
    partial = eqx.tree_at(
        lambda item: item.relation.valid, neighborhood, jnp.asarray(valid)
    )
    stencils = prepare_chart_stencils(
        partial,
        neighborhood.offsets(points, points),
        (MeshfreeFunctional(((2, 0), (0, 2)), (1.0, 1.0)),),
        _policy(approximation, 2, 8),
    )
    weights = np.asarray(stencils.weights[0])
    np.testing.assert_array_equal(weights[:, -3:], 0.0)
    quadratic = points[:, 0] ** 2 - points[:, 0] * points[:, 1] + 2 * points[:, 1] ** 2
    np.testing.assert_allclose(
        MeshfreeOperator(stencils).apply(quadratic), 6.0, atol=1e-8
    )


def _square_stencils(
    approximation: MeshfreeApproximation,
) -> tuple[np.ndarray, PreparedLocalStencils]:
    points = _jittered_grid(2, 6, seed=11)
    stencils = prepare_local_stencils(
        MeshfreeNeighborhoodPlan(points, 12).prepare(),
        points,
        points,
        (
            MeshfreeFunctional(((1, 0),), (1.0,)),
            MeshfreeFunctional(((2, 0), (0, 2)), (1.0, 1.0)),
        ),
        _policy(approximation, 2, 9),
    )
    return points, stencils


@pytest.mark.parametrize("approximation", ["gmls", "phs-rbf-fd"])
def test_refresh_at_anchor_reproduces_admitted_weights(
    approximation: MeshfreeApproximation,
) -> None:
    points, stencils = _square_stencils(approximation)
    refreshed = jax.jit(refresh_local_stencils)(stencils, points, points)
    assert int(refreshed.status) == int(LocalStencilRefreshStatus.ACCEPTED)
    for left, right in zip(stencils.weights, refreshed.stencils.weights, strict=True):
        scale = float(jnp.max(jnp.abs(left)))
        np.testing.assert_allclose(left, right, rtol=0.0, atol=1e-12 * scale)


@pytest.mark.parametrize("approximation", ["gmls", "phs-rbf-fd"])
def test_coordinate_jvp_and_vjp_match_central_differences_inside_trust(
    approximation: MeshfreeApproximation,
) -> None:
    points, stencils = _square_stencils(approximation)
    trust = float(jnp.min(stencils.neighborhood.trust_margin))
    assert trust > 0
    direction = np.random.default_rng(5).normal(size=points.shape)
    direction *= 0.2 * trust / np.max(np.linalg.norm(direction, axis=1))

    def laplacian_weights(coordinates: Array) -> Array:
        refreshed = refresh_local_stencils(stencils, coordinates, coordinates)
        return refreshed.stencils.weights[1]

    base = jnp.asarray(points)
    _, tangent = jax.jvp(laplacian_weights, (base,), (jnp.asarray(direction),))
    # Central-difference points move by 1% of the trust margin: same support.
    step = 0.05
    central = (
        laplacian_weights(base + step * direction)
        - laplacian_weights(base - step * direction)
    ) / (2 * step)
    scale = float(jnp.max(jnp.abs(tangent)))
    np.testing.assert_allclose(tangent, central, atol=1e-6 * scale)
    cotangent = jnp.asarray(np.random.default_rng(6).normal(size=tangent.shape))
    _, pullback = jax.vjp(laplacian_weights, base)
    np.testing.assert_allclose(
        jnp.vdot(pullback(cotangent)[0], direction),
        jnp.vdot(cotangent, tangent),
        rtol=1e-9,
    )
    moved = refresh_local_stencils(stencils, base + direction, base + direction)
    assert bool(moved.accepted)
    assert float(jnp.min(moved.support_margin)) > 0


@pytest.mark.parametrize("approximation", ["gmls", "phs-rbf-fd"])
def test_refreshed_operator_reproduces_and_keeps_weighted_adjoint(
    approximation: MeshfreeApproximation,
) -> None:
    points, stencils = _square_stencils(approximation)
    trust = float(jnp.min(stencils.neighborhood.trust_margin))
    moved = points + 0.5 * trust * np.asarray([0.6, -0.8])
    refreshed = refresh_local_stencils(stencils, moved, moved)
    assert bool(refreshed.accepted)
    operator = MeshfreeOperator(refreshed.stencils, 1)
    quadratic = moved[:, 0] ** 2 + 3 * moved[:, 1] ** 2 - moved[:, 0] * moved[:, 1]
    np.testing.assert_allclose(operator.apply(quadratic), 8.0, atol=1e-7)
    x = jnp.sin(jnp.arange(points.shape[0], dtype=jnp.float64))
    y = jnp.cos(jnp.arange(points.shape[0], dtype=jnp.float64))
    np.testing.assert_allclose(
        jnp.vdot(operator.mv(x), y), jnp.vdot(x, operator.transpose_mv(y)), atol=1e-9
    )


def test_refresh_refuses_support_exit_nonfinite_and_refused_rows() -> None:
    points, stencils = _square_stencils("gmls")
    trust = float(jnp.min(stencils.neighborhood.trust_margin))
    shifted = points.copy()
    shifted[0] += 2.0 * trust
    exited = refresh_local_stencils(stencils, shifted, shifted)
    assert int(exited.status) == int(LocalStencilRefreshStatus.SUPPORT_EXCEEDED)
    assert not bool(exited.accepted)
    nonfinite = points.copy()
    nonfinite[3, 1] = np.nan
    invalid = refresh_local_stencils(stencils, nonfinite, nonfinite)
    assert int(invalid.status) == int(LocalStencilRefreshStatus.INVALID_COORDINATES)
    collinear = np.asarray(stencils.anchor_offsets).copy()
    collinear[..., 1] = 0.0
    refused = refresh_chart_stencils(stencils, collinear, displacement=0.0)
    assert int(refused.status) == int(LocalStencilRefreshStatus.ROW_REFUSED)
    assert np.all(
        np.asarray(refused.stencils.evidence.status)
        == int(MeshfreeRowStatus.RANK_DEFICIENT)
    )
    chart = prepare_chart_stencils(
        stencils.neighborhood,
        stencils.anchor_offsets,
        stencils.functionals,
        stencils.policy,
    )
    with pytest.raises(ValueError, match="no anchored points"):
        refresh_local_stencils(chart, points, points)


def test_tied_selection_admits_only_the_unmoved_anchor() -> None:
    # The third and fourth sources tie at distance two from the target.
    sources = np.asarray([[-1.0], [1.0], [2.0], [-2.0], [3.5]])
    targets = np.asarray([[0.0]])
    neighborhood = MeshfreeNeighborhoodPlan(sources, 3, targets=targets).prepare()
    np.testing.assert_array_equal(neighborhood.trust_margin, 0.0)
    stencils = prepare_local_stencils(
        neighborhood,
        sources,
        targets,
        (MeshfreeFunctional(((1,),), (1.0,)),),
        LocalStencilPolicy(polynomial_degree=1),
    )
    assert bool(refresh_local_stencils(stencils, sources, targets).accepted)
    moved = refresh_local_stencils(stencils, sources, targets + 1e-9)
    assert int(moved.status) == int(LocalStencilRefreshStatus.SUPPORT_EXCEEDED)


def test_candidate_overflow_refuses_neighborhood_preparation() -> None:
    points = _jittered_grid(2, 8, seed=2)
    with pytest.raises(ValueError, match="incomplete"):
        MeshfreeNeighborhoodPlan(points, 12, maximum_candidates=13).prepare()


@pytest.mark.parametrize("count", [40, 300], ids=["below-floor", "ragged-bucket"])
def test_bucketed_storage_matches_logical_capacity_query_and_fit(count: int) -> None:
    from phydrax.discretization.spatial import MortonNeighborQueryPlan

    points = np.random.default_rng(count).uniform(-1.0, 1.0, (count, 2))
    plan = MeshfreeNeighborhoodPlan(points, 12)
    prepared = plan.prepare()
    assert prepared.storage_capacity == ((64, 64) if count == 40 else (512, 512))
    # Independent reference: the exact query at logical capacity in the
    # physical (non-canonical) address, and a brute-force distance sort.
    reference = MortonNeighborQueryPlan(plan.address, count, count, 13).query(
        jnp.asarray(points), jnp.asarray(points)
    )
    np.testing.assert_array_equal(
        prepared.relation.source_indices, np.asarray(reference.source_indices)[:, :12]
    )
    brute = np.sort(np.linalg.norm(points[:, None] - points[None], axis=-1), axis=1)
    np.testing.assert_allclose(prepared.distances, brute[:, :12], rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        prepared.trust_margin, (brute[:, 12] - brute[:, 11]) / 4.0, rtol=0, atol=1e-15
    )
    functionals = (MeshfreeFunctional(((2, 0), (0, 2)), (1.0, 1.0)),)
    policy = _policy("gmls", 2, 64)
    stencils = prepare_local_stencils(prepared, points, points, functionals, policy)
    unpadded, evidence = jax.jit(
        lambda offsets, valid, coefficients: fit_chart_stencils(
            offsets, valid, stencils.terms, coefficients, policy
        )
    )(stencils.offsets, prepared.relation.valid, stencils.coefficients)
    scale = float(jnp.max(jnp.abs(unpadded)))
    # Below the chunk capacity the logical batch differs from storage, so
    # batched SVD rounding may differ at roundoff; otherwise results agree.
    np.testing.assert_allclose(
        stencils.weights[0], unpadded[:, 0, :], rtol=0, atol=1e-13 * scale
    )
    np.testing.assert_array_equal(stencils.evidence.status, evidence.status)
    assert stencils.report.refused_rows == int(jnp.sum(evidence.status != 0))


def test_knn_tie_crossing_refuses_with_nan_sensitivities() -> None:
    # The third and fourth sources tie at distance two: the admitted anchor has
    # no open coordinate neighborhood with its selection, and any motion
    # crosses the tie. Neither publishes a finite (silently masked) derivative.
    sources = np.asarray([[-1.0], [1.0], [2.0], [-2.0], [3.5]])
    targets = np.asarray([[0.0]])
    neighborhood = MeshfreeNeighborhoodPlan(sources, 3, targets=targets).prepare()
    stencils = prepare_local_stencils(
        neighborhood,
        sources,
        targets,
        (MeshfreeFunctional(((1,),), (1.0,)),),
        LocalStencilPolicy(polynomial_degree=1),
    )

    def weights(shift: Array) -> Array:
        refreshed = refresh_local_stencils(stencils, sources, targets + shift)
        return refreshed.stencils.weights[0]

    anchored, tangent = jax.jvp(weights, (jnp.asarray(0.0),), (jnp.asarray(1.0),))
    assert np.all(np.isfinite(anchored)) and np.all(np.isnan(tangent))
    crossed = refresh_local_stencils(stencils, sources, targets + 1e-9)
    assert int(crossed.status) == int(LocalStencilRefreshStatus.SUPPORT_EXCEEDED)
    value, tangent = jax.jvp(weights, (jnp.asarray(1e-9),), (jnp.asarray(1.0),))
    assert np.all(np.isnan(value)) and np.all(np.isnan(tangent))
    _, pullback = jax.vjp(weights, jnp.asarray(1e-9))
    assert np.isnan(pullback(jnp.ones_like(value))[0])
