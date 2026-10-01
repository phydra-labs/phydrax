# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Consumer-visible reproduction, admission, spatial and Hilbert contracts."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.discretization.meshfree._stencils import chart_stencil_kernel
from phydrax.discretization.meshfree._types import (
    MeshfreeApproximation,
    StencilWeightKernel,
)


def _points() -> Array:
    return jnp.asarray([(x, y) for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0)])


def _polynomial(points: Array) -> Array:
    x, y = points[:, 0], points[:, 1]
    return x * x + 2 * x * y + 3 * y * y + 4 * x - 2 * y + 1


@pytest.mark.parametrize("approximation", ["gmls", "phs-rbf-fd"])
@pytest.mark.parametrize("kernel", ["inverse-square", "wendland-c2"])
def test_cross_target_polynomial_and_normal_mixed_reproduction(
    approximation: MeshfreeApproximation, kernel: StencilWeightKernel
) -> None:
    d = phx.discretization
    points = _points()
    targets = jnp.asarray(((0.2, -0.3), (-0.4, 0.6)))
    neighborhood = d.MeshfreeNeighborhoodPlan(points, 9, targets=targets).prepare()
    normal = np.asarray(((0.6, 0.8), (-0.8, 0.6)))
    functionals = (
        d.MeshfreeFunctional(((0, 0),), (1.0,), name="value"),
        d.MeshfreeFunctional(
            ((1, 0), (0, 1)), (1.0, 1.0), row_coefficients=normal, name="normal"
        ),
        d.MeshfreeFunctional(((1, 1),), (1.0,), name="mixed"),
        d.MeshfreeFunctional(((2, 0), (0, 2)), (1.0, 1.0), name="laplacian"),
    )
    stencils = d.prepare_local_stencils(
        neighborhood,
        points,
        targets,
        functionals,
        d.LocalStencilPolicy(approximation=approximation, weight_kernel=kernel),
    )
    values = _polynomial(points)
    gx = 2 * targets[:, 0] + 2 * targets[:, 1] + 4
    gy = 2 * targets[:, 0] + 6 * targets[:, 1] - 2
    expected = (
        _polynomial(targets),
        normal[:, 0] * gx + normal[:, 1] * gy,
        jnp.full((2,), 2.0),
        jnp.full((2,), 8.0),
    )
    for index, reference in enumerate(expected):
        operator = d.MeshfreeOperator(stencils, index)
        np.testing.assert_allclose(
            eqx.filter_jit(operator.apply)(values), reference, atol=2e-10
        )
    vector = jnp.stack((values, 2 * values), axis=-1)
    np.testing.assert_allclose(
        d.MeshfreeOperator(stencils).apply(vector),
        jnp.stack((expected[0], 2 * expected[0]), axis=-1),
        atol=2e-10,
    )


@pytest.mark.parametrize("kernel", ["inverse-square", "wendland-c2"])
def test_overdetermined_gmls_matches_weighted_noisy_data_fit(
    kernel: StencilWeightKernel,
) -> None:
    d = phx.discretization
    points = _points()
    targets = jnp.asarray(((0.2, -0.3), (-0.4, 0.6)))
    neighborhood = d.MeshfreeNeighborhoodPlan(points, 9, targets=targets).prepare()
    stencils = d.prepare_local_stencils(
        neighborhood,
        points,
        targets,
        (d.MeshfreeFunctional(((0, 0),), (1.0,)),),
        d.LocalStencilPolicy(polynomial_degree=1, weight_kernel=kernel),
    )
    values = np.asarray((0.1, 2.0, -0.7, 0.2, 1.4, -1.8, 0.9, 3.1, -0.3))
    expected = []
    for row, target in enumerate(np.asarray(targets)):
        selected = np.asarray(neighborhood.relation.source_indices[row])
        offsets = np.asarray(points)[selected] - target
        radius = np.linalg.norm(offsets, axis=1)
        scale = np.max(radius)
        ratio = radius / scale
        weight = (
            (1.0 - ratio / 1.1) ** 4 * (4.0 * ratio / 1.1 + 1.0)
            if kernel == "wendland-c2"
            else 1.0 / np.maximum(ratio, 0.25) ** 2
        )
        design = np.concatenate((np.ones((9, 1)), offsets / scale), axis=1)
        fitted = np.linalg.lstsq(
            np.sqrt(weight)[:, None] * design,
            np.sqrt(weight) * values[selected],
            rcond=None,
        )[0]
        expected.append(fitted[0])
    np.testing.assert_allclose(
        d.MeshfreeOperator(stencils).apply(jnp.asarray(values)), expected, atol=1e-12
    )


def test_rank_condition_and_amplification_refusal_are_distinct() -> None:
    d = phx.discretization
    # A genuinely full-rank, nearly-collinear cloud violates a low condition
    # threshold, whereas a collinear cloud fails rank even with a loose limit.
    nearly = jnp.asarray(((-1.0, -1.0), (0.0, 1e-4), (1.0, 1.0), (0.3, 0.3 - 2e-4)))
    value = (d.MeshfreeFunctional(((0, 0),), (1.0,)),)
    neighborhood = d.MeshfreeNeighborhoodPlan(
        nearly, 4, targets=jnp.zeros((1, 2))
    ).prepare()
    policy = d.LocalStencilPolicy(
        polynomial_degree=1, condition_limit=2.0, acceptance="mask"
    )
    masked = d.prepare_local_stencils(
        neighborhood, nearly, jnp.zeros((1, 2)), value, policy
    )
    assert int(masked.evidence.rank[0]) == 3
    assert int(masked.evidence.status[0]) == int(d.MeshfreeRowStatus.ILL_CONDITIONED)
    assert np.isfinite(float(masked.evidence.condition[0]))
    np.testing.assert_array_equal(masked.weights[0], 0.0)
    with pytest.raises(ValueError, match="ILL_CONDITIONED"):
        d.prepare_local_stencils(
            neighborhood,
            nearly,
            jnp.zeros((1, 2)),
            value,
            d.LocalStencilPolicy(polynomial_degree=1, condition_limit=2.0),
        )
    line = nearly.at[:, 1].set(0.0)
    relation = d.MeshfreeNeighborhoodPlan(line, 4, targets=jnp.zeros((1, 2))).prepare()
    rank_failed = d.prepare_local_stencils(
        relation,
        line,
        jnp.zeros((1, 2)),
        value,
        d.LocalStencilPolicy(polynomial_degree=1, acceptance="mask"),
    )
    assert int(rank_failed.evidence.status[0]) == int(d.MeshfreeRowStatus.RANK_DEFICIENT)
    points = _points()
    relation = d.MeshfreeNeighborhoodPlan(points, 9, targets=jnp.zeros((1, 2))).prepare()
    with pytest.raises(ValueError, match="EXCESSIVE_AMPLIFICATION"):
        d.prepare_local_stencils(
            relation,
            points,
            jnp.zeros((1, 2)),
            value,
            d.LocalStencilPolicy(amplification_limit=0.1),
        )


def test_device_fixed_support_kernel_has_same_reproduction_and_masked_rank_failure() -> (
    None
):
    d = phx.discretization
    points = _points()
    offsets = jnp.stack((points, points.at[:, 1].set(0.0)))
    coefficients = jnp.ones((2, 1, 1))
    weights, evidence = eqx.filter_jit(chart_stencil_kernel)(
        offsets,
        jnp.ones((2, 9), dtype=bool),
        ((1, 1),),
        coefficients,
        d.LocalStencilPolicy(acceptance="mask"),
    )
    np.testing.assert_allclose(
        jnp.sum(weights[0, 0] * _polynomial(points)), 2.0, atol=1e-10
    )
    assert int(evidence.status[1]) == int(d.MeshfreeRowStatus.RANK_DEFICIENT)
    np.testing.assert_array_equal(weights[1], 0.0)


def test_cross_target_coordinate_transpose_and_weighted_hilbert_adjoint() -> None:
    d = phx.discretization
    points = _points()
    targets = jnp.asarray(((0.2, -0.3), (-0.4, 0.6)))
    relation = d.MeshfreeNeighborhoodPlan(points, 9, targets=targets).prepare()
    stencils = d.prepare_local_stencils(
        relation,
        points,
        targets,
        (d.MeshfreeFunctional(((1, 0),), (1.0,)),),
        d.LocalStencilPolicy(),
    )
    source_mass = jnp.linspace(0.5, 2.0, 9)
    target_mass = jnp.asarray((3.0, 0.7))
    source = phx.linalg.ArraySpace((9,), pairing=phx.linalg.DiagonalPairing(source_mass))
    target = phx.linalg.ArraySpace((2,), pairing=phx.linalg.DiagonalPairing(target_mass))
    operator = d.MeshfreeOperator(stencils, source=source, target=target)
    x = jnp.sin(jnp.arange(9, dtype=jnp.float64))
    y = jnp.asarray((0.3, -0.7))
    transpose = operator.transpose_mv(y)
    np.testing.assert_allclose(
        jnp.vdot(operator.mv(x), y), jnp.vdot(x, transpose), atol=1e-12
    )
    adjoint = operator.adjoint_mv(y)
    np.testing.assert_allclose(
        jnp.vdot(operator.mv(x), target_mass * y),
        jnp.vdot(x, source_mass * adjoint),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        adjoint, operator.transpose_mv(target_mass * y) / source_mass, atol=1e-12
    )
    _, pullback = jax.vjp(operator.apply, x)
    np.testing.assert_allclose(pullback(y)[0], transpose, atol=1e-12)


def test_neighbor_gap_certifies_motion_of_sources_and_targets() -> None:
    d = phx.discretization
    sources = jnp.asarray(((0.0,), (1.0,), (3.0,)))
    targets = jnp.asarray(((0.2,),))
    prepared = d.MeshfreeNeighborhoodPlan(sources, 2, targets=targets).prepare()
    np.testing.assert_allclose(prepared.trust_margin, 0.5)
    np.testing.assert_array_equal(prepared.relation.source_indices, ((0, 1),))
    # At delta .4 < .5 both source and target movement preserve the selected set.
    moved = sources + jnp.asarray(((-0.4,), (-0.4,), (-0.4,)))
    moved_target = targets + 0.4
    again = d.MeshfreeNeighborhoodPlan(moved, 2, targets=moved_target).prepare()
    assert set(np.asarray(again.relation.source_indices[0])) == {0, 1}
    tied = d.MeshfreeNeighborhoodPlan(
        jnp.asarray(((-1.0,), (1.0,), (2.0,))), 1, targets=jnp.zeros((1, 1))
    ).prepare()
    np.testing.assert_array_equal(tied.trust_margin, 0.0)


def test_spatial_admission_duplicates_active_sources_and_radius_capacity() -> None:
    d = phx.discretization
    with pytest.raises(ValueError, match="duplicates"):
        d.MeshfreeNeighborhoodPlan(jnp.asarray(((0.0,), (0.0,))), 1)
    with pytest.raises(ValueError, match="finite"):
        d.MeshfreeNeighborhoodPlan(jnp.asarray(((0.0,), (jnp.nan,))), 1)
    with pytest.raises(TypeError, match="integer"):
        d.MeshfreeNeighborhoodPlan(jnp.asarray(((0.0,), (1.0,))), True)
    points = jnp.asarray(((0.0,), (1.0,), (2.0,), (4.0,)))
    prepared = d.MeshfreeNeighborhoodPlan(
        points,
        2,
        targets=jnp.asarray(((1.1,),)),
        source_active=jnp.asarray((True, False, True, True)),
    ).prepare()
    assert set(np.asarray(prepared.relation.source_indices[0])) == {0, 2}
    edges = d.MeshfreeEdgeRelationPlan(points, 1.1, 2).prepare().relation
    valid = np.asarray(edges.valid)
    pairs = set(
        zip(
            np.asarray(edges.source_indices)[valid],
            np.asarray(edges.target_indices)[valid],
            strict=True,
        )
    )
    assert pairs == {(0, 1), (1, 2)}
    with pytest.raises(ValueError, match="capacity"):
        d.MeshfreeEdgeRelationPlan(points, 1.1, 1).prepare()


def test_point_cloud_mixed_derivative_transpose_and_dual_refusal() -> None:
    d = phx.discretization
    points = _points()
    prepared = d.PointCloudPlan(points, jnp.arange(1.0, 10.0), neighbors=9).prepare()
    values = _polynomial(points)
    np.testing.assert_allclose(
        prepared.mixed_partial_derivative(values, multi_index=(1, 1)), 2.0, atol=1e-10
    )
    np.testing.assert_allclose(prepared.laplacian(values), 8.0, atol=1e-10)
    y = jnp.cos(jnp.arange(9, dtype=jnp.float64))
    np.testing.assert_allclose(
        jnp.vdot(prepared.partial_derivative(values, axis=0), y),
        jnp.vdot(values, prepared.transpose_partial_derivative(y, axis=0)),
        atol=1e-10,
    )
    with pytest.raises(ValueError, match="dual"):
        prepared.divergence(jnp.ones((9, 2)), dual=True)
