# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization import TopologyEpoch
from phydrax.discretization.meshfree._surface_transfer import SurfaceTransferPlan
from phydrax.discretization.meshfree._transfer import (
    PointTransferPlan,
    PointTransferRequest,
    PointTransferStatus,
    PreparedPointTransfer,
)
from phydrax.sparse import EdgeRelation, RowRelation


def _cells(count: int, /) -> tuple[np.ndarray, np.ndarray]:
    """Midpoints and widths of ``count`` uniform cells of [0, 1] (exact to degree 1)."""
    return (np.arange(count) + 0.5) / count, np.full(count, 1.0 / count)


def _simpson(intervals: int, /) -> tuple[np.ndarray, np.ndarray]:
    """Composite Simpson nodes and positive weights of [0, 1] (exact to degree 3)."""
    weights = np.where(np.arange(intervals + 1) % 2 == 1, 4.0, 2.0)
    weights[[0, -1]] = 1.0
    return np.linspace(0.0, 1.0, intervals + 1), weights / (3.0 * intervals)


def _nearest_routes(
    sources: np.ndarray, targets: np.ndarray, neighbors: int, /
) -> tuple[RowRelation, np.ndarray]:
    order = np.argsort(np.abs(targets[:, None] - sources[None, :]), axis=1, kind="stable")
    slots = np.sort(order[:, :neighbors], axis=1).astype(np.int32)
    offsets = (sources[slots] - targets[:, None])[..., None]
    return RowRelation(slots, source_size=sources.size), offsets


def _full_routes(sources: int, targets: int, /) -> EdgeRelation:
    rows, columns = np.divmod(np.arange(sources * targets), sources)
    return EdgeRelation(
        columns.astype(np.int32),
        rows.astype(np.int32),
        source_size=sources,
        target_size=targets,
    )


def _hat_weights(sources: np.ndarray, targets: np.ndarray, /) -> np.ndarray:
    """Nonnegative linear-interpolation base on full routes (not conservative)."""
    width = sources[1] - sources[0]
    return np.maximum(1.0 - np.abs(targets[:, None] - sources[None, :]) / width, 0.0)


def _joint_line(
    sources: int,
    targets: int,
    /,
    *,
    request: PointTransferRequest,
    scale: float = 1.0,
    quadrature: Callable[[int], tuple[np.ndarray, np.ndarray]] = _cells,
) -> PreparedPointTransfer:
    x, old = quadrature(sources)
    y, new = quadrature(targets)
    return PointTransferPlan(
        _full_routes(x.size, y.size),
        _hat_weights(x, y).reshape(-1),
        old,
        scale * new,
        source_id="line-old",
        target_id="line-new",
        request=request,
        offsets=(x[None, :] - y[:, None]).reshape(-1, 1),
    ).prepare()


def test_equal_area_does_not_imply_constant_preservation() -> None:
    relation = RowRelation(np.asarray([[0, 1], [0, 1]], dtype=np.int32), source_size=2)
    route = PointTransferPlan(
        relation,
        np.asarray([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64),
        np.asarray([1.0, 1.0], dtype=np.float64),
        np.asarray([1.0, 1.0], dtype=np.float64),
        source_id="old",
        target_id="new",
        request=PointTransferRequest("conservative-positive"),
    ).prepare()
    assert route.admitted and route.evidence.provider == "conic"
    assert route.transfer is not None
    assert route.transfer.properties.conservative
    assert route.transfer.properties.positivity_preserving
    assert not route.transfer.properties.constant_preserving
    # Minimum change of the over-allocated column splits its excess equally
    # (to the interior-point accuracy of the native conic provider).
    np.testing.assert_allclose(
        route.apply(np.asarray([1.0, 1.0], dtype=np.float64)), [1.5, 0.5], atol=2e-5
    )
    np.testing.assert_allclose(route.evidence.conservation_residual, 0.0, atol=1e-10)


def test_signed_correction_conserves_content_and_has_correct_reverse_maps() -> None:
    relation = RowRelation(
        np.asarray([[0, 1], [0, 1], [0, 1]], dtype=np.int32), source_size=2
    )
    base = np.asarray([[1.1, -0.1], [0.3, 0.7], [-0.2, 1.2]], dtype=np.float64)
    old = np.asarray([2.0, 3.0], dtype=np.float64)
    new = np.asarray([1.0, 2.0, 4.0], dtype=np.float64)
    route = PointTransferPlan(
        relation,
        base,
        old,
        new,
        source_id="old",
        target_id="new",
        request=PointTransferRequest("conservative-signed"),
    ).prepare()
    assert route.admitted and route.evidence.provider == "minimum-norm"
    # Independent minimum-change oracle: each column is one equation, so the
    # correction is the projection of the residual onto that column's weights.
    columns = np.tile([0, 1], 3)
    rows = np.repeat([0, 1, 2], 2)
    flat = base.reshape(-1)
    excess = old - np.bincount(columns, weights=new[rows] * flat, minlength=2)
    norm = np.bincount(columns, weights=new[rows] ** 2, minlength=2)
    expected = flat + new[rows] * (excess / norm)[columns]
    np.testing.assert_allclose(route.evidence.coefficients, expected, atol=1e-10)
    c, y = jnp.array([1.3, -0.4]), jnp.array([0.2, -0.3, 0.7])
    mapped = route.apply(c)
    np.testing.assert_allclose(
        jnp.vdot(route.target_measures, mapped),
        jnp.vdot(route.source_measures, c),
        atol=1e-12,
    )
    assert route.transfer is not None
    dual = route.transfer.dual_pullback_operator
    hilbert = route.transfer.hilbert_adjoint_operator
    assert dual is not None and hilbert is not None
    np.testing.assert_allclose(jnp.vdot(mapped, y), jnp.vdot(c, dual.mv(y)), atol=1e-12)
    np.testing.assert_allclose(
        route.transfer.target.vector_space.inner(mapped, y),
        route.transfer.source.vector_space.inner(c, hilbert.mv(y)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        route.apply_content(route.source_measures * c),
        route.target_measures * mapped,
        atol=1e-12,
    )


def test_uncovered_source_is_refused_with_coverage_evidence() -> None:
    uncovered = EdgeRelation(
        np.asarray([0], dtype=np.int32),
        np.asarray([0], dtype=np.int32),
        source_size=2,
        target_size=1,
    )
    route = PointTransferPlan(
        uncovered,
        np.asarray([1.0], dtype=np.float64),
        np.asarray([1.0, 1.0], dtype=np.float64),
        np.asarray([2.0], dtype=np.float64),
        source_id="old",
        target_id="new",
        request=PointTransferRequest("conservative-signed"),
    ).prepare()
    assert route.evidence.status is PointTransferStatus.UNCOVERED_SOURCE
    assert route.evidence.uncovered_sources == (1,)
    assert route.evidence.provider == "none" and route.transfer is None
    with pytest.raises(ValueError, match="UNCOVERED_SOURCE"):
        route.apply(np.ones(2))


def test_positive_request_corrects_a_signed_base_to_nonnegative_conservation() -> None:
    relation = RowRelation(np.asarray([[0, 1], [0, 1]], dtype=np.int32), source_size=2)
    route = PointTransferPlan(
        relation,
        np.asarray([[1.4, -0.4], [-0.3, 1.3]], dtype=np.float64),
        np.asarray([1.0, 1.0], dtype=np.float64),
        np.asarray([1.0, 1.0], dtype=np.float64),
        source_id="old",
        target_id="new",
        request=PointTransferRequest("conservative-positive"),
    ).prepare()
    assert route.admitted and route.evidence.nonnegative and route.evidence.conservative
    assert route.evidence.minimum_coefficient >= 0.0
    assert route.evidence.correction_norm > 0.0


def test_joint_feasible_positive_constant_conservative_transfer() -> None:
    route = _joint_line(4, 5, request=PointTransferRequest("joint", nonnegative=True))
    evidence = route.evidence
    assert route.admitted and evidence.provider == "conic"
    assert evidence.conservative and evidence.constant_preserving
    assert evidence.nonnegative and evidence.minimum_coefficient >= 0.0
    assert route.transfer is not None
    assert route.transfer.properties.exact_on == ("constant",)
    np.testing.assert_allclose(route.apply(np.ones(4)), np.ones(5), atol=1e-9)
    field = np.asarray([0.3, 1.2, 0.7, 2.0])
    mapped = np.asarray(route.apply(field))
    assert np.all(mapped >= 0.0)
    np.testing.assert_allclose(
        np.vdot(route.target_measures, mapped),
        np.vdot(route.source_measures, field),
        rtol=1e-10,
    )


@pytest.mark.parametrize("nonnegative", (False, True), ids=("signed", "positive"))
def test_unequal_total_measure_obstructs_conservation_with_constants(
    nonnegative: bool,
) -> None:
    route = _joint_line(
        4, 5, request=PointTransferRequest("joint", nonnegative=nonnegative), scale=1.1
    )
    evidence = route.evidence
    assert evidence.status is PointTransferStatus.MEASURE_OBSTRUCTION
    assert evidence.provider == "none" and route.transfer is None
    # Summed equations: conservation rows give sum(m_old), constant rows
    # weighted by m_new give sum(m_new) for the same left side.
    assert evidence.witness_kind == "measure-obstruction"
    np.testing.assert_allclose(evidence.obstruction_defect, 1.0 - 1.1, rtol=1e-12)
    np.testing.assert_allclose(evidence.witness_margin, evidence.obstruction_defect)
    assert evidence.witness_residual <= 1e-14
    conservative = _joint_line(
        4, 5, request=PointTransferRequest("conservative-positive"), scale=1.1
    )
    assert conservative.admitted and not conservative.evidence.constant_preserving


def _starved_row(request: PointTransferRequest, /) -> PreparedPointTransfer:
    """Equal total measure, but target 0 sees only source 0 at a nonzero offset."""
    sources, targets = np.asarray([0.0, 1.0, 2.0]), np.asarray([0.5, 1.5])
    columns = np.asarray([0, 1, 2], dtype=np.int32)
    rows = np.asarray([0, 1, 1], dtype=np.int32)
    return PointTransferPlan(
        EdgeRelation(columns, rows, source_size=3, target_size=2),
        np.asarray([1.0, 0.5, 0.5]),
        np.ones(3),
        np.full(2, 1.5),
        source_id="starved-old",
        target_id="starved-new",
        request=request,
        offsets=(sources[columns] - targets[rows])[:, None],
    ).prepare()


def test_sparse_support_infeasibility_reports_a_left_null_witness() -> None:
    route = _starved_row(PointTransferRequest("joint", moment_degree=1))
    evidence = route.evidence
    assert evidence.status is PointTransferStatus.INFEASIBLE
    assert evidence.provider == "minimum-norm" and evidence.witness_kind == "left-null"
    assert route.transfer is None
    assert evidence.obstruction_defect == 0.0
    assert evidence.witness_residual < 1e-8 < abs(evidence.witness_margin)


def test_sparse_support_infeasibility_reports_a_farkas_ray_when_nonnegative() -> None:
    route = _starved_row(PointTransferRequest("joint", moment_degree=1, nonnegative=True))
    evidence = route.evidence
    assert evidence.status is PointTransferStatus.INFEASIBLE
    assert evidence.provider == "conic" and evidence.witness_kind == "farkas"
    assert evidence.witness_margin < 0.0


def test_high_moments_conflict_with_positivity_but_not_with_signed_weights() -> None:
    line = PointTransferRequest("joint", moment_degree=2)
    signed = _joint_line(6, 4, request=line, quadrature=_simpson)
    assert signed.admitted and signed.evidence.moments_exact
    assert signed.evidence.minimum_coefficient < 0.0
    positive = _joint_line(
        6,
        4,
        request=PointTransferRequest("joint", moment_degree=2, nonnegative=True),
        quadrature=_simpson,
    )
    evidence = positive.evidence
    # sum_e t_e d_e^2 = 0 with t >= 0 and sum_e t_e = 1 needs a coincident
    # source, which the targets at 1/4 and 3/4 lack.
    assert evidence.status is PointTransferStatus.INFEASIBLE
    assert evidence.witness_kind == "farkas" and evidence.witness_margin < 0.0
    assert positive.transfer is None


def test_quadratic_exactness_needs_quadratures_that_agree_on_quadratics() -> None:
    # Midpoint rules of different widths integrate x^2 differently, so the
    # summed equations of a conservative quadratic-exact transfer contradict.
    route = _joint_line(6, 5, request=PointTransferRequest("joint", moment_degree=2))
    evidence = route.evidence
    assert evidence.status is PointTransferStatus.INFEASIBLE
    assert evidence.witness_kind == "left-null"
    assert abs(evidence.obstruction_defect) < 1e-14
    linear = _joint_line(6, 5, request=PointTransferRequest("joint", moment_degree=1))
    assert linear.admitted and linear.evidence.moments_exact


def _signed_quadratic(sources: int, targets: int, /) -> PreparedPointTransfer:
    x, old = _simpson(sources)
    y, new = _simpson(targets)
    relation, offsets = _nearest_routes(x, y, 6)
    return PointTransferPlan(
        relation,
        np.full(relation.route_shape, 1.0 / 6.0),
        old,
        new,
        source_id=f"simpson-{sources}",
        target_id=f"simpson-{targets}",
        request=PointTransferRequest("joint", moment_degree=2),
        offsets=offsets,
    ).prepare()


def test_signed_high_order_route_reproduces_quadratics_and_converges() -> None:
    forward = _signed_quadratic(24, 20)
    evidence = forward.evidence
    assert forward.admitted and evidence.moments_exact and evidence.conservative
    assert forward.transfer is not None
    assert forward.transfer.properties.exact_on == ("constant", "polynomial-degree-2")
    x, _ = _simpson(24)
    y, _ = _simpson(20)
    np.testing.assert_allclose(forward.apply(x**2), y**2, atol=1e-10)
    # Max-norm amplification is the largest absolute row sum of the coefficients.
    rows = np.asarray(forward.relation.target_indices)[np.asarray(forward.relation.valid)]
    row_sums = np.bincount(rows, weights=np.abs(np.asarray(evidence.coefficients)))
    np.testing.assert_allclose(evidence.lebesgue_constant, np.max(row_sums), rtol=1e-14)
    assert evidence.lebesgue_constant >= 1.0

    def error(sources: int, targets: int, /) -> float:
        route = _signed_quadratic(sources, targets)
        source, _ = _simpson(sources)
        target, _ = _simpson(targets)
        mapped = np.asarray(route.apply(np.sin(2 * np.pi * source)))
        return float(np.max(np.abs(mapped - np.sin(2 * np.pi * target))))

    coarse, fine = error(24, 20), error(48, 40)
    # Independently measured point error of a quadratic-exact remap.
    assert fine < coarse / 4.0


def test_joint_correction_is_the_minimum_change_of_all_declared_rows() -> None:
    # Independent dense oracle: the pseudoinverse correction of the stacked
    # conservation, constant and degree-1 moment rows.
    x, old = _simpson(8)
    y, new = _simpson(6)
    relation, offsets = _nearest_routes(x, y, 5)
    base = np.full(relation.route_shape, 0.2)
    route = PointTransferPlan(
        relation,
        base,
        old,
        new,
        source_id="simpson-8",
        target_id="simpson-6",
        request=PointTransferRequest("joint", moment_degree=1),
        offsets=offsets,
    ).prepare()
    assert route.admitted and route.evidence.provider == "minimum-norm"
    slots = np.asarray(relation.source_indices)
    rows = np.repeat(np.arange(y.size), slots.shape[1])
    columns = slots.reshape(-1)
    distance = offsets.reshape(-1)
    scale = np.zeros(y.size)
    np.maximum.at(scale, rows, np.abs(distance))
    matrix = np.zeros((x.size + 2 * y.size, rows.size))
    routes = np.arange(rows.size)
    matrix[columns, routes] = new[rows] / old[columns]
    matrix[x.size + rows, routes] = 1.0
    matrix[x.size + y.size + rows, routes] = distance / scale[rows]
    rhs = np.concatenate((np.ones(x.size), np.ones(y.size), np.zeros(y.size)))
    flat = base.reshape(-1)
    expected = flat + np.linalg.pinv(matrix) @ (rhs - matrix @ flat)
    np.testing.assert_allclose(route.evidence.coefficients, expected, atol=1e-9)


def test_declared_lebesgue_bound_refuses_an_amplifying_transfer() -> None:
    x, old = _simpson(8)
    y, new = _simpson(6)
    relation, offsets = _nearest_routes(x, y, 5)

    def prepare(bound: float, /) -> PreparedPointTransfer:
        return PointTransferPlan(
            relation,
            np.full(relation.route_shape, 0.2),
            old,
            new,
            source_id="simpson-8",
            target_id="simpson-6",
            request=PointTransferRequest("joint", moment_degree=2),
            offsets=offsets,
            lebesgue_bound=bound,
        ).prepare()

    admitted = prepare(1e6)
    assert admitted.admitted
    amplification = admitted.evidence.lebesgue_constant
    refused = prepare(0.5 * (1.0 + amplification))
    assert refused.evidence.status is PointTransferStatus.AMPLIFICATION_EXCEEDED
    assert refused.transfer is None
    assert refused.evidence.lebesgue_constant == amplification


def test_feasible_tensor_quadratic_transfer_is_resolved_on_unbalanced_rows() -> None:
    # Tensor Simpson rules agree on quadratics, so the joint degree-2 system is
    # feasible; conservation rows (Simpson weight ratios up to 4 times the route
    # count) and unit-scale moment rows differ by orders of magnitude in norm.
    def tensor(intervals: int, /) -> tuple[np.ndarray, np.ndarray]:
        nodes, weights = _simpson(intervals)
        x, y = np.meshgrid(nodes, nodes, indexing="ij")
        return np.stack((x.ravel(), y.ravel()), axis=1), np.outer(
            weights, weights
        ).ravel()

    sources, old = tensor(8)
    targets, new = tensor(6)
    distance = np.linalg.norm(targets[:, None] - sources[None], axis=2)
    slots = np.sort(np.argsort(distance, axis=1, kind="stable")[:, :12], axis=1)
    route = PointTransferPlan(
        RowRelation(slots.astype(np.int32), source_size=sources.shape[0]),
        np.full(slots.shape, 1.0 / 12.0),
        old,
        new,
        source_id="simpson-8x8",
        target_id="simpson-6x6",
        request=PointTransferRequest("joint", moment_degree=2),
        offsets=sources[slots] - targets[:, None, :],
    ).prepare()
    assert route.admitted and route.evidence.provider == "minimum-norm"
    assert route.evidence.moments_exact and route.evidence.conservative
    quadratic = sources[:, 0] ** 2 - sources[:, 0] * sources[:, 1]
    np.testing.assert_allclose(
        route.apply(quadratic),
        targets[:, 0] ** 2 - targets[:, 0] * targets[:, 1],
        atol=1e-9,
    )


def test_repeated_remap_conserves_content_with_bounded_measured_error() -> None:
    forward, backward = _signed_quadratic(24, 20), _signed_quadratic(20, 24)
    x, old = _simpson(24)
    exact = 2.0 + np.sin(2 * np.pi * x)
    field = jnp.asarray(exact)
    content = float(np.vdot(old, exact))
    errors = []
    for _ in range(8):
        field = backward.apply(forward.apply(field))
        errors.append(float(np.max(np.abs(np.asarray(field) - exact))))
        np.testing.assert_allclose(float(jnp.vdot(old, field)), content, rtol=1e-12)
    assert errors[-1] < 0.05
    assert errors[-1] <= 8.5 * errors[0]


def test_values_differentiate_through_the_frozen_transfer() -> None:
    route = _signed_quadratic(12, 10)
    assert route.transfer is not None
    rng = np.random.default_rng(3)
    values = jnp.asarray(rng.normal(size=13))
    tangent = jnp.asarray(rng.normal(size=13))
    cotangent = jnp.asarray(rng.normal(size=11))
    primal, jvp = jax.jvp(route.apply, (values,), (tangent,))
    np.testing.assert_allclose(jvp, route.apply(tangent), atol=1e-12)
    _, pullback = jax.vjp(route.apply, values)
    dual = route.transfer.dual_pullback_operator
    assert dual is not None
    np.testing.assert_allclose(pullback(cotangent)[0], dual.mv(cotangent), atol=1e-12)
    transition = route.epoch_transition(
        TopologyEpoch(0, "simpson-12", "line", "serial"),
        TopologyEpoch(1, "simpson-10", "line", "serial"),
    )
    result = transition.apply(values)
    assert bool(result.successful) and bool(result.value_derivative_available)
    assert not bool(result.differentiation_available)
    np.testing.assert_allclose(result.values, primal, atol=1e-12)
    _, epoch_pullback = jax.vjp(lambda v: transition.apply(v).values, values)
    np.testing.assert_allclose(
        epoch_pullback(cotangent)[0], transition.pullback(cotangent), atol=1e-12
    )
    # Hilbert adjoint in the measure pairings, distinct from the coordinate dual.
    adjoint = transition.adjoint(cotangent)
    np.testing.assert_allclose(
        jnp.vdot(route.target_measures * primal, cotangent),
        jnp.vdot(route.source_measures * values, adjoint),
        atol=1e-12,
    )
    assert not np.allclose(adjoint, transition.pullback(cotangent))
    with pytest.raises(ValueError, match="nondifferentiable"):
        transition.require_differentiable_topology()


def _circle(sources: int, targets: int, /) -> tuple[SurfaceTransferPlan, float]:
    """Tangential quadratic-exact remap between two equal-arc circle samplings."""
    old_angles = 2 * np.pi * (np.arange(sources) + 0.5) / sources
    new_angles = 2 * np.pi * (np.arange(targets) + 0.25) / targets
    points = np.stack((np.cos(old_angles), np.sin(old_angles)), axis=1)
    targets_ = np.stack((np.cos(new_angles), np.sin(new_angles)), axis=1)
    distance = np.linalg.norm(targets_[:, None] - points[None], axis=2)
    slots = np.sort(np.argsort(distance, axis=1, kind="stable")[:, :6], axis=1)
    plan = SurfaceTransferPlan(
        RowRelation(slots.astype(np.int32), source_size=sources),
        np.full(slots.shape, 1.0 / 6.0),
        points[slots] - targets_[:, None, :],
        targets_,
        np.full(sources, 2 * np.pi / sources),
        np.full(targets, 2 * np.pi / targets),
        source_id=f"circle-{sources}",
        target_id=f"circle-{targets}",
        request=PointTransferRequest("joint", moment_degree=2),
    )
    route = plan.prepare()
    assert route.admitted and route.evidence.constant_preserving
    assert route.evidence.moments_exact and route.evidence.conservative
    mapped = np.asarray(route.apply(np.cos(3 * old_angles)))
    return plan, float(np.max(np.abs(mapped - np.cos(3 * new_angles))))


def test_surface_transfer_imposes_moments_in_target_tangent_frames() -> None:
    plan, coarse = _circle(24, 20)
    # Routes leave the tangent line of a curved support; moments are tangential.
    assert plan.normal_departure > 0.0
    targets = 2 * np.pi * (np.arange(20) + 0.25) / 20
    normals = np.stack((np.cos(targets), np.sin(targets)), axis=1)
    np.testing.assert_allclose(
        np.sum(np.asarray(plan.frames) * normals[:, None, :], axis=-1), 0.0, atol=1e-14
    )
    _, fine = _circle(48, 40)
    assert fine < coarse / 4.0


def test_transfer_refuses_case_local_routes_without_global_column_identity() -> None:
    relation = RowRelation(
        np.zeros((2, 1, 1), dtype=np.int32), source_size=1, case_shape=(2,)
    )
    with pytest.raises(ValueError, match="unbatched compact target"):
        PointTransferPlan(
            relation,
            np.ones((2, 1, 1)),
            np.ones(1),
            np.ones(2),
            source_id="old",
            target_id="new",
            request=PointTransferRequest("conservative-signed"),
        )


def test_requests_declare_their_constraints_explicitly() -> None:
    with pytest.raises(ValueError, match="conservation only"):
        PointTransferRequest("conservative-signed", nonnegative=True)
    with pytest.raises(ValueError, match="no moments"):
        PointTransferRequest("conservative-positive", moment_degree=1)
    with pytest.raises(ValueError, match="route"):
        PointTransferRequest("high-order")  # ty: ignore[invalid-argument-type]
    assert PointTransferRequest("conservative-positive").nonnegative
