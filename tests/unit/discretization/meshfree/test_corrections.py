# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Learned metric and flux corrections against independent dense references.

Fixture provenance: a jittered 6x6 lattice on the unit square (seed 0, jitter
+-0.1h on interior nodes), radius 2.05h, Dirichlet boundary ring, second-moment
metric. Independent oracles: NumPy pseudoinverse weighted projections, SciPy
``linprog`` (largest feasible minimum weight) and ``minimize`` (positive-margin
QP), and NumPy incidence accumulation from the published edge pairs.

End-to-end training uses ``examples/meshfree_learned_metric_correction.py``: a regular
6x6 lattice (radius 1.5h, positive admitted metric), a log-linear edge-feature
candidate model, the signed projection, and a ``MonotoneEdgeConductance``
conservation solve with the corrected weights as runtime metric; its reference
is central finite differences of the native objective.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax import Array
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

from examples.meshfree_learned_metric_correction import (
    LearnedMetricTraining,
    objective as learned_objective,
    prepare_training,
    train,
    training_loss,
)
from phydrax.discretization.meshfree import (
    MeshfreeCorrectionStatus,
    MeshfreeEdgeFluxCandidate,
    MeshfreeEdgeFluxCorrectionPlan,
    MeshfreeExteriorCalculusPlan,
    MeshfreeMetricCorrection,
    MeshfreeMetricCorrectionPlan,
    MeshfreeMetricCorrectionPolicy,
    PreparedMeshfreeExteriorCalculus,
)
from phydrax.optim import ConicSensitivityStatus
from phydrax.solver import SolverObjective
from phydrax.sparse import EdgeRelation


_TOLERANCE = 1e-8
_MARGIN = 1e-2


class _Dense(NamedTuple):
    design: np.ndarray
    rhs: np.ndarray
    scaling: np.ndarray
    prior: np.ndarray
    pairs: np.ndarray
    points: np.ndarray


@pytest.fixture(scope="module", autouse=True)
def _double_precision() -> Iterator[None]:
    with jax.enable_x64(True):
        yield


@pytest.fixture(scope="module")
def exterior() -> PreparedMeshfreeExteriorCalculus:
    count = 6
    spacing = 1.0 / (count - 1)
    grid = (
        np.stack(
            np.meshgrid(np.arange(count), np.arange(count), indexing="ij"), axis=-1
        ).reshape(-1, 2)
        * spacing
    )
    interior = np.all((grid > 1e-9) & (grid < 1.0 - 1e-9), axis=1)
    jitter = np.random.default_rng(0).uniform(-0.1 * spacing, 0.1 * spacing, grid.shape)
    points = grid + np.where(interior[:, None], jitter, 0.0)
    return MeshfreeExteriorCalculusPlan(
        points,
        2.05 * spacing,
        16 * points.shape[0],
        node_volumes=np.full(points.shape[0], spacing * spacing),
        dirichlet=~interior,
    ).prepare()


@pytest.fixture(scope="module")
def dense(exterior: PreparedMeshfreeExteriorCalculus) -> _Dense:
    system = exterior.metric_system
    relation = system.constraint.relation
    assert isinstance(relation, EdgeRelation)
    design = np.zeros((system.constraint.target.size, system.constraint.source.size))
    np.add.at(
        design,
        (np.asarray(relation.target_indices), np.asarray(relation.source_indices)),
        np.asarray(system.constraint.coefficients),
    )
    return _Dense(
        design,
        np.asarray(system.rhs),
        np.asarray(system.row_scaling),
        np.asarray(system.prior),
        np.asarray(exterior.pairs),
        np.asarray(exterior.points),
    )


def _largest_minimum_weight(dense: _Dense) -> tuple[float, np.ndarray]:
    """SciPy LP: maximize t subject to A w = b and w >= t (t capped at one)."""
    rows, edges = dense.design.shape
    cost = np.zeros(edges + 1)
    cost[-1] = -1.0
    result = linprog(
        cost,
        A_ub=np.hstack((-np.eye(edges), np.ones((edges, 1)))),
        b_ub=np.zeros(edges),
        A_eq=np.hstack((dense.design, np.zeros((rows, 1)))),
        b_eq=dense.rhs,
        bounds=[(None, None)] * edges + [(None, 1.0)],
    )
    assert result.status == 0
    assert result.fun is not None and result.x is not None
    return float(-result.fun), np.asarray(result.x[:edges])


def _weighted_projection(dense: _Dense, candidate: np.ndarray) -> np.ndarray:
    """Independent dense minimum-|w - c|_{Phi^-1} point of {A w = b}."""
    root = np.sqrt(dense.prior)
    shift = np.linalg.pinv(dense.design * root[None, :], rcond=1e-12) @ (
        dense.rhs - dense.design @ candidate
    )
    return candidate + root * shift


def _projector(dense: _Dense, rows: np.ndarray) -> np.ndarray:
    """Dense weighted projector onto the null space of ``rows``."""
    root = np.sqrt(dense.prior)
    scaled = rows * root[None, :]
    return (
        np.eye(root.size)
        - root[:, None] * (np.linalg.pinv(scaled, rcond=1e-12) @ scaled) / root[None, :]
    )


def _normalized_residual(dense: _Dense, weights: np.ndarray) -> float:
    residual = dense.design @ weights - dense.rhs
    return float(np.max(np.abs(residual) / (dense.scaling + np.abs(dense.rhs))))


@pytest.fixture(scope="module")
def positive_point(dense: _Dense) -> np.ndarray:
    largest, weights = _largest_minimum_weight(dense)
    # The fixture must admit strictly positive moment-exact metrics above the margin.
    assert largest > 4 * _MARGIN
    return weights


@pytest.fixture(scope="module")
def signed_plan(
    exterior: PreparedMeshfreeExteriorCalculus,
) -> MeshfreeMetricCorrectionPlan:
    return MeshfreeMetricCorrectionPlan(
        exterior,
        policy=MeshfreeMetricCorrectionPolicy(margin=_MARGIN, tolerance=_TOLERANCE),
    )


@pytest.fixture(scope="module")
def constrained_plan(
    exterior: PreparedMeshfreeExteriorCalculus,
) -> MeshfreeMetricCorrectionPlan:
    return MeshfreeMetricCorrectionPlan(
        exterior,
        policy=MeshfreeMetricCorrectionPolicy(
            margin=_MARGIN, sign="constrained", tolerance=_TOLERANCE
        ),
    )


@pytest.fixture(scope="module")
def learned_candidate(positive_point: np.ndarray, dense: _Dense) -> np.ndarray:
    # A learned perturbation of a positive metric: smooth edge-length feature
    # plus noise, neither of which respects the moment rows.
    lengths = np.linalg.norm(
        dense.points[dense.pairs[:, 1]] - dense.points[dense.pairs[:, 0]], axis=1
    )
    noise = np.random.default_rng(1).standard_normal(lengths.size)
    return positive_point * (1.0 + 0.05 * np.sin(9.0 * lengths)) + 2e-3 * noise


@pytest.fixture(scope="module")
def admitted(
    signed_plan: MeshfreeMetricCorrectionPlan, learned_candidate: np.ndarray
) -> MeshfreeMetricCorrection:
    return signed_plan.correct(learned_candidate)


@pytest.fixture(scope="module")
def conflict_candidate(exterior: PreparedMeshfreeExteriorCalculus) -> np.ndarray:
    # The zero-centered signed metric is moment exact but has negative weights,
    # so its own signed projection is itself and breaks positivity.
    weights = np.asarray(exterior.metric_result.weights)
    assert float(np.min(weights)) < 0.0
    return weights


@pytest.fixture(scope="module")
def constrained(
    constrained_plan: MeshfreeMetricCorrectionPlan, conflict_candidate: np.ndarray
) -> MeshfreeMetricCorrection:
    return constrained_plan.correct(conflict_candidate)


def test_signed_projection_reproduces_every_moment_and_the_dense_reference(
    admitted: MeshfreeMetricCorrection, learned_candidate: np.ndarray, dense: _Dense
) -> None:
    evidence = admitted.evidence
    assert admitted.status is MeshfreeCorrectionStatus.ADMITTED, evidence.provider_status
    assert evidence.provider == "minimum-norm"
    assert evidence.audited_subject == "projection"
    weights = np.asarray(admitted.require_weights())
    # Every original row, independently of the provider and its presolve.
    assert _normalized_residual(dense, learned_candidate) > 1e-3
    assert _normalized_residual(dense, weights) <= _TOLERANCE
    assert evidence.moments_exact
    np.testing.assert_allclose(
        np.asarray(evidence.moment_residual),
        dense.design @ weights - dense.rhs,
        atol=1e-12,
    )
    reference = _weighted_projection(dense, learned_candidate)
    np.testing.assert_allclose(weights, reference, rtol=1e-7, atol=1e-8)
    delta = reference - learned_candidate
    np.testing.assert_allclose(
        evidence.correction_norm, np.sqrt(np.sum(delta * delta / dense.prior)), rtol=1e-6
    )
    assert evidence.positive and evidence.sign_margin >= 0.0
    assert evidence.coercive
    assert evidence.stiffness is not None and bool(evidence.stiffness.spd)
    assert evidence.derivative == "minimum-norm-projector"
    assert evidence.derivative_available


def test_feasible_candidate_is_a_fixed_point(
    signed_plan: MeshfreeMetricCorrectionPlan, admitted: MeshfreeMetricCorrection
) -> None:
    weights = np.asarray(admitted.require_weights())
    again = signed_plan.correct(weights)
    assert again.status is MeshfreeCorrectionStatus.ADMITTED, (
        again.evidence.provider_status
    )
    np.testing.assert_allclose(np.asarray(again.require_weights()), weights, atol=1e-9)
    assert again.evidence.correction_norm <= 1e-8 * float(np.linalg.norm(weights))


def test_positivity_conflict_is_refused_without_publishing_weights(
    signed_plan: MeshfreeMetricCorrectionPlan,
    conflict_candidate: np.ndarray,
    dense: _Dense,
) -> None:
    refused = signed_plan.correct(conflict_candidate)
    evidence = refused.evidence
    assert refused.status is MeshfreeCorrectionStatus.POSITIVITY_CONFLICT
    assert refused.weights is None
    with pytest.raises(ValueError, match="POSITIVITY_CONFLICT"):
        refused.require_weights()
    # The refusal is about sign only: the signed projection was moment exact.
    assert evidence.moments_exact
    assert _normalized_residual(dense, np.asarray(evidence.audited_weights)) <= _TOLERANCE
    assert evidence.signed_minimum_weight == pytest.approx(
        float(np.min(conflict_candidate))
    )
    assert evidence.sign_margin < 0.0 and not evidence.positive
    assert evidence.derivative == "unavailable"


def test_constrained_route_solves_the_positive_margin_qp(
    constrained: MeshfreeMetricCorrection,
    conflict_candidate: np.ndarray,
    positive_point: np.ndarray,
    dense: _Dense,
) -> None:
    evidence = constrained.evidence
    assert constrained.status is MeshfreeCorrectionStatus.ADMITTED, (
        evidence.provider_status,
        evidence.maximum_moment_residual,
        evidence.sign_margin,
    )
    assert evidence.provider == "conic"
    assert evidence.signed_minimum_weight < _MARGIN
    weights = np.asarray(constrained.require_weights())
    assert _normalized_residual(dense, weights) <= _TOLERANCE
    assert float(np.min(weights)) >= _MARGIN - _TOLERANCE
    assert evidence.coercive

    def objective(value: np.ndarray) -> float:
        delta = value - conflict_candidate
        return float(0.5 * np.sum(delta * delta / dense.prior))

    reference = minimize(
        objective,
        positive_point,
        jac=lambda value: (value - conflict_candidate) / dense.prior,
        hess=lambda _: np.diag(1.0 / dense.prior),
        constraints=(LinearConstraint(dense.design, dense.rhs, dense.rhs),),
        bounds=Bounds(np.full(weights.size, _MARGIN), np.full(weights.size, np.inf)),
        method="trust-constr",
        options={"gtol": 1e-12, "xtol": 1e-14, "barrier_tol": 1e-12, "maxiter": 5000},
    )
    assert _normalized_residual(dense, reference.x) <= 1e-6
    np.testing.assert_allclose(objective(weights), reference.fun, rtol=1e-6)
    np.testing.assert_allclose(weights, reference.x, atol=1e-5)


def test_constrained_derivative_is_the_fixed_active_face_projector(
    constrained_plan: MeshfreeMetricCorrectionPlan,
    constrained: MeshfreeMetricCorrection,
    dense: _Dense,
) -> None:
    evidence = constrained.evidence
    assert evidence.derivative == "conic-fixed-active-set"
    assert evidence.sensitivity_status == int(ConicSensitivityStatus.REGULAR_FIXED_ACTIVE)
    assert evidence.derivative_available
    weights = np.asarray(constrained.require_weights())
    active = weights <= _MARGIN + 1e-6
    assert np.any(active)
    direction = np.random.default_rng(2).standard_normal(weights.size)
    tangent, available = constrained_plan.tangent(constrained, direction)
    assert bool(available)
    face = np.vstack((dense.design, np.eye(weights.size)[active]))
    np.testing.assert_allclose(
        np.asarray(tangent), _projector(dense, face) @ direction, atol=1e-6
    )


def test_infeasible_margin_is_refused_with_an_audited_farkas_witness(
    exterior: PreparedMeshfreeExteriorCalculus, positive_point: np.ndarray, dense: _Dense
) -> None:
    largest, _ = _largest_minimum_weight(dense)
    margin = 2.0 * largest
    plan = MeshfreeMetricCorrectionPlan(
        exterior,
        policy=MeshfreeMetricCorrectionPolicy(
            "nonnegative-conic", margin=margin, sign="constrained", tolerance=_TOLERANCE
        ),
    )
    refused = plan.correct(positive_point)
    evidence = refused.evidence
    assert refused.status is MeshfreeCorrectionStatus.INFEASIBLE, (
        evidence.provider_status,
        evidence.witness_residual,
        evidence.witness_margin,
    )
    assert refused.weights is None
    assert evidence.audited_subject == "candidate"
    assert evidence.witness_kind == "farkas" and evidence.witness_valid
    witness = np.asarray(evidence.witness)
    image = dense.design.T @ witness
    scale = float(np.max(np.abs(witness * dense.scaling)))
    assert float(np.min(image)) >= -1e-7 * scale
    # Every w >= margin with A w = b would give <y, b> >= margin * sum(A^T y).
    assert float(witness @ dense.rhs) < margin * float(np.sum(image))
    assert evidence.derivative == "unavailable"


def test_incompatible_moment_rows_are_refused_with_a_left_null_witness() -> None:
    exterior = MeshfreeExteriorCalculusPlan(
        np.asarray([[0.0], [1.0]]), 1.1, 1, node_volumes=np.asarray([1.0, 1.0])
    ).prepare()
    plan = MeshfreeMetricCorrectionPlan(
        exterior, policy=MeshfreeMetricCorrectionPolicy(margin=0.1, tolerance=_TOLERANCE)
    )
    refused = plan.correct(np.asarray([0.7]))
    evidence = refused.evidence
    assert refused.status is MeshfreeCorrectionStatus.MOMENT_INCOMPATIBLE, (
        evidence.provider_status,
        evidence.witness_residual,
        evidence.witness_margin,
    )
    assert refused.weights is None
    assert evidence.witness_kind == "left-null" and evidence.witness_valid
    system = exterior.metric_system
    witness = np.asarray(evidence.witness)
    np.testing.assert_allclose(
        np.asarray(system.constraint.transpose_mv(jnp.asarray(witness))), 0.0, atol=1e-10
    )
    assert abs(float(witness @ np.asarray(system.rhs))) > 1e-3 * float(
        np.max(np.abs(witness))
    )
    projection = plan.project(np.asarray([0.7]))
    assert not bool(projection.admissible)


def _learned_flux(
    points: np.ndarray, pairs: np.ndarray, even: float
) -> tuple[np.ndarray, np.ndarray]:
    """A learned edge flux evaluated in both endpoint orders.

    ``tanh`` of the endpoint difference is orientation odd; ``even`` scales an
    endpoint-symmetric term that a conservative oriented flux cannot contain.
    """

    def model(tail: np.ndarray, head: np.ndarray) -> np.ndarray:
        difference = head - tail
        odd = np.tanh(difference @ np.asarray([3.0, -2.0]))
        return odd + even * np.sum(tail * tail + head * head, axis=1)

    tail, head = points[pairs[:, 0]], points[pairs[:, 1]]
    return model(tail, head), model(head, tail)


def test_flux_correction_is_conservative_and_refuses_broken_parity(
    exterior: PreparedMeshfreeExteriorCalculus, dense: _Dense
) -> None:
    plan = MeshfreeEdgeFluxCorrectionPlan(exterior, tolerance=_TOLERANCE)
    equations = np.asarray(exterior.equation_indices)
    balance = np.zeros(dense.points.shape[0])
    balance[equations] = np.random.default_rng(3).standard_normal(equations.size)
    forward, reverse = _learned_flux(dense.points, dense.pairs, 0.0)
    corrected = plan.correct(MeshfreeEdgeFluxCandidate(forward, reverse), balance)
    assert corrected.evidence.status is MeshfreeCorrectionStatus.ADMITTED, (
        corrected.evidence.provider_status
    )
    assert corrected.evidence.antisymmetric
    flux = np.asarray(corrected.require_flux())
    # Independent oriented accumulation: -F at the tail, +F at the head.
    integrated = np.zeros(dense.points.shape[0])
    np.add.at(integrated, dense.pairs[:, 0], -flux)
    np.add.at(integrated, dense.pairs[:, 1], flux)
    np.testing.assert_allclose(integrated[equations], balance[equations], atol=1e-9)
    incidence = np.zeros((equations.size, flux.size))
    rows = {int(node): index for index, node in enumerate(equations)}
    for edge, (tail, head) in enumerate(dense.pairs):
        if int(tail) in rows:
            incidence[rows[int(tail)], edge] -= 1.0
        if int(head) in rows:
            incidence[rows[int(head)], edge] += 1.0
    minimum = forward + np.linalg.pinv(incidence) @ (
        balance[equations] - incidence @ forward
    )
    np.testing.assert_allclose(flux, minimum, atol=1e-8)

    forward, reverse = _learned_flux(dense.points, dense.pairs, 0.3)
    refused = plan.correct(MeshfreeEdgeFluxCandidate(forward, reverse), balance)
    assert refused.evidence.status is MeshfreeCorrectionStatus.PARITY_CONFLICT
    assert refused.flux is None
    assert not refused.evidence.antisymmetric
    np.testing.assert_allclose(
        refused.evidence.maximum_parity_defect, float(np.max(np.abs(forward + reverse)))
    )


def test_projection_derivative_is_the_weighted_projector_and_failure_has_none(
    signed_plan: MeshfreeMetricCorrectionPlan,
    positive_point: np.ndarray,
    learned_candidate: np.ndarray,
    dense: _Dense,
) -> None:
    feature = jnp.asarray(learned_candidate - positive_point)
    target = jnp.asarray(
        _weighted_projection(dense, positive_point + 0.3 * np.asarray(feature))
    )

    def loss(parameter: Array) -> tuple[Array, Array]:
        projection = signed_plan.project(
            jnp.asarray(positive_point) + parameter * feature
        )
        miss = projection.weights - target
        return jnp.sum(miss * miss), projection.admissible

    parameter = jnp.asarray(1.0)
    (value, admissible), gradient = jax.value_and_grad(loss, has_aux=True)(parameter)
    assert bool(admissible)
    # Independent chain rule through the dense weighted projector.
    projector = _projector(dense, dense.design)
    projected = _weighted_projection(dense, positive_point + np.asarray(feature))
    expected = 2.0 * (projected - np.asarray(target)) @ (projector @ np.asarray(feature))
    np.testing.assert_allclose(float(gradient), expected, rtol=1e-6)
    # The loss is quadratic in the parameter with curvature 2 |P f|^2_2.
    curvature = 2.0 * float(np.sum((projector @ np.asarray(feature)) ** 2))
    stepped, stepped_admissible = loss(parameter - 0.5 * gradient / curvature)
    assert bool(stepped_admissible)
    assert float(stepped) < float(value)

    # A failed native projection publishes no admissible point and no usable
    # derivative: a training step on it is rejected, not silently taken.
    failing = MeshfreeMetricCorrectionPlan(
        signed_plan.exterior,
        policy=MeshfreeMetricCorrectionPolicy(
            margin=_MARGIN, tolerance=_TOLERANCE, maximum_steps=1
        ),
    )

    def failing_loss(parameter: Array) -> tuple[Array, Array]:
        projection = failing.project(jnp.asarray(positive_point) + parameter * feature)
        miss = projection.weights - target
        return jnp.sum(miss * miss), projection.admissible

    (_, failed_admissible), failed_gradient = jax.value_and_grad(
        failing_loss, has_aux=True
    )(parameter)
    assert not bool(failed_admissible)
    assert not bool(jnp.isfinite(failed_gradient))
    assert failing.correct(learned_candidate).status is (
        MeshfreeCorrectionStatus.PROVIDER_UNRESOLVED
    )


@pytest.fixture(scope="module")
def learned() -> tuple[LearnedMetricTraining, SolverObjective]:
    training = prepare_training()
    return training, learned_objective(training)


def test_implicit_training_gradient_through_projection_and_solve_matches_central_difference(
    learned: tuple[LearnedMetricTraining, SolverObjective],
) -> None:
    training, target = learned
    point = jnp.asarray([0.2, -0.1], dtype=jnp.float64)
    direction = jnp.asarray([0.8, 0.6], dtype=jnp.float64)
    projection = training.plan.project(training.model(point)(0.0))
    assert bool(projection.admissible)
    gradient = jax.grad(lambda value: training_loss(training, target, value))(point)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    step = 1e-5
    central = (
        training_loss(training, target, point + step * direction)
        - training_loss(training, target, point - step * direction)
    ) / (2 * step)
    np.testing.assert_allclose(
        float(jnp.vdot(gradient, direction)), float(central), rtol=1e-5
    )
    assert abs(float(central)) > 1e-8


def test_native_training_step_decreases_the_loss_and_keeps_moments_exact(
    learned: tuple[LearnedMetricTraining, SolverObjective],
) -> None:
    training, target = learned
    initial = jnp.zeros((2,), dtype=jnp.float64)
    before = float(training_loss(training, target, initial))
    assert np.isfinite(before) and before > 0.0
    result, coefficients = train(
        training, target, initial, optimizer=optax.lbfgs(), steps=1
    )
    assert result.accepted_updates == 1
    after = float(training_loss(training, target, coefficients))
    assert after < before
    corrected = training.plan.correct(training.model(coefficients)(0.0))
    assert corrected.status is MeshfreeCorrectionStatus.ADMITTED
    assert corrected.evidence.moments_exact and corrected.evidence.positive


def test_failed_projection_and_primal_attempt_is_rejected_not_stepped(
    learned: tuple[LearnedMetricTraining, SolverObjective],
) -> None:
    training, target = learned
    failing = jnp.asarray([6.0, 0.0], dtype=jnp.float64)
    projection = training.plan.project(training.model(failing)(0.0))
    assert bool(projection.successful)
    assert not bool(projection.admissible)
    assert float(jnp.min(projection.weights)) < 0.0
    # The conservation primal refuses the nonpositive metric itself.
    primal = training.prepared.solve(
        parameters={"corrected-metric": projection.weights}, implicit=False
    )
    assert not bool(primal.accepted)
    assert not np.isfinite(float(training_loss(training, target, failing)))
    result, kept = train(
        training,
        target,
        failing,
        optimizer=optax.sgd(1e-3),
        steps=1,
        rejection_budget=1,
    )
    assert result.accepted_updates == 0
    assert result.nonfinite_rejections == 1
    np.testing.assert_array_equal(np.asarray(kept), np.asarray(failing))
