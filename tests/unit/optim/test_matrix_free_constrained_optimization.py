#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.optim._primal_dual import (
    _barrier_schur_metric,
    AbstractPrimalDualKKTSetup,
    PrimalDualEvidence,
    PrimalDualKKTSetupResult,
    PrimalDualNewtonKrylov,
)


class _PartitionedDesign(eqx.Module):
    control: jax.Array
    scale: float = eqx.field(static=True)


def _termination(*, tolerance: Any = 1e-7, steps: Any = 50) -> Any:
    return phx.optim.OptimizationTermination(
        absolute_optimality=tolerance,
        relative_optimality=0.0,
        maximum_steps=steps,
    )


def test_prepared_kkt_renews_dynamic_objective_and_preserves_actual_linear_evidence() -> (
    None
):
    def objective(parameters: jax.Array, target: jax.Array) -> jax.Array:
        return 0.5 * jnp.sum((parameters - target) ** 2)

    def total(parameters: jax.Array, target: jax.Array) -> jax.Array:
        return jnp.sum(parameters, keepdims=True)

    def first(parameters: jax.Array, target: jax.Array) -> jax.Array:
        return parameters[:1]

    problem = phx.optim.MinimizationProblem(
        objective,
        constraints=(
            phx.optim.NonlinearConstraint(
                total, lower=1.0, upper=1.0, constraint_id="sum"
            ),
            phx.optim.NonlinearConstraint(
                first, upper=1.5, constraint_id="first-ceiling"
            ),
        ),
        problem_id="dynamic-matrix-free-kkt",
    )
    method = PrimalDualNewtonKrylov(
        maximum_restoration_steps=0,
        linear_policy=phx.linalg.LinearSolvePolicy(
            phx.linalg.MINRES(),
            tolerance=phx.linalg.TolerancePolicy(
                relative=1.0e-8,
                absolute=1.0e-8,
                max_steps=64,
            ),
            preconditioning=phx.linalg.PreconditioningPolicy(
                phx.linalg.JacobiPreconditionerBuilder(),
            ),
        ),
    )

    @eqx.filter_jit
    def solve(target: jax.Array) -> phx.optim.MinimizationResult:
        return phx.optim.minimize(
            problem,
            jnp.asarray((3.0, 3.0), dtype=np.float64),
            method=method,
            termination=_termination(),
            args=target,
        )

    for target, expected in (((2.0, -1.0), (1.5, -0.5)), ((1.0, 0.0), (1.0, 0.0))):
        result = solve(jnp.asarray(target, dtype=np.float64))
        np.testing.assert_allclose(result.parameters, expected, atol=2.0e-6)
        assert int(result.status) == int(phx.optim.OptimizationStatus.SUCCESS)
        evidence = result.method_evidence
        assert isinstance(evidence, PrimalDualEvidence)
        assert int(evidence.linear_status) == int(phx.linalg.LinearSolveStatus.SUCCESS)
        assert np.isfinite(float(evidence.linear_residual_norm))
        assert int(evidence.linear_iterations) > 0
        assert int(evidence.linear_matvec_count) > 0
        assert int(result.diagnostics.direction_fallbacks) == 0


def test_current_barrier_metric_matches_gram_and_schur_diagonals() -> None:
    def constraints(point: jax.Array) -> tuple[jax.Array, jax.Array]:
        return (
            (point[0] + 2.0 * point[1])[None],
            jnp.stack((3.0 * point[0] - point[1], point[0] + point[1])),
        )

    point = jnp.asarray((0.3, -0.2), dtype=np.float64)
    equality = jnp.zeros(1, dtype=np.float64)
    space = phx.linalg.BlockSpace(
        (
            phx.linalg.ArraySpace((2,), dtype=np.float64),
            phx.linalg.ArraySpace((1,), dtype=np.float64),
        )
    )
    derivative = phx.linalg.JacobianLinearOperator(
        phx.linalg.prepare_linearization(
            constraints,
            point,
            source=space.spaces[0],
        )
    )
    for weights in ((2.0, 4.0), (7.0, 0.5)):
        metric = _barrier_schur_metric(
            derivative,
            jnp.asarray(weights, dtype=np.float64),
            1.0e-8,
            point,
            equality,
            space,
            "actual-barrier-metric",
        )
        first = 1.0e-8 + 9.0 * weights[0] + weights[1]
        second = 1.0e-8 + weights[0] + weights[1]
        np.testing.assert_allclose(
            metric.diagonal,
            (first, second, 1.0 / first + 4.0 / second),
            rtol=1.0e-14,
        )
        correction = phx.linalg.JacobiPreconditionerBuilder().prepare(
            metric,
            materialization=phx.linalg.MaterializationPolicy(),
        )
        applied = correction.apply(
            (jnp.ones(2, dtype=np.float64), jnp.ones(1, dtype=np.float64))
        )
        np.testing.assert_allclose(applied[0], (1.0 / first, 1.0 / second), rtol=1.0e-14)
        np.testing.assert_allclose(
            applied[1], (1.0 / (1.0 / first + 4.0 / second),), rtol=1.0e-14
        )
        assert correction.properties.certifies("positive_definite")
        estimate = phx.linalg.JacobiPreconditionerBuilder().cost_for(metric)
        assert estimate.setup_matvec_count == 0
        assert estimate.storage_bytes == 3 * np.dtype(np.float64).itemsize


def test_compiled_barrier_metric_refreshes_point_and_slack_weights() -> None:
    space = phx.linalg.BlockSpace(
        (
            phx.linalg.ArraySpace((2,), dtype=np.float64),
            phx.linalg.ArraySpace((1,), dtype=np.float64),
        )
    )

    def constraints(point: jax.Array) -> tuple[jax.Array, jax.Array]:
        return (
            (point[0] ** 2 + 2.0 * point[1])[None],
            jnp.stack((3.0 * point[0] ** 2 - point[1], point[0] + point[1])),
        )

    @eqx.filter_jit
    def metric(point: jax.Array, weights: jax.Array) -> jax.Array:
        derivative = phx.linalg.JacobianLinearOperator(
            phx.linalg.prepare_linearization(
                constraints,
                point,
                source=space.spaces[0],
            ),
        )
        return _barrier_schur_metric(
            derivative,
            weights,
            1.0e-8,
            point,
            jnp.zeros(1, dtype=np.float64),
            space,
            "dynamic-barrier-metric",
        ).diagonal

    for coordinate, weights in ((0.3, (2.0, 4.0)), (0.8, (7.0, 0.5))):
        first = 1.0e-8 + (6.0 * coordinate) ** 2 * weights[0] + weights[1]
        second = 1.0e-8 + weights[0] + weights[1]
        actual = metric(
            jnp.asarray((coordinate, -0.2), dtype=np.float64),
            jnp.asarray(weights, dtype=np.float64),
        )
        np.testing.assert_allclose(
            actual,
            (first, second, (2.0 * coordinate) ** 2 / first + 4.0 / second),
            rtol=1.0e-14,
        )


def test_failed_kkt_status_survives_finite_primal_direction() -> None:
    problem = phx.optim.MinimizationProblem(
        lambda parameters, _: jnp.sum(
            jnp.asarray((1.0, 7.0, 31.0), dtype=np.float64) * (parameters - 2.0) ** 2,
        ),
        constraints=(
            phx.optim.NonlinearConstraint(
                lambda parameters, _: jnp.sum(parameters, keepdims=True),
                lower=1.0,
                upper=1.0,
                constraint_id="sum",
            ),
        ),
        problem_id="limited-kkt-evidence",
    )
    result = phx.optim.minimize(
        problem,
        jnp.zeros(3, dtype=np.float64),
        method=PrimalDualNewtonKrylov(
            maximum_restoration_steps=0,
            linear_policy=phx.linalg.LinearSolvePolicy(
                phx.linalg.MINRES(),
                tolerance=phx.linalg.TolerancePolicy(
                    relative=1.0e-14,
                    absolute=1.0e-14,
                    max_steps=1,
                ),
                preconditioning=phx.linalg.PreconditioningPolicy(
                    phx.linalg.JacobiPreconditionerBuilder(),
                ),
            ),
        ),
        termination=_termination(tolerance=1.0e-14, steps=1),
    )
    evidence = result.method_evidence
    assert isinstance(evidence, PrimalDualEvidence)
    assert int(evidence.linear_status) == int(
        phx.linalg.LinearSolveStatus.MAXIMUM_STEPS_REACHED,
    )
    assert float(evidence.linear_residual_norm) > 1.0e-14
    assert int(evidence.linear_iterations) == 1
    assert int(evidence.linear_matvec_count) > 1
    assert np.all(np.isfinite(np.asarray(result.parameters)))
    assert int(result.status) != int(phx.optim.OptimizationStatus.SUCCESS)
    assert int(result.diagnostics.direction_fallbacks) == 0


class _FixedNativeKKTSetup(AbstractPrimalDualKKTSetup):
    operator: phx.sparse.SparseCoordinateOperator

    def __init__(self, operator: phx.sparse.SparseCoordinateOperator, /) -> None:
        self.operator = operator

    def prepare(
        self,
        derivative: phx.linalg.AbstractLinearOperator,
        barrier_weights: jax.Array,
        regularization: float,
        primal: jax.Array,
        equality: jax.Array,
        space: phx.linalg.BlockSpace,
        kkt_operator: phx.linalg.AbstractLinearOperator,
        /,
    ) -> PrimalDualKKTSetupResult:
        del derivative, barrier_weights, regularization, primal, equality, kkt_operator
        if not self.operator.source.compatible(space):
            raise ValueError(
                "The native test correction must preserve the bound block space."
            )
        return PrimalDualKKTSetupResult(
            self.operator,
            jnp.asarray(0, dtype=np.int32),
            jnp.asarray(0, dtype=np.int64),
            jnp.asarray(0, dtype=np.int32),
        )


@pytest.mark.parametrize(
    "diagonal,status,optimization_status",
    (
        (
            0.0,
            phx.linalg.SparseFactorizationStatus.NONPOSITIVE_PIVOT,
            phx.optim.OptimizationStatus.LINEAR_SOLVE_FAILED,
        ),
        (
            np.nan,
            phx.linalg.SparseFactorizationStatus.NONFINITE,
            phx.optim.OptimizationStatus.NONFINITE_EVALUATION,
        ),
    ),
)
def test_failed_native_setup_stops_before_minres_and_preserves_factor_evidence(
    diagonal: float,
    status: phx.linalg.SparseFactorizationStatus,
    optimization_status: phx.optim.OptimizationStatus,
) -> None:
    space = phx.linalg.BlockSpace(
        (
            phx.linalg.ArraySpace((2,), dtype=np.float64),
            phx.linalg.ArraySpace((1,), dtype=np.float64),
        )
    )
    relation = phx.sparse.EdgeRelation(
        np.arange(3, dtype=np.int32),
        np.arange(3, dtype=np.int32),
        source_size=3,
        target_size=3,
    )
    operator = phx.sparse.SparseCoordinateOperator(
        relation,
        jnp.asarray((1.0, 1.0, diagonal), dtype=np.float64),
        source=space,
        target=space,
        properties=phx.linalg.OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        ),
        operator_id="actual-native-factor-status-fixture",
    )
    plan = phx.linalg.prepare_sparse_factorization(
        operator,
        phx.linalg.SparseFactorizationPolicy("cholesky"),
    )
    method = PrimalDualNewtonKrylov(
        maximum_restoration_steps=0,
        kkt_setup=_FixedNativeKKTSetup(operator),
        linear_policy=phx.linalg.LinearSolvePolicy(
            phx.linalg.MINRES(),
            tolerance=phx.linalg.TolerancePolicy(
                relative=1.0e-6, absolute=1.0e-10, max_steps=64
            ),
            preconditioning=phx.linalg.PreconditioningPolicy(
                phx.linalg.SparseFactorizationPreconditionerBuilder(
                    prepared_plan=plan,
                    setup_operator=operator,
                ),
            ),
        ),
    )
    problem = phx.optim.MinimizationProblem(
        lambda point, _: jnp.sum((point - 2.0) ** 2),
        constraints=(
            phx.optim.NonlinearConstraint(
                lambda point, _: jnp.sum(point, keepdims=True),
                lower=1.0,
                upper=1.0,
            ),
        ),
    )
    initial = jnp.asarray((0.0, 0.0), dtype=np.float64)
    result = phx.optim.minimize(
        problem, initial, method=method, termination=_termination(steps=16)
    )
    evidence = result.method_evidence
    assert isinstance(evidence, PrimalDualEvidence)
    assert int(evidence.factorization_status) == int(status)
    assert evidence.factorization_diagnostics is not None
    assert int(result.status) == int(optimization_status)
    assert int(evidence.linear_status) == -1
    assert int(result.diagnostics.linear_solves) == 0
    assert int(result.diagnostics.direction_fallbacks) == 0
    np.testing.assert_array_equal(result.parameters, initial)


def _forbid_explicit_jacobians(monkeypatch: Any) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("The matrix-free method formed an explicit Jacobian.")

    monkeypatch.setattr(jax, "jacrev", forbidden)
    monkeypatch.setattr(jax, "jacfwd", forbidden)


def test_primal_dual_newton_krylov_solves_mixed_constraints_without_jacobians(
    monkeypatch: Any,
) -> None:
    equality = phx.optim.NonlinearConstraint(
        lambda parameters, _: jnp.array([parameters[0] + parameters[1]]),
        lower=1.0,
        upper=1.0,
        constraint_id="sum",
    )
    inequality = phx.optim.NonlinearConstraint(
        lambda parameters, _: jnp.array([parameters[0]]),
        upper=1.5,
        constraint_id="upper-first",
    )
    problem = phx.optim.MinimizationProblem(
        lambda parameters, _: 0.5 * jnp.sum((parameters - jnp.array([2.0, -1.0])) ** 2),
        constraints=(equality, inequality),
        problem_id="mixed-matrix-free",
    )
    _forbid_explicit_jacobians(monkeypatch)

    result = phx.optim.minimize(
        problem,
        jnp.array([3.0, 3.0]),
        method=phx.optim.PrimalDualInteriorPoint(
            mode="matrix-free-centered",
        ),
        termination=_termination(),
    )

    np.testing.assert_allclose(result.parameters, jnp.array([1.5, -0.5]), atol=2e-6)
    assert int(result.status) == int(phx.optim.OptimizationStatus.SUCCESS)
    assert result.provenance.matrix_free
    assert result.diagnostics.jacobian_evaluations == 0
    assert result.diagnostics.hvp_evaluations > 0
    assert result.diagnostics.setup_refreshes == 1
    assert result.diagnostics.numeric_refreshes > 0
    assert result.certificate is not None
    np.testing.assert_allclose(
        result.certificate.equality_multipliers,
        jnp.array([-0.5]),
        atol=2e-6,
    )
    np.testing.assert_allclose(
        result.certificate.inequality_multipliers,
        jnp.array([1.0]),
        atol=2e-6,
    )
    assert result.certificate.active_mask.tolist() == [True]
    assert result.certificate.equality_sources == ("constraint:0:0:equality",)
    assert result.certificate.inequality_sources == ("constraint:1:0:upper",)


def test_primal_dual_contracts() -> None:
    problem = phx.optim.MinimizationProblem(
        lambda parameters, _: jnp.sum((parameters - 2.0) ** 2),
        bounds=phx.optim.Bounds(-jnp.inf, 1.0),
        problem_id="bound-certificate",
    )
    result = phx.optim.minimize(
        problem,
        jnp.array([0.0]),
        method=phx.optim.PrimalDualInteriorPoint(
            mode="matrix-free-centered",
        ),
        termination=_termination(),
    )

    np.testing.assert_allclose(result.parameters, jnp.array([1.0]), atol=2e-6)
    # ty: ignore[unresolved-attribute]
    assert result.certificate.inequality_sources == ("bound:0:upper",)
    np.testing.assert_allclose(
        # ty: ignore[unresolved-attribute]
        result.certificate.inequality_multipliers,
        jnp.array([2.0]),
        atol=2e-6,
    )
    # ty: ignore[unresolved-attribute]
    assert result.certificate.primal_feasibility < 1e-7
    # ty: ignore[unresolved-attribute]
    assert result.certificate.dual_feasibility < 1e-7
    # ty: ignore[unresolved-attribute]
    assert result.certificate.complementarity < 1e-7
    dtype = jnp.asarray(0.0).dtype
    kkt_space = phx.linalg.BlockSpace(
        (
            phx.linalg.ArraySpace((1,), dtype=dtype),
            phx.linalg.ArraySpace((0,), dtype=dtype),
        )
    )
    scale = jnp.asarray(1.0, dtype=dtype)
    inverse_operator = phx.linalg.FunctionLinearOperator(
        lambda blocks: (scale * blocks[0], scale * blocks[1]),
        source=kkt_space,
        target=kkt_space,
        operator_id="jit-primal-dual-function-preconditioner",
    )
    preconditioner = phx.linalg.OperatorPreconditioner(
        inverse_operator,
        positive_definite=True,
    )
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.MINRES(),
        tolerance=phx.linalg.TolerancePolicy(
            relative=1e-8,
            absolute=1e-8,
            max_steps=200,
        ),
        preconditioning=phx.linalg.PreconditioningPolicy(preconditioner),
        differentiation=phx.linalg.DifferentiationPolicy("none"),
    )
    problem = phx.optim.MinimizationProblem(
        lambda parameters, _: jnp.sum((parameters - 2.0) ** 2),
        bounds=phx.optim.Bounds(-jnp.inf, 1.0),
    )
    solve = eqx.filter_jit(
        lambda initial: phx.optim.minimize(
            problem,
            initial,
            method=phx.optim.PrimalDualInteriorPoint(
                mode="matrix-free-centered", linear_policy=policy
            ),
            termination=_termination(),
        )
    )

    result = solve(jnp.array([0.0]))

    np.testing.assert_allclose(result.parameters, jnp.array([1.0]), atol=2e-6)
    assert int(result.status) == int(phx.optim.OptimizationStatus.SUCCESS)
    assert result.diagnostics.numeric_refreshes == (result.diagnostics.linear_solves + 1)
    equality = phx.optim.NonlinearConstraint(
        lambda parameters, _: parameters,
        lower=1.0,
        upper=1.0,
    )
    problem = phx.optim.MinimizationProblem(
        lambda parameters, _: 0.5 * jnp.sum((parameters - 2.0) ** 2),
        constraints=(equality,),
    )
    result = phx.optim.minimize(
        problem,
        jnp.array([0.0]),
        method=phx.optim.PrimalDualInteriorPoint(
            mode="matrix-free-centered",
        ),
        termination=phx.optim.OptimizationTermination(
            absolute_optimality=1e-7,
            relative_optimality=0.0,
            maximum_steps=10,
            maximum_evaluations=1,
        ),
    )

    assert int(result.status) == int(phx.optim.OptimizationStatus.SUCCESS)
    assert result.diagnostics.iterations == 1
    assert result.diagnostics.accepted_steps == 1
    assert result.diagnostics.objective_evaluations > 1
    # ty: ignore[unresolved-attribute]
    assert result.certificate.primal_feasibility <= 1e-7
    # ty: ignore[unresolved-attribute]
    assert result.certificate.dual_feasibility <= 1e-7
    constraint = phx.optim.NonlinearConstraint(
        lambda parameters, _: jnp.repeat(jnp.sum(parameters)[None], 2),
        lower=1.0,
        upper=1.0,
    )
    problem = phx.optim.MinimizationProblem(
        lambda parameters, _: 0.5 * jnp.sum((parameters - jnp.array([2.0, -1.0])) ** 2),
        constraints=(constraint,),
    )

    result = phx.optim.minimize(
        problem,
        jnp.zeros(2),
        method=phx.optim.PrimalDualInteriorPoint(
            mode="matrix-free-centered",
        ),
        termination=_termination(),
    )

    np.testing.assert_allclose(result.parameters, jnp.array([2.0, -1.0]), atol=2e-6)
    assert int(result.status) == int(phx.optim.OptimizationStatus.SUCCESS)
    assert result.diagnostics.primal_feasibility < 1e-7
    impossible = phx.optim.NonlinearConstraint(
        lambda parameters, _: jnp.ones_like(parameters),
        lower=0.0,
        upper=0.0,
    )
    problem = phx.optim.MinimizationProblem(
        lambda parameters, _: jnp.sum(0.0 * parameters),
        constraints=(impossible,),
    )

    result = phx.optim.minimize(
        problem,
        jnp.array([1.0]),
        method=phx.optim.PrimalDualInteriorPoint(
            mode="matrix-free-centered", linear_maximum_steps=3
        ),
        termination=_termination(steps=5),
    )

    assert int(result.status) == int(phx.optim.OptimizationStatus.RESTORATION_FAILED)
    assert result.diagnostics.direction_fallbacks == 1
    assert result.diagnostics.primal_feasibility == 1.0
    np.testing.assert_array_equal(result.parameters, jnp.array([1.0]))
    assert result.diagnostics.accepted_steps == 0
    assert result.diagnostics.rejected_steps == 1
    assert result.diagnostics.setup_refreshes == 1
    assert result.diagnostics.numeric_refreshes == 2
    np.testing.assert_array_equal(
        # ty: ignore[unresolved-attribute]
        result.certificate.stationarity_residual,
        jnp.array([0.0]),
    )


def test_primal_dual_eager_and_filtered_jit_agree_with_large_step_limit() -> None:
    problem = phx.optim.MinimizationProblem(
        lambda parameters, target: jnp.sum((parameters - target) ** 2),
        bounds=phx.optim.Bounds(-jnp.inf, 1.0),
        problem_id="compiled-primal-dual",
    )
    method = phx.optim.PrimalDualInteriorPoint(
        mode="matrix-free-centered",
    )
    termination = _termination(steps=100_000)

    def solve(target: Any) -> Any:
        return phx.optim.minimize(
            problem,
            jnp.array([0.0]),
            method=method,
            termination=termination,
            args=target,
        )

    eager = solve(jnp.array(2.0))
    compiled = eqx.filter_jit(solve)(jnp.array(2.0))

    np.testing.assert_allclose(compiled.parameters, eager.parameters, atol=2e-6)
    np.testing.assert_allclose(
        compiled.certificate.stationarity_residual,
        eager.certificate.stationarity_residual,
        atol=2e-6,
    )
    assert (
        int(compiled.status)
        == int(eager.status)
        == int(phx.optim.OptimizationStatus.SUCCESS)
    )
    assert int(compiled.diagnostics.iterations) == int(eager.diagnostics.iterations)
    assert int(compiled.diagnostics.setup_refreshes) == 1
    assert int(compiled.diagnostics.numeric_refreshes) == int(
        eager.diagnostics.numeric_refreshes
    )
    assert int(compiled.diagnostics.numeric_refreshes) == (
        int(compiled.diagnostics.linear_solves) + 1
    )


def test_reduced_newton_krylov_uses_incremental_state_and_adjoint_actions(
    monkeypatch: Any,
) -> None:
    problem = phx.optim.StateDesignProblem(
        lambda state, design, _: state - design,
        lambda state, design, _: jnp.sum((state - 2.0) ** 2) + 0.1 * jnp.sum(design**2),
        problem_id="reduced-newton-linear",
    )
    _forbid_explicit_jacobians(monkeypatch)

    result = phx.optim.solve_state_design(
        problem,
        jnp.array([0.0]),
        jnp.array([0.0]),
        method=phx.optim.ReducedNewtonKrylov(),
        termination=_termination(tolerance=1e-6, steps=10),
    )

    expected = jnp.array([4.0 / 2.2])
    np.testing.assert_allclose(result.state, expected, atol=2e-6)
    np.testing.assert_allclose(result.design, expected, atol=2e-6)
    assert int(result.status) == int(phx.optim.OptimizationStatus.SUCCESS)
    assert result.provenance.matrix_free
    assert "incremental state" in result.provenance.notes
    assert result.diagnostics.linear_solves > 3
    assert result.diagnostics.setup_refreshes > 0
    assert result.diagnostics.numeric_refreshes > 0


def test_reduced_newton_krylov_jit_supports_partitioned_nested_state_design() -> None:
    wrapped = _PartitionedDesign(jnp.array([0.0]), 2.0)
    initial_design, static_design = eqx.partition(
        wrapped,
        eqx.is_inexact_array,
    )
    initial_state = {"field": jnp.array([0.0])}

    def physical(design: Any) -> Any:
        return eqx.combine(design, static_design)

    problem = phx.optim.StateDesignProblem(
        lambda state, design, _: {
            "field": state["field"] - physical(design).scale * physical(design).control
        },
        lambda state, design, target: (
            jnp.sum((state["field"] - target) ** 2)
            + 0.1 * jnp.sum(physical(design).control ** 2)
        ),
        problem_id="partitioned-reduced-newton",
    )
    method = phx.optim.ReducedNewtonKrylov()
    termination = _termination(tolerance=1e-7, steps=100_000)

    def solve(target: Any) -> Any:
        return phx.optim.solve_state_design(
            problem,
            initial_state,
            initial_design,
            method=method,
            termination=termination,
            args=target,
        )

    eager = solve(jnp.array([2.0]))
    compiled = eqx.filter_jit(solve)(jnp.array([2.0]))
    eager_design = physical(eager.design)
    compiled_design = physical(compiled.design)

    np.testing.assert_allclose(
        compiled_design.control,
        eager_design.control,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        compiled.state["field"],
        eager.state["field"],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        compiled_design.control,
        jnp.array([4.0 / 4.1]),
        atol=2e-6,
    )
    np.testing.assert_allclose(
        compiled.state["field"],
        jnp.array([8.0 / 4.1]),
        atol=2e-6,
    )
    assert (
        int(compiled.status)
        == int(eager.status)
        == int(phx.optim.OptimizationStatus.SUCCESS)
    )
    assert int(compiled.diagnostics.iterations) == int(eager.diagnostics.iterations)
    assert int(compiled.diagnostics.setup_refreshes) == int(
        eager.diagnostics.setup_refreshes
    )
    assert int(compiled.diagnostics.setup_refreshes) > 3
    assert int(compiled.diagnostics.numeric_refreshes) == int(
        eager.diagnostics.numeric_refreshes
    )
