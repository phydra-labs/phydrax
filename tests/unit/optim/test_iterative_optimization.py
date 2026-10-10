#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest
from jax import Array, lax, tree

import phydrax as phx


def _termination(*, steps: Any = 50, tolerance: Any = 1e-8) -> Any:
    return phx.optim.OptimizationTermination(
        absolute_optimality=tolerance,
        relative_optimality=0.0,
        maximum_steps=steps,
    )


def test_minimization_problem_auxiliary_status_and_provenance_contracts() -> None:
    problem = phx.optim.MinimizationProblem(
        lambda value, shift: (jnp.sum((value - shift) ** 2), {"shift": shift}),
        has_aux=True,
        problem_id="auxiliary-quadratic",
    )
    result = phx.optim.minimize(
        problem,
        jnp.array([0.0, 0.0]),
        method=phx.optim.OptimistixMethod(optx.BFGS(rtol=1e-10, atol=1e-10)),
        termination=_termination(),
        args=jnp.array([2.0, -1.0]),
    )

    np.testing.assert_allclose(result.parameters, jnp.array([2.0, -1.0]), atol=1e-7)
    assert result.status == phx.optim.OptimizationStatus.SUCCESS
    assert result.successful
    assert result.provenance.problem_id == "auxiliary-quadratic"
    assert result.provenance.backend == "optimistix"
    np.testing.assert_allclose(result.auxiliary["shift"], jnp.array([2.0, -1.0]))
    # ty: ignore[invalid-argument-type]
    assert phx.optim.optimization_status_message(result.status) == "success"


def test_optimistix_adapter_supports_filtered_jit_status_paths() -> None:
    problem = phx.optim.MinimizationProblem(
        lambda value, target: jnp.sum((value - target) ** 2)
    )
    method = phx.optim.OptimistixMethod(optx.BFGS(rtol=1e-10, atol=1e-10))
    solve = eqx.filter_jit(
        lambda initial, target: phx.optim.minimize(
            problem,
            initial,
            method=method,
            termination=_termination(),
            args=target,
        )
    )

    successful = solve(jnp.array([0.0]), jnp.array([2.0]))
    nonfinite = solve(jnp.array([jnp.nan]), jnp.array([2.0]))

    np.testing.assert_allclose(successful.parameters, jnp.array([2.0]), atol=1e-7)
    assert int(successful.status) == int(phx.optim.OptimizationStatus.SUCCESS)
    assert int(successful.diagnostics.accepted_steps) == -1
    assert int(nonfinite.status) == int(phx.optim.OptimizationStatus.NONFINITE_INPUT)
    assert jnp.isnan(nonfinite.parameters[0])


def test_optimistix_adapter_rejects_unenforceable_evaluation_budget() -> None:
    problem = phx.optim.MinimizationProblem(lambda value, _: jnp.sum(value**2))
    termination = phx.optim.OptimizationTermination(
        maximum_steps=10,
        maximum_evaluations=10,
    )

    with pytest.raises(ValueError, match="cannot enforce maximum_evaluations"):
        phx.optim.minimize(
            problem,
            jnp.array([1.0]),
            method=phx.optim.OptimistixMethod(optx.BFGS(rtol=1e-8, atol=1e-8)),
            termination=termination,
        )


def test_minimization_problem_rejects_non_scalar_and_constrained_backend_mismatch() -> (
    None
):
    vector_problem = phx.optim.MinimizationProblem(lambda value, _: value)
    with pytest.raises(TypeError, match="one real scalar"):
        vector_problem.value(jnp.ones(2))

    constrained = phx.optim.MinimizationProblem(
        lambda value, _: jnp.sum(value**2),
        bounds=phx.optim.Bounds(0.0, 1.0),
    )
    with pytest.raises(ValueError, match="does not translate"):
        phx.optim.minimize(
            constrained,
            jnp.array([0.5]),
            method=phx.optim.OptimistixMethod(optx.BFGS(rtol=1e-8, atol=1e-8)),
        )
    with pytest.raises(ValueError, match="unconstrained"):
        phx.optim.minimize(
            constrained,
            jnp.array([0.5]),
            method=phx.optim.NewtonKrylov(),
        )


@pytest.mark.parametrize(
    "method",
    [phx.optim.GaussNewton(), phx.optim.LevenbergMarquardt()],
)
def test_native_nonlinear_least_squares_methods_solve_nonlinear_residual(
    method: Any,
) -> None:
    problem = phx.optim.NonlinearLeastSquaresProblem(
        lambda value, _: jnp.array([value[0] ** 2 - 4.0, value[1] - 3.0]),
        problem_id="two-residuals",
    )
    result = phx.optim.least_squares(
        problem,
        jnp.array([1.0, 0.0]),
        method=method,
        termination=_termination(steps=30),
        iteration=phx.execution.IterationPlan(
            granularity="attempt",
            observers=(phx.execution.IterationTraceObserver(32),),
        ),
    )

    np.testing.assert_allclose(result.parameters, jnp.array([2.0, 3.0]), atol=1e-6)
    assert result.status == phx.optim.OptimizationStatus.SUCCESS
    assert result.objective < 1e-12
    assert result.diagnostics.residual_evaluations > 0
    assert result.provenance.matrix_free
    assert result.iteration_evidence is not None
    assert int(result.iteration_evidence.observer_outputs[0].stored_count) > 0


@pytest.mark.parametrize("unchanged_residual", (0.0, 1.0e8))
def test_lm_accepts_resolved_decrease_below_total_objective_rounding(
    unchanged_residual: float,
) -> None:
    def residual(parameters: Array, args: None) -> tuple[Array, Array]:
        return jnp.asarray(unchanged_residual, dtype=np.float64), parameters - 1.0

    result = phx.optim.least_squares(
        residual,
        jnp.zeros(1, dtype=np.float64),
        method=phx.optim.LevenbergMarquardt(maximum_trials=4),
        termination=phx.optim.OptimizationTermination(
            absolute_optimality=1.0e-10,
            relative_optimality=0.0,
            maximum_steps=16,
        ),
    )
    np.testing.assert_allclose(
        result.parameters, np.ones(1, dtype=np.float64), atol=1.0e-9
    )
    assert result.status == phx.optim.OptimizationStatus.SUCCESS
    assert result.diagnostics.accepted_steps > 0
    assert result.diagnostics.rejected_steps == 0


class _DirectionalTrialPolicy(phx.optim.AbstractLeastSquaresTrialPolicy):
    scale: Array
    direction_factor: Array
    reverse_image: bool = eqx.field(static=True)

    def __init__(
        self,
        scale: Array,
        *,
        reverse_image: bool = False,
        direction_factor: Array | None = None,
    ) -> None:
        self.scale = scale
        self.direction_factor = (
            jnp.asarray(1.0) if direction_factor is None else direction_factor
        )
        self.reverse_image = reverse_image

    def __call__(
        self,
        parameters: Array,
        direction: Array,
        residual: Array,
        jacobian: phx.linalg.JacobianLinearOperator,
        remaining_model_work: Array,
        args: Any,
    ) -> phx.optim.LeastSquaresTrialResult:
        selected = self.direction_factor * direction
        image = jacobian.mv(selected)
        if self.reverse_image:
            image = tree.map(lambda value: -value, image)
        return phx.optim.LeastSquaresTrialResult(
            direction=selected,
            scale=self.scale,
            model_image=image,
            valid=jnp.asarray(True),
            jvp_actions=1,
            vjp_actions=0,
            model_work_units=0,
            model_visits=0,
            resource_refused=False,
        )

    def termination_valid(self, residual: Array, args: Any) -> Array:
        return jnp.all(residual == 0.0)


@pytest.mark.parametrize("scale", (0.25, 0.5))
def test_lm_trial_policy_scales_actual_step_with_dynamic_state(scale: float) -> None:
    policy = _DirectionalTrialPolicy(jnp.asarray(scale))
    problem = phx.optim.NonlinearLeastSquaresProblem(
        lambda value, target: value - target,
        trial_policy=policy,
        trial_policy_id="exact-directional-test",
        trial_model_work_limit=jnp.asarray(0),
        problem_id="dynamic-trial",
    )
    result = eqx.filter_jit(
        lambda bound, target: phx.optim.least_squares(
            bound,
            jnp.zeros(1),
            args=target,
            method=phx.optim.LevenbergMarquardt(initial_damping=100.0, maximum_trials=4),
            termination=_termination(steps=1, tolerance=0.0),
        )
    )(problem, jnp.ones(1))
    np.testing.assert_allclose(result.parameters, (scale / 101.0,), rtol=1e-8, atol=0.0)
    assert int(result.diagnostics.accepted_steps) == 1
    assert int(result.diagnostics.linear_solves) == 1
    assert float(result.diagnostics.accepted_step_size) == scale
    assert int(result.method_evidence["trial_policy_evaluations"]) == 1
    assert int(result.method_evidence["trial_policy_refusals"]) == 0
    assert int(result.method_evidence["trial_policy_jvp_actions"]) == 1
    assert int(result.method_evidence["trial_policy_vjp_actions"]) == 0
    assert (
        result.provenance.problem_id
        == "dynamic-trial/trial-policy:exact-directional-test"
    )


def test_lm_trial_policy_rejects_outgoing_ascent_without_false_success() -> None:
    problem = phx.optim.NonlinearLeastSquaresProblem(
        lambda value, _: value - 1.0,
        trial_policy=_DirectionalTrialPolicy(jnp.asarray(1.0), reverse_image=True),
        trial_policy_id="outgoing-ascent-test",
        trial_model_work_limit=jnp.asarray(0),
    )
    result = phx.optim.least_squares(
        problem,
        jnp.zeros(1),
        method=phx.optim.LevenbergMarquardt(maximum_trials=4),
        termination=_termination(steps=32, tolerance=0.0),
    )
    np.testing.assert_array_equal(result.parameters, (0.0,))
    assert result.status == phx.optim.OptimizationStatus.STAGNATION
    assert int(result.diagnostics.linear_solves) == 4
    assert int(result.diagnostics.accepted_steps) == 0
    assert int(result.method_evidence["trial_policy_evaluations"]) == 4
    assert int(result.method_evidence["trial_policy_refusals"]) == 4
    assert int(result.method_evidence["trial_policy_jvp_actions"]) == 4
    assert int(result.method_evidence["trial_policy_vjp_actions"]) == 0
    # Refused policies still performed their declared derivative action, but
    # never invoked the actual residual at an inadmissible proposed point.
    assert int(result.diagnostics.globalization_evaluations) == 0


def test_lm_zero_selected_gradient_requires_policy_termination_certificate() -> None:
    problem = phx.optim.NonlinearLeastSquaresProblem(
        lambda value, _: jnp.ones_like(value),
        trial_policy=_DirectionalTrialPolicy(jnp.asarray(1.0)),
        trial_policy_id="nonregular-zero-gradient-test",
        trial_model_work_limit=jnp.asarray(0),
    )
    result = phx.optim.least_squares(
        problem,
        jnp.zeros(1),
        method=phx.optim.LevenbergMarquardt(maximum_trials=4),
        termination=_termination(steps=32, tolerance=0.0),
    )
    assert result.status == phx.optim.OptimizationStatus.STAGNATION
    assert not bool(result.successful)
    assert int(result.diagnostics.linear_solves) == 0
    np.testing.assert_array_equal(result.residual, (1.0,))
    assert int(result.method_evidence["trial_policy_evaluations"]) == 0
    assert int(result.method_evidence["trial_policy_jvp_actions"]) == 0
    assert int(result.method_evidence["trial_policy_vjp_actions"]) == 0


@pytest.mark.parametrize("scale", (0.0, -1.0, 1.01, np.nan))
def test_lm_trial_policy_refuses_invalid_bounds_and_records_actual_calls(
    scale: float,
) -> None:
    problem = phx.optim.NonlinearLeastSquaresProblem(
        lambda value, _: value - 1.0,
        trial_policy=_DirectionalTrialPolicy(jnp.asarray(scale)),
        trial_policy_id="invalid-bound-test",
        trial_model_work_limit=jnp.asarray(0),
    )
    result = phx.optim.least_squares(
        problem,
        jnp.zeros(1),
        method=phx.optim.LevenbergMarquardt(maximum_trials=4),
        termination=_termination(steps=32, tolerance=0.0),
    )
    assert result.status == phx.optim.OptimizationStatus.STAGNATION
    np.testing.assert_array_equal(result.parameters, (0.0,))
    assert int(result.method_evidence["trial_policy_evaluations"]) == 4
    assert int(result.method_evidence["trial_policy_refusals"]) == 4
    assert int(result.method_evidence["trial_policy_jvp_actions"]) == 4
    assert int(result.method_evidence["trial_policy_vjp_actions"]) == 0
    assert int(result.diagnostics.globalization_evaluations) == 0


def test_lm_trial_policy_uses_returned_direction_for_proposal_model_and_norm() -> None:
    problem = phx.optim.NonlinearLeastSquaresProblem(
        lambda value, _: value - 1.0,
        trial_policy=_DirectionalTrialPolicy(
            jnp.asarray(0.5), direction_factor=jnp.asarray(2.0)
        ),
        trial_policy_id="returned-direction-test",
        trial_model_work_limit=jnp.asarray(0),
    )
    result = phx.optim.least_squares(
        problem,
        jnp.zeros(1),
        method=phx.optim.LevenbergMarquardt(initial_damping=100.0, maximum_trials=4),
        termination=_termination(steps=1, tolerance=0.0),
    )
    np.testing.assert_allclose(result.parameters, (1.0 / 101.0,), rtol=1e-8, atol=0.0)
    np.testing.assert_allclose(
        result.diagnostics.final_step_norm,
        jnp.linalg.norm(result.parameters),
        rtol=0.0,
        atol=0.0,
    )
    assert float(result.diagnostics.accepted_step_size) == 0.5
    assert int(result.diagnostics.accepted_steps) == 1
    assert int(result.method_evidence["trial_policy_jvp_actions"]) == 1


@pytest.mark.parametrize("factor", (0.0, np.inf))
def test_lm_trial_policy_refuses_unusable_returned_direction(factor: float) -> None:
    problem = phx.optim.NonlinearLeastSquaresProblem(
        lambda value, _: value - 1.0,
        trial_policy=_DirectionalTrialPolicy(
            jnp.asarray(1.0), direction_factor=jnp.asarray(factor)
        ),
        trial_policy_id="unusable-returned-direction-test",
        trial_model_work_limit=jnp.asarray(0),
    )
    result = phx.optim.least_squares(
        problem,
        jnp.zeros(1),
        method=phx.optim.LevenbergMarquardt(maximum_trials=4),
        termination=_termination(steps=32, tolerance=0.0),
    )
    assert result.status == phx.optim.OptimizationStatus.STAGNATION
    np.testing.assert_array_equal(result.parameters, (0.0,))
    assert int(result.method_evidence["trial_policy_evaluations"]) == 4
    assert int(result.method_evidence["trial_policy_jvp_actions"]) == 4
    assert int(result.method_evidence["trial_policy_vjp_actions"]) == 0
    assert int(result.diagnostics.globalization_evaluations) == 0


class _ConditionalModelTrialPolicy(phx.optim.AbstractLeastSquaresTrialPolicy):
    def __call__(
        self,
        parameters: Array,
        direction: Array,
        residual: Array,
        jacobian: phx.linalg.JacobianLinearOperator,
        remaining_model_work: Array,
        args: Array,
    ) -> phx.optim.LeastSquaresTrialResult:
        image = jacobian.mv(direction)
        first_prediction = -jnp.sum(residual * image) - 0.5 * jnp.sum(image * image)

        def second_model(_: None) -> phx.optim.LeastSquaresTrialResult:
            selected = 0.5 * direction
            selected_image = jacobian.mv(selected)
            prediction = -jnp.sum(residual * selected_image) - 0.5 * jnp.sum(
                selected_image * selected_image
            )
            return phx.optim.LeastSquaresTrialResult(
                direction=selected,
                scale=jnp.asarray(1.0),
                model_image=selected_image,
                valid=(first_prediction > 0.0) & (prediction > 0.0),
                jvp_actions=2,
                vjp_actions=0,
                model_work_units=8 * residual.size,
                model_visits=2,
                resource_refused=False,
            )

        def first_model(_: None) -> phx.optim.LeastSquaresTrialResult:
            return phx.optim.LeastSquaresTrialResult(
                direction=direction,
                scale=jnp.asarray(1.0),
                model_image=image,
                valid=first_prediction > 0.0,
                jvp_actions=1,
                vjp_actions=0,
                model_work_units=4 * residual.size,
                model_visits=1,
                resource_refused=False,
            )

        return lax.cond(args, second_model, first_model, None)

    def termination_valid(self, residual: Array, args: Array) -> Array:
        return jnp.all(residual == 0.0)


@pytest.mark.parametrize("second_model", (False, True))
def test_lm_trial_policy_accumulates_executed_conditional_model_receipts(
    second_model: bool,
) -> None:
    problem = phx.optim.NonlinearLeastSquaresProblem(
        lambda value, _: value - 1.0,
        trial_policy=_ConditionalModelTrialPolicy(),
        trial_policy_id="conditional-model-receipt-test",
        trial_model_work_limit=jnp.asarray(100),
    )
    result = eqx.filter_jit(
        lambda flag: phx.optim.least_squares(
            problem,
            jnp.zeros(1),
            args=flag,
            method=phx.optim.LevenbergMarquardt(initial_damping=100.0, maximum_trials=4),
            termination=_termination(steps=1, tolerance=0.0),
        )
    )(jnp.asarray(second_model))
    evidence = result.method_evidence
    assert bool(evidence["trial_policy_receipts_valid"])
    assert int(evidence["trial_policy_model_visits"]) == (2 if second_model else 1)
    assert int(evidence["trial_policy_jvp_actions"]) == (2 if second_model else 1)
    assert int(evidence["trial_policy_model_work_units"]) > 0
    assert int(result.diagnostics.linear_solves) == 1
    assert int(result.diagnostics.globalization_evaluations) == 1


class _InvalidReceiptTrialPolicy(phx.optim.AbstractLeastSquaresTrialPolicy):
    def __call__(
        self,
        parameters: Array,
        direction: Array,
        residual: Array,
        jacobian: phx.linalg.JacobianLinearOperator,
        remaining_model_work: Array,
        args: Any,
    ) -> phx.optim.LeastSquaresTrialResult:
        image = jacobian.mv(direction)
        return phx.optim.LeastSquaresTrialResult(
            direction=direction,
            scale=jnp.asarray(1.0),
            model_image=image,
            valid=jnp.asarray(True),
            jvp_actions=1,
            vjp_actions=0,
            model_work_units=-1,
            model_visits=0,
            resource_refused=False,
        )

    def termination_valid(self, residual: Array, args: Any) -> Array:
        return jnp.all(residual == 0.0)


def test_lm_trial_policy_invalid_receipt_is_retained_and_cannot_credit_work() -> None:
    problem = phx.optim.NonlinearLeastSquaresProblem(
        lambda value, _: value - 1.0,
        trial_policy=_InvalidReceiptTrialPolicy(),
        trial_policy_id="invalid-work-receipt-test",
        trial_model_work_limit=jnp.asarray(0),
    )
    result = phx.optim.least_squares(
        problem,
        jnp.zeros(1),
        method=phx.optim.LevenbergMarquardt(maximum_trials=4),
        termination=_termination(steps=32, tolerance=0.0),
    )
    assert result.status == phx.optim.OptimizationStatus.INVALID_DIRECTION
    assert not bool(result.method_evidence["trial_policy_receipts_valid"])
    assert int(result.method_evidence["trial_policy_model_work_units"]) < 0
    np.testing.assert_array_equal(result.parameters, (0.0,))
    assert int(result.diagnostics.globalization_evaluations) == 0


class _BudgetedModelTrialPolicy(phx.optim.AbstractLeastSquaresTrialPolicy):
    def __call__(
        self,
        parameters: Array,
        direction: Array,
        residual: Array,
        jacobian: phx.linalg.JacobianLinearOperator,
        remaining_model_work: Array,
        args: Any,
    ) -> phx.optim.LeastSquaresTrialResult:
        required = 4 * residual.size

        def admitted(_: None) -> phx.optim.LeastSquaresTrialResult:
            image = jacobian.mv(direction)
            prediction = -jnp.sum(residual * image) - 0.5 * jnp.sum(image * image)
            return phx.optim.LeastSquaresTrialResult(
                direction=direction,
                scale=jnp.asarray(1.0),
                model_image=image,
                valid=prediction > 0.0,
                jvp_actions=1,
                vjp_actions=0,
                model_work_units=required,
                model_visits=1,
                resource_refused=False,
            )

        def refused(_: None) -> phx.optim.LeastSquaresTrialResult:
            return phx.optim.LeastSquaresTrialResult(
                direction=direction,
                scale=jnp.asarray(0.0),
                model_image=jnp.zeros_like(residual),
                valid=jnp.asarray(False),
                jvp_actions=0,
                vjp_actions=0,
                model_work_units=0,
                model_visits=0,
                resource_refused=True,
            )

        return lax.cond(remaining_model_work >= required, admitted, refused, None)

    def termination_valid(self, residual: Array, args: Any) -> Array:
        return jnp.all(residual == 0.0)


def test_lm_trial_policy_dynamic_work_limit_debits_actual_receipts_and_refuses_resource() -> (
    None
):
    problem = phx.optim.NonlinearLeastSquaresProblem(
        lambda value, _: value - 1.0,
        trial_policy=_BudgetedModelTrialPolicy(),
        trial_policy_id="dynamic-model-work-limit-test",
        trial_model_work_limit=jnp.asarray(4),
    )
    result = eqx.filter_jit(
        lambda initial: phx.optim.least_squares(
            problem,
            initial,
            method=phx.optim.LevenbergMarquardt(initial_damping=100.0, maximum_trials=4),
            termination=_termination(steps=32, tolerance=0.0),
        )
    )(jnp.zeros(1))
    assert result.status == phx.optim.OptimizationStatus.RESOURCE_LIMIT
    assert int(result.diagnostics.accepted_steps) == 1
    assert int(result.diagnostics.globalization_evaluations) == 1
    assert int(result.method_evidence["trial_policy_model_visits"]) == 1
    assert int(result.method_evidence["trial_policy_model_work_units"]) == 4
    assert bool(result.method_evidence["trial_policy_resource_refused"])
    assert bool(result.method_evidence["trial_policy_receipts_valid"])
    assert float(result.parameters[0]) > 0.0


def test_native_minimization_control_stops_at_an_accepted_point() -> None:
    iteration = phx.execution.IterationPlan(
        granularity="attempt",
        observers=(phx.execution.IterationTraceObserver(8),),
        stop_rule=phx.execution.CallableIterationStopRule(
            lambda initial: jnp.asarray(0, dtype=jnp.int32),
            lambda state, record: (
                state + 1,
                record.coordinates.ordinal >= 1,
            ),
            "one-optimization-step",
        ),
    )
    result = phx.optim.minimize(
        lambda value, _: (1.0 - value[0]) ** 2 + 100.0 * (value[1] - value[0] ** 2) ** 2,
        jnp.asarray([-1.2, 1.0]),
        method=phx.optim.NonlinearConjugateGradient(),
        termination=_termination(steps=50),
        iteration=iteration,
    )

    assert int(result.status) == int(phx.optim.OptimizationStatus.USER_STOPPED)
    assert int(result.diagnostics.iterations) == 1
    assert result.iteration_evidence is not None
    assert int(result.iteration_evidence.observer_outputs[0].stored_count) == 1


def test_external_optimization_exposes_terminal_evidence_only() -> None:
    result = phx.optim.minimize(
        lambda value, target: jnp.sum((value - target) ** 2),
        jnp.asarray([0.0]),
        method=phx.optim.OptimistixMethod(optx.BFGS(rtol=1e-10, atol=1e-10)),
        args=jnp.asarray([2.0]),
        iteration=phx.execution.IterationPlan(
            granularity="terminal",
            observers=(phx.execution.IterationTraceObserver(0),),
        ),
    )

    assert result.iteration_evidence is not None
    trace = result.iteration_evidence.observer_outputs[0]
    assert int(trace.stored_count) == 0
    assert int(trace.terminal.status) == int(phx.optim.OptimizationStatus.SUCCESS)


def test_gauss_newton_handles_rectangular_rank_deficient_residual() -> None:
    result = phx.optim.least_squares(
        lambda value, _: jnp.array([value[0] + value[1] - 2.0]),
        jnp.array([0.0, 0.0]),
        method=phx.optim.GaussNewton(),
        termination=_termination(steps=20),
    )

    np.testing.assert_allclose(jnp.sum(result.parameters), 2.0, atol=1e-7)
    assert result.status == phx.optim.OptimizationStatus.SUCCESS
    assert result.diagnostics.linear_solves >= 1


def test_newton_krylov_uses_descent_fallback_for_indefinite_hessian() -> None:
    result = phx.optim.minimize(
        lambda value, _: jnp.sum(value**4 - value**2),
        jnp.array([0.2]),
        method=phx.optim.NewtonKrylov(),
        termination=_termination(steps=40),
    )

    np.testing.assert_allclose(result.parameters, jnp.array([2.0**-0.5]), atol=1e-5)
    assert result.status == phx.optim.OptimizationStatus.SUCCESS
    assert result.diagnostics.direction_fallbacks >= 1
    assert result.diagnostics.hvp_evaluations >= 1
    assert result.provenance.method == "newton-krylov"


def test_newton_krylov_consumes_supplied_hessian_action() -> None:
    diagonal = jnp.asarray((2.0, 5.0))
    target = jnp.asarray((1.5, -0.4))
    problem = phx.optim.MinimizationProblem(
        lambda value, _: 0.5 * jnp.sum(diagonal * (value - target) ** 2),
        hessian_action=lambda _value, direction, _args: diagonal * direction,
        hessian_action_kind="exact",
        problem_id="supplied-hessian-quadratic",
    )

    result = phx.optim.minimize(
        problem,
        jnp.zeros(2),
        method=phx.optim.NewtonKrylov(),
        termination=_termination(steps=8),
    )

    np.testing.assert_allclose(result.parameters, target, atol=1e-8)
    assert bool(result.successful)
    assert problem.hessian_action_kind == "exact"
    assert result.method_evidence == "exact"
    np.testing.assert_allclose(
        problem.apply_hessian(jnp.zeros(2), jnp.ones(2)),
        diagonal,
    )


def test_minimization_problem_requires_hessian_action_identity() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        phx.optim.MinimizationProblem(
            lambda value, _: jnp.sum(value**2),
            hessian_action=lambda _value, direction, _args: direction,
        )


def test_newton_krylov_forcing_controls_inner_accuracy_under_jit() -> None:
    diagonal = jnp.logspace(0.0, 6.0, 24)

    def objective(value: Any, _: Any) -> Any:
        return 0.5 * jnp.sum(diagonal * value**2)

    termination = _termination(steps=1, tolerance=0.0)
    initial = jnp.ones((24,))
    loose_method = phx.optim.NewtonKrylov(
        minimum_forcing=0.5,
        maximum_forcing=0.5,
    )
    tight_method = phx.optim.NewtonKrylov(
        minimum_forcing=1e-8,
        maximum_forcing=1e-8,
    )

    def solve(method: Any) -> Any:
        return phx.optim.minimize(
            objective,
            initial,
            method=method,
            termination=termination,
        )

    loose = solve(loose_method)
    tight = solve(tight_method)
    compiled_loose = eqx.filter_jit(lambda: solve(loose_method))()
    compiled_tight = eqx.filter_jit(lambda: solve(tight_method))()

    assert int(loose.diagnostics.linear_iterations) < int(
        tight.diagnostics.linear_iterations
    )
    assert float(tight.diagnostics.final_optimality_norm) < float(
        loose.diagnostics.final_optimality_norm
    )
    assert int(compiled_loose.diagnostics.linear_iterations) == int(
        loose.diagnostics.linear_iterations
    )
    assert int(compiled_tight.diagnostics.linear_iterations) == int(
        tight.diagnostics.linear_iterations
    )


def test_nonfinite_initial_parameters_return_typed_status() -> None:
    result = phx.optim.minimize(
        lambda value, _: jnp.sum(value**2),
        jnp.array([jnp.nan]),
        method=phx.optim.NewtonKrylov(),
        termination=_termination(),
    )

    assert result.status == phx.optim.OptimizationStatus.NONFINITE_INPUT
    assert not result.successful


def _finite_only_at_nonpositive_parameters(parameters: Any) -> Any:
    value = parameters[0]
    return jnp.where(value <= 0.0, (value - 1.0) ** 2, jnp.nan)


def _finite_only_at_nonpositive_residual(parameters: Any) -> Any:
    value = parameters[0]
    return jnp.where(
        value <= 0.0,
        jnp.asarray([value - 1.0]),
        jnp.asarray([jnp.nan]),
    )


def test_scalar_and_bound_line_searches_report_all_nonfinite_trials() -> None:
    search = phx.optim.ArmijoLineSearch(maximum_steps=2)
    unconstrained = phx.optim.minimize(
        lambda parameters, _: _finite_only_at_nonpositive_parameters(parameters),
        jnp.array([0.0]),
        method=phx.optim.NewtonKrylov(line_search=search),
        termination=_termination(steps=2, tolerance=0.0),
    )
    bounded = phx.optim.minimize(
        phx.optim.MinimizationProblem(
            lambda parameters, _: _finite_only_at_nonpositive_parameters(parameters),
            bounds=phx.optim.Bounds(-1.0, 2.0),
        ),
        jnp.array([0.0]),
        method=phx.optim.ProjectedGradient(line_search=search),
        termination=_termination(steps=2, tolerance=0.0),
    )

    for result in (unconstrained, bounded):
        assert result.status == phx.optim.OptimizationStatus.NONFINITE_EVALUATION
        np.testing.assert_array_equal(result.parameters, jnp.array([0.0]))
        assert result.diagnostics.rejected_steps == 1


@pytest.mark.parametrize(
    "method",
    [
        phx.optim.GaussNewton(line_search=phx.optim.ArmijoLineSearch(maximum_steps=2)),
        phx.optim.LevenbergMarquardt(maximum_trials=2),
    ],
)
def test_least_squares_methods_report_all_nonfinite_trials(method: Any) -> None:
    result = phx.optim.least_squares(
        lambda parameters, _: _finite_only_at_nonpositive_residual(parameters),
        jnp.array([0.0]),
        method=method,
        termination=_termination(steps=2, tolerance=0.0),
    )

    assert result.status == phx.optim.OptimizationStatus.NONFINITE_EVALUATION
    np.testing.assert_array_equal(result.parameters, jnp.array([0.0]))
    assert result.diagnostics.rejected_steps == 1


def test_composite_line_search_reports_all_nonfinite_trials() -> None:
    result = phx.optim.composite_least_squares(
        phx.optim.CompositeLeastSquaresProblem(
            lambda parameters, _: _finite_only_at_nonpositive_residual(parameters),
            lambda parameters, _: 0.05 * jnp.sum(parameters**2),
        ),
        jnp.array([0.0]),
        method=phx.optim.GeneralizedGaussNewton(
            line_search=phx.optim.ArmijoLineSearch(maximum_steps=2)
        ),
        termination=_termination(steps=2, tolerance=0.0),
    )

    assert result.status == phx.optim.OptimizationStatus.NONFINITE_EVALUATION
    np.testing.assert_array_equal(result.parameters, jnp.array([0.0]))
    assert result.diagnostics.rejected_steps == 1


def test_scipy_minimize_accepts_fused_explicit_host_gradients() -> None:
    target = jnp.asarray([1.5, -0.25])
    evaluations = []

    def objective(*_: Any) -> None:
        raise AssertionError("The automatic objective must not run.")

    def explicit(parameters: Any, desired: Any) -> Any:
        evaluations.append(np.asarray(parameters))
        difference = parameters - desired
        return 0.5 * jnp.sum(difference**2), difference

    problem = phx.optim.MinimizationProblem(
        objective,
        explicit_value_and_gradient=explicit,
        derivative_execution="explicit-host",
        problem_id="explicit-host-quadratic",
    )
    result = phx.optim.minimize(
        problem,
        jnp.zeros((2,)),
        args=target,
        method=phx.optim.SciPyMinimize("BFGS", options={"gtol": 1.0e-10}),
        termination=phx.optim.OptimizationTermination(
            absolute_optimality=1.0e-8,
            relative_optimality=0.0,
            maximum_steps=32,
        ),
    )

    assert bool(result.successful)
    np.testing.assert_allclose(result.parameters, target, atol=1.0e-8)
    assert evaluations
    with pytest.raises(ValueError, match="does not support explicit host gradients"):
        phx.optim.minimize(
            problem,
            jnp.zeros((2,)),
            args=target,
            method=phx.optim.ProjectedLBFGS(),
        )


def test_newton_krylov_backtracking_uses_only_remaining_evaluation_budget() -> None:
    method = phx.optim.NewtonKrylov()
    parameters = jnp.asarray([2.0])
    state = method.init(parameters)
    state = eqx.tree_at(
        lambda current: current.objective_evaluations,
        state,
        jnp.asarray(1, dtype=jnp.int32),
    )
    _, next_state, _ = method.step(
        lambda value: jnp.sum(value**4),
        parameters,
        state,
        termination=phx.optim.OptimizationTermination(
            absolute_optimality=0.0,
            relative_optimality=0.0,
            maximum_steps=10,
            maximum_evaluations=2,
        ),
    )

    assert int(next_state.objective_evaluations) == 2
    assert not bool(next_state.metrics.accepted)
