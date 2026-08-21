#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._linear_refresh import LinearRefreshState, prepare_refresh_state
from .._strict import StrictModule
from ..linalg import (
    GeneralizedLSMR,
    IdentityLinearOperator,
    JacobianLinearOperator,
    LeastSquaresProblem as LinearLeastSquaresProblem,
    LinearSolvePolicy,
    LinearSolveStatus,
    prepare_linearization,
    PyTreeSpace,
    ScaledLinearOperator,
    solve as solve_linear,
    TolerancePolicy,
)
from ._iterative._base import AbstractLeastSquaresMethod
from ._iterative._globalization import armijo_backtracking, ArmijoLineSearch
from ._iterative._types import (
    _tree_add_scaled,
    _tree_allfinite,
    _tree_inner,
    _tree_negative,
    _tree_norm,
    _tree_where,
    _validate_real_inexact_tree,
    IterativeStepMetrics,
    LeastSquaresResult,
    NonlinearLeastSquaresProblem,
    OptimizationCapabilities,
    OptimizationDiagnostics,
    OptimizationProvenance,
    OptimizationStatus,
    OptimizationTermination,
)


class LeastSquaresState(StrictModule):
    """Persistent state and exact counters for native least-squares methods."""

    iteration: Array
    initial_optimality_norm: Array
    damping: Array
    accepted_steps: Array
    rejected_steps: Array
    residual_evaluations: Array
    scalar_evaluations: Array
    scalar_gradient_evaluations: Array
    scalar_hvp_evaluations: Array
    jvp_evaluations: Array
    vjp_evaluations: Array
    linear_iterations: Array
    linear_solves: Array
    direction_fallbacks: Array
    setup_refreshes: Array
    numeric_refreshes: Array
    linear_refresh_state: LinearRefreshState | None
    metrics: IterativeStepMetrics

    def __init__(
        self,
        *,
        iteration: Any = 0,
        initial_optimality_norm: Any = jnp.nan,
        damping: Any = 0.0,
        accepted_steps: Any = 0,
        rejected_steps: Any = 0,
        residual_evaluations: Any = 0,
        scalar_evaluations: Any = 0,
        scalar_gradient_evaluations: Any = 0,
        scalar_hvp_evaluations: Any = 0,
        jvp_evaluations: Any = 0,
        vjp_evaluations: Any = 0,
        linear_iterations: Any = 0,
        linear_solves: Any = 0,
        direction_fallbacks: Any = 0,
        setup_refreshes: Any = 0,
        numeric_refreshes: Any = 0,
        linear_refresh_state: LinearRefreshState | None = None,
        metrics: IterativeStepMetrics | None = None,
    ):
        self.iteration = jnp.asarray(iteration, dtype=jnp.int32)
        self.initial_optimality_norm = jnp.asarray(initial_optimality_norm)
        self.damping = jnp.asarray(damping)
        self.accepted_steps = jnp.asarray(accepted_steps, dtype=jnp.int32)
        self.rejected_steps = jnp.asarray(rejected_steps, dtype=jnp.int32)
        self.residual_evaluations = jnp.asarray(residual_evaluations, dtype=jnp.int32)
        self.scalar_evaluations = jnp.asarray(scalar_evaluations, dtype=jnp.int32)
        self.scalar_gradient_evaluations = jnp.asarray(
            scalar_gradient_evaluations, dtype=jnp.int32
        )
        self.scalar_hvp_evaluations = jnp.asarray(scalar_hvp_evaluations, dtype=jnp.int32)
        self.jvp_evaluations = jnp.asarray(jvp_evaluations, dtype=jnp.int32)
        self.vjp_evaluations = jnp.asarray(vjp_evaluations, dtype=jnp.int32)
        self.linear_iterations = jnp.asarray(linear_iterations, dtype=jnp.int32)
        self.linear_solves = jnp.asarray(linear_solves, dtype=jnp.int32)
        self.setup_refreshes = jnp.asarray(setup_refreshes, dtype=jnp.int32)
        self.numeric_refreshes = jnp.asarray(numeric_refreshes, dtype=jnp.int32)
        if linear_refresh_state is not None and not isinstance(
            linear_refresh_state, LinearRefreshState
        ):
            raise TypeError("linear_refresh_state must be a LinearRefreshState or None.")
        self.linear_refresh_state = linear_refresh_state
        self.direction_fallbacks = jnp.asarray(direction_fallbacks, dtype=jnp.int32)
        self.metrics = IterativeStepMetrics() if metrics is None else metrics


class _ResidualModel(StrictModule):
    residual: PyTree[Array]
    jacobian: JacobianLinearOperator
    gradient: PyTree[Array]
    objective: Array
    optimality_norm: Array

    def __init__(
        self,
        *,
        residual: PyTree[Array],
        jacobian: JacobianLinearOperator,
        gradient: PyTree[Array],
        objective: Array,
        optimality_norm: Array,
    ):
        self.residual = residual
        self.jacobian = jacobian
        self.gradient = gradient
        self.objective = jnp.asarray(objective)
        self.optimality_norm = jnp.asarray(optimality_norm)


def _default_linear_policy() -> LinearSolvePolicy:
    return LinearSolvePolicy(
        GeneralizedLSMR(),
        tolerance=TolerancePolicy(relative=1e-6, absolute=1e-10),
    )


def _usable_inexact_linear_status(status: Any, /):
    status_ = jnp.asarray(status, dtype=jnp.int32)
    return (
        (status_ == int(LinearSolveStatus.SUCCESS))
        | (status_ == int(LinearSolveStatus.MAXIMUM_STEPS_REACHED))
        | (status_ == int(LinearSolveStatus.STAGNATION))
        | (status_ == int(LinearSolveStatus.CONDITION_LIMIT_REACHED))
    )


def _prepare_residual_model(
    residual_function: Callable[[PyTree[Any]], PyTree[Any]],
    parameters: PyTree[Any],
    /,
) -> _ResidualModel:
    source = PyTreeSpace(parameters)
    linearization = prepare_linearization(
        residual_function,
        parameters,
        source=source,
    )
    jacobian = JacobianLinearOperator(linearization)
    residual = linearization.primal
    gradient = jacobian.adjoint_mv(residual)
    objective = 0.5 * _tree_inner(residual, residual)
    return _ResidualModel(
        residual=residual,
        jacobian=jacobian,
        gradient=gradient,
        objective=objective,
        optimality_norm=_tree_norm(gradient),
    )


def _linear_least_squares_problem(
    model: _ResidualModel,
    damping: Any | None = None,
    /,
) -> LinearLeastSquaresProblem:
    regularizer = None
    if damping is not None:
        identity = IdentityLinearOperator(model.jacobian.source)
        regularizer = ScaledLinearOperator(identity, jnp.sqrt(damping))
    return LinearLeastSquaresProblem(model.jacobian, regularizer=regularizer)


def _initial_optimality(state: LeastSquaresState, current: Array, /) -> Array:
    return jnp.where(state.iteration == 0, current, state.initial_optimality_norm)


def _converged(
    termination: OptimizationTermination | None,
    initial: Array,
    current: Array,
    /,
):
    if termination is None:
        return jnp.asarray(False)
    return current <= termination.optimality_threshold(initial)


def _stagnated(
    termination: OptimizationTermination | None,
    parameters: PyTree[Any],
    step_norm: Array,
    /,
):
    if termination is None:
        return jnp.asarray(False)
    return step_norm <= termination.step_threshold(_tree_norm(parameters))


class GaussNewton(AbstractLeastSquaresMethod):
    """Matrix-free Gauss–Newton with direct rectangular linear least squares."""

    linear_policy: LinearSolvePolicy
    line_search: ArmijoLineSearch

    def __init__(
        self,
        *,
        linear_policy: LinearSolvePolicy | None = None,
        line_search: ArmijoLineSearch | None = None,
    ):
        policy = _default_linear_policy() if linear_policy is None else linear_policy
        search = ArmijoLineSearch() if line_search is None else line_search
        if not isinstance(policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be a LinearSolvePolicy or None.")
        if not isinstance(search, ArmijoLineSearch):
            raise TypeError("line_search must be an ArmijoLineSearch or None.")
        self.linear_policy = policy
        self.line_search = search

    @property
    def method_id(self) -> str:
        return "gauss-newton"

    @property
    def globalization_id(self) -> str:
        return "armijo"

    @property
    def capabilities(self) -> OptimizationCapabilities:
        return OptimizationCapabilities(
            scalar_objective=False,
            residual_objective=True,
            matrix_free=True,
            prepared_refresh=True,
            implicit_differentiation=True,
        )

    def init(self, parameters: PyTree[Any], /) -> LeastSquaresState:
        parameters = _validate_real_inexact_tree(parameters, name="parameters")
        PyTreeSpace(parameters)
        metric_nan = jnp.asarray(jnp.nan, dtype=_tree_norm(parameters).dtype)
        return LeastSquaresState(
            initial_optimality_norm=metric_nan,
            metrics=IterativeStepMetrics(objective=metric_nan),
        )

    def prepare_state(
        self,
        residual_function: Callable[[PyTree[Any]], PyTree[Any]],
        parameters: PyTree[Any],
        /,
    ) -> LeastSquaresState:
        parameters = _validate_real_inexact_tree(parameters, name="parameters")
        model = _prepare_residual_model(residual_function, parameters)
        metric_nan = jnp.asarray(jnp.nan, dtype=model.objective.dtype)
        _, refresh_state = prepare_refresh_state(
            _linear_least_squares_problem(model),
            self.linear_policy,
        )
        return LeastSquaresState(
            initial_optimality_norm=metric_nan,
            residual_evaluations=1,
            vjp_evaluations=1,
            setup_refreshes=1,
            numeric_refreshes=1,
            linear_refresh_state=refresh_state,
            metrics=IterativeStepMetrics(objective=metric_nan),
        )

    def step(
        self,
        residual_function: Callable[[PyTree[Any]], PyTree[Any]],
        parameters: PyTree[Any],
        state: LeastSquaresState,
        /,
        *,
        termination: OptimizationTermination | None,
    ) -> tuple[PyTree[Any], LeastSquaresState, Array]:
        if not isinstance(state, LeastSquaresState):
            raise TypeError("state must be a LeastSquaresState.")
        model = _prepare_residual_model(residual_function, parameters)
        linear_problem = _linear_least_squares_problem(model)
        if state.linear_refresh_state is None:
            prepared, refresh_state = prepare_refresh_state(
                linear_problem,
                self.linear_policy,
            )
            setup_increment = 1
            bind_increment = 1
        else:
            prepared = None
            refresh_state = state.linear_refresh_state
            setup_increment = 0
            bind_increment = 0
        state_with_refresh = eqx.tree_at(
            lambda value: value.linear_refresh_state,
            state,
            refresh_state,
        )
        _, static_state = eqx.partition(state_with_refresh, eqx.is_array)
        initial_optimality = _initial_optimality(state, model.optimality_norm)
        finite_model = (
            jnp.isfinite(model.objective)
            & jnp.isfinite(model.optimality_norm)
            & _tree_allfinite(model.residual)
            & _tree_allfinite(model.gradient)
        )
        converged = _converged(
            termination,
            initial_optimality,
            model.optimality_norm,
        )

        def terminal_step(_):
            status = jnp.where(
                finite_model,
                int(OptimizationStatus.SUCCESS),
                int(OptimizationStatus.NONFINITE_EVALUATION),
            )
            metrics = IterativeStepMetrics(
                objective=model.objective,
                optimality_norm=model.optimality_norm,
                accepted=finite_model,
                status=status,
            )
            updated = LeastSquaresState(
                iteration=state.iteration + 1,
                initial_optimality_norm=initial_optimality,
                damping=state.damping,
                accepted_steps=state.accepted_steps,
                rejected_steps=(state.rejected_steps + (~finite_model).astype(jnp.int32)),
                residual_evaluations=state.residual_evaluations + 1,
                jvp_evaluations=state.jvp_evaluations,
                vjp_evaluations=state.vjp_evaluations + 1,
                linear_iterations=state.linear_iterations,
                linear_solves=state.linear_solves,
                setup_refreshes=state.setup_refreshes + setup_increment,
                numeric_refreshes=state.numeric_refreshes + bind_increment,
                linear_refresh_state=refresh_state,
                direction_fallbacks=state.direction_fallbacks,
                metrics=metrics,
            )
            dynamic_updated, _ = eqx.partition(updated, eqx.is_array)
            return parameters, dynamic_updated, model.objective

        def gauss_newton_step(_):
            if prepared is None:
                current_prepared, next_refresh_state = refresh_state.refresh(
                    linear_problem
                )
                refresh_increment = 1
            else:
                current_prepared = prepared
                next_refresh_state = refresh_state
                refresh_increment = 0
            linear_result = solve_linear(
                current_prepared,
                _tree_negative(model.residual),
            )
            proposed_direction = linear_result.value
            proposed_directional = _tree_inner(
                model.gradient,
                proposed_direction,
            )
            usable_direction = (
                _usable_inexact_linear_status(linear_result.status)
                & _tree_allfinite(proposed_direction)
                & jnp.isfinite(proposed_directional)
                & (proposed_directional < 0.0)
            )
            direction = _tree_where(
                usable_direction,
                proposed_direction,
                _tree_negative(model.gradient),
            )
            directional = _tree_inner(model.gradient, direction)

            def objective(candidate):
                residual = residual_function(candidate)
                return 0.5 * _tree_inner(residual, residual)

            search = armijo_backtracking(
                objective,
                parameters,
                model.objective,
                direction,
                directional,
                step=_tree_add_scaled,
                contains=_tree_allfinite,
                policy=self.line_search,
            )
            accepted = search.accepted
            step_norm = search.rate * _tree_norm(direction)
            status = jnp.where(
                accepted & _stagnated(termination, parameters, step_norm),
                int(OptimizationStatus.STAGNATION),
                jnp.where(
                    accepted,
                    int(OptimizationStatus.ITERATING),
                    jnp.where(
                        search.finite_candidate_seen,
                        int(OptimizationStatus.LINE_SEARCH_FAILED),
                        int(OptimizationStatus.NONFINITE_EVALUATION),
                    ),
                ),
            )
            linear_iterations = jnp.asarray(linear_result.diagnostics.iterations).reshape(
                ()
            )
            metrics = IterativeStepMetrics(
                objective=search.value,
                optimality_norm=model.optimality_norm,
                step_norm=step_norm,
                accepted_step_size=search.rate,
                globalization_evaluations=search.evaluations,
                accepted=accepted,
                linear_iterations=linear_iterations,
                linear_status=linear_result.status,
                direction_fallback=~usable_direction,
                status=status,
            )
            updated = LeastSquaresState(
                iteration=state.iteration + 1,
                initial_optimality_norm=initial_optimality,
                damping=state.damping,
                accepted_steps=state.accepted_steps + accepted.astype(jnp.int32),
                rejected_steps=(state.rejected_steps + (~accepted).astype(jnp.int32)),
                residual_evaluations=(
                    state.residual_evaluations + 1 + search.evaluations
                ),
                jvp_evaluations=(
                    state.jvp_evaluations + linear_result.diagnostics.matvec_count
                ),
                vjp_evaluations=(
                    state.vjp_evaluations
                    + 1
                    + linear_result.diagnostics.adjoint_matvec_count
                ),
                linear_iterations=state.linear_iterations + linear_iterations,
                linear_solves=state.linear_solves + 1,
                setup_refreshes=state.setup_refreshes + setup_increment,
                numeric_refreshes=(
                    state.numeric_refreshes + bind_increment + refresh_increment
                ),
                linear_refresh_state=next_refresh_state,
                direction_fallbacks=(
                    state.direction_fallbacks + (~usable_direction).astype(jnp.int32)
                ),
                metrics=metrics,
            )
            dynamic_updated, _ = eqx.partition(updated, eqx.is_array)
            return search.parameters, dynamic_updated, search.value

        parameters, dynamic_updated, objective = jax.lax.cond(
            (~finite_model) | converged,
            terminal_step,
            gauss_newton_step,
            None,
        )
        return (
            parameters,
            eqx.combine(dynamic_updated, static_state),
            objective,
        )

    def step_metrics(self, state: LeastSquaresState, /) -> IterativeStepMetrics:
        if not isinstance(state, LeastSquaresState):
            raise TypeError("state must be a LeastSquaresState.")
        return state.metrics

    def solve(
        self,
        problem: NonlinearLeastSquaresProblem,
        initial_parameters: PyTree[Any],
        /,
        *,
        termination: OptimizationTermination,
        args: Any,
    ) -> LeastSquaresResult:
        return _solve_least_squares(
            self,
            problem,
            initial_parameters,
            termination=termination,
            args=args,
        )


class LevenbergMarquardt(AbstractLeastSquaresMethod):
    """Matrix-free damped Gauss–Newton with ratio-based trust control."""

    linear_policy: LinearSolvePolicy
    initial_damping: float = eqx.field(static=True)
    minimum_damping: float = eqx.field(static=True)
    maximum_damping: float = eqx.field(static=True)
    damping_increase: float = eqx.field(static=True)
    damping_decrease: float = eqx.field(static=True)
    acceptance_ratio: float = eqx.field(static=True)
    decrease_ratio: float = eqx.field(static=True)
    increase_ratio: float = eqx.field(static=True)
    maximum_trials: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        linear_policy: LinearSolvePolicy | None = None,
        initial_damping: float = 1e-3,
        minimum_damping: float = 1e-12,
        maximum_damping: float = 1e12,
        damping_increase: float = 2.0,
        damping_decrease: float = 0.5,
        acceptance_ratio: float = 1e-4,
        decrease_ratio: float = 0.75,
        increase_ratio: float = 0.25,
        maximum_trials: int = 12,
    ):
        policy = _default_linear_policy() if linear_policy is None else linear_policy
        if not isinstance(policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be a LinearSolvePolicy or None.")
        damping_values = (
            float(initial_damping),
            float(minimum_damping),
            float(maximum_damping),
        )
        if any(not isfinite(value) or value <= 0.0 for value in damping_values):
            raise ValueError("Levenberg–Marquardt damping values must be positive.")
        if not damping_values[1] <= damping_values[0] <= damping_values[2]:
            raise ValueError("initial_damping must lie within the damping bounds.")
        increase = float(damping_increase)
        decrease = float(damping_decrease)
        if not isfinite(increase) or increase <= 1.0:
            raise ValueError("damping_increase must be finite and greater than one.")
        if not isfinite(decrease) or not 0.0 < decrease < 1.0:
            raise ValueError("damping_decrease must lie strictly between zero and one.")
        ratios = (
            float(acceptance_ratio),
            float(decrease_ratio),
            float(increase_ratio),
        )
        if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in ratios):
            raise ValueError("Trust-region ratios must lie in [0, 1].")
        if not ratios[0] <= ratios[2] < ratios[1]:
            raise ValueError(
                "Ratios must satisfy acceptance_ratio <= increase_ratio < decrease_ratio."
            )
        trials = int(maximum_trials)
        if trials < 1:
            raise ValueError("maximum_trials must be positive.")
        self.linear_policy = policy
        self.initial_damping, self.minimum_damping, self.maximum_damping = damping_values
        self.damping_increase = increase
        self.damping_decrease = decrease
        self.acceptance_ratio, self.decrease_ratio, self.increase_ratio = ratios
        self.maximum_trials = trials

    @property
    def method_id(self) -> str:
        return "levenberg-marquardt"

    @property
    def globalization_id(self) -> str:
        return "damped-trust-region"

    @property
    def capabilities(self) -> OptimizationCapabilities:
        return OptimizationCapabilities(
            scalar_objective=False,
            residual_objective=True,
            matrix_free=True,
            prepared_refresh=True,
            implicit_differentiation=True,
        )

    def init(self, parameters: PyTree[Any], /) -> LeastSquaresState:
        parameters = _validate_real_inexact_tree(parameters, name="parameters")
        PyTreeSpace(parameters)
        metric_nan = jnp.asarray(jnp.nan, dtype=_tree_norm(parameters).dtype)
        damping = jnp.asarray(self.initial_damping, dtype=metric_nan.dtype)
        return LeastSquaresState(
            initial_optimality_norm=metric_nan,
            damping=damping,
            metrics=IterativeStepMetrics(objective=metric_nan),
        )

    def prepare_state(
        self,
        residual_function: Callable[[PyTree[Any]], PyTree[Any]],
        parameters: PyTree[Any],
        /,
    ) -> LeastSquaresState:
        parameters = _validate_real_inexact_tree(parameters, name="parameters")
        model = _prepare_residual_model(residual_function, parameters)
        metric_nan = jnp.asarray(jnp.nan, dtype=model.objective.dtype)
        damping = jnp.asarray(self.initial_damping, dtype=model.objective.dtype)
        _, refresh_state = prepare_refresh_state(
            _linear_least_squares_problem(model, damping),
            self.linear_policy,
        )
        return LeastSquaresState(
            initial_optimality_norm=metric_nan,
            damping=damping,
            residual_evaluations=1,
            vjp_evaluations=1,
            setup_refreshes=1,
            numeric_refreshes=1,
            linear_refresh_state=refresh_state,
            metrics=IterativeStepMetrics(objective=metric_nan),
        )

    def step(
        self,
        residual_function: Callable[[PyTree[Any]], PyTree[Any]],
        parameters: PyTree[Any],
        state: LeastSquaresState,
        /,
        *,
        termination: OptimizationTermination | None,
    ) -> tuple[PyTree[Any], LeastSquaresState, Array]:
        if not isinstance(state, LeastSquaresState):
            raise TypeError("state must be a LeastSquaresState.")
        model = _prepare_residual_model(residual_function, parameters)
        if state.linear_refresh_state is None:
            _, refresh_state = prepare_refresh_state(
                _linear_least_squares_problem(model, state.damping),
                self.linear_policy,
            )
            setup_increment = 1
            bind_increment = 1
        else:
            refresh_state = state.linear_refresh_state
            setup_increment = 0
            bind_increment = 0
        state_with_refresh = eqx.tree_at(
            lambda value: value.linear_refresh_state,
            state,
            refresh_state,
        )
        _, static_state = eqx.partition(state_with_refresh, eqx.is_array)
        initial_optimality = _initial_optimality(state, model.optimality_norm)
        finite_model = (
            jnp.isfinite(model.objective)
            & jnp.isfinite(model.optimality_norm)
            & _tree_allfinite(model.residual)
            & _tree_allfinite(model.gradient)
        )
        converged = _converged(
            termination,
            initial_optimality,
            model.optimality_norm,
        )

        def terminal_step(_):
            status = jnp.where(
                finite_model,
                int(OptimizationStatus.SUCCESS),
                int(OptimizationStatus.NONFINITE_EVALUATION),
            )
            metrics = IterativeStepMetrics(
                objective=model.objective,
                optimality_norm=model.optimality_norm,
                damping=state.damping,
                accepted=finite_model,
                status=status,
            )
            updated = LeastSquaresState(
                iteration=state.iteration + 1,
                initial_optimality_norm=initial_optimality,
                damping=state.damping,
                accepted_steps=state.accepted_steps,
                rejected_steps=(state.rejected_steps + (~finite_model).astype(jnp.int32)),
                residual_evaluations=state.residual_evaluations + 1,
                jvp_evaluations=state.jvp_evaluations,
                vjp_evaluations=state.vjp_evaluations + 1,
                linear_iterations=state.linear_iterations,
                linear_solves=state.linear_solves,
                setup_refreshes=state.setup_refreshes + setup_increment,
                numeric_refreshes=state.numeric_refreshes + bind_increment,
                linear_refresh_state=refresh_state,
                direction_fallbacks=state.direction_fallbacks,
                metrics=metrics,
            )
            dynamic_updated, _ = eqx.partition(updated, eqx.is_array)
            return parameters, dynamic_updated, model.objective

        def lm_step(_):
            dynamic_refresh_state, static_refresh_state = eqx.partition(
                refresh_state,
                eqx.is_array,
            )

            def trial_condition(carry):
                trial, _, accepted, *_ = carry
                return (trial < self.maximum_trials) & (~accepted)

            def trial_body(carry):
                (
                    trial,
                    damping,
                    _,
                    candidate_parameters,
                    candidate_objective,
                    accepted_step_norm,
                    ratio,
                    residual_evaluations,
                    jvp_evaluations,
                    vjp_evaluations,
                    total_linear_iterations,
                    _,
                    usable_linear_step_seen,
                    finite_trial_seen,
                    dynamic_refresh_state_for_trial,
                ) = carry
                refresh_state_for_trial = eqx.combine(
                    dynamic_refresh_state_for_trial,
                    static_refresh_state,
                )
                current_prepared, current_refresh_state = refresh_state_for_trial.refresh(
                    _linear_least_squares_problem(model, damping)
                )
                linear_result = solve_linear(
                    current_prepared,
                    _tree_negative(model.residual),
                )
                direction = linear_result.value
                directional = _tree_inner(model.gradient, direction)
                usable = (
                    _usable_inexact_linear_status(linear_result.status)
                    & _tree_allfinite(direction)
                    & jnp.isfinite(directional)
                    & (directional < 0.0)
                )

                def evaluate_trial(_):
                    linearized_residual = jax.tree.map(
                        lambda residual, change: residual + change,
                        model.residual,
                        model.jacobian.mv(direction),
                    )
                    predicted = model.objective - 0.5 * _tree_inner(
                        linearized_residual,
                        linearized_residual,
                    )
                    proposed = _tree_add_scaled(parameters, direction, 1.0)
                    proposed_residual = residual_function(proposed)
                    proposed_objective = 0.5 * _tree_inner(
                        proposed_residual,
                        proposed_residual,
                    )
                    actual = model.objective - proposed_objective
                    finite_trial = (
                        _tree_allfinite(proposed)
                        & _tree_allfinite(proposed_residual)
                        & jnp.isfinite(proposed_objective)
                        & jnp.isfinite(predicted)
                    )
                    trial_ratio = jnp.where(
                        finite_trial & (predicted > 0.0),
                        actual / predicted,
                        -jnp.inf,
                    )
                    next_damping = jnp.where(
                        trial_ratio > self.decrease_ratio,
                        jnp.maximum(
                            self.minimum_damping,
                            damping * self.damping_decrease,
                        ),
                        jnp.where(
                            trial_ratio < self.increase_ratio,
                            jnp.minimum(
                                self.maximum_damping,
                                damping * self.damping_increase,
                            ),
                            damping,
                        ),
                    )
                    trial_accepted = finite_trial & (trial_ratio >= self.acceptance_ratio)
                    return (
                        next_damping,
                        trial_accepted,
                        _tree_where(
                            trial_accepted,
                            proposed,
                            candidate_parameters,
                        ),
                        jnp.where(
                            trial_accepted,
                            proposed_objective,
                            candidate_objective,
                        ),
                        jnp.where(
                            trial_accepted,
                            _tree_norm(direction),
                            accepted_step_norm,
                        ),
                        trial_ratio,
                        jnp.asarray(1, dtype=jnp.int32),
                        jnp.asarray(1, dtype=jnp.int32),
                        finite_trial,
                    )

                def reject_linear_step(_):
                    return (
                        jnp.minimum(
                            self.maximum_damping,
                            damping * self.damping_increase,
                        ),
                        jnp.asarray(False),
                        candidate_parameters,
                        candidate_objective,
                        accepted_step_norm,
                        ratio,
                        jnp.asarray(0, dtype=jnp.int32),
                        jnp.asarray(0, dtype=jnp.int32),
                        jnp.asarray(False),
                    )

                (
                    next_damping,
                    accepted,
                    next_parameters,
                    next_objective,
                    next_step_norm,
                    next_ratio,
                    residual_increment,
                    jvp_increment,
                    finite_trial,
                ) = jax.lax.cond(
                    usable,
                    evaluate_trial,
                    reject_linear_step,
                    None,
                )
                linear_iterations = jnp.asarray(
                    linear_result.diagnostics.iterations,
                    dtype=jnp.int32,
                ).reshape(())
                dynamic_current_refresh_state, _ = eqx.partition(
                    current_refresh_state,
                    eqx.is_array,
                )
                return (
                    trial + 1,
                    next_damping,
                    accepted,
                    next_parameters,
                    next_objective,
                    next_step_norm,
                    next_ratio,
                    residual_evaluations + residual_increment,
                    jvp_evaluations
                    + jnp.asarray(
                        linear_result.diagnostics.matvec_count,
                        dtype=jnp.int32,
                    )
                    + jvp_increment,
                    vjp_evaluations
                    + jnp.asarray(
                        linear_result.diagnostics.adjoint_matvec_count,
                        dtype=jnp.int32,
                    ),
                    total_linear_iterations + linear_iterations,
                    jnp.asarray(linear_result.status, dtype=jnp.int32),
                    usable_linear_step_seen | usable,
                    finite_trial_seen | finite_trial,
                    dynamic_current_refresh_state,
                )

            initial_trial = (
                jnp.asarray(0, dtype=jnp.int32),
                state.damping,
                jnp.asarray(False),
                parameters,
                model.objective,
                jnp.zeros_like(model.optimality_norm),
                jnp.full_like(model.objective, jnp.nan),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(-1, dtype=jnp.int32),
                jnp.asarray(False),
                jnp.asarray(False),
                dynamic_refresh_state,
            )
            (
                trials,
                damping,
                accepted,
                candidate_parameters,
                candidate_objective,
                accepted_step_norm,
                ratio,
                trial_residual_evaluations,
                trial_jvp_evaluations,
                trial_vjp_evaluations,
                trial_linear_iterations,
                last_linear_status,
                usable_linear_step_seen,
                finite_trial_seen,
                final_refresh_state,
            ) = jax.lax.while_loop(
                trial_condition,
                trial_body,
                initial_trial,
            )
            final_refresh_state = eqx.combine(
                final_refresh_state,
                static_refresh_state,
            )
            status = jnp.where(
                accepted
                & _stagnated(
                    termination,
                    parameters,
                    accepted_step_norm,
                ),
                int(OptimizationStatus.STAGNATION),
                jnp.where(
                    accepted,
                    int(OptimizationStatus.ITERATING),
                    jnp.where(
                        usable_linear_step_seen,
                        jnp.where(
                            finite_trial_seen,
                            int(OptimizationStatus.TRUST_REGION_FAILED),
                            int(OptimizationStatus.NONFINITE_EVALUATION),
                        ),
                        int(OptimizationStatus.LINEAR_SOLVE_FAILED),
                    ),
                ),
            )
            metrics = IterativeStepMetrics(
                objective=candidate_objective,
                optimality_norm=model.optimality_norm,
                step_norm=accepted_step_norm,
                accepted_step_size=accepted.astype(candidate_objective.dtype),
                globalization_evaluations=trial_residual_evaluations,
                accepted=accepted,
                linear_iterations=trial_linear_iterations,
                linear_status=last_linear_status,
                damping=damping,
                reduction_ratio=ratio,
                status=status,
            )
            updated = LeastSquaresState(
                iteration=state.iteration + 1,
                initial_optimality_norm=initial_optimality,
                damping=damping,
                accepted_steps=state.accepted_steps + accepted.astype(jnp.int32),
                rejected_steps=(state.rejected_steps + (~accepted).astype(jnp.int32)),
                residual_evaluations=(
                    state.residual_evaluations + 1 + trial_residual_evaluations
                ),
                jvp_evaluations=(state.jvp_evaluations + trial_jvp_evaluations),
                vjp_evaluations=(state.vjp_evaluations + 1 + trial_vjp_evaluations),
                linear_iterations=(state.linear_iterations + trial_linear_iterations),
                linear_solves=state.linear_solves + trials,
                setup_refreshes=state.setup_refreshes + setup_increment,
                numeric_refreshes=(state.numeric_refreshes + bind_increment + trials),
                linear_refresh_state=final_refresh_state,
                direction_fallbacks=state.direction_fallbacks,
                metrics=metrics,
            )
            dynamic_updated, _ = eqx.partition(updated, eqx.is_array)
            return candidate_parameters, dynamic_updated, candidate_objective

        parameters, dynamic_updated, objective = jax.lax.cond(
            (~finite_model) | converged,
            terminal_step,
            lm_step,
            None,
        )
        return (
            parameters,
            eqx.combine(dynamic_updated, static_state),
            objective,
        )

    def step_metrics(self, state: LeastSquaresState, /) -> IterativeStepMetrics:
        if not isinstance(state, LeastSquaresState):
            raise TypeError("state must be a LeastSquaresState.")
        return state.metrics

    def solve(
        self,
        problem: NonlinearLeastSquaresProblem,
        initial_parameters: PyTree[Any],
        /,
        *,
        termination: OptimizationTermination,
        args: Any,
    ) -> LeastSquaresResult:
        return _solve_least_squares(
            self,
            problem,
            initial_parameters,
            termination=termination,
            args=args,
        )


class _LeastSquaresRun(StrictModule):
    parameters: PyTree[Array]
    state: LeastSquaresState
    status: Array

    def __init__(
        self,
        parameters: PyTree[Any],
        state: LeastSquaresState,
        status: Any,
        /,
    ):
        self.parameters = parameters
        self.state = state
        self.status = jnp.asarray(status, dtype=jnp.int32)


def _run_least_squares_iterations(
    method: AbstractLeastSquaresMethod,
    residual_function,
    initial_parameters: PyTree[Any],
    termination: OptimizationTermination,
    /,
) -> _LeastSquaresRun:
    state = method.prepare_state(residual_function, initial_parameters)
    state, static_state = eqx.partition(state, eqx.is_array)
    initial_status = jnp.where(
        _tree_allfinite(initial_parameters),
        int(OptimizationStatus.ITERATING),
        int(OptimizationStatus.NONFINITE_INPUT),
    ).astype(jnp.int32)

    def condition(carry):
        _, current_state, status = carry
        within_evaluations = (
            jnp.asarray(True)
            if termination.maximum_evaluations is None
            else current_state.residual_evaluations < termination.maximum_evaluations
        )
        return (
            (status == int(OptimizationStatus.ITERATING))
            & (current_state.iteration < termination.maximum_steps)
            & within_evaluations
        )

    def body(carry):
        current_parameters, dynamic_state, _ = carry
        current_state = eqx.combine(dynamic_state, static_state)
        next_parameters, next_state, _ = method.step(
            residual_function,
            current_parameters,
            current_state,
            termination=termination,
        )
        next_status = method.step_metrics(next_state).status
        if termination.maximum_evaluations is not None:
            exhausted = (next_status == int(OptimizationStatus.ITERATING)) & (
                next_state.residual_evaluations >= termination.maximum_evaluations
            )
            next_status = jnp.where(
                exhausted,
                int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED),
                next_status,
            )
        dynamic_next_state, _ = eqx.partition(next_state, eqx.is_array)
        return next_parameters, dynamic_next_state, next_status

    parameters, state, status = jax.lax.while_loop(
        condition,
        body,
        (initial_parameters, state, initial_status),
    )
    state = eqx.combine(state, static_state)
    if termination.maximum_evaluations is not None:
        status = jnp.where(
            (status == int(OptimizationStatus.ITERATING))
            & (state.residual_evaluations >= termination.maximum_evaluations),
            int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED),
            status,
        )
    status = jnp.where(
        status == int(OptimizationStatus.ITERATING),
        int(OptimizationStatus.MAXIMUM_STEPS_REACHED),
        status,
    )
    return _LeastSquaresRun(parameters, state, status)


def _validate_least_squares_inputs(
    problem: NonlinearLeastSquaresProblem,
    initial_parameters: PyTree[Any],
    termination: OptimizationTermination,
    /,
) -> PyTree[Array]:
    if not isinstance(problem, NonlinearLeastSquaresProblem):
        raise TypeError("problem must be a NonlinearLeastSquaresProblem.")
    if not isinstance(termination, OptimizationTermination):
        raise TypeError("termination must be an OptimizationTermination.")
    return _validate_real_inexact_tree(
        initial_parameters,
        name="initial_parameters",
    )


def _package_least_squares_result(
    method: AbstractLeastSquaresMethod,
    problem: NonlinearLeastSquaresProblem,
    run: _LeastSquaresRun,
    residual_function: Callable[[PyTree[Any]], PyTree[Any]],
    termination: OptimizationTermination,
    args: Any,
    /,
) -> LeastSquaresResult:
    parameters, state, status = run.parameters, run.state, run.status
    final_model = _prepare_residual_model(residual_function, parameters)
    residual_evaluations = state.residual_evaluations + 1
    auxiliary = None
    if problem.has_aux:
        _, auxiliary = problem.value(parameters, args)
        residual_evaluations = residual_evaluations + 1
    finite_final = (
        jnp.isfinite(final_model.objective)
        & jnp.isfinite(final_model.optimality_norm)
        & _tree_allfinite(final_model.residual)
        & _tree_allfinite(final_model.gradient)
    )
    initial_nonfinite = status == int(OptimizationStatus.NONFINITE_INPUT)
    status = jnp.where(
        initial_nonfinite,
        status,
        jnp.where(
            ~finite_final,
            int(OptimizationStatus.NONFINITE_EVALUATION),
            jnp.where(
                final_model.optimality_norm
                <= termination.optimality_threshold(state.initial_optimality_norm),
                int(OptimizationStatus.SUCCESS),
                status,
            ),
        ),
    )
    final_metrics = method.step_metrics(state)
    diagnostics = OptimizationDiagnostics(
        iterations=state.iteration,
        accepted_steps=state.accepted_steps,
        rejected_steps=state.rejected_steps,
        objective_evaluations=residual_evaluations,
        residual_evaluations=residual_evaluations,
        jvp_evaluations=state.jvp_evaluations,
        vjp_evaluations=state.vjp_evaluations + 1,
        linear_iterations=state.linear_iterations,
        linear_solves=state.linear_solves,
        setup_refreshes=state.setup_refreshes,
        numeric_refreshes=state.numeric_refreshes,
        globalization_evaluations=(state.residual_evaluations - state.iteration - 1),
        initial_optimality_norm=state.initial_optimality_norm,
        final_optimality_norm=final_model.optimality_norm,
        final_step_norm=final_metrics.step_norm,
        accepted_step_size=final_metrics.accepted_step_size,
        damping=state.damping,
        reduction_ratio=final_metrics.reduction_ratio,
        direction_fallbacks=state.direction_fallbacks,
    )
    return LeastSquaresResult(
        parameters,
        final_model.residual,
        final_model.objective,
        auxiliary,
        status,
        diagnostics,
        OptimizationProvenance(
            problem_id=problem.problem_id,
            method=method.method_id,
            backend="phydrax-native",
            globalization=method.globalization_id,
            matrix_free=True,
            implicit_differentiation=method.capabilities.implicit_differentiation,
        ),
    )


def _solve_least_squares(
    method: AbstractLeastSquaresMethod,
    problem: NonlinearLeastSquaresProblem,
    initial_parameters: PyTree[Any],
    /,
    *,
    termination: OptimizationTermination,
    args: Any,
) -> LeastSquaresResult:
    parameters = _validate_least_squares_inputs(
        problem,
        initial_parameters,
        termination,
    )

    def residual_function(candidate):
        residual, _ = problem.value(candidate, args)
        return residual

    run = _run_least_squares_iterations(
        method,
        residual_function,
        parameters,
        termination,
    )
    return _package_least_squares_result(
        method,
        problem,
        run,
        residual_function,
        termination,
        args,
    )


def least_squares(
    problem_or_residual: NonlinearLeastSquaresProblem | Callable[[PyTree[Any], Any], Any],
    initial_parameters: PyTree[Any],
    /,
    *,
    method: AbstractLeastSquaresMethod | None = None,
    termination: OptimizationTermination | None = None,
    args: Any = None,
    has_aux: bool = False,
) -> LeastSquaresResult:
    """Solve one nonlinear least-squares problem with explicit method semantics."""

    problem = (
        problem_or_residual
        if isinstance(problem_or_residual, NonlinearLeastSquaresProblem)
        else NonlinearLeastSquaresProblem(problem_or_residual, has_aux=has_aux)
    )
    method_ = GaussNewton() if method is None else method
    termination_ = OptimizationTermination() if termination is None else termination
    if not isinstance(method_, AbstractLeastSquaresMethod):
        raise TypeError("method must be an AbstractLeastSquaresMethod or None.")
    return method_.solve(
        problem,
        initial_parameters,
        termination=termination_,
        args=args,
    )


__all__ = [
    "GaussNewton",
    "LeastSquaresState",
    "LevenbergMarquardt",
    "least_squares",
]
