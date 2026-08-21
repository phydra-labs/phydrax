#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from .._linear_refresh import prepare_refresh_state
from ..linalg import (
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    MINRES,
    OperatorProperties,
    PyTreeSpace,
    solve as solve_linear,
    TolerancePolicy,
)
from ._iterative._base import AbstractScalarIterativeMethod
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
    MinimizationProblem,
    MinimizationResult,
    OptimizationCapabilities,
    OptimizationStatus,
    OptimizationTermination,
)
from ._scalar import ScalarIterativeState, solve_scalar_iterative


def _default_newton_linear_policy() -> LinearSolvePolicy:
    return LinearSolvePolicy(
        MINRES(),
        tolerance=TolerancePolicy(relative=1e-3, absolute=1e-10),
    )


def _usable_newton_linear_status(status: Any, /):
    status_ = jnp.asarray(status, dtype=jnp.int32)
    return (
        (status_ == int(LinearSolveStatus.SUCCESS))
        | (status_ == int(LinearSolveStatus.MAXIMUM_STEPS_REACHED))
        | (status_ == int(LinearSolveStatus.STAGNATION))
        | (status_ == int(LinearSolveStatus.CONDITION_LIMIT_REACHED))
    )


def _hessian_system(parameters: PyTree[Any], action, /) -> LinearSystem:
    space = PyTreeSpace(parameters)
    hessian = FunctionLinearOperator(
        action,
        source=space,
        target=space,
        transpose_action=action,
        properties=OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "asserted"},
        ),
        operator_id="objective-hessian",
    )
    return LinearSystem(hessian)


class NewtonKrylov(AbstractScalarIterativeMethod):
    """Matrix-free inexact Newton method with Armijo-globalized fallback."""

    linear_policy: LinearSolvePolicy
    line_search: ArmijoLineSearch
    minimum_forcing: float = eqx.field(static=True)
    maximum_forcing: float = eqx.field(static=True)
    forcing_power: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        linear_policy: LinearSolvePolicy | None = None,
        line_search: ArmijoLineSearch | None = None,
        minimum_forcing: float = 1e-6,
        maximum_forcing: float = 0.5,
        forcing_power: float = 0.5,
    ):
        policy = (
            _default_newton_linear_policy() if linear_policy is None else linear_policy
        )
        search = ArmijoLineSearch() if line_search is None else line_search
        if not isinstance(policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be a LinearSolvePolicy or None.")
        if not isinstance(search, ArmijoLineSearch):
            raise TypeError("line_search must be an ArmijoLineSearch or None.")
        forcing = (
            float(minimum_forcing),
            float(maximum_forcing),
            float(forcing_power),
        )
        if any(not isfinite(value) or value <= 0.0 for value in forcing):
            raise ValueError("Forcing parameters must be positive and finite.")
        if forcing[0] > forcing[1] or forcing[1] >= 1.0:
            raise ValueError("Forcing terms must satisfy minimum <= maximum < one.")
        self.linear_policy = policy
        self.line_search = search
        self.minimum_forcing, self.maximum_forcing, self.forcing_power = forcing

    @property
    def method_id(self) -> str:
        return "newton-krylov"

    @property
    def globalization_id(self) -> str:
        return "armijo"

    @property
    def capabilities(self) -> OptimizationCapabilities:
        return OptimizationCapabilities(
            scalar_objective=True,
            residual_objective=False,
            matrix_free=True,
            prepared_refresh=True,
            implicit_differentiation=True,
        )

    def init(self, parameters: PyTree[Any], /) -> ScalarIterativeState:
        parameters = _validate_real_inexact_tree(parameters, name="parameters")
        _, refresh_state = prepare_refresh_state(
            _hessian_system(parameters, lambda vector: vector),
            self.linear_policy,
        )
        metric_nan = jnp.asarray(jnp.nan, dtype=_tree_norm(parameters).dtype)
        return ScalarIterativeState(
            initial_optimality_norm=metric_nan,
            setup_refreshes=1,
            numeric_refreshes=1,
            linear_refresh_state=refresh_state,
            metrics=IterativeStepMetrics(objective=metric_nan),
        )

    def prepare_state(
        self,
        value_function,
        parameters: PyTree[Any],
        /,
    ) -> ScalarIterativeState:
        if not callable(value_function):
            raise TypeError("value_function must be callable.")
        return self.init(parameters)

    def step(
        self,
        value_function,
        parameters: PyTree[Any],
        state: ScalarIterativeState,
        /,
        *,
        termination: OptimizationTermination | None,
    ) -> tuple[PyTree[Any], ScalarIterativeState, Any]:
        if not callable(value_function):
            raise TypeError("value_function must be callable.")
        if not isinstance(state, ScalarIterativeState):
            raise TypeError("state must be a ScalarIterativeState.")
        if state.linear_refresh_state is None:
            raise ValueError("NewtonKrylov state is missing linear refresh state.")
        _, static_state = eqx.partition(state, eqx.is_array)

        value_and_gradient = jax.value_and_grad(value_function)
        (value, gradient), linearized = jax.linearize(
            value_and_gradient,
            parameters,
        )
        optimality = _tree_norm(gradient)
        initial_optimality = jnp.where(
            state.iteration == 0,
            optimality,
            state.initial_optimality_norm,
        )
        finite = (
            jnp.isfinite(value)
            & jnp.isfinite(optimality)
            & _tree_allfinite(parameters)
            & _tree_allfinite(gradient)
        )
        converged = (
            jnp.asarray(False)
            if termination is None
            else optimality <= termination.optimality_threshold(initial_optimality)
        )

        def terminal_step(_):
            status = jnp.where(
                finite,
                int(OptimizationStatus.SUCCESS),
                int(OptimizationStatus.NONFINITE_EVALUATION),
            )
            metrics = IterativeStepMetrics(
                objective=value,
                optimality_norm=optimality,
                accepted=finite,
                status=status,
            )
            updated = ScalarIterativeState(
                iteration=state.iteration + 1,
                initial_optimality_norm=initial_optimality,
                accepted_steps=state.accepted_steps,
                rejected_steps=state.rejected_steps + (~finite).astype(jnp.int32),
                objective_evaluations=state.objective_evaluations + 1,
                gradient_evaluations=state.gradient_evaluations + 1,
                hvp_evaluations=state.hvp_evaluations,
                linear_solves=state.linear_solves,
                linear_iterations=state.linear_iterations,
                setup_refreshes=state.setup_refreshes,
                numeric_refreshes=state.numeric_refreshes,
                linear_refresh_state=state.linear_refresh_state,
                direction_fallbacks=state.direction_fallbacks,
                metrics=metrics,
            )
            dynamic_updated, _ = eqx.partition(updated, eqx.is_array)
            return parameters, dynamic_updated, value

        def newton_step(_):
            def hessian_action(vector):
                _, hessian_vector = linearized(vector)
                return hessian_vector

            system = _hessian_system(parameters, hessian_action)
            prepared, refresh_state = state.linear_refresh_state.refresh(system)
            relative_optimality = optimality / jnp.maximum(initial_optimality, 1e-30)
            forcing = jnp.clip(
                relative_optimality**self.forcing_power,
                self.minimum_forcing,
                self.maximum_forcing,
            )
            linear_result = solve_linear(
                prepared,
                _tree_negative(gradient),
            )
            proposed_direction = linear_result.value
            proposed_directional = _tree_inner(gradient, proposed_direction)
            usable = (
                _usable_newton_linear_status(linear_result.status)
                & _tree_allfinite(proposed_direction)
                & jnp.isfinite(proposed_directional)
                & (proposed_directional < 0.0)
            )
            direction = _tree_where(
                usable,
                proposed_direction,
                _tree_negative(gradient),
            )
            directional = _tree_inner(gradient, direction)
            search = armijo_backtracking(
                value_function,
                parameters,
                value,
                direction,
                directional,
                step=_tree_add_scaled,
                contains=_tree_allfinite,
                policy=self.line_search,
            )
            accepted = search.accepted
            step_norm = search.rate * _tree_norm(direction)
            stagnated = (
                jnp.asarray(False)
                if termination is None
                else accepted
                & (step_norm <= termination.step_threshold(_tree_norm(parameters)))
            )
            status = jnp.where(
                stagnated,
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
                optimality_norm=optimality,
                step_norm=step_norm,
                accepted_step_size=search.rate,
                globalization_evaluations=search.evaluations,
                accepted=accepted,
                linear_iterations=linear_iterations,
                linear_status=linear_result.status,
                forcing=forcing,
                direction_fallback=~usable,
                status=status,
            )
            updated = ScalarIterativeState(
                iteration=state.iteration + 1,
                initial_optimality_norm=initial_optimality,
                accepted_steps=state.accepted_steps + accepted.astype(jnp.int32),
                rejected_steps=state.rejected_steps + (~accepted).astype(jnp.int32),
                objective_evaluations=(
                    state.objective_evaluations + 1 + search.evaluations
                ),
                gradient_evaluations=state.gradient_evaluations + 1,
                hvp_evaluations=(
                    state.hvp_evaluations + linear_result.diagnostics.matvec_count
                ),
                linear_solves=state.linear_solves + 1,
                linear_iterations=state.linear_iterations + linear_iterations,
                setup_refreshes=state.setup_refreshes,
                numeric_refreshes=state.numeric_refreshes + 1,
                linear_refresh_state=refresh_state,
                direction_fallbacks=(
                    state.direction_fallbacks + (~usable).astype(jnp.int32)
                ),
                metrics=metrics,
            )
            dynamic_updated, _ = eqx.partition(updated, eqx.is_array)
            return search.parameters, dynamic_updated, search.value

        parameters, dynamic_updated, objective = jax.lax.cond(
            (~finite) | converged,
            terminal_step,
            newton_step,
            None,
        )
        return (
            parameters,
            eqx.combine(dynamic_updated, static_state),
            objective,
        )

    def step_metrics(self, state: ScalarIterativeState, /) -> IterativeStepMetrics:
        if not isinstance(state, ScalarIterativeState):
            raise TypeError("state must be a ScalarIterativeState.")
        return state.metrics

    def solve(
        self,
        problem: MinimizationProblem,
        initial_parameters: PyTree[Any],
        /,
        *,
        termination: OptimizationTermination,
        args: Any,
    ) -> MinimizationResult:
        return solve_scalar_iterative(
            self,
            problem,
            initial_parameters,
            termination=termination,
            args=args,
        )


__all__ = ["NewtonKrylov"]
