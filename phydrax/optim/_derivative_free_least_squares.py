#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

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
from ._least_squares import _run_least_squares_iterations, LeastSquaresState


class _FiniteDifferenceResidualModel(eqx.Module):
    residual: PyTree[Array]
    objective: Array
    gradient: PyTree[Array]
    jacobian: Array
    residual_evaluations: Array


def _central_difference_model(
    residual_function,
    parameters: PyTree[Any],
    /,
    *,
    relative_step: float,
    absolute_step: float,
) -> _FiniteDifferenceResidualModel:
    flat_parameters, unravel = ravel_pytree(parameters)
    residual = _validate_real_inexact_tree(residual_function(parameters), name="residual")
    flat_residual, _ = ravel_pytree(residual)
    scales = absolute_step + relative_step * jnp.maximum(1.0, jnp.abs(flat_parameters))
    perturbations = jnp.diag(scales)

    def evaluate(candidate):
        candidate_residual = _validate_real_inexact_tree(
            residual_function(unravel(candidate)), name="residual"
        )
        flat_candidate, _ = ravel_pytree(candidate_residual)
        if flat_candidate.shape != flat_residual.shape:
            raise ValueError(
                "The residual PyTree shape must remain constant during finite differences."
            )
        return flat_candidate

    plus = jax.vmap(evaluate)(flat_parameters + perturbations)
    minus = jax.vmap(evaluate)(flat_parameters - perturbations)
    jacobian = ((plus - minus) / (2.0 * scales[:, None])).T
    flat_gradient = jacobian.T @ flat_residual
    return _FiniteDifferenceResidualModel(
        residual=residual,
        objective=0.5 * jnp.vdot(flat_residual, flat_residual).real,
        gradient=unravel(flat_gradient),
        jacobian=jacobian,
        residual_evaluations=jnp.asarray(1 + 2 * flat_parameters.size, dtype=jnp.int32),
    )


class FiniteDifferenceGaussNewton(AbstractLeastSquaresMethod):
    """Central-difference Gauss-Newton fallback requiring no residual derivatives."""

    line_search: ArmijoLineSearch
    relative_step: float = eqx.field(static=True)
    absolute_step: float = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    max_dense_dimension: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        line_search: ArmijoLineSearch | None = None,
        relative_step: float = 1e-4,
        absolute_step: float = 1e-7,
        regularization: float = 1e-8,
        max_dense_dimension: int = 512,
    ):
        search = ArmijoLineSearch() if line_search is None else line_search
        values = tuple(
            float(value) for value in (relative_step, absolute_step, regularization)
        )
        dimension = int(max_dense_dimension)
        if not isinstance(search, ArmijoLineSearch):
            raise TypeError("line_search must be an ArmijoLineSearch or None.")
        if any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError(
                "Finite-difference controls must be finite and non-negative."
            )
        if values[0] == 0.0 and values[1] == 0.0:
            raise ValueError(
                "At least one finite-difference step control must be positive."
            )
        if dimension < 1:
            raise ValueError("max_dense_dimension must be positive.")
        self.line_search = search
        self.relative_step, self.absolute_step, self.regularization = values
        self.max_dense_dimension = dimension

    @property
    def method_id(self) -> str:
        return "finite-difference-gauss-newton/central"

    @property
    def globalization_id(self) -> str:
        return "armijo"

    @property
    def capabilities(self) -> OptimizationCapabilities:
        return OptimizationCapabilities(
            scalar_objective=False,
            residual_objective=True,
            matrix_free=False,
            prepared_refresh=False,
            implicit_differentiation=False,
        )

    def _model(self, residual_function, parameters, /):
        flat, _ = ravel_pytree(parameters)
        if int(flat.size) > self.max_dense_dimension:
            raise ValueError(
                f"FiniteDifferenceGaussNewton has {flat.size} variables, exceeding "
                f"max_dense_dimension={self.max_dense_dimension}."
            )
        return _central_difference_model(
            residual_function,
            parameters,
            relative_step=self.relative_step,
            absolute_step=self.absolute_step,
        )

    def init(self, parameters: PyTree[Any], /) -> LeastSquaresState:
        parameters = _validate_real_inexact_tree(parameters, name="parameters")
        flat, _ = ravel_pytree(parameters)
        if int(flat.size) > self.max_dense_dimension:
            raise ValueError(
                f"FiniteDifferenceGaussNewton has {flat.size} variables, exceeding "
                f"max_dense_dimension={self.max_dense_dimension}."
            )
        metric_nan = jnp.asarray(jnp.nan, dtype=flat.dtype)
        return LeastSquaresState(
            initial_optimality_norm=metric_nan,
            metrics=IterativeStepMetrics(objective=metric_nan),
        )

    def prepare_state(
        self,
        residual_function,
        parameters: PyTree[Any],
        /,
    ) -> LeastSquaresState:
        if not callable(residual_function):
            raise TypeError("residual_function must be callable.")
        parameters = _validate_real_inexact_tree(parameters, name="parameters")
        model = self._model(residual_function, parameters)
        optimality = _tree_norm(model.gradient)
        return LeastSquaresState(
            initial_optimality_norm=optimality,
            residual_evaluations=model.residual_evaluations,
            metrics=IterativeStepMetrics(
                objective=model.objective,
                optimality_norm=optimality,
            ),
        )

    def step(
        self,
        residual_function,
        parameters: PyTree[Any],
        state: LeastSquaresState,
        /,
        *,
        termination: OptimizationTermination | None,
    ) -> tuple[PyTree[Any], LeastSquaresState, Array]:
        if not callable(residual_function):
            raise TypeError("residual_function must be callable.")
        if not isinstance(state, LeastSquaresState):
            raise TypeError("state must be a LeastSquaresState.")
        _, static_state = eqx.partition(state, eqx.is_array)
        flat_parameters, unravel = ravel_pytree(parameters)
        model_cost = jnp.asarray(1 + 2 * flat_parameters.size, dtype=jnp.int32)
        budget_available = (
            jnp.asarray(True)
            if termination is None or termination.maximum_evaluations is None
            else state.residual_evaluations + 2 * model_cost
            <= termination.maximum_evaluations
        )

        def no_budget(_):
            updated = LeastSquaresState(
                iteration=state.iteration,
                initial_optimality_norm=state.initial_optimality_norm,
                damping=state.damping,
                accepted_steps=state.accepted_steps,
                rejected_steps=state.rejected_steps,
                residual_evaluations=state.residual_evaluations,
                jvp_evaluations=state.jvp_evaluations,
                vjp_evaluations=state.vjp_evaluations,
                linear_iterations=state.linear_iterations,
                linear_solves=state.linear_solves,
                setup_refreshes=state.setup_refreshes,
                numeric_refreshes=state.numeric_refreshes,
                linear_refresh_state=state.linear_refresh_state,
                direction_fallbacks=state.direction_fallbacks,
                metrics=IterativeStepMetrics(
                    objective=state.metrics.objective,
                    optimality_norm=state.metrics.optimality_norm,
                    status=OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED,
                ),
            )
            dynamic, _ = eqx.partition(updated, eqx.is_array)
            return parameters, dynamic, state.metrics.objective

        def evaluate_step(_):
            model = self._model(residual_function, parameters)
            optimality = _tree_norm(model.gradient)
            initial_optimality = jnp.where(
                state.iteration == 0,
                optimality,
                state.initial_optimality_norm,
            )
            finite = (
                jnp.isfinite(model.objective)
                & jnp.isfinite(optimality)
                & _tree_allfinite(model.residual)
                & _tree_allfinite(model.gradient)
                & jnp.all(jnp.isfinite(model.jacobian))
            )
            converged = (
                jnp.asarray(False)
                if termination is None
                else optimality <= termination.optimality_threshold(initial_optimality)
            )

            def finish(_):
                status = jnp.where(
                    finite,
                    int(OptimizationStatus.SUCCESS),
                    int(OptimizationStatus.NONFINITE_EVALUATION),
                )
                updated = LeastSquaresState(
                    iteration=state.iteration + 1,
                    initial_optimality_norm=initial_optimality,
                    damping=state.damping,
                    accepted_steps=state.accepted_steps,
                    rejected_steps=state.rejected_steps + (~finite).astype(jnp.int32),
                    residual_evaluations=(
                        state.residual_evaluations + model.residual_evaluations
                    ),
                    jvp_evaluations=state.jvp_evaluations,
                    vjp_evaluations=state.vjp_evaluations,
                    linear_iterations=state.linear_iterations,
                    linear_solves=state.linear_solves,
                    setup_refreshes=state.setup_refreshes,
                    numeric_refreshes=state.numeric_refreshes,
                    linear_refresh_state=state.linear_refresh_state,
                    direction_fallbacks=state.direction_fallbacks,
                    metrics=IterativeStepMetrics(
                        objective=model.objective,
                        optimality_norm=optimality,
                        accepted=finite,
                        status=status,
                    ),
                )
                dynamic, _ = eqx.partition(updated, eqx.is_array)
                return parameters, dynamic, model.objective

            def gauss_newton(_):
                flat_residual, _ = ravel_pytree(model.residual)
                normal = model.jacobian.T @ model.jacobian
                normal = normal + self.regularization * jnp.eye(
                    flat_parameters.size, dtype=normal.dtype
                )
                flat_gradient, _ = ravel_pytree(model.gradient)
                proposed_flat = jnp.linalg.solve(normal, -flat_gradient)
                proposed = unravel(proposed_flat)
                proposed_directional = _tree_inner(model.gradient, proposed)
                usable = (
                    jnp.all(jnp.isfinite(proposed_flat))
                    & jnp.isfinite(proposed_directional)
                    & (proposed_directional < 0.0)
                )
                direction = _tree_where(usable, proposed, _tree_negative(model.gradient))
                directional = _tree_inner(model.gradient, direction)

                def objective(candidate):
                    residual = residual_function(candidate)
                    return 0.5 * _tree_inner(residual, residual)

                remaining_evaluations = (
                    None
                    if termination is None or termination.maximum_evaluations is None
                    else (
                        termination.maximum_evaluations
                        - state.residual_evaluations
                        - model.residual_evaluations
                        - model_cost
                    )
                )
                search = armijo_backtracking(
                    objective,
                    parameters,
                    model.objective,
                    direction,
                    directional,
                    step=_tree_add_scaled,
                    contains=_tree_allfinite,
                    policy=self.line_search,
                    maximum_evaluations=remaining_evaluations,
                )
                accepted = search.accepted
                step_norm = search.rate * _tree_norm(direction)
                stagnated = (
                    jnp.asarray(False)
                    if termination is None
                    else accepted
                    & (
                        step_norm
                        <= termination.step_threshold(_tree_norm(search.parameters))
                    )
                )
                budget_exhausted = (
                    jnp.asarray(False)
                    if remaining_evaluations is None
                    else search.evaluations
                    >= jnp.maximum(
                        jnp.asarray(remaining_evaluations, dtype=jnp.int32),
                        0,
                    )
                )
                status = jnp.where(
                    stagnated,
                    int(OptimizationStatus.STAGNATION),
                    jnp.where(
                        accepted,
                        int(OptimizationStatus.ITERATING),
                        jnp.where(
                            budget_exhausted,
                            int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED),
                            jnp.where(
                                search.finite_candidate_seen,
                                int(OptimizationStatus.LINE_SEARCH_FAILED),
                                int(OptimizationStatus.NONFINITE_EVALUATION),
                            ),
                        ),
                    ),
                )
                updated = LeastSquaresState(
                    iteration=state.iteration + 1,
                    initial_optimality_norm=initial_optimality,
                    damping=state.damping,
                    accepted_steps=state.accepted_steps + accepted.astype(jnp.int32),
                    rejected_steps=state.rejected_steps + (~accepted).astype(jnp.int32),
                    residual_evaluations=(
                        state.residual_evaluations
                        + model.residual_evaluations
                        + search.evaluations
                    ),
                    jvp_evaluations=state.jvp_evaluations,
                    vjp_evaluations=state.vjp_evaluations,
                    linear_iterations=state.linear_iterations + 1,
                    linear_solves=state.linear_solves + 1,
                    setup_refreshes=state.setup_refreshes,
                    numeric_refreshes=state.numeric_refreshes,
                    linear_refresh_state=state.linear_refresh_state,
                    direction_fallbacks=(
                        state.direction_fallbacks + (~usable).astype(jnp.int32)
                    ),
                    metrics=IterativeStepMetrics(
                        objective=search.value,
                        optimality_norm=optimality,
                        step_norm=step_norm,
                        accepted_step_size=search.rate,
                        globalization_evaluations=search.evaluations,
                        accepted=accepted,
                        linear_iterations=1,
                        direction_fallback=~usable,
                        status=status,
                    ),
                )
                dynamic, _ = eqx.partition(updated, eqx.is_array)
                return search.parameters, dynamic, search.value

            return jax.lax.cond(
                (~finite) | converged,
                finish,
                gauss_newton,
                None,
            )

        next_parameters, dynamic, objective = jax.lax.cond(
            budget_available,
            evaluate_step,
            no_budget,
            None,
        )
        return next_parameters, eqx.combine(dynamic, static_state), objective

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
        if not isinstance(problem, NonlinearLeastSquaresProblem):
            raise TypeError("problem must be a NonlinearLeastSquaresProblem.")
        if not isinstance(termination, OptimizationTermination):
            raise TypeError("termination must be an OptimizationTermination.")
        parameters = _validate_real_inexact_tree(
            initial_parameters, name="initial_parameters"
        )

        def residual_function(candidate):
            residual, _ = problem.value(candidate, args)
            return residual

        run = _run_least_squares_iterations(
            self,
            residual_function,
            parameters,
            termination,
        )
        parameters, state, status = run.parameters, run.state, run.status
        final_model = self._model(residual_function, parameters)
        residual_evaluations = (
            state.residual_evaluations + final_model.residual_evaluations
        )
        auxiliary = None
        if problem.has_aux:
            _, auxiliary = problem.value(parameters, args)
            residual_evaluations = residual_evaluations + 1
        final_optimality = _tree_norm(final_model.gradient)
        finite = (
            _tree_allfinite(parameters)
            & _tree_allfinite(final_model.residual)
            & _tree_allfinite(final_model.gradient)
            & jnp.isfinite(final_model.objective)
            & jnp.isfinite(final_optimality)
        )
        initial_nonfinite = status == int(OptimizationStatus.NONFINITE_INPUT)
        status = jnp.where(
            initial_nonfinite,
            status,
            jnp.where(
                ~finite,
                int(OptimizationStatus.NONFINITE_EVALUATION),
                jnp.where(
                    final_optimality
                    <= termination.optimality_threshold(state.initial_optimality_norm),
                    int(OptimizationStatus.SUCCESS),
                    status,
                ),
            ),
        )
        metrics = self.step_metrics(state)
        diagnostics = OptimizationDiagnostics(
            iterations=state.iteration,
            accepted_steps=state.accepted_steps,
            rejected_steps=state.rejected_steps,
            objective_evaluations=residual_evaluations,
            residual_evaluations=residual_evaluations,
            jacobian_evaluations=state.iteration + 2,
            linear_iterations=state.linear_iterations,
            linear_solves=state.linear_solves,
            globalization_evaluations=jnp.maximum(
                0,
                state.residual_evaluations
                - (state.iteration + 1) * final_model.residual_evaluations,
            ),
            initial_optimality_norm=state.initial_optimality_norm,
            final_optimality_norm=final_optimality,
            final_step_norm=metrics.step_norm,
            accepted_step_size=metrics.accepted_step_size,
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
                method=self.method_id,
                backend="phydrax-native",
                globalization=self.globalization_id,
                matrix_free=False,
                implicit_differentiation=False,
                notes=(
                    "Jacobian columns use central residual differences; no residual "
                    "JVP, VJP, or reverse-mode derivative is evaluated."
                ),
            ),
        )


__all__ = ["FiniteDifferenceGaussNewton"]
