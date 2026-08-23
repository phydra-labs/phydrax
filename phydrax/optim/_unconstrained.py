#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

from .._linear_refresh import LinearRefreshState
from .._strict import StrictModule
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    FunctionLinearOperator,
    HermitianSpectrum,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    PyTreeSpace,
    solve as solve_linear,
)
from ._iterative._base import AbstractScalarIterativeMethod
from ._iterative._globalization import (
    strong_wolfe_line_search,
    StrongWolfeLineSearch,
)
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
from ._scalar import solve_scalar_iterative
from ._trust_region import (
    solve_trust_region_subproblem,
    SteihaugToint,
    TrustRegionQuadraticProblem,
)


class _AbstractScalarExtensionState(StrictModule):
    """Shared accepted-point counters for native scalar extension methods."""

    iteration: Array
    initial_optimality_norm: Array
    accepted_steps: Array
    rejected_steps: Array
    objective_evaluations: Array
    gradient_evaluations: Array
    hvp_evaluations: Array
    linear_solves: Array
    linear_iterations: Array
    setup_refreshes: Array
    numeric_refreshes: Array
    linear_refresh_state: LinearRefreshState | None
    direction_fallbacks: Array
    metrics: IterativeStepMetrics

    def __init__(
        self,
        *,
        iteration: Any = 0,
        initial_optimality_norm: Any = jnp.nan,
        accepted_steps: Any = 0,
        rejected_steps: Any = 0,
        objective_evaluations: Any = 0,
        gradient_evaluations: Any = 0,
        hvp_evaluations: Any = 0,
        linear_solves: Any = 0,
        linear_iterations: Any = 0,
        setup_refreshes: Any = 0,
        numeric_refreshes: Any = 0,
        linear_refresh_state: LinearRefreshState | None = None,
        direction_fallbacks: Any = 0,
        metrics: IterativeStepMetrics | None = None,
    ):
        self.iteration = jnp.asarray(iteration, dtype=jnp.int32)
        self.initial_optimality_norm = jnp.asarray(initial_optimality_norm)
        self.accepted_steps = jnp.asarray(accepted_steps, dtype=jnp.int32)
        self.rejected_steps = jnp.asarray(rejected_steps, dtype=jnp.int32)
        self.objective_evaluations = jnp.asarray(objective_evaluations, dtype=jnp.int32)
        self.gradient_evaluations = jnp.asarray(gradient_evaluations, dtype=jnp.int32)
        self.hvp_evaluations = jnp.asarray(hvp_evaluations, dtype=jnp.int32)
        self.linear_solves = jnp.asarray(linear_solves, dtype=jnp.int32)
        self.linear_iterations = jnp.asarray(linear_iterations, dtype=jnp.int32)
        self.setup_refreshes = jnp.asarray(setup_refreshes, dtype=jnp.int32)
        self.numeric_refreshes = jnp.asarray(numeric_refreshes, dtype=jnp.int32)
        if linear_refresh_state is not None and not isinstance(
            linear_refresh_state, LinearRefreshState
        ):
            raise TypeError("linear_refresh_state must be a LinearRefreshState or None.")
        self.linear_refresh_state = linear_refresh_state
        self.direction_fallbacks = jnp.asarray(direction_fallbacks, dtype=jnp.int32)
        self.metrics = IterativeStepMetrics() if metrics is None else metrics


BetaMethod = Literal[
    "fletcher-reeves",
    "polak-ribiere+",
    "hestenes-stiefel+",
    "dai-yuan",
]


class NonlinearConjugateGradientState(_AbstractScalarExtensionState):
    """Accepted nonlinear-CG point data and the next search direction."""

    value: Array
    gradient: PyTree[Array]
    direction: PyTree[Array]

    def __init__(
        self,
        *,
        value: Any,
        gradient: PyTree[Any],
        direction: PyTree[Any],
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.value = jnp.asarray(value)
        self.gradient = gradient
        self.direction = direction


class NonlinearConjugateGradient(AbstractScalarIterativeMethod):
    """Strong-Wolfe nonlinear CG with safeguarded beta and explicit restarts."""

    line_search: StrongWolfeLineSearch
    beta_method: str = eqx.field(static=True)
    restart_interval: int | None = eqx.field(static=True)
    orthogonality_restart: float = eqx.field(static=True)
    descent_safeguard: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        beta_method: BetaMethod = "polak-ribiere+",
        line_search: StrongWolfeLineSearch | None = None,
        restart_interval: int | None = None,
        orthogonality_restart: float = 0.1,
        descent_safeguard: float = 1e-3,
    ):
        beta = str(beta_method)
        supported = {
            "fletcher-reeves",
            "polak-ribiere+",
            "hestenes-stiefel+",
            "dai-yuan",
        }
        if beta not in supported:
            raise ValueError(f"beta_method must be one of {sorted(supported)}.")
        search = StrongWolfeLineSearch() if line_search is None else line_search
        if not isinstance(search, StrongWolfeLineSearch):
            raise TypeError("line_search must be a StrongWolfeLineSearch or None.")
        interval = None if restart_interval is None else int(restart_interval)
        orthogonality = float(orthogonality_restart)
        safeguard = float(descent_safeguard)
        if interval is not None and interval < 1:
            raise ValueError("restart_interval must be positive or None.")
        if not isfinite(orthogonality) or not 0.0 <= orthogonality < 1.0:
            raise ValueError("orthogonality_restart must lie in [0, 1).")
        if not isfinite(safeguard) or not 0.0 < safeguard < 1.0:
            raise ValueError("descent_safeguard must lie in (0, 1).")
        self.line_search = search
        self.beta_method = beta
        self.restart_interval = interval
        self.orthogonality_restart = orthogonality
        self.descent_safeguard = safeguard

    @property
    def method_id(self) -> str:
        return f"nonlinear-conjugate-gradient/{self.beta_method}"

    @property
    def globalization_id(self) -> str:
        return "strong-wolfe"

    @property
    def capabilities(self) -> OptimizationCapabilities:
        return OptimizationCapabilities(
            scalar_objective=True,
            residual_objective=False,
            matrix_free=True,
            prepared_refresh=False,
            implicit_differentiation=True,
        )

    def init(self, parameters: PyTree[Any], /) -> NonlinearConjugateGradientState:
        parameters = _validate_real_inexact_tree(parameters, name="parameters")
        zeros = jax.tree.map(jnp.zeros_like, parameters)
        metric_nan = jnp.asarray(jnp.nan, dtype=_tree_norm(parameters).dtype)
        return NonlinearConjugateGradientState(
            value=metric_nan,
            gradient=zeros,
            direction=zeros,
            initial_optimality_norm=metric_nan,
            metrics=IterativeStepMetrics(objective=metric_nan),
        )

    def prepare_state(
        self,
        value_function,
        parameters: PyTree[Any],
        /,
    ) -> NonlinearConjugateGradientState:
        if not callable(value_function):
            raise TypeError("value_function must be callable.")
        parameters = _validate_real_inexact_tree(parameters, name="parameters")
        value, gradient = jax.value_and_grad(value_function)(parameters)
        optimality = _tree_norm(gradient)
        return NonlinearConjugateGradientState(
            value=value,
            gradient=gradient,
            direction=_tree_negative(gradient),
            initial_optimality_norm=optimality,
            objective_evaluations=1,
            gradient_evaluations=1,
            metrics=IterativeStepMetrics(
                objective=value,
                optimality_norm=optimality,
            ),
        )

    def _beta(
        self,
        new_gradient: PyTree[Any],
        old_gradient: PyTree[Any],
        old_direction: PyTree[Any],
        /,
    ) -> Array:
        difference = jax.tree.map(
            lambda new, old: new - old,
            new_gradient,
            old_gradient,
        )
        new_squared = _tree_inner(new_gradient, new_gradient)
        old_squared = _tree_inner(old_gradient, old_gradient)
        tiny = jnp.asarray(1e-30, dtype=jnp.asarray(new_squared).dtype)
        if self.beta_method == "fletcher-reeves":
            return new_squared / jnp.maximum(old_squared, tiny)
        numerator = _tree_inner(new_gradient, difference)
        if self.beta_method == "polak-ribiere+":
            return jnp.maximum(0.0, numerator / jnp.maximum(old_squared, tiny))
        denominator = _tree_inner(old_direction, difference)
        safe_denominator = jnp.where(
            jnp.abs(denominator) > tiny,
            denominator,
            jnp.ones_like(denominator),
        )
        if self.beta_method == "hestenes-stiefel+":
            return jnp.where(
                denominator > tiny,
                jnp.maximum(0.0, numerator / safe_denominator),
                0.0,
            )
        return jnp.where(
            denominator > tiny,
            new_squared / safe_denominator,
            0.0,
        )

    def step(
        self,
        value_function,
        parameters: PyTree[Any],
        state: _AbstractScalarExtensionState,
        /,
        *,
        termination: OptimizationTermination | None,
    ) -> tuple[PyTree[Any], NonlinearConjugateGradientState, Any]:
        if not callable(value_function):
            raise TypeError("value_function must be callable.")
        if not isinstance(state, NonlinearConjugateGradientState):
            raise TypeError("state must be a NonlinearConjugateGradientState.")
        _, static_state = eqx.partition(state, eqx.is_array)
        optimality = _tree_norm(state.gradient)
        finite = (
            jnp.isfinite(state.value)
            & _tree_allfinite(parameters)
            & _tree_allfinite(state.gradient)
        )
        converged = (
            jnp.asarray(False)
            if termination is None
            else optimality
            <= termination.optimality_threshold(state.initial_optimality_norm)
        )

        def terminal_step(_):
            status = jnp.where(
                finite,
                int(OptimizationStatus.SUCCESS),
                int(OptimizationStatus.NONFINITE_EVALUATION),
            )
            updated = NonlinearConjugateGradientState(
                value=state.value,
                gradient=state.gradient,
                direction=state.direction,
                iteration=state.iteration + 1,
                initial_optimality_norm=state.initial_optimality_norm,
                accepted_steps=state.accepted_steps,
                rejected_steps=state.rejected_steps + (~finite).astype(jnp.int32),
                objective_evaluations=state.objective_evaluations,
                gradient_evaluations=state.gradient_evaluations,
                hvp_evaluations=state.hvp_evaluations,
                linear_solves=state.linear_solves,
                linear_iterations=state.linear_iterations,
                setup_refreshes=state.setup_refreshes,
                numeric_refreshes=state.numeric_refreshes,
                linear_refresh_state=state.linear_refresh_state,
                direction_fallbacks=state.direction_fallbacks,
                metrics=IterativeStepMetrics(
                    objective=state.value,
                    optimality_norm=optimality,
                    accepted=finite,
                    status=status,
                ),
            )
            dynamic, _ = eqx.partition(updated, eqx.is_array)
            return parameters, dynamic, state.value

        def conjugate_gradient_step(_):
            proposed_directional = _tree_inner(state.gradient, state.direction)
            valid_stored_direction = (
                _tree_allfinite(state.direction)
                & jnp.isfinite(proposed_directional)
                & (
                    proposed_directional
                    <= -self.descent_safeguard * optimality * optimality
                )
            )
            direction = _tree_where(
                valid_stored_direction,
                state.direction,
                _tree_negative(state.gradient),
            )
            search = strong_wolfe_line_search(
                jax.value_and_grad(value_function),
                parameters,
                state.value,
                state.gradient,
                direction,
                step=_tree_add_scaled,
                contains=_tree_allfinite,
                policy=self.line_search,
                maximum_evaluations=(
                    None
                    if termination is None or termination.maximum_evaluations is None
                    else termination.maximum_evaluations - state.objective_evaluations
                ),
            )
            accepted = search.accepted
            new_optimality = _tree_norm(search.gradient)
            beta = self._beta(search.gradient, state.gradient, direction)
            conjugate_direction = _tree_add_scaled(
                _tree_negative(search.gradient),
                direction,
                beta,
            )
            new_squared = _tree_inner(search.gradient, search.gradient)
            orthogonality = jnp.abs(
                _tree_inner(search.gradient, state.gradient)
            ) / jnp.maximum(new_squared, 1e-30)
            periodic_restart = (
                jnp.asarray(False)
                if self.restart_interval is None
                else ((state.iteration + 1) % self.restart_interval) == 0
            )
            restart = periodic_restart | (orthogonality >= self.orthogonality_restart)
            conjugate_directional = _tree_inner(
                search.gradient,
                conjugate_direction,
            )
            safeguarded_restart = (
                ~jnp.isfinite(beta)
                | ~_tree_allfinite(conjugate_direction)
                | (conjugate_directional > -self.descent_safeguard * new_squared)
            )
            restart = restart | safeguarded_restart
            next_direction = _tree_where(
                restart,
                _tree_negative(search.gradient),
                conjugate_direction,
            )
            next_direction = _tree_where(
                accepted,
                next_direction,
                state.direction,
            )
            next_gradient = _tree_where(
                accepted,
                search.gradient,
                state.gradient,
            )
            next_value = jnp.where(accepted, search.value, state.value)
            step_norm = search.rate * _tree_norm(direction)
            stagnated = (
                jnp.asarray(False)
                if termination is None
                else accepted
                & (step_norm <= termination.step_threshold(_tree_norm(search.parameters)))
                & (
                    new_optimality
                    > termination.optimality_threshold(state.initial_optimality_norm)
                )
            )
            budget_exhausted = (
                jnp.asarray(False)
                if termination is None or termination.maximum_evaluations is None
                else state.objective_evaluations + search.evaluations
                >= termination.maximum_evaluations
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
            fallback = (~valid_stored_direction) | (accepted & restart)
            updated = NonlinearConjugateGradientState(
                value=next_value,
                gradient=next_gradient,
                direction=next_direction,
                iteration=state.iteration + 1,
                initial_optimality_norm=state.initial_optimality_norm,
                accepted_steps=state.accepted_steps + accepted.astype(jnp.int32),
                rejected_steps=state.rejected_steps + (~accepted).astype(jnp.int32),
                objective_evaluations=(state.objective_evaluations + search.evaluations),
                gradient_evaluations=(state.gradient_evaluations + search.evaluations),
                hvp_evaluations=state.hvp_evaluations,
                linear_solves=state.linear_solves,
                linear_iterations=state.linear_iterations,
                setup_refreshes=state.setup_refreshes,
                numeric_refreshes=state.numeric_refreshes,
                linear_refresh_state=state.linear_refresh_state,
                direction_fallbacks=(
                    state.direction_fallbacks + fallback.astype(jnp.int32)
                ),
                metrics=IterativeStepMetrics(
                    objective=next_value,
                    optimality_norm=jnp.where(accepted, new_optimality, optimality),
                    step_norm=step_norm,
                    accepted_step_size=search.rate,
                    globalization_evaluations=search.evaluations,
                    accepted=accepted,
                    direction_fallback=fallback,
                    status=status,
                ),
            )
            dynamic, _ = eqx.partition(updated, eqx.is_array)
            return search.parameters, dynamic, next_value

        next_parameters, dynamic_state, objective = jax.lax.cond(
            (~finite) | converged,
            terminal_step,
            conjugate_gradient_step,
            None,
        )
        return next_parameters, eqx.combine(dynamic_state, static_state), objective

    def step_metrics(
        self, state: _AbstractScalarExtensionState, /
    ) -> IterativeStepMetrics:
        if not isinstance(state, NonlinearConjugateGradientState):
            raise TypeError("state must be a NonlinearConjugateGradientState.")
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


class DenseNewtonDoglegState(_AbstractScalarExtensionState):
    """Accepted-point scalar state with a persistent trust-region radius."""

    trust_radius: Array

    def __init__(self, *, trust_radius: Any, **kwargs: Any):
        super().__init__(**kwargs)
        self.trust_radius = jnp.asarray(trust_radius)


class DenseNewtonDogleg(AbstractScalarIterativeMethod):
    """Dense Newton dogleg method with ratio-based trust-region acceptance."""

    initial_radius: float = eqx.field(static=True)
    maximum_radius: float = eqx.field(static=True)
    minimum_radius: float = eqx.field(static=True)
    acceptance_ratio: float = eqx.field(static=True)
    shrink_ratio: float = eqx.field(static=True)
    expansion_ratio: float = eqx.field(static=True)
    shrink_factor: float = eqx.field(static=True)
    expansion_factor: float = eqx.field(static=True)
    minimum_curvature: float = eqx.field(static=True)
    max_dense_dimension: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        initial_radius: float = 1.0,
        maximum_radius: float = 1e4,
        minimum_radius: float = 1e-12,
        acceptance_ratio: float = 1e-4,
        shrink_ratio: float = 0.25,
        expansion_ratio: float = 0.75,
        shrink_factor: float = 0.25,
        expansion_factor: float = 2.0,
        minimum_curvature: float = 1e-10,
        max_dense_dimension: int = 512,
    ):
        values = tuple(
            float(value)
            for value in (
                initial_radius,
                maximum_radius,
                minimum_radius,
                acceptance_ratio,
                shrink_ratio,
                expansion_ratio,
                shrink_factor,
                expansion_factor,
                minimum_curvature,
            )
        )
        dimension = int(max_dense_dimension)
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Trust-region controls must be positive and finite.")
        if not values[2] <= values[0] <= values[1]:
            raise ValueError(
                "Radii must satisfy minimum_radius <= initial_radius <= maximum_radius."
            )
        if not 0.0 < values[3] < values[4] < values[5] < 1.0:
            raise ValueError("Ratios must satisfy acceptance < shrink < expansion < one.")
        if not 0.0 < values[6] < 1.0 or values[7] <= 1.0:
            raise ValueError("Radius factors must shrink below and expand above one.")
        if dimension < 1:
            raise ValueError("max_dense_dimension must be positive.")
        (
            self.initial_radius,
            self.maximum_radius,
            self.minimum_radius,
            self.acceptance_ratio,
            self.shrink_ratio,
            self.expansion_ratio,
            self.shrink_factor,
            self.expansion_factor,
            self.minimum_curvature,
        ) = values
        self.max_dense_dimension = dimension

    @property
    def method_id(self) -> str:
        return "dense-newton-dogleg"

    @property
    def globalization_id(self) -> str:
        return "trust-region-ratio"

    @property
    def capabilities(self) -> OptimizationCapabilities:
        return OptimizationCapabilities(
            scalar_objective=True,
            residual_objective=False,
            matrix_free=False,
            prepared_refresh=False,
            implicit_differentiation=True,
        )

    def init(self, parameters: PyTree[Any], /) -> DenseNewtonDoglegState:
        parameters = _validate_real_inexact_tree(parameters, name="parameters")
        flat, _ = ravel_pytree(parameters)
        if int(flat.size) > self.max_dense_dimension:
            raise ValueError(
                f"DenseNewtonDogleg has {flat.size} variables, exceeding "
                f"max_dense_dimension={self.max_dense_dimension}."
            )
        metric_nan = jnp.asarray(jnp.nan, dtype=flat.dtype)
        return DenseNewtonDoglegState(
            trust_radius=jnp.asarray(self.initial_radius, dtype=flat.dtype),
            initial_optimality_norm=metric_nan,
            metrics=IterativeStepMetrics(objective=metric_nan),
        )

    def prepare_state(
        self,
        value_function,
        parameters: PyTree[Any],
        /,
    ) -> DenseNewtonDoglegState:
        if not callable(value_function):
            raise TypeError("value_function must be callable.")
        return self.init(parameters)

    def step(
        self,
        value_function,
        parameters: PyTree[Any],
        state: _AbstractScalarExtensionState,
        /,
        *,
        termination: OptimizationTermination | None,
    ) -> tuple[PyTree[Any], DenseNewtonDoglegState, Any]:
        if not callable(value_function):
            raise TypeError("value_function must be callable.")
        if not isinstance(state, DenseNewtonDoglegState):
            raise TypeError("state must be a DenseNewtonDoglegState.")
        flat_parameters, unravel = ravel_pytree(parameters)
        if int(flat_parameters.size) > self.max_dense_dimension:
            raise ValueError(
                f"DenseNewtonDogleg has {flat_parameters.size} variables, exceeding "
                f"max_dense_dimension={self.max_dense_dimension}."
            )

        def flat_objective(candidate):
            return value_function(unravel(candidate))

        value, flat_gradient = jax.value_and_grad(flat_objective)(flat_parameters)
        optimality = jnp.linalg.norm(flat_gradient)
        initial_optimality = jnp.where(
            state.iteration == 0,
            optimality,
            state.initial_optimality_norm,
        )
        finite = (
            jnp.isfinite(value)
            & jnp.isfinite(optimality)
            & jnp.all(jnp.isfinite(flat_gradient))
            & _tree_allfinite(parameters)
        )
        converged = (
            jnp.asarray(False)
            if termination is None
            else optimality <= termination.optimality_threshold(initial_optimality)
        )
        budget_allows_candidate = (
            jnp.asarray(True)
            if termination is None or termination.maximum_evaluations is None
            else state.objective_evaluations + 2 <= termination.maximum_evaluations
        )
        _, static_state = eqx.partition(state, eqx.is_array)

        def terminal_step(_):
            status = jnp.where(
                ~finite,
                int(OptimizationStatus.NONFINITE_EVALUATION),
                jnp.where(
                    converged,
                    int(OptimizationStatus.SUCCESS),
                    int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED),
                ),
            )
            updated = DenseNewtonDoglegState(
                trust_radius=state.trust_radius,
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
                metrics=IterativeStepMetrics(
                    objective=value,
                    optimality_norm=optimality,
                    step_norm=state.metrics.step_norm,
                    accepted_step_size=state.metrics.accepted_step_size,
                    globalization_evaluations=(state.metrics.globalization_evaluations),
                    accepted=finite,
                    damping=state.trust_radius,
                    reduction_ratio=state.metrics.reduction_ratio,
                    status=status,
                ),
            )
            dynamic, _ = eqx.partition(updated, eqx.is_array)
            return parameters, dynamic, value

        def trust_region_step(_):
            hessian = jax.hessian(flat_objective)(flat_parameters)
            hessian = 0.5 * (hessian + hessian.T)
            spectrum = HermitianSpectrum(hessian)
            eigenvalues = spectrum.eigenvalues
            positive_definite = spectrum.valid & (
                jnp.min(eigenvalues) >= self.minimum_curvature
            )
            newton_direction = solve_linear(
                LinearSystem(DenseLinearOperator(hessian)),
                -flat_gradient,
                policy=LinearSolvePolicy(DenseLU()),
            ).value
            newton_usable = positive_definite & jnp.all(jnp.isfinite(newton_direction))
            gradient_squared = jnp.vdot(flat_gradient, flat_gradient).real
            gradient_norm = jnp.sqrt(gradient_squared)
            gradient_curvature = jnp.vdot(
                flat_gradient,
                hessian @ flat_gradient,
            ).real
            boundary_scale = state.trust_radius / jnp.maximum(gradient_norm, 1e-30)
            cauchy_scale = jnp.where(
                gradient_curvature > 0.0,
                jnp.minimum(
                    gradient_squared / jnp.maximum(gradient_curvature, 1e-30),
                    boundary_scale,
                ),
                boundary_scale,
            )
            cauchy_direction = -cauchy_scale * flat_gradient
            newton_inside = newton_usable & (
                jnp.linalg.norm(newton_direction) <= state.trust_radius
            )
            dogleg_delta = newton_direction - cauchy_direction
            quadratic_a = jnp.vdot(dogleg_delta, dogleg_delta).real
            quadratic_b = 2.0 * jnp.vdot(cauchy_direction, dogleg_delta).real
            quadratic_c = (
                jnp.vdot(cauchy_direction, cauchy_direction).real
                - state.trust_radius * state.trust_radius
            )
            discriminant = jnp.maximum(
                quadratic_b * quadratic_b - 4.0 * quadratic_a * quadratic_c,
                0.0,
            )
            dogleg_rate = jnp.clip(
                (-quadratic_b + jnp.sqrt(discriminant))
                / jnp.maximum(2.0 * quadratic_a, 1e-30),
                0.0,
                1.0,
            )
            boundary_direction = cauchy_direction + dogleg_rate * dogleg_delta
            direction = jnp.where(
                newton_inside,
                newton_direction,
                jnp.where(newton_usable, boundary_direction, cauchy_direction),
            )
            predicted_reduction = -(
                jnp.vdot(flat_gradient, direction).real
                + 0.5 * jnp.vdot(direction, hessian @ direction).real
            )
            candidate_flat = flat_parameters + direction
            candidate_value, candidate_gradient = jax.value_and_grad(flat_objective)(
                candidate_flat
            )
            actual_reduction = value - candidate_value
            ratio = actual_reduction / jnp.maximum(predicted_reduction, 1e-30)
            finite_candidate = (
                jnp.isfinite(candidate_value)
                & jnp.all(jnp.isfinite(candidate_gradient))
                & jnp.all(jnp.isfinite(direction))
            )
            accepted = (
                finite_candidate
                & jnp.isfinite(predicted_reduction)
                & (predicted_reduction > 0.0)
                & (ratio >= self.acceptance_ratio)
            )
            step_norm = jnp.linalg.norm(direction)
            shrink = (~finite_candidate) | (ratio < self.shrink_ratio)
            expand = (
                finite_candidate
                & (ratio > self.expansion_ratio)
                & (step_norm >= 0.9 * state.trust_radius)
            )
            next_radius = jnp.where(
                shrink,
                jnp.maximum(
                    self.minimum_radius,
                    self.shrink_factor * state.trust_radius,
                ),
                jnp.where(
                    expand,
                    jnp.minimum(
                        self.maximum_radius,
                        self.expansion_factor * state.trust_radius,
                    ),
                    state.trust_radius,
                ),
            )
            accepted_flat = jnp.where(accepted, candidate_flat, flat_parameters)
            accepted_parameters = unravel(accepted_flat)
            accepted_optimality = jnp.where(
                accepted,
                jnp.linalg.norm(candidate_gradient),
                optimality,
            )
            stagnated = (
                jnp.asarray(False)
                if termination is None
                else accepted
                & (
                    step_norm
                    <= termination.step_threshold(jnp.linalg.norm(accepted_flat))
                )
                & (
                    accepted_optimality
                    > termination.optimality_threshold(initial_optimality)
                )
            )
            failed = (~accepted) & (next_radius <= self.minimum_radius)
            status = jnp.where(
                stagnated,
                int(OptimizationStatus.STAGNATION),
                jnp.where(
                    failed,
                    int(OptimizationStatus.TRUST_REGION_FAILED),
                    int(OptimizationStatus.ITERATING),
                ),
            )
            fallback = ~newton_usable
            updated = DenseNewtonDoglegState(
                trust_radius=next_radius,
                iteration=state.iteration + 1,
                initial_optimality_norm=initial_optimality,
                accepted_steps=state.accepted_steps + accepted.astype(jnp.int32),
                rejected_steps=state.rejected_steps + (~accepted).astype(jnp.int32),
                objective_evaluations=state.objective_evaluations + 2,
                gradient_evaluations=state.gradient_evaluations + 2,
                hvp_evaluations=(state.hvp_evaluations + flat_parameters.size),
                linear_solves=state.linear_solves + 1,
                linear_iterations=state.linear_iterations + 1,
                setup_refreshes=state.setup_refreshes,
                numeric_refreshes=state.numeric_refreshes,
                linear_refresh_state=state.linear_refresh_state,
                direction_fallbacks=(
                    state.direction_fallbacks + fallback.astype(jnp.int32)
                ),
                metrics=IterativeStepMetrics(
                    objective=jnp.where(accepted, candidate_value, value),
                    optimality_norm=accepted_optimality,
                    step_norm=step_norm,
                    accepted_step_size=jnp.where(accepted, 1.0, 0.0),
                    accepted=accepted,
                    linear_iterations=1,
                    damping=next_radius,
                    reduction_ratio=ratio,
                    direction_fallback=fallback,
                    status=status,
                ),
            )
            dynamic, _ = eqx.partition(updated, eqx.is_array)
            return (
                accepted_parameters,
                dynamic,
                jnp.where(accepted, candidate_value, value),
            )

        next_parameters, dynamic_state, objective = jax.lax.cond(
            (~finite) | converged | (~budget_allows_candidate),
            terminal_step,
            trust_region_step,
            None,
        )
        return next_parameters, eqx.combine(dynamic_state, static_state), objective

    def step_metrics(
        self, state: _AbstractScalarExtensionState, /
    ) -> IterativeStepMetrics:
        if not isinstance(state, DenseNewtonDoglegState):
            raise TypeError("state must be a DenseNewtonDoglegState.")
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


class NewtonTrustRegionState(_AbstractScalarExtensionState):
    """Accepted-point matrix-free trust-region state."""

    trust_radius: Array

    def __init__(self, *, trust_radius: Any, **kwargs: Any):
        super().__init__(**kwargs)
        self.trust_radius = jnp.asarray(trust_radius)


class NewtonTrustRegion(AbstractScalarIterativeMethod):
    """Matrix-free Newton method with a Steihaug--Toint trust region."""

    subproblem: SteihaugToint
    initial_radius: float = eqx.field(static=True)
    maximum_radius: float = eqx.field(static=True)
    minimum_radius: float = eqx.field(static=True)
    acceptance_ratio: float = eqx.field(static=True)
    shrink_ratio: float = eqx.field(static=True)
    expansion_ratio: float = eqx.field(static=True)
    shrink_factor: float = eqx.field(static=True)
    expansion_factor: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        subproblem: SteihaugToint | None = None,
        initial_radius: float = 1.0,
        maximum_radius: float = 1e4,
        minimum_radius: float = 1e-12,
        acceptance_ratio: float = 1e-4,
        shrink_ratio: float = 0.25,
        expansion_ratio: float = 0.75,
        shrink_factor: float = 0.25,
        expansion_factor: float = 2.0,
    ):
        subproblem_ = SteihaugToint() if subproblem is None else subproblem
        if not isinstance(subproblem_, SteihaugToint):
            raise TypeError("subproblem must be SteihaugToint or None.")
        values = tuple(
            float(value)
            for value in (
                initial_radius,
                maximum_radius,
                minimum_radius,
                acceptance_ratio,
                shrink_ratio,
                expansion_ratio,
                shrink_factor,
                expansion_factor,
            )
        )
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Trust-region controls must be positive and finite.")
        if not values[2] <= values[0] <= values[1]:
            raise ValueError(
                "Radii must satisfy minimum_radius <= initial_radius <= maximum_radius."
            )
        if not 0.0 < values[3] < values[4] < values[5] < 1.0:
            raise ValueError("Ratios must satisfy acceptance < shrink < expansion < one.")
        if not 0.0 < values[6] < 1.0 or values[7] <= 1.0:
            raise ValueError("Radius factors must shrink below and expand above one.")
        self.subproblem = subproblem_
        (
            self.initial_radius,
            self.maximum_radius,
            self.minimum_radius,
            self.acceptance_ratio,
            self.shrink_ratio,
            self.expansion_ratio,
            self.shrink_factor,
            self.expansion_factor,
        ) = values

    @property
    def method_id(self) -> str:
        return "newton-trust-region/steihaug-toint"

    @property
    def globalization_id(self) -> str:
        return "trust-region-ratio"

    @property
    def capabilities(self) -> OptimizationCapabilities:
        return OptimizationCapabilities(
            scalar_objective=True,
            residual_objective=False,
            matrix_free=True,
            prepared_refresh=False,
            implicit_differentiation=True,
        )

    def init(self, parameters: PyTree[Any], /) -> NewtonTrustRegionState:
        parameters_ = _validate_real_inexact_tree(parameters, name="parameters")
        metric_nan = jnp.asarray(
            jnp.nan,
            dtype=jax.tree.leaves(parameters_)[0].dtype,
        )
        return NewtonTrustRegionState(
            trust_radius=jnp.asarray(self.initial_radius, dtype=metric_nan.dtype),
            initial_optimality_norm=metric_nan,
            metrics=IterativeStepMetrics(objective=metric_nan),
        )

    def prepare_state(
        self,
        value_function,
        parameters: PyTree[Any],
        /,
    ) -> NewtonTrustRegionState:
        if not callable(value_function):
            raise TypeError("value_function must be callable.")
        return self.init(parameters)

    def step(
        self,
        value_function,
        parameters: PyTree[Any],
        state: _AbstractScalarExtensionState,
        /,
        *,
        termination: OptimizationTermination | None,
    ) -> tuple[PyTree[Any], NewtonTrustRegionState, Any]:
        if not callable(value_function):
            raise TypeError("value_function must be callable.")
        if not isinstance(state, NewtonTrustRegionState):
            raise TypeError("state must be a NewtonTrustRegionState.")
        parameters_ = _validate_real_inexact_tree(parameters, name="parameters")
        value, gradient = jax.value_and_grad(value_function)(parameters_)
        optimality = _tree_norm(gradient)
        initial_optimality = jnp.where(
            state.iteration == 0,
            optimality,
            state.initial_optimality_norm,
        )
        finite = (
            jnp.isfinite(value)
            & jnp.isfinite(optimality)
            & _tree_allfinite(parameters_)
            & _tree_allfinite(gradient)
        )
        converged = (
            jnp.asarray(False)
            if termination is None
            else optimality <= termination.optimality_threshold(initial_optimality)
        )
        budget_allows_candidate = (
            jnp.asarray(True)
            if termination is None or termination.maximum_evaluations is None
            else state.objective_evaluations + 2 <= termination.maximum_evaluations
        )
        _, static_state = eqx.partition(state, eqx.is_array)

        def terminal_step(_):
            status = jnp.where(
                ~finite,
                int(OptimizationStatus.NONFINITE_EVALUATION),
                jnp.where(
                    converged,
                    int(OptimizationStatus.SUCCESS),
                    int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED),
                ),
            ).astype(jnp.int32)
            updated = NewtonTrustRegionState(
                trust_radius=state.trust_radius,
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
                metrics=IterativeStepMetrics(
                    objective=value,
                    optimality_norm=optimality,
                    step_norm=state.metrics.step_norm,
                    accepted_step_size=state.metrics.accepted_step_size,
                    globalization_evaluations=(state.metrics.globalization_evaluations),
                    accepted=finite,
                    damping=state.trust_radius,
                    reduction_ratio=state.metrics.reduction_ratio,
                    status=status,
                ),
            )
            dynamic, _ = eqx.partition(updated, eqx.is_array)
            return parameters_, dynamic, value

        def trust_region_step(_):
            linearized_gradient, hessian_action = jax.linearize(
                jax.grad(value_function),
                parameters_,
            )
            space = PyTreeSpace(parameters_)
            hessian = FunctionLinearOperator(
                hessian_action,
                source=space,
                target=space,
                properties=OperatorProperties(
                    self_adjoint=True,
                    evidence={"self_adjoint": "construction"},
                ),
                operator_id="objective-hessian-action",
                closure_convert=False,
            )
            subproblem = solve_trust_region_subproblem(
                TrustRegionQuadraticProblem(
                    hessian,
                    linearized_gradient,
                    state.trust_radius,
                ),
                method=self.subproblem,
            )
            direction = subproblem.step
            candidate = _tree_add_scaled(parameters_, direction, 1.0)
            candidate_value, candidate_gradient = jax.value_and_grad(value_function)(
                candidate
            )
            predicted = subproblem.diagnostics.predicted_reduction
            actual = value - candidate_value
            ratio = actual / jnp.maximum(predicted, 1e-30)
            candidate_finite = (
                jnp.isfinite(candidate_value)
                & _tree_allfinite(candidate)
                & _tree_allfinite(candidate_gradient)
            )
            accepted = (
                subproblem.successful
                & candidate_finite
                & jnp.isfinite(predicted)
                & (predicted > 0.0)
                & jnp.isfinite(ratio)
                & (ratio >= self.acceptance_ratio)
            )
            step_norm = _tree_norm(direction)
            shrink = (
                ~subproblem.successful
                | ~candidate_finite
                | ~jnp.isfinite(ratio)
                | (ratio < self.shrink_ratio)
            )
            expand = (
                accepted
                & (ratio > self.expansion_ratio)
                & subproblem.diagnostics.boundary_hit
            )
            next_radius = jnp.where(
                shrink,
                jnp.maximum(
                    self.minimum_radius,
                    self.shrink_factor * state.trust_radius,
                ),
                jnp.where(
                    expand,
                    jnp.minimum(
                        self.maximum_radius,
                        self.expansion_factor * state.trust_radius,
                    ),
                    state.trust_radius,
                ),
            )
            accepted_parameters = _tree_where(accepted, candidate, parameters_)
            accepted_optimality = jnp.where(
                accepted,
                _tree_norm(candidate_gradient),
                optimality,
            )
            stagnated = (
                jnp.asarray(False)
                if termination is None
                else accepted
                & (
                    step_norm
                    <= termination.step_threshold(_tree_norm(accepted_parameters))
                )
                & (
                    accepted_optimality
                    > termination.optimality_threshold(initial_optimality)
                )
            )
            failed = (~accepted) & (next_radius <= self.minimum_radius)
            status = jnp.where(
                stagnated,
                int(OptimizationStatus.STAGNATION),
                jnp.where(
                    failed,
                    int(OptimizationStatus.TRUST_REGION_FAILED),
                    int(OptimizationStatus.ITERATING),
                ),
            ).astype(jnp.int32)
            updated = NewtonTrustRegionState(
                trust_radius=next_radius,
                iteration=state.iteration + 1,
                initial_optimality_norm=initial_optimality,
                accepted_steps=state.accepted_steps + accepted.astype(jnp.int32),
                rejected_steps=state.rejected_steps + (~accepted).astype(jnp.int32),
                objective_evaluations=state.objective_evaluations + 2,
                gradient_evaluations=state.gradient_evaluations + 3,
                hvp_evaluations=(
                    state.hvp_evaluations + subproblem.diagnostics.hessian_actions
                ),
                linear_solves=state.linear_solves + 1,
                linear_iterations=(
                    state.linear_iterations + subproblem.diagnostics.iterations
                ),
                setup_refreshes=state.setup_refreshes,
                numeric_refreshes=state.numeric_refreshes,
                linear_refresh_state=state.linear_refresh_state,
                direction_fallbacks=state.direction_fallbacks,
                metrics=IterativeStepMetrics(
                    objective=jnp.where(accepted, candidate_value, value),
                    optimality_norm=accepted_optimality,
                    step_norm=step_norm,
                    accepted_step_size=jnp.where(accepted, 1.0, 0.0),
                    accepted=accepted,
                    linear_iterations=subproblem.diagnostics.iterations,
                    damping=next_radius,
                    reduction_ratio=ratio,
                    status=status,
                ),
            )
            dynamic, _ = eqx.partition(updated, eqx.is_array)
            return (
                accepted_parameters,
                dynamic,
                jnp.where(accepted, candidate_value, value),
            )

        next_parameters, dynamic_state, objective = jax.lax.cond(
            (~finite) | converged | (~budget_allows_candidate),
            terminal_step,
            trust_region_step,
            None,
        )
        return next_parameters, eqx.combine(dynamic_state, static_state), objective

    def step_metrics(
        self,
        state: _AbstractScalarExtensionState,
        /,
    ) -> IterativeStepMetrics:
        if not isinstance(state, NewtonTrustRegionState):
            raise TypeError("state must be a NewtonTrustRegionState.")
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


__all__ = [
    "BetaMethod",
    "DenseNewtonDogleg",
    "DenseNewtonDoglegState",
    "NewtonTrustRegion",
    "NewtonTrustRegionState",
    "NonlinearConjugateGradient",
    "NonlinearConjugateGradientState",
]
