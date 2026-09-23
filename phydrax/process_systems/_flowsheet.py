#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Equation-oriented steady flowsheet compilation and Newton execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..ein import contract
from ..linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve


@dataclass(frozen=True, slots=True)
class FlowsheetSolveResult:
    variables: Array
    residual: Array
    scaled_residual_norm: Array
    iterations: int
    successful: Array


@dataclass(frozen=True, slots=True)
class EquationOrientedFlowsheet:
    """Square differentiable residual system with scaled damped Newton solves."""

    residual_function: Callable[[Array], Array]
    variable_scale: Array
    residual_scale: Array
    lower_bounds: Array
    upper_bounds: Array

    @classmethod
    def create(
        cls,
        residual_function: Callable[[Array], Array],
        variable_scale: ArrayLike,
        residual_scale: ArrayLike,
        /,
        *,
        lower_bounds: ArrayLike | None = None,
        upper_bounds: ArrayLike | None = None,
    ) -> EquationOrientedFlowsheet:
        if not callable(residual_function):
            raise TypeError("Flowsheet residual must be callable.")
        variable = np.asarray(variable_scale, dtype=np.float64)
        residual = np.asarray(residual_scale, dtype=np.float64)
        if not np.all(np.isfinite(variable)) or not np.all(np.isfinite(residual)):
            raise ValueError("Flowsheet scales must be finite.")
        if variable.ndim != 1 or variable.size == 0 or np.any(variable <= 0):
            raise ValueError("Flowsheet variable scales must be a positive vector.")
        if residual.shape != variable.shape or np.any(residual <= 0):
            raise ValueError("Flowsheet residual scales must be positive and aligned.")
        lower = (
            np.full_like(variable, -np.inf)
            if lower_bounds is None
            else np.broadcast_to(
                np.asarray(lower_bounds, dtype=np.float64), variable.shape
            )
        )
        upper = (
            np.full_like(variable, np.inf)
            if upper_bounds is None
            else np.broadcast_to(
                np.asarray(upper_bounds, dtype=np.float64), variable.shape
            )
        )
        if np.any(np.isnan(lower)) or np.any(np.isnan(upper)) or np.any(lower >= upper):
            raise ValueError("Flowsheet variable bounds must have nonempty interiors.")
        return cls(
            residual_function,
            jnp.asarray(variable),
            jnp.asarray(residual),
            jnp.asarray(lower),
            jnp.asarray(upper),
        )

    def residual(self, variables: ArrayLike, /) -> Array:
        value = jnp.asarray(self.residual_function(jnp.asarray(variables)))
        if value.shape != self.residual_scale.shape:
            raise ValueError("Flowsheet residual output has incompatible shape.")
        return value

    def solve(
        self,
        initial_variables: ArrayLike,
        /,
        *,
        tolerance: float = 1e-10,
        maximum_iterations: int = 32,
        maximum_line_search_steps: int = 12,
    ) -> FlowsheetSolveResult:
        value = jnp.asarray(initial_variables)
        if value.shape != self.variable_scale.shape:
            raise ValueError("Flowsheet initial variables have incompatible shape.")
        if (
            not np.isfinite(tolerance)
            or tolerance <= 0
            or isinstance(maximum_iterations, bool)
            or not isinstance(maximum_iterations, int)
            or maximum_iterations <= 0
            or isinstance(maximum_line_search_steps, bool)
            or not isinstance(maximum_line_search_steps, int)
            or maximum_line_search_steps <= 0
        ):
            raise ValueError("Flowsheet nonlinear controls must be finite and positive.")
        value = jnp.clip(value, self.lower_bounds, self.upper_bounds)

        def norm_at(candidate):
            candidate_residual = self.residual(candidate)
            scaled = candidate_residual / self.residual_scale
            return jnp.sqrt(jnp.real(contract("i,i->", jnp.conj(scaled), scaled)))

        def nonlinear_step(iteration, state):
            current, converged, failed, completed = state
            residual = self.residual(current)
            scaled = residual / self.residual_scale
            norm = jnp.sqrt(jnp.real(contract("i,i->", jnp.conj(scaled), scaled)))
            already_converged = jnp.isfinite(norm) & (norm <= tolerance)
            active = ~converged & ~failed & ~already_converged
            jacobian = jax.jacfwd(self.residual_function)(current)
            scaled_jacobian = (
                jacobian * self.variable_scale[None, :] / self.residual_scale[:, None]
            )
            correction = solve(
                LinearSystem(DenseLinearOperator(scaled_jacobian)),
                -scaled,
                policy=LinearSolvePolicy(DenseLU()),
            )
            linear_successful = correction.successful & jnp.all(
                jnp.isfinite(correction.value)
            )
            direction = self.variable_scale * correction.value

            def line_search(_, search_state):
                accepted, accepted_norm, found, step = search_state
                candidate = jnp.clip(
                    current + step * direction,
                    self.lower_bounds,
                    self.upper_bounds,
                )
                candidate_norm = norm_at(candidate)
                improve = ~found & jnp.isfinite(candidate_norm) & (candidate_norm < norm)
                return (
                    jnp.where(improve, candidate, accepted),
                    jnp.where(improve, candidate_norm, accepted_norm),
                    found | improve,
                    step * 0.5,
                )

            accepted, accepted_norm, found, _ = jax.lax.fori_loop(
                0,
                maximum_line_search_steps,
                line_search,
                (
                    current,
                    norm,
                    jnp.asarray(False),
                    jnp.asarray(1.0, dtype=current.dtype),
                ),
            )
            commit = active & linear_successful & found
            next_value = jnp.where(commit, accepted, current)
            next_converged = (
                converged | already_converged | (commit & (accepted_norm <= tolerance))
            )
            next_failed = failed | (
                active & (~linear_successful | ~found | ~jnp.isfinite(norm))
            )
            next_completed = jnp.where(active, iteration + 1, completed)
            return next_value, next_converged, next_failed, next_completed

        value, _, failed, completed = jax.lax.fori_loop(
            0,
            maximum_iterations,
            nonlinear_step,
            (
                value,
                jnp.asarray(False),
                jnp.asarray(False),
                jnp.asarray(0, dtype=jnp.int32),
            ),
        )
        final_residual = self.residual(value)
        final_scaled = final_residual / self.residual_scale
        final_norm = jnp.sqrt(
            jnp.real(contract("i,i->", jnp.conj(final_scaled), final_scaled))
        )
        successful = ~failed & jnp.isfinite(final_norm) & (final_norm <= tolerance)
        return FlowsheetSolveResult(
            value,
            final_residual,
            final_norm,
            completed,
            successful,
        )


__all__ = ["EquationOrientedFlowsheet", "FlowsheetSolveResult"]
