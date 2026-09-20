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
        variable = np.asarray(variable_scale, dtype=float)
        residual = np.asarray(residual_scale, dtype=float)
        if variable.ndim != 1 or variable.size == 0 or np.any(variable <= 0):
            raise ValueError("Flowsheet variable scales must be a positive vector.")
        if residual.shape != variable.shape or np.any(residual <= 0):
            raise ValueError("Flowsheet residual scales must be positive and aligned.")
        lower = (
            np.full_like(variable, -np.inf)
            if lower_bounds is None
            else np.broadcast_to(np.asarray(lower_bounds, dtype=float), variable.shape)
        )
        upper = (
            np.full_like(variable, np.inf)
            if upper_bounds is None
            else np.broadcast_to(np.asarray(upper_bounds, dtype=float), variable.shape)
        )
        if np.any(lower >= upper):
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
        if tolerance <= 0 or maximum_iterations <= 0 or maximum_line_search_steps <= 0:
            raise ValueError("Flowsheet nonlinear controls must be positive.")
        value = jnp.clip(value, self.lower_bounds, self.upper_bounds)
        completed = 0
        for iteration in range(maximum_iterations):
            residual = self.residual(value)
            scaled = residual / self.residual_scale
            norm = jnp.sqrt(jnp.real(contract("i,i->", jnp.conj(scaled), scaled)))
            if bool(norm <= tolerance):
                completed = iteration
                break
            jacobian = jax.jacfwd(self.residual_function)(value)
            scaled_jacobian = (
                jacobian * self.variable_scale[None, :] / self.residual_scale[:, None]
            )
            correction = solve(
                LinearSystem(DenseLinearOperator(scaled_jacobian)),
                -scaled,
                policy=LinearSolvePolicy(DenseLU()),
            )
            direction = self.variable_scale * correction.value
            accepted = value
            accepted_norm = norm
            step = 1.0
            for _ in range(maximum_line_search_steps):
                candidate = jnp.clip(
                    value + step * direction, self.lower_bounds, self.upper_bounds
                )
                candidate_scaled = self.residual(candidate) / self.residual_scale
                candidate_norm = jnp.sqrt(
                    jnp.real(
                        contract(
                            "i,i->",
                            jnp.conj(candidate_scaled),
                            candidate_scaled,
                        )
                    )
                )
                if bool(candidate_norm < accepted_norm):
                    accepted = candidate
                    accepted_norm = candidate_norm
                    break
                step *= 0.5
            value = accepted
            completed = iteration + 1
        final_residual = self.residual(value)
        final_scaled = final_residual / self.residual_scale
        final_norm = jnp.sqrt(
            jnp.real(contract("i,i->", jnp.conj(final_scaled), final_scaled))
        )
        successful = jnp.isfinite(final_norm) & (final_norm <= tolerance)
        return FlowsheetSolveResult(
            value,
            final_residual,
            final_norm,
            completed,
            successful,
        )


__all__ = ["EquationOrientedFlowsheet", "FlowsheetSolveResult"]
