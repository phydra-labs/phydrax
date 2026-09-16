#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    DenseLinearOperator,
    LinearSystem,
    SmallLinearSolvePlan,
    solve,
    solve_small_linear,
)


class LocalRootDiagnostics(StrictModule):
    converged: Array
    iterations: Array
    residual: Array
    derivative: Array
    finite: Array


class VectorLocalRootDiagnostics(StrictModule):
    converged: Array
    iterations: Array
    residual: Array
    residual_norm: Array
    jacobian: Array
    condition_estimate: Array
    finite: Array


class VectorLocalRootPlan(StrictModule, NonTrainableState):
    """Bounded one-to-three component root with implicit-function derivatives."""

    dimension: int = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        /,
        *,
        maximum_steps: int = 30,
        tolerance: float = 1.0e-10,
        plan_id: str,
    ):
        dimension_ = int(dimension)
        steps = int(maximum_steps)
        tolerance_ = float(tolerance)
        identifier = str(plan_id)
        if (
            dimension_ < 1
            or dimension_ > 64
            or steps <= 0
            or not np.isfinite(tolerance_)
            or tolerance_ <= 0.0
            or not identifier
        ):
            raise ValueError("Vector local root configuration is invalid.")
        self.dimension = dimension_
        self.maximum_steps = steps
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "vector-local-root",
                "dimension": dimension_,
                "maximum_steps": steps,
                "tolerance": tolerance_,
                "declared_id": identifier,
            }
        )

    def _solve_linear(self, matrix, right_hand_side):
        if self.dimension <= 3:
            result = solve_small_linear(
                SmallLinearSolvePlan(self.dimension),
                matrix,
                right_hand_side,
            )
            return result.value, result
        result = solve(
            LinearSystem(DenseLinearOperator(matrix)),
            right_hand_side,
        )
        return result.value, result

    def _newton(self, residual, guess):
        def body(_, value):
            current = residual(value)
            jacobian = jax.jacfwd(residual)(value)
            delta, solve = self._solve_linear(jacobian, current)
            candidate = value - delta
            converged = jnp.linalg.norm(current) <= self.tolerance
            return jnp.where(converged | ~solve.successful, value, candidate)

        return jax.lax.fori_loop(0, self.maximum_steps, body, guess)

    def solve(self, residual: Callable[[Array], Array], initial: ArrayLike, /):
        if not callable(residual):
            raise TypeError("residual must be callable.")
        initial_ = jnp.asarray(initial)
        if initial_.shape != (self.dimension,):
            raise ValueError(
                f"Vector local root initial state must have shape {(self.dimension,)}."
            )

        def solve_fn(function, guess):
            return self._newton(function, guess)

        def tangent_solve(linearize, right_hand_side):
            basis = jnp.eye(self.dimension, dtype=right_hand_side.dtype)
            matrix = jax.vmap(linearize)(basis).T
            value, _ = self._solve_linear(matrix, right_hand_side)
            return value

        return jax.lax.custom_root(residual, initial_, solve_fn, tangent_solve)

    def solve_with_diagnostics(self, residual, initial, /):
        root = self.solve(residual, initial)
        value = residual(root)
        jacobian = jax.jacfwd(residual)(root)
        identity = jnp.eye(self.dimension, dtype=root.dtype)
        if self.dimension <= 3:
            inverse = solve_small_linear(
                SmallLinearSolvePlan(self.dimension), jacobian, identity
            )
            condition_estimate = inverse.condition_estimate
        else:
            inverse = solve(
                LinearSystem(DenseLinearOperator(jacobian)),
                identity,
            )
            condition_estimate = inverse.diagnostics.condition_estimate
        norm = jnp.linalg.norm(value)
        finite = (
            jnp.all(jnp.isfinite(root))
            & jnp.all(jnp.isfinite(value))
            & jnp.all(jnp.isfinite(jacobian))
        )
        converged = finite & jnp.all(inverse.successful) & (norm <= self.tolerance)
        return root, VectorLocalRootDiagnostics(
            converged,
            jnp.asarray(self.maximum_steps, dtype=jnp.int32),
            value,
            norm,
            jacobian,
            condition_estimate,
            finite,
        )


class LocalRootPlan(StrictModule, NonTrainableState):
    """Bounded scalar local root with implicit-function derivatives."""

    maximum_steps: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    minimum_derivative: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_steps: int = 25,
        tolerance: float = 1.0e-10,
        minimum_derivative: float = 1.0e-12,
        plan_id: str,
    ):
        steps = int(maximum_steps)
        tolerance_ = float(tolerance)
        derivative = float(minimum_derivative)
        identifier = str(plan_id)
        if (
            steps <= 0
            or not np.isfinite(tolerance_)
            or tolerance_ <= 0.0
            or not np.isfinite(derivative)
            or derivative <= 0.0
            or not identifier
        ):
            raise ValueError("Local constitutive root configuration is invalid.")
        self.maximum_steps = steps
        self.tolerance = tolerance_
        self.minimum_derivative = derivative
        self.plan_id = canonical_fingerprint(
            {
                "kind": "local-constitutive-root",
                "maximum_steps": steps,
                "tolerance": tolerance_,
                "minimum_derivative": derivative,
                "declared_id": identifier,
            }
        )

    def _newton(self, residual: Callable[[Array], Array], guess: Array) -> Array:
        def body(_, value):
            current = residual(value)
            derivative = jax.grad(residual)(value)
            safe = jnp.where(
                jnp.abs(derivative) >= self.minimum_derivative,
                derivative,
                jnp.copysign(self.minimum_derivative, derivative + 1.0e-30),
            )
            candidate = value - current / safe
            return jnp.where(jnp.abs(current) <= self.tolerance, value, candidate)

        return jax.lax.fori_loop(0, self.maximum_steps, body, guess)

    def solve(self, residual: Callable[[Array], Array], initial: ArrayLike, /) -> Array:
        if not callable(residual):
            raise TypeError("residual must be callable.")
        initial_ = jnp.asarray(initial)
        if initial_.shape != ():
            raise ValueError("Local constitutive roots are scalar.")

        def solve_fn(function, guess):
            return self._newton(function, guess)

        def tangent_solve(linearize, right_hand_side):
            derivative = jax.grad(linearize)(jnp.zeros_like(right_hand_side))
            safe = jnp.where(
                jnp.abs(derivative) >= self.minimum_derivative,
                derivative,
                jnp.copysign(self.minimum_derivative, derivative + 1.0e-30),
            )
            return right_hand_side / safe

        return jax.lax.custom_root(residual, initial_, solve_fn, tangent_solve)

    def solve_with_diagnostics(
        self, residual: Callable[[Array], Array], initial: ArrayLike, /
    ) -> tuple[Array, LocalRootDiagnostics]:
        root = self.solve(residual, initial)
        value = residual(root)
        derivative = jax.grad(residual)(root)
        finite = jnp.isfinite(root) & jnp.isfinite(value) & jnp.isfinite(derivative)
        converged = (
            finite
            & (jnp.abs(value) <= self.tolerance)
            & (jnp.abs(derivative) >= self.minimum_derivative)
        )
        return root, LocalRootDiagnostics(
            converged,
            jnp.asarray(self.maximum_steps, dtype=jnp.int32),
            value,
            derivative,
            finite,
        )


__all__ = [
    "LocalRootDiagnostics",
    "LocalRootPlan",
    "VectorLocalRootDiagnostics",
    "VectorLocalRootPlan",
]
