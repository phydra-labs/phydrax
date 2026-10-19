#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, overload

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._custom_root import custom_root
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    DenseLinearOperator,
    factorize,
    LinearSolveResult,
    LinearSystem,
    SmallLinearSolvePlan,
    SmallLinearSolveResult,
    solve,
    solve_small_linear,
)
from ._domain_cond import domain_cond
from ._newton import RootLineSearch


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
    linear_solve_status: Array
    linear_solve_rank: Array
    conditioning_available: Array


class VectorLocalRootPlan(StrictModule, NonTrainableState):
    """Bounded one-to-64 component root with implicit-function derivatives.

    ``solve_with_diagnostics`` reports accepted Newton updates, excluding
    iterations after convergence and unsuccessful or refused trial steps.
    """

    dimension: int = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    line_search: RootLineSearch
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        /,
        *,
        maximum_steps: int = 30,
        tolerance: float = 1.0e-10,
        plan_id: str,
        line_search: RootLineSearch | None = None,
    ) -> None:
        dimension_ = int(dimension)
        steps = int(maximum_steps)
        tolerance_ = float(tolerance)
        identifier = str(plan_id)
        search = RootLineSearch() if line_search is None else line_search
        if not isinstance(search, RootLineSearch):
            raise TypeError("line_search must be a RootLineSearch.")
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
        self.line_search = search
        self.plan_id = canonical_fingerprint(
            {
                "kind": "vector-local-root",
                "dimension": dimension_,
                "maximum_steps": steps,
                "tolerance": tolerance_,
                "declared_id": identifier,
                "line_search": {
                    "initial_rate": search.initial_rate,
                    "contraction": search.contraction,
                    "sufficient_decrease": search.sufficient_decrease,
                    "minimum_rate": search.minimum_rate,
                    "maximum_steps": search.maximum_steps,
                },
            }
        )

    def _solve_linear(
        self, matrix: Array, right_hand_side: Array
    ) -> tuple[Array, SmallLinearSolveResult | LinearSolveResult]:
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

    def _backtracking_step(
        self,
        residual: Callable[[Array], Array],
        admissible: Callable[[Array], Array],
        value: Array,
        direction: Array,
        merit: Array,
        slope: Array,
    ) -> Array:
        policy = self.line_search
        initial = (
            jnp.asarray(policy.initial_rate, dtype=merit.dtype),
            value,
            jnp.asarray(False, dtype=jnp.bool_),
        )

        def body(
            _: int | Array, carry: tuple[Array, Array, Array]
        ) -> tuple[Array, Array, Array]:
            rate, accepted_value, finished = carry

            def unchanged(_: None) -> tuple[Array, Array, Array]:
                return carry

            def attempt(_: None) -> tuple[Array, Array, Array]:
                candidate = value + rate * direction
                feasible = jnp.all(jnp.isfinite(candidate)) & admissible(candidate)

                def evaluate(_: None) -> Array:
                    result = residual(candidate)
                    return 0.5 * jnp.sum(jnp.abs(result) ** 2)

                def refused(_: None) -> Array:
                    return jnp.asarray(jnp.inf, dtype=merit.dtype)

                candidate_merit = domain_cond(feasible, evaluate, refused)
                accepted = jnp.isfinite(candidate_merit) & (
                    candidate_merit <= merit + policy.sufficient_decrease * rate * slope
                )
                next_rate = jnp.maximum(rate * policy.contraction, policy.minimum_rate)
                return (
                    next_rate,
                    jnp.where(accepted, candidate, accepted_value),
                    accepted | (rate <= policy.minimum_rate),
                )

            return domain_cond(~finished, attempt, unchanged)

        _, accepted_value, _ = jax.lax.fori_loop(0, policy.maximum_steps, body, initial)
        return accepted_value

    def _newton(
        self,
        residual: Callable[[Array], Array],
        guess: Array,
        admissible: Callable[[Array], Array] | None,
    ) -> tuple[Array, Array]:
        zero = jnp.asarray(0, dtype=jnp.int32)

        def body(_: int | Array, carry: tuple[Array, Array]) -> tuple[Array, Array]:
            value, iterations = carry
            current = residual(value)
            jacobian = jax.jacfwd(residual)(value)
            delta, solve_result = self._solve_linear(jacobian, current)
            candidate = value - delta
            converged = jnp.linalg.norm(current) <= self.tolerance
            update = ~converged & solve_result.successful
            return (
                jnp.where(update, candidate, value),
                iterations + update.astype(jnp.int32),
            )

        if admissible is None:
            return jax.lax.fori_loop(
                0,
                self.maximum_steps,
                body,
                (guess, zero),
            )

        def guarded_body(
            _: int | Array, carry: tuple[Array, Array]
        ) -> tuple[Array, Array]:
            value, iterations = carry

            def unchanged(_: None) -> tuple[Array, Array]:
                return carry

            def evaluate(_: None) -> tuple[Array, Array]:
                current = residual(value)
                merit = 0.5 * jnp.sum(jnp.abs(current) ** 2)

                def update(_: None) -> tuple[Array, Array]:
                    jacobian = jax.jacfwd(residual)(value)
                    delta, evidence = self._solve_linear(jacobian, current)
                    direction = -delta
                    slope = jnp.real(jnp.vdot(current, jacobian @ direction))

                    def globalize(_: None) -> Array:
                        trial = self._backtracking_step(
                            residual, admissible, value, direction, merit, slope
                        )

                        def accepted(_: None) -> Array:
                            return trial

                        def steepest_descent(_: None) -> Array:
                            # A Newton direction can point out of an active
                            # physical face even when the root is feasible.
                            # Use another genuine residual descent search,
                            # never project/clip the candidate into validity.
                            gradient = jnp.conj(jacobian).T @ current
                            descent = -gradient
                            descent_slope = -jnp.real(jnp.vdot(gradient, gradient))
                            return self._backtracking_step(
                                residual,
                                admissible,
                                value,
                                descent,
                                merit,
                                descent_slope,
                            )

                        return domain_cond(
                            jnp.any(trial != value), accepted, steepest_descent
                        )

                    trial = domain_cond(
                        jnp.all(evidence.successful)
                        & jnp.all(jnp.isfinite(delta))
                        & jnp.isfinite(slope)
                        & (slope < 0.0),
                        globalize,
                        lambda _: value,
                    )
                    accepted_update = jnp.any(trial != value)
                    return (
                        trial,
                        iterations + accepted_update.astype(jnp.int32),
                    )

                return domain_cond(
                    jnp.all(jnp.isfinite(current)) & (2.0 * merit > self.tolerance**2),
                    update,
                    unchanged,
                )

            return domain_cond(
                jnp.all(jnp.isfinite(value)) & admissible(value),
                evaluate,
                unchanged,
            )

        return jax.lax.fori_loop(
            0,
            self.maximum_steps,
            guarded_body,
            (guess, zero),
        )

    def _solve(
        self,
        residual: Callable[[Array], Array],
        initial: ArrayLike,
        /,
        *,
        admissible: Callable[[Array], Array] | None = None,
    ) -> tuple[Array, Array]:
        if not callable(residual):
            raise TypeError("residual must be callable.")
        if admissible is not None and not callable(admissible):
            raise TypeError("admissible must be callable.")
        initial_ = jnp.asarray(initial)
        if initial_.shape != (self.dimension,):
            raise ValueError(
                f"Vector local root initial state must have shape {(self.dimension,)}."
            )
        if not jnp.issubdtype(initial_.dtype, jnp.inexact):
            raise TypeError("Vector local root initial state must have an inexact dtype.")

        def solve_fn(
            function: Callable[[Array], Array], guess: Array
        ) -> tuple[Array, Array]:
            return self._newton(function, guess, admissible)

        def tangent_solve(
            linearize: Callable[[Array], Array], right_hand_side: Array
        ) -> Array:
            basis = jnp.eye(self.dimension, dtype=right_hand_side.dtype)
            matrix = jax.vmap(linearize)(basis).T

            def solve_tangent(operator: Array, right: Array) -> Array:
                value, evidence = self._solve_linear(operator, right)
                # An undefined derivative is a nonfinite numeric trial, not a
                # host exception inside the enclosing nonlinear iteration.
                # Root diagnostics retain the native matrix-failure evidence.
                # Linearity fixes the image of a zero right-hand side at zero
                # for every operator, so a masked or inactive root contributes
                # an exact zero rather than an undefined derivative.
                undefined = jnp.where(
                    jnp.all(right == 0),
                    jnp.zeros_like(value),
                    jnp.full_like(value, jnp.nan),
                )
                return jnp.where(jnp.all(evidence.successful), value, undefined)

            # Native solve evidence may depend on the RHS. The IFT action is
            # linear nonetheless; explicitly own both actions rather than
            # transposing diagnostic predicates.
            return jax.lax.custom_linear_solve(
                linearize,
                right_hand_side,
                solve=lambda _, right: solve_tangent(matrix, right),
                transpose_solve=lambda _, right: solve_tangent(matrix.T, right),
            )

        return custom_root(
            residual,
            initial_,
            solve_fn,
            tangent_solve,
            has_aux=True,
        )

    def solve(
        self,
        residual: Callable[[Array], Array],
        initial: ArrayLike,
        /,
        *,
        admissible: Callable[[Array], Array] | None = None,
    ) -> Array:
        return self._solve(residual, initial, admissible=admissible)[0]

    @overload
    def solve_with_diagnostics(
        self,
        residual: Callable[[Array], Array],
        initial: ArrayLike,
        /,
        *,
        admissible: Callable[[Array], Array] | None = None,
        has_aux: Literal[False] = False,
    ) -> tuple[Array, VectorLocalRootDiagnostics]: ...

    @overload
    def solve_with_diagnostics[T](
        self,
        residual: Callable[[Array], tuple[Array, T]],
        initial: ArrayLike,
        /,
        *,
        admissible: Callable[[Array], Array] | None = None,
        has_aux: Literal[True],
    ) -> tuple[Array, VectorLocalRootDiagnostics, T]: ...

    def solve_with_diagnostics(
        self,
        residual: Callable[[Array], Any],
        initial: ArrayLike,
        /,
        *,
        admissible: Callable[[Array], Array] | None = None,
        has_aux: bool = False,
    ) -> (
        tuple[Array, VectorLocalRootDiagnostics]
        | tuple[Array, VectorLocalRootDiagnostics, Any]
    ):
        if has_aux:

            def evaluate(value: Array) -> tuple[Array, Any]:
                output = residual(value)
                if not isinstance(output, tuple) or len(output) != 2:
                    raise TypeError(
                        "has_aux=True requires residual to return a (residual, auxiliary) tuple."
                    )
                return output

            def residual_only(value: Array) -> Array:
                return evaluate(value)[0]

            root, iterations = self._solve(
                residual_only,
                initial,
                admissible=admissible,
            )
            value, auxiliary = evaluate(root)
            jacobian = jax.jacfwd(residual_only)(root)
        else:
            root, iterations = self._solve(
                residual,
                initial,
                admissible=admissible,
            )
            value = residual(root)
            jacobian = jax.jacfwd(residual)(root)
        if self.dimension <= 3:
            identity = jnp.eye(self.dimension, dtype=root.dtype)
            inverse = solve_small_linear(
                SmallLinearSolvePlan(self.dimension), jacobian, identity
            )
            condition_estimate = inverse.condition_estimate
            status = inverse.status
            rank = inverse.rank
        else:
            inverse = factorize(DenseLinearOperator(jacobian)).materialize_inverse()
            condition_estimate = inverse.diagnostics.condition_estimate
            status = inverse.status
            rank = inverse.diagnostics.rank
        norm = jnp.linalg.norm(value)
        finite = (
            jnp.all(jnp.isfinite(root))
            & jnp.all(jnp.isfinite(value))
            & jnp.all(jnp.isfinite(jacobian))
        )
        conditioning_available = jnp.all(inverse.successful) & jnp.isfinite(
            condition_estimate
        )
        converged = finite & jnp.all(inverse.successful) & (norm <= self.tolerance)
        if admissible is not None:
            converged = converged & admissible(root)
        diagnostics = VectorLocalRootDiagnostics(
            converged,
            iterations,
            value,
            norm,
            jacobian,
            condition_estimate,
            finite,
            status,
            rank,
            conditioning_available,
        )
        if has_aux:
            return root, diagnostics, auxiliary
        return root, diagnostics


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
    ) -> None:
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
        def body(_: int | Array, value: Array) -> Array:
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
        if not jnp.issubdtype(initial_.dtype, jnp.inexact):
            raise TypeError(
                "Local constitutive root initial state must have an inexact dtype."
            )

        def solve_fn(function: Callable[[Array], Array], guess: Array) -> Array:
            return self._newton(function, guess)

        def tangent_solve(
            linearize: Callable[[Array], Array], right_hand_side: Array
        ) -> Array:
            derivative = jax.grad(linearize)(jnp.zeros_like(right_hand_side))
            valid = jnp.isfinite(derivative) & (
                jnp.abs(derivative) >= self.minimum_derivative
            )
            safe = jnp.where(valid, derivative, 1.0)

            def solve_linear(_: Callable[[Array], Array], right: Array) -> Array:
                # A refused derivative maps a zero right-hand side to
                # zero (safe is one there) and every other one to NaN.
                value = right / safe
                return jnp.where(
                    valid | (right == 0), value, jnp.full_like(value, jnp.nan)
                )

            return jax.lax.custom_linear_solve(
                linearize,
                right_hand_side,
                solve=solve_linear,
                transpose_solve=solve_linear,
            )

        return custom_root(residual, initial_, solve_fn, tangent_solve)

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
