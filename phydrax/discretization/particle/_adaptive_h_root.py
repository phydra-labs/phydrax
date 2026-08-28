#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class AdaptiveHRootPlan(StrictModule, NonTrainableState):
    eta: float = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    minimum_h: float = eqx.field(static=True)
    maximum_h: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        eta: float,
        dimension: int,
        minimum_h: float,
        maximum_h: float,
        /,
        *,
        tolerance: float = 1e-10,
        maximum_iterations: int = 30,
    ):
        if eta <= 0.0 or dimension <= 0 or minimum_h <= 0.0 or maximum_h < minimum_h:
            raise ValueError("Adaptive-h root parameters are invalid.")
        if tolerance <= 0.0 or maximum_iterations <= 0:
            raise ValueError("Adaptive-h root solve controls are invalid.")
        self.eta = float(eta)
        self.dimension = int(dimension)
        self.minimum_h = float(minimum_h)
        self.maximum_h = float(maximum_h)
        self.tolerance = float(tolerance)
        self.maximum_iterations = int(maximum_iterations)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "adaptive-h-root-plan",
                "eta": eta,
                "dimension": dimension,
                "minimum_h": minimum_h,
                "maximum_h": maximum_h,
                "tolerance": tolerance,
                "maximum_iterations": maximum_iterations,
            }
        )


class AdaptiveHRootResult(StrictModule):
    smoothing_length: Array
    density: Array
    residual: Array
    derivative: Array
    bracket_width: Array
    iterations: Array
    function_evaluations: Array
    derivative_evaluations: Array
    bound_active: Array
    converged: Array
    successful: Array


def solve_adaptive_h_root(
    plan: AdaptiveHRootPlan,
    mass: ArrayLike,
    density_function: Callable[[Array], Array],
    initial_h: ArrayLike,
    /,
) -> AdaptiveHRootResult:
    mass_ = jnp.asarray(mass)
    initial = jnp.clip(jnp.asarray(initial_h), plan.minimum_h, plan.maximum_h)
    lower = jnp.full(initial.shape, plan.minimum_h, initial.dtype)
    upper = jnp.full(initial.shape, plan.maximum_h, initial.dtype)

    def residual(h):
        density = density_function(h)
        target = plan.eta * (mass_ / density) ** (1.0 / plan.dimension)
        return h - target, density

    def body(_, carry):
        h, lo, hi, _, function_count, derivative_count = carry
        value, _ = residual(h)
        derivative = jax.jvp(
            lambda current: residual(current)[0], (h,), (jnp.ones_like(h),)
        )[1]
        converged = jnp.abs(value) <= plan.tolerance
        newton = h - value / jnp.where(jnp.abs(derivative) > 1e-14, derivative, 1.0)
        bisect = 0.5 * (lo + hi)
        use_newton = (newton >= lo) & (newton <= hi) & jnp.isfinite(newton)
        proposal = jnp.where(use_newton, newton, bisect)
        candidate = jnp.where(converged, h, proposal)
        candidate_value, _ = residual(candidate)
        lo = jnp.where(~converged & (candidate_value < 0.0), candidate, lo)
        hi = jnp.where(~converged & (candidate_value >= 0.0), candidate, hi)
        return (
            candidate,
            lo,
            hi,
            jnp.max(jnp.abs(candidate_value)),
            function_count + 2,
            derivative_count + 1,
        )

    h, lower, upper, maximum_residual, function_count, derivative_count = (
        jax.lax.fori_loop(
            0,
            plan.maximum_iterations,
            body,
            (
                initial,
                lower,
                upper,
                jnp.asarray(jnp.inf, initial.dtype),
                jnp.asarray(0, jnp.int32),
                jnp.asarray(0, jnp.int32),
            ),
        )
    )
    value, density = residual(h)
    derivative = jax.jvp(lambda current: residual(current)[0], (h,), (jnp.ones_like(h),))[
        1
    ]
    residual_max = jnp.max(jnp.abs(value))
    converged = residual_max <= plan.tolerance
    finite = jnp.all(jnp.isfinite(h) & jnp.isfinite(density) & jnp.isfinite(derivative))
    bound_active = (h <= plan.minimum_h) | (h >= plan.maximum_h)
    return AdaptiveHRootResult(
        h,
        density,
        residual_max,
        derivative,
        jnp.max(upper - lower),
        jnp.asarray(plan.maximum_iterations, jnp.int32),
        function_count + 1,
        derivative_count + 1,
        bound_active,
        converged,
        converged & finite,
    )


def adaptive_h_implicit_tangent(
    result: AdaptiveHRootResult,
    parameter_residual_derivative: ArrayLike,
    /,
) -> Array:
    """Implicit h tangent for a converged root with a frozen bound active set."""

    derivative = jnp.asarray(parameter_residual_derivative)
    tangent = -derivative / jnp.where(
        jnp.abs(result.derivative) > 1e-14, result.derivative, 1.0
    )
    return jnp.where(result.bound_active | ~result.successful, 0.0, tangent)


__all__ = [
    "AdaptiveHRootPlan",
    "AdaptiveHRootResult",
    "adaptive_h_implicit_tangent",
    "solve_adaptive_h_root",
]
