#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._strict import StrictModule
from ...._trainable import NonTrainableState


class BoundedResidualAdaptationPolicy(StrictModule, NonTrainableState):
    """Fixed-weight test-time adaptation of one small context vector."""

    iterations: int
    learning_rate: float
    maximum_update_norm: float
    gradient_clip_norm: float

    def __init__(
        self,
        *,
        iterations: int = 16,
        learning_rate: float = 1.0e-2,
        maximum_update_norm: float = 1.0,
        gradient_clip_norm: float = 10.0,
    ):
        count = int(iterations)
        if count < 0:
            raise ValueError("iterations must be nonnegative.")
        values = tuple(
            float(value)
            for value in (learning_rate, maximum_update_norm, gradient_clip_norm)
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Adaptation scales must be finite and positive.")
        self.iterations = count
        self.learning_rate, self.maximum_update_norm, self.gradient_clip_norm = values


class TestTimeAdaptationResult(StrictModule):
    context: Array
    objective_history: Array
    initial_objective: Array
    final_objective: Array
    update_norm: Array
    accepted: Array


def adapt_operator_context(
    initial_context: ArrayLike,
    residual_objective: Callable[[Array], Array],
    /,
    *,
    policy: BoundedResidualAdaptationPolicy | None = None,
    lower_bound: ArrayLike | None = None,
    upper_bound: ArrayLike | None = None,
    jit: bool = True,
) -> TestTimeAdaptationResult:
    """Adapt only a bounded context; model weights remain captured and frozen."""

    if not callable(residual_objective):
        raise TypeError("residual_objective must be callable.")
    resolved = BoundedResidualAdaptationPolicy() if policy is None else policy
    if not isinstance(resolved, BoundedResidualAdaptationPolicy):
        raise TypeError("policy must be BoundedResidualAdaptationPolicy or None.")
    initial = jnp.asarray(initial_context, dtype=float)
    if initial.size == 0 or bool(jnp.any(~jnp.isfinite(initial))):
        raise ValueError("initial_context must be finite and non-empty.")
    lower = (
        None
        if lower_bound is None
        else jnp.broadcast_to(jnp.asarray(lower_bound), initial.shape)
    )
    upper = (
        None
        if upper_bound is None
        else jnp.broadcast_to(jnp.asarray(upper_bound), initial.shape)
    )
    if lower is not None and bool(jnp.any(~jnp.isfinite(lower))):
        raise ValueError("lower_bound must be finite.")
    if upper is not None and bool(jnp.any(~jnp.isfinite(upper))):
        raise ValueError("upper_bound must be finite.")
    if lower is not None and upper is not None and bool(jnp.any(lower > upper)):
        raise ValueError("lower_bound cannot exceed upper_bound.")

    def checked_objective(context):
        value = jnp.asarray(residual_objective(context))
        if value.shape != () or jnp.iscomplexobj(value):
            raise ValueError("residual_objective must return one real scalar.")
        return eqx.error_if(
            value, ~jnp.isfinite(value), "Residual objective is nonfinite."
        )

    value_and_grad = jax.value_and_grad(checked_objective)

    def step(context):
        value, gradient = value_and_grad(context)
        gradient_norm = jnp.sqrt(jnp.sum(gradient * gradient))
        scale = jnp.minimum(
            1.0, resolved.gradient_clip_norm / jnp.maximum(gradient_norm, 1.0e-30)
        )
        candidate = context - resolved.learning_rate * scale * gradient
        displacement = candidate - initial
        displacement_norm = jnp.sqrt(jnp.sum(displacement * displacement))
        radius_scale = jnp.minimum(
            1.0,
            resolved.maximum_update_norm / jnp.maximum(displacement_norm, 1.0e-30),
        )
        candidate = initial + radius_scale * displacement
        if lower is not None:
            candidate = jnp.maximum(candidate, lower)
        if upper is not None:
            candidate = jnp.minimum(candidate, upper)
        return candidate, value

    run_step = eqx.filter_jit(step) if jit else step
    initial_objective = checked_objective(initial)
    current = initial
    best = initial
    best_objective = initial_objective
    history = [initial_objective]
    for _ in range(resolved.iterations):
        candidate, _ = run_step(current)
        candidate_objective = checked_objective(candidate)
        improved = candidate_objective < best_objective
        best = jnp.where(improved, candidate, best)
        best_objective = jnp.where(improved, candidate_objective, best_objective)
        current = candidate
        history.append(candidate_objective)
    update = best - initial
    update_norm = jnp.sqrt(jnp.sum(update * update))
    accepted = best_objective <= initial_objective
    return TestTimeAdaptationResult(
        context=jnp.where(accepted, best, initial),
        objective_history=jnp.stack(history),
        initial_objective=initial_objective,
        final_objective=jnp.where(accepted, best_objective, initial_objective),
        update_norm=jnp.where(accepted, update_norm, 0.0),
        accepted=accepted,
    )


__all__ = [
    "BoundedResidualAdaptationPolicy",
    "TestTimeAdaptationResult",
    "adapt_operator_context",
]
