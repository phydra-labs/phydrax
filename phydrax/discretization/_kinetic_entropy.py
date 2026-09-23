#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


KineticEntropyRootStrategy = Literal["exact", "asymptotic", "hybrid"]

KINETIC_ENTROPY_SUCCESS = 0
KINETIC_ENTROPY_NONFINITE = 1
KINETIC_ENTROPY_NONPOSITIVE = 2
KINETIC_ENTROPY_INACTIVE = 3
KINETIC_ENTROPY_NO_BRACKET = 4
KINETIC_ENTROPY_NONCONVERGED = 5


class KineticEntropyRootPlan(StrictModule, NonTrainableState):
    """Safeguarded nontrivial entropy root for one population direction."""

    strategy: KineticEntropyRootStrategy = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    approximation_tolerance: float = eqx.field(static=True)
    positivity_margin: float = eqx.field(static=True)
    minimum_root: float = eqx.field(static=True)
    maximum_root: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        strategy: KineticEntropyRootStrategy = "exact",
        maximum_steps: int = 24,
        residual_tolerance: float = 1.0e-11,
        approximation_tolerance: float = 1.0e-7,
        positivity_margin: float = 1.0e-7,
        minimum_root: float = 1.0,
        maximum_root: float = 4.0,
    ):
        if strategy not in ("exact", "asymptotic", "hybrid"):
            raise ValueError(f"Unknown kinetic entropy root strategy {strategy!r}.")
        steps = int(maximum_steps)
        residual = float(residual_tolerance)
        approximation = float(approximation_tolerance)
        margin = float(positivity_margin)
        lower = float(minimum_root)
        upper = float(maximum_root)
        if steps < 2:
            raise ValueError("maximum_steps must be at least two.")
        if any(
            not np.isfinite(value) or value <= 0.0
            for value in (residual, approximation, margin, lower, upper)
        ):
            raise ValueError("Entropy-root controls must be finite and positive.")
        if margin >= 1.0 or lower >= upper:
            raise ValueError("Entropy-root interval or positivity margin is invalid.")
        self.strategy = strategy
        self.maximum_steps = steps
        self.residual_tolerance = residual
        self.approximation_tolerance = approximation
        self.positivity_margin = margin
        self.minimum_root = lower
        self.maximum_root = upper
        self.plan_id = canonical_fingerprint(
            {
                "kind": "kinetic-entropy-root",
                "strategy": strategy,
                "maximum_steps": steps,
                "residual_tolerance": residual,
                "approximation_tolerance": approximation,
                "positivity_margin": margin,
                "minimum_root": lower,
                "maximum_root": upper,
            }
        )


class KineticEntropyRootEvidence(StrictModule):
    alpha: Array
    positivity_upper_bound: Array
    residual: Array
    iterations: Array
    newton_steps: Array
    bisection_steps: Array
    used_approximation: Array
    minimum_population: Array
    bracketed: Array
    status: Array
    successful: Array


class KineticEntropyRootResult(StrictModule):
    mirror_populations: Array
    evidence: KineticEntropyRootEvidence


def _entropy(populations: Array, base_measure: Array) -> Array:
    safe = jnp.where(populations > 0.0, populations, 1.0)
    return jnp.sum(
        jnp.where(
            populations > 0.0,
            populations * (jnp.log(safe / base_measure) - 1.0),
            jnp.inf,
        ),
        axis=-1,
    )


def solve_kinetic_entropy_root(
    plan: KineticEntropyRootPlan,
    populations: ArrayLike,
    direction: ArrayLike,
    /,
    *,
    base_measure: ArrayLike | None = None,
    initial: ArrayLike | None = None,
) -> KineticEntropyRootResult:
    """Solve the nontrivial constant-entropy root with explicit route evidence."""
    if not isinstance(plan, KineticEntropyRootPlan):
        raise TypeError("plan must be a KineticEntropyRootPlan.")
    values = jnp.asarray(populations)
    delta = jnp.asarray(direction, dtype=values.dtype)
    if values.ndim < 1 or delta.shape != values.shape:
        raise ValueError("populations and direction must have equal non-scalar shape.")
    q = values.shape[-1]
    base = (
        jnp.ones((q,), dtype=values.dtype)
        if base_measure is None
        else jnp.asarray(base_measure, dtype=values.dtype)
    )
    if base.shape != (q,):
        raise ValueError("base_measure must have shape (population_count,).")
    if initial is None:
        initial_alpha = jnp.full(values.shape[:-1], 2.0, dtype=values.dtype)
    else:
        initial_alpha = jnp.broadcast_to(
            jnp.asarray(initial, dtype=values.dtype), values.shape[:-1]
        )

    finite = (
        jnp.all(jnp.isfinite(values), axis=-1)
        & jnp.all(jnp.isfinite(delta), axis=-1)
        & jnp.all(jnp.isfinite(base) & (base > 0.0))
        & jnp.isfinite(initial_alpha)
    )
    positive = jnp.all(values > 0.0, axis=-1)
    direction_scale = jnp.max(jnp.abs(delta), axis=-1)
    inactive = direction_scale <= (
        32.0 * jnp.finfo(values.dtype).eps * jnp.maximum(jnp.max(values, axis=-1), 1.0)
    )
    ratios = jnp.where(delta < 0.0, -values / delta, jnp.inf)
    positivity_upper = jnp.minimum(
        jnp.min(ratios, axis=-1) * (1.0 - plan.positivity_margin),
        plan.maximum_root,
    )
    entropy_initial = _entropy(values, base)

    def residual_at(alpha: Array) -> Array:
        candidate = values + alpha[..., None] * delta
        return _entropy(candidate, base) - entropy_initial

    def derivative_at(alpha: Array) -> Array:
        candidate = values + alpha[..., None] * delta
        safe = jnp.where(candidate > 0.0, candidate, 1.0)
        return jnp.sum(delta * jnp.log(safe / base), axis=-1)

    linear = jnp.sum(
        delta * jnp.log(jnp.where(values > 0.0, values, 1.0) / base), axis=-1
    )
    quadratic = jnp.sum(delta * delta / jnp.where(values > 0.0, values, 1.0), axis=-1)
    approximate_alpha = jnp.where(quadratic > 0.0, -2.0 * linear / quadratic, 2.0)
    approximate_alpha = jnp.minimum(
        jnp.maximum(approximate_alpha, plan.minimum_root), positivity_upper
    )
    approximate_residual = jnp.abs(residual_at(approximate_alpha))
    approximate_valid = (
        finite
        & positive
        & ~inactive
        & (positivity_upper > plan.minimum_root)
        & jnp.isfinite(approximate_alpha)
        & jnp.isfinite(approximate_residual)
        & (approximate_residual <= plan.approximation_tolerance)
    )
    use_approximation = approximate_valid & (plan.strategy != "exact")

    lower = jnp.full(values.shape[:-1], plan.minimum_root, dtype=values.dtype)
    upper = positivity_upper
    lower_residual = residual_at(lower)
    upper_residual = residual_at(upper)
    bracketed = (
        finite
        & positive
        & ~inactive
        & (upper > lower)
        & jnp.isfinite(lower_residual)
        & jnp.isfinite(upper_residual)
        & (lower_residual <= 0.0)
        & (upper_residual >= 0.0)
    )
    exact_active = bracketed & ~use_approximation & (plan.strategy != "asymptotic")
    alpha = jnp.minimum(jnp.maximum(initial_alpha, lower), upper)
    newton_count = jnp.zeros(values.shape[:-1], dtype=jnp.int32)
    bisection_count = jnp.zeros(values.shape[:-1], dtype=jnp.int32)
    iteration_count = jnp.zeros(values.shape[:-1], dtype=jnp.int32)

    def body(_, state):
        (
            current,
            lo,
            hi,
            active,
            newton_steps,
            bisection_steps,
            iterations,
        ) = state
        residual = residual_at(current)
        derivative = derivative_at(current)
        next_lo = jnp.where(active & (residual <= 0.0), current, lo)
        next_hi = jnp.where(active & (residual > 0.0), current, hi)
        newton = current - residual / jnp.where(derivative != 0.0, derivative, 1.0)
        newton_usable = (
            active
            & jnp.isfinite(newton)
            & jnp.isfinite(derivative)
            & (derivative != 0.0)
            & (newton > next_lo)
            & (newton < next_hi)
        )
        midpoint = 0.5 * (next_lo + next_hi)
        candidate = jnp.where(newton_usable, newton, midpoint)
        candidate_residual = jnp.abs(residual_at(candidate))
        converged = active & (candidate_residual <= plan.residual_tolerance)
        next_active = active & ~converged
        return (
            jnp.where(active, candidate, current),
            next_lo,
            next_hi,
            next_active,
            newton_steps + newton_usable.astype(jnp.int32),
            bisection_steps + (active & ~newton_usable).astype(jnp.int32),
            iterations + active.astype(jnp.int32),
        )

    (
        exact_alpha,
        _,
        _,
        exact_active,
        newton_count,
        bisection_count,
        iteration_count,
    ) = jax.lax.fori_loop(
        0,
        plan.maximum_steps,
        body,
        (
            alpha,
            lower,
            upper,
            exact_active,
            newton_count,
            bisection_count,
            iteration_count,
        ),
    )
    del exact_active
    selected_alpha = jnp.where(use_approximation, approximate_alpha, exact_alpha)
    selected_alpha = jnp.where(inactive & finite & positive, 2.0, selected_alpha)
    residual = jnp.abs(residual_at(selected_alpha))
    exact_success = (
        bracketed
        & (plan.strategy != "asymptotic")
        & (residual <= plan.residual_tolerance)
    )
    root_success = jnp.where(use_approximation, approximate_valid, exact_success)
    successful = (finite & positive & inactive) | root_success
    mirror = values + selected_alpha[..., None] * delta
    minimum_population = jnp.min(mirror, axis=-1)
    successful &= jnp.all(jnp.isfinite(mirror), axis=-1) & (minimum_population > 0.0)
    status = jnp.where(
        ~finite,
        KINETIC_ENTROPY_NONFINITE,
        jnp.where(
            ~positive,
            KINETIC_ENTROPY_NONPOSITIVE,
            jnp.where(
                inactive,
                KINETIC_ENTROPY_INACTIVE,
                jnp.where(
                    ~bracketed & ~use_approximation,
                    KINETIC_ENTROPY_NO_BRACKET,
                    jnp.where(
                        successful,
                        KINETIC_ENTROPY_SUCCESS,
                        KINETIC_ENTROPY_NONCONVERGED,
                    ),
                ),
            ),
        ),
    )
    mirror = jnp.where(successful[..., None], mirror, values)
    return KineticEntropyRootResult(
        mirror_populations=mirror,
        evidence=KineticEntropyRootEvidence(
            alpha=selected_alpha,
            positivity_upper_bound=positivity_upper,
            residual=jnp.where(inactive, 0.0, residual),
            iterations=iteration_count,
            newton_steps=newton_count,
            bisection_steps=bisection_count,
            used_approximation=use_approximation,
            minimum_population=minimum_population,
            bracketed=bracketed | inactive,
            status=jnp.asarray(status, dtype=jnp.int32),
            successful=successful,
        ),
    )


__all__ = [
    "KINETIC_ENTROPY_INACTIVE",
    "KINETIC_ENTROPY_NONCONVERGED",
    "KINETIC_ENTROPY_NO_BRACKET",
    "KINETIC_ENTROPY_NONFINITE",
    "KINETIC_ENTROPY_NONPOSITIVE",
    "KINETIC_ENTROPY_SUCCESS",
    "KineticEntropyRootEvidence",
    "KineticEntropyRootPlan",
    "KineticEntropyRootResult",
    "KineticEntropyRootStrategy",
    "solve_kinetic_entropy_root",
]
