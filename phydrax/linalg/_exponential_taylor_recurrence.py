#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-capacity differentiable scaled Taylor action and degree selection."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ._exponential_taylor_planning import TaylorExponentialPlan
from ._generated_exponential_taylor_thresholds import TAYLOR_THRESHOLDS
from ._operators import AbstractLinearOperator


def select_taylor_degree(
    norm: Array,
    alpha_p: Array,
    plan: TaylorExponentialPlan,
    action_factor: int,
    /,
) -> tuple[Array, Array, Array]:
    """Choose minimum-cost theta-admissible (power, degree, scaling) candidate."""
    resource = plan.policy.resources
    tolerance_row = int(np.ceil(-np.log2(plan.policy.error_tolerance))) - 1
    theta = jnp.asarray(
        TAYLOR_THRESHOLDS[tolerance_row, : resource.max_degree], dtype=norm.dtype
    )
    degrees = jnp.arange(1, resource.max_degree + 1, dtype=jnp.int32)
    powers = jnp.arange(1, alpha_p.shape[0] + 2, dtype=jnp.int32)
    selector_norms = jnp.concatenate((norm[None], alpha_p))
    quotient = selector_norms[:, None] / theta[None, :]
    # Clamp before conversion: out-of-range float -> int32 conversion is undefined.
    scalings = jnp.maximum(
        1,
        jnp.ceil(jnp.minimum(quotient, resource.max_scaling_count + 1)).astype(jnp.int32),
    )
    work = action_factor * scalings * (degrees[None, :] + 1)
    minimum_degree = jnp.maximum(1, powers * (powers - 1) - 1)
    admissible = (
        jnp.isfinite(selector_norms[:, None])
        & (selector_norms[:, None] >= 0)
        & jnp.isfinite(theta[None, :])
        & (theta[None, :] > 0)
        & (degrees[None, :] >= minimum_degree[:, None])
        & (scalings <= resource.max_scaling_count)
        & (work <= resource.max_action_matvec_count)
    )
    costs = jnp.where(admissible, work, jnp.iinfo(jnp.int32).max)
    selected = jnp.argmin(costs.reshape((-1,)))
    return (
        degrees[selected % resource.max_degree],
        scalings.reshape((-1,))[selected],
        jnp.any(admissible),
    )


def _coordinate_action(operator: AbstractLinearOperator, coordinates: Array, /) -> Array:
    """Extend a real operator complex-linearly without invalid space casts."""
    space = operator.source
    native_dtype = jax.tree.leaves(space.structure())[0].dtype
    if jnp.issubdtype(coordinates.dtype, jnp.complexfloating) and not jnp.issubdtype(
        native_dtype, jnp.complexfloating
    ):
        real = operator.target.flatten(
            operator.mv(space.unflatten(jnp.real(coordinates).astype(native_dtype)))
        )
        imaginary = operator.target.flatten(
            operator.mv(space.unflatten(jnp.imag(coordinates).astype(native_dtype)))
        )
        return real + 1j * imaginary
    return operator.target.flatten(
        operator.mv(space.unflatten(coordinates.astype(native_dtype)))
    )


def taylor_recurrence(
    operator: AbstractLinearOperator,
    coordinates: Array,
    scale: Array,
    trace_shift: Array,
    degree: Array,
    scaling_count: Array,
    max_degree: int,
    max_scaling_count: int,
    /,
) -> tuple[Array, Array, Array]:
    """Full-degree recurrence even for zero primal RHS, with bounded loops.

    A shift is applied only inside actions; exp(scale*trace_shift) is applied
    once after all segments. The extra Taylor term is diagnostic, never used to
    terminate the recurrence, so a zero primal vector retains its exact JVP.
    """
    small_scale = scale / scaling_count.astype(scale.dtype)
    one = jnp.asarray(1.0, dtype=coordinates.real.dtype)
    norm_before = jnp.sum(jnp.abs(coordinates))

    def segment(
        index: Array, carry: tuple[Array, Array, Array]
    ) -> tuple[Array, Array, Array]:

        def calculate(state: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
            start, estimate, healthy = state

            def term_step(
                order: Array, partial: tuple[Array, Array, Array]
            ) -> tuple[Array, Array, Array]:

                def next_term(
                    values: tuple[Array, Array, Array],
                ) -> tuple[Array, Array, Array]:
                    prior, summation, _ = values
                    applied = _coordinate_action(operator, prior)
                    updated = (small_scale / order.astype(small_scale.dtype)) * (
                        applied - trace_shift * prior
                    )
                    summation = summation + jnp.where(
                        order <= degree, updated, jnp.zeros_like(updated)
                    )
                    return (
                        updated,
                        summation,
                        jnp.where(
                            order == degree + 1, jnp.sum(jnp.abs(updated)), one * 0
                        ),
                    )

                return jax.lax.cond(order <= degree + 1, next_term, lambda v: v, partial)

            term, result, tail = jax.lax.fori_loop(
                1,
                max_degree + 2,
                term_step,
                (start, start, jnp.asarray(0.0, dtype=one.dtype)),
            )
            del term
            return (
                result,
                estimate + tail,
                healthy & jnp.all(jnp.isfinite(result)) & jnp.isfinite(tail),
            )

        return jax.lax.cond(index < scaling_count, calculate, lambda state: state, carry)

    value, tail_sum, finite = jax.lax.fori_loop(
        0,
        max_scaling_count,
        segment,
        (coordinates, jnp.asarray(0.0, dtype=one.dtype), jnp.asarray(True)),
    )
    multiplier = jnp.exp(scale * trace_shift)
    value = multiplier * value
    tail_sum = jnp.abs(multiplier) * tail_sum
    finite = finite & jnp.all(jnp.isfinite(value)) & jnp.isfinite(norm_before)
    return value, tail_sum, finite
