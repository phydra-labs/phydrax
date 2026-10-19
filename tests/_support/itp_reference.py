#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Frozen pre-cutover edge ITP arithmetic for bitwise differential evidence.

The reference is the 51-step, unit-interval kernel read before native ownership
migration. It is test-only and pins callback evaluation, branch, and reduction
order; independent closed-form tests separately validate the native method.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from phydrax.geometry._contracts import GeometryKernel
from phydrax.geometry.design._schema import DesignState


_ITP_BRACKET_TOLERANCE = 2.0**-50
_ITP_MAXIMUM_ITERATIONS = 51
_ITP_K1 = 0.2
_ITP_K2 = 2.0
type _ReferenceCarry = tuple[Array, Array, Array, Array, Array, Array, Array]


@eqx.filter_jit
def reference_edge_roots(
    kernel: GeometryKernel,
    state: DesignState,
    lower_points: Array,
    upper_points: Array,
    lower_values: Array,
    upper_values: Array,
    tolerance: Array,
) -> tuple[Array, Array]:
    """Bracketed ITP root isolation over every lattice crossing in one program.

    Each lane solves f(lower + t (upper - lower)) = 0 on t in [0, 1]. ITP keeps
    bisection's worst-case iteration bound while converging superlinearly on
    smooth fields. Lanes freeze once |f| <= tolerance.
    """

    direction = upper_points - lower_points

    def field(parameter: Array) -> Array:
        return kernel.boundary_field(state, lower_points + parameter[:, None] * direction)

    count = lower_points.shape[0]
    initial = (
        jnp.asarray(0, dtype=jnp.int32),
        jnp.zeros((count,), dtype=lower_points.dtype),
        jnp.ones((count,), dtype=lower_points.dtype),
        lower_values,
        upper_values,
        jnp.zeros((count,), dtype=lower_points.dtype),
        jnp.zeros((count,), dtype=jnp.bool_),
    )

    def active(carry: _ReferenceCarry) -> Array:
        _, left, right, _, _, _, converged = carry
        return ~converged & (right - left > 2.0 * _ITP_BRACKET_TOLERANCE)

    def condition(carry: _ReferenceCarry) -> Array:
        return (carry[0] < _ITP_MAXIMUM_ITERATIONS) & jnp.any(active(carry))

    def body(carry: _ReferenceCarry) -> _ReferenceCarry:
        iteration, left, right, left_value, right_value, root, converged = carry
        running = active(carry)
        width = right - left
        middle = 0.5 * (left + right)
        radius = jnp.maximum(
            _ITP_BRACKET_TOLERANCE
            * jnp.exp2(
                (_ITP_MAXIMUM_ITERATIONS - 1 - iteration).astype(lower_points.dtype)
            )
            - 0.5 * width,
            0.0,
        )
        truncation = _ITP_K1 * width**_ITP_K2
        falsi = (right_value * left - left_value * right) / (right_value - left_value)
        falsi = jnp.where(jnp.isfinite(falsi), falsi, middle)
        sigma = jnp.sign(middle - falsi)
        truncated = jnp.where(
            truncation <= jnp.abs(middle - falsi), falsi + sigma * truncation, middle
        )
        candidate = jnp.where(
            jnp.abs(truncated - middle) <= radius, truncated, middle - sigma * radius
        )
        value = field(candidate)
        same_as_left = (value < 0.0) == (left_value < 0.0)
        move_left = running & same_as_left
        move_right = running & ~same_as_left
        hit = running & (jnp.abs(value) <= tolerance)
        return (
            iteration + 1,
            jnp.where(move_left, candidate, left),
            jnp.where(move_right, candidate, right),
            jnp.where(move_left, value, left_value),
            jnp.where(move_right, value, right_value),
            jnp.where(hit, candidate, root),
            converged | hit,
        )

    _, left, right, _, _, root, converged = jax.lax.while_loop(condition, body, initial)
    parameter = jnp.where(converged, root, 0.5 * (left + right))
    points = lower_points + parameter[:, None] * direction
    return points, jnp.abs(kernel.boundary_field(state, points))
