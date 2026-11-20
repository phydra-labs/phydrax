#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Fixed IEEE operation schedules shared by JAX routes and native meshcore kernels.

XLA CPU contracts a product feeding an addition into one fused multiply-add
and lowers reductions in an order of its own, both depending on the
surrounding fusion. Every product here is separated by an optimization
barrier and every sum is an explicit left-to-right chain, so a jitted, eager,
or vmapped evaluation rounds exactly like the native kernels (compiled with
``-ffp-contract=off``) that spell the same expressions. Native counterparts:
``native/meshcore/src/bvh_query.cpp``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def product(first: Array | float, second: Array | float, /) -> Array:
    """``first * second`` rounded on its own (never fused into a later sum)."""
    return jax.lax.optimization_barrier(jnp.multiply(first, second))


def ordered_sum(values: Array, /) -> Array:
    """Left-to-right sum over the trailing axis: ``((v0 + v1) + v2) + ...``."""
    total = values[..., 0]
    for index in range(1, values.shape[-1]):
        total = total + values[..., index]
    return total


def dot3(first: Array, second: Array, /) -> Array:
    """``(u0 v0 + u1 v1) + u2 v2`` over the trailing axis of length 3."""
    terms = product(first, second)
    return (terms[..., 0] + terms[..., 1]) + terms[..., 2]


def cross3(first: Array, second: Array, /) -> Array:
    """``(u1 v2 - u2 v1, u2 v0 - u0 v2, u0 v1 - u1 v0)`` over the trailing axis."""
    # Both product vectors cross one barrier together.
    leading, trailing = jax.lax.optimization_barrier(
        (
            first[..., (1, 2, 0)] * second[..., (2, 0, 1)],
            first[..., (2, 0, 1)] * second[..., (1, 2, 0)],
        )
    )
    return leading - trailing


def quotient(numerator: Array, denominator: Array | float, /) -> Array:
    """``numerator / denominator`` as one IEEE division.

    XLA rewrites a division by a constant or by a broadcast into a product with
    a reciprocal; the barrier hides the denominator's origin.
    """
    hidden = jax.lax.optimization_barrier(
        jnp.broadcast_to(jnp.asarray(denominator, dtype=numerator.dtype), numerator.shape)
    )
    return jax.lax.optimization_barrier(numerator / hidden)


def root(value: Array, /) -> Array:
    """``sqrt(value)`` rounded on its own (never rewritten into a reciprocal root)."""
    return jax.lax.optimization_barrier(jnp.sqrt(value))


def norm3(values: Array, /) -> Array:
    """Euclidean norm ``sqrt(dot3(u, u))`` over the trailing axis of length 3."""
    return root(dot3(values, values))


__all__ = ["cross3", "dot3", "norm3", "ordered_sum", "product", "quotient", "root"]
