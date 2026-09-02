#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike


def dense_inverse(
    matrix: ArrayLike,
    /,
    *,
    positive_definite: bool = False,
) -> Array:
    """Materialize a batched dense inverse through one reusable factorization."""
    value = jnp.asarray(matrix)
    if value.ndim < 2 or value.shape[-2] != value.shape[-1]:
        raise ValueError("dense_inverse requires square trailing matrix axes.")
    if not jnp.issubdtype(value.dtype, jnp.inexact):
        value = value.astype(float)
    dimension = value.shape[-1]
    identity = jnp.broadcast_to(jnp.eye(dimension, dtype=value.dtype), value.shape)
    if not positive_definite:
        return jnp.linalg.solve(value, identity)

    factor = jnp.linalg.cholesky(value)
    batch_shape = value.shape[:-2]
    batch_count = prod(batch_shape) if batch_shape else 1
    flat_factor = factor.reshape((batch_count, dimension, dimension))
    flat_identity = identity.reshape((batch_count, dimension, dimension))

    def solve_one(cholesky, right_hand_side):
        intermediate = jsp.linalg.solve_triangular(
            cholesky,
            right_hand_side,
            lower=True,
        )
        return jsp.linalg.solve_triangular(
            jnp.conj(cholesky.T),
            intermediate,
            lower=False,
        )

    inverse = jax.vmap(solve_one)(flat_factor, flat_identity).reshape(value.shape)
    return 0.5 * (inverse + jnp.conj(jnp.swapaxes(inverse, -1, -2)))


__all__ = ["dense_inverse"]
