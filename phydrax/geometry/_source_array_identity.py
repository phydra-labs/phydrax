#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Bit-exact dynamic source-array identity checks compatible with JAX."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


@jax.custom_jvp
def _fixed_source_array(value: Array, /) -> Array:
    return value


@_fixed_source_array.defjvp
def _reject_source_jvp(
    primals: tuple[Array], tangents: tuple[Array], /
) -> tuple[Array, Array]:
    del primals, tangents
    raise ValueError("Fixed source geometry differentiation is not supported.")


def _floating_bits(value: Array, /) -> Array:
    width = value.dtype.itemsize
    unsigned = {1: jnp.uint8, 2: jnp.uint16, 4: jnp.uint32, 8: jnp.uint64}.get(width)
    if unsigned is None:
        raise TypeError(f"Unsupported source floating-point width: {width} bytes.")
    return jax.lax.bitcast_convert_type(value, unsigned)


def source_arrays_equal(first: ArrayLike, second: ArrayLike, /) -> Array:
    """Compare exact source representations, including zero signs and NaN payloads.

    Matching shape and dtype are required. Complex components are checked
    independently because JAX's bitcast primitive does not accept complex input.
    """
    left, right = jnp.asarray(first), jnp.asarray(second)
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("Source arrays must have the same shape and dtype.")
    if (jnp.issubdtype(left.dtype, jnp.floating)
            or jnp.issubdtype(left.dtype, jnp.complexfloating)):
        left, right = _fixed_source_array(left), _fixed_source_array(right)
    if jnp.issubdtype(left.dtype, jnp.complexfloating):
        return jnp.all(_floating_bits(jnp.real(left)) == _floating_bits(jnp.real(right))) & jnp.all(
            _floating_bits(jnp.imag(left)) == _floating_bits(jnp.imag(right))
        )
    if jnp.issubdtype(left.dtype, jnp.floating):
        return jnp.all(_floating_bits(left) == _floating_bits(right))
    if jnp.issubdtype(left.dtype, jnp.integer) or jnp.issubdtype(left.dtype, jnp.bool_):
        return jnp.array_equal(left, right)
    raise TypeError(f"Unsupported source array dtype: {left.dtype}.")
