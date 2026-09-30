#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


# The reduced layout stores upper faces and omits lower faces. Zero wall
# extension gives forward = -backward.T, shared by Maxwell and PIC continuity.
def forward_difference(value: Array, axis: int, spacing: float, periodic: bool) -> Array:
    if periodic:
        shifted = jnp.roll(value, -1, axis=axis)
    else:
        pad = [(0, 0)] * value.ndim
        pad[axis] = (0, 1)
        shifted = jnp.pad(
            jnp.take(value, jnp.arange(1, value.shape[axis]), axis=axis), pad
        )
    return (shifted - value) / spacing


def backward_difference(value: Array, axis: int, spacing: float, periodic: bool) -> Array:
    if periodic:
        previous = jnp.roll(value, 1, axis=axis)
    else:
        pad = [(0, 0)] * value.ndim
        pad[axis] = (1, 0)
        previous = jnp.pad(
            jnp.take(value, jnp.arange(value.shape[axis] - 1), axis=axis), pad
        )
    return (value - previous) / spacing
