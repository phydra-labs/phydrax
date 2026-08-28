#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


class FieldJet(StrictModule):
    value: Array
    gradient: Array | None
    divergence: Array | None
    curl: Array | None

    def __init__(
        self,
        value: ArrayLike,
        /,
        *,
        gradient: ArrayLike | None = None,
        divergence: ArrayLike | None = None,
        curl: ArrayLike | None = None,
    ):
        self.value = jnp.asarray(value)
        self.gradient = None if gradient is None else jnp.asarray(gradient)
        self.divergence = None if divergence is None else jnp.asarray(divergence)
        self.curl = None if curl is None else jnp.asarray(curl)


def symmetric_gradient(gradient: ArrayLike, /) -> Array:
    gradient_ = jnp.asarray(gradient)
    if gradient_.shape[-1] != gradient_.shape[-2]:
        raise ValueError("Symmetric gradient requires square value/coordinate axes.")
    return 0.5 * (gradient_ + jnp.swapaxes(gradient_, -1, -2))


def divergence(gradient: ArrayLike, /) -> Array:
    gradient_ = jnp.asarray(gradient)
    if gradient_.shape[-1] != gradient_.shape[-2]:
        raise ValueError("Divergence requires matching value/coordinate dimensions.")
    return jnp.trace(gradient_, axis1=-2, axis2=-1)


def curl(gradient: ArrayLike, /) -> Array:
    gradient_ = jnp.asarray(gradient)
    if gradient_.shape[-2:] == (2, 2):
        return gradient_[..., 1, 0] - gradient_[..., 0, 1]
    if gradient_.shape[-2:] == (3, 3):
        return jnp.stack(
            (
                gradient_[..., 2, 1] - gradient_[..., 1, 2],
                gradient_[..., 0, 2] - gradient_[..., 2, 0],
                gradient_[..., 1, 0] - gradient_[..., 0, 1],
            ),
            axis=-1,
        )
    raise ValueError("Curl requires a two- or three-dimensional vector gradient.")


def normal_trace(value: ArrayLike, normal: ArrayLike, /) -> Array:
    value_ = jnp.asarray(value)
    normal_ = jnp.asarray(normal)
    if value_.shape[-1] != normal_.shape[-1]:
        raise ValueError("Normal trace value and normal dimensions must match.")
    return jnp.sum(value_ * normal_, axis=-1)


def tangential_trace(value: ArrayLike, normal: ArrayLike, /) -> Array:
    value_ = jnp.asarray(value)
    normal_ = jnp.asarray(normal)
    if value_.shape[-1] == 2:
        tangent = jnp.stack((-normal_[..., 1], normal_[..., 0]), axis=-1)
        return jnp.sum(value_ * tangent, axis=-1)
    if value_.shape[-1] == 3:
        return value_ - jnp.sum(value_ * normal_, axis=-1, keepdims=True) * normal_
    raise ValueError("Tangential trace requires a two- or three-dimensional value.")


def jump(plus: ArrayLike, minus: ArrayLike, /) -> Array:
    plus_ = jnp.asarray(plus)
    minus_ = jnp.asarray(minus)
    if plus_.shape != minus_.shape:
        raise ValueError("Jump operands must have identical shape.")
    return plus_ - minus_


def average(plus: ArrayLike, minus: ArrayLike, /) -> Array:
    plus_ = jnp.asarray(plus)
    minus_ = jnp.asarray(minus)
    if plus_.shape != minus_.shape:
        raise ValueError("Average operands must have identical shape.")
    return 0.5 * (plus_ + minus_)


__all__ = [
    "FieldJet",
    "average",
    "curl",
    "divergence",
    "jump",
    "normal_trace",
    "symmetric_gradient",
    "tangential_trace",
]
