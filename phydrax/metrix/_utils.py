#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


def _coordinates(points: ArrayLike, dimension: int, /) -> Array:
    values = jnp.asarray(points)
    if values.ndim < 1:
        raise ValueError("Coordinates must have at least one axis.")
    if values.shape[-1] != dimension:
        raise ValueError(
            f"Coordinates must have trailing dimension {dimension}; got {values.shape}."
        )
    return values


def _pointwise_array(
    function: Callable[[Array], Any],
    points: ArrayLike,
    dimension: int,
    /,
) -> Array:
    values = _coordinates(points, dimension)
    if values.ndim == 1:
        return jnp.asarray(function(values))
    leading_shape = values.shape[:-1]
    flattened = values.reshape((-1, dimension))
    result = jax.vmap(function)(flattened)
    return jnp.asarray(result).reshape(leading_shape + result.shape[1:])


def _pointwise_jacfwd(
    function: Callable[[Array], Any],
    points: ArrayLike,
    dimension: int,
    /,
) -> Array:
    return _pointwise_array(jax.jacfwd(function), points, dimension)


def _pointwise_jacrev(
    function: Callable[[Array], Any],
    points: ArrayLike,
    dimension: int,
    /,
) -> Array:
    return _pointwise_array(jax.jacrev(function), points, dimension)
