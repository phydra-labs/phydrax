#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isqrt

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


def symmetric_packed_dimension(matrix_dimension: int, /) -> int:
    dimension = int(matrix_dimension)
    if dimension <= 0:
        raise ValueError("matrix_dimension must be positive.")
    return dimension * (dimension + 1) // 2


def symmetric_matrix_dimension(packed_dimension: int, /) -> int:
    packed = int(packed_dimension)
    if packed <= 0:
        raise ValueError("packed_dimension must be positive.")
    discriminant = 1 + 8 * packed
    root = isqrt(discriminant)
    if root * root != discriminant or (root - 1) % 2:
        raise ValueError("packed_dimension must be triangular: d * (d + 1) / 2.")
    return (root - 1) // 2


def svec(matrix: ArrayLike, /) -> Array:
    """Pack a real symmetric matrix in Frobenius-orthonormal coordinates."""
    values = jnp.asarray(matrix)
    if jnp.issubdtype(values.dtype, jnp.complexfloating):
        raise TypeError("svec requires real-valued matrices.")
    if values.ndim < 2 or values.shape[-2] != values.shape[-1]:
        raise ValueError("svec requires square trailing matrix axes.")
    values = values.astype(jnp.result_type(values, 0.0))
    dimension = int(values.shape[-1])
    rows, columns = jnp.triu_indices(dimension)
    packed = values[..., rows, columns]
    scale = jnp.where(rows == columns, 1.0, jnp.sqrt(2.0)).astype(packed.dtype)
    return packed * scale


def smat(vector: ArrayLike, /, *, matrix_dimension: int | None = None) -> Array:
    """Unpack Frobenius-orthonormal coordinates to a real symmetric matrix."""
    values = jnp.asarray(vector)
    if jnp.issubdtype(values.dtype, jnp.complexfloating):
        raise TypeError("smat requires real-valued coordinates.")
    if values.ndim == 0:
        raise ValueError("smat requires a trailing packed-coordinate axis.")
    values = values.astype(jnp.result_type(values, 0.0))
    dimension = (
        symmetric_matrix_dimension(int(values.shape[-1]))
        if matrix_dimension is None
        else int(matrix_dimension)
    )
    expected = symmetric_packed_dimension(dimension)
    if int(values.shape[-1]) != expected:
        raise ValueError(
            f"smat expected trailing packed dimension {expected}; got {values.shape}."
        )
    rows, columns = jnp.triu_indices(dimension)
    scale = jnp.where(rows == columns, 1.0, jnp.sqrt(2.0)).astype(values.dtype)
    unscaled = values / scale
    matrix = jnp.zeros(values.shape[:-1] + (dimension, dimension), dtype=values.dtype)
    matrix = matrix.at[..., rows, columns].set(unscaled)
    matrix = matrix.at[..., columns, rows].set(unscaled)
    return matrix


__all__ = ["smat", "svec", "symmetric_matrix_dimension", "symmetric_packed_dimension"]
