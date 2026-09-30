#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Exterior powers with polynomial determinant derivatives at singular matrices."""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from math import comb, prod

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ._small_batched import determinant_small_linear, SmallLinearSolvePlan
from .backends._jax_dense import _prepare_lu, dense_lu_slogdet


@lru_cache(maxsize=4)
def _small_plan(size: int, /) -> SmallLinearSolvePlan:
    return SmallLinearSolvePlan(size)


def _determinant_value(matrix: Array, /) -> Array:
    size = matrix.shape[-1]
    if size == 0:
        return jnp.ones(matrix.shape[:-2], dtype=matrix.dtype)
    if size <= 4:
        return determinant_small_linear(_small_plan(size), matrix)
    # Reuse the native batched LU owner, but never its singular slogdet JVP.
    state = _prepare_lu(matrix, matrix.shape[:-2])
    sign, log_abs = dense_lu_slogdet(state)
    return sign * jnp.exp(log_abs)


@jax.custom_jvp
def _determinant(matrix: Array, /) -> Array:
    return _determinant_value(matrix)


@_determinant.defjvp
def _determinant_jvp(
    primals: tuple[Array], tangents: tuple[Array], /
) -> tuple[Array, Array]:
    (matrix,), (tangent,) = primals, tangents
    size = matrix.shape[-1]
    value = _determinant(matrix)
    if size == 0:
        return value, jnp.zeros_like(value)
    if size == 1:
        return value, tangent[..., 0, 0]
    # Cofactors, not det(A)*A^{-T}: correct even at rank size-1, for complex
    # matrices as a holomorphic derivative, and recursively for higher AD.
    subsets = np.asarray(tuple(combinations(range(size), size - 1)), dtype=np.int32)[::-1]
    minors = matrix[..., subsets[:, None, :, None], subsets[None, :, None, :]]
    parity = (np.arange(size)[:, None] + np.arange(size)[None, :]) % 2
    signs = jnp.asarray(1 - 2 * parity, dtype=matrix.dtype).reshape(
        (1,) * len(matrix.shape[:-2]) + (size, size)
    )
    cofactors = _determinant(minors) * jnp.broadcast_to(signs, matrix.shape)
    return value, jnp.sum(cofactors * tangent, axis=(-2, -1))


def compound_matrix(
    matrix: ArrayLike, degree: int, /, *, maximum_minor_entries: int = 1 << 24
) -> Array:
    """Return lexicographically ordered degree-k minors of a batched matrix.

    The budget counts gathered minor entries across all batches, or output
    entries at degrees zero and one. Basis/minor allocation follows admission.
    """
    if not isinstance(degree, int) or isinstance(degree, bool) or degree < 0:
        raise ValueError("degree must be a static nonnegative integer.")
    if not isinstance(maximum_minor_entries, int) or maximum_minor_entries < 0:
        raise ValueError("maximum_minor_entries must be a nonnegative integer.")
    value = jnp.asarray(matrix)
    if value.ndim < 2:
        raise ValueError("matrix must have at least two axes.")
    if not jnp.issubdtype(value.dtype, jnp.inexact):
        value = value.astype(jnp.float64)
    rows, columns = value.shape[-2:]
    row_count = comb(rows, degree) if degree <= rows else 0
    column_count = comb(columns, degree) if degree <= columns else 0
    output_entries = prod(value.shape[:-2]) * row_count * column_count
    if output_entries * max(1, degree * degree) > maximum_minor_entries:
        raise ValueError("compound matrix exceeds maximum_minor_entries.")
    shape = value.shape[:-2] + (row_count, column_count)
    if degree == 0:
        return jnp.ones(shape, dtype=value.dtype)
    if not row_count or not column_count:
        return jnp.zeros(shape, dtype=value.dtype)
    if degree == 1:
        return value
    row_indices = np.asarray(tuple(combinations(range(rows), degree)), dtype=np.int32)
    column_indices = np.asarray(
        tuple(combinations(range(columns), degree)), dtype=np.int32
    )
    minors = value[..., row_indices[:, None, :, None], column_indices[None, :, None, :]]
    return _determinant(minors)


__all__ = ["compound_matrix"]
