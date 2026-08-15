#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._operators import AbstractLinearOperator


SparseFormat = Literal["csr"]


class SparseStorage(StrictModule):
    """Canonical CSR storage owned by a sparse linear operator."""

    values: Array
    indices: Array
    indptr: Array
    shape: tuple[int, int] = eqx.field(static=True)
    format: SparseFormat = eqx.field(static=True)
    index_width: int = eqx.field(static=True)
    sorted_indices: bool = eqx.field(static=True)
    canonical: bool = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        indices: ArrayLike,
        indptr: ArrayLike,
        /,
        *,
        shape: tuple[int, int],
        sorted_indices: bool = True,
        canonical: bool = True,
    ):
        values_ = jnp.asarray(values)
        indices_ = jnp.asarray(indices)
        indptr_ = jnp.asarray(indptr)
        shape_values = tuple(int(size) for size in shape)
        if len(shape_values) != 2 or any(size < 0 for size in shape_values):
            raise ValueError("Sparse storage shape must contain nonnegative dimensions.")
        shape_ = (shape_values[0], shape_values[1])
        if values_.ndim != 1 or indices_.shape != values_.shape:
            raise ValueError("Sparse values and indices must be matching vectors.")
        if indptr_.shape != (shape_[0] + 1,):
            raise ValueError("CSR indptr length must equal the row count plus one.")
        if not jnp.issubdtype(values_.dtype, jnp.inexact):
            raise TypeError("Sparse values must use an inexact dtype.")
        if not jnp.issubdtype(indices_.dtype, jnp.integer) or not jnp.issubdtype(
            indptr_.dtype, jnp.integer
        ):
            raise TypeError("Sparse indices and indptr must use integer dtypes.")
        if indices_.dtype != indptr_.dtype:
            raise TypeError("Sparse indices and indptr must use the same dtype.")
        width = int(indices_.dtype.itemsize * 8)
        if width not in (32, 64):
            raise TypeError("Sparse index width must be 32 or 64 bits.")
        self.values = values_
        self.indices = indices_
        self.indptr = indptr_
        self.shape = shape_
        self.format = "csr"
        self.index_width = width
        self.sorted_indices = bool(sorted_indices)
        self.canonical = bool(canonical)


class AbstractSparseLinearOperator(AbstractLinearOperator):
    """Linear operator exposing validated canonical sparse storage."""

    @abc.abstractmethod
    def sparse_storage(self, /) -> SparseStorage:
        raise NotImplementedError

    def _assemble_diagonal(self, /) -> Array:
        storage = self.sparse_storage()
        rows = jnp.repeat(
            jnp.arange(storage.shape[0], dtype=storage.indptr.dtype),
            jnp.diff(storage.indptr),
            total_repeat_length=storage.values.size,
        )
        diagonal_entries = jnp.where(
            rows == storage.indices,
            storage.values,
            jnp.zeros((), dtype=storage.values.dtype),
        )
        diagonal = jnp.zeros((storage.shape[0],), dtype=storage.values.dtype)
        return diagonal.at[rows].add(diagonal_entries)


__all__ = ["AbstractSparseLinearOperator", "SparseFormat", "SparseStorage"]
