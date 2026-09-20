#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._strict import StrictModule


class PivotedCholeskyResult(StrictModule):
    factor: Array
    pivots: Array
    residual_diagonal: Array
    successful: Array


class RandomizedNystromResult(StrictModule):
    factor: Array
    gram_eigenvalues: Array
    effective_rank: Array


def pivoted_cholesky_factor(
    matrix_element: Callable[[Array, Array], Array],
    /,
    *,
    size: int,
    rank: int,
) -> PivotedCholeskyResult:
    """Prepare a pivoted low-rank Cholesky factor from matrix elements."""

    size_ = int(size)
    rank_ = int(rank)
    if size_ <= 0:
        raise ValueError("size must be positive.")
    if rank_ <= 0 or rank_ > size_:
        raise ValueError("rank must lie in [1, size].")
    indices = jnp.arange(size_, dtype=jnp.int32)
    diagonal = jax.vmap(matrix_element)(indices, indices)
    factor = jnp.zeros((size_, rank_), dtype=diagonal.dtype)
    permutation = indices

    def swap_rows(array, left, right):
        left_row = array[left]
        right_row = array[right]
        return array.at[left].set(right_row).at[right].set(left_row)

    def step(column_index, state):
        factor_, permutation_, successful_ = state
        permuted_diagonal = diagonal[permutation_]
        residual = permuted_diagonal - jnp.sum(jnp.abs(factor_) ** 2, axis=1)
        candidates = jnp.where(
            indices >= column_index,
            residual,
            jnp.asarray(-jnp.inf, dtype=residual.dtype),
        )
        pivot_index = jnp.argmax(candidates)
        factor_ = swap_rows(factor_, column_index, pivot_index)
        permutation_ = swap_rows(permutation_, column_index, pivot_index)

        pivot_residual = diagonal[permutation_[column_index]] - jnp.sum(
            jnp.abs(factor_[column_index]) ** 2
        )
        valid = jnp.isfinite(pivot_residual) & (pivot_residual > 0.0)
        pivot = jnp.sqrt(jnp.where(valid, pivot_residual, 1.0))
        matrix_column = jax.vmap(
            lambda row: matrix_element(permutation_[row], permutation_[column_index])
        )(indices)
        values = (matrix_column - factor_ @ jnp.conj(factor_[column_index])) / pivot
        values = jnp.where((indices >= column_index) & valid, values, 0.0)
        factor_ = factor_.at[:, column_index].set(values)
        return factor_, permutation_, successful_ & valid & jnp.all(jnp.isfinite(values))

    factor, permutation, successful = jax.lax.fori_loop(
        0,
        rank_,
        step,
        (factor, permutation, jnp.asarray(True)),
    )
    inverse_permutation = jnp.argsort(permutation)
    factor = factor[inverse_permutation]
    residual = diagonal - jnp.sum(jnp.abs(factor) ** 2, axis=1)
    return PivotedCholeskyResult(
        factor,
        permutation[:rank_],
        residual,
        successful,
    )


def randomized_nystrom_factor(
    action: Callable[[Array], Array],
    probes: Array,
    /,
    *,
    eigenvalue_relative_tolerance: float,
) -> RandomizedNystromResult:
    """Prepare a positive-semidefinite Nyström factor from fixed probes."""

    probes_ = jnp.asarray(probes)
    if probes_.ndim != 2 or probes_.shape[1] == 0:
        raise ValueError("probes must have shape (dimension, positive sketch size).")
    images = jax.vmap(action, in_axes=1, out_axes=1)(probes_)
    if images.shape != probes_.shape:
        raise ValueError("action must preserve each probe vector shape.")
    gram = probes_.T.conj() @ images
    gram = 0.5 * (gram + gram.T.conj())
    eigenvalues, eigenvectors = jnp.linalg.eigh(gram)
    scale = jnp.max(jnp.abs(eigenvalues))
    threshold = jnp.asarray(eigenvalue_relative_tolerance, dtype=scale.dtype) * scale
    retained = eigenvalues >= threshold
    inverse_roots = jnp.where(retained, eigenvalues ** (-0.5), 0.0)
    factor = images @ (jnp.where(retained, eigenvectors, 0.0) * inverse_roots)
    return RandomizedNystromResult(
        factor,
        eigenvalues,
        jnp.sum(retained, dtype=jnp.int32),
    )


__all__ = [
    "PivotedCholeskyResult",
    "RandomizedNystromResult",
    "pivoted_cholesky_factor",
    "randomized_nystrom_factor",
]
