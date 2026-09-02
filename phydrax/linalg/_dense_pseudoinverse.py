#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._strict import StrictModule
from ._policies import RankPolicy
from ._rank import numerical_rank_data


class DensePseudoinverseFactors(StrictModule):
    matrix: Array
    left_vectors: Array
    singular_values: Array
    right_adjoint: Array
    retained: Array
    rank: Array
    rank_cutoff: Array
    condition_estimate: Array
    finite: Array
    hermitian: bool = eqx.field(static=True)


def factor_pseudoinverse(
    matrix: Array,
    rank_policy: RankPolicy,
    /,
    *,
    hermitian: bool = False,
) -> DensePseudoinverseFactors:
    """Compute one batched economy factorization for Moore-Penrose operations."""
    value = jnp.asarray(matrix)
    if value.ndim < 2:
        raise ValueError("matrix must have at least two dimensions.")
    if not jnp.issubdtype(value.dtype, jnp.inexact):
        value = value.astype(float)
    rows, columns = (int(size) for size in value.shape[-2:])
    if hermitian and rows != columns:
        raise ValueError("Hermitian pseudoinverse requires square matrices.")
    if hermitian:
        value = 0.5 * (value + jnp.conj(jnp.swapaxes(value, -1, -2)))
    left, singular_values, right_adjoint = jnp.linalg.svd(
        value,
        full_matrices=False,
        hermitian=hermitian,
    )
    rank = numerical_rank_data(singular_values, rows, columns, rank_policy)
    finite = rank.finite & jnp.all(jnp.isfinite(value), axis=(-2, -1))
    return DensePseudoinverseFactors(
        matrix=value,
        left_vectors=left,
        singular_values=singular_values,
        right_adjoint=right_adjoint,
        retained=rank.retained,
        rank=rank.rank,
        rank_cutoff=rank.cutoff,
        condition_estimate=rank.condition_estimate,
        finite=finite,
        hermitian=hermitian,
    )


def apply_pseudoinverse(
    factors: DensePseudoinverseFactors,
    rhs: Array,
    /,
) -> Array:
    """Apply A⁺ without materializing its dense coordinate matrix."""
    value = jnp.asarray(rhs)
    vector_rhs = value.ndim == factors.left_vectors.ndim - 1
    if vector_rhs:
        value = value[..., None]
    projected = jnp.matmul(
        jnp.conj(jnp.swapaxes(factors.left_vectors, -1, -2)),
        value,
    )
    safe = jnp.where(
        factors.retained,
        factors.singular_values,
        jnp.ones_like(factors.singular_values),
    )
    reciprocal = jnp.where(factors.retained, 1.0 / safe, 0.0)
    scaled = reciprocal[..., :, None] * projected
    result = jnp.matmul(
        jnp.conj(jnp.swapaxes(factors.right_adjoint, -1, -2)),
        scaled,
    )
    return result[..., 0] if vector_rhs else result


def materialize_pseudoinverse(
    factors: DensePseudoinverseFactors,
    /,
) -> Array:
    """Materialize A⁺ directly from economy factors with one matrix product."""
    safe = jnp.where(
        factors.retained,
        factors.singular_values,
        jnp.ones_like(factors.singular_values),
    )
    reciprocal = jnp.where(factors.retained, 1.0 / safe, 0.0)
    scaled_right = (
        jnp.conj(jnp.swapaxes(factors.right_adjoint, -1, -2)) * reciprocal[..., None, :]
    )
    value = jnp.matmul(
        scaled_right,
        jnp.conj(jnp.swapaxes(factors.left_vectors, -1, -2)),
    )
    return fixed_rank_pseudoinverse_value(
        factors.matrix,
        value,
        factors.hermitian,
    )


def fixed_rank_pseudoinverse_value(
    matrix: Array,
    value: Array,
    hermitian: bool,
    /,
) -> Array:
    return (
        _fixed_rank_hermitian_pseudoinverse(matrix, value)
        if hermitian
        else _fixed_rank_pseudoinverse(matrix, value)
    )


def _adjoint(value: Array, /) -> Array:
    return jnp.conj(jnp.swapaxes(value, -1, -2))


def _pseudoinverse_tangent(
    matrix: Array,
    pseudoinverse: Array,
    tangent: Array,
    /,
) -> Array:
    rows, columns = matrix.shape[-2:]
    pseudoinverse_adjoint = _adjoint(pseudoinverse)
    tangent_adjoint = _adjoint(tangent)
    result = -pseudoinverse @ tangent @ pseudoinverse
    if rows >= columns:
        target_identity = jnp.eye(rows, dtype=matrix.dtype)
        target_complement = target_identity - matrix @ pseudoinverse
        result = (
            result
            + pseudoinverse @ pseudoinverse_adjoint @ tangent_adjoint @ target_complement
        )
    if columns >= rows:
        source_identity = jnp.eye(columns, dtype=matrix.dtype)
        source_complement = source_identity - pseudoinverse @ matrix
        result = (
            result
            + source_complement @ tangent_adjoint @ pseudoinverse_adjoint @ pseudoinverse
        )
    return result


@jax.custom_jvp
def _fixed_rank_pseudoinverse(matrix: Array, value: Array, /) -> Array:
    del matrix
    return value


@_fixed_rank_pseudoinverse.defjvp
def _fixed_rank_pseudoinverse_jvp(primals, tangents):
    matrix, value = primals
    matrix_tangent, _ = tangents
    return value, _pseudoinverse_tangent(matrix, value, matrix_tangent)


@jax.custom_jvp
def _fixed_rank_hermitian_pseudoinverse(
    matrix: Array,
    value: Array,
    /,
) -> Array:
    del matrix
    return value


@_fixed_rank_hermitian_pseudoinverse.defjvp
def _fixed_rank_hermitian_pseudoinverse_jvp(primals, tangents):
    matrix, value = primals
    matrix_tangent, _ = tangents
    matrix_tangent = 0.5 * (matrix_tangent + _adjoint(matrix_tangent))
    return value, _pseudoinverse_tangent(matrix, value, matrix_tangent)


__all__ = [
    "DensePseudoinverseFactors",
    "apply_pseudoinverse",
    "factor_pseudoinverse",
    "materialize_pseudoinverse",
]
