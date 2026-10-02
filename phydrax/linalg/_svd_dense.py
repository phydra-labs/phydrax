# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array

from ._materialization import materialize
from ._spaces import (
    _coordinate_pairing_matrix,
    _coordinate_pairing_weights,
    _has_diagonal_pairing,
    AbstractVectorSpace,
)
from ._svd_contracts import DenseSVDState, SVDProblem, SVDSolvePlan, SVDSolveStatus


def diagonal_scales(space: AbstractVectorSpace, /) -> tuple[Array, Array]:
    weights = _coordinate_pairing_weights(space)
    valid = (
        jnp.all(jnp.isfinite(weights))
        & jnp.all(weights.real > 0)
        & jnp.all(weights.imag == 0)
    )
    return jnp.sqrt(weights.real).astype(weights.dtype), valid


def pairing_factor(space: AbstractVectorSpace, /) -> tuple[Array, Array]:
    pairing = _coordinate_pairing_matrix(space)
    scale = jnp.maximum(jnp.max(jnp.abs(pairing)), 1)
    tolerance = 64 * max(space.size, 1) * jnp.finfo(pairing.real.dtype).eps * scale
    valid = jnp.all(jnp.isfinite(pairing)) & (
        jnp.max(jnp.abs(pairing - pairing.conj().T)) <= tolerance
    )
    factor = jax.lax.cond(
        valid,
        lambda matrix: jnp.linalg.cholesky(matrix, symmetrize_input=False),
        lambda matrix: jnp.full_like(matrix, jnp.nan),
        pairing,
    )
    valid = valid & jnp.all(jnp.isfinite(factor)) & jnp.all(jnp.diag(factor).real > 0)
    return factor, valid


def restore_coordinates(factor: Array, columns: Array, diagonal: bool, /) -> Array:
    if diagonal:
        return columns / factor[:, None]
    return jsp.linalg.solve_triangular(factor.conj().T, columns, lower=False)


def prepare_dense(problem: SVDProblem, plan: SVDSolvePlan, /) -> DenseSVDState:
    operator = problem.operator
    diagonal = _has_diagonal_pairing(operator.source) and _has_diagonal_pairing(
        operator.target
    )
    if diagonal:
        source, source_valid = diagonal_scales(operator.source)
        target, target_valid = diagonal_scales(operator.target)
    else:
        source, source_valid = pairing_factor(operator.source)
        target, target_valid = pairing_factor(operator.target)
    matrix = materialize(operator, plan.policy.materialization)
    valid = source_valid & target_valid & jnp.all(jnp.isfinite(matrix))

    def transform(value: Array) -> Array:
        if diagonal:
            return target[:, None] * value / source[None, :]
        scaled = jsp.linalg.solve_triangular(source, value.conj().T, lower=True).conj().T
        return target.conj().T @ scaled

    reduced = jax.lax.cond(
        valid, transform, lambda value: jnp.full_like(value, jnp.nan), matrix
    )
    valid = valid & jnp.all(jnp.isfinite(reduced))
    status = jnp.where(
        valid, int(SVDSolveStatus.SUCCESS), int(SVDSolveStatus.PREPARATION_FAILED)
    ).astype(jnp.int32)
    return DenseSVDState(reduced, source, target, status, diagonal)


def thin_decomposition(matrix: Array, available: Array, /) -> tuple[Array, Array, Array]:
    rows, columns = matrix.shape
    rank = min(rows, columns)

    def decompose(value: Array) -> tuple[Array, Array, Array]:
        left, values, right_adjoint = jnp.linalg.svd(value, full_matrices=False)
        return left, values, right_adjoint.conj().T

    def unavailable(value: Array) -> tuple[Array, Array, Array]:
        return (
            jnp.full((rows, rank), jnp.nan, value.dtype),
            jnp.full((rank,), jnp.nan, value.real.dtype),
            jnp.full((columns, rank), jnp.nan, value.dtype),
        )

    return jax.lax.cond(available, decompose, unavailable, matrix)
