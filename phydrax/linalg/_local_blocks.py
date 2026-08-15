#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule


LocalBlockFactorizationKind = Literal["lu", "weighted-cholesky"]


class LocalBlockFactorization(StrictModule):
    """Immutable batched factorization of equal-sized local blocks."""

    factors: Array
    pivots: Array
    metric_sqrt: Array
    failed_blocks: Array
    kind: LocalBlockFactorizationKind = eqx.field(static=True)
    num_blocks: int = eqx.field(static=True)
    block_size: int = eqx.field(static=True)

    def __init__(
        self,
        factors: Array,
        pivots: Array,
        metric_sqrt: Array,
        failed_blocks: Array,
        /,
        *,
        kind: LocalBlockFactorizationKind,
    ):
        self.factors = factors
        self.pivots = pivots
        self.metric_sqrt = metric_sqrt
        self.failed_blocks = failed_blocks
        self.kind = kind
        self.num_blocks = int(factors.shape[0])
        self.block_size = int(factors.shape[1])


def prepare_local_block_factorization(
    blocks: ArrayLike,
    /,
    *,
    positive_definite: bool = False,
    metric_weights: ArrayLike | None = None,
) -> LocalBlockFactorization:
    """Factor square local blocks using batched LU or pairing-correct Cholesky."""
    matrices = jnp.asarray(blocks)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("Local block factorization requires square blocks.")
    if any(int(size) < 1 for size in matrices.shape):
        raise ValueError("Local block dimensions must be positive.")
    if not jnp.issubdtype(matrices.dtype, jnp.inexact):
        matrices = matrices.astype(float)
    num_blocks, block_size, _ = matrices.shape
    real_dtype = jnp.empty((), dtype=matrices.dtype).real.dtype
    epsilon = jnp.finfo(real_dtype).eps
    scale = jnp.max(jnp.abs(matrices), axis=(-2, -1))
    tolerance = epsilon * block_size * jnp.maximum(scale, 1.0)
    finite = jnp.all(jnp.isfinite(matrices), axis=(-2, -1))

    if positive_definite:
        weights = (
            jnp.ones((num_blocks, block_size), dtype=real_dtype)
            if metric_weights is None
            else jnp.asarray(metric_weights, dtype=real_dtype)
        )
        if weights.shape != (num_blocks, block_size):
            raise ValueError("metric_weights must match the grouped local block layout.")
        valid_weights = jnp.all(jnp.isfinite(weights) & (weights > 0.0), axis=-1)
        metric_sqrt = jnp.sqrt(weights)
        transformed = metric_sqrt[:, :, None] * matrices / metric_sqrt[:, None, :]
        symmetry_error = jnp.max(
            jnp.abs(transformed - jnp.conj(jnp.swapaxes(transformed, -1, -2))),
            axis=(-2, -1),
        )
        factors = jnp.linalg.cholesky(transformed)
        factor_diagonal = jnp.real(jnp.diagonal(factors, axis1=-2, axis2=-1))
        failed = (
            ~finite
            | ~valid_weights
            | (symmetry_error > tolerance)
            | jnp.any(~jnp.isfinite(factors), axis=(-2, -1))
            | jnp.any(factor_diagonal <= tolerance[:, None], axis=-1)
        )
        pivots = jnp.zeros((num_blocks, block_size), dtype=jnp.int32)
        return LocalBlockFactorization(
            factors,
            pivots,
            metric_sqrt,
            failed,
            kind="weighted-cholesky",
        )

    factors, pivots = jsp.linalg.lu_factor(matrices)
    factor_diagonal = jnp.diagonal(factors, axis1=-2, axis2=-1)
    failed = (
        ~finite
        | jnp.any(~jnp.isfinite(factors), axis=(-2, -1))
        | jnp.any(jnp.abs(factor_diagonal) <= tolerance[:, None], axis=-1)
    )
    return LocalBlockFactorization(
        factors,
        pivots,
        jnp.ones((num_blocks, block_size), dtype=real_dtype),
        failed,
        kind="lu",
    )


def solve_local_blocks(
    factorization: LocalBlockFactorization,
    right_hand_side: ArrayLike,
    /,
) -> tuple[Array, Array]:
    """Apply one local factorization and return the solution and failure flag."""
    if not isinstance(factorization, LocalBlockFactorization):
        raise TypeError("factorization must be a LocalBlockFactorization.")
    rhs = jnp.asarray(right_hand_side)
    expected_prefix = (factorization.num_blocks, factorization.block_size)
    if rhs.shape[:2] != expected_prefix:
        raise ValueError(
            f"right_hand_side must begin with shape {expected_prefix}; got {rhs.shape}."
        )
    trailing_shape = rhs.shape[2:]
    columns = rhs.reshape(expected_prefix + (-1,))

    if factorization.kind == "lu":
        solution = jsp.linalg.lu_solve(
            (factorization.factors, factorization.pivots),
            columns,
        )
    else:
        weighted = columns * factorization.metric_sqrt[:, :, None]
        intermediate = jsp.linalg.solve_triangular(
            factorization.factors,
            weighted,
            lower=True,
        )
        transformed_solution = jsp.linalg.solve_triangular(
            jnp.conj(jnp.swapaxes(factorization.factors, -1, -2)),
            intermediate,
            lower=False,
        )
        solution = transformed_solution / factorization.metric_sqrt[:, :, None]

    failed = jnp.any(factorization.failed_blocks) | jnp.any(~jnp.isfinite(solution))
    solution = jnp.where(failed, jnp.zeros((), dtype=solution.dtype), solution)
    return solution.reshape(expected_prefix + trailing_shape), failed


__all__ = [
    "LocalBlockFactorization",
    "LocalBlockFactorizationKind",
    "prepare_local_block_factorization",
    "solve_local_blocks",
]
