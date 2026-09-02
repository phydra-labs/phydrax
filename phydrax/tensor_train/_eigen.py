#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._core import TensorTrain, TensorTrainOperator, tt_svd
from ._local import solve_spd


class BlockTensorTrainEigenPlan(StrictModule):
    """Static block inverse-iteration, rank, dense, and accuracy budgets."""

    mode_sizes: tuple[int, ...] = eqx.field(static=True)
    block_size: int = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    max_rank: int = eqx.field(static=True)
    compression_relative_tolerance: float = eqx.field(static=True)
    residual_relative_tolerance: float = eqx.field(static=True)
    orthogonality_tolerance: float = eqx.field(static=True)
    inverse_shift: float = eqx.field(static=True)
    max_dense_entries: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode_sizes: Sequence[int],
        /,
        *,
        block_size: int,
        iterations: int,
        max_rank: int,
        compression_relative_tolerance: float,
        residual_relative_tolerance: float,
        orthogonality_tolerance: float,
        inverse_shift: float,
        max_dense_entries: int,
    ):
        modes = tuple(int(size) for size in mode_sizes)
        block = int(block_size)
        iteration_count = int(iterations)
        rank = int(max_rank)
        compression_tolerance = float(compression_relative_tolerance)
        residual_tolerance = float(residual_relative_tolerance)
        orthogonality = float(orthogonality_tolerance)
        shift = float(inverse_shift)
        dense_limit = int(max_dense_entries)
        dimension = prod(modes) if modes else 0
        if not modes or any(size <= 0 for size in modes):
            raise ValueError("Block TT eigen modes must be nonempty and positive.")
        if block <= 0 or block > dimension or iteration_count <= 0 or rank <= 0:
            raise ValueError(
                "Block size, iterations, and rank must be positive and feasible."
            )
        if min(compression_tolerance, residual_tolerance, orthogonality, shift) < 0.0:
            raise ValueError(
                "Block TT eigen tolerances and inverse shift must be non-negative."
            )
        if dense_limit < dimension**2:
            raise ValueError("Block TT eigen dense resource budget is infeasible.")
        self.mode_sizes = modes
        self.block_size = block
        self.iterations = iteration_count
        self.max_rank = rank
        self.compression_relative_tolerance = compression_tolerance
        self.residual_relative_tolerance = residual_tolerance
        self.orthogonality_tolerance = orthogonality
        self.inverse_shift = shift
        self.max_dense_entries = dense_limit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "block-tensor-train-eigen-plan",
                "modes": modes,
                "block_size": block,
                "iterations": iteration_count,
                "max_rank": rank,
                "compression_relative_tolerance": compression_tolerance,
                "residual_relative_tolerance": residual_tolerance,
                "orthogonality_tolerance": orthogonality,
                "inverse_shift": shift,
                "max_dense_entries": dense_limit,
            }
        )


class BlockTensorTrainEigenEvidence(StrictModule):
    true_residual_norms: Array
    relative_residual_norms: Array
    gram_matrix: Array
    orthogonality_error: Array
    compression_frobenius_bounds: Array
    iteration_count: int = eqx.field(static=True)

    def __init__(
        self,
        true_residual_norms: Array,
        relative_residual_norms: Array,
        gram_matrix: Array,
        orthogonality_error: Array,
        compression_frobenius_bounds: Array,
        /,
        *,
        iteration_count: int,
    ):
        residuals = jnp.asarray(true_residual_norms)
        relative = jnp.asarray(relative_residual_norms)
        gram = jnp.asarray(gram_matrix)
        bounds = jnp.asarray(compression_frobenius_bounds)
        if residuals.ndim != 1 or relative.shape != residuals.shape:
            raise ValueError("Block eigen residual evidence has inconsistent shapes.")
        if (
            gram.shape != (residuals.size, residuals.size)
            or bounds.shape != residuals.shape
        ):
            raise ValueError(
                "Block eigen Gram or compression evidence has inconsistent shape."
            )
        self.true_residual_norms = residuals
        self.relative_residual_norms = relative
        self.gram_matrix = gram
        self.orthogonality_error = jnp.asarray(orthogonality_error)
        self.compression_frobenius_bounds = bounds
        self.iteration_count = int(iteration_count)


class BlockTensorTrainEigenResult(StrictModule):
    eigenvalues: Array
    eigenvectors: tuple[TensorTrain, ...]
    evidence: BlockTensorTrainEigenEvidence
    converged: bool = eqx.field(static=True)
    status: str = eqx.field(static=True)

    def __init__(
        self,
        eigenvalues: Array,
        eigenvectors: Sequence[TensorTrain],
        evidence: BlockTensorTrainEigenEvidence,
        converged: bool,
        /,
    ):
        values = jnp.asarray(eigenvalues)
        vectors = tuple(eigenvectors)
        if values.shape != (len(vectors),):
            raise ValueError("Block eigenvalue and eigenvector counts must agree.")
        self.eigenvalues = values
        self.eigenvectors = vectors
        self.evidence = evidence
        self.converged = bool(converged)
        self.status = "converged" if self.converged else "iteration_budget_exhausted"


def smallest_eigenpairs(
    operator: TensorTrainOperator,
    plan: BlockTensorTrainEigenPlan,
    /,
) -> BlockTensorTrainEigenResult:
    """Compute a bounded block of smallest SPD eigenpairs by inverse iteration."""
    if (
        operator.input_mode_sizes != plan.mode_sizes
        or operator.output_mode_sizes != plan.mode_sizes
    ):
        raise ValueError("Block TT eigen operator does not match the plan modes.")
    matrix = operator.to_matrix(max_entries=plan.max_dense_entries)
    dimension = matrix.shape[0]
    rows = jnp.arange(dimension, dtype=jnp.float32)[:, None] + 1
    columns = jnp.arange(plan.block_size, dtype=jnp.float32)[None, :] + 1
    initial = jnp.sin(rows * columns) + jnp.cos(rows * (columns + 0.5))
    vectors, _ = jnp.linalg.qr(initial.astype(matrix.dtype), mode="reduced")
    shifted = matrix + jnp.asarray(plan.inverse_shift, dtype=matrix.dtype) * jnp.eye(
        dimension, dtype=matrix.dtype
    )
    for _ in range(plan.iterations):
        inverse_images = solve_spd(shifted, vectors)
        vectors, _ = jnp.linalg.qr(inverse_images, mode="reduced")
        projected = ein.contract("ia,ij,jb->ab", jnp.conj(vectors), matrix, vectors)
        values, rotation = jnp.linalg.eigh(projected)
        vectors = ein.contract("ia,ab->ib", vectors, rotation)
    trains: list[TensorTrain] = []
    bounds: list[Array] = []
    for column in range(plan.block_size):
        decomposition = tt_svd(
            vectors[:, column].reshape(plan.mode_sizes),
            max_ranks=plan.max_rank,
            relative_tolerance=plan.compression_relative_tolerance,
        )
        trains.append(decomposition.tensor)
        bounds.append(decomposition.evidence.frobenius_error_bound)
    compressed = jnp.stack(
        tuple(
            train.to_dense(max_entries=plan.max_dense_entries).reshape((-1,))
            for train in trains
        ),
        axis=1,
    )
    gram = ein.contract("ia,ib->ab", jnp.conj(compressed), compressed)
    identity = jnp.eye(plan.block_size, dtype=gram.dtype)
    orthogonality_error = jnp.sqrt(jnp.sum(jnp.abs(gram - identity) ** 2))
    applied = ein.contract("ij,jb->ib", matrix, compressed)
    residual = applied - compressed * values[None, :]
    residual_norms = jnp.sqrt(jnp.sum(jnp.abs(residual) ** 2, axis=0))
    matrix_norm = jnp.sqrt(jnp.sum(jnp.abs(matrix) ** 2))
    relative_residuals = residual_norms / jnp.where(matrix_norm > 0, matrix_norm, 1)
    evidence = BlockTensorTrainEigenEvidence(
        residual_norms,
        relative_residuals,
        gram,
        orthogonality_error,
        jnp.stack(bounds),
        iteration_count=plan.iterations,
    )
    converged = bool(
        np.asarray(jnp.max(relative_residuals) <= plan.residual_relative_tolerance)
    ) and bool(np.asarray(orthogonality_error <= plan.orthogonality_tolerance))
    return BlockTensorTrainEigenResult(values, tuple(trains), evidence, converged)


__all__ = [
    "BlockTensorTrainEigenEvidence",
    "BlockTensorTrainEigenPlan",
    "BlockTensorTrainEigenResult",
    "smallest_eigenpairs",
]
