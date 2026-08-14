#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
from jaxtyping import Array

from ..._strict import StrictModule


class KrylovBreakdownStatus(IntEnum):
    NONE = 0
    HAPPY = 1
    NEAR_BREAKDOWN = 2
    NONFINITE_ACTION = 3
    LOSS_OF_ORTHOGONALITY = 4
    RANK_DEFICIENT_START = 5
    STAGNATION = 6


class KrylovDecomposition(StrictModule):
    """Fixed-capacity decomposition with a dynamic effective dimension."""

    basis: Array
    projected: Array
    residual_vector: Array
    residual_norm: Array
    effective_dimension: Array
    breakdown_status: Array
    orthogonality_error: Array
    matvec_count: Array
    adjoint_matvec_count: Array
    method: str = eqx.field(static=True)
    provenance: str = eqx.field(static=True)


class BlockKrylovDecomposition(StrictModule):
    """Fixed-capacity block Arnoldi decomposition with column deflation."""

    basis: Array
    projected: Array
    initial_factor: Array
    block_ranks: Array
    effective_dimension: Array
    breakdown_status: Array
    orthogonality_error: Array
    matvec_count: Array
    block_size: int = eqx.field(static=True)
    max_blocks: int = eqx.field(static=True)
    method: str = eqx.field(static=True)
    provenance: str = eqx.field(static=True)


class GolubKahanDecomposition(StrictModule):
    """Fixed-capacity two-basis bidiagonalization artifact."""

    left_basis: Array
    right_basis: Array
    diagonal: Array
    superdiagonal: Array
    effective_dimension: Array
    breakdown_status: Array
    left_orthogonality_error: Array
    right_orthogonality_error: Array
    matvec_count: Array
    adjoint_matvec_count: Array
    provenance: str = eqx.field(static=True)


class KrylovIterationData(StrictModule):
    """Common fixed-shape diagnostics returned by native iterative solvers."""

    iterations: Array
    residual_norm: Array
    normal_residual_norm: Array
    condition_estimate: Array
    breakdown_status: Array
    basis_bytes: int = eqx.field(static=True)


__all__ = [
    "BlockKrylovDecomposition",
    "GolubKahanDecomposition",
    "KrylovBreakdownStatus",
    "KrylovDecomposition",
    "KrylovIterationData",
]
