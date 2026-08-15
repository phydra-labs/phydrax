#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Reusable breakdown-safe Krylov decompositions and diagnostics."""

from ._decompositions import (
    arnoldi,
    block_arnoldi,
    golub_kahan,
    lanczos,
    Orthogonalization,
)
from ._projection import (
    KrylovProjectionCostEstimate,
    KrylovProjectionMethod,
    KrylovProjectionPlan,
    KrylovProjectionPolicy,
    KrylovProjectionResourcePolicy,
    plan_krylov_projection,
    prepare_krylov_projection,
    PreparedKrylovProjection,
    refresh_krylov_projection,
)
from ._results import (
    BlockKrylovDecomposition,
    GolubKahanDecomposition,
    KrylovBreakdownStatus,
    KrylovDecomposition,
    KrylovIterationData,
)


__all__ = [
    "BlockKrylovDecomposition",
    "GolubKahanDecomposition",
    "KrylovBreakdownStatus",
    "KrylovDecomposition",
    "KrylovIterationData",
    "KrylovProjectionCostEstimate",
    "KrylovProjectionMethod",
    "KrylovProjectionPlan",
    "KrylovProjectionPolicy",
    "KrylovProjectionResourcePolicy",
    "PreparedKrylovProjection",
    "Orthogonalization",
    "block_arnoldi",
    "arnoldi",
    "golub_kahan",
    "lanczos",
    "plan_krylov_projection",
    "prepare_krylov_projection",
    "refresh_krylov_projection",
]
