#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

from .._strict import StrictModule
from ..integration import (
    factorized_bilinear_form,
    FactorizedBilinearEvaluation,
    FactorizedBilinearTerm,
    IntegrationRealization,
    SeparableIntegrationBatch,
)
from ..linalg.eigen import (
    block_rayleigh_trace,
    BlockRayleighEvaluation,
    ReducedRitzResult,
    solve_reduced_ritz,
)


class FactorizedVariationalEigenspaceResult(StrictModule):
    """Factor-preserving form assembly followed by native reduced Ritz extraction."""

    stiffness: FactorizedBilinearEvaluation
    mass: FactorizedBilinearEvaluation
    block: BlockRayleighEvaluation
    reduced: ReducedRitzResult

    @property
    def eigenvalues(self):
        return self.reduced.eigenvalues

    @property
    def successful(self):
        return (
            self.stiffness.valid
            & self.mass.valid
            & self.block.valid
            & self.reduced.successful
        )


def factorized_variational_eigenspace(
    stiffness_terms: Sequence[FactorizedBilinearTerm],
    mass_terms: Sequence[FactorizedBilinearTerm],
    realization_or_batch: IntegrationRealization | SeparableIntegrationBatch,
    /,
    *,
    count: int | None = None,
    which: str = "smallest-algebraic",
    tolerance: float = 1e-10,
) -> FactorizedVariationalEigenspaceResult:
    """Assemble and solve one high-dimensional factorized Hermitian trial space."""
    stiffness = factorized_bilinear_form(stiffness_terms, realization_or_batch)
    mass = factorized_bilinear_form(mass_terms, realization_or_batch)
    if stiffness.value.shape != mass.value.shape:
        raise ValueError("Factorized stiffness and mass matrices must share one shape.")
    block = block_rayleigh_trace(
        stiffness.value,
        mass.value,
        tolerance=tolerance,
    )
    reduced = solve_reduced_ritz(
        block.stiffness,
        block.mass,
        count=count,
        which=which,
        tolerance=tolerance,
    )
    return FactorizedVariationalEigenspaceResult(
        stiffness=stiffness,
        mass=mass,
        block=block,
        reduced=reduced,
    )


__all__ = [
    "FactorizedVariationalEigenspaceResult",
    "factorized_variational_eigenspace",
]
