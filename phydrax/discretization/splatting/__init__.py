#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Measure-aware structured transfer between particles and tensor grids."""

from ._assignment import (
    AbstractStructuredSplatAssignment,
    MultilinearSplatAssignment,
    SplatAssignmentCapabilities,
    SplatAssignmentState,
)
from ._bspline import TensorBSplineSplatAssignment
from ._structured import (
    ParticleGridSplatPlan,
    ParticleGridSplatState,
    PreparedParticleGridSplat,
)
from ._types import (
    ParticleGridSplatBudget,
    SplatAccumulation,
    SplatBalanceEvidence,
    SplatBoundaryPolicy,
    SplatDepositResult,
    SplatExecutionPolicy,
    SplatGeometryAD,
    SplatReconstructionResult,
)


__all__ = [
    "AbstractStructuredSplatAssignment",
    "MultilinearSplatAssignment",
    "ParticleGridSplatBudget",
    "ParticleGridSplatPlan",
    "ParticleGridSplatState",
    "PreparedParticleGridSplat",
    "SplatAssignmentCapabilities",
    "SplatAssignmentState",
    "SplatAccumulation",
    "SplatBalanceEvidence",
    "SplatBoundaryPolicy",
    "SplatDepositResult",
    "SplatExecutionPolicy",
    "SplatGeometryAD",
    "SplatReconstructionResult",
    "TensorBSplineSplatAssignment",
]
