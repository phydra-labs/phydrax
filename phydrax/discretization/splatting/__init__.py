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
from ._cpdi import (
    AffineCPDISplatAssignment,
    CPDI2AssignmentInput,
    CPDI2SplatAssignment,
    CPDIAssignmentInput,
)
from ._gimp import GIMPAssignmentInput, UniformGIMPSplatAssignment
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
    SplatRouteScatterResult,
)


__all__ = [
    "AbstractStructuredSplatAssignment",
    "AffineCPDISplatAssignment",
    "CPDI2AssignmentInput",
    "CPDI2SplatAssignment",
    "CPDIAssignmentInput",
    "GIMPAssignmentInput",
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
    "SplatRouteScatterResult",
    "TensorBSplineSplatAssignment",
    "UniformGIMPSplatAssignment",
]
