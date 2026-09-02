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
from ._mesh import (
    MeshCompactKernelSplatAssignment,
    MeshPartitionPolicy,
    MeshSplatBoundaryPolicy,
    MeshSplatDepositResult,
    MeshSplatGatherResult,
    MeshSplatGeometryAD,
    MeshSplatRouteEvidence,
    MeshSplatRoutes,
    MeshSplatTarget,
    ParticleGridSplatEpoch,
    ParticleGridSplatEpochTransition,
    prepare_particle_grid_splat_transition,
    PreparedMeshParticleGridSplat,
    SimplicialBarycentricSplatAssignment,
)
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
    "MeshCompactKernelSplatAssignment",
    "MeshPartitionPolicy",
    "MeshSplatBoundaryPolicy",
    "MeshSplatDepositResult",
    "MeshSplatGatherResult",
    "MeshSplatGeometryAD",
    "MeshSplatRouteEvidence",
    "MeshSplatRoutes",
    "MeshSplatTarget",
    "ParticleGridSplatBudget",
    "ParticleGridSplatPlan",
    "ParticleGridSplatState",
    "PreparedParticleGridSplat",
    "ParticleGridSplatEpoch",
    "ParticleGridSplatEpochTransition",
    "PreparedMeshParticleGridSplat",
    "SimplicialBarycentricSplatAssignment",
    "prepare_particle_grid_splat_transition",
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
