#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-capacity block AMR topology, FillPatch, and distributed execution."""

from ._ale import (
    VariablePatchALEPlan,
    VariablePatchALEStepEvidence,
    VariablePatchALEStepGeometry,
)
from ._composite import CompositeAMRCellLayout
from ._core import (
    BlockHierarchyPlan,
    BlockHierarchyState,
    BlockHierarchyTopology,
    BlockLevelPlan,
    BlockLevelState,
    BlockMetadata,
)
from ._distributed import (
    BlockAMRPartitionPlan,
    BlockAMRStableIDMigrationPlan,
    DistributedBlockAMRResourceEvidence,
    PreparedDistributedBlockAMRHierarchy,
)
from ._embedded import (
    VariablePatchEmbeddedBoundaryEvidence,
    VariablePatchEmbeddedBoundaryMetrics,
    VariablePatchEmbeddedBoundaryPlan,
)
from ._entities import (
    VariablePatchEntityBucketView,
    VariablePatchEntityComplex,
    VariablePatchEntityComplexPlan,
)
from ._entity_runtime import (
    VariablePatchEntityExecutionPlan,
    VariablePatchEntityFieldState,
    VariablePatchEntityRoute,
)
from ._entity_transfer import (
    CompatibleEntityTransfer,
    CompatibleEntityTransferEvidence,
    CompatibleEntityTransferFamily,
)
from ._fd_halo import (
    FDAMRFillPatchPlan,
    FDAMRFillPatchResult,
    FDAMRFillPatchWorkspace,
    FDAMRPhysicalBoundaryRequest,
    FillPatchSource,
)
from ._fd_runtime import FDAMRHierarchyPlan, PreparedFDAMRHierarchy
from ._fd_transfer import (
    AMRAxisEntity,
    AMREntityTransferPlan,
    AMREntityTransferReport,
)
from ._geometry import VariablePatchGeometryPlan, VariablePatchGeometryState
from ._patches import (
    BlockHierarchyCapacityPlan,
    LogicalPatchBox,
    PatchBucketPlan,
    PatchShapeSignature,
)
from ._reflux import FluxRegister
from ._topology_compiler import (
    BlockTopologyCompileEvidence,
    BlockTopologyCompiler,
    BlockTopologyCompileResult,
    BlockTopologyCompileStatus,
    BlockTopologyRouteGraph,
)
from ._topology_transfer import (
    BlockFieldTopologyTransition,
    BlockFieldTopologyTransitionResult,
)
from ._variable import (
    VariablePatchCompileEvidence,
    VariablePatchCompileResult,
    VariablePatchCompileStatus,
    VariablePatchFieldState,
    VariablePatchHierarchyPlan,
    VariablePatchHierarchyTopology,
    VariablePatchLevelMetadata,
    VariablePatchLevelPlan,
    VariablePatchTopologyCompiler,
)
from ._variable_distributed import (
    PreparedVariablePatchPartition,
    VariablePatchPartitionEvidence,
    VariablePatchPartitionPlan,
)
from ._variable_runtime import (
    VariablePatchFillPatchPlan,
    VariablePatchFillPatchResult,
    VariablePatchFillPatchWorkspace,
    VariablePatchFillSource,
    VariablePatchHierarchyState,
    VariablePatchPhysicalBoundaryRequest,
)


__all__ = [
    "AMRAxisEntity",
    "AMREntityTransferPlan",
    "AMREntityTransferReport",
    "BlockAMRPartitionPlan",
    "BlockAMRStableIDMigrationPlan",
    "BlockFieldTopologyTransition",
    "BlockFieldTopologyTransitionResult",
    "BlockHierarchyPlan",
    "BlockHierarchyState",
    "BlockHierarchyTopology",
    "BlockLevelPlan",
    "BlockLevelState",
    "BlockMetadata",
    "BlockTopologyCompileEvidence",
    "BlockTopologyCompileResult",
    "BlockTopologyCompileStatus",
    "BlockTopologyCompiler",
    "BlockTopologyRouteGraph",
    "CompositeAMRCellLayout",
    "DistributedBlockAMRResourceEvidence",
    "FDAMRFillPatchPlan",
    "FDAMRFillPatchResult",
    "FDAMRFillPatchWorkspace",
    "FDAMRHierarchyPlan",
    "FDAMRPhysicalBoundaryRequest",
    "FillPatchSource",
    "FluxRegister",
    "PreparedDistributedBlockAMRHierarchy",
    "PreparedFDAMRHierarchy",
    "BlockHierarchyCapacityPlan",
    "LogicalPatchBox",
    "PatchBucketPlan",
    "PatchShapeSignature",
    "VariablePatchCompileEvidence",
    "VariablePatchCompileResult",
    "VariablePatchCompileStatus",
    "VariablePatchFieldState",
    "VariablePatchHierarchyPlan",
    "VariablePatchHierarchyTopology",
    "VariablePatchLevelMetadata",
    "VariablePatchLevelPlan",
    "VariablePatchTopologyCompiler",
    "VariablePatchFillPatchPlan",
    "VariablePatchFillPatchResult",
    "VariablePatchFillPatchWorkspace",
    "VariablePatchFillSource",
    "VariablePatchHierarchyState",
    "VariablePatchPhysicalBoundaryRequest",
    "VariablePatchEntityBucketView",
    "VariablePatchEntityComplex",
    "VariablePatchEntityComplexPlan",
    "VariablePatchGeometryPlan",
    "VariablePatchGeometryState",
    "VariablePatchEmbeddedBoundaryEvidence",
    "VariablePatchEmbeddedBoundaryMetrics",
    "VariablePatchEmbeddedBoundaryPlan",
    "VariablePatchEntityExecutionPlan",
    "VariablePatchEntityFieldState",
    "VariablePatchEntityRoute",
    "CompatibleEntityTransfer",
    "CompatibleEntityTransferEvidence",
    "CompatibleEntityTransferFamily",
    "VariablePatchALEPlan",
    "VariablePatchALEStepEvidence",
    "VariablePatchALEStepGeometry",
    "PreparedVariablePatchPartition",
    "VariablePatchPartitionEvidence",
    "VariablePatchPartitionPlan",
]
