#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-capacity block AMR topology, FillPatch, and distributed execution."""

from ._adaptive_implicit import (
    AdaptiveImplicitSamplingEvidence,
    AdaptiveImplicitSamplingPlan,
    CertifiedImplicitBody,
    PreparedAdaptiveImplicitSampling,
)
from ._ale import (
    VariablePatchALEPlan,
    VariablePatchALEStepEvidence,
    VariablePatchALEStepGeometry,
)
from ._canonical import (
    BlockAMRResourceEvidence,
    BlockAMRResourcePlan,
    canonicalize_patch_hierarchy,
    CanonicalPatchBucket,
    CanonicalPatchHierarchy,
    CanonicalPatchLevel,
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
from ._cut_cochain import CutCellCochainPlan, CutCellCochainState
from ._cut_cochain_transfer import (
    CutCellCochainTransferEvidence,
    CutCellCochainTransferPlan,
)
from ._cut_complex import (
    CutCellSignTopology,
    EmbeddedLevelSetBody,
    EmbeddedLevelSetBodySet,
    MultivaluedCutCellComplex,
    MultivaluedCutCellEvidence,
    MultivaluedCutCellPlan,
)
from ._cut_complex_2d import (
    MultivaluedCutCell2DComplex,
    MultivaluedCutCell2DEvidence,
    MultivaluedCutCell2DPlan,
)
from ._cut_distributed import (
    DistributedCutCellEvidence,
    DistributedCutCellPartitionPlan,
    DistributedCutCellState,
    PreparedDistributedCutCellComplex,
)
from ._cut_transition import (
    MultivaluedCutCellTransition,
    MultivaluedCutCellTransitionResult,
)
from ._derivatives import (
    BlockAMRDerivativeEvidence,
    BlockAMRDerivativePolicy,
    EventAwareCutCellDerivativePlan,
    FrozenCutCellDerivativeResult,
    FrozenCutCellTransitionDerivativePlan,
    MappedGeometryDerivativePlan,
    RelaxedHierarchyBlendPlan,
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
from ._mapped_geometry import (
    CanonicalMappedGeometryEvidence,
    CanonicalMappedGeometryPlan,
    CanonicalMappedGeometryState,
    MappedMortarEvidence,
    MappedMortarFluxPlan,
    MappedMortarFluxResult,
    MappedMortarGeometry,
    MappedMortarPlan,
    PatchCoordinateMapSet,
)
from ._patches import (
    BlockHierarchyCapacityPlan,
    LogicalPatchBox,
    PatchBucketPlan,
    PatchShapeSignature,
)
from ._reference_parity import (
    BlockAMRReferenceParityEvidence,
    BlockAMRReferenceParityPlan,
)
from ._reflux import FluxRegister
from ._signature_cache import (
    PatchExecutableCachePlan,
    PatchExecutableCacheState,
    PatchExecutableInstallResult,
    PatchExecutableSignature,
    PatchSignaturePolicy,
    PreparedPatchExecutable,
)
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
    PatchClusteringPolicy,
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
    "AdaptiveImplicitSamplingEvidence",
    "AdaptiveImplicitSamplingPlan",
    "BlockAMRPartitionPlan",
    "BlockAMRStableIDMigrationPlan",
    "BlockFieldTopologyTransition",
    "BlockFieldTopologyTransitionResult",
    "BlockHierarchyPlan",
    "BlockHierarchyState",
    "BlockAMRResourceEvidence",
    "BlockAMRResourcePlan",
    "BlockAMRReferenceParityEvidence",
    "BlockAMRReferenceParityPlan",
    "BlockHierarchyTopology",
    "BlockLevelPlan",
    "BlockLevelState",
    "BlockMetadata",
    "BlockTopologyCompileEvidence",
    "BlockTopologyCompileResult",
    "BlockTopologyCompileStatus",
    "BlockTopologyCompiler",
    "BlockTopologyRouteGraph",
    "CanonicalPatchBucket",
    "CanonicalPatchHierarchy",
    "CanonicalMappedGeometryEvidence",
    "CanonicalMappedGeometryPlan",
    "CanonicalMappedGeometryState",
    "CanonicalPatchLevel",
    "canonicalize_patch_hierarchy",
    "CertifiedImplicitBody",
    "CompositeAMRCellLayout",
    "DistributedBlockAMRResourceEvidence",
    "BlockAMRDerivativeEvidence",
    "BlockAMRDerivativePolicy",
    "DistributedCutCellEvidence",
    "DistributedCutCellPartitionPlan",
    "DistributedCutCellState",
    "FDAMRFillPatchPlan",
    "FDAMRFillPatchResult",
    "FDAMRFillPatchWorkspace",
    "FDAMRHierarchyPlan",
    "FDAMRPhysicalBoundaryRequest",
    "FillPatchSource",
    "FluxRegister",
    "PreparedDistributedBlockAMRHierarchy",
    "PreparedDistributedCutCellComplex",
    "PreparedFDAMRHierarchy",
    "BlockHierarchyCapacityPlan",
    "LogicalPatchBox",
    "PatchBucketPlan",
    "PatchShapeSignature",
    "PatchClusteringPolicy",
    "CutCellCochainPlan",
    "CutCellCochainState",
    "CutCellCochainTransferEvidence",
    "CutCellCochainTransferPlan",
    "CutCellSignTopology",
    "EventAwareCutCellDerivativePlan",
    "FrozenCutCellDerivativeResult",
    "FrozenCutCellTransitionDerivativePlan",
    "EmbeddedLevelSetBody",
    "EmbeddedLevelSetBodySet",
    "MultivaluedCutCellComplex",
    "MultivaluedCutCell2DComplex",
    "MultivaluedCutCell2DEvidence",
    "MultivaluedCutCell2DPlan",
    "MappedMortarEvidence",
    "MappedMortarGeometry",
    "MappedMortarFluxPlan",
    "MappedMortarFluxResult",
    "MappedMortarPlan",
    "PatchCoordinateMapSet",
    "MappedGeometryDerivativePlan",
    "MultivaluedCutCellEvidence",
    "MultivaluedCutCellPlan",
    "PreparedAdaptiveImplicitSampling",
    "MultivaluedCutCellTransition",
    "MultivaluedCutCellTransitionResult",
    "PatchExecutableCachePlan",
    "PatchExecutableCacheState",
    "PatchExecutableInstallResult",
    "PatchExecutableSignature",
    "PatchSignaturePolicy",
    "PreparedPatchExecutable",
    "RelaxedHierarchyBlendPlan",
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
