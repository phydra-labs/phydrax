#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-capacity adaptive block hierarchies and conservative synchronization."""

from ._core import (
    BlockHierarchyPlan,
    BlockHierarchyState,
    BlockLevelPlan,
    BlockLevelState,
    BlockMetadata,
)
from ._fd_halo import FDAMRHaloPlan, FDAMRHaloWorkspace
from ._fd_runtime import (
    AMRMigrationPlan,
    AMRMigrationResult,
    AMRSubcycleResult,
    ConservativeAMRSubcyclingPlan,
    FDAMRHierarchyPlan,
    FDRegridPlan,
    FDRegridResult,
    PreparedFDAMRHierarchy,
)
from ._fd_transfer import (
    AMRAxisEntity,
    AMREntityTransferPlan,
    AMREntityTransferReport,
)
from ._refinement import FixedCapacityRefinementPlan, RefinementDecision
from ._reflux import FluxRegister
from ._transfer import ConservativeBlockTransfer
from ._two_level import CoarseFineFluxRegister, TwoLevelAMRPlan, TwoLevelAMRState


__all__ = [
    "AMRAxisEntity",
    "AMREntityTransferPlan",
    "AMREntityTransferReport",
    "AMRMigrationPlan",
    "AMRMigrationResult",
    "AMRSubcycleResult",
    "BlockHierarchyPlan",
    "BlockHierarchyState",
    "BlockLevelPlan",
    "BlockLevelState",
    "BlockMetadata",
    "ConservativeBlockTransfer",
    "FDAMRHaloPlan",
    "FDAMRHaloWorkspace",
    "FDAMRHierarchyPlan",
    "ConservativeAMRSubcyclingPlan",
    "FDRegridPlan",
    "FDRegridResult",
    "FixedCapacityRefinementPlan",
    "FluxRegister",
    "RefinementDecision",
    "PreparedFDAMRHierarchy",
    "CoarseFineFluxRegister",
    "TwoLevelAMRPlan",
    "TwoLevelAMRState",
]
