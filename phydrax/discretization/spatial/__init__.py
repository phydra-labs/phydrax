#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical sparse spatial addressing, point hierarchies, and voxel grids."""

from ._dyadic import (
    AdaptiveDyadicGridPlan,
    DyadicAdaptationEvidence,
    DyadicCellTopology,
    DyadicTopologyEvidence,
    DyadicTopologyTransition,
)
from ._dyadic_transfer import (
    DyadicCellTransferPlan,
    DyadicFieldTransferResult,
)
from ._level_octree import (
    SparseLevelOctree,
    SparseLevelOctreeEvidence,
    SparseLevelOctreePlan,
)
from ._morton import (
    canonical_morton_order,
    morton_decode_integer,
    morton_encode_integer,
    MortonAddressPlan,
    MortonCellGeometry,
    MortonEncoding,
)
from ._point_hierarchy import (
    MortonHierarchyBuildEvidence,
    MortonHierarchyTransition,
    MortonPointHierarchyPlan,
    MortonPointHierarchyState,
)
from ._voxel import (
    PreparedSparseVoxelGrid,
    SparseVoxelBuildEvidence,
    SparseVoxelDepositResult,
    SparseVoxelField,
    SparseVoxelGridPlan,
    SparseVoxelLookup,
    SparseVoxelQueryResult,
)


__all__ = [
    "AdaptiveDyadicGridPlan",
    "DyadicAdaptationEvidence",
    "DyadicCellTopology",
    "DyadicCellTransferPlan",
    "DyadicFieldTransferResult",
    "DyadicTopologyEvidence",
    "DyadicTopologyTransition",
    "MortonAddressPlan",
    "MortonCellGeometry",
    "MortonEncoding",
    "MortonHierarchyBuildEvidence",
    "MortonHierarchyTransition",
    "MortonPointHierarchyPlan",
    "MortonPointHierarchyState",
    "PreparedSparseVoxelGrid",
    "SparseVoxelBuildEvidence",
    "SparseVoxelDepositResult",
    "SparseVoxelField",
    "SparseVoxelGridPlan",
    "SparseVoxelLookup",
    "SparseVoxelQueryResult",
    "SparseLevelOctree",
    "SparseLevelOctreeEvidence",
    "SparseLevelOctreePlan",
    "canonical_morton_order",
    "morton_decode_integer",
    "morton_encode_integer",
]
