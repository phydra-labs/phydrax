#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical sparse spatial addressing, point hierarchies, and voxel grids."""

from ._block_sparse import (
    BlockKeyOrdering,
    SparseBlockBuildEvidence,
    SparseBlockLookup,
    SparseBlockTopologyPlan,
    SparseBlockTopologyState,
    SparseBlockTransition,
)
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
from ._neighbor_query import (
    MortonNeighborQueryEvidence,
    MortonNeighborQueryPlan,
    MortonNeighborQueryResult,
    MortonRadiusRelationEvidence,
    MortonRadiusRelationPlan,
    MortonRadiusRelationResult,
    SpatialDistanceBackend,
)
from ._plane_distributed import (
    DistributedMortonNeighborEvidence,
    DistributedMortonNeighborQueryPlan,
    DistributedMortonNeighborResult,
)
from ._point_hierarchy import (
    MortonHierarchyBuildEvidence,
    MortonHierarchyTransition,
    MortonPointHierarchyPlan,
    MortonPointHierarchyState,
)
from ._primitive_bounds import (
    MortonPrimitiveBoundsEvidence,
    MortonPrimitiveBoundsPlan,
    MortonPrimitiveBoundsState,
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
    "BlockKeyOrdering",
    "AdaptiveDyadicGridPlan",
    "DyadicAdaptationEvidence",
    "DistributedMortonNeighborEvidence",
    "DistributedMortonNeighborQueryPlan",
    "DistributedMortonNeighborResult",
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
    "MortonNeighborQueryEvidence",
    "MortonNeighborQueryPlan",
    "MortonNeighborQueryResult",
    "MortonRadiusRelationEvidence",
    "MortonRadiusRelationPlan",
    "MortonRadiusRelationResult",
    "MortonPrimitiveBoundsEvidence",
    "MortonPrimitiveBoundsPlan",
    "MortonPrimitiveBoundsState",
    "SparseBlockBuildEvidence",
    "SparseBlockLookup",
    "SparseBlockTopologyPlan",
    "SparseBlockTopologyState",
    "SpatialDistanceBackend",
    "SparseBlockTransition",
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
