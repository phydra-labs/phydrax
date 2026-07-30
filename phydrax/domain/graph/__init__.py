#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Graph domains and graph-domain batches."""

from ._batch import (
    GRAPH_ENTITY_INDEX_KEY,
    GRAPH_GRAPH_INDEX_KEY,
    GraphBatch,
)
from ._cochain import as_cochain_field, cochain_field_spec
from ._components import (
    BoundaryEdges,
    BoundaryNodes,
    CochainCellRegion,
    CochainCells,
    Edges,
    EdgeSet,
    EdgeType,
    Globals,
    InterfaceEdges,
    InteriorNodes,
    Nodes,
    NodeSet,
    NodeType,
)
from ._dataset import (
    GRAPH_DATASET_INDEX_KEY,
    GRAPH_ENTITY_OFFSET_KEY,
    GRAPH_SAMPLE_INDEX_KEY,
    GraphDatasetDomain,
    GraphDatasetMeasureMode,
)
from ._domain import GraphDomain
from ._trajectory import (
    GRAPH_TRAJECTORY_TIME_INDEX_KEY,
    GraphTrajectoryDatasetDomain,
    GraphTrajectoryMeasure,
    GraphTrajectorySampling,
)


__all__ = [
    "GRAPH_DATASET_INDEX_KEY",
    "GRAPH_ENTITY_OFFSET_KEY",
    "GRAPH_ENTITY_INDEX_KEY",
    "GRAPH_GRAPH_INDEX_KEY",
    "GRAPH_SAMPLE_INDEX_KEY",
    "GRAPH_TRAJECTORY_TIME_INDEX_KEY",
    "as_cochain_field",
    "cochain_field_spec",
    "CochainCellRegion",
    "CochainCells",
    "BoundaryEdges",
    "BoundaryNodes",
    "EdgeSet",
    "EdgeType",
    "Edges",
    "Globals",
    "GraphBatch",
    "GraphDatasetDomain",
    "GraphDatasetMeasureMode",
    "GraphDomain",
    "GraphTrajectoryDatasetDomain",
    "GraphTrajectoryMeasure",
    "GraphTrajectorySampling",
    "InteriorNodes",
    "InterfaceEdges",
    "NodeSet",
    "NodeType",
    "Nodes",
]
