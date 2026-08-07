#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Graph domains and graph-domain batches."""

from ._batch import (
    GRAPH_ENTITY_INDEX_KEY,
    GRAPH_GRAPH_INDEX_KEY,
    GraphBatch,
)
from ._cochain import (
    as_cochain_field,
    cochain_field_spec,
    has_cochain_field_spec,
    with_cochain_field_spec,
)
from ._components import (
    BoundaryEdges,
    BoundaryNodes,
    CochainCellRegion,
    CochainCells,
    Edges,
    EdgeSet,
    EdgeType,
    Globals,
    graph_component_indices,
    graph_component_indices_for_graph,
    graph_component_kind,
    GraphComponentKind,
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
    graph_trajectory_default_quadrature_total_weight,
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
    "GraphComponentKind",
    "graph_component_indices",
    "graph_component_indices_for_graph",
    "graph_component_kind",
    "as_cochain_field",
    "cochain_field_spec",
    "has_cochain_field_spec",
    "with_cochain_field_spec",
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
    "graph_trajectory_default_quadrature_total_weight",
    "InteriorNodes",
    "InterfaceEdges",
    "NodeSet",
    "NodeType",
    "Nodes",
]
