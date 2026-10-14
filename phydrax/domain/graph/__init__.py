#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Graph domains and graph-domain batches."""

from typing import TYPE_CHECKING

from ..._deferred_exports import deferred_exports


# Re-exports load on first use, so importing one owner never imports the
# whole package and its import cycles.
_DEFERRED_EXPORTS = {
    "GRAPH_ENTITY_INDEX_KEY": "._batch",
    "GRAPH_GRAPH_INDEX_KEY": "._batch",
    "GraphBatch": "._batch",
    "as_cochain_field": "._cochain",
    "cochain_form_type": "._cochain",
    "cochain_representation": "._cochain",
    "CochainRepresentation": "._cochain",
    "has_cochain_form_type": "._cochain",
    "BoundaryEdges": "._components",
    "BoundaryNodes": "._components",
    "CochainCellRegion": "._components",
    "CochainCells": "._components",
    "Edges": "._components",
    "EdgeSet": "._components",
    "EdgeType": "._components",
    "Globals": "._components",
    "graph_component_indices": "._components",
    "graph_component_indices_for_graph": "._components",
    "graph_component_kind": "._components",
    "GraphComponentKind": "._components",
    "InterfaceEdges": "._components",
    "InteriorNodes": "._components",
    "Nodes": "._components",
    "NodeSet": "._components",
    "NodeType": "._components",
    "GRAPH_DATASET_INDEX_KEY": "._dataset",
    "GRAPH_ENTITY_OFFSET_KEY": "._dataset",
    "GRAPH_SAMPLE_INDEX_KEY": "._dataset",
    "GraphDatasetDomain": "._dataset",
    "GraphDatasetMeasureMode": "._dataset",
    "GraphDomain": "._domain",
    "GraphModel": "._model",
    "GraphModelOutput": "._model",
    "GraphRolloutModel": "._model",
    "graph_trajectory_default_quadrature_total_weight": "._trajectory",
    "GRAPH_TRAJECTORY_TIME_INDEX_KEY": "._trajectory",
    "GraphTrajectoryDatasetDomain": "._trajectory",
    "GraphTrajectoryMeasure": "._trajectory",
    "GraphTrajectorySampling": "._trajectory",
}


__getattr__, __dir__ = deferred_exports(__name__, globals(), _DEFERRED_EXPORTS)


if TYPE_CHECKING:
    from ._batch import (
        GRAPH_ENTITY_INDEX_KEY,
        GRAPH_GRAPH_INDEX_KEY,
        GraphBatch,
    )
    from ._cochain import (
        as_cochain_field,
        cochain_form_type,
        cochain_representation,
        CochainRepresentation,
        has_cochain_form_type,
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
    from ._model import GraphModel, GraphModelOutput, GraphRolloutModel
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
    "CochainRepresentation",
    "cochain_form_type",
    "cochain_representation",
    "has_cochain_form_type",
    "CochainCellRegion",
    "CochainCells",
    "BoundaryEdges",
    "BoundaryNodes",
    "EdgeSet",
    "EdgeType",
    "Edges",
    "Globals",
    "GraphBatch",
    "GraphModel",
    "GraphModelOutput",
    "GraphRolloutModel",
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
