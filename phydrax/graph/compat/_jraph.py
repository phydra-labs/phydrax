from __future__ import annotations

import importlib
import importlib.util
from typing import Any

from .._ir import GraphIR


_FORWARDED_SYMBOLS = (
    "ArrayTree",
    "DeepSets",
    "GraphConvolution",
    "GraphMapFeatures",
    "InteractionNetwork",
    "RelationNetwork",
    "GraphNetGAT",
    "GAT",
    "GraphsTuple",
    "GraphNetwork",
    "NodeFeatures",
    "AggregateEdgesToNodesFn",
    "AggregateNodesToGlobalsFn",
    "AggregateEdgesToGlobalsFn",
    "AttentionLogitFn",
    "AttentionReduceFn",
    "GNUpdateEdgeFn",
    "GNUpdateNodeFn",
    "GNUpdateGlobalFn",
    "InteractionUpdateNodeFn",
    "InteractionUpdateEdgeFn",
    "EmbedEdgeFn",
    "EmbedNodeFn",
    "EmbedGlobalFn",
    "GATAttentionQueryFn",
    "GATAttentionLogitFn",
    "GATNodeUpdateFn",
    "batch",
    "batch_np",
    "unbatch",
    "unbatch_np",
    "pad_with_graphs",
    "get_number_of_padding_with_graphs_graphs",
    "get_number_of_padding_with_graphs_nodes",
    "get_number_of_padding_with_graphs_edges",
    "unpad_with_graphs",
    "get_node_padding_mask",
    "get_edge_padding_mask",
    "get_graph_padding_mask",
    "segment_max",
    "segment_max_or_constant",
    "segment_min_or_constant",
    "segment_softmax",
    "segment_sum",
    "partition_softmax",
    "concatenated_args",
    "get_fully_connected_graph",
    "dynamically_batch",
    "with_zero_out_padding_outputs",
    "zero_out_padding",
    "sparse_matrix_to_graphs_tuple",
)


def is_available() -> bool:
    return importlib.util.find_spec("jraph") is not None


def require_jraph() -> Any:
    if importlib.util.find_spec("jraph") is None:
        raise ImportError(
            "jraph is required for `vertax.compat.jraph`. "
            "Install with `pip install jraph` or `vertax[compat]`."
        )
    return importlib.import_module("jraph")


def to_vertax(graph: Any, /, *, validate: bool = True) -> GraphIR:
    if isinstance(graph, GraphIR):
        if validate:
            graph.validate()
        return graph

    jraph = require_jraph()
    if isinstance(graph, jraph.GraphsTuple):
        return GraphIR.from_jraph_tuple(graph, validate=validate)

    raise TypeError("Expected GraphIR or jraph.GraphsTuple.")


def to_jraph(graph: Any, /) -> Any:
    if isinstance(graph, GraphIR):
        return graph.as_jraph_tuple()

    jraph = require_jraph()
    if isinstance(graph, jraph.GraphsTuple):
        return graph

    raise TypeError("Expected GraphIR or jraph.GraphsTuple.")


def __getattr__(name: str) -> Any:
    if name in _FORWARDED_SYMBOLS:
        module = require_jraph()
        return module.__dict__[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_FORWARDED_SYMBOLS))


__all__ = (
    "is_available",
    "require_jraph",
    "to_vertax",
    "to_jraph",
) + _FORWARDED_SYMBOLS
