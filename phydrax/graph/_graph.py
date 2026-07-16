from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Mapping
from typing import Any

from ._ir import GraphIR


Graph = GraphIR


def is_graph_like(value: Any, /) -> bool:
    if isinstance(value, GraphIR):
        return True
    if importlib.util.find_spec("jraph") is None:
        return False
    jraph = importlib.import_module("jraph")
    return isinstance(value, jraph.GraphsTuple)


def ensure_graph(value: Any, /, *, validate: bool = True) -> GraphIR:
    if isinstance(value, GraphIR):
        if validate:
            value.validate()
        return value

    if importlib.util.find_spec("jraph") is not None:
        jraph = importlib.import_module("jraph")
        if isinstance(value, jraph.GraphsTuple):
            return GraphIR.from_jraph_tuple(value, validate=validate)

    raise TypeError("Expected GraphIR or jraph.GraphsTuple input.")


def graph_counts(graph: GraphIR, /) -> Mapping[str, int]:
    graph.validate()
    return {
        "n_graph": graph.num_graphs,
        "n_node": graph.num_nodes,
        "n_edge": graph.num_edges,
    }


__all__ = [
    "Graph",
    "GraphIR",
    "ensure_graph",
    "is_graph_like",
    "graph_counts",
]
