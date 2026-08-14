from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ._graph import ensure_graph
from ._ir import GraphIR
from ._kernels import segment_sum
from ._typed import edge_type_ids, node_type_indices


def _as_int_vector(name: str, value: Any, /) -> jnp.ndarray:
    arr = jnp.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f"{name} must have shape (n,), got {arr.shape!r}.")
    if not jnp.issubdtype(arr.dtype, jnp.integer):
        raise TypeError(f"{name} must use integer dtype.")
    return arr.astype(jnp.int32)


def _infer_count(name: str, indices: jnp.ndarray, value: int | None, /) -> int:
    if value is not None:
        n = int(value)
        if n < 0:
            raise ValueError(f"{name} must be non-negative.")
    elif int(indices.shape[0]) == 0:
        raise ValueError(f"{name} must be provided when there are no incidences.")
    else:
        n = int(jnp.max(indices)) + 1
    if int(indices.shape[0]) > 0:
        idx_np = np.asarray(indices)
        if np.any(idx_np < 0) or np.any(idx_np >= n):
            raise ValueError(f"{name} incidence indices must be in [0, {n}).")
    return n


def _hyperedges_to_incidence(
    hyperedges: Sequence[Any],
    /,
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    node_parts: list[int] = []
    hyperedge_parts: list[int] = []
    for hyperedge_index, nodes in enumerate(hyperedges):
        arr = np.asarray(nodes, dtype=np.int32).reshape((-1,))
        if np.any(arr < 0):
            raise ValueError("Hypergraph node indices must be non-negative.")
        node_parts.extend(int(i) for i in arr.tolist())
        hyperedge_parts.extend([hyperedge_index] * int(arr.shape[0]))
    return (
        jnp.asarray(node_parts, dtype=jnp.int32),
        jnp.asarray(hyperedge_parts, dtype=jnp.int32),
        len(hyperedges),
    )


def _combine_features(
    name: str,
    first: Any | None,
    second: Any | None,
    n_first: int,
    n_second: int,
    /,
) -> jnp.ndarray | None:
    if first is None and second is None:
        return None
    if first is None:
        second_arr = jnp.asarray(second, dtype=float)
        if int(second_arr.shape[0]) != n_second:
            raise ValueError(f"{name} hyperedge feature leading axis must be {n_second}.")
        first_arr = jnp.zeros((n_first,) + second_arr.shape[1:], dtype=second_arr.dtype)
    elif second is None:
        first_arr = jnp.asarray(first, dtype=float)
        if int(first_arr.shape[0]) != n_first:
            raise ValueError(f"{name} node feature leading axis must be {n_first}.")
        second_arr = jnp.zeros((n_second,) + first_arr.shape[1:], dtype=first_arr.dtype)
    else:
        first_arr = jnp.asarray(first, dtype=float)
        second_arr = jnp.asarray(second, dtype=float)
        if int(first_arr.shape[0]) != n_first:
            raise ValueError(f"{name} node feature leading axis must be {n_first}.")
        if int(second_arr.shape[0]) != n_second:
            raise ValueError(f"{name} hyperedge feature leading axis must be {n_second}.")
        if first_arr.shape[1:] != second_arr.shape[1:]:
            raise ValueError(
                f"{name} node and hyperedge features must share trailing shape."
            )
    return jnp.concatenate([first_arr, second_arr], axis=0)


class HypergraphBipartiteGraph(eqx.Module):
    """A hypergraph represented as a typed bipartite `GraphIR`."""

    graph: GraphIR
    original_nodes: jnp.ndarray
    hyperedge_nodes: jnp.ndarray
    incidence_edges: jnp.ndarray
    reverse_incidence_edges: jnp.ndarray
    original_node_type: int = eqx.field(static=True)
    hyperedge_node_type: int = eqx.field(static=True)
    incidence_edge_type: int = eqx.field(static=True)
    reverse_incidence_edge_type: int = eqx.field(static=True)

    def __init__(
        self,
        graph: GraphIR,
        /,
        *,
        original_nodes: Any,
        hyperedge_nodes: Any,
        incidence_edges: Any,
        reverse_incidence_edges: Any,
        original_node_type: int,
        hyperedge_node_type: int,
        incidence_edge_type: int,
        reverse_incidence_edge_type: int,
    ):
        self.graph = graph
        self.original_nodes = jnp.asarray(original_nodes, dtype=jnp.int32)
        self.hyperedge_nodes = jnp.asarray(hyperedge_nodes, dtype=jnp.int32)
        self.incidence_edges = jnp.asarray(incidence_edges, dtype=jnp.int32)
        self.reverse_incidence_edges = jnp.asarray(
            reverse_incidence_edges, dtype=jnp.int32
        )
        self.original_node_type = int(original_node_type)
        self.hyperedge_node_type = int(hyperedge_node_type)
        self.incidence_edge_type = int(incidence_edge_type)
        self.reverse_incidence_edge_type = int(reverse_incidence_edge_type)

    def original_nodes_component(self):
        from ..domain.graph import NodeType

        return NodeType(self.original_node_type)

    def hyperedge_nodes_component(self):
        from ..domain.graph import NodeType

        return NodeType(self.hyperedge_node_type)

    def incidence_edges_component(self):
        from ..domain.graph import EdgeType

        return EdgeType(self.incidence_edge_type)

    def reverse_incidence_edges_component(self):
        from ..domain.graph import EdgeType

        return EdgeType(self.reverse_incidence_edge_type)


def incidence_to_bipartite_graph(
    node_indices: Any,
    hyperedge_indices: Any,
    /,
    *,
    num_nodes: int | None = None,
    num_hyperedges: int | None = None,
    node_features: Any | None = None,
    hyperedge_features: Any | None = None,
    incidence_weight: Any | None = None,
    globals: Any = None,
    add_reverse_edges: bool = True,
    original_node_type: int = 0,
    hyperedge_node_type: int = 1,
    incidence_edge_type: int = 0,
    reverse_incidence_edge_type: int = 1,
    validate: bool = True,
) -> HypergraphBipartiteGraph:
    """Represent hypergraph incidence as a typed bipartite `GraphIR`."""
    node_idx = _as_int_vector("node_indices", node_indices)
    hyper_idx = _as_int_vector("hyperedge_indices", hyperedge_indices)
    if int(node_idx.shape[0]) != int(hyper_idx.shape[0]):
        raise ValueError("node_indices and hyperedge_indices must have the same length.")
    n_node = _infer_count("num_nodes", node_idx, num_nodes)
    n_hyperedge = _infer_count("num_hyperedges", hyper_idx, num_hyperedges)
    n_total_node = n_node + n_hyperedge
    hyper_nodes = n_node + hyper_idx

    senders = node_idx
    receivers = hyper_nodes
    edge_type = jnp.full(
        (int(node_idx.shape[0]),), int(incidence_edge_type), dtype=jnp.int32
    )
    edge_node_index = node_idx
    edge_hyperedge_index = hyper_idx
    if add_reverse_edges:
        senders = jnp.concatenate([senders, hyper_nodes], axis=0)
        receivers = jnp.concatenate([receivers, node_idx], axis=0)
        edge_type = jnp.concatenate(
            [
                edge_type,
                jnp.full(
                    (int(node_idx.shape[0]),),
                    int(reverse_incidence_edge_type),
                    dtype=jnp.int32,
                ),
            ],
            axis=0,
        )
        edge_node_index = jnp.concatenate([edge_node_index, node_idx], axis=0)
        edge_hyperedge_index = jnp.concatenate([edge_hyperedge_index, hyper_idx], axis=0)

    feature = _combine_features(
        "hypergraph",
        node_features,
        hyperedge_features,
        n_node,
        n_hyperedge,
    )
    nodes = {
        "type": jnp.concatenate(
            [
                jnp.full((n_node,), int(original_node_type), dtype=jnp.int32),
                jnp.full((n_hyperedge,), int(hyperedge_node_type), dtype=jnp.int32),
            ],
            axis=0,
        ),
        "is_original_node": jnp.concatenate(
            [jnp.ones((n_node,), dtype=bool), jnp.zeros((n_hyperedge,), dtype=bool)],
            axis=0,
        ),
        "is_hyperedge": jnp.concatenate(
            [jnp.zeros((n_node,), dtype=bool), jnp.ones((n_hyperedge,), dtype=bool)],
            axis=0,
        ),
        "node_index": jnp.concatenate(
            [
                jnp.arange(n_node, dtype=jnp.int32),
                jnp.full((n_hyperedge,), -1, dtype=jnp.int32),
            ],
            axis=0,
        ),
        "hyperedge_index": jnp.concatenate(
            [
                jnp.full((n_node,), -1, dtype=jnp.int32),
                jnp.arange(n_hyperedge, dtype=jnp.int32),
            ],
            axis=0,
        ),
    }
    if feature is not None:
        nodes["features"] = feature

    if incidence_weight is None:
        weight = jnp.ones((int(node_idx.shape[0]),), dtype=float)
    else:
        weight = jnp.asarray(incidence_weight, dtype=float).reshape((-1,))
        if int(weight.shape[0]) != int(node_idx.shape[0]):
            raise ValueError("incidence_weight must match the number of incidences.")
    if add_reverse_edges:
        weight = jnp.concatenate([weight, weight], axis=0)
    edges = {
        "type": edge_type,
        "incidence_weight": weight,
        "node_index": edge_node_index,
        "hyperedge_index": edge_hyperedge_index,
    }
    n_edge = int(senders.shape[0])
    graph = GraphIR(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        globals=globals,
        n_node=jnp.asarray([n_total_node], dtype=jnp.int32),
        n_edge=jnp.asarray([n_edge], dtype=jnp.int32),
        validate=validate,
    )
    n_incidence = int(node_idx.shape[0])
    return HypergraphBipartiteGraph(
        graph,
        original_nodes=jnp.arange(n_node, dtype=jnp.int32),
        hyperedge_nodes=n_node + jnp.arange(n_hyperedge, dtype=jnp.int32),
        incidence_edges=jnp.arange(n_incidence, dtype=jnp.int32),
        reverse_incidence_edges=jnp.arange(n_incidence, n_edge, dtype=jnp.int32),
        original_node_type=original_node_type,
        hyperedge_node_type=hyperedge_node_type,
        incidence_edge_type=incidence_edge_type,
        reverse_incidence_edge_type=reverse_incidence_edge_type,
    )


def hypergraph_to_bipartite_graph(
    hyperedges: Sequence[Any],
    /,
    *,
    num_nodes: int | None = None,
    node_features: Any | None = None,
    hyperedge_features: Any | None = None,
    incidence_weight: Any | None = None,
    globals: Any = None,
    add_reverse_edges: bool = True,
    original_node_type: int = 0,
    hyperedge_node_type: int = 1,
    incidence_edge_type: int = 0,
    reverse_incidence_edge_type: int = 1,
    validate: bool = True,
) -> HypergraphBipartiteGraph:
    """Convert a sequence of hyperedge node lists into a bipartite `GraphIR`."""
    node_idx, hyper_idx, n_hyperedge = _hyperedges_to_incidence(hyperedges)
    return incidence_to_bipartite_graph(
        node_idx,
        hyper_idx,
        num_nodes=num_nodes,
        num_hyperedges=n_hyperedge,
        node_features=node_features,
        hyperedge_features=hyperedge_features,
        incidence_weight=incidence_weight,
        globals=globals,
        add_reverse_edges=add_reverse_edges,
        original_node_type=original_node_type,
        hyperedge_node_type=hyperedge_node_type,
        incidence_edge_type=incidence_edge_type,
        reverse_incidence_edge_type=reverse_incidence_edge_type,
        validate=validate,
    )


def _node_features(graph: GraphIR, input_key: str | None, /) -> jnp.ndarray:
    if graph.nodes is None:
        raise ValueError("HypergraphConvolution requires node features.")
    if input_key is None:
        if isinstance(graph.nodes, Mapping):
            raise TypeError("mapping-valued hypergraph nodes require input_key.")
        arr = jnp.asarray(graph.nodes, dtype=float)
    else:
        if not isinstance(graph.nodes, Mapping):
            raise TypeError("input_key requires mapping-valued hypergraph nodes.")
        if input_key not in graph.nodes:
            raise KeyError(f"Graph nodes do not contain input_key {input_key!r}.")
        arr = jnp.asarray(graph.nodes[input_key], dtype=float)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError("HypergraphConvolution node features must be rank-1 or rank-2.")
    return arr


def _edge_weight(graph: GraphIR, edge_weight_key: str | None, /) -> jnp.ndarray:
    if graph.senders is None:
        raise ValueError("HypergraphConvolution requires explicit graph edges.")
    if edge_weight_key is None:
        out = jnp.ones((graph.senders.shape[0],), dtype=float)
    else:
        if not isinstance(graph.edges, Mapping):
            raise TypeError("edge_weight_key requires mapping-valued graph edges.")
        if edge_weight_key not in graph.edges:
            raise KeyError(
                f"Graph edges do not contain edge_weight_key {edge_weight_key!r}."
            )
        out = jnp.asarray(graph.edges[edge_weight_key], dtype=float).reshape((-1,))
    if graph.edge_mask is not None:
        out = out * graph.edge_mask.astype(out.dtype)
    return out


def _with_node_output(
    graph: GraphIR, value: jnp.ndarray, output_key: str | None, /
) -> Any:
    if output_key is None:
        return value
    nodes = {} if graph.nodes is None else dict(graph.nodes)
    nodes[output_key] = value
    return nodes


def _mask_nodes(nodes: jnp.ndarray, graph: GraphIR, /) -> jnp.ndarray:
    if graph.node_mask is None:
        return nodes
    return nodes * graph.node_mask.astype(nodes.dtype)[:, None]


class HypergraphConvolution(eqx.Module):
    """Two-stage hypergraph convolution over a bipartite hypergraph graph."""

    input_key: str | None = eqx.field(static=True)
    output_key: str | None = eqx.field(static=True)
    node_type_key: str = eqx.field(static=True)
    edge_type_key: str = eqx.field(static=True)
    edge_weight_key: str | None = eqx.field(static=True)
    original_node_type: int = eqx.field(static=True)
    hyperedge_node_type: int = eqx.field(static=True)
    incidence_edge_type: int = eqx.field(static=True)
    reverse_incidence_edge_type: int = eqx.field(static=True)
    normalize_hyperedges: bool = eqx.field(static=True)
    normalize_nodes: bool = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        input_key: str | None = "features",
        output_key: str | None = None,
        node_type_key: str = "type",
        edge_type_key: str = "type",
        edge_weight_key: str | None = "incidence_weight",
        original_node_type: int = 0,
        hyperedge_node_type: int = 1,
        incidence_edge_type: int = 0,
        reverse_incidence_edge_type: int = 1,
        normalize_hyperedges: bool = True,
        normalize_nodes: bool = True,
    ):
        self.input_key = input_key
        self.output_key = output_key
        self.node_type_key = str(node_type_key)
        self.edge_type_key = str(edge_type_key)
        self.edge_weight_key = edge_weight_key
        self.original_node_type = int(original_node_type)
        self.hyperedge_node_type = int(hyperedge_node_type)
        self.incidence_edge_type = int(incidence_edge_type)
        self.reverse_incidence_edge_type = int(reverse_incidence_edge_type)
        self.normalize_hyperedges = bool(normalize_hyperedges)
        self.normalize_nodes = bool(normalize_nodes)

    def __call__(self, graph: GraphIR) -> GraphIR:
        graph = ensure_graph(graph, validate=False)
        if graph.senders is None or graph.receivers is None:
            raise ValueError("HypergraphConvolution requires explicit senders/receivers.")
        x = _node_features(graph, self.input_key)
        n = int(x.shape[0])
        weights = _edge_weight(graph, self.edge_weight_key)
        edge_types = edge_type_ids(graph, type_key=self.edge_type_key)
        original = node_type_indices(
            graph, self.original_node_type, type_key=self.node_type_key
        )
        hyper = node_type_indices(
            graph, self.hyperedge_node_type, type_key=self.node_type_key
        )
        is_incidence = edge_types == self.incidence_edge_type
        is_reverse = edge_types == self.reverse_incidence_edge_type

        incidence_weight = jnp.where(is_incidence, weights, 0.0)
        hyper_messages = x[graph.senders] * incidence_weight[:, None]
        hyper_state = segment_sum(hyper_messages, graph.receivers, n)
        if self.normalize_hyperedges:
            hyper_degree = segment_sum(incidence_weight, graph.receivers, n)
            hyper_scale = jnp.where(hyper_degree > 0, 1.0 / hyper_degree, 0.0)
            hyper_state = hyper_state * hyper_scale[:, None]

        reverse_weight = jnp.where(is_reverse, weights, 0.0)
        node_messages = hyper_state[graph.senders] * reverse_weight[:, None]
        out = segment_sum(node_messages, graph.receivers, n)
        if self.normalize_nodes:
            node_degree = segment_sum(reverse_weight, graph.receivers, n)
            node_scale = jnp.where(node_degree > 0, 1.0 / node_degree, 0.0)
            out = out * node_scale[:, None]
        original_mask = jnp.zeros((n,), dtype=bool).at[original].set(True)
        hyper_mask = jnp.zeros((n,), dtype=bool).at[hyper].set(True)
        combined = jnp.where(original_mask[:, None], out, jnp.zeros_like(out))
        combined = jnp.where(hyper_mask[:, None], hyper_state, combined)
        combined = _mask_nodes(combined, graph)
        return graph.replace(
            nodes=_with_node_output(graph, combined, self.output_key), validate=False
        )


__all__ = [
    "HypergraphBipartiteGraph",
    "HypergraphConvolution",
    "hypergraph_to_bipartite_graph",
    "incidence_to_bipartite_graph",
]
