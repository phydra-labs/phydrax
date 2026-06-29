from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ._graph import ensure_graph
from ._ir import GraphIR
from ._kernels import segment_sum


GraphFlow = Literal["source_to_target", "target_to_source"]


def _type_ids(type_ids: Any, /) -> jnp.ndarray:
    arr = jnp.asarray(type_ids)
    if arr.ndim == 0:
        arr = arr.reshape((1,))
    if arr.ndim != 1:
        raise ValueError(f"Graph type ids must be scalar or rank-1, got {arr.shape!r}.")
    if not jnp.issubdtype(arr.dtype, jnp.integer):
        raise TypeError("Graph type ids must be integer-valued.")
    return arr.astype(jnp.int32)


def _mapping_type_ids(payload: Any, type_key: str, kind: str, /) -> jnp.ndarray:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{kind} type ids require mapping-valued graph {kind}.")
    if type_key not in payload:
        raise KeyError(f"Graph {kind} payload does not contain type key {type_key!r}.")
    arr = jnp.asarray(payload[type_key])
    if arr.ndim == 2 and int(arr.shape[1]) == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(
            f"Graph {kind} type ids must have shape (n,) or (n, 1); got {arr.shape!r}."
        )
    if not jnp.issubdtype(arr.dtype, jnp.integer):
        raise TypeError(f"Graph {kind} type ids must use integer dtype.")
    return arr.astype(jnp.int32)


def node_type_ids(graph: GraphIR, /, *, type_key: str = "type") -> jnp.ndarray:
    """Return integer node type ids stored in mapping-valued graph nodes."""
    return _mapping_type_ids(ensure_graph(graph, validate=False).nodes, type_key, "nodes")


def edge_type_ids(graph: GraphIR, /, *, type_key: str = "type") -> jnp.ndarray:
    """Return integer edge type ids stored in mapping-valued graph edges."""
    return _mapping_type_ids(ensure_graph(graph, validate=False).edges, type_key, "edges")


def _type_indices(
    ids: jnp.ndarray,
    wanted: Any,
    /,
    *,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    wanted_arr = _type_ids(wanted)
    keep = jnp.any(ids[:, None] == wanted_arr[None, :], axis=1)
    if mask is not None:
        keep = keep & mask
    return jnp.asarray(np.nonzero(np.asarray(keep))[0], dtype=jnp.int32)


def node_type_indices(
    graph: GraphIR,
    type_ids: Any,
    /,
    *,
    type_key: str = "type",
) -> jnp.ndarray:
    """Return node indices whose type id is in `type_ids`."""
    graph = ensure_graph(graph, validate=False)
    return _type_indices(
        node_type_ids(graph, type_key=type_key),
        type_ids,
        mask=graph.node_mask,
    )


def edge_type_indices(
    graph: GraphIR,
    type_ids: Any,
    /,
    *,
    type_key: str = "type",
) -> jnp.ndarray:
    """Return edge indices whose type id is in `type_ids`."""
    graph = ensure_graph(graph, validate=False)
    return _type_indices(
        edge_type_ids(graph, type_key=type_key),
        type_ids,
        mask=graph.edge_mask,
    )


def typed_nodes_component(
    type_ids: Any,
    /,
    *,
    type_key: str = "type",
    name: str | None = None,
) -> Any:
    """Return a graph-domain component selecting nodes by type id."""
    from ..domain.graph import NodeType

    return NodeType(type_ids, type_key=type_key, name=name)


def typed_edges_component(
    type_ids: Any,
    /,
    *,
    type_key: str = "type",
    name: str | None = None,
) -> Any:
    """Return a graph-domain component selecting edges by type id."""
    from ..domain.graph import EdgeType

    return EdgeType(type_ids, type_key=type_key, name=name)


def _as_2d(name: str, value: Any, /) -> jnp.ndarray:
    arr = jnp.asarray(value)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be rank-1 or rank-2, got shape {arr.shape!r}.")
    return arr


def _node_features(graph: GraphIR, input_key: str | None, /) -> jnp.ndarray:
    if graph.nodes is None:
        raise ValueError("RelationalGraphConvolution requires node features.")
    if input_key is None:
        if isinstance(graph.nodes, Mapping):
            raise TypeError(
                "mapping-valued graph nodes require input_key for "
                "RelationalGraphConvolution."
            )
        return _as_2d("nodes", graph.nodes)
    if not isinstance(graph.nodes, Mapping):
        raise TypeError("input_key requires mapping-valued graph nodes.")
    if input_key not in graph.nodes:
        raise KeyError(f"Graph nodes do not contain input_key {input_key!r}.")
    return _as_2d(f"nodes[{input_key!r}]", graph.nodes[input_key])


def _with_node_output(graph: GraphIR, value: jnp.ndarray, output_key: str | None, /) -> Any:
    if output_key is None:
        return value
    nodes = {} if graph.nodes is None else dict(graph.nodes)
    nodes[output_key] = value
    return nodes


def _oriented_edges(graph: GraphIR, flow: GraphFlow, /) -> tuple[jnp.ndarray, jnp.ndarray]:
    if graph.senders is None or graph.receivers is None:
        raise ValueError("RelationalGraphConvolution requires explicit senders/receivers.")
    if flow == "source_to_target":
        return graph.senders, graph.receivers
    if flow == "target_to_source":
        return graph.receivers, graph.senders
    raise ValueError("flow must be 'source_to_target' or 'target_to_source'.")


def _edge_weights(graph: GraphIR, edge_weight_key: str | None, /) -> jnp.ndarray:
    if graph.senders is None:
        raise ValueError("RelationalGraphConvolution requires explicit edges.")
    if edge_weight_key is None:
        out = jnp.ones((graph.senders.shape[0],), dtype=float)
    else:
        if not isinstance(graph.edges, Mapping):
            raise TypeError("edge_weight_key requires mapping-valued graph edges.")
        if edge_weight_key not in graph.edges:
            raise KeyError(f"Graph edges do not contain edge_weight_key {edge_weight_key!r}.")
        out = jnp.asarray(graph.edges[edge_weight_key], dtype=float)
        if out.ndim == 2 and int(out.shape[1]) == 1:
            out = out[:, 0]
        if out.ndim != 1:
            raise ValueError("edge weights must have shape (n_edge,) or (n_edge, 1).")
    if graph.edge_mask is not None:
        out = out * graph.edge_mask.astype(out.dtype)
    return out


def _relation_transform(nodes: jnp.ndarray, relation_weights: jnp.ndarray) -> jnp.ndarray:
    if relation_weights.ndim == 1:
        return nodes * relation_weights[:, None]
    if relation_weights.ndim == 2:
        return nodes * relation_weights
    if relation_weights.ndim == 3:
        return jnp.einsum("ef,efo->eo", nodes, relation_weights)
    raise ValueError(
        "relation_weights must have shape (R,), (R, F), or (R, F, O)."
    )


def _self_transform(nodes: jnp.ndarray, self_weight: Any) -> jnp.ndarray:
    weight = jnp.asarray(self_weight, dtype=nodes.dtype)
    if weight.ndim == 0:
        return nodes * weight
    if weight.ndim == 1:
        return nodes * weight
    if weight.ndim == 2:
        return nodes @ weight
    raise ValueError("self_weight must be scalar, rank-1, or rank-2.")


def _mask_nodes(nodes: jnp.ndarray, mask: jnp.ndarray | None, /) -> jnp.ndarray:
    if mask is None:
        return nodes
    return nodes * mask.astype(nodes.dtype)[:, None]


class RelationalGraphConvolution(eqx.Module):
    """Relation-specific graph convolution for typed/heterogeneous graphs.

    Edge type ids choose a relation weight for each message. Relation weights may
    be scalar per relation `(R,)`, channel-wise `(R, F)`, or dense matrices
    `(R, F, O)`.
    """

    relation_weights: jnp.ndarray
    self_weight: Any
    edge_type_key: str = eqx.field(static=True)
    edge_weight_key: str | None = eqx.field(static=True)
    input_key: str | None = eqx.field(static=True)
    output_key: str | None = eqx.field(static=True)
    flow: GraphFlow = eqx.field(static=True)
    normalize: bool = eqx.field(static=True)

    def __init__(
        self,
        relation_weights: Any,
        /,
        *,
        self_weight: Any = None,
        edge_type_key: str = "type",
        edge_weight_key: str | None = None,
        input_key: str | None = None,
        output_key: str | None = None,
        flow: GraphFlow = "source_to_target",
        normalize: bool = False,
    ):
        weights = jnp.asarray(relation_weights, dtype=float)
        if weights.ndim not in (1, 2, 3):
            raise ValueError("relation_weights must have shape (R,), (R, F), or (R, F, O).")
        if int(weights.shape[0]) <= 0:
            raise ValueError("relation_weights must contain at least one relation.")
        self.relation_weights = weights
        self.self_weight = None if self_weight is None else jnp.asarray(self_weight, dtype=float)
        self.edge_type_key = str(edge_type_key)
        self.edge_weight_key = edge_weight_key
        self.input_key = input_key
        self.output_key = output_key
        self.flow = flow
        self.normalize = bool(normalize)

    def __call__(self, graph: GraphIR) -> GraphIR:
        graph = ensure_graph(graph, validate=False)
        nodes = _node_features(graph, self.input_key)
        source, target = _oriented_edges(graph, self.flow)
        edge_types = edge_type_ids(graph, type_key=self.edge_type_key)

        edge_scale = _edge_weights(graph, self.edge_weight_key)
        if self.normalize:
            n = int(nodes.shape[0])
            keys = edge_types * n + target
            degree = segment_sum(
                edge_scale,
                keys,
                int(self.relation_weights.shape[0]) * n,
            )[keys]
            edge_scale = jnp.where(degree > 0, edge_scale / degree, 0.0)

        messages = _relation_transform(nodes[source], self.relation_weights[edge_types])
        messages = messages * edge_scale[:, None]
        out = segment_sum(messages, target, int(nodes.shape[0]))
        if self.self_weight is not None:
            out = out + _self_transform(nodes, self.self_weight)
        out = _mask_nodes(out, graph.node_mask)
        return graph.replace(nodes=_with_node_output(graph, out, self.output_key), validate=False)


__all__ = [
    "GraphFlow",
    "RelationalGraphConvolution",
    "edge_type_ids",
    "edge_type_indices",
    "node_type_ids",
    "node_type_indices",
    "typed_edges_component",
    "typed_nodes_component",
]
