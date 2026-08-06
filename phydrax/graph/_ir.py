from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Sequence
from typing import Any, Literal

import equinox as eqx
import jax.core as jcore
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from .._trainable import NonTrainableState
from ..sparse import EdgeRelation


_MISSING = object()


def _contains_tracer(tree: Any, /) -> bool:
    for leaf in jtu.tree_leaves(tree):
        if isinstance(leaf, jcore.Tracer):
            return True
    return False


def _ensure_int_vector(name: str, value: Any, /) -> jnp.ndarray:
    arr = jnp.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be rank-1, got shape {arr.shape!r}.")
    if not jnp.issubdtype(arr.dtype, jnp.integer):
        raise TypeError(f"{name} must be integer dtype, got {arr.dtype!r}.")
    return arr.astype(jnp.int32)


def _leaf_leading_size(tree: Any, /) -> int | None:
    if tree is None:
        return None
    leaves = jtu.tree_leaves(tree)
    if not leaves:
        return None
    first = jnp.asarray(leaves[0])
    if first.ndim == 0:
        raise ValueError("Feature leaves must have rank >= 1.")
    size = int(first.shape[0])
    for leaf in leaves[1:]:
        arr = jnp.asarray(leaf)
        if arr.ndim == 0:
            raise ValueError("Feature leaves must have rank >= 1.")
        if int(arr.shape[0]) != size:
            raise ValueError(
                "All leaves in a feature pytree must share the same leading axis size."
            )
    return size


def _concat_trees(values: Sequence[Any], /) -> Any:
    if len(values) == 1:
        return values[0]
    return jtu.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *values)


def _split_tree(tree: Any, offsets: jnp.ndarray, /) -> list[Any]:
    if tree is None:
        return []
    leaves, treedef = jtu.tree_flatten(tree)
    if len(leaves) == 0:
        return [tree]

    split_leaves = [jnp.split(jnp.asarray(leaf), offsets, axis=0) for leaf in leaves]
    n_parts = len(split_leaves[0])
    return [
        jtu.tree_unflatten(treedef, [parts[i] for parts in split_leaves])
        for i in range(n_parts)
    ]


def _require_uniform_presence(name: str, values: Sequence[Any], /) -> None:
    present = [v is not None for v in values]
    if not all(present) and any(present):
        raise ValueError(f"All graphs must either have `{name}` or all omit it.")


class GraphIR(eqx.Module, NonTrainableState):
    """Canonical execution representation for sparse batched graphs.

    Semantics follow a counts-first sparse layout:

    - `n_node[g]`: number of nodes in graph g
    - `n_edge[g]`: number of edges in graph g
    - `senders[k]`, `receivers[k]`: absolute node indices for edge k
    - node/edge/global features may be arbitrary pytrees with matching leading axes
    """

    nodes: Any
    edges: Any
    senders: jnp.ndarray | None
    receivers: jnp.ndarray | None
    globals: Any
    n_node: jnp.ndarray
    n_edge: jnp.ndarray
    node_mask: jnp.ndarray | None
    edge_mask: jnp.ndarray | None
    graph_mask: jnp.ndarray | None

    def __init__(
        self,
        *,
        nodes: Any = None,
        edges: Any = None,
        senders: Any | None = None,
        receivers: Any | None = None,
        globals: Any = None,
        n_node: Any,
        n_edge: Any,
        node_mask: Any | None = None,
        edge_mask: Any | None = None,
        graph_mask: Any | None = None,
        validate: bool = True,
    ):
        self.nodes = nodes
        self.edges = edges
        self.senders = None if senders is None else _ensure_int_vector("senders", senders)
        self.receivers = (
            None if receivers is None else _ensure_int_vector("receivers", receivers)
        )
        self.globals = globals
        self.n_node = _ensure_int_vector("n_node", n_node)
        self.n_edge = _ensure_int_vector("n_edge", n_edge)
        self.node_mask = None if node_mask is None else jnp.asarray(node_mask, dtype=bool)
        self.edge_mask = None if edge_mask is None else jnp.asarray(edge_mask, dtype=bool)
        self.graph_mask = (
            None if graph_mask is None else jnp.asarray(graph_mask, dtype=bool)
        )
        if validate:
            self.validate()

    @property
    def num_graphs(self) -> int:
        return int(self.n_node.shape[0])

    @property
    def num_nodes(self) -> int:
        if _contains_tracer(self.n_node):
            raise RuntimeError(
                "`GraphIR.num_nodes` is only available in eager mode. "
                "In jitted code, infer total node count from feature/index shapes."
            )
        return int(np.asarray(self.n_node).sum())

    @property
    def num_edges(self) -> int:
        if _contains_tracer(self.n_edge):
            raise RuntimeError(
                "`GraphIR.num_edges` is only available in eager mode. "
                "In jitted code, infer total edge count from feature/index shapes."
            )
        return int(np.asarray(self.n_edge).sum())

    @property
    def edge_index(self) -> jnp.ndarray:
        if self.senders is None or self.receivers is None:
            return jnp.zeros((2, 0), dtype=jnp.int32)
        return jnp.stack([self.senders, self.receivers], axis=0)

    def edge_relation(
        self,
        /,
        *,
        node_count: int | None = None,
        flow: Literal["source_to_target", "target_to_source"] = "source_to_target",
    ) -> EdgeRelation:
        """Return the graph routes as a sparse relation without copying payloads."""
        if self.senders is None or self.receivers is None:
            raise ValueError("GraphIR has no explicit edge relation.")
        if flow == "source_to_target":
            source, target = self.senders, self.receivers
        elif flow == "target_to_source":
            source, target = self.receivers, self.senders
        else:
            raise ValueError("flow must be 'source_to_target' or 'target_to_source'.")

        if node_count is not None:
            count = int(node_count)
        elif self.node_mask is not None:
            count = int(self.node_mask.shape[0])
        else:
            payload_count = _leaf_leading_size(self.nodes)
            if payload_count is not None:
                count = payload_count
            elif _contains_tracer(self.n_node):
                raise ValueError(
                    "node_count is required for a payload-free traced GraphIR."
                )
            else:
                count = int(np.asarray(self.n_node).sum())
        return EdgeRelation(
            source,
            target,
            source_size=count,
            target_size=count,
            valid=self.edge_mask,
        )

    def validate(self, *, strict: bool = True) -> None:
        if self.n_node.shape != self.n_edge.shape:
            raise ValueError("`n_node` and `n_edge` must have identical shapes.")

        if (self.senders is None) != (self.receivers is None):
            raise ValueError(
                "`senders` and `receivers` must be both None or both arrays."
            )

        if self.senders is not None and self.receivers is not None:
            if self.senders.shape != self.receivers.shape:
                raise ValueError("`senders` and `receivers` must have identical shapes.")

        node_size = _leaf_leading_size(self.nodes) if self.nodes is not None else None
        edge_size = _leaf_leading_size(self.edges) if self.edges is not None else None
        global_size = (
            _leaf_leading_size(self.globals) if self.globals is not None else None
        )

        if self.graph_mask is not None:
            if int(self.graph_mask.shape[0]) != self.num_graphs:
                raise ValueError("`graph_mask` length must match number of graphs.")
        if self.nodes is not None:
            if node_size is None:
                raise ValueError(
                    "Node feature tree must contain at least one rank >= 1 leaf."
                )
        if self.edges is not None:
            if edge_size is None:
                raise ValueError(
                    "Edge feature tree must contain at least one rank >= 1 leaf."
                )
        if self.globals is not None:
            if global_size is None:
                raise ValueError(
                    "Global feature tree must contain at least one rank >= 1 leaf."
                )

        if self.node_mask is not None:
            if self.node_mask.ndim != 1:
                raise ValueError("`node_mask` must be rank-1.")
            if node_size is not None and int(self.node_mask.shape[0]) != node_size:
                raise ValueError(
                    "`node_mask` length must match node feature leading size."
                )
        if self.edge_mask is not None:
            if self.edge_mask.ndim != 1:
                raise ValueError("`edge_mask` must be rank-1.")
            if self.senders is not None and int(self.edge_mask.shape[0]) != int(
                self.senders.shape[0]
            ):
                raise ValueError(
                    "`edge_mask` length must match senders/receivers length."
                )
        if self.graph_mask is not None and self.graph_mask.ndim != 1:
            raise ValueError("`graph_mask` must be rank-1.")

        if _contains_tracer((self.n_node, self.n_edge, self.senders, self.receivers)):
            if strict and self.senders is not None and edge_size is not None:
                if edge_size != int(self.senders.shape[0]):
                    raise ValueError(
                        "Edge feature leading size must equal sender/receiver length."
                    )
            if strict and global_size is not None and global_size != self.num_graphs:
                raise ValueError("Global feature leading size must equal `len(n_node)`.")
            return

        n_node_np = np.asarray(self.n_node)
        n_edge_np = np.asarray(self.n_edge)
        if np.any(n_node_np < 0):
            raise ValueError("`n_node` must be non-negative.")
        if np.any(n_edge_np < 0):
            raise ValueError("`n_edge` must be non-negative.")

        n_nodes = int(n_node_np.sum())
        n_edges = int(n_edge_np.sum())

        if self.senders is None or self.receivers is None:
            if strict and n_edges != 0:
                raise ValueError(
                    "Edge count is non-zero but senders/receivers are missing."
                )
        else:
            if strict and int(self.senders.shape[0]) != n_edges:
                raise ValueError("Edge index length must match `sum(n_edge)`.")
            if int(self.senders.size) > 0:
                senders_np = np.asarray(self.senders)
                receivers_np = np.asarray(self.receivers)
                if np.any(senders_np < 0) or np.any(receivers_np < 0):
                    raise ValueError("Edge indices must be non-negative.")
                if n_nodes == 0:
                    raise ValueError("Edges are present but node count is zero.")
                if np.any(senders_np >= n_nodes) or np.any(receivers_np >= n_nodes):
                    raise ValueError("Edge indices must be in [0, sum(n_node)).")
                node_start = 0
                edge_start = 0
                for graph_index, (node_count, edge_count) in enumerate(
                    zip(n_node_np, n_edge_np, strict=True)
                ):
                    node_end = node_start + int(node_count)
                    edge_end = edge_start + int(edge_count)
                    for endpoint_name, endpoints in (
                        ("sender", senders_np),
                        ("receiver", receivers_np),
                    ):
                        graph_endpoints = endpoints[edge_start:edge_end]
                        offending = np.flatnonzero(
                            (graph_endpoints < node_start) | (graph_endpoints >= node_end)
                        )
                        if offending.size:
                            local_edge = int(offending[0])
                            edge_position = edge_start + local_edge
                            endpoint = int(graph_endpoints[local_edge])
                            raise ValueError(
                                f"Graph {graph_index} edge position {edge_position} "
                                f"has {endpoint_name} {endpoint}; required node interval "
                                f"[{node_start}, {node_end})."
                            )
                    node_start = node_end
                    edge_start = edge_end

        if strict and node_size is not None and node_size != n_nodes:
            raise ValueError("Node feature leading size must equal `sum(n_node)`.")
        if strict and edge_size is not None and edge_size != n_edges:
            raise ValueError("Edge feature leading size must equal `sum(n_edge)`.")
        if strict and global_size is not None and global_size != self.num_graphs:
            raise ValueError("Global feature leading size must equal `len(n_node)`.")

    def replace(
        self,
        *,
        nodes: Any = _MISSING,
        edges: Any = _MISSING,
        senders: Any = _MISSING,
        receivers: Any = _MISSING,
        globals: Any = _MISSING,
        n_node: Any = _MISSING,
        n_edge: Any = _MISSING,
        node_mask: Any = _MISSING,
        edge_mask: Any = _MISSING,
        graph_mask: Any = _MISSING,
        validate: bool = False,
    ) -> "GraphIR":
        return GraphIR(
            nodes=self.nodes if nodes is _MISSING else nodes,
            edges=self.edges if edges is _MISSING else edges,
            senders=self.senders if senders is _MISSING else senders,
            receivers=self.receivers if receivers is _MISSING else receivers,
            globals=self.globals if globals is _MISSING else globals,
            n_node=self.n_node if n_node is _MISSING else n_node,
            n_edge=self.n_edge if n_edge is _MISSING else n_edge,
            node_mask=self.node_mask if node_mask is _MISSING else node_mask,
            edge_mask=self.edge_mask if edge_mask is _MISSING else edge_mask,
            graph_mask=self.graph_mask if graph_mask is _MISSING else graph_mask,
            validate=validate,
        )

    def as_jraph_tuple(self) -> Any:
        if importlib.util.find_spec("jraph") is None:
            raise ImportError(
                "jraph is required for `phydrax.graph.GraphIR.as_jraph_tuple`; "
                "install it with `pip install jraph`."
            )
        jraph = importlib.import_module("jraph")
        return jraph.GraphsTuple(
            nodes=self.nodes,
            edges=self.edges,
            senders=self.senders,
            receivers=self.receivers,
            globals=self.globals,
            n_node=self.n_node,
            n_edge=self.n_edge,
        )

    @classmethod
    def from_jraph_tuple(cls, graph: Any, /, *, validate: bool = True) -> "GraphIR":
        return cls(
            nodes=graph.nodes,
            edges=graph.edges,
            senders=graph.senders,
            receivers=graph.receivers,
            globals=graph.globals,
            n_node=graph.n_node,
            n_edge=graph.n_edge,
            validate=validate,
        )


def batch_graphs(graphs: Sequence[GraphIR], /, *, validate: bool = True) -> GraphIR:
    if len(graphs) == 0:
        raise ValueError("`batch_graphs` requires at least one graph.")

    _require_uniform_presence("nodes", [g.nodes for g in graphs])
    _require_uniform_presence("edges", [g.edges for g in graphs])
    _require_uniform_presence("globals", [g.globals for g in graphs])
    _require_uniform_presence("node_mask", [g.node_mask for g in graphs])
    _require_uniform_presence("edge_mask", [g.edge_mask for g in graphs])
    _require_uniform_presence("graph_mask", [g.graph_mask for g in graphs])

    n_node = jnp.concatenate([g.n_node for g in graphs], axis=0)
    n_edge = jnp.concatenate([g.n_edge for g in graphs], axis=0)

    nodes = None
    if graphs[0].nodes is not None:
        nodes = _concat_trees([g.nodes for g in graphs])

    edges = None
    if graphs[0].edges is not None:
        edges = _concat_trees([g.edges for g in graphs])

    globals_ = None
    if graphs[0].globals is not None:
        globals_ = _concat_trees([g.globals for g in graphs])
    node_masks = tuple(graph.node_mask for graph in graphs)
    node_mask = (
        None
        if node_masks[0] is None
        else jnp.concatenate(
            tuple(mask for mask in node_masks if mask is not None), axis=0
        )
    )
    edge_masks = tuple(graph.edge_mask for graph in graphs)
    edge_mask = (
        None
        if edge_masks[0] is None
        else jnp.concatenate(
            tuple(mask for mask in edge_masks if mask is not None), axis=0
        )
    )
    graph_masks = tuple(graph.graph_mask for graph in graphs)
    graph_mask = (
        None
        if graph_masks[0] is None
        else jnp.concatenate(
            tuple(mask for mask in graph_masks if mask is not None),
            axis=0,
        )
    )

    offsets = [0]
    running = 0
    for g in graphs[:-1]:
        running += g.num_nodes
        offsets.append(running)

    senders = None
    receivers = None
    if graphs[0].senders is not None and graphs[0].receivers is not None:
        sender_parts = []
        receiver_parts = []
        for offset, graph in zip(offsets, graphs, strict=True):
            if graph.senders is None or graph.receivers is None:
                raise ValueError(
                    "All graphs must either provide senders/receivers or omit both."
                )
            sender_parts.append(graph.senders + int(offset))
            receiver_parts.append(graph.receivers + int(offset))
        senders = jnp.concatenate(sender_parts, axis=0)
        receivers = jnp.concatenate(receiver_parts, axis=0)

    return GraphIR(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        globals=globals_,
        n_node=n_node,
        n_edge=n_edge,
        node_mask=node_mask,
        edge_mask=edge_mask,
        graph_mask=graph_mask,
        validate=validate,
    )


def unbatch_graph(graph: GraphIR, /, *, validate: bool = True) -> tuple[GraphIR, ...]:
    graph.validate()
    n_graphs = graph.num_graphs
    if n_graphs == 0:
        return ()

    node_offsets = jnp.cumsum(graph.n_node[:-1], axis=0)
    edge_offsets = jnp.cumsum(graph.n_edge[:-1], axis=0)

    node_splits = _split_tree(graph.nodes, node_offsets)
    edge_splits = _split_tree(graph.edges, edge_offsets)
    global_splits = _split_tree(graph.globals, jnp.arange(1, n_graphs, dtype=jnp.int32))
    node_mask_splits = (
        [None] * n_graphs
        if graph.node_mask is None
        else list(jnp.split(graph.node_mask, node_offsets, axis=0))
    )
    edge_mask_splits = (
        [None] * n_graphs
        if graph.edge_mask is None
        else list(jnp.split(graph.edge_mask, edge_offsets, axis=0))
    )
    graph_mask_splits = (
        [None] * n_graphs
        if graph.graph_mask is None
        else list(jnp.split(graph.graph_mask, jnp.arange(1, n_graphs), axis=0))
    )

    if graph.nodes is None:
        node_splits = [None] * n_graphs
    if graph.edges is None:
        edge_splits = [None] * n_graphs
    if graph.globals is None:
        global_splits = [None] * n_graphs

    if graph.senders is None or graph.receivers is None:
        sender_splits = [None] * n_graphs
        receiver_splits = [None] * n_graphs
    else:
        sender_splits = list(jnp.split(graph.senders, edge_offsets, axis=0))
        receiver_splits = list(jnp.split(graph.receivers, edge_offsets, axis=0))
        node_offsets_list = [0] + [int(x) for x in node_offsets]
        for i, offset in enumerate(node_offsets_list):
            sender_splits[i] = sender_splits[i] - offset
            receiver_splits[i] = receiver_splits[i] - offset

    out = []
    for i in range(n_graphs):
        g = GraphIR(
            nodes=node_splits[i],
            edges=edge_splits[i],
            senders=sender_splits[i],
            receivers=receiver_splits[i],
            globals=global_splits[i],
            n_node=graph.n_node[i : i + 1],
            n_edge=graph.n_edge[i : i + 1],
            node_mask=node_mask_splits[i],
            edge_mask=edge_mask_splits[i],
            graph_mask=graph_mask_splits[i],
            validate=validate,
        )
        out.append(g)

    return tuple(out)


def batch(graphs: Sequence[GraphIR], /, *, validate: bool = True) -> GraphIR:
    """jraph-style alias for `batch_graphs`."""
    return batch_graphs(graphs, validate=validate)


def batch_np(graphs: Sequence[GraphIR], /, *, validate: bool = True) -> GraphIR:
    """NumPy-named parity alias for `batch_graphs`."""
    return batch_graphs(graphs, validate=validate)


def unbatch(graph: GraphIR, /, *, validate: bool = True) -> list[GraphIR]:
    """jraph-style alias for `unbatch_graph`."""
    return list(unbatch_graph(graph, validate=validate))


def unbatch_np(graph: GraphIR, /, *, validate: bool = True) -> list[GraphIR]:
    """NumPy-named parity alias for `unbatch_graph`."""
    return list(unbatch_graph(graph, validate=validate))


__all__ = [
    "GraphIR",
    "batch_graphs",
    "unbatch_graph",
    "batch",
    "batch_np",
    "unbatch",
    "unbatch_np",
]
