from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from phydrax._strict import StrictModule

from ..sparse import EdgeRelation, gather_routes, route_reduce
from ._graph import ensure_graph
from ._ir import GraphIR


GraphPoolReduce = Literal["sum", "mean"]


def _tree_leading_size(tree: Any, /) -> int:
    leaves = jtu.tree_leaves(tree)
    if not leaves:
        raise ValueError("Feature tree must contain at least one array leaf.")
    return jnp.asarray(leaves[0]).shape[0]


def _mask_tree(tree: Any, mask: jnp.ndarray | None, /) -> Any:
    if mask is None:
        return tree

    def mask_leaf(value):
        arr = jnp.asarray(value)
        leaf_mask = mask
        while leaf_mask.ndim < arr.ndim:
            leaf_mask = jnp.expand_dims(leaf_mask, axis=-1)
        return arr * leaf_mask.astype(arr.dtype)

    return jtu.tree_map(mask_leaf, tree)


def _membership_relation(
    cluster_ids: jnp.ndarray, valid: jnp.ndarray, cluster_count: int, /
) -> EdgeRelation:
    """Route every fine entity onto its cluster; excluded entities stay inert."""
    count = cluster_ids.shape[0]
    return EdgeRelation(
        jnp.arange(count, dtype=jnp.int32),
        jnp.where(valid, cluster_ids, 0),
        source_size=count,
        target_size=cluster_count,
        valid=valid,
    )


def _pool_reduction(reduce: GraphPoolReduce, /) -> GraphPoolReduce:
    if reduce not in ("sum", "mean"):
        raise ValueError("Graph pool reduce must be 'sum' or 'mean'.")
    return reduce


def _valid_node_mask(graph: GraphIR, cluster_ids: jnp.ndarray, /) -> jnp.ndarray:
    valid = cluster_ids >= 0
    if graph.node_mask is not None:
        valid = valid & graph.node_mask
    return valid


def pool_graph_by_cluster(
    graph: GraphIR,
    cluster_ids: Any,
    /,
    *,
    reduce_nodes: GraphPoolReduce = "mean",
    reduce_edges: GraphPoolReduce = "mean",
    drop_self_edges: bool = True,
) -> GraphIR:
    """Pool a materialized graph into a coarse graph using node cluster ids.

    `cluster_ids[i]` gives the coarse node for fine node `i`; `-1` excludes a
    node. Node payloads and coalesced edge payloads are reduced over fine-to-
    coarse membership relations, so excluded nodes and masked edges are inert.
    Coarse topology is prepared on the host; this helper expects a single
    materialized graph.
    """
    node_reduction = _pool_reduction(reduce_nodes)
    edge_reduction = _pool_reduction(reduce_edges)
    graph = ensure_graph(graph, validate=False)
    if graph.n_node.shape[0] != 1:
        raise ValueError(
            "pool_graph_by_cluster currently expects one materialized graph."
        )
    if graph.nodes is None:
        raise ValueError("pool_graph_by_cluster requires node features.")
    if graph.senders is None or graph.receivers is None:
        raise ValueError("pool_graph_by_cluster requires explicit senders/receivers.")

    cluster_ids = jnp.asarray(cluster_ids, dtype=jnp.int32)
    n_nodes = _tree_leading_size(graph.nodes)
    if cluster_ids.shape[0] != n_nodes:
        raise ValueError(
            "cluster_ids length must match the node feature leading size; "
            f"got {cluster_ids.shape[0]} for {n_nodes} nodes."
        )
    valid_nodes = _valid_node_mask(graph, cluster_ids)
    valid_cluster_ids = cluster_ids[valid_nodes]
    if valid_cluster_ids.shape[0] == 0:
        raise ValueError("cluster_ids must select at least one node.")
    n_cluster = int(jnp.max(valid_cluster_ids)) + 1

    nodes = route_reduce(
        _membership_relation(cluster_ids, valid_nodes, n_cluster),
        graph.nodes,
        reduction=node_reduction,
    )

    coarse_senders = cluster_ids[graph.senders]
    coarse_receivers = cluster_ids[graph.receivers]
    edge_valid = (coarse_senders >= 0) & (coarse_receivers >= 0)
    if graph.edge_mask is not None:
        edge_valid = edge_valid & graph.edge_mask
    if drop_self_edges:
        edge_valid = edge_valid & (coarse_senders != coarse_receivers)

    valid_edge_np = np.asarray(edge_valid)
    coarse_senders_np = np.asarray(coarse_senders[edge_valid], dtype=np.int32)
    coarse_receivers_np = np.asarray(coarse_receivers[edge_valid], dtype=np.int32)

    if coarse_senders_np.size == 0:
        senders = jnp.zeros((0,), dtype=jnp.int32)
        receivers = jnp.zeros((0,), dtype=jnp.int32)
        edges = None
        if graph.edges is not None:
            edges = jtu.tree_map(lambda x: jnp.asarray(x)[:0], graph.edges)
    else:
        keys = coarse_senders_np * n_cluster + coarse_receivers_np
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        senders = jnp.asarray(unique_keys // n_cluster, dtype=jnp.int32)
        receivers = jnp.asarray(unique_keys % n_cluster, dtype=jnp.int32)
        edges = None
        if graph.edges is not None:
            coarse_edges = np.zeros(valid_edge_np.shape, dtype=np.int32)
            coarse_edges[valid_edge_np] = inverse
            edges = route_reduce(
                _membership_relation(
                    jnp.asarray(coarse_edges), edge_valid, unique_keys.shape[0]
                ),
                graph.edges,
                reduction=edge_reduction,
            )

    return GraphIR(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        globals=graph.globals,
        n_node=jnp.asarray([n_cluster], dtype=jnp.int32),
        n_edge=jnp.asarray([senders.shape[0]], dtype=jnp.int32),
        validate=False,
    )


def unpool_nodes_by_cluster(
    coarse_nodes: Any,
    cluster_ids: Any,
    /,
    *,
    fill_value: float = 0.0,
) -> Any:
    """Broadcast coarse node features back to fine nodes by cluster id."""
    cluster_ids = jnp.asarray(cluster_ids, dtype=jnp.int32)
    valid = cluster_ids >= 0
    membership = _membership_relation(
        cluster_ids, valid, _tree_leading_size(coarse_nodes)
    )
    lifted = gather_routes(membership.transpose(), coarse_nodes)

    def fill_leaf(value):
        fill = jnp.asarray(fill_value, dtype=value.dtype)
        mask = valid.reshape(valid.shape + (1,) * (value.ndim - 1))
        return jnp.where(mask, value, fill)

    return jtu.tree_map(fill_leaf, lifted)


class GraphClusterPool(StrictModule):
    """`GraphIR -> GraphIR` cluster pooling block."""

    cluster_ids: jnp.ndarray
    reduce_nodes: GraphPoolReduce = eqx.field(static=True)
    reduce_edges: GraphPoolReduce = eqx.field(static=True)
    drop_self_edges: bool = eqx.field(static=True)

    def __init__(
        self,
        cluster_ids: Any,
        /,
        *,
        reduce_nodes: GraphPoolReduce = "mean",
        reduce_edges: GraphPoolReduce = "mean",
        drop_self_edges: bool = True,
    ):
        self.cluster_ids = jnp.asarray(cluster_ids, dtype=jnp.int32)
        self.reduce_nodes = reduce_nodes
        self.reduce_edges = reduce_edges
        self.drop_self_edges = bool(drop_self_edges)

    def __call__(self, graph: GraphIR) -> GraphIR:
        return pool_graph_by_cluster(
            graph,
            self.cluster_ids,
            reduce_nodes=self.reduce_nodes,
            reduce_edges=self.reduce_edges,
            drop_self_edges=self.drop_self_edges,
        )


class GraphMultiscaleBlock(StrictModule):
    """Pool, process on a coarse graph, unpool, and fuse with fine nodes."""

    cluster_ids: jnp.ndarray
    coarse_block: Callable[[GraphIR], GraphIR]
    fine_block: Callable[[GraphIR], GraphIR] | None
    fusion_fn: Callable[[Any, Any], Any] | None
    reduce_nodes: GraphPoolReduce = eqx.field(static=True)
    reduce_edges: GraphPoolReduce = eqx.field(static=True)
    residual: bool = eqx.field(static=True)

    def __init__(
        self,
        cluster_ids: Any,
        coarse_block: Callable[[GraphIR], GraphIR],
        /,
        *,
        fine_block: Callable[[GraphIR], GraphIR] | None = None,
        fusion_fn: Callable[[Any, Any], Any] | None = None,
        reduce_nodes: GraphPoolReduce = "mean",
        reduce_edges: GraphPoolReduce = "mean",
        residual: bool = True,
    ):
        self.cluster_ids = jnp.asarray(cluster_ids, dtype=jnp.int32)
        self.coarse_block = coarse_block
        self.fine_block = fine_block
        self.fusion_fn = fusion_fn
        self.reduce_nodes = reduce_nodes
        self.reduce_edges = reduce_edges
        self.residual = bool(residual)

    def __call__(self, graph: GraphIR) -> GraphIR:
        graph = ensure_graph(graph, validate=False)
        fine = graph if self.fine_block is None else self.fine_block(graph)
        if not isinstance(fine, GraphIR):
            raise TypeError("GraphMultiscaleBlock fine_block must return GraphIR.")
        if fine.nodes is None:
            raise ValueError("GraphMultiscaleBlock requires fine node features.")

        coarse = pool_graph_by_cluster(
            fine,
            self.cluster_ids,
            reduce_nodes=self.reduce_nodes,
            reduce_edges=self.reduce_edges,
        )
        coarse = self.coarse_block(coarse)
        if not isinstance(coarse, GraphIR):
            raise TypeError("GraphMultiscaleBlock coarse_block must return GraphIR.")
        if coarse.nodes is None:
            raise ValueError("GraphMultiscaleBlock coarse_block output must have nodes.")

        lifted = unpool_nodes_by_cluster(coarse.nodes, self.cluster_ids)
        if self.fusion_fn is not None:
            nodes = self.fusion_fn(fine.nodes, lifted)
        elif self.residual:
            nodes = jtu.tree_map(lambda x, y: x + y, fine.nodes, lifted)
        else:
            nodes = lifted
        nodes = _mask_tree(nodes, fine.node_mask)
        return fine.replace(nodes=nodes, validate=False)


__all__ = [
    "GraphClusterPool",
    "GraphMultiscaleBlock",
    "GraphPoolReduce",
    "pool_graph_by_cluster",
    "unpool_nodes_by_cluster",
]
