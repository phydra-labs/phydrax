from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from ._graph import ensure_graph
from ._ir import GraphIR
from ._kernels import segment_sum


GraphPoolReduce = Literal["sum", "mean"]


def _tree_leading_size(tree: Any, /) -> int:
    leaves = jtu.tree_leaves(tree)
    if not leaves:
        raise ValueError("Feature tree must contain at least one array leaf.")
    return int(jnp.asarray(leaves[0]).shape[0])


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


def _tree_segment_sum(tree: Any, segment_ids: jnp.ndarray, num_segments: int, /) -> Any:
    return jtu.tree_map(lambda x: segment_sum(x, segment_ids, num_segments), tree)


def _tree_segment_reduce(
    tree: Any,
    segment_ids: jnp.ndarray,
    num_segments: int,
    reduce: GraphPoolReduce,
    /,
) -> Any:
    summed = _tree_segment_sum(tree, segment_ids, num_segments)
    if reduce == "sum":
        return summed
    if reduce != "mean":
        raise ValueError("Graph pool reduce must be 'sum' or 'mean'.")
    counts = segment_sum(
        jnp.ones((segment_ids.shape[0],), dtype=float), segment_ids, num_segments
    )

    def divide(value):
        scale = jnp.where(counts > 0, 1.0 / counts, 0.0)
        while scale.ndim < value.ndim:
            scale = jnp.expand_dims(scale, axis=-1)
        return value * scale.astype(value.dtype)

    return jtu.tree_map(divide, summed)


def _tree_take(tree: Any, index: jnp.ndarray, /) -> Any:
    return jtu.tree_map(lambda x: x[index], tree)


def _filter_tree(tree: Any, mask: np.ndarray, /) -> Any:
    mask_jnp = jnp.asarray(mask, dtype=bool)
    return jtu.tree_map(lambda x: jnp.asarray(x)[mask_jnp], tree)


def _coalesce_tree(
    tree: Any, inverse: np.ndarray, n_unique: int, reduce: GraphPoolReduce, /
) -> Any:
    inverse_jnp = jnp.asarray(inverse, dtype=jnp.int32)
    return _tree_segment_reduce(tree, inverse_jnp, n_unique, reduce)


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
    node. This helper currently expects a single materialized graph.
    """
    graph = ensure_graph(graph, validate=False)
    if int(graph.n_node.shape[0]) != 1:
        raise ValueError(
            "pool_graph_by_cluster currently expects one materialized graph."
        )
    if graph.nodes is None:
        raise ValueError("pool_graph_by_cluster requires node features.")
    if graph.senders is None or graph.receivers is None:
        raise ValueError("pool_graph_by_cluster requires explicit senders/receivers.")

    cluster_ids = jnp.asarray(cluster_ids, dtype=jnp.int32)
    n_nodes = _tree_leading_size(graph.nodes)
    if int(cluster_ids.shape[0]) != n_nodes:
        raise ValueError(
            "cluster_ids length must match the node feature leading size; "
            f"got {cluster_ids.shape[0]} for {n_nodes} nodes."
        )
    valid_nodes = _valid_node_mask(graph, cluster_ids)
    valid_cluster_ids = cluster_ids[valid_nodes]
    if int(valid_cluster_ids.shape[0]) == 0:
        raise ValueError("cluster_ids must select at least one node.")
    n_cluster = int(jnp.max(valid_cluster_ids)) + 1

    valid_node_indices = jnp.where(valid_nodes)[0]
    nodes = _tree_segment_reduce(
        _tree_take(graph.nodes, valid_node_indices),
        valid_cluster_ids,
        n_cluster,
        reduce_nodes,
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
            edges = _filter_tree(
                graph.edges, np.zeros((int(graph.senders.shape[0]),), dtype=bool)
            )
    else:
        keys = coarse_senders_np * n_cluster + coarse_receivers_np
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        senders = jnp.asarray(unique_keys // n_cluster, dtype=jnp.int32)
        receivers = jnp.asarray(unique_keys % n_cluster, dtype=jnp.int32)
        edges = None
        if graph.edges is not None:
            filtered_edges = _filter_tree(graph.edges, valid_edge_np)
            edges = _coalesce_tree(
                filtered_edges, inverse, int(unique_keys.shape[0]), reduce_edges
            )

    return GraphIR(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        globals=graph.globals,
        n_node=jnp.asarray([n_cluster], dtype=jnp.int32),
        n_edge=jnp.asarray([int(senders.shape[0])], dtype=jnp.int32),
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
    safe_ids = jnp.where(valid, cluster_ids, 0)

    def unpool_leaf(value):
        arr = jnp.asarray(value)
        gathered = arr[safe_ids]
        fill = jnp.asarray(fill_value, dtype=arr.dtype)
        mask = valid
        while mask.ndim < gathered.ndim:
            mask = jnp.expand_dims(mask, axis=-1)
        return jnp.where(mask, gathered, fill)

    return jtu.tree_map(unpool_leaf, coarse_nodes)


class GraphClusterPool(eqx.Module):
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


class GraphMultiscaleBlock(eqx.Module):
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
