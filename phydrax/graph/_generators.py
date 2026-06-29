from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import jax.tree_util as jtu

from ._ir import GraphIR


def _tree_leading_size(tree: Any, /) -> int | None:
    if tree is None:
        return None
    leaves = jtu.tree_leaves(tree)
    if not leaves:
        return None
    return int(jnp.asarray(leaves[0]).shape[0])


def get_fully_connected_graph(
    n_node_per_graph: int,
    n_graph: int,
    *,
    node_features: Any | None = None,
    global_features: Any | None = None,
    add_self_edges: bool = True,
) -> GraphIR:
    """Build a fully connected directed graph batch."""
    n_node_per_graph = int(n_node_per_graph)
    n_graph = int(n_graph)
    if n_node_per_graph < 0 or n_graph < 1:
        raise ValueError("`n_node_per_graph` must be >= 0 and `n_graph` must be >= 1.")

    if node_features is not None:
        n_nodes_feat = _tree_leading_size(node_features)
        expected = n_node_per_graph * n_graph
        if n_nodes_feat != expected:
            raise ValueError(
                "Number of nodes is not equal to n_node_per_graph * n_graph."
            )
    if global_features is not None:
        n_globals = _tree_leading_size(global_features)
        if n_globals != n_graph:
            raise ValueError("The number of globals is not equal to n_graph.")

    if n_node_per_graph == 0:
        per_graph_edges = 0
        senders = jnp.zeros((0,), dtype=jnp.int32)
        receivers = jnp.zeros((0,), dtype=jnp.int32)
    else:
        base = jnp.arange(n_node_per_graph, dtype=jnp.int32)
        tmp_senders, tmp_receivers = jnp.meshgrid(base, base, indexing="ij")
        if not add_self_edges:
            mask = tmp_senders != tmp_receivers
            tmp_senders = tmp_senders[mask]
            tmp_receivers = tmp_receivers[mask]
        else:
            tmp_senders = tmp_senders.reshape(-1)
            tmp_receivers = tmp_receivers.reshape(-1)

        per_graph_edges = int(tmp_senders.shape[0])
        sender_parts = []
        receiver_parts = []
        for graph_idx in range(n_graph):
            offset = graph_idx * n_node_per_graph
            sender_parts.append(tmp_senders + offset)
            receiver_parts.append(tmp_receivers + offset)

        if sender_parts:
            senders = jnp.concatenate(sender_parts, axis=0)
            receivers = jnp.concatenate(receiver_parts, axis=0)
        else:
            senders = jnp.zeros((0,), dtype=jnp.int32)
            receivers = jnp.zeros((0,), dtype=jnp.int32)

    return GraphIR(
        nodes=node_features,
        edges=None,
        senders=senders,
        receivers=receivers,
        globals=global_features,
        n_node=jnp.asarray([n_node_per_graph] * n_graph, dtype=jnp.int32),
        n_edge=jnp.asarray([per_graph_edges] * n_graph, dtype=jnp.int32),
        validate=True,
    )


def sparse_matrix_to_graph(
    senders: jnp.ndarray,
    receivers: jnp.ndarray,
    values: jnp.ndarray,
    n_node: jnp.ndarray,
) -> GraphIR:
    """Create `GraphIR` from COO sparse matrix representation."""
    senders = jnp.asarray(senders, dtype=jnp.int32)
    receivers = jnp.asarray(receivers, dtype=jnp.int32)
    values = jnp.asarray(values, dtype=jnp.int32)
    n_node = jnp.asarray(n_node, dtype=jnp.int32)

    if senders.shape != receivers.shape:
        raise ValueError("`senders` and `receivers` must have identical shapes.")
    if values.ndim != 1:
        raise ValueError("`values` must be rank-1.")
    if senders.ndim != 1:
        raise ValueError("`senders` and `receivers` must be rank-1.")
    if int(values.shape[0]) != int(senders.shape[0]):
        raise ValueError("`values` length must match sender/receiver length.")
    if n_node.ndim != 1:
        raise ValueError("`n_node` must be rank-1.")
    if int(n_node.shape[0]) != 1:
        raise ValueError("`sparse_matrix_to_graph` currently supports exactly one graph.")
    if jnp.any(values < 0):
        raise ValueError("`values` must be non-negative.")

    n_edge_total = int(jnp.sum(values))
    repeated_senders = jnp.repeat(
        senders,
        values,
        axis=0,
        total_repeat_length=n_edge_total,
    )
    repeated_receivers = jnp.repeat(
        receivers,
        values,
        axis=0,
        total_repeat_length=n_edge_total,
    )

    return GraphIR(
        nodes=None,
        edges=None,
        senders=repeated_senders,
        receivers=repeated_receivers,
        globals=None,
        n_node=n_node,
        n_edge=jnp.asarray([n_edge_total], dtype=jnp.int32),
        validate=True,
    )


sparse_matrix_to_graphs_tuple = sparse_matrix_to_graph


__all__ = [
    "get_fully_connected_graph",
    "sparse_matrix_to_graph",
    "sparse_matrix_to_graphs_tuple",
]
