from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.core as jcore
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from ._ir import batch_graphs, GraphIR


def _pad_tree_leading(tree, target: int):
    if tree is None:
        return None

    def _pad_leaf(x: jnp.ndarray) -> jnp.ndarray:
        pad = target - int(x.shape[0])
        if pad < 0:
            raise ValueError("Cannot pad to a smaller leading dimension.")
        if pad == 0:
            return x
        zeros = jnp.zeros((pad,) + x.shape[1:], dtype=x.dtype)
        return jnp.concatenate([x, zeros], axis=0)

    return jtu.tree_map(_pad_leaf, tree)


def _trim_tree_leading(tree, count: int):
    if tree is None:
        return None
    return jtu.tree_map(lambda x: x[:count], tree)


def _contains_tracer(tree) -> bool:
    for leaf in jtu.tree_leaves(tree):
        if isinstance(leaf, jcore.Tracer):
            return True
    return False


def _tree_leading_size(tree) -> int | None:
    if tree is None:
        return None
    leaves = jtu.tree_leaves(tree)
    if not leaves:
        return None
    return int(jnp.asarray(leaves[0]).shape[0])


def _node_count(graph: GraphIR) -> int:
    n = _tree_leading_size(graph.nodes)
    if n is not None:
        return n
    if _contains_tracer(graph.n_node):
        raise ValueError(
            "Cannot infer total node count from traced `n_node` without node features. "
            "Provide node features for jitted packing."
        )
    return int(np.asarray(graph.n_node).sum())


def _edge_count(graph: GraphIR) -> int:
    if graph.senders is not None:
        return int(graph.senders.shape[0])
    n = _tree_leading_size(graph.edges)
    if n is not None:
        return n
    if _contains_tracer(graph.n_edge):
        raise ValueError(
            "Cannot infer total edge count from traced `n_edge` without senders/edges. "
            "Provide edge index/features for jitted packing."
        )
    return int(np.asarray(graph.n_edge).sum())


def _graph_count(graph: GraphIR) -> int:
    return int(graph.n_node.shape[0])


class LayoutPlan(eqx.Module):
    """Static shape budget for packed graph execution."""

    max_nodes: int = eqx.field(static=True)
    max_edges: int = eqx.field(static=True)
    max_graphs: int = eqx.field(static=True)

    def __init__(self, *, max_nodes: int, max_edges: int, max_graphs: int):
        self.max_nodes = int(max_nodes)
        self.max_edges = int(max_edges)
        self.max_graphs = int(max_graphs)
        if self.max_nodes < 0 or self.max_edges < 0 or self.max_graphs < 1:
            raise ValueError("Invalid layout maxima.")

    @classmethod
    def from_graphs(
        cls,
        graphs: Sequence[GraphIR],
        /,
        *,
        multiple: int = 1,
    ) -> "LayoutPlan":
        if len(graphs) == 0:
            raise ValueError("`LayoutPlan.from_graphs` requires at least one graph.")
        if multiple <= 0:
            raise ValueError("`multiple` must be positive.")

        max_nodes = max(g.num_nodes for g in graphs)
        max_edges = max(g.num_edges for g in graphs)
        max_graphs = max(g.num_graphs for g in graphs)

        def _round_up(x: int) -> int:
            if x % multiple == 0:
                return x
            return ((x // multiple) + 1) * multiple

        return cls(
            max_nodes=_round_up(max_nodes),
            max_edges=_round_up(max_edges),
            max_graphs=_round_up(max_graphs),
        )

    def pack(self, graph: GraphIR, /) -> GraphIR:
        num_nodes = _node_count(graph)
        num_edges = _edge_count(graph)
        num_graphs = _graph_count(graph)

        if num_nodes > self.max_nodes:
            raise ValueError("Graph exceeds max_nodes in layout plan.")
        if num_edges > self.max_edges:
            raise ValueError("Graph exceeds max_edges in layout plan.")
        if num_graphs > self.max_graphs:
            raise ValueError("Graph exceeds max_graphs in layout plan.")

        n_node = jnp.pad(
            graph.n_node,
            (0, self.max_graphs - num_graphs),
            mode="constant",
            constant_values=0,
        )
        n_edge = jnp.pad(
            graph.n_edge,
            (0, self.max_graphs - num_graphs),
            mode="constant",
            constant_values=0,
        )

        senders = None
        receivers = None
        if graph.senders is not None:
            senders = jnp.pad(
                graph.senders,
                (0, self.max_edges - num_edges),
                mode="constant",
                constant_values=0,
            )
            receivers = jnp.pad(
                graph.receivers,
                (0, self.max_edges - num_edges),
                mode="constant",
                constant_values=0,
            )

        node_mask = jnp.arange(self.max_nodes, dtype=jnp.int32) < num_nodes
        edge_mask = jnp.arange(self.max_edges, dtype=jnp.int32) < num_edges
        graph_mask = jnp.arange(self.max_graphs, dtype=jnp.int32) < num_graphs

        return GraphIR(
            nodes=_pad_tree_leading(graph.nodes, self.max_nodes),
            edges=_pad_tree_leading(graph.edges, self.max_edges),
            senders=senders,
            receivers=receivers,
            globals=_pad_tree_leading(graph.globals, self.max_graphs),
            n_node=n_node,
            n_edge=n_edge,
            node_mask=node_mask,
            edge_mask=edge_mask,
            graph_mask=graph_mask,
            validate=False,
        )

    def unpack(self, graph: GraphIR, /) -> GraphIR:
        if _contains_tracer(graph.graph_mask):
            raise RuntimeError(
                "LayoutPlan.unpack is not jittable because output shape depends on masks."
            )
        if graph.graph_mask is None:
            raise ValueError("Packed graph must include graph_mask.")

        n_graph = int(jnp.sum(graph.graph_mask.astype(jnp.int32)))
        n_node = graph.n_node[:n_graph]
        n_edge = graph.n_edge[:n_graph]

        real_nodes = int(jnp.sum(n_node))
        real_edges = int(jnp.sum(n_edge))

        senders = None
        receivers = None
        if graph.senders is not None:
            senders = graph.senders[:real_edges]
            receivers = graph.receivers[:real_edges]

        return GraphIR(
            nodes=_trim_tree_leading(graph.nodes, real_nodes),
            edges=_trim_tree_leading(graph.edges, real_edges),
            senders=senders,
            receivers=receivers,
            globals=_trim_tree_leading(graph.globals, n_graph),
            n_node=n_node,
            n_edge=n_edge,
            validate=True,
        )


def pack_graphs(
    graphs: Sequence[GraphIR],
    plan: LayoutPlan,
    /,
) -> GraphIR:
    batched = batch_graphs(graphs, validate=True)
    return plan.pack(batched)


__all__ = ["LayoutPlan", "pack_graphs"]
