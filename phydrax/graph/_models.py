from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from ._graph import ensure_graph, GraphIR
from ._kernels import segment_softmax, segment_sum


ArrayTree = Any


def _tree_index(tree: ArrayTree, index: jnp.ndarray) -> ArrayTree:
    return jtu.tree_map(lambda x: x[index], tree)


def _tree_leading_size(tree: ArrayTree) -> int:
    leaves = jtu.tree_leaves(tree)
    if not leaves:
        raise ValueError("Feature tree must contain at least one array leaf.")
    return int(jnp.asarray(leaves[0]).shape[0])


def _tree_repeat(tree: ArrayTree, repeats: jnp.ndarray, total_repeat_length: int) -> ArrayTree:
    return jtu.tree_map(
        lambda x: jnp.repeat(x, repeats, axis=0, total_repeat_length=total_repeat_length),
        tree,
    )


def _tree_segment(
    tree: ArrayTree,
    segment_ids: jnp.ndarray,
    num_segments: int,
    aggregate_fn: Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray],
) -> ArrayTree:
    return jtu.tree_map(lambda x: aggregate_fn(x, segment_ids, num_segments), tree)


class GraphNetwork(eqx.Module):
    """Graph network block over `GraphIR`."""

    update_edge_fn: Callable | None
    update_node_fn: Callable | None
    update_global_fn: Callable | None

    aggregate_edges_for_nodes_fn: Callable = eqx.field(static=True)
    aggregate_nodes_for_globals_fn: Callable = eqx.field(static=True)
    aggregate_edges_for_globals_fn: Callable = eqx.field(static=True)
    attention_logit_fn: Callable | None
    attention_normalize_fn: Callable = eqx.field(static=True)
    attention_reduce_fn: Callable | None

    def __init__(
        self,
        update_edge_fn: Callable | None,
        update_node_fn: Callable | None,
        update_global_fn: Callable | None = None,
        *,
        aggregate_edges_for_nodes_fn: Callable = segment_sum,
        aggregate_nodes_for_globals_fn: Callable = segment_sum,
        aggregate_edges_for_globals_fn: Callable = segment_sum,
        attention_logit_fn: Callable | None = None,
        attention_normalize_fn: Callable = segment_softmax,
        attention_reduce_fn: Callable | None = None,
    ):
        if (attention_logit_fn is None) != (attention_reduce_fn is None):
            raise ValueError(
                "attention_logit_fn and attention_reduce_fn must both be provided or both be None."
            )

        self.update_edge_fn = update_edge_fn
        self.update_node_fn = update_node_fn
        self.update_global_fn = update_global_fn
        self.aggregate_edges_for_nodes_fn = aggregate_edges_for_nodes_fn
        self.aggregate_nodes_for_globals_fn = aggregate_nodes_for_globals_fn
        self.aggregate_edges_for_globals_fn = aggregate_edges_for_globals_fn
        self.attention_logit_fn = attention_logit_fn
        self.attention_normalize_fn = attention_normalize_fn
        self.attention_reduce_fn = attention_reduce_fn

    def __call__(self, graph: GraphIR) -> GraphIR:
        graph = ensure_graph(graph, validate=False)

        if graph.senders is None or graph.receivers is None:
            raise ValueError("GraphNetwork requires explicit senders and receivers.")

        if graph.nodes is None:
            raise ValueError("GraphNetwork requires node features.")

        nodes = graph.nodes
        edges = graph.edges
        globals_ = graph.globals
        senders = graph.senders
        receivers = graph.receivers
        n_node = graph.n_node
        n_edge = graph.n_edge

        sum_n_node = _tree_leading_size(nodes)
        sum_n_edge = int(senders.shape[0])

        if self.update_edge_fn is not None:
            sent = _tree_index(nodes, senders)
            recv = _tree_index(nodes, receivers)
            glob_edge = None
            if globals_ is not None:
                glob_edge = _tree_repeat(globals_, n_edge, total_repeat_length=sum_n_edge)
            edges = self.update_edge_fn(edges, sent, recv, glob_edge)

        if self.attention_logit_fn is not None:
            if edges is None:
                raise ValueError("Attention requires edge features.")
            sent = _tree_index(nodes, senders)
            recv = _tree_index(nodes, receivers)
            glob_edge = None
            if globals_ is not None:
                glob_edge = _tree_repeat(globals_, n_edge, total_repeat_length=sum_n_edge)
            logits = self.attention_logit_fn(edges, sent, recv, glob_edge)
            normalize = functools.partial(
                self.attention_normalize_fn,
                segment_ids=receivers,
                num_segments=sum_n_node,
            )
            weights = jtu.tree_map(normalize, logits)
            attention_reduce_fn = self.attention_reduce_fn
            if attention_reduce_fn is None:
                raise RuntimeError("GraphNetwork attention reducer invariant was violated.")
            edges = attention_reduce_fn(edges, weights)

        if self.update_node_fn is not None:
            if edges is None:
                raise ValueError("Node update requires edge features.")
            sent_aggr = _tree_segment(
                edges,
                senders,
                sum_n_node,
                self.aggregate_edges_for_nodes_fn,
            )
            recv_aggr = _tree_segment(
                edges,
                receivers,
                sum_n_node,
                self.aggregate_edges_for_nodes_fn,
            )
            glob_node = None
            if globals_ is not None:
                glob_node = _tree_repeat(globals_, n_node, total_repeat_length=sum_n_node)
            nodes = self.update_node_fn(nodes, sent_aggr, recv_aggr, glob_node)

        if self.update_global_fn is not None:
            n_graph = int(n_node.shape[0])
            graph_idx = jnp.arange(n_graph, dtype=jnp.int32)
            node_gr_idx = jnp.repeat(
                graph_idx,
                n_node,
                axis=0,
                total_repeat_length=sum_n_node,
            )
            edge_gr_idx = jnp.repeat(
                graph_idx,
                n_edge,
                axis=0,
                total_repeat_length=sum_n_edge,
            )

            node_aggr = _tree_segment(
                nodes,
                node_gr_idx,
                n_graph,
                self.aggregate_nodes_for_globals_fn,
            )
            edge_aggr = None
            if edges is not None:
                edge_aggr = _tree_segment(
                    edges,
                    edge_gr_idx,
                    n_graph,
                    self.aggregate_edges_for_globals_fn,
                )
            globals_ = self.update_global_fn(node_aggr, edge_aggr, globals_)

        return graph.replace(nodes=nodes, edges=edges, globals=globals_, validate=False)


class InteractionNetwork(eqx.Module):
    """Interaction network convenience wrapper."""

    update_edge_fn: Callable
    update_node_fn: Callable
    aggregate_edges_for_nodes_fn: Callable = eqx.field(static=True)
    include_sent_messages_in_node_update: bool = eqx.field(static=True)

    def __init__(
        self,
        update_edge_fn: Callable,
        update_node_fn: Callable,
        *,
        aggregate_edges_for_nodes_fn: Callable = segment_sum,
        include_sent_messages_in_node_update: bool = False,
    ):
        self.update_edge_fn = update_edge_fn
        self.update_node_fn = update_node_fn
        self.aggregate_edges_for_nodes_fn = aggregate_edges_for_nodes_fn
        self.include_sent_messages_in_node_update = include_sent_messages_in_node_update

    def __call__(self, graph: GraphIR) -> GraphIR:
        if self.include_sent_messages_in_node_update:

            def node_fn(nodes, sent, recv, globals_):
                del globals_
                return self.update_node_fn(nodes, sent, recv)

        else:

            def node_fn(nodes, sent, recv, globals_):
                del sent, globals_
                return self.update_node_fn(nodes, recv)

        def edge_fn(edges, sent, recv, globals_):
            del globals_
            return self.update_edge_fn(edges, sent, recv)

        net = GraphNetwork(
            update_edge_fn=edge_fn,
            update_node_fn=node_fn,
            update_global_fn=None,
            aggregate_edges_for_nodes_fn=self.aggregate_edges_for_nodes_fn,
        )
        return net(graph)


class RelationNetwork(eqx.Module):
    """Relation network convenience wrapper."""

    update_edge_fn: Callable
    update_global_fn: Callable
    aggregate_edges_for_globals_fn: Callable = eqx.field(static=True)

    def __init__(
        self,
        update_edge_fn: Callable,
        update_global_fn: Callable,
        *,
        aggregate_edges_for_globals_fn: Callable = segment_sum,
    ):
        self.update_edge_fn = update_edge_fn
        self.update_global_fn = update_global_fn
        self.aggregate_edges_for_globals_fn = aggregate_edges_for_globals_fn

    def __call__(self, graph: GraphIR) -> GraphIR:
        def edge_fn(edges, sent, recv, globals_):
            del edges, globals_
            return self.update_edge_fn(sent, recv)

        def global_fn(node_aggr, edge_aggr, globals_):
            del node_aggr, globals_
            return self.update_global_fn(edge_aggr)

        net = GraphNetwork(
            update_edge_fn=edge_fn,
            update_node_fn=None,
            update_global_fn=global_fn,
            aggregate_edges_for_globals_fn=self.aggregate_edges_for_globals_fn,
        )
        return net(graph)


class DeepSets(eqx.Module):
    """DeepSets convenience wrapper."""

    update_node_fn: Callable
    update_global_fn: Callable
    aggregate_nodes_for_globals_fn: Callable = eqx.field(static=True)

    def __init__(
        self,
        update_node_fn: Callable,
        update_global_fn: Callable,
        *,
        aggregate_nodes_for_globals_fn: Callable = segment_sum,
    ):
        self.update_node_fn = update_node_fn
        self.update_global_fn = update_global_fn
        self.aggregate_nodes_for_globals_fn = aggregate_nodes_for_globals_fn

    def __call__(self, graph: GraphIR) -> GraphIR:
        def node_fn(nodes, sent, recv, globals_):
            del sent, recv
            return self.update_node_fn(nodes, globals_)

        def global_fn(node_aggr, edge_aggr, globals_):
            del edge_aggr, globals_
            return self.update_global_fn(node_aggr)

        net = GraphNetwork(
            update_edge_fn=None,
            update_node_fn=node_fn,
            update_global_fn=global_fn,
            aggregate_nodes_for_globals_fn=self.aggregate_nodes_for_globals_fn,
        )
        return net(graph)


class GraphNetGAT(eqx.Module):
    """GraphNetwork with attention required."""

    update_edge_fn: Callable
    update_node_fn: Callable
    attention_logit_fn: Callable
    attention_reduce_fn: Callable
    update_global_fn: Callable | None
    aggregate_edges_for_nodes_fn: Callable = eqx.field(static=True)
    aggregate_nodes_for_globals_fn: Callable = eqx.field(static=True)
    aggregate_edges_for_globals_fn: Callable = eqx.field(static=True)
    attention_normalize_fn: Callable = eqx.field(static=True)

    def __init__(
        self,
        update_edge_fn: Callable,
        update_node_fn: Callable,
        attention_logit_fn: Callable,
        attention_reduce_fn: Callable,
        *,
        update_global_fn: Callable | None = None,
        aggregate_edges_for_nodes_fn: Callable = segment_sum,
        aggregate_nodes_for_globals_fn: Callable = segment_sum,
        aggregate_edges_for_globals_fn: Callable = segment_sum,
        attention_normalize_fn: Callable = segment_softmax,
    ):
        if attention_logit_fn is None or attention_reduce_fn is None:
            raise ValueError(
                "`None` value not supported for `attention_logit_fn` or "
                "`attention_reduce_fn` in GraphNetGAT."
            )
        self.update_edge_fn = update_edge_fn
        self.update_node_fn = update_node_fn
        self.attention_logit_fn = attention_logit_fn
        self.attention_reduce_fn = attention_reduce_fn
        self.update_global_fn = update_global_fn
        self.aggregate_edges_for_nodes_fn = aggregate_edges_for_nodes_fn
        self.aggregate_nodes_for_globals_fn = aggregate_nodes_for_globals_fn
        self.aggregate_edges_for_globals_fn = aggregate_edges_for_globals_fn
        self.attention_normalize_fn = attention_normalize_fn

    def __call__(self, graph: GraphIR) -> GraphIR:
        net = GraphNetwork(
            update_edge_fn=self.update_edge_fn,
            update_node_fn=self.update_node_fn,
            update_global_fn=self.update_global_fn,
            aggregate_edges_for_nodes_fn=self.aggregate_edges_for_nodes_fn,
            aggregate_nodes_for_globals_fn=self.aggregate_nodes_for_globals_fn,
            aggregate_edges_for_globals_fn=self.aggregate_edges_for_globals_fn,
            attention_logit_fn=self.attention_logit_fn,
            attention_normalize_fn=self.attention_normalize_fn,
            attention_reduce_fn=self.attention_reduce_fn,
        )
        return net(graph)


class GAT(eqx.Module):
    """Graph attention network layer over `GraphIR`."""

    attention_query_fn: Callable
    attention_logit_fn: Callable
    node_update_fn: Callable | None

    def __init__(
        self,
        attention_query_fn: Callable,
        attention_logit_fn: Callable,
        node_update_fn: Callable | None = None,
    ):
        self.attention_query_fn = attention_query_fn
        self.attention_logit_fn = attention_logit_fn
        self.node_update_fn = node_update_fn

    def __call__(self, graph: GraphIR) -> GraphIR:
        graph = ensure_graph(graph, validate=False)
        if graph.nodes is None:
            raise ValueError("GAT requires node features.")
        if graph.senders is None or graph.receivers is None:
            raise ValueError("GAT requires explicit senders and receivers.")

        nodes = graph.nodes
        senders = graph.senders
        receivers = graph.receivers
        sum_n_node = _tree_leading_size(nodes)

        queried_nodes = self.attention_query_fn(nodes)
        sent_attributes = queried_nodes[senders]
        received_attributes = queried_nodes[receivers]

        softmax_logits = self.attention_logit_fn(
            sent_attributes,
            received_attributes,
            graph.edges,
        )
        weights = segment_softmax(softmax_logits, segment_ids=receivers, num_segments=sum_n_node)
        messages = sent_attributes * weights
        updated_nodes = segment_sum(messages, receivers, num_segments=sum_n_node)

        if self.node_update_fn is None:
            updated_nodes = jnp.reshape(
                jax.nn.leaky_relu(updated_nodes),
                (updated_nodes.shape[0], -1),
            )
        else:
            updated_nodes = self.node_update_fn(updated_nodes)

        return graph.replace(nodes=updated_nodes, validate=False)


class GraphConvolution(eqx.Module):
    """Graph convolution layer over `GraphIR`."""

    update_node_fn: Callable
    aggregate_nodes_fn: Callable = eqx.field(static=True)
    add_self_edges: bool = eqx.field(static=True)
    symmetric_normalization: bool = eqx.field(static=True)

    def __init__(
        self,
        update_node_fn: Callable,
        *,
        aggregate_nodes_fn: Callable = segment_sum,
        add_self_edges: bool = False,
        symmetric_normalization: bool = True,
    ):
        self.update_node_fn = update_node_fn
        self.aggregate_nodes_fn = aggregate_nodes_fn
        self.add_self_edges = bool(add_self_edges)
        self.symmetric_normalization = bool(symmetric_normalization)

    def __call__(self, graph: GraphIR) -> GraphIR:
        graph = ensure_graph(graph, validate=False)
        if graph.nodes is None:
            raise ValueError("GraphConvolution requires node features.")
        if graph.senders is None or graph.receivers is None:
            raise ValueError("GraphConvolution requires explicit senders and receivers.")

        nodes = self.update_node_fn(graph.nodes)
        senders = graph.senders
        receivers = graph.receivers
        total_num_nodes = _tree_leading_size(nodes)

        if self.add_self_edges:
            self_edges = jnp.arange(total_num_nodes, dtype=jnp.int32)
            conv_senders = jnp.concatenate((senders, self_edges), axis=0)
            conv_receivers = jnp.concatenate((receivers, self_edges), axis=0)
        else:
            conv_senders = senders
            conv_receivers = receivers

        if self.symmetric_normalization:
            ones = jnp.ones_like(conv_senders, dtype=jnp.float32)
            sender_degree = segment_sum(ones, conv_senders, total_num_nodes)
            receiver_degree = segment_sum(ones, conv_receivers, total_num_nodes)

            nodes = jtu.tree_map(
                lambda x: x
                * jax.lax.rsqrt(jnp.maximum(sender_degree, 1.0)).astype(x.dtype)[:, None],
                nodes,
            )
            nodes = jtu.tree_map(
                lambda x: self.aggregate_nodes_fn(
                    x[conv_senders],
                    conv_receivers,
                    total_num_nodes,
                ),
                nodes,
            )
            nodes = jtu.tree_map(
                lambda x: x
                * jax.lax.rsqrt(jnp.maximum(receiver_degree, 1.0)).astype(x.dtype)[:, None],
                nodes,
            )
        else:
            nodes = jtu.tree_map(
                lambda x: self.aggregate_nodes_fn(
                    x[conv_senders],
                    conv_receivers,
                    total_num_nodes,
                ),
                nodes,
            )

        return graph.replace(nodes=nodes, validate=False)


class GraphMapFeatures(eqx.Module):
    """Apply independent feature maps to node/edge/global fields."""

    embed_edge_fn: Callable | None
    embed_node_fn: Callable | None
    embed_global_fn: Callable | None

    def __init__(
        self,
        embed_edge_fn: Callable | None = None,
        embed_node_fn: Callable | None = None,
        embed_global_fn: Callable | None = None,
    ):
        self.embed_edge_fn = embed_edge_fn
        self.embed_node_fn = embed_node_fn
        self.embed_global_fn = embed_global_fn

    def __call__(self, graph: GraphIR) -> GraphIR:
        graph = ensure_graph(graph, validate=False)
        nodes = graph.nodes if self.embed_node_fn is None else self.embed_node_fn(graph.nodes)
        edges = graph.edges if self.embed_edge_fn is None else self.embed_edge_fn(graph.edges)
        globals_ = (
            graph.globals
            if self.embed_global_fn is None
            else self.embed_global_fn(graph.globals)
        )
        return graph.replace(nodes=nodes, edges=edges, globals=globals_, validate=False)


def graph_map_features(
    embed_edge_fn: Callable | None = None,
    embed_node_fn: Callable | None = None,
    embed_global_fn: Callable | None = None,
) -> GraphMapFeatures:
    return GraphMapFeatures(
        embed_edge_fn=embed_edge_fn,
        embed_node_fn=embed_node_fn,
        embed_global_fn=embed_global_fn,
    )


__all__ = [
    "GraphNetwork",
    "InteractionNetwork",
    "RelationNetwork",
    "DeepSets",
    "GraphNetGAT",
    "GAT",
    "GraphConvolution",
    "GraphMapFeatures",
    "graph_map_features",
]
