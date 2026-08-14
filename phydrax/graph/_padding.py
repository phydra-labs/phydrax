from __future__ import annotations

import functools
from collections.abc import Callable, Generator, Iterator
from typing import Any

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from ._ir import batch_graphs, GraphIR


def _tree_leading_size(tree: Any, /) -> int | None:
    if tree is None:
        return None
    leaves = jtu.tree_leaves(tree)
    if not leaves:
        return None
    return int(jnp.asarray(leaves[0]).shape[0])


def _pad_tree_leading(tree: Any, pad_amount: int, /) -> Any:
    if tree is None:
        return None
    if pad_amount < 0:
        raise ValueError("pad amount must be non-negative.")
    if pad_amount == 0:
        return tree
    return jtu.tree_map(
        lambda x: jnp.concatenate(
            [x, jnp.zeros((pad_amount,) + x.shape[1:], dtype=x.dtype)],
            axis=0,
        ),
        tree,
    )


def _trim_tree_leading(tree: Any, keep: int, /) -> Any:
    if tree is None:
        return None
    return jtu.tree_map(lambda x: x[:keep], tree)


def _total_nodes(graph: GraphIR, /) -> int:
    n = _tree_leading_size(graph.nodes)
    if n is not None:
        return n
    return int(np.asarray(graph.n_node).sum())


def _total_edges(graph: GraphIR, /) -> int:
    if graph.senders is not None:
        return int(graph.senders.shape[0])
    n = _tree_leading_size(graph.edges)
    if n is not None:
        return n
    return int(np.asarray(graph.n_edge).sum())


def _graph_count(graph: GraphIR, /) -> int:
    return int(graph.n_node.shape[0])


def _mask_from_padding_length(*, padding_length: int, full_length: int) -> jnp.ndarray:
    valid_length = int(full_length) - int(padding_length)
    return jnp.arange(full_length, dtype=jnp.int32) < valid_length


def pad_with_graphs(
    graph: GraphIR,
    n_node: int,
    n_edge: int,
    n_graph: int = 2,
) -> GraphIR:
    """Pad a graph with a dummy graph and empty graphs.

    This mirrors jraph's `pad_with_graphs` semantics:
    - append one dummy graph with all padding nodes/edges
    - append trailing empty graphs
    """
    graph.validate()

    if n_graph < 2:
        raise ValueError(
            f"n_graph is {n_graph}, which is smaller than minimum value of 2."
        )

    n_node = int(n_node)
    n_edge = int(n_edge)
    n_graph = int(n_graph)

    num_nodes = _total_nodes(graph)
    num_edges = _total_edges(graph)
    num_graphs = _graph_count(graph)

    pad_n_node = n_node - num_nodes
    pad_n_edge = n_edge - num_edges
    pad_n_graph = n_graph - num_graphs
    if pad_n_node <= 0 or pad_n_edge < 0 or pad_n_graph <= 0:
        raise RuntimeError(
            "Given graph is too large for the given padding. difference: "
            f"n_node {pad_n_node}, n_edge {pad_n_edge}, n_graph {pad_n_graph}"
        )

    pad_n_empty_graph = pad_n_graph - 1

    n_node_out = jnp.concatenate(
        [
            graph.n_node,
            jnp.asarray([pad_n_node], dtype=jnp.int32),
            jnp.zeros((pad_n_empty_graph,), dtype=jnp.int32),
        ],
        axis=0,
    )
    n_edge_out = jnp.concatenate(
        [
            graph.n_edge,
            jnp.asarray([pad_n_edge], dtype=jnp.int32),
            jnp.zeros((pad_n_empty_graph,), dtype=jnp.int32),
        ],
        axis=0,
    )

    if graph.senders is None or graph.receivers is None:
        senders = jnp.zeros((pad_n_edge,), dtype=jnp.int32)
        receivers = jnp.zeros((pad_n_edge,), dtype=jnp.int32)
    else:
        pad_index = jnp.asarray(num_nodes, dtype=jnp.int32)
        senders = jnp.concatenate(
            [
                graph.senders,
                jnp.full((pad_n_edge,), pad_index, dtype=jnp.int32),
            ],
            axis=0,
        )
        receivers = jnp.concatenate(
            [
                graph.receivers,
                jnp.full((pad_n_edge,), pad_index, dtype=jnp.int32),
            ],
            axis=0,
        )

    out = GraphIR(
        nodes=_pad_tree_leading(graph.nodes, pad_n_node),
        edges=_pad_tree_leading(graph.edges, pad_n_edge),
        senders=senders,
        receivers=receivers,
        globals=_pad_tree_leading(graph.globals, pad_n_graph),
        n_node=n_node_out,
        n_edge=n_edge_out,
        node_mask=jnp.arange(n_node, dtype=jnp.int32) < num_nodes,
        edge_mask=jnp.arange(n_edge, dtype=jnp.int32) < num_edges,
        graph_mask=jnp.arange(n_graph, dtype=jnp.int32) < num_graphs,
        validate=True,
    )
    return out


def get_number_of_padding_with_graphs_graphs(padded_graph: GraphIR) -> int:
    if padded_graph.graph_mask is not None:
        graph_mask = np.asarray(padded_graph.graph_mask, dtype=bool)
        return int(graph_mask.shape[0] - graph_mask.sum())

    n_node = np.asarray(padded_graph.n_node)
    n_trailing_empty = 0
    for n in n_node[::-1]:
        if int(n) == 0:
            n_trailing_empty += 1
        else:
            break
    return n_trailing_empty + 1


def get_number_of_padding_with_graphs_nodes(padded_graph: GraphIR) -> int:
    if padded_graph.node_mask is not None:
        node_mask = np.asarray(padded_graph.node_mask, dtype=bool)
        return int(node_mask.shape[0] - node_mask.sum())

    n_padding_graph = get_number_of_padding_with_graphs_graphs(padded_graph)
    return int(np.asarray(padded_graph.n_node)[-n_padding_graph])


def get_number_of_padding_with_graphs_edges(padded_graph: GraphIR) -> int:
    if padded_graph.edge_mask is not None:
        edge_mask = np.asarray(padded_graph.edge_mask, dtype=bool)
        return int(edge_mask.shape[0] - edge_mask.sum())

    n_padding_graph = get_number_of_padding_with_graphs_graphs(padded_graph)
    return int(np.asarray(padded_graph.n_edge)[-n_padding_graph])


def unpad_with_graphs(padded_graph: GraphIR) -> GraphIR:
    n_padding_graph = get_number_of_padding_with_graphs_graphs(padded_graph)
    n_padding_node = get_number_of_padding_with_graphs_nodes(padded_graph)
    n_padding_edge = get_number_of_padding_with_graphs_edges(padded_graph)

    total_nodes = _total_nodes(padded_graph)
    total_edges = _total_edges(padded_graph)
    real_nodes = total_nodes - n_padding_node
    real_edges = total_edges - n_padding_edge

    n_node = padded_graph.n_node[:-n_padding_graph]
    n_edge = padded_graph.n_edge[:-n_padding_graph]

    senders = None
    receivers = None
    if padded_graph.senders is not None and padded_graph.receivers is not None:
        senders = padded_graph.senders[:real_edges]
        receivers = padded_graph.receivers[:real_edges]

    return GraphIR(
        nodes=_trim_tree_leading(padded_graph.nodes, real_nodes),
        edges=_trim_tree_leading(padded_graph.edges, real_edges),
        senders=senders,
        receivers=receivers,
        globals=_trim_tree_leading(padded_graph.globals, int(n_node.shape[0])),
        n_node=n_node,
        n_edge=n_edge,
        validate=True,
    )


def get_node_padding_mask(padded_graph: GraphIR) -> jnp.ndarray:
    if padded_graph.node_mask is not None:
        return padded_graph.node_mask

    n_padding_node = get_number_of_padding_with_graphs_nodes(padded_graph)
    total_num_nodes = _tree_leading_size(padded_graph.nodes)
    if total_num_nodes is None:
        raise ValueError("`padded_graph` must have at least one array of node features.")
    return _mask_from_padding_length(
        padding_length=n_padding_node,
        full_length=total_num_nodes,
    )


def get_edge_padding_mask(padded_graph: GraphIR) -> jnp.ndarray:
    if padded_graph.edge_mask is not None:
        return padded_graph.edge_mask

    n_padding_edge = get_number_of_padding_with_graphs_edges(padded_graph)
    total_num_edges = _total_edges(padded_graph)
    return _mask_from_padding_length(
        padding_length=n_padding_edge,
        full_length=total_num_edges,
    )


def get_graph_padding_mask(padded_graph: GraphIR) -> jnp.ndarray:
    if padded_graph.graph_mask is not None:
        return padded_graph.graph_mask

    n_padding_graph = get_number_of_padding_with_graphs_graphs(padded_graph)
    total_num_graphs = int(padded_graph.n_node.shape[0])
    return _mask_from_padding_length(
        padding_length=n_padding_graph,
        full_length=total_num_graphs,
    )


def _expand_trailing_dimensions(array: jnp.ndarray, template: jnp.ndarray) -> jnp.ndarray:
    missing_dims = len(template.shape) - len(array.shape)
    return jnp.reshape(array, array.shape + (1,) * missing_dims)


def _mask_tree(tree: Any, mask: jnp.ndarray) -> Any:
    if tree is None:
        return None
    return jtu.tree_map(lambda x: _expand_trailing_dimensions(mask, x) * x, tree)


def zero_out_padding(graph: GraphIR) -> GraphIR:
    node_mask = get_node_padding_mask(graph)
    edge_mask = get_edge_padding_mask(graph)
    graph_mask = get_graph_padding_mask(graph)

    return graph.replace(
        nodes=_mask_tree(graph.nodes, node_mask),
        edges=_mask_tree(graph.edges, edge_mask),
        globals=_mask_tree(graph.globals, graph_mask),
        validate=False,
    )


def with_zero_out_padding_outputs(
    graph_net: Callable[[GraphIR], GraphIR],
) -> Callable[[GraphIR], GraphIR]:
    @functools.wraps(graph_net)
    def wrapper(graph: GraphIR) -> GraphIR:
        return zero_out_padding(graph_net(graph))

    return wrapper


def _graph_size(graph: GraphIR) -> tuple[int, int, int]:
    return _total_nodes(graph), _total_edges(graph), _graph_count(graph)


def _is_over_batch_size(
    graph: GraphIR,
    graph_batch_size: tuple[int, int, int],
) -> bool:
    n_node, n_edge, n_graph = _graph_size(graph)
    b_node, b_edge, b_graph = graph_batch_size
    return n_node > b_node or n_edge > b_edge or n_graph > b_graph


def dynamically_batch(
    graphs_iterator: Iterator[GraphIR],
    n_node: int,
    n_edge: int,
    n_graph: int,
) -> Generator[GraphIR, None, None]:
    if n_graph < 2:
        raise ValueError(
            "The number of graphs in a batch size must be greater or equal "
            f"to `2` for padding with graphs, got {n_graph}."
        )

    valid_batch_size = (n_node - 1, n_edge, n_graph - 1)
    accumulated_graphs: list[GraphIR] = []
    num_acc_nodes = 0
    num_acc_edges = 0
    num_acc_graphs = 0

    for element in graphs_iterator:
        if not isinstance(element, GraphIR):
            raise RuntimeError("`dynamically_batch` expects an iterator of `GraphIR`.")

        element_nodes, element_edges, element_graphs = _graph_size(element)
        if _is_over_batch_size(element, valid_batch_size):
            if accumulated_graphs:
                batched_graph = batch_graphs(accumulated_graphs, validate=True)
                yield pad_with_graphs(batched_graph, n_node, n_edge, n_graph)

            graph_size = {
                "n_node": element_nodes,
                "n_edge": element_edges,
                "n_graph": element_graphs,
            }
            batch_size = {
                "n_node": valid_batch_size[0],
                "n_edge": valid_batch_size[1],
                "n_graph": valid_batch_size[2],
            }
            raise RuntimeError(
                "Found graph bigger than batch size. "
                f"Valid Batch Size: {batch_size}, Graph Size: {graph_size}"
            )

        if not accumulated_graphs:
            accumulated_graphs = [element]
            num_acc_nodes = element_nodes
            num_acc_edges = element_edges
            num_acc_graphs = element_graphs
            continue

        would_exceed = (
            num_acc_graphs + element_graphs > n_graph - 1
            or num_acc_nodes + element_nodes > n_node - 1
            or num_acc_edges + element_edges > n_edge
        )
        if would_exceed:
            batched_graph = batch_graphs(accumulated_graphs, validate=True)
            yield pad_with_graphs(batched_graph, n_node, n_edge, n_graph)
            accumulated_graphs = [element]
            num_acc_nodes = element_nodes
            num_acc_edges = element_edges
            num_acc_graphs = element_graphs
        else:
            accumulated_graphs.append(element)
            num_acc_nodes += element_nodes
            num_acc_edges += element_edges
            num_acc_graphs += element_graphs

    if accumulated_graphs:
        batched_graph = batch_graphs(accumulated_graphs, validate=True)
        yield pad_with_graphs(batched_graph, n_node, n_edge, n_graph)


__all__ = [
    "pad_with_graphs",
    "get_number_of_padding_with_graphs_graphs",
    "get_number_of_padding_with_graphs_nodes",
    "get_number_of_padding_with_graphs_edges",
    "unpad_with_graphs",
    "get_node_padding_mask",
    "get_edge_padding_mask",
    "get_graph_padding_mask",
    "zero_out_padding",
    "with_zero_out_padding_outputs",
    "dynamically_batch",
]
