import jax.numpy as jnp
import pytest

import phydrax.graph as vx


def _make_graph(n_nodes: int, n_edges: int | None = None) -> vx.GraphIR:
    if n_edges is None:
        n_edges = n_nodes
    senders = jnp.arange(n_edges, dtype=jnp.int32) % n_nodes
    receivers = jnp.roll(senders, -1)
    return vx.GraphIR(
        nodes=jnp.arange(float(n_nodes)).reshape(n_nodes, 1),
        edges=jnp.ones((n_edges, 1)),
        senders=senders,
        receivers=receivers,
        globals=jnp.ones((1, 1)),
        n_node=jnp.asarray([n_nodes], dtype=jnp.int32),
        n_edge=jnp.asarray([n_edges], dtype=jnp.int32),
    )


def test_pad_with_graphs_and_unpad_roundtrip():
    graph = _make_graph(3, 3)
    padded = vx.pad_with_graphs(graph, n_node=6, n_edge=5, n_graph=3)

    assert padded.n_node.tolist() == [3, 3, 0]
    assert padded.n_edge.tolist() == [3, 2, 0]
    assert int(jnp.sum(padded.node_mask.astype(jnp.int32))) == 3
    assert int(jnp.sum(padded.edge_mask.astype(jnp.int32))) == 3
    assert int(jnp.sum(padded.graph_mask.astype(jnp.int32))) == 1

    assert vx.get_number_of_padding_with_graphs_graphs(padded) == 2
    assert vx.get_number_of_padding_with_graphs_nodes(padded) == 3
    assert vx.get_number_of_padding_with_graphs_edges(padded) == 2

    unpadded = vx.unpad_with_graphs(padded)
    assert jnp.array_equal(unpadded.n_node, graph.n_node)
    assert jnp.array_equal(unpadded.n_edge, graph.n_edge)
    assert jnp.array_equal(unpadded.nodes, graph.nodes)
    assert jnp.array_equal(unpadded.edges, graph.edges)


def test_padding_masks_fallback_without_stored_masks():
    graph = _make_graph(3, 3)
    padded = vx.pad_with_graphs(graph, n_node=6, n_edge=5, n_graph=3)
    no_masks = padded.replace(
        node_mask=None,
        edge_mask=None,
        graph_mask=None,
        validate=False,
    )

    node_mask = vx.get_node_padding_mask(no_masks)
    edge_mask = vx.get_edge_padding_mask(no_masks)
    graph_mask = vx.get_graph_padding_mask(no_masks)

    assert int(jnp.sum(node_mask.astype(jnp.int32))) == 3
    assert int(jnp.sum(edge_mask.astype(jnp.int32))) == 3
    assert int(jnp.sum(graph_mask.astype(jnp.int32))) == 1


def test_pad_with_graphs_rejects_invalid_limits():
    graph = _make_graph(3, 3)
    with pytest.raises(ValueError):
        vx.pad_with_graphs(graph, n_node=8, n_edge=8, n_graph=1)
    with pytest.raises(RuntimeError):
        vx.pad_with_graphs(graph, n_node=3, n_edge=5, n_graph=3)


def test_zero_out_padding_and_wrapper():
    graph = _make_graph(3, 3)
    padded = vx.pad_with_graphs(graph, n_node=6, n_edge=5, n_graph=3)
    poisoned = padded.replace(
        nodes=jnp.ones_like(padded.nodes),
        edges=jnp.ones_like(padded.edges),
        globals=jnp.ones_like(padded.globals),
        validate=False,
    )

    zeroed = vx.zero_out_padding(poisoned)
    node_mask = vx.get_node_padding_mask(poisoned)
    edge_mask = vx.get_edge_padding_mask(poisoned)
    graph_mask = vx.get_graph_padding_mask(poisoned)

    assert jnp.all(zeroed.nodes[node_mask] == 1)
    assert jnp.all(zeroed.nodes[~node_mask] == 0)
    assert jnp.all(zeroed.edges[edge_mask] == 1)
    assert jnp.all(zeroed.edges[~edge_mask] == 0)
    assert jnp.all(zeroed.globals[graph_mask] == 1)
    assert jnp.all(zeroed.globals[~graph_mask] == 0)

    wrapped = vx.with_zero_out_padding_outputs(
        lambda g: g.replace(
            nodes=jnp.ones_like(g.nodes),
            edges=jnp.ones_like(g.edges),
            globals=jnp.ones_like(g.globals),
            validate=False,
        )
    )
    wrapped_out = wrapped(padded)
    assert jnp.all(wrapped_out.nodes[~node_mask] == 0)
    assert jnp.all(wrapped_out.edges[~edge_mask] == 0)
    assert jnp.all(wrapped_out.globals[~graph_mask] == 0)


def test_dynamically_batch_batches_and_pads():
    graphs = [_make_graph(2, 2), _make_graph(2, 2), _make_graph(1, 1)]
    batches = list(vx.dynamically_batch(iter(graphs), n_node=6, n_edge=6, n_graph=3))

    assert len(batches) == 2
    assert int(jnp.sum(vx.get_graph_padding_mask(batches[0]).astype(jnp.int32))) == 2
    assert int(jnp.sum(vx.get_graph_padding_mask(batches[1]).astype(jnp.int32))) == 1


def test_dynamically_batch_raises_for_oversized_graph():
    graphs = [_make_graph(6, 1)]
    with pytest.raises(RuntimeError):
        list(vx.dynamically_batch(iter(graphs), n_node=6, n_edge=10, n_graph=3))


def test_dynamically_batch_rejects_n_graph_lt_two():
    graphs = [_make_graph(2, 1)]
    with pytest.raises(ValueError):
        list(vx.dynamically_batch(iter(graphs), n_node=6, n_edge=10, n_graph=1))
