import jax.numpy as jnp

import phydrax.graph as vx


def _make_graph() -> vx.GraphIR:
    return vx.GraphIR(
        nodes=jnp.array([[0.0], [1.0], [2.0]]),
        edges=jnp.array([[1.0], [1.5], [2.0]]),
        senders=jnp.array([0, 1, 2], dtype=jnp.int32),
        receivers=jnp.array([1, 2, 0], dtype=jnp.int32),
        globals=jnp.array([[3.0]]),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([3], dtype=jnp.int32),
    )


def test_graph_ir_counts_and_edge_index():
    graph = _make_graph()
    assert graph.num_graphs == 1
    assert graph.num_nodes == 3
    assert graph.num_edges == 3
    assert graph.edge_index.shape == (2, 3)


def test_batch_unbatch_graphs_roundtrip():
    g1 = _make_graph()
    g2 = _make_graph().replace(nodes=jnp.array([[4.0], [5.0], [6.0]]), validate=True)

    batched = vx.batch_graphs((g1, g2))
    pieces = vx.unbatch_graph(batched)

    assert len(pieces) == 2
    assert pieces[0].num_nodes == g1.num_nodes
    assert pieces[1].num_edges == g2.num_edges
    assert jnp.array_equal(pieces[0].senders, g1.senders)
    assert jnp.array_equal(pieces[1].receivers, g2.receivers)


def test_graph_counts_helper():
    graph = _make_graph()
    counts = vx.graph_counts(graph)
    assert counts == {"n_graph": 1, "n_node": 3, "n_edge": 3}
