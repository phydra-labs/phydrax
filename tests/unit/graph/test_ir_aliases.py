import jax.numpy as jnp

import phydrax.graph as vx


def _make_graph(n_nodes: int) -> vx.GraphIR:
    senders = jnp.arange(n_nodes, dtype=jnp.int32)
    receivers = jnp.roll(senders, -1)
    return vx.GraphIR(
        nodes=jnp.arange(float(n_nodes)).reshape(n_nodes, 1),
        edges=jnp.ones((n_nodes, 1)),
        senders=senders,
        receivers=receivers,
        globals=jnp.ones((1, 1)),
        n_node=jnp.asarray([n_nodes], dtype=jnp.int32),
        n_edge=jnp.asarray([n_nodes], dtype=jnp.int32),
    )


def test_batch_aliases_match_batch_graphs():
    g1 = _make_graph(2)
    g2 = _make_graph(3)

    canonical = vx.batch_graphs((g1, g2))
    b1 = vx.batch((g1, g2))
    b2 = vx.batch_np((g1, g2))

    assert jnp.array_equal(canonical.n_node, b1.n_node)
    assert jnp.array_equal(canonical.n_edge, b1.n_edge)
    assert jnp.array_equal(canonical.edge_index, b1.edge_index)
    assert jnp.array_equal(canonical.n_node, b2.n_node)
    assert jnp.array_equal(canonical.n_edge, b2.n_edge)
    assert jnp.array_equal(canonical.edge_index, b2.edge_index)


def test_unbatch_aliases_match_unbatch_graph():
    g1 = _make_graph(2)
    g2 = _make_graph(3)
    batched = vx.batch_graphs((g1, g2))

    canonical = vx.unbatch_graph(batched)
    u1 = vx.unbatch(batched)
    u2 = vx.unbatch_np(batched)

    assert isinstance(u1, list)
    assert isinstance(u2, list)
    assert len(u1) == len(canonical)
    assert len(u2) == len(canonical)
    for got, expected in zip(u1, canonical, strict=True):
        assert jnp.array_equal(got.n_node, expected.n_node)
        assert jnp.array_equal(got.n_edge, expected.n_edge)
        assert jnp.array_equal(got.edge_index, expected.edge_index)
    for got, expected in zip(u2, canonical, strict=True):
        assert jnp.array_equal(got.n_node, expected.n_node)
        assert jnp.array_equal(got.n_edge, expected.n_edge)
        assert jnp.array_equal(got.edge_index, expected.edge_index)
