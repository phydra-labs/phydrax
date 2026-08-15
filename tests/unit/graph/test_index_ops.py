import jax.numpy as jnp

import phydrax.graph as vx


def test_coalesce_add_merges_duplicate_edges():
    edge_index = jnp.array([[0, 0, 1], [1, 1, 0]], dtype=jnp.int32)
    edge_attr = jnp.array([1.0, 2.0, 3.0])

    out_index, out_attr = vx.coalesce(edge_index, edge_attr, reduce="add")
    assert out_attr is not None
    assert out_index.shape == (2, 2)
    assert out_attr.shape == (2,)
    assert jnp.allclose(jnp.sort(out_attr), jnp.array([3.0, 3.0]))


def test_to_undirected_makes_symmetric_pairs():
    edge_index = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)
    out_index, _ = vx.to_undirected(edge_index)

    pairs = set(tuple(x.tolist()) for x in out_index.T)
    assert (0, 1) in pairs
    assert (1, 0) in pairs
    assert (1, 2) in pairs
    assert (2, 1) in pairs


def test_add_remaining_self_loops():
    edge_index = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    out_index, _ = vx.add_remaining_self_loops(edge_index, num_nodes=2)
    pairs = set(tuple(x.tolist()) for x in out_index.T)
    assert (0, 0) in pairs
    assert (1, 1) in pairs


def test_dense_edge_roundtrip():
    edge_index = jnp.array([[0, 1, 2], [1, 2, 0]], dtype=jnp.int32)
    edge_attr = jnp.array([2.0, 3.0, 4.0])

    dense = vx.to_dense_adj(edge_index, edge_attr, num_nodes=3)
    back_index, back_attr = vx.to_edge_index(dense)

    assert back_index.shape[0] == 2
    assert back_attr is not None
    assert int(back_index.shape[1]) == 3
