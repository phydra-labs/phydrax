import jax.numpy as jnp
import pytest

import phydrax.graph as vx


def test_get_fully_connected_graph_no_self_edges():
    graph = vx.get_fully_connected_graph(3, 2, add_self_edges=False)
    assert graph.n_node.tolist() == [3, 3]
    assert graph.n_edge.tolist() == [6, 6]
    assert int(graph.senders.shape[0]) == 12
    assert int(graph.receivers.shape[0]) == 12


def test_get_fully_connected_graph_with_features():
    node_features = jnp.arange(12.0).reshape(6, 2)
    global_features = jnp.arange(2.0).reshape(2, 1)
    graph = vx.get_fully_connected_graph(
        3,
        2,
        node_features=node_features,
        global_features=global_features,
        add_self_edges=True,
    )
    assert graph.nodes.shape == (6, 2)
    assert graph.globals.shape == (2, 1)
    assert graph.n_edge.tolist() == [9, 9]


def test_get_fully_connected_graph_feature_shape_checks():
    with pytest.raises(ValueError):
        vx.get_fully_connected_graph(
            3,
            2,
            node_features=jnp.ones((5, 2)),
        )
    with pytest.raises(ValueError):
        vx.get_fully_connected_graph(
            3,
            2,
            global_features=jnp.ones((1, 2)),
        )


def test_sparse_matrix_to_graph_repeats_indices():
    graph = vx.sparse_matrix_to_graph(
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 0], dtype=jnp.int32),
        values=jnp.array([2, 1], dtype=jnp.int32),
        n_node=jnp.array([2], dtype=jnp.int32),
    )
    assert graph.n_edge.tolist() == [3]
    assert graph.senders.tolist() == [0, 0, 1]
    assert graph.receivers.tolist() == [1, 1, 0]


def test_sparse_matrix_to_graph_validation():
    with pytest.raises(ValueError):
        vx.sparse_matrix_to_graph(
            senders=jnp.array([0, 1], dtype=jnp.int32),
            receivers=jnp.array([1], dtype=jnp.int32),
            values=jnp.array([1, 1], dtype=jnp.int32),
            n_node=jnp.array([2], dtype=jnp.int32),
        )
    with pytest.raises(ValueError):
        vx.sparse_matrix_to_graph(
            senders=jnp.array([0], dtype=jnp.int32),
            receivers=jnp.array([1], dtype=jnp.int32),
            values=jnp.array([-1], dtype=jnp.int32),
            n_node=jnp.array([2], dtype=jnp.int32),
        )
