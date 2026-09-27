import jax.numpy as jnp
import pytest

import phydrax.graph as vx


def test_generators_scenario_1() -> None:
    graph = vx.get_fully_connected_graph(3, 2, add_self_edges=False)
    assert graph.senders is not None
    assert graph.receivers is not None
    assert graph.n_node.tolist() == [3, 3]
    assert graph.n_edge.tolist() == [6, 6]
    assert graph.senders.shape[0] == 12
    assert graph.receivers.shape[0] == 12
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
    graph = vx.sparse_matrix_to_graph(
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 0], dtype=jnp.int32),
        values=jnp.array([2, 1], dtype=jnp.int32),
        n_node=jnp.array([2], dtype=jnp.int32),
    )
    assert graph.senders is not None
    assert graph.receivers is not None
    assert graph.n_edge.tolist() == [3]
    assert graph.senders.tolist() == [0, 0, 1]
    assert graph.receivers.tolist() == [1, 1, 0]
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
