import jax.numpy as jnp

import phydrax.graph as vx


def _make_data(offset: float) -> vx.Data:
    return vx.Data(
        x=jnp.array([[offset], [offset + 1.0]]),
        edge_index=jnp.array([[0, 1], [1, 0]], dtype=jnp.int32),
        edge_attr=jnp.array([[1.0], [2.0]]),
        y=jnp.array([offset]),
    )


def test_data_to_graph_ir():
    data = _make_data(0.0)
    graph = data.to_graph_ir()
    assert graph.num_nodes == 2
    assert graph.num_edges == 2
    assert graph.nodes.shape == (2, 1)


def test_batch_from_data_list_and_back():
    d1 = _make_data(0.0)
    d2 = _make_data(10.0)

    batch = vx.Batch.from_data_list([d1, d2])
    assert batch.x is not None
    assert batch.edge_index is not None
    assert batch.num_graphs == 2
    assert batch.x.shape == (4, 1)
    assert batch.edge_index.shape == (2, 4)

    recovered = batch.to_data_list()
    assert recovered[0].x is not None
    assert recovered[1].edge_index is not None
    assert len(recovered) == 2
    assert recovered[0].x.shape == (2, 1)
    assert recovered[1].edge_index.shape == (2, 2)
