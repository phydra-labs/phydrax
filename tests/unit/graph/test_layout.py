import jax
import jax.numpy as jnp
import pytest

import phydrax.graph as vx


def _make_graph(n_nodes: int = 3) -> vx.GraphIR:
    senders = jnp.arange(n_nodes, dtype=jnp.int32)
    receivers = jnp.roll(senders, -1)
    return vx.GraphIR(
        nodes=jnp.arange(float(n_nodes)).reshape(n_nodes, 1),
        edges=jnp.ones((n_nodes, 1)),
        senders=senders,
        receivers=receivers,
        globals=jnp.array([[1.0]]),
        n_node=jnp.array([n_nodes], dtype=jnp.int32),
        n_edge=jnp.array([n_nodes], dtype=jnp.int32),
    )


def test_layout_pack_and_unpack():
    graph = _make_graph(3)
    plan = vx.LayoutPlan(max_nodes=8, max_edges=8, max_graphs=4)

    packed = plan.pack(graph)
    assert packed.node_mask is not None
    assert int(jnp.sum(packed.node_mask)) == 3
    assert packed.n_node.shape[0] == 4

    restored = plan.unpack(packed)
    assert restored.num_nodes == graph.num_nodes
    assert restored.num_edges == graph.num_edges


def test_layout_pack_jit_runs():
    graph = _make_graph(3)
    plan = vx.LayoutPlan(max_nodes=8, max_edges=8, max_graphs=4)

    packed = jax.jit(plan.pack)(graph)
    assert packed.node_mask is not None
    assert packed.edge_mask is not None
    assert packed.graph_mask is not None
    assert packed.n_node.shape[0] == 4


def test_layout_unpack_jit_raises_shape_error():
    graph = _make_graph(3)
    plan = vx.LayoutPlan(max_nodes=8, max_edges=8, max_graphs=4)
    packed = plan.pack(graph)

    with pytest.raises(RuntimeError):
        jax.jit(plan.unpack)(packed)


def test_pack_graphs_helper():
    g1 = _make_graph(2)
    g2 = _make_graph(3)
    plan = vx.LayoutPlan(max_nodes=8, max_edges=8, max_graphs=4)
    packed = vx.pack_graphs((g1, g2), plan)
    assert packed.num_graphs == 4
    assert packed.graph_mask is not None
    assert int(jnp.sum(packed.graph_mask)) == 2
