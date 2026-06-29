import jax
import jax.numpy as jnp

import phydrax.graph as vx


def _make_graph() -> vx.GraphIR:
    return vx.GraphIR(
        nodes=jnp.array([[0.0], [1.0], [2.0]]),
        edges=jnp.array([[0.5], [0.5], [0.5]]),
        senders=jnp.array([0, 1, 2], dtype=jnp.int32),
        receivers=jnp.array([1, 2, 0], dtype=jnp.int32),
        globals=jnp.array([[0.0]]),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([3], dtype=jnp.int32),
    )


def test_graph_network_shapes():
    graph = _make_graph()

    net = vx.GraphNetwork(
        update_edge_fn=lambda e, s, r, g: e + s + r,
        update_node_fn=lambda n, s, r, g: n + s + r,
        update_global_fn=lambda n, e, g: jnp.mean(n, axis=0, keepdims=True),
    )

    out = net(graph)
    assert out.nodes.shape == (3, 1)
    assert out.edges.shape == (3, 1)
    assert out.globals.shape == (1, 1)


def test_graph_network_jit_runs():
    graph = _make_graph()
    net = vx.GraphNetwork(
        update_edge_fn=lambda e, s, r, g: e + s + r,
        update_node_fn=lambda n, s, r, g: n + s + r,
        update_global_fn=lambda n, e, g: jnp.mean(n, axis=0, keepdims=True),
    )

    out = jax.jit(net)(graph)
    assert out.nodes.shape == (3, 1)
    assert out.edges.shape == (3, 1)
    assert out.globals.shape == (1, 1)


def test_interaction_network_runs():
    graph = _make_graph()
    net = vx.InteractionNetwork(
        update_edge_fn=lambda e, s, r: e + s + r,
        update_node_fn=lambda n, r: n + r,
    )

    out = net(graph)
    assert out.nodes.shape == (3, 1)
    assert out.edges.shape == (3, 1)


def test_relation_network_runs():
    graph = _make_graph()
    net = vx.RelationNetwork(
        update_edge_fn=lambda s, r: s + r,
        update_global_fn=lambda e: jnp.mean(e, axis=0, keepdims=True),
    )

    out = net(graph)
    assert out.edges.shape == (3, 1)
    assert out.globals.shape == (1, 1)


def test_relation_network_jit_runs():
    graph = _make_graph()
    net = vx.RelationNetwork(
        update_edge_fn=lambda s, r: s + r,
        update_global_fn=lambda e: jnp.mean(e, axis=0, keepdims=True),
    )

    out = jax.jit(net)(graph)
    assert out.edges.shape == (3, 1)
    assert out.globals.shape == (1, 1)


def test_deepsets_runs():
    graph = _make_graph()
    net = vx.DeepSets(
        update_node_fn=lambda n, g: n + g,
        update_global_fn=lambda n: jnp.mean(n, axis=0, keepdims=True),
    )

    out = net(graph)
    assert out.nodes.shape == (3, 1)
    assert out.globals.shape == (1, 1)


def test_deepsets_jit_runs():
    graph = _make_graph()
    net = vx.DeepSets(
        update_node_fn=lambda n, g: n + g,
        update_global_fn=lambda n: jnp.mean(n, axis=0, keepdims=True),
    )

    out = jax.jit(net)(graph)
    assert out.nodes.shape == (3, 1)
    assert out.globals.shape == (1, 1)


def test_graphnet_gat_runs():
    graph = _make_graph()
    net = vx.GraphNetGAT(
        update_edge_fn=lambda e, s, r, g: e + s + r,
        update_node_fn=lambda n, s, r, g: n + s + r,
        attention_logit_fn=lambda e, s, r, g: e,
        attention_reduce_fn=lambda e, w: e * w,
        update_global_fn=lambda n, e, g: jnp.mean(n, axis=0, keepdims=True),
    )

    out = net(graph)
    assert out.nodes.shape == (3, 1)
    assert out.edges.shape == (3, 1)
    assert out.globals.shape == (1, 1)


def test_graphnet_gat_jit_runs():
    graph = _make_graph()
    net = vx.GraphNetGAT(
        update_edge_fn=lambda e, s, r, g: e + s + r,
        update_node_fn=lambda n, s, r, g: n + s + r,
        attention_logit_fn=lambda e, s, r, g: e,
        attention_reduce_fn=lambda e, w: e * w,
        update_global_fn=lambda n, e, g: jnp.mean(n, axis=0, keepdims=True),
    )

    out = jax.jit(net)(graph)
    assert out.nodes.shape == (3, 1)
    assert out.edges.shape == (3, 1)
    assert out.globals.shape == (1, 1)


def test_gat_runs():
    graph = vx.GraphIR(
        nodes=jnp.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]),
        edges=jnp.array([[0.5], [0.5], [0.5]]),
        senders=jnp.array([0, 1, 2], dtype=jnp.int32),
        receivers=jnp.array([1, 2, 0], dtype=jnp.int32),
        globals=jnp.array([[0.0]]),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([3], dtype=jnp.int32),
    )
    net = vx.GAT(
        attention_query_fn=lambda n: n,
        attention_logit_fn=lambda s, r, e: jnp.sum(s + r, axis=-1, keepdims=True),
        node_update_fn=lambda n: n,
    )

    out = net(graph)
    assert out.nodes.shape == (3, 2)


def test_gat_jit_runs():
    graph = vx.GraphIR(
        nodes=jnp.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]),
        edges=jnp.array([[0.5], [0.5], [0.5]]),
        senders=jnp.array([0, 1, 2], dtype=jnp.int32),
        receivers=jnp.array([1, 2, 0], dtype=jnp.int32),
        globals=jnp.array([[0.0]]),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([3], dtype=jnp.int32),
    )
    net = vx.GAT(
        attention_query_fn=lambda n: n,
        attention_logit_fn=lambda s, r, e: jnp.sum(s + r, axis=-1, keepdims=True),
        node_update_fn=lambda n: n,
    )

    out = jax.jit(net)(graph)
    assert out.nodes.shape == (3, 2)


def test_graph_convolution_runs():
    graph = _make_graph()
    net = vx.GraphConvolution(
        update_node_fn=lambda n: n + 1.0,
        add_self_edges=True,
        symmetric_normalization=True,
    )

    out = net(graph)
    assert out.nodes.shape == (3, 1)


def test_graph_convolution_jit_runs():
    graph = _make_graph()
    net = vx.GraphConvolution(
        update_node_fn=lambda n: n + 1.0,
        add_self_edges=True,
        symmetric_normalization=True,
    )

    out = jax.jit(net)(graph)
    assert out.nodes.shape == (3, 1)


def test_graph_map_features_runs():
    graph = _make_graph()
    mapper = vx.graph_map_features(
        embed_node_fn=lambda n: n + 1.0,
        embed_edge_fn=lambda e: e * 2.0,
        embed_global_fn=lambda g: g - 1.0,
    )
    out = mapper(graph)
    assert float(out.nodes[0, 0]) == 1.0
    assert float(out.edges[0, 0]) == 1.0
    assert float(out.globals[0, 0]) == -1.0


def test_graph_map_features_jit_runs():
    graph = _make_graph()
    mapper = vx.graph_map_features(
        embed_node_fn=lambda n: n + 1.0,
        embed_edge_fn=lambda e: e * 2.0,
        embed_global_fn=lambda g: g - 1.0,
    )

    out = jax.jit(mapper)(graph)
    assert out.nodes.shape == (3, 1)
    assert out.edges.shape == (3, 1)
    assert out.globals.shape == (1, 1)
