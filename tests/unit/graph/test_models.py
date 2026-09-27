import jax
import jax.numpy as jnp

import phydrax.graph as vx
from tests._support.assertions import assert_tree_close


def _make_graph() -> vx.GraphIR:
    return vx.GraphIR(
        nodes=jnp.asarray([[0.0], [1.0], [2.0]]),
        edges=jnp.asarray([[0.5], [0.5], [0.5]]),
        senders=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        receivers=jnp.asarray([1, 2, 0], dtype=jnp.int32),
        globals=jnp.asarray([[0.0]]),
        n_node=jnp.asarray([3], dtype=jnp.int32),
        n_edge=jnp.asarray([3], dtype=jnp.int32),
    )


def test_graph_contracts() -> None:
    graph = _make_graph()
    cases = (
        (
            "graph-network",
            vx.GraphNetwork(
                update_edge_fn=lambda edge, sent, received, glob: edge + sent + received,
                update_node_fn=lambda node, sent, received, glob: node + sent + received,
                update_global_fn=lambda node, edge, glob: jnp.mean(
                    node, axis=0, keepdims=True
                ),
            ),
            ((3, 1), (3, 1), (1, 1)),
        ),
        (
            "interaction-network",
            vx.InteractionNetwork(
                update_edge_fn=lambda edge, sent, received: edge + sent + received,
                update_node_fn=lambda node, received: node + received,
            ),
            ((3, 1), (3, 1), (1, 1)),
        ),
        (
            "relation-network",
            vx.RelationNetwork(
                update_edge_fn=lambda sent, received: sent + received,
                update_global_fn=lambda edge: jnp.mean(edge, axis=0, keepdims=True),
            ),
            ((3, 1), (3, 1), (1, 1)),
        ),
        (
            "deep-sets",
            vx.DeepSets(
                update_node_fn=lambda node, glob: node + glob,
                update_global_fn=lambda node: jnp.mean(node, axis=0, keepdims=True),
            ),
            ((3, 1), (3, 1), (1, 1)),
        ),
        (
            "graphnet-gat",
            vx.GraphNetGAT(
                update_edge_fn=lambda edge, sent, received, glob: edge + sent + received,
                update_node_fn=lambda node, sent, received, glob: node + sent + received,
                attention_logit_fn=lambda edge, sent, received, glob: edge,
                attention_reduce_fn=lambda edge, weight: edge * weight,
                update_global_fn=lambda node, edge, glob: jnp.mean(
                    node, axis=0, keepdims=True
                ),
            ),
            ((3, 1), (3, 1), (1, 1)),
        ),
        (
            "graph-convolution",
            vx.GraphConvolution(
                update_node_fn=lambda node: node + 1.0,
                add_self_edges=True,
                symmetric_normalization=True,
            ),
            ((3, 1), (3, 1), (1, 1)),
        ),
    )
    for case_id, model, expected_shapes in cases:
        eager = model(graph)
        compiled = jax.jit(model)(graph)
        assert (eager.nodes.shape, eager.edges.shape, eager.globals.shape) == (
            expected_shapes
        ), case_id
        assert_tree_close(compiled, eager, rtol=1e-12, atol=1e-12)
    unpadded = vx.GraphIR(
        nodes=jnp.asarray([[1.0], [2.0]]),
        edges=jnp.asarray([[0.5]]),
        senders=jnp.asarray([0], dtype=jnp.int32),
        receivers=jnp.asarray([1], dtype=jnp.int32),
        globals=jnp.asarray([[3.0]]),
        n_node=jnp.asarray([2], dtype=jnp.int32),
        n_edge=jnp.asarray([1], dtype=jnp.int32),
    )
    padded = vx.GraphIR(
        nodes=jnp.asarray([[1.0], [2.0], [jnp.nan]]),
        edges=jnp.asarray([[0.5], [jnp.nan], [jnp.nan]]),
        senders=jnp.asarray([0, 2, 2], dtype=jnp.int32),
        receivers=jnp.asarray([1, 2, 2], dtype=jnp.int32),
        globals=jnp.asarray([[3.0], [jnp.nan]]),
        n_node=jnp.asarray([2, 1], dtype=jnp.int32),
        n_edge=jnp.asarray([1, 2], dtype=jnp.int32),
        node_mask=jnp.asarray([True, True, False]),
        edge_mask=jnp.asarray([True, False, False]),
        graph_mask=jnp.asarray([True, False]),
    )
    model = vx.GraphNetwork(
        update_edge_fn=lambda edge, sent, received, glob: (
            edge + sent + received + glob + 1.0
        ),
        update_node_fn=lambda node, sent, received, glob: (
            node + sent + received + glob + 1.0
        ),
        update_global_fn=lambda node, edge, glob: node + edge + glob + 1.0,
        attention_logit_fn=lambda edge, sent, received, glob: (
            edge + sent + received + glob
        ),
        attention_reduce_fn=lambda edge, weight: edge * weight,
    )
    expected = model(unpadded)
    actual = model(padded)
    assert jnp.allclose(actual.nodes[:2], expected.nodes)
    assert jnp.allclose(actual.edges[:1], expected.edges)
    assert jnp.allclose(actual.globals[:1], expected.globals)
    assert jnp.array_equal(actual.nodes[2:], jnp.zeros((1, 1)))
    assert jnp.array_equal(actual.edges[1:], jnp.zeros((2, 1)))
    assert jnp.array_equal(actual.globals[1:], jnp.zeros((1, 1)))
    graph = vx.GraphIR(
        nodes=jnp.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]),
        edges=jnp.asarray([[0.5], [0.5], [0.5]]),
        senders=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        receivers=jnp.asarray([1, 2, 0], dtype=jnp.int32),
        globals=jnp.asarray([[0.0]]),
        n_node=jnp.asarray([3], dtype=jnp.int32),
        n_edge=jnp.asarray([3], dtype=jnp.int32),
    )
    model = vx.GAT(
        attention_query_fn=lambda node: node,
        attention_logit_fn=lambda sent, received, edge: jnp.sum(
            sent + received, axis=-1, keepdims=True
        ),
        node_update_fn=lambda node: node,
    )
    eager = model(graph)
    compiled = jax.jit(model)(graph)
    assert eager.nodes.shape == (3, 2)
    assert_tree_close(compiled, eager, rtol=1e-12, atol=1e-12)
    graph = _make_graph()
    mapper = vx.graph_map_features(
        embed_node_fn=lambda node: node + 1.0,
        embed_edge_fn=lambda edge: edge * 2.0,
        embed_global_fn=lambda glob: glob - 1.0,
    )
    eager = mapper(graph)
    compiled = jax.jit(mapper)(graph)
    assert float(eager.nodes[0, 0]) == 1.0
    assert float(eager.edges[0, 0]) == 1.0
    assert float(eager.globals[0, 0]) == -1.0
    assert_tree_close(compiled, eager, rtol=1e-12, atol=1e-12)
