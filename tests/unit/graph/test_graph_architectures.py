from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx


def _feature_graph() -> phx.graph.GraphIR:
    return phx.graph.GraphIR(
        nodes=jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        ),
        edges=jnp.array(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [-1.0, 0.0, 1.0],
                [0.0, -1.0, 1.0],
            ]
        ),
        senders=jnp.array([0, 1, 2, 3], dtype=jnp.int32),
        receivers=jnp.array([1, 2, 3, 0], dtype=jnp.int32),
        n_node=jnp.array([4], dtype=jnp.int32),
        n_edge=jnp.array([4], dtype=jnp.int32),
    )


def test_graph_architectures_scenario_1() -> None:
    mlp = phx.graph.RowMLP(2, 3, width_size=4, depth=2, key=jr.key(0))

    out = mlp(jnp.ones((5, 2)))

    assert out.shape == (5, 3)
    assert jnp.all(jnp.isfinite(out))
    model = phx.graph.MeshGraphNet(
        node_in_size=2,
        edge_in_size=3,
        node_out_size=1,
        latent_size=8,
        hidden_size=8,
        processor_steps=2,
        key=jr.key(1),
    )

    out = model(_feature_graph())

    assert out.nodes.shape == (4, 1)
    assert out.edges.shape == (4, 8)
    assert jnp.all(jnp.isfinite(out.nodes))
    graph = phx.graph.GraphIR(
        nodes=jnp.array([[0.0], [1.0], [0.0]]),
        edges=jnp.array([[1.0], [0.0]]),
        senders=jnp.array([0, 2], dtype=jnp.int32),
        receivers=jnp.array([1, 2], dtype=jnp.int32),
        n_node=jnp.array([2], dtype=jnp.int32),
        n_edge=jnp.array([1], dtype=jnp.int32),
        node_mask=jnp.array([True, True, False]),
        edge_mask=jnp.array([True, False]),
        validate=False,
    )
    model = phx.graph.MeshGraphNet(
        node_in_size=1,
        edge_in_size=1,
        node_out_size=1,
        latent_size=4,
        hidden_size=4,
        processor_steps=1,
        key=jr.key(2),
    )

    out = model(graph)

    assert out.node_mask is not None
    assert out.edge_mask is not None
    assert jnp.allclose(out.nodes[2], jnp.zeros((1,)))
    assert jnp.allclose(out.edges[1], jnp.zeros((4,)))
    graph = _feature_graph()
    domain = phx.domain.GraphDomain(graph)
    component = domain.component({"graph": phx.domain.Nodes()})
    batch = component.sample(
        phx.domain.PointSampling(4, layout=phx.domain.SampleLayout((("graph",),)))
    )
    model = phx.graph.MeshGraphNet(
        node_in_size=2,
        edge_in_size=3,
        node_out_size=1,
        latent_size=4,
        hidden_size=4,
        processor_steps=1,
        key=jr.key(3),
    )

    f = domain.GraphModel(model)

    assert f(batch).data.shape == (4, 1)
    graph = phx.graph.GraphIR(
        nodes=jnp.array([[1.0], [3.0], [5.0], [7.0]]),
        edges=jnp.array([[1.0], [3.0], [9.0]]),
        senders=jnp.array([0, 1, 2], dtype=jnp.int32),
        receivers=jnp.array([2, 3, 3], dtype=jnp.int32),
        n_node=jnp.array([4], dtype=jnp.int32),
        n_edge=jnp.array([3], dtype=jnp.int32),
    )

    coarse = phx.graph.pool_graph_by_cluster(graph, jnp.array([0, 0, 1, 1]))
    assert coarse.senders is not None
    assert coarse.receivers is not None

    assert coarse.num_nodes == 2
    assert coarse.num_edges == 1
    assert jnp.allclose(coarse.nodes[:, 0], jnp.array([2.0, 6.0]))
    assert jnp.allclose(coarse.edges[:, 0], jnp.array([2.0]))
    assert jnp.allclose(coarse.senders, jnp.array([0], dtype=jnp.int32))
    assert jnp.allclose(coarse.receivers, jnp.array([1], dtype=jnp.int32))


def test_graph_multiscale_block_unpools_coarse_update() -> None:
    graph = phx.graph.GraphIR(
        nodes=jnp.array([[1.0], [3.0], [5.0], [7.0]]),
        edges=jnp.array([[1.0], [3.0]]),
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([2, 3], dtype=jnp.int32),
        n_node=jnp.array([4], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )

    def coarse_shift(coarse: Any) -> Any:
        return coarse.replace(nodes=coarse.nodes + 10.0, validate=False)

    block = phx.graph.GraphMultiscaleBlock(
        jnp.array([0, 0, 1, 1]),
        coarse_shift,
        residual=False,
    )

    out = block(graph)

    assert jnp.allclose(out.nodes[:, 0], jnp.array([12.0, 12.0, 16.0, 16.0]))


def test_streamed_mesh_graph_net_block_matches_dense_outputs_and_combined_loss() -> None:
    senders = np.asarray([0, 1, 2, 0, 2, 3, 4, 5, 3, 5, 4], dtype=np.int32)
    receivers = np.asarray([1, 2, 0, 2, 1, 4, 5, 3, 5, 4, 3], dtype=np.int32)
    edge_mask = np.asarray([True] * 9 + [False, False])
    rng = np.random.default_rng(5)
    graph = phx.graph.GraphIR(
        nodes=jnp.asarray(rng.normal(size=(6, 3))),
        edges=jnp.asarray(np.where(edge_mask[:, None], rng.normal(size=(11, 3)), 1.0e3)),
        senders=senders,
        receivers=receivers,
        globals=jnp.asarray(rng.normal(size=(2, 2))),
        n_node=[3, 3],
        n_edge=[5, 6],
        edge_mask=edge_mask,
    )
    node_graph = np.repeat([0, 1], [3, 3])
    edge_graph = np.repeat([0, 1], [5, 6])
    # One receiver per tile and two events per fragment split every receiver row.
    block = phx.graph.MeshGraphNetBlock(
        3,
        global_size=2,
        execution=phx.sparse.StreamedRelationPlan(receiver_tile=1, edge_tile=2),
        key=jr.key(9),
    )

    def dense(block: Any, nodes: Any) -> Any:
        globals_ = graph.globals
        edges = graph.edges + block.edge_mlp(
            jnp.concatenate(
                [graph.edges, nodes[senders], nodes[receivers], globals_[edge_graph]],
                -1,
            )
        )
        edges = jnp.where(edge_mask[:, None], edges, 0.0)
        received = jax.ops.segment_sum(edges, receivers, 6)
        updated = nodes + block.node_mlp(
            jnp.concatenate([nodes, received, globals_[node_graph]], -1)
        )
        return updated, edges

    def streamed(block: Any, nodes: Any) -> Any:
        out = block(graph.replace(nodes=nodes, validate=False))
        return out.nodes, out.edges

    observed, expected = streamed(block, graph.nodes), dense(block, graph.nodes)
    for actual, reference in zip(observed, expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)

    def combined_loss(evaluate: Any) -> Any:
        def loss(block: Any, nodes: Any) -> Any:
            updated, edges = evaluate(block, nodes)
            return jnp.sum(updated**2) + jnp.sum(jnp.sin(edges))

        return loss

    grad = eqx.filter_grad(combined_loss(streamed))(block, graph.nodes)
    reference_grad = eqx.filter_grad(combined_loss(dense))(block, graph.nodes)
    for actual, reference in zip(
        jax.tree_util.tree_leaves(grad),
        jax.tree_util.tree_leaves(reference_grad),
        strict=True,
    ):
        np.testing.assert_allclose(actual, reference, rtol=1e-11, atol=1e-11)
    node_grad = jax.grad(combined_loss(streamed), argnums=1)(block, graph.nodes)
    np.testing.assert_allclose(
        node_grad,
        jax.grad(combined_loss(dense), argnums=1)(block, graph.nodes),
        rtol=1e-11,
        atol=1e-11,
    )
