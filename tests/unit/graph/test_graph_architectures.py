import jax.numpy as jnp
import jax.random as jr

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


def test_row_mlp_maps_rows():
    mlp = phx.graph.RowMLP(2, 3, width_size=4, depth=2, key=jr.key(0))

    out = mlp(jnp.ones((5, 2)))

    assert out.shape == (5, 3)
    assert jnp.all(jnp.isfinite(out))


def test_mesh_graph_net_outputs_node_predictions():
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


def test_mesh_graph_net_preserves_padding_masks():
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


def test_mesh_graph_net_wraps_as_domain_graph_model():
    graph = _feature_graph()
    domain = phx.domain.GraphDomain(graph)
    component = domain.component({"graph": phx.domain.Nodes()})
    batch = component.sample(phx.domain.PointSampling(4, layout=phx.domain.SampleLayout((("graph",),))))
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


def test_pool_graph_by_cluster_coalesces_edges_and_means_features():
    graph = phx.graph.GraphIR(
        nodes=jnp.array([[1.0], [3.0], [5.0], [7.0]]),
        edges=jnp.array([[1.0], [3.0], [9.0]]),
        senders=jnp.array([0, 1, 2], dtype=jnp.int32),
        receivers=jnp.array([2, 3, 3], dtype=jnp.int32),
        n_node=jnp.array([4], dtype=jnp.int32),
        n_edge=jnp.array([3], dtype=jnp.int32),
    )

    coarse = phx.graph.pool_graph_by_cluster(graph, jnp.array([0, 0, 1, 1]))

    assert coarse.num_nodes == 2
    assert coarse.num_edges == 1
    assert jnp.allclose(coarse.nodes[:, 0], jnp.array([2.0, 6.0]))
    assert jnp.allclose(coarse.edges[:, 0], jnp.array([2.0]))
    assert jnp.allclose(coarse.senders, jnp.array([0], dtype=jnp.int32))
    assert jnp.allclose(coarse.receivers, jnp.array([1], dtype=jnp.int32))


def test_graph_multiscale_block_unpools_coarse_update():
    graph = phx.graph.GraphIR(
        nodes=jnp.array([[1.0], [3.0], [5.0], [7.0]]),
        edges=jnp.array([[1.0], [3.0]]),
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([2, 3], dtype=jnp.int32),
        n_node=jnp.array([4], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )

    def coarse_shift(coarse):
        return coarse.replace(nodes=coarse.nodes + 10.0, validate=False)

    block = phx.graph.GraphMultiscaleBlock(
        jnp.array([0, 0, 1, 1]),
        coarse_shift,
        residual=False,
    )

    out = block(graph)

    assert jnp.allclose(out.nodes[:, 0], jnp.array([12.0, 12.0, 16.0, 16.0]))
