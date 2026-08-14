#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import opt_einsum as oe

import phydrax as phx


def _triangle_graph(positions=None) -> phx.graph.GraphIR:
    if positions is None:
        positions = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
    return phx.graph.GraphIR(
        nodes={
            "positions": positions,
            "features": jnp.array([[1.0], [2.0], [3.0]]),
        },
        senders=jnp.array([0, 1, 2], dtype=jnp.int32),
        receivers=jnp.array([1, 2, 0], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([3], dtype=jnp.int32),
    )


def test_euclidean_edge_features_are_rigid_motion_invariant():
    graph = phx.graph.euclidean_edge_features(_triangle_graph())
    rotation = jnp.array([[0.0, -1.0], [1.0, 0.0]])
    translation = jnp.array([3.0, -2.0])
    moved_positions = _triangle_graph().nodes["positions"] @ rotation.T + translation
    moved = phx.graph.euclidean_edge_features(_triangle_graph(moved_positions))

    assert jnp.allclose(graph.edges["distance"], moved.edges["distance"])
    assert jnp.allclose(
        graph.edges["squared_distance"],
        moved.edges["squared_distance"],
    )
    expected_relative = oe.contract("ij,ej->ei", rotation, graph.edges["relative"])
    assert jnp.allclose(moved.edges["relative"], expected_relative)


def test_gaussian_radial_basis_expands_distances():
    out = phx.graph.gaussian_radial_basis(
        jnp.array([[0.0], [1.0]]),
        jnp.array([0.0, 1.0]),
        gamma=2.0,
    )

    assert out.shape == (2, 2)
    assert jnp.allclose(out[0], jnp.array([1.0, jnp.exp(-2.0)]))
    assert jnp.allclose(out[1], jnp.array([jnp.exp(-2.0), 1.0]))


def test_equivariant_graph_convolution_respects_rigid_motion():
    graph = _triangle_graph()
    moved_positions = graph.nodes["positions"] @ jnp.array([[0.0, -1.0], [1.0, 0.0]]).T
    moved = _triangle_graph(moved_positions + jnp.array([4.0, 5.0]))
    model = phx.graph.EquivariantGraphConvolution(
        input_key="features",
        scalar_output_key="scalar",
        vector_output_key="vector",
    )

    out = model(graph)
    moved_out = model(moved)
    rotation = jnp.array([[0.0, -1.0], [1.0, 0.0]])
    expected_vector = oe.contract("ij,njf->nif", rotation, out.nodes["vector"])

    assert jnp.allclose(moved_out.nodes["scalar"], out.nodes["scalar"])
    assert jnp.allclose(moved_out.nodes["vector"], expected_vector)


def test_equivariant_graph_convolution_supports_radial_weights_and_normalization():
    graph = phx.graph.euclidean_edge_features(_triangle_graph())

    def radial(edges, distance, unit, sent, recv):
        del distance, unit, sent, recv
        return 1.0 / edges["distance"][:, 0]

    out = phx.graph.EquivariantGraphConvolution(
        radial,
        input_key="features",
        scalar_output_key="scalar",
        vector_output_key="vector",
        normalize=True,
    )(graph)

    assert jnp.allclose(out.nodes["scalar"][:, 0], jnp.array([3.0, 1.0, 2.0]))
    assert out.nodes["vector"].shape == (3, 2, 1)


def test_equivariant_graph_convolution_wraps_as_graph_model():
    graph = _triangle_graph()
    domain = phx.domain.GraphDomain(graph)
    nodes = domain.component({"graph": phx.domain.Nodes()})
    batch = nodes.sample(
        phx.domain.PointSampling(3, layout=phx.domain.SampleLayout((("graph",),)))
    )

    @domain.Function("graph")
    def u(node):
        return node.get("features")[0]

    model = domain.GraphModel(
        phx.graph.EquivariantGraphConvolution(
            input_key="u",
            scalar_output_key="scalar",
            vector_output_key="vector",
        ),
        input_fn=u,
        input_key="u",
        output_key="scalar",
    )

    assert jnp.allclose(model(batch).data[:, 0], jnp.array([3.0, 1.0, 2.0]))
