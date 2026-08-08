import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx


def _right_triangle():
    vertices = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    faces = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    return vertices, faces


def test_mesh_geometry_primitives_for_right_triangle():
    vertices, faces = _right_triangle()

    assert jnp.allclose(phx.graph.mesh_face_areas(vertices, faces), jnp.array([0.5]))
    assert jnp.allclose(
        phx.graph.mesh_lumped_vertex_areas(vertices, faces),
        jnp.full((3,), 1.0 / 6.0),
    )
    assert jnp.allclose(
        phx.graph.mesh_face_normals(vertices, faces),
        jnp.array([[0.0, 0.0, 1.0]]),
    )
    assert jnp.allclose(
        phx.graph.mesh_vertex_normals(vertices, faces),
        jnp.tile(jnp.array([[0.0, 0.0, 1.0]]), (3, 1)),
    )


def test_mesh_cotangent_weights_for_right_triangle():
    vertices, faces = _right_triangle()

    senders, receivers, weights = phx.graph.mesh_cotangent_weights(vertices, faces)
    lookup = {
        (int(sender), int(receiver)): float(weight)
        for sender, receiver, weight in zip(
            np.asarray(senders),
            np.asarray(receivers),
            np.asarray(weights),
            strict=True,
        )
    }

    assert lookup[(0, 1)] == lookup[(1, 0)] == 0.5
    assert lookup[(0, 2)] == lookup[(2, 0)] == 0.5
    assert abs(lookup[(1, 2)]) < 1e-7
    assert abs(lookup[(2, 1)]) < 1e-7


def test_mesh_to_cotangent_graph_attaches_mass_and_weights():
    vertices, faces = _right_triangle()

    bundle = phx.graph.mesh_to_cotangent_graph(vertices, faces)

    assert bundle.graph.num_nodes == 3
    assert bundle.graph.num_edges == 6
    assert "mass" in bundle.graph.nodes
    assert "cotangent_weight" in bundle.graph.edges
    assert bundle.graph.nodes["mass"].shape == (3,)
    assert bundle.graph.edges["cotangent_weight"].shape == (6,)
    assert jnp.allclose(bundle.graph.nodes["mass"], jnp.full((3,), 1.0 / 6.0))


def test_mesh_cotangent_laplacian_zero_for_constant_field():
    vertices, faces = _right_triangle()
    graph = phx.graph.mesh_to_cotangent_graph(vertices, faces).graph
    graph = graph.replace(
        nodes={**graph.nodes, "u": jnp.ones((3,))},
        validate=False,
    )

    out = phx.graph.MeshCotangentLaplacian(
        sign="neighbor_minus_self", input_key="u", output_key="lap_u"
    )(graph)

    assert jnp.allclose(out.nodes["lap_u"], jnp.zeros((3,)))


def test_mesh_cotangent_laplacian_known_linear_field_on_open_triangle():
    vertices, faces = _right_triangle()
    graph = phx.graph.mesh_to_cotangent_graph(vertices, faces).graph
    graph = graph.replace(
        nodes={**graph.nodes, "u": vertices[:, 0]},
        validate=False,
    )

    out = phx.graph.MeshCotangentLaplacian(
        sign="neighbor_minus_self", input_key="u", output_key="lap_u"
    )(graph)

    assert jnp.allclose(out.nodes["lap_u"], jnp.array([3.0, -3.0, 0.0]))


def test_mesh_cotangent_laplacian_integrates_with_graph_model_keys():
    vertices, faces = _right_triangle()
    bundle = phx.graph.mesh_to_cotangent_graph(vertices, faces)
    domain = phx.domain.GraphDomain(bundle.graph)
    nodes = domain.component({"graph": phx.domain.Nodes()})
    structure = phx.domain.SampleLayout((("graph",),))

    @domain.Function("graph")
    def u(node):
        del node
        return 1.0

    def residual(f):
        return domain.GraphModel(
            phx.graph.MeshCotangentLaplacian(
                sign="neighbor_minus_self", input_key="u", output_key="lap_u"
            ),
            input_fn=f,
            input_key="u",
            output_key="lap_u",
        )

    condition = phx.conditions.Residual("u", nodes, residual)
    source = phx.integration.per_step(
        phx.integration.mean_over(nodes),
        phx.domain.PointSampling(bundle.graph.num_nodes, layout=structure),
    )
    term = phx.terms.ResidualPenalty(condition, source)

    assert term.loss({"u": u}, key=jr.key(0)) < 1e-12
