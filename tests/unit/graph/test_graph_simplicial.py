#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _single_triangle() -> phx.graph.SimplicialComplexGraph:
    return phx.graph.triangle_mesh_to_simplicial_graph(
        jnp.array([[0, 1, 2]], dtype=jnp.int32),
        vertex_features=jnp.array([[1.0], [2.0], [3.0]]),
    )


def test_triangle_mesh_to_simplicial_graph_builds_signed_cell_complex():
    bundle = _single_triangle()
    graph = bundle.graph

    assert graph.num_nodes == 7
    assert graph.num_edges == 18
    assert jnp.allclose(
        graph.nodes["type"],
        jnp.array([0, 0, 0, 1, 1, 1, 2], dtype=jnp.int32),
    )
    assert jnp.allclose(bundle.edge_vertices, jnp.array([[0, 1], [0, 2], [1, 2]]))
    assert jnp.allclose(bundle.face_edges, jnp.array([[0, 2, 1]], dtype=jnp.int32))
    assert jnp.allclose(bundle.face_edge_signs, jnp.array([[1.0, 1.0, -1.0]]))
    assert jnp.allclose(
        graph.edges["type"],
        jnp.array(
            [0, 0, 0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 1, 1, 1, 3, 3, 3],
            dtype=jnp.int32,
        ),
    )


def test_simplicial_bundle_components_select_cells_and_incidences():
    bundle = _single_triangle()
    domain = phx.domain.GraphDomain(bundle.graph, measure="count")
    structure = phx.domain.ProductStructure((("graph",),))
    vertices = domain.component({"graph": bundle.vertex_cells_component()})
    edges = domain.component({"graph": bundle.edge_cells_component()})
    incidence = domain.component({"graph": bundle.edge_to_face_component()})

    vertex_batch = vertices.sample(3, structure=structure)
    edge_batch = edges.sample(3, structure=structure)
    incidence_batch = incidence.sample(3, structure=structure)

    assert jnp.allclose(vertex_batch["graph"]["features"].data[:, 0], jnp.array([1.0, 2.0, 3.0]))
    assert jnp.allclose(edge_batch["graph"]["features"].data[:, 0], jnp.zeros((3,)))
    assert jnp.allclose(
        incidence_batch["graph"]["incidence_sign"].data,
        jnp.array([1.0, 1.0, -1.0]),
    )
    assert vertices.measure() == 3.0


def test_simplicial_hodge_laplacian_zero_for_constant_zero_form():
    graph = _single_triangle().graph
    graph = graph.replace(nodes={**graph.nodes, "u": jnp.ones((7,))}, validate=False)

    out = phx.graph.SimplicialHodgeLaplacian(0, input_key="u", output_key="lap_u")(graph)

    assert jnp.allclose(out.nodes["lap_u"], jnp.zeros((7,)))


def test_simplicial_hodge_laplacian_known_zero_form_on_triangle():
    graph = _single_triangle().graph
    u = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    graph = graph.replace(nodes={**graph.nodes, "u": u}, validate=False)

    out = phx.graph.SimplicialHodgeLaplacian(0, input_key="u", output_key="lap_u")(graph)

    assert jnp.allclose(
        out.nodes["lap_u"],
        jnp.array([-1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
    )


def test_simplicial_hodge_laplacian_known_one_form_circulation():
    graph = _single_triangle().graph
    alpha = jnp.array([0.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0])
    graph = graph.replace(nodes={**graph.nodes, "alpha": alpha}, validate=False)

    out = phx.graph.SimplicialHodgeLaplacian(1, input_key="alpha", output_key="lap_alpha")(
        graph
    )

    assert jnp.allclose(
        out.nodes["lap_alpha"],
        jnp.array([0.0, 0.0, 0.0, 3.0, -3.0, 3.0, 0.0]),
    )


def test_simplicial_hodge_laplacian_integrates_with_graph_model_and_constraints():
    bundle = _single_triangle()
    domain = phx.domain.GraphDomain(bundle.graph)
    vertices = domain.component({"graph": bundle.vertex_cells_component()})
    structure = phx.domain.ProductStructure((("graph",),))
    table = jnp.array([0.0, 1.0, 0.0])

    @domain.Function("graph")
    def u(cell):
        return jnp.where(cell["cell_dim"] == 0, table[cell["local_index"]], 0.0)

    def residual(f):
        return domain.GraphModel(
            phx.graph.SimplicialHodgeLaplacian(0, input_key="u", output_key="lap_u"),
            input_fn=f,
            input_key="u",
            output_key="lap_u",
        )

    model = residual(u)
    batch = vertices.sample(3, structure=structure)
    constraint = phx.constraints.FunctionalConstraint.from_operator(
        component=vertices,
        operator=lambda f: residual(f) - model,
        constraint_vars="u",
        num_points=3,
        structure=structure,
    )

    assert jnp.allclose(model(batch).data, jnp.array([-1.0, 2.0, -1.0]))
    assert constraint.loss({"u": u}, key=jr.key(0)) < 1e-12
