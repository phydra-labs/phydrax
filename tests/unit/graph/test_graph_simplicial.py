#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _single_triangle() -> phx.graph.SimplicialComplexGraph:
    return phx.graph.triangle_mesh_to_simplicial_graph(
        jnp.array([[0, 1, 2]], dtype=jnp.int32),
        vertex_features=jnp.array([[1.0], [2.0], [3.0]]),
    )


def _metric_graph(bundle: phx.graph.SimplicialComplexGraph) -> phx.graph.GraphIR:
    realization = phx.discretization.CochainDiscretization(
        bundle.topology,
        tuple(
            phx.discretization.DiagonalHodge(jnp.ones((entities.count,)))
            for entities in bundle.topology.entity_sets
        ),
    )
    graph = phx.graph.CochainComplexIR(realization).graph
    return graph.replace(nodes={**bundle.graph.nodes, **graph.nodes}, validate=False)


def test_graph_simplicial_scenario_1() -> None:
    bundle = _single_triangle()
    graph = bundle.graph

    assert graph.num_nodes == 7
    assert graph.num_edges == 18
    assert jnp.allclose(
        graph.nodes["type"],
        jnp.array([0, 0, 0, 1, 1, 1, 2], dtype=jnp.int32),
    )
    edge_identities = [tuple(map(int, edge)) for edge in bundle.edge_vertices.tolist()]
    assert set(edge_identities) == {(0, 1), (0, 2), (1, 2)}
    oriented_boundary = {
        edge_identities[int(edge)]: float(sign)
        for edge, sign in zip(
            bundle.face_edges[0].tolist(), bundle.face_edge_signs[0].tolist(), strict=True
        )
    }
    assert oriented_boundary == {(0, 1): 1.0, (1, 2): 1.0, (0, 2): -1.0}
    bundle = _single_triangle()
    domain = phx.domain.GraphDomain(bundle.graph, measure="count")
    structure = phx.domain.SampleLayout((("graph",),))
    vertices = domain.component({"graph": bundle.vertex_cells_component()})
    edges = domain.component({"graph": bundle.edge_cells_component()})
    incidence = domain.component({"graph": bundle.edge_to_face_component()})

    vertex_batch = vertices.sample(phx.domain.PointSampling(3, layout=structure))
    edge_batch = edges.sample(phx.domain.PointSampling(3, layout=structure))
    incidence_batch = incidence.sample(phx.domain.PointSampling(3, layout=structure))

    assert jnp.allclose(
        vertex_batch["graph"]["features"].data[:, 0], jnp.array([1.0, 2.0, 3.0])
    )
    assert jnp.allclose(edge_batch["graph"]["features"].data[:, 0], jnp.zeros((3,)))
    assert sorted(
        map(float, incidence_batch["graph"]["incidence_sign"].data.tolist())
    ) == [-1.0, 1.0, 1.0]
    # ty: ignore[unresolved-attribute]
    assert vertices.mass.value == 3.0
    graph = _metric_graph(_single_triangle())
    graph = graph.replace(nodes={**graph.nodes, "u": jnp.ones((7,))}, validate=False)

    out = phx.graph.CochainHodgeLaplacian(0, input_key="u", output_key="lap_u")(graph)

    assert jnp.allclose(out.nodes["lap_u"], jnp.zeros((7,)))
    graph = _metric_graph(_single_triangle())
    u = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    graph = graph.replace(nodes={**graph.nodes, "u": u}, validate=False)

    out = phx.graph.CochainHodgeLaplacian(0, input_key="u", output_key="lap_u")(graph)

    assert jnp.allclose(
        out.nodes["lap_u"],
        jnp.array([-1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
    )
    graph = _metric_graph(_single_triangle())
    alpha = jnp.array([0.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0])
    graph = graph.replace(nodes={**graph.nodes, "alpha": alpha}, validate=False)

    out = phx.graph.CochainHodgeLaplacian(1, input_key="alpha", output_key="lap_alpha")(
        graph
    )

    assert jnp.allclose(
        out.nodes["lap_alpha"],
        jnp.array([0.0, 0.0, 0.0, 3.0, -3.0, 3.0, 0.0]),
    )


def test_hodge_laplacian_integrates_with_graph_model_and_constraints() -> None:
    bundle = _single_triangle()
    domain = phx.domain.GraphDomain(_metric_graph(bundle))
    vertices = domain.component({"graph": bundle.vertex_cells_component()})
    structure = phx.domain.SampleLayout((("graph",),))
    table = jnp.array([0.0, 1.0, 0.0])

    @domain.Function("graph")
    def u(cell: Any) -> Any:
        return jnp.where(cell["cell_dim"] == 0, table[cell["local_index"]], 0.0)

    def residual(f: Any) -> Any:
        return domain.GraphModel(
            phx.graph.CochainHodgeLaplacian(0, input_key="u", output_key="lap_u"),
            input_fn=f,
            input_key="u",
            output_key="lap_u",
        )

    model = residual(u)
    batch = vertices.sample(phx.domain.PointSampling(3, layout=structure))
    condition = phx.conditions.Residual("u", vertices, lambda f: residual(f) - model)
    source = phx.integration.per_step(
        phx.integration.mean_over(vertices),
        phx.domain.PointSampling(3, layout=structure),
    )
    term = phx.terms.ResidualPenalty(condition, source)

    assert jnp.allclose(jnp.asarray(model(batch).data), jnp.array([-1.0, 2.0, -1.0]))
    assert term.loss({"u": u}, key=jr.key(0)) < 1e-12
