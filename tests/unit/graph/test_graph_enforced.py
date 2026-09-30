#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

import phydrax as phx
from tests._support.cochain import triangle_cochain_lowering


def _line_graph() -> phx.graph.GraphIR:
    return phx.graph.GraphIR(
        nodes=jnp.array([[0.0], [1.0], [2.0]]),
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 2], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )


def _graphs() -> tuple[phx.graph.GraphIR, phx.graph.GraphIR]:
    graph0 = phx.graph.GraphIR(
        nodes=jnp.array([[0.0], [1.0]]),
        senders=jnp.array([0], dtype=jnp.int32),
        receivers=jnp.array([1], dtype=jnp.int32),
        n_node=jnp.array([2], dtype=jnp.int32),
        n_edge=jnp.array([1], dtype=jnp.int32),
    )
    graph1 = phx.graph.GraphIR(
        nodes=jnp.array([[2.0], [4.0], [8.0]]),
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 2], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )
    return graph0, graph1


def test_enforce_graph_values_overwrites_boundary_nodes_and_satisfies_residual() -> None:
    graph = _line_graph()
    domain = phx.domain.GraphDomain(graph)
    structure = phx.domain.SampleLayout((("graph",),))
    nodes = domain.component({"graph": phx.domain.Nodes()})
    # ty: ignore[invalid-argument-type]
    boundary = domain.component({"graph": phx.domain.BoundaryNodes([0, 2])})
    node_batch = nodes.sample(phx.domain.PointSampling(graph.num_nodes, layout=structure))

    @domain.Function("graph")
    def u(node: Any) -> Any:
        return node[0]

    hard_u = phx.enforcement.enforce_graph_values(u, boundary, target=5.0)
    assert jnp.allclose(jnp.asarray(hard_u(node_batch).data), jnp.array([5.0, 1.0, 5.0]))

    condition = phx.conditions.Residual("u", boundary, lambda f: f - 5.0)
    source = phx.integration.per_step(
        phx.integration.mean_over(boundary),
        phx.domain.PointSampling(2, layout=structure),
    )
    term = phx.terms.ResidualPenalty(condition, source)
    assert term.loss({"u": hard_u}) < 1e-12


def test_enforce_graph_values_is_seen_by_graph_gradient_full_node_view() -> None:
    graph = phx.graph.GraphIR(
        nodes=jnp.zeros((2, 1)),
        senders=jnp.array([0], dtype=jnp.int32),
        receivers=jnp.array([1], dtype=jnp.int32),
        n_node=jnp.array([2], dtype=jnp.int32),
        n_edge=jnp.array([1], dtype=jnp.int32),
    )
    domain = phx.domain.GraphDomain(graph)
    edge_batch = domain.component({"graph": phx.domain.Edges()}).sample(
        phx.domain.PointSampling(
            graph.num_edges, layout=phx.domain.SampleLayout((("graph",),))
        )
    )
    # ty: ignore[invalid-argument-type]
    left = domain.component({"graph": phx.domain.BoundaryNodes([0])})

    @domain.Function("graph")
    def u(node: Any) -> float:
        del node
        return 0.0

    hard_u = phx.enforcement.enforce_graph_values(u, left, target=2.0)

    assert jnp.allclose(
        jnp.asarray(phx.operators.graph_gradient(hard_u)(edge_batch).data), -2.0
    )


def test_enforce_graph_values_supports_edge_and_global_components() -> None:
    graph = phx.graph.GraphIR(
        nodes=jnp.zeros((2, 1)),
        edges=jnp.array([[2.0], [3.0]]),
        globals=jnp.array([[4.0]]),
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 0], dtype=jnp.int32),
        n_node=jnp.array([2], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )
    domain = phx.domain.GraphDomain(graph)
    structure = phx.domain.SampleLayout((("graph",),))
    edge_batch = domain.component({"graph": phx.domain.Edges()}).sample(
        phx.domain.PointSampling(graph.num_edges, layout=structure)
    )
    global_batch = domain.component({"graph": phx.domain.Globals()}).sample(
        phx.domain.PointSampling(graph.num_graphs, layout=structure)
    )

    @domain.Function("graph")
    def flux(edge: Any) -> Any:
        return edge[0]

    @domain.Function("graph")
    def scale(global_: Any) -> Any:
        return global_[0]

    hard_flux = phx.enforcement.enforce_graph_values(
        flux,
        # ty: ignore[invalid-argument-type]
        domain.component({"graph": phx.domain.EdgeSet([1])}),
        target=-1.0,
    )
    hard_scale = phx.enforcement.enforce_graph_values(
        scale,
        domain.component({"graph": phx.domain.Globals()}),
        target=9.0,
    )

    assert jnp.allclose(jnp.asarray(hard_flux(edge_batch).data), jnp.array([2.0, -1.0]))
    assert jnp.allclose(jnp.asarray(hard_scale(global_batch).data), jnp.array([9.0]))


def test_enforce_graph_values_uses_local_indices_for_graph_dataset_batches() -> None:
    domain = phx.domain.GraphDatasetDomain(_graphs())
    full_nodes = domain.points_from_indices(
        # ty: ignore[invalid-argument-type]
        [0, 1],
        component=phx.domain.Nodes(),
        structure=phx.domain.SampleLayout((("graph",),)),
    )
    # ty: ignore[invalid-argument-type]
    boundary = domain.component({"graph": phx.domain.BoundaryNodes([1])})

    @domain.Function("graph")
    def u(node: Any) -> Any:
        return node[0]

    hard_u = phx.enforcement.enforce_graph_values(u, boundary, target=7.0)

    assert jnp.allclose(
        jnp.asarray(hard_u(full_nodes).data), jnp.array([0.0, 7.0, 2.0, 7.0, 8.0])
    )


def test_enforce_graph_values_supports_time_dependent_graph_trajectory_targets() -> None:
    domain = phx.domain.GraphTrajectoryDatasetDomain(
        _graphs(),
        jnp.array([3, 5], dtype=jnp.int32),
        dt=0.5,
    )
    component = domain.component(
        {"graph": phx.domain.Nodes(), "t": phx.domain.Interior()}
    )
    batch = domain.points_from_case_time(
        [0, 1],
        [0.5, 1.0],
        component=component,
        structure=phx.domain.SampleLayout((("graph", "t"),)),
    )
    boundary = domain.component(
        # ty: ignore[invalid-argument-type]
        {"graph": phx.domain.BoundaryNodes([1]), "t": phx.domain.Interior()}
    )

    @domain.Function("graph", "t")
    def u(node: Any, t: Any) -> float:
        del node, t
        return 0.0

    @domain.Function("graph", "t")
    def target(node: Any, t: Any) -> Any:
        del node
        return 10.0 + t

    hard_u = phx.enforcement.enforce_graph_values(u, boundary, target=target)

    assert jnp.allclose(
        jnp.asarray(hard_u(batch).data), jnp.array([0.0, 10.5, 0.0, 11.0, 0.0])
    )


def test_graph_value_enforcement_integrates_with_functional_solver() -> None:
    graph = _line_graph()
    domain = phx.domain.GraphDomain(graph)
    structure = phx.domain.SampleLayout((("graph",),))
    # ty: ignore[invalid-argument-type]
    boundary = domain.component({"graph": phx.domain.BoundaryNodes([0, 2])})
    node_batch = domain.component({"graph": phx.domain.Nodes()}).sample(
        phx.domain.PointSampling(graph.num_nodes, layout=structure)
    )

    @domain.Function("graph")
    def u(node: Any) -> Any:
        return node[0]

    functions = {"u": u}
    condition = phx.conditions.Dirichlet("u", boundary, target=5.0)
    spec = phx.enforcement.EnforcementSpec(condition)
    program = phx.enforcement.compile(functions, [spec])
    solver = phx.solver.FunctionalSolver(
        functions=functions,
        terms=(),
        enforcement=program,
    )

    assert jnp.allclose(
        jnp.asarray(solver["u"](node_batch).data), jnp.array([5.0, 1.0, 5.0])
    )


def _cochain_complex_with_interior_vertex() -> Any:
    vertices = jnp.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )
    faces = jnp.asarray(
        [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        dtype=jnp.int32,
    )
    return triangle_cochain_lowering(vertices, faces)


class _TrainableCellValues(eqx.Module):
    values: jax.Array = phx.parameter_field()

    def __call__(self, graph: Any) -> Any:
        nodes = dict(graph.nodes)
        nodes["candidate"] = self.values
        return graph.replace(nodes=nodes, validate=False)


def test_enforce_cochain_values_preserves_signed_semantics_and_rejects_mismatch() -> None:
    complex_ir = _cochain_complex_with_interior_vertex()
    domain = phx.domain.GraphDomain(complex_ir.graph)
    structure = phx.domain.SampleLayout((("graph",),))
    edge_spec = phx.exterior.FormType(2, 1)
    vertex_spec = phx.exterior.FormType(2, 0)

    @domain.Function("graph")
    def raw(cell: Any) -> Any:
        return 2.0 + cell["local_index"]

    @domain.Function("graph")
    def target_raw(cell: Any) -> Any:
        return -3.0 - cell["local_index"]

    edge_form = phx.domain.as_cochain_field(raw, edge_spec, representation="cochain")
    target = phx.domain.as_cochain_field(target_raw, edge_spec, representation="cochain")
    vertex_form = phx.domain.as_cochain_field(raw, vertex_spec, representation="cochain")
    boundary = domain.component({"graph": phx.domain.CochainCells(1, region="boundary")})
    all_edges = domain.component({"graph": phx.domain.CochainCells(1)}).sample(
        phx.domain.PointSampling(complex_ir.cell_counts[1], layout=structure)
    )
    hard = phx.enforcement.enforce_cochain_values(
        edge_form,
        boundary,
        target=target,
    )
    boundary_mask = jnp.asarray(all_edges["graph"]["boundary"].data)
    hard_values = jnp.asarray(hard(all_edges).data)
    base_values = jnp.asarray(edge_form(all_edges).data)
    target_values = jnp.asarray(target(all_edges).data)

    assert phx.domain.cochain_form_type(hard) == edge_spec
    assert jnp.allclose(hard_values[boundary_mask], target_values[boundary_mask])
    assert jnp.allclose(hard_values[~boundary_mask], base_values[~boundary_mask])
    with pytest.raises((TypeError, ValueError)):
        phx.enforcement.enforce_cochain_values(
            edge_form,
            boundary,
            target=vertex_form,
        )


def test_hard_cochain_boundary_remains_exact_during_solver_optimization() -> None:
    complex_ir = _cochain_complex_with_interior_vertex()
    domain = phx.domain.GraphDomain(complex_ir.graph)
    structure = phx.domain.SampleLayout((("graph",),))
    zero_spec = phx.exterior.FormType(2, 0)
    candidate = domain.GraphModel(
        _TrainableCellValues(jnp.zeros((complex_ir.num_cells,))),
        output_key="candidate",
    )
    field = phx.domain.as_cochain_field(candidate, zero_spec, representation="cochain")
    boundary = domain.component({"graph": phx.domain.CochainCells(0, region="boundary")})
    field = phx.enforcement.enforce_cochain_values(field, boundary, target=0.0)

    exact_vertices = jnp.asarray([0.0, 0.0, 0.0, 0.0, 1.0])
    exact = jnp.where(
        complex_ir.graph.nodes["cell_dim"] == 0,
        exact_vertices[jnp.clip(complex_ir.graph.nodes["local_index"], 0, 4)],
        0.0,
    )
    forcing_values = phx.graph.cochain_hodge_laplacian(
        complex_ir.graph,
        exact,
        0,
        boundary="absolute",
    )

    @domain.Function("graph")
    def forcing_raw(cell: Any) -> Any:
        index = jnp.where(cell["cell_dim"] == 0, cell["local_index"], 0)
        return forcing_values[index]

    forcing = phx.domain.as_cochain_field(
        forcing_raw, zero_spec, representation="cochain"
    )
    interior = domain.component({"graph": phx.domain.CochainCells(0, region="interior")})
    term = phx.terms.CochainResidualTerm(
        component=interior,
        residual=lambda functions: (
            phx.operators.cochain_hodge_laplacian(
                functions["u"],
                boundary="absolute",
            )
            - forcing
        ),
        fields=("u",),
        sampling=phx.domain.PointSampling(1, layout=structure),
        reduction="metric_sum",
        sampling_mode="fixed",
    )
    solver = phx.solver.FunctionalSolver(functions={"u": field}, terms=(term,))
    initial_loss = solver.loss()
    trained = solver.solve(
        num_iter=40,
        optim=optax.adam(0.1),
        seed=3,
        keep_best=True,
        log_every=0,
    )
    final_loss = trained.loss()
    vertices = domain.component({"graph": phx.domain.CochainCells(0)}).sample(
        phx.domain.PointSampling(complex_ir.cell_counts[0], layout=structure)
    )
    prediction = trained["u"](vertices).data
    boundary_mask = vertices["graph"]["boundary"].data

    assert final_loss < 0.01 * initial_loss
    assert jnp.all(prediction[boundary_mask] == 0.0)
    assert jnp.allclose(prediction[~boundary_mask], 1.0, atol=0.15)
