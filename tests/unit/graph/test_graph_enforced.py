#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

import phydrax as phx


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


def test_enforce_graph_values_overwrites_boundary_nodes_and_satisfies_constraint():
    graph = _line_graph()
    domain = phx.domain.GraphDomain(graph)
    structure = phx.domain.ProductStructure((("graph",),))
    nodes = domain.component({"graph": phx.domain.Nodes()})
    boundary = domain.component({"graph": phx.domain.BoundaryNodes([0, 2])})
    node_batch = nodes.sample(graph.num_nodes, structure=structure)

    @domain.Function("graph")
    def u(node):
        return node[0]

    hard_u = phx.constraints.enforce_graph_values(u, boundary, target=5.0)
    assert jnp.allclose(hard_u(node_batch).data, jnp.array([5.0, 1.0, 5.0]))

    constraint = phx.constraints.FunctionalConstraint.from_operator(
        component=boundary,
        operator=lambda f: f - 5.0,
        constraint_vars="u",
        num_points=2,
        structure=structure,
    )
    assert constraint.loss({"u": hard_u}) < 1e-12


def test_enforce_graph_values_is_seen_by_graph_gradient_full_node_view():
    graph = phx.graph.GraphIR(
        nodes=jnp.zeros((2, 1)),
        senders=jnp.array([0], dtype=jnp.int32),
        receivers=jnp.array([1], dtype=jnp.int32),
        n_node=jnp.array([2], dtype=jnp.int32),
        n_edge=jnp.array([1], dtype=jnp.int32),
    )
    domain = phx.domain.GraphDomain(graph)
    edge_batch = domain.component({"graph": phx.domain.Edges()}).sample(
        graph.num_edges,
        structure=phx.domain.ProductStructure((("graph",),)),
    )
    left = domain.component({"graph": phx.domain.BoundaryNodes([0])})

    @domain.Function("graph")
    def u(node):
        del node
        return 0.0

    hard_u = phx.constraints.enforce_graph_values(u, left, target=2.0)

    assert jnp.allclose(phx.operators.graph_gradient(hard_u)(edge_batch).data, -2.0)


def test_enforce_graph_values_supports_edge_and_global_components():
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
    structure = phx.domain.ProductStructure((("graph",),))
    edge_batch = domain.component({"graph": phx.domain.Edges()}).sample(
        graph.num_edges,
        structure=structure,
    )
    global_batch = domain.component({"graph": phx.domain.Globals()}).sample(
        graph.num_graphs,
        structure=structure,
    )

    @domain.Function("graph")
    def flux(edge):
        return edge[0]

    @domain.Function("graph")
    def scale(global_):
        return global_[0]

    hard_flux = phx.constraints.enforce_graph_values(
        flux,
        domain.component({"graph": phx.domain.EdgeSet([1])}),
        target=-1.0,
    )
    hard_scale = phx.constraints.enforce_graph_values(
        scale,
        domain.component({"graph": phx.domain.Globals()}),
        target=9.0,
    )

    assert jnp.allclose(hard_flux(edge_batch).data, jnp.array([2.0, -1.0]))
    assert jnp.allclose(hard_scale(global_batch).data, jnp.array([9.0]))


def test_enforce_graph_values_uses_local_indices_for_graph_dataset_batches():
    domain = phx.domain.GraphDatasetDomain(_graphs())
    full_nodes = domain.points_from_indices(
        [0, 1],
        component=phx.domain.Nodes(),
        structure=phx.domain.ProductStructure((("graph",),)),
    )
    boundary = domain.component({"graph": phx.domain.BoundaryNodes([1])})

    @domain.Function("graph")
    def u(node):
        return node[0]

    hard_u = phx.constraints.enforce_graph_values(u, boundary, target=7.0)

    assert jnp.allclose(hard_u(full_nodes).data, jnp.array([0.0, 7.0, 2.0, 7.0, 8.0]))


def test_enforce_graph_values_supports_time_dependent_graph_trajectory_targets():
    domain = phx.domain.GraphTrajectoryDatasetDomain(
        _graphs(),
        jnp.array([3, 5], dtype=jnp.int32),
        dt=0.5,
    )
    component = domain.component({"graph": phx.domain.Nodes(), "t": phx.domain.Interior()})
    batch = domain.points_from_case_time(
        [0, 1],
        [0.5, 1.0],
        component=component,
        structure=phx.domain.ProductStructure((("graph", "t"),)),
    )
    boundary = domain.component(
        {"graph": phx.domain.BoundaryNodes([1]), "t": phx.domain.Interior()}
    )

    @domain.Function("graph", "t")
    def u(node, t):
        del node, t
        return 0.0

    @domain.Function("graph", "t")
    def target(node, t):
        del node
        return 10.0 + t

    hard_u = phx.constraints.enforce_graph_values(u, boundary, target=target)

    assert jnp.allclose(hard_u(batch).data, jnp.array([0.0, 10.5, 0.0, 11.0, 0.0]))


def test_graph_value_enforcement_integrates_with_functional_solver_terms():
    graph = _line_graph()
    domain = phx.domain.GraphDomain(graph)
    structure = phx.domain.ProductStructure((("graph",),))
    boundary = domain.component({"graph": phx.domain.BoundaryNodes([0, 2])})
    node_batch = domain.component({"graph": phx.domain.Nodes()}).sample(
        graph.num_nodes,
        structure=structure,
    )

    @domain.Function("graph")
    def u(node):
        return node[0]

    term = phx.solver.SingleFieldEnforcedConstraint(
        "u",
        boundary,
        lambda f: phx.constraints.enforce_graph_values(f, boundary, target=5.0),
    )
    solver = phx.solver.FunctionalSolver(
        functions={"u": u},
        constraints=(),
        constraint_terms=[term],
    )

    assert jnp.allclose(solver["u"](node_batch).data, jnp.array([5.0, 1.0, 5.0]))


def _cochain_complex_with_interior_vertex():
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
    return phx.graph.triangle_mesh_to_cochain_complex(vertices, faces)


class _TrainableCellValues(eqx.Module):
    values: jax.Array

    def __call__(self, graph):
        nodes = dict(graph.nodes)
        nodes["candidate"] = self.values
        return graph.replace(nodes=nodes, validate=False)


def test_enforce_cochain_values_preserves_signed_semantics_and_rejects_mismatch():
    complex_ir = _cochain_complex_with_interior_vertex()
    domain = phx.domain.GraphDomain(complex_ir.graph)
    structure = phx.domain.ProductStructure((("graph",),))
    edge_spec = phx.graph.CochainFieldSpec(
        1,
        cell_orientation="signed",
        sampling="cell_integral",
    )
    vertex_spec = phx.graph.CochainFieldSpec(
        0,
        cell_orientation="invariant",
        sampling="point_value",
    )

    @domain.Function("graph")
    def raw(cell):
        return 2.0 + cell["local_index"]

    @domain.Function("graph")
    def target_raw(cell):
        return -3.0 - cell["local_index"]

    edge_form = phx.domain.as_cochain_field(raw, edge_spec)
    target = phx.domain.as_cochain_field(target_raw, edge_spec)
    vertex_form = phx.domain.as_cochain_field(raw, vertex_spec)
    boundary = domain.component(
        {"graph": phx.domain.CochainCells(1, region="boundary")}
    )
    all_edges = domain.component({"graph": phx.domain.CochainCells(1)}).sample(
        complex_ir.cell_counts[1],
        structure=structure,
    )
    hard = phx.constraints.enforce_cochain_values(
        edge_form,
        boundary,
        target=target,
    )
    boundary_mask = all_edges["graph"]["boundary"].data
    hard_values = hard(all_edges).data
    base_values = edge_form(all_edges).data
    target_values = target(all_edges).data

    assert phx.domain.cochain_field_spec(hard) == edge_spec
    assert jnp.allclose(hard_values[boundary_mask], target_values[boundary_mask])
    assert jnp.allclose(hard_values[~boundary_mask], base_values[~boundary_mask])
    with pytest.raises(ValueError, match="same degree, side, orientation"):
        phx.constraints.enforce_cochain_values(
            edge_form,
            boundary,
            target=vertex_form,
        )


def test_hard_cochain_boundary_remains_exact_during_solver_optimization():
    complex_ir = _cochain_complex_with_interior_vertex()
    domain = phx.domain.GraphDomain(complex_ir.graph)
    structure = phx.domain.ProductStructure((("graph",),))
    zero_spec = phx.graph.CochainFieldSpec(
        0,
        cell_orientation="invariant",
        sampling="point_value",
    )
    candidate = domain.GraphModel(
        _TrainableCellValues(jnp.zeros((complex_ir.num_cells,))),
        output_key="candidate",
    )
    field = phx.domain.as_cochain_field(candidate, zero_spec)
    boundary = domain.component(
        {"graph": phx.domain.CochainCells(0, region="boundary")}
    )
    field = phx.constraints.enforce_cochain_values(field, boundary, target=0.0)

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
        boundary_policy="absolute",
    )

    @domain.Function("graph")
    def forcing_raw(cell):
        index = jnp.where(cell["cell_dim"] == 0, cell["local_index"], 0)
        return forcing_values[index]

    forcing = phx.domain.as_cochain_field(forcing_raw, zero_spec)
    interior = domain.component(
        {"graph": phx.domain.CochainCells(0, region="interior")}
    )
    constraint = phx.constraints.CochainResidualConstraint(
        component=interior,
        residual=lambda functions: phx.operators.cochain_hodge_laplacian(
            functions["u"],
            boundary_policy="absolute",
        )
        - forcing,
        constraint_vars=("u",),
        num_points=1,
        structure=structure,
        reduction="metric_sum",
        sampling_mode="fixed",
    )
    solver = phx.solver.FunctionalSolver(
        functions={"u": field},
        constraints=(constraint,),
    )
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
        complex_ir.cell_counts[0],
        structure=structure,
    )
    prediction = trained["u"](vertices).data
    boundary_mask = vertices["graph"]["boundary"].data

    assert final_loss < 0.01 * initial_loss
    assert jnp.all(prediction[boundary_mask] == 0.0)
    assert jnp.allclose(prediction[~boundary_mask], 1.0, atol=0.15)
