#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _square_complex():
    vertices = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return phx.graph.triangle_mesh_to_cochain_complex(vertices, faces)


def _annulus_complex():
    outer = np.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    vertices = np.concatenate((outer, 0.4 * outer), axis=0)
    faces = np.asarray(
        [(index, (index + 1) % 4, 4 + (index + 1) % 4) for index in range(4)]
        + [(index, 4 + (index + 1) % 4, 4 + index) for index in range(4)],
        dtype=np.int32,
    )
    return phx.graph.triangle_mesh_to_cochain_complex(vertices, faces)


def _degree_values(complex_ir, degree, values):
    packed = jnp.zeros((complex_ir.num_cells,), dtype=jnp.asarray(values).dtype)
    start = complex_ir.cell_offsets[degree]
    return packed.at[start : start + complex_ir.cell_counts[degree]].set(values)


def _degree_slice(complex_ir, degree):
    start = complex_ir.cell_offsets[degree]
    return slice(start, start + complex_ir.cell_counts[degree])


def test_triangle_mesh_cochain_complex_has_exact_incidence_and_metric_graph():
    complex_ir = _square_complex()
    boundary_1 = complex_ir.incidences[0].scipy_matrix()
    boundary_2 = complex_ir.incidences[1].scipy_matrix()

    assert complex_ir.cell_counts == (4, 5, 2)
    assert complex_ir.graph.num_nodes == 11
    assert complex_ir.graph.num_edges == 32
    assert np.array_equal((boundary_1 @ boundary_2).toarray(), np.zeros((4, 2)))
    assert all(np.all(np.asarray(star) > 0.0) for star in complex_ir.hodge_stars)
    assert np.count_nonzero(np.asarray(complex_ir.boundary_masks[0])) == 4
    assert np.count_nonzero(np.asarray(complex_ir.boundary_masks[1])) == 4
    assert np.count_nonzero(np.asarray(complex_ir.boundary_masks[2])) == 0
    complex_ir.validate()


def test_sparse_dec_operators_satisfy_exactness_adjointness_and_positive_energy():
    complex_ir = _square_complex()
    graph = complex_ir.graph
    zero_form = _degree_values(complex_ir, 0, jnp.asarray([0.3, -0.2, 0.7, 1.1]))
    one_form = _degree_values(
        complex_ir,
        1,
        jnp.asarray([0.5, -0.4, 0.8, 0.2, -0.6]),
    )

    derivative = phx.graph.cochain_exterior_derivative(graph, zero_form, 0)
    second_derivative = phx.graph.cochain_exterior_derivative(graph, derivative, 1)
    codifferential = phx.graph.cochain_codifferential(graph, one_form, 1)
    star = jnp.asarray(graph.nodes["hodge_star"])
    left_inner_product = jnp.sum(star * derivative * one_form)
    right_inner_product = jnp.sum(star * zero_form * codifferential)

    lower = phx.graph.cochain_hodge_laplacian(graph, one_form, 1, component="lower")
    upper = phx.graph.cochain_hodge_laplacian(graph, one_form, 1, component="upper")
    complete = phx.graph.cochain_hodge_laplacian(graph, one_form, 1)
    energy = jnp.sum(star * one_form * complete)

    assert jnp.allclose(second_derivative, 0.0, atol=1e-12)
    assert jnp.allclose(left_inner_product, right_inner_product, atol=1e-12)
    assert jnp.allclose(complete, lower + upper, atol=1e-12)
    assert energy >= -1e-12


def test_graphir_dec_wrappers_match_functional_operators():
    complex_ir = _square_complex()
    values = _degree_values(complex_ir, 0, jnp.asarray([0.0, 1.0, 2.0, 3.0]))
    graph = complex_ir.graph.replace(
        nodes={**complex_ir.graph.nodes, "potential": values},
        validate=False,
    )

    differentiated = phx.graph.CochainExteriorDerivative(
        0,
        input_key="potential",
        output_key="gradient",
    )(graph)
    laplacian = phx.graph.CochainHodgeLaplacian(
        0,
        input_key="potential",
        output_key="laplacian",
    )(graph)

    assert jnp.allclose(
        differentiated.nodes["gradient"],
        phx.graph.cochain_exterior_derivative(graph, values, 0),
    )
    assert jnp.allclose(
        laplacian.nodes["laplacian"],
        phx.graph.cochain_hodge_laplacian(graph, values, 0),
    )


def test_harmonic_preprocessing_recovers_disconnected_and_annulus_betti_numbers():
    disconnected = phx.graph.triangle_mesh_to_cochain_complex(
        np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [3.0, 0.0],
                [4.0, 0.0],
                [3.0, 1.0],
            ]
        ),
        np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
    )
    annulus = _annulus_complex()

    disconnected_harmonics = phx.graph.compute_harmonic_subspace(
        disconnected, max_modes=4
    )
    absolute = phx.graph.compute_harmonic_subspace(annulus, max_modes=4)
    relative = phx.graph.compute_harmonic_subspace(
        annulus,
        boundary_policy="relative",
        max_modes=4,
    )

    assert disconnected_harmonics.ranks == (2, 0, 0)
    assert absolute.ranks == (1, 1, 0)
    assert relative.ranks == (0, 1, 1)


def test_harmonic_projection_is_metric_orthogonal_idempotent_and_laplacian_null():
    base = _annulus_complex()
    harmonics = phx.graph.compute_harmonic_subspace(base, max_modes=3)
    complex_ir = base.with_harmonic_subspace(harmonics)
    graph = complex_ir.graph
    one_form = _degree_values(
        complex_ir,
        1,
        jnp.linspace(-1.0, 1.0, complex_ir.cell_counts[1]),
    )

    projected = phx.graph.cochain_harmonic_projection(graph, one_form, 1)
    projected_twice = phx.graph.cochain_harmonic_projection(graph, projected, 1)
    residual = phx.graph.cochain_hodge_laplacian(graph, projected, 1)
    basis = harmonics.bases[1][:, : harmonics.ranks[1]]
    metric = complex_ir.hodge_stars[1]

    assert jnp.allclose(projected_twice, projected, atol=1e-10)
    assert jnp.allclose(residual, 0.0, atol=1e-9)
    assert jnp.allclose(
        basis.T @ (metric[:, None] * basis),
        jnp.eye(harmonics.ranks[1]),
        atol=1e-9,
    )


def test_orientation_changes_conjugate_exterior_codifferential_and_laplacian():
    complex_ir = _square_complex()
    signs = (
        np.asarray([1.0, -1.0, 1.0, -1.0]),
        np.asarray([-1.0, 1.0, -1.0, 1.0, -1.0]),
        np.asarray([1.0, -1.0]),
    )
    reoriented = phx.graph.reorient_cochain_complex(complex_ir, signs)
    zero_form = jnp.asarray([0.2, -0.5, 0.7, 1.3])
    one_form = jnp.asarray([0.4, -0.1, 0.8, -0.3, 0.6])
    packed_zero = _degree_values(complex_ir, 0, zero_form)
    packed_one = _degree_values(complex_ir, 1, one_form)
    reoriented_zero = _degree_values(
        reoriented,
        0,
        phx.graph.reorient_cochain(zero_form, signs[0]),
    )
    reoriented_one = _degree_values(
        reoriented,
        1,
        phx.graph.reorient_cochain(one_form, signs[1]),
    )

    derivative = phx.graph.cochain_exterior_derivative(complex_ir.graph, packed_zero, 0)
    transformed_derivative = phx.graph.cochain_exterior_derivative(
        reoriented.graph, reoriented_zero, 0
    )
    codifferential = phx.graph.cochain_codifferential(complex_ir.graph, packed_one, 1)
    transformed_codifferential = phx.graph.cochain_codifferential(
        reoriented.graph, reoriented_one, 1
    )
    laplacian = phx.graph.cochain_hodge_laplacian(complex_ir.graph, packed_one, 1)
    transformed_laplacian = phx.graph.cochain_hodge_laplacian(
        reoriented.graph, reoriented_one, 1
    )

    assert jnp.allclose(
        transformed_derivative[_degree_slice(reoriented, 1)],
        phx.graph.reorient_cochain(derivative[_degree_slice(complex_ir, 1)], signs[1]),
        atol=1e-12,
    )
    assert jnp.allclose(
        transformed_codifferential[_degree_slice(reoriented, 0)],
        phx.graph.reorient_cochain(
            codifferential[_degree_slice(complex_ir, 0)], signs[0]
        ),
        atol=1e-12,
    )
    assert jnp.allclose(
        transformed_laplacian[_degree_slice(reoriented, 1)],
        phx.graph.reorient_cochain(laplacian[_degree_slice(complex_ir, 1)], signs[1]),
        atol=1e-12,
    )


def test_relative_boundary_policy_masks_boundary_cochains():
    complex_ir = _square_complex()
    zero_form = _degree_values(complex_ir, 0, jnp.ones((4,)))
    one_form = _degree_values(complex_ir, 1, jnp.ones((5,)))

    relative_derivative = phx.graph.cochain_exterior_derivative(
        complex_ir.graph,
        zero_form,
        0,
        boundary_policy="relative",
    )
    relative_codifferential = phx.graph.cochain_codifferential(
        complex_ir.graph,
        one_form,
        1,
        boundary_policy="relative",
    )

    assert jnp.allclose(relative_derivative, 0.0)
    assert jnp.allclose(relative_codifferential, 0.0)


def _centered_square_complex():
    vertices = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )
    faces = np.asarray(
        [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        dtype=np.int32,
    )
    return phx.graph.triangle_mesh_to_cochain_complex(vertices, faces)


def test_cochain_cells_select_degree_boundary_and_padded_dataset_offsets():
    small = _square_complex()
    large = _centered_square_complex()
    structure = phx.domain.SampleLayout((("graph",),))

    fixed_domain = phx.domain.GraphDomain(large.graph)
    edges = fixed_domain.component({"graph": phx.domain.CochainCells(1)}).sample(
        phx.domain.PointSampling(large.cell_counts[1], layout=structure)
    )
    boundary_vertices = fixed_domain.component(
        {"graph": phx.domain.CochainCells(0, region="boundary")}
    ).sample(phx.domain.PointSampling(4, layout=structure))
    interior_vertices = fixed_domain.component(
        {"graph": phx.domain.CochainCells(0, region="interior")}
    ).sample(phx.domain.PointSampling(1, layout=structure))

    assert jnp.all(edges["graph"]["cell_dim"].data == 1)
    assert jnp.all(boundary_vertices["graph"]["boundary"].data)
    assert jnp.array_equal(
        interior_vertices["graph"]["local_index"].data,
        jnp.asarray([4], dtype=jnp.int32),
    )

    base_dataset = phx.domain.GraphDatasetDomain((small.graph, large.graph))
    dataset = base_dataset.with_layout(base_dataset.layout_for_batch_size(2, multiple=4))
    dataset_batch = dataset.points_from_indices(
        [0, 1],
        component=phx.domain.CochainCells(0, region="interior"),
        structure=structure,
    )

    assert dataset_batch.graph.node_mask is not None
    assert jnp.array_equal(
        dataset_batch[phx.domain.graph.GRAPH_DATASET_INDEX_KEY].data,
        jnp.asarray([1], dtype=jnp.int32),
    )
    assert jnp.array_equal(
        dataset_batch["graph"]["local_index"].data,
        jnp.asarray([4], dtype=jnp.int32),
    )

    base_trajectory = phx.domain.GraphTrajectoryDatasetDomain(
        (small.graph, large.graph),
        jnp.asarray([2, 3], dtype=jnp.int32),
        dt=0.5,
    )
    trajectory = base_trajectory.with_layout(
        base_trajectory.layout_for_batch_size(2, multiple=4)
    )
    trajectory_component = trajectory.component(
        {
            "graph": phx.domain.CochainCells(0, region="interior"),
            "t": phx.domain.Interior(),
        }
    )
    trajectory_batch = trajectory.points_from_case_time(
        [0, 1],
        [0.5, 0.5],
        component=trajectory_component,
        structure=phx.domain.SampleLayout((("graph", "t"),)),
    )

    assert trajectory_batch.graph.node_mask is not None
    assert jnp.array_equal(
        trajectory_batch[phx.domain.graph.GRAPH_DATASET_INDEX_KEY].data,
        jnp.asarray([1], dtype=jnp.int32),
    )
    assert jnp.array_equal(
        trajectory_batch["graph"]["local_index"].data,
        jnp.asarray([4], dtype=jnp.int32),
    )
    assert jnp.allclose(trajectory_batch["t"].data, 0.5)


def test_cochain_field_masks_other_degrees_and_preserves_compatible_metadata():
    complex_ir = _square_complex()
    domain = phx.domain.GraphDomain(complex_ir.graph)
    structure = phx.domain.SampleLayout((("graph",),))
    all_cells = domain.component({"graph": phx.domain.Nodes()}).sample(
        phx.domain.PointSampling(complex_ir.num_cells, layout=structure)
    )
    zero_spec = phx.graph.CochainFieldSpec(
        0,
        cell_orientation="invariant",
        sampling="point_value",
    )
    one_spec = phx.graph.CochainFieldSpec(
        1,
        cell_orientation="signed",
        sampling="cell_integral",
    )

    @domain.Function("graph")
    def raw(cell):
        return 1.0 + cell["local_index"]

    zero_form = phx.domain.as_cochain_field(raw, zero_spec)
    another_zero_form = phx.domain.as_cochain_field(2.0 * raw, zero_spec)
    one_form = phx.domain.as_cochain_field(raw, one_spec)
    values = zero_form(all_cells).data
    degree = all_cells["graph"]["cell_dim"].data

    assert jnp.all(values[degree != 0] == 0.0)
    assert phx.domain.cochain_field_spec(3.0 * zero_form - another_zero_form) == zero_spec
    with pytest.raises(ValueError, match="no declared cochain field semantics"):
        phx.domain.cochain_field_spec(zero_form + one_form)


def test_domain_cochain_dec_is_exact_and_matches_sparse_graph_operators():
    complex_ir = _square_complex()
    domain = phx.domain.GraphDomain(complex_ir.graph)
    structure = phx.domain.SampleLayout((("graph",),))
    all_cells = domain.component({"graph": phx.domain.Nodes()}).sample(
        phx.domain.PointSampling(complex_ir.num_cells, layout=structure)
    )
    edge_batch = domain.component({"graph": phx.domain.CochainCells(1)}).sample(
        phx.domain.PointSampling(complex_ir.cell_counts[1], layout=structure)
    )
    face_batch = domain.component({"graph": phx.domain.CochainCells(2)}).sample(
        phx.domain.PointSampling(complex_ir.cell_counts[2], layout=structure)
    )
    zero_spec = phx.graph.CochainFieldSpec(
        0,
        cell_orientation="invariant",
        sampling="point_value",
    )

    @domain.Function("graph")
    def raw(cell):
        return 0.25 + cell["local_index"]

    zero_form = phx.domain.as_cochain_field(raw, zero_spec)
    derivative = phx.operators.cochain_exterior_derivative(zero_form)
    second_derivative = phx.operators.cochain_exterior_derivative(derivative)
    laplacian = phx.operators.cochain_hodge_laplacian(zero_form)
    packed = zero_form(all_cells).data
    expected_derivative = phx.graph.cochain_exterior_derivative(
        complex_ir.graph,
        packed,
        0,
    )
    expected_laplacian = phx.graph.cochain_hodge_laplacian(
        complex_ir.graph,
        packed,
        0,
    )
    edge_indices = edge_batch[phx.domain.graph.GRAPH_ENTITY_INDEX_KEY].data
    vertex_batch = domain.component({"graph": phx.domain.CochainCells(0)}).sample(
        phx.domain.PointSampling(complex_ir.cell_counts[0], layout=structure)
    )
    vertex_indices = vertex_batch[phx.domain.graph.GRAPH_ENTITY_INDEX_KEY].data

    assert phx.domain.cochain_field_spec(derivative).degree == 1
    assert phx.domain.cochain_field_spec(second_derivative).degree == 2
    assert jnp.allclose(derivative(edge_batch).data, expected_derivative[edge_indices])
    assert jnp.allclose(second_derivative(face_batch).data, 0.0, atol=1e-12)
    assert jnp.allclose(laplacian(vertex_batch).data, expected_laplacian[vertex_indices])


def test_domain_cochain_laplacian_is_equivariant_to_cell_reorientation():
    complex_ir = _square_complex()
    signs = (
        np.asarray([1.0, -1.0, 1.0, -1.0]),
        np.asarray([-1.0, 1.0, -1.0, 1.0, -1.0]),
        np.asarray([1.0, -1.0]),
    )
    reoriented = phx.graph.reorient_cochain_complex(complex_ir, signs)
    values = jnp.asarray([0.4, -0.1, 0.8, -0.3, 0.6])
    transformed_values = phx.graph.reorient_cochain(values, signs[1])
    one_spec = phx.graph.CochainFieldSpec(
        1,
        cell_orientation="signed",
        sampling="cell_integral",
    )
    structure = phx.domain.SampleLayout((("graph",),))

    def laplacian_values(bundle, coefficients):
        domain = phx.domain.GraphDomain(bundle.graph)

        @domain.Function("graph")
        def raw(cell):
            index = jnp.where(cell["cell_dim"] == 1, cell["local_index"], 0)
            return coefficients[index]

        one_form = phx.domain.as_cochain_field(raw, one_spec)
        batch = domain.component({"graph": phx.domain.CochainCells(1)}).sample(
            phx.domain.PointSampling(bundle.cell_counts[1], layout=structure)
        )
        return phx.operators.cochain_hodge_laplacian(one_form)(batch).data

    original = laplacian_values(complex_ir, values)
    transformed = laplacian_values(reoriented, transformed_values)

    assert jnp.allclose(
        transformed,
        phx.graph.reorient_cochain(original, signs[1]),
        atol=1e-12,
    )


def test_cochain_metric_reductions_ignore_padding_and_compose_segment_weights():
    values = jnp.asarray([1.0, 3.0, 1000.0, 2.0])
    metric = jnp.asarray([1.0, 3.0, 1000.0, 2.0])
    graph_index = jnp.asarray([0, 0, -1, 1], dtype=jnp.int32)
    active = jnp.asarray([True, True, False, True])

    graph_mean = phx.graph.cochain_metric_reduce(
        values,
        metric,
        graph_index,
        n_graph=2,
        reduction="graph_mean",
        entity_mask=active,
    )
    metric_mean = phx.graph.cochain_metric_reduce(
        values,
        metric,
        graph_index,
        n_graph=2,
        reduction="metric_mean",
        entity_mask=active,
    )
    metric_sum = phx.graph.cochain_metric_reduce(
        values,
        metric,
        graph_index,
        n_graph=2,
        reduction="metric_sum",
        entity_mask=active,
    )
    weighted_sum = phx.graph.cochain_metric_reduce(
        values,
        metric,
        graph_index,
        n_graph=2,
        reduction="metric_sum",
        segment_weight=jnp.asarray([2.0, 2.0, 1000.0, 4.0]),
        entity_mask=active,
    )

    assert jnp.allclose(graph_mean, 2.0)
    assert jnp.allclose(metric_mean, 2.25)
    assert jnp.allclose(metric_sum, 7.0)
    assert jnp.allclose(weighted_sum, 18.0)


@pytest.mark.parametrize(
    ("measure", "expected"),
    [
        ("time_integral_average", 1.5),
        ("time_integral_sum", 3.0),
    ],
)
def test_cochain_residual_constraint_composes_graph_and_time_measures(
    measure,
    expected,
):
    complex_ir = _square_complex()
    base = phx.domain.GraphTrajectoryDatasetDomain(
        (complex_ir.graph, complex_ir.graph),
        jnp.asarray([2, 3], dtype=jnp.int32),
        dt=1.0,
        measure=measure,
    )
    domain = base.with_layout(base.layout_for_batch_size(2, multiple=4))
    component = domain.component(
        {
            "graph": phx.domain.CochainCells(0),
            "t": phx.domain.Interior(),
        }
    )
    structure = phx.domain.SampleLayout((("graph", "t"),))
    zero_spec = phx.graph.CochainFieldSpec(
        0,
        cell_orientation="invariant",
        sampling="point_value",
    )

    @domain.Function("graph", "t")
    def unit_residual(cell, time):
        return jnp.ones_like(time) + 0.0 * cell["local_index"]

    residual = phx.domain.as_cochain_field(unit_residual, zero_spec)
    batch = domain.points_from_case_time(
        [0, 1],
        [0.5, 1.0],
        component=component,
        structure=structure,
    )
    constraint = phx.terms.CochainResidualTerm(
        component=component,
        residual=lambda functions: functions["u"],
        fields=("u",),
        sampling=phx.domain.PointSampling(2, layout=structure),
        reduction="graph_mean",
    )

    assert jnp.allclose(
        constraint.loss({"u": residual}, batch=batch),
        expected,
    )
