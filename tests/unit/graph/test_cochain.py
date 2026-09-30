#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _triangle_lower(vertices: Any, faces: Any) -> phx.graph.CochainComplexIR:
    topology = phx.discretization.polygonal_cell_complex(faces, None, len(vertices))
    hodges = phx.discretization.simplicial_dual_hodges(
        topology, vertices, dual="barycentric"
    )
    geometry, _ = phx.discretization.simplicial_cell_geometry(topology)
    coordinates = tuple(
        jnp.mean(jnp.asarray(vertices)[cells], axis=1) for cells in geometry
    )
    realization = phx.discretization.CochainDiscretization(
        topology,
        hodges,
        boundary_masks=tuple(
            entities.subset("boundary").mask for entities in topology.entity_sets
        ),
        coordinates=coordinates,
    )
    return phx.graph.CochainComplexIR(realization)


def _reorient_lower(
    base: phx.graph.CochainComplexIR, signs: Any
) -> phx.graph.CochainComplexIR:
    realization = base.discretization
    return phx.graph.CochainComplexIR(
        phx.discretization.CochainDiscretization(
            phx.discretization.reorient_cell_complex(realization.topology, signs),
            realization.hodges,
            boundary_masks=realization.boundary_masks,
            coordinates=realization.coordinates,
            primal_measures=realization.primal_measures,
            dual_measures=realization.dual_measures,
        )
    )


def _square_complex() -> Any:
    vertices = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return _triangle_lower(vertices, faces)


def _annulus_complex() -> Any:
    outer = np.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    vertices = np.concatenate((outer, 0.4 * outer), axis=0)
    faces = np.asarray(
        [(index, (index + 1) % 4, 4 + (index + 1) % 4) for index in range(4)]
        + [(index, 4 + (index + 1) % 4, 4 + index) for index in range(4)],
        dtype=np.int32,
    )
    return _triangle_lower(vertices, faces)


def _degree_values(complex_ir: Any, degree: Any, values: Any) -> Any:
    packed = jnp.zeros((complex_ir.num_cells,), dtype=jnp.asarray(values).dtype)
    start = complex_ir.cell_offsets[degree]
    return packed.at[start : start + complex_ir.cell_counts[degree]].set(values)


def _degree_slice(complex_ir: Any, degree: Any) -> Any:
    start = complex_ir.cell_offsets[degree]
    return slice(start, start + complex_ir.cell_counts[degree])


def test_lowered_graph_preserves_oriented_boundary_and_metric_adjoint() -> None:
    complex_ir = _square_complex()
    b1 = complex_ir.discretization.topology.incidences[0].scipy_boundary().toarray()
    b2 = complex_ir.discretization.topology.incidences[1].scipy_boundary().toarray()
    assert np.array_equal(b1 @ b2, np.zeros((4, 2)))
    values = _degree_values(complex_ir, 0, jnp.asarray([0.0, 1.0, 3.0, 2.0]))
    differentiated = phx.graph.cochain_exterior_derivative(complex_ir.graph, values, 0)
    np.testing.assert_allclose(
        differentiated[_degree_slice(complex_ir, 1)], b1.T @ np.asarray(values[:4])
    )


def test_sparse_dec_operators_satisfy_exactness_adjointness_and_positive_energy() -> None:
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

    lower = phx.graph.cochain_hodge_laplacian(graph, one_form, 1, part="lower")
    upper = phx.graph.cochain_hodge_laplacian(graph, one_form, 1, part="upper")
    complete = phx.graph.cochain_hodge_laplacian(graph, one_form, 1)
    energy = jnp.sum(star * one_form * complete)

    assert jnp.allclose(second_derivative, 0.0, atol=1e-12)
    assert jnp.allclose(left_inner_product, right_inner_product, atol=1e-12)
    assert jnp.allclose(complete, lower + upper, atol=1e-12)
    assert energy >= -1e-12


def test_graphir_dec_wrappers_apply_oriented_incidence_and_weighted_laplacian() -> None:
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

    b1 = jnp.asarray(
        complex_ir.discretization.topology.incidences[0].scipy_boundary().toarray()
    )
    weights = complex_ir.discretization.hodge_diagonal(1)
    expected_d = b1.T @ values[:4]
    expected_delta_d = (
        b1 @ (weights * expected_d)
    ) / complex_ir.discretization.hodge_diagonal(0)
    assert jnp.allclose(differentiated.nodes["gradient"][4:9], expected_d)
    assert jnp.allclose(laplacian.nodes["laplacian"][:4], expected_delta_d)


def test_harmonic_preprocessing_recovers_disconnected_and_annulus_betti_numbers() -> None:
    disconnected = _triangle_lower(
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

    disconnected_harmonics = tuple(
        phx.exterior.validate_harmonic_cohomology(disconnected.discretization, k)[0]
        for k in range(3)
    )
    absolute = tuple(
        phx.exterior.validate_harmonic_cohomology(annulus.discretization, k)[0]
        for k in range(3)
    )
    relative = tuple(
        phx.exterior.validate_harmonic_cohomology(
            annulus.discretization, k, boundary="relative"
        )[0]
        for k in range(3)
    )
    disconnected_graph = phx.graph.CochainComplexIR(
        disconnected.discretization, harmonic=disconnected_harmonics
    ).graph
    absolute_graph = phx.graph.CochainComplexIR(
        annulus.discretization, harmonic=absolute
    ).graph
    relative_graph = phx.graph.CochainComplexIR(
        annulus.discretization, boundary="relative", harmonic=relative
    ).graph
    assert jnp.array_equal(
        disconnected_graph.globals["harmonic_rank"],
        jnp.asarray([[2, 0, 0]], dtype=jnp.int32),
    )
    assert jnp.array_equal(
        absolute_graph.globals["harmonic_rank"], jnp.asarray([[1, 1, 0]], dtype=jnp.int32)
    )
    assert jnp.array_equal(
        relative_graph.globals["harmonic_rank"], jnp.asarray([[0, 1, 1]], dtype=jnp.int32)
    )


def test_harmonic_projection_is_metric_orthogonal_idempotent_and_laplacian_null() -> None:
    base = _annulus_complex()
    harmonics = tuple(
        phx.exterior.validate_harmonic_cohomology(base.discretization, k)[0]
        for k in range(3)
    )
    complex_ir = phx.graph.CochainComplexIR(base.discretization, harmonic=harmonics)
    graph = complex_ir.graph
    one_form = _degree_values(
        complex_ir,
        1,
        jnp.linspace(-1.0, 1.0, complex_ir.cell_counts[1]),
    )

    projected = phx.graph.cochain_harmonic_projection(graph, one_form, 1)
    projected_twice = phx.graph.cochain_harmonic_projection(graph, projected, 1)
    residual = phx.graph.cochain_hodge_laplacian(graph, projected, 1)
    basis = harmonics[1].basis
    metric = complex_ir.discretization.hodge_diagonal(1)

    assert jnp.allclose(projected_twice, projected, atol=1e-10)
    assert jnp.allclose(residual, 0.0, atol=1e-9)
    assert jnp.allclose(
        basis.T @ (metric[:, None] * basis),
        jnp.eye(harmonics[1].dimension),
        atol=1e-9,
    )


def test_orientation_changes_conjugate_exterior_codifferential_and_laplacian() -> None:
    complex_ir = _square_complex()
    signs = (
        np.ones((4,), dtype=np.float64),
        np.asarray([-1.0, 1.0, -1.0, 1.0, -1.0]),
        np.asarray([1.0, -1.0]),
    )
    reoriented = _reorient_lower(complex_ir, signs)
    zero_form = jnp.asarray([0.2, -0.5, 0.7, 1.3])
    one_form = jnp.asarray([0.4, -0.1, 0.8, -0.3, 0.6])
    packed_zero = _degree_values(complex_ir, 0, zero_form)
    packed_one = _degree_values(complex_ir, 1, one_form)
    reoriented_zero = _degree_values(
        reoriented,
        0,
        phx.discretization.reorient_cochain(zero_form, signs[0], cell_axis=0),
    )
    reoriented_one = _degree_values(
        reoriented,
        1,
        phx.discretization.reorient_cochain(one_form, signs[1], cell_axis=0),
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
        phx.discretization.reorient_cochain(
            derivative[_degree_slice(complex_ir, 1)], signs[1], cell_axis=0
        ),
        atol=1e-12,
    )
    assert jnp.allclose(
        transformed_codifferential[_degree_slice(reoriented, 0)],
        phx.discretization.reorient_cochain(
            codifferential[_degree_slice(complex_ir, 0)], signs[0], cell_axis=0
        ),
        atol=1e-12,
    )
    assert jnp.allclose(
        transformed_laplacian[_degree_slice(reoriented, 1)],
        phx.discretization.reorient_cochain(
            laplacian[_degree_slice(complex_ir, 1)], signs[1], cell_axis=0
        ),
        atol=1e-12,
    )


def test_relative_boundary_policy_masks_boundary_cochains() -> None:
    complex_ir = _square_complex()
    zero_form = _degree_values(complex_ir, 0, jnp.ones((4,)))
    one_form = _degree_values(complex_ir, 1, jnp.ones((5,)))

    relative_derivative = phx.graph.cochain_exterior_derivative(
        complex_ir.graph,
        zero_form,
        0,
        boundary="relative",
    )
    relative_codifferential = phx.graph.cochain_codifferential(
        complex_ir.graph,
        one_form,
        1,
        boundary="relative",
    )

    assert jnp.allclose(relative_derivative, 0.0)
    assert jnp.allclose(relative_codifferential, 0.0)


def _centered_square_complex() -> Any:
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
    return _triangle_lower(vertices, faces)


def test_cochain_cells_select_degree_boundary_and_padded_dataset_offsets() -> None:
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
        jnp.asarray(interior_vertices["graph"]["local_index"].data),
        jnp.asarray([4], dtype=jnp.int32),
    )

    base_dataset = phx.domain.GraphDatasetDomain((small.graph, large.graph))
    dataset = base_dataset.with_layout(base_dataset.layout_for_batch_size(2, multiple=4))
    dataset_batch = dataset.points_from_indices(
        # ty: ignore[invalid-argument-type]
        [0, 1],
        component=phx.domain.CochainCells(0, region="interior"),
        structure=structure,
    )

    assert dataset_batch.graph.node_mask is not None
    assert jnp.array_equal(
        jnp.asarray(dataset_batch[phx.domain.graph.GRAPH_DATASET_INDEX_KEY].data),
        jnp.asarray([1], dtype=jnp.int32),
    )
    assert jnp.array_equal(
        jnp.asarray(dataset_batch["graph"]["local_index"].data),
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
        jnp.asarray(trajectory_batch[phx.domain.graph.GRAPH_DATASET_INDEX_KEY].data),
        jnp.asarray([1], dtype=jnp.int32),
    )
    assert jnp.array_equal(
        jnp.asarray(trajectory_batch["graph"]["local_index"].data),
        jnp.asarray([4], dtype=jnp.int32),
    )
    assert jnp.allclose(jnp.asarray(trajectory_batch["t"].data), 0.5)


def test_graph_builder_returns_metric_realization_with_weighted_path_laplacian() -> None:
    graph = phx.graph.GraphIR(
        nodes=jnp.zeros((3, 1)),
        edges={"conductance": jnp.asarray([2.5, 2.5, 3.25, 3.25])},
        senders=jnp.asarray([0, 1, 1, 2], dtype=jnp.int32),
        receivers=jnp.asarray([1, 0, 2, 1], dtype=jnp.int32),
        n_node=jnp.asarray([3], dtype=jnp.int32),
        n_edge=jnp.asarray([4], dtype=jnp.int32),
    )
    realization = phx.graph.graph_to_cochain_complex(graph, edge_weight_key="conductance")
    values = jnp.asarray([0, 1, 4], dtype=jnp.int32)
    expected = jnp.asarray([-2.5, -7.25, 9.75])
    assert jnp.allclose(realization.hodge_laplacian(0, values), expected)
    lower = phx.graph.CochainComplexIR(realization)
    packed = _degree_values(lower, 0, values)
    assert jnp.allclose(
        phx.graph.cochain_hodge_laplacian(lower.graph, packed, 0)[:3], expected
    )


def test_full_gram_graph_action_survives_batch_padding_and_unbatch() -> None:
    topology = phx.discretization.polygonal_cell_complex(
        jnp.asarray([[0, 1, 2]], dtype=jnp.int32), None, 3
    )
    gram = np.asarray([[2.0, 0.3, 0.0], [0.3, 3.0, 0.4], [0.0, 0.4, 4.0]])
    rows, columns = np.triu_indices(3)
    realization = phx.discretization.CochainDiscretization(
        topology,
        (
            phx.discretization.SparseHodge(rows, columns, gram[rows, columns], 3),
            phx.discretization.DiagonalHodge(jnp.asarray([1.5, 2.0, 2.5])),
            phx.discretization.DiagonalHodge(jnp.ones((1,))),
        ),
    )
    graph = phx.graph.CochainComplexIR(realization).graph
    derivative = np.asarray(topology.incidences[0].scipy_boundary().toarray()).T
    expected = np.linalg.solve(gram, derivative.T @ np.diag([1.5, 2.0, 2.5]) @ derivative)
    first = np.asarray([1.0, -2.0, 0.75])
    second = np.asarray([-0.5, 0.3, 1.25])
    values = jnp.concatenate(
        (
            jnp.asarray(first),
            jnp.zeros(4),
            jnp.asarray(second),
            jnp.zeros(4),
            jnp.zeros(2),
        )
    )
    batched = phx.graph.batch_graphs((graph, graph))
    padded = phx.graph.pad_with_graphs(
        batched, n_node=16, n_edge=batched.num_edges, n_graph=3
    )
    actual = eqx.filter_jit(phx.graph.cochain_hodge_laplacian)(padded, values, 0)
    np.testing.assert_allclose(actual[:3], expected @ first, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(actual[7:10], expected @ second, rtol=1e-11, atol=1e-12)
    np.testing.assert_array_equal(actual[14:], np.zeros(2))
    restored = phx.graph.unbatch_graph(phx.graph.unpad_with_graphs(padded))[1]
    restored_action = phx.graph.cochain_hodge_laplacian(restored, values[7:14], 0)
    np.testing.assert_allclose(
        restored_action[:3], expected @ second, rtol=1e-11, atol=1e-12
    )
    assert (
        restored.cochain_bindings[0].discretization.prepared_id == realization.prepared_id
    )


def test_native_graph_codifferential_preserves_nonconvergence_evidence() -> None:
    topology = phx.discretization.polygonal_cell_complex(
        jnp.asarray([[0, 1, 2]], dtype=jnp.int32), None, 3
    )
    gram = np.asarray([[2.0, 0.3, 0.0], [0.3, 3.0, 0.4], [0.0, 0.4, 4.0]])
    rows, columns = np.triu_indices(3)
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.PCG(),
        tolerance=phx.linalg.TolerancePolicy(relative=1e-12, absolute=0.0, max_steps=1),
        failure=phx.linalg.FailurePolicy("status"),
    )
    owner = phx.discretization.CochainDiscretization(
        topology,
        (
            phx.discretization.SparseHodge(
                rows, columns, gram[rows, columns], 3, policy=policy
            ),
            phx.discretization.DiagonalHodge(jnp.ones(3)),
            phx.discretization.DiagonalHodge(jnp.ones(1)),
        ),
    )
    graph = phx.graph.CochainComplexIR(owner).graph
    edges = jnp.asarray([1.0, -2.0, 0.75])
    rhs = jnp.asarray(topology.incidences[0].scipy_boundary().toarray()) @ edges
    space = owner.hilbert_complex().space(0)
    if not isinstance(space, phx.linalg.ArraySpace):
        raise TypeError("The cell realization must prepare a native array space.")
    pairing = space.pairing
    if not isinstance(pairing, phx.linalg.OperatorPairing):
        raise TypeError("A coupled sparse metric must prepare an operator pairing.")
    if pairing.prepared_inverse is None:
        raise ValueError("A coupled sparse metric inverse must be prepared.")
    result = phx.linalg.solve(pairing.prepared_inverse, rhs)
    assert not bool(result.successful)
    assert int(result.status) == int(phx.linalg.LinearSolveStatus.MAXIMUM_STEPS_REACHED)
    values = jnp.concatenate((jnp.zeros(3), edges, jnp.zeros(1)))
    with pytest.raises(eqx.EquinoxRuntimeError):
        eqx.filter_jit(phx.graph.cochain_codifferential)(
            graph, values, 1
        ).block_until_ready()


def test_graph_lowering_embedding_identity_refuses_foreign_harmonic_frame() -> None:
    base = _square_complex().discretization
    shifted = phx.discretization.CochainDiscretization(
        base.topology,
        base.hodges,
        boundary_masks=base.boundary_masks,
        coordinates=tuple(
            None if coordinates is None else coordinates + 1.0
            for coordinates in base.coordinates
        ),
    )
    assert (
        phx.graph.CochainComplexIR(base).fingerprint
        != phx.graph.CochainComplexIR(shifted).fingerprint
    )
    harmonic = phx.exterior.validate_harmonic_cohomology(base, 0)[0]
    with pytest.raises(ValueError):
        phx.graph.CochainComplexIR(shifted, harmonic=(harmonic, None, None))


def test_lowering_preserves_multiaxis_routes_and_inactive_cells() -> None:
    vertices = phx.discretization.EntitySet(
        "vertices",
        0,
        jnp.arange(3, dtype=jnp.int32),
        active_mask=jnp.asarray([True, False, True]),
    )
    edges = phx.discretization.EntitySet("edges", 1, jnp.arange(1, dtype=jnp.int32))
    incidence = phx.discretization.OrientedIncidence(
        1,
        vertices,
        edges,
        phx.sparse.EdgeRelation(
            jnp.asarray([0, 2], dtype=jnp.int32),
            jnp.asarray([0, 0], dtype=jnp.int32),
            source_size=3,
            target_size=1,
        ),
        jnp.asarray([-1.0, 1.0]),
    )
    realization = phx.discretization.CochainDiscretization(
        phx.discretization.CellComplexTopology((vertices, edges), (incidence,)),
        (
            phx.discretization.DiagonalHodge(jnp.ones((3,))),
            phx.discretization.DiagonalHodge(jnp.ones((1,))),
        ),
    )
    lower = phx.graph.CochainComplexIR(realization)
    values = jnp.asarray([[1.0, 2.0], [100.0, 200.0], [4.0, 8.0], [0.0, 0.0]])
    assert jnp.allclose(
        phx.graph.cochain_exterior_derivative(lower.graph, values, 0),
        jnp.asarray([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [3.0, 6.0]]),
    )
    assert jnp.allclose(
        phx.graph.cochain_hodge_laplacian(lower.graph, values, 0),
        jnp.asarray([[-3.0, -6.0], [0.0, 0.0], [3.0, 6.0], [0.0, 0.0]]),
    )
    block = phx.nn.operator.architectures.TopologicalCochainBlock(
        2,
        (0, 1),
        dimension=1,
        residual_scale=0.0,
        routes=phx.nn.operator.architectures.TopologicalRouteConfig(
            exterior_derivative=False,
            codifferential=False,
            lower_laplacian=False,
            upper_laplacian=False,
        ),
    )
    padded = phx.graph.pad_with_graphs(
        lower.graph, n_node=6, n_edge=lower.graph.num_edges, n_graph=2
    )
    padded_values = jnp.concatenate((values, jnp.zeros((2, 2))))
    expected = np.asarray([[1.0, 2.0], [0.0, 0.0], [4.0, 8.0], [0.0, 0.0]])
    np.testing.assert_array_equal(block(padded, padded_values)[:4], expected)
    restored = phx.graph.unpad_with_graphs(padded)
    np.testing.assert_array_equal(block(restored, values), expected)


def test_graph_harmonic_projection_preserves_complex_field_coefficients() -> None:
    realization = _annulus_complex().discretization
    harmonic = phx.exterior.validate_harmonic_cohomology(realization, 1)[0]
    lower = phx.graph.CochainComplexIR(realization, harmonic=(None, harmonic, None))
    coefficients = jnp.linspace(
        -1.0, 1.0, realization.cell_counts[1]
    ) + 1j * jnp.linspace(0.0, 2.0, realization.cell_counts[1])
    field = _degree_values(lower, 1, coefficients)
    projected = phx.graph.cochain_harmonic_projection(lower.graph, field, 1)
    basis = harmonic.basis
    expected = basis @ (basis.conj().T @ (realization.hodge_diagonal(1) * coefficients))
    assert jnp.allclose(projected[_degree_slice(lower, 1)], expected, atol=1e-10)
    assert jnp.allclose(
        phx.graph.cochain_hodge_laplacian(lower.graph, projected, 1), 0.0, atol=1e-9
    )
