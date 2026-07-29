#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _square_complex():
    vertices = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return phx.graph.triangle_mesh_to_cochain_complex(vertices, faces)


def _annulus_complex():
    outer = np.asarray(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    )
    vertices = np.concatenate((outer, 0.4 * outer), axis=0)
    faces = np.asarray(
        [
            (index, (index + 1) % 4, 4 + (index + 1) % 4)
            for index in range(4)
        ]
        + [
            (index, 4 + (index + 1) % 4, 4 + index)
            for index in range(4)
        ],
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

    lower = phx.graph.cochain_hodge_laplacian(
        graph, one_form, 1, component="lower"
    )
    upper = phx.graph.cochain_hodge_laplacian(
        graph, one_form, 1, component="upper"
    )
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

    derivative = phx.graph.cochain_exterior_derivative(
        complex_ir.graph, packed_zero, 0
    )
    transformed_derivative = phx.graph.cochain_exterior_derivative(
        reoriented.graph, reoriented_zero, 0
    )
    codifferential = phx.graph.cochain_codifferential(
        complex_ir.graph, packed_one, 1
    )
    transformed_codifferential = phx.graph.cochain_codifferential(
        reoriented.graph, reoriented_one, 1
    )
    laplacian = phx.graph.cochain_hodge_laplacian(
        complex_ir.graph, packed_one, 1
    )
    transformed_laplacian = phx.graph.cochain_hodge_laplacian(
        reoriented.graph, reoriented_one, 1
    )

    assert jnp.allclose(
        transformed_derivative[_degree_slice(reoriented, 1)],
        phx.graph.reorient_cochain(
            derivative[_degree_slice(complex_ir, 1)], signs[1]
        ),
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
        phx.graph.reorient_cochain(
            laplacian[_degree_slice(complex_ir, 1)], signs[1]
        ),
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
