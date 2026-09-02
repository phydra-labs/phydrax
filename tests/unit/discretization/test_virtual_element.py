#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
import pytest

import phydrax as phx


def _single_polygon(points, degree=1):
    coordinates = jnp.asarray(points, dtype=float)
    mesh = phx.discretization.CellMesh.from_polygons(
        coordinates, (tuple(range(len(points))),)
    )
    field = phx.discretization.VirtualElementFieldSpec(
        "u", phx.discretization.conforming_h1_virtual_element(degree)
    )
    return phx.discretization.VirtualElementPlan(mesh, field).prepare()


def _two_cell_space(factory, degree=1):
    coordinates = jnp.asarray(
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (2.0, 1.0),
        ),
        dtype=float,
    )
    mesh = phx.discretization.CellMesh.from_polygons(
        coordinates,
        ((0, 1, 4, 3), (1, 2, 5, 4)),
    )
    field = phx.discretization.VirtualElementFieldSpec("v", factory(degree))
    return phx.discretization.VirtualElementPlan(mesh, field).prepare()


def _affine_vector_coefficients(space, components):
    projection = space.default_runtime.projections[0]
    geometry = space.default_runtime.geometries[0]
    exponents = [tuple(int(value) for value in row) for row in projection.basis.exponents]
    constant = exponents.index((0, 0))
    x_term = exponents.index((1, 0))
    y_term = exponents.index((0, 1))
    polynomial_count = projection.basis.feature_count
    coefficients = jnp.zeros(
        (geometry.centroids.shape[0], 2 * polynomial_count),
        dtype=geometry.centroids.dtype,
    )
    for component, (offset, x_slope, y_slope) in enumerate(components):
        start = component * polynomial_count
        coefficients = coefficients.at[:, start + constant].set(
            offset
            + x_slope * geometry.centroids[:, 0]
            + y_slope * geometry.centroids[:, 1]
        )
        coefficients = coefficients.at[:, start + x_term].set(
            x_slope * geometry.characteristic_lengths
        )
        coefficients = coefficients.at[:, start + y_term].set(
            y_slope * geometry.characteristic_lengths
        )
    return coefficients


def _global_state_from_local(space, local):
    routes = np.asarray(space.dof_map.cell_dofs[0])
    orientations = np.asarray(space.dof_map.orientations[0])
    oriented = np.asarray(local) * orientations
    state = np.full((space.dof_map.global_dof_count,), np.nan)
    for cell in range(routes.shape[0]):
        for local_dof, global_dof in enumerate(routes[cell]):
            candidate = oriented[cell, local_dof]
            if np.isnan(state[global_dof]):
                state[global_dof] = candidate
            else:
                np.testing.assert_allclose(state[global_dof], candidate, atol=2.0e-10)
    assert not np.any(np.isnan(state))
    return jnp.asarray(state)


def test_polygon_mesh_canonicalizes_orientation_and_arbitrary_arity():
    coordinates = jnp.asarray(
        ((0.0, 0.0), (0.0, 1.0), (0.5, 1.4), (1.0, 1.0), (1.0, 0.0))
    )
    mesh = phx.discretization.CellMesh.from_polygons(
        coordinates, ((0, 1, 2, 3, 4),), cell_global_ids=jnp.asarray((17,))
    )
    connectivity = mesh.connectivity

    assert mesh.blocks[0].cell_kind == "polygon"
    assert mesh.blocks[0].arity == 5
    assert connectivity.cell_vertices.shape == (1, 5)
    assert connectivity.cell_count == 1
    assert connectivity.polygon_count == 1
    assert int(mesh.topology.entity_sets[2].entity_ids[0]) == 17
    assert jnp.all(connectivity.boundary_edges)


def test_polygon_mesh_rejects_self_intersection_during_vem_preparation():
    coordinates = jnp.asarray(((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)))
    mesh = phx.discretization.CellMesh(
        coordinates,
        (
            phx.discretization.CellBlock(
                "bow", "quadrilateral", jnp.asarray(((0, 1, 2, 3),))
            ),
        ),
    )
    field = phx.discretization.VirtualElementFieldSpec(
        "u", phx.discretization.conforming_h1_virtual_element(1)
    )
    with np.testing.assert_raises(ValueError):
        phx.discretization.VirtualElementPlan(mesh, field).prepare()


def test_virtual_element_projector_budget_is_checked_during_planning():
    coordinates = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    mesh = phx.discretization.CellMesh.from_polygons(coordinates, ((0, 1, 2, 3),))
    field = phx.discretization.VirtualElementFieldSpec(
        "u", phx.discretization.conforming_hdiv_virtual_element(1)
    )
    budget = phx.discretization.VirtualElementResourceBudget(maximum_projector_bytes=1)
    with pytest.raises(ValueError, match="projector storage budget"):
        phx.discretization.VirtualElementPlan(mesh, field, resource_budget=budget)


def test_virtual_element_dof_layout_and_edge_orientation():
    coordinates = jnp.asarray(
        ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0))
    )
    mesh = phx.discretization.CellMesh.from_polygons(
        coordinates,
        ((0, 1, 4, 3), (1, 2, 5, 4)),
    )
    field = phx.discretization.VirtualElementFieldSpec(
        "u", phx.discretization.conforming_h1_virtual_element(3)
    )
    space = phx.discretization.VirtualElementPlan(mesh, field).prepare()
    dof_map = space.dof_map
    expected = coordinates.shape[0] + 2 * mesh.connectivity.edges.shape[0] + 3 * 2

    assert dof_map.global_dof_count == expected
    assert dof_map.cell_dofs[0].shape == (2, 15)
    assert dof_map.cell_dof_count == 6
    assert not jnp.any(dof_map.point_dof_valid[-6:])
    assert space.field_space.representation == "functional"
    assert space.field_space.conformity == "H1"


def test_h1_and_enhanced_l2_projectors_reproduce_polynomials():
    points = ((0.0, 0.0), (1.0, 0.0), (1.2, 0.8), (0.5, 1.3), (-0.2, 0.8))
    for degree in (1, 2, 3):
        space = _single_polygon(points, degree)
        projection = space.default_runtime.projections[0]
        evidence = projection.evidence

        assert jnp.max(evidence.h1_reproduction_error) < 5.0e-10
        assert jnp.max(evidence.l2_reproduction_error) < 5.0e-10
        assert jnp.max(evidence.h1_idempotence_error) < 5.0e-10
        assert jnp.max(evidence.l2_idempotence_error) < 5.0e-10
        assert jnp.all(evidence.factorization_valid)


def test_stabilization_annihilates_polynomial_image():
    space = _single_polygon(
        ((0.0, 0.0), (1.0, 0.0), (1.1, 0.8), (0.4, 1.2), (-0.1, 0.7)),
        2,
    )
    projection = space.default_runtime.projections[0]
    coefficients = projection.h1_coefficients
    consistent = oe.contract(
        "cai,cab,cbj->cij", coefficients, projection.gradient_gram, coefficients
    )
    stabilized = phx.discretization.stabilize_virtual_element_tensor(
        projection,
        consistent,
        phx.discretization.VirtualElementStabilizationPolicy("dofi_dofi"),
        projector="h1",
    )

    assert jnp.max(stabilized.evidence.polynomial_leakage) < 1.0e-9
    assert jnp.max(stabilized.evidence.symmetry_error) < 1.0e-12
    assert jnp.min(stabilized.evidence.minimum_kernel_eigenvalue) > -1.0e-10


def test_polygon_geometry_refresh_has_finite_coordinate_gradient():
    space = _single_polygon(
        ((0.0, 0.0), (1.0, 0.0), (1.2, 0.7), (0.5, 1.2), (-0.1, 0.7)),
        1,
    )

    def total_area(coordinates):
        runtime = space.prepare_runtime(coordinates, numeric_version="gradient")
        return jnp.sum(runtime.geometries[0].areas)

    gradient = jax.grad(total_area)(space.mesh.coordinates)
    assert gradient.shape == space.mesh.coordinates.shape
    assert jnp.all(jnp.isfinite(gradient))


def test_virtual_element_families_have_distinct_entity_topologies():
    coordinates = jnp.asarray(
        ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0))
    )
    mesh = phx.discretization.CellMesh.from_polygons(
        coordinates,
        ((0, 1, 4, 3), (1, 2, 5, 4)),
    )
    factories = (
        phx.discretization.conforming_h1_virtual_element,
        phx.discretization.conforming_hdiv_virtual_element,
        phx.discretization.conforming_hcurl_virtual_element,
        phx.discretization.discontinuous_l2_virtual_element,
    )
    spaces = []
    for factory in factories:
        field = phx.discretization.VirtualElementFieldSpec("u", factory(1))
        spaces.append(phx.discretization.VirtualElementPlan(mesh, field).prepare())

    h1, hdiv, hcurl, l2 = spaces
    assert h1.field_space.conformity == "H1"
    assert hdiv.field_space.conformity == "Hdiv"
    assert hcurl.field_space.conformity == "Hcurl"
    assert l2.field_space.conformity == "L2"
    assert h1.field_space.layout.names == ("vertices",)
    assert hdiv.field_space.layout.names == ("edges", "cells")
    assert hcurl.field_space.layout.names == ("edges", "cells")
    assert l2.field_space.layout.names == ("cells",)
    assert h1.field_space.representation == "functional"
    assert hdiv.field_space.representation == "flux_moment"
    assert hcurl.field_space.representation == "circulation_moment"
    assert l2.field_space.representation == "polynomial_moment"
    assert h1.dof_map.global_dof_count == 6
    assert hdiv.dof_map.global_dof_count == 18
    assert hcurl.dof_map.global_dof_count == 18
    assert l2.dof_map.global_dof_count == 6
    assert not jnp.any(hdiv.dof_map.point_dof_valid)
    assert not jnp.any(hcurl.dof_map.point_dof_valid)
    assert not jnp.any(l2.dof_map.point_dof_valid)
    for traced in (h1, hdiv, hcurl):
        assert phx.discretization.DiscretizationCapability.TRACE in traced.capabilities
        assert traced.field_space.trace_space_id is not None

    for space in (hdiv, hcurl):
        orientation = space.dof_map.orientations[0]
        np.testing.assert_allclose(
            np.asarray(orientation[0, 2:4]),
            np.asarray((1.0, 1.0)),
        )
        np.testing.assert_allclose(
            np.asarray(orientation[1, 6:8]),
            np.asarray((-1.0, 1.0)),
        )


def test_moment_virtual_element_projectors_reproduce_exact_sequence_polynomials():
    points = ((0.0, 0.0), (1.0, 0.0), (1.2, 0.8), (0.5, 1.3), (-0.2, 0.8))
    coordinates = jnp.asarray(points, dtype=float)
    mesh = phx.discretization.CellMesh.from_polygons(
        coordinates, (tuple(range(len(points))),)
    )
    for factory, differential_kind in (
        (phx.discretization.conforming_hdiv_virtual_element, "divergence"),
        (phx.discretization.conforming_hcurl_virtual_element, "curl"),
    ):
        element = factory(2)
        field = phx.discretization.VirtualElementFieldSpec("v", element)
        space = phx.discretization.VirtualElementPlan(mesh, field).prepare()
        projection = space.default_runtime.projections[0]
        exponents = [
            tuple(int(value) for value in row) for row in projection.basis.exponents
        ]
        x_index = exponents.index((1, 0))
        y_index = exponents.index((0, 1))
        polynomial_count = projection.basis.feature_count
        coefficients = jnp.zeros((2 * polynomial_count,))
        if differential_kind == "divergence":
            coefficients = coefficients.at[x_index].set(1.0)
            coefficients = coefficients.at[polynomial_count + y_index].set(1.0)
        else:
            coefficients = coefficients.at[y_index].set(-1.0)
            coefficients = coefficients.at[polynomial_count + x_index].set(1.0)
        dofs = projection.dof_matrix[0] @ coefficients
        differential = projection.differential_coefficients[0] @ dofs
        scale = space.default_runtime.geometries[0].characteristic_lengths[0]

        assert projection.projection_kind == "vector_L2"
        assert projection.polynomial_value_shape == (2,)
        assert projection.differential_kind == differential_kind
        assert jnp.abs(differential[0] - 2.0 / scale) < 5.0e-10
        assert jnp.max(jnp.abs(differential[1:])) < 5.0e-10
        assert jnp.max(projection.evidence.l2_reproduction_error) < 5.0e-10
        assert jnp.max(projection.evidence.differential_reproduction_error) < 5.0e-10

    element = phx.discretization.discontinuous_l2_virtual_element(2)
    field = phx.discretization.VirtualElementFieldSpec("q", element)
    space = phx.discretization.VirtualElementPlan(mesh, field).prepare()
    projection = space.default_runtime.projections[0]
    polynomial = jnp.arange(projection.basis.feature_count, dtype=float)
    dofs = projection.dof_matrix[0] @ polynomial
    recovered = projection.l2_coefficients[0] @ dofs

    assert projection.projection_kind == "L2"
    assert projection.differential_kind == "none"
    assert projection.differential_coefficients.shape[-2] == 0
    np.testing.assert_allclose(
        np.asarray(recovered), np.asarray(polynomial), atol=5.0e-10
    )


@pytest.mark.parametrize(
    ("factory", "differential_kind", "expected_differential"),
    (
        (phx.discretization.conforming_hdiv_virtual_element, "divergence", 5.0),
        (phx.discretization.conforming_hcurl_virtual_element, "curl", 2.0),
    ),
)
def test_vector_reconstruction_orients_shared_edges_and_exposes_exact_trace(
    factory, differential_kind, expected_differential
):
    space = _two_cell_space(factory)
    projection = space.default_runtime.projections[0]
    coefficients = _affine_vector_coefficients(
        space,
        ((1.0, 2.0, -1.0), (-0.5, 1.0, 3.0)),
    )
    local = oe.contract("cia,ca->ci", projection.dof_matrix, coefficients)
    state = _global_state_from_local(space, local)
    reconstruction = phx.equations.project_virtual_element_field(space, state)

    shared_points = jnp.asarray(
        (
            ((1.0, 0.1), (1.0, 0.5), (1.0, 0.9)),
            ((1.0, 0.1), (1.0, 0.5), (1.0, 0.9)),
        )
    )
    value, differential = phx.equations.evaluate_virtual_element_reconstruction(
        reconstruction, space, 0, shared_points
    )
    expected_value = jnp.stack(
        (
            1.0 + 2.0 * shared_points[..., 0] - shared_points[..., 1],
            -0.5 + shared_points[..., 0] + 3.0 * shared_points[..., 1],
        ),
        axis=-1,
    )
    np.testing.assert_allclose(value, expected_value, atol=2.0e-9)
    np.testing.assert_allclose(differential, expected_differential, atol=2.0e-9)
    assert projection.differential_kind == differential_kind

    topology_edges = np.asarray(space.mesh.connectivity.edges)
    shared_edge = next(
        index
        for index, edge in enumerate(topology_edges)
        if set(int(value) for value in edge) == {1, 4}
    )
    parameters = jnp.asarray((-0.75, 0.0, 0.75))
    endpoints = topology_edges[shared_edge]
    start = space.mesh.coordinates[endpoints[0]]
    stop = space.mesh.coordinates[endpoints[1]]
    edge_points = (
        0.5 * (1.0 - parameters[:, None]) * start
        + 0.5 * (1.0 + parameters[:, None]) * stop
    )
    vector = jnp.stack(
        (
            1.0 + 2.0 * edge_points[:, 0] - edge_points[:, 1],
            -0.5 + edge_points[:, 0] + 3.0 * edge_points[:, 1],
        ),
        axis=-1,
    )
    tangent = (stop - start) / jnp.linalg.norm(stop - start)
    direction = (
        jnp.asarray((tangent[1], -tangent[0]))
        if differential_kind == "divergence"
        else tangent
    )
    expected_trace = vector @ direction
    trace = phx.equations.evaluate_virtual_element_trace(
        reconstruction,
        space,
        jnp.asarray((shared_edge,)),
        parameters,
    )
    np.testing.assert_allclose(trace[0], expected_trace, atol=2.0e-9)

    constraint = phx.discretization.virtual_element_dirichlet_constraint(space, "v")
    prescribed = jnp.linspace(0.1, 0.9, constraint.constrained_dofs.size)
    lift = constraint.lift(prescribed)
    np.testing.assert_allclose(lift[constraint.constrained_dofs], prescribed)
    constant_lift = constraint.lift(2.0)
    np.testing.assert_allclose(
        constant_lift[constraint.constrained_dofs],
        jnp.where(constraint.trace_modes == 0, 2.0, 0.0),
    )
    with pytest.raises(ValueError, match="prescribed DOF moments"):
        constraint.lift(lambda points: points[:, 0])


def test_discontinuous_l2_reconstructs_cell_polynomials_without_a_trace():
    space = _two_cell_space(phx.discretization.discontinuous_l2_virtual_element)
    projection = space.default_runtime.projections[0]
    geometry = space.default_runtime.geometries[0]
    exponents = [tuple(int(value) for value in row) for row in projection.basis.exponents]
    coefficients = jnp.zeros(
        (geometry.centroids.shape[0], projection.basis.feature_count)
    )
    coefficients = coefficients.at[:, exponents.index((0, 0))].set(
        1.0 + 2.0 * geometry.centroids[:, 0] - geometry.centroids[:, 1]
    )
    coefficients = coefficients.at[:, exponents.index((1, 0))].set(
        2.0 * geometry.characteristic_lengths
    )
    coefficients = coefficients.at[:, exponents.index((0, 1))].set(
        -geometry.characteristic_lengths
    )
    local = oe.contract("cia,ca->ci", projection.dof_matrix, coefficients)
    state = _global_state_from_local(space, local)
    reconstruction = phx.equations.project_virtual_element_field(space, state)
    points = geometry.centroids[:, None, :]
    value, differential = phx.equations.evaluate_virtual_element_reconstruction(
        reconstruction, space, 0, points
    )

    expected = 1.0 + 2.0 * points[..., 0] - points[..., 1]
    np.testing.assert_allclose(value, expected, atol=2.0e-9)
    assert differential is None
    assert phx.discretization.DiscretizationCapability.TRACE not in space.capabilities
    assert (
        phx.discretization.DiscretizationCapability.BOUNDARY_INTEGRAL
        not in space.capabilities
    )
    with pytest.raises(ValueError, match="no boundary trace"):
        phx.equations.evaluate_virtual_element_trace(
            reconstruction,
            space,
            jnp.asarray(((0,),)),
            jnp.asarray(((0.0,),)),
        )
    with pytest.raises(ValueError, match="no boundary trace"):
        phx.discretization.virtual_element_dirichlet_constraint(space, "v")
    with pytest.raises(ValueError, match="H1 projector is undefined"):
        phx.discretization.stabilize_virtual_element_tensor(
            projection,
            jnp.asarray(0.0),
            phx.discretization.VirtualElementStabilizationPolicy(),
            projector="h1",
        )
