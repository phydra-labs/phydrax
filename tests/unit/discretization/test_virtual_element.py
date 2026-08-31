#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe

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
