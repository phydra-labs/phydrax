#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import SpatialCoordinateContract
from phydrax.discretization import CellMesh
from phydrax.geometry.surface import SurfaceMetadata, SurfaceModel
from phydrax.operators.integral.layer_potential._scalar_screens3d import (
    prepare_scalar_screen_calderon_dp0_3d,
    prepare_scalar_screen_hypersingular_dp0_3d,
    prepare_scalar_screen_junction_solve_3d,
    prepare_scalar_screen_single_layer_dp0_3d,
    scalar_screen_junction_evidence_3d,
    ScalarCrackSideMetadata3D,
    UnsupportedScalarScreenJunctionError,
)
from phydrax.operators.integral.layer_potential._scalar_trace import (
    UnsupportedScalarBoundarySpaceError,
)


def _metadata(source_id="unit-screen"):
    return SurfaceMetadata(
        source_id=source_id,
        source_revision="r1",
        coordinate_contract=SpatialCoordinateContract.si(),
        provenance=("unit-test",),
    )


def _square_screen():
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        )
    )
    faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    return SurfaceModel.from_triangles(vertices, faces, _metadata())


def _closed_tetrahedron():
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    faces = np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int32)
    return SurfaceModel.from_triangles(vertices, faces, _metadata("unit-closed"))


def _junction_surface():
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    faces = np.asarray(((0, 1, 2), (1, 0, 3), (0, 1, 4)), dtype=np.int32)
    mesh = CellMesh.from_triangles(vertices, faces)
    return SurfaceModel(mesh, _metadata("unit-junction"))


def test_finite_open_screen_solve_and_edge_evidence():
    prepared = prepare_scalar_screen_single_layer_dp0_3d(_square_screen())
    expected_density = jnp.asarray((0.75, -0.25))
    boundary_values = prepared.forward(expected_density)
    result = prepared.solve_dirichlet(boundary_values)

    assert prepared.topology.boundary_edge_count == 4
    assert prepared.topology.interior_edge_count == 1
    assert prepared.topology.junctions.junction_edge_count == 0
    assert prepared.assembly_report.exact_dense_actions
    assert bool(prepared.assembly_report.finite)
    assert bool(result.valid)
    np.testing.assert_allclose(result.density, expected_density, rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(
        result.predicted_boundary_values, boundary_values, rtol=2.0e-5, atol=2.0e-6
    )


def test_dense_forward_transpose_and_adjoint_are_exact_algebraic_actions():
    prepared = prepare_scalar_screen_single_layer_dp0_3d(_square_screen())
    left = jnp.asarray((0.3, -0.7))
    right = jnp.asarray((1.2, 0.4))

    np.testing.assert_allclose(
        jnp.vdot(left, prepared.forward(right)),
        jnp.vdot(prepared.transpose(left), right),
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        prepared.adjoint(left),
        jnp.conj(prepared.strong_operator.matrix.T) @ left,
        rtol=0.0,
        atol=0.0,
    )


def test_crack_sides_remain_distinct_and_jump_density_route_is_explicit():
    sides = ScalarCrackSideMetadata3D("rock-minus", "rock-plus")
    prepared = prepare_scalar_screen_single_layer_dp0_3d(
        _square_screen(), crack_sides=sides
    )
    density = jnp.asarray((2.0, -3.0))

    assert prepared.crack_sides.side_names == ("rock-minus", "rock-plus")
    assert prepared.crack_sides.name("minus") != prepared.crack_sides.name("plus")
    assert prepared.crack_sides.derivative_jump == "q_plus-q_minus=-density"
    np.testing.assert_array_equal(prepared.crack_jump_density(density), -density)


def test_closed_surface_w_calderon_and_junction_misuse_fail_closed():
    closed = _closed_tetrahedron()
    with pytest.raises(ValueError, match="open support"):
        prepare_scalar_screen_single_layer_dp0_3d(closed)
    with pytest.raises(UnsupportedScalarBoundarySpaceError, match="hypersingular W"):
        prepare_scalar_screen_hypersingular_dp0_3d(closed)
    with pytest.raises(
        UnsupportedScalarBoundarySpaceError, match="Closed-surface Calderón"
    ):
        prepare_scalar_screen_calderon_dp0_3d(closed)

    with pytest.raises(ValueError, match="edge-manifold"):
        _junction_surface()

    screen = _square_screen()
    evidence = scalar_screen_junction_evidence_3d(screen)
    assert evidence.junction_edges == ()
    assert evidence.incident_faces == ()
    assert evidence.maximum_incidence == 0
    with pytest.raises(UnsupportedScalarScreenJunctionError) as solve_error:
        prepare_scalar_screen_junction_solve_3d(screen)
    assert solve_error.value.evidence.evidence_id == evidence.evidence_id
