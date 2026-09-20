#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.geometry.surface._g1_multipatch import (
    BiquinticGluingBasis,
    BiquinticGluingConstraints,
    G1PatchStatus,
    GluingContinuity,
    PatchEdge,
    PatchTopologyError,
    PatchVertexKind,
    PreparedHalfEdgePatchTopology,
    UnsupportedG1TopologyError,
)


def _two_patch_topology():
    return PreparedHalfEdgePatchTopology(
        ("left", "right"),
        (
            ("a", "b", "e", "d"),
            ("b", "c", "f", "e"),
        ),
    )


def _planar_two_patch_coefficients():
    parameter = jnp.linspace(0.0, 1.0, 6)
    u, v = jnp.meshgrid(parameter, parameter, indexing="ij")
    zero = jnp.zeros_like(u)
    left = jnp.stack((u, v, zero), axis=-1)
    right = jnp.stack((1.0 + u, v, zero), axis=-1)
    return jnp.stack((left, right))


def test_two_patch_shared_edge_has_canonical_reversed_halfedge_orientation():
    topology = _two_patch_topology()
    shared = np.asarray(topology.shared_halfedges)[0]

    assert topology.patch_labels == ("left", "right")
    assert tuple(np.asarray(topology.halfedge_orientation)[shared]) == (1, -1)
    assert tuple(np.asarray(topology.halfedge_local_edges)[shared]) == (
        int(PatchEdge.U_MAX),
        int(PatchEdge.U_MIN),
    )
    assert np.asarray(topology.halfedge_twins)[shared[0]] == shared[1]
    assert np.asarray(topology.halfedge_twins)[shared[1]] == shared[0]

    constraints = BiquinticGluingConstraints(topology, GluingContinuity.G0)
    residual = constraints.residual(_planar_two_patch_coefficients())
    assert constraints.constraint_count == 6
    assert jnp.allclose(residual, 0.0)


def test_g0_and_g1_constraints_and_sampled_seam_jets_vanish_for_planar_join():
    topology = _two_patch_topology()
    coefficients = _planar_two_patch_coefficients()
    g0 = BiquinticGluingConstraints(topology, GluingContinuity.G0)
    g1 = BiquinticGluingConstraints(topology, GluingContinuity.G1)

    assert g0.status is G1PatchStatus.SUPPORTED
    assert g1.status is G1PatchStatus.SUPPORTED
    assert jnp.allclose(g0.residual(coefficients), 0.0)
    assert jnp.allclose(g1.residual(coefficients), 0.0)

    evidence = g1.seam_evidence(
        coefficients,
        jnp.asarray([0.0, 0.13, 0.5, 0.81, 1.0]),
        tolerance=2.0e-6,
    )
    assert evidence.path_supported
    assert evidence.finite
    assert evidence.g0_geometry_accepted
    assert evidence.g1_geometry_accepted
    assert evidence.value_residual.shape == (1, 5, 3)
    assert evidence.tangent_residual.shape == (1, 5, 3)
    assert evidence.transverse_residual.shape == (1, 5, 3)
    assert evidence.maximum_value_residual == pytest.approx(0.0, abs=2.0e-6)
    assert evidence.maximum_first_derivative_residual == pytest.approx(0.0, abs=2.0e-6)


def test_regular_four_patch_vertex_is_supported():
    topology = PreparedHalfEdgePatchTopology(
        ("lower_left", "lower_right", "upper_left", "upper_right"),
        (
            ("a", "b", "e", "d"),
            ("b", "c", "f", "e"),
            ("d", "e", "h", "g"),
            ("e", "f", "i", "h"),
        ),
    )

    center = topology.vertex_labels.index("e")
    assert topology.vertex_kind("e") is PatchVertexKind.REGULAR_INTERIOR
    assert np.asarray(topology.vertex_valence)[center] == 4
    assert topology.status is G1PatchStatus.SUPPORTED
    assert topology.unsupported_vertex_labels == ()
    assert (
        BiquinticGluingConstraints(topology, GluingContinuity.G1).status
        is G1PatchStatus.SUPPORTED
    )


def test_extraordinary_boundary_vertex_is_classified_and_g1_basis_fails_closed():
    topology = PreparedHalfEdgePatchTopology(
        ("lower_left", "right", "upper"),
        (
            ("a", "b", "x", "d"),
            ("b", "c", "e", "x"),
            ("d", "x", "g", "f"),
        ),
    )
    constraints = BiquinticGluingConstraints(topology, GluingContinuity.G1)

    assert topology.vertex_kind("x") is PatchVertexKind.EXTRAORDINARY_BOUNDARY
    assert topology.status is G1PatchStatus.EXTRAORDINARY_BOUNDARY_UNSUPPORTED
    assert topology.unsupported_vertex_labels == ("x",)
    assert constraints.status is G1PatchStatus.EXTRAORDINARY_BOUNDARY_UNSUPPORTED
    with pytest.raises(UnsupportedG1TopologyError) as captured:
        BiquinticGluingBasis(constraints)
    assert captured.value.status is G1PatchStatus.EXTRAORDINARY_BOUNDARY_UNSUPPORTED
    assert captured.value.unsupported_vertex_labels == ("x",)


def test_native_nullspace_reconstructs_and_projects_g1_coefficients():
    constraints = BiquinticGluingConstraints(_two_patch_topology(), GluingContinuity.G1)
    basis = BiquinticGluingBasis(constraints, relative_rank_tolerance=1.0e-6)

    assert basis.rank == 12
    assert basis.nullity == 60
    assert basis.nullspace_residual < 2.0e-5

    free = jnp.linspace(-0.75, 1.25, basis.nullity * 2).reshape((basis.nullity, 2))
    reconstructed = basis.reconstruct(free)
    assert reconstructed.shape == (2, 6, 6, 2)
    assert jnp.max(jnp.abs(constraints.residual(reconstructed))) < 2.0e-5
    assert jnp.allclose(basis.coordinates(reconstructed), free, atol=2.0e-5)

    raw = jnp.sin(jnp.arange(2 * 6 * 6 * 2, dtype="float64")).reshape((2, 6, 6, 2))
    projection = basis.project(raw)
    assert projection.finite
    assert projection.coefficients.shape == raw.shape
    assert jnp.max(jnp.abs(projection.residual_after)) < 2.0e-5
    assert jnp.max(jnp.abs(projection.residual_after)) < jnp.max(
        jnp.abs(projection.residual_before)
    )


def test_malformed_shared_edge_orientation_is_rejected_before_preparation():
    with pytest.raises(PatchTopologyError, match="opposite") as captured:
        PreparedHalfEdgePatchTopology(
            ("first", "second"),
            (
                ("a", "b", "c", "d"),
                ("a", "b", "f", "e"),
            ),
        )
    assert captured.value.status is G1PatchStatus.MALFORMED_TOPOLOGY
