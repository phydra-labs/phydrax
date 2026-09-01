#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _diamond_mesh():
    coordinates = jnp.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [0.0, 0.0],
        ]
    )
    triangles = jnp.asarray(
        [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        dtype=jnp.int32,
    )
    return phx.discretization.CellMesh.from_triangles(coordinates, triangles)


def _discretization():
    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        phx.discretization.lagrange_element("triangle", 1),
    )
    return phx.discretization.FiniteElementPlan(_diamond_mesh(), field).prepare()


def _circle_motion():
    geometry = phx.geometry.Circle(
        (0.0, 0.0),
        1.0,
        feature_id="circle",
    ).compile()
    projection = phx.geometry.ImplicitPointProjectionPlan(
        geometry,
        _diamond_mesh().coordinates[:4],
        0.3,
        source_id="circle-boundary",
    )
    motion = phx.discretization.FiniteElementMeshMotionPlan(
        _discretization(),
        projection,
    )
    return geometry, motion


def test_runtime_rejects_changed_coordinate_count():
    discretization = _discretization()

    with pytest.raises(ValueError, match="preserve coordinate shape"):
        discretization.prepare_runtime(
            discretization.mesh.coordinates[:-1],
            numeric_version="invalid",
        )


def test_harmonic_mesh_motion_preserves_topology_and_has_shape_derivative():
    geometry, motion = _circle_motion()
    radius_index = geometry.schema.index(phx.geometry.ParameterId("circle", "radius"))
    state = geometry.state.replace_at(radius_index, jnp.asarray(1.1))
    result = eqx.filter_jit(motion.realize)(state)

    assert bool(result.accepted)
    assert result.runtime.topology_id == motion.topology_id
    assert result.runtime.geometry_layout_id == motion.geometry_layout_id
    assert jnp.allclose(result.coordinates[:4], 1.1 * motion.reference_coordinates[:4])
    assert jnp.allclose(result.coordinates[4], jnp.zeros((2,)), atol=1.0e-7)
    assert result.evidence.geometry.minimum_relative_jacobian > 1.0

    def coordinate_sum(radius):
        design = geometry.state.replace_at(radius_index, radius)
        return jnp.sum(motion.realize(design).proposed_coordinates ** 2)

    derivative = jax.grad(coordinate_sum)(jnp.asarray(1.0))
    assert jnp.isfinite(derivative)
    assert derivative > 0.0


def test_invalid_boundary_motion_returns_base_runtime_and_rejected_evidence():
    geometry, motion = _circle_motion()
    radius_index = geometry.schema.index(phx.geometry.ParameterId("circle", "radius"))
    expired = geometry.state.replace_at(radius_index, jnp.asarray(1.8))

    result = eqx.filter_jit(motion.realize)(expired)

    assert not bool(result.accepted)
    assert bool(result.refresh_required)
    assert jnp.array_equal(result.coordinates, motion.reference_coordinates)
    assert jnp.array_equal(result.runtime.coordinates, motion.reference_coordinates)


class _InvertingProvider(eqx.Module):
    reference_points: jax.Array
    mapping_id: str = eqx.field(static=True)

    def __init__(self, reference_points):
        self.reference_points = jnp.asarray(reference_points)
        self.mapping_id = "inverting-boundary"

    def realize(self, design, /):
        del design
        proposed = self.reference_points.at[0].set(jnp.asarray([-1.5, 0.0]))
        return phx.discretization.FiniteElementBoundaryRealization(
            proposed,
            proposed,
            accepted=True,
            refresh_required=False,
            status=0,
            mapping_id=self.mapping_id,
        )


def test_signed_jacobian_rejects_orientation_reversal():
    discretization = _discretization()
    provider = _InvertingProvider(discretization.mesh.coordinates[:4])
    motion = phx.discretization.FiniteElementMeshMotionPlan(
        discretization,
        provider,
        policy=phx.discretization.FiniteElementMeshMotionPolicy(
            maximum_displacement_fraction=10.0
        ),
    )

    result = motion.realize(jnp.asarray(0.0))

    assert not bool(result.accepted)
    assert not bool(result.evidence.geometry.orientation_preserved)
    assert jnp.array_equal(result.coordinates, motion.reference_coordinates)
