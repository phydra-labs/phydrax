from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import pytest

from phydrax.applications.contact._rod_capsule import (
    prepare_reduced_rod_contact_participant,
    RodCapsuleGeometryPlan,
)
from phydrax.applications.solid_mechanics._rod_dynamics import (
    prepare_rod,
    RodPlan,
)
from phydrax.applications.solid_mechanics._rod_reduced_basis import (
    RodStrainBasisPlan,
)
from phydrax.applications.solid_mechanics._rod_reduction import (
    prepare_reduced_rod,
    ReducedRodPlan,
)
from phydrax.discretization.contact._implicit_geometry import PlaneContactGeometry
from phydrax.discretization.contact._surface import (
    CollisionFeatureKind,
    CollisionFeaturePolicy,
)


def _spatial_rod(*, stiffness_scale: float = 1.0):
    dtype = jnp.float32
    segment_count = 5
    return prepare_rod(
        RodPlan(
            jnp.asarray(
                ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)),
                dtype=jnp.int32,
            ),
            jnp.asarray(
                (
                    (0.0, 0.0, 1.0),
                    (1.0, 0.0, 1.0),
                    (2.0, 0.0, 1.0),
                    (3.0, 0.0, 1.0),
                    (4.0, 0.0, 1.0),
                    (5.0, 0.0, 1.0),
                ),
                dtype=dtype,
            ),
            jnp.broadcast_to(jnp.eye(3, dtype=dtype), (segment_count, 3, 3)),
            jnp.ones((segment_count + 1,), dtype=dtype),
            jnp.broadcast_to(jnp.eye(3, dtype=dtype), (segment_count, 3, 3)),
            stiffness_scale
            * jnp.broadcast_to(
                jnp.diag(jnp.asarray((80.0, 50.0, 40.0), dtype=dtype)),
                (segment_count, 3, 3),
            ),
            stiffness_scale
            * jnp.broadcast_to(
                jnp.diag(jnp.asarray((7.0, 8.0, 9.0), dtype=dtype)),
                (segment_count - 1, 3, 3),
            ),
        )
    )


def _planar_rod():
    dtype = jnp.float32
    return prepare_rod(
        RodPlan(
            jnp.asarray(((0, 1),), dtype=jnp.int32),
            jnp.asarray(((0.0, 0.0), (1.0, 0.0)), dtype=dtype),
            jnp.asarray((jnp.eye(2, dtype=dtype),)),
            jnp.ones((2,), dtype=dtype),
            jnp.ones((1,), dtype=dtype),
            jnp.asarray((jnp.eye(2, dtype=dtype),)),
            jnp.empty((0, 1, 1), dtype=dtype),
        )
    )


def _geometry(rod=None):
    source = _spatial_rod() if rod is None else rod
    plan = RodCapsuleGeometryPlan(
        jnp.asarray((0.2, 0.25, 0.15, 0.3, 0.1), dtype=jnp.float32),
        participant_id=3,
        body_id=7,
        material_id=11,
        patch_id=13,
        solver_clearance=jnp.asarray((0.01, 0.02, 0.03, 0.04, 0.05), dtype=jnp.float32),
    )
    return source, plan, plan.prepare(source)


def _reduced(rod):
    basis = RodStrainBasisPlan.shifted_legendre(
        0,
        dimension=3,
        component_scales=jnp.ones((6,), dtype=jnp.float32),
    )
    return prepare_reduced_rod(rod, ReducedRodPlan(basis))


def _skew_configuration(rod):
    positions = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.5, 0.5, 0.3),
            (0.5, -1.0, 1.0),
            (0.5, 1.0, 1.0),
            (1.5, 1.0, 1.0),
        ),
        dtype=jnp.float32,
    )
    orientations = jnp.broadcast_to(
        jnp.asarray((1.0, 0.0, 0.0, 0.0), dtype=jnp.float32),
        (rod.plan.segment_count, 4),
    )
    return positions, orientations


def _plane(normal, offset):
    policy = CollisionFeaturePolicy(
        jnp.asarray((100,), dtype=jnp.int64),
        jnp.asarray((int(CollisionFeatureKind.ANALYTIC),), dtype=jnp.int32),
        participant_ids=17,
        body_ids=19,
        material_ids=23,
        patch_ids=29,
        static_mask=True,
        provenance_id="test-plane-feature",
    )
    return PlaneContactGeometry(normal, offset, feature_policy=policy)


def test_plan_owns_exact_feature_radius_labels_proxy_and_adjacency_policy():
    rod, plan, geometry = _geometry()
    features = geometry.surface_plan.feature_policy
    edge_slice = features.edge_slice

    assert geometry.rod.prepared_id == rod.prepared_id
    assert jnp.array_equal(
        features.feature_ids[features.vertex_slice], plan.node_feature_ids
    )
    assert jnp.array_equal(
        features.feature_ids[edge_slice],
        plan.segment_feature_ids[geometry.surface_edge_order],
    )
    assert jnp.array_equal(
        features.physical_radius[edge_slice],
        plan.segment_radii[geometry.surface_edge_order],
    )
    assert jnp.all(features.physical_radius[features.vertex_slice] > 0.0)
    assert jnp.all(
        features.contact_extent == features.physical_radius + features.solver_clearance
    )
    assert jnp.all(features.participant_ids == 3)
    assert jnp.all(features.body_ids == 7)
    assert jnp.all(features.material_ids == 11)
    assert jnp.all(features.patch_ids == 13)
    assert jnp.all(features.proxy_error == 0.0)
    assert jnp.array_equal(
        geometry.surface_plan.pair_policy.excluded_vertex_pairs,
        rod.plan.segment_node_ids,
    )

    repeated = RodCapsuleGeometryPlan(
        jnp.asarray((0.2, 0.25, 0.15, 0.3, 0.1), dtype=jnp.float32),
        participant_id=3,
        body_id=7,
        material_id=11,
        patch_id=13,
        solver_clearance=jnp.asarray((0.01, 0.02, 0.03, 0.04, 0.05), dtype=jnp.float32),
    )
    changed = RodCapsuleGeometryPlan(
        jnp.asarray((0.2, 0.25, 0.16, 0.3, 0.1), dtype=jnp.float32),
        participant_id=3,
        body_id=7,
        material_id=11,
        patch_id=13,
        solver_clearance=jnp.asarray((0.01, 0.02, 0.03, 0.04, 0.05), dtype=jnp.float32),
    )
    assert repeated.plan_id == plan.plan_id
    assert repeated.prepare(rod).prepared_id == geometry.prepared_id
    assert changed.plan_id != plan.plan_id


def test_capsule_plane_witness_is_exact_for_endpoint_support_and_penetration_gap():
    rod, _, geometry = _geometry()
    positions, orientations = _skew_configuration(rod)
    positions = positions.at[0].set(jnp.asarray((0.0, 0.0, 2.0)))
    positions = positions.at[1].set(jnp.asarray((1.0, 0.0, 1.0)))
    plane = _plane(jnp.asarray((0.0, 0.0, 2.0)), 0.0)

    witness = geometry.capsule_plane_witness(
        (positions, orientations),
        jnp.asarray((0,), dtype=jnp.int32),
        plane,
    )

    assert witness.valid[0]
    assert witness.axial_coordinates[0] == pytest.approx(1.0)
    assert jnp.allclose(witness.normal[0], jnp.asarray((0.0, 0.0, 1.0)))
    assert jnp.allclose(witness.centerline_witness[0], jnp.asarray((1.0, 0.0, 1.0)))
    assert jnp.allclose(witness.capsule_witness[0], jnp.asarray((1.0, 0.0, 0.8)))
    assert jnp.allclose(witness.plane_witness[0], jnp.asarray((1.0, 0.0, 0.0)))
    assert witness.signed_centerline_distance[0] == pytest.approx(1.0)
    assert witness.gap[0] == pytest.approx(0.8)
    assert jnp.allclose(
        witness.capsule_witness[0] - witness.plane_witness[0],
        witness.gap[0] * witness.normal[0],
    )


def test_capsule_capsule_witness_is_exact_and_rigid_motion_invariant():
    rod, _, geometry = _geometry()
    configuration = _skew_configuration(rod)
    pairs = jnp.asarray(((0, 3),), dtype=jnp.int32)

    witness = geometry.capsule_capsule_witness(configuration, pairs)

    assert witness.valid[0]
    assert not witness.adjacent[0]
    assert witness.left_axial_coordinates[0] == pytest.approx(0.5)
    assert witness.right_axial_coordinates[0] == pytest.approx(0.5)
    assert witness.centerline_distance[0] == pytest.approx(1.0)
    assert witness.gap[0] == pytest.approx(0.5)
    assert jnp.allclose(witness.normal[0], jnp.asarray((0.0, 0.0, -1.0)))
    assert jnp.allclose(witness.left_capsule_witness[0], jnp.asarray((0.5, 0.0, 0.2)))
    assert jnp.allclose(witness.right_capsule_witness[0], jnp.asarray((0.5, 0.0, 0.7)))
    assert jnp.allclose(
        witness.left_capsule_witness[0] - witness.right_capsule_witness[0],
        witness.gap[0] * witness.normal[0],
    )

    rotation = jnp.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=jnp.float32,
    )
    translation = jnp.asarray((2.0, -3.0, 0.7), dtype=jnp.float32)
    moved_positions = configuration[0] @ rotation.T + translation
    moved = geometry.capsule_capsule_witness(
        (moved_positions, configuration[1]),
        pairs,
    )
    assert moved.gap[0] == pytest.approx(witness.gap[0])
    assert jnp.allclose(moved.normal[0], rotation @ witness.normal[0])
    assert jnp.allclose(
        moved.left_capsule_witness[0],
        rotation @ witness.left_capsule_witness[0] + translation,
    )
    assert jnp.allclose(
        moved.right_capsule_witness[0],
        rotation @ witness.right_capsule_witness[0] + translation,
    )


def test_spin_surface_velocity_and_native_effort_pullback_preserve_power_and_wrench():
    rod, _, geometry = _geometry()
    state = rod.initialize_state()
    configuration = rod.configuration_from_state(state)
    linear = jnp.zeros_like(state.velocities)
    angular = jnp.zeros_like(state.angular_velocities).at[0, 2].set(2.0)
    segment_indices = jnp.asarray((0,), dtype=jnp.int32)
    coordinates = jnp.asarray((0.25,), dtype=jnp.float32)
    offsets = jnp.asarray(((0.0, 0.2, 0.0),), dtype=jnp.float32)
    effort = jnp.asarray(((3.0, 0.0, 0.0),), dtype=jnp.float32)

    velocity = geometry.surface_velocity(
        configuration,
        (linear, angular),
        segment_indices,
        coordinates,
        offsets,
    )
    native_effort = geometry.native_effort_pullback(
        configuration,
        segment_indices,
        coordinates,
        offsets,
        effort,
    )

    assert jnp.allclose(velocity[0], jnp.asarray((-0.4, 0.0, 0.0)))
    assert jnp.allclose(jnp.sum(native_effort[0], axis=0), effort[0])
    assert jnp.allclose(native_effort[1][0], jnp.asarray((0.0, 0.0, -0.6)))
    surface_position = geometry.surface_positions(
        configuration,
        segment_indices,
        coordinates,
        offsets,
    )[0]
    surface_moment = jnp.cross(surface_position, effort[0])
    nodal_moment = jnp.sum(jnp.cross(configuration[0], native_effort[0]), axis=0)
    material_moment = native_effort[1][0]
    assert jnp.allclose(nodal_moment + material_moment, surface_moment)
    surface_power = jnp.sum(effort * velocity)
    native_power = rod.effort_space.pair(native_effort, (linear, angular))
    assert surface_power == pytest.approx(native_power)


def test_reduced_surface_pullback_is_the_true_dual_of_native_lift_and_spin_map():
    rod, _, geometry = _geometry()
    reduced = _reduced(rod)
    participant = prepare_reduced_rod_contact_participant(reduced, geometry)
    coefficients = reduced.initialize_state().coefficients
    rates = jnp.asarray((0.3, -0.2, 0.1, 0.4, -0.5, 0.25), dtype=jnp.float32)
    segment_indices = jnp.asarray((3,), dtype=jnp.int32)
    coordinates = jnp.asarray((0.5,), dtype=jnp.float32)
    offsets = jnp.asarray(((0.0, 0.3, 0.0),), dtype=jnp.float32)
    effort = jnp.asarray(((1.25, -0.75, 0.5),), dtype=jnp.float32)

    evidence = participant.surface_duality_evidence(
        coefficients,
        rates,
        segment_indices,
        coordinates,
        offsets,
        effort,
    )
    pulled = participant.surface_effort_pullback(
        coefficients,
        segment_indices,
        coordinates,
        offsets,
        effort,
    )
    velocity = participant.surface_velocity(
        coefficients,
        rates,
        segment_indices,
        coordinates,
        offsets,
    )

    assert evidence.finite
    assert evidence.valid
    assert evidence.surface_power == pytest.approx(jnp.sum(effort * velocity))
    assert evidence.reduced_power == pytest.approx(
        reduced.reduced_effort_space.pair(pulled, rates)
    )
    assert evidence.native_residual == pytest.approx(0.0, abs=2.0e-6)
    assert evidence.reduced_residual == pytest.approx(0.0, abs=2.0e-6)

    node_effort = jnp.arange(rod.plan.node_count * 3, dtype=jnp.float32).reshape((-1, 3))
    generic = participant.duality_evidence(coefficients, rates, node_effort)
    assert generic.finite
    assert generic.valid


def test_adjacent_capsules_are_filtered_and_ambiguous_geometry_is_rejected():
    rod, _, geometry = _geometry()
    adjacent = geometry.capsule_capsule_witness(
        rod.configuration_from_state(rod.initialize_state()),
        jnp.asarray(((0, 1),), dtype=jnp.int32),
    )
    assert adjacent.adjacent[0]
    assert not adjacent.valid[0]

    common = dict(participant_id=0, body_id=0, material_id=0, patch_id=0)
    with pytest.raises(ValueError, match="constant circular radius"):
        RodCapsuleGeometryPlan(jnp.asarray(((0.2, 0.3),)), **common)
    with pytest.raises(ValueError, match="positive"):
        RodCapsuleGeometryPlan(jnp.asarray((0.0,)), **common)
    with pytest.raises(ValueError, match="zero proxy_error"):
        RodCapsuleGeometryPlan(jnp.asarray((0.2,)), proxy_error=1.0e-3, **common)
    with pytest.raises(TypeError, match="participant_id"):
        RodCapsuleGeometryPlan(
            jnp.asarray((0.2,)),
            participant_id=jnp.asarray((0, 1)),
            body_id=0,
            material_id=0,
            patch_id=0,
        )
    with pytest.raises(ValueError, match="spatial"):
        RodCapsuleGeometryPlan(jnp.asarray((0.2,)), **common).prepare(_planar_rod())

    segment_indices = jnp.asarray((0,), dtype=jnp.int32)
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="exact circular capsule boundary",
    ):
        geometry.surface_velocity(
            rod.configuration_from_state(rod.initialize_state()),
            rod.velocity_from_state(rod.initialize_state()),
            segment_indices,
            jnp.asarray((0.5,), dtype=jnp.float32),
            jnp.zeros((1, 3), dtype=jnp.float32),
        )


def test_reduced_participant_rejects_geometry_owned_by_a_different_native_rod():
    first_rod, _, geometry = _geometry()
    second_rod = _spatial_rod(stiffness_scale=2.0)
    reduction = _reduced(second_rod)

    assert first_rod.prepared_id != second_rod.prepared_id
    with pytest.raises(ValueError, match="same PreparedRod"):
        prepare_reduced_rod_contact_participant(reduction, geometry)
