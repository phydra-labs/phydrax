#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications.contact._articulated import (
    build_contact_velocity_operator,
    build_delassus_operator,
    prepare_articulated_contact,
    solve_articulated_contact,
)


def _robot_plan():
    return phx.discretization.CollisionSurfacePlan(
        jnp.asarray((0,), dtype=jnp.int64),
        ambient_dimension=2,
        allow_isolated_vertices=True,
        pair_policy=phx.discretization.ContactPairPolicy(
            1,
            body_ids=jnp.asarray((0,), dtype=jnp.int64),
            material_ids=jnp.zeros((1,), dtype=jnp.int64),
        ),
    )


def _ground_plan():
    return phx.discretization.CollisionSurfacePlan(
        jnp.asarray((1, 2), dtype=jnp.int64),
        ambient_dimension=2,
        edges=jnp.asarray(((0, 1),), dtype=jnp.int32),
        pair_policy=phx.discretization.ContactPairPolicy(
            2,
            body_ids=jnp.ones((2,), dtype=jnp.int64),
            material_ids=jnp.zeros((2,), dtype=jnp.int64),
            static_mask=jnp.ones((2,), dtype=bool),
        ),
    )


def _articulated_case():
    configuration_space = phx.linalg.ArraySpace((1,), dtype=np.float64)
    tangent_space = phx.linalg.ArraySpace((2,), dtype=np.float64)
    plan = _robot_plan()

    def positions(configuration):
        return jnp.stack(
            (jnp.asarray(0.0, dtype=configuration.dtype), configuration[0])
        )[None, :]

    def velocities(configuration, velocity):
        del configuration
        return jnp.stack(
            (jnp.asarray(0.0, dtype=velocity.dtype), velocity[0])
        )[None, :]

    def force_pullback(configuration, surface_force):
        del configuration
        return jnp.asarray(
            (surface_force[0, 1], 0.0), dtype=surface_force.dtype
        )

    participant = phx.applications.contact.make_articulated_contact_participant(
        plan,
        configuration_space,
        positions,
        tangent_space=tangent_space,
        velocity_action=velocities,
        pullback_action=force_pullback,
        participant_id="articulated-test-robot",
    )
    ground_space = phx.linalg.ArraySpace((0,), dtype=np.float64)
    ground = phx.discretization.FunctionContactParticipant(
        _ground_plan(),
        ground_space,
        lambda state: jnp.asarray(
            ((-1.0, 0.0), (1.0, 0.0)), dtype=state.dtype
        ),
        participant_id="articulated-test-ground",
    )
    configuration = jnp.asarray((0.05,), dtype=jnp.float64)
    free_velocity = jnp.asarray((-1.0, 0.37), dtype=jnp.float64)
    scene = phx.discretization.ContactParticipantScene((participant, ground))
    ground_state = ground_space.zeros()
    world_positions = scene.positions((configuration, ground_state))
    world_velocities = scene.velocities(
        (configuration, ground_state), (free_velocity, ground_state)
    )
    search = phx.discretization.SweepAndPruneContactSearchPlan(
        edge_vertex_capacity=1,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
    )
    candidates = search.build(scene, world_positions)
    kinematics = phx.discretization.evaluate_contact_kinematics(
        scene,
        candidates,
        world_positions,
        world_velocities,
        0.01,
        rest_positions=world_positions,
        activation_distance=0.1,
    )
    materials = phx.applications.contact.ContactMaterialPairTable.uniform(
        normal_stiffness=1.0,
        static_friction=0.0,
        dynamic_friction=0.0,
        restitution=0.0,
    )
    inverse_mass = phx.linalg.DenseLinearOperator(
        jnp.asarray(((0.5, 0.0), (0.0, 0.0)), dtype=jnp.float64),
        source=tangent_space,
        target=tangent_space,
    )
    return (
        participant,
        configuration,
        free_velocity,
        kinematics,
        materials,
        inverse_mass,
    )


def test_function_participant_preserves_legacy_space_and_requires_distinct_actions():
    plan = _robot_plan()
    legacy_space = phx.linalg.ArraySpace((1, 2), dtype=np.float64)

    def identity_positions(configuration):
        return configuration

    legacy = phx.discretization.FunctionContactParticipant(
        plan, legacy_space, identity_positions, participant_id="legacy-participant-test"
    )
    state = jnp.asarray(((1.0, 2.0),), dtype=jnp.float64)
    rate = jnp.asarray(((0.25, -0.5),), dtype=jnp.float64)
    force = jnp.asarray(((3.0, -2.0),), dtype=jnp.float64)

    assert legacy.tangent_space.compatible(legacy.source_space)
    np.testing.assert_allclose(legacy.velocities(state, rate), rate)
    np.testing.assert_allclose(legacy.force_pullback(state, force), force)
    assert bool(legacy.duality_evidence(state, rate, force).valid)

    distinct = phx.linalg.ArraySpace((3,), dtype=np.float64)
    with pytest.raises(ValueError, match="Distinct configuration and tangent spaces"):
        phx.discretization.FunctionContactParticipant(
            plan,
            legacy_space,
            identity_positions,
            tangent_space=distinct,
        )


def test_distinct_configuration_and_tangent_spaces_validate_velocity_and_effort():
    participant, configuration, velocity, _, _, _ = _articulated_case()
    world_velocity = participant.velocities(configuration, velocity)
    surface_force = jnp.asarray(((0.0, 2.0),), dtype=jnp.float64)
    evidence = participant.duality_evidence(
        configuration, velocity, surface_force
    )

    assert participant.source_space.size == 1
    assert participant.tangent_space.size == 2
    assert world_velocity.shape == (1, 2)
    assert participant.force_pullback(configuration, surface_force).shape == (2,)
    assert bool(evidence.valid)
    with pytest.raises(ValueError, match="Vector must have shape"):
        participant.velocities(configuration, jnp.zeros((1,), dtype=jnp.float64))


def test_delassus_composition_matches_dense_g_minv_g_adjoint():
    participant, configuration, _, kinematics, _, inverse_mass = _articulated_case()
    velocity_operator = build_contact_velocity_operator(
        participant, configuration, kinematics
    )
    delassus = build_delassus_operator(velocity_operator, inverse_mass)
    policy = phx.linalg.MaterializationPolicy(max_entries=64, max_bytes=4096)
    velocity_matrix = phx.linalg.materialize(velocity_operator, policy)
    inverse_mass_matrix = phx.linalg.materialize(inverse_mass, policy)
    delassus_matrix = phx.linalg.materialize(delassus, policy)

    np.testing.assert_allclose(
        delassus_matrix,
        velocity_matrix @ inverse_mass_matrix @ velocity_matrix.T,
        atol=1.0e-12,
    )

    wrong_space = phx.linalg.ArraySpace((3,), dtype=np.float64)
    wrong_inverse_mass = phx.linalg.DenseLinearOperator(
        jnp.eye(3, dtype=jnp.float64), source=wrong_space, target=wrong_space
    )
    with pytest.raises(ValueError, match="contact tangent space"):
        build_delassus_operator(velocity_operator, wrong_inverse_mass)


def test_frictionless_articulated_impact_applies_constrained_generalized_impulse():
    participant, configuration, free, kinematics, materials, inverse_mass = (
        _articulated_case()
    )
    prepared = prepare_articulated_contact(
        participant,
        configuration,
        free,
        kinematics,
        materials,
        inverse_mass,
    )
    result = solve_articulated_contact(prepared)

    assert bool(result.evidence.successful)
    assert bool(result.evidence.preparation.successful)
    assert bool(result.evidence.cone.successful)
    assert bool(result.evidence.duality.valid)
    assert bool(result.evidence.contact_certificate_valid)
    assert result.evidence.cone.complementarity_defect <= 1.0e-8
    assert result.evidence.cone.cone_defect <= 1.0e-8
    assert result.evidence.minimum_post_normal_velocity >= -1.0e-10
    assert result.post_contact_velocity[0, 0] >= -1.0e-10
    np.testing.assert_allclose(result.post_velocity[1], free[1], atol=1.0e-12)
    np.testing.assert_allclose(
        result.evidence.duality.contact_power,
        result.evidence.duality.generalized_power,
        atol=1.0e-12,
    )


def test_unsuccessful_articulated_cone_solve_fails_closed():
    participant, configuration, free, kinematics, materials, inverse_mass = (
        _articulated_case()
    )
    prepared = prepare_articulated_contact(
        participant,
        configuration,
        free,
        kinematics,
        materials,
        inverse_mass,
    )
    result = solve_articulated_contact(
        prepared,
        solver=phx.applications.contact.ContactConeSolverPlan(maximum_iterations=1),
    )

    assert not bool(result.evidence.successful)
    assert not bool(result.evidence.cone.successful)
    assert bool(result.evidence.fail_closed)
    np.testing.assert_allclose(result.impulse, 0.0, atol=0.0)
    np.testing.assert_allclose(result.velocity_update, 0.0, atol=0.0)
    np.testing.assert_allclose(result.post_velocity, free, atol=0.0)
