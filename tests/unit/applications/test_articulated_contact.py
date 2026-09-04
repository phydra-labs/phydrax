#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications.contact._articulated import (
    apply_articulated_contact_impulse,
    build_contact_velocity_operator,
    build_delassus_operator,
    prepare_articulated_contact,
    solve_articulated_contact,
)


def _robot_plan():
    return phx.discretization.CollisionSurfacePlan(
        jnp.asarray((0,), dtype=jnp.int64),
        ambient_dimension=2,
        participant_ids=0,
        body_ids=0,
        material_ids=0,
        pair_policy=phx.discretization.ContactPairPolicy(
            1, allowed_participant_pairs=jnp.asarray(((0, 1),), dtype=jnp.int64)
        ),
    )


def _ground_plan():
    return phx.discretization.CollisionSurfacePlan(
        jnp.asarray((1, 2), dtype=jnp.int64),
        ambient_dimension=2,
        edges=jnp.asarray(((0, 1),), dtype=jnp.int32),
        participant_ids=1,
        body_ids=1,
        material_ids=0,
        static_mask=True,
        pair_policy=phx.discretization.ContactPairPolicy(
            2, allowed_participant_pairs=jnp.asarray(((0, 1),), dtype=jnp.int64)
        ),
    )


def _articulated_case():
    configuration_space = phx.linalg.ArraySpace((1,), dtype=np.float64)
    tangent_space = phx.linalg.ArraySpace((2,), dtype=np.float64)
    plan = _robot_plan()

    def positions(configuration):
        return jnp.stack((jnp.asarray(0.0, dtype=configuration.dtype), configuration[0]))[
            None, :
        ]

    def velocities(configuration, velocity):
        del configuration
        return jnp.stack((jnp.asarray(0.0, dtype=velocity.dtype), velocity[0]))[None, :]

    def effort_pullback(configuration, surface_effort):
        del configuration
        return jnp.asarray((surface_effort[0, 1], 0.0), dtype=surface_effort.dtype)

    participant = phx.applications.contact.make_articulated_contact_participant(
        plan,
        configuration_space,
        positions,
        tangent_space=tangent_space,
        velocity_action=velocities,
        effort_pullback_action=effort_pullback,
        participant_id="articulated-test-robot",
    )
    ground_space = phx.linalg.ArraySpace((0,), dtype=np.float64)
    ground = phx.discretization.FunctionContactParticipant(
        _ground_plan(),
        ground_space,
        lambda state: jnp.asarray(((-1.0, 0.0), (1.0, 0.0)), dtype=state.dtype),
        tangent_space=ground_space,
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
        source=phx.linalg.DualSpace(tangent_space),
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


def test_function_participant_uses_explicit_equal_spaces_and_true_efforts():
    plan = _robot_plan()
    space = phx.linalg.ArraySpace((1, 2), dtype=np.float64)

    def identity_positions(configuration):
        return configuration

    participant = phx.discretization.FunctionContactParticipant(
        plan,
        space,
        identity_positions,
        tangent_space=space,
        participant_id="equal-space-participant-test",
    )
    state = jnp.asarray(((1.0, 2.0),), dtype=jnp.float64)
    rate = jnp.asarray(((0.25, -0.5),), dtype=jnp.float64)
    effort = jnp.asarray(((3.0, -2.0),), dtype=jnp.float64)

    assert participant.tangent_space.compatible(participant.source_space)
    np.testing.assert_allclose(participant.velocities(state, rate), rate)
    np.testing.assert_allclose(participant.effort_pullback(state, effort), effort)
    assert participant.effort_space.compatible(phx.linalg.DualSpace(space))
    assert bool(participant.duality_evidence(state, rate, effort).valid)

    distinct = phx.linalg.ArraySpace((3,), dtype=np.float64)
    with pytest.raises(ValueError, match="Distinct configuration and tangent spaces"):
        phx.discretization.FunctionContactParticipant(
            plan,
            space,
            identity_positions,
            tangent_space=distinct,
        )


def test_distinct_configuration_and_tangent_spaces_validate_velocity_and_effort():
    participant, configuration, velocity, _, _, _ = _articulated_case()
    world_velocity = participant.velocities(configuration, velocity)
    surface_effort = jnp.asarray(((0.0, 2.0),), dtype=jnp.float64)
    evidence = participant.duality_evidence(configuration, velocity, surface_effort)

    assert participant.source_space.size == 1
    assert participant.tangent_space.size == 2
    assert world_velocity.shape == (1, 2)
    assert participant.effort_pullback(configuration, surface_effort).shape == (2,)
    assert bool(evidence.valid)
    with pytest.raises(ValueError, match="Vector must have shape"):
        participant.velocities(configuration, jnp.zeros((1,), dtype=jnp.float64))


def test_delassus_composition_matches_dense_g_minv_g_dual_transpose():
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


@pytest.mark.parametrize("material_id", (-1, 1))
def test_active_route_without_in_range_mechanical_material_fails_closed(material_id):
    participant, configuration, free, kinematics, materials, inverse_mass = (
        _articulated_case()
    )
    changed = eqx.tree_at(
        lambda epoch: epoch.batches[0].left_material_ids,
        kinematics,
        jnp.full_like(kinematics.batches[0].left_material_ids, material_id),
    )
    prepared = prepare_articulated_contact(
        participant,
        configuration,
        free,
        changed,
        materials,
        inverse_mass,
    )
    result = solve_articulated_contact(prepared)

    assert not bool(prepared.evidence.material_law_complete)
    assert not bool(result.evidence.successful)
    assert bool(result.evidence.fail_closed)
    np.testing.assert_array_equal(result.impulse, jnp.zeros_like(result.impulse))
    np.testing.assert_array_equal(result.post_velocity, free)


def test_out_of_range_material_is_allowed_only_on_padding():
    participant, configuration, free, kinematics, materials, inverse_mass = (
        _articulated_case()
    )
    padded = eqx.tree_at(
        lambda epoch: (
            epoch.batches[0].valid,
            epoch.batches[0].left_material_ids,
        ),
        kinematics,
        (
            jnp.zeros_like(kinematics.batches[0].valid),
            jnp.full_like(kinematics.batches[0].left_material_ids, -1),
        ),
    )
    prepared = prepare_articulated_contact(
        participant,
        configuration,
        free,
        padded,
        materials,
        inverse_mass,
    )
    result = solve_articulated_contact(prepared)

    assert bool(prepared.evidence.material_law_complete)
    assert bool(result.evidence.successful)
    np.testing.assert_array_equal(result.impulse, jnp.zeros_like(result.impulse))
    np.testing.assert_array_equal(result.post_velocity, free)


def test_active_route_with_unavailable_mechanical_law_fails_closed():
    participant, configuration, free, kinematics, materials, inverse_mass = (
        _articulated_case()
    )
    unavailable = eqx.tree_at(
        lambda table: table.mechanical_available,
        materials,
        jnp.zeros_like(materials.mechanical_available),
    )
    prepared = prepare_articulated_contact(
        participant,
        configuration,
        free,
        kinematics,
        unavailable,
        inverse_mass,
    )
    result = solve_articulated_contact(prepared)

    assert not bool(result.evidence.preparation.material_law_complete)
    assert not bool(result.evidence.cone.material_law_complete)
    assert not bool(result.evidence.successful)
    np.testing.assert_array_equal(result.impulse, jnp.zeros_like(result.impulse))
    np.testing.assert_array_equal(result.post_velocity, free)


def test_indefinite_delassus_is_spectrally_rejected_and_rolls_back():
    participant, configuration, free, kinematics, materials, _ = _articulated_case()
    inverse_mass = phx.linalg.DenseLinearOperator(
        jnp.asarray(((-0.5, 0.0), (0.0, 0.0)), dtype=jnp.float64),
        source=participant.tangent_space,
        target=participant.tangent_space,
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

    assert prepared.evidence.minimum_delassus_eigenvalue < 0.0
    assert not bool(prepared.evidence.delassus_spectral_valid)
    assert not bool(result.evidence.successful)
    assert bool(result.evidence.fail_closed)
    np.testing.assert_array_equal(result.impulse, jnp.zeros_like(result.impulse))
    np.testing.assert_array_equal(result.post_velocity, free)


def test_stale_cone_numeric_revision_cannot_apply():
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
    cone_result = phx.applications.contact.solve_contact_cone(prepared.program)
    changed_program = eqx.tree_at(
        lambda program: program.effective_mass,
        prepared.program,
        2.0 * prepared.program.effective_mass,
    )
    changed_prepared = eqx.tree_at(lambda value: value.program, prepared, changed_program)
    result = apply_articulated_contact_impulse(changed_prepared, cone_result)

    assert not bool(result.evidence.numeric_revision_matches)
    assert not bool(result.evidence.successful)
    assert bool(result.evidence.fail_closed)
    np.testing.assert_array_equal(result.impulse, jnp.zeros_like(result.impulse))
    np.testing.assert_array_equal(result.post_velocity, free)


def _single_contact_program(
    tangential_velocity,
    *,
    static_friction,
    dynamic_friction,
    effective_normal=1.0,
    restitution=0.0,
):
    return phx.applications.contact.ContactConeProgram(
        jnp.asarray(((-1.0, tangential_velocity),), dtype=jnp.float64),
        jnp.asarray(((effective_normal, 0.0), (0.0, 1.0)), dtype=jnp.float64),
        jnp.zeros((2,), dtype=jnp.float64),
        jnp.asarray((dynamic_friction,), dtype=jnp.float64),
        jnp.asarray((1,), dtype=jnp.int64),
        jnp.asarray((True,)),
        1,
        "single-contact-law-test",
        static_friction=jnp.asarray((static_friction,), dtype=jnp.float64),
        restitution=jnp.asarray((restitution,), dtype=jnp.float64),
    )


def test_signorini_and_coulomb_evidence_use_one_static_and_sliding_law():
    sticking = phx.applications.contact.solve_contact_cone(
        _single_contact_program(0.2, static_friction=0.5, dynamic_friction=0.3)
    )
    sliding = phx.applications.contact.solve_contact_cone(
        _single_contact_program(1.0, static_friction=0.5, dynamic_friction=0.3)
    )

    assert bool(sticking.evidence.successful)
    assert bool(sliding.evidence.successful)
    np.testing.assert_allclose(sticking.impulse, ((1.0, -0.2),), atol=1.0e-8)
    np.testing.assert_allclose(sticking.contact_law_velocity, 0.0, atol=1.0e-8)
    np.testing.assert_allclose(sliding.impulse, ((1.0, -0.3),), atol=1.0e-8)
    np.testing.assert_allclose(sliding.contact_law_velocity, ((0.0, 0.7),), atol=1.0e-8)
    assert sticking.evidence.complementarity_defect <= 1.0e-8
    assert sliding.evidence.complementarity_defect <= 1.0e-8
    assert sticking.evidence.maximum_dissipation_defect <= 1.0e-8
    assert sliding.evidence.maximum_dissipation_defect <= 1.0e-8
    assert bool(sticking.evidence.dissipative)
    assert bool(sliding.evidence.dissipative)


def test_unequal_mass_frictionless_impact_and_singular_psd_route_are_supported():
    inverse_effective_mass = 1.0 / 2.0 + 1.0 / 3.0
    restitution = 0.25
    closing_velocity = -1.2
    program = phx.applications.contact.ContactConeProgram(
        jnp.asarray(
            (((1.0 + restitution) * closing_velocity, 0.0),),
            dtype=jnp.float64,
        ),
        jnp.asarray(((inverse_effective_mass, 0.0), (0.0, 0.0)), dtype=jnp.float64),
        jnp.zeros((2,), dtype=jnp.float64),
        jnp.zeros((1,), dtype=jnp.float64),
        jnp.asarray((1,), dtype=jnp.int64),
        jnp.asarray((True,)),
        1,
        "unequal-mass-frictionless-impact",
        restitution=jnp.asarray((restitution,), dtype=jnp.float64),
    )
    result = phx.applications.contact.solve_contact_cone(program)

    expected_impulse = -(1.0 + restitution) * closing_velocity
    expected_impulse /= inverse_effective_mass
    assert bool(result.evidence.successful)
    np.testing.assert_allclose(
        result.impulse[0, 0], expected_impulse, rtol=1.0e-8, atol=1.0e-8
    )
    np.testing.assert_allclose(result.impulse[0, 1], 0.0, atol=0.0)
    np.testing.assert_allclose(result.contact_law_velocity, 0.0, atol=1.0e-8)


def test_fixed_route_cone_jit_and_vmap_match_eager():
    program = _single_contact_program(0.2, static_friction=0.5, dynamic_friction=0.3)
    eager = phx.applications.contact.solve_contact_cone(program)
    compiled = jax.jit(phx.applications.contact.solve_contact_cone)(program)
    free_batch = jnp.asarray((((-1.0, 0.2),), ((-1.0, 1.0),)), dtype=jnp.float64)

    def solve_free(free_velocity):
        changed = eqx.tree_at(lambda value: value.free_velocity, program, free_velocity)
        return phx.applications.contact.solve_contact_cone(changed).impulse

    mapped = jax.vmap(solve_free)(free_batch)
    eager_mapped = jnp.stack(tuple(solve_free(value) for value in free_batch))

    np.testing.assert_allclose(compiled.impulse, eager.impulse, atol=1.0e-10)
    np.testing.assert_array_equal(compiled.evidence.successful, eager.evidence.successful)
    np.testing.assert_allclose(mapped, eager_mapped, atol=1.0e-10)
