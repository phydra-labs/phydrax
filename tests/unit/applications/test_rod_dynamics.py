from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.applications.solid_mechanics._rod_dynamics import (
    evaluate_endpoint_attachment,
    evaluate_rod,
    prepare_rod,
    prepare_rod_dynamics,
    rod_potential_energy,
    RodDynamicsPlan,
    RodEndpointAttachment,
    RodPlan,
    RodState,
)


def _planar_rod(*, inextensible: bool = False):
    plan = RodPlan(
        jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))),
        jnp.broadcast_to(jnp.eye(2), (2, 2, 2)),
        jnp.asarray((1.0, 1.5, 1.0)),
        jnp.asarray((0.2, 0.3)),
        jnp.broadcast_to(jnp.diag(jnp.asarray((100.0, 30.0))), (2, 2, 2)),
        jnp.asarray((((5.0,),),)),
        inextensible=inextensible,
    )
    return prepare_rod(plan)


def _spatial_rod():
    plan = RodPlan(
        jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
        jnp.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 2.0))),
        jnp.broadcast_to(jnp.eye(3), (2, 3, 3)),
        jnp.asarray((1.0, 1.0, 1.0)),
        jnp.broadcast_to(jnp.diag(jnp.asarray((0.2, 0.3, 0.4))), (2, 3, 3)),
        jnp.broadcast_to(jnp.diag(jnp.asarray((20.0, 20.0, 100.0))), (2, 3, 3)),
        jnp.asarray((((3.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 6.0)),)),
    )
    return prepare_rod(plan)


def _one_segment_inextensible_rod():
    return prepare_rod(
        RodPlan(
            jnp.asarray(((0, 1),), dtype=jnp.int32),
            jnp.asarray(((0.0, 0.0), (1.0, 0.0))),
            jnp.eye(2)[None, ...],
            jnp.ones((2,)),
            jnp.ones((1,)),
            jnp.eye(2)[None, ...],
            jnp.zeros((0, 1, 1)),
            inextensible=True,
        )
    )


def test_rod_exposes_shared_collision_surface_map():
    rod = _planar_rod()
    surface = rod.collision_surface(physical_radius=0.01)
    scene = phx.discretization.PreparedCollisionScene((surface,))
    state = jnp.zeros((rod.plan.node_count, rod.plan.dimension))
    epoch = phx.discretization.SweepAndPruneContactSearchPlan(
        edge_vertex_capacity=8,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
    ).build(scene, scene.positions(state))

    assert surface.plan.edge_count == rod.plan.segment_count
    assert surface.plan.physical_radius == pytest.approx(0.01)
    assert jnp.array_equal(scene.positions(state), rod.plan.rest_positions)
    assert bool(epoch.successful)


def test_rest_state_has_zero_energy_and_planar_rigid_motion_is_objective():
    rod = _planar_rod()
    rest = rod.initialize_state()
    rest_evaluation = evaluate_rod(rod, rest)
    angle = jnp.asarray(0.63)
    rotation = jnp.asarray(
        ((jnp.cos(angle), -jnp.sin(angle)), (jnp.sin(angle), jnp.cos(angle)))
    )
    moved = RodState(
        rest.positions @ rotation.T + jnp.asarray((4.0, -2.0)),
        rest.velocities,
        rest.orientations + angle,
        rest.angular_velocities,
    )
    moved_evaluation = evaluate_rod(rod, moved)

    assert rest_evaluation.valid
    assert rest_evaluation.potential_energy == pytest.approx(0.0, abs=2.0e-6)
    assert moved_evaluation.valid
    assert moved_evaluation.potential_energy == pytest.approx(
        rest_evaluation.potential_energy, abs=2.0e-5
    )
    assert jnp.allclose(moved_evaluation.stretch_shear_strain, 0.0, atol=2.0e-6)
    assert jnp.allclose(moved_evaluation.bend_twist_strain, 0.0, atol=2.0e-6)


def test_spatial_rest_state_has_zero_energy_and_stable_preparation_identity():
    first = _spatial_rod()
    second = _spatial_rod()
    rest = first.initialize_state()
    evaluation = evaluate_rod(first, rest)
    angle = jnp.asarray(0.41)
    quaternion = jnp.asarray((jnp.cos(0.5 * angle), jnp.sin(0.5 * angle), 0.0, 0.0))
    rotation = jnp.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, jnp.cos(angle), -jnp.sin(angle)),
            (0.0, jnp.sin(angle), jnp.cos(angle)),
        )
    )
    moved = RodState(
        rest.positions @ rotation.T + jnp.asarray((2.0, -3.0, 1.0)),
        rest.velocities,
        jnp.broadcast_to(quaternion, rest.orientations.shape),
        rest.angular_velocities,
    )
    moved_evaluation = evaluate_rod(first, moved)

    assert first.plan.plan_id == second.plan.plan_id
    assert first.prepared_id == second.prepared_id
    assert evaluation.valid
    assert moved_evaluation.valid
    assert evaluation.potential_energy == pytest.approx(0.0, abs=2.0e-6)
    assert moved_evaluation.potential_energy == pytest.approx(0.0, abs=2.0e-5)
    assert jnp.allclose(evaluation.internal_forces, 0.0, atol=2.0e-6)
    assert jnp.allclose(evaluation.internal_moments, 0.0, atol=2.0e-6)


def test_energy_gradient_is_dual_to_reported_force_and_moment():
    rod = _planar_rod()
    rest = rod.initialize_state()
    state = RodState(
        rest.positions.at[1, 1].set(0.17),
        rest.velocities,
        rest.orientations + jnp.asarray((0.04, 0.19)),
        rest.angular_velocities,
    )
    position_direction = jnp.asarray(((0.1, -0.2), (-0.3, 0.4), (0.2, -0.1)))
    angle_direction = jnp.asarray((0.35, -0.25))
    _, directional_derivative = jax.jvp(
        lambda positions, angles: rod_potential_energy(rod, positions, angles),
        (state.positions, state.orientations),
        (position_direction, angle_direction),
    )
    evaluation = evaluate_rod(rod, state)
    negative_virtual_work = -(
        jnp.sum(evaluation.internal_forces * position_direction)
        + jnp.sum(evaluation.internal_moments * angle_direction)
    )

    assert evaluation.valid
    assert directional_derivative == pytest.approx(
        negative_virtual_work, rel=2.0e-5, abs=2.0e-5
    )
    assert evaluation.resultant_force_residual == pytest.approx(0.0, abs=2.0e-5)


def test_planar_elastica_bend_produces_equal_opposite_restoring_moments():
    rod = _planar_rod()
    rest = rod.initialize_state()
    angle = jnp.asarray(0.2)
    curved_positions = jnp.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0 + jnp.cos(angle), jnp.sin(angle)))
    )
    state = RodState(
        curved_positions,
        rest.velocities,
        jnp.asarray((0.0, angle)),
        rest.angular_velocities,
    )
    evaluation = evaluate_rod(rod, state)

    assert evaluation.valid
    assert evaluation.bend_twist_strain[0, 0] == pytest.approx(angle)
    assert evaluation.internal_moments[0] == pytest.approx(
        -evaluation.internal_moments[1], rel=2.0e-5
    )
    assert evaluation.internal_moments[1] < 0.0
    assert evaluation.potential_energy > 0.0


def test_spatial_twist_is_pure_torsion_with_restoring_material_moment():
    rod = _spatial_rod()
    rest = rod.initialize_state()
    twist = jnp.asarray(0.25)
    twist_quaternion = jnp.asarray((jnp.cos(0.5 * twist), 0.0, 0.0, jnp.sin(0.5 * twist)))
    state = RodState(
        rest.positions,
        rest.velocities,
        rest.orientations.at[1].set(twist_quaternion),
        rest.angular_velocities,
    )
    evaluation = evaluate_rod(rod, state)

    assert evaluation.valid
    assert jnp.allclose(evaluation.stretch_shear_strain, 0.0, atol=2.0e-6)
    assert jnp.allclose(evaluation.bend_twist_strain[0, :2], 0.0, atol=2.0e-6)
    assert evaluation.bend_twist_strain[0, 2] == pytest.approx(twist, rel=2.0e-5)
    assert evaluation.internal_moments[0, 2] == pytest.approx(
        -evaluation.internal_moments[1, 2], rel=2.0e-5
    )
    assert evaluation.internal_moments[1, 2] < 0.0


def test_inextensible_symplectic_step_projects_length_and_axial_velocity():
    rod = _one_segment_inextensible_rod()
    rest = rod.initialize_state()
    state = RodState(
        rest.positions,
        jnp.asarray(((-1.0, 0.0), (1.0, 0.0))),
        rest.orientations,
        rest.angular_velocities,
    )
    dynamics = prepare_rod_dynamics(
        rod,
        RodDynamicsPlan(
            maximum_time_step=0.2,
            maximum_nodal_displacement=1.0,
            projection_iterations=2,
        ),
    )
    result = dynamics.step(state, jnp.asarray(0.1))
    accepted = result.accepted_state
    segment = accepted.positions[1] - accepted.positions[0]
    direction = segment / jnp.linalg.norm(segment)
    relative_axial_velocity = jnp.dot(
        accepted.velocities[1] - accepted.velocities[0], direction
    )

    assert result.successful
    assert jnp.linalg.norm(segment) == pytest.approx(1.0, abs=2.0e-6)
    assert relative_axial_velocity == pytest.approx(0.0, abs=2.0e-6)
    assert result.accepted_evaluation.inextensibility_valid


def test_endpoint_attachment_reports_action_reaction_about_body_center():
    rod = _planar_rod()
    rest = rod.initialize_state()
    angle = jnp.asarray(0.18)
    state = RodState(
        rest.positions.at[2].set(
            jnp.asarray((1.0, 0.0)) + 1.08 * jnp.asarray((jnp.cos(angle), jnp.sin(angle)))
        ),
        rest.velocities,
        jnp.asarray((0.0, angle)),
        rest.angular_velocities,
    )
    evaluation = evaluate_rod(rod, state)
    attachment = RodEndpointAttachment("end", 7, jnp.asarray((0.2, -0.1)))
    response = evaluate_endpoint_attachment(
        rod, state, evaluation, attachment, jnp.asarray(0.31)
    )

    assert response.valid
    assert jnp.linalg.norm(response.force_on_rigid) > 0.0
    assert jnp.allclose(response.force_on_rod, -response.force_on_rigid, atol=2.0e-6)
    assert jnp.allclose(response.force_balance, 0.0, atol=2.0e-6)
    assert jnp.allclose(response.moment_balance, 0.0, atol=2.0e-6)


def test_plan_rejects_invalid_ids_rest_lengths_frames_mass_inertia_and_stiffness():
    segments = jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32)
    positions = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)))
    frames = jnp.broadcast_to(jnp.eye(2), (2, 2, 2))
    masses = jnp.ones((3,))
    inertias = jnp.ones((2,))
    stretch = jnp.broadcast_to(jnp.eye(2), (2, 2, 2))
    bend = jnp.ones((1, 1, 1))

    with pytest.raises(ValueError, match="ordered"):
        RodPlan(
            jnp.asarray(((0, 1), (0, 2)), dtype=jnp.int32),
            positions,
            frames,
            masses,
            inertias,
            stretch,
            bend,
        )
    with pytest.raises(ValueError, match="rest_lengths"):
        RodPlan(
            segments,
            positions,
            frames,
            masses,
            inertias,
            stretch,
            bend,
            rest_lengths=jnp.asarray((1.0, 1.1)),
        )
    with pytest.raises(ValueError, match="orthonormal"):
        RodPlan(
            segments,
            positions,
            frames.at[0, 0, 0].set(2.0),
            masses,
            inertias,
            stretch,
            bend,
        )
    with pytest.raises(ValueError, match="node_masses"):
        RodPlan(
            segments,
            positions,
            frames,
            masses.at[1].set(0.0),
            inertias,
            stretch,
            bend,
        )
    with pytest.raises(ValueError, match="segment_inertias"):
        RodPlan(
            segments,
            positions,
            frames,
            masses,
            inertias.at[0].set(0.0),
            stretch,
            bend,
        )
    with pytest.raises(ValueError, match="positive"):
        RodPlan(
            segments,
            positions,
            frames,
            masses,
            inertias,
            stretch.at[0, 0, 0].set(-1.0),
            bend,
        )


def test_chart_and_current_zero_length_fail_with_finite_evidence():
    rod = _planar_rod()
    rest = rod.initialize_state()
    collapsed = RodState(
        rest.positions.at[1].set(rest.positions[0]),
        rest.velocities,
        rest.orientations,
        rest.angular_velocities,
    )
    collapsed_evaluation = evaluate_rod(rod, collapsed)
    chart_boundary = RodState(
        rest.positions,
        rest.velocities,
        jnp.asarray((0.0, jnp.pi)),
        rest.angular_velocities,
    )
    chart_evaluation = evaluate_rod(rod, chart_boundary)

    assert collapsed_evaluation.finite
    assert not collapsed_evaluation.nondegenerate
    assert not collapsed_evaluation.valid
    assert chart_evaluation.finite
    assert not chart_evaluation.chart_valid
    assert not chart_evaluation.valid

    with pytest.raises(ValueError, match="rest segment"):
        RodPlan(
            jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
            jnp.asarray(((0.0, 0.0), (0.0, 0.0), (1.0, 0.0))),
            jnp.broadcast_to(jnp.eye(2), (2, 2, 2)),
            jnp.ones((3,)),
            jnp.ones((2,)),
            jnp.broadcast_to(jnp.eye(2), (2, 2, 2)),
            jnp.ones((1, 1, 1)),
        )


def test_evaluation_and_step_are_jittable_and_rejected_candidate_rolls_back():
    rod = _spatial_rod()
    state = rod.initialize_state()
    compiled_evaluate = jax.jit(lambda current: evaluate_rod(rod, current))
    evaluation = compiled_evaluate(state)
    dynamics = prepare_rod_dynamics(
        rod,
        RodDynamicsPlan(
            maximum_time_step=0.05,
            maximum_nodal_displacement=0.5,
            maximum_angular_increment=0.5,
        ),
    )
    compiled_step = jax.jit(lambda current, dt: dynamics.step(current, dt))
    result = compiled_step(state, jnp.asarray(0.2))

    assert evaluation.valid
    assert not result.successful
    assert not result.evidence.time_step_valid
    assert jnp.array_equal(result.accepted_state.positions, state.positions)
    assert jnp.array_equal(result.accepted_state.velocities, state.velocities)
    assert jnp.array_equal(result.accepted_state.orientations, state.orientations)
    assert result.accepted_evaluation.valid
