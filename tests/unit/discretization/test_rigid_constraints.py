#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _quaternion_z(angle):
    return jnp.asarray([jnp.cos(0.5 * angle), 0.0, 0.0, jnp.sin(0.5 * angle)])


def _prepared_bodies(count=3, *, fixed_mask=None, dimension=3):
    ids = jnp.arange(100, 100 + count, dtype=jnp.int64)
    particles = phx.discretization.ParticleSetPlan(
        ids,
        jnp.ones((count,)),
        ambient_dimension=dimension,
    ).prepare()
    inertia = (
        jnp.ones((count,))
        if dimension == 2
        else jnp.stack(tuple(jnp.eye(3) for _ in range(count)))
    )
    fixed = (
        jnp.zeros((count,), dtype=bool) if fixed_mask is None else jnp.asarray(fixed_mask)
    )
    bodies = phx.discretization.RigidBodySetPlan(
        jnp.zeros((count,), dtype=jnp.int32),
        inertia,
        fixed_mask=fixed,
    ).prepare(particles)
    return ids, bodies


def _reference(bodies, position):
    position = jnp.asarray(position)
    if bodies.ambient_dimension == 2:
        orientation = jnp.zeros((bodies.capacity, 1))
        angular = jnp.zeros((bodies.capacity, 1))
    else:
        orientation = jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * bodies.capacity)
        angular = jnp.zeros_like(position)
    return bodies.kinematics(
        position,
        jnp.zeros_like(position),
        orientation,
        angular,
    )


def _constant_load(force, torque):
    load = phx.discretization.RigidBodyLoad(jnp.asarray(force), jnp.asarray(torque))

    def evaluate(time, kinematics, args):
        del time, kinematics, args
        return load

    return evaluate


def test_joint_plan_validation_and_static_scope():
    with pytest.raises(ValueError, match="nonzero"):
        phx.discretization.HingeJointSetPlan(
            jnp.asarray([1]),
            jnp.asarray([100]),
            jnp.asarray([101]),
            jnp.zeros((1, 3)),
            jnp.zeros((1, 3)),
        )
    with pytest.raises(ValueError, match="globally unique"):
        phx.discretization.RigidJointGraphPlan(
            fixed=phx.discretization.FixedJointSetPlan(
                jnp.asarray([1]), jnp.asarray([100]), jnp.asarray([101])
            ),
            ball=phx.discretization.BallJointSetPlan(
                jnp.asarray([1]),
                jnp.asarray([100]),
                jnp.asarray([101]),
                jnp.zeros((1, 3)),
            ),
        )

    _, bodies_2d = _prepared_bodies(2, fixed_mask=[True, False], dimension=2)
    reference_2d = _reference(bodies_2d, [[0.0, 0.0], [1.0, 0.0]])
    graph = phx.discretization.RigidJointGraphPlan(
        fixed=phx.discretization.FixedJointSetPlan(
            jnp.asarray([1]), jnp.asarray([100]), jnp.asarray([101])
        )
    )
    with pytest.raises(ValueError, match="three-dimensional"):
        graph.prepare(bodies_2d, reference_2d)

    _, fixed_bodies = _prepared_bodies(2, fixed_mask=[True, True])
    fixed_reference = _reference(fixed_bodies, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="mobile endpoint"):
        graph.prepare(fixed_bodies, fixed_reference)

    _, one_mobile = _prepared_bodies(2, fixed_mask=[True, False])
    one_reference = _reference(one_mobile, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    overconstrained = phx.discretization.RigidJointGraphPlan(
        fixed=phx.discretization.FixedJointSetPlan(
            jnp.asarray([2]), jnp.asarray([100]), jnp.asarray([101])
        ),
        ball=phx.discretization.BallJointSetPlan(
            jnp.asarray([3]),
            jnp.asarray([100]),
            jnp.asarray([101]),
            jnp.asarray([[0.5, 0.0, 0.0]]),
        ),
    )
    with pytest.raises(ValueError, match="rows exceed"):
        overconstrained.prepare(one_mobile, one_reference)


def test_fixed_and_hinge_residuals_are_objective_and_preserve_hinge_spin():
    _, bodies = _prepared_bodies(3, fixed_mask=[True, False, False])
    reference = _reference(
        bodies,
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    )
    graph = phx.discretization.RigidJointGraphPlan(
        fixed=phx.discretization.FixedJointSetPlan(
            jnp.asarray([10]), jnp.asarray([100]), jnp.asarray([101])
        ),
        hinge=phx.discretization.HingeJointSetPlan(
            jnp.asarray([11]),
            jnp.asarray([101]),
            jnp.asarray([102]),
            jnp.asarray([[1.5, 0.0, 0.0]]),
            jnp.asarray([[0.0, 0.0, 1.0]]),
        ),
    ).prepare(bodies, reference)
    assert (
        phx.discretization.rigid_joint_maximum_residual(graph.residuals(reference))
        < 1.0e-12
    )

    angle = jnp.asarray(0.4)
    quaternion = _quaternion_z(angle)
    cosine, sine = jnp.cos(angle), jnp.sin(angle)
    rotation = jnp.asarray([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    translation = jnp.asarray([0.3, -0.2, 0.7])
    transformed_position = reference.position @ rotation.T + translation
    transformed = bodies.kinematics(
        transformed_position,
        reference.velocity,
        jnp.broadcast_to(quaternion, reference.orientation.shape),
        reference.angular_velocity,
    )
    assert (
        phx.discretization.rigid_joint_maximum_residual(graph.residuals(transformed))
        < 1.0e-12
    )

    axial = transformed.orientation.at[2].set(_quaternion_z(angle + jnp.asarray(0.3)))
    axial_state = bodies.kinematics(
        transformed.position,
        transformed.velocity,
        axial,
        transformed.angular_velocity,
    )
    residuals = graph.residuals(axial_state)
    assert jnp.max(jnp.abs(residuals.hinge_axis)) < 1.0e-12
    assert jnp.min(graph.hinge_alignment(axial_state)) > 0.99


def test_empty_graph_matches_unconstrained_rigid_kdk():
    _, bodies = _prepared_bodies(2, fixed_mask=[False, False])
    reference = _reference(bodies, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    force = jnp.asarray([[1.0, 0.0, 0.0], [0.0, -2.0, 0.0]])
    torque = jnp.asarray([[0.0, 0.0, 0.5], [0.0, 0.2, 0.0]])
    load_function = _constant_load(force, torque)
    load = phx.discretization.RigidBodyLoad(force, torque)
    expected = phx.discretization.rigid_body_kick_drift_kick(
        bodies,
        reference,
        load,
        jnp.asarray(0.0),
        jnp.asarray(1.0e-3),
        load_function,
        None,
    )
    dynamics = phx.discretization.RigidConstraintDynamicsPlan(
        phx.discretization.RigidJointGraphPlan()
    ).prepare(
        bodies,
        reference,
        external_load=load_function,
        external_load_id="constant-load",
    )
    state = dynamics.initialize_state(
        reference.position,
        reference.velocity,
        reference.orientation,
        reference.angular_velocity,
    )
    result = dynamics.step(state, jnp.asarray(0.0), jnp.asarray(1.0e-3))
    assert result.successful
    assert jnp.allclose(
        result.accepted_state.kinematics.position, expected.kinematics.position
    )
    assert jnp.allclose(
        result.accepted_state.kinematics.velocity, expected.kinematics.velocity
    )
    assert jnp.allclose(
        result.accepted_state.kinematics.orientation, expected.kinematics.orientation
    )
    assert jnp.allclose(
        result.accepted_state.kinematics.angular_velocity,
        expected.kinematics.angular_velocity,
    )


def test_fixed_joint_projects_pose_and_velocity_globally():
    _, bodies = _prepared_bodies(2, fixed_mask=[True, False])
    reference = _reference(bodies, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    graph = phx.discretization.RigidJointGraphPlan(
        fixed=phx.discretization.FixedJointSetPlan(
            jnp.asarray([20]), jnp.asarray([100]), jnp.asarray([101])
        )
    )
    dynamics = phx.discretization.RigidConstraintDynamicsPlan(graph).prepare(
        bodies, reference
    )
    perturbed_orientation = reference.orientation.at[1].set(_quaternion_z(0.05))
    state = dynamics.initialize_state(
        reference.position.at[1].add(jnp.asarray([0.03, -0.02, 0.01])),
        jnp.asarray([[0.0, 0.0, 0.0], [0.2, -0.1, 0.05]]),
        perturbed_orientation,
        jnp.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]]),
    )
    result = dynamics.step(state, jnp.asarray(0.0), jnp.asarray(1.0e-3))
    assert result.successful
    assert (
        result.evaluation.diagnostics.maximum_position_residual
        <= dynamics.solver.position_tolerance
    )
    assert (
        result.evaluation.diagnostics.maximum_velocity_residual
        <= dynamics.solver.velocity_tolerance
    )
    assert jnp.allclose(
        result.accepted_state.kinematics.position[0], state.kinematics.position[0]
    )
    assert jnp.allclose(
        result.accepted_state.kinematics.orientation[0], state.kinematics.orientation[0]
    )
    assert result.evaluation.diagnostics.quaternion_defect <= 1.0e-12


def test_ball_and_hinge_steps_are_jittable_and_certified():
    _, bodies = _prepared_bodies(3, fixed_mask=[True, False, False])
    reference = _reference(
        bodies,
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    )
    graph = phx.discretization.RigidJointGraphPlan(
        ball=phx.discretization.BallJointSetPlan(
            jnp.asarray([30]),
            jnp.asarray([100]),
            jnp.asarray([101]),
            jnp.asarray([[0.5, 0.0, 0.0]]),
        ),
        hinge=phx.discretization.HingeJointSetPlan(
            jnp.asarray([31]),
            jnp.asarray([101]),
            jnp.asarray([102]),
            jnp.asarray([[1.5, 0.0, 0.0]]),
            jnp.asarray([[0.0, 0.0, 1.0]]),
        ),
    )

    def gravity(time, kinematics, args):
        del time, args
        return phx.discretization.RigidBodyLoad(
            jnp.broadcast_to(jnp.asarray([0.0, -9.81, 0.0]), kinematics.position.shape),
            jnp.zeros_like(kinematics.angular_velocity),
        )

    dynamics = phx.discretization.RigidConstraintDynamicsPlan(graph).prepare(
        bodies,
        reference,
        external_load=gravity,
        external_load_id="gravity",
    )
    state = dynamics.initialize_state(
        reference.position,
        reference.velocity,
        reference.orientation,
        reference.angular_velocity.at[2, 2].set(0.2),
    )
    step = eqx.filter_jit(
        lambda current: dynamics.step(current, jnp.asarray(0.0), jnp.asarray(1.0e-3))
    )
    result = step(state)
    assert result.successful
    assert result.evaluation.diagnostics.maximum_position_residual < 1.0e-8
    assert result.evaluation.diagnostics.maximum_velocity_residual < 1.0e-8
    assert result.accepted_state.kinematics.angular_velocity[2, 2] != 0.0
    assert result.evaluation.diagnostics.velocity_projection_energy_increase < 1.0e-10


def test_invalid_step_and_nonfinite_load_roll_back_atomically():
    _, bodies = _prepared_bodies(2, fixed_mask=[True, False])
    reference = _reference(bodies, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    graph = phx.discretization.RigidJointGraphPlan()

    def bad_load(time, kinematics, args):
        del time, args
        return phx.discretization.RigidBodyLoad(
            jnp.full_like(kinematics.position, jnp.nan),
            jnp.zeros_like(kinematics.angular_velocity),
        )

    dynamics = phx.discretization.RigidConstraintDynamicsPlan(graph).prepare(
        bodies,
        reference,
        external_load=bad_load,
        external_load_id="nonfinite-load",
    )
    state = dynamics.initialize_state(
        reference.position,
        reference.velocity,
        reference.orientation,
        reference.angular_velocity,
    )
    result = dynamics.step(state, jnp.asarray(0.0), jnp.asarray(1.0e-3))
    assert not result.successful
    assert jnp.allclose(
        result.accepted_state.kinematics.position, state.kinematics.position
    )
    assert int(result.rejection_reasons) & int(
        phx.discretization.RigidConstraintRejectionReason.INITIAL_LOAD
    )

    valid_dynamics = phx.discretization.RigidConstraintDynamicsPlan(graph).prepare(
        bodies, reference
    )
    invalid_step = valid_dynamics.step(state, jnp.asarray(0.0), jnp.asarray(0.0))
    assert not invalid_step.successful
    assert jnp.allclose(
        invalid_step.accepted_state.kinematics.position, state.kinematics.position
    )
    assert int(invalid_step.rejection_reasons) & int(
        phx.discretization.RigidConstraintRejectionReason.INVALID_STEP
    )

    invalid_orientation_state = valid_dynamics.initialize_state(
        reference.position,
        reference.velocity,
        reference.orientation.at[1].set(jnp.zeros((4,))),
        reference.angular_velocity,
    )
    invalid_orientation = valid_dynamics.step(
        invalid_orientation_state,
        jnp.asarray(0.0),
        jnp.asarray(1.0e-3),
    )
    assert not invalid_orientation.successful
    assert int(invalid_orientation.rejection_reasons) & int(
        phx.discretization.RigidConstraintRejectionReason.INVALID_STATE
    )


def test_successful_constraint_step_has_implicit_load_derivative():
    _, bodies = _prepared_bodies(2, fixed_mask=[True, False])
    reference = _reference(bodies, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    graph = phx.discretization.RigidJointGraphPlan(
        ball=phx.discretization.BallJointSetPlan(
            jnp.asarray([40]),
            jnp.asarray([100]),
            jnp.asarray([101]),
            jnp.asarray([[0.5, 0.0, 0.0]]),
        )
    )

    def parameterized_load(time, kinematics, vertical_force):
        del time
        return phx.discretization.RigidBodyLoad(
            jnp.asarray([[0.0, 0.0, 0.0], [0.0, vertical_force, 0.0]]),
            jnp.zeros_like(kinematics.angular_velocity),
        )

    dynamics = phx.discretization.RigidConstraintDynamicsPlan(graph).prepare(
        bodies,
        reference,
        external_load=parameterized_load,
        external_load_id="vertical-force",
    )
    state = dynamics.initialize_state(
        reference.position,
        reference.velocity,
        reference.orientation,
        reference.angular_velocity,
    )

    def observable(vertical_force):
        result = dynamics.step(
            state,
            jnp.asarray(0.0),
            jnp.asarray(1.0e-3),
            vertical_force,
        )
        return result.candidate_state.kinematics.position[1, 1]

    value = observable(jnp.asarray(-1.0))
    derivative = jax.grad(observable)(jnp.asarray(-1.0))
    epsilon = jnp.asarray(1.0e-4)
    finite_difference = (
        observable(jnp.asarray(-1.0) + epsilon) - observable(jnp.asarray(-1.0) - epsilon)
    ) / (2.0 * epsilon)
    assert jnp.isfinite(value)
    assert derivative != 0.0
    assert jnp.allclose(derivative, finite_difference, rtol=1.0e-5, atol=1.0e-10)


def test_redundant_joint_rows_fail_rank_qualification():
    _, bodies = _prepared_bodies(3, fixed_mask=[True, False, False])
    reference = _reference(
        bodies,
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    )
    graph = phx.discretization.RigidJointGraphPlan(
        ball=phx.discretization.BallJointSetPlan(
            jnp.asarray([50, 51]),
            jnp.asarray([100, 100]),
            jnp.asarray([101, 101]),
            jnp.asarray([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        )
    )
    dynamics = phx.discretization.RigidConstraintDynamicsPlan(graph).prepare(
        bodies, reference
    )
    state = dynamics.initialize_state(
        reference.position,
        reference.velocity,
        reference.orientation,
        reference.angular_velocity,
    )
    result = dynamics.step(state, jnp.asarray(0.0), jnp.asarray(1.0e-3))
    assert not result.successful
    assert result.evaluation.diagnostics.constraint_rank < graph.constraint_count
    assert int(result.rejection_reasons) & int(
        phx.discretization.RigidConstraintRejectionReason.RANK_OR_CONDITION
    )
    assert jnp.allclose(
        result.accepted_state.kinematics.position, state.kinematics.position
    )
