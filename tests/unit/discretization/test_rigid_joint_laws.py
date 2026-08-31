#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.discretization.particle._rigid_joint_laws import (
    accept_rigid_joint_hinge_coordinate,
    CompliantRigidJointLawPlan,
    DissipativeRigidJointLawPlan,
    evaluate_rigid_joint_law_compatibility,
    RigidJointCoordinate,
    RigidJointEffortMotorPlan,
    RigidJointPDServoPlan,
)


def _quaternion_z(angle):
    return jnp.asarray([jnp.cos(0.5 * angle), 0.0, 0.0, jnp.sin(0.5 * angle)])


def _prepared_mechanism():
    body_ids = jnp.arange(100, 104, dtype=jnp.int64)
    particles = phx.discretization.ParticleSetPlan(
        body_ids,
        jnp.ones((4,)),
        ambient_dimension=3,
    ).prepare()
    bodies = phx.discretization.RigidBodySetPlan(
        jnp.zeros((4,), dtype=jnp.int32),
        jnp.stack(tuple(jnp.eye(3) for _ in range(4))),
        fixed_mask=jnp.asarray([True, False, False, False]),
    ).prepare(particles)
    position = jnp.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    reference = bodies.kinematics(
        position,
        jnp.zeros_like(position),
        jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 4),
        jnp.zeros_like(position),
    )
    graph = phx.discretization.RigidJointGraphPlan(
        fixed=phx.discretization.FixedJointSetPlan(
            jnp.asarray([10]), body_ids[:1], body_ids[1:2]
        ),
        ball=phx.discretization.BallJointSetPlan(
            jnp.asarray([20]),
            body_ids[1:2],
            body_ids[2:3],
            jnp.asarray([[1.5, 0.0, 0.0]]),
        ),
        hinge=phx.discretization.HingeJointSetPlan(
            jnp.asarray([30]),
            body_ids[2:3],
            body_ids[3:4],
            jnp.asarray([[2.5, 0.0, 0.0]]),
            jnp.asarray([[0.0, 0.0, 1.0]]),
        ),
    ).prepare(bodies, reference)
    return bodies, reference, graph


def _moved(reference, *, body, angle=0.0, angular_velocity=0.0):
    orientation = reference.orientation.at[body].set(_quaternion_z(angle))
    angular = reference.angular_velocity.at[body, 2].set(angular_velocity)
    return phx.discretization.RigidBodyKinematics(
        reference.position,
        reference.velocity,
        orientation,
        angular,
    )


def test_law_plans_validate_physics_fingerprints_and_hard_compatibility():
    _, _, graph = _prepared_mechanism()
    with pytest.raises(ValueError, match="positive semidefinite"):
        CompliantRigidJointLawPlan(
            [20],
            RigidJointCoordinate.BALL_ORIENTATION,
            jnp.diag(jnp.asarray([1.0, -1.0, 1.0])),
        )
    with pytest.raises(ValueError, match="symmetric"):
        DissipativeRigidJointLawPlan(
            [20],
            RigidJointCoordinate.BALL_ORIENTATION,
            jnp.asarray([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        )
    with pytest.raises(ValueError, match="strictly positive"):
        CompliantRigidJointLawPlan(
            [20],
            RigidJointCoordinate.BALL_ORIENTATION,
            1.0,
            coordinate_scale=0.0,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        RigidJointPDServoPlan(
            [30],
            0.0,
            proportional_gain=-1.0,
            derivative_gain=0.0,
            effort_limit=1.0,
        )
    with pytest.raises(ValueError, match="strictly positive"):
        RigidJointEffortMotorPlan([30], 1.0, effort_limit=0.0)

    first = CompliantRigidJointLawPlan(
        [20],
        RigidJointCoordinate.BALL_ORIENTATION,
        jnp.diag(jnp.asarray([1.0, 2.0, 3.0])),
    )
    same = CompliantRigidJointLawPlan(
        [20],
        RigidJointCoordinate.BALL_ORIENTATION,
        jnp.diag(jnp.asarray([1.0, 2.0, 3.0])),
    )
    changed = CompliantRigidJointLawPlan(
        [20],
        RigidJointCoordinate.BALL_ORIENTATION,
        jnp.diag(jnp.asarray([1.0, 2.0, 4.0])),
    )
    assert first.plan_id == same.plan_id
    assert first.plan_id != changed.plan_id
    assert first.prepare(graph).prepared_id == same.prepare(graph).prepared_id

    hard_fixed = CompliantRigidJointLawPlan(
        [10], RigidJointCoordinate.FIXED_TRANSLATION, 1.0
    )
    hard_ball = DissipativeRigidJointLawPlan([20], RigidJointCoordinate.BALL_ANCHOR, 1.0)
    fixed_evidence = evaluate_rigid_joint_law_compatibility(hard_fixed, graph)
    ball_evidence = evaluate_rigid_joint_law_compatibility(hard_ball, graph)
    assert fixed_evidence.joint_found[0]
    assert fixed_evidence.joint_kind_matches[0]
    assert not fixed_evidence.coordinate_is_free[0]
    assert not fixed_evidence.valid
    assert not ball_evidence.valid
    with pytest.raises(ValueError, match="incompatible joint IDs"):
        hard_fixed.prepare(graph)
    with pytest.raises(ValueError, match="incompatible joint IDs"):
        hard_ball.prepare(graph)


def test_compliance_energy_gradient_and_equal_opposite_wrench():
    _, reference, graph = _prepared_mechanism()
    law = CompliantRigidJointLawPlan(
        [20],
        RigidJointCoordinate.BALL_ORIENTATION,
        jnp.diag(jnp.asarray([0.0, 0.0, 8.0])),
    ).prepare(graph)
    angle = jnp.asarray(0.25)

    def energy(value):
        return law.evaluate(_moved(reference, body=2, angle=value)).stored_energy

    kinematics = _moved(reference, body=2, angle=angle)
    evaluation = law.evaluate(kinematics)
    energy_gradient = jax.grad(energy)(angle)
    total_origin_torque = jnp.sum(
        evaluation.load.torque + jnp.cross(kinematics.position, evaluation.load.force),
        axis=0,
    )

    assert jnp.allclose(evaluation.coordinate[0, 2], angle, atol=1.0e-7)
    assert jnp.allclose(evaluation.stored_energy, 0.5 * 8.0 * angle**2)
    assert jnp.allclose(evaluation.load.torque[2, 2], -energy_gradient, atol=1.0e-6)
    assert jnp.allclose(jnp.sum(evaluation.load.force, axis=0), 0.0, atol=1.0e-7)
    assert jnp.allclose(total_origin_torque, 0.0, atol=1.0e-6)
    assert evaluation.evidence.valid


def test_damping_is_nonnegative_and_removes_mechanical_power():
    _, reference, graph = _prepared_mechanism()
    law = DissipativeRigidJointLawPlan(
        [20],
        RigidJointCoordinate.BALL_ORIENTATION,
        jnp.diag(jnp.asarray([0.0, 0.0, 5.0])),
    ).prepare(graph)
    angular = reference.angular_velocity.at[1, 2].set(-0.25).at[2, 2].set(0.75)
    kinematics = phx.discretization.RigidBodyKinematics(
        reference.position,
        reference.velocity,
        reference.orientation,
        angular,
    )
    evaluation = law.evaluate(kinematics)
    mechanical_power = jnp.sum(evaluation.load.torque * angular)

    assert jnp.allclose(evaluation.rate[0, 2], 1.0)
    assert evaluation.dissipation_rate >= 0.0
    assert jnp.allclose(evaluation.dissipation_rate, 5.0)
    assert jnp.allclose(mechanical_power, -evaluation.dissipation_rate)
    assert jnp.allclose(evaluation.load.torque[1], -evaluation.load.torque[2])
    assert evaluation.evidence.valid


def test_zero_law_is_exact_zero_baseline():
    _, reference, graph = _prepared_mechanism()
    law = CompliantRigidJointLawPlan(
        jnp.empty((0,), dtype=jnp.int64),
        RigidJointCoordinate.BALL_ORIENTATION,
        0.0,
    ).prepare(graph)
    evaluation = law.evaluate(reference)

    assert evaluation.coordinate.shape == (0, 3)
    assert jnp.array_equal(evaluation.load.force, jnp.zeros_like(evaluation.load.force))
    assert jnp.array_equal(evaluation.load.torque, jnp.zeros_like(evaluation.load.torque))
    assert evaluation.stored_energy == 0.0
    assert evaluation.dissipation_rate == 0.0
    assert evaluation.actuator_source_power == 0.0
    assert evaluation.evidence.valid


def test_hinge_effort_has_consistent_sign_work_and_unwrapped_state():
    _, reference, graph = _prepared_mechanism()
    motor = RigidJointEffortMotorPlan([30], 3.0, effort_limit=5.0).prepare(graph)
    state = motor.initialize_state(reference)
    kinematics = _moved(reference, body=3, angle=0.4, angular_velocity=2.0)
    evaluation = motor.evaluate(kinematics, state)
    mechanical_power = jnp.sum(
        evaluation.load.force * kinematics.velocity
        + evaluation.load.torque * kinematics.angular_velocity
    )

    assert jnp.allclose(evaluation.coordinate, jnp.asarray([[0.4]]), atol=1.0e-6)
    assert jnp.allclose(evaluation.rate, jnp.asarray([[2.0]]), atol=1.0e-6)
    assert jnp.allclose(evaluation.load.torque[2, 2], -3.0, atol=1.0e-6)
    assert jnp.allclose(evaluation.load.torque[3, 2], 3.0, atol=1.0e-6)
    assert jnp.allclose(evaluation.actuator_source_power, 6.0, atol=1.0e-6)
    assert jnp.allclose(mechanical_power, evaluation.actuator_source_power, atol=1.0e-6)

    near_branch = motor.evaluate(_moved(reference, body=3, angle=3.0), state)
    crossed = motor.evaluate(
        _moved(reference, body=3, angle=3.2), near_branch.accepted_state
    )
    assert jnp.allclose(
        crossed.candidate_state.unwrapped_coordinate,
        jnp.asarray([[3.2]]),
        atol=1.0e-6,
    )
    rolled_back = accept_rigid_joint_hinge_coordinate(
        near_branch.accepted_state, crossed.candidate_state, False
    )
    assert jnp.array_equal(
        rolled_back.unwrapped_coordinate,
        near_branch.accepted_state.unwrapped_coordinate,
    )

    compiled = eqx.filter_jit(motor.evaluate)(kinematics, state)
    assert jnp.allclose(compiled.load.torque, evaluation.load.torque)
    assert compiled.evidence.valid


def test_pd_servo_saturates_and_reports_source_power():
    _, reference, graph = _prepared_mechanism()
    servo = RigidJointPDServoPlan(
        [30],
        1.0,
        proportional_gain=10.0,
        derivative_gain=0.0,
        effort_limit=2.0,
    ).prepare(graph)
    state = servo.initialize_state(reference)
    kinematics = _moved(reference, body=3, angular_velocity=0.5)
    evaluation = servo.evaluate(kinematics, state)

    assert jnp.allclose(evaluation.effort, jnp.asarray([[2.0]]))
    assert evaluation.saturated[0, 0]
    assert jnp.allclose(evaluation.saturation_margin, jnp.asarray([[-8.0]]))
    assert jnp.allclose(evaluation.actuator_source_power, 1.0)
    assert evaluation.evidence.valid


def test_chart_and_nonfinite_failures_are_explicit_and_rollback_state():
    _, reference, graph = _prepared_mechanism()
    motor = RigidJointEffortMotorPlan([30], 1.0, effort_limit=2.0).prepare(graph)
    motor_state = motor.initialize_state(reference)
    chart_failure = motor.evaluate(_moved(reference, body=3, angle=jnp.pi), motor_state)

    assert chart_failure.chart_margin[0] <= motor.plan.chart_tolerance
    assert not chart_failure.evidence.chart_valid
    assert not chart_failure.successful
    assert not chart_failure.candidate_state.chart_valid[0]
    assert jnp.array_equal(
        chart_failure.accepted_state.unwrapped_coordinate,
        motor_state.unwrapped_coordinate,
    )

    nonfinite_actuator = phx.discretization.RigidBodyKinematics(
        reference.position,
        reference.velocity,
        reference.orientation,
        reference.angular_velocity.at[3, 2].set(jnp.nan),
    )
    actuator_failure = motor.evaluate(nonfinite_actuator, motor_state)
    assert not actuator_failure.evidence.finite
    assert jnp.array_equal(
        actuator_failure.accepted_state.unwrapped_coordinate,
        motor_state.unwrapped_coordinate,
    )

    damper = DissipativeRigidJointLawPlan(
        [20],
        RigidJointCoordinate.BALL_ORIENTATION,
        jnp.eye(3),
    ).prepare(graph)
    nonfinite = phx.discretization.RigidBodyKinematics(
        reference.position,
        reference.velocity,
        reference.orientation,
        reference.angular_velocity.at[2, 2].set(jnp.nan),
    )
    failure = damper.evaluate(nonfinite)

    assert not failure.evidence.finite
    assert not failure.evidence.valid
    assert not failure.successful
