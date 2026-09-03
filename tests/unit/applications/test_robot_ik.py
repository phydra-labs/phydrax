#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications.robotics._ik import (
    FrameInverseKinematicsPlan,
    FrameOrientationTask,
    FramePoseTask,
    FramePositionTask,
    InverseKinematicsStatus,
)
from phydrax.discretization.particle._reduced_articulation import (
    ReducedArticulationPlan,
)


def _quaternion_z(angle):
    return jnp.asarray(
        (jnp.cos(0.5 * angle), 0.0, 0.0, jnp.sin(0.5 * angle))
    )


def _one_joint_articulation(kind):
    body_ids = jnp.asarray((100, 101), dtype=jnp.int64)
    particles = phx.discretization.ParticleSetPlan(
        body_ids,
        jnp.ones((2,)),
        ambient_dimension=3,
    ).prepare()
    bodies = phx.discretization.RigidBodySetPlan(
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.broadcast_to(jnp.eye(3), (2, 3, 3)),
        fixed_mask=jnp.asarray((True, False)),
    ).prepare(particles)
    reference = bodies.kinematics(
        jnp.zeros((2, 3)),
        jnp.zeros((2, 3)),
        jnp.asarray(((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))),
        jnp.zeros((2, 3)),
    )
    joint_ids = jnp.asarray((200,), dtype=jnp.int64)
    if kind == "hinge":
        joint = phx.discretization.HingeJointSetPlan(
            joint_ids,
            body_ids[:1],
            body_ids[1:],
            jnp.zeros((1, 3)),
            jnp.asarray(((0.0, 0.0, 1.0),)),
        )
        graph_plan = phx.discretization.RigidJointGraphPlan(hinge=joint)
    elif kind == "prismatic":
        joint = phx.discretization.PrismaticJointSetPlan(
            joint_ids,
            body_ids[:1],
            body_ids[1:],
            jnp.zeros((1, 3)),
            jnp.asarray(((1.0, 0.0, 0.0),)),
        )
        graph_plan = phx.discretization.RigidJointGraphPlan(prismatic=joint)
    else:
        raise ValueError("Unknown test joint kind.")
    graph = graph_plan.prepare(bodies, reference)
    articulation = ReducedArticulationPlan(
        100,
        joint_ids,
        body_ids[:1],
        body_ids[1:],
    ).prepare(graph, reference)
    return articulation


def _termination():
    return phx.optim.OptimizationTermination(
        absolute_optimality=1.0e-9,
        relative_optimality=0.0,
        absolute_step=1.0e-12,
        relative_step=0.0,
        maximum_steps=64,
    )


def _tip_transform():
    return jnp.eye(4).at[0, 3].set(1.0)


@pytest.mark.parametrize("task_kind", ("position", "orientation", "pose"))
def test_one_hinge_reaches_analytic_frame_targets(task_kind):
    articulation = _one_joint_articulation("hinge")
    angle = jnp.asarray(0.6)
    target_position = jnp.asarray((jnp.cos(angle), jnp.sin(angle), 0.0))
    target_orientation = _quaternion_z(angle)
    if task_kind == "position":
        task = FramePositionTask(
            101,
            "tool",
            target_position,
            local_transform=_tip_transform(),
            tolerance=2.0e-6,
            task_id="tool-position",
        )
    elif task_kind == "orientation":
        task = FrameOrientationTask(
            101,
            "tool",
            target_orientation,
            local_transform=_tip_transform(),
            tolerance=2.0e-6,
            task_id="tool-orientation",
        )
    else:
        task = FramePoseTask(
            101,
            "tool",
            target_position,
            target_orientation,
            local_transform=_tip_transform(),
            tolerance=2.0e-6,
            task_id="tool-pose",
        )
    result = FrameInverseKinematicsPlan(articulation, (task,)).solve(
        jnp.zeros((1,)),
        method=phx.optim.LevenbergMarquardt(),
        termination=_termination(),
    )

    assert bool(result.successful)
    assert int(result.status) == int(InverseKinematicsStatus.SUCCESS)
    np.testing.assert_allclose(result.configuration, jnp.asarray((angle,)), atol=2.0e-6)
    assert bool(result.kinematics.finite)
    assert bool(result.task_residuals[0].feasible)


def test_prismatic_target_and_joint_bounds_are_respected():
    articulation = _one_joint_articulation("prismatic")
    reachable = FramePositionTask(
        101,
        "slider",
        jnp.asarray((0.4, 0.0, 0.0)),
        tolerance=2.0e-6,
        task_id="slider-reachable",
    )
    reachable_result = FrameInverseKinematicsPlan(articulation, (reachable,)).solve(
        jnp.zeros((1,)),
        method=phx.optim.GaussNewton(),
        termination=_termination(),
    )
    assert bool(reachable_result.successful)
    np.testing.assert_allclose(
        reachable_result.configuration, jnp.asarray((0.4,)), atol=2.0e-6
    )

    beyond_limit = FramePositionTask(
        101,
        "slider",
        jnp.asarray((0.8, 0.0, 0.0)),
        tolerance=2.0e-6,
        task_id="slider-beyond-limit",
    )
    bounds = phx.optim.Bounds(jnp.asarray((-0.1,)), jnp.asarray((0.3,)))
    bounded_result = FrameInverseKinematicsPlan(
        articulation, (beyond_limit,)
    ).solve(
        jnp.zeros((1,)),
        method=phx.optim.BoundedLevenbergMarquardt(),
        termination=_termination(),
        joint_bounds=bounds,
    )

    assert bool(bounded_result.feasibility.joint_bounds_satisfied)
    assert float(bounded_result.configuration[0]) <= 0.3 + 1.0e-7
    assert not bool(bounded_result.successful)
    assert int(bounded_result.status) == int(InverseKinematicsStatus.INFEASIBLE)
    assert float(bounded_result.task_residuals[0].bound_violation) > 0.4


def test_conflicting_tasks_report_residual_without_false_success():
    articulation = _one_joint_articulation("prismatic")
    left = FramePositionTask(
        101,
        "slider",
        jnp.asarray((0.25, 0.0, 0.0)),
        task_id="left-target",
    )
    right = FramePositionTask(
        101,
        "slider",
        jnp.asarray((0.75, 0.0, 0.0)),
        task_id="right-target",
    )
    result = FrameInverseKinematicsPlan(articulation, (left, right)).solve(
        jnp.zeros((1,)),
        method=phx.optim.GaussNewton(),
        termination=_termination(),
    )

    assert bool(result.optimizer.successful)
    assert not bool(result.successful)
    assert int(result.status) == int(InverseKinematicsStatus.INFEASIBLE)
    assert not bool(result.feasibility.task_bounds_satisfied)
    residuals = jnp.stack(tuple(value.residual[0] for value in result.task_residuals))
    np.testing.assert_allclose(jnp.abs(residuals), jnp.asarray((0.25, 0.25)), atol=2.0e-6)


def test_pose_residual_is_invariant_to_target_quaternion_sign():
    articulation = _one_joint_articulation("hinge")
    target_angle = jnp.asarray(0.7)
    target_position = jnp.asarray(
        (jnp.cos(target_angle), jnp.sin(target_angle), 0.0)
    )
    quaternion = _quaternion_z(target_angle)
    positive = FramePoseTask(
        101,
        "tool",
        target_position,
        quaternion,
        local_transform=_tip_transform(),
        task_id="positive-quaternion",
    )
    negative = FramePoseTask(
        101,
        "tool",
        target_position,
        -quaternion,
        local_transform=_tip_transform(),
        task_id="negative-quaternion",
    )
    positive_plan = FrameInverseKinematicsPlan(articulation, (positive,))
    negative_plan = FrameInverseKinematicsPlan(articulation, (negative,))
    configuration = jnp.asarray((0.2,))

    np.testing.assert_allclose(
        positive_plan.residual(configuration),
        negative_plan.residual(configuration),
        atol=2.0e-7,
    )


def test_pi_rotation_chart_failure_is_typed_and_fails_closed():
    articulation = _one_joint_articulation("hinge")
    task = FrameOrientationTask(
        101,
        "tool",
        _quaternion_z(jnp.asarray(jnp.pi)),
        task_id="pi-target",
    )
    result = FrameInverseKinematicsPlan(articulation, (task,)).solve(
        jnp.zeros((1,)),
        method=phx.optim.GaussNewton(),
        termination=_termination(),
    )

    assert bool(result.optimizer.successful)
    assert not bool(result.chart.valid)
    assert not bool(result.successful)
    assert int(result.status) == int(InverseKinematicsStatus.CHART_INVALID)


def test_posture_residual_uses_articulation_configuration_difference():
    articulation = _one_joint_articulation("hinge")
    task = FrameOrientationTask(
        101,
        "tool",
        _quaternion_z(jnp.asarray(0.1)),
        task_id="tool-orientation",
    )
    plan = FrameInverseKinematicsPlan(
        articulation,
        (task,),
        posture_configuration=jnp.asarray((0.3,)),
        posture_weight=jnp.asarray((2.0,)),
    )

    evaluation = plan.evaluate(jnp.asarray((0.1,)))
    np.testing.assert_allclose(evaluation.posture_residual, jnp.asarray((-0.4,)))
    assert evaluation.residual.shape == (4,)


def test_residual_jit_and_implicit_local_solution_sensitivity():
    articulation = _one_joint_articulation("prismatic")
    initial = jnp.zeros((1,))
    termination = _termination()
    method = phx.optim.GaussNewton()

    def make_plan(target):
        return FrameInverseKinematicsPlan(
            articulation,
            (
                FramePositionTask(
                    101,
                    "slider",
                    target,
                    task_id="differentiable-target",
                ),
            ),
        )

    target = jnp.asarray((0.35, 0.0, 0.0))
    plan = make_plan(target)
    compiled_residual = eqx.filter_jit(plan.residual)(jnp.asarray((0.1,)))
    np.testing.assert_allclose(
        compiled_residual, jnp.asarray((-0.25, 0.0, 0.0)), atol=2.0e-7
    )

    sensitivity = jax.jacfwd(
        lambda value: make_plan(value).implicit_solution(
            initial,
            method=method,
            termination=termination,
        )
    )(target)
    np.testing.assert_allclose(
        sensitivity,
        jnp.asarray(((1.0, 0.0, 0.0),)),
        atol=2.0e-6,
    )
