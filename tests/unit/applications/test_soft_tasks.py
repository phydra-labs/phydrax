from __future__ import annotations

from math import prod

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications.robotics._soft_tasks import (
    ContinuumDifferentialIKPlan,
    ContinuumIKStatus,
    ContinuumInverseKinematicsPlan,
    ContinuumOrientationTask,
    ContinuumPoseTask,
    ContinuumPositionTask,
    ContinuumPostureTask,
    ContinuumShapeTask,
    SmoothReducedRodTrajectoryPlan,
)
from phydrax.applications.solid_mechanics._rod_dynamics import prepare_rod, RodPlan
from phydrax.applications.solid_mechanics._rod_plant import prepare_reduced_rod_plant
from phydrax.applications.solid_mechanics._rod_reconstruction import (
    prepare_rod_reconstruction,
    RodFrameQueryPlan,
    RodReconstructionPlan,
)
from phydrax.applications.solid_mechanics._rod_reduced_basis import RodStrainBasisPlan
from phydrax.applications.solid_mechanics._rod_reduced_dynamics import (
    prepare_reduced_rod_dynamics,
    ReducedRodDenseCholeskyPlan,
)
from phydrax.applications.solid_mechanics._rod_reduced_integrators import (
    ReducedRodSemiImplicitVelocityEuler,
)
from phydrax.applications.solid_mechanics._rod_reduction import (
    prepare_reduced_rod,
    ReducedRodPlan,
)
from phydrax.dynamics import PlantStateVectorCodec, StateLayout, TimeGrid
from phydrax.linalg import ArraySpace
from phydrax.metrix import QuaternionPoseStateGeometry


_POSE = QuaternionPoseStateGeometry(convention="body", tolerance=1.0e-9)


def _spatial_reconstruction(*, queries=(0.0, 0.25, 0.5, 0.75, 1.0)):
    dtype = jnp.float64
    segment_count = 6
    arc = jnp.linspace(0.0, 1.0, segment_count + 1, dtype=dtype)
    positions = jnp.stack((jnp.zeros_like(arc), jnp.zeros_like(arc), arc), axis=-1)
    segment_ids = jnp.stack(
        (
            jnp.arange(segment_count, dtype=jnp.int32),
            jnp.arange(1, segment_count + 1, dtype=jnp.int32),
        ),
        axis=-1,
    )
    rod = prepare_rod(
        RodPlan(
            segment_ids,
            positions,
            jnp.broadcast_to(jnp.eye(3, dtype=dtype), (segment_count, 3, 3)),
            jnp.ones((segment_count + 1,), dtype=dtype),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((0.2, 0.3, 0.4), dtype=dtype)),
                (segment_count, 3, 3),
            ),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((20.0, 25.0, 100.0), dtype=dtype)),
                (segment_count, 3, 3),
            ),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((4.0, 5.0, 7.0), dtype=dtype)),
                (segment_count - 1, 3, 3),
            ),
        )
    )
    basis = RodStrainBasisPlan.piecewise_constant(
        jnp.asarray((0.0, 1.0), dtype=dtype),
        dimension=3,
        component_scales=jnp.ones((6,), dtype=dtype),
    )
    reduction = prepare_reduced_rod(
        rod,
        ReducedRodPlan(
            basis,
            base_policy="fixed",
            fixed_base_position=jnp.asarray((0.2, -0.1, 0.3), dtype=dtype),
            fixed_base_orientation=jnp.asarray((1.0, 0.0, 0.0, 0.0), dtype=dtype),
        ),
    )
    reconstruction = prepare_rod_reconstruction(
        reduction,
        RodReconstructionPlan(
            RodFrameQueryPlan(jnp.asarray(queries, dtype=dtype)),
            method="pcs",
        ),
    )
    return reduction, reconstruction


def _termination(maximum_steps=80):
    return phx.optim.OptimizationTermination(
        absolute_optimality=1.0e-9,
        relative_optimality=0.0,
        absolute_step=1.0e-11,
        relative_step=0.0,
        maximum_steps=maximum_steps,
    )


def _constant_curvature_target(reconstruction):
    return jnp.asarray((0.05, -0.03, 0.08, 0.14, -0.09, 0.21), dtype=jnp.float64)


def test_all_continuum_task_contracts_match_an_analytic_constant_curvature_target():
    reduction, reconstruction = _spatial_reconstruction()
    target_coefficients = _constant_curvature_target(reconstruction)
    target_poses = reconstruction.pose(target_coefficients)
    analytic_endpoint = _POSE.retract(
        jnp.concatenate((reduction.base_orientation, reduction.base_position)),
        target_coefficients
        + jnp.asarray((0.0, 0.0, 1.0, 0.0, 0.0, 0.0), dtype=jnp.float64),
    )
    np.testing.assert_allclose(target_poses[-1], analytic_endpoint, atol=2.0e-11)

    tasks = (
        ContinuumPositionTask(
            reconstruction, 1.0, target_poses[-1, 4:], tolerance=2.0e-8
        ),
        ContinuumOrientationTask(
            reconstruction, 1.0, target_poses[-1, :4], tolerance=2.0e-8
        ),
        ContinuumPoseTask(
            reconstruction,
            0.75,
            target_poses[-2, 4:],
            target_poses[-2, :4],
            tolerance=2.0e-8,
        ),
        ContinuumShapeTask(reconstruction, target_poses[:, 4:], tolerance=2.0e-8),
        ContinuumPostureTask(reconstruction, target_coefficients, tolerance=2.0e-8),
    )
    plan = ContinuumInverseKinematicsPlan(reconstruction, tasks)
    evaluation = plan.evaluate(target_coefficients)

    assert evaluation.reconstruction_valid
    assert evaluation.chart_valid
    assert evaluation.feasible
    np.testing.assert_allclose(
        evaluation.residual, jnp.zeros_like(evaluation.residual), atol=2.0e-10
    )
    assert len({task.task_id for task in tasks}) == len(tasks)
    assert all(
        task.reconstruction_id == reconstruction.reconstruction_id for task in tasks
    )


def test_continuum_pose_ik_solves_local_nls_and_retains_native_accepted_evidence():
    _reduction, reconstruction = _spatial_reconstruction(queries=(0.0, 1.0))
    target_coefficients = _constant_curvature_target(reconstruction)
    target_pose = reconstruction.pose(target_coefficients)[-1]
    task = ContinuumPoseTask(
        reconstruction,
        1.0,
        target_pose[4:],
        target_pose[:4],
        tolerance=2.0e-6,
    )
    plan = ContinuumInverseKinematicsPlan(reconstruction, (task,))
    result = plan.solve_least_squares(
        jnp.zeros_like(target_coefficients),
        method=phx.optim.LevenbergMarquardt(),
        termination=_termination(),
    )

    assert result.optimizer.successful
    assert result.successful
    assert int(result.status) == int(ContinuumIKStatus.SUCCESS)
    assert result.accepted_reconstruction.valid
    assert result.accepted_reconstruction.interpretation.startswith(
        "observational/reference"
    )
    assert result.accepted_reconstruction.native_evaluation.valid
    np.testing.assert_allclose(
        result.accepted_state.coefficients, target_coefficients, atol=3.0e-6
    )


def test_quaternion_sign_has_one_content_identity_and_one_pose_residual():
    _reduction, reconstruction = _spatial_reconstruction(queries=(0.0, 1.0))
    coefficients = _constant_curvature_target(reconstruction)
    pose = reconstruction.pose(coefficients)[-1]
    positive = ContinuumPoseTask(
        reconstruction, 1.0, pose[4:], pose[:4], tolerance=1.0e-7
    )
    negative = ContinuumPoseTask(
        reconstruction, 1.0, pose[4:], -pose[:4], tolerance=1.0e-7
    )
    positive_plan = ContinuumInverseKinematicsPlan(reconstruction, (positive,))
    negative_plan = ContinuumInverseKinematicsPlan(reconstruction, (negative,))

    assert positive.task_id == negative.task_id
    np.testing.assert_array_equal(
        positive.target_orientation, negative.target_orientation
    )
    np.testing.assert_allclose(
        positive_plan.residual(jnp.zeros_like(coefficients)),
        negative_plan.residual(jnp.zeros_like(coefficients)),
        atol=2.0e-12,
    )


def test_conflicting_tasks_report_infeasible_candidate_and_roll_back_source():
    _reduction, reconstruction = _spatial_reconstruction(queries=(0.0, 1.0))
    source = jnp.zeros((6,), dtype=jnp.float64)
    position = reconstruction.pose(source)[-1, 4:]
    offset = jnp.asarray((0.12, 0.0, 0.0), dtype=source.dtype)
    left = ContinuumPositionTask(reconstruction, 1.0, position - offset, tolerance=1.0e-6)
    right = ContinuumPositionTask(
        reconstruction, 1.0, position + offset, tolerance=1.0e-6
    )
    result = ContinuumInverseKinematicsPlan(
        reconstruction, (left, right)
    ).solve_least_squares(
        source,
        method=phx.optim.GaussNewton(),
        termination=_termination(),
    )

    assert result.optimizer.successful
    assert not result.successful
    assert int(result.status) == int(ContinuumIKStatus.INFEASIBLE)
    assert not result.feasibility.task_bounds_satisfied
    np.testing.assert_array_equal(result.accepted_state.coefficients, source)
    assert result.candidate_evaluation.maximum_task_violation > 0.1


def test_coefficient_bounds_are_separate_from_task_feasibility_and_fail_closed():
    _reduction, reconstruction = _spatial_reconstruction(queries=(0.0, 1.0))
    source = jnp.zeros((6,), dtype=jnp.float64)
    target = source.at[0].set(0.25)
    task = ContinuumPostureTask(reconstruction, target, tolerance=1.0e-7)
    coefficient_bounds = phx.optim.Bounds(
        -0.04 * jnp.ones_like(source), 0.04 * jnp.ones_like(source)
    )
    result = ContinuumInverseKinematicsPlan(reconstruction, (task,)).solve_least_squares(
        source,
        method=phx.optim.BoundedLevenbergMarquardt(),
        termination=_termination(),
        coefficient_bounds=coefficient_bounds,
    )

    assert result.optimizer.successful
    assert result.feasibility.coefficient_bounds_satisfied
    assert not result.feasibility.task_bounds_satisfied
    assert not result.successful
    np.testing.assert_array_equal(result.accepted_state.coefficients, source)
    assert result.candidate_state.coefficients[0] <= 0.04 + 1.0e-7


def test_continuum_sqp_problem_exposes_exact_task_constraints_and_solves_posture():
    _reduction, reconstruction = _spatial_reconstruction(queries=(0.0, 1.0))
    source = jnp.zeros((6,), dtype=jnp.float64)
    target = jnp.asarray((0.02, -0.01, 0.0, 0.01, 0.0, -0.02), dtype=source.dtype)
    task = ContinuumPostureTask(reconstruction, target, tolerance=2.0e-6)
    plan = ContinuumInverseKinematicsPlan(reconstruction, (task,))
    problem = plan.sqp_problem(coefficient_bounds=phx.optim.Bounds(-0.1, 0.1))
    lower, upper = problem.constraints[0].bounds(problem.constraints[0].value(source))

    assert len(problem.constraints) == 1
    assert lower.shape == target.shape
    assert upper.shape == target.shape
    result = plan.solve_sqp(
        source,
        method=phx.optim.SQP(),
        termination=_termination(),
        coefficient_bounds=phx.optim.Bounds(-0.1, 0.1),
    )
    assert result.optimizer.certificate is not None
    assert result.successful
    np.testing.assert_allclose(result.accepted_state.coefficients, target, atol=2.0e-6)


def test_differential_ik_compiles_native_qp_with_velocity_and_one_step_bounds():
    _reduction, reconstruction = _spatial_reconstruction(queries=(0.0, 1.0))
    source = jnp.zeros((6,), dtype=jnp.float64)
    target = source.at[0].set(0.2)
    inverse = ContinuumInverseKinematicsPlan(
        reconstruction,
        (ContinuumPostureTask(reconstruction, target, tolerance=1.0e-6),),
    )
    differential = ContinuumDifferentialIKPlan(
        inverse,
        correction_gain=2.0,
        velocity_regularization=1.0e-7,
        time_step=0.1,
    )
    velocity_bounds = phx.optim.Bounds(-0.03, 0.03)
    coefficient_bounds = phx.optim.Bounds(-0.002, 0.002)
    compilation = differential.compile(
        source,
        velocity_bounds=velocity_bounds,
        coefficient_bounds=coefficient_bounds,
    )

    assert isinstance(compilation.program, phx.optim.QuadraticProgram)
    assert compilation.task_jacobian.shape == (6, 6)
    assert compilation.program.num_user_inequalities == 12
    np.testing.assert_allclose(compilation.task_jacobian, jnp.eye(6), atol=1.0e-12)

    result = differential.solve(
        source,
        velocity_bounds=velocity_bounds,
        coefficient_bounds=coefficient_bounds,
    )
    assert result.optimizer.successful
    assert result.successful
    assert result.bounds.velocity_bounds_satisfied
    assert result.bounds.coefficient_bounds_satisfied
    assert 0.0 < result.accepted_velocity[0] <= 0.0200001
    np.testing.assert_allclose(
        result.reduced_task_effort,
        compilation.task_jacobian.T
        @ (result.achieved_task_rate - result.desired_task_rate),
        atol=2.0e-10,
    )


def _passive_axial_plant_and_reconstruction():
    dtype = jnp.float32
    rod = prepare_rod(
        RodPlan(
            jnp.asarray(((0, 1),), dtype=jnp.int32),
            jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=dtype),
            jnp.eye(3, dtype=dtype)[None, ...],
            jnp.asarray((1.0, 1.5), dtype=dtype),
            jnp.diag(jnp.asarray((0.2, 0.3, 0.4), dtype=dtype))[None, ...],
            jnp.diag(jnp.asarray((40.0, 10.0, 10.0), dtype=dtype))[None, ...],
            jnp.zeros((0, 3, 3), dtype=dtype),
        )
    )
    basis = RodStrainBasisPlan.shifted_legendre(
        0,
        dimension=3,
        components=("nu_x",),
        component_scales=jnp.ones((6,), dtype=dtype),
    )
    reduction = prepare_reduced_rod(
        rod,
        ReducedRodPlan(
            basis,
            base_policy="fixed",
            fixed_base_position=jnp.zeros((3,), dtype=dtype),
            fixed_base_orientation=jnp.asarray((1.0, 0.0, 0.0, 0.0), dtype=dtype),
        ),
    )
    dynamics = prepare_reduced_rod_dynamics(reduction, ReducedRodDenseCholeskyPlan())
    plant = prepare_reduced_rod_plant(
        dynamics,
        ReducedRodSemiImplicitVelocityEuler(
            maximum_step_size=0.002,
            energy_balance_tolerance=1.0,
        ),
    )
    reconstruction = prepare_rod_reconstruction(
        reduction,
        RodReconstructionPlan(
            RodFrameQueryPlan(jnp.asarray((0.0, 1.0), dtype=dtype)),
            method="gvs",
        ),
    )
    return plant, reconstruction


def test_passive_trajectory_uses_complete_codec_and_authoritative_accepted_replay():
    plant, reconstruction = _passive_axial_plant_and_reconstruction()
    parameters = plant.bind_parameters()
    reset = plant.reset(jnp.asarray((11, 7), dtype=jnp.uint32), parameters)
    initial_coefficients = reset.accepted_state.payload.reduced_state.coefficients
    target_positions = reconstruction.pose(initial_coefficients)[:, 4:]
    inverse = ContinuumInverseKinematicsPlan(
        reconstruction,
        (
            ContinuumShapeTask(
                reconstruction,
                target_positions,
                bounds=phx.optim.Bounds(-0.02, 0.02),
            ),
        ),
    )
    state_size = sum(prod(leaf.shape) for leaf in plant.state_schema.leaves)
    state_space = ArraySpace(
        (state_size,), dtype=plant.initial_state.reduced_state.values.dtype
    )
    codec = PlantStateVectorCodec(
        plant.state_schema,
        StateLayout(
            (state_size,),
            local_space=state_space,
            tangent_space=state_space,
            layout_id=f"test-state:{plant.plant_id}",
        ),
        plant.initial_state,
        semantic_provenance=plant.semantic_provenance,
        numeric_revision=plant.numeric_revision,
        executable_signature=plant.execution_signature,
    )
    trajectory = SmoothReducedRodTrajectoryPlan(
        plant,
        codec,
        parameters,
        inverse,
        TimeGrid(
            jnp.asarray((0.0, 0.001), dtype=jnp.float32),
            time_id="soft-task-passive-replay",
        ),
        profile="passive",
    )
    replay = trajectory.accepted_replay(reset.accepted_state)

    assert not trajectory.controlled
    assert replay.replay.successful
    assert replay.successful
    assert replay.chart_valid
    assert replay.final_task_feasible
    assert replay.codec_id == codec.codec_id
    assert len(replay.replay.step_results) == 1
    assert replay.replay.final_state is replay.replay.accepted_states[-1]
    np.testing.assert_array_equal(
        replay.encoded_states[-1],
        codec.encode_point(replay.replay.final_state.payload).vector,
    )
