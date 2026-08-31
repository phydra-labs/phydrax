#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _materials(*, rolling=0.0):
    return phx.equations.DEMMaterialTable(
        jnp.asarray([2.0e5]),
        jnp.asarray([0.25]),
        jnp.asarray([[0.8]]),
        jnp.asarray([[0.4]]),
        rolling_friction=jnp.asarray([[rolling]]),
    )


def _compile(
    normal,
    *,
    tangential=None,
    rolling=None,
    barriers=(),
    neighborhood=None,
    execution=None,
    materials=None,
    maximum_overlap=0.3,
):
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([10, 20]), jnp.ones((2,)), ambient_dimension=2
    ).prepare()
    spheres = phx.discretization.RigidSphereSetPlan(
        jnp.asarray([0.5, 0.5]), jnp.asarray([0, 0])
    )
    contact = phx.discretization.DEMContactModelPlan(
        normal, tangential=tangential, rolling=rolling
    )
    method = phx.discretization.SoftSphereDEMMethodPlan(
        contact, maximum_overlap_fraction=maximum_overlap
    )
    problem = phx.equations.DiscreteElementProblemIR(
        "extended-dem",
        _materials() if materials is None else materials,
        gravity=jnp.zeros((2,)),
        barriers=barriers,
    )
    return phx.equations.compile_discrete_element_problem(
        problem,
        particles,
        spheres,
        method,
        neighborhood=(
            phx.discretization.DenseParticleNeighborhoodPlan(1)
            if neighborhood is None
            else neighborhood
        ),
        execution=execution,
    )


def test_energy_ledger_is_source_resolved_rejection_safe_and_qualifiable():
    compiled = _compile(phx.discretization.LinearSpringDashpotNormalPlan(1.0e4))
    state = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0], [0.95, 0.0]]),
        jnp.zeros((2, 2)),
    )
    detail = compiled.dynamics.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        jnp.asarray(1.0e-4),
        None,
    )

    assert detail.successful
    assert detail.energy.accepted
    assert jnp.abs(detail.energy.energy_residual) < 1.0e-12
    assert detail.accepted_state.energy.accepted_steps == 1
    assert detail.accepted_state.energy.cumulative_boundary_contact_work.shape == (0,)
    diagnostics = compiled.diagnostics(1.0e-4, detail.accepted_state)
    artifact = phx.discretization.qualify_dem(
        diagnostics,
        phx.discretization.DEMQualificationProfile(
            maximum_overlap_fraction=0.3,
            energy_balance_tolerance=1.0e-10,
        ),
    )
    assert artifact.qualified

    strict = _compile(
        phx.discretization.LinearSpringDashpotNormalPlan(1.0e4),
        maximum_overlap=0.01,
    )
    strict_state = strict.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0], [0.95, 0.0]]),
        jnp.zeros((2, 2)),
    )
    rejected = strict.dynamics.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        strict_state,
        jnp.asarray(1.0e-4),
        None,
    )
    assert not rejected.successful
    assert rejected.rejection_reasons & int(phx.discretization.DEMRejectionReason.OVERLAP)
    assert rejected.accepted_state.energy.accepted_steps == 0


def test_verlet_fused_hierarchical_and_batched_paths_preserve_authority():
    box = phx.discretization.ParticleBox(
        jnp.asarray([-1.0, -1.0]),
        jnp.asarray([1.0, 1.0]),
        periodic_axes=(False, False),
    )
    base = phx.discretization.CellListParticleNeighborhoodPlan(1.2, 2, 1, box)
    verlet = phx.discretization.VerletParticleNeighborhoodPlan(base, 1.0, 0.2)
    execution = phx.discretization.ParticleExecutionPolicy(
        realization="cell_edge_list", kernel_backend="verlet_fused"
    )
    compiled = _compile(
        phx.discretization.LinearSpringDashpotNormalPlan(1.0e4),
        neighborhood=verlet,
        execution=execution,
    )
    positions = jnp.asarray([[-0.45, 0.0], [0.45, 0.0]])
    state = compiled.initialize_state(0.0, positions, jnp.zeros((2, 2)))
    detail = compiled.dynamics.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        jnp.asarray(1.0e-5),
        None,
    )
    assert detail.successful
    assert not detail.accepted_state.neighborhood_cache.rebuilt
    assert detail.accepted_state.neighborhood_cache.rebuild_count == 1

    batch = phx.discretization.initialize_dem_batch(
        compiled.dynamics,
        jnp.asarray(0.0),
        jnp.stack((positions, positions + jnp.asarray([0.0, 0.05]))),
        jnp.zeros((2, 2, 2)),
    )
    batched = phx.discretization.batch_step_detailed(
        compiled.dynamics,
        batch,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        jnp.asarray(1.0e-5),
        phx.discretization.DEMBatchExecutionPlan(),
    )
    assert jnp.all(batched.successful)

    hierarchy = phx.discretization.HierarchicalRadiusParticleNeighborhoodPlan(
        phx.discretization.CellListParticleNeighborhoodPlan(1.1, 2, 1, box),
        jnp.asarray([0.5, 0.5]),
        jnp.asarray([0.1, 0.6]),
        skin=0.1,
    )
    particles = compiled.dynamics.bodies.particles
    hierarchy_state = hierarchy.prepare(particles).build(positions)
    assert hierarchy_state.successful
    assert hierarchy_state.pair_count == 1


def test_rolling_smooth_sensitivity_and_checkpoint_replay_are_operational():
    compiled = _compile(
        phx.discretization.LinearSpringDashpotNormalPlan(1.0e4),
        rolling=phx.discretization.ConstantRollingResistancePlan(),
        materials=_materials(rolling=0.1),
    )
    state = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0], [0.9, 0.0]]),
        jnp.zeros((2, 2)),
        jnp.asarray([[1.0], [-1.0]]),
    )
    evaluation = compiled.dynamics.evaluate(
        jnp.asarray(0.0), state, jnp.asarray(1.0e-4), None
    )
    assert evaluation.particle_contact.rolling_dissipated_work[0] > 0.0
    assert jnp.allclose(
        evaluation.particle_contact.rolling_torque_left,
        -evaluation.particle_contact.rolling_torque_right,
    )

    policy = phx.discretization.DEMSensitivityPolicy(
        activation_margin=1.0e-12,
        no_tension_margin=1.0e-12,
        friction_margin=1.0e-12,
        frame_margin=1.0e-12,
        acceptance_margin=1.0e-12,
        neighborhood_margin=1.0e-12,
    )
    diagnostics = compiled.diagnostics(0.0, state)
    result = phx.discretization.sharp_branchwise_jvp(
        lambda value: value**2,
        jnp.asarray(2.0),
        jnp.asarray(1.0),
        diagnostics,
        policy,
    )
    assert result.usable
    assert jnp.isclose(result.sensitivity, 4.0)

    replay = phx.discretization.checkpointed_dem_rollout(
        compiled.dynamics,
        state,
        t0=0.0,
        step_size=1.0e-5,
        step_count=2,
        checkpoint=phx.discretization.DEMCheckpointPolicy(1),
    )
    assert replay.successful
    assert replay.replay.successful.shape == (2,)

    smooth = _compile(
        phx.discretization.SmoothPenaltyNormalPlan(
            1.0e4, gap_smoothing=1.0e-3, force_smoothing=1.0e-3
        ),
        tangential=phx.discretization.SmoothCoulombTangentialPlan(
            2.5e3, direction_smoothing=1.0e-4
        ),
    )
    smooth_state = smooth.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0], [0.9, 0.0]]),
        jnp.zeros((2, 2)),
    )
    assert smooth.diagnostics(0.0, smooth_state).successful


def test_moving_servo_curvature_dmt_plastic_and_contact_thermal_models():
    geometry = phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()

    def motion(base, time, points, args):
        del time, args
        return phx.discretization.DEMBarrierMotion(
            base,
            jnp.asarray([0.1, 0.0]),
            jnp.asarray([0.0]),
            jnp.zeros((2,)),
            jnp.asarray(True),
        )

    barrier = phx.discretization.ImplicitDEMBarrier(
        geometry,
        phx.discretization.DEMBarrierSide.INTERIOR,
        0,
        barrier_id="moving-square",
        motion=phx.discretization.PrescribedDEMBarrierMotionPlan(
            motion, motion_id="moving-square-motion"
        ),
    )
    moving = _compile(
        phx.discretization.LinearSpringDashpotNormalPlan(1.0e4),
        barriers=(barrier,),
    )
    moving_state = moving.initialize_state(
        0.0,
        jnp.asarray([[0.9, 0.0], [0.0, 0.0]]),
        jnp.zeros((2, 2)),
    )
    moving_eval = moving.dynamics.evaluate(
        jnp.asarray(0.0), moving_state, jnp.asarray(1.0e-4), None
    )
    assert jnp.isfinite(moving_eval.boundaries[0].wall_power)

    servo = phx.discretization.ServoDEMBarrierMotionPlan(
        jnp.asarray([1.0, 0.0]),
        1.0,
        proportional_gain=0.1,
        velocity_limit=1.0,
        geometry_function=lambda value, displacement: value,
        motion_id="servo",
    )
    servo_state = servo.update(
        servo.initialize(), jnp.asarray([0.0, 0.0]), jnp.asarray(0.1)
    )
    assert servo_state.displacement > 0.0

    circle = phx.discretization.ImplicitDEMBarrier(
        phx.geometry.Circle((0.0, 0.0), 2.0).compile(),
        phx.discretization.DEMBarrierSide.INTERIOR,
        0,
        barrier_id="curved-hertz",
    )
    hertz = _compile(phx.discretization.HertzNormalContactPlan(), barriers=(circle,))
    hertz_state = hertz.initialize_state(
        0.0,
        jnp.asarray([[1.6, 0.0], [0.0, 0.0]]),
        jnp.zeros((2, 2)),
    )
    assert hertz.diagnostics(0.0, hertz_state).successful

    dmt = _compile(phx.discretization.DMTAdhesiveNormalPlan(0.05, 0.1))
    dmt_state = dmt.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0], [1.05, 0.0]]),
        jnp.zeros((2, 2)),
    )
    assert dmt_state.loads.total.force[0, 0] > 0.0

    plastic = _compile(
        phx.discretization.ThorntonLinearPlasticNormalPlan(1.0e4, 2.0e4, 2.0e3, 0.02)
    )
    plastic_state = plastic.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0], [0.95, 0.0]]),
        jnp.zeros((2, 2)),
    )
    plastic_step = plastic.dynamics.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        plastic_state,
        jnp.asarray(1.0e-4),
        None,
    )
    assert plastic_step.candidate_state.particle_history.normal_plastic_overlap[0] > 0.0

    thermal_plan = phx.discretization.LumpedContactThermalPlan(
        jnp.asarray([1.0]), jnp.asarray([[2.0]])
    )
    thermal_state = thermal_plan.initialize(
        plastic.dynamics.bodies, jnp.asarray([300.0, 400.0])
    )
    thermal_response = thermal_plan.evaluate(
        plastic.dynamics.bodies, plastic_step.evaluation, thermal_state
    )
    assert jnp.isclose(jnp.sum(thermal_response.temperature_rate), 0.0)
    assert thermal_response.entropy_production >= 0.0
