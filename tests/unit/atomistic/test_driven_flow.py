import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _periodic_dynamics():
    cell = phx.discretization.PeriodicCell(jnp.eye(3) * 10.0)
    system = phx.atomistic.AtomisticSystemPlan(
        [0, 1],
        [0, 0],
        [1.0, 1.0],
        phx.atomistic.AtomisticUnitSystem.reduced(),
        atom_type_ids=[0, 0],
        element_mask=[False, False],
        cell=cell,
    ).prepare()
    potential = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.LennardJonesPotential([0.1], [1.0], 2.5)]
    ).prepare(system)
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1, box=cell).prepare(
        system.particles
    )
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        neighborhood,
        phx.atomistic.VelocityVerletPlan(1.0e-3),
    ).prepare()
    return cell, dynamics


def test_evolving_cell_and_lees_edwards_remap_preserve_cartesian_positions():
    flow = phx.atomistic.driven_flow
    cell = phx.discretization.PeriodicCell(jnp.eye(3) * 10.0)
    plan = flow.EvolvingFlowCellPlan(cell)
    state = plan.initialize()
    gradient = jnp.asarray([[0.0, 0.1, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    advanced = plan.step(state, gradient, 6.0)
    assert advanced.accepted
    np.testing.assert_allclose(advanced.accepted_state.vectors[1, 0], 6.0)

    positions = jnp.asarray([[1.0, 2.0, 3.0]])
    images = jnp.zeros((1, 3), dtype=jnp.int32)
    remapped = flow.LeesEdwardsRemapPlan(plan).apply(
        advanced.accepted_state, positions, images
    )
    assert remapped.remapped & remapped.successful
    np.testing.assert_allclose(remapped.accepted_state.vectors[1, 0], -4.0)
    old_unwrapped = positions + images @ advanced.accepted_state.vectors
    new_unwrapped = (
        remapped.accepted_positions
        + remapped.accepted_image_counts @ remapped.accepted_state.vectors
    )
    np.testing.assert_allclose(new_unwrapped, old_unwrapped)


def test_planar_kraynik_reinelt_recurrence_resets_lattice():
    flow = phx.atomistic.driven_flow
    plan = flow.PlanarKraynikReineltPlan(0.2, 25.0, 5.0)
    state = plan.initialize()
    advanced = plan.step(state, plan.generalized.period)
    assert advanced.accepted
    positions = jnp.asarray([[0.5, 0.5, 0.5]])
    images = jnp.zeros((1, 3), dtype=jnp.int32)
    remapped = plan.remap_if_due(advanced.accepted_state, positions, images)
    assert remapped.remapped & remapped.successful
    np.testing.assert_allclose(
        remapped.accepted_state.vectors,
        plan.generalized.cell_plan.cell.vectors,
        rtol=1.0e-9,
        atol=1.0e-9,
    )
    assert plan.generalized.certification_residual < 1.0e-10


def test_sllod_uses_peculiar_momenta_and_rolls_back_failed_cell_step():
    flow = phx.atomistic.driven_flow
    cell, dynamics = _periodic_dynamics()
    cell_plan = flow.EvolvingFlowCellPlan(cell)
    runtime = flow.SLLODIntegratorPlan(
        1.0e-3, thermostat="gaussian-isokinetic", maximum_displacement=1.0
    ).prepare(dynamics, cell_plan)
    positions = jnp.asarray([[1.0, 1.0, 1.0], [4.0, 1.0, 1.0]])
    velocities = jnp.asarray([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    state = runtime.initialize(positions, velocities)
    accepted = runtime.step(state, jnp.zeros((3, 3)))
    assert accepted.accepted & accepted.successful
    np.testing.assert_allclose(
        accepted.peculiar_kinetic_energy,
        0.5 * jnp.sum(state.peculiar_momenta**2),
        rtol=1.0e-8,
    )

    rejected = runtime.step(state, jnp.eye(3))
    assert ~rejected.accepted
    assert rejected.accepted_state.successful
    np.testing.assert_array_equal(rejected.accepted_state.positions, state.positions)
    np.testing.assert_array_equal(
        rejected.accepted_state.flow_cell.vectors, state.flow_cell.vectors
    )


def test_total_driven_stress_uses_peculiar_momentum_and_tension_sign():
    _, dynamics = _periodic_dynamics()
    thermodynamic = phx.atomistic.AtomisticThermodynamicStatePlan(
        phx.atomistic.AtomisticPhaseSpaceMeasurePlan(dynamics.system),
        ensemble="nve",
    ).prepare(dynamics)
    positions = jnp.asarray([[1.0, 1.0, 1.0], [4.0, 1.0, 1.0]])
    velocities = jnp.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    state = dynamics.initialize_state(
        positions,
        thermodynamic,
        velocity=velocities,
        key=jax.random.key(17),
    )
    result = phx.atomistic.atomistic_driven_stress(
        phx.atomistic.AtomisticDrivenStressPlan(),
        dynamics,
        state,
        jnp.zeros((3, 3)),
    )
    assert result.successful & result.complete
    np.testing.assert_allclose(result.kinetic_pressure_tensor[0, 0], 2.0e-3)
    np.testing.assert_allclose(result.total_cauchy_stress[0, 0], -2.0e-3)
    np.testing.assert_allclose(result.pressure, 2.0e-3 / 3.0)


def test_shear_protocol_work_ledger_and_steady_rheology():
    flow = phx.atomistic.driven_flow
    protocol = flow.HomogeneousFlowProtocolPlan("steady-shear", rate=2.0)
    evaluated = protocol.evaluate(0.5)
    assert evaluated.successful
    np.testing.assert_allclose(evaluated.velocity_gradient[0, 1], 2.0)
    np.testing.assert_allclose(evaluated.accumulated_strain, 1.0)

    ledger_plan = flow.DrivenWorkLedgerPlan()
    ledger = flow.driven_work_ledger_step(
        ledger_plan,
        ledger_plan.initialize(2.0),
        2.0,
        0.5,
        1.0,
    )
    assert ledger.accepted & ledger.successful
    np.testing.assert_allclose(ledger.accepted_state.accumulated_flow_work, 1.0)

    stress = np.zeros((32, 3, 3))
    stress[:, 0, 1] = 4.0
    stress[:, 0, 0] = 3.0
    stress[:, 1, 1] = 1.0
    stress[:, 2, 2] = 0.5
    result = flow.shear_rheology(flow.ShearRheologyPlan(), stress, 2.0)
    assert result.successful
    np.testing.assert_allclose(result.apparent_viscosity, 2.0)
    np.testing.assert_allclose(result.first_normal_stress_difference, 2.0)
    np.testing.assert_allclose(result.second_normal_stress_difference, 0.5)


def test_laos_recovers_storage_loss_harmonics_and_cycle_work():
    flow = phx.atomistic.driven_flow
    amplitude = 2.0
    frequency = 3.0
    period = 2.0 * np.pi / frequency
    time = np.linspace(0.0, 4.0 * period, 801)
    strain = amplitude * np.sin(frequency * time)
    storage = 5.0
    loss = 2.0
    stress = amplitude * (
        storage * np.sin(frequency * time) + loss * np.cos(frequency * time)
    )
    result = flow.analyze_laos(
        flow.LAOSAnalysisPlan(
            amplitude,
            frequency,
            harmonic_count=5,
            discard_cycles=1,
            minimum_analysis_cycles=3,
            closure_tolerance=1.0e-8,
        ),
        time,
        strain,
        stress,
    )
    assert result.successful
    np.testing.assert_allclose(result.storage_moduli[0], storage, rtol=1.0e-10)
    np.testing.assert_allclose(result.loss_moduli[0], loss, rtol=1.0e-10)
    np.testing.assert_allclose(result.harmonic_magnitudes[1:], 0.0, atol=1.0e-10)
    np.testing.assert_allclose(
        result.dissipated_energy_per_cycle,
        np.pi * amplitude**2 * loss,
        rtol=1.0e-4,
    )
