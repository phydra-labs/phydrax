import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _constrained_baoab():
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    topology = phx.atomistic.MolecularTopologyPlan(
        constraints=[[10, 20]], constraint_distances=[1.0]
    )
    system = phx.atomistic.AtomisticSystemPlan(
        [10, 20],
        [1, 1],
        [1.0, 1.0],
        units,
        atom_type_ids=[0, 0],
        topology=topology,
    ).prepare()
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1).prepare(
        system.particles
    )
    potential = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.LennardJonesPotential([0.1], [0.8], 2.0)]
    ).prepare(system)
    constraints = phx.atomistic.DistanceConstraintPlan(
        maximum_iterations=64, tolerance=1.0e-9
    ).prepare(system)
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        neighborhood,
        phx.atomistic.BAOABLangevinPlan(1.0e-3, 1.0, 0.5),
        constraints=constraints,
    ).prepare()
    return dynamics


def test_stable_particle_noise_permutes_with_stable_ids():
    key = jax.random.key_data(jax.random.key(7))
    ids = jnp.asarray([30, 10, 20])
    permutation = jnp.asarray([1, 2, 0])
    first = phx.atomistic.stable_particle_normals(
        key,
        ids,
        4,
        operator_id=2,
        realization_id=3,
        dtype=jnp.float64,
    )
    second = phx.atomistic.stable_particle_normals(
        key,
        ids[permutation],
        4,
        operator_id=2,
        realization_id=3,
        dtype=jnp.float64,
    )
    np.testing.assert_array_equal(second, first[permutation])


def test_baoab_and_rattle_preserve_distance_and_velocity_tangent():
    dynamics = _constrained_baoab()
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    state = dynamics.initialize_state(
        positions,
        velocity=jnp.asarray([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]]),
        key=jax.random.key(8),
    )
    step = dynamics.step_detailed(state)
    assert bool(step.successful)
    unwrapped = dynamics._unwrapped(
        step.accepted_state.kinematics, step.accepted_state.cell_vectors
    )
    distance = jnp.sqrt(jnp.sum((unwrapped[0] - unwrapped[1]) ** 2))
    np.testing.assert_allclose(distance, 1.0, atol=1.0e-8)
    assert float(step.accepted_state.constraint_velocity_residual) <= 1.0e-9
    assert bool(jnp.all(jnp.isfinite(step.accepted_state.thermostat_state)))


def test_thermodynamic_observer_and_trajectory_adapter_are_typed():
    dynamics = _constrained_baoab()
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    state = dynamics.initialize_state(
        positions, velocity=jnp.zeros_like(positions), key=jax.random.key(9)
    )
    accumulator = phx.atomistic.ThermodynamicAccumulator.empty(positions.dtype)
    accumulator = accumulator.update(dynamics, state)
    summary = phx.atomistic.summarize_thermodynamics(accumulator)
    assert int(summary.count) == 1
    rollout = phx.atomistic.AtomisticRolloutPlan(
        dynamics, phx.atomistic.AtomisticTrajectoryPlan(2)
    ).rollout(state)
    data = phx.atomistic.atomistic_trajectory_data(rollout.trajectory, dynamics)
    assert data.states.shape == (3, 2, 2, 3)
    assert bool(jnp.all(data.sample_valid))
