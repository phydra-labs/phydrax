import jax
import jax.numpy as jnp

import phydrax as phx


def test_periodic_nvt_segmented_workflow_is_replayable_end_to_end():
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    cell = phx.discretization.PeriodicCell(4.0 * jnp.eye(3))
    system = phx.atomistic.AtomisticSystemPlan(
        [0, 1, 2],
        [1, 1, 1],
        [1.0, 1.0, 1.0],
        units,
        atom_type_ids=[0, 0, 0],
        cell=cell,
    ).prepare()
    base = phx.discretization.MetricCellListParticleNeighborhoodPlan(1.6, 3, 3, cell)
    neighborhood = phx.discretization.VerletParticleNeighborhoodPlan(
        base, 1.4, 0.2
    ).prepare(system.particles)
    potential = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.LennardJonesPotential([0.2], [0.8], 1.4, switch_distance=1.2)]
    ).prepare(system)
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        neighborhood,
        phx.atomistic.BAOABLangevinPlan(2.0e-4, 1.0, 0.2),
    ).prepare()
    positions = cell.cartesian(
        jnp.asarray([[0.1, 0.1, 0.1], [0.35, 0.1, 0.1], [0.6, 0.1, 0.1]])
    )
    initial = dynamics.initialize_state(
        positions, velocity=jnp.zeros_like(positions), key=jax.random.key(99)
    )
    rollout = phx.atomistic.AtomisticRolloutPlan(
        dynamics,
        phx.atomistic.AtomisticTrajectoryPlan(5, sample_stride=2),
        replay=phx.atomistic.AtomisticReplayPolicy("step"),
    )
    segmented = phx.atomistic.run_atomistic_segments(rollout, initial, 2)
    direct = phx.atomistic.AtomisticRolloutPlan(
        dynamics,
        phx.atomistic.AtomisticTrajectoryPlan(10, retention="final"),
        replay=phx.atomistic.AtomisticReplayPolicy("step"),
    ).rollout(initial)
    assert bool(segmented.successful)
    assert bool(direct.successful)
    assert int(segmented.final_state.step_index) == 10
    assert jnp.allclose(
        segmented.final_state.kinematics.positions,
        direct.final_state.kinematics.positions,
        rtol=0.0,
        atol=0.0,
    )
