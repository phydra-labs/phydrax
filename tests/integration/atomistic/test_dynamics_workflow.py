import jax
import jax.numpy as jnp

import phydrax as phx
from tests._support.atomistic_bundles import (
    prepare_program,
    role_capacity,
    search_capacity,
)


def test_periodic_nvt_segmented_workflow_is_replayable_end_to_end() -> None:
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    cell = phx.discretization.PeriodicCell(4.0 * jnp.eye(3))
    system = phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [0, 1, 2],
        # ty: ignore[invalid-argument-type]
        [1, 1, 1],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0, 1.0],
        units,
        # ty: ignore[invalid-argument-type]
        atom_type_ids=[0, 0, 0],
        cell=cell,
    ).prepare()
    # Explicit resources of the former metric cell list (radius 1.6 = cutoff
    # 1.4 plus skin 0.2, three atoms per cell, three pairs).
    potential = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(
            # ty: ignore[invalid-argument-type]
            [phx.atomistic.LennardJonesPotential([0.2], [0.8], 1.4, switch_distance=1.2)]
        ),
        system,
        search=phx.atomistic.AtomisticCellListImageSearch(maximum_candidate_slots=4096),
        skin=0.2,
        search_capacities=(search_capacity(6, particles_per_cell=3, images=27),),
        role_capacities=(role_capacity(3, 2),),
    )
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        phx.atomistic.BAOABLangevinPlan(2.0e-4, 0.2),
    ).prepare()
    thermodynamic = phx.atomistic.AtomisticThermodynamicStatePlan(
        phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system),
        ensemble="nvt",
        temperature=1.0,
    ).prepare(dynamics)
    positions = cell.cartesian(
        jnp.asarray([[0.1, 0.1, 0.1], [0.35, 0.1, 0.1], [0.6, 0.1, 0.1]])
    )
    initial = dynamics.initialize_state(
        positions,
        thermodynamic,
        velocity=jnp.zeros_like(positions),
        key=jax.random.key(99),
    )
    rollout = phx.atomistic.AtomisticRolloutPlan(
        dynamics,
        thermodynamic,
        phx.atomistic.AtomisticTrajectoryPlan(5, sample_stride=2),
        replay=phx.atomistic.AtomisticReplayPolicy("step"),
    )
    segmented = phx.atomistic.run_atomistic_segments(rollout, initial, 2)
    direct = phx.atomistic.AtomisticRolloutPlan(
        dynamics,
        thermodynamic,
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
