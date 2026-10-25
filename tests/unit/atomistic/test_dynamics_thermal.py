from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from tests._support.atomistic_bundles import (
    prepare_program,
    role_capacity,
    search_capacity,
)


# Finite two-atom pair: nonperiodic search frame enclosing the trajectory and
# the one-pair storage of the former dense neighborhood.
_FRAME = phx.discretization.PeriodicCell(
    20.0 * jnp.eye(3), origin=jnp.full((3,), -10.0), periodic_axes=(False,) * 3
)


def _constrained_baoab() -> Any:
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    topology = phx.atomistic.MolecularTopologyPlan(
        constraints=[[10, 20]], constraint_distances=[1.0]
    )
    system = phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [10, 20],
        # ty: ignore[invalid-argument-type]
        [1, 1],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        units,
        # ty: ignore[invalid-argument-type]
        atom_type_ids=[0, 0],
        topology=topology,
    ).prepare()
    potential = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(
            # ty: ignore[invalid-argument-type]
            [phx.atomistic.LennardJonesPotential([0.1], [0.8], 2.0)]
        ),
        system,
        search=phx.atomistic.AtomisticCellListImageSearch(maximum_candidate_slots=4096),
        skin=0.5,
        search_capacities=(search_capacity(8, particles_per_cell=2, images=27),),
        role_capacities=(role_capacity(1, 1),),
        frame=_FRAME,
    )
    constraints = phx.atomistic.DistanceConstraintPlan(
        maximum_iterations=64, tolerance=1.0e-9
    ).prepare(system)
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        phx.atomistic.BAOABLangevinPlan(1.0e-3, 0.5),
        constraints=constraints,
    ).prepare()
    thermodynamic = phx.atomistic.AtomisticThermodynamicStatePlan(
        phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system),
        ensemble="nvt",
        temperature=1.0,
    ).prepare(dynamics)
    return dynamics, thermodynamic


def test_dynamics_thermal_scenario_1() -> None:
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
    dynamics, thermodynamic = _constrained_baoab()
    assert thermodynamic.constraint_manifold_id == dynamics.constraints.prepared_id
    assert (
        thermodynamic.phase_space_measure_id
        != thermodynamic.declared_phase_space_measure_id
    )
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    state = dynamics.initialize_state(
        positions,
        thermodynamic,
        velocity=jnp.asarray([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]]),
        key=jax.random.key(8),
    )
    step = dynamics.step_detailed(state, thermodynamic)
    assert bool(step.successful)
    unwrapped = dynamics._unwrapped(
        step.accepted_state.kinematics, step.accepted_state.cell_vectors
    )
    distance = jnp.sqrt(jnp.sum((unwrapped[0] - unwrapped[1]) ** 2))
    np.testing.assert_allclose(distance, 1.0, atol=1.0e-8)
    assert float(step.accepted_state.constraint_velocity_residual) <= 1.0e-9
    # BAOAB owns no chain, memory or cell controller; its exchanged heat is
    # booked in the ledger over the constrained kinetic degrees of freedom.
    assert thermodynamic.degrees_of_freedom == 5
    controllers = step.accepted_state.controllers
    assert controllers.particle_chain is None
    assert controllers.gle is None
    assert controllers.cell is None
    ledger = step.accepted_state.energy
    assert bool(jnp.isfinite(ledger.thermostat_heat))
    assert bool(jnp.isfinite(ledger.reservoir_heat))
    assert int(ledger.accepted_steps) == 1
    dynamics, thermodynamic = _constrained_baoab()
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    state = dynamics.initialize_state(
        positions,
        thermodynamic,
        velocity=jnp.zeros_like(positions),
        key=jax.random.key(9),
    )
    accumulator = phx.atomistic.ThermodynamicAccumulator.empty(positions.dtype)
    accumulator = accumulator.update(dynamics, state, thermodynamic)
    summary = phx.atomistic.summarize_thermodynamics(accumulator)
    assert int(summary.count) == 1
    rollout = phx.atomistic.AtomisticRolloutPlan(
        dynamics, thermodynamic, phx.atomistic.AtomisticTrajectoryPlan(2)
    ).rollout(state)
    data = phx.atomistic.atomistic_trajectory_data(rollout.trajectory, dynamics)
    assert data.states.shape == (3, 2, 2, 3)
    assert bool(jnp.all(data.sample_valid))
