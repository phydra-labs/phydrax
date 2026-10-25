from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications import cellular_mechanics as cm, polymer_liquids as pl
from phydrax.atomistic._stochastic_runtime import GeneralizedLangevinPlan
from tests._support.atomistic_bundles import (
    prepare_program,
    role_capacity,
    search_capacity,
)


def _runtime(integrator: Any = None, **ensemble: Any) -> Any:
    system = phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [10, 20, 30, 40],
        # ty: ignore[invalid-argument-type]
        [0, 0, 0, 0],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0, 1.0, 1.0],
        phx.atomistic.AtomisticUnitSystem.reduced(),
        # ty: ignore[invalid-argument-type]
        atom_type_ids=[0, 0, 0, 0],
        # ty: ignore[invalid-argument-type]
        element_mask=[False, False, False, False],
        # ty: ignore[invalid-argument-type]
        molecule_ids=[0, 0, 0, 0],
    ).prepare()
    # Finite four-bead chain: explicit nonperiodic frame and the six-pair
    # storage of the former dense neighborhood.
    potential = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(
            # ty: ignore[invalid-argument-type]
            [phx.atomistic.LennardJonesPotential([0.01], [0.5], 1.0)]
        ),
        system,
        search=phx.atomistic.AtomisticCellListImageSearch(maximum_candidate_slots=4096),
        skin=0.5,
        search_capacities=(search_capacity(16, particles_per_cell=4, images=27),),
        role_capacities=(role_capacity(6, 3),),
        frame=phx.discretization.PeriodicCell(
            20.0 * jnp.eye(3), origin=jnp.full((3,), -10.0), periodic_axes=(False,) * 3
        ),
    )
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        phx.atomistic.VelocityVerletPlan(1.0e-3) if integrator is None else integrator,
    ).prepare()
    thermodynamic = phx.atomistic.AtomisticThermodynamicStatePlan(
        phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system),
        **(ensemble or {"ensemble": "nve"}),
    ).prepare(dynamics)
    positions = jnp.asarray(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.4, 0.0, 0.0], [3.6, 0.0, 0.0]]
    )
    state = dynamics.initialize_state(
        positions,
        thermodynamic,
        velocity=jnp.zeros_like(positions),
        key=jax.random.key(7),
    )
    return dynamics, thermodynamic, state, positions


def test_polymer_dynamics_scenario_1() -> None:
    dynamics, thermodynamic, atomistic, positions = _runtime()
    chromatin = cm.ChromatinDynamicsPlan(
        4,
        1,
        ambient_dimension=3,
        capture_distance=10.0,
        spring_stiffness=0.1,
        spring_rest_length=1.0,
    ).prepare()
    # ty: ignore[invalid-argument-type]
    chromatin_state = chromatin.initialize(left=[0], right=[3])
    coupling = cm.ChromatinAtomisticCouplingPlan(
        # ty: ignore[invalid-argument-type]
        [10, 20, 30, 40],
        maximum_spring_energy=100.0,
    ).prepare(dynamics, chromatin)
    state = coupling.initialize(atomistic, chromatin_state)

    first = coupling.step(state, thermodynamic, jax.random.key(9))
    checkpoint = coupling.checkpoint(state)
    replay = coupling.step(coupling.restore(checkpoint), thermodynamic, jax.random.key(9))

    assert first.successful & replay.successful
    np.testing.assert_allclose(
        first.accepted_state.atomistic.kinematics.positions,
        replay.accepted_state.atomistic.kinematics.positions,
    )
    np.testing.assert_allclose(
        first.evidence.initial_springs.energy,
        chromatin.springs.energy(chromatin_state.relations, positions),
    )
    dynamics, _, _, positions = _runtime()
    runtime = phx.atomistic.OverdampedAtomisticPlan(
        1.0e-3, 0.2, 1.0, realization_id=11
    ).prepare(dynamics)
    state = runtime.initialize(positions, key=jax.random.key(3))

    left = runtime.step(state)
    right = runtime.step(state)

    assert left.successful & right.successful
    np.testing.assert_array_equal(
        left.accepted_state.positions, right.accepted_state.positions
    )
    assert int(left.accepted_state.step_index) == 1
    # Generalized Langevin is an in-schedule canonical NVT method over the
    # discrete FDT-checked half-step transition; temperature comes from the row.
    decay = 0.9
    transition = np.eye(2) * decay
    noise = np.eye(2) * np.sqrt(1.0 - decay**2)
    gle = GeneralizedLangevinPlan(1.0e-3, transition, noise, realization_id=5)
    np.testing.assert_allclose(gle.covariance_residual(), 0.0, atol=1.0e-12)
    dynamics, thermodynamic, atomistic, _ = _runtime(gle, ensemble="nvt", temperature=1.0)
    assert atomistic.controllers.gle is not None
    assert atomistic.controllers.gle.memory.shape == (4, 3, 1)

    left = dynamics.step_detailed(atomistic, thermodynamic)
    right = dynamics.step_detailed(atomistic, thermodynamic)

    assert left.successful & right.successful
    np.testing.assert_array_equal(
        left.accepted_state.kinematics.momenta,
        right.accepted_state.kinematics.momenta,
    )
    np.testing.assert_array_equal(
        left.accepted_state.controllers.gle.memory,
        right.accepted_state.controllers.gle.memory,
    )
    assert left.accepted_state.controllers.gle.memory.shape == (4, 3, 1)
    assert bool(jnp.any(left.accepted_state.controllers.gle.memory != 0.0))
    assert bool(jnp.isfinite(left.accepted_state.energy.reservoir_heat))


def test_equilibrium_rheology_retains_correlation_uncertainty_evidence() -> None:
    time = np.arange(64, dtype="float64")
    shear = np.sin(0.2 * time) + 0.2 * np.cos(0.7 * time)
    stress = np.zeros((64, 3, 3), dtype="float64")
    stress[:, 0, 1] = shear
    stress[:, 1, 0] = shear
    result = pl.polymer_green_kubo_viscosity(
        pl.PolymerStressCorrelationPlan(
            maximum_frames=64,
            maximum_lag=8,
            block_count=4,
            minimum_origins=16,
            maximum_relative_standard_error=1.0e9,
            maximum_stationarity_drift=1.0e9,
        ),
        stress,
        0.1,
        10.0,
        1.0,
        1.0,
    )
    assert result.successful
    assert float(result.shear_correlation[0]) > 0.0
    assert jnp.isfinite(result.viscosity)
