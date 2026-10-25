from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from tests._support.atomistic_bundles import (
    bundle_state,
    prepare_program,
    role_capacity,
    search_capacity,
)


# Two charged atoms in a cubic cell: one cell-list source over the real-space
# radius plus skin, and one stored pair route of the former dense neighborhood.
_PERIODIC: dict[str, Any] = {
    "search": phx.atomistic.AtomisticCellListImageSearch(maximum_candidate_slots=4096),
    "skin": 0.5,
    "search_capacities": (search_capacity(8, particles_per_cell=2, images=27),),
    "role_capacities": (role_capacity(1, 1),),
}


def _system(*, cell: Any = None) -> Any:
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    return phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [0, 1],
        # ty: ignore[invalid-argument-type]
        [1, 1],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        units,
        # ty: ignore[invalid-argument-type]
        atom_type_ids=[0, 0],
        # ty: ignore[invalid-argument-type]
        charges=[1.0, -1.0],
        cell=cell,
    ).prepare()


def test_finite_direct_coulomb_has_balanced_forces() -> None:
    system = _system()
    # Finite direct Coulomb: the complete pair relation of two atoms.
    potential = prepare_program(
        phx.atomistic.AtomisticPotentialProgram([phx.atomistic.DirectCoulombPotential()]),
        system,
        maximum_pairs=1,
    )
    positions = jnp.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    evaluation = potential.evaluate(positions, bundle_state(potential, positions))
    np.testing.assert_allclose(evaluation.energy, -0.5, atol=1.0e-12)
    np.testing.assert_allclose(jnp.sum(evaluation.forces, axis=0), 0.0, atol=1.0e-12)


def test_periodic_pme_matches_ewald_reference() -> None:
    cell = phx.discretization.PeriodicCell(6.0 * jnp.eye(3))
    system = _system(cell=cell)
    positions = jnp.asarray([[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]])
    reference = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(
            [phx.atomistic.EwaldReferencePotential(0.8, 2.5, 5)]
        ),
        system,
        **_PERIODIC,
    )
    pme = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(
            [phx.atomistic.ParticleMeshEwaldPotential(0.8, 2.5, (16, 16, 16))]
        ),
        system,
        **_PERIODIC,
    )
    expected = reference.evaluate(positions, bundle_state(reference, positions))
    observed = pme.evaluate(positions, bundle_state(pme, positions))
    assert bool(expected.successful)
    assert bool(observed.successful)
    np.testing.assert_allclose(observed.energy, expected.energy, rtol=2e-1, atol=2e-1)
    np.testing.assert_allclose(jnp.sum(observed.forces, axis=0), 0.0, atol=1.0e-3)


def test_pme_monte_carlo_cell_move_commits_controller_state() -> None:
    cell = phx.discretization.PeriodicCell(6.0 * jnp.eye(3))
    system = _system(cell=cell)
    potential = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(
            [phx.atomistic.ParticleMeshEwaldPotential(0.8, 2.5, (8, 8, 8))]
        ),
        system,
        **_PERIODIC,
    )
    # A Monte Carlo volume move is scheduled beside a canonical NVT method.
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        phx.atomistic.BAOABLangevinPlan(1.0e-3, 1.0),
    ).prepare()
    thermodynamic = phx.atomistic.AtomisticThermodynamicStatePlan(
        phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system),
        ensemble="npt",
        temperature=1.0,
        pressure=0.0,
    ).prepare(dynamics)
    positions = jnp.asarray([[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]])
    state = dynamics.initialize_state(
        positions,
        thermodynamic,
        velocity=jnp.zeros_like(positions),
        key=jax.random.key(14),
    )
    move = phx.atomistic.apply_isotropic_monte_carlo_barostat(
        dynamics,
        state,
        thermodynamic,
        phx.atomistic.IsotropicMonteCarloBarostatPlan(0.01),
    )
    assert bool(move.successful)
    assert bool(jnp.isfinite(move.energy_after))
    assert int(move.accepted_state.controllers.move_counter) == 1


def test_molecular_volume_measure_controls_monte_carlo_cell_move() -> None:
    cell = phx.discretization.PeriodicCell(5.0 * jnp.eye(3))
    system = _system(cell=cell)
    potential = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(
            # ty: ignore[invalid-argument-type]
            [phx.atomistic.LennardJonesPotential([0.2], [1.0], 2.0)]
        ),
        system,
        **_PERIODIC,
    )
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        phx.atomistic.BAOABLangevinPlan(1.0e-3, 1.0),
    ).prepare()
    measure = phx.atomistic.AtomisticPhaseSpaceMeasurePlan(
        system,
        scaled_entity_count=1,
        volume_coordinate_convention="molecular-center",
    )
    thermodynamic = phx.atomistic.AtomisticThermodynamicStatePlan(
        measure,
        ensemble="npt",
        temperature=1.0,
        pressure=0.1,
    ).prepare(dynamics)
    assert thermodynamic.scaled_entity_count == 1
    assert thermodynamic.volume_coordinate_convention == "molecular-center"
    positions = jnp.asarray([[1.0, 1.0, 1.0], [2.2, 1.0, 1.0]])
    state = dynamics.initialize_state(
        positions,
        thermodynamic,
        velocity=jnp.zeros_like(positions),
        key=jax.random.key(11),
    )
    move = phx.atomistic.apply_isotropic_monte_carlo_barostat(
        dynamics,
        state,
        thermodynamic,
        phx.atomistic.IsotropicMonteCarloBarostatPlan(0.05),
    )
    assert bool(move.successful)
    assert bool(jnp.isfinite(move.log_acceptance_probability))
    assert float(move.volume_after) > 0.0
    assert move.accepted_state.cell_vectors.shape == (3, 3)
    assert int(move.accepted_state.controllers.move_counter) == 1
