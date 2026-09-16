import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _polymer_runtime():
    cell = phx.discretization.PeriodicCell(jnp.eye(3) * 10.0)
    particle_ids = np.asarray([10, 20, 30, 40])
    topology = phx.atomistic.MolecularTopologyPlan(
        bonds=[[10, 20], [20, 30], [30, 40]],
        bond_type_ids=[0, 0, 0],
    )
    system = phx.atomistic.AtomisticSystemPlan(
        particle_ids,
        [0, 0, 0, 0],
        [1.0, 1.0, 1.0, 1.0],
        phx.atomistic.AtomisticUnitSystem.reduced(),
        atom_type_ids=[0, 0, 0, 0],
        element_mask=[False, False, False, False],
        molecule_ids=[0, 0, 0, 0],
        topology=topology,
        cell=cell,
    ).prepare()
    cutoff = 2.0 ** (1.0 / 6.0)
    potential = phx.atomistic.AtomisticPotentialProgram(
        [
            phx.atomistic.FiniteExtensibleNonlinearElasticBondPotential([30.0], [1.5]),
            phx.atomistic.LennardJonesPotential(
                [1.0], [1.0], cutoff, shift_energy_at_cutoff=True
            ),
        ]
    ).prepare(system)
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(6, box=cell).prepare(
        system.particles
    )
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        neighborhood,
        phx.atomistic.BAOABLangevinPlan(1.0e-3, 1.0, realization_id=3),
    ).prepare()
    thermodynamic = phx.atomistic.AtomisticThermodynamicStatePlan(
        phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system),
        ensemble="nvt",
        temperature=1.0,
    ).prepare(dynamics)
    layout = phx.atomistic.PolymerChainLayoutPlan(
        [[0, 1, 2, 3]], [[True, True, True, True]], maximum_frames=4
    )
    profile = phx.applications.polymer_liquids.KremerGrestProfilePlan(
        production_steps=10,
        maximum_particles=4,
        maximum_chains=1,
        minimum_fene_margin=0.2,
    ).prepare(dynamics, layout)
    return dynamics, thermodynamic, layout, profile


def test_shifted_lennard_jones_is_continuous_wca_parameterization():
    cutoff = 2.0 ** (1.0 / 6.0)
    dynamics, thermodynamic, _, _ = _polymer_runtime()
    positions = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [cutoff, 0.0, 0.0],
            [2.0 * cutoff, 0.0, 0.0],
            [3.0 * cutoff, 0.0, 0.0],
        ]
    )
    state = dynamics.initialize_state(
        positions,
        thermodynamic,
        velocity=jnp.zeros_like(positions),
        key=jax.random.key(0),
    )
    # Only the LJ term is zero at its cutoff; FENE bonds still contribute.
    context = dynamics.potential.context(
        state.kinematics.positions,
        state.neighborhood,
        unwrapped_positions=dynamics._unwrapped(state.kinematics),
        species=state.species,
        cell=dynamics.system.cell,
        fractional_positions=dynamics.system.cell.fractional(state.kinematics.positions),
        cell_vectors=state.cell_vectors,
    )
    lj = dynamics.potential.terms[1].energy(context)
    np.testing.assert_allclose(lj.energy, 0.0, atol=1.0e-12)

    with pytest.raises(ValueError, match="mutually exclusive"):
        phx.atomistic.LennardJonesPotential(
            [1.0],
            [1.0],
            cutoff,
            switch_distance=1.0,
            shift_energy_at_cutoff=True,
        )


def test_fene_domain_and_kremer_grest_profile_evidence():
    dynamics, thermodynamic, _, profile = _polymer_runtime()
    positions = jnp.asarray(
        [[1.0, 1.0, 1.0], [1.96, 1.0, 1.0], [2.92, 1.0, 1.0], [3.88, 1.0, 1.0]]
    )
    state = dynamics.initialize_state(
        positions,
        thermodynamic,
        velocity=jnp.zeros_like(positions),
        key=jax.random.key(1),
    )
    evidence = phx.applications.polymer_liquids.kremer_grest_evidence(
        profile, state, thermodynamic
    )
    assert evidence.successful
    np.testing.assert_allclose(evidence.maximum_bond_fraction, 0.96 / 1.5)
    fene = dynamics.potential.terms[0]

    def fene_energy(coordinates):
        context = dynamics.potential.context(
            coordinates,
            state.neighborhood,
            unwrapped_positions=coordinates,
            species=state.species,
            cell=dynamics.system.cell,
            fractional_positions=dynamics.system.cell.fractional(coordinates),
            cell_vectors=state.cell_vectors,
        )
        return fene.energy(context).energy

    ratio_squared = (0.96 / 1.5) ** 2
    expected_energy = 3.0 * (-0.5 * 30.0 * 1.5**2 * np.log(1.0 - ratio_squared))
    expected_force = 30.0 * 0.96 / (1.0 - ratio_squared)
    np.testing.assert_allclose(fene_energy(positions), expected_energy)
    force = -jax.grad(fene_energy)(positions)
    np.testing.assert_allclose(force[0, 0], expected_force)
    np.testing.assert_allclose(jnp.sum(force, axis=0), 0.0, atol=1.0e-12)

    invalid_positions = positions.at[1, 0].set(2.6)
    neighborhood = dynamics.neighborhood.build(invalid_positions)
    invalid = dynamics.potential.evaluate(
        invalid_positions,
        neighborhood,
        unwrapped_positions=invalid_positions,
        species=dynamics.system.plan.atom_type_ids,
        cell=dynamics.system.cell,
        fractional_positions=dynamics.system.cell.fractional(invalid_positions),
        cell_vectors=dynamics.system.cell.vectors,
    )
    assert not invalid.successful
    assert jnp.isnan(invalid.energy)


def test_polymer_observables_retain_normalization_and_contour_semantics():
    _, _, layout, _ = _polymer_runtime()
    positions = jnp.asarray(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]
    )
    conformation = phx.atomistic.polymer_conformation(layout, positions)
    assert conformation.successful
    np.testing.assert_allclose(conformation.centers_of_mass, [[[1.5, 0.0, 0.0]]])
    np.testing.assert_allclose(conformation.end_to_end_squared, [[9.0]])
    np.testing.assert_allclose(conformation.radius_of_gyration_squared, [[1.25]])

    contour_plan = phx.atomistic.PolymerContourStatisticsPlan(layout, 2, 1.1)
    contour = phx.atomistic.polymer_contour_statistics(contour_plan, positions)
    assert contour.successful
    np.testing.assert_allclose(contour.internal_distance_squared, [[[1.0, 4.0]]])
    np.testing.assert_allclose(contour.contact_probability, [[[1.0, 0.0]]])

    debye = phx.atomistic.debye_scattering(
        phx.atomistic.DebyeScatteringPlan(
            [0.0, 1.0], maximum_frames=1, maximum_particles=4, block_size=2
        ),
        positions,
    )
    assert debye.successful
    np.testing.assert_allclose(debye.values[0], 4.0)

    partial = phx.atomistic.partial_structure_factors(
        phx.atomistic.PartialStructureFactorPlan(
            [[0.0, 0.0, 0.0]],
            1,
            maximum_frames=1,
            maximum_particles=4,
        ),
        positions,
        [0, 0, 0, 0],
    )
    assert partial.successful
    np.testing.assert_allclose(partial.values, [[[4.0]]])
