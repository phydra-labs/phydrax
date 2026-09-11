import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _case():
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    target = np.asarray(
        [[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.24, 0.92, 0.0]],
        dtype=float,
    )
    structure = phx.atomistic.AtomicStructure(
        [8, 1, 1],
        target + np.asarray([[0.05, -0.03, 0.02], [0.02, 0.04, -0.01], [-0.03, 0.01, 0.03]]),
        [15.999, 1.008, 1.008],
        units.scale,
        particle_ids=[101, 102, 103],
        name="water-fixture",
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        structure, units, molecule_ids=[0, 0, 0]
    )
    stiffness = 2.0

    def evaluate(positions, _cell):
        coordinate = jnp.asarray(positions)
        delta = coordinate - jnp.asarray(target)
        energy = 0.5 * stiffness * jnp.sum(delta**2)
        forces = -stiffness * delta
        return phx.chemistry.PotentialEnergySurfaceEvaluation(
            energy,
            forces,
            None,
            True,
            provider_id="harmonic-surface",
            source_result_id=phx.atomistic.AtomicStructure(
                [8, 1, 1],
                coordinate,
                [15.999, 1.008, 1.008],
                units.scale,
                particle_ids=[101, 102, 103],
            ).structure_id,
        )

    surface = phx.chemistry.CallablePotentialEnergySurface(
        evaluate,
        system.system_id,
        units,
        "harmonic-surface",
        phx.chemistry.PotentialEnergySurfaceCapabilities(),
    )
    return units, target, structure, system, surface


def test_optimization_hessian_vibration_and_rrho_form_one_typed_workflow():
    units, target, structure, system, surface = _case()
    convergence = phx.chemistry.MolecularGeometryConvergencePlan(
        maximum_force=1.0e-7,
        rms_force=1.0e-7,
        maximum_steps=64,
        maximum_evaluations=256,
    )
    optimized = phx.chemistry.MolecularGeometryOptimizationPlan(
        system, surface, convergence=convergence
    ).run(structure)

    assert bool(optimized.successful)
    np.testing.assert_allclose(optimized.final_structure.positions, target, atol=1.0e-7)

    hessian = phx.chemistry.MolecularHessianPlan(
        system,
        surface,
        displacement=1.0e-4,
        antisymmetry_tolerance=1.0e-10,
    ).evaluate(optimized.final_structure)
    assert bool(hessian.successful)
    np.testing.assert_allclose(
        np.asarray(hessian.hessian).reshape((9, 9)), 2.0 * np.eye(9), atol=1.0e-9
    )

    vibration = phx.chemistry.VibrationalAnalysisPlan(system).evaluate(
        optimized.final_structure, hessian
    )
    assert bool(vibration.successful)
    assert vibration.external_mode_count == 6
    assert vibration.internal_mode_count == 3
    assert vibration.stationary_point is phx.chemistry.StationaryPointKind.MINIMUM
    assert np.all(np.asarray(vibration.wavenumbers) > 0.0)

    pressure = 101325.0 * float(
        phx.units.conversion_factor(phx.units.PASCAL, units.pressure_unit)
    )
    thermochemistry = phx.chemistry.HarmonicThermochemistryPlan(
        system,
        298.15,
        pressure,
        symmetry_number=2,
        electronic_degeneracy=1,
    ).evaluate(
        optimized.final_structure,
        vibration,
        optimized.final_evaluation.energy,
    )
    molar = phx.chemistry.to_molar_thermochemistry(
        thermochemistry, phx.units.KILOJOULE_PER_MOLE
    )

    assert bool(thermochemistry.successful)
    assert np.isfinite(float(thermochemistry.gibbs_energy))
    assert np.isfinite(float(molar.gibbs_energy))
    assert molar.energy_unit == phx.units.KILOJOULE_PER_MOLE


def test_native_atomistic_program_adapts_to_the_same_surface_contract():
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    system = phx.atomistic.AtomisticSystemPlan(
        [1, 2],
        [1, 1],
        [1.0, 1.0],
        units,
        atom_type_ids=[0, 0],
        molecule_ids=[0, 0],
    ).prepare()
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1).prepare(
        system.particles
    )
    potential = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.LennardJonesPotential([0.2], [1.0], 2.5)]
    ).prepare(system)
    surface = phx.chemistry.AtomisticPotentialEnergySurface(
        system, potential, neighborhood
    )
    result = surface.evaluate(np.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]))

    assert bool(result.successful)
    assert np.isfinite(float(result.energy))
    assert np.all(np.isfinite(np.asarray(result.forces)))
