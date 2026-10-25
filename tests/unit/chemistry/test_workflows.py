from typing import Any

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint


def _case() -> Any:
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    target = np.asarray(
        [[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.24, 0.92, 0.0]],
        dtype="float64",
    )
    structure = phx.atomistic.AtomicStructure(
        # ty: ignore[invalid-argument-type]
        [8, 1, 1],
        target
        + np.asarray([[0.05, -0.03, 0.02], [0.02, 0.04, -0.01], [-0.03, 0.01, 0.03]]),
        # ty: ignore[invalid-argument-type]
        [15.999, 1.008, 1.008],
        units.scale,
        # ty: ignore[invalid-argument-type]
        particle_ids=[101, 102, 103],
        name="water-fixture",
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        structure, units, molecule_ids=[0, 0, 0]
    )
    stiffness = 2.0

    def evaluate(positions: Any, _cell: Any) -> Any:
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
                # ty: ignore[invalid-argument-type]
                [8, 1, 1],
                coordinate,
                # ty: ignore[invalid-argument-type]
                [15.999, 1.008, 1.008],
                units.scale,
                # ty: ignore[invalid-argument-type]
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


def test_geometry_optimization_preserves_explicit_mass_accumulation_precision() -> None:
    _, target, source, system, surface = _case()
    structure = phx.atomistic.AtomicStructure(
        source.atomic_numbers,
        source.positions,
        source.masses,
        source.scale,
        particle_ids=source.particle_ids,
        name=source.name,
        mass_accumulation_dtype=jnp.float32,
    )
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
    assert (
        optimized.final_structure.particles.mass_measure.total_mass.dtype == jnp.float32
    )


def test_workflows_scenario_1() -> None:
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
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    system = phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [1, 2],
        # ty: ignore[invalid-argument-type]
        [1, 1],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        units,
        # ty: ignore[invalid-argument-type]
        atom_type_ids=[0, 0],
        # ty: ignore[invalid-argument-type]
        molecule_ids=[0, 0],
        topology=phx.atomistic.MolecularTopologyPlan(bonds=[[1, 2]]),
    ).prepare()
    potential = phx.atomistic.AtomisticPotentialProgram(
        # ty: ignore[invalid-argument-type]
        [phx.atomistic.HarmonicBondPotential([4.0], [1.0])]
    ).prepare(
        system,
        neighborhood_bundle=phx.atomistic.AtomisticNeighborhoodBundlePlan((), (), ()),
    )
    surface = phx.chemistry.AtomisticPotentialEnergySurface(system, potential)
    result = surface.evaluate(np.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]))

    assert bool(result.successful)
    assert np.isfinite(float(result.energy))
    assert np.all(np.isfinite(np.asarray(result.forces)))
    np.testing.assert_allclose(
        result.forces, [[0.8, 0.0, 0.0], [-0.8, 0.0, 0.0]], atol=1e-12
    )
    np.testing.assert_allclose(result.energy, 0.5 * 4.0 * 0.2**2, atol=1e-12)
    coordinate = np.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    expected_source = canonical_fingerprint(
        {
            "kind": "atomistic-surface-evaluation",
            "surface": surface.surface_id,
            "arrays": array_tree_fingerprint(
                {
                    "positions": np.asarray(
                        coordinate, dtype=system.plan.coordinate_dtype
                    ),
                    "energy": np.asarray(result.energy),
                    "forces": np.asarray(result.forces),
                }
            ),
        }
    )
    expected_evaluation = canonical_fingerprint(
        {
            "kind": "potential-energy-surface-evaluation",
            "provider": surface.provider_id,
            "source_result": expected_source,
            "successful": True,
            "arrays": array_tree_fingerprint(
                {
                    "energy": np.asarray(result.energy),
                    "forces": np.asarray(result.forces),
                    "hessian": None,
                }
            ),
        }
    )
    assert result.source_result_id == expected_source
    assert result.evaluation_id == expected_evaluation
    repeated = surface.evaluate(coordinate)
    assert repeated.evaluation_id == result.evaluation_id
    translated = surface.evaluate(coordinate + 1.0)
    assert bool(translated.successful)
    assert translated.source_result_id != result.source_result_id
    assert translated.evaluation_id != result.evaluation_id
    changed = surface.evaluate(coordinate * 1.1)
    assert bool(changed.successful)
    assert changed.source_result_id != result.source_result_id
    assert changed.evaluation_id != result.evaluation_id
