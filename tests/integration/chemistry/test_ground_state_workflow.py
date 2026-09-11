import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _calculation_and_provider():
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    structure = phx.atomistic.AtomicStructure(
        [1, 1],
        [[-0.35, 0.0, 0.0], [0.35, 0.0, 0.0]],
        [1.008, 1.008],
        units.scale,
        particle_ids=[7, 9],
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        structure, units, molecule_ids=[0, 0]
    )
    calculation = phx.chemistry.ElectronicCalculationPlan(
        system,
        phx.chemistry.MolecularElectronicStatePlan(0, 1),
        phx.chemistry.ElectronicModelChemistryPlan(
            phx.chemistry.ElectronicMethodPlan(
                phx.chemistry.ElectronicMethodFamily.CUSTOM_EXTERNAL,
                "harmonic-electronic",
                phx.chemistry.ElectronicReferenceKind.RESTRICTED,
            )
        ),
        phx.chemistry.ElectronicPropertyRequest.energy_and_forces(),
    )

    def evaluate(plan, positions, cell):
        del cell
        coordinate = jnp.asarray(positions)
        energy = 0.5 * jnp.sum(coordinate**2)
        return phx.chemistry.make_electronic_evaluation(
            plan,
            "harmonic-electronic-provider",
            coordinate,
            energy,
            forces=-coordinate,
        )

    capabilities = phx.chemistry.ElectronicProviderCapabilities(
        (phx.chemistry.ElectronicProperty.ENERGY, phx.chemistry.ElectronicProperty.FORCES),
        (phx.chemistry.ElectronicReferenceKind.RESTRICTED,),
    )
    provider = phx.chemistry.CallableElectronicProvider(
        evaluate, "harmonic-electronic-provider", capabilities
    )
    return units, structure, system, calculation, provider


def test_result_archive_and_born_oppenheimer_adapter_roundtrip(tmp_path):
    _, structure, system, calculation, provider = _calculation_and_provider()
    prepared = provider.prepare(calculation)
    result = prepared.evaluate(structure.positions)
    path = tmp_path / "electronic-result.phx"
    phx.chemistry.write_electronic_result_archive(path, result, run_id="fixture-run")
    restored = phx.chemistry.read_electronic_result_archive(path)

    assert restored.result_id == result.result_id
    np.testing.assert_allclose(restored.energy, result.energy)
    np.testing.assert_allclose(restored.forces, result.forces)

    lifecycle = phx.chemistry.chemistry_lifecycle(
        calculation,
        result,
        provider_capability_id=provider.capabilities.capabilities_id,
    )
    assert lifecycle.run.status == "completed"
    assert lifecycle.result.result_id == result.result_id

    surface = phx.chemistry.ElectronicPotentialEnergySurface(prepared)
    adapter = phx.chemistry.SurfaceExternalAtomisticProvider(surface)
    dynamics = phx.atomistic.BornOppenheimerVelocityVerletPlan(
        system.prepare(), adapter, 1.0e-3
    )
    state = dynamics.initialize(structure.positions, velocity=jnp.zeros_like(structure.positions))
    step = dynamics.step(state)

    assert bool(step.successful)
    assert int(step.state.step_index) == 1


def test_real_ase_calculator_executes_through_typed_provider():
    ase = pytest.importorskip("ase")
    from ase.calculators.emt import EMT

    del ase
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    structure = phx.atomistic.AtomicStructure(
        [29, 29],
        [[0.0, 0.0, 0.0], [2.6, 0.0, 0.0]],
        [63.546, 63.546],
        units.scale,
        particle_ids=[21, 22],
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        structure, units, molecule_ids=[0, 0]
    )
    calculation = phx.chemistry.ElectronicCalculationPlan(
        system,
        phx.chemistry.MolecularElectronicStatePlan(0, 1),
        phx.chemistry.ElectronicModelChemistryPlan(
            phx.chemistry.ElectronicMethodPlan(
                phx.chemistry.ElectronicMethodFamily.CUSTOM_EXTERNAL,
                "emt",
                phx.chemistry.ElectronicReferenceKind.RESTRICTED,
            )
        ),
        phx.chemistry.ElectronicPropertyRequest.energy_and_forces(),
    )
    provider = phx.chemistry.interchange.ASECalculatorProvider(
        EMT,
        "ase-emt",
        phx.chemistry.interchange.ASEElectronicStateBinding.invariant(),
        model_chemistry_id=calculation.model_chemistry.model_chemistry_id,
    )
    result = provider.prepare(calculation).evaluate(structure.positions)

    assert bool(result.successful)
    assert np.isfinite(float(result.energy))
    assert np.all(np.isfinite(np.asarray(result.forces)))


def test_real_pyscf_rhf_preserves_native_force_order():
    pytest.importorskip("pyscf")
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    structure = phx.atomistic.AtomicStructure(
        [1, 1],
        [[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]],
        [1.008, 1.008],
        units.scale,
        particle_ids=[31, 37],
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        structure, units, molecule_ids=[0, 0]
    )
    calculation = phx.chemistry.ElectronicCalculationPlan(
        system,
        phx.chemistry.MolecularElectronicStatePlan(0, 1),
        phx.chemistry.ElectronicModelChemistryPlan(
            phx.chemistry.ElectronicMethodPlan(
                phx.chemistry.ElectronicMethodFamily.HARTREE_FOCK,
                "hf",
                phx.chemistry.ElectronicReferenceKind.RESTRICTED,
            ),
            basis=phx.chemistry.BasisSetReference(
                "sto-3g", "pyscf-basis-library"
            ),
        ),
        phx.chemistry.ElectronicPropertyRequest.energy_and_forces(),
    )
    result = phx.chemistry.interchange.PySCFProvider(
        convergence_tolerance=1.0e-11
    ).prepare(calculation).evaluate(structure.positions)

    assert bool(result.successful)
    np.testing.assert_array_equal(result.header.stable_particle_ids, [31, 37])
    np.testing.assert_allclose(
        np.asarray(result.forces)[0],
        -np.asarray(result.forces)[1],
        rtol=1.0e-10,
        atol=1.0e-10,
    )
