from typing import Any

import numpy as np
import pytest

import phydrax as phx


def _units() -> Any:
    return phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()


def _system(numbers: Any, masses: Any, *, charges: Any = None) -> Any:
    units = _units()
    return phx.atomistic.AtomisticSystemPlan(
        np.arange(len(numbers), dtype=np.int64),
        numbers,
        masses,
        units,
        charges=charges,
        molecule_ids=np.zeros(len(numbers), dtype=np.int32),
    )


def _model() -> Any:
    return phx.chemistry.ElectronicModelChemistryPlan(
        phx.chemistry.HartreeFockMethodPlan(
            phx.chemistry.ElectronicReferenceKind.RESTRICTED
        ),
        basis=phx.chemistry.BasisSetReference("sto-3g", "provider-library"),
    )


def test_contracts_scenario_1() -> None:
    neutral = _system([8, 1], [15.999, 1.008], charges=[7.5, -7.5])
    prepared = phx.chemistry.MolecularElectronicSectorPlan(0, 2).prepare(neutral)

    assert prepared.electron_count == 9
    assert prepared.alpha_electron_count == 5
    assert prepared.beta_electron_count == 4

    charged = phx.chemistry.MolecularElectronicSectorPlan(1, 1).prepare(neutral)
    assert charged.electron_count == 8
    assert charged.alpha_electron_count == charged.beta_electron_count == 4
    hydrogen = _system([1], [1.008])
    with pytest.raises(ValueError, match="incompatible parity"):
        phx.chemistry.MolecularElectronicSectorPlan(0, 1).prepare(hydrogen)
    radical = _system([8, 1], [15.999, 1.008])
    state = phx.chemistry.MolecularElectronicSectorPlan(0, 2)
    request = phx.chemistry.GroundStateTaskPlan.energy_and_forces()
    with pytest.raises(ValueError, match="equal alpha and beta"):
        phx.chemistry.ElectronicCalculationPlan(radical, state, _model(), request)

    unrestricted = phx.chemistry.ElectronicModelChemistryPlan(
        phx.chemistry.HartreeFockMethodPlan(
            phx.chemistry.ElectronicReferenceKind.UNRESTRICTED
        ),
        basis=phx.chemistry.BasisSetReference("sto-3g", "provider-library"),
    )
    calculation = phx.chemistry.ElectronicCalculationPlan(
        radical, state, unrestricted, request
    )
    assert calculation.state.alpha_electron_count == 5
    assert calculation.state.beta_electron_count == 4


def test_contracts_scenario_2() -> None:
    system = _system([1, 1], [1.008, 1.008])
    state = phx.chemistry.MolecularElectronicSectorPlan(0, 1)
    request = phx.chemistry.GroundStateTaskPlan.energy_and_forces()
    calculation = phx.chemistry.ElectronicCalculationPlan(
        system, state, _model(), request
    )
    capabilities = phx.chemistry.ElectronicProviderCapabilities.molecular_ground_state(
        (
            phx.chemistry.ElectronicProperty.ENERGY,
            phx.chemistry.ElectronicProperty.FORCES,
        ),
        (phx.chemistry.ElectronicReferenceKind.RESTRICTED,),
    )

    def unused(*_: Any) -> None:
        raise AssertionError("preparation must not evaluate the provider")

    # ty: ignore[invalid-argument-type]
    first = phx.chemistry.CallableElectronicProvider(unused, "first", capabilities)
    # ty: ignore[invalid-argument-type]
    second = phx.chemistry.CallableElectronicProvider(unused, "second", capabilities)

    assert calculation.model_chemistry.model_chemistry_id == _model().model_chemistry_id
    assert first.provider_id != second.provider_id
    assert (
        first.prepare(calculation).prepared_id != second.prepare(calculation).prepared_id
    )
    with pytest.raises(ValueError, match="also request forces"):
        phx.chemistry.GroundStateTaskPlan(
            (
                phx.chemistry.ElectronicProperty.ENERGY,
                phx.chemistry.ElectronicProperty.HESSIAN,
            )
        )
    system = _system([1, 1], [1.008, 1.008])
    calculation = phx.chemistry.ElectronicCalculationPlan(
        system,
        phx.chemistry.MolecularElectronicSectorPlan(0, 1),
        _model(),
        phx.chemistry.GroundStateTaskPlan.energy_forces_and_hessian(),
    )
    capabilities = phx.chemistry.ElectronicProviderCapabilities.molecular_ground_state(
        (
            phx.chemistry.ElectronicProperty.ENERGY,
            phx.chemistry.ElectronicProperty.FORCES,
        ),
        (phx.chemistry.ElectronicReferenceKind.RESTRICTED,),
    )
    with pytest.raises(phx.chemistry.ElectronicCapabilityError):
        capabilities.require(calculation)
    forward = phx.atomistic.single_system_energy_to_molar_factor(
        phx.units.ELECTRONVOLT,
        phx.units.KILOJOULE_PER_MOLE,
        constant_set_id="codata-2018",
    )
    inverse = phx.atomistic.molar_energy_to_single_system_factor(
        phx.units.KILOJOULE_PER_MOLE,
        phx.units.ELECTRONVOLT,
        constant_set_id="codata-2018",
    )

    np.testing.assert_allclose(forward * inverse, 1.0, rtol=1.0e-15)
    assert phx.units.INVERSE_CENTIMETER.dimension == phx.units.LENGTH**-1


def test_contracts_scenario_3() -> None:
    with pytest.raises(TypeError, match="state_index"):
        phx.chemistry.MolecularElectronicSectorPlan(
            0,
            1,
            # ty: ignore[unknown-argument]
            state_index=1,
        )
    system = _system([1, 1], [1.008, 1.008])
    calculation = phx.chemistry.ElectronicCalculationPlan(
        system,
        phx.chemistry.MolecularElectronicSectorPlan(0, 1),
        _model(),
        phx.chemistry.GroundStateTaskPlan.energy_and_forces(),
    )
    capabilities = phx.chemistry.ElectronicProviderCapabilities.molecular_ground_state(
        (
            phx.chemistry.ElectronicProperty.ENERGY,
            phx.chemistry.ElectronicProperty.FORCES,
        ),
        (phx.chemistry.ElectronicReferenceKind.RESTRICTED,),
    )

    def unexpected(*_: Any) -> None:
        raise AssertionError("Unsupported context must not invoke the evaluator.")

    prepared = phx.chemistry.CallableElectronicProvider(
        # ty: ignore[invalid-argument-type]
        unexpected,
        "context-fixture",
        capabilities,
    ).prepare(calculation)
    context = phx.chemistry.ElectronicEvaluationContext(
        # ty: ignore[invalid-argument-type]
        [[0.0, 0.0, -0.35], [0.0, 0.0, 0.35]],
        external_field=phx.chemistry.ExternalFieldState(
            # ty: ignore[invalid-argument-type]
            [0.0, 0.0, 0.01],
            system.units,
        ),
    )

    with pytest.raises(phx.chemistry.ElectronicCapabilityError):
        prepared.evaluate_context(context)
