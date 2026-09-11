import numpy as np
import pytest

import phydrax as phx


def _units():
    return phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()


def _system(numbers, masses, *, charges=None):
    units = _units()
    return phx.atomistic.AtomisticSystemPlan(
        np.arange(len(numbers), dtype=np.int64),
        numbers,
        masses,
        units,
        charges=charges,
        molecule_ids=np.zeros(len(numbers), dtype=np.int32),
    )


def _model():
    return phx.chemistry.ElectronicModelChemistryPlan(
        phx.chemistry.ElectronicMethodPlan(
            phx.chemistry.ElectronicMethodFamily.HARTREE_FOCK,
            "hf",
            phx.chemistry.ElectronicReferenceKind.RESTRICTED,
        ),
        basis=phx.chemistry.BasisSetReference("sto-3g", "provider-library"),
    )


def test_electronic_state_derives_spin_population_without_using_site_charges():
    neutral = _system([8, 1], [15.999, 1.008], charges=[7.5, -7.5])
    prepared = phx.chemistry.MolecularElectronicStatePlan(0, 2).prepare(neutral)

    assert prepared.electron_count == 9
    assert prepared.alpha_electron_count == 5
    assert prepared.beta_electron_count == 4

    charged = phx.chemistry.MolecularElectronicStatePlan(1, 1).prepare(neutral)
    assert charged.electron_count == 8
    assert charged.alpha_electron_count == charged.beta_electron_count == 4


def test_impossible_electron_spin_parity_is_rejected():
    hydrogen = _system([1], [1.008])
    with pytest.raises(ValueError, match="incompatible parity"):
        phx.chemistry.MolecularElectronicStatePlan(0, 1).prepare(hydrogen)


def test_open_shell_state_requires_an_open_shell_reference():
    radical = _system([8, 1], [15.999, 1.008])
    state = phx.chemistry.MolecularElectronicStatePlan(0, 2)
    request = phx.chemistry.ElectronicPropertyRequest.energy_and_forces()
    with pytest.raises(ValueError, match="equal alpha and beta"):
        phx.chemistry.ElectronicCalculationPlan(radical, state, _model(), request)

    unrestricted = phx.chemistry.ElectronicModelChemistryPlan(
        phx.chemistry.ElectronicMethodPlan(
            phx.chemistry.ElectronicMethodFamily.HARTREE_FOCK,
            "hf",
            phx.chemistry.ElectronicReferenceKind.UNRESTRICTED,
        ),
        basis=phx.chemistry.BasisSetReference("sto-3g", "provider-library"),
    )
    calculation = phx.chemistry.ElectronicCalculationPlan(
        radical, state, unrestricted, request
    )
    assert calculation.state.alpha_electron_count == 5
    assert calculation.state.beta_electron_count == 4


def test_model_provider_and_property_identities_are_independent():
    system = _system([1, 1], [1.008, 1.008])
    state = phx.chemistry.MolecularElectronicStatePlan(0, 1)
    request = phx.chemistry.ElectronicPropertyRequest.energy_and_forces()
    calculation = phx.chemistry.ElectronicCalculationPlan(system, state, _model(), request)
    capabilities = phx.chemistry.ElectronicProviderCapabilities(
        (phx.chemistry.ElectronicProperty.ENERGY, phx.chemistry.ElectronicProperty.FORCES),
        (phx.chemistry.ElectronicReferenceKind.RESTRICTED,),
    )

    def unused(*_):
        raise AssertionError("preparation must not evaluate the provider")

    first = phx.chemistry.CallableElectronicProvider(unused, "first", capabilities)
    second = phx.chemistry.CallableElectronicProvider(unused, "second", capabilities)

    assert calculation.model_chemistry.model_chemistry_id == _model().model_chemistry_id
    assert first.provider_id != second.provider_id
    assert first.prepare(calculation).prepared_id != second.prepare(calculation).prepared_id


def test_property_and_provider_capability_mismatches_fail_before_execution():
    with pytest.raises(ValueError, match="also request forces"):
        phx.chemistry.ElectronicPropertyRequest(
            (
                phx.chemistry.ElectronicProperty.ENERGY,
                phx.chemistry.ElectronicProperty.HESSIAN,
            )
        )
    system = _system([1, 1], [1.008, 1.008])
    calculation = phx.chemistry.ElectronicCalculationPlan(
        system,
        phx.chemistry.MolecularElectronicStatePlan(0, 1),
        _model(),
        phx.chemistry.ElectronicPropertyRequest.energy_forces_and_hessian(),
    )
    capabilities = phx.chemistry.ElectronicProviderCapabilities(
        (phx.chemistry.ElectronicProperty.ENERGY, phx.chemistry.ElectronicProperty.FORCES),
        (phx.chemistry.ElectronicReferenceKind.RESTRICTED,),
    )
    with pytest.raises(phx.chemistry.ElectronicCapabilityError, match="hessian"):
        capabilities.require(calculation)


def test_single_system_and_molar_energy_conversion_remain_explicit_inverses():
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
