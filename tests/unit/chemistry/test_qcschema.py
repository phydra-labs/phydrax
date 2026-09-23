import numpy as np
import pytest

import phydrax as phx


def _calculation():
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    system = phx.atomistic.AtomisticSystemPlan(
        [41, 73],
        [1, 1],
        [1.008, 1.008],
        units,
        molecule_ids=[0, 0],
    )
    state = phx.chemistry.MolecularElectronicSectorPlan(0, 1)
    model = phx.chemistry.ElectronicModelChemistryPlan(
        phx.chemistry.HartreeFockMethodPlan(
            phx.chemistry.ElectronicReferenceKind.RESTRICTED
        ),
        basis=phx.chemistry.BasisSetReference("sto-3g", "provider-library"),
    )
    return phx.chemistry.ElectronicCalculationPlan(
        system,
        state,
        model,
        phx.chemistry.GroundStateTaskPlan.energy_and_forces(),
    )


def test_qcschema_roundtrip_preserves_state_order_units_and_force_sign():
    calculation = _calculation()
    positions = np.asarray([[0.0, 0.0, -0.35], [0.0, 0.0, 0.35]])
    payload, exported = phx.chemistry.interchange.electronic_calculation_to_qcschema(
        calculation, positions
    )

    assert exported.status == phx.interchange.AdapterStatus.LOSSLESS
    assert payload["driver"] == "gradient"
    assert payload["molecule"]["molecular_charge"] == 0.0
    assert payload["molecule"]["molecular_multiplicity"] == 1
    assert payload["molecule"]["fix_com"] is True
    assert payload["molecule"]["fix_orientation"] is True
    assert payload["molecule"]["extras"]["phydrax"]["stable_particle_ids"] == [41, 73]

    gradient = np.asarray([[0.0, 0.0, 0.02], [0.0, 0.0, -0.02]])
    record = {
        "success": True,
        "return_result": gradient.tolist(),
        "properties": {"return_energy": -1.1},
        "molecule": payload["molecule"],
        "extras": payload["extras"],
        "provenance": {"creator": "analytic-qcschema-fixture", "version": "1"},
    }
    result, imported = phx.chemistry.interchange.electronic_evaluation_from_qcschema(
        calculation,
        "qcschema-fixture",
        positions,
        record,
    )

    energy_factor = float(
        phx.units.conversion_factor(phx.units.HARTREE, phx.units.ELECTRONVOLT)
    )
    length_factor = float(phx.units.conversion_factor(phx.units.BOHR, phx.units.ANGSTROM))
    np.testing.assert_allclose(result.energy, -1.1 * energy_factor)
    np.testing.assert_allclose(result.forces, -gradient * energy_factor / length_factor)
    assert imported.status == phx.interchange.AdapterStatus.LOSSLESS
    assert result.header.geometry_id == phx.chemistry.electronic_geometry_id(
        calculation.system, positions
    )


def test_qcschema_missing_native_identity_fails_closed():
    calculation = _calculation()
    positions = np.asarray([[0.0, 0.0, -0.35], [0.0, 0.0, 0.35]])
    record = {
        "success": True,
        "return_result": np.zeros((2, 3)).tolist(),
        "properties": {"return_energy": -1.0},
        "molecule": {"extras": {}},
        "provenance": {"creator": "fixture", "version": "1"},
    }

    with pytest.raises(ValueError, match="exact calculation"):
        phx.chemistry.interchange.electronic_evaluation_from_qcschema(
            calculation,
            "qcschema-fixture",
            positions,
            record,
        )
