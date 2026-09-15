import numpy as np
import pytest

import phydrax as phx


def _calculation(task):
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    cell = phx.discretization.PeriodicCell(
        5.0 * np.eye(3), periodic_axes=(True, True, True)
    )
    system = phx.atomistic.AtomisticSystemPlan(
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([1, 1], dtype=np.int32),
        np.asarray([1.008, 1.008]),
        units,
        molecule_ids=np.zeros(2, dtype=np.int32),
        cell=cell,
    )
    model = phx.chemistry.ElectronicModelChemistryPlan(
        phx.chemistry.HartreeFockMethodPlan(
            phx.chemistry.ElectronicReferenceKind.RESTRICTED
        ),
        basis=phx.chemistry.BasisSetReference("periodic-fixture", "test"),
    )
    return phx.chemistry.ElectronicCalculationPlan(
        system,
        phx.chemistry.PeriodicElectronicSectorPlan(2.0),
        model,
        task,
    )


def _capabilities(task, properties):
    return phx.chemistry.ElectronicProviderCapabilities(
        phx.chemistry.ElectronicTheoryCapabilities(
            (phx.chemistry.ElectronicMethodFamily.HARTREE_FOCK,),
            (phx.chemistry.ElectronicReferenceKind.RESTRICTED,),
        ),
        phx.chemistry.ElectronicGeometryCapabilities(
            finite=False, periodic_ranks=(3,), variable_cell=True
        ),
        phx.chemistry.ElectronicObservableCapabilities(
            (task.task_kind,), properties, derivative_orders=(0, 1)
        ),
    )


def test_generic_provider_returns_complete_periodic_ground_state_payload():
    properties = (
        phx.chemistry.ElectronicProperty.ENERGY,
        phx.chemistry.ElectronicProperty.FORCES,
        phx.chemistry.ElectronicProperty.STRESS,
        phx.chemistry.ElectronicProperty.DENSITY_MATRIX,
        phx.chemistry.ElectronicProperty.POLARIZATION,
    )
    task = phx.chemistry.GroundStateTaskPlan(properties)
    calculation = _calculation(task)

    def evaluate(plan, positions, cell_vectors):
        return phx.chemistry.make_electronic_evaluation(
            plan,
            "periodic-provider",
            positions,
            -1.0,
            forces=np.zeros((2, 3)),
            stress=np.diag([1.0, 2.0, 3.0]),
            density_matrices=np.asarray([[[2.0]]]),
            polarization=np.asarray([0.1, 0.2, 0.3]),
            cell_vectors=cell_vectors,
        )

    provider = phx.chemistry.CallableElectronicProvider(
        evaluate,
        "periodic-provider",
        _capabilities(task, properties),
    )
    prepared = provider.prepare(calculation)
    result = prepared.evaluate(np.zeros((2, 3)), 5.0 * np.eye(3))

    assert isinstance(result, phx.chemistry.ElectronicPeriodicEvaluation)
    assert bool(result.successful)
    np.testing.assert_allclose(result.stress, np.diag([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(result.density_matrices, [[[2.0]]])
    assert result.header.provider_id == "periodic-provider"
    assert result.header.geometry_id


def test_generic_provider_rejects_incomplete_periodic_payload():
    properties = (
        phx.chemistry.ElectronicProperty.ENERGY,
        phx.chemistry.ElectronicProperty.STRESS,
    )
    task = phx.chemistry.GroundStateTaskPlan(properties)
    calculation = _calculation(task)

    with pytest.raises(ValueError, match="omitted requested properties"):
        phx.chemistry.make_electronic_evaluation(
            calculation,
            "periodic-provider",
            np.zeros((2, 3)),
            -1.0,
            cell_vectors=5.0 * np.eye(3),
        )


def test_generic_provider_returns_band_structure_payload():
    task = phx.chemistry.BandStructureTaskPlan(
        np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    )
    calculation = _calculation(task)
    bands = np.asarray([[-1.0, 1.0], [-0.5, 0.5]])

    result = phx.chemistry.make_electronic_evaluation(
        calculation,
        "periodic-provider",
        np.zeros((2, 3)),
        -1.0,
        band_energies=bands,
        cell_vectors=5.0 * np.eye(3),
    )

    assert isinstance(result, phx.chemistry.ElectronicPeriodicEvaluation)
    np.testing.assert_allclose(result.band_energies, bands)
