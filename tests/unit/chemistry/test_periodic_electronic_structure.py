import numpy as np
import pytest

import phydrax as phx


def _cell():
    return phx.discretization.PeriodicCell(
        5.0 * np.eye(3),
        periodic_axes=(True, True, True),
    )


def test_gamma_point_native_periodic_scf_closes_population_and_energy():
    model = phx.chemistry.PeriodicAOModelPlan(
        [[0, 0, 0]],
        [[[-1.0]]],
        [[[1.0]]],
        [0.0],
        [2.0],
        0.0,
        phx.units.ELECTRONVOLT,
    )
    plan = phx.chemistry.NativePeriodicSCFPlan(
        _cell(),
        phx.chemistry.KPointMeshPlan.monkhorst_pack((1, 1, 1)),
        phx.chemistry.PeriodicElectronicSectorPlan(2.0),
        model,
    )
    result = plan.evaluate()

    assert bool(result.successful)
    np.testing.assert_allclose(result.energy, -2.0, atol=1.0e-12)
    np.testing.assert_allclose(result.occupations, [[2.0]])
    np.testing.assert_allclose(result.populations, [2.0], atol=1.0e-12)
    np.testing.assert_allclose(result.free_energy, result.energy)


def test_k_point_smearing_preserves_fractional_electron_count():
    model = phx.chemistry.PeriodicAOModelPlan(
        [[-1, 0, 0], [0, 0, 0], [1, 0, 0]],
        [[[-0.2]], [[-1.0]], [[-0.2]]],
        [[[0.0]], [[1.0]], [[0.0]]],
        [0.0],
        [1.0],
        0.0,
        phx.units.ELECTRONVOLT,
    )
    mesh = phx.chemistry.KPointMeshPlan.monkhorst_pack((4, 1, 1))
    result = phx.chemistry.NativePeriodicSCFPlan(
        _cell(),
        mesh,
        phx.chemistry.PeriodicElectronicSectorPlan(1.0),
        model,
        smearing_energy=0.05,
    ).evaluate()

    electron_count = float(
        np.sum(np.asarray(mesh.weights)[:, None] * np.asarray(result.occupations))
    )
    assert bool(result.successful)
    np.testing.assert_allclose(electron_count, 1.0, atol=1.0e-10)
    assert float(result.entropy) > 0.0
    assert float(result.free_energy) < float(result.energy)


def test_zero_smearing_rejects_overlapping_periodic_bands():
    model = phx.chemistry.PeriodicAOModelPlan(
        [[-1, 0, 0], [0, 0, 0], [1, 0, 0]],
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[-1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        [0.0, 0.0],
        [2.0, 0.0],
        0.0,
        phx.units.ELECTRONVOLT,
    )
    mesh = phx.chemistry.KPointMeshPlan(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        [0.5, 0.5],
        mesh_shape=(2, 1, 1),
        shift=(0.0, 0.0, 0.0),
    )
    plan = phx.chemistry.NativePeriodicSCFPlan(
        _cell(),
        mesh,
        phx.chemistry.PeriodicElectronicSectorPlan(2.0),
        model,
    )

    with pytest.raises(ValueError, match="insulating band gap"):
        plan.evaluate()


def test_periodic_result_payload_roundtrips_through_production_archive(tmp_path):
    model = phx.chemistry.PeriodicAOModelPlan(
        [[0, 0, 0]],
        [[[-1.0]]],
        [[[1.0]]],
        [0.0],
        [2.0],
        0.0,
        phx.units.ELECTRONVOLT,
    )
    result = phx.chemistry.NativePeriodicSCFPlan(
        _cell(),
        phx.chemistry.KPointMeshPlan.monkhorst_pack((1, 1, 1)),
        phx.chemistry.PeriodicElectronicSectorPlan(2.0),
        model,
    ).evaluate()
    archive_plan = phx.chemistry.ProductionChemistryArchivePlan(
        "periodic-scf",
        {
            "energy": phx.units.ELECTRONVOLT,
            "occupations": "1",
            "orbital_energies": phx.units.ELECTRONVOLT,
        },
        result.model_id,
    )
    path = tmp_path / "periodic-scf.phx"
    archive_plan.write(
        path,
        result.result_id,
        "periodic-run",
        {
            "energy": result.energy,
            "occupations": result.occupations,
            "orbital_energies": result.orbital_energies,
        },
    )
    restored = archive_plan.open(path, expected_result_id=result.result_id)

    np.testing.assert_allclose(restored.arrays["energy"], result.energy)
    np.testing.assert_allclose(restored.arrays["occupations"], result.occupations)
