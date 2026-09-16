import numpy as np
import pytest

import phydrax as phx
from phydrax.chemistry._state import PeriodicElectronicSectorPlan
from phydrax.chemistry.periodic._model_scf import NativePeriodicSCFPlan
from phydrax.chemistry.periodic._orbital_model import (
    PeriodicBlochGauge,
    PeriodicHubbardMeanFieldPlan,
    PeriodicOrbitalBasisPlan,
    PeriodicOrbitalPencilPlan,
)
from phydrax.discretization import ReciprocalMeshPlan
from phydrax.operators.periodic import (
    periodic_translation_family_from_dense_blocks,
    PeriodicFourierConvention,
)


def _cell():
    return phx.discretization.PeriodicCell(
        5.0 * np.eye(3),
        periodic_axes=(True, True, True),
    )


def _model(translations, hamiltonian_blocks, overlap_blocks, hubbard, reference):
    cell = _cell()
    orbitals = len(hubbard)
    basis = PeriodicOrbitalBasisPlan(
        cell,
        tuple(f"ao-{index}" for index in range(orbitals)),
        np.zeros((orbitals, 3)),
        phx.units.ANGSTROM,
        PeriodicBlochGauge("lattice"),
    )
    h = periodic_translation_family_from_dense_blocks(
        translations,
        np.asarray(hamiltonian_blocks).reshape((-1, orbitals, 1, orbitals, 1)),
    )
    s = periodic_translation_family_from_dense_blocks(
        translations,
        np.asarray(overlap_blocks).reshape((-1, orbitals, 1, orbitals, 1)),
    )
    pencil = PeriodicOrbitalPencilPlan(
        basis, h.plan, h.state, s.plan, s.state, phx.units.ELECTRONVOLT
    ).prepare()
    mean_field = PeriodicHubbardMeanFieldPlan(
        basis, hubbard, reference, 0.0, phx.units.ELECTRONVOLT
    )
    return cell, pencil, mean_field


def test_orbital_contract_rejects_unit_and_fourier_mismatches():
    cell = _cell()
    with pytest.raises(ValueError, match="length dimension"):
        PeriodicOrbitalBasisPlan(
            cell,
            ("ao",),
            [[0.0, 0.0, 0.0]],
            phx.units.ELECTRONVOLT,
            PeriodicBlochGauge("lattice"),
        )

    basis = PeriodicOrbitalBasisPlan(
        cell,
        ("ao",),
        [[0.0, 0.0, 0.0]],
        phx.units.ANGSTROM,
        PeriodicBlochGauge("lattice"),
    )
    block = np.ones((1, 1, 1, 1, 1))
    hamiltonian = periodic_translation_family_from_dense_blocks(
        [[0, 0, 0]],
        block,
        convention=PeriodicFourierConvention(1),
    )
    overlap = periodic_translation_family_from_dense_blocks(
        [[0, 0, 0]],
        block,
        convention=PeriodicFourierConvention(-1),
    )
    with pytest.raises(ValueError, match="one Fourier convention"):
        PeriodicOrbitalPencilPlan(
            basis,
            hamiltonian.plan,
            hamiltonian.state,
            overlap.plan,
            overlap.state,
            phx.units.ELECTRONVOLT,
        )

    with pytest.raises(ValueError, match="energy dimension"):
        PeriodicOrbitalPencilPlan.orthonormal(
            basis,
            hamiltonian.plan,
            hamiltonian.state,
            phx.units.ANGSTROM,
        )
    with pytest.raises(ValueError, match="energy dimension"):
        PeriodicHubbardMeanFieldPlan(
            basis,
            [0.0],
            [0.0],
            0.0,
            phx.units.ANGSTROM,
        )


def test_gamma_point_native_periodic_scf_closes_population_and_energy():
    cell, pencil, mean_field = _model([[0, 0, 0]], [[[-1.0]]], [[[1.0]]], [0.0], [2.0])
    mesh = ReciprocalMeshPlan.monkhorst_pack(cell, (1, 1, 1))
    result = NativePeriodicSCFPlan(
        cell, mesh, PeriodicElectronicSectorPlan(2.0), pencil, mean_field
    ).evaluate()

    assert bool(result.successful)
    np.testing.assert_allclose(result.energy, -2.0, atol=1.0e-12)
    np.testing.assert_allclose(result.occupations, [[2.0]])
    np.testing.assert_allclose(result.populations, [2.0], atol=1.0e-12)
    np.testing.assert_allclose(result.free_energy, result.energy)


def test_k_point_smearing_preserves_fractional_electron_count():
    cell, pencil, mean_field = _model(
        [[-1, 0, 0], [0, 0, 0], [1, 0, 0]],
        [[[-0.2]], [[-1.0]], [[-0.2]]],
        [[[0.0]], [[1.0]], [[0.0]]],
        [0.0],
        [1.0],
    )
    mesh = ReciprocalMeshPlan.monkhorst_pack(cell, (4, 1, 1))
    result = NativePeriodicSCFPlan(
        cell,
        mesh,
        PeriodicElectronicSectorPlan(1.0),
        pencil,
        mean_field,
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
    cell, pencil, mean_field = _model(
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
    )
    mesh = ReciprocalMeshPlan(
        cell,
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        [0.5, 0.5],
        mesh_shape=(2, 1, 1),
        shift=(0.0, 0.0, 0.0),
    )
    plan = NativePeriodicSCFPlan(
        cell, mesh, PeriodicElectronicSectorPlan(2.0), pencil, mean_field
    )

    with pytest.raises(ValueError, match="insulating band gap"):
        plan.evaluate()


def test_periodic_result_payload_roundtrips_through_production_archive(tmp_path):
    cell, pencil, mean_field = _model([[0, 0, 0]], [[[-1.0]]], [[[1.0]]], [0.0], [2.0])
    mesh = ReciprocalMeshPlan.monkhorst_pack(cell, (1, 1, 1))
    result = NativePeriodicSCFPlan(
        cell, mesh, PeriodicElectronicSectorPlan(2.0), pencil, mean_field
    ).evaluate()
    archive_plan = phx.chemistry.ProductionChemistryArchivePlan(
        "periodic-scf",
        {
            "energy": phx.units.ELECTRONVOLT,
            "occupations": "1",
            "orbital_energies": phx.units.ELECTRONVOLT,
        },
        result.pencil_id,
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
