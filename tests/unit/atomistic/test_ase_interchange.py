import gc
import weakref

import numpy as np
import pytest

import phydrax as phx
from phydrax.interchange import AdapterError, AdapterStatus, require_lossless
from phydrax.units import ANGSTROM, ELECTRONVOLT, JOULE, METER


SCALE = phx.atomistic.AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)


@pytest.fixture
def ase():
    return pytest.importorskip("ase")


def _identified_atoms(ase, *, cell=None, pbc=False):
    atoms = ase.Atoms(
        numbers=[8, 1, 6],
        positions=[[0.1, 0.2, 0.3], [1.2, -0.4, 0.5], [-0.2, 1.1, 0.7]],
        masses=[15.999, 2.014, 13.003],
        cell=cell,
        pbc=pbc,
        info={phx.atomistic.interchange.ASE_SOURCE_ID_INFO: "source-catalogue-entry"},
    )
    atoms.new_array(
        phx.atomistic.interchange.ASE_PARTICLE_ID_ARRAY,
        np.asarray([71, 19, 44], dtype=np.int64),
    )
    return atoms


def test_nonperiodic_structure_roundtrip_preserves_content_and_provenance(ase):
    cell = np.asarray([[7.0, 0.0, 0.0], [0.2, 8.0, 0.0], [0.1, 0.3, 9.0]])
    source = _identified_atoms(ase, cell=cell, pbc=False)

    structure, imported = phx.atomistic.interchange.from_ase_atoms(source, SCALE)

    assert imported.status == AdapterStatus.LOSSLESS
    assert imported.source_id == "source-catalogue-entry"
    assert imported.target_id == structure.structure_id
    np.testing.assert_array_equal(structure.atomic_numbers, [8, 1, 6])
    np.testing.assert_array_equal(structure.positions, source.positions)
    np.testing.assert_array_equal(structure.masses, source.get_masses())
    np.testing.assert_array_equal(structure.cell, cell)
    np.testing.assert_array_equal(structure.periodic_axes, [False, False, False])
    np.testing.assert_array_equal(structure.particle_ids, [71, 19, 44])

    restored, exported = phx.atomistic.interchange.to_ase_atoms(structure)

    assert exported.status == AdapterStatus.LOSSLESS
    assert restored.calc is None
    assert restored.info[phx.atomistic.interchange.ASE_SOURCE_ID_INFO] == (
        structure.structure_id
    )
    np.testing.assert_array_equal(restored.numbers, source.numbers)
    np.testing.assert_array_equal(restored.positions, source.positions)
    np.testing.assert_array_equal(restored.get_masses(), source.get_masses())
    np.testing.assert_array_equal(restored.cell.array, source.cell.array)
    np.testing.assert_array_equal(restored.pbc, source.pbc)
    np.testing.assert_array_equal(
        restored.arrays[phx.atomistic.interchange.ASE_PARTICLE_ID_ARRAY],
        [71, 19, 44],
    )

    roundtripped, roundtrip_report = phx.atomistic.interchange.from_ase_atoms(
        restored, SCALE
    )
    assert roundtrip_report.status == AdapterStatus.LOSSLESS
    assert roundtrip_report.source_id == structure.structure_id
    assert roundtripped.structure_id == structure.structure_id


def test_triclinic_periodic_cell_is_preserved_exactly(ase):
    cell = np.asarray([[4.1, 0.0, 0.0], [1.2, 3.7, 0.0], [-0.4, 0.8, 5.3]])
    source = _identified_atoms(ase, cell=cell, pbc=True)

    structure, report = phx.atomistic.interchange.from_ase_atoms(source, SCALE)
    restored, _ = phx.atomistic.interchange.to_ase_atoms(structure)

    assert report.status == AdapterStatus.LOSSLESS
    np.testing.assert_array_equal(structure.cell, cell)
    np.testing.assert_array_equal(structure.periodic_axes, [True, True, True])
    np.testing.assert_array_equal(restored.cell.array, cell)
    np.testing.assert_array_equal(restored.pbc, [True, True, True])


def test_partial_periodicity_accepts_independent_periodic_vectors(ase):
    cell = np.asarray([[3.0, 0.0, 0.0], [0.7, 2.6, 0.0], [0.0, 0.0, 0.0]])
    source = _identified_atoms(ase, cell=cell, pbc=[True, True, False])

    structure, report = phx.atomistic.interchange.from_ase_atoms(source, SCALE)
    restored, _ = phx.atomistic.interchange.to_ase_atoms(structure)

    assert report.status == AdapterStatus.LOSSLESS
    np.testing.assert_array_equal(structure.cell, cell)
    np.testing.assert_array_equal(structure.periodic_axes, [True, True, False])
    np.testing.assert_array_equal(restored.cell.array, cell)
    np.testing.assert_array_equal(restored.pbc, [True, True, False])


def test_default_ids_are_deterministic_and_declared_as_synthesized(ase):
    source = ase.Atoms(
        "H2O",
        positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, 0.7, 0.0]],
    )

    first, first_report = phx.atomistic.interchange.from_ase_atoms(source, SCALE)
    second, second_report = phx.atomistic.interchange.from_ase_atoms(source.copy(), SCALE)

    np.testing.assert_array_equal(first.particle_ids, [0, 1, 2])
    np.testing.assert_array_equal(second.particle_ids, [0, 1, 2])
    assert first.cell is None
    assert first.periodic_axes is None
    restored, _ = phx.atomistic.interchange.to_ase_atoms(first)
    np.testing.assert_array_equal(restored.cell.array, np.zeros((3, 3)))
    np.testing.assert_array_equal(restored.pbc, [False, False, False])
    roundtripped, _ = phx.atomistic.interchange.from_ase_atoms(restored, SCALE)
    assert roundtripped.cell is None
    assert roundtripped.periodic_axes is None
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    system = phx.atomistic.AtomisticSystemPlan.from_structure(roundtripped, units)
    assert system.cell is None
    assert first.structure_id == second.structure_id
    assert first_report.source_id == second_report.source_id
    assert first_report.status == AdapterStatus.DECLARED_LOSS
    assert [loss.path for loss in first_report.losses] == [
        f"arrays.{phx.atomistic.interchange.ASE_PARTICLE_ID_ARRAY}"
    ]
    with pytest.raises(AdapterError) as error:
        require_lossless(first_report)
    assert error.value.status == AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC


def test_explicit_ids_follow_exact_atom_reordering(ase):
    source = _identified_atoms(ase)
    original, original_report = phx.atomistic.interchange.from_ase_atoms(source, SCALE)
    order = np.asarray([2, 0, 1])
    reordered, reordered_report = phx.atomistic.interchange.from_ase_atoms(
        source[order], SCALE
    )

    assert original_report.status == AdapterStatus.LOSSLESS
    assert reordered_report.status == AdapterStatus.LOSSLESS
    assert reordered_report.source_id == original_report.source_id
    np.testing.assert_array_equal(reordered.particle_ids, [44, 71, 19])
    np.testing.assert_array_equal(
        reordered.atomic_numbers, np.asarray(original.atomic_numbers)[order]
    )
    np.testing.assert_array_equal(
        reordered.positions, np.asarray(original.positions)[order]
    )
    np.testing.assert_array_equal(reordered.masses, np.asarray(original.masses)[order])
    assert reordered.structure_id != original.structure_id


def test_explicit_source_provenance_is_preserved_and_conflicts_are_rejected(ase):
    source = _identified_atoms(ase)
    source.info.pop(phx.atomistic.interchange.ASE_SOURCE_ID_INFO)

    _, report = phx.atomistic.interchange.from_ase_atoms(
        source, SCALE, source_id="external-relaxation-42"
    )
    assert report.source_id == "external-relaxation-42"

    source.info[phx.atomistic.interchange.ASE_SOURCE_ID_INFO] = "different-source"
    with pytest.raises(AdapterError) as error:
        phx.atomistic.interchange.from_ase_atoms(
            source, SCALE, source_id="external-relaxation-42"
        )
    assert error.value.status == AdapterStatus.INCONSISTENT_SOURCE


def test_unsupported_state_is_fully_declared_and_require_lossless_rejects(ase):
    from ase.constraints import FixAtoms

    source = _identified_atoms(ase)
    source.set_velocities(np.ones((3, 3)))
    source.set_initial_charges([0.2, -0.1, -0.1])
    source.new_array("custom_labels", np.asarray([3, 4, 5], dtype=np.int32))
    source.set_constraint(FixAtoms(indices=[0]))
    source.info["workflow_note"] = object()

    _, report = phx.atomistic.interchange.from_ase_atoms(source, SCALE)

    assert report.status == AdapterStatus.DECLARED_LOSS
    assert {loss.path for loss in report.losses} == {
        "arrays.custom_labels",
        "arrays.initial_charges",
        "arrays.momenta",
        "constraints",
        "info.workflow_note",
    }
    assert all(loss.direction == "import" for loss in report.losses)
    with pytest.raises(AdapterError):
        require_lossless(report)


@pytest.mark.parametrize(
    ("location", "name"),
    [
        ("info", "occupancy"),
        ("info", "topology"),
        ("info", "units"),
        ("array", "spins"),
    ],
)
def test_required_unsupported_semantics_are_rejected(ase, location, name):
    source = _identified_atoms(ase)
    if location == "info":
        source.info[name] = {"opaque": object()}
    else:
        source.new_array(name, np.ones((len(source),), dtype=float))

    with pytest.raises(AdapterError) as error:
        phx.atomistic.interchange.from_ase_atoms(source, SCALE)

    assert error.value.status in (
        AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
        AdapterStatus.INCONSISTENT_SOURCE,
    )


def test_incompatible_units_and_malformed_periodic_cells_are_rejected(ase):
    source = _identified_atoms(
        ase,
        cell=[[2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        pbc=[True, True, False],
    )
    with pytest.raises(AdapterError) as malformed:
        phx.atomistic.interchange.from_ase_atoms(source, SCALE)
    assert malformed.value.status == AdapterStatus.MALFORMED_SOURCE

    valid = _identified_atoms(ase)
    incompatible = phx.atomistic.AtomisticScaleContract(METER, JOULE)
    with pytest.raises(AdapterError) as units:
        phx.atomistic.interchange.from_ase_atoms(valid, incompatible)
    assert units.value.status == AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC


def test_calculator_state_is_reported_but_never_retained(ase):
    from ase.calculators.singlepoint import SinglePointCalculator

    source = _identified_atoms(ase)
    calculator = SinglePointCalculator(source, energy=-1.25)
    calculator_reference = weakref.ref(calculator)
    source.calc = calculator

    structure, report = phx.atomistic.interchange.from_ase_atoms(source, SCALE)
    assert [loss.path for loss in report.losses] == ["calculator"]
    assert report.status == AdapterStatus.DECLARED_LOSS

    restored, _ = phx.atomistic.interchange.to_ase_atoms(structure)
    assert restored.calc is None
    source.calc = None
    del calculator
    gc.collect()
    assert calculator_reference() is None


def test_optional_dependency_failure_and_public_exports(monkeypatch):
    import phydrax.atomistic.interchange._ase as ase_adapter

    monkeypatch.setattr(ase_adapter.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(AdapterError) as error:
        phx.atomistic.interchange.require_ase()
    assert error.value.status == AdapterStatus.OPTIONAL_DEPENDENCY_UNAVAILABLE

    assert {
        "ASE_PARTICLE_ID_ARRAY",
        "ASE_SOURCE_ID_INFO",
        "from_ase_atoms",
        "is_ase_available",
        "require_ase",
        "to_ase_atoms",
    } <= set(phx.atomistic.interchange.__all__)
