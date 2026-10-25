#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Optional RDKit graph admission: declared losses, stereo conversion and refusals.

RDKit is an optional host parser.  Where it is installed it also serves as an
isolated test oracle: an RDKit-embedded geometry checks that the chiral-tag
conversion agrees with stereo measured from coordinates.  RDKit coordinates
never enter the native conformer workflow.
"""

import importlib.util

import pytest

from phydrax.atomistic.crystal._atom_types import molecular_correspondences
from phydrax.atomistic.crystal._input import ElementTable
from phydrax.atomistic.interchange._rdkit import (
    is_rdkit_available,
    molecular_component_from_rdkit,
    molecular_component_from_smiles,
    require_rdkit,
)
from phydrax.interchange import AdapterError, AdapterStatus


MASSES = ElementTable(
    "mass",
    {1: 1.008, 6: 12.011, 8: 15.999, 9: 18.998, 17: 35.45, 35: 79.904},
    provenance="CIAAW 2021 conventional standard atomic weights",
)
rdkit_installed = importlib.util.find_spec("rdkit") is not None


@pytest.mark.skipif(rdkit_installed, reason="RDKit is installed")
def test_missing_rdkit_is_an_explicit_optional_dependency_status() -> None:
    assert not is_rdkit_available()
    with pytest.raises(AdapterError) as caught:
        require_rdkit()
    assert caught.value.status == AdapterStatus.OPTIONAL_DEPENDENCY_UNAVAILABLE


def test_smiles_admission_declares_synthesized_hydrogens() -> None:
    pytest.importorskip("rdkit")
    component, molecule, report = molecular_component_from_smiles(
        "OCC", masses=MASSES, spin_multiplicity=1
    )
    assert component.atom_count == 9
    assert molecule.GetNumAtoms() == 9
    assert report.status == AdapterStatus.DECLARED_LOSS
    assert {loss.path for loss in report.losses} == {"hydrogens"}
    assert component.atom_classes.count(component.atom_classes[0]) == 1


def test_tetrahedral_tags_agree_with_measured_geometry() -> None:
    pytest.importorskip("rdkit")
    chem = require_rdkit()
    from rdkit.Chem import rdDistGeom

    right, _, _ = molecular_component_from_smiles(
        "F[C@](Cl)(Br)[H]", masses=MASSES, spin_multiplicity=1
    )
    left, _, _ = molecular_component_from_smiles(
        "F[C@@](Cl)(Br)[H]", masses=MASSES, spin_multiplicity=1
    )
    assert right.stereo_chiral and right.component_id != left.component_id
    mirror = molecular_correspondences(
        right, left, max_mappings=2, max_nodes=10_000, mirror=True
    )
    assert mirror.status == "complete"
    oracle = chem.AddHs(chem.MolFromSmiles("F[C@](Cl)(Br)[H]"))
    assert rdDistGeom.EmbedMolecule(oracle, randomSeed=7) == 0
    measured, _, report = molecular_component_from_rdkit(
        oracle, masses=MASSES, spin_multiplicity=1, stereo_source="conformer"
    )
    assert measured.component_id == right.component_id
    assert any("source conformer" in item for item in report.assumptions)


def test_cip_labeled_double_bonds_need_an_explicit_conversion() -> None:
    pytest.importorskip("rdkit")
    with pytest.raises(AdapterError) as caught:
        molecular_component_from_smiles("F/C=C/F", masses=MASSES, spin_multiplicity=1)
    assert caught.value.status == AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
    trans, _, report = molecular_component_from_smiles(
        "F/C=C/F",
        masses=MASSES,
        spin_multiplicity=1,
        ez_conversion="legacy-cip-rank",
    )
    # Heavy atoms F0, C1, C2, F3 precede the added hydrogens; the fluorines are
    # the smallest substituents of each end and lie trans.
    assert trans.double_bond_stereo == ((0, 1, 2, 3, -1),)
    assert any("legacy CIP" in item for item in report.assumptions)


def test_implicit_hydrogens_are_refused_without_an_explicit_request() -> None:
    pytest.importorskip("rdkit")
    chem = require_rdkit()
    with pytest.raises(AdapterError) as caught:
        molecular_component_from_rdkit(
            chem.MolFromSmiles("CO"), masses=MASSES, spin_multiplicity=1
        )
    assert caught.value.status == AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
