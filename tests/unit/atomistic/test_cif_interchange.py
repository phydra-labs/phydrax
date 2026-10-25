#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native CIF admission: exact symmetry, settings, multiplicities, losses and refusals.

Controls: rock-salt Fm-3m generated from its Hall symbol (multiplicity 4 per
site, 192 operations), a P2_1/c block with explicit operations (general versus
inversion-center multiplicity), exact parsing of operation strings and bond
symmetry codes, enumerated losses, and refusal of partial occupancy, disorder,
label-inferred elements, oxidation-state labels, nonclosed operations, Hall
disagreement, overlapping sites, magnetic content and cell/operation metric
inconsistency.
"""

from fractions import Fraction

import numpy as np
import pytest

from phydrax.atomistic.interchange._cif import (
    cif_crystal_structure,
    parse_symmetry_operation,
)
from phydrax.interchange import AdapterError, AdapterStatus


ROCK_SALT = """
data_NaCl
_cell_length_a 5.6402
_cell_length_b 5.6402
_cell_length_c 5.6402
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 225
_space_group_name_Hall '-F 4 2 3'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0 0 0
Cl1 Cl 0.5 0.5 0.5
"""

MONOCLINIC_HEADER = """
data_monoclinic
_cell_length_a 5.0(1)
_cell_length_b 6.0
_cell_length_c 7.0
_cell_angle_alpha 90
_cell_angle_beta 100.0
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1 21/c 1'
loop_
_space_group_symop_operation_xyz
'x, y, z'
'-x, y+1/2, -z+1/2'
'-x, -y, -z'
'x, -y+1/2, z+1/2'
"""

MONOCLINIC_SITES = """
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_U_iso_or_equiv
C1 C 0.1 0.2 0.3 1 0.02
O1 O 0 0 0 1.0 0.03
loop_
_geom_bond_atom_site_label_1
_geom_bond_atom_site_label_2
_geom_bond_site_symmetry_2
C1 O1 2_655
"""


def site_block(rows: str, extra_columns: str = "", extra_items: str = "") -> str:
    return (
        MONOCLINIC_HEADER
        + extra_items
        + "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n"
        + "_atom_site_fract_y\n_atom_site_fract_z\n"
        + extra_columns
        + rows
    )


def refused(text: str) -> AdapterStatus:
    with pytest.raises(AdapterError) as caught:
        cif_crystal_structure(text)
    return caught.value.status


def test_rock_salt_expands_from_its_hall_symbol() -> None:
    structure = cif_crystal_structure(ROCK_SALT)
    assert structure.setting is not None
    assert structure.setting.international_number == 225
    assert len(structure.operations) == 192
    assert structure.site_multiplicities == (4, 4)
    assert structure.expanded_fractional.shape == (8, 3)
    assert sorted(structure.expanded_atomic_numbers.tolist()) == [11] * 4 + [17] * 4
    sodium = structure.expanded_fractional[structure.expanded_site_indices == 0]
    expected = {(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)}
    assert {tuple(np.round(row, 9)) for row in sodium} == expected
    assert structure.report.status == AdapterStatus.LOSSLESS
    assert any("Hall symbol" in item for item in structure.report.assumptions)
    np.testing.assert_allclose(
        np.diag(structure.cell_vectors @ structure.cell_vectors.T), 5.6402**2
    )


def test_explicit_operations_identify_p21c_with_exact_multiplicities() -> None:
    structure = cif_crystal_structure(MONOCLINIC_HEADER + MONOCLINIC_SITES)
    assert structure.setting is not None
    assert structure.setting.international_number == 14
    assert structure.site_multiplicities == (4, 2)
    assert structure.expanded_fractional.shape == (6, 3)
    assert np.all(
        (structure.expanded_fractional >= 0.0) & (structure.expanded_fractional < 1.0)
    )
    (bond,) = structure.bonds
    assert (bond.first, bond.second) == (0, 1)
    assert bond.first_operation == 0 and bond.second_operation == 1
    assert bond.second_translation == (1, 0, 0) and bond.order is None
    assert structure.report.status == AdapterStatus.DECLARED_LOSS
    paths = {loss.path for loss in structure.report.losses}
    assert "standard-uncertainties" in paths
    assert "_symmetry" in paths and "_atom" in paths
    assert structure.cell_lengths == (5.0, 6.0, 7.0)


def test_symmetry_operation_strings_parse_exactly() -> None:
    rotation, translation = parse_symmetry_operation("-x+1/2, y+1/2, -z+1/2")
    assert rotation == (-1, 0, 0, 0, 1, 0, 0, 0, -1)
    assert translation == (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2))
    rotation, translation = parse_symmetry_operation("x-y, x, z+1/6")
    assert rotation == (1, -1, 0, 1, 0, 0, 0, 0, 1)
    assert translation == (Fraction(0), Fraction(0), Fraction(1, 6))
    _, shifted = parse_symmetry_operation("x, y, z-0.25")
    assert shifted == (Fraction(0), Fraction(0), Fraction(3, 4))
    with pytest.raises(AdapterError) as caught:
        parse_symmetry_operation("x, y")
    assert caught.value.status == AdapterStatus.MALFORMED_SOURCE


def test_unresolved_occupancy_and_disorder_are_refused() -> None:
    partial = site_block("C1 C 0.1 0.2 0.3 0.5\n", "_atom_site_occupancy\n")
    assert refused(partial) == AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
    disordered = site_block("C1 C 0.1 0.2 0.3 1\n", "_atom_site_disorder_group\n")
    assert refused(disordered) == AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
    overlapping = site_block("C1 C 0.1 0.2 0.3\nC2 C 0.1 0.2 0.3\n")
    assert refused(overlapping) == AdapterStatus.INCONSISTENT_SOURCE


def test_element_identity_is_never_inferred() -> None:
    unlabeled = (
        MONOCLINIC_HEADER
        + "loop_\n_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n"
        + "_atom_site_fract_z\nC1 0.1 0.2 0.3\n"
    )
    assert refused(unlabeled) == AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
    charged = site_block("O1 O2- 0.1 0.2 0.3\n")
    assert refused(charged) == AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
    dropped = cif_crystal_structure(charged, charged_type_symbols="drop")
    (loss,) = [
        item for item in dropped.report.losses if item.path.startswith("_atom_site_type")
    ]
    assert loss.changes_interpretation
    assert dropped.site_atomic_numbers == (8,)


def test_inconsistent_symmetry_and_cell_are_refused() -> None:
    nonclosed = """
data_bad
_cell_length_a 5
_cell_length_b 6
_cell_length_c 7
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_space_group_symop_operation_xyz
'x, y, z'
'-x, y+1/3, -z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.1 0.2 0.3
"""
    assert refused(nonclosed) == AdapterStatus.INCONSISTENT_SOURCE
    disagreeing = site_block(
        "C1 C 0.1 0.2 0.3\n", extra_items="_space_group_name_Hall 'P 2yb'\n"
    )
    assert refused(disagreeing) == AdapterStatus.INCONSISTENT_SOURCE
    tetragonal = nonclosed.replace(
        "'x, y, z'\n'-x, y+1/3, -z'", "'x, y, z'\n'-y, x, z'\n'-x, -y, z'\n'y, -x, z'"
    )
    assert refused(tetragonal) == AdapterStatus.INCONSISTENT_SOURCE
    magnetic = site_block(
        "C1 C 0.1 0.2 0.3\n", extra_items="_space_group_magn_name_BNS 'P2_1/c'\n"
    )
    assert refused(magnetic) == AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
    two_blocks = ROCK_SALT + ROCK_SALT.replace("data_NaCl", "data_copy")
    assert refused(two_blocks) == AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
    assert cif_crystal_structure(two_blocks, block="copy").block_name == "copy"
