#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact space-group settings: group laws, Hall derivation, transforms and chirality.

Independent controls are International Tables facts (crystal-system number
ranges, operation counts, 65 Sohncke and 92 centrosymmetric types, the 11
enantiomorphic pairs, centering multiplicities, origin choice 2 at an
inversion centre) and the natively parsed Hall notation, never the table
under test echoed back.
"""

from fractions import Fraction

import numpy as np
import pytest

from phydrax.atomistic.crystal import (
    admitted_hall_numbers,
    HallSymbol,
    SettingTransformation,
    SPACE_GROUP_DATA_ID,
    SpaceGroupSetting,
    standard_space_group_setting,
)


HALL_NUMBERS = tuple(range(1, 531))
_CENTERING_COUNT = {
    "P": 1,
    "A": 2,
    "B": 2,
    "C": 2,
    "I": 2,
    "R": 3,
    "S": 3,
    "T": 3,
    "F": 4,
}
_CRYSTAL_SYSTEM_RANGES = (
    (1, 2, "triclinic"),
    (3, 15, "monoclinic"),
    (16, 74, "orthorhombic"),
    (75, 142, "tetragonal"),
    (143, 167, "trigonal"),
    (168, 194, "hexagonal"),
    (195, 230, "cubic"),
)
# International Tables Vol. A: the 11 enantiomorphic pairs of chiral types.
_ENANTIOMORPHIC_PAIRS = frozenset(
    {
        (76, 78), (91, 95), (92, 96), (144, 145), (151, 153), (152, 154),
        (169, 170), (171, 172), (178, 179), (180, 181), (212, 213),
    }
)  # fmt: skip
_INVERSION = (-1, 0, 0, 0, -1, 0, 0, 0, -1)
_ZERO = (Fraction(0), Fraction(0), Fraction(0))
_REVERSAL = SettingTransformation(((-1, 0, 0), (0, -1, 0), (0, 0, -1)))


def _reduced(vector: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    return tuple(value % 1 for value in vector)


@pytest.mark.parametrize("hall_number", HALL_NUMBERS, ids=lambda n: f"hall-{n}")
def test_admitted_setting_obeys_space_group_laws_and_multiplicity(
    hall_number: int,
) -> None:
    setting = SpaceGroupSetting(hall_number)
    products, shifts = setting.multiplication_table()
    count = setting.operation_count
    identity = np.arange(count)
    assert setting.exact_operations()[0] == ((1, 0, 0, 0, 1, 0, 0, 0, 1), _ZERO)
    np.testing.assert_array_equal(products[0], identity)
    np.testing.assert_array_equal(products[:, 0], identity)
    # Every row and column of a group table is a permutation and every
    # operation has a two-sided inverse.
    assert all(sorted(row) == list(identity) for row in products)
    assert all(sorted(column) == list(identity) for column in products.T)
    inverses = np.argwhere(products == 0)
    assert inverses.shape == (count, 2)
    assert np.all(products[inverses[:, 1], inverses[:, 0]] == 0)
    # Associativity of the reduced table, including the integer lattice shifts.
    rng = np.random.default_rng(hall_number)
    triples = rng.integers(0, count, size=(64, 3))
    for i, j, k in triples:
        left = products[products[i, j], k]
        right = products[i, products[j, k]]
        assert left == right
    symbol = HallSymbol(setting.hall_symbol or "")
    assert setting.centering_count == _CENTERING_COUNT[symbol.lattice]
    assert count == setting.point_group_order * setting.centering_count
    assert 48 % setting.point_group_order == 0
    assert (
        len(set(setting.rotations[: setting.point_group_order]))
        == setting.point_group_order
    )
    assert set(setting.determinants) <= {-1, 1}
    assert shifts.dtype == np.int64


@pytest.mark.parametrize("hall_number", HALL_NUMBERS, ids=lambda n: f"hall-{n}")
def test_native_hall_closure_reproduces_admitted_operations(hall_number: int) -> None:
    setting = SpaceGroupSetting(hall_number)
    assert setting.hall_symbol is not None
    derived = HallSymbol(setting.hall_symbol).group()
    assert derived.operations == setting.exact_operations()
    assert derived.point_group_order == setting.point_group_order
    assert hall_number in setting.hall_numbers
    assert setting.data_id == SPACE_GROUP_DATA_ID


def test_standard_settings_follow_crystal_system_ranges() -> None:
    for first, last, system in _CRYSTAL_SYSTEM_RANGES:
        for number in range(first, last + 1):
            setting = standard_space_group_setting(number)
            assert (number, setting.crystal_system) == (number, system)
            assert setting.international_number == number


def test_standard_settings_count_sohncke_and_centrosymmetric_types() -> None:
    settings = [standard_space_group_setting(n) for n in range(1, 231)]
    assert sum(setting.sohncke for setting in settings) == 65
    assert sum(setting.centrosymmetric for setting in settings) == 92
    assert not any(setting.sohncke and setting.centrosymmetric for setting in settings)


@pytest.mark.parametrize(
    ("number", "operations", "point_order", "centering"),
    (
        (1, 1, 1, 1),
        (14, 4, 4, 1),
        (19, 4, 4, 1),
        (166, 36, 12, 3),
        (194, 24, 24, 1),
        (225, 192, 48, 4),
        (230, 96, 48, 2),
    ),
    ids=("P1", "P21/c", "P212121", "R-3m-hexagonal", "P63/mmc", "Fm-3m", "Ia-3d"),
)
def test_standard_general_position_multiplicity(
    number: int, operations: int, point_order: int, centering: int
) -> None:
    setting = standard_space_group_setting(number)
    assert setting.operation_count == operations
    assert setting.point_group_order == point_order
    assert setting.centering_count == centering


def test_two_origin_groups_standardize_on_the_inversion_centre() -> None:
    two_origin = []
    for number in range(1, 231):
        settings = [SpaceGroupSetting(h) for h in admitted_hall_numbers(number)]
        origins = {setting.origin_choice for setting in settings}
        if {1, 2} <= origins:
            two_origin.append(number)
            standard = standard_space_group_setting(number)
            assert standard.origin_choice == 2
            assert (_INVERSION, _ZERO) in standard.exact_operations()
            first = next(s for s in settings if s.origin_choice == 1)
            assert first.centrosymmetric
            assert (_INVERSION, _ZERO) not in first.exact_operations()
    assert len(two_origin) == 24


def test_rhombohedral_groups_standardize_on_hexagonal_axes() -> None:
    for number in (146, 148, 155, 160, 161, 166, 167):
        standard = standard_space_group_setting(number)
        assert standard.rhombohedral_axes == "hexagonal"
        assert standard.centering_count == 3
        rhombohedral = [
            SpaceGroupSetting(h)
            for h in admitted_hall_numbers(number)
            if SpaceGroupSetting(h).rhombohedral_axes == "rhombohedral"
        ]
        assert len(rhombohedral) == 1
        assert rhombohedral[0].centering_count == 1
        assert rhombohedral[0].operation_count == standard.point_group_order


def test_hexagonal_to_rhombohedral_axes_transform_matches_admitted_setting() -> None:
    third = Fraction(1, 3)
    obverse = SettingTransformation(
        ((2 * third, -third, -third), (third, third, -2 * third), (third, third, third))
    )
    hexagonal = standard_space_group_setting(166)
    rhombohedral = hexagonal.transformed(obverse)
    assert rhombohedral.international_number == 166
    assert rhombohedral.rhombohedral_axes == "rhombohedral"
    assert rhombohedral.centering_count == 1
    assert rhombohedral.operation_count == 12
    assert rhombohedral.transformation.determinant == third
    back = SpaceGroupSetting(
        rhombohedral.hall_numbers[0],
        transformation=SettingTransformation(((1, 0, 1), (-1, 1, 1), (0, -1, 1))),
    )
    assert back.setting_id == hexagonal.setting_id


def test_monoclinic_axis_permutation_moves_the_unique_axis() -> None:
    b_unique = standard_space_group_setting(14)
    assert b_unique.unique_axis == "b"
    assert (-1, 0, 0, 0, 1, 0, 0, 0, -1) in b_unique.rotations
    # (a', b', c') = (c, a, b): the old b axis becomes the new c axis.
    cyclic = SettingTransformation(((0, 1, 0), (0, 0, 1), (1, 0, 0)))
    c_unique = b_unique.transformed(cyclic)
    assert c_unique.international_number == 14
    assert (-1, 0, 0, 0, -1, 0, 0, 0, 1) in c_unique.rotations
    assert c_unique.unique_axis is not None and c_unique.unique_axis.endswith("c")
    assert c_unique.hall_numbers
    assert c_unique.setting_id != b_unique.setting_id


def test_origin_shift_conjugates_translations_exactly() -> None:
    standard = standard_space_group_setting(19)
    shift = (Fraction(1, 4), Fraction(1, 8), Fraction(0))
    moved = standard.transformed(SettingTransformation(None, shift))
    expected = {
        (
            rotation,
            _reduced(
                tuple(
                    translation[i]
                    + sum(rotation[3 * i + j] * shift[j] for j in range(3))
                    - shift[i]
                    for i in range(3)
                )
            ),
        )
        for rotation, translation in standard.exact_operations()
    }
    assert set(moved.exact_operations()) == expected
    assert moved.international_number == 19
    assert moved.hall_numbers == ()
    assert moved.hall_symbol is None
    # (R - I) p doubles the shift along reversed axes: denominators 2 and 4.
    assert moved.translation_denominator == 4
    restored = moved.transformed(SettingTransformation(None, tuple(-v for v in shift)))
    assert restored.setting_id == standard.setting_id


def test_composed_transformation_equals_sequential_transforms() -> None:
    setting = SpaceGroupSetting(81)
    first = SettingTransformation(
        ((0, 1, 0), (0, 0, 1), (1, 0, 0)), (Fraction(1, 2), 0, 0)
    )
    second = SettingTransformation(
        ((1, 0, 0), (0, 1, 0), (1, 0, 1)), (0, Fraction(1, 4), 0)
    )
    sequential = setting.transformed(first).transformed(second)
    composed = setting.transformed(first.followed_by(second))
    assert sequential.setting_id == composed.setting_id
    assert sequential.operation_count == setting.operation_count


def test_supercell_turns_lattice_translations_into_centering() -> None:
    doubled = standard_space_group_setting(1).transformed(
        SettingTransformation(((2, 0, 0), (0, 1, 0), (0, 0, 1)))
    )
    assert doubled.operation_count == 2
    assert doubled.centering_vectors == (
        _ZERO,
        (Fraction(1, 2), Fraction(0), Fraction(0)),
    )
    with pytest.raises(ValueError, match="maximum_operation_count"):
        standard_space_group_setting(1).transformed(
            SettingTransformation(((2, 0, 0), (0, 2, 0), (0, 0, 2))),
            maximum_operation_count=4,
        )


def test_invalid_changes_of_basis_are_refused() -> None:
    with pytest.raises(ValueError, match="lattice translation"):
        standard_space_group_setting(14).transformed(
            SettingTransformation(((Fraction(1, 2), 0, 0), (0, 1, 0), (0, 0, 1)))
        )
    with pytest.raises(ValueError, match="not invariant"):
        standard_space_group_setting(75).transformed(
            SettingTransformation(((2, 0, 0), (0, 1, 0), (0, 0, 1)))
        )
    with pytest.raises(ValueError, match="nonsingular"):
        SettingTransformation(((1, 0, 0), (0, 1, 0), (1, 1, 0)))
    with pytest.raises(TypeError, match="exact"):
        SettingTransformation(((1.0, 0, 0), (0, 1, 0), (0, 0, 1)))  # ty: ignore[invalid-argument-type]


def test_orientation_reversal_maps_chiral_types_to_their_enantiomorphs() -> None:
    swapped = set()
    for number in range(1, 231):
        standard = standard_space_group_setting(number)
        mirrored = standard.transformed(_REVERSAL)
        assert mirrored.sohncke == standard.sohncke
        assert mirrored.crystal_system == standard.crystal_system
        if mirrored.international_number != number:
            swapped.add(tuple(sorted((number, mirrored.international_number))))
    assert swapped == _ENANTIOMORPHIC_PAIRS


@pytest.mark.parametrize("number", (80, 98, 210), ids=("I41", "I4122", "F4132"))
def test_achiral_sohncke_types_keep_their_number_under_reversal(number: int) -> None:
    standard = standard_space_group_setting(number)
    assert standard.sohncke
    mirrored = standard.transformed(_REVERSAL)
    assert mirrored.international_number == number
    assert mirrored.transformation.orientation_preserving is False


@pytest.mark.parametrize(
    "number", (1, 2, 14, 19, 62, 142, 166, 194, 227, 230), ids=lambda n: f"sg-{n}"
)
def test_general_position_orbit_has_full_multiplicity_and_matches_table(
    number: int,
) -> None:
    setting = standard_space_group_setting(number)
    point = np.asarray([0.1234, 0.3571, 0.7913])
    images = np.asarray(setting.apply(point, dtype=np.float64))
    assert images.shape == (setting.operation_count, 3)
    wrapped = np.round(np.mod(images, 1.0), 9) % 1.0
    assert len({tuple(row) for row in wrapped}) == setting.operation_count
    products, shifts = setting.multiplication_table()
    twice = np.asarray(setting.apply(images, dtype=np.float64))
    # g_i(g_j(x)) = g_k(x) + n_ij exactly in fractional coordinates.
    np.testing.assert_allclose(twice, images[products] + shifts, rtol=0.0, atol=1.0e-12)


@pytest.mark.parametrize(
    "symbol",
    ("Q 2", "P 5", "P 2 3", "P 31 2 (x,y,z)", "P 2 (0 0 1", "P 2q"),
    ids=("lattice", "order", "axis", "change-of-basis", "unclosed", "translation"),
)
def test_hall_symbol_grammar_refusals(symbol: str) -> None:
    with pytest.raises(ValueError):
        HallSymbol(symbol)


def test_hall_symbol_kind_and_range_refusals() -> None:
    with pytest.raises(TypeError, match="string"):
        HallSymbol(3)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="1..530"):
        SpaceGroupSetting(531)
    with pytest.raises(ValueError, match="1..230"):
        standard_space_group_setting(0)
