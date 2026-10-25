#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact site symmetry, special positions and bounded site-orbit enumeration.

Independent controls are International Tables Vol. A facts: the number of
Wyckoff positions and their multiplicities per space-group type (P-1: 8 x 1
and 2; P2/m: 8 x 1, 6 x 2 and 4; C2/m: 4 x 2, 5 x 4 and 8; P4/mmm: 4 x 1,
4 x 2, 7 x 4, 5 x 8 and 16), their invariance under setting changes, and the
explicit stabilizers of inversion centres, twofold axes and mirror planes.
No Wyckoff letters are compared: none are produced without an admitted
letter table.
"""

from collections import Counter
from fractions import Fraction

import numpy as np
import pytest

from phydrax.atomistic.crystal import (
    admitted_hall_numbers,
    enumerate_site_orbits,
    SettingTransformation,
    site_of_point,
    SitePosition,
    SiteSymmetryTable,
    SpaceGroupSetting,
    standard_space_group_setting,
)


_HALF = Fraction(1, 2)
# International Tables Vol. A: Wyckoff multiplicities of each type.
_MULTIPLICITIES = {
    2: Counter({1: 8, 2: 1}),
    10: Counter({1: 8, 2: 6, 4: 1}),
    12: Counter({2: 4, 4: 5, 8: 1}),
    123: Counter({1: 4, 2: 4, 4: 7, 8: 5, 16: 1}),
}


def _table(setting: SpaceGroupSetting) -> SiteSymmetryTable:
    return enumerate_site_orbits(setting, maximum_sites=4096, maximum_systems=200_000)


@pytest.mark.parametrize("number", sorted(_MULTIPLICITIES), ids=lambda n: f"type-{n}")
def test_site_orbits_reproduce_wyckoff_multiplicities(number: int) -> None:
    setting = standard_space_group_setting(number)
    table = _table(setting)
    assert table.complete and table.status == "complete"
    assert (
        Counter(orbit.multiplicity for orbit in table.orbits) == _MULTIPLICITIES[number]
    )
    for orbit in table.orbits:
        site = orbit.representative
        assert site.multiplicity * site.stabilizer_order == setting.operation_count
        assert len(site.coset_representatives) == site.multiplicity
        assert site.coset_representatives[0] == 0
        assert table.orbit_of(site) is orbit
    general = [orbit for orbit in table.orbits if orbit.dimension == 3]
    assert len(general) == 1 and general[0].stabilizer_order == 1


@pytest.mark.parametrize(
    "hall_number", admitted_hall_numbers(10), ids=lambda n: f"hall-{n}"
)
def test_site_orbit_multiplicities_are_setting_invariant(hall_number: int) -> None:
    table = _table(SpaceGroupSetting(hall_number))
    assert table.complete
    assert Counter(orbit.multiplicity for orbit in table.orbits) == _MULTIPLICITIES[10]


def test_inversion_centres_follow_an_origin_shift() -> None:
    reference = standard_space_group_setting(2)
    shift = (Fraction(1, 4), Fraction(0), Fraction(1, 3))
    shifted = reference.transformed(SettingTransformation(origin_shift=shift))
    table = _table(shifted)
    centres = {
        orbit.representative.point for orbit in table.orbits if orbit.dimension == 0
    }
    expected = {
        tuple((Fraction(a, 2) - shift[i]) % 1 for i, a in enumerate(corner))
        for corner in np.ndindex(2, 2, 2)
    }
    assert centres == expected


def test_special_point_stabilizer_multiplicity_and_orbit() -> None:
    setting = standard_space_group_setting(10)
    centre = site_of_point(setting, (0, _HALF, 0))
    assert centre.special and centre.dimension == 0
    assert centre.stabilizer_order == 4 and centre.multiplicity == 1
    assert centre.improper_stabilizer
    assert len(centre.orbit()) == 1
    point = (Fraction(1, 7), Fraction(2, 9), Fraction(3, 11))
    general = site_of_point(setting, point)
    assert not general.special and general.multiplicity == 4 and general.dimension == 3
    assert general.point == (0, 0, 0)
    orbit = general.orbit(point)
    assert len({image for _, image, _ in orbit}) == 4
    for index, image, shift in orbit:
        rotation, translation = setting.exact_operations()[index]
        moved = [
            sum(rotation[3 * r + c] * point[c] for c in range(3)) + translation[r]
            for r in range(3)
        ]
        assert tuple(moved[r] + shift[r] for r in range(3)) == image
        assert all(0 <= value < 1 for value in image)
    mirror = site_of_point(setting, (Fraction(1, 7), 0, Fraction(2, 5)))
    assert (
        len({image for _, image, _ in mirror.orbit((Fraction(1, 5), Fraction(1, 3)))})
        == 2
    )


def test_mirror_plane_and_axis_sites_have_exact_free_directions() -> None:
    setting = standard_space_group_setting(10)
    mirror = site_of_point(setting, (Fraction(1, 7), 0, Fraction(2, 5)))
    assert (
        mirror.dimension == 2
        and mirror.stabilizer_order == 2
        and mirror.multiplicity == 2
    )
    assert mirror.improper_stabilizer
    span = np.asarray(mirror.direction_matrix(), dtype=float)
    assert np.linalg.matrix_rank(span) == 2
    assert np.allclose(span[1], 0.0)
    axis = site_of_point(setting, (0, Fraction(3, 10), _HALF))
    assert axis.dimension == 1 and axis.stabilizer_order == 2
    assert not axis.improper_stabilizer
    assert np.array_equal(np.abs(axis.direction_matrix()[:, 0]), [0, 1, 0])
    on_site = np.asarray(axis.point_at(np.asarray([0.37]), dtype=np.float64))
    assert np.allclose(np.asarray(axis.constraint_residual(on_site)), 0.0)
    off_site = on_site + np.asarray([0.01, 0.0, 0.0])
    assert np.max(np.abs(np.asarray(axis.constraint_residual(off_site)))) > 1e-3


def test_subspace_that_is_not_a_fixed_set_is_refused() -> None:
    setting = standard_space_group_setting(10)
    with pytest.raises(ValueError, match="not the fixed set"):
        SitePosition(setting, (0, 0, 0), [(1, 0, 0)])


def test_site_keys_identify_sites_modulo_lattice_translations() -> None:
    setting = standard_space_group_setting(2)
    first = site_of_point(setting, (_HALF, 0, 0))
    translated = site_of_point(setting, (Fraction(3, 2), -1, 2))
    other = site_of_point(setting, (0, _HALF, 0))
    assert first.site_key == translated.site_key
    assert first.site_key != other.site_key


def test_exhausted_site_budget_is_reported_incomplete() -> None:
    setting = standard_space_group_setting(123)
    table = enumerate_site_orbits(setting, maximum_sites=3, maximum_systems=200_000)
    assert not table.complete and table.status == "site-limit"
    missing = site_of_point(setting, (_HALF, _HALF, _HALF))
    orbit = table.orbit_of(missing)
    assert orbit is None or orbit.multiplicity == 1
    limited = enumerate_site_orbits(setting, maximum_sites=4096, maximum_systems=2)
    assert not limited.complete and limited.status == "system-limit"
