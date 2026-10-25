#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Asymmetric-unit admission, special-site multiplicity, chirality and expansion.

Independent controls: Z and Z' from site stabilizer orders (an inversion
centre halves Z'), the sign of a tetrahedral triple product under proper and
improper operations, rigid-body distance invariance of every copy computed
with an independent NumPy inverse, whole-molecule branches across a skew
cell face, and the geometric compatibility of a molecule with a twofold axis.
"""

from collections.abc import Sequence
from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.atomistic.crystal import (
    AsymmetricUnitPlan,
    AsymmetricUnitSite,
    AsymmetricUnitState,
    expand_asymmetric_unit,
    metric_from_cell_parameters,
    site_of_point,
    standard_space_group_setting,
)
from tests._support.crystal_geometry_molecules import (
    argon,
    chlorofluorobromomethane,
    nontrivial_automorphism,
    trans_dichloroethene,
)


_GENERAL = (Fraction(1, 5), Fraction(1, 3), Fraction(1, 7))


def _state(
    plan: AsymmetricUnitPlan,
    centers: Sequence[Sequence[float] | np.ndarray],
    quaternions: Sequence[Sequence[float] | np.ndarray],
    lengths: tuple[float, float, float],
    angles: tuple[float, float, float],
) -> AsymmetricUnitState:
    metric = metric_from_cell_parameters(
        jnp.asarray(lengths, dtype=jnp.float64), jnp.asarray(angles, dtype=jnp.float64)
    )
    return AsymmetricUnitState(
        jnp.asarray(centers, dtype=jnp.float64),
        jnp.asarray(quaternions, dtype=jnp.float64),
        plan.chart.coefficients(metric),
        jnp.zeros((len(centers),), dtype=jnp.int32),
    )


def _identity(count: int) -> tuple[int, ...]:
    return tuple(range(count))


def test_inversion_centre_halves_z_prime_without_duplicate_copies() -> None:
    setting = standard_space_group_setting(2)
    pool = trans_dichloroethene()
    swap = nontrivial_automorphism(pool)
    centre = site_of_point(setting, (0, 0, 0))
    special = AsymmetricUnitPlan(
        setting,
        [AsymmetricUnitSite(pool, centre, site_permutations=[_identity(6), swap])],
        chirality="enantiopure",
    )
    assert special.copy_count == 1 and special.atom_count == 6
    assert special.component_z_prime == ((pool.component.component_id, Fraction(1, 2)),)
    assert special.molecules_per_cell(pool.component.component_id) == 1
    general = AsymmetricUnitPlan(
        setting,
        [AsymmetricUnitSite(pool, site_of_point(setting, _GENERAL))],
        chirality="enantiopure",
    )
    assert general.copy_count == 2 and general.atom_count == 12
    assert general.molecules_per_cell(pool.component.component_id) == 2
    expansion = expand_asymmetric_unit(
        special,
        _state(
            special, [(0.0, 0.0, 0.0)], [(0.9, 0.1, -0.3, 0.2)], (6, 7, 8), (80, 95, 100)
        ),
    )
    positions = np.asarray(expansion.positions)
    assert np.allclose(positions, -positions[list(swap)], atol=1e-10)
    assert float(expansion.orientation_residual[0]) < 1e-10


def test_one_atom_component_occupies_a_special_point() -> None:
    setting = standard_space_group_setting(2)
    plan = AsymmetricUnitPlan(
        setting,
        [
            AsymmetricUnitSite(
                argon(),
                site_of_point(setting, (Fraction(1, 2), 0, 0)),
                site_permutations=[(0,), (0,)],
            )
        ],
        chirality="enantiopure",
    )
    assert plan.copy_count == 1
    expansion = expand_asymmetric_unit(
        plan,
        _state(plan, [(0.5, 0.0, 0.0)], [(1.0, 0.0, 0.0, 0.0)], (5, 5, 5), (90, 90, 90)),
    )
    assert np.allclose(np.asarray(expansion.fractional), [[0.5, 0.0, 0.0]])


def _triple_product(positions: np.ndarray) -> float:
    centre = positions[0]
    return float(np.linalg.det(np.stack(positions[1:4] - centre)))


def test_improper_copies_are_explicit_enantiomers() -> None:
    pool = chlorofluorobromomethane()
    centric = standard_space_group_setting(2)
    site = AsymmetricUnitSite(pool, site_of_point(centric, _GENERAL))
    with pytest.raises(ValueError, match="enantiomer"):
        AsymmetricUnitPlan(centric, [site], chirality="enantiopure")
    racemic = AsymmetricUnitPlan(centric, [site], chirality="racemic")
    assert racemic.copy_handedness.tolist() == [1, -1]
    assert racemic.copy_enantiomer.tolist() == [False, True]
    expansion = expand_asymmetric_unit(
        racemic,
        _state(
            racemic, [(0.2, 0.3, 0.4)], [(0.8, 0.2, 0.5, -0.1)], (7, 8, 9), (85, 95, 105)
        ),
    )
    positions = np.asarray(expansion.positions)
    signs = [np.sign(_triple_product(positions[5 * c : 5 * c + 5])) for c in range(2)]
    assert signs[0] == -signs[1]
    sohncke = standard_space_group_setting(4)
    chiral_site = AsymmetricUnitSite(pool, site_of_point(sohncke, _GENERAL))
    pure = AsymmetricUnitPlan(sohncke, [chiral_site], chirality="enantiopure")
    assert pure.copy_handedness.tolist() == [1, 1]
    with pytest.raises(ValueError, match="racemic"):
        AsymmetricUnitPlan(sohncke, [chiral_site], chirality="racemic")
    centred = AsymmetricUnitSite(
        pool,
        site_of_point(centric, (0, 0, 0)),
        site_permutations=[_identity(5), _identity(5)],
    )
    with pytest.raises(ValueError, match="stereo-chiral"):
        AsymmetricUnitPlan(centric, [centred], chirality="racemic")


def test_boundary_crossing_molecule_stays_whole_in_a_skew_cell() -> None:
    setting = standard_space_group_setting(1)
    pool = trans_dichloroethene()
    plan = AsymmetricUnitPlan(
        setting,
        [AsymmetricUnitSite(pool, site_of_point(setting, _GENERAL))],
        chirality="enantiopure",
    )
    expansion = expand_asymmetric_unit(
        plan,
        _state(
            plan,
            [(1.985, 0.02, 0.5)],
            [(1.0, 0.0, 0.0, 0.0)],
            (7.1, 8.3, 9.2),
            (78, 96, 107),
        ),
    )
    assert np.array_equal(np.asarray(expansion.copy_center_shifts), [[-1, 0, 0]])
    assert np.allclose(np.asarray(expansion.copy_centers), [[0.985, 0.02, 0.5]])
    fractional = np.asarray(expansion.fractional)
    positions = np.asarray(expansion.positions)
    assert np.any(fractional >= 1.0) and np.any(fractional[:, 0] < 1.0)
    local = np.asarray(pool.conformers[0].positions)
    bonds = pool.component.bonds
    for i, j in bonds:
        assert np.isclose(
            np.linalg.norm(positions[i] - positions[j]),
            np.linalg.norm(local[i] - local[j]),
            rtol=1e-10,
        )
    vectors = np.asarray(expansion.cell.vectors)
    wrapped = (fractional - np.floor(fractional)) @ vectors
    broken = [
        abs(np.linalg.norm(wrapped[i] - wrapped[j]) - np.linalg.norm(local[i] - local[j]))
        for i, j in bonds
    ]
    assert max(broken) > 1.0


def test_skew_cell_copies_are_rigid_images_of_the_asymmetric_unit() -> None:
    setting = standard_space_group_setting(4)
    pool = chlorofluorobromomethane()
    plan = AsymmetricUnitPlan(
        setting,
        [AsymmetricUnitSite(pool, site_of_point(setting, _GENERAL))],
        chirality="enantiopure",
    )
    quaternion = np.asarray((0.7, -0.3, 0.5, 0.4))
    quaternion /= np.linalg.norm(quaternion)
    centre = np.asarray((0.31, 0.77, 0.12))
    expansion = expand_asymmetric_unit(
        plan, _state(plan, [centre], [quaternion], (6.5, 7.5, 8.5), (90, 113, 90))
    )
    vectors = np.asarray(expansion.cell.vectors)
    w, x, y, z = quaternion
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    offsets = np.asarray(pool.conformers[0].positions) @ rotation.T
    fractional_offsets = offsets @ np.linalg.inv(vectors)
    operations = setting.exact_operations()
    shifts = np.asarray(expansion.copy_center_shifts)
    positions = np.asarray(expansion.positions)
    for copy, operation in enumerate(plan.copy_operation.tolist()):
        matrix = np.asarray(operations[operation][0], dtype=float).reshape(3, 3)
        translation = np.asarray([float(v) for v in operations[operation][1]])
        expected = (
            (centre + fractional_offsets) @ matrix.T + translation + shifts[copy]
        ) @ vectors
        assert np.allclose(positions[5 * copy : 5 * copy + 5], expected, atol=1e-10)
        cartesian = np.asarray(expansion.cell.cartesian_rotations[operation])
        relative = positions[5 * copy : 5 * copy + 5] - np.asarray(
            expansion.copy_center_positions[copy]
        )
        assert np.allclose(relative, offsets @ cartesian.T, atol=1e-10)


def test_site_action_admission_refuses_non_automorphisms_and_non_actions() -> None:
    pool = trans_dichloroethene()
    swap = nontrivial_automorphism(pool)
    centric = standard_space_group_setting(2)
    with pytest.raises(ValueError, match="admitted"):
        AsymmetricUnitPlan(
            centric,
            [
                AsymmetricUnitSite(
                    pool,
                    site_of_point(centric, (0, 0, 0)),
                    site_permutations=[_identity(6), (0, 1, 3, 2, 4, 5)],
                )
            ],
            chirality="enantiopure",
        )
    with pytest.raises(ValueError, match="special-position"):
        AsymmetricUnitSite(pool, site_of_point(centric, (0, 0, 0)))
    prism = standard_space_group_setting(10)
    centre = site_of_point(prism, (0, 0, 0))
    assert centre.stabilizer_order == 4
    with pytest.raises(ValueError, match="group action"):
        AsymmetricUnitPlan(
            prism,
            [
                AsymmetricUnitSite(
                    pool, centre, site_permutations=[_identity(6), swap, swap, swap]
                )
            ],
            chirality="enantiopure",
        )


def test_twofold_axis_orientation_compatibility_is_reported() -> None:
    setting = standard_space_group_setting(3)
    pool = trans_dichloroethene()
    swap = nontrivial_automorphism(pool)
    axis = site_of_point(setting, (0, 0, 0))
    assert axis.dimension == 1 and axis.stabilizer_order == 2
    plan = AsymmetricUnitPlan(
        setting,
        [AsymmetricUnitSite(pool, axis, site_permutations=[_identity(6), swap])],
        chirality="enantiopure",
    )
    assert plan.copy_count == 1
    half = np.sqrt(0.5)
    aligned = expand_asymmetric_unit(
        plan,
        _state(
            plan, [(0.0, 0.37, 0.0)], [(half, -half, 0.0, 0.0)], (6, 7, 8), (90, 104, 90)
        ),
    )
    assert float(aligned.orientation_residual[0]) < 1e-10
    assert float(aligned.center_residual[0]) < 1e-12
    misaligned = expand_asymmetric_unit(
        plan,
        _state(
            plan, [(0.0, 0.37, 0.0)], [(1.0, 0.0, 0.0, 0.0)], (6, 7, 8), (90, 104, 90)
        ),
    )
    assert float(misaligned.orientation_residual[0]) > 0.1
    off_axis = expand_asymmetric_unit(
        plan,
        _state(
            plan, [(0.05, 0.37, 0.0)], [(half, -half, 0.0, 0.0)], (6, 7, 8), (90, 104, 90)
        ),
    )
    assert float(off_axis.center_residual[0]) > 0.01


def test_provenance_names_every_atom_and_copy() -> None:
    setting = standard_space_group_setting(14)
    pool = trans_dichloroethene()
    plan = AsymmetricUnitPlan(
        setting,
        [AsymmetricUnitSite(pool, site_of_point(setting, _GENERAL))],
        chirality="enantiopure",
    )
    assert plan.copy_count == 4 and plan.atom_count == 24
    assert plan.atom_copy.tolist() == [c for c in range(4) for _ in range(6)]
    assert plan.atom_local.tolist() == list(range(6)) * 4
    assert plan.atom_operation.tolist() == [
        o for o in plan.copy_operation.tolist() for _ in range(6)
    ]
    assert sorted(plan.copy_operation.tolist()) == list(range(4))
    assert set(plan.copy_handedness.tolist()) == {1, -1}
    assert not plan.copy_enantiomer.any()
