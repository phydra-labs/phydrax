#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import PeriodicCell


_G = phx.geometry
_IDENTITY = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
_INVERSION = ((-1, 0, 0), (0, -1, 0), (0, 0, -1))
_SKEW_CELL = ((1.3, 0.0, 0.0), (0.4, 0.9, 0.0), (-0.2, 0.3, 1.7))

type _Formula = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def _gyroid(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)


def _gyroid_mirror(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.cos(x) * np.sin(y) + np.cos(y) * np.sin(z) + np.cos(z) * np.sin(x)


def _schwarz_p(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.cos(x) + np.cos(y) + np.cos(z)


def _schwarz_d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return (
        np.sin(x) * np.sin(y) * np.sin(z)
        + np.sin(x) * np.cos(y) * np.cos(z)
        + np.cos(x) * np.sin(y) * np.cos(z)
        + np.cos(x) * np.cos(y) * np.sin(z)
    )


def _iwp(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return 2.0 * (
        np.cos(x) * np.cos(y) + np.cos(y) * np.cos(z) + np.cos(z) * np.cos(x)
    ) - (np.cos(2 * x) + np.cos(2 * y) + np.cos(2 * z))


def _lidinoid(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return (
        0.5
        * (
            np.sin(2 * x) * np.cos(y) * np.sin(z)
            + np.sin(2 * y) * np.cos(z) * np.sin(x)
            + np.sin(2 * z) * np.cos(x) * np.sin(y)
        )
        - 0.5
        * (
            np.cos(2 * x) * np.cos(2 * y)
            + np.cos(2 * y) * np.cos(2 * z)
            + np.cos(2 * z) * np.cos(2 * x)
        )
        + 0.15
    )


@pytest.fixture(scope="module")
def schwarz_p_levels() -> _G.TrigonometricCriticalLevels:
    return _G.certify_trigonometric_critical_levels(
        _G.TrigonometricTable.family("schwarz_p")
    )


@pytest.fixture(scope="module")
def gyroid_levels() -> _G.TrigonometricCriticalLevels:
    return _G.certify_trigonometric_critical_levels(
        _G.TrigonometricTable.family("gyroid")
    )


@pytest.mark.parametrize(
    ("family", "formula"),
    [
        pytest.param("gyroid", _gyroid, id="gyroid"),
        pytest.param("gyroid_mirror", _gyroid_mirror, id="gyroid-mirror"),
        pytest.param("schwarz_p", _schwarz_p, id="schwarz-p"),
        pytest.param("schwarz_d", _schwarz_d, id="schwarz-d"),
        pytest.param("iwp", _iwp, id="iwp"),
        pytest.param("lidinoid", _lidinoid, id="lidinoid"),
    ],
)
def test_named_family_tables_evaluate_their_cited_formulas(
    family: _G.TrigonometricFamily, formula: _Formula
) -> None:
    table = _G.TrigonometricTable.family(family)
    reduced = np.random.default_rng(7).uniform(-1.5, 2.5, size=(64, 3))
    expected = formula(*(2.0 * np.pi * reduced[:, axis] for axis in range(3)))

    values = np.asarray(table.evaluate_reduced(jnp.asarray(reduced)))

    np.testing.assert_allclose(values, expected, rtol=0.0, atol=1.0e-13)


def test_skew_lattice_field_and_lattice_equivalent_points_are_bitwise_periodic() -> None:
    table = _G.TrigonometricTable.family("gyroid")
    cell = PeriodicCell(np.asarray(_SKEW_CELL, dtype=np.float64))
    reduced = np.random.default_rng(3).integers(0, 2**20, size=(32, 3)) / 2.0**20
    shifts = np.random.default_rng(4).integers(-5, 6, size=(32, 3)).astype(np.float64)
    faces = np.asarray([[0.0, 0.25, 0.5], [1.0, 0.25, 0.5], [-3.0, 1.25, 4.5]])

    base = np.asarray(table.evaluate_reduced(jnp.asarray(reduced)))
    shifted = np.asarray(table.evaluate_reduced(jnp.asarray(reduced + shifts)))
    face_values = np.asarray(table.evaluate_reduced(jnp.asarray(faces)))
    source = _G.TrigonometricLevelSet(table, cell, level=0.25)
    points = np.asarray(cell.cartesian(jnp.asarray(reduced)))
    field = np.asarray(source.compile().boundary_field(jnp.asarray(points)))
    independent = _gyroid(*(2.0 * np.pi * reduced[:, axis] for axis in range(3)))

    assert np.array_equal(base, shifted)
    assert face_values[0] == face_values[1] == face_values[2]
    np.testing.assert_allclose(field, independent - 0.25, rtol=0.0, atol=1.0e-12)
    dyadic = _G.TrigonometricLevelSet(table, PeriodicCell(2.0 * np.eye(3)), level=0.25)
    cartesian = 2.0 * reduced
    assert np.array_equal(
        np.asarray(dyadic.compile().boundary_field(jnp.asarray(cartesian))),
        np.asarray(
            dyadic.compile().boundary_field(jnp.asarray(cartesian + 2.0 * shifts))
        ),
    )


def test_schwarz_p_critical_levels_enclose_plus_minus_one_and_three_and_refuse_level_one(
    schwarz_p_levels: _G.TrigonometricCriticalLevels,
) -> None:
    table = _G.TrigonometricTable.family("schwarz_p")
    cell = PeriodicCell(np.eye(3))
    enclosures = np.stack((schwarz_p_levels.value_lower, schwarz_p_levels.value_upper), 1)

    assert schwarz_p_levels.resolved
    assert enclosures.shape == (4, 2)
    assert np.all(enclosures[:, 0] <= (-3.0, -1.0, 1.0, 3.0))
    assert np.all(enclosures[:, 1] >= (-3.0, -1.0, 1.0, 3.0))
    assert np.all(enclosures[:, 1] - enclosures[:, 0] < 1.0e-9)
    with pytest.raises(_G.TrigonometricSingularLevelError) as refusal:
        _G.TrigonometricLevelSet(table, cell, level=1.0, critical_levels=schwarz_p_levels)
    assert refusal.value.enclosure[0] <= 1.0 <= refusal.value.enclosure[1]
    assert refusal.value.separation.outcome == "quarter_point"
    with pytest.raises(_G.TrigonometricSingularLevelError):
        _G.TrigonometricLevelSet(
            table,
            cell,
            region="sheet",
            sheet=_G.TrigonometricSheetBand(1.0),
            critical_levels=schwarz_p_levels,
        )
    accepted = _G.TrigonometricLevelSet(
        table, cell, level=0.5, critical_levels=schwarz_p_levels
    ).compile()
    assert bool(accepted.validity().accepted)


def test_gyroid_level_at_a_krawczyk_critical_value_is_refused_with_evidence(
    gyroid_levels: _G.TrigonometricCriticalLevels,
) -> None:
    separation = gyroid_levels.separate(1.5)

    assert separation.outcome == "critical_point"
    assert separation.evidence_value is not None
    assert separation.evidence_value[0] <= 1.5 <= separation.evidence_value[1]
    assert separation.evidence_box is not None
    point = 0.5 * separation.evidence_box[0] + 0.5 * separation.evidence_box[1]
    gyroid = _G.TrigonometricTable.family("gyroid")
    gradient = jax.grad(lambda u: gyroid.evaluate_reduced(u))(jnp.asarray(point))
    assert float(jnp.max(jnp.abs(gradient))) < 1.0e-9
    assert gyroid_levels.separate(1.4).separated


def test_iwp_non_isolated_critical_set_is_resolved_only_on_demand() -> None:
    table = _G.TrigonometricTable.family("iwp")
    levels = _G.certify_trigonometric_critical_levels(table)
    cell = PeriodicCell(np.eye(3))

    # The coarse global cover keeps the body-diagonal set (value 3) as a sound
    # unresolved enclosure without refining it for unrelated levels.
    assert not levels.resolved
    assert levels.unresolved_boxes.shape[0] < 20_000
    widths = levels.unresolved_values[:, 1] - levels.unresolved_values[:, 0]
    assert np.all(widths <= levels.value_resolution)
    assert levels.enclosure_meeting(3.0) is not None
    assert levels.separate(0.0).refined_boxes == 0

    exact = levels.separate(3.0)
    assert exact.outcome == "quarter_point"
    assert exact.evidence_box is not None
    with pytest.raises(_G.TrigonometricSingularLevelError):
        _G.TrigonometricLevelSet(table, cell, level=3.0, critical_levels=levels)

    # A level inside the coarse enclosure but solvably separated from 3 is
    # resolved by refining only the boxes meeting it.
    near = 3.001
    assert levels.enclosure_meeting(near) is not None
    separated = levels.separate(near)
    assert separated.separated and separated.refined_boxes > 0
    assert separated.critical_levels.separation(near) > 0.0
    source = _G.TrigonometricLevelSet(table, cell, level=near, critical_levels=levels)
    assert source.critical_levels.separation(near) > 0.0
    assert bool(source.compile().validity().accepted)


def test_table_symmetries_are_proved_exactly_from_the_coefficients() -> None:
    gyroid = _G.TrigonometricTable.family("gyroid")
    schwarz_p = _G.TrigonometricTable.family("schwarz_p")
    half = Fraction(1, 2)

    assert gyroid.symmetry_sign(_INVERSION, (0, 0, 0)) == -1
    assert schwarz_p.symmetry_sign(_IDENTITY, (half, half, half)) == -1
    assert gyroid.symmetry_sign(_IDENTITY, (half, 0, 0)) == 0
    assert (
        gyroid.composed(((1, 0, 0), (0, 0, 1), (0, 1, 0)), (0, 0, 0)).table_id
        == _G.TrigonometricTable.family("gyroid_mirror").table_id
    )
    with pytest.raises(ValueError, match="non-quarter turn"):
        gyroid.composed(_IDENTITY, (Fraction(1, 3), 0, 0))


def test_sheet_thickness_bound_increases_and_solved_half_width_meets_the_target(
    gyroid_levels: _G.TrigonometricCriticalLevels,
) -> None:
    table = _G.TrigonometricTable.family("gyroid")
    cell = PeriodicCell(np.diag((1.0, 1.25, 0.8)))
    certificate = _G.certify_trigonometric_sheet_thickness(
        table, cell, level=0.0, maximum_half_width=0.5, resolution_level=5
    )
    half_widths = (0.05, 0.1, 0.2, 0.3, 0.5)
    thicknesses = [certificate.minimum_thickness(value) for value in half_widths]
    target = 0.6 * thicknesses[2]

    solved = certificate.half_width_for_thickness(target)
    source = _G.TrigonometricLevelSet(
        table,
        cell,
        region="sheet",
        sheet=_G.TrigonometricWallThickness(target, resolution_level=5),
        critical_levels=gyroid_levels,
    )

    assert np.all(np.diff(thicknesses) > 0.0)
    for value, thickness in zip(half_widths, thicknesses, strict=True):
        uniform = 2.0 * value / certificate.gradient_norm_upper(value)
        assert thickness >= uniform * (1.0 - 1.0e-12)
    assert certificate.gradient_norm_upper(0.5) >= certificate.sampled_gradient_norm
    assert certificate.minimum_thickness(solved) >= target
    assert certificate.minimum_thickness(solved * (1.0 - 1.0e-9)) < target
    assert source.half_width is not None
    assert source.half_width >= solved * (1.0 - 1.0e-9)
    assert source.sheet_thickness is not None
    assert source.sheet_thickness.minimum_thickness(source.half_width) >= target
