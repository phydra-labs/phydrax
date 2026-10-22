from contextlib import AbstractContextManager
from fractions import Fraction

import equinox as eqx
import numpy as np
import pytest

from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.discretization._periodic_exact import (
    exact_periodic_element,
    exact_periodic_generators,
    owned_periodic_preparation,
    PeriodicIsometryGroup,
    PeriodicIsometryIdentityError,
)
from phydrax.discretization._periodic_topology import periodic_preparation_bounds


def _action_budget(
    group: PeriodicIsometryGroup, exponents: tuple[tuple[int, ...], ...], /
) -> AbstractContextManager[CoordinateEnclosureBudget]:
    work, storage, queries = periodic_preparation_bounds(
        group, len(exponents), np.asarray(exponents, dtype=np.int64)
    )
    return owned_periodic_preparation(work, storage, queries_upper=queries)


def _screw() -> np.ndarray:
    matrix = np.diag((1.0, -1.0, -1.0, 1.0))
    matrix[:3, 3] = 1.0
    return matrix


def test_original_screw_source_and_signed_exponents() -> None:
    source = _screw()
    group = PeriodicIsometryGroup(source[None])
    assert group.orders == (0,)
    assert group.linear_orders == (2,)
    assert group.translation_periods == (2,)
    assert np.asarray(group.generators).tobytes() == source[None].tobytes()
    with _action_budget(
        group, ((2,), (-7,), (-2,), (-1,), (0,), (1,), (2,), (7,), (-1,))
    ) as budget:
        matrices, orders = exact_periodic_generators(group, budget=budget)
        square = exact_periodic_element(matrices, orders, (2,), budget=budget)
        assert tuple(row[-1] for row in square[:-1]) == (
            Fraction(2),
            Fraction(0),
            Fraction(0),
        )
        for exponent in (-7, -2, -1, 0, 1, 2, 7):
            exact = exact_periodic_element(matrices, orders, (exponent,), budget=budget)
            expected = np.linalg.matrix_power(source, exponent)
            np.testing.assert_array_equal(np.asarray(exact, dtype=np.float64), expected)
            np.testing.assert_array_equal(group.element(np.array((exponent,))), expected)
        inverse = exact_periodic_element(matrices, orders, (-1,), budget=budget)
        assert tuple(row[-1] for row in inverse[:-1]) == (
            Fraction(-1),
            Fraction(1),
            Fraction(1),
        )


def test_finite_and_translation_source_laws() -> None:
    finite = _screw()
    finite[0, 3] = 0.0
    group = PeriodicIsometryGroup(finite[None])
    assert (group.orders, group.linear_orders, group.translation_periods) == (
        (2,),
        (2,),
        (0,),
    )
    translation = np.eye(4)
    translation[0, 3] = -0.25
    group = PeriodicIsometryGroup(translation[None])
    assert (group.orders, group.linear_orders, group.translation_periods) == (
        (0,),
        (1,),
        (1,),
    )
    with _action_budget(group, ((-3,),)) as budget:
        matrices, orders = exact_periodic_generators(group, budget=budget)
        assert exact_periodic_element(matrices, orders, (-3,), budget=budget)[0][
            -1
        ] == Fraction(3, 4)


def test_exact_source_refusals_and_relations() -> None:
    with pytest.raises(
        PeriodicIsometryIdentityError, match="PeriodicIsometryGroup\\.generators"
    ):
        PeriodicIsometryGroup(np.eye(4)[None])
    rounded = _screw()
    rounded[1, 1] = np.nextafter(-1.0, 0.0)
    with pytest.raises(
        PeriodicIsometryIdentityError, match="PeriodicIsometryGroup\\.generators"
    ):
        PeriodicIsometryGroup(rounded[None], tolerance=1.0)
    source = _screw()
    translation = np.eye(4)
    translation[0, 3] = 1.0
    with pytest.raises(PeriodicIsometryIdentityError, match="exact_group_solve\\.matrix"):
        PeriodicIsometryGroup(np.stack((source, translation)))
    finite = source.copy()
    finite[0, 3] = 0.0
    with pytest.raises(
        PeriodicIsometryIdentityError, match="PeriodicIsometryGroup\\.exponents"
    ):
        PeriodicIsometryGroup(np.stack((finite, finite)))
    noncommuting = np.eye(4)
    noncommuting[1, 3] = 1.0
    with pytest.raises(
        PeriodicIsometryIdentityError, match="PeriodicIsometryGroup\\.generators"
    ):
        PeriodicIsometryGroup(np.stack((source, noncommuting)))


def test_group_restoration_authenticates_source_metadata() -> None:
    group = PeriodicIsometryGroup(_screw()[None], tolerance=0.0)
    group.validate_restored()
    for name, replacement in (
        ("orders", (2,)),
        ("linear_orders", (1,)),
        ("translation_periods", (1,)),
        ("group_id", "forged"),
        ("ambient_dimension", 2),
    ):
        tampered = eqx.tree_at(lambda value: value.generators, group, group.generators)
        object.__setattr__(tampered, name, replacement)
        with pytest.raises(ValueError, match="PeriodicIsometryGroup\\."):
            tampered.validate_restored()
    other = PeriodicIsometryGroup(_screw()[None], tolerance=1.0e-9)
    assert other.group_id != group.group_id
    changed = _screw()
    changed[0, 3] = np.nextafter(1.0, 2.0)
    assert PeriodicIsometryGroup(changed[None]).group_id != group.group_id


def test_unproven_linear_order_is_specific_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.discretization._periodic_exact as canonical

    monkeypatch.setattr(canonical, "MAXIMUM_ROTATION_ORDER", 1)
    with pytest.raises(
        PeriodicIsometryIdentityError,
        match="PeriodicIsometryGroup\\.linear_orders",
    ):
        PeriodicIsometryGroup(_screw()[None])


def test_faithful_screw_finite_cosets_preserve_original_actions() -> None:
    screw = _screw()
    finite = screw.copy()
    finite[0, 3] = 0.0
    group = PeriodicIsometryGroup(np.stack((screw, finite)))
    assert group.orders == (0, 2)
    assert group.linear_orders == (2, 2)
    assert group.translation_periods == (2, 0)
    with _action_budget(group, ((-3, 1),)) as budget:
        matrices, orders = exact_periodic_generators(group, budget=budget)
        composed = exact_periodic_element(matrices, orders, (-3, 1), budget=budget)
        expected = np.linalg.matrix_power(screw, -3) @ finite
        np.testing.assert_array_equal(np.asarray(composed, dtype=np.float64), expected)


def test_exact_admission_and_action_charge_original_work_budget() -> None:
    with owned_periodic_preparation(0, 1 << 20) as empty:
        with pytest.raises(CoordinateEnclosureResourceError):
            PeriodicIsometryGroup(_screw()[None], budget=empty)
    group = PeriodicIsometryGroup(_screw()[None])
    with owned_periodic_preparation(100000, 1 << 20) as budget:
        matrices, orders = exact_periodic_generators(group, budget=budget)
        before = budget.work_units
        exact_periodic_element(matrices, orders, (-7,), budget=budget)
        assert budget.work_units > before > 0


def test_exact_group_admission_reuses_only_current_request_bound_content() -> None:
    group = PeriodicIsometryGroup(_screw()[None])
    with _action_budget(group, ((0,),)) as budget:
        admitted = exact_periodic_generators(group, budget=budget)
        charged = budget.work_units
        assert charged > 0
        assert exact_periodic_generators(group, budget=budget) is admitted
        assert budget.work_units == charged
        changed = eqx.tree_at(
            lambda value: value.generators, group,
            group.generators.at[0, 0, 3].add(1.0),
        )
        with pytest.raises(PeriodicIsometryIdentityError, match="bind current generators"):
            exact_periodic_generators(changed, budget=budget)
        assert budget.work_units == charged
    with _action_budget(group, ((0,),)) as fresh:
        reconstructed = exact_periodic_generators(group, budget=fresh)
        assert reconstructed == admitted and reconstructed is not admitted
        assert fresh.work_units == charged
