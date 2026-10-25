#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from fractions import Fraction

import pytest

import phydrax.discretization._sphere_chart_deformation as owner
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.geometry._sphere_material_atlas import SphereProjectiveActionBank
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.sphere_material import accepted_sphere_material


@contextmanager
def _arithmetic_budget(work: int) -> Iterator[CoordinateEnclosureBudget]:
    ticket = HostBatchTicket(work, bytes_upper=16 * 1024**2)
    budget = CoordinateEnclosureBudget(
        ticket, maximum_work_units=work, maximum_memory_bytes=16 * 1024**2
    )
    with ticket, budget.activate():
        # The scalar helpers require their caller to admit expansion storage.
        budget.reserve(0, 4096)
        yield budget


@pytest.mark.parametrize(
    ("row", "point", "expected", "work"),
    (
        ((7, 0, 0), (Fraction(2, 5), Fraction(1, 7)), Fraction(7), 0),
        ((7, 2, 3), (Fraction(0), Fraction(0)), Fraction(7), 0),
        ((0, 1, 0), (Fraction(2, 5), Fraction(1, 7)), Fraction(2, 5), 0),
        ((0, 2, 0), (Fraction(1), Fraction(0)), Fraction(2), 0),
        ((3, 1, 0), (Fraction(2, 5), Fraction(0)), Fraction(17, 5), 1),
        ((0, 2, 0), (Fraction(2, 5), Fraction(0)), Fraction(4, 5), 1),
        ((3, 2, 5), (Fraction(1, 4), Fraction(1, 3)), Fraction(31, 6), 4),
        ((1, -1, 1), (Fraction(1), Fraction(0)), Fraction(0), 1),
    ),
)
def test_affine_row_omits_only_actual_identity_arithmetic(
    row: tuple[int, int, int],
    point: tuple[Fraction, Fraction],
    expected: Fraction,
    work: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact_row = tuple(Fraction(value) for value in row)
    events: list[str] = []
    with _arithmetic_budget(work) as budget:
        for name in ("__add__", "__mul__"):
            original = getattr(Fraction, name)

            def observed(
                left: Fraction,
                right: Fraction | int,
                *,
                _name: str = name,
                _original: Callable[[Fraction, Fraction | int], Fraction] = original,
            ) -> Fraction:
                # The operation's debit must precede the Fraction operation.
                assert budget.work_units == len(events) + 1
                events.append(_name)
                return _original(left, right)

            monkeypatch.setattr(Fraction, name, observed)
        actual = owner._sphere_affine_row(exact_row, point, budget=budget)
        assert actual == expected
        assert budget.work_units == len(events) == work
        if not point[0] and not point[1]:
            assert actual is exact_row[0]
    monkeypatch.undo()
    if work:
        with (
            _arithmetic_budget(work - 1) as limited,
            pytest.raises(CoordinateEnclosureResourceError) as refused,
        ):
            owner._sphere_affine_row(exact_row, point, budget=limited)
        assert refused.value.resource == "coefficient_work"


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected", "work"),
    (
        (Fraction(0), Fraction(7, 3), Fraction(0), 0),
        (Fraction(7, 3), Fraction(7, 3), Fraction(1), 0),
        (Fraction(5, 7), Fraction(1), Fraction(5, 7), 0),
        (Fraction(5, 7), Fraction(7, 3), Fraction(15, 49), 1),
    ),
)
def test_image_quotient_admits_each_executed_division(
    numerator: Fraction,
    denominator: Fraction,
    expected: Fraction,
    work: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    divisions = []
    original = Fraction.__truediv__
    with _arithmetic_budget(work) as budget:
        def observed(left: Fraction, right: Fraction | int) -> Fraction:
            assert budget.work_units == len(divisions) + 1
            divisions.append((left, right))
            return original(left, right)

        monkeypatch.setattr(Fraction, "__truediv__", observed)
        assert owner._sphere_image_quotient(
            numerator, denominator, budget=budget
        ) == expected
        assert budget.work_units == len(divisions) == work
    monkeypatch.undo()
    if work:
        with (
            _arithmetic_budget(work - 1) as limited,
            pytest.raises(CoordinateEnclosureResourceError) as refused,
        ):
            owner._sphere_image_quotient(numerator, denominator, budget=limited)
        assert refused.value.resource == "coefficient_work"


def test_authenticated_reference_images_preserve_checks_and_exact_work_boundary() -> None:
    _, _, atlas = accepted_sphere_material(1.2)
    with coordinate_reference_budget() as budget, budget.live_storage() as storage:
        actions = SphereProjectiveActionBank.authenticate(
            atlas, atlas, budget=budget, storage=storage
        )
        identifier = int(atlas.cell_global_ids[0])
        mapping = atlas.projective_reference_map(
            identifier, atlas, identifier, budget=budget, actions=actions
        )
        other_mapping = atlas.projective_reference_map(
            identifier, atlas, int(atlas.cell_global_ids[1]), budget=budget, actions=actions
        )
    points = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
        (Fraction(1, 4), Fraction(1, 3)),
    )
    expected = mapping.reference_images(points)

    def evaluate(work: int) -> int:
        with _arithmetic_budget(work) as ledger:
            result = tuple(
                owner._sphere_reference_image(mapping, point, budget=ledger)
                for point in points
            )
        assert result == expected
        return ledger.work_units

    work = evaluate(10**6)
    assert evaluate(work) == work
    with pytest.raises(CoordinateEnclosureResourceError) as refused:
        evaluate(work - 1)
    assert refused.value.resource == "coefficient_work"
    for point in (
        (Fraction(-1), Fraction(0)),
        (Fraction(0), Fraction(-1)),
        (Fraction(3, 4), Fraction(1, 2)),
    ):
        with pytest.raises(ValueError, match="source reference triangle"):
            mapping.reference_images((point,))
        with (
            _arithmetic_budget(100) as ledger,
            pytest.raises(ValueError, match="source reference triangle"),
        ):
            owner._sphere_reference_image(mapping, point, budget=ledger)
    with (
        _arithmetic_budget(100) as ledger,
        pytest.raises(TypeError, match="Fraction coordinates"),
    ):
        # Deliberately ill-typed input: the runtime refusal of non-Fraction coordinates is the contract.
        owner._sphere_reference_image(mapping, (0, Fraction(0)), budget=ledger)  # ty: ignore[invalid-argument-type]
    interior = (Fraction(1, 3), Fraction(1, 3))
    with pytest.raises(ValueError, match="target direction cone"):
        other_mapping.reference_images((interior,))
    with (
        _arithmetic_budget(100) as ledger,
        pytest.raises(ValueError, match="target direction cone"),
    ):
        owner._sphere_reference_image(other_mapping, interior, budget=ledger)
