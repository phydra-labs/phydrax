from dataclasses import replace
from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.discretization._exact_plc_geometry import (
    ExactPlcBudget,
    ExactPlcCellGeometrySource,
)
from tests._support.coordinate_references import coordinate_reference_budget


def source(*, budget: CoordinateEnclosureBudget) -> ExactPlcCellGeometrySource:
    return ExactPlcCellGeometrySource(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        np.array([[0, 1, 2], [0, 1, 3]]),
        np.array([[0, 1], [0, 2]]),
        np.empty(0, dtype=np.int8),
        np.empty(0, dtype=np.int64),
        np.empty((0, 2)),
        domain_source_id="candidate-source",
        domain_source_revision="original",
        source_triangle_ids=np.array([8, 2]),
        source_triangle_bounds=np.zeros(2),
        source_segment_ids=np.array([9, 1]),
        source_segment_bounds=np.zeros(2),
        budget=budget,
    )


def budget(*, budget: CoordinateEnclosureBudget) -> ExactPlcBudget:
    return ExactPlcBudget(10_000_000, 4096, budget=budget)


def test_binary_membership_reuses_exact_predicates_without_changing_witnesses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    carriers = np.asarray(
        ((0.25, 0.25, 0.0), (0.5, 0.5, 0.0), (0.0, 0.0, 0.0),
         (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (0.25, 0.0, 0.25))
    )
    strata = np.asarray((2, 2, 2, 1, 1, 2), dtype=np.int8)
    rows = np.asarray((0, 0, 0, 0, 0, 1), dtype=np.int64)
    parameters = np.asarray(
        ((0.25, 0.25), (0.5, 0.5), (0.0, 0.0),
         (0.5, 0.0), (1.0, 0.0), (0.25, 0.25))
    )
    with coordinate_reference_budget() as ledger:
        owner = source(budget=ledger)
        before = ledger.work_units
        fast = owner.prepare_witnesses(
            carriers, strata, rows, parameters, budget=ledger
        )
        fast_work = ledger.work_units - before
    monkeypatch.setattr(
        ExactPlcCellGeometrySource, "_binary64_memberships",
        lambda self, rounded, strata, rows, exact_work: np.zeros(
            rounded.shape[0], dtype=np.bool_
        ),
    )
    with coordinate_reference_budget() as ledger:
        before = ledger.work_units
        exact = owner.prepare_witnesses(
            carriers, strata, rows, parameters, budget=ledger
        )
        exact_work = ledger.work_units - before
    assert fast.vertices == exact.vertices
    assert fast.on_entity == exact.on_entity
    assert fast.maximum_rounding_error == exact.maximum_rounding_error
    assert fast_work < exact_work


@pytest.mark.parametrize("carrier,stratum", (((0.25, 0.25, 0.01), 2), ((1.25, 0.0, 0.0), 1)))
def test_binary_membership_keeps_off_entity_rational_refusal(
    carrier: tuple[float, float, float], stratum: int,
) -> None:
    with coordinate_reference_budget() as ledger:
        owner = source(budget=ledger)
        with pytest.raises(ValueError, match="rounded_vertices"):
            owner.prepare_witnesses(
                np.asarray((carrier,)), np.asarray((stratum,), dtype=np.int8),
                np.asarray((0,), dtype=np.int64), np.asarray(((0.0, 0.0),)),
                budget=ledger,
            )


@pytest.mark.parametrize(
    "point,stratum,row",
    [
        ((0, 0, 0), 1, 0),
        ((Fraction(1, 2), 0, 0), 1, 0),
        ((Fraction(1, 4), Fraction(1, 4), 0), 2, 0),
        ((Fraction(1, 4), 0, Fraction(1, 4)), 2, 1),
        ((Fraction(1, 4), Fraction(1, 4), Fraction(1, 4)), 0, -1),
        ((Fraction(1, 3), Fraction(1, 3), 0), 2, 0),
        ((1 - Fraction(1, 2**55), 0, 0), 1, 0),
        ((1 + Fraction(1, 2**55), 0, 0), 0, -1),
    ],
)
def test_index_matches_brute_source_membership(
    point: tuple[int | Fraction, int | Fraction, int | Fraction],
    stratum: int,
    row: int,
) -> None:
    with coordinate_reference_budget(queries_upper=2) as ledger:
        owner = source(budget=ledger)
        exact_point = (Fraction(point[0]), Fraction(point[1]), Fraction(point[2]))
        locator = owner.prepare_witness_locator(budget(budget=ledger), budget=ledger)
        brute = owner.locate_witness(exact_point, budget(budget=ledger), budget=ledger)
        indexed = owner.locate_witness(
            exact_point, budget(budget=ledger), locator=locator, budget=ledger
        )
        assert indexed == brute
        assert indexed[:2] == (stratum, row)


def test_locator_refuses_replaced_source_bits_and_forged_rows() -> None:
    with coordinate_reference_budget(queries_upper=4) as ledger:
        owner = source(budget=ledger)
        locator = owner.prepare_witness_locator(budget(budget=ledger), budget=ledger)
        point = (Fraction(0), Fraction(0), Fraction(0))
        changed = eqx.tree_at(
            lambda value: value.source_points,
            owner,
            owner.source_points.at[0, 0].set(jnp.nextafter(0.0, 1.0)),
        )
        with pytest.raises(ValueError, match=r"ExactPlcCellGeometrySource\.locator"):
            changed.locate_witness(
                point, budget(budget=ledger), locator=locator, budget=ledger
            )
        with pytest.raises(ValueError, match=r"ExactPlcCellGeometrySource\.locator"):
            owner.locate_witness(
                point,
                budget(budget=ledger),
                locator=replace(locator, packed=None),
                budget=ledger,
            )
        original_ids = owner.source_triangle_ids
        object.__setattr__(owner, "source_triangle_ids", original_ids.at[0].set(99))
        with pytest.raises(ValueError, match=r"ExactPlcCellGeometrySource\.locator"):
            owner.locate_witness(
                point, budget(budget=ledger), locator=locator, budget=ledger
            )
        object.__setattr__(owner, "source_triangle_ids", original_ids)
        object.__setattr__(owner, "domain_source_revision", "mutated")
        with pytest.raises(ValueError, match=r"ExactPlcCellGeometrySource\.locator"):
            owner.locate_witness(
                point, budget(budget=ledger), locator=locator, budget=ledger
            )


def test_preparation_and_query_obey_declared_work_cap() -> None:
    with coordinate_reference_budget(queries_upper=1) as ledger:
        owner = source(budget=ledger)
        with pytest.raises(CoordinateEnclosureResourceError):
            owner.prepare_witness_locator(
                ExactPlcBudget(1, 4096, budget=ledger), budget=ledger
            )
        locator = owner.prepare_witness_locator(budget(budget=ledger), budget=ledger)
        with pytest.raises(CoordinateEnclosureResourceError):
            owner.locate_witness(
                (Fraction(0), Fraction(0), Fraction(0)),
                ExactPlcBudget(16, 4096, budget=ledger),
                locator=locator,
                budget=ledger,
            )


def test_disjoint_rows_preserve_closed_row_candidates() -> None:
    with coordinate_reference_budget(queries_upper=2) as ledger:
        count = 256
        points = np.array(
            [
                [4.0 * row + x, y, 0.0]
                for row in range(count)
                for x, y in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
            ]
        )
        owner = ExactPlcCellGeometrySource(
            points,
            np.arange(3 * count).reshape(count, 3),
            np.empty((0, 2), dtype=np.int64),
            np.empty(0, dtype=np.int8),
            np.empty(0, dtype=np.int64),
            np.empty((0, 2)),
            domain_source_id="disjoint-candidate-source",
            domain_source_revision="original",
            source_triangle_ids=np.arange(count)[::-1],
            source_triangle_bounds=np.zeros(count),
            source_segment_ids=np.empty(0, dtype=np.int64),
            source_segment_bounds=np.empty(0),
            budget=ledger,
        )
        locator = owner.prepare_witness_locator(budget(budget=ledger), budget=ledger)
        point = (Fraction(4 * (count - 1)) + Fraction(1, 4), Fraction(1, 4), Fraction(0))
        brute_budget, indexed_budget = budget(budget=ledger), budget(budget=ledger)
        brute = owner.locate_witness(point, brute_budget, budget=ledger)
        indexed = owner.locate_witness(
            point, indexed_budget, locator=locator, budget=ledger
        )
        assert indexed == brute
        assert indexed[:2] == (2, count - 1)
        assert indexed_budget.work < brute_budget.work // 8


@pytest.mark.parametrize("maximum_bits", (64, 4096))
def test_prepared_planes_reject_only_exactly_noncoplanar_candidates(
    maximum_bits: int,
) -> None:
    with coordinate_reference_budget(queries_upper=3) as ledger:
        count = 96
        points = np.asarray(
            [
                point
                for row in range(count)
                for point in (
                    (0.0, 0.0, 0.0),
                    (1.0, 1.0, 0.0),
                    (0.0, 1.0, 1.0 + row / 128.0),
                )
            ],
            dtype=np.float64,
        )
        owner = ExactPlcCellGeometrySource(
            points,
            np.arange(3 * count, dtype=np.int64).reshape(count, 3),
            np.empty((0, 2), dtype=np.int64),
            np.empty(0, dtype=np.int8),
            np.empty(0, dtype=np.int64),
            np.empty((0, 2), dtype=np.float64),
            domain_source_id="overlapping-box-candidate-source",
            domain_source_revision="original",
            source_triangle_ids=np.arange(count, dtype=np.int64),
            source_triangle_bounds=np.zeros(count, dtype=np.float64),
            source_segment_ids=np.empty(0, dtype=np.int64),
            source_segment_bounds=np.empty(0, dtype=np.float64),
            budget=ledger,
        )
        locator = owner.prepare_witness_locator(
            ExactPlcBudget(10_000_000, maximum_bits, budget=ledger), budget=ledger
        )
        # Every row's box contains this point, but none of its exact planes do.
        point = (Fraction(1, 4),) * 3
        brute_work = ExactPlcBudget(10_000_000, maximum_bits, budget=ledger)
        indexed_work = ExactPlcBudget(10_000_000, maximum_bits, budget=ledger)
        brute = owner.locate_witness(point, brute_work, budget=ledger)
        indexed = owner.locate_witness(
            point, indexed_work, locator=locator, budget=ledger
        )
        assert indexed == brute
        assert indexed[:2] == (0, -1)
        if maximum_bits == 4096:
            assert all(normal is not None for normal in locator.normals)
            assert indexed_work.work < brute_work.work // 4
        else:
            # A tighter original bit cap keeps the original predicate instead
            # of constructing larger prepared rational operands.
            assert all(normal is None for normal in locator.normals)
        coplanar = (Fraction(1, 4), Fraction(1, 4), Fraction(0))
        assert owner.locate_witness(
            coplanar,
            ExactPlcBudget(10_000_000, maximum_bits, budget=ledger),
            locator=locator,
            budget=ledger,
        )[:2] == (2, 0)


@pytest.mark.parametrize("offset", [float(2**900), np.nextafter(0.0, 1.0)])
def test_extreme_source_rows_refuse_storage_before_fraction_materialization(
    offset: float,
) -> None:
    with coordinate_reference_budget(queries_upper=8) as ledger:
        import sys

        from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget

        next_coordinate = np.nextafter(offset, np.inf)
        points = np.array(
            [
                [offset, offset, offset],
                [next_coordinate, offset, offset],
                [offset, next_coordinate, offset],
                [offset, offset, next_coordinate],
            ]
        )
        triangles = np.array([[0, 1, 2]])
        segments = np.empty((0, 2), dtype=np.int64)
        strata, rows = np.zeros(4, dtype=np.int8), np.full(4, -1, dtype=np.int64)
        parameters = np.zeros((4, 2))
        triangle_ids, triangle_bounds = np.array([42]), np.zeros(1)
        segment_ids, segment_bounds = np.empty(0, dtype=np.int64), np.empty(0)
        owner = ExactPlcCellGeometrySource(
            points,
            triangles,
            segments,
            strata,
            rows,
            parameters,
            domain_source_id="extreme-candidate-source",
            domain_source_revision="original",
            source_triangle_ids=triangle_ids,
            source_triangle_bounds=triangle_bounds,
            source_segment_ids=segment_ids,
            source_segment_bounds=segment_bounds,
            budget=ledger,
        )
        # Independent Python owner sizes, not the implementation's byte formula.
        corners = tuple(
            tuple(Fraction(float(value)) for value in point) for point in points[:3]
        )
        row_bytes = sys.getsizeof(corners) + sum(
            sys.getsizeof(corner)
            + sum(
                sys.getsizeof(value)
                + sys.getsizeof(value.numerator)
                + sys.getsizeof(value.denominator)
                for value in corner
            )
            for corner in corners
        )
        source_bytes = sum(
            value.nbytes
            for value in (
                points,
                triangles,
                segments,
                strata,
                rows,
                parameters,
                triangle_ids,
                triangle_bounds,
                segment_ids,
                segment_bounds,
            )
        )
        limited = CoordinateEnclosureBudget(
            HostBatchTicket(10_000_000, bytes_upper=source_bytes + row_bytes - 1),
            maximum_work_units=10_000_000,
            maximum_memory_bytes=source_bytes + row_bytes - 1,
        )
        exact_budget = budget(budget=limited)
        with (
            limited.ticket,
            limited.activate(),
            pytest.raises(CoordinateEnclosureResourceError) as refusal,
        ):
            owner.prepare_witness_locator(exact_budget, budget=limited)
        assert refusal.value.resource == "polynomial_storage"
        assert refusal.value.limit == source_bytes + row_bytes - 1
        assert refusal.value.completed <= refusal.value.limit < refusal.value.requested
        locator = owner.prepare_witness_locator(budget(budget=ledger), budget=ledger)
        for point in points:
            exact_point = (
                Fraction(float(point[0])),
                Fraction(float(point[1])),
                Fraction(float(point[2])),
            )
            assert owner.locate_witness(
                exact_point,
                budget(budget=ledger),
                locator=locator,
                budget=ledger,
            ) == owner.locate_witness(exact_point, budget(budget=ledger), budget=ledger)


@pytest.mark.parametrize(
    "point,bound,admitted",
    [
        # Exactly representable on the slanted row: no displacement to bound.
        ((1, Fraction(3, 4), Fraction(1, 4)), 0.0, True),
        # Round-to-nearest of 1/3 leaves the plane y = 3z.
        ((1, 1, Fraction(1, 3)), 0.0, False),
        ((1, 1, Fraction(1, 3)), 1.0e-15, True),
        # Off every source row: no row binds the point.
        ((1, 1, 1), 1.0, False),
    ],
)
def test_located_witness_admission_uses_original_row_bound(
    point: tuple[int | Fraction, int | Fraction, int | Fraction],
    bound: float,
    admitted: bool,
) -> None:
    with coordinate_reference_budget(queries_upper=2) as ledger:
        owner = ExactPlcCellGeometrySource(
            np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 1.0]]),
            np.array([[0, 1, 2]]),
            np.empty((0, 2), dtype=np.int64),
            np.empty(0, dtype=np.int8),
            np.empty(0, dtype=np.int64),
            np.empty((0, 2)),
            domain_source_id="slanted-source",
            domain_source_revision="original",
            source_triangle_ids=np.array([5]),
            source_triangle_bounds=np.array([bound]),
            source_segment_ids=np.empty(0, dtype=np.int64),
            source_segment_bounds=np.empty(0),
            budget=ledger,
        )
        exact_point = (Fraction(point[0]), Fraction(point[1]), Fraction(point[2]))
        locator = owner.prepare_witness_locator(budget(budget=ledger), budget=ledger)
        assert (
            owner.admits_located_witness(
                exact_point, budget(budget=ledger), locator=locator, budget=ledger
            )
            is admitted
        )
