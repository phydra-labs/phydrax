from dataclasses import replace
from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization._coordinate_enclosure import CoordinateEnclosureResourceError
from phydrax.discretization._exact_plc_geometry import (
    _ExactPlcBudget,
    ExactPlcCellGeometrySource,
)


def source() -> ExactPlcCellGeometrySource:
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
    )


def budget() -> _ExactPlcBudget:
    return _ExactPlcBudget(10_000_000, 4096)


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
def test_index_matches_canonical_brute(
    point: tuple[int | Fraction, int | Fraction, int | Fraction],
    stratum: int,
    row: int,
) -> None:
    owner = source()
    exact_point = (Fraction(point[0]), Fraction(point[1]), Fraction(point[2]))
    locator = owner._prepare_witness_locator(budget())
    brute = owner._locate_witness(exact_point, budget())
    indexed = owner._locate_witness(exact_point, budget(), locator=locator)
    assert indexed == brute
    assert indexed[:2] == (stratum, row)


def test_locator_refuses_replaced_source_bits_and_forged_rows() -> None:
    owner = source()
    locator = owner._prepare_witness_locator(budget())
    point = (Fraction(0), Fraction(0), Fraction(0))
    changed = eqx.tree_at(
        lambda value: value.source_points,
        owner,
        owner.source_points.at[0, 0].set(jnp.nextafter(0.0, 1.0)),
    )
    with pytest.raises(ValueError, match="original source"):
        changed._locate_witness(point, budget(), locator=locator)
    with pytest.raises(ValueError, match="original source"):
        owner._locate_witness(point, budget(), locator=replace(locator, packed=None))
    original_ids = owner.source_triangle_ids
    object.__setattr__(owner, "source_triangle_ids", original_ids.at[0].set(99))
    with pytest.raises(ValueError, match="original source"):
        owner._locate_witness(point, budget(), locator=locator)
    object.__setattr__(owner, "source_triangle_ids", original_ids)
    object.__setattr__(owner, "domain_source_revision", "mutated")
    with pytest.raises(ValueError, match="original source"):
        owner._locate_witness(point, budget(), locator=locator)


def test_preparation_and_query_obey_original_work_cap() -> None:
    owner = source()
    with pytest.raises(CoordinateEnclosureResourceError):
        owner._prepare_witness_locator(_ExactPlcBudget(1, 4096))
    locator = owner._prepare_witness_locator(budget())
    with pytest.raises(CoordinateEnclosureResourceError):
        owner._locate_witness(
            (Fraction(0), Fraction(0), Fraction(0)),
            _ExactPlcBudget(16, 4096),
            locator=locator,
        )


def test_disjoint_rows_reduce_only_complete_candidates() -> None:
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
    )
    locator = owner._prepare_witness_locator(budget())
    point = (Fraction(4 * (count - 1)) + Fraction(1, 4), Fraction(1, 4), Fraction(0))
    brute_budget, indexed_budget = budget(), budget()
    brute = owner._locate_witness(point, brute_budget)
    indexed = owner._locate_witness(point, indexed_budget, locator=locator)
    assert indexed == brute
    assert indexed[:2] == (2, count - 1)
    assert indexed_budget.work < brute_budget.work // 8


@pytest.mark.parametrize("offset", [float(2**900), np.nextafter(0.0, 1.0)])
def test_extreme_source_rows_refuse_storage_before_fraction_materialization(
    offset: float,
) -> None:
    import sys

    from phydrax.discretization._coordinate_enclosure import CoordinateEnclosureBudget

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
    )
    # Independent actual Python owner sizes, not the implementation's byte formula.
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
    ledger = CoordinateEnclosureBudget(10_000_000, source_bytes + row_bytes - 1)
    exact_budget = budget()
    with ledger.activate(), pytest.raises(CoordinateEnclosureResourceError):
        owner._prepare_witness_locator(exact_budget)
    # The exact row has not yet materialized, so no rational bit receipt exists.
    assert exact_budget.bits == 0
    locator = owner._prepare_witness_locator(budget())
    for point in points:
        exact_point = (
            Fraction(float(point[0])),
            Fraction(float(point[1])),
            Fraction(float(point[2])),
        )
        assert owner._locate_witness(
            exact_point, budget(), locator=locator
        ) == owner._locate_witness(exact_point, budget())


def test_canonical_bvh_pack_charges_actual_leaf_payload_slots() -> None:
    from phydrax._bvh import _build_split_levels, _median_split, _pack, BVHBuildPolicy

    lower = np.arange(21, dtype=np.float64).reshape((7, 3))
    upper = lower + 0.25
    policy = BVHBuildPolicy(leaf_size=4)
    levels = _build_split_levels(7, 4, _median_split(lower, upper, 0.5 * (lower + upper)))
    events, storage = [], []
    packed = _pack(
        lower,
        upper,
        levels,
        policy,
        np.dtype(np.float64),
        _charge_work=events.append,
        _reserve_storage=storage.append,
    )
    node_count = packed.left.size
    payload_slots = packed.leaf_items.size
    assert payload_slots == 8
    assert events[0] == 12 * node_count + 4 * lower.size + 3 * payload_slots
    assert events[0] != 12 * node_count + 4 * lower.size + 3 * (7 * policy.leaf_size)
    assert storage[0] >= packed.leaf_items.nbytes
