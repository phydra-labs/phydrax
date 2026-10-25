# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Exact periodic graph equivalence, live demand and pre-allocation refusals."""

import gc
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction
from itertools import product
from math import ceil, floor
from random import Random
from sys import getsizeof
from types import SimpleNamespace
from typing import cast, NoReturn

import numpy as np
import pytest

from phydrax._meshcore import NativeExecutionScope
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.discretization._exact_algebraic_geometry import RadicalElement
from phydrax.meshing.providers import _native_periodic_algebraic as algebraic


type RationalPoint = tuple[Fraction, Fraction]
type RationalSegment = tuple[RationalPoint, RationalPoint]


@contextmanager
def _budget(memory: int = 8_000_000) -> Iterator[CoordinateEnclosureBudget]:
    ticket = HostBatchTicket(10_000_000, bytes_upper=memory)
    with ticket, CoordinateEnclosureBudget(
        ticket, maximum_work_units=10_000_000, maximum_memory_bytes=memory
    ).activate() as budget:
        yield budget


def _geometry(
    rows: list[RationalSegment], shear: Fraction = Fraction(0)
) -> tuple[algebraic._Grid, list[algebraic.Edge], list[algebraic._Segment]]:
    zero, one = RadicalElement(), RadicalElement(1)
    # Predicate-only frames exercise the exact lattice independently of source
    # recovery; the full construction tests below use actual recovered grids.
    grid = cast(algebraic._Grid, SimpleNamespace(
        vectors=((one, zero), (RadicalElement(shear), one))
    ))

    def point(value: RationalPoint) -> algebraic.Point:
        return RadicalElement(value[0] + shear * value[1]), RadicalElement(value[1])

    segments = [
        algebraic._Segment(
            point(a), point(b),
            (RadicalElement(a[0]), RadicalElement(a[1])),
            (RadicalElement(b[0]), RadicalElement(b[1])),
        )
        for a, b in rows
    ]
    return grid, [(2 * i, 2 * i + 1, 0, 0) for i in range(len(rows))], segments


def _orientation(a: RationalPoint, b: RationalPoint, c: RationalPoint) -> Fraction:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _strictly_inside(a: RationalPoint, b: RationalPoint, p: RationalPoint) -> bool:
    return (
        p not in (a, b)
        and _orientation(a, b, p) == 0
        and all(min(a[k], b[k]) <= p[k] <= max(a[k], b[k]) for k in range(2))
    )


def _reference_conflict(first: RationalSegment, second: RationalSegment) -> bool:
    # Independent finite exhaustive deck-image reference. Test coordinates are
    # between -1/8 and 9/8, so shifts outside [-3, 3]^2 cannot meet.
    a, b = first
    for image in product(range(-3, 4), repeat=2):
        c, d = tuple(
            (p[0] + image[0], p[1] + image[1]) for p in second
        )
        signs = (_orientation(a, b, c), _orientation(a, b, d),
                 _orientation(c, d, a), _orientation(c, d, b))
        if signs[0] * signs[1] < 0 and signs[2] * signs[3] < 0:
            return True
        if all(value == 0 for value in signs):
            axis = 0 if a[0] != b[0] else 1
            if max(min(a[axis], b[axis]), min(c[axis], d[axis])) < min(
                max(a[axis], b[axis]), max(c[axis], d[axis])
            ):
                return True
        if any((_strictly_inside(a, b, c), _strictly_inside(a, b, d),
                _strictly_inside(c, d, a), _strictly_inside(c, d, b))):
            return True
    return False


def _reference_graph(rows: list[RationalSegment]) -> list[int]:
    adjacency = [0] * len(rows)
    for i, row in enumerate(rows):
        for j in range(i):
            if _reference_conflict(row, rows[j]):
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    return adjacency


def _exhaustive_selection(
    adjacency: list[int], penalties: list[tuple[int, int, int]], short: int, long: int
) -> int | None:
    independent = [
        mask for mask in range(1 << len(adjacency))
        if all(not (mask & neighbors) for i, neighbors in enumerate(adjacency)
               if mask & (1 << i))
    ]
    cardinality = max(mask.bit_count() for mask in independent)
    feasible = []
    for mask in independent:
        if mask.bit_count() != cardinality:
            continue
        tails = tuple(sum(p[k] for i, p in enumerate(penalties) if mask & (1 << i))
                      for k in range(3))
        if tails[0] <= short and tails[1] <= long:
            feasible.append((tails[2], mask))
    return min(feasible)[1] if feasible else None


def _small_rows(seed: int) -> list[RationalSegment]:
    random = Random(seed)
    rows = []
    while len(rows) < 7:
        a = (Fraction(random.randrange(-1, 10), 8), Fraction(random.randrange(-1, 10), 8))
        b = (Fraction(random.randrange(-1, 10), 8), Fraction(random.randrange(-1, 10), 8))
        if a != b:
            rows.append((a, b))
    return rows


@pytest.mark.parametrize("shear", (Fraction(0), Fraction(1, 4)))
@pytest.mark.parametrize("seed", (3, 17, 29, 43))
def test_periodic_graph_matches_independent_exhaustive_reference(
    shear: Fraction, seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _small_rows(seed)
    grid, edges, segments = _geometry(rows, shear)
    expected = _reference_graph(rows)
    original = algebraic._conflict
    visited: list[tuple[int, int]] = []
    indices = {id(segment): index for index, segment in enumerate(segments)}

    def recorded(
        first: algebraic._Segment, second: algebraic._Segment,
        grid: algebraic._Grid, work: algebraic._Work,
    ) -> bool:
        visited.append((indices[id(first)], indices[id(second)]))
        return original(first, second, grid, work)

    monkeypatch.setattr(algebraic, "_conflict", recorded)
    with _budget() as budget:
        work = algebraic._Work(budget)
        with budget.temporary_scope(), work.box_scope():
            actual = algebraic._conflict_graph(segments, grid, work)
            assert actual == expected
        assert visited == sorted(visited)
        assert len(visited) == len(set(visited))
        assert work.counts.get("edge_conflict_queries", 0) == len(visited)
        assert not work.boxes
        assert budget.temporary_bytes_upper == 0
        # Powers of two make every selected-subset cost distinct, independently
        # checking objective and both quantile tails without a tie oracle.
        penalties = [(int(i % 3 == 0), int(i % 3 == 1), 1 << i) for i in range(len(edges))]
        field_penalties = [(lo, hi, RadicalElement(cost)) for lo, hi, cost in penalties]
        for short, long in ((7, 7), (1, 1), (0, 0)):
            expected_mask = _exhaustive_selection(expected, penalties, short, long)
            with budget.temporary_scope():
                if expected_mask is None:
                    with pytest.raises(ValueError, match="no maximum-cardinality completion"):
                        algebraic._compatible_edges(
                            edges, segments, field_penalties, grid, work,
                            short_quota=short, long_quota=long,
                        )
                else:
                    selected = algebraic._compatible_edges(
                        edges, segments, field_penalties, grid, work,
                        short_quota=short, long_quota=long,
                    )
                    assert selected == tuple(
                        edge for i, edge in enumerate(edges) if expected_mask & (1 << i)
                    )
                    del selected
            assert not work.boxes
            assert budget.temporary_bytes_upper == 0


def test_equal_cost_ties_keep_original_first_included_vertex() -> None:
    q = Fraction
    rows = [
        ((q(1, 8), q(1, 8)), (q(3, 8), q(3, 8))),
        ((q(1, 8), q(3, 8)), (q(3, 8), q(1, 8))),
        ((q(5, 8), q(5, 8)), (q(7, 8), q(7, 8))),
        ((q(5, 8), q(7, 8)), (q(7, 8), q(5, 8))),
    ]
    grid, edges, segments = _geometry(rows)
    assert _reference_graph(rows) == [2, 1, 8, 4]
    with _budget() as budget:
        selected = algebraic._compatible_edges(
            edges, segments, [(0, 0, RadicalElement()) for _ in edges],
            grid, algebraic._Work(budget),
        )
        assert selected == (edges[0], edges[2])


def test_bins_preserve_seam_collinearity_and_subulp_conflicts() -> None:
    q = Fraction
    rows = [
        ((q(7, 8), q(1, 2)), (q(9, 8), q(1, 2))),
        ((q(0), q(1, 4)), (q(0), q(3, 4))),
        ((q(0), q(1, 2)), (q(1, 4), q(1, 2))),
        ((q(1, 4), q(1, 2)), (q(1, 4), q(3, 4))),
    ]
    grid, _, segments = _geometry(rows)
    delta = RadicalElement.sqrt(2) / 2**80
    a = (RadicalElement(1) - delta, RadicalElement(Fraction(1, 4)))
    b = (RadicalElement(1) + delta, RadicalElement(Fraction(3, 4)))
    segments.append(algebraic._Segment(a, b, a, b))
    with _budget() as budget:
        work = algebraic._Work(budget)
        with budget.temporary_scope(), work.box_scope():
            expected = [0] * len(segments)
            for i, segment in enumerate(segments):
                for j in range(i):
                    if algebraic._conflict(segment, segments[j], grid, work):
                        expected[i] |= 1 << j
                        expected[j] |= 1 << i
            assert expected[:4] != [0, 0, 0, 0]
            assert algebraic._conflict_graph(segments, grid, work) == expected


def _disjoint_rows(count: int) -> list[RationalSegment]:
    return [
        ((Fraction(i % 20, 20), Fraction(i // 20, 20)),
         (Fraction(i % 20, 20) + Fraction(1, 100), Fraction(i // 20, 20)))
        for i in range(count)
    ]


def test_large_local_graph_uses_live_demand_and_releases_search_banks() -> None:
    grid, edges, segments = _geometry(_disjoint_rows(280))
    assert 1024 * len(edges) ** 2 > 4_000_000
    with _budget(4_000_000) as budget:
        work = algebraic._Work(budget)
        for _ in range(2):
            with budget.temporary_scope():
                result = algebraic._compatible_edges(
                    edges, segments, [(0, 0, RadicalElement()) for _ in edges], grid, work
                )
                assert result == tuple(edges)
                # Only the returned reference tuple remains charged; adjacency,
                # bins, box owners, recursion states and DP frontiers are gone.
                assert budget.temporary_bytes_upper == 64 + 16 * len(edges)
                assert not work.boxes
                del result
            assert budget.temporary_bytes_upper == 0
        assert work.counts.get("edge_conflict_queries", 0) == 0
        assert work.counts["independent_graph_vertex_visits"] == 4 * len(edges)
        assert 0 < budget.peak_bytes_upper < 4_000_000


def test_recursive_search_releases_owners_without_cyclic_collection() -> None:
    grid, edges, segments = _geometry(_disjoint_rows(8))
    enabled = gc.isenabled()
    gc.disable()
    try:
        with _budget() as budget:
            work = algebraic._Work(budget)
            owner = weakref.ref(work)
            algebraic._compatible_edges(
                edges, segments, [(0, 0, RadicalElement()) for _ in edges], grid, work
            )
            del work
            assert owner() is None
    finally:
        if enabled:
            gc.enable()


def test_adjacency_refuses_before_any_pair_predicate(monkeypatch: pytest.MonkeyPatch) -> None:
    grid, _, segments = _geometry(_disjoint_rows(280))

    def forbidden(
        first: algebraic._Segment, second: algebraic._Segment,
        grid: algebraic._Grid, work: algebraic._Work,
    ) -> NoReturn:
        raise AssertionError("unadmitted adjacency reached a predicate")

    monkeypatch.setattr(algebraic, "_conflict", forbidden)
    with _budget(20_000) as budget:
        work = algebraic._Work(budget)
        with budget.temporary_scope(), work.box_scope():
            with pytest.raises(CoordinateEnclosureResourceError) as error:
                algebraic._conflict_graph(segments, grid, work)
            assert error.value.resource == "polynomial_storage"
            assert not work.boxes
            assert not work.counts
        assert budget.temporary_bytes_upper == 0


def test_box_cache_outlives_interval_scratch_but_not_its_owner() -> None:
    with _budget() as budget:
        work = algebraic._Work(budget)
        point = (RadicalElement.sqrt(2), RadicalElement(Fraction(1, 3)))
        with work.box_scope():
            with budget.temporary_scope():
                first = work.box(point, point)
            retained = budget.temporary_bytes_upper
            charged_work = budget.work_units
            assert retained > 0
            assert budget.peak_bytes_upper > retained
            with budget.temporary_scope():
                assert work.box(point, point) is first
            assert budget.work_units == charged_work
            assert budget.temporary_bytes_upper == retained
            del first
        assert not work.boxes
        assert budget.temporary_bytes_upper == 0


def test_box_compaction_releases_discarded_keys_without_recomputing_survivors() -> None:
    _, _, segments = _geometry(_disjoint_rows(20))
    with _budget() as budget:
        work = algebraic._Work(budget)
        with work.box_scope():
            for segment in segments:
                work.box(segment.fa, segment.fb)
            before = budget.temporary_bytes_upper
            kept = segments[0]
            enclosure = work.box(kept.fa, kept.fb)
            work.discard_boxes_except([kept])
            assert len(work.boxes) == 1
            assert 0 < budget.temporary_bytes_upper < before
            charged_work = budget.work_units
            assert work.box(kept.fa, kept.fb) is enclosure
            assert budget.work_units == charged_work
            work.discard_boxes_except([])
            assert not work.boxes
            assert budget.temporary_bytes_upper == 256
            del enclosure
        assert budget.temporary_bytes_upper == 0


def test_target_graph_matches_exhaustive_pair_order_and_performed_predicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zero, one = RadicalElement(), RadicalElement(1)
    grid = cast(algebraic._Grid, SimpleNamespace(
        vectors=((one, zero), (zero, one)),
        dual=((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))),
        protected=(),
        radius=Fraction(1, 4),
        maximum=Fraction(3, 8),
    ))
    points = tuple(
        (RadicalElement(Fraction(x, 4)), RadicalElement(Fraction(y, 4)))
        for x, y in product(range(4), repeat=2)
    )
    with _budget() as budget:
        work = algebraic._Work(budget)
        indexed = algebraic._target_graph(points, grid, work, (0.25, 0.5))
        indexed_counts = dict(work.counts)
        assert len(indexed) == 3 * len(points)
        assert not work.boxes

    def exhaustive(
        self: algebraic._PeriodicBins, segment: algebraic._Segment, *, stop: int | None = None
    ) -> Iterator[int]:
        yield from range(len(self.segments) if stop is None else stop)

    # Restore the original all-pairs traversal at every predicate consumer:
    # protected, mandatory, visibility, radius and optional conflict graphs.
    monkeypatch.setattr(algebraic._PeriodicBins, "candidates", exhaustive)
    with _budget() as budget:
        work = algebraic._Work(budget)
        expected = algebraic._target_graph(points, grid, work, (0.25, 0.5))
        assert indexed == expected
        assert indexed_counts["edge_conflict_queries"] < work.counts["edge_conflict_queries"]
        assert indexed_counts["root_edge_visibility_pairs"] < work.counts["root_edge_visibility_pairs"]
        for name in (
            "root_orbit_pairs", "bounded_edge_candidates", "periodic_edge_image_pairs",
            "orientation_predicates", "mandatory_edge_classifications",
            "independent_graph_states", "independent_graph_vertex_visits",
            "completion_tail_states",
        ):
            assert indexed_counts.get(name, 0) == work.counts.get(name, 0)


@pytest.mark.parametrize("width", (0, 1, 29, 30, 31, 280, 8192))
def test_bigint_bound_includes_python_digits_and_header(width: int) -> None:
    assert algebraic._bitset_bytes(width) >= getsizeof((1 << width) - 1)


@pytest.mark.parametrize("resolution", (4, 8))
def test_original_source_action_construction_fits_frozen_cap(resolution: int) -> None:
    from tools.meshing_qualification._controls import QualificationControls
    from tools.meshing_qualification.scenarios.periodic import (
        _periodic_qualification_fixture,
    )

    source, specification, _, case = _periodic_qualification_fixture(
        QualificationControls(
            periodic_case="skew-material-feature-exact-algebraic", resolution=resolution
        )
    )
    limits = specification.limits
    assert limits.maximum_scratch_bytes == case["resource_limits"]["maximum_scratch_bytes"]
    assert limits.maximum_scratch_bytes == 81_920_000
    assert specification.size_compliance.absolute_tolerance == 0
    assert specification.size_compliance.relative_tolerance == 0
    with NativeExecutionScope(limits.native_limits()) as scope:
        initial = scope.batch(0)
        budget = CoordinateEnclosureBudget(
            initial, maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        initial.settle()
        with budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size)):
            carrier = algebraic.construct_exact_algebraic_carrier(
                source, specification, scope=scope, budget=budget
            )
            count = 4 * resolution**2 + 3 * resolution
            assert len(carrier.exact_coordinates) == count
            assert carrier.simplices.shape == (2 * count, 3)
            assert set(carrier.regions) == {0, 1}
            assert budget.temporary_bytes_upper == 0
            assert 0 < budget.retained_basis_bytes <= budget.peak_bytes_upper < 81_920_000
            assert budget.work_units <= limits.maximum_work_units
            assert dict(carrier.work_evidence)["root_orbit_pairs"] == count * (count - 1) // 2
            vectors: tuple[algebraic.Point, algebraic.Point] = (
                (
                    RadicalElement(Fraction(float(case["vectors"][0][0]))),
                    RadicalElement(Fraction(float(case["vectors"][0][1]))),
                ),
                (
                    RadicalElement(Fraction(float(case["vectors"][1][0]))),
                    RadicalElement(Fraction(float(case["vectors"][1][1]))),
                ),
            )
            edges = set()
            for roots, images in zip(carrier.simplices, carrier.shifts, strict=True):
                for a, b in ((0, 1), (1, 2), (2, 0)):
                    edges.add(algebraic._key(
                        int(roots[a]), (int(images[a, 0]), int(images[a, 1])),
                        int(roots[b]), (int(images[b, 0]), int(images[b, 1])),
                    )[0])
            assert len(edges) == 3 * count
            squares = sorted(
                algebraic._square(
                    carrier.exact_coordinates[a],
                    algebraic._lift(carrier.exact_coordinates[b], (sx, sy), vectors),
                )
                for a, b, sx, sy in edges
            )
            assert squares[-1] <= Fraction(3, 2 * resolution) ** 2
            for quantile in (0.5, 0.95):
                rank = quantile * (len(squares) - 1)
                assert squares[floor(rank)] == squares[ceil(rank)] == Fraction(1, resolution) ** 2
            assert all(not array.flags.writeable for array in (
                carrier.points, carrier.simplices, carrier.shifts, carrier.regions
            ))
            assert np.isfinite(carrier.points).all()
