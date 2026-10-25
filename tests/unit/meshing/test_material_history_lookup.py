#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Prepared chronological material lookup against the complete reverse walk."""

from collections.abc import Sequence
from fractions import Fraction
from itertools import permutations, product

import numpy as np

import phydrax.meshing._tetra_metric as owner
from phydrax._bvh import bvh_overlap_pair_blocks, prepare_bvh
from phydrax._meshcore import TetMeshArrays
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry._implicit_cavity_atlas import Box, ImplicitCavityAtlas, Point, Tet
from phydrax.geometry._rational import rational_vector
from phydrax.linalg._small_batched import prepare_exact_small_linear_actions
from phydrax.meshing._failures import MeshingFailure
from phydrax.meshing._tetra_metric import CurvedCavityTransition, TetraMetricSource
from tests._support.coordinate_references import coordinate_reference_budget


_UNLIMITED = 10**9
_SHARED_TRACE = "implicit material charts must agree on every shared inverse trace"
_Stencil = tuple[np.ndarray, np.ndarray, np.ndarray]
_Outcome = tuple[object, ...]
# The lookup reads only chart rows; the paired snapshots stay empty.
_EMPTY_SNAPSHOT = TetMeshArrays(
    np.empty((0, 3), dtype=np.float64),
    np.empty((0, 4), dtype=np.int32),
    np.empty((0,), dtype=np.int32),
    np.empty((0, 3), dtype=np.int32),
    np.empty((0,), dtype=np.int32),
    np.empty((0, 2), dtype=np.int32),
    np.empty((0,), dtype=np.int32),
    np.empty((0,), dtype=np.int32),
    np.empty((0,), dtype=np.float64),
)


def _point(x: Fraction | int, y: Fraction | int, z: Fraction | int) -> Point:
    return Fraction(x), Fraction(y), Fraction(z)


def _volume(tetrahedron: Tet) -> Fraction:
    a, b, c, d = tetrahedron
    u, v, w = (tuple(p[axis] - a[axis] for axis in range(3)) for p in (b, c, d))
    return (
        u[0] * (v[1] * w[2] - v[2] * w[1])
        - u[1] * (v[0] * w[2] - v[2] * w[0])
        + u[2] * (v[0] * w[1] - v[1] * w[0])
    )


def _box(points: Sequence[Point]) -> Box:
    return (
        _point(*(min(point[axis] for point in points) for axis in range(3))),
        _point(*(max(point[axis] for point in points) for axis in range(3))),
    )


def _inside(point: Point, box: Box) -> bool:
    return all(box[0][axis] <= point[axis] <= box[1][axis] for axis in range(3))


def _transition(
    old: tuple[Tet, ...], new: tuple[Tet, ...], chronology: int
) -> CurvedCavityTransition:
    """Wrap positive paired chart rows with their exact closed row and union boxes."""
    assert all(_volume(a) > 0 and _volume(b) > 0 for a, b in zip(old, new, strict=True))
    old_bounds = tuple(_box(row) for row in old)
    new_bounds = tuple(_box(row) for row in new)
    atlas = ImplicitCavityAtlas(
        old, new, f"material-chart-{chronology}", (), (), "material-source",
        "material-discovery", chronology,
        f"material-chart-{chronology - 1}" if chronology else "",
        (0, 1), old[0][0], None, (), (), old_bounds, new_bounds,
        _box(tuple(corner for box in old_bounds for corner in box)),
        _box(tuple(corner for box in new_bounds for corner in box)),
    )
    identifiers = np.empty((0,), dtype=np.int64)
    return CurvedCavityTransition(
        atlas, _EMPTY_SNAPSHOT, _EMPTY_SNAPSHOT, identifiers, identifiers
    )


def _bipyramid(
    center: Point,
    bottom: tuple[Fraction, Fraction],
    top: tuple[Fraction, Fraction],
    chronology: int,
    *,
    disagreeing_top: Fraction | None = None,
) -> CurvedCavityTransition:
    """Anchor cones over a triangular bipyramid whose apexes move vertically.

    ``bottom`` and ``top`` are (old, new) apex distances from the rim plane.
    Consecutive cones share the anchor-apex axis, so axis points carry shared
    traces. ``disagreeing_top`` replaces only the first cone's new top height.
    """
    rim = tuple(
        _point(center[0] + dx, center[1] + dy, center[2])
        for dx, dy in (
            (Fraction(1), Fraction(0)),
            (Fraction(-1, 2), Fraction(2, 3)),
            (Fraction(-1, 2), Fraction(-2, 3)),
        )
    )

    def apex(height: Fraction) -> Point:
        return _point(center[0], center[1], center[2] + height)

    old: list[Tet] = []
    new: list[Tet] = []
    for side in range(3):
        first, second = rim[side], rim[(side + 1) % 3]
        moved = top[1] if disagreeing_top is None or side else disagreeing_top
        old.append((center, apex(top[0]), first, second))
        new.append((center, apex(moved), first, second))
        old.append((center, apex(-bottom[0]), second, first))
        new.append((center, apex(-bottom[1]), second, first))
    return _transition(tuple(old), tuple(new), chronology)


def _history() -> list[CurvedCavityTransition]:
    """Overlapping edits: expansion, bottom contraction, offset top contraction."""
    one = Fraction(1)
    return [
        _bipyramid(_point(0, 0, 0), (one, one), (one, Fraction(5, 3)), 0),
        _bipyramid(_point(0, 0, 2), (one, Fraction(1, 3)), (one, one), 1),
        _bipyramid(
            _point(Fraction(1, 3), Fraction(1, 5), 1), (one, one), (one, Fraction(1, 2)), 2
        ),
    ]


def _bank(history: Sequence[CurvedCavityTransition]) -> tuple[Point, ...]:
    """Nonbinary grid points plus every chart vertex, without repetition."""
    grid = (
        _point(x, y, z)
        for x, y, z in product(
            (0, Fraction(1, 7), Fraction(1, 3), Fraction(-1, 2)),
            (0, Fraction(1, 5), Fraction(-1, 3)),
            (
                Fraction(-1, 2), 0, Fraction(1, 3), 1, Fraction(5, 4), Fraction(3, 2),
                Fraction(5, 3), Fraction(7, 4), 2, Fraction(5, 2),
            ),
        )
    )
    vertices = (
        point
        for record in history
        for chart in (record.atlas.old_tets, record.atlas.new_tets)
        for row in chart
        for point in row
    )
    return tuple(dict.fromkeys((*grid, *vertices)))


def _source_tables(budget: CoordinateEnclosureBudget) -> TetraMetricSource:
    """Six positive Kuhn tetrahedra of the binary64 cube [-4, 4]^3."""
    points = np.asarray(tuple(product((-4.0, 4.0), repeat=3)), dtype=np.float64)
    paths = []
    for order in permutations(range(3)):
        corner = [0, 0, 0]
        path = [0]
        for axis in order:
            corner[axis] = 1
            path.append(4 * corner[0] + 2 * corner[1] + corner[2])
        paths.append(path)
    cells = np.asarray(paths, dtype=np.int32)
    negative = np.linalg.det(points[cells[:, 1:]] - points[cells[:, :1]]) < 0
    cells[negative] = cells[negative][:, (0, 2, 1, 3)]
    return TetraMetricSource(
        points, cells, np.arange(8, dtype=np.int64), np.arange(6, dtype=np.int64),
        budget=budget,
    )


def _reference_weights(
    corners: np.ndarray, point: Sequence[Fraction], budget: CoordinateEnclosureBudget
) -> tuple[Fraction, ...]:
    """The historical per-pair rational elimination of one tetrahedron."""
    if corners.dtype != object:
        corners = np.asarray(tuple(rational_vector(row) for row in corners), dtype=object)
    solved = prepare_exact_small_linear_actions(
        tuple(
            tuple(corners[column + 1, axis] - corners[0, axis] for column in range(3))
            for axis in range(3)
        ),
        tuple((point[axis] - corners[0, axis],) for axis in range(3)),
        budget=budget,
    )
    assert solved.actions is not None and solved.determinant > 0
    reference = tuple(row[0] for row in solved.actions)
    return (Fraction(1) - sum(reference, Fraction(0)), *reference)


def _reference_stencil(
    tables: TetraMetricSource,
    bank: Sequence[Point],
    history: Sequence[CurvedCavityTransition],
    maximum_pairs: int,
    counter: list[int],
    budget: CoordinateEnclosureBudget,
    trace: list[tuple[int, int, int]],
) -> _Stencil:
    """The complete reverse walk over every edit and point, then source location.

    ``trace`` receives (edit, point, nonnegative new rows) for every visit
    that charged at least one pair.
    """
    material: list[tuple[Fraction, ...]] = list(bank)
    admitted = np.ones((len(material),), dtype=np.bool_)
    work = 0

    def charge() -> None:
        nonlocal work
        work += 1
        counter[0] += 1
        if work > maximum_pairs:
            raise owner.source_location_budget(maximum_pairs, work)

    for chronology in range(len(history) - 1, -1, -1):
        atlas = history[chronology].atlas
        old, new = atlas.old_tets, atlas.new_tets
        for index, point in enumerate(material):
            image = None
            visited, accepted = False, 0
            hull = owner._inside_chart_box(point, atlas.new_hull, budget)
            for row in (range(len(new)) if hull else range(0)):
                if not owner._inside_chart_box(point, atlas.new_bounds[row], budget):
                    continue
                visited = True
                charge()
                coefficients = _reference_weights(
                    np.asarray(new[row], dtype=object), point, budget
                )
                if min(coefficients) < 0:
                    continue
                accepted += 1
                candidate = tuple(
                    sum(
                        (w * v[axis] for w, v in zip(coefficients, old[row], strict=True)),
                        Fraction(0),
                    )
                    for axis in range(3)
                )
                if image is not None and candidate != image:
                    raise ValueError(_SHARED_TRACE)
                image = candidate
            if image is not None:
                material[index] = image
            elif owner._inside_chart_box(point, atlas.old_hull, budget):
                for tetrahedron, box in zip(old, atlas.old_bounds, strict=True):
                    if not owner._inside_chart_box(point, box, budget):
                        continue
                    visited = True
                    charge()
                    if min(
                        _reference_weights(np.asarray(tetrahedron, dtype=object), point, budget)
                    ) >= 0:
                        admitted[index] = False
                        break
            if visited:
                trace.append((chronology, index, accepted))
    remaining = maximum_pairs - work
    tree = owner._metric_source_bvh(tables, budget=budget)
    with owner._exact_table_boxes(
        np.asarray(material, dtype=object)[:, None, :], budget=budget
    ) as (lower, upper):
        queries = prepare_bvh(lower, upper, dtype=np.float64)
    selected = np.full((len(material),), -1, dtype=np.int64)
    selected_ids = np.full((len(material),), np.iinfo(np.int64).max, dtype=np.int64)
    weights = np.zeros((len(material), 4), dtype=np.float64)
    located = 0
    for cells, slots in bvh_overlap_pair_blocks(tree, queries, include_touching=True):
        for cell, query in zip(cells.tolist(), slots.tolist(), strict=True):
            located += 1
            counter[0] += 1
            if located > remaining:
                raise owner.source_location_budget(remaining, located)
            coefficients = _reference_weights(tables.corners[cell], material[query], budget)
            if min(coefficients) < 0 or tables.cell_global_ids[cell] >= selected_ids[query]:
                continue
            selected[query], selected_ids[query] = cell, tables.cell_global_ids[cell]
            weights[query] = tuple(float(value) for value in coefficients)
    covered = selected >= 0
    weights[~covered, 0] = 1.0
    weights /= np.sum(weights, axis=1, keepdims=True)
    return (
        tables.vertex_global_ids[tables.cells[np.maximum(selected, 0)]],
        weights,
        np.broadcast_to(covered[:, None], weights.shape) & admitted[:, None],
    )


def _refusal(error: Exception, counter: list[int]) -> _Outcome:
    if isinstance(error, MeshingFailure):
        evidence = error.evidence
        return (
            "refused", type(error), str(error), evidence.category,
            evidence.requested, evidence.achieved, counter[0],
        )
    return ("refused", type(error), str(error), counter[0])


def _located(stencil: _Stencil, counter: list[int]) -> _Outcome:
    indices, weights, valid = stencil
    return ("located", indices.tolist(), weights.tolist(), valid.tolist(), counter[0])


def _reference_outcome(
    tables: TetraMetricSource,
    bank: Sequence[Point],
    history: Sequence[CurvedCavityTransition],
    maximum_pairs: int,
    budget: CoordinateEnclosureBudget,
    trace: list[tuple[int, int, int]] | None = None,
) -> _Outcome:
    counter = [0]
    try:
        stencil = _reference_stencil(
            tables, bank, history, maximum_pairs, counter, budget,
            [] if trace is None else trace,
        )
    except (MeshingFailure, ValueError) as error:
        return _refusal(error, counter)
    return _located(stencil, counter)


def _prepared_outcome(
    tables: TetraMetricSource,
    bank: Sequence[Point],
    history: Sequence[CurvedCavityTransition],
    maximum_pairs: int,
    budget: CoordinateEnclosureBudget,
    index: owner._MaterialHistoryIndex | None = None,
) -> _Outcome:
    counter = [0]
    try:
        stencil = owner._material_source_stencil(
            tables, np.asarray(bank, dtype=np.float64), history, maximum_pairs,
            require_coverage=False, work_counter=counter, exact_points=tuple(bank),
            prepared_history=index, budget=budget,
        )
    except (MeshingFailure, ValueError) as error:
        return _refusal(error, counter)
    return _located(stencil, counter)


def _measured_outcome(
    tables: TetraMetricSource,
    bank: Sequence[Point],
    history: Sequence[CurvedCavityTransition],
    budget: CoordinateEnclosureBudget,
    index: owner._MaterialHistoryIndex,
) -> tuple[_Outcome, int]:
    before = budget.work_units
    outcome = _prepared_outcome(tables, bank, history, _UNLIMITED, budget, index)
    return outcome, budget.work_units - before


def test_prepared_lookup_matches_complete_reverse_walk() -> None:
    history = _history()
    bank = _bank(history)
    trace: list[tuple[int, int, int]] = []
    with coordinate_reference_budget() as budget:
        tables = _source_tables(budget)
        expected = _reference_outcome(tables, bank, history, _UNLIMITED, budget, trace)
        actual = _prepared_outcome(tables, bank, history, _UNLIMITED, budget)
    assert expected[0] == "located"
    assert actual == expected
    valid = np.asarray(expected[3], dtype=np.bool_)[:, 0]
    changing = bank.index(_point(0, 0, Fraction(7, 4)))
    removed = bank.index(_point(0, 0, Fraction(3, 2)))
    earliest = history[0].atlas
    # Above the earliest support, this point reaches it only through edit 1's inverse.
    assert not _inside(bank[changing], earliest.new_hull)
    assert not _inside(bank[changing], earliest.old_hull)
    visits = {(chronology, index): accepted for chronology, index, accepted in trace}
    assert visits[(1, changing)] > 1 and (0, changing) in visits
    assert valid[changing] and not valid[removed]
    assert 0 < np.count_nonzero(valid) < valid.size


def test_prepared_lookup_follows_appended_reordered_and_truncated_histories() -> None:
    history = _history()
    bank = _bank(history)
    live = owner._CavityHistory()
    with coordinate_reference_budget() as budget, budget.live_storage() as storage:
        tables = _source_tables(budget)
        index = owner._MaterialHistoryIndex(storage)
        for count in range(len(history) + 1):
            if count:
                live.append(history[count - 1])
            expected = _reference_outcome(tables, bank, live, _UNLIMITED, budget)
            assert expected[0] == "located"
            leased = budget._live_bytes_upper
            assert _prepared_outcome(tables, bank, live, _UNLIMITED, budget) == expected
            # A call-scoped lookup releases everything it prepared.
            assert budget._live_bytes_upper == leased
            appended, appended_work = _measured_outcome(tables, bank, live, budget, index)
            bound = storage.bound
            repeated, repeated_work = _measured_outcome(tables, bank, live, budget, index)
            assert appended == repeated == expected
            # The unchanged append-only history prepares nothing twice.
            assert storage.bound == bound
            assert repeated_work < appended_work if count else repeated_work == appended_work
        # Ordinary sequences are authenticated in full; a reordered, truncated
        # or empty history keeps exactly a fresh lookup's preparation.
        for other in ([history[1], history[0], history[2]], [history[0]], []):
            expected = _reference_outcome(tables, bank, other, _UNLIMITED, budget)
            assert _prepared_outcome(tables, bank, other, _UNLIMITED, budget, index) == expected
            with budget.live_storage() as fresh:
                restarted = owner._MaterialHistoryIndex(fresh)
                assert _prepared_outcome(
                    tables, bank, other, _UNLIMITED, budget, restarted
                ) == expected
                assert storage.bound == fresh.bound
        assert storage.bound == 0




def test_prepared_lookup_keeps_shared_trace_refusal() -> None:
    one = Fraction(1)
    disagreeing = [
        _bipyramid(
            _point(0, 0, 0), (one, one), (one, Fraction(1, 3)), 0,
            disagreeing_top=Fraction(1, 2),
        )
    ]
    # The first point is located; the axis point has two different inverses.
    bank = (_point(Fraction(1, 7), 0, Fraction(-1, 3)), _point(0, 0, Fraction(1, 4)))
    with coordinate_reference_budget() as budget:
        tables = _source_tables(budget)
        expected = _reference_outcome(tables, bank, disagreeing, _UNLIMITED, budget)
        actual = _prepared_outcome(tables, bank, disagreeing, _UNLIMITED, budget)
    assert expected[:3] == ("refused", ValueError, _SHARED_TRACE)
    assert actual == expected


def test_prepared_lookup_refuses_low_work_at_the_reverse_walk_pair() -> None:
    history = _history()
    bank = (
        _point(0, 0, Fraction(7, 4)),
        _point(0, 0, Fraction(3, 2)),
        _point(Fraction(1, 7), Fraction(1, 5), Fraction(1, 3)),
        _point(Fraction(1, 3), Fraction(1, 5), Fraction(5, 4)),
    )
    with coordinate_reference_budget() as budget:
        tables = _source_tables(budget)
        complete = _reference_outcome(tables, bank, history, _UNLIMITED, budget)
        assert complete[0] == "located"
        total = complete[-1]
        assert isinstance(total, int) and total > len(bank)
        for maximum_pairs in range(total):
            expected = _reference_outcome(tables, bank, history, maximum_pairs, budget)
            assert expected[0] == "refused" and expected[-1] == maximum_pairs + 1
            assert _prepared_outcome(tables, bank, history, maximum_pairs, budget) == expected
        assert _prepared_outcome(tables, bank, history, total, budget) == complete
