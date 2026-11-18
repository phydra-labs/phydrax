"""The candidate sequence of a relocated vertex against its definition.

The reference below is the definition of the sequence written out one goal
and one carrier at a time: goals of the pulls in order with dyadic
backtracking, projection, removal of repeated goals, the screen, the exact
carriers of the mode, removal of repeated stencils, the tensor of a stencil
and the evaluation of a position. It takes exact carriers from
``MetricSource.carriers`` and lengths, band quanta, determinants, qualities
and tensors from the measurement kernels, each of which has its own
reference tests; what is checked here is their composition and its order.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from fractions import Fraction

import numpy as np
import pytest

from phydrax._meshcore import (
    metric_band_quanta,
    metric_hull_frame,
    metric_interpolations,
    metric_lengths,
    metric_logarithms,
    metric_project_goals,
    metric_relative_determinants,
    metric_tetra_quality,
    MetricCarrierOrder,
    MetricRelocationCandidate,
    MetricRelocationEnd,
    MetricRelocationSearch,
    MetricRelocationStar,
    MetricSource,
)


_FLOOR = 2.0**-22
_IMPROVEMENT = 1.0e-6
_UNIT = np.eye(3, dtype=np.float64)

type _Simplices = tuple[tuple[tuple[int, ...], bool], ...]
type _Group = tuple[tuple[object, ...], ...]


def _source(graded: bool, /) -> tuple[MetricSource, np.ndarray, np.ndarray, np.ndarray]:
    # Two cells on both sides of the face (1, 2, 3), corners at sevenths: the
    # images of dyadic combinations are in general not binary64 values.
    exact = tuple(
        tuple(Fraction(value, 7) for value in point)
        for point in ((0, 0, 0), (9, 0, 0), (0, 10, 0), (0, 0, 11), (8, 9, 10))
    )
    points = np.asarray([[float(value) for value in point] for point in exact])
    scales = (4.0, 4.5, 5.0, 3.5, 4.25) if graded else (4.0,) * 5
    metrics = np.stack([scale * _UNIT for scale in scales])
    logarithms, valid = metric_logarithms(metrics)
    assert np.all(valid)
    cells = np.asarray(((0, 1, 2, 3), (4, 1, 3, 2)), dtype=np.int64)
    return (
        MetricSource(points, exact, cells, metrics, logarithms),
        points,
        metrics,
        logarithms,
    )


def _oriented(points: np.ndarray, cell: tuple[int, int, int, int], /) -> tuple[int, ...]:
    a, b, c, d = (points[row] for row in cell)
    positive = float(np.linalg.det(np.stack((b - a, c - a, d - a)))) > 0.0
    return cell if positive else (cell[0], cell[1], cell[3], cell[2])


def _star(
    points: np.ndarray,
    metrics: np.ndarray,
    vertex: np.ndarray,
    tensor: np.ndarray,
    corners: tuple[int, ...],
    cells: tuple[tuple[int, int, int, int], ...],
    measured: tuple[bool, ...],
    /,
) -> MetricRelocationStar:
    """The vertex with the source points ``corners`` as neighbors; ``cells``
    name star rows (0 is the vertex) and are oriented here."""
    star_points = np.concatenate((vertex[None], points[list(corners)]))
    star_metrics = np.concatenate((tensor[None], metrics[list(corners)]))
    star_cells = np.asarray(
        [_oriented(star_points, cell) for cell in cells], dtype=np.int64
    )
    spokes = np.asarray([(0, row) for row in range(1, len(corners) + 1)], dtype=np.int64)
    lengths = metric_lengths(star_points, star_metrics, spokes)
    quality = float(np.min(metric_tetra_quality(star_points, star_metrics, star_cells)))
    return MetricRelocationStar(
        star_points,
        star_metrics,
        np.asarray(measured, dtype=np.bool_),
        lengths,
        quality,
        star_cells,
        _FLOOR,
        _IMPROVEMENT,
    )


def _violations(
    lengths: np.ndarray, /
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    above, below = metric_band_quanta(lengths)
    if not (np.all(np.isfinite(above)) and np.all(np.isfinite(below))):
        return None
    return (
        tuple(sorted((float(value) for value in above if value != 0.0), reverse=True)),
        tuple(sorted((float(value) for value in below if value != 0.0), reverse=True)),
    )


class _Unmeasured(Exception):
    pass


def _reference(
    source: MetricSource,
    source_metrics: np.ndarray,
    source_logarithms: np.ndarray,
    star: MetricRelocationStar,
    pulls: np.ndarray,
    frame: tuple[np.ndarray, np.ndarray] | None,
    simplices: _Simplices | None,
    /,
    *,
    deep: bool,
    refine: bool,
    band_only: bool,
    constant_field: bool,
) -> Iterator[_Group]:
    points, metrics, cells = star.points, star.metrics, star.cells
    vertex = points[0]
    spokes = np.asarray([(0, row) for row in range(1, points.shape[0])], dtype=np.int64)
    before = _violations(star.before_lengths[star.measured])
    assert before is not None

    def floor_oriented(position: np.ndarray) -> bool:
        moved = points.copy()
        moved[0] = position
        determinants, relative = metric_relative_determinants(moved[cells])
        return bool(np.all(determinants > 0.0) and np.all(relative > _FLOOR))

    def evaluate(
        position: np.ndarray, tensor: np.ndarray
    ) -> tuple[float, np.ndarray] | None:
        if any(np.array_equal(position, other) for other in points[1:]):
            return None
        moved, moved_metrics = points.copy(), metrics.copy()
        moved[0], moved_metrics[0] = position, tensor
        lengths = metric_lengths(moved, moved_metrics, spokes)
        after = _violations(lengths[star.measured])
        if after is None:
            raise _Unmeasured
        if after > before or (band_only and after == before):
            return None
        if not floor_oriented(position):
            return None
        quality = float(np.min(metric_tetra_quality(moved, moved_metrics, cells)))
        if after == before and not quality > star.before_quality + _IMPROVEMENT:
            return None
        return quality, lengths

    def goals() -> Iterator[np.ndarray]:
        seen: set[bytes] = set()
        for pull in pulls:
            step = 0
            while deep or step <= 5:
                goal = vertex + np.ldexp(pull, -step)
                if np.array_equal(goal, vertex):
                    break
                step += 1
                if frame is not None:
                    goal = metric_project_goals(goal[None], frame[0], frame[1])[0]
                key = goal.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                if constant_field:
                    if evaluate(goal, metrics[0]) is None:
                        continue
                elif not floor_oriented(goal):
                    continue
                yield goal

    examined: set[tuple[tuple[int, Fraction], ...]] = set()

    def group(goal: np.ndarray, remaining: bool) -> _Group:
        if refine:
            carriers = source.carriers(
                goal,
                vertex,
                simplices,
                MetricCarrierOrder.FINE_FIRST,
                54 if remaining else 1,
            )
            carriers = carriers[1:] if remaining else carriers
        elif deep:
            carriers = source.carriers(
                goal, vertex, simplices, MetricCarrierOrder.COARSE_THEN_FINE, 2
            )
        else:
            carriers = source.carriers(
                goal, vertex, simplices, MetricCarrierOrder.COARSE_FIRST, 1
            )
        found = []
        for carrier in carriers:
            if any(np.array_equal(carrier.point, other) for other in points):
                continue
            stencil = tuple(
                (row, Fraction(weight, 1 << carrier.bits))
                for row, weight in zip(carrier.rows, carrier.weights, strict=True)
            )
            if stencil in examined:
                continue
            examined.add(stencil)
            rows = np.asarray(carrier.rows, dtype=np.int64)
            weights = np.asarray(
                [weight / (1 << carrier.bits) for weight in carrier.weights],
                dtype=np.float64,
            )
            tensors, valid = metric_interpolations(
                np.asarray((0, rows.size), dtype=np.int64),
                rows,
                weights,
                source_metrics,
                source_logarithms,
            )
            assert valid[0]
            evaluated = evaluate(carrier.point, tensors[0])
            if evaluated is None:
                continue
            found.append(
                (
                    carrier.rows,
                    carrier.weights,
                    carrier.bits,
                    carrier.point.tobytes(),
                    tensors[0].tobytes(),
                    evaluated[0],
                    evaluated[1].tobytes(),
                )
            )
        return tuple(found)

    kept = []
    for goal in goals():
        if refine:
            kept.append(goal)
        found = group(goal, False)
        if found:
            yield found
    for goal in kept:
        found = group(goal, True)
        if found:
            yield found


def _record(candidate: MetricRelocationCandidate, /) -> tuple[object, ...]:
    return (
        candidate.rows,
        candidate.weights,
        candidate.bits,
        candidate.point.tobytes(),
        candidate.metric.tobytes(),
        candidate.quality,
        candidate.lengths.tobytes(),
    )


def _pulls(star: MetricRelocationStar, /) -> np.ndarray:
    # Toward and away from every neighbor, along the axes, and repeats: a
    # generated stream of directions, some of them equal, some of them zero.
    generator = np.random.default_rng(20261118)
    vertex = star.points[0]
    toward = [0.5 * (other - vertex) for other in star.points[1:]]
    away = [-0.25 * (other - vertex) for other in star.points[1:]]
    axes = [0.2 * axis for axis in (*_UNIT, *(-_UNIT))]
    drawn = list(generator.normal(scale=0.2, size=(12, 3)))
    return np.asarray([*toward, *axes, np.zeros(3), *away, *toward[:2], *drawn])


_MODES = (
    pytest.param(False, False, id="coarsest"),
    pytest.param(True, False, id="deep"),
    pytest.param(False, True, id="refine"),
    pytest.param(True, True, id="deep-refine"),
)


@pytest.mark.parametrize(("deep", "refine"), _MODES)
@pytest.mark.parametrize("band_only", [False, True], ids=["any-gain", "band-only"])
@pytest.mark.parametrize("graded", [False, True], ids=["constant-field", "graded-field"])
def test_interior_candidates_follow_their_definition(
    deep: bool, refine: bool, band_only: bool, graded: bool
) -> None:
    source, points, metrics, logarithms = _source(graded)
    try:
        # A vertex of the first cell, off its center so that spokes violate
        # the band on both sides; the last spoke is not measured.
        vertex = np.asarray((0.2, 0.3, 0.25))
        star = _star(
            points,
            metrics,
            vertex,
            metrics[0],
            (0, 1, 2, 3),
            ((0, 2, 3, 4), (1, 0, 3, 4), (1, 2, 0, 4), (1, 2, 3, 0)),
            (True, True, True, False),
        )
        pulls = _pulls(star)
        mode = {
            "deep": deep,
            "refine": refine,
            "band_only": band_only,
            "constant_field": not graded,
        }
        expected = list(
            _reference(source, metrics, logarithms, star, pulls, None, None, **mode)
        )
        search = source.relocation_search(star, pulls, None, None, **mode)
        groups = []
        while True:
            group, end = search.next()
            if end is MetricRelocationEnd.EXHAUSTED:
                break
            assert end is MetricRelocationEnd.GROUP and group
            groups.append(tuple(_record(candidate) for candidate in group))
        assert groups == expected
        # Exhausted stays exhausted.
        assert search.next() == ((), MetricRelocationEnd.EXHAUSTED)
        assert sum(map(len, groups)) > (0 if band_only and not deep else 2)
    finally:
        source.close()


@pytest.mark.parametrize(("deep", "refine"), _MODES)
@pytest.mark.parametrize("declared", [False, True], ids=["exact-face", "declared-face"])
def test_boundary_candidates_follow_their_definition(
    deep: bool, refine: bool, declared: bool
) -> None:
    source, points, metrics, logarithms = _source(True)
    try:
        # A vertex on the face (1, 2, 3) shared by the two cells: it moves in
        # the face, goals are projected onto its plane.
        face = (1, 2, 3)
        weights = np.asarray((0.5, 0.25, 0.25))
        vertex = weights @ points[list(face)]
        star = _star(
            points,
            metrics,
            vertex,
            metrics[1],
            (0, 1, 2, 3, 4),
            (
                (1, 0, 3, 4),
                (1, 2, 0, 4),
                (1, 2, 3, 0),
                (5, 0, 3, 4),
                (5, 2, 0, 4),
                (5, 2, 3, 0),
            ),
            (True,) * 5,
        )
        _, projector = metric_hull_frame(points[list(face)])
        frame = (points[face[0]], projector)
        simplices: _Simplices = ((face, declared),)
        pulls = _pulls(star)
        mode = {
            "deep": deep,
            "refine": refine,
            "band_only": False,
            "constant_field": False,
        }
        expected = list(
            _reference(source, metrics, logarithms, star, pulls, frame, simplices, **mode)
        )
        search = source.relocation_search(star, pulls, frame, simplices, **mode)
        groups = []
        while True:
            group, end = search.next()
            if end is MetricRelocationEnd.EXHAUSTED:
                break
            assert end is MetricRelocationEnd.GROUP
            groups.append(tuple(_record(candidate) for candidate in group))
        assert groups == expected
        # Over sevenths no interior combination of the face is a binary64
        # point: only a declared face has carriers.
        assert bool(groups) == declared
    finally:
        source.close()


def test_two_searches_do_not_share_their_position() -> None:
    source, points, metrics, _ = _source(False)
    try:
        star = _star(
            points,
            metrics,
            np.asarray((0.2, 0.3, 0.25)),
            metrics[0],
            (0, 1, 2, 3),
            ((0, 2, 3, 4), (1, 0, 3, 4), (1, 2, 0, 4), (1, 2, 3, 0)),
            (True,) * 4,
        )
        pulls = _pulls(star)
        mode = {"deep": True, "refine": True, "band_only": False, "constant_field": True}

        def drain(search: MetricRelocationSearch) -> list[_Group]:
            groups = []
            while True:
                group, end = search.next()
                if end is MetricRelocationEnd.EXHAUSTED:
                    return groups
                groups.append(tuple(_record(candidate) for candidate in group))

        alone = drain(source.relocation_search(star, pulls, None, None, **mode))
        assert len(alone) > 3
        # The same search advanced in turns with a second one.
        first = source.relocation_search(star, pulls, None, None, **mode)
        second = source.relocation_search(star, pulls[::-1].copy(), None, None, **mode)
        taken = []
        for _ in range(len(alone)):
            group, end = first.next()
            assert end is MetricRelocationEnd.GROUP
            taken.append(tuple(_record(candidate) for candidate in group))
            second.next()
        assert taken == alone
    finally:
        source.close()


def test_a_position_without_a_measured_length_is_returned_for_report() -> None:
    source, points, metrics, _ = _source(False)
    try:
        # A neighbor so far away that its squared metric length is finite at
        # the vertex and overflows one step further out.
        far = np.asarray((0.2, 0.3, 0.25 + 6.0e153))
        star_points = np.concatenate(
            (np.asarray(((0.2, 0.3, 0.25),)), points[:4], far[None])
        )
        star_metrics = np.broadcast_to(4.0 * _UNIT, (6, 3, 3)).copy()
        cells = np.asarray(
            [
                _oriented(star_points, cell)
                for cell in ((0, 2, 3, 4), (1, 0, 3, 4), (1, 2, 0, 4), (1, 2, 3, 0))
            ],
            dtype=np.int64,
        )
        spokes = np.asarray([(0, row) for row in range(1, 6)], dtype=np.int64)
        lengths = metric_lengths(star_points, star_metrics, spokes)
        assert math.isfinite(float(lengths[-1]))
        star = MetricRelocationStar(
            star_points,
            star_metrics,
            np.ones((5,), dtype=np.bool_),
            lengths,
            float(np.min(metric_tetra_quality(star_points, star_metrics, cells))),
            cells,
            _FLOOR,
            _IMPROVEMENT,
        )
        pulls = np.asarray(((0.0, 0.0, -2.0e153),))
        search = source.relocation_search(
            star,
            pulls,
            None,
            None,
            deep=False,
            refine=False,
            band_only=False,
            constant_field=True,
        )
        group, end = search.next()
        assert end is MetricRelocationEnd.UNMEASURED_GOAL
        assert len(group) == 1 and group[0].rows == ()
        np.testing.assert_array_equal(group[0].point, star_points[0] + pulls[0])
        assert not math.isfinite(float(group[0].lengths[-1]))
        # A star that is unmeasured as it stands has no search.
        with pytest.raises(ValueError):
            source.relocation_search(
                star._replace(before_lengths=np.asarray((1.0, 1.0, 1.0, 1.0, np.inf))),
                pulls,
                None,
                None,
                deep=False,
                refine=False,
                band_only=False,
                constant_field=True,
            )
    finally:
        source.close()


def test_a_search_ends_with_its_source() -> None:
    source, points, metrics, _ = _source(False)
    star = _star(
        points,
        metrics,
        np.asarray((0.2, 0.3, 0.25)),
        metrics[0],
        (0, 1, 2, 3),
        ((0, 2, 3, 4), (1, 0, 3, 4), (1, 2, 0, 4), (1, 2, 3, 0)),
        (True,) * 4,
    )
    search = source.relocation_search(
        star,
        _pulls(star),
        None,
        None,
        deep=False,
        refine=False,
        band_only=False,
        constant_field=True,
    )
    source.close()
    with pytest.raises(RuntimeError, match="closed"):
        search.next()
    with pytest.raises(RuntimeError, match="closed"):
        source.relocation_search(
            star,
            _pulls(star),
            None,
            None,
            deep=False,
            refine=False,
            band_only=False,
            constant_field=True,
        )
    search.close()
