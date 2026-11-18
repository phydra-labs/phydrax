"""Positions and moves of a collapse survivor against independent references.

The verdict of a position is checked on stars whose lengths and orientations
follow by hand. The moves of a survivor are checked against exact rational
arithmetic (the carrier is the correctly rounded image of an exact dyadic
combination of source points, closer to its goal than the survivor) and
against the separate measurement kernels for their tensors and lengths.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from phydrax._meshcore import (
    metric_collapse_positions,
    metric_interpolations,
    metric_lengths,
    metric_logarithms,
    metric_survivor_goals,
    MetricCarrierOrder,
    MetricCollapseStar,
    MetricPositionVerdict,
    MetricSearchEnd,
    MetricSource,
)


_UPPER = float(np.sqrt(2.0))
_FLOOR = 2.0**-22
_UNIT = np.eye(3, dtype=np.float64)


def _axis_star(before: tuple[float, ...], /) -> MetricCollapseStar:
    # The survivor far away, one spoke to the origin and one cell (origin,
    # survivor, e_y, e_z) that is positive for a survivor at (x, 0, 0), x > 0.
    return MetricCollapseStar(
        np.asarray(
            (
                (9.0, 9.0, 9.0),
                (8.0, 8.0, 8.0),
                (0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        ),
        np.broadcast_to(_UNIT, (5, 3, 3)),
        np.asarray((2,), dtype=np.int64),
        np.asarray(before, dtype=np.float64),
        np.asarray(((2, 0, 3, 4),), dtype=np.int64),
        _FLOOR,
    )


def _verdicts(
    star: MetricCollapseStar, positions: list[tuple[float, float, float]], /
) -> list[str]:
    verdicts, _, _ = metric_collapse_positions(
        star, np.asarray(positions), np.broadcast_to(_UNIT, (len(positions), 3, 3))
    )
    return [MetricPositionVerdict(int(value)).name for value in verdicts]


def test_position_verdicts_are_decided_in_their_stated_order() -> None:
    star = _axis_star((3.0,))
    # Coincidence precedes every measurement.
    assert _verdicts(star, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (3.0, 0.0, 0.0)]) == [
        "COINCIDENT",
        "ADMITTED",
        "NOT_DECREASING",
    ]
    # An unmeasured length precedes the comparison of violations.
    unmeasured = star._replace(before_lengths=np.asarray((3.0, np.nan)))
    assert _verdicts(unmeasured, [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0)]) == [
        "COINCIDENT",
        "UNMEASURED",
    ]
    # The violations precede the cells: a negative cell is only reached by a
    # position that decreases them.
    inverted = star._replace(cells=np.asarray(((0, 2, 3, 4),), dtype=np.int64))
    assert _verdicts(inverted, [(1.0, 0.0, 0.0), (3.0, 0.0, 0.0)]) == [
        "NOT_ORIENTED",
        "NOT_DECREASING",
    ]
    # The predicate domain precedes the orientation it guards.
    points = np.array(inverted.points)
    points[4, 1] = 2.0**-130
    assert _verdicts(inverted._replace(points=points), [(1.0, 0.0, 0.0)]) == [
        "OUTSIDE_DOMAIN"
    ]
    # The determinant floor counts as orientation; quality comes last.
    assert _verdicts(star._replace(determinant_floor=2.0), [(1.0, 0.0, 0.0)]) == [
        "NOT_ORIENTED"
    ]
    metrics = np.array(star.metrics)
    metrics[3, 0, 0] = -1.0
    assert _verdicts(star._replace(metrics=metrics), [(1.0, 0.0, 0.0)]) == ["NO_QUALITY"]


def test_positions_report_their_spoke_lengths_and_worst_quality() -> None:
    star = _axis_star((3.0,))
    verdicts, qualities, lengths = metric_collapse_positions(
        star,
        np.asarray(((1.0, 0.0, 0.0), (0.5, 0.0, 0.0))),
        np.broadcast_to(_UNIT, (2, 3, 3)),
    )
    # 0.5 is below the band: one short violation, fewer than one long one.
    assert verdicts.tolist() == [MetricPositionVerdict.ADMITTED] * 2
    assert lengths.tolist() == [[1.0], [0.5]]
    # A second, elongated cell lowers the reported quality.
    points = np.concatenate((star.points, ((0.0, 0.0, 9.0),)))
    two = star._replace(
        points=points,
        metrics=np.broadcast_to(_UNIT, (6, 3, 3)),
        cells=np.asarray(((2, 0, 3, 4), (2, 0, 3, 5)), dtype=np.int64),
    )
    _, worst, _ = metric_collapse_positions(
        two, np.asarray(((1.0, 0.0, 0.0),)), _UNIT[None]
    )
    assert 0.0 < worst[0] < qualities[0]


def _source() -> tuple[
    MetricSource, tuple[tuple[Fraction, ...], ...], np.ndarray, np.ndarray
]:
    # One cell with corners at thirds: images of dyadic combinations are in
    # general not binary64 values.
    exact = tuple(
        tuple(Fraction(value, 3) for value in point)
        for point in ((0, 0, 0), (25, 0, 0), (0, 25, 0), (0, 0, 25))
    )
    points = np.asarray([[float(value) for value in point] for point in exact])
    metrics = np.stack([(1.0 + row / 8.0) * _UNIT for row in range(4)])
    logarithms, valid = metric_logarithms(metrics)
    assert np.all(valid)
    source = MetricSource(
        points, exact, np.asarray(((0, 1, 2, 3),), dtype=np.int64), metrics, logarithms
    )
    return source, exact, metrics, logarithms


def _interior_star(
    before: tuple[float, ...], spokes: tuple[int, ...], /
) -> MetricCollapseStar:
    # The survivor at (1, 1, 1), the removed vertex at (2, 1, 1) and link
    # vertices above, behind, below, ahead and far away from the survivor.
    return MetricCollapseStar(
        np.asarray(
            (
                (1.0, 1.0, 1.0),
                (2.0, 1.0, 1.0),
                (1.0, 2.5, 1.0),
                (1.0, 1.0, 2.5),
                (0.25, 0.5, 0.5),
                (2.5, 1.0, 1.0),
                (1.0, 1.0, -3.0),
            )
        ),
        np.broadcast_to(_UNIT, (7, 3, 3)),
        np.asarray(spokes, dtype=np.int64),
        np.asarray(before, dtype=np.float64),
        np.asarray(((0, 2, 3, 5),), dtype=np.int64),
        _FLOOR,
    )


def _squared_distance(a: np.ndarray, b: np.ndarray, /) -> Fraction:
    return sum(
        (
            (Fraction(float(x)) - Fraction(float(y))) ** 2
            for x, y in zip(a, b, strict=True)
        ),
        Fraction(0),
    )


def test_survivor_moves_are_exact_carriers_with_the_measured_tensors_and_lengths() -> (
    None
):
    source, exact, metrics, logarithms = _source()
    # The far spoke (length 4 before) stays above the band at every carrier
    # but below the longest edge before: the search runs to its end.
    star = _interior_star((6.0, 1.5, 1.5, 1.5), (2, 3, 5, 6))
    try:
        moves, end = source.survivor_moves(star, None, True)
    finally:
        source.close()
    goals, valid = metric_survivor_goals(star.points, star.metrics[:2], None)
    assert np.all(valid)
    assert end is MetricSearchEnd.EXHAUSTED
    assert len(moves) > 3
    assert [move.goal for move in moves] == sorted(move.goal for move in moves)
    for move in moves:
        # An exact dyadic combination of distinct source points in ascending
        # row order, and its correctly rounded image.
        assert list(move.rows) == sorted(set(move.rows))
        assert all(weight > 0 for weight in move.weights)
        assert sum(move.weights) == 1 << move.bits
        image = [
            sum(
                (
                    Fraction(weight, 1 << move.bits) * exact[row][axis]
                    for row, weight in zip(move.rows, move.weights, strict=True)
                ),
                Fraction(0),
            )
            for axis in range(3)
        ]
        assert move.point.tolist() == [float(value) for value in image]
        # Closer to its goal than the survivor.
        assert _squared_distance(move.point, goals[move.goal]) < _squared_distance(
            star.points[0], goals[move.goal]
        )
        # The tensor and the spoke lengths are those of the measurement kernels.
        tensor, has_tensor = metric_interpolations(
            np.asarray((0, len(move.rows)), dtype=np.int64),
            np.asarray(move.rows, dtype=np.int64),
            np.asarray([weight / (1 << move.bits) for weight in move.weights]),
            metrics,
            logarithms,
        )
        assert has_tensor[0]
        assert move.metric.tobytes() == tensor[0].tobytes()
        moved_points = np.array(star.points)
        moved_points[0] = move.point
        moved_metrics = np.array(star.metrics)
        moved_metrics[0] = move.metric
        spokes = np.stack((np.zeros_like(star.spokes), star.spokes), axis=1)
        assert (
            move.lengths.tobytes()
            == metric_lengths(moved_points, moved_metrics, spokes).tobytes()
        )
        # It is an admitted position of the star with the reported quality.
        verdicts, qualities, _ = metric_collapse_positions(
            star, move.point[None], move.metric[None]
        )
        assert verdicts[0] == MetricPositionVerdict.ADMITTED
        assert qualities[0] == move.quality


def test_survivor_search_ends_as_its_star_demands() -> None:
    source, _, _, _ = _source()
    try:
        # One spoke, ahead of the survivor: the first move that brings it
        # into the band ends the search.
        moves, end = source.survivor_moves(_interior_star((1.5,), (5,)), None, True)
        assert end is MetricSearchEnd.IN_BAND
        assert moves and moves[-1].lengths[0] <= _UPPER
        assert all(move.lengths[0] > _UPPER for move in moves[:-1])
        # The coarsest carrier of each goal only.
        coarse, _ = source.survivor_moves(
            _interior_star((3.0, 1.5, 1.5, 1.5), (2, 3, 5)), None, False
        )
        assert 0 < len(coarse) <= 3
        assert len({move.goal for move in coarse}) == len(coarse)
        # An unmeasured length before: the first carrier is referred, not a move.
        referred, end = source.survivor_moves(
            _interior_star((0.0,), (2, 3, 5)), None, True
        )
        assert end is MetricSearchEnd.UNMEASURED
        assert len(referred) == 1
        # On the face x = 0 of the cell every move stays in the face.
        face = MetricCollapseStar(
            np.asarray(
                (
                    (0.0, 1.0, 1.0),
                    (0.0, 2.0, 1.0),
                    (0.0, 1.0, 2.5),
                    (1.5, 1.0, 1.0),
                    (0.0, 2.5, 1.0),
                )
            ),
            np.broadcast_to(_UNIT, (5, 3, 3)),
            np.asarray((2, 3, 4), dtype=np.int64),
            np.asarray((3.0, 1.5, 1.5, 1.5)),
            np.asarray(((0, 4, 2, 3),), dtype=np.int64),
            _FLOOR,
        )
        # The face is declared: its carriers need not be exact images.
        on_face, _ = source.survivor_moves(face, (((0, 2, 3), True),), True)
        assert on_face
        assert all(
            move.point[0] == 0.0 and set(move.rows) <= {0, 2, 3} for move in on_face
        )
        # Undeclared, the inexact images of the thirds are not admitted.
        exact_only, _ = source.survivor_moves(face, (((0, 2, 3), False),), True)
        carriers = source.carriers(
            np.asarray((0.0, 1.5, 1.0)),
            face.points[0],
            (((0, 2, 3), False),),
            MetricCarrierOrder.COARSE_FIRST,
            53,
        )
        assert all(carrier.exact for carrier in carriers)
        assert len(exact_only) <= len(on_face)
    finally:
        source.close()


def test_collapse_bindings_refuse_malformed_stars() -> None:
    source, _, _, _ = _source()
    star = _interior_star((3.0,), (2, 3, 5))
    position = np.asarray(((1.5, 1.0, 1.0),))
    try:
        with pytest.raises(ValueError, match="at least one cell"):
            metric_collapse_positions(
                star._replace(cells=np.empty((0, 4), dtype=np.int64)),
                position,
                _UNIT[None],
            )
        with pytest.raises(ValueError, match="one tensor per point"):
            metric_collapse_positions(
                star._replace(metrics=star.metrics[:5]), position, _UNIT[None]
            )
        with pytest.raises(ValueError, match="one tensor per position"):
            metric_collapse_positions(star, position, np.broadcast_to(_UNIT, (2, 3, 3)))
        with pytest.raises(ValueError, match="n >= 2"):
            source.survivor_moves(
                star._replace(
                    points=star.points[:1],
                    metrics=star.metrics[:1],
                    spokes=np.empty((0,), dtype=np.int64),
                    cells=np.zeros((1, 4), dtype=np.int64),
                ),
                None,
                True,
            )
        # A spoke to the survivor itself and a cell naming an absent point.
        with pytest.raises(ValueError, match="collapse_positions"):
            metric_collapse_positions(
                star._replace(spokes=np.asarray((0,), dtype=np.int64)),
                position,
                _UNIT[None],
            )
        with pytest.raises(ValueError, match="collapse_positions"):
            metric_collapse_positions(
                star._replace(cells=np.asarray(((0, 2, 3, 7),), dtype=np.int64)),
                position,
                _UNIT[None],
            )
    finally:
        source.close()
    with pytest.raises(RuntimeError, match="closed"):
        source.survivor_moves(star, None, True)
