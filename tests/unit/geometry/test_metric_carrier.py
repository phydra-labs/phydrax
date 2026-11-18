"""Exact carriers of the native metric source against their definition.

The reference below evaluates the definition of a carrier with Python
rationals: exact barycentric weights of the goal, partial sums rounded half to
even on each dyadic grid, and the correctly rounded image of the exact
combination. It shares no code with the native search, which works on
integers over one common denominator.
"""

from __future__ import annotations

from fractions import Fraction
from math import frexp, sqrt

import numpy as np
import pytest

from phydrax._meshcore import MetricCarrier, MetricCarrierOrder, MetricSource


type _Point = tuple[Fraction, Fraction, Fraction]
type _Simplex = tuple[tuple[int, ...], bool]


def _dot(a: tuple[Fraction, ...], b: tuple[Fraction, ...], /) -> Fraction:
    return sum((x * y for x, y in zip(a, b, strict=True)), Fraction(0))


def _determinant(
    a: tuple[Fraction, ...], b: tuple[Fraction, ...], c: tuple[Fraction, ...], /
) -> Fraction:
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _cell_weights(corners: list[_Point], query: _Point, /) -> tuple[Fraction, ...] | None:
    """Barycentric weights of ``query`` in a tetrahedron; ``None`` outside."""
    edges = [
        tuple(x - o for x, o in zip(corner, corners[0], strict=True))
        for corner in corners[1:]
    ]
    offset = tuple(x - o for x, o in zip(query, corners[0], strict=True))
    volume = _determinant(edges[0], edges[1], edges[2])
    if volume == 0:
        return None
    tail = (
        _determinant(offset, edges[1], edges[2]) / volume,
        _determinant(edges[0], offset, edges[2]) / volume,
        _determinant(edges[0], edges[1], offset) / volume,
    )
    weights = (1 - sum(tail, Fraction(0)), *tail)
    return weights if all(weight >= 0 for weight in weights) else None


def _simplex_weights(
    corners: list[_Point], query: _Point, /
) -> tuple[Fraction, ...] | None:
    """Weights of the orthogonal projection on an edge or a triangle."""
    edges = [
        tuple(x - o for x, o in zip(corner, corners[0], strict=True))
        for corner in corners[1:]
    ]
    offset = tuple(x - o for x, o in zip(query, corners[0], strict=True))
    if len(edges) == 1:
        gram = _dot(edges[0], edges[0])
        if gram == 0:
            return None
        tail: tuple[Fraction, ...] = (_dot(edges[0], offset) / gram,)
    else:
        g00, g01, g11 = (
            _dot(edges[0], edges[0]),
            _dot(edges[0], edges[1]),
            _dot(edges[1], edges[1]),
        )
        determinant = g00 * g11 - g01 * g01
        if determinant == 0:
            return None
        r0, r1 = _dot(edges[0], offset), _dot(edges[1], offset)
        tail = ((r0 * g11 - r1 * g01) / determinant, (r1 * g00 - r0 * g01) / determinant)
    weights = (1 - sum(tail, Fraction(0)), *tail)
    return weights if all(weight >= 0 for weight in weights) else None


def _reference(
    exact: tuple[_Point, ...],
    points: np.ndarray,
    cells: np.ndarray,
    target: np.ndarray,
    current: np.ndarray,
    simplices: tuple[_Simplex, ...] | None,
    order: MetricCarrierOrder,
    limit: int,
    cell: int | None,
    /,
) -> list[tuple[tuple[int, ...], tuple[int, ...], int, list[float], bool]]:
    query = (
        Fraction(float(target[0])),
        Fraction(float(target[1])),
        Fraction(float(target[2])),
    )
    located: tuple[tuple[int, ...], tuple[Fraction, ...], bool] | None = None
    if simplices is None:
        for index in range(len(cells)) if cell is None else (cell,):
            rows = tuple(int(row) for row in cells[index])
            weights = _cell_weights([exact[row] for row in rows], query)
            if weights is not None:
                located = rows, weights, False
                break
        goal = [float(value) for value in target]
    else:
        for rows, declared in simplices:
            if len(rows) not in (2, 3):
                continue
            weights = _simplex_weights([exact[row] for row in rows], query)
            if weights is not None:
                located = rows, weights, declared
                break
        if located is None:
            return []
        goal = [
            float(_dot(located[1], tuple(exact[row][axis] for row in located[0])))
            for axis in range(3)
        ]
    if located is None:
        return []
    rows, weights, declared = located
    moved = [goal[axis] - float(current[axis]) for axis in range(3)]
    distance = moved[0] * moved[0] + moved[1] * moved[1] + moved[2] * moved[2]
    support = sorted(set(rows))
    span = max(
        max(points[row][axis] for row in support)
        - min(points[row][axis] for row in support)
        for axis in range(3)
    )
    if not distance > 0.0 or not span > 0.0:
        return []
    first = min(max(frexp(span / sqrt(distance))[1] - 1, 1), 53)

    def at(
        bits: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...], int, list[float], bool] | None:
        scale = 1 << bits
        merged = dict.fromkeys(support, 0)
        previous = 0
        running = Fraction(0)
        for row, weight in zip(rows[:-1], weights[:-1], strict=True):
            running += weight
            # ``round`` of a rational is half to even.
            quotient = min(max(round(running * scale), previous), scale)
            merged[row] += quotient - previous
            previous = quotient
        merged[rows[-1]] += scale - previous
        image = [
            sum(
                (Fraction(merged[row], scale) * exact[row][axis] for row in support),
                Fraction(0),
            )
            for axis in range(3)
        ]
        rounded = [float(value) for value in image]
        is_exact = all(
            Fraction(value) == reference
            for value, reference in zip(rounded, image, strict=True)
        )
        delta = [rounded[axis] - goal[axis] for axis in range(3)]
        if not delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2] < distance:
            return None
        if simplices is not None and not is_exact and not declared:
            return None
        present = [row for row in support if merged[row]]
        return (
            tuple(present),
            tuple(merged[row] for row in present),
            bits,
            rounded,
            is_exact,
        )

    found = []
    match order:
        case MetricCarrierOrder.COARSE_FIRST:
            resolutions = list(range(first, 54))
        case MetricCarrierOrder.FINE_FIRST:
            resolutions = list(range(53, first - 1, -1))
        case MetricCarrierOrder.COARSE_THEN_FINE:
            # Up to the first carrier, then from the finest down.
            resolutions = []
            for bits in range(first, 54):
                resolutions.append(bits)
                if at(bits) is not None:
                    resolutions.extend(range(53, first - 1, -1))
                    break
    for bits in resolutions:
        carrier = at(bits)
        if carrier is not None:
            found.append(carrier)
    return found[:limit]


def _same(
    carriers: tuple[MetricCarrier, ...],
    expected: list[tuple[tuple[int, ...], tuple[int, ...], int, list[float], bool]],
    /,
) -> None:
    assert [
        (
            carrier.rows,
            carrier.weights,
            carrier.bits,
            carrier.point.tolist(),
            carrier.exact,
        )
        for carrier in carriers
    ] == expected


def _bank(
    denominator: int, /
) -> tuple[tuple[_Point, ...], np.ndarray, np.ndarray, MetricSource]:
    # Two cells on both sides of the face (1, 2, 3); with a denominator that
    # is not a power of two the exact points are not binary64 values.
    integers = ((0, 0, 0), (9, 0, 0), (0, 10, 0), (0, 0, 11), (8, 9, 10))
    exact = tuple(
        (Fraction(x, denominator), Fraction(y, denominator), Fraction(z, denominator))
        for x, y, z in integers
    )
    points = np.asarray([[float(value) for value in point] for point in exact])
    cells = np.asarray(((4, 1, 2, 3), (0, 1, 2, 3)), dtype=np.int64)
    tensors = np.broadcast_to(np.eye(3), (5, 3, 3))
    return (
        exact,
        points,
        cells,
        MetricSource(points, exact, cells, tensors, np.zeros((5, 3, 3))),
    )


@pytest.mark.parametrize("denominator", [8, 7], ids=["dyadic-points", "rational-points"])
def test_carriers_in_cells_follow_their_definition(denominator: int) -> None:
    exact, points, cells, source = _bank(denominator)
    generator = np.random.default_rng(20261117)
    located = 0
    try:
        for _ in range(60):
            weights = generator.dirichlet(np.ones(4))
            target = weights @ points[cells[int(generator.integers(2))]]
            current = target + generator.normal(scale=0.2, size=3)
            for order in MetricCarrierOrder:
                for cell in (None, 0, 1):
                    expected = _reference(
                        exact, points, cells, target, current, None, order, 54, cell
                    )
                    _same(
                        source.carriers(target, current, None, order, 54, cell=cell),
                        expected,
                    )
                    located += bool(expected)
            few = _reference(
                exact,
                points,
                cells,
                target,
                current,
                None,
                MetricCarrierOrder.COARSE_THEN_FINE,
                2,
                None,
            )
            _same(
                source.carriers(
                    target, current, None, MetricCarrierOrder.COARSE_THEN_FINE, 2
                ),
                few,
            )
        # A target outside every cell has no carrier.
        outside = np.asarray((5.0, 5.0, 5.0))
        assert (
            source.carriers(outside, points[0], None, MetricCarrierOrder.COARSE_FIRST, 54)
            == ()
        )
    finally:
        source.close()
    assert located > 100


@pytest.mark.parametrize("denominator", [8, 7], ids=["dyadic-points", "rational-points"])
def test_carriers_on_simplices_follow_their_definition(denominator: int) -> None:
    exact, points, cells, source = _bank(denominator)
    generator = np.random.default_rng(20261118)
    faces = ((1, 2, 3), (0, 1, 2), (0, 2, 3), (4, 1, 2))
    edges = ((1, 2), (0, 3), (4, 3))
    admitted = 0
    try:
        for _ in range(60):
            chosen: list[_Simplex] = []
            for _ in range(int(generator.integers(1, 4))):
                kind = int(generator.integers(5))
                if kind == 0:
                    chosen.append((edges[int(generator.integers(3))], False))
                elif kind == 1:
                    # A degenerate edge and a lone vertex locate nothing.
                    chosen.append(
                        ((2, 2), False) if generator.integers(2) else ((3,), False)
                    )
                else:
                    chosen.append(
                        (faces[int(generator.integers(4))], bool(generator.integers(2)))
                    )
            simplices = tuple(chosen)
            anchor = points[list(simplices[int(generator.integers(len(simplices)))][0])]
            weights = generator.dirichlet(np.ones(len(anchor)))
            if generator.integers(4) == 0:
                weights = weights * 1.6 - 0.6 / len(anchor)  # may leave the simplex
            target = weights @ anchor + generator.normal(scale=0.05, size=3)
            current = anchor[0] + generator.normal(scale=0.1, size=3)
            for order in MetricCarrierOrder:
                expected = _reference(
                    exact, points, cells, target, current, simplices, order, 54, None
                )
                _same(source.carriers(target, current, simplices, order, 54), expected)
                admitted += len(expected)
    finally:
        source.close()
    assert admitted > 100


def test_carrier_source_refuses_malformed_input() -> None:
    exact, points, cells, source = _bank(8)
    target = np.asarray((0.25, 0.25, 0.25))
    try:
        with pytest.raises(ValueError, match="shape"):
            source.carriers(
                target[:2], points[0], None, MetricCarrierOrder.COARSE_FIRST, 1
            )
        with pytest.raises(ValueError, match="carriers"):
            source.carriers(
                target, points[0], None, MetricCarrierOrder.COARSE_FIRST, 1, cell=2
            )
        with pytest.raises(ValueError, match="carriers"):
            source.carriers(
                target, points[0], (((0, 9), False),), MetricCarrierOrder.COARSE_FIRST, 1
            )
        with pytest.raises(ValueError, match="finite"):
            source.carriers(
                np.asarray((np.inf, 0.0, 0.0)),
                points[0],
                None,
                MetricCarrierOrder.COARSE_FIRST,
                1,
            )
        with pytest.raises(ValueError, match="one tensor per point"):
            MetricSource(points, exact, cells, np.zeros((4, 3, 3)), np.zeros((4, 3, 3)))
        with pytest.raises(ValueError, match="three rationals"):
            MetricSource(
                points, exact[:4], cells, np.zeros((5, 3, 3)), np.zeros((5, 3, 3))
            )
    finally:
        source.close()
    with pytest.raises(RuntimeError, match="closed"):
        source.carriers(target, points[0], None, MetricCarrierOrder.COARSE_FIRST, 1)
