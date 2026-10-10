#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Certified tangent-cone contacts of adjacent codimension-one mapped patches.

Two polynomial patches ``A`` and ``B`` of dimension ``d - 1`` in ``R^d`` share a
vertex or an edge and agree identically on it. Each chart is re-based so that
the shared entity sits at the parameter origin (vertex) or on ``t = 0``
(edge, parameterized from its smaller to its larger global vertex id).

Vertex lemma. With ``P`` the shared image, ``F_A(x) - P = sum_i x_i Q_i(x)`` for
exact polynomial quotients ``Q_i`` and ``x >= 0`` on the chart. If a vector
``w`` is strictly positive on enclosures of every ``Q_i`` over an ``A`` piece
and strictly negative on those of ``B`` over a ``B`` piece, then
``w . (F_A(x) - P) > 0`` unless ``x = 0`` and ``w . (F_B(y) - P) < 0`` unless
``y = 0``: the two pieces meet at most in ``P``.

Edge lemma (``d = 3``). Write ``F_A(s, t) = g(s) + t D_A(s, t)`` and
``F_B(s, u) = g(s) + u D_B(s, u)`` with exact quotients ``D``. A coincidence of
two pieces whose edge parameters lie in ``I`` gives
``t D_A - u D_B = (s' - s) G`` with ``G`` the componentwise mean-value vector of
``g'`` on ``I``, hence ``t (D_A x G) = u (D_B x G)``. A vector ``w`` strictly
positive on the interval enclosure of ``D_A x G`` and strictly negative on that
of ``D_B x G`` forces ``t = u = 0``: the pieces meet only on the shared edge.

Both lemmas hold for arbitrary sub-pieces, so adaptive subdivision closes
curved contacts; nearly affine patches whose affine counterparts have a
positive angular gap certify on their root pieces. Coincident (folded) tangent
cones never admit a separating ``w`` and stay undecided. Enclosures are
outward-rounded Bernstein bounds; every separation sign is decided exactly on
rational interval endpoints.
"""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from fractions import Fraction
from typing import NamedTuple, TYPE_CHECKING

import numpy as np

from ..discretization._coordinate_enclosure import (
    _COORDINATE_BUDGET,
    affine_arguments,
    derivative,
    ExpressionComposition as ArgumentComposition,
    Polynomial,
    RationalPolynomial,
)
from ..discretization._reference_cell import reference_cell_topology


if TYPE_CHECKING:
    from ._mapped_embedding import _Cell
    from ._mesh_certificates import MeshCertificateLimits


type _Box = tuple[tuple[Fraction, ...], tuple[Fraction, ...]]
type _Chart = tuple[np.ndarray, np.ndarray]


def _entity_chart(cell: _Cell, shared: tuple[int, ...]) -> _Chart | None:
    """Affine reference chart with the shared entity at the origin or on ``t = 0``."""
    reference = np.asarray(reference_cell_topology(cell.kind).vertices, dtype=np.float64)
    local = [cell.vertices.index(vertex) for vertex in shared]
    count = len(cell.vertices)
    start = local[0]
    if cell.kind == "interval":
        if len(shared) != 1:
            return None
        return reference[start], (reference[1 - start] - reference[start])[:, None]
    if len(shared) == 1:
        if cell.kind == "triangle":
            first, second = (start + 1) % 3, (start + 2) % 3
        elif cell.kind == "quadrilateral":
            first, second = (start + 1) % 4, (start - 1) % 4
        else:
            return None
    elif len(shared) == 2:
        first = local[1]
        if cell.kind == "triangle":
            second = 3 - start - first
        elif cell.kind == "quadrilateral":
            if (first - start) % count not in (1, count - 1):
                return None
            second = (start - 1) % 4 if (start + 1) % 4 == first else (start + 1) % 4
        else:
            return None
    else:
        return None
    matrix = np.stack(
        (reference[first] - reference[start], reference[second] - reference[start]),
        axis=1,
    )
    return reference[start], matrix


def _rebased(cell: _Cell, chart: _Chart) -> _Cell:
    from ._mapped_embedding import _Cell

    composition = ArgumentComposition(affine_arguments(*chart))
    return _Cell(
        tuple(composition(value) for value in cell.coordinates),
        cell.domain,
        cell.kind,
        cell.dimension,
        cell.vertices,
        cell.reference_vertices,
    )


def _quotient(polynomial: Polynomial, axis: int, *, free: tuple[int, ...]) -> Polynomial:
    """Exact ``(p - p|_{x_axis = 0}) / x_axis`` restricted to zero ``free`` exponents."""
    budget = _COORDINATE_BUDGET.get()
    if budget is not None:
        budget.reserve(len(polynomial), 256 + 384 * len(polynomial))
    return {
        tuple(value - int(i == axis) for i, value in enumerate(index)): coefficient
        for index, coefficient in polynomial.items()
        if index[axis] and not any(index[other] for other in free)
    }


def _corners(domain: str, dimension: int) -> np.ndarray:
    if domain == "simplex":
        return np.vstack((np.zeros((1, dimension)), np.eye(dimension, dtype=np.float64)))
    grid = np.indices((2,) * dimension, dtype=np.float64).reshape(dimension, -1)
    return grid.T


def _box(cell: _Cell, chart: _Chart) -> _Box:
    from ._mapped_embedding import _bounds

    lower, upper = _bounds(cell, *chart)
    return (
        tuple(Fraction(float(value)) for value in lower),
        tuple(Fraction(float(value)) for value in upper),
    )


def _product(
    first: tuple[Fraction, Fraction], second: tuple[Fraction, Fraction]
) -> tuple[Fraction, Fraction]:
    values = tuple(a * b for a in first for b in second)
    return min(values), max(values)


def _cross(first: _Box, second: _Box) -> _Box:
    """Interval enclosure of ``{a x b}`` over two coordinate boxes."""
    lower, upper = [], []
    for axis in range(3):
        i, j = (axis + 1) % 3, (axis + 2) % 3
        left = _product((first[0][i], first[1][i]), (second[0][j], second[1][j]))
        right = _product((first[0][j], first[1][j]), (second[0][i], second[1][i]))
        lower.append(left[0] - right[1])
        upper.append(left[1] - right[0])
    return tuple(lower), tuple(upper)


def _direction(boxes: list[_Box]) -> np.ndarray | None:
    total = np.zeros((len(boxes[0][0]),), dtype=np.float64)
    for lower, upper in boxes:
        middle = np.asarray(
            [float(a + b) for a, b in zip(lower, upper, strict=True)], dtype=np.float64
        )
        norm = float(np.linalg.norm(middle))
        if not norm or not np.isfinite(norm):
            return None
        total += middle / norm
    norm = float(np.linalg.norm(total))
    return total / norm if norm and np.isfinite(norm) else None


def _strict_side(boxes: list[_Box], normal: tuple[Fraction, ...], sign: int) -> bool:
    budget = _COORDINATE_BUDGET.get()
    for lower, upper in boxes:
        if budget is not None:
            budget.reserve(4 * len(normal))
        extreme = sum(
            (
                min(w * a, w * b) if sign > 0 else max(w * a, w * b)
                for w, a, b in zip(normal, lower, upper, strict=True)
            ),
            Fraction(0),
        )
        if extreme * sign <= 0:
            return False
    return True


def _separated_cones(first: list[_Box], second: list[_Box]) -> bool:
    """Exact strict separation of two box-generated cones by one rounded vector."""
    a, b = _direction(first), _direction(second)
    if a is None or b is None:
        return False
    candidate = a - b
    if not np.all(np.isfinite(candidate)) or not np.any(candidate):
        return False
    normal = tuple(Fraction(float(value)) for value in candidate)
    return _strict_side(first, normal, 1) and _strict_side(second, normal, -1)


def _parameter_range(chart: _Chart, domain: str) -> tuple[Fraction, Fraction]:
    origin, matrix = chart
    values = [
        Fraction(float(value))
        for value in (origin + _corners(domain, matrix.shape[1]) @ matrix.T)[:, 0]
    ]
    return min(values), max(values)


class _Cones(NamedTuple):
    """Re-based patches and the exact quotient expressions of one contact."""

    patches: tuple[_Cell, _Cell]
    generators: tuple[tuple[_Cell, ...], tuple[_Cell, ...]]
    # ``g'`` of the shared edge trace; ``None`` for a shared vertex.
    tangent: _Cell | None


def _prepare_cones(first: _Cell, second: _Cell, /) -> _Cones | None:
    """Applicability, exact shared-entity identity and quotient expressions."""
    from ._mapped_embedding import _Cell

    dimension = first.dimension
    if (
        dimension not in (1, 2)
        or second.dimension != dimension
        or len(first.coordinates) != dimension + 1
        or len(second.coordinates) != dimension + 1
    ):
        return None
    shared = tuple(sorted(set(first.vertices).intersection(second.vertices)))
    if not shared or (len(shared) == 2 and dimension != 2):
        return None
    if any(
        isinstance(value, RationalPolynomial)
        for cell in (first, second)
        for value in cell.coordinates
    ):
        return None
    charts = (_entity_chart(first, shared), _entity_chart(second, shared))
    if charts[0] is None or charts[1] is None:
        return None
    patches = (_rebased(first, charts[0]), _rebased(second, charts[1]))
    polynomials = tuple(
        tuple(value for value in patch.coordinates if isinstance(value, dict))
        for patch in patches
    )
    budget = _COORDINATE_BUDGET.get()
    if budget is not None:
        budget.reserve(sum(len(value) for value in (*polynomials[0], *polynomials[1])))

    def generator(patch: _Cell, values: tuple[Polynomial, ...]) -> _Cell:
        return _Cell(
            values,
            patch.domain,
            patch.kind,
            dimension,
            patch.vertices,
            patch.reference_vertices,
        )

    if len(shared) == 2:
        traces = tuple(
            tuple(
                {index: value for index, value in polynomial.items() if not index[1]}
                for polynomial in coordinates
            )
            for coordinates in polynomials
        )
        if traces[0] != traces[1]:
            return None
        first_edge, second_edge = (
            (generator(patch, tuple(_quotient(value, 1, free=()) for value in values)),)
            for patch, values in zip(patches, polynomials, strict=True)
        )
        tangent = _Cell(
            tuple(derivative(value, 0) for value in traces[0]),
            "box",
            "quadrilateral",
            2,
            first.vertices,
            first.reference_vertices,
        )
        return _Cones(patches, (first_edge, second_edge), tangent)
    origins = tuple(
        tuple(value.get((0,) * dimension, Fraction(0)) for value in values)
        for values in polynomials
    )
    if origins[0] != origins[1]:
        return None
    first_vertex, second_vertex = (
        tuple(
            generator(
                patch,
                tuple(
                    _quotient(value, axis, free=tuple(range(axis))) for value in values
                ),
            )
            for axis in range(dimension)
        )
        for patch, values in zip(patches, polynomials, strict=True)
    )
    return _Cones(patches, (first_vertex, second_vertex), None)


def _piece_cones(cones: _Cones, a: _Chart, b: _Chart, /) -> tuple[list[_Box], list[_Box]]:
    """Box generators whose strict separation proves the two pieces disjoint."""
    if cones.tangent is None:
        return (
            [_box(cell, a) for cell in cones.generators[0]],
            [_box(cell, b) for cell in cones.generators[1]],
        )
    range_a = _parameter_range(a, cones.patches[0].domain)
    range_b = _parameter_range(b, cones.patches[1].domain)
    low = min(range_a[0], range_b[0])
    high = max(range_a[1], range_b[1])
    hull = (
        np.asarray((float(low), 0.0), dtype=np.float64),
        np.asarray(((float(high - low), 0.0), (0.0, 0.0)), dtype=np.float64),
    )
    slope = _box(cones.tangent, hull)
    return (
        [_cross(_box(cones.generators[0][0], a), slope)],
        [_cross(_box(cones.generators[1][0], b), slope)],
    )


def tangent_cone_contact(
    first: _Cell,
    second: _Cell,
    limits: MeshCertificateLimits,
    maximum_work: int,
) -> tuple[bool, int] | None:
    """Prove that adjacent codimension-one patches meet only on their shared entity.

    Returns ``None`` when the lemma does not apply (not codimension one, no or
    an unsupported shared entity, rational expressions, nonidentical traces),
    otherwise ``(proved, work)`` with ``work`` charged subdivision pieces.
    """
    from ._mapped_embedding import _bounds, _children

    cones = _prepare_cones(first, second)
    if cones is None:
        return None
    dimension = first.dimension
    identity = (
        np.zeros((dimension,), dtype=np.float64),
        np.eye(dimension, dtype=np.float64),
    )
    budget = _COORDINATE_BUDGET.get()
    chart_bytes = 1024 + 32 * dimension**2
    work = 0
    with nullcontext() if budget is None else budget.live_storage() as live:
        if live is not None:
            live.set_bound(1024 + 2 * chart_bytes)
        active: deque[tuple[_Chart, _Chart, int]] = deque(((identity, identity, 0),))
        while active:
            with nullcontext() if budget is None else budget.temporary_scope():
                a, b, depth = active.popleft()
                work += 1
                if work > maximum_work:
                    return False, work - 1
                if budget is not None:
                    budget.reserve(0, 4096 + 512 * (dimension + 1) * (2 + 2 * dimension))
                box_a = _bounds(cones.patches[0], *a)
                box_b = _bounds(cones.patches[1], *b)
                if (
                    np.any(box_a[1] < box_b[0])
                    or np.any(box_b[1] < box_a[0])
                    or _separated_cones(*_piece_cones(cones, a, b))
                ):
                    if live is not None:
                        live.set_bound(1024 + chart_bytes * (len(active) + 1))
                    continue
                if depth == limits.maximum_subdivision_depth:
                    return False, work
                children: tuple[tuple[_Chart, _Chart, int], ...]
                if np.max(box_a[1] - box_a[0]) >= np.max(box_b[1] - box_b[0]):
                    children = tuple(
                        (child, b, depth + 1) for child in _children(cones.patches[0], *a)
                    )
                else:
                    children = tuple(
                        (a, child, depth + 1) for child in _children(cones.patches[1], *b)
                    )
                if len(active) + len(children) > maximum_work - work:
                    return False, work
                if live is not None:
                    live.set_bound(1024 + chart_bytes * (len(active) + len(children) + 1))
                active.extend(children)
    return True, work
