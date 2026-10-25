#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Certified tangent-cone contacts of adjacent codimension-one mapped patches.

Two polynomial patches ``A`` and ``B`` of dimension ``d - 1`` in ``R^d`` share a
vertex or an edge and agree identically on it. Each chart is re-based so that
the shared entity sits at the parameter origin (vertex) or on ``t = 0``
(edge, parameterized from its smaller to its larger global vertex id).

Vertex lemma. With ``P`` the shared image, ``F_A(x) - P = sum_i x_i Q_i(x)`` for
coefficientwise polynomial quotients ``Q_i`` and ``x >= 0`` on the chart. If a vector
``w`` is strictly positive on enclosures of every ``Q_i`` over an ``A`` piece
and strictly negative on those of ``B`` over a ``B`` piece, then
``w . (F_A(x) - P) > 0`` unless ``x = 0`` and ``w . (F_B(y) - P) < 0`` unless
``y = 0``: the two pieces meet at most in ``P``.

Edge lemma (``d = 3``). Write ``F_A(s, t) = g(s) + t D_A(s, t)`` and
``F_B(s, u) = g(s) + u D_B(s, u)`` with coefficientwise quotients ``D``. A coincidence of
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

Mesh certificate checks
-----------------------
[
{"checks":["mapped_cell_contact","mapped_boundary_injectivity"],"owner":"phydrax.geometry._tangent_cone_contact.tangent_cone_contact","premises":[{"tag":"X","statement":"Rebased shared vertex or edge expressions agree exactly and quotient generators recover every outgoing source displacement."},{"tag":"N","statement":"Every generator box has one strict sign against a fixed rounded separating vector; edge contacts also enclose the shared-edge mean-value chord direction."}],"conclusion":"A separated tangent-cone decision proves that the closed codimension-one patches meet only on their shared entity; an unresolved result is a fallback request, not a negative geometry conclusion.","citation":{"kind":"mathematical","reference":"Rockafellar, Convex Analysis","locator":"Section 11, separation theorems","url":"https://doi.org/10.1515/9781400873173"},"failure_statuses":[]}
]
End mesh certificate checks
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from typing import final

import numpy as np

import phydrax.geometry._embedding_cells as embedding_cells_owner

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._coordinate_enclosure import (
    affine_arguments,
    BernsteinDomain,
    derivative,
    ExpressionComposition as ArgumentComposition,
    Polynomial,
    RationalPolynomial,
)
from ..discretization._reference_cell import reference_cell_topology
from ..typing import checked
from ._embedding_cells import Cell
from ._embedding_state import ContactDecision, MeshCertificateLimits
from ._rational import rational_vector


type _Box = tuple[tuple[Fraction, ...], tuple[Fraction, ...]]
type _Chart = tuple[np.ndarray, np.ndarray]

# The piece workspace holds two patches and their quotient-generator bounds.
_CONE_PIECE_BASE_BYTES = 4096
_CONE_GENERATOR_COMPONENT_BYTES = 512


def _entity_chart(cell: Cell, shared: tuple[int, ...]) -> _Chart | None:
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


def _rebased(
    cell: embedding_cells_owner.Cell, chart: _Chart, *, budget: CoordinateEnclosureBudget
) -> embedding_cells_owner.Cell:

    composition = ArgumentComposition(
        affine_arguments(*chart, budget=budget), budget=budget
    )
    return cell.composed(composition, budget=budget)


def _quotient(
    polynomial: Polynomial,
    axis: int,
    *,
    free: tuple[int, ...],
    budget: CoordinateEnclosureBudget,
) -> Polynomial:
    """Divide ``p - p|_{x_axis = 0}`` coefficientwise by ``x_axis`` with zero ``free`` exponents."""

    budget.reserve(len(polynomial), 256 + 384 * len(polynomial))
    return {
        tuple(value - int(i == axis) for i, value in enumerate(index)): coefficient
        for index, coefficient in polynomial.items()
        if index[axis] and not any(index[other] for other in free)
    }


def _corners(domain: BernsteinDomain, dimension: int) -> np.ndarray:
    if domain == "simplex":
        return np.vstack((np.zeros((1, dimension)), np.eye(dimension, dtype=np.float64)))
    grid = np.indices((2,) * dimension, dtype=np.float64).reshape(dimension, -1)
    return grid.T


def _box(cell: Cell, chart: _Chart, *, budget: CoordinateEnclosureBudget) -> _Box:

    lower, upper = embedding_cells_owner.bounds(cell, *chart, budget=budget)
    return (
        rational_vector(lower),
        rational_vector(upper),
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


def _strict_side(
    boxes: list[_Box],
    normal: tuple[Fraction, ...],
    sign: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:

    for lower, upper in boxes:
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


def _separated_cones(
    first: list[_Box], second: list[_Box], *, budget: CoordinateEnclosureBudget
) -> bool:
    """Prove strict separation of box-generated cones using rational arithmetic."""
    a, b = _direction(first), _direction(second)
    if a is None or b is None:
        return False
    candidate = a - b
    if not np.all(np.isfinite(candidate)) or not np.any(candidate):
        return False
    normal = rational_vector(candidate)
    return _strict_side(first, normal, 1, budget=budget) and _strict_side(
        second, normal, -1, budget=budget
    )


def _parameter_range(chart: _Chart, domain: BernsteinDomain) -> tuple[Fraction, Fraction]:
    origin, matrix = chart
    values = [
        Fraction(float(value))
        for value in (origin + _corners(domain, matrix.shape[1]) @ matrix.T)[:, 0]
    ]
    return min(values), max(values)


@final
class _Cones(StrictModule, NonTrainableState):
    """Re-based patches and the rational quotient expressions of one contact."""

    patches: tuple[Cell, Cell]
    generators: tuple[tuple[Cell, ...], tuple[Cell, ...]]
    # ``g'`` of the shared edge trace; ``None`` for a shared vertex.
    tangent: Cell | None


def _prepare_cones(
    first: embedding_cells_owner.Cell,
    second: embedding_cells_owner.Cell,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _Cones | None:
    """Check applicability, shared-entity coefficient identity, and quotient expressions."""

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
    patches = (
        _rebased(first, charts[0], budget=budget),
        _rebased(second, charts[1], budget=budget),
    )
    polynomials = tuple(
        tuple(value for value in patch.coordinates if isinstance(value, dict))
        for patch in patches
    )

    budget.reserve(sum(len(value) for value in (*polynomials[0], *polynomials[1])))

    def generator(
        patch: embedding_cells_owner.Cell, values: tuple[Polynomial, ...]
    ) -> embedding_cells_owner.Cell:
        return replace(patch, coordinates=values, corner_images=None)

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
            (
                generator(
                    patch,
                    tuple(
                        _quotient(value, 1, free=(), budget=budget) for value in values
                    ),
                ),
            )
            for patch, values in zip(patches, polynomials, strict=True)
        )
        tangent = embedding_cells_owner.Cell(
            tuple(derivative(value, 0, budget=budget) for value in traces[0]),
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
                    _quotient(value, axis, free=tuple(range(axis)), budget=budget)
                    for value in values
                ),
            )
            for axis in range(dimension)
        )
        for patch, values in zip(patches, polynomials, strict=True)
    )
    return _Cones(patches, (first_vertex, second_vertex), None)


def _piece_cones(
    cones: _Cones, a: _Chart, b: _Chart, /, *, budget: CoordinateEnclosureBudget
) -> tuple[list[_Box], list[_Box]]:
    """Box generators whose strict separation proves the two pieces disjoint."""
    if cones.tangent is None:
        return (
            [_box(cell, a, budget=budget) for cell in cones.generators[0]],
            [_box(cell, b, budget=budget) for cell in cones.generators[1]],
        )
    range_a = _parameter_range(a, cones.patches[0].domain)
    range_b = _parameter_range(b, cones.patches[1].domain)
    low = min(range_a[0], range_b[0])
    high = max(range_a[1], range_b[1])
    hull = (
        np.asarray((float(low), 0.0), dtype=np.float64),
        np.asarray(((float(high - low), 0.0), (0.0, 0.0)), dtype=np.float64),
    )
    slope = _box(cones.tangent, hull, budget=budget)
    return (
        [_cross(_box(cones.generators[0][0], a, budget=budget), slope)],
        [_cross(_box(cones.generators[1][0], b, budget=budget), slope)],
    )


@checked
def tangent_cone_contact(
    first: Cell,
    second: Cell,
    limits: MeshCertificateLimits,
    maximum_work: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> ContactDecision | None:
    """Prove that adjacent codimension-one patches meet only on their shared entity.

    Returns ``None`` when the lemma does not apply (not codimension one, no or
    an unsupported shared entity, rational expressions, nonidentical traces),
    otherwise a ``ContactDecision`` with charged subdivision work.
    """

    cones = _prepare_cones(first, second, budget=budget)
    if cones is None:
        return None
    dimension = first.dimension
    identity = (
        np.zeros((dimension,), dtype=np.float64),
        np.eye(dimension, dtype=np.float64),
    )

    work = 0
    with budget.live_storage() as live:
        active = embedding_cells_owner.BisectionWorklist(
            dimension, dimension, live, first=identity, second=identity
        )
        while active:
            with budget.temporary_scope():
                a, b, depth = active.pop()
                work += 1
                if work > maximum_work:
                    return ContactDecision("unresolved", work - 1)
                budget.reserve(
                    0,
                    _CONE_PIECE_BASE_BYTES
                    + _CONE_GENERATOR_COMPONENT_BYTES
                    * (dimension + 1)
                    * (2 + 2 * dimension),
                )
                # At the root, the exact quotient-cone lemma can discharge
                # the whole contact without preparing either image box.
                # Boundary broadphase already needed whole-face enclosures;
                # repeat them here only when the cone proof needs subdivision.
                root_separated = depth == 0 and _separated_cones(
                    *_piece_cones(cones, a, b, budget=budget), budget=budget
                )
                if root_separated:
                    active.release_piece()
                    continue
                box_a = embedding_cells_owner.bounds(cones.patches[0], *a, budget=budget)
                box_b = embedding_cells_owner.bounds(cones.patches[1], *b, budget=budget)
                if (
                    np.any(box_a[1] < box_b[0])
                    or np.any(box_b[1] < box_a[0])
                    or (
                        depth != 0
                        and _separated_cones(
                            *_piece_cones(cones, a, b, budget=budget), budget=budget
                        )
                    )
                ):
                    active.release_piece()
                    continue
                if depth == limits.maximum_subdivision_depth:
                    return ContactDecision("unresolved", work)
                children: tuple[tuple[_Chart, _Chart, int], ...]
                if np.max(box_a[1] - box_a[0]) >= np.max(box_b[1] - box_b[0]):
                    children = tuple(
                        (child, b, depth + 1)
                        for child in embedding_cells_owner.children(cones.patches[0], *a)
                    )
                else:
                    children = tuple(
                        (a, child, depth + 1)
                        for child in embedding_cells_owner.children(cones.patches[1], *b)
                    )
                if len(active) + len(children) > maximum_work - work:
                    return ContactDecision("unresolved", work)
                active.bound_children(len(children))
                active.extend(children)
    return ContactDecision("separated", work)
