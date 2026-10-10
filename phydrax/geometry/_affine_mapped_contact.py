#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Closed affine simplex contacts proved from complete source expressions."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from ..discretization._coordinate_enclosure import (
    _solve_exact,
    expression_evaluate,
    RationalPolynomial,
)


if TYPE_CHECKING:
    from ._mapped_embedding import _Cell


def affine_simplex_contact(
    first: _Cell,
    second: _Cell,
    maximum_work: int,
    *,
    authoritative_first_traces: tuple[tuple[int, ...], ...] = (),
) -> tuple[str, int] | None:
    if (
        first.kind not in ("triangle", "tetrahedron")
        or second.kind != first.kind
        or len(first.coordinates) != first.dimension
        or len(second.coordinates) != first.dimension
    ):
        return None
    if any(
        isinstance(value, RationalPolynomial) or any(sum(index) > 1 for index in value)
        for cell in (first, second)
        for value in cell.coordinates
    ):
        return None
    dimension = first.dimension
    for trace in authoritative_first_traces:
        if (
            not trace
            or len(set(trace)) > dimension
            or not set(trace).issubset(first.vertices)
        ):
            raise ValueError(
                "Authoritative affine contact must be a proper first-simplex trace."
            )
    vertices = tuple(
        tuple(
            tuple(expression_evaluate(value, point) for value in cell.coordinates)
            for point in cell.reference_vertices
        )
        for cell in (first, second)
    )
    planes = []
    for points in vertices:
        matrix = [
            [points[j + 1][i] - points[0][i] for j in range(dimension)]
            for i in range(dimension)
        ]
        inverse = _solve_exact(
            matrix,
            [[Fraction(i == j) for j in range(dimension)] for i in range(dimension)],
        )
        rows = [
            (
                tuple(row),
                -sum((a * x for a, x in zip(row, points[0], strict=True)), Fraction(0)),
            )
            for row in inverse
        ]
        normal = tuple(
            -sum((row[axis] for row in inverse), Fraction(0)) for axis in range(dimension)
        )
        offset = Fraction(1) - sum((row[1] for row in rows), Fraction(0))
        planes.extend(((normal, offset), *rows))
    candidates = set()
    work = 0
    from ..discretization._coordinate_enclosure import _COORDINATE_BUDGET
    from ._mesh_certificates import _det2, _det3

    budget = _COORDINATE_BUDGET.get()
    bits = max(
        (
            abs(value.numerator).bit_length() + value.denominator.bit_length()
            for normal, offset in planes
            for value in (*normal, offset)
        ),
        default=1,
    )
    storage_upper = (
        4 * dimension**2 * (128 + 8 * ((12 * (dimension + 1) * bits + 29) // 30))
    )

    for selected in combinations(planes, dimension):
        if work >= maximum_work:
            return "mapped_intersection_piece_budget", work
        work += 1
        if budget is not None:
            budget.reserve(
                6 * dimension**3 * (dimension + 1) + 4 * dimension * len(planes),
                storage_upper,
            )
        matrix = [list(row[0]) for row in selected]
        determinant = (
            _det2(*tuple(np.asarray(row, dtype=object) for row in matrix))
            if dimension == 2
            else _det3(*tuple(np.asarray(row, dtype=object) for row in matrix))
        )
        if not determinant:
            continue
        point = tuple(
            row[0] for row in _solve_exact(matrix, [[-row[1]] for row in selected])
        )
        if all(
            sum((a * x for a, x in zip(normal, point, strict=True)), offset) >= 0
            for normal, offset in planes
        ):
            candidates.add(point)
    shared = set(first.vertices).intersection(second.vertices)
    # Auxiliary simplices need not triangulate an authored parent face in the
    # same way. A caller may supply only independently authenticated proper
    # parent traces. ALL intersection vertices must lie in ONE such convex
    # trace; accepting a union point-by-point could hide a volume overlap.
    for trace in (tuple(shared), *authoritative_first_traces):
        supported = set(trace)
        accepted = True
        for point in candidates:
            for identifier, (normal, offset) in zip(
                first.vertices,
                planes[: dimension + 1],
                strict=True,
            ):
                if identifier in supported:
                    continue
                if budget is not None:
                    budget.admit_work_bound(dimension)
                    budget.reserve(dimension)
                value = sum((a * x for a, x in zip(normal, point, strict=True)), offset)
                if value:
                    accepted = False
                    break
            if not accepted:
                break
        if accepted:
            return "separated", work
    return "overlap", work
