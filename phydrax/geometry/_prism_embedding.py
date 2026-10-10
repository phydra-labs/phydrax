#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Continuous piecewise injectivity of a convex two-prism reference atlas."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from ..discretization._coordinate_enclosure import (
    _solve_exact,
    Expression,
    expression_reference_evaluate,
    expression_scale,
    expression_sum,
)


if TYPE_CHECKING:
    from ._mapped_embedding import _Cell
    from ._mesh_certificates import MeshCertificateLimits


def adjacent_prism_injectivity(
    first: _Cell, second: _Cell, limits: MeshCertificateLimits, maximum_work: int
) -> tuple[bool, int]:
    from ._mapped_embedding import (
        _children,
        _jacobian_midpoint,
        _projected_jacobian,
        _strict_spd,
    )

    if first.kind != second.kind or first.kind != "prism" or len(first.coordinates) != 3:
        return False, 0
    shared = sorted(set(first.vertices[:3]).intersection(second.vertices[:3]))
    if len(shared) != 2 or len(set(first.vertices).intersection(second.vertices)) != 4:
        return False, 0
    a = tuple(
        tuple(
            expression_reference_evaluate(value, point, first.domain)
            for value in first.coordinates
        )
        for point in first.reference_vertices
    )
    b = tuple(
        tuple(
            expression_reference_evaluate(value, point, second.domain)
            for value in second.coordinates
        )
        for point in second.reference_vertices
    )
    baseline = [[a[j][i] - a[0][i] for j in (1, 2, 3)] for i in range(3)]
    from ._mesh_certificates import _det3

    if not _det3(*tuple(np.asarray(row, dtype=object) for row in baseline)):
        return False, 0
    charts = []
    for point in b[:3]:
        result = _solve_exact(baseline, [[point[i] - a[0][i]] for i in range(3)])
        if result[2][0]:
            return False, 0
        charts.append(tuple(row[0] for row in result[:2]))
    reference = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    p, q = (reference[first.vertices.index(vertex)] for vertex in shared)
    c = reference[
        next(i for i, vertex in enumerate(first.vertices[:3]) if vertex not in shared)
    ]
    d = charts[
        next(i for i, vertex in enumerate(second.vertices[:3]) if vertex not in shared)
    ]

    def turn(
        x: tuple[Fraction, ...], y: tuple[Fraction, ...], z: tuple[Fraction, ...]
    ) -> Fraction:
        return (y[0] - x[0]) * (z[1] - x[1]) - (y[1] - x[1]) * (z[0] - x[0])

    side = turn(p, q, c)
    boundary = (c, p, d, q)
    if (
        not side
        or side * turn(p, q, d) >= 0
        or any(
            side * turn(boundary[i], boundary[(i + 1) % 4], boundary[(i + 2) % 4]) < 0
            for i in range(4)
        )
    ):
        return False, 0
    matrix = [
        [charts[j + 1][i] - charts[0][i] for j in range(2)] + [Fraction(0)]
        for i in range(2)
    ] + [[Fraction(0), Fraction(0), Fraction(1)]]
    inverse = _solve_exact(
        matrix, [[Fraction(i == j) for j in range(3)] for i in range(3)]
    )
    midpoint = _jacobian_midpoint(first)
    if midpoint is None or np.linalg.matrix_rank(midpoint) != 3:
        return False, 0
    projection = np.linalg.solve(midpoint, np.eye(3, dtype=np.float64))
    work = 0
    for cell, chart_inverse in (
        (first, [[Fraction(i == j) for j in range(3)] for i in range(3)]),
        (second, inverse),
    ):
        local = _projected_jacobian(cell, projection)
        if local is None:
            return False, work
        jacobian = tuple(
            tuple(
                expression_sum(
                    tuple(
                        expression_scale(local[i][k], chart_inverse[k][j])
                        for k in range(3)
                    )
                )
                for j in range(3)
            )
            for i in range(3)
        )
        active = [(np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64), 0)]
        symmetric_parts: dict[tuple[int, int], Expression] = {}
        while active:
            origin, piece, depth = active.pop()
            if work >= maximum_work:
                return False, work
            work += 1
            if _strict_spd(jacobian, "prism", 3, origin, piece, symmetric_parts):
                continue
            if depth == limits.maximum_subdivision_depth:
                return False, work
            active.extend(
                (start, child, depth + 1)
                for start, child in _children(cell, origin, piece)
            )
    return True, work
