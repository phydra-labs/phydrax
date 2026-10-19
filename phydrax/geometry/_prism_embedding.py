#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Continuous piecewise injectivity of a convex two-prism reference atlas.

Mesh certificate checks
-----------------------
[
{"checks":["mapped_cell_contact"],"owner":"phydrax.geometry._prism_embedding.adjacent_prism_injectivity","premises":[{"tag":"X","statement":"The two prism base triangles glue across their shared edge into a convex quadrilateral and the corresponding source facet traces agree."},{"tag":"N","statement":"Every source piece has a strictly positive symmetric Jacobian in one common physical projection."}],"conclusion":"A successful two-prism proof is injective on the full glued atlas, so contact is confined to its shared facet.","citation":{"kind":"mathematical","reference":"Gale and Nikaido, The Jacobian matrix and global univalence of mappings (1965)","locator":"Positive quasi-definite Jacobian criterion, pp. 81-93","url":"https://doi.org/10.1007/BF01360282"},"failure_statuses":[]}
]
End mesh certificate checks
"""

from __future__ import annotations

from fractions import Fraction
from typing import final

import numpy as np

import phydrax.geometry._embedding_cells as embedding_cells_owner
import phydrax.geometry._mapped_embedding as mapped_embedding_owner
import phydrax.linalg._small_batched as small_batched_owner
from phydrax.linalg._exact import solve_exact

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._coordinate_enclosure import (
    ExactScalar,
    Expression,
    expression_reference_evaluate,
    expression_scale,
    expression_sum,
)
from ..discretization._exact_algebraic_geometry import (
    RadicalElement,
    reserve_algebraic_work,
)
from ..typing import checked
from ._embedding_cells import Cell
from ._embedding_state import MeshCertificateLimits


@final
class _PrismAtlas(StrictModule, NonTrainableState):
    projection: np.ndarray
    inverse: list[list[ExactScalar]]


def _prism_atlas(
    first: Cell, second: Cell, shared: list[int], *, budget: CoordinateEnclosureBudget
) -> _PrismAtlas | None:
    """Prepare the compatible prism chart and verify convex base gluing."""
    a = tuple(
        tuple(
            expression_reference_evaluate(value, point, first.domain, budget=budget)
            for value in first.coordinates
        )
        for point in first.reference_vertices
    )
    b = tuple(
        tuple(
            expression_reference_evaluate(value, point, second.domain, budget=budget)
            for value in second.coordinates
        )
        for point in second.reference_vertices
    )
    if any(isinstance(value, RadicalElement) for point in (*a, *b) for value in point):
        # Four 3-by-3 solves, one determinant, and chart differences fit 512
        # scalar operations and 256 simultaneously live field elements.
        reserve_algebraic_work((*a, *b), 512, 256, 6, budget=budget)
    baseline: list[list[ExactScalar]] = [
        [a[j][i] - a[0][i] for j in (1, 2, 3)] for i in range(3)
    ]

    if not small_batched_owner.cofactor_determinant_3(
        *tuple(np.asarray(row, dtype=object) for row in baseline)
    ):
        return None
    charts = []
    for point in b[:3]:
        result = solve_exact(baseline, [[point[i] - a[0][i]] for i in range(3)])
        if result[2][0]:
            return None
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

    if not embedding_cells_owner.convex_triangle_union(p, q, c, d, budget=budget):
        return None
    matrix: list[list[ExactScalar]] = [
        [charts[j + 1][i] - charts[0][i] for j in range(2)] + [Fraction(0)]
        for i in range(2)
    ]
    matrix.append([Fraction(0), Fraction(0), Fraction(1)])
    identity: list[list[ExactScalar]] = [
        [Fraction(i == j) for j in range(3)] for i in range(3)
    ]
    inverse = solve_exact(matrix, identity)
    midpoint = mapped_embedding_owner.jacobian_midpoint(first, budget=budget)
    if midpoint is None or np.linalg.matrix_rank(midpoint) != 3:
        return None
    projection = np.linalg.solve(midpoint, np.eye(3, dtype=np.float64))
    return _PrismAtlas(projection, inverse)


@checked
def adjacent_prism_injectivity(
    first: Cell,
    second: Cell,
    limits: MeshCertificateLimits,
    maximum_work: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[bool, int]:

    if first.kind != second.kind or first.kind != "prism" or len(first.coordinates) != 3:
        return False, 0
    shared = sorted(set(first.vertices[:3]).intersection(second.vertices[:3]))
    if len(shared) != 2 or len(set(first.vertices).intersection(second.vertices)) != 4:
        return False, 0
    atlas = _prism_atlas(first, second, shared, budget=budget)
    if atlas is None:
        return False, 0
    projection, inverse = atlas.projection, atlas.inverse
    work = 0
    for cell, chart_inverse in (
        (first, [[Fraction(i == j) for j in range(3)] for i in range(3)]),
        (second, inverse),
    ):
        local = mapped_embedding_owner.projected_jacobian(cell, projection, budget=budget)
        if local is None:
            return False, work
        jacobian = tuple(
            tuple(
                expression_sum(
                    tuple(
                        expression_scale(local[i][k], chart_inverse[k][j], budget=budget)
                        for k in range(3)
                    ),
                    budget=budget,
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
            if mapped_embedding_owner.strict_spd(
                jacobian, "prism", 3, origin, piece, symmetric_parts, budget=budget
            ):
                continue
            if depth == limits.maximum_subdivision_depth:
                return False, work
            active.extend(
                (start, child, depth + 1)
                for start, child in embedding_cells_owner.children(cell, origin, piece)
            )
    return True, work
