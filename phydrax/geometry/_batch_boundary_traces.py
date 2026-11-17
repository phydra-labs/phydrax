#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Operation-local batched exact simplex face comparisons."""

from __future__ import annotations

from fractions import Fraction
from math import lcm

import numpy as np

from ..discretization._coordinate_enclosure import Polynomial
from ._embedding_cells import EmbeddingCell
from ._restricted_embedding import (
    _integer_polynomials,
    _principal_lattice,
    _rational_simplex_map,
    TracePointValues,
)


type FaceRequest = tuple[EmbeddingCell, EmbeddingCell, tuple[int, ...]]
type _Job = tuple[
    int,
    int,
    EmbeddingCell,
    int,
    tuple[tuple[int, ...], ...],
    int,
    tuple[dict[tuple[int, ...], int], ...],
]


def batch_simplex_face_equal(
    requests: tuple[FaceRequest, ...], values: TracePointValues, /
) -> tuple[bool | None, ...]:
    """Compare eligible face maps with arbitrary-precision integer batches.

    ``None`` leaves non-simplex, non-rational and other unsupported maps to the
    original exact restriction route. Every denominator and numerator belongs
    to one cell; no arithmetic or identity depends on the other batch members.
    """
    results: list[bool | None] = [None] * len(requests)
    groups: dict[tuple[int, int, int], list[_Job]] = {}
    maps: dict[int, tuple[int, tuple[Polynomial, ...]] | None] = {}
    programs: dict[
        tuple[int, int], tuple[int, tuple[dict[tuple[int, ...], int], ...]]
    ] = {}

    def cell_map(cell: EmbeddingCell, /) -> tuple[int, tuple[Polynomial, ...]] | None:
        key = id(cell)
        if key not in maps:
            maps[key] = _rational_simplex_map(cell)
        return maps[key]

    for index, (first, second, identifiers) in enumerate(requests):
        if len(identifiers) != 3 or first.dimension != 3 or second.dimension != 3:
            continue
        if first.affine and second.affine:
            continue
        left = cell_map(first)
        right = left if first is second else cell_map(second)
        if left is None or right is None or len(left[1]) != len(right[1]):
            continue
        if not set(identifiers) <= set(first.vertex_ids) or not set(identifiers) <= set(
            second.vertex_ids
        ):
            continue
        corners = tuple(
            tuple(cell.vertex_ids.index(identifier) for identifier in identifiers)
            for cell in (first, second)
        )
        if any(
            first.corner_images[a] != second.corner_images[b]
            for a, b in zip(*corners, strict=True)
        ):
            results[index] = False
            continue
        order = max(left[0], right[0], 1)
        if order == 1:
            results[index] = True
            continue
        # Bound object-array workspace; larger lattices retain the scalar route.
        if order > 20:
            continue
        sides = (
            ((2, first, left, corners[0]),)
            if first is second
            else ((0, first, left, corners[0]), (1, second, right, corners[1]))
        )
        for side, cell, mapping, indices in sides:
            references = tuple(cell.reference_vertices[position] for position in indices)
            reference_scale = lcm(
                *(coordinate.denominator for point in references for coordinate in point)
            )
            integer_references = tuple(
                tuple(
                    value.numerator * (reference_scale // value.denominator)
                    for value in point
                )
                for point in references
            )
            point_scale = order * reference_scale
            program_key = (id(cell), point_scale)
            if program_key not in programs:
                denominator, terms = _integer_polynomials(mapping[1], point_scale)
                programs[program_key] = denominator, tuple(dict(row) for row in terms)
            denominator, integer_terms = programs[program_key]
            jobs = groups.setdefault((order, cell.dimension, len(mapping[1])), [])
            jobs.append(
                (
                    index,
                    side,
                    cell,
                    point_scale,
                    integer_references,
                    denominator,
                    integer_terms,
                )
            )
    for (order, dimension, components), jobs in groups.items():
        weights = tuple(row for row in _principal_lattice(3, order) if order not in row)
        references = np.asarray([job[4] for job in jobs], dtype=np.object_)
        numerators = np.matmul(
            np.asarray(weights, dtype=np.object_)[None, :, :], references
        )
        totals = np.zeros((len(jobs), len(weights), components), dtype=np.object_)
        exponents = sorted(
            {exponent for job in jobs for polynomial in job[6] for exponent in polynomial}
        )
        for exponent in exponents:
            coefficients = np.asarray(
                [[polynomial.get(exponent, 0) for polynomial in job[6]] for job in jobs],
                dtype=np.object_,
            )
            monomial = np.ones((len(jobs), len(weights)), dtype=np.object_)
            for axis in range(dimension):
                if exponent[axis]:
                    monomial *= numerators[:, :, axis] ** exponent[axis]
            totals += monomial[:, :, None] * coefficients[:, None, :]
        pending: dict[int, tuple[np.ndarray, int]] = {}
        for row, (index, side, cell, point_scale, _, denominator, _) in enumerate(jobs):
            for point_index in range(len(weights)):
                point = tuple(
                    Fraction(value, point_scale) for value in numerators[row, point_index]
                )
                key = (id(cell), point)
                if key not in values:
                    values[key] = tuple(
                        Fraction(value, denominator) for value in totals[row, point_index]
                    )
            if side == 2:
                results[index] = True
            elif side == 0:
                pending[index] = (totals[row], denominator)
            else:
                left_values, left_denominator = pending.pop(index)
                results[index] = bool(
                    np.all(left_values * denominator == totals[row] * left_denominator)
                )
    return tuple(results)
