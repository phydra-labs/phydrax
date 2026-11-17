#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Operation-local batched exact simplex face comparisons."""

from __future__ import annotations

from fractions import Fraction
from math import lcm

from ..discretization._coordinate_enclosure import Polynomial
from ._embedding_cells import EmbeddingCell
from ._restricted_embedding import (
    _integer_polynomials,
    _principal_lattice,
    _rational_simplex_map,
    TracePointValues,
)


type FaceRequest = tuple[EmbeddingCell, EmbeddingCell, tuple[int, ...]]


def batch_simplex_face_equal(
    requests: tuple[FaceRequest, ...], values: TracePointValues, /
) -> tuple[bool | None, ...]:
    """Compare eligible face maps with arbitrary-precision integer batches.

    ``None`` leaves non-simplex, non-rational and other unsupported maps to the
    original exact restriction route. Every denominator and numerator belongs
    to one cell; no arithmetic or identity depends on the other batch members.
    """
    results: list[bool | None] = [None] * len(requests)
    lattices: dict[int, tuple[tuple[int, ...], ...]] = {}
    maps: dict[int, tuple[int, tuple[Polynomial, ...]] | None] = {}
    programs: dict[
        tuple[int, int],
        tuple[
            int,
            tuple[tuple[tuple[tuple[int, ...], int], ...], ...],
            tuple[int, ...],
        ],
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
        sides = ((first, left, corners[0]),) if first is second else (
            (first, left, corners[0]), (second, right, corners[1])
        )
        if order not in lattices:
            lattices[order] = tuple(
                row for row in _principal_lattice(3, order) if order not in row
            )
        side_values = []
        for cell, mapping, indices in sides:
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
                max_powers = tuple(
                    max(
                        (exponent[axis] for polynomial in terms for exponent, _ in polynomial),
                        default=0,
                    )
                    for axis in range(cell.dimension)
                )
                programs[program_key] = denominator, terms, max_powers
            denominator, integer_terms, max_powers = programs[program_key]
            evaluated = []
            for weights in lattices[order]:
                numerators = tuple(
                    sum(
                        weights[corner] * integer_references[corner][axis]
                        for corner in range(3)
                    )
                    for axis in range(cell.dimension)
                )
                point = tuple(Fraction(value, point_scale) for value in numerators)
                key = (id(cell), point)
                found = values.get(key)
                if found is None:
                    powers = tuple(
                        tuple(numerator**power for power in range(max_power + 1))
                        for numerator, max_power in zip(numerators, max_powers, strict=True)
                    )
                    result = []
                    for polynomial in integer_terms:
                        total = 0
                        for exponent, coefficient in polynomial:
                            term = coefficient
                            for axis, power in enumerate(exponent):
                                if power:
                                    term *= powers[axis][power]
                            total += term
                        result.append(Fraction(total, denominator))
                    found = tuple(result)
                    values[key] = found
                evaluated.append(found)
            side_values.append(evaluated)
        results[index] = first is second or side_values[0] == side_values[1]
    return tuple(results)
