#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Operation-local batched exact simplex face comparisons."""

from __future__ import annotations

from fractions import Fraction
from math import lcm

from ..discretization._coordinate_enclosure import Polynomial
from ._embedding_cells import EmbeddingCell
from ._exact_polyhedral_geometry import Point3
from ._restricted_embedding import (
    _integer_polynomials,
    _principal_lattice,
    _rational_simplex_map,
    TracePointValues,
)


type FaceRequest = tuple[EmbeddingCell, EmbeddingCell, tuple[int, ...]]
type SimplexMaps = dict[int, tuple[int, tuple[Polynomial, ...]] | None]
type IntegerPrograms = dict[
    tuple[int, int],
    tuple[
        int,
        tuple[tuple[tuple[tuple[int, ...], int], ...], ...],
        tuple[int, ...],
    ],
]

type QuadraticBank = tuple[int, tuple[tuple[int, ...], ...]]
type QuadraticBanks = dict[int, QuadraticBank | None]

_TETRAHEDRON_REFERENCES = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
_TETRAHEDRON_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_TETRAHEDRON_EDGE_INDEX = {edge: index for index, edge in enumerate(_TETRAHEDRON_EDGES)}


def _edge_index(first: int, second: int, /) -> int:
    edge = (first, second) if first < second else (second, first)
    return _TETRAHEDRON_EDGE_INDEX[edge]


def quadratic_tetrahedron_bank(
    cell: EmbeddingCell,
    mapping: tuple[int, tuple[Polynomial, ...]] | None,
    banks: QuadraticBanks,
    /,
) -> QuadraticBank | None:
    """Integer numerators at all six edge midpoints over one denominator."""
    key = id(cell)
    if key in banks:
        return banks[key]
    if (
        mapping is None
        or mapping[0] > 2
        or len(mapping[1]) != 3
        or cell.dimension != 3
        or len(cell.vertex_ids) != 4
        or len(set(cell.vertex_ids)) != 4
        or cell.reference_vertices != _TETRAHEDRON_REFERENCES
        or any(
            type(coordinate) is not Fraction
            for point in cell.reference_vertices
            for coordinate in point
        )
        or any(
            len(exponent) != 3
            or any(type(power) is not int or power < 0 for power in exponent)
            for polynomial in mapping[1]
            for exponent in polynomial
        )
    ):
        banks[key] = None
        return None
    denominator, polynomials = _integer_polynomials(mapping[1], 2)
    axes: list[list[int]] = []
    for polynomial in polynomials:
        coefficients = [0] * 8
        for exponent, coefficient in polynomial:
            mask = sum(1 << axis for axis, power in enumerate(exponent) if power)
            coefficients[mask] += coefficient
        axes.append(coefficients)
    rows = tuple(
        tuple(
            coefficients[0]
            + (coefficients[1 << (first - 1)] if first else 0)
            + coefficients[1 << (second - 1)]
            + (coefficients[(1 << (first - 1)) | (1 << (second - 1))] if first else 0)
            for coefficients in axes
        )
        for first, second in _TETRAHEDRON_EDGES
    )
    bank = denominator, rows
    banks[key] = bank
    return bank


def quadratic_face_class(
    cell: EmbeddingCell,
    vertex_ids: tuple[int, ...],
    points: tuple[Point3, ...],
    maps: SimplexMaps,
    banks: QuadraticBanks,
    /,
) -> str | None:
    """Classify an exact quadratic tetrahedron face using integer determinants."""
    if (
        cell.affine
        or len(vertex_ids) != 3
        or len(set(vertex_ids)) != 3
        or len(points) != 3
        or len(cell.corner_images) != len(cell.vertex_ids)
        or any(len(point) != 3 for point in points)
    ):
        return None
    if not set(vertex_ids) <= set(cell.vertex_ids):
        return None
    corners = tuple(cell.vertex_ids.index(identifier) for identifier in vertex_ids)
    if any(
        cell.corner_images[index] != point
        for index, point in zip(corners, points, strict=True)
    ):
        return None
    rational_points: list[tuple[Fraction, ...]] = []
    for point in points:
        rational_point: list[Fraction] = []
        for value in point:
            if not isinstance(value, Fraction) or type(value) is not Fraction:
                return None
            rational_point.append(value)
        rational_points.append(tuple(rational_point))
    if id(cell) not in maps:
        maps[id(cell)] = _rational_simplex_map(cell)
    mapping = maps[id(cell)]
    if mapping is None or mapping[0] != 2:
        return None
    bank = quadratic_tetrahedron_bank(cell, mapping, banks)
    if bank is None:
        return None
    denominator, values = bank
    scale = lcm(*(value.denominator for point in rational_points for value in point))
    scaled = tuple(
        tuple(value.numerator * (scale // value.denominator) for value in point)
        for point in rational_points
    )
    edges = tuple(
        values[_edge_index(corners[first], corners[second])]
        for first, second in ((0, 1), (0, 2), (1, 2))
    )
    affine = all(
        2 * edge[axis] * scale
        == (scaled[first][axis] + scaled[second][axis]) * denominator
        for edge, (first, second) in zip(edges, ((0, 1), (0, 2), (1, 2)), strict=True)
        for axis in range(3)
    )
    left = tuple(scaled[1][axis] - scaled[0][axis] for axis in range(3))
    right = tuple(scaled[2][axis] - scaled[0][axis] for axis in range(3))
    normal = (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )
    if affine:
        return "affine" if any(normal) else "degenerate"
    if not any(normal):
        return "undecided"
    if any(
        sum(
            normal[axis] * (edge[axis] * scale - scaled[0][axis] * denominator)
            for axis in range(3)
        )
        for edge in edges
    ):
        return "nonplanar"
    return "undecided"


def batch_simplex_face_equal(
    requests: tuple[FaceRequest, ...],
    values: TracePointValues,
    maps: SimplexMaps,
    programs: IntegerPrograms,
    banks: QuadraticBanks,
    /,
) -> tuple[bool | None, ...]:
    """Compare eligible face maps with arbitrary-precision integer batches.

    ``None`` leaves non-simplex, non-rational and other unsupported maps to the
    original exact restriction route. Standard quadratic tetrahedra use six
    integer midpoint values per cell; other rational simplex maps use the
    general principal-lattice route. The caller keeps all cells alive for the
    operation-local caches.
    """
    results: list[bool | None] = [None] * len(requests)
    lattices: dict[int, tuple[tuple[int, ...], ...]] = {}

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
        if order == 2 and len(set(identifiers)) == 3:
            first_bank = quadratic_tetrahedron_bank(first, left, banks)
            second_bank = (
                first_bank
                if first is second
                else quadratic_tetrahedron_bank(second, right, banks)
            )
            if first_bank is not None and second_bank is not None:
                if first is second:
                    results[index] = True
                    continue
                first_denominator, first_values = first_bank
                second_denominator, second_values = second_bank
                results[index] = all(
                    all(
                        left_value * second_denominator == right_value * first_denominator
                        for left_value, right_value in zip(
                            first_values[_edge_index(corners[0][a], corners[0][b])],
                            second_values[_edge_index(corners[1][a], corners[1][b])],
                            strict=True,
                        )
                    )
                    for a, b in ((0, 1), (0, 2), (1, 2))
                )
                continue
        # Bound object-array workspace; larger lattices retain the scalar route.
        if order > 20:
            continue
        sides = (
            ((first, left, corners[0]),)
            if first is second
            else ((first, left, corners[0]), (second, right, corners[1]))
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
                        (
                            exponent[axis]
                            for polynomial in terms
                            for exponent, _ in polynomial
                        ),
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
                        for numerator, max_power in zip(
                            numerators, max_powers, strict=True
                        )
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
