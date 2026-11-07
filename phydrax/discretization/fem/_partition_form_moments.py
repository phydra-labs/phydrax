"""Exact coarse-entity pieces induced by affine fine reference cells."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ...linalg._exact import solve_exact
from .._nested_reference import NestedCellReferenceMap
from .._nested_reference_types import (
    exact_reference_cell_vertices,
    ReferenceCellPartition,
    validate_reference_cell_partition,
)
from .._reference_cell import reference_cell_topology
from ._form_elements import _form_entity_chart, FormBasis


Point = tuple[Fraction, ...]
Simplex = tuple[Point, ...]


@dataclass(frozen=True, slots=True)
class PartitionedReferenceEntities:
    """One fine cell's disjoint contributions to coarse reference entities."""

    fine_cell_id: int
    coarse_cell_id: int
    fine_cell_kind: str
    coarse_cell_kind: str
    coarse_to_fine: NestedCellReferenceMap
    pieces: tuple[tuple[tuple[int, ...], tuple[Simplex, ...]], ...]


def _sub(a: Point, b: Point) -> Point:
    return tuple(x - y for x, y in zip(a, b, strict=True))


def _dot(a: Point, b: Point) -> Fraction:
    return sum((x * y for x, y in zip(a, b, strict=True)), Fraction(0))


def _cross(a: Point, b: Point) -> Point:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


_PLANAR_KINDS = ("triangle", "simplex:2")


def _affine(kind: str, vertices: tuple[Point, ...]) -> tuple[Point, tuple[Point, ...]]:
    reference = tuple(
        tuple(Fraction(float(x)) for x in v)
        for v in reference_cell_topology(kind).vertices
    )
    basis = (
        (0, 1, 2)
        if kind in _PLANAR_KINDS
        else (0, 1, 2, 3)
        if kind in ("tetrahedron", "simplex:3", "prism")
        else (0, 1, 3, 4)
    )
    dimension = len(basis) - 1
    coefficient = solve_exact(
        [[Fraction(1), *reference[i]] for i in basis], [list(vertices[i]) for i in basis]
    )
    origin = tuple(coefficient[0])
    jacobian = tuple(
        tuple(coefficient[j + 1][i] for j in range(dimension)) for i in range(dimension)
    )
    return origin, jacobian


def _planes(kind: str, vertices: tuple[Point, ...]) -> tuple[tuple[Point, Fraction], ...]:
    """Inward half-spaces ``normal · x <= offset`` bounding a convex cell image."""
    dimension = len(vertices[0])
    center = tuple(
        sum((p[i] for p in vertices), Fraction(0)) / len(vertices)
        for i in range(dimension)
    )
    planes = []
    topology = reference_cell_topology(kind)
    for face in topology.entities[dimension - 1]:
        if dimension == 2:
            a, b = vertices[face[0]], vertices[face[1]]
            normal = (a[1] - b[1], b[0] - a[0])
        else:
            a, b, c = (vertices[face[i]] for i in range(3))
            normal = _cross(_sub(b, a), _sub(c, a))
        offset = _dot(normal, a)
        if _dot(normal, center) > offset:
            normal = tuple(-v for v in normal)
            offset = -offset
        planes.append((normal, offset))
    return tuple(planes)


def _clip_polygon(
    polygon: tuple[Point, ...], constraints: tuple[tuple[Point, Fraction], ...]
) -> tuple[Point, ...]:
    for normal, offset in constraints:
        if not polygon:
            break
        output = []
        previous = polygon[-1]
        before = _dot(normal, previous) - offset
        for current in polygon:
            after = _dot(normal, current) - offset
            if (before < 0 and after > 0) or (before > 0 and after < 0):
                weight = before / (before - after)
                output.append(
                    tuple(
                        a + weight * (b - a)
                        for a, b in zip(previous, current, strict=True)
                    )
                )
            if after <= 0:
                output.append(current)
            previous, before = current, after
        cleaned = []
        for point in output:
            if not cleaned or point != cleaned[-1]:
                cleaned.append(point)
        if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
            cleaned.pop()
        polygon = tuple(cleaned)
    return polygon


def _inverse_map(
    fine_id: int,
    coarse_id: int,
    fine_kind: str,
    coarse_kind: str,
    origin: Point,
    jacobian: tuple[Point, ...],
) -> NestedCellReferenceMap:
    dimension = len(origin)
    inverse = solve_exact(
        jacobian,
        [[Fraction(int(i == j)) for j in range(dimension)] for i in range(dimension)],
    )
    offset = tuple(
        -sum((inverse[i][j] * origin[j] for j in range(dimension)), Fraction(0))
        for i in range(dimension)
    )
    zero = (0,) * dimension
    arguments = tuple(
        {
            zero: offset[i],
            **{
                tuple(int(k == j) for k in range(dimension)): inverse[i][j]
                for j in range(dimension)
                if inverse[i][j]
            },
        }
        for i in range(dimension)
    )
    coarse_vertices = reference_cell_topology(coarse_kind).vertices
    vertices = tuple(
        tuple(
            offset[i]
            + sum(
                (inverse[i][j] * Fraction(float(point[j])) for j in range(dimension)),
                Fraction(0),
            )
            for i in range(dimension)
        )
        for point in coarse_vertices
    )
    return NestedCellReferenceMap(
        coarse_id, fine_id, coarse_kind, fine_kind, vertices, arguments
    )


PartitionCell = tuple[int, str, str, Simplex]
"""One fine cell: ``(fine_id, fine_kind, coarse_kind, corners in coarse reference)``."""


def index_reference_cell_partition(
    partition: ReferenceCellPartition, /
) -> dict[int, tuple[PartitionCell, ...]]:
    """Validate a partition once and group its fine cells by coarse parent ID."""
    validate_reference_cell_partition(partition)
    grouped: dict[int, list[PartitionCell]] = {}
    for group in partition.maps:
        exact = exact_reference_cell_vertices(group)
        for fine, coarse, vertices in zip(
            np.asarray(group.target_cell_ids),
            np.asarray(group.source_cell_ids),
            exact,
            strict=True,
        ):
            grouped.setdefault(int(coarse), []).append(
                (int(fine), group.target_cell_kind, group.source_cell_kind, vertices)
            )
    return {
        coarse: tuple(sorted(cells, key=lambda item: item[0]))
        for coarse, cells in grouped.items()
    }


def prepare_partitioned_reference_entities(
    partition: ReferenceCellPartition, coarse_cell_id: int, basis: FormBasis, /
) -> tuple[PartitionedReferenceEntities, ...]:
    """Clip every coarse entity against its fine cells, assigning shared traces once."""
    cells = index_reference_cell_partition(partition).get(coarse_cell_id, ())
    return prepare_indexed_partitioned_reference_entities(cells, coarse_cell_id, basis)


def prepare_indexed_partitioned_reference_entities(
    cells: tuple[PartitionCell, ...], coarse_cell_id: int, basis: FormBasis, /
) -> tuple[PartitionedReferenceEntities, ...]:
    """Clip coarse entities against one parent's validated, ID-sorted fine cells.

    ``cells`` must come from :func:`index_reference_cell_partition`. Vertices
    are owned by their least fine owner, coarse edges (and 3D facets) are split
    into uniquely owned fine-trace pieces, and the coarse cell is the union of
    its fine cells.
    """
    dimension = basis.dimension
    if dimension not in (2, 3):
        raise ValueError(
            "Reference partitions require two- or three-dimensional form bases."
        )
    if not cells:
        raise ValueError("The partition does not contain the requested coarse cell.")
    coarse_kind = cells[0][2]
    if any(
        reference_cell_topology(kind).dimension != dimension
        or reference_cell_topology(parent).dimension != dimension
        for _, kind, parent, _ in cells
    ):
        raise ValueError("Reference partition cells and form basis dimensions differ.")
    if len(reference_cell_topology(coarse_kind).vertices) != len(
        basis.entity_vertices[0]
    ):
        raise ValueError("Coarse cell and form basis reference cells differ.")
    planes = {fine: _planes(kind, vertices) for fine, kind, _, vertices in cells}
    results: dict[int, dict[tuple[int, ...], list[Simplex]]] = {
        cell[0]: {} for cell in cells
    }
    for m, level in enumerate(basis.entity_vertices):
        for face in level:
            _, origin_float, jac_float = _form_entity_chart(dimension, basis.family, face)
            origin = tuple(Fraction(int(v)) for v in origin_float)
            jac = tuple(tuple(Fraction(int(v)) for v in row) for row in jac_float)
            if m == dimension:
                for fine, kind, _, vertices in cells:
                    a, matrix = _affine(kind, vertices)
                    if kind in _PLANAR_KINDS:
                        local = (((0, 0), (1, 0), (0, 1)),)
                    elif kind in ("tetrahedron", "simplex:3"):
                        local = (((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),)
                    elif kind == "prism":
                        from ._exact_form_moments import _prism_simplices

                        local = _prism_simplices()
                    elif kind == "hexahedron":
                        from ._exact_form_moments import _cube_simplices

                        local = _cube_simplices(3)
                    else:
                        apex = (Fraction(1, 2), Fraction(1, 2), Fraction(1))
                        local = (
                            ((0, 0, 0), (1, 0, 0), (1, 1, 0), apex),
                            ((0, 0, 0), (1, 1, 0), (0, 1, 0), apex),
                        )
                    simplices: list[Simplex] = [
                        tuple(
                            tuple(
                                a[i]
                                + sum(
                                    (matrix[i][j] * p[j] for j in range(dimension)),
                                    Fraction(0),
                                )
                                for i in range(dimension)
                            )
                            for p in simplex
                        )
                        for simplex in local
                    ]
                    results[fine][face] = simplices
                continue
            constraints = {}
            for fine, _, _, _ in cells:
                constraints[fine] = tuple(
                    (
                        tuple(
                            sum(
                                (normal[i] * jac[i][j] for i in range(dimension)),
                                Fraction(0),
                            )
                            for j in range(m)
                        ),
                        offset - _dot(normal, origin),
                    )
                    for normal, offset in planes[fine]
                )
            if m == 0:
                owners = [
                    fine
                    for fine, _, _, _ in cells
                    if all(offset >= 0 for _, offset in constraints[fine])
                ]
                if not owners:
                    raise ValueError("Partition does not cover a coarse vertex.")
                results[min(owners)][face] = [((),)]
            elif m == 1:
                intervals = {}
                cuts = {Fraction(0), Fraction(1)}
                for fine, _, _, _ in cells:
                    low, high = Fraction(0), Fraction(1)
                    for normal, offset in constraints[fine]:
                        factor = normal[0]
                        if factor > 0:
                            high = min(high, offset / factor)
                        elif factor < 0:
                            low = max(low, offset / factor)
                        elif offset < 0:
                            high = low - 1
                    if low < high:
                        intervals[fine] = (low, high)
                        cuts.update((low, high))
                points = sorted(cuts)
                for low, high in zip(points, points[1:]):
                    midpoint = (low + high) / 2
                    owners = [
                        fine for fine, (a, b) in intervals.items() if a <= midpoint <= b
                    ]
                    if not owners:
                        raise ValueError("Partition does not cover a coarse edge.")
                    results[min(owners)].setdefault(face, []).append(((low,), (high,)))
            else:
                initial = (
                    (
                        (Fraction(0), Fraction(0)),
                        (Fraction(1), Fraction(0)),
                        (Fraction(1), Fraction(1)),
                        (Fraction(0), Fraction(1)),
                    )
                    if len(face) == 4
                    else (
                        (Fraction(0), Fraction(0)),
                        (Fraction(1), Fraction(0)),
                        (Fraction(0), Fraction(1)),
                    )
                )
                area = Fraction(0)
                for fine, _, _, _ in cells:
                    polygon = _clip_polygon(initial, constraints[fine])
                    pieces: list[Simplex] = []
                    for i in range(1, len(polygon) - 1):
                        a, b, c = polygon[0], polygon[i], polygon[i + 1]
                        if (b[0] - a[0]) * (c[1] - a[1]) != (b[1] - a[1]) * (c[0] - a[0]):
                            pieces.append((a, b, c))
                            area += (
                                abs(
                                    (b[0] - a[0]) * (c[1] - a[1])
                                    - (b[1] - a[1]) * (c[0] - a[0])
                                )
                                / 2
                            )
                    if pieces:
                        results[fine][face] = pieces
                if area != (Fraction(1) if len(face) == 4 else Fraction(1, 2)):
                    raise ValueError(
                        "Partition does not cover a coarse facet exactly once."
                    )
    return tuple(
        PartitionedReferenceEntities(
            fine,
            coarse_cell_id,
            kind,
            coarse_kind,
            _inverse_map(
                fine, coarse_cell_id, kind, coarse_kind, *_affine(kind, vertices)
            ),
            tuple((face, tuple(pieces)) for face, pieces in results[fine].items()),
        )
        for fine, kind, _, vertices in cells
    )
