#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Closed affine contacts with field-arithmetic simplex and convex-polytope proofs.

Mesh certificate checks
-----------------------
[
{"checks":["mapped_intersection_piece_budget"],"owner":"phydrax.geometry._affine_mapped_contact.affine_simplex_contact","premises":[{"tag":"X","statement":"The next independent bounding-plane intersection would exceed maximum_work."}],"conclusion":"The affine-simplex intersection is unresolved with its completed plane-enumeration work preserved.","citation":{"kind":"definition","reference":"Mapped embedding proof and resource contract","locator":"phydrax.geometry._affine_mapped_contact.affine_simplex_contact","url":""},"failure_statuses":["unresolved"]},
{"checks":["mapped_intersection_piece_budget"],"owner":"phydrax.geometry._affine_mapped_contact.affine_polytope_contact","premises":[{"tag":"X","statement":"The next exact separating-axis test would exceed maximum_work."}],"conclusion":"The affine-polytope contact is unresolved with its completed axis-test work preserved.","citation":{"kind":"definition","reference":"Mapped embedding proof and resource contract","locator":"phydrax.geometry._affine_mapped_contact.affine_polytope_contact","url":""},"failure_statuses":["unresolved"]},
{"checks":["mapped_cell_contact"],"owner":"phydrax.geometry._affine_mapped_contact.contact_decision","premises":[{"tag":"X","statement":"The complete source actions are affine, shared identities are authenticated, and any additional allowed simplex support is one proper convex parent trace."},{"tag":"N","statement":"Intersection vertices or convex separating axes are evaluated in the exact ordered rational or algebraic coefficient field."}],"conclusion":"A separated affine contact decision admits only the authenticated shared convex trace.","citation":{"kind":"mathematical","reference":"Rockafellar, Convex Analysis","locator":"Section 11, separation theorems","url":"https://doi.org/10.1515/9781400873173"},"failure_statuses":[]}
]
End mesh certificate checks
"""

from __future__ import annotations

from contextlib import nullcontext
from fractions import Fraction
from itertools import combinations, product
from typing import assert_never, final, Literal

import equinox as eqx
import numpy as np

import phydrax.linalg._small_batched as small_batched_owner
from phydrax.linalg._exact import solve_exact

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._coordinate_enclosure import (
    Expression,
    expression_evaluate,
    RationalPolynomial,
)
from ..discretization._exact_algebraic_geometry import (
    RadicalElement,
    reserve_algebraic_work,
)
from ..discretization._reference_cell import reference_cell_topology
from ..typing import checked, parse
from ._embedding_cells import Cell
from ._embedding_state import ContactDecision, MeshCertificateLimits
from ._rational import rational_cross
from ._tangent_cone_contact import tangent_cone_contact


def _cross(first: np.ndarray, second: np.ndarray, /) -> np.ndarray:
    return np.asarray(
        (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        ),
        dtype=object,
    )


def interiors_overlap(
    first: np.ndarray,
    second: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> bool:
    """Apply the separating-axis theorem to field-valued triangles/tetrahedra."""
    with budget.temporary_scope() if budget is not None else nullcontext():
        if budget is not None:
            reserve_algebraic_work(
                tuple(first.flat) + tuple(second.flat),
                4096 if first.shape[1] == 3 else 256,
                1024 if first.shape[1] == 3 else 128,
                4,
                budget=budget,
            )

        dimension = first.shape[1]
        if dimension == 2:
            edges = [
                corners[b] - corners[a]
                for corners in (first, second)
                for a, b in ((0, 1), (1, 2), (2, 0))
            ]
            axes = [np.asarray((-edge[1], edge[0]), dtype=object) for edge in edges]
        else:
            pairs = tuple(combinations(range(4), 2))
            first_edges = [first[b] - first[a] for a, b in pairs]
            second_edges = [second[b] - second[a] for a, b in pairs]
            axes = [
                _cross(corners[b] - corners[a], corners[c] - corners[a])
                for corners in (first, second)
                for a, b, c in combinations(range(4), 3)
            ]
            axes.extend(_cross(a, b) for a in first_edges for b in second_edges)
        for axis in axes:
            if not np.any(axis):
                continue
            left, right = first @ axis, second @ axis
            if max(left) <= min(right) or max(right) <= min(left):
                return False
        return True


@final
class PreparedTriangleSAT(StrictModule, NonTrainableState):
    """Retain request-local triangle axes and their projection bounds."""

    corners: np.ndarray
    axes: tuple[np.ndarray, ...]
    bounds: tuple[tuple[Fraction | RadicalElement, Fraction | RadicalElement], ...]


def prepare_triangle_sat(
    corners: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    storage_precharged: bool = False,
) -> PreparedTriangleSAT:
    """Prepare triangle actions; prepaid storage borrows the caller's SAT workspace."""
    reserve_algebraic_work(
        tuple(corners.flat), 256, 0 if storage_precharged else 128, 4, budget=budget
    )
    axes = tuple(
        np.asarray((-edge[1], edge[0]), dtype=object)
        for a, b in ((0, 1), (1, 2), (2, 0))
        for edge in (corners[b] - corners[a],)
    )
    bounds = tuple(
        (min(projections), max(projections))
        for axis in axes
        for projections in (corners @ axis,)
    )
    return PreparedTriangleSAT(corners, axes, bounds)


def prepared_triangle_interiors_overlap(
    first: PreparedTriangleSAT,
    second: PreparedTriangleSAT,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    storage_precharged: bool = False,
) -> bool:
    """Apply every cross projection in the unprepared predicate's decision order."""
    with budget.temporary_scope():
        reserve_algebraic_work(
            tuple(first.corners.flat) + tuple(second.corners.flat),
            256,
            0 if storage_precharged else 128,
            4,
            budget=budget,
        )
        for owner, other in ((first, second), (second, first)):
            for axis, (lower, upper) in zip(owner.axes, owner.bounds, strict=True):
                if not np.any(axis):
                    continue
                projections = other.corners @ axis
                # Preserve the first/second comparison order even for second axes.
                if owner is first:
                    if upper <= min(projections) or max(projections) <= lower:
                        return False
                elif max(projections) <= lower or upper <= min(projections):
                    return False
        return True


type ContactMethod = Literal["affine_simplex", "affine_polytope", "tangent_cone"]


type Point = tuple[Fraction | RadicalElement, ...]
type Plane = tuple[Point, Fraction | RadicalElement]

# Bounds retain Python Fraction/PyLong owners and the rational elimination workspace.
_FRACTION_OWNER_BYTES = 128
_PYLONG_LIMB_BITS = 30
_POLYTOPE_PREPARATION_WORK = 2048
_POLYTOPE_PREPARATION_BYTES = 32768
_POLYTOPE_AXIS_WORK = 256


@final
class _AffineImages(StrictModule, NonTrainableState):
    first: tuple[Point, ...] = eqx.field(static=True)
    second: tuple[Point, ...] = eqx.field(static=True)


@final
class _Intersection(StrictModule, NonTrainableState):
    points: set[Point] = eqx.field(static=True)
    work: int = eqx.field(static=True)
    exhausted: bool = eqx.field(static=True)


def affine_expressions(values: tuple[Expression, ...], /) -> bool:
    """Return whether every source coordinate is a polynomial of total degree one."""
    return not any(
        isinstance(value, RationalPolynomial) or any(sum(index) > 1 for index in value)
        for value in values
    )


def _images(
    first: Cell, second: Cell, *, budget: CoordinateEnclosureBudget
) -> _AffineImages:
    values = tuple(
        tuple(
            tuple(
                expression_evaluate(value, point, budget=budget)
                for value in cell.coordinates
            )
            for point in cell.reference_vertices
        )
        for cell in (first, second)
    )
    return _AffineImages(values[0], values[1])


@checked
def affine_simplex_interiors_overlap(
    first: Cell, second: Cell, /, *, budget: CoordinateEnclosureBudget
) -> bool | None:
    """Decide open-interior overlap of full-dimensional affine simplices."""
    if (
        first.kind not in ("triangle", "tetrahedron")
        or second.kind != first.kind
        or len(first.coordinates) != first.dimension
        or len(second.coordinates) != first.dimension
        or not affine_expressions(first.coordinates)
        or not affine_expressions(second.coordinates)
    ):
        return None
    budget.reserve(1, _POLYTOPE_PREPARATION_BYTES)
    budget.reserve_geometry_queries(1)
    images = _images(first, second, budget=budget)
    left = np.asarray(images.first, dtype=object)
    right = np.asarray(images.second, dtype=object)
    if np.any(np.max(left, axis=0) <= np.min(right, axis=0)) or np.any(
        np.max(right, axis=0) <= np.min(left, axis=0)
    ):
        return False
    return interiors_overlap(
        left,
        right,
        budget=budget
        if any(
            isinstance(value, RadicalElement)
            for row in (*images.first, *images.second)
            for value in row
        )
        else None,
    )


def _simplex_planes(points: tuple[Point, ...], dimension: int, /) -> tuple[Plane, ...]:
    """Return inward barycentric half-spaces in reference-vertex order."""
    matrix = [
        [points[j + 1][i] - points[0][i] for j in range(dimension)]
        for i in range(dimension)
    ]
    inverse = solve_exact(
        matrix,
        [[Fraction(i == j) for j in range(dimension)] for i in range(dimension)],
    )
    rows = tuple(
        (
            tuple(row),
            -sum((a * x for a, x in zip(row, points[0], strict=True)), Fraction(0)),
        )
        for row in inverse
    )
    normal = tuple(
        -sum((row[axis] for row in inverse), Fraction(0)) for axis in range(dimension)
    )
    offset = Fraction(1) - sum((row[1] for row in rows), Fraction(0))
    return ((normal, offset), *rows)


def _intersection_vertices(
    planes: tuple[Plane, ...],
    dimension: int,
    maximum_work: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _Intersection:
    """Enumerate feasible intersections of independent bounding hyperplanes."""
    candidates: set[Point] = set()
    work = 0
    algebraic = any(
        isinstance(value, RadicalElement)
        for normal, offset in planes
        for value in (*normal, offset)
    )
    bits = max(
        (
            value.coefficient_bits
            if isinstance(value, RadicalElement)
            else abs(value.numerator).bit_length() + value.denominator.bit_length()
            for normal, offset in planes
            for value in (*normal, offset)
        ),
        default=1,
    )
    storage_upper = (
        4
        * dimension**2
        * (
            _FRACTION_OWNER_BYTES
            + 8
            * ((12 * (dimension + 1) * bits + _PYLONG_LIMB_BITS - 1) // _PYLONG_LIMB_BITS)
        )
    )
    for selected in combinations(planes, dimension):
        if work >= maximum_work:
            return _Intersection(candidates, work, True)
        work += 1
        budget.reserve(
            6 * dimension**3 * (dimension + 1) + 4 * dimension * len(planes),
            storage_upper,
        )
        if algebraic:
            reserve_algebraic_work(
                tuple((*normal, offset) for normal, offset in planes),
                6 * dimension**3 * (dimension + 1) + 4 * dimension * len(planes),
                8 * dimension**2,
                2 * dimension,
                budget=budget,
            )
        matrix = [list(row[0]) for row in selected]
        match dimension:
            case 2:
                determinant = small_batched_owner.cofactor_determinant_2(
                    *tuple(np.asarray(row, dtype=object) for row in matrix)
                )
            case 3:
                determinant = small_batched_owner.cofactor_determinant_3(
                    *tuple(np.asarray(row, dtype=object) for row in matrix)
                )
            case _:
                raise ValueError(
                    f"affine_simplex_contact.dimension must be 2 or 3; got {dimension}"
                )
        if not determinant:
            continue
        point = tuple(
            row[0] for row in solve_exact(matrix, [[-row[1]] for row in selected])
        )
        if all(
            sum((a * x for a, x in zip(normal, point, strict=True)), offset) >= 0
            for normal, offset in planes
        ):
            candidates.add(point)
    return _Intersection(candidates, work, False)


def _supported_intersection(
    first: Cell,
    planes: tuple[Plane, ...],
    candidates: set[Point],
    traces: tuple[tuple[int, ...], ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Require every intersection vertex to belong to one convex shared trace."""
    # Accepting a union point-by-point could hide a volume overlap.
    for trace in traces:
        supported = set(trace)
        accepted = True
        for point in candidates:
            for identifier, (normal, offset) in zip(
                first.vertices, planes[: first.dimension + 1], strict=True
            ):
                if identifier in supported:
                    continue
                if any(
                    isinstance(value, RadicalElement)
                    for value in (*normal, offset, *point)
                ):
                    reserve_algebraic_work(
                        (*normal, offset, *point),
                        4 * first.dimension,
                        2 * first.dimension,
                        2,
                        budget=budget,
                    )
                budget.admit_work_bound(first.dimension)
                budget.reserve(first.dimension)
                value = sum((a * x for a, x in zip(normal, point, strict=True)), offset)
                if value:
                    accepted = False
                    break
            if not accepted:
                break
        if accepted:
            return True
    return False


@checked
def affine_simplex_contact(
    first: Cell,
    second: Cell,
    maximum_work: int,
    *,
    authenticated_first_traces: tuple[tuple[int, ...], ...] = (),
    budget: CoordinateEnclosureBudget,
) -> ContactDecision | None:
    """Classify closed affine simplices using their barycentric half-spaces.

    A separated decision permits only one shared convex trace. Additional
    first-cell traces must have been authenticated by the caller's partition
    proof; a union of traces cannot certify a volume intersection.
    """
    if (
        first.kind not in ("triangle", "tetrahedron")
        or second.kind != first.kind
        or len(first.coordinates) != first.dimension
        or len(second.coordinates) != first.dimension
    ):
        return None
    if not affine_expressions(first.coordinates) or not affine_expressions(
        second.coordinates
    ):
        return None
    dimension = first.dimension
    for trace in authenticated_first_traces:
        if (
            not trace
            or len(set(trace)) > dimension
            or not set(trace).issubset(first.vertices)
        ):
            raise ValueError(
                "affine_simplex_contact.authenticated_first_traces must contain proper first-simplex traces; got invalid trace"
            )
    images = _images(first, second, budget=budget)
    algebraic = any(
        isinstance(value, RadicalElement)
        for row in (*images.first, *images.second)
        for value in row
    )
    if algebraic and not authenticated_first_traces:
        separated = affine_polytope_contact(first, second, maximum_work, budget=budget)
        if separated is not None and separated.status == "separated":
            return separated
    if algebraic:
        reserve_algebraic_work(
            (*images.first, *images.second),
            32 * dimension**3,
            16 * dimension**2,
            2 * dimension,
            budget=budget,
        )
    planes = (
        *_simplex_planes(images.first, dimension),
        *_simplex_planes(images.second, dimension),
    )
    intersection = _intersection_vertices(planes, dimension, maximum_work, budget=budget)
    if intersection.exhausted:
        return ContactDecision(
            "unresolved", intersection.work, check="mapped_intersection_piece_budget"
        )
    shared = tuple(set(first.vertices).intersection(second.vertices))
    separated = _supported_intersection(
        first,
        planes,
        intersection.points,
        (shared, *authenticated_first_traces),
        budget=budget,
    )
    return ContactDecision("separated" if separated else "overlap", intersection.work)


@checked
def shared_affine_parent_traces(
    first: Cell, second: Cell, /
) -> tuple[tuple[int, ...], ...]:
    """Return shared reference entities and convex parent-facet subdivisions."""
    first_topology = reference_cell_topology(first.kind)
    second_topology = reference_cell_topology(second.kind)
    second_vertices = set(second.vertices)
    result: dict[tuple[int, ...], None] = {}
    for entities in first_topology.entities[:-1]:
        for entity in entities:
            trace = tuple(first.vertices[index] for index in entity)
            if set(trace).issubset(second_vertices):
                result[trace] = None
    first_facets = first_topology.entities[first.dimension - 1]
    second_facets = second_topology.entities[second.dimension - 1]
    for first_facet in first_facets:
        first_trace = tuple(first.vertices[index] for index in first_facet)
        first_set = set(first_trace)
        for second_facet in second_facets:
            second_trace = tuple(second.vertices[index] for index in second_facet)
            second_set = set(second_trace)
            if second_set.issubset(first_set):
                support = tuple(
                    identifier for identifier in first_trace if identifier in second_set
                )
                if support:
                    result[support] = None
    return tuple(result)


def _subtract(left: Point, right: Point, /) -> Point:
    return tuple(a - b for a, b in zip(left, right, strict=True))


def _normalized_axis(axis: Point, /) -> Point | None:
    pivot = next((value for value in axis if value), None)
    return None if pivot is None else tuple(value / pivot for value in axis)


def _polytope_axes(
    first: Cell, second: Cell, images: _AffineImages, /
) -> tuple[Point, ...] | None:
    """Prepare facet normals followed by edge cross-products in insertion order."""
    axes: dict[Point, None] = {}
    edge_vectors: list[tuple[Point, ...]] = []
    for cell, points in zip((first, second), (images.first, images.second), strict=True):
        topology = reference_cell_topology(cell.kind)
        edges = tuple(
            _subtract(points[edge[1]], points[edge[0]]) for edge in topology.entities[1]
        )
        if first.dimension == 2:
            for edge in edges:
                normal = (-edge[1], edge[0])
                if any(normal):
                    axes[normal] = None
            continue
        edge_vectors.append(edges)
        for facet in topology.entities[2]:
            origin = points[facet[0]]
            normal = next(
                (
                    value
                    for left, right in product(facet[1:], repeat=2)
                    if left < right
                    for value in (
                        rational_cross(
                            _subtract(points[left], origin),
                            _subtract(points[right], origin),
                        ),
                    )
                    if any(value)
                ),
                None,
            )
            if normal is None:
                return None
            normalized = _normalized_axis(normal)
            if normalized is not None:
                axes[normalized] = None
    if first.dimension == 2:
        return tuple(axes)
    for first_edge in edge_vectors[0]:
        for second_edge in edge_vectors[1]:
            normalized = _normalized_axis(rational_cross(first_edge, second_edge))
            if normalized is not None:
                axes[normalized] = None
    return tuple(axes)


def _axis_separates(
    axis: Point, images: _AffineImages, first: Cell, traces: tuple[set[int], ...], /
) -> bool:
    projections = tuple(
        tuple(
            sum(
                (weight * value for weight, value in zip(axis, point, strict=True)),
                Fraction(0),
            )
            for point in points
        )
        for points in (images.first, images.second)
    )
    first_min, first_max = min(projections[0]), max(projections[0])
    second_min, second_max = min(projections[1]), max(projections[1])
    if first_max < second_min or second_max < first_min:
        return True
    first_support: tuple[int, ...] = ()
    if first_max == second_min:
        first_support = tuple(
            first.vertices[index]
            for index, value in enumerate(projections[0])
            if value == first_max
        )
    elif second_max == first_min:
        first_support = tuple(
            first.vertices[index]
            for index, value in enumerate(projections[0])
            if value == first_min
        )
    return bool(first_support) and any(
        set(first_support).issubset(trace) for trace in traces
    )


@checked
def affine_polytope_contact(
    first: Cell, second: Cell, maximum_work: int, /, *, budget: CoordinateEnclosureBudget
) -> ContactDecision | None:
    """Classify affine simplices and prisms by field-arithmetic convex separating axes."""
    if (
        first.kind not in ("triangle", "tetrahedron", "prism")
        or second.kind not in ("triangle", "tetrahedron", "prism")
        or first.dimension not in (2, 3)
        or second.dimension != first.dimension
    ):
        return None
    if not affine_expressions(first.coordinates) or not affine_expressions(
        second.coordinates
    ):
        return None
    if first.dimension == 2 and not any(
        isinstance(coefficient, RadicalElement)
        for cell in (first, second)
        for expression in cell.coordinates
        if not isinstance(expression, RationalPolynomial)
        for coefficient in expression.values()
    ):
        return None
    if maximum_work <= 0:
        return ContactDecision("unresolved", 0, check="mapped_intersection_piece_budget")
    budget.reserve(_POLYTOPE_PREPARATION_WORK, _POLYTOPE_PREPARATION_BYTES)
    images = _images(first, second, budget=budget)
    algebraic = any(
        isinstance(value, RadicalElement)
        for row in (*images.first, *images.second)
        for value in row
    )
    if algebraic:
        reserve_algebraic_work(
            (*images.first, *images.second),
            128 * first.dimension,
            64 * first.dimension,
            2 * first.dimension,
            budget=budget,
        )
    axes = _polytope_axes(first, second, images)
    if axes is None:
        return None
    traces = tuple(set(trace) for trace in shared_affine_parent_traces(first, second))
    work = 0
    for axis in axes:
        if work >= maximum_work:
            return ContactDecision(
                "unresolved", work, check="mapped_intersection_piece_budget"
            )
        work += 1
        budget.reserve(_POLYTOPE_AXIS_WORK)
        if algebraic:
            reserve_algebraic_work(
                (*images.first, *images.second, axis),
                64 * first.dimension,
                32 * first.dimension,
                3,
                budget=budget,
            )
        if _axis_separates(axis, images, first, traces):
            return ContactDecision("separated", work)
    return None if first.dimension == 2 else ContactDecision("overlap", work)


@checked
def contact_decision(
    first: Cell,
    second: Cell,
    maximum_work: int,
    /,
    *,
    method: ContactMethod,
    budget: CoordinateEnclosureBudget,
    limits: MeshCertificateLimits | None = None,
    authenticated_first_traces: tuple[tuple[int, ...], ...] = (),
) -> ContactDecision | None:
    """Dispatch one rational contact proof with a shared status, check and work record.

    The method selects a mathematical proof, not a fallback policy. None means
    that the selected proof's geometric hypotheses do not apply. Tangent-cone
    subdivision requires explicit limits; every method requires an explicit
    coordinate budget. Callers retain proof order and accumulate record.work.
    """
    method = parse(method, ContactMethod, "contact_decision.method")
    match method:
        case "affine_simplex":
            return affine_simplex_contact(
                first,
                second,
                maximum_work,
                authenticated_first_traces=authenticated_first_traces,
                budget=budget,
            )
        case "affine_polytope":
            return affine_polytope_contact(first, second, maximum_work, budget=budget)
        case "tangent_cone":
            if limits is None:
                raise ValueError(
                    "contact_decision.limits must bound tangent-cone subdivision; got None"
                )
            return tangent_cone_contact(
                first, second, limits, maximum_work, budget=budget
            )
        case _:
            assert_never(method)
