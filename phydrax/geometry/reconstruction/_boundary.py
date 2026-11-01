#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Continuous boundary queries for reconstructed piecewise-linear surfaces."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import final, TYPE_CHECKING

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import finite_real_scalar
from .._source_boundary_records import (
    SourceBoundaryChartCover,
    SourceBoundaryDistance,
    SourceBoundarySamples,
)
from ..simplicial._regions import MeshRegion
from ._core import ReconstructedGeometrySource
from ._sheet import AuthoredTriangleSheet


if TYPE_CHECKING:
    from .._domain_boundary import MeshingDomainBoundarySource
    from .._meshing_domain import MeshingDomain, ValidatedMeshingDomain


_ExactPoint = tuple[Fraction, Fraction, Fraction]
_ExactTriangle = tuple[_ExactPoint, _ExactPoint, _ExactPoint]

_BVH_LEAF_SIZE = 8


@dataclass(frozen=True, slots=True)
class _TriangleBVHNode:
    lower: _ExactPoint
    upper: _ExactPoint
    left: int | None
    right: int | None
    triangle_indices: tuple[int, ...]


def _triangle_bvh(
    triangles: tuple[_ExactTriangle, ...], /
) -> tuple[tuple[_TriangleBVHNode, ...], int]:
    bounds = tuple(
        (
            tuple(min(point[axis] for point in triangle) for axis in range(3)),
            tuple(max(point[axis] for point in triangle) for axis in range(3)),
        )
        for triangle in triangles
    )
    nodes: list[_TriangleBVHNode] = []

    def visit(indices: list[int], /) -> int:
        lower: _ExactPoint = (
            min(bounds[index][0][0] for index in indices),
            min(bounds[index][0][1] for index in indices),
            min(bounds[index][0][2] for index in indices),
        )
        upper: _ExactPoint = (
            max(bounds[index][1][0] for index in indices),
            max(bounds[index][1][1] for index in indices),
            max(bounds[index][1][2] for index in indices),
        )
        if len(indices) <= _BVH_LEAF_SIZE:
            nodes.append(_TriangleBVHNode(lower, upper, None, None, tuple(indices)))
            return len(nodes) - 1
        axis = max(
            range(3),
            key=lambda dimension: upper[dimension] - lower[dimension],
        )
        ordered = sorted(
            indices,
            key=lambda index: (
                bounds[index][0][axis] + bounds[index][1][axis],
                index,
            ),
        )
        middle = len(ordered) // 2
        left = visit(ordered[:middle])
        right = visit(ordered[middle:])
        nodes.append(_TriangleBVHNode(lower, upper, left, right, ()))
        return len(nodes) - 1

    root = visit(list(range(len(triangles))))
    return tuple(nodes), root


def _box_distance_squared(
    point: _ExactPoint,
    lower: _ExactPoint,
    upper: _ExactPoint,
    /,
) -> Fraction:
    squared = Fraction(0)
    for coordinate, first, last in zip(point, lower, upper, strict=True):
        difference = (
            first - coordinate
            if coordinate < first
            else (coordinate - last if coordinate > last else Fraction(0))
        )
        squared += difference * difference
    return squared


def _nearest_triangle_distance_squared(
    point: _ExactPoint,
    triangles: tuple[_ExactTriangle, ...],
    nodes: tuple[_TriangleBVHNode, ...],
    root: int,
    /,
) -> Fraction:
    pending = [(_box_distance_squared(point, nodes[root].lower, nodes[root].upper), root)]
    nearest: Fraction | None = None
    while pending:
        bound, index = heapq.heappop(pending)
        if nearest is not None and bound >= nearest:
            break
        node = nodes[index]
        if node.triangle_indices:
            for triangle_index in node.triangle_indices:
                distance = _point_triangle_distance_squared(
                    point, triangles[triangle_index]
                )
                if nearest is None or distance < nearest:
                    nearest = distance
                    if nearest == 0:
                        return nearest
            continue
        for child in (node.left, node.right):
            if child is None:
                continue
            child_node = nodes[child]
            child_bound = _box_distance_squared(point, child_node.lower, child_node.upper)
            if nearest is None or child_bound < nearest:
                heapq.heappush(pending, (child_bound, child))
    if nearest is None:
        raise RuntimeError("The reconstructed triangle boundary BVH is empty.")
    return nearest


def _point(values: np.ndarray, /) -> _ExactPoint:
    return (
        Fraction.from_float(float(values[0])),
        Fraction.from_float(float(values[1])),
        Fraction.from_float(float(values[2])),
    )


def _subtract(first: _ExactPoint, second: _ExactPoint, /) -> _ExactPoint:
    return (
        first[0] - second[0],
        first[1] - second[1],
        first[2] - second[2],
    )


def _midpoint(first: _ExactPoint, second: _ExactPoint, /) -> _ExactPoint:
    return (
        (first[0] + second[0]) / 2,
        (first[1] + second[1]) / 2,
        (first[2] + second[2]) / 2,
    )


def _dot(first: _ExactPoint, second: _ExactPoint, /) -> Fraction:
    return sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))


def _cross(first: _ExactPoint, second: _ExactPoint, /) -> _ExactPoint:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _segment_distance_squared(
    point: _ExactPoint,
    first: _ExactPoint,
    last: _ExactPoint,
    /,
) -> Fraction:
    edge = _subtract(last, first)
    offset = _subtract(point, first)
    length_squared = _dot(edge, edge)
    parameter = _dot(offset, edge) / length_squared
    if parameter <= 0:
        closest = first
    elif parameter >= 1:
        closest = last
    else:
        closest = (
            first[0] + parameter * edge[0],
            first[1] + parameter * edge[1],
            first[2] + parameter * edge[2],
        )
    return _dot(_subtract(point, closest), _subtract(point, closest))


def _point_triangle_distance_squared(
    point: _ExactPoint, triangle: _ExactTriangle, /
) -> Fraction:
    first, second, third = triangle
    edge_first = _subtract(second, first)
    edge_second = _subtract(third, first)
    offset = _subtract(point, first)
    dot_first_first = _dot(edge_first, edge_first)
    dot_first_second = _dot(edge_first, edge_second)
    dot_second_second = _dot(edge_second, edge_second)
    dot_offset_first = _dot(offset, edge_first)
    dot_offset_second = _dot(offset, edge_second)
    denominator = dot_first_first * dot_second_second - dot_first_second**2
    first_weight = (
        dot_second_second * dot_offset_first - dot_first_second * dot_offset_second
    ) / denominator
    second_weight = (
        dot_first_first * dot_offset_second - dot_first_second * dot_offset_first
    ) / denominator
    if first_weight >= 0 and second_weight >= 0 and first_weight + second_weight <= 1:
        normal = _cross(edge_first, edge_second)
        signed = _dot(offset, normal)
        return signed**2 / _dot(normal, normal)
    return min(
        _segment_distance_squared(point, first, second),
        _segment_distance_squared(point, second, third),
        _segment_distance_squared(point, third, first),
    )


def _sqrt_fraction_bounds(value: Fraction, /) -> tuple[float, float]:
    """Outward binary64 enclosure of the exact square root of a rational."""
    if value < 0:
        raise ValueError("A squared distance cannot be negative.")
    if value == 0:
        return 0.0, 0.0
    maximum = float(np.finfo(np.float64).max)
    maximum_squared = Fraction.from_float(maximum) ** 2
    if value > maximum_squared:
        return maximum, math.inf
    numerator, denominator = value.numerator, value.denominator
    exponent = (numerator.bit_length() - denominator.bit_length()) // 2
    if exponent >= 0:
        scaled = value / (1 << (2 * exponent))
    else:
        scaled = value * (1 << (-2 * exponent))
    try:
        candidate = math.ldexp(math.sqrt(float(scaled)), exponent)
    except OverflowError:
        candidate = maximum
    if math.isinf(candidate):
        candidate = maximum
    squared = Fraction.from_float(candidate) ** 2
    if squared <= value:
        lower = candidate
        upper = candidate
        while Fraction.from_float(upper) ** 2 < value:
            lower = upper
            upper = math.nextafter(upper, math.inf)
    else:
        upper = candidate
        lower = math.nextafter(candidate, -math.inf)
        while lower >= 0.0 and Fraction.from_float(lower) ** 2 > value:
            upper = lower
            lower = math.nextafter(lower, -math.inf)
        lower = max(0.0, lower)
    return lower, upper


def _outward_float(value: Fraction, /) -> float:
    """Round an exact rational toward positive infinity."""
    try:
        result = float(value)
    except OverflowError:
        return math.inf
    if math.isfinite(result) and Fraction.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def reconstructed_triangle_arrays(
    source: ReconstructedGeometrySource | AuthoredTriangleSheet, /
) -> tuple[np.ndarray, np.ndarray]:
    """The vertex and face arrays a reconstructed triangle query is built over."""
    if isinstance(source, AuthoredTriangleSheet):
        return (
            np.asarray(source.repaired_vertices, dtype=np.float64),
            np.asarray(source.repaired_triangles, dtype=np.int64),
        )
    if isinstance(source, ReconstructedGeometrySource):
        region = source.source
        if not isinstance(region, MeshRegion):
            raise TypeError("Reconstructed triangle boundaries require a MeshRegion.")
        return (
            np.asarray(region.vertices, dtype=np.float64),
            np.asarray(region.faces, dtype=np.int64),
        )
    raise TypeError(
        "source must be a ReconstructedGeometrySource or AuthoredTriangleSheet."
    )


def domain_binds_source(
    domain: MeshingDomain,
    source: ReconstructedGeometrySource | AuthoredTriangleSheet,
    vertices: np.ndarray,
    /,
) -> bool:
    """Whether `domain` carries the source's identity, facet IDs and corners."""
    return (
        domain.source_id == source.source_id
        and domain.source_revision == source.source_revision
        and np.array_equal(
            np.asarray(domain.source_indices[2], dtype=np.int64),
            np.asarray(source.persistent_facet_ids, dtype=np.int64),
        )
        and np.array_equal(np.asarray(domain.corner_points), vertices)
    )


def _chart_cover_triangle_keys(
    values: np.ndarray, /
) -> tuple[tuple[tuple[Fraction, ...], ...], ...]:
    return tuple(
        sorted(
            tuple(
                sorted(
                    tuple(Fraction.from_float(float(value)) for value in point)
                    for point in triangle
                )
            )
            for triangle in values
        )
    )


def _prove_domain_query(
    source: ReconstructedGeometrySource | AuthoredTriangleSheet,
    vertices: np.ndarray,
    faces: np.ndarray,
    validated: ValidatedMeshingDomain | None,
    /,
) -> tuple[MeshingDomainBoundarySource, SourceBoundaryChartCover] | str:
    """The whole-domain boundary query of the source's persistent MeshingDomain.

    Proves that the domain carries the source's identity, facet IDs and corners
    and that its exact chart cover is the query's physical triangles (the
    query's own chart cover, ``vertices[faces]``, is complete and exact);
    returns the refusal otherwise. ``validated`` is the caller's validation of
    a domain; it spares the content fingerprint only for that very object.
    """
    from .._domain_boundary import MeshingDomainBoundarySource

    domain = source.meshing_domain
    if domain is None:
        return "Reconstructed source has no authoritative persistent MeshingDomain."
    if not domain_binds_source(domain, source, vertices):
        return "Reconstructed MeshingDomain changes persistent source vertex/facet IDs."
    try:
        query = MeshingDomainBoundarySource(
            domain, tuple(range(len(domain.patches))), validated=validated
        )
    except ValueError as error:
        return str(error)
    domain_cover = query.boundary_chart_cover()
    if (
        not domain_cover.complete
        or domain_cover.semantics != "certified"
        or np.any(domain_cover.deviation_bounds != 0.0)
    ):
        return "Reconstructed source charts lack complete exact physical coverage."
    if _chart_cover_triangle_keys(vertices[faces]) != _chart_cover_triangle_keys(
        domain_cover.simplices
    ):
        return (
            "Reconstructed query and persistent source domain cover different "
            "physical triangles."
        )
    return query, domain_cover


@final
class ReconstructedTriangleBoundarySource(StrictModule, NonTrainableState):
    """Exact continuous query over a reconstructed or authored triangle complex.

    Authored sheets are queried through their repaired complex, which is the
    bitwise-identical point set of the authored triangles (welding, duplicate
    removal, and orientation never move a point), so distances are exact
    distances to the authored sheet.

    Construction also prepares the whole-domain boundary query of the source's
    persistent MeshingDomain once (``domain_query``) and its exact chart cover
    (``domain_cover``), with the proof that this cover is the query's physical
    triangles, or records why it cannot (``domain_refusal``). Source-fidelity
    validation reuses them after checking that the source still reproduces
    these triangles and still retains a domain of that content, instead of
    recomputing the domain's chart cover.
    """

    source: ReconstructedGeometrySource | AuthoredTriangleSheet = eqx.field(repr=False)
    _vertices: np.ndarray = eqx.field(repr=False)
    _faces: np.ndarray = eqx.field(repr=False)
    _triangles: tuple[_ExactTriangle, ...] = eqx.field(repr=False)
    _nodes: tuple[_TriangleBVHNode, ...] = eqx.field(repr=False)
    _root_index: int = eqx.field(repr=False)
    domain_query: MeshingDomainBoundarySource | None = eqx.field(repr=False)
    domain_cover: SourceBoundaryChartCover | None = eqx.field(repr=False)
    domain_refusal: str | None = eqx.field(static=True, repr=False)

    def __init__(
        self,
        source: ReconstructedGeometrySource | AuthoredTriangleSheet,
        /,
        *,
        validated: ValidatedMeshingDomain | None = None,
    ) -> None:
        vertices, faces = reconstructed_triangle_arrays(source)
        triangles = tuple(
            (
                _point(vertices[int(face[0])]),
                _point(vertices[int(face[1])]),
                _point(vertices[int(face[2])]),
            )
            for face in faces.tolist()
        )
        if not triangles:
            raise ValueError("A reconstructed triangle boundary must be non-empty.")
        nodes, root = _triangle_bvh(triangles)
        vertices = np.array(vertices, dtype=np.float64, copy=True)
        faces = np.array(faces, dtype=np.int64, copy=True)
        vertices.setflags(write=False)
        faces.setflags(write=False)
        self.source = source
        self._vertices = vertices
        self._faces = faces
        self._triangles = triangles
        self._nodes = nodes
        self._root_index = root
        prepared = _prove_domain_query(source, vertices, faces, validated)
        if isinstance(prepared, str):
            self.domain_query = None
            self.domain_cover = None
            self.domain_refusal = prepared
        else:
            self.domain_query, self.domain_cover = prepared
            self.domain_refusal = None

    @property
    def source_id(self) -> str:
        return self.source.source_id

    @property
    def source_revision(self) -> str:
        return self.source.source_revision

    @property
    def ambient_dimension(self) -> int:
        return 3

    def boundary_distance(self, points: ArrayLike, /) -> SourceBoundaryDistance:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
            raise ValueError("Boundary query points must be a finite (N, 3) array.")
        lower = np.empty((values.shape[0],), dtype=np.float64)
        upper = np.empty_like(lower)
        for row in range(values.shape[0]):
            exact_point = _point(values[row])
            squared = _nearest_triangle_distance_squared(
                exact_point,
                self._triangles,
                self._nodes,
                self._root_index,
            )
            lower[row], upper[row] = _sqrt_fraction_bounds(squared)
        return SourceBoundaryDistance(lower, upper, "certified")

    def boundary_cover(self, maximum_radius: float, /) -> SourceBoundarySamples:
        """Return a complete exact-triangle cover at a physical radius."""
        if isinstance(maximum_radius, (bool, np.bool_)):
            raise TypeError("maximum_radius must be a positive finite physical distance.")
        radius_limit = finite_real_scalar(maximum_radius, "maximum_radius")
        if radius_limit <= 0.0:
            raise ValueError(
                "maximum_radius must be a positive finite physical distance."
            )
        radius_squared = Fraction.from_float(radius_limit) ** 2
        points: list[np.ndarray] = []
        covering_radii: list[float] = []
        point_errors: list[float] = []
        for original in self._triangles:
            pending = [original]
            while pending:
                triangle = pending.pop()
                exact_sample: _ExactPoint = (
                    sum((point[0] for point in triangle), Fraction(0)) / 3,
                    sum((point[1] for point in triangle), Fraction(0)) / 3,
                    sum((point[2] for point in triangle), Fraction(0)) / 3,
                )
                sample = np.asarray(
                    tuple(float(value) for value in exact_sample), dtype=np.float64
                )
                represented_sample = _point(sample)
                sample_error = _subtract(exact_sample, represented_sample)
                point_error_squared = _dot(sample_error, sample_error)
                if point_error_squared > radius_squared:
                    raise ArithmeticError(
                        "Reconstructed boundary cover cannot represent a source point "
                        "within the requested radius."
                    )
                point_error = _sqrt_fraction_bounds(point_error_squared)[1]
                covering_squared = max(
                    _dot(difference, difference)
                    for difference in (
                        _subtract(vertex, exact_sample) for vertex in triangle
                    )
                )
                covering_radius = _sqrt_fraction_bounds(covering_squared)[1]
                total_radius = math.inf
                if math.isfinite(covering_radius) and math.isfinite(point_error):
                    total_radius = _outward_float(
                        Fraction.from_float(covering_radius)
                        + Fraction.from_float(point_error)
                    )
                if total_radius <= radius_limit:
                    points.append(sample)
                    covering_radii.append(covering_radius)
                    point_errors.append(point_error)
                    continue

                first, second, third = triangle
                first_second = _midpoint(first, second)
                second_third = _midpoint(second, third)
                third_first = _midpoint(third, first)
                children = (
                    (first, first_second, third_first),
                    (first_second, second, second_third),
                    (third_first, second_third, third),
                    (first_second, second_third, third_first),
                )
                pending.extend(reversed(children))
        return SourceBoundarySamples(
            np.asarray(points, dtype=np.float64).reshape((-1, self.ambient_dimension)),
            np.asarray(covering_radii, dtype=np.float64),
            np.asarray(point_errors, dtype=np.float64),
            "certified",
            complete=True,
        )

    def boundary_chart_cover(self, /) -> SourceBoundaryChartCover:
        simplices = self._vertices[self._faces]
        return SourceBoundaryChartCover(
            simplices,
            np.zeros((simplices.shape[0],), dtype=np.float64),
            "certified",
            True,
            self.source_id,
            self.source_revision,
        )


__all__ = ["ReconstructedTriangleBoundarySource"]
