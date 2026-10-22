#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Exact physical intersections of independent affine boundary chart chains."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key
from typing import TypeVar

import equinox as eqx
import numpy as np

import phydrax.geometry._embedding_state as embedding_state_owner
import phydrax.geometry._source_boundary_records as source_boundary_records_owner
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateLiveStorage,
    reserve_polynomial,
)
from phydrax.geometry._planar_chain_embedding import (
    _compare_identity_coordinates,
    _group_prepared_triangle,
    _IdentityKey,
    _PlanarOperation,
    _PlanarTopology,
    _PreparedPlanarOperands,
    _PreparedTriangle,
    _prove_planar_embedding,
    _transport_planar_embedding,
    PlanarEmbedding,
    prepare_planar_topology,
)
from phydrax.geometry._planar_primitives import (
    _prepared_polygon_halfplanes,
    _prepared_supports_separated,
    binary64_fraction_bits,
    convex_hull,
    fraction_points,
    intersection_polygon,
    polygon_bounds,
    PolygonIntersectionPreparation,
    PolygonSidePreparation,
    prepared_plane_key,
    prepared_signed_measure,
    project,
)

from .. import _meshcore
from .._host_storage import declared_integer_storage_bytes, immutable_host_array_layout
from .._strict import StrictModule
from ..typing import checked
from ._embedding_state import MeshCertificateLimits


_T = TypeVar("_T")
_Point = tuple[Fraction, ...]
_Triangle = tuple[_Point, ...]
_PlaneKey = tuple[_Point, tuple[int, ...], int, int]


@dataclass(frozen=True, slots=True)
class _PolygonBounds:
    """One operation's exact box, bound to immutable output and scalar owners."""

    owner: CoordinateLiveStorage
    polygon: _Triangle
    coefficients: tuple[tuple[int, int], ...]
    bounds: tuple[_Point, _Point]


@dataclass(frozen=True, slots=True)
class _PointProfile:
    """Numeric bit bound tied to this preparation and exact coefficient owners."""

    owner: _Preparation
    points: _Triangle | _Point
    coefficients: tuple[tuple[Fraction, int, int], ...]
    bits: int


class _Preparation:
    """Rational coefficient preparation over the admitted proof batch."""

    def __init__(
        self,
        *,
        budget: CoordinateEnclosureBudget,
        cache_storage: CoordinateLiveStorage,
        side_storage: CoordinateLiveStorage,
        areas: Mapping[tuple[_Triangle, _Triangle], Fraction] | None = None,
        basis_storage: CoordinateLiveStorage | None = None,
    ) -> None:
        self.budget = budget
        self.clip_calls = 0
        self.plane_comparisons = 0
        self.plane_mismatches = 0
        self.zero_area_pairs = 0
        self.identity_rows = 0
        self.identity_key_bytes = 0
        self.cache_storage = cache_storage
        self.cache_bytes_upper = 2560
        cache_storage.set_bound(self.cache_bytes_upper)
        self.sides = PolygonSidePreparation(budget, side_storage)
        self.point_profiles: dict[int, _PointProfile] = {}
        self.projections: dict[tuple[int, tuple[int, ...]], _Triangle] = {}
        self.bounds_cache: dict[int, _PolygonBounds] = {}
        self.measure_cache: dict[int, tuple[_Triangle, Fraction]] = {}
        self.polygon_cache: dict[int, PolygonIntersectionPreparation] = {}
        self.vertex_cache: dict[tuple[float, ...], _Point] = {}
        # Request-proved areas are read only; this proof's new areas are
        # retained here and published to the request owner after success.
        self.known_areas: Mapping[tuple[_Triangle, _Triangle], Fraction] = (
            {} if areas is None else areas
        )
        self.earned_areas: dict[tuple[_Triangle, _Triangle], Fraction] = {}
        self.identity_mapping: tuple[int, ...] | None = None
        self.basis_storage = basis_storage
        self.basis_objects: dict[int, object] = {}
        if basis_storage is not None:
            basis_storage.set_bound(256)

    def area(self, first: _Triangle, second: _Triangle, /) -> Fraction:
        """Return one exact pair area, proving each distinct exact pair once."""
        self._point_profile(first)
        self._point_profile(second)
        self.budget.reserve(1, 96)
        key = first, second
        result = self.known_areas.get(key)
        if result is None:
            result = self.earned_areas.get(key)
        if result is None:
            # Intersection area is symmetric even when the oriented boundary
            # loops differ. Reuse the numeric result, not a tiling verdict:
            # each caller still performs its own exact accumulated comparison.
            self.budget.reserve(1, 96)
            reverse = second, first
            result = self.known_areas.get(reverse)
            if result is None:
                result = self.earned_areas.get(reverse)
        if result is None:
            result = _intersection(first, second, self)
            # The projected key rows are already retained preparations. The
            # entry adds a dictionary slot, the key pair and one Fraction.
            self.cache_bytes_upper += 1024 + 2 * declared_integer_storage_bytes(
                self.bits((result,))
            )
            self.cache_storage.set_bound(self.cache_bytes_upper, work=1)
            self.earned_areas[key] = result
        return result

    def stage(
        self,
        bound: int,
        terms: int,
        bits: int,
        operation: Callable[[], tuple[_T, int]],
        /,
    ) -> _T:
        self.budget.admit_work_bound(bound)
        reserve_polynomial(0, terms, 3, bits, budget=self.budget)
        value, actual = operation()
        self.budget.reserve(actual)
        return value

    def retain(self, values: tuple[object, ...], /) -> None:
        if self.basis_storage is None:
            self.budget.retain_basis(values)
            return
        with self.budget.temporary_scope():
            seen, size = self.budget._object_storage_profile(
                values, self.basis_objects, retained=False,
            )
            self.basis_storage.set_bound(self.basis_storage.bound + size)
            self.basis_objects.update(seen)

    def close(self) -> None:
        """Drop construction-only graphs, including point-profile owner cycles."""
        self.point_profiles.clear()
        self.projections.clear()
        self.bounds_cache.clear()
        self.measure_cache.clear()
        self.polygon_cache.clear()
        self.vertex_cache.clear()
        self.earned_areas.clear()
        self.basis_objects.clear()
        self.identity_mapping = None

    def _point_profile(self, values: tuple[object, ...], /) -> _PointProfile | None:
        prepared = self.point_profiles.get(id(values))
        if prepared is None:
            return None
        if prepared.owner is not self or prepared.points is not values:
            raise ValueError("AffineChartChain.bits must bind the current operation and point owner")
        if any(
            value.numerator is not numerator or value.denominator is not denominator
            for value, numerator, denominator in prepared.coefficients
        ):
            raise ValueError("AffineChartChain.bits must bind unchanged point owners; got mutated coefficients")
        return prepared

    def bits(self, values: tuple[object, ...], /) -> int:
        prepared = self._point_profile(values)
        if prepared is not None:
            self.budget.reserve(1)
            return prepared.bits
        pending = list(values)
        result = 1
        while pending:
            self.budget.reserve(1)
            value = pending.pop()
            prepared = self._point_profile(value) if isinstance(value, tuple) else None
            if prepared is not None:
                result = max(result, prepared.bits)
                continue
            match value:
                case Fraction():
                    result = max(
                        result,
                        abs(value.numerator).bit_length(),
                        value.denominator.bit_length(),
                    )
                case tuple():
                    pending.extend(value)
                case _:
                    continue
        return result

    def retain_cache_entry(self) -> None:
        # Graphs are retained separately; dictionary slots, identity keys and
        # resizing coexist with them until the proof's cache owner closes.
        self.cache_bytes_upper += 256
        self.cache_storage.set_bound(self.cache_bytes_upper, work=1)

    def prepare_points(self, points: _Triangle | _Point, /) -> None:
        bits = self.bits(points)
        pending: list[object] = [points]
        coefficients: list[tuple[Fraction, int, int]] = []
        while pending:
            value = pending.pop()
            if type(value) is tuple:
                pending.extend(value)
            elif type(value) is Fraction:
                coefficients.append((value, value.numerator, value.denominator))
            else:
                raise ValueError("AffineChartChain.bits must bind immutable exact point owners")
        record = _PointProfile(self, points, tuple(coefficients), bits)
        self.retain((points, bits, record.coefficients))
        self.retain_cache_entry()
        self.point_profiles[id(points)] = record

    def vertex(self, row: np.ndarray, point_bits: int, /) -> _Point:
        self.budget.reserve(row.size, 256 + 384 * row.size)
        key = tuple(float(value) for value in row)
        point = self.vertex_cache.get(key)
        if point is None:
            reserve_polynomial(
                row.size, row.size, row.size, point_bits, budget=self.budget
            )
            point = fraction_points((key,))[0]
            self.prepare_points(point)
            self.retain((key,))
            self.retain_cache_entry()
            self.vertex_cache[key] = point
        return point

    def projected(self, points: _Triangle, axes: tuple[int, ...], /) -> _Triangle:
        self.budget.reserve(1)
        key = (id(points), axes)
        result = self.projections.get(key)
        if result is None:
            result = project(points, axes, budget=self.budget)
            self.prepare_points(result)
            self.retain((key,))
            self.retain_cache_entry()
            self.projections[key] = result
        return result

    def bounds(self, points: _Triangle, /) -> tuple[_Point, _Point]:
        """Reuse numeric boxes only for this operation's unchanged exact owner.

        The first box still pays every original min/max comparison. Retained
        integer owners detect even deliberate mutation of a Fraction's backing
        fields; neither value-equal polygons nor another request inherit a box.
        """
        self.budget.reserve(1)
        known = self.bounds_cache.get(id(points))
        if known is not None:
            if known.owner is not self.cache_storage or known.polygon is not points:
                raise ValueError("AffineChartChain.bounds must bind the current operation and polygon owner")
            self.budget.reserve(len(known.coefficients))
            if any(
                value.numerator is not numerator or value.denominator is not denominator
                for value, (numerator, denominator) in zip(
                    (value for point in points for value in point),
                    known.coefficients,
                    strict=True,
                )
            ):
                raise ValueError("AffineChartChain.bounds must bind unchanged point owners; got mutated coefficients")
            return known.bounds
        self.budget.reserve(1 + len(points) + sum(map(len, points)))
        if type(points) is not tuple or any(
            type(point) is not tuple or any(type(value) is not Fraction for value in point)
            for point in points
        ):
            raise ValueError("AffineChartChain.bounds must bind immutable exact polygon and point owners")
        result = polygon_bounds(points, budget=self.budget)
        coefficients = tuple(
            (value.numerator, value.denominator) for point in points for value in point
        )
        record = _PolygonBounds(self.cache_storage, points, coefficients, result)
        self.retain((record, points, result, coefficients))
        self.retain_cache_entry()
        self.bounds_cache[id(points)] = record
        return result

    def signed_measure(self, points: _Triangle, /) -> Fraction:
        """Retain a numeric shoelace result, never an area-completeness verdict."""
        self.budget.reserve(1)
        known = self.measure_cache.get(id(points))
        if known is not None and known[0] is points:
            return known[1]
        bits = 8 * self.bits((points,)) + 64
        with self.budget.temporary_scope():
            measure = self.stage(
                13, 12, bits, lambda: (prepared_signed_measure(points), 13),
            )
        self.cache_bytes_upper += 768 + 2 * declared_integer_storage_bytes(bits)
        self.cache_storage.set_bound(self.cache_bytes_upper, work=1)
        self.measure_cache[id(points)] = points, measure
        return measure

    def polygon(self, points: _Triangle, /) -> PolygonIntersectionPreparation:
        self.budget.reserve(1)
        result = self.polygon_cache.get(id(points))
        if result is None:
            lower, upper = self.bounds(points)
            with self.budget.temporary_scope():
                measure = self.signed_measure(points)
                self.budget.reserve(1)
                orientation = 1 if measure > 0 else -1
                halfplanes, bits = _prepared_polygon_halfplanes(
                    points, orientation, budget=self.budget
                )
                result = PolygonIntersectionPreparation(
                    points, lower, upper, halfplanes, bits
                )
            self.retain(
                (
                    result,
                    points,
                    result.lower,
                    result.upper,
                    result.halfplanes,
                    result.bits,
                )
            )
            self.retain_cache_entry()
            self.polygon_cache[id(points)] = result
        return result


def _plane(points: _Triangle, preparation: _Preparation) -> _PlaneKey | None:
    def operation() -> tuple[_PlaneKey | None, int]:
        result = prepared_plane_key(points)
        # Two edge rows, their cross product, normalized coefficients and
        # offset have thirteen simultaneously live rational terms in 3D.
        return result, 15 if result is None else 25

    return preparation.stage(25, 13, 8 * preparation.bits(points) + 64, operation)


def _same_triangle(
    first: _Triangle, second: _Triangle, preparation: _Preparation,
) -> bool:
    """Prove exact cyclic corner equality, allowing reversed boundary orientation."""
    if len(first) != 3 or len(second) != 3:
        return False
    for slot, point in enumerate(second):
        preparation.budget.reserve(len(first[0]))
        if first[0] != point:
            continue
        for direction in (1, -1):
            preparation.budget.reserve(len(first[1]))
            if first[1] != second[(slot + direction) % 3]:
                continue
            preparation.budget.reserve(len(first[2]))
            if first[2] == second[(slot + 2 * direction) % 3]:
                return True
    return False


def _intersection(
    first: _Triangle, second: _Triangle, preparation: _Preparation
) -> Fraction:
    """Prove zero-area contacts before exact physical positive-area clipping."""
    first_lower, first_upper = preparation.bounds(first)
    second_lower, second_upper = preparation.bounds(second)
    preparation.budget.reserve(2 * len(first_lower))
    if any(
        first_upper[axis] <= second_lower[axis]
        or second_upper[axis] <= first_lower[axis]
        for axis in range(len(first_lower))
    ):
        return Fraction(0)
    if _same_triangle(first, second, preparation):
        # This is equality of the actual exact operands, not an atlas ID or
        # coverage verdict. Intersecting this triangle with itself needs its
        # rank and measure, but no reference solve, clipping or reconstruction.
        measure = preparation.signed_measure(first)
        preparation.budget.reserve(1)
        area = abs(measure)
        if area == 0:
            raise ValueError(
                "AffineChartChain.source_triangle must span a reference plane; got singular source transform"
            )
        return area
    if _prepared_supports_separated(
        preparation.polygon(first),
        preparation.polygon(second),
        budget=preparation.budget,
        sides=preparation.sides,
    ):
        return Fraction(0)
    with preparation.budget.temporary_scope():
        return _physical_intersection(first, second, preparation)[2]


def _physical_intersection(
    first: _Triangle, second: _Triangle, preparation: _Preparation,
) -> tuple[_Triangle, tuple[int, ...], Fraction]:
    """Construct a canonical exact loop and independently check every support."""
    for points in (first, second):
        preparation.budget.reserve(2 + len(points) + sum(map(len, points)), 256)
        if len(points) != 3 or any(
            len(point) != 2 or any(not isinstance(value, Fraction) for value in point)
            for point in points
        ):
            raise ValueError("AffineChartChain.triangle must contain three exact planar corners")
        preparation.budget.reserve(1)
        if preparation.signed_measure(points) == 0:
            raise ValueError(
                "AffineChartChain.source_triangle must span a reference plane; got singular source transform"
            )
    source = preparation.polygon(first)
    target = preparation.polygon(second)
    if source.polygon is not first or target.polygon is not second:
        raise ValueError("AffineChartChain.preparation must bind the current exact polygons")
    preparation.clip_calls += 1
    polygon = convex_hull(
        intersection_polygon(
            first, second, budget=preparation.budget, target_preparation=target,
        ),
        budget=preparation.budget,
    )
    if len(polygon) < 3:
        return (), (), Fraction(0)
    planes = (
        source.halfplanes[2], source.halfplanes[0], source.halfplanes[1],
        *target.halfplanes,
    )
    bits = preparation.bits((polygon, planes))
    preparation.budget.reserve(0, 1024)
    masks = []
    for point in polygon:
        def inspect_point() -> tuple[int, int]:
            actual = 0
            mask = 0
            for slot, plane in enumerate(planes):
                actual += 6
                residual = plane[0] * point[0] + plane[1] * point[1] - plane[2]
                if residual > 0:
                    raise ValueError(
                        "AffineChartChain.vertices must satisfy every retained halfplane"
                    )
                if residual == 0:
                    actual += 2
                    mask |= 1 << slot
            return mask, actual

        masks.append(preparation.stage(
            8 * len(planes), 4, 4 * bits + 64, inspect_point,
        ))
    labels = []
    for first_mask, second_mask in zip(masks, (*masks[1:], masks[0]), strict=True):
        preparation.budget.reserve(5, 128)
        common = first_mask & second_mask
        if not common:
            raise ValueError("AffineChartChain.edges must lie on a retained exact support")
        labels.append((common & -common).bit_length() - 1)
    measure = preparation.stage(
        4 * len(polygon) + 1, 6, 4 * bits + 64,
        lambda: (prepared_signed_measure(polygon), 4 * len(polygon) + 1),
    )
    preparation.budget.reserve(1)
    if measure <= 0:
        raise ValueError("AffineChartChain.polygon must have positive exact signed area")
    return polygon, tuple(labels), measure


def _sqrt_lower(value: Fraction, /) -> float:
    """Greatest practical binary64 lower bound of a rational nonnegative root."""
    if value <= 0:
        return 0.0
    exponent = (value.numerator.bit_length() - value.denominator.bit_length()) // 2
    scaled = value / (Fraction(2) ** (2 * exponent))
    root = math.sqrt(float(scaled))
    try:
        candidate = math.ldexp(root, exponent)
    except OverflowError:
        candidate = np.finfo(np.float64).max
    if not math.isfinite(candidate):
        candidate = np.finfo(np.float64).max
    while candidate > 0.0 and Fraction.from_float(candidate) ** 2 > value:
        candidate = float(np.nextafter(candidate, -math.inf))
    following = float(np.nextafter(candidate, math.inf))
    while math.isfinite(following) and Fraction.from_float(following) ** 2 <= value:
        candidate = following
        following = float(np.nextafter(candidate, math.inf))
    return candidate


class _CandidateChains(StrictModule):
    target: tuple[tuple[int, int], ...] = eqx.field(static=True)
    source: tuple[tuple[int, int], ...] = eqx.field(static=True)
    cross: tuple[tuple[int, int], ...] = eqx.field(static=True)


class _PreparedChains(StrictModule):
    target: tuple[tuple[_Triangle, _PlaneKey | None], ...]
    source: tuple[tuple[_Triangle, _PlaneKey | None], ...]
    records: dict[int, tuple[_Triangle, _PlaneKey | None]]
    point_bits: int = eqx.field(static=True)


@dataclass(slots=True)
class _SourceIdentityOperation:
    pending: _SourceIdentityOperands | None = None


@dataclass(frozen=True, slots=True)
class _SourceIdentityOperands:
    """One-use source-key import, issued only after current cover authentication."""

    operation: _SourceIdentityOperation
    topology: _PlanarTopology
    simplices: np.ndarray
    layout: tuple[object, ...]
    preparation: _Preparation
    rows: tuple[np.ndarray, ...]
    records: tuple[tuple[_Triangle, _PlaneKey | None], ...]
    owners: dict[int, tuple[_Triangle, _PlaneKey | None]]

    def consume(
        self, rows: tuple[np.ndarray, ...], prepared: _PreparedChains,
        preparation: _Preparation,
    ) -> Mapping[_IdentityKey, tuple[int, ...]]:
        pending, self.operation.pending = self.operation.pending, None
        if (
            pending is not self or self.preparation is not preparation or self.rows is not rows
            or self.records is not prepared.source or self.owners is not prepared.records
        ):
            raise ValueError("AffineChartChain.source_keys must bind current ordered owners once")
        topology, budget = self.topology, preparation.budget
        budget.reserve(6, topology.storage_bytes)
        if (
            topology.row_count != len(rows)
            or not topology.bindings or len(topology.bindings[0]) != 4
            or topology.bindings[0][0] is not topology.row_count
            or topology.bindings[0][1] is not topology.planes
            or topology.bindings[0][2] is not topology.identity_keys
            or topology.bindings[0][3] is not topology.storage_bytes
            or immutable_host_array_layout(self.simplices) != self.layout
            or self.simplices.shape != (len(rows), 3, 3)
        ):
            raise ValueError("AffineChartChain.source_keys must bind unchanged schedule and cover owners")
        address = self.simplices.__array_interface__["data"][0]
        for index, (row, record) in enumerate(zip(rows, self.records, strict=True)):
            budget.reserve(7)
            layout = immutable_host_array_layout(row)
            if (
                layout is None
                or layout[:3] != (self.simplices.dtype, (3, 3), self.simplices.strides[1:])
                or layout[3] != address + index * self.simplices.strides[0]
                or layout[4] != self.simplices.strides[0] or layout[-1] is not self.layout[-1]
                or self.owners.get(id(row)) is not record
            ):
                raise ValueError("AffineChartChain.source_keys must preserve current cover row order")
            points, plane = record
            budget.reserve(17)
            if (
                plane is None or len(points) != 3 or any(len(point) != 3 for point in points)
                or preparation._point_profile(points) is None
                or preparation._point_profile(plane[0]) is None
            ):
                raise ValueError("AffineChartChain.source_keys must bind unchanged exact source owners")
        return topology.identity_keys


def _source_identity_operands(
    topology: _PlanarTopology | None, simplices: np.ndarray,
    rows: tuple[np.ndarray, ...], prepared: _PreparedChains, preparation: _Preparation,
) -> _SourceIdentityOperands | None:
    """Seal the closed caller's just-authenticated immutable source-cover role."""
    if topology is None:
        return None
    layout = immutable_host_array_layout(simplices)
    if layout is None:
        return None
    preparation.budget.reserve(1, 512)
    operation = _SourceIdentityOperation()
    operands = _SourceIdentityOperands(
        operation, topology, simplices, layout, preparation, rows, prepared.source, prepared.records,
    )
    operation.pending = operands
    return operands


class _ChainDecision(StrictModule):
    complete: bool = eqx.field(static=True)
    separation: float | None = eqx.field(static=True)


def boxes(rows: tuple[np.ndarray, ...], /) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray([np.min(row, axis=0) for row in rows])
    upper = np.asarray([np.max(row, axis=0) for row in rows])
    return np.nextafter(lower, -np.inf), np.nextafter(upper, np.inf)


def _chain_candidates(
    target_rows: tuple[np.ndarray, ...],
    source_rows: tuple[np.ndarray, ...],
    limits: MeshCertificateLimits,
    *,
    matched_rows: int | None = None,
    target_embedded: bool = False,
    source_embedded: bool = False,
    theorem_obligations: int = 0,
) -> _CandidateChains | None:
    capacity = min(
        limits.maximum_candidate_pairs,
        limits.maximum_distance_evaluations,
    )
    capacity -= theorem_obligations
    if capacity < 0:
        return None
    # A proved bijection replaces cross-chain search, not either partition's
    # independent nonoverlap proof. Its matched rows still own candidate slots.
    if matched_rows is not None:
        if matched_rows > min(capacity, limits.maximum_source_samples):
            return None
        capacity -= matched_rows
    target_boxes, source_boxes = boxes(target_rows), boxes(source_rows)
    if target_embedded:
        target_first = target_second = np.empty(0, dtype=np.int64)
        target_exhausted = False
    else:
        target_first, target_second, target_exhausted = embedding_state_owner.candidate_pairs(
            *target_boxes, capacity,
        )
    remaining = capacity - target_first.size
    if source_embedded:
        source_first = source_second = np.empty(0, dtype=np.int64)
        source_exhausted = False
    else:
        source_first, source_second, source_exhausted = embedding_state_owner.candidate_pairs(
            *source_boxes, max(remaining, 0),
        )
    remaining -= source_first.size
    if matched_rows is None:
        cross_first, cross_second, cross_exhausted = embedding_state_owner.candidate_pairs(
            *target_boxes,
            max(remaining, 0),
            other=source_boxes,
        )
        if cross_exhausted:
            return None
        cross_pairs = tuple(
            (int(first), int(second))
            for first, second in zip(cross_first, cross_second, strict=True)
        )
    else:
        cross_pairs = ()
    if target_exhausted or source_exhausted:
        return None
    target_pairs = tuple(
        (int(first), int(second))
        for first, second in zip(target_first, target_second, strict=True)
    )
    source_pairs = tuple(
        (int(first), int(second))
        for first, second in zip(source_first, source_second, strict=True)
    )
    return _CandidateChains(target_pairs, source_pairs, cross_pairs)


def _prepare_chain_records(
    target_rows: tuple[np.ndarray, ...],
    source_rows: tuple[np.ndarray, ...],
    preparation: _Preparation,
) -> _PreparedChains:
    all_rows = (*target_rows, *source_rows)
    scalar_count = sum(row.size for row in all_rows)

    def point_bits_operation() -> tuple[int, int]:
        bits = max(
            (
                binary64_fraction_bits(float(value))
                for row in all_rows
                for value in row.flat
            ),
            default=1,
        )
        return bits, scalar_count

    point_bits = preparation.stage(scalar_count, 1, 64, point_bits_operation)

    def exact_points(row: np.ndarray, /) -> _Triangle:
        preparation.budget.reserve(0, 256 + 16 * row.shape[0])
        points = tuple(preparation.vertex(vertex, point_bits) for vertex in row)
        preparation.prepare_points(points)
        return points

    with preparation.budget.temporary_scope():
        target_records = tuple(
            (points, _plane(points, preparation))
            for points in (exact_points(row) for row in target_rows)
        )
        source_records = tuple(
            (points, _plane(points, preparation))
            for points in (exact_points(row) for row in source_rows)
        )
        for _, plane in (*target_records, *source_records):
            if plane is not None:
                preparation.prepare_points(plane[0])
    preparation.retain((target_records, source_records))
    records = {
        id(row): record
        for rows, prepared in (
            (target_rows, target_records),
            (source_rows, source_records),
        )
        for row, record in zip(rows, prepared, strict=True)
    }
    return _PreparedChains(target_records, source_records, records, point_bits)


def _exact_overlap(
    first_record: tuple[_Triangle, _PlaneKey | None],
    second_record: tuple[_Triangle, _PlaneKey | None],
    preparation: _Preparation,
) -> bool | None:
    first, first_plane = first_record
    second, second_plane = second_record
    if first_plane is None or second_plane is None:
        return None
    if first_plane[0] != second_plane[0]:
        return False
    with preparation.budget.temporary_scope():
        return (
            preparation.area(
                preparation.projected(first, first_plane[1]),
                preparation.projected(second, first_plane[1]),
            )
            != 0
        )


def _chain_separation(
    target_rows: tuple[np.ndarray, ...],
    source_rows: tuple[np.ndarray, ...],
    records: dict[int, tuple[_Triangle, _PlaneKey | None]],
    maximum_deviation: float,
    tolerance: float,
    preparation: _Preparation,
) -> _ChainDecision:
    complete = True
    separation = None
    separated = True
    closest_lower = math.inf
    for source_row in source_rows:
        with preparation.budget.temporary_scope():
            target, _ = records[id(target_rows[0])]
            source, plane = records[id(source_row)]
            if plane is None:
                complete = separated = False
                break
            proved_plane = plane

            def separation_operation() -> tuple[float | None, int]:
                threshold = Fraction(tolerance) + Fraction(maximum_deviation)
                normal, offset = proved_plane[0][:-1], proved_plane[0][-1]
                signed = (
                    sum(
                        (
                            value * coordinate
                            for value, coordinate in zip(
                                normal,
                                target[0],
                                strict=True,
                            )
                        ),
                        Fraction(0),
                    )
                    + offset
                )
                norm_squared = sum(
                    (value * value for value in normal),
                    Fraction(0),
                )
                distance = signed * signed / norm_squared
                if distance <= threshold * threshold:
                    return None, 19
                lower = _sqrt_lower(distance)
                return max(
                    0.0,
                    float(np.nextafter(lower - maximum_deviation, -math.inf)),
                ), 21

            candidate = preparation.stage(
                21,
                16,
                16 * preparation.bits((target[0], plane[0])) + 256,
                separation_operation,
            )
        if candidate is None:
            separated = False
            break
        closest_lower = min(closest_lower, candidate)
    if complete and separated:
        separation = closest_lower
        complete = False
    return _ChainDecision(complete, separation)


def _chain_areas_equal(
    rows: tuple[np.ndarray, ...],
    areas: list[Fraction],
    records: dict[int, tuple[_Triangle, _PlaneKey | None]],
    preparation: _Preparation,
    *,
    matched_areas: Mapping[int, Fraction] | None = None,
) -> bool:
    for row, area in zip(rows, areas, strict=True):
        if matched_areas is None:
            points, plane = records[id(row)]
            if plane is None:
                return False
            with preparation.budget.temporary_scope():
                projected = preparation.projected(points, plane[1])
                measure = preparation.signed_measure(projected)
                preparation.budget.reserve(1)
                expected = abs(measure)
        else:
            # Only the nonescaping bijection operation supplies these numeric
            # areas, after authenticating both complete corner/plane owners.
            preparation.budget.reserve(1)
            expected = matched_areas[id(row)]
        if area != expected:
            return False
    return True


def _chain_accumulator_bits(point_bits: int, pair_count: int) -> int:
    """Bound exact intersection areas and every partial sum, without profiling.

    Binary64 coordinates have power-of-two denominators. Scale all coordinates
    by Q=2**(b-1), b=point_bits: integer coordinates have magnitude <2**(2*b).
    Integer supporting lines then have A,B <2**(2*b+1), C <2**(4*b+1).
    Their intersection has homogeneous denominator |W| <2**(4*b+3).
    An intersection polygon has at most six vertices. Its shoelace denominator
    divides 2*Q**2 times the product of their nonzero W's (original vertices use
    W=1), so has at most 26*b+19 bits. Its area is <2**(2*b+3),
    giving at most 28*b+22 numerator bits. Thus 32*b+256 bounds either integer.

    For n areas, a product of denominators bounds the sum's denominator.
    Each numerator after multiplication by the other denominators is <2**(n*B);
    their sum is <n*2**(n*B). Reduction can only decrease both integer sizes.
    In particular, logarithmic growth alone does NOT bound the denominator:
    different intersections can introduce coprime factors.
    """
    count = max(1, pair_count)
    return count * (32 * point_bits + 256) + count.bit_length()


def _chain_identity_mapping(
    target_rows: tuple[np.ndarray, ...],
    source_rows: tuple[np.ndarray, ...],
    prepared: _PreparedChains,
    preparation: _Preparation,
    *,
    source_keys: _SourceIdentityOperands | None = None,
) -> tuple[int, ...] | None:
    """Prove a fresh bijection, never reuse a chain or material verdict.

    The caller has already authenticated one source-boundary scope and its
    role. These operands have no independent material labels: the theorem is
    local to that invocation, not transferable between scoped chain owners.
    Exact endpoints determine every edge support. A triangle's six vertex
    permutations are precisely its cyclic/reversed boundary orderings.
    """
    budget = preparation.budget
    compiled_source = (
        None if source_keys is None else source_keys.consume(source_rows, prepared, preparation)
    )
    budget.reserve(1)
    count = len(target_rows)
    if not count or count != len(source_rows):
        return None
    if len(prepared.target) != count or len(prepared.source) != count:
        return None
    # Keys reference existing exact scalars; include row lists, sort wrappers,
    # merge scratch, index integers, mapping and all transient key tuples.
    key_bytes = 2048 + (4096 if compiled_source is None else 2048) * count
    budget.reserve(0, key_bytes)
    preparation.identity_key_bytes = max(preparation.identity_key_bytes, key_bytes)

    def compare(first: tuple[Fraction, ...], second: tuple[Fraction, ...]) -> int:
        return _compare_identity_coordinates(first, second, preparation)

    def keys(
        rows: tuple[np.ndarray, ...],
        records: tuple[tuple[_Triangle, _PlaneKey | None], ...],
        *, sort_rows: bool = True,
    ) -> list[tuple[tuple[Fraction, ...], int]] | None:
        result = []
        for index, (row, record) in enumerate(zip(rows, records, strict=True)):
            points, plane = record
            budget.reserve(1)
            if prepared.records.get(id(row)) is not record:
                budget.reserve(17)
                if prepared.records.get(id(row)) != record:
                    raise ValueError("AffineChartChain.identity must bind unchanged record owners")
            if plane is None:
                return None
            if not points or row.shape != (len(points), len(points[0])):
                raise ValueError("AffineChartChain.identity must bind unchanged row owners")
            budget.reserve(1 + len(points) + sum(map(len, points)) + len(plane[0]))
            if (
                len(points) != 3
                or any(len(point) != 3 for point in points)
                or preparation._point_profile(points) is None
                or preparation._point_profile(plane[0]) is None
            ):
                return None
            # A current ndarray owner may have changed since exact preparation.
            # Fraction/float equality is exact, not a binary64 round trip.
            for point, current in zip(points, row, strict=True):
                for value, coordinate in zip(point, current, strict=True):
                    budget.reserve(1)
                    if value != float(coordinate):
                        raise ValueError("AffineChartChain.identity must bind unchanged row owners")
            ordered = sorted(points, key=cmp_to_key(compare))
            budget.reserve(13)
            # Normalized plane coefficients are orientation-independent. The
            # final sign in _PlaneKey deliberately changes under reversal;
            # axes and projection orientation are fixed by the normal.
            key = tuple(value for point in ordered for value in point) + plane[0]
            result.append((key, index))
        if sort_rows:
            result.sort(key=cmp_to_key(lambda first, second: compare(first[0], second[0])))
        return result

    target = keys(target_rows, prepared.target, sort_rows=compiled_source is None)
    source = keys(source_rows, prepared.source) if compiled_source is None else None
    if target is None or (source is None and compiled_source is None):
        return None
    budget.reserve(count)
    mapping = [0] * count
    budget.reserve(count if compiled_source is not None else 0)
    seen = [False] * count if compiled_source is not None else []
    for index, (target_key, first) in enumerate(target):
        if compiled_source is None:
            assert source is not None
            source_key, second = source[index]
            if compare(target_key, source_key):
                return None
            duplicate = bool(index) and (
                compare(target[index - 1][0], target_key) == 0
                or compare(source[index - 1][0], source_key) == 0
            )
        else:
            # The source lookup contains full exact integer-pair keys and every
            # incident source row, including duplicates. Equality is exact;
            # neither a digest nor a cached match can authorize a row.
            budget.reserve(26)
            encoded_key = tuple(
                component for value in target_key
                for component in (value.numerator, value.denominator)
            )
            budget.reserve(4)
            incidence = compiled_source.get(encoded_key, ())
            if len(incidence) != 1:
                return None
            second = incidence[0]
            duplicate = seen[second]
            seen[second] = True
            # One hit per target and one use per singleton source incidence,
            # with equal complete row counts, prove both key uniqueness and
            # exhaustiveness afresh without sorting either full role again.
        if duplicate:
            return None
        target_plane = prepared.target[first][1]
        source_plane = prepared.source[second][1]
        assert target_plane is not None and source_plane is not None
        budget.reserve(3)
        if target_plane[1:3] != source_plane[1:3]:
            return None
        mapping[first] = second
    budget.reserve(count)
    return tuple(mapping)


def _chain_identity_tiling(
    target_rows: tuple[np.ndarray, ...],
    source_rows: tuple[np.ndarray, ...],
    prepared: _PreparedChains,
    preparation: _Preparation,
    *,
    source_keys: _SourceIdentityOperands | None = None,
) -> bool | None:
    """Prove matched areas; publication still requires both partition proofs."""
    preparation.identity_mapping = None
    with preparation.budget.temporary_scope():
        mapping = _chain_identity_mapping(
            target_rows, source_rows, prepared, preparation,
            source_keys=source_keys,
        )
        if mapping is None:
            return None
        preparation.identity_rows = len(mapping)
        preparation.budget.reserve(4 * len(mapping), 1024 + 288 * len(mapping))
        target_area = [Fraction(0)] * len(mapping)
        source_area = [Fraction(0)] * len(mapping)
        matched_areas: dict[int, Fraction] = {}
        for first, second in enumerate(mapping):
            # Complete exact keys authenticated both operands before this
            # operation. Vertex permutations preserve unsigned projected area,
            # and the projection axes also matched. Compute that numeric area
            # once; do not export a tiling verdict or seed a preparation cache.
            points, plane = prepared.target[first]
            assert plane is not None
            projected = preparation.projected(points, plane[1])
            measure = preparation.signed_measure(projected)
            bits = preparation.bits((measure,))
            reserve_polynomial(1, 1, 1, bits, budget=preparation.budget)
            area = abs(measure)
            preparation.budget.reserve(4)
            target_area[first] = source_area[second] = area
            matched_areas[id(target_rows[first])] = area
            matched_areas[id(source_rows[second])] = area
            if area <= 0:
                return False
        complete = _chain_areas_equal(
            target_rows, target_area, prepared.records, preparation,
            matched_areas=matched_areas,
        ) and _chain_areas_equal(
            source_rows, source_area, prepared.records, preparation,
            matched_areas=matched_areas,
        )
        if complete:
            # Retain only the authenticated numeric permutation, never the
            # identity/coverage decision. The closed theorem caller consumes
            # it immediately; every identity invocation replaces it.
            preparation.cache_bytes_upper += 256 + len(mapping) * (
                8 + declared_integer_storage_bytes(len(mapping))
            )
            preparation.cache_storage.set_bound(preparation.cache_bytes_upper, work=1)
            preparation.identity_mapping = mapping
        return complete


def _chain_embedding_proofs(
    target_rows: tuple[np.ndarray, ...],
    source_rows: tuple[np.ndarray, ...],
    prepared: _PreparedChains,
    preparation: _Preparation,
    capacity: int,
    *,
    topology: _PlanarTopology | None = None,
    source_keys: _SourceIdentityOperands | None = None,
) -> tuple[bool | None, PlanarEmbedding, PlanarEmbedding]:
    """Authenticate and immediately spend role-local numeric capabilities.

    The exact bijection authenticates every row, Fraction owner and canonical
    plane before either token exists. No external callback occurs from there
    through token consumption. Tokens are neither stored in preparation nor
    returned, and unsuccessful identity authentication takes the raw path.
    """
    identity = _chain_identity_tiling(
        target_rows, source_rows, prepared, preparation,
        source_keys=source_keys,
    )
    remaining = max(0, capacity - (len(target_rows) if identity is not None else 0))
    mapping = preparation.identity_mapping
    preparation.identity_mapping = None
    with preparation.budget.temporary_scope():
        # Role batches, projected tuple slots, canonical plane index, numeric
        # permutation, operation and tokens coexist with topology scratch.
        preparation.budget.reserve(0, 2048 + 1536 * len(target_rows) if identity is True else 512)
        batches: tuple[tuple[_PreparedTriangle, ...], ...] = ()
        if identity is True:
            assert mapping is not None
            plane_groups: dict[_Point, _PreparedTriangle | list[_PreparedTriangle]] = {}
            target_operands: list[_PreparedTriangle] = []
            source_operands: list[_PreparedTriangle | None] = [None] * len(source_rows)
            preparation.budget.reserve(2 * len(mapping))
            for first, second in enumerate(mapping):
                points, plane = prepared.target[first]
                source_points, source_plane = prepared.source[second]
                assert plane is not None and source_plane is not None
                preparation.budget.reserve(5)
                projected = preparation.projected(points, plane[1])
                measure = preparation.signed_measure(projected)
                operand = _PreparedTriangle(points, plane, projected, measure, plane[0])
                _group_prepared_triangle(plane_groups, operand, preparation)
                target_operands.append(operand)
                # Exact preparation shares immutable vertex owners. Reorder
                # their already projected numeric operands by the authenticated
                # bijection, not by another Fraction canonicalization.
                preparation.budget.reserve(10)
                vertices = {id(point): flat for point, flat in zip(points, projected, strict=True)}
                source_projection = []
                for point in source_points:
                    flat = vertices.get(id(point))
                    if flat is None:
                        # A general caller may supply equal but distinct exact
                        # owners. Such inputs retain the fully prepared route.
                        source_projected = preparation.projected(source_points, source_plane[1])
                        break
                    source_projection.append(flat)
                else:
                    source_projected = tuple(source_projection)
                source_measure = measure
                if plane[3] != source_plane[3]:
                    reserve_polynomial(
                        1, 1, 1, 8 * prepared.point_bits + 64, budget=preparation.budget,
                    )
                    source_measure = -measure
                source_operands[second] = _PreparedTriangle(
                    source_points, source_plane, source_projected, source_measure, plane[0],
                )
            batches = (
                tuple(target_operands),
                tuple(operand for operand in source_operands if operand is not None),
            )
        results: list[PlanarEmbedding] = []
        tokens: list[_PreparedPlanarOperands] = []
        try:
            for role, (rows, records) in enumerate((
                (target_rows, prepared.target), (source_rows, prepared.source),
            )):
                # Preserve the pairwise fallback for other tiny chains, but a
                # single simplex has a canonical vacuous nonoverlap theorem.
                if 1 < len(rows) <= 3 or not rows:
                    results.append(PlanarEmbedding(False, 0, 0, 0, 0))
                    continue
                token = None
                if identity is True:
                    preparation.budget.reserve(1)
                    operation = _PlanarOperation()
                    token = _PreparedPlanarOperands(
                        operation, preparation, rows, records, prepared.records, batches[role],
                        groups=plane_groups if role == 0 else None,
                        topology=topology if role == 0 else None,
                        source_mapping=mapping if role == 0 and topology is not None else None,
                    )
                    operation.pending = token
                    tokens.append(token)
                if role == 1 and token is not None and results[0].complete:
                    assert mapping is not None
                    result = _transport_planar_embedding(
                        rows, records, prepared.records, preparation,
                        capacity=remaining, source=token, target=tokens[0],
                        mapping=mapping, evidence=results[0],
                    )
                else:
                    result = _prove_planar_embedding(
                        rows, records, prepared.records, preparation,
                        capacity=remaining, point_bits=prepared.point_bits,
                        _prepared=token,
                    )
                results.append(result)
                remaining = max(0, remaining - result.obligations)
            return identity, results[0], results[1]
        finally:
            for token in tokens:
                token.operation.close()


def _chain_tiling(
    target_rows: tuple[np.ndarray, ...],
    source_rows: tuple[np.ndarray, ...],
    cross_pairs: tuple[tuple[int, int], ...],
    prepared: _PreparedChains,
    preparation: _Preparation,
) -> bool:
    complete = True
    row_count = len(target_rows) + len(source_rows)
    zero_bytes = (
        512 + declared_integer_storage_bytes(0) + declared_integer_storage_bytes(1)
    )
    # Four list headers, their slots, and one admitted-byte counter per row.
    # Only the counter's logarithmic metadata depends on the possible sum
    # length; Fraction storage itself is admitted from the current values.
    counter_bits = (
        prepared.point_bits.bit_length() + len(cross_pairs).bit_length() + 16
    )
    schedule_bytes = 1024 + row_count * (
        32 + declared_integer_storage_bytes(counter_bits)
    )
    accumulator_bytes = schedule_bytes + row_count * zero_bytes
    with preparation.budget.live_storage() as area_storage:
        area_storage.set_bound(accumulator_bytes, work=row_count)
        target_area = [Fraction(0) for _ in target_rows]
        source_area = [Fraction(0) for _ in source_rows]
        target_bytes = [zero_bytes] * len(target_rows)
        source_bytes = [zero_bytes] * len(source_rows)
        # Stable streaming partition: compare the canonical coefficients, not
        # hashes or cached verdicts. No pair index is allocated, and every
        # broad-phase candidate remains charged by the caller. In particular,
        # distinct supporting planes prove zero surface area even when their
        # triangles intersect along a line.
        for first, second in cross_pairs:
            target, target_plane = prepared.records[id(target_rows[first])]
            source, source_plane = prepared.records[id(source_rows[second])]
            if target_plane is None or source_plane is None:
                complete = False
                break
            preparation.budget.reserve(1)
            preparation.plane_comparisons += 1
            if target_plane[0] != source_plane[0]:
                preparation.plane_mismatches += 1
                continue
            with preparation.budget.temporary_scope():
                area = preparation.area(
                    preparation.projected(target, target_plane[1]),
                    preparation.projected(source, target_plane[1]),
                )
                if area == 0:
                    preparation.zero_area_pairs += 1
                    continue
                bits = preparation.bits(
                    (area, target_area[first], source_area[second])
                )
                # For a/b + c/d, |ad + cb| < 2**(2*bits + 1) and
                # bd < 2**(2*bits). Reduction can only shrink either integer.
                # Admit both results together before either ordered addition.
                result_bytes = (
                    512
                    + declared_integer_storage_bytes(2 * bits + 1)
                    + declared_integer_storage_bytes(2 * bits)
                )
                next_bytes = (
                    accumulator_bytes
                    - target_bytes[first]
                    - source_bytes[second]
                    + 2 * result_bytes
                )

                def addition_operation() -> tuple[None, int]:
                    area_storage.set_bound(next_bytes)
                    target_area[first] += area
                    source_area[second] += area
                    return None, 2

                preparation.stage(2, 2, 8 * bits + 64, addition_operation)
                # These values now exist. Keep only their actual declared
                # integer storage, never an optimistic post-allocation bound.
                target_bytes[first] = (
                    512
                    + declared_integer_storage_bytes(
                        target_area[first].numerator.bit_length()
                    )
                    + declared_integer_storage_bytes(
                        target_area[first].denominator.bit_length()
                    )
                )
                source_bytes[second] = (
                    512
                    + declared_integer_storage_bytes(
                        source_area[second].numerator.bit_length()
                    )
                    + declared_integer_storage_bytes(
                        source_area[second].denominator.bit_length()
                    )
                )
                accumulator_bytes = (
                    next_bytes
                    - 2 * result_bytes
                    + target_bytes[first]
                    + source_bytes[second]
                )
                area_storage.set_bound(accumulator_bytes)
        if complete:
            for rows, areas in ((target_rows, target_area), (source_rows, source_area)):
                if not _chain_areas_equal(rows, areas, prepared.records, preparation):
                    complete = False
                    break
    return complete


def _prepare_source_planar_topology(
    simplices: np.ndarray, budget: CoordinateEnclosureBudget,
) -> _PlanarTopology | None:
    """Prepare source numerical topology under its genuine proof request."""
    with (
        budget.temporary_scope(),
        budget.live_storage() as cache,
        budget.live_storage() as sides,
        budget.live_storage() as basis,
    ):
        budget.reserve(len(simplices), 256 + 128 * len(simplices))
        rows = tuple(np.asarray(triangle, dtype=np.float64) for triangle in simplices)
        preparation = _Preparation(
            budget=budget, cache_storage=cache, side_storage=sides, basis_storage=basis,
        )
        try:
            prepared = _prepare_chain_records((), rows, preparation)
            try:
                topology = prepare_planar_topology(prepared.source, preparation)
            finally:
                del prepared
        finally:
            preparation.close()
            del preparation, rows
        # Only the compact schedule survives. Its retained-owner traversal is
        # the existing one, after all construction profile indices are gone.
    if topology is not None:
        budget.retain_owner(topology)
    return topology


@checked
def prove_affine_chart_chain(
    proxies: list[tuple[np.ndarray, float, float]],
    cover: source_boundary_records_owner.SourceBoundaryChartCover,
    limits: MeshCertificateLimits,
    tolerance: float,
    *,
    source_identity: tuple[str, str],
    areas: Mapping[tuple[_Triangle, _Triangle], Fraction] | None = None,
    earned_areas: dict[tuple[_Triangle, _Triangle], Fraction] | None = None,
) -> tuple[
    float | None, source_boundary_records_owner.SourceBoundaryChartCover, float | None
]:
    """Prove that the affine mesh proxies and source chart chain tile one PLC.

    Each proxy carries the certified Bernstein upper bound of its measured map
    minus the affine map through its binary64 corners, so corner rounding is
    included. A proved tiling returns the outward-rounded two-sided bound
    ``cover + proxy`` deviation; ``None`` means the chain was not proved.
    ``areas`` supplies exact projected triangle-pair areas already proved in the
    same request; ``earned_areas`` receives the pairs newly proved here.
    ``source_identity`` is the caller-authenticated scoped source owner. The
    actual proxies are this invocation's target role; no bijection theorem is
    exported or shared between owners, roles, requests or material scopes.
    """

    if source_identity != (cover.source_id, cover.source_revision):
        raise ValueError(
            "AffineChartChain.source_identity must bind the authenticated scoped source owner"
        )
    if not all(
        math.isfinite(deviation) and math.isfinite(rounding)
        for _, deviation, rounding in proxies
    ):
        return None, cover, None
    proxy_deviation = max((deviation for _, deviation, _ in proxies), default=0.0)
    cover_deviation = float(np.max(cover.deviation_bounds, initial=0.0))
    maximum_deviation = (
        cover_deviation
        if proxy_deviation == 0.0
        else float(np.nextafter(cover_deviation + proxy_deviation, math.inf))
    )
    target_rows = tuple(
        np.asarray(corners, dtype=np.float64) for corners, _, _ in proxies
    )
    source_rows = tuple(
        np.asarray(triangle, dtype=np.float64) for triangle in cover.simplices
    )

    # Completeness/semantics and the source identity are authenticated by
    # _chart_fidelity for this one scoped chart/material chain. Do not use a
    # cover ID, byte-identical rows or another scope's theorem as a verdict.
    if not cover.complete or cover.semantics != "certified":
        return None, cover, None

    native = _meshcore.NativeExecutionScope(limits.native_limits())
    complete = True
    separation = None
    with native:
        initial = native.remaining().remaining_work_units
        remaining = native.remaining()
        ticket = native.batch(
            min(limits.maximum_work_units, remaining.remaining_work_units),
            queries_upper=min(
                limits.maximum_distance_evaluations, remaining.remaining_geometry_queries
            ),
            bytes_upper=min(
                limits.maximum_scratch_bytes, remaining.remaining_managed_bytes
            ),
        )
        view = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        with (
            ticket,
            ticket.native_target(),
            view.measurement() as measurement,
            view.activate(),
            view.live_storage() as cache_storage,
            view.live_storage() as side_storage,
        ):
            preparation = _Preparation(
                budget=view,
                cache_storage=cache_storage,
                side_storage=side_storage,
                areas=areas,
            )
            prepared = _prepare_chain_records(target_rows, source_rows, preparation)
            from ._domain_boundary import _authenticated_chart_topology

            topology = _authenticated_chart_topology(cover, view)
            # Authenticate complete exact keys before paying for broad cross
            # candidates. A matched-area result is not a partition verdict:
            # separation and both independent nonoverlap proofs still follow.
            identity, target_embedding, source_embedding = _chain_embedding_proofs(
                target_rows, source_rows, prepared, preparation,
                min(limits.maximum_candidate_pairs, limits.maximum_distance_evaluations),
                topology=topology,
                source_keys=_source_identity_operands(
                    topology, cover.simplices, source_rows, prepared, preparation,
                ),
            )
            matched_rows = None if identity is None else len(target_rows)
            theorem_obligations = target_embedding.obligations + source_embedding.obligations
            candidates = _chain_candidates(
                target_rows, source_rows, limits, matched_rows=matched_rows,
                target_embedded=target_embedding.complete,
                source_embedded=source_embedding.complete,
                theorem_obligations=theorem_obligations,
            )
            if candidates is None:
                return None, cover, None
            target_pairs, source_pairs, cross_pairs = (
                candidates.target, candidates.source, candidates.cross,
            )
            requested = (
                len(target_pairs) + len(source_pairs)
                + (len(cross_pairs) if matched_rows is None else matched_rows)
            )
            preparation.stage(requested, 0, 1, lambda: (None, requested))
            complete = not any(
                plane is None
                for records in (prepared.target, prepared.source)
                for _, plane in records
            )
            if complete and math.isfinite(maximum_deviation):
                decision = _chain_separation(
                    target_rows,
                    source_rows,
                    prepared.records,
                    maximum_deviation,
                    tolerance,
                    preparation,
                )
                complete, separation = decision.complete, decision.separation
            if complete:
                for chain, pairs in (
                    (target_rows, target_pairs),
                    (source_rows, source_pairs),
                ):
                    for first, second in pairs:
                        overlap = _exact_overlap(
                            prepared.records[id(chain[first])],
                            prepared.records[id(chain[second])],
                            preparation,
                        )
                        if overlap is None or overlap:
                            complete = False
                            break
                    if not complete:
                        break

            if complete:
                complete = (
                    _chain_tiling(
                        target_rows, source_rows, cross_pairs, prepared, preparation,
                    )
                    if identity is None else identity
                )
        consumed = initial - native.remaining().remaining_work_units
    if earned_areas is not None:
        earned_areas.update(preparation.earned_areas)
    evidence = native.evidence
    if evidence is None:
        raise RuntimeError(
            "AffineChartChain.native_scope must retain closed execution evidence; got absent evidence"
        )
    resources: tuple[tuple[source_boundary_records_owner.AffineChainResourceMetric, int, int], ...] = (
        ("total_work", consumed, limits.maximum_work_units),
        (
            "required_work",
            max(consumed, measurement.required_work_units),
            limits.maximum_work_units,
        ),
        (
            "coefficient_work",
            measurement.work_units,
            limits.maximum_work_units,
        ),
        (
            "coefficient_bytes_upper",
            measurement.peak_bytes_upper,
            limits.maximum_scratch_bytes,
        ),
        (
            "retained_coefficient_bytes",
            measurement.retained_basis_bytes,
            limits.maximum_scratch_bytes,
        ),
        (
            "native_peak_bytes",
            int(evidence.memory_counters.peak_bytes),
            limits.maximum_scratch_bytes,
        ),
        (
            "identity_rows",
            preparation.identity_rows,
            limits.maximum_source_samples,
        ),
        (
            "identity_key_bytes_upper",
            preparation.identity_key_bytes,
            limits.maximum_scratch_bytes,
        ),
        (
            "embedding_obligations",
            theorem_obligations,
            limits.maximum_candidate_pairs,
        ),
        ("embedding_target", int(target_embedding.complete), 1),
        ("embedding_source", int(source_embedding.complete), 1),
        (
            "embedding_components",
            target_embedding.components + source_embedding.components,
            limits.maximum_source_samples,
        ),
        (
            "embedding_holes",
            target_embedding.holes + source_embedding.holes,
            limits.maximum_source_samples,
        ),
        (
            "embedding_boundary_edges",
            target_embedding.boundary_edges + source_embedding.boundary_edges,
            6 * limits.maximum_source_samples,
        ),
        (
            "plane_comparisons",
            preparation.plane_comparisons,
            limits.maximum_candidate_pairs,
        ),
        (
            "plane_mismatches",
            preparation.plane_mismatches,
            limits.maximum_candidate_pairs,
        ),
        (
            "zero_area_pairs",
            preparation.zero_area_pairs,
            limits.maximum_candidate_pairs,
        ),
        # Filtering is streamed in original order: no grouping/index storage.
        ("plane_index_bytes", 0, limits.maximum_scratch_bytes),
        (
            "exact_clip_calls",
            preparation.clip_calls,
            limits.maximum_candidate_pairs,
        ),
        (
            "native_clip_work",
            0,  # Physical exact clipping performs no native reference-clip work.
            limits.maximum_work_units,
        ),
        ("native_primitive_counter_available", 0, 1),
    )
    proven_cover = source_boundary_records_owner.SourceBoundaryChartCover(
        cover.simplices,
        cover.deviation_bounds,
        cover.semantics,
        cover.complete,
        cover.source_id,
        cover.source_revision,
        findings=cover.findings,
        resource_counts=cover.resource_counts,
        affine_chain_resources=(
            *(row.as_source() for row in cover.affine_chain_resources),
            *(
                source_boundary_records_owner.AffineChainResourceEvidence(
                    metric, "target", cover.cover_id, used, maximum,
                )
                for metric, used, maximum in resources
            ),
        ),
        restriction_ids=cover.restriction_ids,
    )
    return (maximum_deviation if complete else None), proven_cover, separation
