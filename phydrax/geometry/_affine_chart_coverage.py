#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Native rational intersections of independent affine boundary chart chains."""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction
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
from phydrax.geometry._planar_primitives import (
    _prepared_separated,
    binary64_fraction_bits,
    fraction_points,
    PolygonIntersectionPreparation,
    PolygonSidePreparation,
    prepare_polygon_intersection,
    prepared_plane_key,
    prepared_project,
    prepared_signed_measure,
    project,
)

from .. import _meshcore
from .._host_storage import declared_integer_storage_bytes
from .._strict import StrictModule
from ..linalg._small_batched import (
    ExactSmallLinearActions,
    prepare_exact_small_linear_actions,
)
from ..typing import checked
from ._embedding_state import MeshCertificateLimits


_T = TypeVar("_T")
_Point = tuple[Fraction, ...]
_Triangle = tuple[_Point, ...]
_PlaneKey = tuple[_Point, tuple[int, ...], int, int]
_Halfplane = tuple[Fraction, Fraction, Fraction]


class _Preparation:
    """Rational coefficient preparation over the admitted proof batch."""

    def __init__(
        self,
        ticket: _meshcore.NativeBatchTicket,
        /,
        *,
        budget: CoordinateEnclosureBudget,
        cache_storage: CoordinateLiveStorage,
        side_storage: CoordinateLiveStorage,
    ) -> None:
        self.ticket = ticket
        self.budget = budget
        self.clip_calls = 0
        self.clip_work = 0
        self.cache_storage = cache_storage
        self.cache_bytes_upper = 2560
        cache_storage.set_bound(self.cache_bytes_upper)
        self.sides = PolygonSidePreparation(budget, side_storage)
        self.point_profiles: dict[int, tuple[_Triangle | _Point, int]] = {}
        self.projections: dict[tuple[int, tuple[int, ...]], _Triangle] = {}
        self.polygon_cache: dict[int, PolygonIntersectionPreparation] = {}
        self.action_cache: dict[tuple[_Triangle, _Triangle], ExactSmallLinearActions] = {}
        self.vertex_cache: dict[tuple[float, ...], _Point] = {}

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

    def exact_actions(
        self,
        matrix: tuple[tuple[Fraction, ...], ...],
        right: tuple[tuple[Fraction, ...], ...],
        /,
    ) -> ExactSmallLinearActions:
        self.budget.reserve(1, 96)
        key = matrix, right
        result = self.action_cache.get(key)
        if result is None:
            result = prepare_exact_small_linear_actions(matrix, right, budget=self.budget)
            bits = self.bits((matrix, right, result.actions, result.determinant))
            rows = (*matrix, *right, *(result.actions or ()))
            count = 1 + sum(len(row) for row in rows)
            # For 1–4 dimensions, 8192 bytes cover row headers, the record and
            # status/digest/scalar metadata. The coefficient bound includes
            # row references, one Fraction and two bounded integer payloads.
            self.cache_bytes_upper += 8192 + count * (
                512 + 2 * declared_integer_storage_bytes(bits)
            )
            self.cache_storage.set_bound(self.cache_bytes_upper, work=1)
            self.action_cache[key] = result
        return result

    def retain(self, values: tuple[object, ...], /) -> None:
        self.budget.retain_basis(values)

    def bits(self, values: tuple[object, ...], /) -> int:
        pending = list(values)
        result = 1
        while pending:
            self.budget.reserve(1)
            value = pending.pop()
            prepared = self.point_profiles.get(id(value))
            if prepared is not None and prepared[0] is value:
                result = max(result, prepared[1])
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
        self.retain((points, bits))
        self.retain_cache_entry()
        self.point_profiles[id(points)] = (points, bits)

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

    def polygon(self, points: _Triangle, /) -> PolygonIntersectionPreparation:
        self.budget.reserve(1)
        result = self.polygon_cache.get(id(points))
        if result is None:
            with self.budget.temporary_scope():
                orientation = self.stage(
                    13,
                    12,
                    4 * self.bits(points) + 64,
                    lambda: (1 if prepared_signed_measure(points) > 0 else -1, 13),
                )
                result = prepare_polygon_intersection(
                    points, orientation, budget=self.budget
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


def _intersection(
    first: _Triangle, second: _Triangle, preparation: _Preparation
) -> Fraction:
    """Prove zero-area contacts before native positive-area clipping."""
    if _prepared_separated(
        preparation.polygon(first),
        preparation.polygon(second),
        budget=preparation.budget,
        sides=preparation.sides,
    ):
        return Fraction(0)
    bits = 4 * preparation.bits((first, second)) + 64
    with preparation.budget.temporary_scope():

        def mapping_input_operation() -> tuple[
            tuple[tuple[tuple[Fraction, ...], ...], tuple[tuple[Fraction, ...], ...]],
            int,
        ]:
            matrix = tuple(
                tuple(first[column + 1][axis] - first[0][axis] for column in range(2))
                for axis in range(2)
            )
            right = tuple(
                tuple(point[axis] - first[0][axis] for point in second)
                for axis in range(2)
            )
            return (matrix, right), 10

        matrix, right = preparation.stage(
            10,
            16,
            bits,
            mapping_input_operation,
        )
        construction = preparation.exact_actions(matrix, right)
        if construction.actions is None:
            raise ValueError(
                "AffineChartChain.source_triangle must span a reference plane; got singular source transform"
            )
        vertices = tuple(
            tuple(construction.actions[axis][column] for axis in range(2))
            for column in range(3)
        )
        determinant = construction.determinant

        def plane_operation() -> tuple[tuple[_Halfplane, ...], int]:
            orientation = 1 if prepared_signed_measure(vertices) > 0 else -1
            rows = []
            for a, b in zip(vertices, (*vertices[1:], vertices[0]), strict=True):
                dx, dy = b[0] - a[0], b[1] - a[1]
                rows.append(
                    (
                        orientation * dy,
                        -orientation * dx,
                        orientation * (dy * a[0] - dx * a[1]),
                    )
                )
            planes = (
                (Fraction(-1), Fraction(0), Fraction(0)),
                (Fraction(0), Fraction(-1), Fraction(0)),
                (Fraction(1), Fraction(1), Fraction(1)),
                *rows,
            )
            return planes, 13 + 3 * 8

        planes = preparation.stage(37, 36, bits, plane_operation)

        memory = preparation.ticket.remaining_managed_bytes
        work_remaining = preparation.ticket.remaining_work_units
        if memory <= 0 or work_remaining <= 0:
            preparation.ticket.refuse(
                1 if work_remaining <= 0 else 0,
                bytes_upper=preparation.ticket.bytes_upper + 1 if memory <= 0 else 0,
            )
        preparation.clip_calls += 1
        supports, labels, work = _meshcore.clip_reference_triangle_exact(
            planes,
            maximum_scratch_bytes=memory,
            maximum_work_units=work_remaining,
        )
        preparation.clip_work += work
        if supports.shape[0] < 3:
            return Fraction(0)
        planes_bits = preparation.bits(planes)
        polygon_bits = 1

        polygon = []
        for first_support, second_support in supports:
            if not (0 <= first_support < 6 and 0 <= second_support < 6):
                raise ValueError(
                    "AffineChartChain.native_supports must name retained halfplanes; got out-of-range support index"
                )
            a, b = planes[int(first_support)], planes[int(second_support)]
            construction = preparation.exact_actions(
                (a[:2], b[:2]),
                ((a[2],), (b[2],)),
            )
            if construction.actions is None:
                raise ValueError(
                    "AffineChartChain.native_supports must reconstruct one intersection point; got singular support pair"
                )
            point = construction.actions[0][0], construction.actions[1][0]
            point_bits = preparation.bits((point,))
            polygon_bits = max(polygon_bits, point_bits)

            def point_validation_operation() -> tuple[None, int]:
                for plane in planes:
                    if plane[0] * point[0] + plane[1] * point[1] > plane[2]:
                        raise ValueError(
                            "AffineChartChain.native_vertices must satisfy every retained halfplane; got an outside vertex"
                        )
                return None, 4 * len(planes)

            preparation.stage(
                4 * len(planes),
                4,
                3 * max(point_bits, planes_bits) + 32,
                point_validation_operation,
            )
            polygon.append(point)

        def reconstruction_operation() -> tuple[Fraction, int]:
            actual = 0
            for slot, label in enumerate(labels):
                if not 0 <= label < 6:
                    raise ValueError(
                        "AffineChartChain.native_labels must name retained edges; got out-of-range edge index"
                    )
                plane = planes[int(label)]
                for point in (
                    polygon[slot],
                    polygon[(slot + 1) % len(polygon)],
                ):
                    actual += 4
                    if plane[0] * point[0] + plane[1] * point[1] != plane[2]:
                        raise ValueError(
                            "AffineChartChain.native_edges must lie on their labeled support; got inconsistent endpoints"
                        )
            area = prepared_signed_measure(tuple(polygon))
            actual += 4 * len(polygon) + 1
            if area <= 0 or len(set(polygon)) != len(polygon):
                raise ValueError(
                    "AffineChartChain.native_polygon must have distinct vertices and positive area; got a degenerate or reversed polygon"
                )
            return area * abs(determinant), actual + 1

        # Reconstructed vertices and source planes are already owned above.
        # Edge checks and shoelace accumulation stream over already owned
        # vertices. Two products, their difference, one accumulator, and the
        # final determinant product require at most six extra rational terms.
        return preparation.stage(
            80,
            6,
            4
            * max(
                polygon_bits,
                planes_bits,
                preparation.bits((determinant,)),
            )
            + 64,
            reconstruction_operation,
        )


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
) -> _CandidateChains | None:
    capacity = min(
        limits.maximum_candidate_pairs,
        limits.maximum_distance_evaluations,
    )
    target_boxes, source_boxes = boxes(target_rows), boxes(source_rows)
    target_first, target_second, target_exhausted = embedding_state_owner.candidate_pairs(
        *target_boxes,
        capacity,
    )
    remaining = capacity - target_first.size
    source_first, source_second, source_exhausted = embedding_state_owner.candidate_pairs(
        *source_boxes,
        max(remaining, 0),
    )
    remaining -= source_first.size
    cross_first, cross_second, cross_exhausted = embedding_state_owner.candidate_pairs(
        *target_boxes,
        max(remaining, 0),
        other=source_boxes,
    )
    if target_exhausted or source_exhausted or cross_exhausted:
        return None
    target_pairs = tuple(
        (int(first), int(second))
        for first, second in zip(target_first, target_second, strict=True)
    )
    source_pairs = tuple(
        (int(first), int(second))
        for first, second in zip(source_first, source_second, strict=True)
    )
    cross_pairs = tuple(
        (int(first), int(second))
        for first, second in zip(cross_first, cross_second, strict=True)
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
            _intersection(
                preparation.projected(first, first_plane[1]),
                preparation.projected(second, first_plane[1]),
                preparation,
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
) -> bool:
    for row, area in zip(rows, areas, strict=True):
        points, plane = records[id(row)]
        if plane is None:
            return False
        with preparation.budget.temporary_scope():
            expected = preparation.stage(
                14,
                12,
                8 * preparation.bits(points) + 64,
                lambda: (
                    abs(prepared_signed_measure(prepared_project(points, plane[1]))),
                    14,
                ),
            )
        if area != expected:
            return False
    return True


def _chain_tiling(
    target_rows: tuple[np.ndarray, ...],
    source_rows: tuple[np.ndarray, ...],
    cross_pairs: tuple[tuple[int, int], ...],
    prepared: _PreparedChains,
    preparation: _Preparation,
) -> bool:
    complete = True
    point_bits = prepared.point_bits
    target_area = [Fraction(0) for _ in target_rows]
    source_area = [Fraction(0) for _ in source_rows]
    accumulator_bits = (
        32 * point_bits
        + 256
        + max(1, (max(len(target_rows), len(source_rows)) + 1).bit_length())
    )
    accumulator_bytes = 1024 + (len(target_area) + len(source_area)) * (
        512 + 2 * ((accumulator_bits + 7) // 8)
    )
    with preparation.budget.live_storage() as area_storage:
        area_storage.set_bound(
            accumulator_bytes,
            work=len(target_area) + len(source_area),
        )
        for first, second in cross_pairs:
            with preparation.budget.temporary_scope():
                target, target_plane = prepared.records[id(target_rows[first])]
                source, source_plane = prepared.records[id(source_rows[second])]
                if target_plane is None or source_plane is None:
                    complete = False
                    break
                area = (
                    Fraction(0)
                    if target_plane[0] != source_plane[0]
                    else _intersection(
                        preparation.projected(target, target_plane[1]),
                        preparation.projected(source, target_plane[1]),
                        preparation,
                    )
                )

                def addition_operation() -> tuple[None, int]:
                    target_area[first] += area
                    source_area[second] += area
                    return None, 2

                preparation.stage(
                    2,
                    2,
                    8
                    * preparation.bits(
                        (
                            area,
                            target_area[first],
                            source_area[second],
                        )
                    )
                    + 64,
                    addition_operation,
                )
        if complete:
            for rows, areas in ((target_rows, target_area), (source_rows, source_area)):
                if not _chain_areas_equal(rows, areas, prepared.records, preparation):
                    complete = False
                    break
    return complete


@checked
def prove_affine_chart_chain(
    proxies: list[tuple[np.ndarray, float, float]],
    cover: source_boundary_records_owner.SourceBoundaryChartCover,
    limits: MeshCertificateLimits,
    tolerance: float,
) -> tuple[
    float | None, source_boundary_records_owner.SourceBoundaryChartCover, float | None
]:
    """Prove that the affine mesh proxies and source chart chain tile one PLC.

    Each proxy carries the certified Bernstein upper bound of its measured map
    minus the affine map through its binary64 corners, so corner rounding is
    included. A proved tiling returns the outward-rounded two-sided bound
    ``cover + proxy`` deviation; ``None`` means the chain was not proved.
    """

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

    # A certified cover built from these rational affine proxy rows already owns
    # its completeness and nonoverlap proof. Byte-identical ordered rows need
    # no second all-pairs clipping pass merely to rediscover that same chain.
    if len(target_rows) == len(source_rows) and all(
        np.array_equal(target, source)
        for target, source in zip(target_rows, source_rows, strict=True)
    ):
        return maximum_deviation, cover, None

    candidates = _chain_candidates(target_rows, source_rows, limits)
    if candidates is None:
        return None, cover, None
    target_pairs, source_pairs, cross_pairs = (
        candidates.target,
        candidates.source,
        candidates.cross,
    )
    requested = len(target_pairs) + len(source_pairs) + len(cross_pairs)
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
                ticket,
                budget=view,
                cache_storage=cache_storage,
                side_storage=side_storage,
            )
            preparation.stage(
                requested,
                0,
                1,
                lambda: (None, requested),
            )
            prepared = _prepare_chain_records(target_rows, source_rows, preparation)
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
                complete = _chain_tiling(
                    target_rows, source_rows, cross_pairs, prepared, preparation
                )
        consumed = initial - native.remaining().remaining_work_units
    evidence = native.evidence
    if evidence is None:
        raise RuntimeError(
            "AffineChartChain.native_scope must retain closed execution evidence; got absent evidence"
        )
    resources = (
        *cover.resource_counts,
        ("affine_chain_total_work", consumed, limits.maximum_work_units),
        (
            "affine_chain_coefficient_work",
            measurement.work_units,
            limits.maximum_work_units,
        ),
        (
            "affine_chain_coefficient_bytes_upper",
            measurement.peak_bytes_upper,
            limits.maximum_scratch_bytes,
        ),
        (
            "affine_chain_retained_coefficient_bytes",
            measurement.retained_basis_bytes,
            limits.maximum_scratch_bytes,
        ),
        (
            "affine_chain_native_peak_bytes",
            int(evidence.memory_counters.peak_bytes),
            limits.maximum_scratch_bytes,
        ),
        (
            "affine_chain_exact_clip_calls",
            preparation.clip_calls,
            limits.maximum_candidate_pairs,
        ),
        (
            "affine_chain_native_clip_work",
            preparation.clip_work,
            limits.maximum_work_units,
        ),
        ("affine_chain_native_primitive_counter_available", 0, 1),
    )
    proven_cover = source_boundary_records_owner.SourceBoundaryChartCover(
        cover.simplices,
        cover.deviation_bounds,
        cover.semantics,
        cover.complete,
        cover.source_id,
        cover.source_revision,
        findings=cover.findings,
        resource_counts=resources,
    )
    return (maximum_deviation if complete else None), proven_cover, separation
