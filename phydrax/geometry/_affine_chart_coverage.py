#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Native exact intersections of independent affine boundary chart chains."""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from .. import _meshcore
from ..discretization import _coordinate_enclosure as algebra
from ..linalg._small_batched import (
    ExactSmallLinearActions,
    prepare_exact_small_linear_actions,
)
from ._planar_coverage import (
    _binary64_fraction_bits,
    _fraction_points,
    _plane_key,
    _project,
    _signed_measure,
    project,
)


if TYPE_CHECKING:
    from ._mesh_certificates import MeshCertificateLimits, SourceBoundaryChartCover

_T = TypeVar("_T")
_Point = tuple[Fraction, ...]
_Triangle = tuple[_Point, ...]
_PlaneKey = tuple[_Point, tuple[int, ...], int, int]
_Halfplane = tuple[Fraction, Fraction, Fraction]


class _Preparation:
    """Coefficient preparation borrows the original native remaining allowance."""

    def __init__(self, native: _meshcore.NativeExecutionBudget, /) -> None:
        self.native = native
        remaining = native.remaining()
        self.budget = algebra.CoordinateEnclosureBudget(
            remaining.remaining_work_units,
            remaining.remaining_scratch_bytes,
        )
        self.clip_calls = 0
        self.clip_work = 0

    def stage(
        self,
        bound: int,
        terms: int,
        bits: int,
        operation: Callable[[], tuple[_T, int]],
        /,
    ) -> _T:
        remaining = self.native.remaining()
        self.budget.maximum_work_units = (
            self.budget.work_units + remaining.remaining_work_units
        )
        self.budget.maximum_memory_bytes = remaining.remaining_scratch_bytes
        self.native.admit_work_bound(bound)
        before = self.budget.work_units
        with self.budget.activate():
            algebra._reserve_polynomial(0, terms, 3, bits)
            value, actual = operation()
            self.budget.reserve(actual)
        self.native.charge(work=self.budget.work_units - before)
        return value

    def exact_actions(
        self,
        matrix: tuple[tuple[Fraction, ...], ...],
        right: tuple[tuple[Fraction, ...], ...],
        /,
    ) -> ExactSmallLinearActions:
        """Delegate exact solve preparation and evidence to its native owner."""
        remaining = self.native.remaining()
        self.budget.maximum_work_units = (
            self.budget.work_units + remaining.remaining_work_units
        )
        self.budget.maximum_memory_bytes = remaining.remaining_scratch_bytes
        before = self.budget.work_units
        with self.budget.activate():
            result = prepare_exact_small_linear_actions(
                matrix,
                right,
                coordinate_budget=self.budget,
            )
        self.native.charge(work=self.budget.work_units - before)
        return result

    def retain(self, values: tuple[object, ...], /) -> None:
        remaining = self.native.remaining()
        self.budget.maximum_work_units = (
            self.budget.work_units + remaining.remaining_work_units
        )
        self.budget.maximum_memory_bytes = remaining.remaining_scratch_bytes
        before = self.budget.work_units
        try:
            self.budget.retain_basis(values)
        finally:
            self.native.charge(work=self.budget.work_units - before)

    def bits(self, values: tuple[object, ...], /) -> int:
        remaining = self.native.remaining()
        self.budget.maximum_work_units = (
            self.budget.work_units + remaining.remaining_work_units
        )
        before = self.budget.work_units
        pending = list(values)
        result = 1
        try:
            while pending:
                self.budget.reserve(1)
                value = pending.pop()
                if isinstance(value, Fraction):
                    result = max(
                        result,
                        abs(value.numerator).bit_length(),
                        value.denominator.bit_length(),
                    )
                elif isinstance(value, tuple):
                    pending.extend(value)
        finally:
            self.native.charge(work=self.budget.work_units - before)
        return result


def _plane(points: _Triangle, preparation: _Preparation) -> _PlaneKey | None:
    def operation() -> tuple[_PlaneKey | None, int]:
        result = _plane_key(points)
        # Two edge rows, their cross product, normalized coefficients and
        # offset have thirteen simultaneously live rational terms in 3D.
        return result, 15 if result is None else 25

    return preparation.stage(25, 13, 8 * preparation.bits(points) + 64, operation)


def _intersection(
    first: _Triangle, second: _Triangle, preparation: _Preparation
) -> Fraction:
    """Construct exact coefficient planes; native code owns every clip decision."""
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
                "An affine chart intersection requires a nondegenerate source triangle."
            )
        vertices = tuple(
            tuple(construction.actions[axis][column] for axis in range(2))
            for column in range(3)
        )
        determinant = construction.determinant

        def plane_operation() -> tuple[tuple[_Halfplane, ...], int]:
            orientation = 1 if _signed_measure(vertices) > 0 else -1
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
        remaining = preparation.native.remaining()
        live_host = (
            preparation.budget.retained_basis_bytes
            + preparation.budget.temporary_bytes_upper
        )
        memory = remaining.remaining_scratch_bytes - live_host
        if memory <= 0 or remaining.remaining_work_units <= 0:
            raise _meshcore.MeshcoreError(
                _meshcore.MeshcoreStatus.CAPACITY_EXCEEDED,
                "Affine clipping has no original remaining work or scratch.",
            )
        preparation.clip_calls += 1
        supports, labels, work = _meshcore.clip_reference_triangle_exact(
            planes,
            maximum_scratch_bytes=memory,
            maximum_work_units=remaining.remaining_work_units,
        )
        preparation.clip_work += work
        if supports.shape[0] < 3:
            return Fraction(0)

        polygon = []
        for first_support, second_support in supports:
            if not (0 <= first_support < 6 and 0 <= second_support < 6):
                raise ValueError(
                    "Native affine clipping names an absent exact halfplane."
                )
            a, b = planes[int(first_support)], planes[int(second_support)]
            construction = preparation.exact_actions(
                (a[:2], b[:2]),
                ((a[2],), (b[2],)),
            )
            if construction.actions is None:
                raise ValueError(
                    "Native affine boundary supports must reconstruct a unique exact point."
                )
            point = construction.actions[0][0], construction.actions[1][0]

            def point_validation_operation() -> tuple[None, int]:
                for plane in planes:
                    if plane[0] * point[0] + plane[1] * point[1] > plane[2]:
                        raise ValueError(
                            "Native affine boundary leaves an original exact halfplane."
                        )
                return None, 4 * len(planes)

            preparation.stage(
                4 * len(planes),
                4,
                3 * preparation.bits((point, planes)) + 32,
                point_validation_operation,
            )
            polygon.append(point)

        def reconstruction_operation() -> tuple[Fraction, int]:
            actual = 0
            for slot, label in enumerate(labels):
                if not 0 <= label < 6:
                    raise ValueError(
                        "Native affine boundary names an absent original edge."
                    )
                plane = planes[int(label)]
                for point in (
                    polygon[slot],
                    polygon[(slot + 1) % len(polygon)],
                ):
                    actual += 4
                    if plane[0] * point[0] + plane[1] * point[1] != plane[2]:
                        raise ValueError(
                            "Native affine edge lost its original exact support."
                        )
            area = _signed_measure(tuple(polygon))
            actual += 4 * len(polygon) + 1
            if area <= 0 or len(set(polygon)) != len(polygon):
                raise ValueError(
                    "Native affine intersection must have distinct, positively oriented vertices."
                )
            return area * abs(determinant), actual + 1

        # Reconstructed vertices and source planes are already owned above.
        # Edge checks and shoelace accumulation stream over already owned
        # vertices. Two products, their difference, one accumulator, and the
        # final determinant product require at most six extra rational terms.
        return preparation.stage(
            80,
            6,
            4 * preparation.bits((tuple(polygon), planes, determinant)) + 64,
            reconstruction_operation,
        )


def _sqrt_lower(value: Fraction, /) -> float:
    """Greatest practical binary64 lower bound of an exact nonnegative root."""
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


def prove_affine_chart_chain(
    proxies: list[tuple[np.ndarray, float, float]],
    cover: SourceBoundaryChartCover,
    limits: MeshCertificateLimits,
    tolerance: float,
) -> tuple[float | None, SourceBoundaryChartCover, float | None]:
    """Prove that the affine mesh proxies and source chart chain tile one PLC.

    Each proxy carries the certified Bernstein upper bound of its actual map
    minus the affine map through its binary64 corners, so corner rounding is
    included. A proved tiling returns the outward-rounded two-sided bound
    ``cover + proxy`` deviation; ``None`` means the chain was not proved.
    """
    from ._mesh_certificates import _candidate_pairs, SourceBoundaryChartCover

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

    # A certified cover built from these exact affine proxy rows already owns
    # its completeness and nonoverlap proof. Byte-identical ordered rows need
    # no second all-pairs clipping pass merely to rediscover that same chain.
    if len(target_rows) == len(source_rows) and all(
        np.array_equal(target, source)
        for target, source in zip(target_rows, source_rows, strict=True)
    ):
        return maximum_deviation, cover, None

    def boxes(rows: tuple[np.ndarray, ...], /) -> tuple[np.ndarray, np.ndarray]:
        lower = np.asarray([np.min(row, axis=0) for row in rows])
        upper = np.asarray([np.max(row, axis=0) for row in rows])
        return np.nextafter(lower, -np.inf), np.nextafter(upper, np.inf)

    capacity = min(
        limits.maximum_candidate_pairs,
        limits.maximum_distance_evaluations,
    )
    target_boxes, source_boxes = boxes(target_rows), boxes(source_rows)
    target_first, target_second, target_exhausted = _candidate_pairs(
        *target_boxes,
        capacity,
    )
    remaining = capacity - target_first.size
    source_first, source_second, source_exhausted = _candidate_pairs(
        *source_boxes,
        max(remaining, 0),
    )
    remaining -= source_first.size
    cross_first, cross_second, cross_exhausted = _candidate_pairs(
        *target_boxes,
        max(remaining, 0),
        other=source_boxes,
    )
    if target_exhausted or source_exhausted or cross_exhausted:
        return None, cover, None
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
    requested = len(target_pairs) + len(source_pairs) + len(cross_pairs)
    native = _meshcore.NativeExecutionBudget(
        max_work=limits.maximum_work_units,
        max_geometry_queries=limits.maximum_distance_evaluations,
        max_cavity_cells=0,
        max_scratch_bytes=limits.maximum_scratch_bytes,
        max_wall_seconds=math.inf,
    )
    complete = True
    separation = None
    with native:
        initial = native.remaining().remaining_work_units
        preparation = _Preparation(native)
        preparation.stage(
            requested,
            0,
            1,
            lambda: (None, requested),
        )
        all_rows = (*target_rows, *source_rows)
        scalar_count = sum(row.size for row in all_rows)

        def point_bits_operation() -> tuple[int, int]:
            bits = max(
                (
                    _binary64_fraction_bits(float(value))
                    for row in all_rows
                    for value in row.flat
                ),
                default=1,
            )
            return bits, scalar_count

        point_bits = preparation.stage(scalar_count, 1, 64, point_bits_operation)

        def exact_points(row: np.ndarray, /) -> _Triangle:
            def operation() -> tuple[_Triangle, int]:
                return _fraction_points(row), row.size

            return preparation.stage(row.size, row.size, point_bits, operation)

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

        def exact_record(row: np.ndarray, /) -> tuple[_Triangle, _PlaneKey | None]:
            return records[id(row)]

        if any(
            plane is None
            for records_ in (target_records, source_records)
            for _, plane in records_
        ):
            complete = False

        def exact_overlap(
            first_row: np.ndarray, second_row: np.ndarray, /
        ) -> bool | None:
            first, first_plane = exact_record(first_row)
            second, second_plane = exact_record(second_row)
            if first_plane is None or second_plane is None:
                return None
            if first_plane[0] != second_plane[0]:
                return False
            with preparation.budget.temporary_scope():
                return (
                    _intersection(
                        project(first, first_plane[1]),
                        project(second, first_plane[1]),
                        preparation,
                    )
                    != 0
                )

        if complete and math.isfinite(maximum_deviation):
            separated = True
            closest_lower = math.inf
            for source_row in source_rows:
                with preparation.budget.temporary_scope():
                    target, _ = exact_record(target_rows[0])
                    source, plane = exact_record(source_row)
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

        if complete:
            for chain, pairs in (
                (target_rows, target_pairs),
                (source_rows, source_pairs),
            ):
                for first, second in pairs:
                    overlap = exact_overlap(chain[first], chain[second])
                    if overlap is None or overlap:
                        complete = False
                        break
                if not complete:
                    break

        if complete:
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
                        target, target_plane = exact_record(target_rows[first])
                        source, source_plane = exact_record(source_rows[second])
                        if target_plane is None or source_plane is None:
                            complete = False
                            break
                        area = (
                            Fraction(0)
                            if target_plane[0] != source_plane[0]
                            else _intersection(
                                project(target, target_plane[1]),
                                project(source, target_plane[1]),
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
                    for rows, areas in (
                        (target_rows, target_area),
                        (source_rows, source_area),
                    ):
                        for row, area in zip(rows, areas, strict=True):
                            points, plane = exact_record(row)
                            if plane is None:
                                complete = False
                                break
                            with preparation.budget.temporary_scope():
                                expected = preparation.stage(
                                    14,
                                    12,
                                    8 * preparation.bits(points) + 64,
                                    lambda: (
                                        abs(_signed_measure(_project(points, plane[1]))),
                                        14,
                                    ),
                                )
                            if area != expected:
                                complete = False
                                break
                        if not complete:
                            break
        consumed = initial - native.remaining().remaining_work_units
    evidence = native.evidence
    if evidence is None:
        raise RuntimeError("An affine clipping scope must retain its native evidence.")
    resources = (
        *cover.resource_counts,
        ("affine_chain_total_work", consumed, limits.maximum_work_units),
        (
            "affine_chain_coefficient_work",
            preparation.budget.work_units,
            limits.maximum_work_units,
        ),
        (
            "affine_chain_coefficient_bytes_upper",
            preparation.budget.peak_bytes_upper,
            limits.maximum_scratch_bytes,
        ),
        (
            "affine_chain_retained_coefficient_bytes",
            preparation.budget.retained_basis_bytes,
            limits.maximum_scratch_bytes,
        ),
        (
            "affine_chain_native_peak_bytes",
            int(evidence.memory_evidence[2]),
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
    proven_cover = SourceBoundaryChartCover(
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
