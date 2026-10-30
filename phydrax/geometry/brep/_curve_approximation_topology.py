#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Continuous fitted-branch embedding and trim-separation certificates."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ..._fingerprint import canonical_fingerprint
from ...typing import parse
from ._curve_approximation_bounds import CoupledApproximationEvidence
from ._intersection_curve import IntersectionCurve, IntersectionSide
from ._patches import AbstractCurve, BSplineCurve


class BranchTopologyCertificationError(ValueError):
    """A fitted branch cannot be separated or locally embedded continuously."""

    def __init__(self, message: str, /, *, interval: tuple[float, float]) -> None:
        self.interval = interval
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class BranchApproximationTopologyEvidence:
    """Whole-domain boxes and local-axis witnesses for a fitted source branch."""

    source_branch_id: str
    approximation_id: str
    parameter_cells: np.ndarray
    spatial_homotopy_boxes: np.ndarray
    first_homotopy_boxes: np.ndarray
    second_homotopy_boxes: np.ndarray
    local_axes: np.ndarray
    separated_pairs: int
    minimum_separation_lower_bound: float
    evidence_id: str = field(init=False)

    def __post_init__(self) -> None:
        cells = np.asarray(self.parameter_cells, dtype=np.float64)
        spatial = np.asarray(self.spatial_homotopy_boxes, dtype=np.float64)
        first = np.asarray(self.first_homotopy_boxes, dtype=np.float64)
        second = np.asarray(self.second_homotopy_boxes, dtype=np.float64)
        axes = np.asarray(self.local_axes, dtype=np.int64)
        count = cells.shape[0]
        if (
            cells.ndim != 2
            or cells.shape[1] != 2
            or count == 0
            or spatial.shape != (count, 2, 3)
            or first.shape != (count, 2, 2)
            or second.shape != (count, 2, 2)
            or axes.shape != (count, 3)
            or any(not np.all(np.isfinite(value)) for value in (cells, spatial, first, second))
            or np.any(cells[:, 0] >= cells[:, 1])
            or isinstance(self.separated_pairs, bool)
            or self.separated_pairs < 0
            or not math.isfinite(self.minimum_separation_lower_bound)
            or self.minimum_separation_lower_bound < 0.0
        ):
            raise ValueError("Branch topology evidence has malformed interval data.")
        immutable = tuple(
            np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
            for value in (cells, spatial, first, second, axes)
        )
        for name, value in zip(
            (
                "parameter_cells",
                "spatial_homotopy_boxes",
                "first_homotopy_boxes",
                "second_homotopy_boxes",
                "local_axes",
            ),
            immutable,
            strict=True,
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "evidence_id",
            canonical_fingerprint(
                {
                    "kind": "branch-approximation-topology",
                    "source_branch_id": self.source_branch_id,
                    "approximation_id": self.approximation_id,
                    "parameter_cells": immutable[0],
                    "spatial_boxes": immutable[1],
                    "first_boxes": immutable[2],
                    "second_boxes": immutable[3],
                    "local_axes": immutable[4],
                    "separated_pairs": self.separated_pairs,
                    "minimum_separation_lower_bound": self.minimum_separation_lower_bound,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class BranchTrimSeparationEvidence:
    """Positive parameter-space separation of a fitted branch from one trim."""

    topology_evidence_id: str
    trim_identity: str
    side: IntersectionSide
    separation_lower_bound: float
    interval_evidence: np.ndarray
    evidence_id: str = field(init=False)

    def __post_init__(self) -> None:
        value = float(self.separation_lower_bound)
        rows = np.asarray(self.interval_evidence, dtype=np.float64)
        if (
            not math.isfinite(value)
            or value <= 0.0
            or rows.ndim != 2
            or rows.shape[1] != 4
            or not np.all(np.isfinite(rows))
        ):
            raise ValueError("Branch trim-separation evidence must be finite and strictly positive.")
        immutable = np.frombuffer(rows.tobytes(), dtype=np.float64).reshape(rows.shape)
        object.__setattr__(self, "separation_lower_bound", value)
        object.__setattr__(self, "interval_evidence", immutable)
        object.__setattr__(
            self,
            "evidence_id",
            canonical_fingerprint(
                {
                    "kind": "branch-trim-separation",
                    "topology_evidence_id": self.topology_evidence_id,
                    "trim_identity": self.trim_identity,
                    "side": self.side,
                    "separation_lower_bound": value,
                    "interval_evidence": immutable,
                }
            ),
        )


def _union(first: np.ndarray, second: np.ndarray, /) -> np.ndarray:
    return np.stack((np.minimum(first[0], second[0]), np.maximum(first[1], second[1])))


def _overlap(first: np.ndarray, second: np.ndarray, /) -> bool:
    return bool(np.all(first[0] <= second[1]) and np.all(second[0] <= first[1]))


def _separation(first: np.ndarray, second: np.ndarray, /) -> float:
    gaps = np.maximum(np.maximum(first[0] - second[1], second[0] - first[1]), 0.0)
    return float(np.nextafter(np.max(gaps), -np.inf)) if np.any(gaps > 0.0) else 0.0


def _source_parameters(source: IntersectionCurve, first: float, last: float, /) -> np.ndarray:
    boxes = np.asarray(source.parameter_enclosures(first, last), dtype=np.float64)
    return np.stack((np.min(boxes[:, 0], axis=0), np.max(boxes[:, 1], axis=0)))


def _homotopy_boxes(
    source: IntersectionCurve,
    curve: BSplineCurve,
    first_pcurve: BSplineCurve,
    second_pcurve: BSplineCurve,
    first: float,
    last: float,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spatial = _union(source.bounding_box(first, last), curve.bounding_box(first, last))
    parameters = _source_parameters(source, first, last)
    first_fit = first_pcurve.bounding_box(first, last)
    second_fit = second_pcurve.bounding_box(first, last)
    return (
        spatial,
        _union(parameters[:, :2], first_fit),
        _union(parameters[:, 2:], second_fit),
    )


def _monotone_derivative(curve: BSplineCurve, first: float, last: float, /) -> tuple[int, int] | None:
    lower, upper = curve.derivative_bounds(first, last)
    for axis in range(curve.ambient_dimension):
        if lower[axis] > 0.0:
            return axis, 1
        if upper[axis] < 0.0:
            return axis, -1
    return None


def _cell_axes(
    source: IntersectionCurve,
    curve: BSplineCurve,
    first_pcurve: BSplineCurve,
    second_pcurve: BSplineCurve,
    first: float,
    last: float,
    /,
) -> tuple[int, int, int] | None:
    if not source.is_c1_on(first, last):
        return None
    chart = min(int(math.floor(first + 0.5 * (last - first))), source.num_charts - 1)
    source_axis = int(source.chart_axes[chart])
    start = np.asarray(source.chart_start[chart], dtype=np.float64)
    end = np.asarray(source.chart_end[chart], dtype=np.float64)
    sign = 1 if end[source_axis] > start[source_axis] else -1
    pcurve, uv_axis = (first_pcurve, source_axis) if source_axis < 2 else (second_pcurve, source_axis - 2)
    lower, upper = pcurve.derivative_bounds(first, last)
    if sign > 0 and lower[uv_axis] <= 0.0 or sign < 0 and upper[uv_axis] >= 0.0:
        return None
    spatial_axis = _monotone_derivative(curve, first, last)
    if spatial_axis is None:
        return None
    return source_axis, uv_axis + 1 if source_axis < 2 else uv_axis + 3, spatial_axis[0] + 1


def _split(interval: tuple[float, float], /) -> tuple[tuple[float, float], tuple[float, float]] | None:
    first, last = interval
    middle = first + 0.5 * (last - first)
    if not first < middle < last:
        return None
    return (first, middle), (middle, last)


def _adjacent(first: tuple[float, float], second: tuple[float, float], /) -> bool:
    return first[1] == second[0] or second[1] == first[0]


def _approximation_identity(
    curve: BSplineCurve,
    first_pcurve: BSplineCurve,
    second_pcurve: BSplineCurve,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "coupled-intersection-fit",
            "curves": tuple(
                value.identity_payload() for value in (curve, first_pcurve, second_pcurve)
            ),
        }
    )


def certify_branch_approximation_topology(
    source: IntersectionCurve,
    curve: BSplineCurve,
    first_pcurve: BSplineCurve,
    second_pcurve: BSplineCurve,
    approximation: CoupledApproximationEvidence,
    /,
) -> BranchApproximationTopologyEvidence:
    """Certify fitted local graph axes and continuous nonlocal branch separation."""
    if not source.fully_certified or source.branch_id != approximation.source_branch_id:
        raise ValueError("Branch topology evidence must bind the same certified source branch.")
    fit_identity = _approximation_identity(curve, first_pcurve, second_pcurve)
    if fit_identity != approximation.approximation_id:
        raise ValueError("Branch topology evidence must bind the certified fitted carriers.")
    if source.start_kind in ("singular", "uncertified") or source.end_kind in ("singular", "uncertified"):
        raise BranchTopologyCertificationError(
            "Fitted branch topology requires regular or closed exact source endpoints.",
            interval=(0.0, float(source.num_charts)),
        )
    intervals = [tuple(float(value) for value in row[:2]) for row in approximation.interval_evidence]
    while True:
        intervals.sort()
        records: list[tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[int, int, int] | None]] = []
        for first, last in intervals:
            boxes = _homotopy_boxes(
                source, curve, first_pcurve, second_pcurve, first, last
            )
            axes = _cell_axes(
                source, curve, first_pcurve, second_pcurve, first, last
            )
            records.append((boxes, axes))
        offenders: set[int] = {
            index for index, (_, axes) in enumerate(records) if axes is None
        }
        minimum_separation = math.inf
        separated_pairs = 0
        for first_index, first_interval in enumerate(intervals):
            for second_index in range(first_index + 1, len(intervals)):
                second_interval = intervals[second_index]
                if _adjacent(first_interval, second_interval) or (
                    source.closed and first_index == 0 and second_index == len(intervals) - 1
                ):
                    continue
                first_boxes, _ = records[first_index]
                second_boxes, _ = records[second_index]
                spaces_separate = tuple(
                    _separation(first_box, second_box)
                    for first_box, second_box in zip(first_boxes, second_boxes, strict=True)
                )
                if all(value > 0.0 for value in spaces_separate):
                    separated_pairs += 1
                    minimum_separation = min(minimum_separation, *spaces_separate)
                else:
                    offenders.update((first_index, second_index))
        if not offenders:
            break
        selected = max(offenders, key=lambda index: intervals[index][1] - intervals[index][0])
        split = _split(intervals[selected])
        if split is None:
            raise BranchTopologyCertificationError(
                "Fitted branch topology remains unresolved at parameter resolution.",
                interval=intervals[selected],
            )
        intervals[selected : selected + 1] = list(split)
    endpoint_boxes = (
        source.bounding_box(0.0, 0.0),
        source.bounding_box(float(source.num_charts), float(source.num_charts)),
    )
    for parameter, root_box in zip((0.0, float(source.num_charts)), endpoint_boxes, strict=True):
        if not _overlap(np.asarray(curve.bounding_box(parameter, parameter)), np.asarray(root_box)):
            raise BranchTopologyCertificationError(
                "A fitted branch endpoint lost its exact source-root neighborhood.",
                interval=(parameter, parameter),
            )
    spatial = np.asarray([record[0][0] for record in records], dtype=np.float64)
    first_boxes = np.asarray([record[0][1] for record in records], dtype=np.float64)
    second_boxes = np.asarray([record[0][2] for record in records], dtype=np.float64)
    axes = np.asarray([record[1] for record in records], dtype=np.int64)
    return BranchApproximationTopologyEvidence(
        source.branch_id,
        fit_identity,
        np.asarray(intervals, dtype=np.float64),
        spatial,
        first_boxes,
        second_boxes,
        axes,
        separated_pairs,
        0.0 if not math.isfinite(minimum_separation) else minimum_separation,
    )


def certify_branch_trim_separation(
    evidence: BranchApproximationTopologyEvidence,
    trim: AbstractCurve,
    side: IntersectionSide,
    /,
    *,
    first: float,
    last: float,
) -> BranchTrimSeparationEvidence:
    """Certify a fitted p-curve homotopy avoids one unrelated trim segment."""
    side_ = parse(side, IntersectionSide, "side")
    if not math.isfinite(first) or not math.isfinite(last) or not first < last:
        raise ValueError("Trim-separation range must be finite and increasing.")
    trim_box = np.asarray(trim.bounding_box(first, last), dtype=np.float64)
    if trim_box.shape != (2, 2) or not np.all(np.isfinite(trim_box)):
        raise ValueError("Trim-separation requires a finite two-dimensional trim enclosure.")
    boxes = evidence.first_homotopy_boxes if side_ == "first" else evidence.second_homotopy_boxes
    rows: list[tuple[float, float, float, float]] = []
    separation = math.inf
    for interval, box in zip(evidence.parameter_cells, boxes, strict=True):
        lower = _separation(box, trim_box)
        if lower <= 0.0:
            raise BranchTopologyCertificationError(
                "A fitted branch p-curve is not separated from the declared face trim.",
                interval=(float(interval[0]), float(interval[1])),
            )
        separation = min(separation, lower)
        rows.append((float(interval[0]), float(interval[1]), first, last))
    return BranchTrimSeparationEvidence(
        evidence.evidence_id,
        canonical_fingerprint(
            {
                "kind": "trim-carrier",
                "type": type(trim).__qualname__,
                "range": (first, last),
                "box": trim_box,
            }
        ),
        side_,
        separation,
        np.asarray(rows, dtype=np.float64),
    )


__all__ = [
    "BranchApproximationTopologyEvidence",
    "BranchTopologyCertificationError",
    "BranchTrimSeparationEvidence",
    "certify_branch_approximation_topology",
    "certify_branch_trim_separation",
]
