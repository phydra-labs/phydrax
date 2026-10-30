#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Continuous interval bounds for fitted native B-Rep intersection branches."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._validation import positive_finite_float
from ._intersection_curve import IntersectionCurve
from ._patches import AbstractSurfacePatch, BSplineCurve


@dataclass(frozen=True, slots=True)
class CoupledApproximationEvidence:
    """Outward whole-domain error bounds for a fitted curve and both p-curves."""

    source_branch_id: str
    approximation_id: str
    distance_bound: float
    parameter_bound: float
    first_correspondence_bound: float
    second_correspondence_bound: float
    cells: int
    interval_evidence: np.ndarray
    endpoint_root_bounds: tuple[float, float]
    evidence_id: str = field(init=False)

    def __post_init__(self) -> None:
        values = (
            self.distance_bound,
            self.parameter_bound,
            self.first_correspondence_bound,
            self.second_correspondence_bound,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(
                "Coupled approximation bounds must be finite and nonnegative."
            )
        if (
            isinstance(self.cells, bool)
            or not isinstance(self.cells, int)
            or self.cells < 1
        ):
            raise ValueError("Coupled approximation cells must be a positive integer.")
        rows = np.asarray(self.interval_evidence, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != 6 or not np.all(np.isfinite(rows)):
            raise ValueError(
                "Coupled approximation interval evidence must have finite six-column rows."
            )
        roots = tuple(float(value) for value in self.endpoint_root_bounds)
        if len(roots) != 2 or any(
            not math.isfinite(value) or value < 0.0 for value in roots
        ):
            raise ValueError(
                "Coupled approximation root bounds must be two finite nonnegative values."
            )
        immutable = np.frombuffer(rows.tobytes(), dtype=np.float64).reshape(rows.shape)
        object.__setattr__(self, "interval_evidence", immutable)
        object.__setattr__(self, "endpoint_root_bounds", roots)
        object.__setattr__(
            self,
            "evidence_id",
            canonical_fingerprint(
                {
                    "kind": "coupled-intersection-approximation",
                    "source_branch_id": self.source_branch_id,
                    "approximation_id": self.approximation_id,
                    "bounds": values,
                    "cells": self.cells,
                    "interval_evidence": immutable,
                    "endpoint_root_bounds": roots,
                }
            ),
        )


class ApproximationCertificationError(ValueError):
    """A continuous approximation certificate cannot meet its declared bound."""

    def __init__(self, message: str, /, *, interval: tuple[float, float]) -> None:
        self.interval = interval
        super().__init__(message)


def _identity(curves: tuple[BSplineCurve, BSplineCurve, BSplineCurve], /) -> str:
    return canonical_fingerprint(
        {
            "kind": "coupled-intersection-fit",
            "curves": tuple(curve.identity_payload() for curve in curves),
        }
    )


def _piecewise_cubic(samples: np.ndarray, /) -> BSplineCurve:
    if samples.ndim != 3 or samples.shape[0] < 1 or samples.shape[1] < 4:
        raise ValueError(
            "Intersection fitting samples must have shape (charts, samples>=4, dimension)."
        )
    chart_count, sample_count, dimension = samples.shape
    parameter = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
    first_basis = 3.0 * parameter * (1.0 - parameter) ** 2
    second_basis = 3.0 * parameter**2 * (1.0 - parameter)
    design = np.column_stack((first_basis[1:-1], second_basis[1:-1]))
    controls: list[np.ndarray] = []
    for chart in range(chart_count):
        values = samples[chart]
        start, end = values[0], values[-1]
        fixed = ((1.0 - parameter) ** 3)[:, None] * start + parameter[:, None] ** 3 * end
        interior = np.linalg.lstsq(design, values[1:-1] - fixed[1:-1], rcond=None)[0]
        segment = np.stack((start, interior[0], interior[1], end))
        controls.extend(segment if chart == 0 else segment[1:])
    knots = [0.0] * 4
    for chart in range(1, chart_count):
        knots.extend((float(chart),) * 3)
    knots.extend((float(chart_count),) * 4)
    control_values = np.asarray(controls, dtype=np.float64).reshape(-1, dimension)
    return BSplineCurve(
        control_values,
        np.ones((control_values.shape[0],), dtype=np.float64),
        np.asarray(knots, dtype=np.float64),
        3,
    )


def propose_intersection_approximation(
    source: IntersectionCurve,
    /,
    *,
    endpoint_points: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[BSplineCurve, BSplineCurve, BSplineCurve]:
    """Build deterministic per-source-chart cubic proposals; samples do not certify."""
    if not source.fully_certified:
        raise ValueError("Intersection fitting requires a fully certified source branch.")
    sample_fractions = np.linspace(0.0, 1.0, 17, dtype=np.float64)
    first_samples: list[np.ndarray] = []
    second_samples: list[np.ndarray] = []
    spatial_samples: list[np.ndarray] = []
    for chart in range(source.num_charts):
        parameters = jnp.asarray(chart + sample_fractions, dtype=jnp.float64)
        points = source.evaluate_coupled(parameters)
        spatial_samples.append(np.asarray(points.point, dtype=np.float64))
        first_samples.append(np.asarray(points.first_parameters, dtype=np.float64))
        second_samples.append(np.asarray(points.second_parameters, dtype=np.float64))
    spatial = np.stack(spatial_samples)
    first = np.stack(first_samples)
    second = np.stack(second_samples)
    if endpoint_points is not None:
        endpoints = np.asarray(endpoint_points, dtype=np.float64)
        if endpoints.shape != (2, 3) or not np.all(np.isfinite(endpoints)):
            raise ValueError(
                "Intersection fit endpoints must be finite (2, 3) coordinates."
            )
        if source.closed and not np.array_equal(endpoints[0], endpoints[1]):
            raise ValueError("A closed intersection fit must retain one shared endpoint.")
        spatial[0, 0] = endpoints[0]
        spatial[-1, -1] = endpoints[1]
    elif source.closed:
        spatial[-1, -1] = spatial[0, 0]
    return (
        _piecewise_cubic(spatial),
        _piecewise_cubic(first),
        _piecewise_cubic(second),
    )


def _box_difference_upper(first: np.ndarray, second: np.ndarray, /) -> float:
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if a.ndim != 2 or b.shape != a.shape or a.shape[0] != 2:
        raise ValueError("Approximation bounds require aligned (2, dimension) boxes.")
    lower = np.nextafter(a[0] - b[1], -np.inf)
    upper = np.nextafter(a[1] - b[0], np.inf)
    magnitude = np.maximum(np.abs(lower), np.abs(upper))
    squared = np.float64(0.0)
    for value in magnitude:
        term = np.nextafter(np.float64(value * value), np.inf)
        squared = np.nextafter(squared + term, np.inf)
    return float(np.nextafter(np.sqrt(squared), np.inf))


def _parameter_enclosures(
    source: IntersectionCurve, first: float, last: float, /
) -> np.ndarray:
    boxes = np.asarray(source.parameter_enclosures(first, last), dtype=np.float64)
    if boxes.ndim != 3 or boxes.shape[1:] != (2, 4):
        raise ValueError(
            "The source intersection branch returned malformed parameter enclosures."
        )
    return boxes


def _parameter_hull(boxes: np.ndarray, /) -> np.ndarray:
    return np.stack((np.min(boxes[:, 0], axis=0), np.max(boxes[:, 1], axis=0)))


def source_point_box(
    source: IntersectionCurve, first: float, last: float, /
) -> np.ndarray:
    """Enclose source branch points over ``[first, last]`` on both carriers.

    Each certified parameter enclosure is lifted through both source surfaces;
    the true point lies in both lifts, so their intersection is a sound and
    subrange-convergent enclosure.
    """
    if first == last and first == round(first):
        # Branch nodes carry their own certified root boxes.
        return np.asarray(source.bounding_box(first, last), dtype=np.float64)
    return _lifted_point_box(source, _parameter_enclosures(source, first, last))


def _lifted_point_box(source: IntersectionCurve, boxes: np.ndarray, /) -> np.ndarray:
    lower = np.full((3,), np.inf, dtype=np.float64)
    upper = np.full((3,), -np.inf, dtype=np.float64)
    for box in boxes:
        first_lift = np.asarray(
            source.first.patch.bounding_box(
                _positive_parameter_box(box[:, :2], source.first.patch)
            ),
            dtype=np.float64,
        )
        second_lift = np.asarray(
            source.second.patch.bounding_box(
                _positive_parameter_box(box[:, 2:], source.second.patch)
            ),
            dtype=np.float64,
        )
        box_lower = np.maximum(first_lift[0], second_lift[0])
        box_upper = np.minimum(first_lift[1], second_lift[1])
        if np.any(box_lower > box_upper):
            raise ValueError(
                "The source intersection branch lifts to disjoint surface enclosures."
            )
        lower = np.minimum(lower, box_lower)
        upper = np.maximum(upper, box_upper)
    return np.stack((lower, upper))


def _positive_parameter_box(
    box: np.ndarray, patch: AbstractSurfacePatch, /
) -> np.ndarray:
    result = np.asarray(box, dtype=np.float64).copy()
    domain = patch.parameter_domain
    for axis in range(2):
        if result[0, axis] < result[1, axis]:
            continue
        value = result[0, axis]
        lower = float(np.nextafter(value, -np.inf))
        upper = float(np.nextafter(value, np.inf))
        if domain is not None:
            lower = max(lower, domain[0][axis])
            upper = min(upper, domain[1][axis])
        if not lower < upper:
            raise ValueError(
                "A fitted p-curve lift has no positive-width surface enclosure."
            )
        result[:, axis] = (lower, upper)
    return result


def _cell_bounds(
    source: IntersectionCurve,
    curve: BSplineCurve,
    first_pcurve: BSplineCurve,
    second_pcurve: BSplineCurve,
    first: float,
    last: float,
    /,
) -> tuple[float, float, float, float]:
    enclosures = _parameter_enclosures(source, first, last)
    source_box = _lifted_point_box(source, enclosures)
    curve_box = curve.bounding_box(first, last)
    distance = _box_difference_upper(source_box, curve_box)
    parameters = _parameter_hull(enclosures)
    first_fit_box = first_pcurve.bounding_box(first, last)
    second_fit_box = second_pcurve.bounding_box(first, last)
    first_parameter_bound = _box_difference_upper(parameters[:, :2], first_fit_box)
    second_parameter_bound = _box_difference_upper(parameters[:, 2:], second_fit_box)
    first_lift = source.first.patch.bounding_box(
        _positive_parameter_box(first_fit_box, source.first.patch)
    )
    second_lift = source.second.patch.bounding_box(
        _positive_parameter_box(second_fit_box, source.second.patch)
    )
    return (
        max(distance, _box_difference_upper(first_lift, curve_box)),
        max(distance, _box_difference_upper(second_lift, curve_box)),
        max(first_parameter_bound, second_parameter_bound),
        distance,
    )


def _endpoint_root_bound(
    source: IntersectionCurve, curve: BSplineCurve, parameter: float, /
) -> float:
    source_box = source_point_box(source, parameter, parameter)
    fit_box = curve.bounding_box(parameter, parameter)
    return _box_difference_upper(source_box, fit_box)


def certify_coupled_approximation(
    source: IntersectionCurve,
    curve: BSplineCurve,
    first_pcurve: BSplineCurve,
    second_pcurve: BSplineCurve,
    /,
    *,
    distance_tolerance: float,
    parameter_tolerance: float,
) -> CoupledApproximationEvidence:
    """Certify 3D, UV, and both surface-lift errors over every closed parameter cell."""
    if not source.fully_certified:
        raise ValueError(
            "Approximation certification requires a fully certified source branch."
        )
    fits = (curve, first_pcurve, second_pcurve)
    domain = (0.0, float(source.num_charts))
    if any(fit.parameter_domain != domain for fit in fits):
        raise ValueError(
            "Fitted intersection curves must retain the source parameter domain."
        )
    if tuple(fit.ambient_dimension for fit in fits) != (3, 2, 2):
        raise ValueError(
            "Fitted intersection curves must have ambient dimensions (3, 2, 2)."
        )
    distance_limit = positive_finite_float(distance_tolerance, "distance_tolerance")
    parameter_limit = positive_finite_float(parameter_tolerance, "parameter_tolerance")
    pending = [
        (float(chart), float(chart + 1), 0)
        for chart in reversed(range(source.num_charts))
    ]
    records: list[tuple[float, ...]] = []
    while pending:
        first, last, depth = pending.pop()
        first_lift, second_lift, parameter_bound, distance_bound = _cell_bounds(
            source, curve, first_pcurve, second_pcurve, first, last
        )
        if (
            distance_bound <= distance_limit
            and parameter_bound <= parameter_limit
            and first_lift <= distance_limit
            and second_lift <= distance_limit
        ):
            records.append(
                (first, last, distance_bound, parameter_bound, first_lift, second_lift)
            )
            continue
        middle = first + 0.5 * (last - first)
        if not first < middle < last:
            raise ApproximationCertificationError(
                "Continuous coupled approximation bounds remain above tolerance at parameter resolution.",
                interval=(first, last),
            )
        pending.append((middle, last, depth + 1))
        pending.append((first, middle, depth + 1))
    endpoint_bounds = (
        _endpoint_root_bound(source, curve, 0.0),
        _endpoint_root_bound(source, curve, domain[1]),
    )
    maxima = np.max(np.asarray(records, dtype=np.float64)[:, 2:], axis=0)
    interval_rows = np.asarray(records, dtype=np.float64)
    source_identity = source.branch_id
    approximation_identity = _identity(fits)
    return CoupledApproximationEvidence(
        source_identity,
        approximation_identity,
        float(maxima[0]),
        float(maxima[1]),
        float(maxima[2]),
        float(maxima[3]),
        len(records),
        interval_rows,
        endpoint_bounds,
    )


__all__ = [
    "ApproximationCertificationError",
    "CoupledApproximationEvidence",
    "certify_coupled_approximation",
    "propose_intersection_approximation",
]
