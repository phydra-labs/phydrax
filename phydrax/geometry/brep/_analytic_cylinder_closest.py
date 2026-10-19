#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Compute closed-form closest points on finite cylinder walls spanning a full turn.

Rational arithmetic preserves the dyadic query and source frame without rounding.
Root bounds follow integer square-root inequalities; angles use the interval
implementation's four-ulp elementary-function error bound. An axial query has a
closest circular fiber with NON_ISOLATED status instead of an angular parameter.
"""

from __future__ import annotations

from fractions import Fraction
from math import isqrt

import numpy as np

from ..._meshcore import NativeBatchTicket
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked
from .._interval_enclosure import interval_multiply, interval_subtract, monotone
from .._rational import fraction_endpoint
from ._analytic_source_regions import (
    AnalyticClosestBatch,
    complete_cylinder_region,
    CylinderSourceRegion,
)
from ._model import BRepModel
from ._patches import LineCurve
from ._projection_contracts import BRepProjectionStatus
from ._query_types import BRepQueryBudget, Tolerances


def sqrt_bounds(value: Fraction, /) -> tuple[Fraction, Fraction]:
    """Enclose a rational square root by 60 relative binary digits.

    For a positive value, floor(sqrt(floor(value * 2**(2*k)))) is the
    lower integer endpoint, computed with exact arithmetic. The neighboring
    integer is the upper endpoint; a square has identical endpoints.
    Scaling avoids float underflow.
    """
    if value < 0:
        raise ValueError(f"sqrt_bounds.value must be nonnegative; got {value!r}")
    if value == 0:
        return Fraction(0), Fraction(0)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    shift = 60 - exponent // 2
    scale = Fraction(2) ** shift
    square = value * scale * scale
    root = isqrt(square.numerator // square.denominator)
    lower = Fraction(root) / scale
    upper = lower if lower * lower == value else Fraction(root + 1) / scale
    return lower, upper


def _rational_interval(value: Fraction, /) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(
            fraction_endpoint(value, upper=False, saturate=True), dtype=np.float64
        ),
        np.asarray(fraction_endpoint(value, upper=True, saturate=True), dtype=np.float64),
    )


def angle_bounds(x: Fraction, y: Fraction, /) -> tuple[float, float]:
    """Enclose the angle in [0, 2*pi] without a branch-cut hull."""
    if y == 0 and x > 0:
        return 0.0, 0.0
    atan = monotone(np.arctan, True, 4)
    pi = interval_multiply(
        atan(_rational_interval(Fraction(1))), _rational_interval(Fraction(4))
    )
    half_pi = interval_multiply(pi, _rational_interval(Fraction(1, 2)))
    ax, ay = abs(x), abs(y)
    if ax >= ay:
        angle = atan(_rational_interval(ay / ax))
    else:
        angle = interval_subtract(half_pi, atan(_rational_interval(ax / ay)))
    if x < 0:
        angle = interval_subtract(pi, angle)
    if y < 0:
        angle = interval_subtract(
            interval_multiply(pi, _rational_interval(Fraction(2))), angle
        )
    return max(0.0, float(angle[0])), float(angle[1])


def linear_root_quotient(
    base: Fraction, coefficient: Fraction, root: tuple[Fraction, Fraction], /
) -> tuple[Fraction, Fraction]:
    first = base + coefficient / root[0]
    second = base + coefficient / root[1]
    return min(first, second), max(first, second)


def _distance_bounds(
    radial: tuple[Fraction, Fraction], radius: Fraction, axial_delta: Fraction, /
) -> tuple[float, float]:
    first, last = radial[0] - radius, radial[1] - radius
    minimum = Fraction(0) if first <= 0 <= last else min(abs(first), abs(last))
    maximum = max(abs(first), abs(last))
    lower = sqrt_bounds(minimum * minimum + axial_delta * axial_delta)[0]
    upper = sqrt_bounds(maximum * maximum + axial_delta * axial_delta)[1]
    return fraction_endpoint(lower, upper=False, saturate=True), fraction_endpoint(
        upper, upper=True, saturate=True
    )


def vector_error(
    values: np.ndarray, bounds: tuple[tuple[Fraction, Fraction], ...], /
) -> float:
    squared = Fraction(0)
    for value, (lower, upper) in zip(values, bounds, strict=True):
        represented = Fraction(float(value))
        error = max(abs(represented - lower), abs(represented - upper))
        squared += error * error
    return fraction_endpoint(sqrt_bounds(squared)[1], upper=True, saturate=True)


def _point_fields(
    origin: tuple[Fraction, ...],
    axes: tuple[tuple[Fraction, ...], ...],
    radius: Fraction,
    x: Fraction,
    y: Fraction,
    height: Fraction,
    radial: tuple[Fraction, Fraction],
    orientation: Fraction,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    midpoint = (radial[0] + radial[1]) / 2
    bounds = tuple(
        linear_root_quotient(
            origin[k] + height * axes[2][k],
            radius * (x * axes[0][k] + y * axes[1][k]),
            radial,
        )
        for k in range(3)
    )
    point = np.asarray(
        [
            fraction_endpoint(
                (lower + upper) / 2, upper=(lower + upper) < 0, saturate=True
            )
            for lower, upper in bounds
        ],
        dtype=np.float64,
    )
    first = np.asarray(
        [float(radius * (-y * axes[0][k] + x * axes[1][k]) / midpoint) for k in range(3)],
        dtype=np.float64,
    )
    second = np.asarray([float(value) for value in axes[2]], dtype=np.float64)
    cross = (
        axes[0][1] * axes[1][2] - axes[0][2] * axes[1][1],
        axes[0][2] * axes[1][0] - axes[0][0] * axes[1][2],
        axes[0][0] * axes[1][1] - axes[0][1] * axes[1][0],
    )
    handedness = sum((a * b for a, b in zip(cross, axes[2], strict=True)), Fraction(0))
    normal = np.asarray(
        [
            float(orientation * handedness * (x * axes[0][k] + y * axes[1][k]) / midpoint)
            for k in range(3)
        ],
        dtype=np.float64,
    )
    return point, normal, first, second, vector_error(point, bounds)


def _stationarity(
    query: np.ndarray,
    point: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    lower_active: bool,
    upper_active: bool,
    distance: float,
    distance_tolerance: float,
    /,
) -> float:
    """Return the normalized executed-point KKT residual for the finite wall."""
    delta = point - query
    angular = float(delta @ first)
    axial = float(delta @ second)
    if lower_active:
        axial = min(axial, 0.0)
    if upper_active:
        axial = max(axial, 0.0)
    denominator = max(
        float(np.hypot(np.linalg.norm(first), np.linalg.norm(second)))
        * max(distance, distance_tolerance),
        float(np.finfo(np.float64).tiny),
    )
    return float(np.hypot(angular, axial)) / denominator


def _wall_edges(
    model: BRepModel, lateral: int, cap_faces: tuple[int, int], /
) -> tuple[tuple[int, int], int]:
    geometry = model.geometry
    if geometry is None:
        raise ValueError("BRepModel.geometry must be present for a cylinder; got None")
    rings = tuple(
        geometry.coedge_edges[geometry.face_loops[cap][0][0]] for cap in cap_faces
    )
    seam = next(
        geometry.coedge_edges[coedge]
        for coedge in geometry.face_loops[lateral][0]
        if geometry.coedge_edges[coedge] not in rings
    )
    return (rings[0], rings[1]), seam


def _seam_parameter(
    model: BRepModel,
    edge: int,
    height: Fraction,
    origin: tuple[Fraction, ...],
    axis: tuple[Fraction, ...],
    /,
) -> float:
    geometry = model.geometry
    if geometry is None:
        raise ValueError("BRepModel.geometry must be present for a cylinder; got None")
    match geometry.curves[geometry.edge_curves[edge]]:
        case LineCurve() as curve:
            line_origin = tuple(Fraction(float(v)) for v in np.asarray(curve.origin))
            direction = tuple(Fraction(float(v)) for v in np.asarray(curve.direction))
        case _:
            raise ValueError(
                "BRepGeometry.curves must contain a LineCurve for the cylinder seam; "
                f"got {type(geometry.curves[geometry.edge_curves[edge]]).__name__}"
            )
    offset = sum(
        ((a - b) * c for a, b, c in zip(line_origin, origin, axis, strict=True)),
        Fraction(0),
    )
    speed = sum((a * b for a, b in zip(direction, axis, strict=True)), Fraction(0))
    return float((height - offset) / speed)


def _batch_arrays(count: int, face: int, /) -> dict[str, np.ndarray]:
    """Allocate the immutable-result fields with explicit host dtypes."""
    return {
        "points": np.full((count, 3), np.nan, dtype=np.float64),
        "parameters": np.full((count, 2), np.nan, dtype=np.float64),
        "distances": np.full((count,), np.inf, dtype=np.float64),
        "faces": np.full((count,), face, dtype=np.int32),
        "status": np.full(
            (count,), int(BRepProjectionStatus.NON_ISOLATED), dtype=np.int8
        ),
        "normals": np.full((count, 3), np.nan, dtype=np.float64),
        "first_derivatives": np.full((count, 3), np.nan, dtype=np.float64),
        "second_derivatives": np.full((count, 3), np.nan, dtype=np.float64),
        "stationarity": np.full((count,), np.nan, dtype=np.float64),
        "distance_lower_bounds": np.zeros((count,), dtype=np.float64),
        "unresolved": np.zeros((count,), dtype=np.bool_),
        "refinements": np.zeros((count,), dtype=np.int32),
        "parameter_bounds": np.full((count,), np.inf, dtype=np.float64),
        "edge_indices": np.full((count,), -1, dtype=np.int32),
        "edge_parameters": np.full((count,), np.nan, dtype=np.float64),
        "point_error_bounds": np.full((count,), np.inf, dtype=np.float64),
        "query_operations": np.ones((count,), dtype=np.int64),
    }


def _fill_row(
    data: dict[str, np.ndarray],
    row: int,
    query: np.ndarray,
    region: CylinderSourceRegion,
    model: BRepModel,
    origin: tuple[Fraction, ...],
    rings: tuple[int, int],
    seam: int,
    tolerances: Tolerances,
    /,
) -> None:
    relative = tuple(Fraction(float(query[k])) - origin[k] for k in range(3))
    x, y, z = (
        sum((a * b for a, b in zip(relative, axis, strict=True)), Fraction(0))
        for axis in region.axes
    )
    height = min(max(z, region.lower_height), region.upper_height)
    radial = sqrt_bounds(x * x + y * y)
    lower, upper = _distance_bounds(radial, region.radius, z - height)
    data["distances"][row] = upper
    data["distance_lower_bounds"][row] = lower
    data["parameters"][row, 1] = float(height)
    lower_active, upper_active = (
        height == region.lower_height,
        height == region.upper_height,
    )
    if lower_active or upper_active:
        data["edge_indices"][row] = rings[0 if lower_active else 1]
    if x == 0 and y == 0:
        # The full circular fiber is certified, but has no distinguished point,
        # angle, normal or differential. Infinite representative errors are not
        # unfinished root work and do not change its NON_ISOLATED classification.
        return
    angle_lower, angle_upper = angle_bounds(x, y)
    angle = (angle_lower + angle_upper) / 2
    angular_error = max(
        abs(Fraction(angle) - Fraction(angle_lower)),
        abs(Fraction(angle_upper) - Fraction(angle)),
    )
    axial_error = abs(Fraction(float(height)) - height)
    parameter_error = fraction_endpoint(
        sqrt_bounds(angular_error * angular_error + axial_error * axial_error)[1],
        upper=True,
        saturate=True,
    )
    point, normal, first, second, point_error = _point_fields(
        origin,
        region.axes,
        region.radius,
        x,
        y,
        height,
        radial,
        Fraction(float(model.orientation[region.lateral_face])),
    )
    data["points"][row] = point
    data["parameters"][row, 0] = angle
    data["normals"][row] = normal
    data["first_derivatives"][row] = first
    data["second_derivatives"][row] = second
    data["point_error_bounds"][row] = point_error
    data["parameter_bounds"][row] = parameter_error
    data["stationarity"][row] = _stationarity(
        query,
        point,
        first,
        second,
        lower_active,
        upper_active,
        upper,
        tolerances.distance,
    )
    if lower_active or upper_active:
        data["edge_parameters"][row] = angle
    elif y == 0 and x > 0:
        data["edge_indices"][row] = seam
        data["edge_parameters"][row] = _seam_parameter(
            model, seam, height, origin, region.axes[2]
        )
    failed = (
        upper - lower > tolerances.distance
        or point_error > tolerances.distance
        or parameter_error > tolerances.parametric
        or not np.isfinite(data["stationarity"][row])
    )
    data["unresolved"][row] = failed
    data["status"][row] = (
        int(BRepProjectionStatus.FAILED)
        if failed
        else int(BRepProjectionStatus.SEAM)
        if data["edge_indices"][row] >= 0
        else int(BRepProjectionStatus.UNIQUE)
    )


def closest_workspace_bytes(count: int, /) -> int:
    """Bound one streaming row of exact arithmetic on binary64 inputs and buffers."""
    # Binary64 dyadics have at most 1075 numerator/denominator bits. Fixed-depth
    # dots, root quotients, world boxes and error norms use at most 64 times that
    # bound and 128 live Fractions. Four-byte/15-bit limbs cover both CPython builds.
    intermediate_bits = 64 * 1075
    integer_bytes = 64 + 4 * ((intermediate_bits + 14) // 15)
    exact_workspace_bytes = 128 * (64 + 2 * integer_bytes)
    # Arrays require 182 bytes/row; 256 includes row temporaries. Metadata includes
    # array headers, dictionaries, tuples and the immutable result constructor.
    return 256 * count + 16384 + exact_workspace_bytes


@checked
def closest_complete_cylinder(
    model: BRepModel,
    face: int,
    points: np.ndarray,
    tolerances: Tolerances,
    budget: BRepQueryBudget,
    *,
    coordinate_budget: CoordinateEnclosureBudget,
) -> AnalyticClosestBatch | None:
    """Return directed closest-point evidence for an admitted cylinder wall.

    Each row visits the source once. Rational frame coordinates give rho and the
    native-height clamp h; the squared minimum is (rho-radius)**2 + (z-h)**2.
    Integer-root bounds and rational point boxes enclose the minimum and the
    executed representative separately. Boundary ownership uses height and
    zero-phase seam predicates evaluated with exact arithmetic, without tolerance.
    """
    region = complete_cylinder_region(model, budget=coordinate_budget)
    if region is None or region.lateral_face != face:
        return None
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or points.dtype != np.dtype(np.float64)
        or not np.all(np.isfinite(points))
    ):
        raise ValueError(
            "closest_complete_cylinder.points must be finite float64 (N, 3); "
            f"got dtype {points.dtype}, shape {points.shape}"
        )
    count = points.shape[0]
    # The dispatcher charges logical input points once across selected faces;
    # this helper charges only completed source-face operations.
    budget.admit(count, 0, closest_workspace_bytes(count))
    match coordinate_budget.ticket:
        case NativeBatchTicket() as query_ticket:
            if count > query_ticket.remaining_geometry_queries:
                query_ticket.refuse(0, queries_upper=count)
        case _:
            query_ticket = None
    data = _batch_arrays(count, face)
    origin = tuple(Fraction(float(v)) for v in np.asarray(region.patch.origin))
    rings, seam = _wall_edges(model, face, region.cap_faces)
    for row, query in enumerate(points):
        if query_ticket is not None:
            query_ticket.consume(geometry_queries=1)
        _fill_row(data, row, query, region, model, origin, rings, seam, tolerances)
        budget.consume(1)
    return AnalyticClosestBatch(**data)
