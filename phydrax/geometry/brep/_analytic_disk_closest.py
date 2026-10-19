#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Compute closed-form closest queries for source-plane disks with a full-turn rim.

Projection and clipping use rational frame coordinates without rounding. Integer
square-root inequalities enclose radial normalization, distances, and representation
errors. The circle parameter uses the interval implementation's function error bound.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from ..._meshcore import NativeBatchTicket
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked
from .._rational import fraction_endpoint, rational_dot
from ._analytic_cylinder_closest import (
    angle_bounds,
    linear_root_quotient,
    sqrt_bounds,
    vector_error,
)
from ._analytic_source_regions import AnalyticClosestBatch, complete_disk_region
from ._model import BRepModel
from ._projection_contracts import BRepProjectionStatus
from ._query_types import BRepQueryBudget, Tolerances


type _Vector = tuple[Fraction, ...]
type _Bounds = tuple[tuple[Fraction, Fraction], ...]


def _disk_coordinates(
    point: np.ndarray, center: _Vector, axes: tuple[_Vector, ...], normal: _Vector, /
) -> tuple[_Vector, Fraction]:
    delta = tuple(
        Fraction(float(value)) - origin
        for value, origin in zip(point, center, strict=True)
    )
    x, y = (rational_dot(delta, axis) for axis in axes)
    return (x, y), rational_dot(delta, normal)


def _disk_bounds(
    coordinates: _Vector,
    height: Fraction,
    center: _Vector,
    axes: tuple[_Vector, ...],
    center_uv: _Vector,
    radius: Fraction,
    /,
) -> tuple[_Bounds, _Bounds, tuple[float, float], bool]:
    x, y = coordinates
    squared = x * x + y * y
    rim = squared >= radius * radius
    if squared > radius * radius:
        radial = sqrt_bounds(squared)
        uv = tuple(
            linear_root_quotient(base, radius * value, radial)
            for base, value in zip(center_uv, coordinates, strict=True)
        )
        world = tuple(
            linear_root_quotient(
                center[k], radius * (x * axes[0][k] + y * axes[1][k]), radial
            )
            for k in range(3)
        )
        gap = (max(Fraction(0), radial[0] - radius), radial[1] - radius)
        lower = sqrt_bounds(height * height + gap[0] * gap[0])[0]
        upper = sqrt_bounds(height * height + gap[1] * gap[1])[1]
    else:
        uv = tuple(
            (base + value, base + value)
            for base, value in zip(center_uv, coordinates, strict=True)
        )
        projected = tuple(center[k] + x * axes[0][k] + y * axes[1][k] for k in range(3))
        world = tuple((value, value) for value in projected)
        lower = upper = abs(height)
    return (
        world,
        uv,
        (
            fraction_endpoint(lower, upper=False, saturate=True),
            fraction_endpoint(upper, upper=True, saturate=True),
        ),
        rim,
    )


def _representative(bounds: _Bounds, /) -> np.ndarray:
    return np.asarray(
        [
            fraction_endpoint((lower + upper) / 2, upper=False, saturate=True)
            for lower, upper in bounds
        ],
        dtype=np.float64,
    )


def _scratch_bytes(count: int, coefficients: tuple[Fraction, ...], /) -> int:
    """Bound batch banks and one streaming row's rational intermediates.

    Binary64 input ratios occupy at most 2200 numerator/denominator bits.
    The fixed projection expression has fewer than 256 simultaneous rational
    objects and at most 32 input-bit-lengths per numerator or denominator.
    Integer storage uses the conservative two-integer bound of two bytes per
    seven bits plus object headers; no query-sized rational bank is retained.
    """
    if count == 0:
        return 4096
    bits = max(
        (
            value.numerator.bit_length() + value.denominator.bit_length()
            for value in coefficients
        ),
        default=1,
    )
    return count * 182 + 256 * (256 + (64 * (bits + 2200) + 6) // 7)


@checked
def closest_complete_disk(
    model: BRepModel,
    face: int,
    points: np.ndarray,
    tolerances: Tolerances,
    budget: BRepQueryBudget,
    *,
    coordinate_budget: CoordinateEnclosureBudget,
) -> AnalyticClosestBatch | None:
    """Return a source-disk closest certificate, or decline the source.

    In rational orthonormal coordinates, the solution is the plane projection
    when x²+y² <= r², otherwise its radial clipping to the source circle.
    The convex disk KKT multiplier is zero in the interior and rho/r-1 on
    the rim; consequently the constrained stationarity residual is exactly zero.
    Radius clipping is enclosed rather than inferred from a floating comparison.
    """
    source = complete_disk_region(model, face, budget=coordinate_budget)
    if source is None:
        return None
    count = points.shape[0]
    axes = source.plane_axes
    origin = tuple(Fraction(float(value)) for value in np.asarray(source.patch.origin))
    offset = tuple(a - b for a, b in zip(source.center, origin, strict=True))
    center_uv = tuple(rational_dot(offset, axis) for axis in axes)
    normal = (
        axes[0][1] * axes[1][2] - axes[0][2] * axes[1][1],
        axes[0][2] * axes[1][0] - axes[0][0] * axes[1][2],
        axes[0][0] * axes[1][1] - axes[0][1] * axes[1][0],
    )
    coefficients = (
        source.center
        + origin
        + source.plane_axes[0]
        + source.plane_axes[1]
        + source.circle_axes[0]
        + source.circle_axes[1]
        + (source.radius,)
    )
    # The caller accounts for logical input points once across selected faces.
    budget.admit(count, 0, _scratch_bytes(count, coefficients))
    match coordinate_budget.ticket:
        case NativeBatchTicket() as query_ticket:
            if count > query_ticket.remaining_geometry_queries:
                query_ticket.refuse(0, queries_upper=count)
        case _:
            query_ticket = None
    projected = np.empty((count, 3), dtype=np.float64)
    parameters = np.empty((count, 2), dtype=np.float64)
    distances = np.empty(count, dtype=np.float64)
    lower_bounds = np.empty(count, dtype=np.float64)
    status = np.empty(count, dtype=np.int8)
    unresolved = np.empty(count, dtype=np.bool_)
    parameter_bounds = np.empty(count, dtype=np.float64)
    point_errors = np.empty(count, dtype=np.float64)
    edge_indices = np.full(count, -1, dtype=np.int32)
    edge_parameters = np.full(count, np.nan, dtype=np.float64)
    first = np.broadcast_to(
        np.asarray([float(value) for value in axes[0]], dtype=np.float64), (count, 3)
    ).copy()
    second = np.broadcast_to(
        np.asarray([float(value) for value in axes[1]], dtype=np.float64), (count, 3)
    ).copy()
    oriented_normal = np.asarray(
        [float(value) for value in normal], dtype=np.float64
    ) * float(model.orientation[face])
    normals = np.broadcast_to(oriented_normal, (count, 3)).copy()
    circle_coordinates = tuple(
        tuple(rational_dot(axis, circle_axis) for axis in axes)
        for circle_axis in source.circle_axes
    )
    for row, point in enumerate(points):
        if query_ticket is not None:
            query_ticket.consume(geometry_queries=1)
        coordinates, height = _disk_coordinates(point, source.center, axes, normal)
        world, uv, distance, rim = _disk_bounds(
            coordinates, height, source.center, axes, center_uv, source.radius
        )
        projected[row] = _representative(world)
        parameters[row] = _representative(uv)
        lower_bounds[row], distances[row] = distance
        finite = bool(
            np.all(np.isfinite(projected[row])) and np.all(np.isfinite(parameters[row]))
        )
        point_errors[row] = vector_error(projected[row], world) if finite else np.inf
        parameter_bounds[row] = vector_error(parameters[row], uv) if finite else np.inf
        if rim:
            edge_indices[row] = source.edge
            circle_x, circle_y = (
                rational_dot(coordinates, axis) for axis in circle_coordinates
            )
            angle_lower, angle_upper = angle_bounds(circle_x, circle_y)
            edge_parameters[row] = (angle_lower + angle_upper) / 2
        unresolved[row] = (
            not finite
            or not np.isfinite(distances[row])
            or distances[row] - lower_bounds[row] > tolerances.distance
            or point_errors[row] > tolerances.distance
            or parameter_bounds[row] > tolerances.parametric
        )
        status[row] = int(
            BRepProjectionStatus.FAILED
            if unresolved[row]
            else BRepProjectionStatus.SEAM
            if rim
            else BRepProjectionStatus.UNIQUE
        )
        budget.consume(1)
    return AnalyticClosestBatch(
        points=projected,
        parameters=parameters,
        distances=distances,
        faces=np.full(count, source.face, dtype=np.int32),
        status=status,
        normals=normals,
        first_derivatives=first,
        second_derivatives=second,
        stationarity=np.zeros(count, dtype=np.float64),
        distance_lower_bounds=lower_bounds,
        unresolved=unresolved,
        refinements=np.zeros(count, dtype=np.int32),
        parameter_bounds=parameter_bounds,
        edge_indices=edge_indices,
        edge_parameters=edge_parameters,
        point_error_bounds=point_errors,
        query_operations=np.ones(count, dtype=np.int64),
    )
