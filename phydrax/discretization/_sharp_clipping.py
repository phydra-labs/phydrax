#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host-certified exact linear-edge clipping shared by embedded geometry frontends."""

from __future__ import annotations

import numpy as np


def unique_points(points, tolerance, /):
    unique = []
    for point in points:
        if not any(np.linalg.norm(point - existing) <= tolerance for existing in unique):
            unique.append(point)
    return unique


def clip_positive_polygon(vertices, values, /):
    """Clip one counter-clockwise polygon to a linearly interpolated positive field."""
    output = []
    intersections = []
    for index in range(vertices.shape[0]):
        start = vertices[index]
        stop = vertices[(index + 1) % vertices.shape[0]]
        start_value = float(values[index])
        stop_value = float(values[(index + 1) % vertices.shape[0]])
        start_inside = start_value >= 0.0
        stop_inside = stop_value >= 0.0
        if start_inside:
            output.append(start)
        if start_inside != stop_inside:
            fraction = start_value / (start_value - stop_value)
            point = start + fraction * (stop - start)
            output.append(point)
            intersections.append(point)
    scale = max(np.max(np.linalg.norm(vertices - vertices[0], axis=-1)), 1.0)
    tolerance = 128.0 * np.finfo(np.float64).eps * scale
    output = unique_points(output, tolerance)
    intersections = unique_points(intersections, tolerance)
    return np.asarray(output), intersections


def open_positive_segment(start, stop, start_value, stop_value, /):
    """Return positive-subsegment fraction, measure, centroid, and endpoints."""
    start_ = np.asarray(start, dtype=np.float64)
    stop_ = np.asarray(stop, dtype=np.float64)
    first = float(start_value)
    second = float(stop_value)
    length = float(np.linalg.norm(stop_ - start_))
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("Embedded background segment must have positive finite measure.")
    if first == 0.0 or second == 0.0:
        raise ValueError(
            "Embedded background-face crossing through a vertex is ambiguous."
        )
    if first > 0.0 and second > 0.0:
        return 1.0, length, 0.5 * (start_ + stop_), start_, stop_
    if first < 0.0 and second < 0.0:
        zero = np.zeros_like(start_)
        return 0.0, 0.0, zero, zero, zero
    crossing = first / (first - second)
    point = start_ + crossing * (stop_ - start_)
    if first > 0.0:
        fraction = crossing
        first_open, second_open = start_, point
    else:
        fraction = 1.0 - crossing
        first_open, second_open = point, stop_
    return (
        fraction,
        fraction * length,
        0.5 * (first_open + second_open),
        first_open,
        second_open,
    )


def polygon_measure_centroid(vertices, /):
    """Return positive polygon area and centroid, or a zero empty-polygon measure."""
    if vertices.shape[0] < 3:
        return 0.0, np.zeros((2,))
    following = np.roll(vertices, -1, axis=0)
    cross = vertices[:, 0] * following[:, 1] - following[:, 0] * vertices[:, 1]
    twice_area = np.sum(cross)
    if twice_area <= 0.0:
        raise ValueError("Clipped fluid polygon must retain positive orientation.")
    area = 0.5 * twice_area
    centroid = np.sum((vertices + following) * cross[:, None], axis=0) / (
        3.0 * twice_area
    )
    return area, centroid


__all__ = [
    "clip_positive_polygon",
    "open_positive_segment",
    "polygon_measure_centroid",
    "unique_points",
]
