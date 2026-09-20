#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import numpy as np


def signed_area2(points: np.ndarray, /) -> float:
    points_ = np.asarray(points, dtype=np.float64)
    return float(
        np.sum(
            points_[:, 0] * np.roll(points_[:, 1], -1)
            - np.roll(points_[:, 0], -1) * points_[:, 1]
        )
    )


def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray, /) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_intersect(a, b, c, d, tolerance: float, /) -> bool:
    first = orientation(a, b, c)
    second = orientation(a, b, d)
    third = orientation(c, d, a)
    fourth = orientation(c, d, b)
    return (
        (first > tolerance and second < -tolerance)
        or (first < -tolerance and second > tolerance)
    ) and (
        (third > tolerance and fourth < -tolerance)
        or (third < -tolerance and fourth > tolerance)
    )


def validate_simple_polygon(
    points: np.ndarray,
    /,
    *,
    require_counter_clockwise: bool = True,
) -> None:
    points_ = np.asarray(points, dtype=np.float64)
    if points_.ndim != 2 or points_.shape[0] < 3 or points_.shape[1] != 2:
        raise ValueError("Polygon vertices must have shape (count >= 3, 2).")
    if not np.all(np.isfinite(points_)):
        raise ValueError("Polygon vertices must contain only finite values.")
    count = points_.shape[0]
    scale = max(float(np.max(np.abs(points_))), 1.0)
    tolerance = 128.0 * np.finfo(np.float64).eps * scale * scale
    area = signed_area2(points_)
    if abs(area) <= tolerance:
        raise ValueError("Polygon cells must have positive area.")
    if require_counter_clockwise and area < 0.0:
        raise ValueError("Polygon cells must be counter-clockwise with positive area.")
    edges = np.roll(points_, -1, axis=0) - points_
    if np.any(np.sum(edges * edges, axis=1) <= tolerance):
        raise ValueError("Polygon cells cannot contain zero-length edges.")
    for first in range(count):
        first_next = (first + 1) % count
        for second in range(first + 1, count):
            second_next = (second + 1) % count
            if first in (second, second_next) or first_next in (second, second_next):
                continue
            if _segments_intersect(
                points_[first],
                points_[first_next],
                points_[second],
                points_[second_next],
                tolerance,
            ):
                raise ValueError(
                    "Polygon cells must be simple and non-self-intersecting."
                )


__all__ = ["orientation", "signed_area2", "validate_simple_polygon"]
