"""The recorded metric edit goals against independent references.

``native/meshcore/tests/metric_goal_vectors.inc`` holds the inputs and the
output bits that the native test requires on every platform. These tests hold
the recorded outputs to the definitions of the goals, evaluated here with the
platform's libm and LAPACK and by exhaustive search. They read the recorded
file only, so they are independent of the kernels.
"""

from __future__ import annotations

import math
import re
import struct
from itertools import combinations
from pathlib import Path

import numpy as np


_VECTORS = (
    Path(__file__).resolve().parents[3]
    / "native"
    / "meshcore"
    / "tests"
    / "metric_goal_vectors.inc"
)


def _binary64(bits: int, /) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def _rows(kind: str, /) -> list[list[int]]:
    rows = []
    for line in _VECTORS.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(rf"PHX_MG_{kind}\((.*)\)", line)
        if match is not None:
            rows.append([int(field, 16) for field in match[1].split(", ")])
    return rows


def _values(fields: list[int], /) -> np.ndarray:
    return np.asarray([_binary64(bits) for bits in fields], dtype=np.float64)


def _is_metric(tensor: np.ndarray, /) -> bool:
    return bool(np.all(np.isfinite(tensor))) and bool(np.linalg.eigvalsh(tensor)[0] > 0.0)


def _reference_parameter(
    p: np.ndarray, q: np.ndarray, metric_p: np.ndarray, metric_q: np.ndarray, /
) -> float:
    delta = q - p
    with np.errstate(all="ignore"):
        first = float(delta @ metric_p @ delta)
        second = float(delta @ metric_q @ delta)
        if not (math.isfinite(first + second) and first > 0.0 and second > 0.0):
            return 0.5
        ratio = math.sqrt(second / first)
    if ratio == 1.0 or ratio == 0.0 or not math.isfinite(ratio):
        return 0.5
    return math.log1p(0.5 * (ratio - 1.0)) / math.log1p(ratio - 1.0)


def _smallest_ball(points: np.ndarray, /) -> tuple[np.ndarray, float]:
    """Smallest enclosing ball by search over every support of up to four points."""
    best_center = points[0]
    best_radius = math.inf
    scale = max(float(np.max(np.abs(points))), 1.0)
    for size in range(1, min(4, len(points)) + 1):
        for support in combinations(range(len(points)), size):
            corner = points[list(support)]
            directions = corner[1:] - corner[0]
            if size == 1:
                center = corner[0]
            else:
                coefficients = np.linalg.lstsq(
                    directions @ directions.T,
                    0.5 * np.sum(directions * directions, axis=1),
                    rcond=None,
                )[0]
                center = corner[0] + directions.T @ coefficients
            distances = np.sum((points - center) ** 2, axis=1)
            radius = float(np.max(np.sum((corner - center) ** 2, axis=1)))
            if (
                np.all(distances <= radius + 1.0e-9 * scale * scale)
                and radius < best_radius
            ):
                best_center, best_radius = center, radius
    return best_center, best_radius


def _reference_center(points: np.ndarray, metric: np.ndarray, /) -> np.ndarray:
    factor = np.linalg.cholesky(metric).T
    center, _ = _smallest_ball(points @ factor.T)
    return np.linalg.solve(factor, center)


def _reference_projection(corners: np.ndarray, goal: np.ndarray, /) -> np.ndarray:
    span = (corners[1:] - corners[0]).T
    coefficients = np.linalg.lstsq(span, goal - corners[0], rcond=2.0**-20)[0]
    return corners[0] + span @ coefficients


def test_recorded_midpoints_follow_the_geometric_size_model() -> None:
    interior = 0
    arithmetic = 0
    for row in _rows("MIDPOINT"):
        p = _values(row[0:3])
        q = _values(row[3:6])
        metric_p = _values(row[6:15]).reshape((3, 3))
        metric_q = _values(row[15:24]).reshape((3, 3))
        parameter = _binary64(row[24])
        point = _values(row[25:28])
        reference = _reference_parameter(p, q, metric_p, metric_q)
        assert math.isclose(parameter, reference, rel_tol=1.0e-12)
        assert 0.0 < parameter < 1.0
        with np.errstate(all="ignore"):
            assert np.array_equal(point, p + parameter * (q - p))
        if parameter == 0.5:
            arithmetic += 1
        else:
            # With a local length density l_p r^s along the edge (the size
            # varies geometrically), the length up to t is half the whole when
            # r^t - 1 = (r - 1) / 2.
            ratio = math.sqrt(
                float((q - p) @ metric_q @ (q - p)) / float((q - p) @ metric_p @ (q - p))
            )
            assert math.isclose(
                ratio**parameter - 1.0, 0.5 * (ratio - 1.0), rel_tol=1.0e-9
            )
            interior += 1
    assert interior > 35
    assert arithmetic >= 6


def test_recorded_centers_are_centers_of_the_smallest_metric_ball() -> None:
    found = 0
    refused = 0
    for row in _rows("CENTER"):
        count = row[0]
        metric = _values(row[1:10]).reshape((3, 3))
        points = _values(row[10 : 10 + 3 * count]).reshape((count, 3))
        valid = row[10 + 3 * count]
        center = _values(row[11 + 3 * count :])
        with np.errstate(all="ignore"):
            representable = (
                count > 0
                and _is_metric(metric)
                and bool(np.all(np.isfinite(points)))
                and bool(np.isfinite(np.max(np.abs(points), initial=0.0) ** 2))
            )
        assert bool(valid) == representable
        if not valid:
            assert not center.any()
            refused += 1
            continue
        reference = _reference_center(points, metric)
        extent = float(np.max(np.abs(points))) + 1.0
        assert np.allclose(center, reference, rtol=0.0, atol=1.0e-9 * extent)
        found += 1
    assert found > 40
    assert refused == 4


def test_recorded_hull_projections_are_orthogonal_projections() -> None:
    projected_rows = 0
    refused = 0
    for row in _rows("HULL"):
        corner_count = row[0]
        corners = _values(row[1:10]).reshape((3, 3))[:corner_count]
        goal = _values(row[10:13])
        valid = row[13]
        projected = _values(row[14:17])
        with np.errstate(all="ignore"):
            edges = corners[1:] - corners[0]
            representable = bool(
                np.all(np.isfinite(goal)) and np.all(np.isfinite(edges @ edges.T))
            )
        assert bool(valid) == representable
        if not valid:
            assert not projected.any()
            refused += 1
            continue
        reference = _reference_projection(corners, goal)
        extent = float(np.max(np.abs(np.concatenate((corners.ravel(), goal))))) + 1.0
        assert np.allclose(projected, reference, rtol=0.0, atol=1.0e-10 * extent)
        # The residual is orthogonal to every edge of the hull.
        assert np.allclose(edges @ (goal - projected), 0.0, atol=1.0e-9 * extent * extent)
        projected_rows += 1
    assert projected_rows > 35
    assert refused == 2


def test_recorded_survivor_goals_compose_the_midpoint_the_center_and_the_projection() -> (
    None
):
    complete = 0
    without_center = 0
    for row in _rows("SURVIVOR"):
        count, corner_count = row[0], row[1]
        metrics = _values(row[2:20]).reshape((2, 3, 3))
        corners = _values(row[20:29]).reshape((3, 3))[:corner_count]
        points = _values(row[29 : 29 + 3 * count]).reshape((count, 3))
        valid = row[29 + 3 * count : 32 + 3 * count]
        goals = _values(row[32 + 3 * count :]).reshape((3, 3))
        survivor, removed = points[0], points[1]
        parameter = _reference_parameter(survivor, removed, metrics[0], metrics[1])
        references: list[np.ndarray | None] = [
            survivor + parameter * (removed - survivor),
            survivor + 0.5 * parameter * (removed - survivor),
        ]
        if count > 2 and _is_metric(metrics[0]) and _is_metric(metrics[1]):
            logarithms = []
            for tensor in metrics:
                values, vectors = np.linalg.eigh(tensor)
                logarithms.append((vectors * np.log(values)) @ vectors.T)
            values, vectors = np.linalg.eigh(0.5 * (logarithms[0] + logarithms[1]))
            references.append(
                _reference_center(points[2:], (vectors * np.exp(values)) @ vectors.T)
            )
        else:
            references.append(None)
        extent = float(np.max(np.abs(points))) + 1.0
        for goal, flag, reference in zip(goals, valid, references, strict=True):
            assert bool(flag) == (reference is not None)
            if reference is None:
                assert not goal.any()
                continue
            if corner_count:
                reference = _reference_projection(corners, reference)
            assert np.allclose(goal, reference, rtol=0.0, atol=1.0e-8 * extent)
        complete += all(valid)
        without_center += valid == [1, 1, 0]
    assert complete > 30
    assert without_center == 2
