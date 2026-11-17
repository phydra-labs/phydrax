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


def test_recorded_hull_frames_are_orthonormal_bases_and_projectors() -> None:
    full = 0
    reduced = 0
    for row in _rows("FRAME"):
        corner_count = row[0]
        corners = _values(row[1:10]).reshape((3, 3))[:corner_count]
        rank = row[10]
        basis = _values(row[11:17]).reshape((2, 3))
        projector = _values(row[17:26]).reshape((3, 3))
        assert not basis[rank:].any()
        basis = basis[:rank]
        edges = corners[1:] - corners[0]
        assert rank == int(np.linalg.matrix_rank(edges)) if edges.any() else rank == 0
        assert np.allclose(basis @ basis.T, np.eye(rank), rtol=0.0, atol=1.0e-14)
        assert np.array_equal(projector, projector.T)
        # The projector of the span of the edges, however they are factored.
        reference = edges.T @ np.linalg.pinv(edges.T)
        assert np.allclose(projector, reference, rtol=0.0, atol=1.0e-12)
        full += rank == corner_count - 1
        reduced += rank < corner_count - 1
    assert full > 20
    assert reduced == 3


def test_recorded_goal_projections_apply_their_projector() -> None:
    rows = _rows("PROJECT")
    for row in rows:
        origin = _values(row[0:3])
        projector = _values(row[3:12]).reshape((3, 3))
        goal = _values(row[12:15])
        projected = _values(row[15:18])
        reference = origin + projector @ (goal - origin)
        extent = float(np.max(np.abs(np.concatenate((origin, goal))))) + 1.0
        assert np.allclose(projected, reference, rtol=0.0, atol=1.0e-13 * extent)
    assert len(rows) > 100


def _reference_pulls(
    points: np.ndarray,
    metric: np.ndarray,
    lengths: np.ndarray,
    measured: np.ndarray,
    deep: bool,
    interior: bool,
    basis: np.ndarray,
    projector: np.ndarray | None,
    jacobian: np.ndarray | None,
    /,
) -> list[np.ndarray]:
    """The pulls by their definitions, with LAPACK for the least-squares step."""
    lower = float.fromhex("0x1.6a09e667f3bccp-1")
    upper = float.fromhex("0x1.6a09e667f3bcdp+0")
    spokes = points[1:] - points[0]
    forces = (1.0 - 1.0 / lengths)[:, None] * spokes
    outside = np.flatnonzero(measured & ((lengths < lower) | (lengths > upper)))
    mean = forces.mean(axis=0)
    pulls: list[np.ndarray] = []
    searching = deep and outside.size > 0
    if searching and jacobian is not None:
        residual = np.clip(lengths[measured], lower, upper) - lengths[measured]
        step = np.linalg.lstsq(jacobian[:, measured].T, residual, rcond=2.0**-20)[0]
        joint = basis.T @ step
        if np.any(joint):
            pulls.append(joint)
        for spoke in outside:
            direction = basis.T @ jacobian[:, spoke]
            norm = float(np.linalg.norm(direction))
            if norm > 0.0:
                direction = direction * (float(np.linalg.norm(forces[spoke])) / norm)
                pulls.append(direction if lengths[spoke] < lower else -direction)
    if not deep:
        pulls.append(mean)
    pulls.extend(forces[outside])
    if deep:
        pulls.append(mean)
    if searching:
        active = np.flatnonzero(measured)
        margins = np.minimum(
            np.abs(lengths[active] - lower), np.abs(lengths[active] - upper)
        )
        active = active[np.argsort(margins, kind="stable")]
        normals = spokes[active] @ metric
        directions = forces[outside]
        if projector is not None:
            normals = normals @ projector
            directions = directions @ projector
        for direction in directions:
            for normal, length in zip(normals, lengths[active], strict=True):
                component = float(direction @ normal)
                squared = float(normal @ normal)
                crosses = (length < 1.0 and component > 0.0) or (
                    length > 1.0 and component < 0.0
                )
                if crosses and squared > 0.0:
                    pulls.append(direction - (component / squared) * normal)
            if interior:
                for left, right in combinations(range(len(normals)), 2):
                    crossed = np.cross(normals[left], normals[right])
                    squared = float(crossed @ crossed)
                    if squared > 0.0:
                        pulls.append((float(direction @ crossed) / squared) * crossed)
    return pulls


def test_recorded_relocation_pulls_follow_their_definitions() -> None:
    deep_rows = 0
    gradient_rows = 0
    boundary_rows = 0
    for row in _rows("PULLS"):
        count, deep, interior, basis_count, has_projector, has_jacobian = row[:6]
        cursor = 6
        metric = _values(row[cursor : cursor + 9]).reshape((3, 3))
        cursor += 9
        points = _values(row[cursor : cursor + 3 * (count + 1)]).reshape((count + 1, 3))
        cursor += 3 * (count + 1)
        lengths = _values(row[cursor : cursor + count])
        cursor += count
        measured = np.asarray(row[cursor : cursor + count], dtype=np.bool_)
        cursor += count
        basis = _values(row[cursor : cursor + 9]).reshape((3, 3))[:basis_count]
        cursor += 9
        projector = _values(row[cursor : cursor + 9]).reshape((3, 3))
        cursor += 9
        jacobian = _values(row[cursor : cursor + 3 * count]).reshape((3, count))[
            :basis_count
        ]
        cursor += 3 * count
        written = row[cursor]
        pulls = _values(row[cursor + 1 :]).reshape((written, 3))
        reference = _reference_pulls(
            points,
            metric,
            lengths,
            measured,
            bool(deep),
            bool(interior),
            basis,
            projector if has_projector else None,
            jacobian if has_jacobian else None,
        )
        assert written == len(reference)
        scale = float(np.max(np.abs(np.asarray(reference)))) if reference else 1.0
        assert np.allclose(
            pulls,
            np.asarray(reference).reshape((-1, 3)),
            rtol=1.0e-9,
            atol=1.0e-9 * scale,
        )
        deep_rows += bool(deep) and written > count
        gradient_rows += bool(has_jacobian)
        boundary_rows += not interior
    assert deep_rows > 10
    assert gradient_rows > 5
    assert boundary_rows > 10
