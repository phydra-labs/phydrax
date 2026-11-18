"""Consumer-facing quality checks against exact arithmetic on represented coordinates."""

import math
from fractions import Fraction
from itertools import product
from typing import Any

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest

import phydrax as phx


_CUBE_FACES = (
    (0, 1, 3, 2),
    (4, 6, 7, 5),
    (0, 4, 5, 1),
    (2, 3, 7, 6),
    (0, 2, 6, 4),
    (1, 5, 7, 3),
)


def _ideal(kind: str) -> np.ndarray:
    root3 = math.sqrt(3.0)
    triangle = np.array(((0, 0), (1, 0), (0.5, root3 / 2)))
    if kind == "triangle":
        return triangle
    if kind == "polygon":
        angles = np.arange(6) * math.pi / 3
        return np.stack((np.cos(angles), np.sin(angles)), axis=1)
    if kind == "tetrahedron":
        return np.array(
            (
                (0, 0, 0),
                (1, 0, 0),
                (0.5, root3 / 2, 0),
                (0.5, root3 / 6, math.sqrt(2 / 3)),
            )
        )
    if kind == "prism":
        return np.concatenate(
            (
                np.column_stack((triangle, np.zeros(3))),
                np.column_stack((triangle, np.ones(3))),
            )
        )
    if kind == "polyhedron":
        return np.array(list(product((0.0, 1.0), repeat=3)))
    points = np.array(phx.discretization.reference_cell_topology(kind).vertices)
    if kind == "pyramid":
        points[4, 2] = math.sqrt(0.5)
    return points


def _mesh(kind: str, points: np.ndarray) -> Any:
    if kind == "polyhedron":
        return phx.discretization.CellMesh.from_polyhedra(points, (_CUBE_FACES,))
    return phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "cells", kind, np.arange(len(points), dtype=np.int32)[None, :]
            ),
        ),
    )


def _det(
    a: tuple[Fraction, ...], b: tuple[Fraction, ...], c: tuple[Fraction, ...]
) -> Fraction:
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _sub(a: tuple[Fraction, ...], b: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    return tuple(x - y for x, y in zip(a, b, strict=True))


def _reference_measure(kind: str, points: np.ndarray) -> float:
    # Fractions exactly represent the binary64 input, including irrational ideal
    # coordinates after rounding. Only the final square root is rounded.
    p = [tuple(Fraction(float(x)) for x in row) for row in points]
    if kind == "interval":
        return math.sqrt(float(sum(x * x for x in _sub(p[1], p[0]))))
    if kind in ("triangle", "quadrilateral", "polygon"):
        if len(p[0]) == 2:
            return float(
                sum(
                    a[0] * b[1] - a[1] * b[0]
                    for a, b in zip(p, p[1:] + p[:1], strict=True)
                )
                / 2
            )
        cross_sum = [Fraction(0), Fraction(0), Fraction(0)]
        for a, b in zip(p, p[1:] + p[:1], strict=True):
            for i in range(3):
                cross_sum[i] += (
                    a[(i + 1) % 3] * b[(i + 2) % 3] - a[(i + 2) % 3] * b[(i + 1) % 3]
                )
        return math.sqrt(float(sum(x * x for x in cross_sum))) / 2
    faces = (
        _CUBE_FACES
        if kind == "polyhedron"
        else phx.discretization.reference_cell_topology(kind).entities[2]
    )
    center = tuple(sum((row[i] for row in p), Fraction(0)) / len(p) for i in range(3))
    # Convex, planar-faced cells: tetrahedralize each face to the interior point.
    volume6 = Fraction(0)
    for face in faces:
        a = _sub(p[face[0]], center)
        for i in range(1, len(face) - 1):
            volume6 += abs(
                _det(a, _sub(p[face[i]], center), _sub(p[face[i + 1]], center))
            )
    return float(volume6 / 6)


_CASES = [
    (kind, embedded)
    for kind in (
        "interval",
        "triangle",
        "quadrilateral",
        "polygon",
        "tetrahedron",
        "hexahedron",
        "prism",
        "pyramid",
        "polyhedron",
    )
    for embedded in (
        (False, True) if kind in ("triangle", "quadrilateral", "polygon") else (False,)
    )
]


@pytest.mark.parametrize("kind,embedded", _CASES)
@pytest.mark.parametrize("distorted", (False, True), ids=("ideal", "distorted"))
def test_quality_measure_against_exact_reference_and_jit_gradient(
    kind: str, embedded: bool, distorted: bool
) -> None:
    points = _ideal(kind)
    if distorted:
        points = points * np.array((1.75, 0.75, 1.25)[: points.shape[1]])
    if embedded:
        # Isometric embedding in a tilted plane, not merely a zero z coordinate.
        points = np.column_stack((points[:, 0], points[:, 1] * 0.6, points[:, 1] * 0.8))
    mesh = _mesh(kind, points)
    x = jnp.asarray(points)
    evaluate = lambda coordinates: phx.meshing.evaluate_cell_quality(mesh, coordinates)
    eager = evaluate(x)
    compiled = jax.jit(evaluate)(x)
    expected = _reference_measure(kind, points)
    # The oracle is exact through rational accumulation; the implementation uses
    # binary64 reductions and, for solids, floating-point quadrature.
    np.testing.assert_allclose(eager.measures, [expected], rtol=3e-13, atol=3e-14)
    for field in (
        "measures",
        "mean_ratios",
        "aspect_ratios",
        "radius_ratios",
        "scaled_jacobian",
        "minimum_angle",
        "maximum_angle",
        "condition_number",
        "sliver_measures",
        "warpage",
        "metric_quality",
    ):
        np.testing.assert_allclose(
            getattr(compiled, field),
            getattr(eager, field),
            rtol=3e-13,
            atol=3e-14,
            equal_nan=True,
        )
    assert bool(eager.sampled_valid[0])
    assert bool(compiled.sampled_valid[0])
    if not distorted and kind != "polyhedron":
        np.testing.assert_allclose(eager.mean_ratios, [1.0], rtol=3e-13, atol=3e-14)

    objective = lambda coordinates: evaluate(coordinates).measures[0]
    gradient = jax.grad(objective)(x)
    compiled_gradient = jax.jit(jax.grad(objective))(x)
    np.testing.assert_allclose(compiled_gradient, gradient, rtol=3e-13, atol=3e-14)
    # Euler's identity supplies an independent derivative oracle for homogeneous
    # length, area, and volume, without finite-difference cancellation.
    dimension = (
        1
        if kind == "interval"
        else 2
        if kind in ("triangle", "quadrilateral", "polygon")
        else 3
    )
    np.testing.assert_allclose(
        jnp.sum(gradient * x), dimension * expected, rtol=5e-13, atol=5e-14
    )


@pytest.mark.parametrize("embedded", (False, True))
def test_triangle_shape_against_closed_form(embedded: bool) -> None:
    points = np.array(((0.0, 0.0), (2.0, 0.0), (0.5, 1.5)))
    if embedded:
        points = np.column_stack((points[:, 0], 0.6 * points[:, 1], 0.8 * points[:, 1]))
    quality = phx.meshing.evaluate_cell_quality(_mesh("triangle", points))
    area = _reference_measure("triangle", points)
    lengths = [math.dist(points[i], points[(i + 1) % 3]) for i in range(3)]
    # For triangles all corner mean ratios coincide: 4 sqrt(3) A / sum(l^2).
    ratio = 4 * math.sqrt(3) * area / sum(length * length for length in lengths)
    radius = 16 * area * area / (sum(lengths) * math.prod(lengths))
    angles = [
        math.acos(
            sum(
                float(a) * float(b)
                for a, b in zip(
                    points[(i + 1) % 3] - points[i],
                    points[(i - 1) % 3] - points[i],
                    strict=True,
                )
            )
            / (lengths[i] * lengths[(i - 1) % 3])
        )
        for i in range(3)
    ]
    np.testing.assert_allclose(quality.mean_ratios, [ratio], rtol=3e-13)
    np.testing.assert_allclose(
        quality.aspect_ratios, [max(lengths) / min(lengths)], rtol=3e-13
    )
    np.testing.assert_allclose(quality.radius_ratios, [radius], rtol=3e-13)
    np.testing.assert_allclose(
        quality.scaled_jacobian,
        [min(map(math.sin, angles)) / math.sin(math.pi / 3)],
        rtol=3e-13,
    )
    np.testing.assert_allclose(quality.minimum_angle, [min(angles)], rtol=3e-13)
    np.testing.assert_allclose(quality.maximum_angle, [max(angles)], rtol=3e-13)
    np.testing.assert_allclose(quality.condition_number, [1.0 / ratio], rtol=3e-13)
    np.testing.assert_array_equal(quality.warpage, [0.0])
    assert np.isnan(np.asarray(quality.sliver_measures)).all()
    assert np.isnan(np.asarray(quality.metric_quality)).all()


@pytest.mark.parametrize(
    ("kind", "minimum_angle", "maximum_angle"),
    (
        ("triangle", math.pi / 3.0, math.pi / 3.0),
        ("quadrilateral", math.pi / 2.0, math.pi / 2.0),
        ("tetrahedron", math.acos(1.0 / 3.0), math.acos(1.0 / 3.0)),
        ("hexahedron", math.pi / 2.0, math.pi / 2.0),
        ("prism", math.pi / 3.0, math.pi / 2.0),
    ),
)
def test_ideal_quality_against_geometric_constants(
    kind: str, minimum_angle: float, maximum_angle: float
) -> None:
    quality = phx.meshing.evaluate_cell_quality(_mesh(kind, _ideal(kind)))
    for value in (
        quality.mean_ratios,
        quality.aspect_ratios,
        quality.scaled_jacobian,
        quality.condition_number,
    ):
        np.testing.assert_allclose(value, [1.0], rtol=3e-13, atol=3e-14)
    np.testing.assert_allclose(quality.minimum_angle, [minimum_angle], atol=3e-14)
    np.testing.assert_allclose(quality.maximum_angle, [maximum_angle], atol=3e-14)
    np.testing.assert_allclose(quality.warpage, [0.0], atol=3e-14)
    if kind in ("triangle", "tetrahedron"):
        np.testing.assert_allclose(quality.radius_ratios, [1.0], rtol=3e-13)
    else:
        assert np.isnan(np.asarray(quality.radius_ratios)).all()
    if kind == "tetrahedron":
        np.testing.assert_allclose(quality.sliver_measures, [1.0], rtol=3e-13)
    else:
        assert np.isnan(np.asarray(quality.sliver_measures)).all()


def _high_precision_corner_shapes(
    kind: str, points: np.ndarray
) -> tuple[float, float, float]:
    """Independent 100-digit Gram-matrix and edge-normalization reference."""
    ideal = _ideal(kind)
    dimension = ideal.shape[1]
    count = len(ideal)
    if dimension == 2:
        rows = [(i, ((i + 1) % count, (i - 1) % count)) for i in range(count)]
    else:
        edges = phx.discretization.reference_cell_topology(kind).entities[1]
        adjacency: list[list[int]] = [[] for _ in range(count)]
        for first, second in edges:
            adjacency[first].append(second)
            adjacency[second].append(first)
        rows = []
        for vertex, neighbors in enumerate(adjacency):
            if len(neighbors) == 3:
                rows.append((vertex, tuple(neighbors)))
            else:
                cycle = sorted(neighbors)
                rows.extend(
                    (vertex, tuple(cycle[(start + offset) % 4] for offset in range(3)))
                    for start in range(4)
                )
    with mp.workdps(100):
        actual_points = [[mp.mpf(float(x)) for x in point] for point in points]
        ideal_points = [[mp.mpf(float(x)) for x in point] for point in ideal]
        if dimension == 2 and points.shape[1] == 3:
            first = [actual_points[1][i] - actual_points[0][i] for i in range(3)]
            second = [actual_points[-1][i] - actual_points[0][i] for i in range(3)]
            normal = [
                first[(i + 1) % 3] * second[(i + 2) % 3]
                - first[(i + 2) % 3] * second[(i + 1) % 3]
                for i in range(3)
            ]
            normal_length = mp.sqrt(sum(value * value for value in normal))
            normal = [value / normal_length for value in normal]
        else:
            normal = []
        ratios = []
        scaled = []
        conditions = []
        for vertex, neighbors in rows:
            frame = mp.matrix(
                [
                    [actual_points[n][i] - actual_points[vertex][i] for n in neighbors]
                    for i in range(points.shape[1])
                ]
            )
            reference = mp.matrix(
                [
                    [ideal_points[n][i] - ideal_points[vertex][i] for n in neighbors]
                    for i in range(dimension)
                ]
            )
            if mp.det(reference) < 0:
                frame[:, 1], frame[:, 2] = frame[:, 2], frame[:, 1]
                reference[:, 1], reference[:, 2] = reference[:, 2], reference[:, 1]
            if normal:
                cross = [
                    frame[(i + 1) % 3, 0] * frame[(i + 2) % 3, 1]
                    - frame[(i + 2) % 3, 0] * frame[(i + 1) % 3, 1]
                    for i in range(3)
                ]
                determinant = sum(cross[i] * normal[i] for i in range(3))
            else:
                determinant = mp.det(frame)
            frame_lengths = [
                mp.sqrt(sum(frame[i, j] ** 2 for i in range(points.shape[1])))
                for j in range(dimension)
            ]
            ideal_lengths = [
                mp.sqrt(sum(reference[i, j] ** 2 for i in range(dimension)))
                for j in range(dimension)
            ]
            scaled.append(
                determinant
                / mp.fprod(frame_lengths)
                / (mp.det(reference) / mp.fprod(ideal_lengths))
            )
            target = frame * reference**-1
            gram = target.T * target
            trace = sum(gram[i, i] for i in range(dimension))
            gram_det = mp.det(gram)
            ratios.append(dimension * mp.root(gram_det, dimension) / trace)
            adjugate_trace = (
                trace
                if dimension == 2
                else sum(
                    gram[i, i] * gram[j, j] - gram[i, j] * gram[j, i]
                    for i in range(3)
                    for j in range(i + 1, 3)
                )
            )
            conditions.append(mp.sqrt(trace * adjugate_trace / gram_det) / dimension)
        return float(min(ratios)), float(min(scaled)), float(max(conditions))


@pytest.mark.parametrize(
    "kind",
    (
        "triangle",
        "quadrilateral",
        "polygon",
        "tetrahedron",
        "hexahedron",
        "prism",
        "pyramid",
    ),
)
def test_distorted_corner_quality_against_high_precision(kind: str) -> None:
    points = _ideal(kind)
    if points.shape[1] == 2:
        transform = np.asarray(((1.25, 0.2), (0.15, 0.8)), dtype=np.float64)
    else:
        transform = np.asarray(
            ((1.25, 0.2, 0.1), (0.0, 0.8, 0.15), (0.0, 0.0, 1.4)),
            dtype=np.float64,
        )
    points = points @ transform.T
    quality = phx.meshing.evaluate_cell_quality(_mesh(kind, points))
    ratio, scaled, condition = _high_precision_corner_shapes(kind, points)
    np.testing.assert_allclose(quality.mean_ratios, [ratio], rtol=3e-13)
    np.testing.assert_allclose(quality.scaled_jacobian, [scaled], rtol=3e-13)
    np.testing.assert_allclose(quality.condition_number, [condition], rtol=3e-13)
