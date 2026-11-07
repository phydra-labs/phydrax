import math
from dataclasses import replace
from fractions import Fraction
from itertools import product
from typing import Any

import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._motion_validity import reference_corner_frames
from phydrax.meshing import _audit, _quality
from phydrax.meshing._quality import CellQualityReport, CellQualityReportArchiveRecipe


def _single(kind: str, coordinates: Any, /) -> Any:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    return phx.discretization.CellMesh(
        coordinates,
        (
            phx.discretization.CellBlock(
                "cells", kind, np.arange(coordinates.shape[0], dtype=np.int32)[None, :]
            ),
        ),
    )


def _reference(kind: str, /) -> np.ndarray:
    return np.asarray(
        phx.discretization.reference_cell_topology(kind).vertices, dtype=np.float64
    )


def _exact(points: np.ndarray, /) -> list[tuple[Fraction, ...]]:
    return [tuple(Fraction(float(value)) for value in row) for row in points]


def _sub(first: Any, second: Any, /) -> tuple[Fraction, ...]:
    return tuple(a - b for a, b in zip(first, second, strict=True))


def _dot(first: Any, second: Any, /) -> Fraction:
    return sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))


def _det3(a: Any, b: Any, c: Any, /) -> Fraction:
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


# Independent Fraction oracles --------------------------------------------------------


def _triangle_ratio_below(points: np.ndarray, threshold: float, /) -> bool:
    """Closed form q = 4 sqrt(3) A / sum(l^2) of the planar triangle."""
    a, b, c = _exact(points)
    area2 = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if area2 <= 0:
        return threshold > 0.0
    lengths = sum(_dot(_sub(q, p), _sub(q, p)) for p, q in ((a, b), (b, c), (c, a)))
    # q < r  <=>  2 sqrt(3) A2 < r L  <=>  12 A2^2 < r^2 L^2.
    return 12 * area2 * area2 < Fraction(threshold) ** 2 * lengths * lengths


def _tetrahedron_ratio_below(points: np.ndarray, threshold: float, /) -> bool:
    """Closed form q = 12 (3V)^(2/3) / sum(l^2) of the tetrahedron."""
    p = _exact(points)
    volume6 = _det3(_sub(p[1], p[0]), _sub(p[2], p[0]), _sub(p[3], p[0]))
    if volume6 <= 0:
        return threshold > 0.0
    lengths = sum(
        _dot(_sub(p[j], p[i]), _sub(p[j], p[i]))
        for i in range(4)
        for j in range(i + 1, 4)
    )
    # q^3 = 1728 * 9 V^2 / L^3 = 432 V6^2 / L^3.
    return 432 * volume6 * volume6 < Fraction(threshold) ** 3 * lengths**3


def _box_ratio_below(kind: str, points: np.ndarray, threshold: float, /) -> bool:
    """Quadrilateral/hexahedron corners: q = d det(F)^(2/d) / |F|^2 (unit ideal)."""
    vertices, neighbors = reference_corner_frames(kind)
    p = _exact(points)
    r = Fraction(threshold)
    for vertex, row in zip(vertices, neighbors, strict=True):
        frames = [_sub(p[int(n)], p[int(vertex)]) for n in row]
        trace = sum(_dot(frame, frame) for frame in frames)
        if len(frames) == 2:
            det = frames[0][0] * frames[1][1] - frames[0][1] * frames[1][0]
            if det <= 0 or 2 * det < r * trace:
                return True
        else:
            det = _det3(*frames)
            if det <= 0 or 27 * det * det < r**3 * trace**3:
                return True
    return False


def _hexahedron_volume(points: np.ndarray, /) -> Fraction:
    """Exact integral of det J of the trilinear map by monomial expansion."""
    reference = _reference("hexahedron")
    p = _exact(points)

    def multiply(first: dict, second: dict) -> dict:
        result: dict = {}
        for (a, b, c), x in first.items():
            for (d, e, f), y in second.items():
                key = (a + d, b + e, c + f)
                result[key] = result.get(key, Fraction(0)) + x * y
        return result

    def add(first: dict, second: dict, sign: int = 1) -> dict:
        result = dict(first)
        for key, value in second.items():
            result[key] = result.get(key, Fraction(0)) + sign * value
        return result

    def factor(axis: int, high: bool) -> dict:
        unit = tuple(1 if index == axis else 0 for index in range(3))
        return (
            {unit: Fraction(1)} if high else {(0, 0, 0): Fraction(1), unit: Fraction(-1)}
        )

    def derivative(poly: dict, axis: int) -> dict:
        result: dict = {}
        for key, value in poly.items():
            if key[axis]:
                lowered = tuple(k - (i == axis) for i, k in enumerate(key))
                result[lowered] = result.get(lowered, Fraction(0)) + key[axis] * value
        return result

    shapes = []
    for vertex in reference:
        shape = {(0, 0, 0): Fraction(1)}
        for axis in range(3):
            shape = multiply(shape, factor(axis, bool(vertex[axis] > 0.5)))
        shapes.append(shape)
    j = [[{} for _ in range(3)] for _ in range(3)]
    for shape, point in zip(shapes, p, strict=True):
        for column in range(3):
            gradient = derivative(shape, column)
            for row in range(3):
                j[row][column] = add(
                    j[row][column], {k: v * point[row] for k, v in gradient.items()}
                )
    det = {}
    for (a, b, c), sign in (
        ((0, 1, 2), 1),
        ((1, 2, 0), 1),
        ((2, 0, 1), 1),
        ((2, 1, 0), -1),
        ((1, 0, 2), -1),
        ((0, 2, 1), -1),
    ):
        det = add(det, multiply(multiply(j[0][a], j[1][b]), j[2][c]), sign)
    return sum(
        (value / ((k[0] + 1) * (k[1] + 1) * (k[2] + 1)) for k, value in det.items()),
        Fraction(0),
    )


# Exact decisions at near-threshold cells ----------------------------------------------


def _near_thresholds(value: float, /) -> tuple[float, ...]:
    below = np.nextafter(value, -np.inf)
    above = np.nextafter(value, np.inf)
    return (
        float(np.nextafter(below, -np.inf)),
        float(below),
        float(value),
        float(above),
        float(np.nextafter(above, np.inf)),
    )


@pytest.mark.parametrize(
    ("kind", "oracle"),
    (
        ("triangle", _triangle_ratio_below),
        ("tetrahedron", _tetrahedron_ratio_below),
        ("quadrilateral", lambda points, r: _box_ratio_below("quadrilateral", points, r)),
        ("hexahedron", lambda points, r: _box_ratio_below("hexahedron", points, r)),
    ),
)
def test_mean_ratio_gate_is_exact_at_the_sampled_threshold(
    kind: str, oracle: Any
) -> None:
    rng = np.random.default_rng(7)
    reference = _reference(kind)
    for _ in range(12):
        points = reference + rng.normal(scale=0.07, size=reference.shape)
        mesh = _single(kind, points)
        evaluation = phx.meshing.evaluate_cell_quality(mesh)
        sampled = float(evaluation.mean_ratios[0])
        for threshold in _near_thresholds(sampled):
            report = phx.meshing.summarize_cell_quality(
                mesh, evaluation, minimum_mean_ratio=threshold
            )
            assert bool(report.mean_ratio_failure_ids) is oracle(points, threshold)


def test_inverted_corner_fails_every_positive_mean_ratio_threshold() -> None:
    points = _reference("quadrilateral")
    points[2] = (0.2, 0.2)  # reflex corner opposite vertex 0
    mesh = _single("quadrilateral", points)
    evaluation = phx.meshing.evaluate_cell_quality(mesh)
    assert phx.meshing.summarize_cell_quality(
        mesh, evaluation, minimum_mean_ratio=1.0e-300
    ).mean_ratio_failure_ids
    assert not phx.meshing.summarize_cell_quality(
        mesh, evaluation, minimum_mean_ratio=0.0
    ).mean_ratio_failure_ids


def test_measure_gate_is_inclusive_and_exact() -> None:
    triangle = _single("triangle", ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    evaluation = phx.meshing.evaluate_cell_quality(triangle)
    assert phx.meshing.summarize_cell_quality(
        triangle, evaluation, minimum_measure=0.5
    ).measure_failure_ids == (0,)
    assert not phx.meshing.summarize_cell_quality(
        triangle, evaluation, minimum_measure=float(np.nextafter(0.5, 0.0))
    ).measure_failure_ids
    # The tetrahedron volume 1/6 is not a binary64 value: the exact comparison
    # against the threshold nearest 1/6 decides on the side of the true volume.
    tetrahedron = _single("tetrahedron", _reference("tetrahedron"))
    evaluation = phx.meshing.evaluate_cell_quality(tetrahedron)
    nearest = 1.0 / 6.0
    for threshold in _near_thresholds(nearest):
        report = phx.meshing.summarize_cell_quality(
            tetrahedron, evaluation, minimum_measure=threshold
        )
        assert bool(report.measure_failure_ids) is (Fraction(1, 6) <= Fraction(threshold))


def test_multilinear_volume_matches_independent_jacobian_integral() -> None:
    rng = np.random.default_rng(11)
    for _ in range(6):
        points = _reference("hexahedron") + rng.normal(scale=0.08, size=(8, 3))
        mesh = _single("hexahedron", points)
        evaluation = phx.meshing.evaluate_cell_quality(mesh)
        volume = _hexahedron_volume(points)
        assert float(volume) == pytest.approx(float(evaluation.measures[0]), rel=1e-13)
        for threshold in _near_thresholds(float(volume)):
            report = phx.meshing.summarize_cell_quality(
                mesh, evaluation, minimum_measure=threshold
            )
            assert bool(report.measure_failure_ids) is (volume <= Fraction(threshold))
    for kind in ("prism", "pyramid"):
        points = _reference(kind) + rng.normal(scale=0.05, size=_reference(kind).shape)
        mesh = _single(kind, points)
        evaluation = phx.meshing.evaluate_cell_quality(mesh)
        sampled = float(evaluation.measures[0])
        assert phx.meshing.summarize_cell_quality(
            mesh, evaluation, minimum_measure=sampled * (1.0 + 1.0e-12)
        ).measure_failure_ids
        assert not phx.meshing.summarize_cell_quality(
            mesh, evaluation, minimum_measure=sampled * (1.0 - 1.0e-12)
        ).measure_failure_ids


def test_aspect_gate_compares_squared_lengths_exactly() -> None:
    rectangle = _single("quadrilateral", ((0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (0.0, 1.0)))
    evaluation = phx.meshing.evaluate_cell_quality(rectangle)
    assert not phx.meshing.summarize_cell_quality(
        rectangle, evaluation, maximum_aspect_ratio=3.0
    ).aspect_ratio_failure_ids
    assert phx.meshing.summarize_cell_quality(
        rectangle, evaluation, maximum_aspect_ratio=float(np.nextafter(3.0, 0.0))
    ).aspect_ratio_failure_ids


@pytest.mark.parametrize("arity", (5, 7, 8, 9, 10, 12))
def test_ideal_polygon_cosine_enclosure_is_rigorous(arity: int) -> None:
    lower, upper = _quality._polygon_cosine(arity, 128)
    target = -math.cos(2.0 * math.pi / arity)
    assert lower <= Fraction(target) + Fraction(1, 1 << 50)
    assert upper >= Fraction(target) - Fraction(1, 1 << 50)
    assert 0 < upper - lower < Fraction(1, 1 << 120)
    if arity in _quality._QUADRATIC_POLYGON_COSINES:
        a, b, c = _quality._QUADRATIC_POLYGON_COSINES[arity]
        values = (a * lower * lower + b * lower + c, a * upper * upper + b * upper + c)
        assert min(values) < 0 < max(values)


def test_polygon_ratio_decides_an_exact_algebraic_tie() -> None:
    # Octagon corner Gram diag(1, 2) at r = 2/3: P(c) = 4 (2 c^2 - 1) vanishes at
    # the ideal cosine -sqrt(2)/2, so the ratio equals r and is not below it.
    gram = ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(2)))
    tie = Fraction(2, 3)
    assert not _quality._polygon_ratio_fails(gram, Fraction(2), tie, 8)
    assert _quality._polygon_ratio_fails(gram, Fraction(2), tie + Fraction(1, 10**30), 8)
    assert not _quality._polygon_ratio_fails(
        gram, Fraction(2), tie - Fraction(1, 10**30), 8
    )


def _single_polygon(points: Any, /) -> Any:
    points = np.asarray(points, dtype=np.float64)
    return phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "p", "polygon", np.arange(len(points), dtype=np.int32)[None, :]
            ),
        ),
    )


@pytest.mark.parametrize("arity", (5, 6, 7, 8))
def test_polygon_mean_ratio_gate_brackets_the_sampled_ratio(arity: int) -> None:
    rng = np.random.default_rng(arity)
    angles = 2.0 * np.pi * np.arange(arity) / arity
    regular = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    for _ in range(6):
        mesh = _single_polygon(regular + rng.normal(scale=0.04, size=regular.shape))
        evaluation = phx.meshing.evaluate_cell_quality(mesh)
        sampled = float(evaluation.mean_ratios[0])
        for factor in (1.0 - 1.0e-9, 1.0 + 1.0e-9):
            report = phx.meshing.summarize_cell_quality(
                mesh, evaluation, minimum_mean_ratio=sampled * factor
            )
            assert bool(report.mean_ratio_failure_ids) is (factor > 1.0)


def test_hexagon_mean_ratio_matches_fraction_oracle() -> None:
    points = np.asarray(
        ((1.0, 0.0), (0.5, 0.75), (-0.5, 0.75), (-1.0, 0.0), (-0.5, -0.75), (0.5, -0.75))
    )
    mesh = _single_polygon(points)
    evaluation = phx.meshing.evaluate_cell_quality(mesh)
    for threshold in _near_thresholds(float(evaluation.mean_ratios[0])):
        report = phx.meshing.summarize_cell_quality(
            mesh, evaluation, minimum_mean_ratio=threshold
        )
        assert bool(report.mean_ratio_failure_ids) is _hexagon_ratio_below(
            points, threshold
        )


def _hexagon_ratio_below(points: np.ndarray, threshold: float, /) -> bool:
    """Hexagon corners relative to unit-edge ideal Gram [[1, -1/2], [-1/2, 1]]."""
    p = _exact(points)
    r = Fraction(threshold)
    for vertex in range(6):
        e0 = _sub(p[(vertex + 1) % 6], p[vertex])
        e1 = _sub(p[(vertex - 1) % 6], p[vertex])
        det = e0[0] * e1[1] - e0[1] * e1[0]
        if det <= 0:
            return True
        g00, g01, g11 = _dot(e0, e0), _dot(e0, e1), _dot(e1, e1)
        # q = 2 sqrt(det C / det K) / tr(C K^-1), det K = 3/4,
        # K^-1 = (4/3) [[1, 1/2], [1/2, 1]].
        trace = Fraction(4, 3) * (g00 + g11 + g01)
        if 4 * det * det / Fraction(3, 4) < r * r * trace * trace:
            return True
    return False


def test_polyhedral_gates_match_star_tetrahedra() -> None:
    rng = np.random.default_rng(5)
    cube = np.asarray(list(product((0.0, 1.0), repeat=3)))
    points = cube + rng.normal(scale=0.03, size=cube.shape)
    faces = (
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    )
    mesh = phx.discretization.CellMesh.from_polyhedra(points, (faces,))
    evaluation = phx.meshing.evaluate_cell_quality(mesh)
    measure = float(evaluation.measures[0])
    ratio = float(evaluation.mean_ratios[0])
    aspect = float(evaluation.aspect_ratios[0])
    for factor, fails in ((1.0 - 1.0e-9, False), (1.0 + 1.0e-9, True)):
        report = phx.meshing.summarize_cell_quality(
            mesh,
            evaluation,
            minimum_measure=measure * factor,
            minimum_mean_ratio=ratio * factor,
            maximum_aspect_ratio=aspect / factor,
        )
        assert bool(report.measure_failure_ids) is fails
        assert bool(report.mean_ratio_failure_ids) is fails
        assert bool(report.aspect_ratio_failure_ids) is fails


# Identity ------------------------------------------------------------------------------


_DIAGNOSTICS = (
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
)


def _ulp_perturbed(evaluation: Any, /) -> Any:
    """Simulate another platform's XLA rounding: every diagnostic moves one ulp."""
    return replace(
        evaluation,
        **{
            name: np.nextafter(
                np.asarray(getattr(evaluation, name), dtype=np.float64), np.inf
            )
            for name in _DIAGNOSTICS
        },
    )


def _two_triangles() -> Any:
    return phx.discretization.CellMesh.from_simplices(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)), dtype=np.float64),
        np.asarray(((0, 1, 2), (1, 3, 2)), dtype=np.int32),
        dimension=2,
    )


def test_ulp_perturbed_diagnostics_keep_every_identity(monkeypatch: Any) -> None:
    mesh = _two_triangles()
    contract = phx.SpatialCoordinateContract.si()
    reference = phx.meshing.certify_cell_mesh(mesh, contract)
    evaluate = _audit.evaluate_cell_quality
    monkeypatch.setattr(
        _audit,
        "evaluate_cell_quality",
        lambda *args, **kwargs: _ulp_perturbed(evaluate(*args, **kwargs)),
    )
    perturbed = phx.meshing.certify_cell_mesh(mesh, contract)
    assert not np.array_equal(
        np.asarray(perturbed.quality.evaluation.mean_ratios),
        np.asarray(reference.quality.evaluation.mean_ratios),
    )
    assert perturbed.quality.report_id == reference.quality.report_id
    assert perturbed.audit.report_id == reference.audit.report_id
    assert perturbed.trace.trace_id == reference.trace.trace_id
    assert perturbed.provenance.semantic_id == reference.provenance.semantic_id
    assert perturbed.result_id == reference.result_id


def test_changed_gate_decision_changes_report_and_audit_identity() -> None:
    mesh = _two_triangles()
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    evaluation = phx.meshing.evaluate_cell_quality(mesh)
    # Both right triangles have mean ratio sqrt(3)/2 exactly.
    ratio = math.sqrt(3.0) / 2.0
    below, above = float(np.nextafter(ratio, 0.0)), float(np.nextafter(ratio, 1.0))
    passed, failed = (
        phx.meshing.summarize_cell_quality(mesh, evaluation, minimum_mean_ratio=value)
        for value in (below, above)
    )
    assert passed.mean_ratio_failure_ids == ()
    assert failed.mean_ratio_failure_ids == (0, 1)
    assert failed.failed_gates == ("minimum_mean_ratio",)
    # The decided failure set alone separates the identities.
    tampered = CellQualityReport(
        evaluation,
        mesh_id=passed.mesh_id,
        gate_thresholds=passed.gate_thresholds,
        measure_failure_ids=(),
        mean_ratio_failure_ids=(0,),
        aspect_ratio_failure_ids=(),
    )
    assert len({passed.report_id, failed.report_id, tampered.report_id}) == 3
    audits = [
        phx.meshing.audit_cell_mesh(
            mesh,
            geometry,
            policy=phx.meshing.CellMeshAuditPolicy(minimum_mean_ratio=value),
        )
        for value in (below, above)
    ]
    assert audits[0].passed and not audits[1].passed
    assert audits[1].issues == ("minimum_mean_ratio",)
    assert audits[0].report_id != audits[1].report_id


def test_stored_report_rebuilds_exact_decisions_and_rejects_tampering() -> None:
    mesh = _two_triangles()
    evaluation = phx.meshing.evaluate_cell_quality(mesh)
    report = phx.meshing.summarize_cell_quality(mesh, evaluation, minimum_mean_ratio=0.9)
    restored = CellQualityReport.from_stored_fields(report)
    assert restored.report_id == report.report_id
    assert restored.mean_ratio_failure_ids == (0, 1)
    recipe = CellQualityReportArchiveRecipe.from_report(report)
    with pytest.raises(ValueError, match="differs from recomputed"):
        CellQualityReport.from_stored_fields(recipe._replace(mean_ratio_failure_ids=(0,)))
    with pytest.raises(ValueError, match="not bound"):
        phx.meshing.summarize_cell_quality(
            _single("triangle", ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))), evaluation
        )


# Exact scaled-Jacobian gate -------------------------------------------------------


def _scaled_below(kind: str, points: np.ndarray, threshold: float, /) -> bool:
    """Corner det(F) / (|f1| |f2| |f3| s_K) < s; s_K^2 is 1 (box) or 1/2 (regular)."""
    vertices, neighbors = reference_corner_frames(kind)
    p = _exact(points)
    ideal = Fraction(1) if kind == "hexahedron" else Fraction(1, 2)
    bound = Fraction(threshold) ** 2 * ideal
    for vertex, row in zip(vertices, neighbors, strict=True):
        frames = [_sub(p[int(n)], p[int(vertex)]) for n in row]
        det = _det3(*frames)
        lengths = math.prod(_dot(frame, frame) for frame in frames)
        if det <= 0 or det * det < bound * lengths:
            return True
    return False


def _jacobian_gates(kind: str, points: Any, threshold: float, /) -> Any:
    return _quality.exact_cell_gate_failures(
        kind,
        np.asarray(points, dtype=np.float64)[None],
        minimum_measure=0.0,
        minimum_scaled_jacobian=threshold,
        minimum_mean_ratio=0.0,
        maximum_aspect_ratio=math.inf,
    )


@pytest.mark.parametrize("kind", ("hexahedron", "tetrahedron"))
def test_scaled_jacobian_gate_is_exact_at_the_sampled_threshold(kind: str) -> None:
    rng = np.random.default_rng(11)
    reference = _reference(kind)
    for _ in range(12):
        points = reference + rng.normal(scale=0.07, size=reference.shape)
        evaluation = phx.meshing.evaluate_cell_quality(_single(kind, points))
        sampled = float(evaluation.scaled_jacobian[0])
        for threshold in _near_thresholds(sampled):
            gates = _jacobian_gates(kind, points, threshold)
            assert bool(gates.scaled_jacobian[0]) is _scaled_below(
                kind, points, threshold
            )
            assert not gates.nonpositive_corner[0]


def test_scaled_jacobian_gate_decides_the_cube_tie_and_inverted_corners() -> None:
    cube = _reference("hexahedron")
    sampled = float(
        phx.meshing.evaluate_cell_quality(_single("hexahedron", cube)).scaled_jacobian[0]
    )
    # One ulp low on another platform, the former float rule rejected the cube.
    assert float(np.nextafter(sampled, 0.0)) < 1.0
    assert not _jacobian_gates("hexahedron", cube, 1.0).scaled_jacobian[0]
    assert _jacobian_gates(
        "hexahedron", cube, float(np.nextafter(1.0, 2.0))
    ).scaled_jacobian[0]
    vertices, neighbors = reference_corner_frames("hexahedron")
    corner = int(vertices[0])
    inverted = cube.copy()
    # Pull the corner through the plane of its three neighbours.
    inverted[corner] = (
        1.5 * np.mean(cube[np.asarray(neighbors[0])], axis=0) - 0.5 * (cube[corner])
    )
    p = _exact(inverted)
    frames = [_sub(p[int(n)], p[corner]) for n in neighbors[0]]
    assert _det3(*frames) <= 0
    gates = _jacobian_gates("hexahedron", inverted, 1.0e-300)
    assert gates.nonpositive_corner[0] and gates.scaled_jacobian[0]
    with pytest.raises(ValueError, match="finite and positive"):
        _jacobian_gates("hexahedron", cube, 0.0)
