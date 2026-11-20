# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Differential surface checks against the Python exact chart partition.

The native oriented-chain condition deliberately strengthens the Python reference:
that reference uses absolute per-triangle area and accepts a single flipped row.
"""

from __future__ import annotations

import ctypes
from collections import defaultdict
from fractions import Fraction
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from phydrax._meshcore import load_meshcore
from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry_validity import certify_cell_geometry_validity
from phydrax.geometry._exact_polyhedral_geometry import triangle_contact_points3
from phydrax.geometry._global_embedding import certify_global_embedding
from phydrax.geometry._source_fidelity import _require_reference_chart_partition


type _Point2 = tuple[Fraction, Fraction]
type _Triangle2 = tuple[_Point2, _Point2, _Point2]
type _Point3 = tuple[Fraction, Fraction, Fraction]


class _ChartFailure(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int32), ("patch", ctypes.c_int64),
                ("row", ctypes.c_int64), ("other", ctypes.c_int64)]


def _chart_verdict(uv: NDArray[np.float64]) -> bool:
    function = load_meshcore()["phx_mc_verify_chart_partition"]
    uv = np.ascontiguousarray(uv, dtype=np.float64)
    triangle_patch = np.zeros(len(uv), dtype=np.int32)
    use_patch = np.zeros(4, dtype=np.int64)
    use_range = np.tile(np.asarray((0.0, 1.0), dtype=np.float64), (4, 1))
    chord_use = np.arange(4, dtype=np.int64)
    chord_loop = np.zeros(4, dtype=np.int64)
    chord_uv = np.asarray((((0, 0), (1, 0)), ((1, 0), (1, 1)),
                           ((1, 1), (0, 1)), ((0, 1), (0, 0))), dtype=np.float64)
    evidence = (_ChartFailure * 128)()
    count = ctypes.c_int64()
    status = function(1, len(uv), triangle_patch.ctypes.data, uv.ctypes.data,
                      4, use_patch.ctypes.data, use_range.ctypes.data, 1, 4,
                      chord_use.ctypes.data, chord_loop.ctypes.data, chord_uv.ctypes.data,
                      use_range.ctypes.data, 128, ctypes.addressof(evidence),
                      ctypes.addressof(count))
    if status != 0:
        raise RuntimeError(f"chart verifier refused input: {status}")
    return count.value == 0


def _triangle_rows(uv: NDArray[np.float64]) -> tuple[_Triangle2, ...]:
    def point(row: NDArray[np.float64]) -> _Point2:
        return Fraction.from_float(float(row[0])), Fraction.from_float(float(row[1]))

    return tuple((point(row[0]), point(row[1]), point(row[2])) for row in uv)


def _python_chart_verdict(uv: NDArray[np.float64], domain: NDArray[np.float64]) -> bool:
    try:
        _require_reference_chart_partition(_triangle_rows(uv), _triangle_rows(domain))
    except ValueError:
        return False
    return True


def _signed_area(triangle: NDArray[np.float64]) -> Fraction:
    a, b, c = _triangle_rows(triangle[np.newaxis, :, :])[0]
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def test_chart_partition_differential() -> None:
    good = np.asarray((((0, 0), (1, 0), (1, 1)),
                       ((0, 0), (1, 1), (0, 1))), dtype=np.float64)
    missing = good[:1].copy()
    perturbed = good.copy()
    perturbed[1, 1, 0] = 0.75
    folded = np.asarray((good[0], good[0, ::-1]), dtype=np.float64)
    for candidate, expected in ((good, True), (missing, False),
                                (perturbed, False), (folded, False)):
        assert _python_chart_verdict(candidate, good) is expected
        assert _chart_verdict(candidate) is expected

    flipped = good.copy()
    flipped[1] = flipped[1, ::-1]
    assert _signed_area(flipped[0]) > 0 > _signed_area(flipped[1])
    assert _python_chart_verdict(flipped, good)
    assert not _chart_verdict(flipped)


class _SurfaceFinding(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int32), ("first", ctypes.c_int64),
                ("second", ctypes.c_int64), ("classification", ctypes.c_int32)]


class _SurfaceVerdict(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int32), ("finding_count", ctypes.c_int64)]


def _surface_verdict(points: NDArray[np.float64], triangles: NDArray[np.int64]) -> bool:
    function = load_meshcore()["phx_mc_verify_surface_embedding"]
    points = np.ascontiguousarray(points, dtype=np.float64)
    triangles = np.ascontiguousarray(triangles, dtype=np.int64)
    evidence = (_SurfaceFinding * 512)()
    verdict = _SurfaceVerdict()
    status = function(len(points), points.ctypes.data, len(triangles), triangles.ctypes.data,
                      512, ctypes.addressof(evidence), ctypes.addressof(verdict))
    if status != 0:
        raise RuntimeError(f"surface verifier refused input: {status}")
    return verdict.finding_count == 0


def _reference_surface(points: NDArray[np.float64], triangles: NDArray[np.int64]) -> bool:
    exact: tuple[_Point3, ...] = tuple(
        (Fraction.from_float(float(p[0])), Fraction.from_float(float(p[1])),
         Fraction.from_float(float(p[2]))) for p in points
    )
    edges: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for row in triangles:
        a, b, c = (int(row[0]), int(row[1]), int(row[2]))
        for x, y in ((a, b), (b, c), (c, a)):
            edges[min(x, y), max(x, y)].append((x, y))
    if any(len(rows) != 2 or rows[0] != rows[1][::-1] for rows in edges.values()):
        return False
    for first, second in combinations(range(len(triangles)), 2):
        ids1 = (int(triangles[first, 0]), int(triangles[first, 1]), int(triangles[first, 2]))
        ids2 = (int(triangles[second, 0]), int(triangles[second, 1]), int(triangles[second, 2]))
        common = set(ids1) & set(ids2)
        contacts = triangle_contact_points3(
            (exact[ids1[0]], exact[ids1[1]], exact[ids1[2]]),
            (exact[ids2[0]], exact[ids2[1]], exact[ids2[2]]),
        )
        if set(contacts) != {exact[i] for i in common}:
            return False
    return True


def test_surface_embedding_differential() -> None:
    points = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
                        dtype=np.float64)
    triangles = np.asarray(((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)),
                           dtype=np.int64)
    flipped = triangles.copy()
    flipped[0] = flipped[0, ::-1]
    nonmanifold = np.concatenate((triangles, triangles[:1]))
    for rows, expected in ((triangles, True), (triangles[:-1], False),
                           (flipped, False), (nonmanifold, False)):
        assert _reference_surface(points, rows) is expected
        assert _surface_verdict(points, rows) is expected

    crossing_points = np.concatenate((points, points + np.asarray((0.25, 0.25, 0.25))))
    crossing_triangles = np.concatenate((triangles, triangles + 4))
    assert not _reference_surface(crossing_points, crossing_triangles)
    assert not _surface_verdict(crossing_points, crossing_triangles)


def _python_embedding(points: NDArray[np.float64], triangles: NDArray[np.int64]) -> bool:
    mesh = CellMesh(points, (CellBlock("triangles", "triangle", triangles.astype(np.int32)),))
    geometry = CellGeometrySpec.affine(mesh)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    return certify_global_embedding(mesh, geometry, validity).status == "certified"


def test_sphere_and_crossing_python_embedding() -> None:
    points = np.asarray(((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0),
                         (0, 0, 1), (0, 0, -1)), dtype=np.float64)
    triangles = np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4),
                            (1, 0, 5), (2, 1, 5), (3, 2, 5), (0, 3, 5)),
                           dtype=np.int64)
    assert _python_embedding(points, triangles)
    assert _surface_verdict(points, triangles)
    crossing_points = np.concatenate((points, points + np.asarray((0.5, 0.25, 0.125))))
    crossing_triangles = np.concatenate((triangles, triangles + 6))
    assert not _python_embedding(crossing_points, crossing_triangles)
    assert not _surface_verdict(crossing_points, crossing_triangles)
