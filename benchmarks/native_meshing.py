"""Native meshing kernels against the measured best reference per row.

Run as `python -m benchmarks.native_meshing` from the repository root. Each row times one native kernel on a fixed generated input, single threaded, with
host timing (the kernels return host arrays). The ceiling of a row is the best
reference time measured on the development machine for the same input
(`native_meshing_ceilings.json`, with its provenance); a row passes when the
kernel's minimum time is at most its ceiling times the row's allowed ratio.
Ratios above one are interim acceptance targets of the plan
`docs/plans/native-meshing-kernels.md`; the final target of every row is 1.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from benchmarks._runtime import capture_environment, measure_host
from phydrax import _meshcore
from phydrax.geometry._interval_enclosure import prepare_interval_function


_CEILINGS = Path(__file__).with_name("native_meshing_ceilings.json")


def _none_unmet(result: object, /) -> int:
    return 0


def _no_quality(result: object, /) -> tuple[int, dict[str, float]]:
    return 0, {}


@dataclass(frozen=True, slots=True)
class _Row:
    name: str
    size: int
    run: Callable[[], object]
    count: Callable[[object], int]
    # Cells left off the row's quality targets; a row with any does not pass.
    unmet: Callable[[object], int] = _none_unmet
    # Independent measurement of the last result (not timed): the number of
    # targets it misses, and the measured values.
    quality: Callable[[object], tuple[int, dict[str, float]]] = _no_quality


def _uniform_points(count: int, dimension: int, seed: int, /) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.random((count, dimension), dtype=np.float64)


def _cells(result: object, /) -> int:
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("A Delaunay kernel returns (cells, regions).")
    cells = result[0]
    if not isinstance(cells, np.ndarray):
        raise TypeError("Delaunay cells must be a NumPy array.")
    return cells.shape[0]


def _delaunay_rows(sizes: tuple[int, ...], /) -> tuple[_Row, ...]:
    rows: list[_Row] = []
    for size in sizes:
        planar = _uniform_points(size, 2, 20261119)
        spatial = _uniform_points(size, 3, 20261119)
        rows.append(
            _Row(
                "delaunay-2d",
                size,
                lambda points=planar: _meshcore.delaunay_2d(points),
                _cells,
            )
        )
        rows.append(
            _Row(
                "delaunay-3d",
                size,
                lambda points=spatial: _meshcore.delaunay_3d(points),
                _cells,
            )
        )
    return tuple(rows)


def _rows(sizes: tuple[int, ...], names: frozenset[str] | None, /) -> tuple[_Row, ...]:
    # Families prepare their inputs only when one of their rows is selected.
    families: tuple[tuple[frozenset[str], Callable[[], tuple[_Row, ...]]], ...] = (
        (frozenset({"delaunay-2d", "delaunay-3d"}), lambda: _delaunay_rows(sizes)),
        (frozenset({"implicit-sphere", "implicit-torus"}), _implicit_rows),
        (frozenset({"recover-sphere", "recover-torus"}), _recovery_rows),
        (frozenset({"surface-remesh-sphere", "surface-remesh-torus"}), _remesh_rows),
        (
            frozenset(
                {
                    "surface-to-tets-sphere",
                    "surface-to-tets-torus",
                    "surface-to-tets-grid-torus",
                }
            ),
            _volume_rows,
        ),
    )
    rows: list[_Row] = []
    for family, build in families:
        if names is None or family & names:
            rows.extend(row for row in build() if names is None or row.name in names)
    return tuple(rows)


def _icosphere(level: int, /) -> tuple[np.ndarray, np.ndarray]:
    # The ledger's unit icosphere: 20 * 4**level outward triangles.
    golden = (1.0 + 5.0**0.5) / 2.0
    vertices = np.asarray(
        [
            [-1, golden, 0], [1, golden, 0], [-1, -golden, 0], [1, -golden, 0],
            [0, -1, golden], [0, 1, golden], [0, -1, -golden], [0, 1, -golden],
            [golden, 0, -1], [golden, 0, 1], [-golden, 0, -1], [-golden, 0, 1],
        ],
        dtype=np.float64,
    )  # fmt: skip
    faces = np.asarray(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9],
            [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2],
            [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
            [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )  # fmt: skip
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    for _ in range(level):
        edges = np.sort(
            np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1
        )
        unique, inverse = np.unique(edges, axis=0, return_inverse=True)
        middle = vertices[unique[:, 0]] + vertices[unique[:, 1]]
        middle /= np.linalg.norm(middle, axis=1, keepdims=True)
        count = faces.shape[0]
        inverse = inverse.reshape((-1,)) + vertices.shape[0]
        a, b, c = inverse[:count], inverse[count : 2 * count], inverse[2 * count :]
        vertices = np.concatenate((vertices, middle))
        faces = np.concatenate(
            (
                np.stack((faces[:, 0], a, c), axis=1),
                np.stack((faces[:, 1], b, a), axis=1),
                np.stack((faces[:, 2], c, b), axis=1),
                np.stack((a, b, c), axis=1),
            )
        )
    return vertices, faces


def _uv_torus(around: int, /) -> tuple[np.ndarray, np.ndarray]:
    # The ledger's torus R = 1, r = 0.4 with an (around x around / 2) grid.
    across = around // 2
    u = 2.0 * np.pi * np.arange(around) / around
    w = 2.0 * np.pi * np.arange(across) / across
    grid_u, grid_w = np.meshgrid(u, w, indexing="ij")
    ring = 1.0 + 0.4 * np.cos(grid_w)
    vertices = np.stack(
        (ring * np.cos(grid_u), ring * np.sin(grid_u), 0.4 * np.sin(grid_w)), axis=-1
    ).reshape((-1, 3))
    i, j = np.meshgrid(np.arange(around), np.arange(across), indexing="ij")
    a = (i * across + j).reshape((-1,))
    b = (((i + 1) % around) * across + j).reshape((-1,))
    c = (((i + 1) % around) * across + (j + 1) % across).reshape((-1,))
    d = (i * across + (j + 1) % across).reshape((-1,))
    faces = np.concatenate((np.stack((a, b, c), axis=1), np.stack((a, c, d), axis=1)))
    return vertices, faces


def _recover(
    vertices: np.ndarray, faces: np.ndarray, facets: np.ndarray, /
) -> _meshcore.PlcRecovery3D:
    # The surface's triangles are polygons of the given PLC facets, all
    # bounding region 0.
    count = faces.shape[0]
    facet_count = int(facets.max()) + 1
    return _meshcore.recover_plc_3d(
        vertices,
        np.arange(0, 3 * count + 1, 3, dtype=np.int64),
        faces.reshape((-1,)).astype(np.int32),
        facets.astype(np.int32),
        np.tile(np.asarray([[-1, 0]], dtype=np.int32), (facet_count, 1)),
        boundary_policy=_meshcore.PlcBoundaryPolicy.FIXED,
    )


def _facets(vertices: np.ndarray, faces: np.ndarray, /) -> np.ndarray:
    # One facet for the whole surface (as an imported triangulated surface
    # is), unless recovery would then retriangulate coplanar neighbors (the
    # grid torus's quads are planar trapezoids): the reference tool keeps the
    # given triangles, so each triangle is then its own facet. Checked once,
    # outside the timing.
    single = np.zeros((faces.shape[0],), dtype=np.int32)
    recovered = _recover(vertices, faces, single)
    _recovered_cells(recovered)
    boundary = np.unique(
        np.sort(np.asarray(recovered.faces).reshape((-1, 3)), axis=1), axis=0
    )
    if np.array_equal(boundary, np.unique(np.sort(faces, axis=1), axis=0)):
        return single
    return np.arange(faces.shape[0], dtype=np.int32)


def _recovered_cells(result: object, /) -> int:
    if not isinstance(result, _meshcore.PlcRecovery3D):
        raise TypeError("Boundary recovery returns a PlcRecovery3D.")
    if result.status != _meshcore.MeshcoreStatus.OK:
        raise RuntimeError(f"Boundary recovery returned {result.status!r}.")
    return result.tetrahedra.shape[0]


def _recovery_rows() -> tuple[_Row, ...]:
    # Closed surfaces to a constrained tetrahedralization with the boundary
    # fixed; the row size is the triangle count.
    rows: list[_Row] = []
    for level in (4, 5, 6, 7):
        vertices, faces = _icosphere(level)
        facets = _facets(vertices, faces)
        rows.append(
            _Row(
                "recover-sphere",
                faces.shape[0],
                lambda vertices=vertices, faces=faces, facets=facets: _recover(
                    vertices, faces, facets
                ),
                _recovered_cells,
            )
        )
    for around in (100, 300):
        vertices, faces = _uv_torus(around)
        facets = _facets(vertices, faces)
        rows.append(
            _Row(
                "recover-torus",
                faces.shape[0],
                lambda vertices=vertices, faces=faces, facets=facets: _recover(
                    vertices, faces, facets
                ),
                _recovered_cells,
            )
        )
    return tuple(rows)


class _VolumeResult(NamedTuple):
    cells: int
    unmet: int


def _surface_to_tets(
    vertices: np.ndarray, faces: np.ndarray, facets: np.ndarray, size: float, /
) -> _VolumeResult:
    # The whole task Gmsh's generate(3) performs from a surface: boundary
    # recovery, state creation (adopted from the recovery), refinement,
    # improvement and export.
    recovered = _recover(vertices, faces, facets)
    _recovered_cells(recovered)
    mesh = _meshcore.TetMesh3D.from_recovery(
        recovered, boundary_policy=_meshcore.PlcBoundaryPolicy.FIXED
    )
    sizes = np.full((recovered.points.shape[0],), size, dtype=np.float64)
    refined = mesh.refine(radius_edge_bound=2.0, sizes=sizes)
    # UNRESOLVED leaves cells for improvement; any other status is a failure.
    if refined.status not in (
        _meshcore.MeshcoreStatus.OK,
        _meshcore.MeshcoreStatus.UNRESOLVED,
    ):
        raise RuntimeError(f"Refinement failed with status {refined.status!r}.")
    improved = mesh.improve(radius_edge_bound=2.0, min_dihedral_degrees=15.0)
    return _VolumeResult(mesh.arrays().tetrahedra.shape[0], improved.unmet_cells)


def _stuffed_torus_surface(spacing: float, /) -> tuple[np.ndarray, np.ndarray]:
    # A well-shaped closed torus surface (R = 1, r = 0.4): the outward boundary
    # of the isosurface-stuffed solid at the given lattice spacing.
    def torus(x: Array) -> Array:
        ring = jnp.sqrt(x[0] * x[0] + x[1] * x[1]) - 1.0
        return jnp.stack((jnp.sqrt(ring * ring + x[2] * x[2]) - 0.4,))

    program = prepare_interval_function(
        torus, 3, batch_capacity=1, execution="batched"
    ).native_program
    if program is None:
        raise RuntimeError("The torus rows need the native interval program.")
    upper = np.asarray((1.5, 1.5, 0.5), dtype=np.float64)
    stuffed = program.stuff_tetrahedra(-upper, upper, spacing)
    cells = stuffed.cells
    faces = np.concatenate(
        (
            cells[:, [1, 2, 3]],
            cells[:, [0, 3, 2]],
            cells[:, [0, 1, 3]],
            cells[:, [0, 2, 1]],
        )
    )
    _, inverse, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    boundary = faces[counts[inverse.reshape((-1,))] == 1]
    used, local = np.unique(boundary, return_inverse=True)
    return stuffed.points[used], local.reshape(boundary.shape).astype(np.int64)


def _volume_rows() -> tuple[_Row, ...]:
    # Surface to tetrahedra end to end (recovery, state creation, refinement to
    # radius-edge 2, improvement to radius-edge 2 and dihedral 15 degrees,
    # export) with the boundary fixed, against Gmsh HXT generate(3) on the same
    # surface. The circumradius target 1.75 sqrt(3/8) h, h the mean surface
    # edge, matches HXT's element counts. The sphere and grid torus row sizes
    # are the triangle counts, the stuffed torus row size the stuffing spacing
    # in thousandths. The grid torus (cocircular rows, every edge constrained)
    # is a hard input for both tools.
    rows: list[_Row] = []
    shapes = (
        ("surface-to-tets-sphere", faces.shape[0], (vertices, faces))
        for vertices, faces in (_icosphere(5), _icosphere(6))
    )
    tori = (
        ("surface-to-tets-torus", thousandths, _stuffed_torus_surface(1e-3 * thousandths))
        for thousandths in (67, 22)
    )
    grids = (
        ("surface-to-tets-grid-torus", faces.shape[0], (vertices, faces))
        for vertices, faces in (_uv_torus(100), _uv_torus(300))
    )
    for name, size, (vertices, faces) in (*shapes, *tori, *grids):
        edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        spacing = float(
            np.mean(np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1))
        )
        target = 1.75 * math.sqrt(3.0 / 8.0) * spacing
        facets = _facets(vertices, faces)
        rows.append(
            _Row(
                name,
                size,
                lambda vertices=vertices, faces=faces, facets=facets, target=target: (
                    _surface_to_tets(vertices, faces, facets, target)
                ),
                _volume_cells,
                _volume_unmet,
            )
        )
    return tuple(rows)


def _volume_cells(result: object, /) -> int:
    if not isinstance(result, _VolumeResult):
        raise TypeError("The volume row returns a _VolumeResult.")
    return result.cells


def _volume_unmet(result: object, /) -> int:
    if not isinstance(result, _VolumeResult):
        raise TypeError("The volume row returns a _VolumeResult.")
    return result.unmet


def _implicit_rows() -> tuple[_Row, ...]:
    # The external comparison's sphere (radius 0.371) and torus (major 0.742,
    # minor 0.371, i.e. inner 0.371 and outer 1.113) through the analytic
    # profile route's construction: stuffing at the route's spacing, lattice
    # acceptance (exact volume certificate, topology, certified facet
    # deviation) and the fixed-boundary quality pass when the stuffed cells
    # need it, ending at the constructed arrays as the references end at their
    # native meshes. Publication as a CellMesh and certification are not
    # timed. The row size is the tolerance in thousandths.
    from phydrax._physical import SpatialCoordinateContract
    from phydrax.geometry import Sphere, Torus
    from phydrax.geometry.implicit import AnalyticImplicitProfile
    from phydrax.meshing import (
        CellFamilyPolicy,
        CellMeshingTarget,
        VolumeFillStrategy,
        VolumeMeshingSpec,
    )
    from phydrax.meshing._implicit_volume import PreparedStuffedImplicitVolume
    from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
    from phydrax.meshing._volume_generation import NativeVolumeSchedule
    from phydrax.meshing.providers._native_implicit import prepare_native_implicit
    from phydrax.meshing.providers._native_options import NativeMeshingOptions

    contract = SpatialCoordinateContract.si()
    rows: list[_Row] = []
    for name, geometry in (
        ("implicit-sphere", Sphere((0.0, 0.0, 0.0), 0.371, feature_id="sphere")),
        ("implicit-torus", Torus((0.0, 0.0, 0.0), 0.371, 1.113, feature_id="torus")),
    ):
        profile = AnalyticImplicitProfile(
            geometry.compile(), contract, source_id=name, source_revision="benchmark"
        )
        scope = MeshingScope(
            profile.source_id,
            profile.source_revision,
            MeshingEntityKind.GEOMETRY,
            2,
            profile.scope_entity_set_id(2),
            profile.scope_entity_ids(2),
        )
        for thousandths in (20, 10):
            specification = VolumeMeshingSpec(
                CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
                scope,
                VolumeFillStrategy.SIMPLEX,
                size_controls=(),
                maximum_deviation=1e-3 * thousandths,
            )
            prepared = prepare_native_implicit(
                profile,
                specification,
                NativeMeshingOptions(
                    "implicit_stuffing", volume_schedule=NativeVolumeSchedule()
                ),
                contract,
            )
            if not isinstance(prepared, PreparedStuffedImplicitVolume):
                raise TypeError("An analytic profile volume prepares the stuffing route.")
            rows.append(_Row(name, thousandths, prepared.construct, _constructed_cells))
    return tuple(rows)


def _remesh_rows() -> tuple[_Row, ...]:
    # The ledger's surface remeshing rows: the unit icosphere (81,920
    # triangles) and the R = 1, r = 0.4 grid torus (90,000) remeshed toward
    # 1e4 triangles, i.e. the edge h of an equilateral triangle of area A/1e4,
    # within tolerance h^2/4 and a 30 degree minimum angle. The row times the
    # construction; certification (zero_set_certify) is a separate entry and
    # is not timed. The row size is the input triangle count. Quality is
    # measured independently against the exact surface.
    def sphere(x: Array) -> Array:
        return x[0] * x[0] + x[1] * x[1] + x[2] * x[2] - 1.0

    def torus(x: Array) -> Array:
        ring = jnp.sqrt(x[0] * x[0] + x[1] * x[1]) - 1.0
        return ring * ring + x[2] * x[2] - 0.16

    rows: list[_Row] = []
    for name, function, shape, area, (vertices, faces) in (
        ("surface-remesh-sphere", sphere, "sphere", 4.0 * math.pi, _icosphere(6)),
        ("surface-remesh-torus", torus, "torus", 4.0 * math.pi**2 * 0.4, _uv_torus(300)),
    ):
        program = prepare_interval_function(
            function, 3, batch_capacity=1, execution="native"
        ).native_program
        if program is None:
            raise RuntimeError(
                f"The {name} field must lower to a native interval program."
            )
        value = program.box_program()
        edge = math.sqrt(4.0 * area / (math.sqrt(3.0) * 1e4))
        tolerance = edge * edge / 4.0
        rows.append(
            _Row(
                name,
                faces.shape[0],
                lambda vertices=vertices, faces=faces, value=value, edge=edge, tolerance=tolerance: (
                    _meshcore.zero_set_remesh(
                        vertices, faces, value, tolerance, tolerance, edge, 30.0
                    )
                ),
                _remeshed_triangles,
                _remesh_unmet,
                lambda result, shape=shape, tolerance=tolerance: _remesh_quality(
                    result, shape, tolerance
                ),
            )
        )
    return tuple(rows)


def _remeshed_triangles(result: object, /) -> int:
    if not isinstance(result, _meshcore.ZeroSetRemeshResult):
        raise TypeError("The remesh row returns a ZeroSetRemeshResult.")
    return result.triangles.shape[0]


def _remesh_unmet(result: object, /) -> int:
    # Triangles below the minimum angle plus triangles sampled beyond the
    # tolerance by the construction itself (counters 4 and 6).
    if not isinstance(result, _meshcore.ZeroSetRemeshResult):
        raise TypeError("The remesh row returns a ZeroSetRemeshResult.")
    return int(result.counters[4] + result.counters[6])


def _exact_distance(points: np.ndarray, shape: str, /) -> np.ndarray:
    match shape:
        case "sphere":
            return np.abs(np.linalg.norm(points, axis=-1) - 1.0)
        case "torus":
            ring = np.linalg.norm(points[..., :2], axis=-1) - 1.0
            return np.abs(np.hypot(ring, points[..., 2]) - 0.4)
        case _:
            raise ValueError(f"Unknown remesh row shape {shape!r}.")


def _surface_samples(shape: str, count: int, /) -> np.ndarray:
    generator = np.random.default_rng(20261119)
    match shape:
        case "sphere":
            directions = generator.normal(size=(count, 3))
            return directions / np.linalg.norm(directions, axis=1, keepdims=True)
        case "torus":
            u, v = 2.0 * np.pi * generator.random((2, count))
            ring = 1.0 + 0.4 * np.cos(v)
            return np.stack((ring * np.cos(u), ring * np.sin(u), 0.4 * np.sin(v)), axis=1)
        case _:
            raise ValueError(f"Unknown remesh row shape {shape!r}.")


def _point_triangle_distances(point: np.ndarray, corners: np.ndarray, /) -> np.ndarray:
    # Exact Euclidean distance from one point to each triangle (rows of corners).
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    first, second = b - a, c - a
    normal = np.cross(first, second)
    squared_normal = np.sum(normal * normal, axis=1)
    offset = point - a
    height = np.sum(offset * normal, axis=1)
    projected = offset - (height / squared_normal)[:, None] * normal
    aa, ab, bb = (
        np.sum(x * y, axis=1)
        for x, y in ((first, first), (first, second), (second, second))
    )
    pa, pb = np.sum(projected * first, axis=1), np.sum(projected * second, axis=1)
    determinant = aa * bb - ab * ab
    s, t = (bb * pa - ab * pb) / determinant, (aa * pb - ab * pa) / determinant
    inside = (s >= 0.0) & (t >= 0.0) & (s + t <= 1.0)
    squared = np.where(inside, height * height / squared_normal, np.inf)
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        along = np.clip(
            np.sum((point - start) * edge, axis=1) / np.sum(edge * edge, axis=1), 0, 1
        )
        closest = start + along[:, None] * edge
        squared = np.minimum(squared, np.sum((point - closest) ** 2, axis=1))
    return np.sqrt(squared)


def _remesh_quality(
    result: object, shape: str, tolerance: float, /
) -> tuple[int, dict[str, float]]:
    # Angles of every triangle; mesh -> surface at 15 barycentric points per
    # triangle and surface -> mesh at 4000 points of the exact surface (exact
    # distance to the triangles whose centroid lies within twice the longest
    # edge), both against the exact distance function.
    from scipy.spatial import cKDTree

    if not isinstance(result, _meshcore.ZeroSetRemeshResult):
        raise TypeError("The remesh row returns a ZeroSetRemeshResult.")
    corners = result.vertices[result.triangles]
    angles = []
    for k in range(3):
        first = corners[:, (k + 1) % 3] - corners[:, k]
        second = corners[:, (k + 2) % 3] - corners[:, k]
        cosine = np.sum(first * second, axis=1) / (
            np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
        )
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    angle = np.stack(angles, axis=1)
    weights = np.asarray(
        [(i / 4, j / 4, 1 - (i + j) / 4) for i in range(5) for j in range(5 - i)]
    )
    mesh_to_surface = float(
        np.max(_exact_distance(np.einsum("wk,tkd->twd", weights, corners), shape))
    )
    centroids = corners.mean(axis=1)
    reach = 2.0 * float(
        np.max(np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2))
    )
    tree = cKDTree(centroids)
    surface_to_mesh = 0.0
    for point in _surface_samples(shape, 4000):
        nearby = tree.query_ball_point(point, reach)
        surface_to_mesh = max(
            surface_to_mesh,
            float(np.min(_point_triangle_distances(point, corners[nearby]))),
        )
    below = int(np.sum(angle.min(axis=1) < 30.0))
    missed = below + int(mesh_to_surface > tolerance) + int(surface_to_mesh > tolerance)
    return missed, {
        "minimum_angle_degrees": float(angle.min()),
        "maximum_angle_degrees": float(angle.max()),
        "below_minimum_angle": float(below),
        "mesh_to_surface": mesh_to_surface,
        "surface_to_mesh": surface_to_mesh,
        "tolerance": tolerance,
    }


def _constructed_cells(result: object, /) -> int:
    from phydrax.meshing._implicit_volume import StuffedImplicitConstruction

    if not isinstance(result, StuffedImplicitConstruction):
        raise TypeError("The stuffing construction returns its constructed arrays.")
    return result.tetrahedra.shape[0]


def _ceiling(
    ceilings: Mapping[str, Any], row: _Row, elements: int, /
) -> tuple[float, float, str] | None:
    # A ceiling is a reference time, or a reference throughput converted to the
    # time the reference takes for this row's element count.
    entry = ceilings["rows"].get(row.name, {}).get(str(row.size))
    if entry is None:
        return None
    seconds = (
        float(entry["seconds"])
        if "seconds" in entry
        else elements / float(entry["elements_per_second"])
    )
    return seconds, float(entry["allowed_ratio"]), str(entry["tool"])


def _measure(row: _Row, repeats: int, /) -> dict[str, Any]:
    samples: list[float] = []
    result: object = None
    for _ in range(repeats):
        result, seconds = measure_host(row.run)
        samples.append(seconds)
    missed, quality = row.quality(result)
    return {
        "elements": row.count(result),
        "unmet": row.unmet(result),
        "quality_missed": missed,
        "quality": quality,
        "seconds": samples,
        "minimum_seconds": min(samples),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[1_000, 10_000, 100_000, 1_000_000]
    )
    parser.add_argument(
        "--rows", nargs="+", default=None, help="run only the named rows (all by default)"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise RuntimeError("Run with OMP_NUM_THREADS=1: rows are single-thread rows.")
    ceilings = json.loads(_CEILINGS.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    failed = False
    names = None if arguments.rows is None else frozenset(arguments.rows)
    for row in _rows(tuple(arguments.sizes), names):
        measured = _measure(row, arguments.repeats)
        record: dict[str, Any] = {"row": row.name, "size": row.size, **measured}
        ceiling = _ceiling(ceilings, row, measured["elements"])
        if ceiling is not None:
            seconds, allowed, tool = ceiling
            ratio = measured["minimum_seconds"] / seconds
            meets = measured["unmet"] == 0 and measured["quality_missed"] == 0
            record |= {
                "reference_tool": tool,
                "reference_seconds": seconds,
                "ratio": ratio,
                "allowed_ratio": allowed,
                "passed": ratio <= allowed and meets,
            }
            failed = failed or ratio > allowed or not meets
        records.append(record)
        print(json.dumps(record), flush=True)
    if arguments.output is not None:
        arguments.output.write_text(
            json.dumps(
                {
                    "environment": capture_environment().to_dict(),
                    "meshcore": _meshcore.meshcore_identity(),
                    "ceilings_provenance": ceilings["provenance"],
                    "rows": records,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
