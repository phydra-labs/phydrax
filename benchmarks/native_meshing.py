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
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from benchmarks._meshing_surfaces import icosphere
from benchmarks._runtime import capture_environment, measure_host
from phydrax import _meshcore
from phydrax.geometry._interval_enclosure import prepare_interval_function


if TYPE_CHECKING:
    from phydrax.discretization import PreparedSimplicialCellLocator


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
    domain: str | None = None
    # Independent measurement of the last result (not timed): the number of
    # targets it misses, and the measured values.
    quality: Callable[[object], tuple[int, dict[str, float]]] = _no_quality
    # The timed seconds of one run when the row times a stage of a larger
    # call by the kernel's own stage clock; None times the whole call.
    stage_seconds: Callable[[object], float] | None = None


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


def _planar_cells(result: object, /) -> int:
    if not isinstance(result, _meshcore.ConstrainedDelaunay2D):
        raise TypeError("A planar constrained kernel returns ConstrainedDelaunay2D.")
    return result.triangles.shape[0]


def _planar_unmet(result: object, /) -> int:
    if not isinstance(result, _meshcore.ConstrainedDelaunay2D):
        raise TypeError("A planar constrained kernel returns ConstrainedDelaunay2D.")
    return int(result.status != _meshcore.MeshcoreStatus.OK)


def _planar_minimum_angle(result: _meshcore.ConstrainedDelaunay2D, /) -> float:
    triangles = result.points[result.triangles]
    minimum = 180.0
    for k in range(3):
        a = triangles[:, (k + 1) % 3] - triangles[:, k]
        b = triangles[:, (k + 2) % 3] - triangles[:, k]
        cross = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        dot = np.sum(a * b, axis=1)
        minimum = min(minimum, float(np.degrees(np.arctan2(cross, dot)).min()))
    return minimum


def _planar_rows(sizes: tuple[int, ...], /) -> tuple[_Row, ...]:
    fixture = Path(__file__).with_name("native_meshing_inputs")
    ring_points = np.load(fixture / "q2ring_pts.npy", allow_pickle=False)
    ring_segments = np.load(fixture / "q2ring_seg.npy", allow_pickle=False)
    if ring_points.dtype != np.float64 or ring_points.shape != (206, 2):
        raise ValueError("The published ring points must be float64 with shape (206, 2).")
    if ring_segments.dtype != np.int32 or ring_segments.shape != (205, 2):
        raise ValueError("The published ring segments must be int32 with shape (205, 2).")
    square_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    square_segments = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
    rows: list[_Row] = []
    for size in sizes:
        count = int(4 * np.sqrt(size))
        angles = 2 * np.pi * np.arange(count) / count
        ring = np.column_stack((0.5 + 0.3 * np.cos(angles), 0.5 + 0.3 * np.sin(angles)))
        points = np.vstack((ring, _uniform_points(size - count, 2, 777 + size)))
        segments = np.column_stack(
            (np.arange(count), (np.arange(count) + 1) % count)
        ).astype(np.int32)
        rows.append(
            _Row(
                "cdt-2d",
                size,
                lambda p=points, s=segments: _meshcore.constrained_delaunay_2d(
                    p, s, keep_convex_hull=True
                ),
                _planar_cells,
                _planar_unmet,
                "ring",
            )
        )
        for domain, p, s in (
            ("square", square_points, square_segments),
            ("ring-slit", ring_points, ring_segments),
        ):
            rows.append(
                _Row(
                    "ruppert-2d",
                    size,
                    lambda p=p, s=s, n=size: _meshcore.constrained_delaunay_2d(
                        p, s, keep_convex_hull=True, min_angle=20.0, max_area=1.5 / n
                    ),
                    _planar_cells,
                    _planar_unmet,
                    domain,
                )
            )
    return tuple(rows)


def _regular_rows(sizes: tuple[int, ...], /) -> tuple[_Row, ...]:
    # The ledger's weighted points: uniform in the unit cube, weights uniform
    # in [0, (0.5 n^(-1/3))^2), so some points are redundant.
    rows: list[_Row] = []
    for size in sizes:
        points = _uniform_points(size, 3, 20261119)
        limit = (0.5 * size ** (-1.0 / 3.0)) ** 2
        weights = limit * np.random.default_rng(20261120).random(size, dtype=np.float64)
        rows.append(
            _Row(
                "regular-3d",
                size,
                lambda points=points, weights=weights: _meshcore.regular_3d(
                    points, weights
                ),
                _cells,
            )
        )
    return tuple(rows)


def _periodic_rows(sizes: tuple[int, ...], /) -> tuple[_Row, ...]:
    # Uniform points on the unit 3-torus at the default initial margin.
    lattice = np.eye(3, dtype=np.float64)
    rows: list[_Row] = []
    for size in sizes:
        points = _uniform_points(size, 3, 20261121)
        rows.append(
            _Row(
                "periodic-3d",
                size,
                lambda points=points: _meshcore.periodic_delaunay(
                    points, points, lattice, lattice
                ),
                _cells,
            )
        )
    return tuple(rows)


def _rows(
    sizes: tuple[int, ...],
    names: frozenset[str] | None,
    improve_inputs: Path | None,
    query_inputs: Path | None,
    /,
) -> tuple[_Row, ...]:
    # Families prepare their inputs only when one of their rows is selected.
    families: tuple[tuple[frozenset[str], Callable[[], tuple[_Row, ...]]], ...] = (
        (frozenset({"delaunay-2d", "delaunay-3d"}), lambda: _delaunay_rows(sizes)),
        (frozenset({"cdt-2d", "ruppert-2d"}), lambda: _planar_rows(sizes)),
        (frozenset({"regular-3d"}), lambda: _regular_rows(sizes)),
        (frozenset({"periodic-3d"}), lambda: _periodic_rows(sizes)),
        (frozenset({"implicit-sphere", "implicit-torus"}), _implicit_rows),
        (frozenset({"recover-sphere", "recover-torus"}), _recovery_rows),
        (
            frozenset(
                {
                    "surface-remesh-sphere-1e3",
                    "surface-remesh-sphere",
                    "surface-remesh-sphere-1e5",
                    "surface-remesh-torus",
                }
            ),
            _remesh_rows,
        ),
        (
            frozenset(
                {
                    "mesh-remesh-torus-1e3",
                    "mesh-remesh-torus",
                    "mesh-remesh-torus-1e5",
                    "mesh-remesh-noisy-torus-1e3",
                    "mesh-remesh-noisy-torus",
                    "mesh-remesh-noisy-torus-1e5",
                }
            ),
            _mesh_remesh_rows,
        ),
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
        (frozenset({"cad-quadric", "cad-quadric-p2"}), _cad_rows),
        (frozenset({"improve-sphere"}), lambda: _improve_rows(improve_inputs)),
    )
    queries = frozenset(
        {"query-closest", "query-signed", "query-winding", "query-locate"}
    )
    if query_inputs is not None:
        families += ((queries, lambda: _query_rows(sizes, query_inputs)),)
    elif names is not None and queries & names:
        raise ValueError(
            "The query rows read the ledger's F10 inputs: pass --query-inputs."
        )
    rows: list[_Row] = []
    for family, build in families:
        if names is None or family & names:
            rows.extend(row for row in build() if names is None or row.name in names)
    return tuple(rows)


def _uv_torus(
    around: int, major: float = 1.0, minor: float = 0.4, /
) -> tuple[np.ndarray, np.ndarray]:
    # The ledger's grid torus (R = 1, r = 0.4 unless given) with an
    # (around x around / 2) grid.
    across = around // 2
    u = 2.0 * np.pi * np.arange(around) / around
    w = 2.0 * np.pi * np.arange(across) / across
    grid_u, grid_w = np.meshgrid(u, w, indexing="ij")
    ring = major + minor * np.cos(grid_w)
    vertices = np.stack(
        (ring * np.cos(grid_u), ring * np.sin(grid_u), minor * np.sin(grid_w)), axis=-1
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
        vertices, faces = icosphere(level)
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


def _restricted_torus_surface(edge: float, /) -> tuple[np.ndarray, np.ndarray]:
    # A well-shaped closed torus surface (R = 1, r = 0.4): the restricted
    # Delaunay surface of the solid with facet edges at most `edge`, facet
    # angles of at least 30 degrees and surface ball centers within edge / 8
    # of the facets (the edge bound governs), without interior targets. Edges
    # 0.072 and 0.025 give 14,010 and 123,098 triangles of mean edge 0.0524
    # and 0.0177.
    def torus(x: Array) -> Array:
        ring = jnp.sqrt(x[0] * x[0] + x[1] * x[1]) - 1.0
        return jnp.stack((jnp.sqrt(ring * ring + x[2] * x[2]) - 0.4,))

    program = prepare_interval_function(
        torus, 3, batch_capacity=1, execution="batched"
    ).native_program
    if program is None:
        raise RuntimeError("The torus rows need the native interval program.")
    upper = np.asarray((1.4, 1.4, 0.4), dtype=np.float64)
    criteria = _meshcore.RestrictedDelaunayCriteria(
        edge, edge / 8.0, 30.0, math.inf, None, 0.0, 0.0, 0.8
    )
    volume = program.restricted_tetrahedra(
        -upper, upper, criteria, 1, 0, np.empty((0, 3), dtype=np.float64)
    )
    if volume.status is not _meshcore.MeshcoreStatus.OK or not volume.topology_matched:
        raise RuntimeError("The torus rows need a resolved restricted torus surface.")
    boundary = volume.boundary_faces
    used, local = np.unique(boundary, return_inverse=True)
    return volume.points[used], local.reshape(boundary.shape).astype(np.int64)


def _volume_rows() -> tuple[_Row, ...]:
    # Surface to tetrahedra end to end (recovery, state creation, refinement to
    # radius-edge 2, improvement to radius-edge 2 and dihedral 15 degrees,
    # export) with the boundary fixed, against Gmsh HXT generate(3) on the same
    # surface. Both tools get the same request h, the mean surface edge (HXT's
    # MeshSize; ours the circumradius bound). The sphere and grid torus row sizes
    # are the triangle counts, the restricted torus row size its facet edge
    # bound in thousandths. The grid torus (cocircular rows, every edge
    # constrained) is a hard input for both tools.
    rows: list[_Row] = []
    shapes = (
        ("surface-to-tets-sphere", faces.shape[0], (vertices, faces))
        for vertices, faces in (icosphere(5), icosphere(6))
    )
    tori = (
        (
            "surface-to-tets-torus",
            thousandths,
            _restricted_torus_surface(1e-3 * thousandths),
        )
        for thousandths in (72, 25)
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
        target = spacing
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


# Gmsh 4.15.2 HXT (2D=6, 3D=10) tetrahedra of the quadric at the same h, and
# Netgen's optimized linear mesh under Curve(2), on the LIN machine (reference
# report); the count band is reported, not enforced.
_GMSH_QUADRIC_TETS = {0.2293: 1_054, 0.1061: 8_493, 0.04925: 75_913, 0.02286: 717_939}
_NETGEN_QUADRIC_TETS = {0.1061: 6_095, 0.04925: 67_533}


def _cad_rows() -> tuple[_Row, ...]:
    # The repository's quadric fixture (z cylinder r 1, x cylinder r 0.6) as
    # one native call: curves, faces, fixed-boundary volume (and P2 for the
    # -p2 rows, timed by its stage clock against Netgen's Curve(2)). The row
    # size is h in units of 1e-5.
    import phydrax as phx
    from phydrax.geometry._meshing_domain import MeshingDomain
    from phydrax.meshing.providers._native_cad_bundle import cad_geometry_bundle

    contract = phx.SpatialCoordinateContract.si()
    vertical, transverse = (
        phx.geometry.brep.brep_cylinder(
            radius,
            height,
            base_center=base,
            axis=axis,
            coordinate_contract=contract,
            source_id=f"qualification:native-cad-quadric-{name}",
        )
        for name, radius, height, base, axis in (
            ("vertical", 1.0, 2.4, (0.0, 0.0, -1.2), (0.0, 0.0, 1.0)),
            ("transverse", 0.6, 3.0, (-1.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
        )
    )
    solid = phx.geometry.brep.boolean_brep(
        vertical,
        transverse,
        "intersection",
        policy=phx.geometry.brep.BRepBooleanPolicy(),
    ).model
    model = cad_geometry_bundle(MeshingDomain.from_brep(solid)).model
    faces, curves = model.surface_nodes.size, model.curve_nodes.size

    def run(size: float, order: int) -> _meshcore.CadSolidResult:
        return _meshcore.cad_mesh_solid(
            model,
            np.full((faces,), size),
            np.full((faces,), 0.4),
            np.full((curves,), np.inf),
            geometry_order=order,
        )

    rows: list[_Row] = []
    for size, tets in _GMSH_QUADRIC_TETS.items():
        rows.append(
            _Row(
                "cad-quadric",
                round(size * 1e5),
                lambda size=size: run(size, 1),
                _cad_cells,
                _cad_unmet,
                quality=lambda result, tets=tets: _cad_quality(result, tets),
            )
        )
    for size, tets in _NETGEN_QUADRIC_TETS.items():
        rows.append(
            _Row(
                "cad-quadric-p2",
                round(size * 1e5),
                lambda size=size: run(size, 2),
                _cad_cells,
                _cad_unmet,
                quality=lambda result, tets=tets: _cad_quality(result, tets),
                stage_seconds=lambda result: (
                    float(_cad_stats(result)["quadratic_ns"]) * 1e-9
                ),
            )
        )
    return tuple(rows)


def _cad_stats(result: object, /) -> Any:
    if not isinstance(result, _meshcore.CadSolidResult):
        raise TypeError("The CAD row returns a CadSolidResult.")
    return result.volume.stats[0]


def _cad_cells(result: object, /) -> int:
    if not isinstance(result, _meshcore.CadSolidResult):
        raise TypeError("The CAD row returns a CadSolidResult.")
    return result.volume.tetrahedra.shape[0]


def _cad_unmet(result: object, /) -> int:
    stats = _cad_stats(result)
    quadratic = stats["quadratic"]
    return int(
        stats["unmet_cells"] + quadratic["final_invalid"] + quadratic["final_unresolved"]
    )


def _dihedral_degrees(points: np.ndarray, cells: np.ndarray, /) -> np.ndarray:
    corners = points[cells]
    pairs = (
        (0, 1, 2, 3),
        (0, 2, 1, 3),
        (0, 3, 1, 2),
        (1, 2, 0, 3),
        (1, 3, 0, 2),
        (2, 3, 0, 1),
    )
    angles = []
    for a, b, c, d in pairs:
        # The dihedral at edge ab between faces abc and abd.
        axis = corners[:, b] - corners[:, a]
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        u = corners[:, c] - corners[:, a]
        w = corners[:, d] - corners[:, a]
        u -= np.sum(u * axis, axis=1, keepdims=True) * axis
        w -= np.sum(w * axis, axis=1, keepdims=True) * axis
        cosine = np.sum(u * w, axis=1) / (
            np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1)
        )
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return np.stack(angles, axis=1)


def _radius_edge(points: np.ndarray, cells: np.ndarray, /) -> np.ndarray:
    corners = points[cells]
    origin = corners[:, 0]
    edges = corners[:, 1:] - origin[:, None, :]
    rhs = 0.5 * np.sum(edges * edges, axis=2)
    center = np.linalg.solve(edges, rhs[..., None])[..., 0]
    radius = np.linalg.norm(center, axis=1)
    index = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    shortest = np.min(
        np.stack(
            [np.linalg.norm(corners[:, i] - corners[:, j], axis=1) for i, j in index],
            axis=1,
        ),
        axis=1,
    )
    return radius / shortest


def _cad_quality(result: object, reference_tets: int, /) -> tuple[int, dict[str, float]]:
    # Dihedral range and radius-edge measured here; the tetrahedron count
    # against the reference's (+-20 % band reported, not a target); the
    # native stage times.
    if not isinstance(result, _meshcore.CadSolidResult):
        raise TypeError("The CAD row returns a CadSolidResult.")
    points, cells = result.volume.points, result.volume.tetrahedra
    dihedral = _dihedral_degrees(points, cells)
    radius_edge = _radius_edge(points, cells)
    stats = _cad_stats(result)
    measured = {
        "minimum_dihedral_degrees": float(dihedral.min()),
        "maximum_dihedral_degrees": float(dihedral.max()),
        "maximum_radius_edge": float(radius_edge.max()),
        "reference_tets": float(reference_tets),
        "tets_over_reference": cells.shape[0] / reference_tets,
        "surface_triangles": float(result.surface.triangles.shape[0]),
        "boundary_splits": float(stats["boundary_splits"]),
        **{
            f"{stage}_seconds": float(stats[f"{stage}_ns"]) * 1e-9
            for stage in ("curve", "face", "assembly", "volume", "quadratic")
        },
    }
    missed = int(np.sum(np.min(dihedral, axis=1) < 15.5)) + int(np.sum(radius_edge > 2.0))
    return missed, measured


class _ImproveResult(NamedTuple):
    mesh: _meshcore.TetMesh3D
    unmet: int


def _improve(
    points: np.ndarray, cells: np.ndarray, faces: np.ndarray, dihedral: float, /
) -> _ImproveResult:
    # State creation on the given mesh with its boundary fixed, then
    # improvement to radius-edge 2 and the given minimum dihedral.
    empty = np.empty((0,), dtype=np.int32)
    mesh = _meshcore.TetMesh3D(
        points,
        cells,
        np.zeros((cells.shape[0],), dtype=np.int32),
        faces,
        np.zeros((faces.shape[0],), dtype=np.int32),
        np.empty((0, 2), dtype=np.int32),
        empty,
        boundary_policy=_meshcore.PlcBoundaryPolicy.FIXED,
    )
    run = mesh.improve(radius_edge_bound=2.0, min_dihedral_degrees=dihedral)
    if run.status not in (
        _meshcore.MeshcoreStatus.OK,
        _meshcore.MeshcoreStatus.UNRESOLVED,
    ):
        raise RuntimeError(f"Improvement failed with status {run.status!r}.")
    return _ImproveResult(mesh, run.unmet_cells)


def _boundary_faces(cells: np.ndarray, /) -> np.ndarray:
    # Faces used by one cell, oriented outward as in that cell.
    faces = np.concatenate(
        (
            cells[:, [1, 2, 3]],
            cells[:, [0, 3, 2]],
            cells[:, [0, 1, 3]],
            cells[:, [0, 2, 1]],
        )
    )
    _, first, count = np.unique(
        np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True
    )
    return faces[first[count == 1]]


def _improve_rows(directory: Path | None, /) -> tuple[_Row, ...]:
    # Improvement passes on a given mesh: the TetGen tetrahedra of the
    # icosphere at levels 3, 4 and 5 (minimum dihedral about 1.5 degrees),
    # prepared by `benchmarks/native_meshing_mmg3d.py prepare DIR`, which also
    # times the reference (MMG3D -optim -nosurf, library time) on the same
    # input. Timed: state creation plus improvement; the row size is the
    # input tetrahedron count. The dihedral target of each size is the
    # minimum dihedral MMG3D achieves on it (rounded up to 0.1 degree, kept
    # with the ceiling), so the row is at equal or better quality.
    if directory is None:
        raise RuntimeError(
            "The improve-sphere rows need --improve-inputs DIR from "
            "`benchmarks/native_meshing_mmg3d.py prepare DIR`."
        )
    targets = json.loads(_CEILINGS.read_text(encoding="utf-8"))["rows"]["improve-sphere"]
    rows: list[_Row] = []
    for level in (3, 4, 5):
        given = np.load(directory / f"improve-sphere-{level}.npz")
        points = np.ascontiguousarray(given["points"], dtype=np.float64)
        cells = np.ascontiguousarray(given["cells"], dtype=np.int32)
        faces = _boundary_faces(cells)
        dihedral = float(targets[str(cells.shape[0])]["minimum_dihedral_degrees"])
        rows.append(
            _Row(
                "improve-sphere",
                cells.shape[0],
                lambda points=points, cells=cells, faces=faces, dihedral=dihedral: (
                    _improve(points, cells, faces, dihedral)
                ),
                _improved_cells,
                _improved_unmet,
                quality=lambda result, points=points, faces=faces, dihedral=dihedral: (
                    _improved_quality(result, points, faces, dihedral)
                ),
            )
        )
    return tuple(rows)


def _improved_result(result: object, /) -> _ImproveResult:
    if not isinstance(result, _ImproveResult):
        raise TypeError("The improve row returns an _ImproveResult.")
    return result


def _improved_cells(result: object, /) -> int:
    return _improved_result(result).mesh.arrays().tetrahedra.shape[0]


def _improved_unmet(result: object, /) -> int:
    return _improved_result(result).unmet


def _improved_quality(
    result: object, given_points: np.ndarray, given_faces: np.ndarray, target: float, /
) -> tuple[int, dict[str, float]]:
    # Targets: the minimum dihedral `target`, radius-edge 2, the given closed
    # boundary kept exactly (FIXED: its triangles, by vertex coordinates), no
    # inverted cell (the input's positive orientation).
    arrays = _improved_result(result).mesh.arrays()
    points = np.asarray(arrays.points)
    cells = np.asarray(arrays.tetrahedra, dtype=np.int64)
    dihedral = _dihedral_degrees(points, cells)
    radius_edge = _radius_edge(points, cells)
    corners = points[cells]
    edges = corners[:, 1:] - corners[:, :1]
    volume = np.einsum("ij,ij->i", edges[:, 0], np.cross(edges[:, 1], edges[:, 2]))
    given = {tuple(point): row for row, point in enumerate(given_points.tolist())}
    boundary = _boundary_faces(cells)
    rows = np.asarray(
        [
            given.get(tuple(point), -1)
            for point in points[boundary.reshape((-1,))].tolist()
        ],
        dtype=np.int64,
    ).reshape(boundary.shape)
    closed = bool(np.all(rows >= 0)) and np.array_equal(
        np.unique(np.sort(rows, axis=1), axis=0),
        np.unique(np.sort(given_faces, axis=1), axis=0),
    )
    inverted = int(np.sum(volume <= 0.0))
    measured = {
        "target_dihedral_degrees": target,
        "minimum_dihedral_degrees": float(dihedral.min()),
        "maximum_dihedral_degrees": float(dihedral.max()),
        "maximum_radius_edge": float(radius_edge.max()),
        "closed_boundary": float(closed),
        "inverted": float(inverted),
    }
    missed = (
        int(np.sum(np.min(dihedral, axis=1) < target))
        + int(np.sum(radius_edge > 2.0))
        + inverted
        + int(not closed)
    )
    return missed, measured


def _implicit_rows() -> tuple[_Row, ...]:
    # The external comparison's sphere (radius 0.371) and torus (major 0.742,
    # minor 0.371, i.e. inner 0.371 and outer 1.113) through the analytic
    # profile route's construction: restricted Delaunay refinement of the zero
    # set from the interval box tree's seeds to the certified facet bound, of
    # the interior cells in the same triangulation to radius-edge 2, sliver
    # removal to 15 degrees and the fixed-boundary quality fallback, ending at
    # the constructed arrays as the references end at their native meshes.
    # Publication as a CellMesh and certification are not timed. The row size
    # is the tolerance in thousandths.
    from phydrax._physical import SpatialCoordinateContract
    from phydrax.geometry import Sphere, Torus
    from phydrax.geometry.implicit import AnalyticImplicitProfile
    from phydrax.meshing import (
        CellFamilyPolicy,
        CellMeshingTarget,
        VolumeFillStrategy,
        VolumeMeshingSpec,
    )
    from phydrax.meshing._implicit_volume import PreparedRestrictedImplicitVolume
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
                    "implicit_restricted_delaunay", volume_schedule=NativeVolumeSchedule()
                ),
                contract,
            )
            if not isinstance(prepared, PreparedRestrictedImplicitVolume):
                raise TypeError(
                    "An analytic profile volume prepares the restricted route."
                )
            rows.append(_Row(name, thousandths, prepared.construct, _constructed_cells))
    return tuple(rows)


def _remesh_rows() -> tuple[_Row, ...]:
    # The ledger's surface remeshing rows: unit icospheres (20,480 to 1,310,720
    # triangles) and the R = 1, r = 0.4 grid torus (90,000) remeshed toward a
    # target triangle count N, i.e. the edge h of an equilateral triangle of
    # area A/N, within tolerance h^2/4 and a 30 degree minimum angle. The row
    # name carries the target (no suffix: 1e4; `-1e3`, `-1e5`); the row size is
    # the input triangle count. The row times the construction; the quality
    # stage (untimed) certifies it with zero_set_certify, a refusal counting
    # as a miss, and measures the certified surface independently against the
    # exact surface.
    def sphere(x: Array) -> Array:
        return x[0] * x[0] + x[1] * x[1] + x[2] * x[2] - 1.0

    def torus(x: Array) -> Array:
        ring = jnp.sqrt(x[0] * x[0] + x[1] * x[1]) - 1.0
        return ring * ring + x[2] * x[2] - 0.16

    # Per field: f and grad f as one-box programs (construction), and f with
    # its branch-evidence events (the certificate's value program).
    programs = []
    for label, function in (("sphere", sphere), ("torus", torus)):
        value_prepared = prepare_interval_function(
            function, 3, batch_capacity=1, execution="native"
        )
        gradient_program = prepare_interval_function(
            jax.grad(function), 3, batch_capacity=1, execution="native"
        ).native_program
        if value_prepared.native_program is None or gradient_program is None:
            raise RuntimeError(
                f"The {label} field and its gradient must lower to native "
                "interval programs."
            )
        programs.append(
            (
                value_prepared.native_program.box_program(),
                gradient_program.box_program(),
                value_prepared.native_branches().branch_box_program(),
            )
        )
    (
        (sphere_value, sphere_gradient, sphere_certified),
        (torus_value, torus_gradient, torus_certified),
    ) = programs
    spheres = {level: icosphere(level) for level in (5, 6, 7, 8)}
    sphere_rows = (
        ("surface-remesh-sphere-1e3", 5, 1e3),
        ("surface-remesh-sphere-1e3", 6, 1e3),
        ("surface-remesh-sphere", 6, 1e4),
        ("surface-remesh-sphere", 7, 1e4),
        ("surface-remesh-sphere-1e5", 7, 1e5),
        ("surface-remesh-sphere-1e5", 8, 1e5),
    )
    rows: list[_Row] = []
    # The certificate's domains: boxes holding each zero set with a margin.
    sphere_domain = np.array(
        [[-1.25, -1.25, -1.25], [1.25, 1.25, 1.25]], dtype=np.float64
    )
    torus_domain = np.array([[-1.5, -1.5, -0.5], [1.5, 1.5, 0.5]], dtype=np.float64)
    for name, value, gradient, certified, domain, distance, samples, area, (
        vertices,
        faces,
    ), target in (
        *(
            (
                name,
                sphere_value,
                sphere_gradient,
                sphere_certified,
                sphere_domain,
                _sphere_distance,
                _sphere_samples,
                4.0 * math.pi,
                spheres[level],
                target,
            )
            for name, level, target in sphere_rows
        ),
        (
            "surface-remesh-torus",
            torus_value,
            torus_gradient,
            torus_certified,
            torus_domain,
            _torus_distance,
            _torus_samples,
            4.0 * math.pi**2 * 0.4,
            _uv_torus(300),
            1e4,
        ),
    ):
        edge = math.sqrt(4.0 * area / (math.sqrt(3.0) * target))
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
                quality=lambda result, certified=certified, gradient=gradient, domain=domain, tolerance=tolerance, edge=edge, distance=distance, samples=samples: (
                    _zero_set_quality(
                        result,
                        certified,
                        gradient,
                        domain,
                        tolerance,
                        edge,
                        distance,
                        samples,
                    )
                ),
            )
        )
    return tuple(rows)


# The ledger's F7 torus: R = 0.742, r = 0.371 on a 1000 x 500 grid
# (1,000,000 triangles), and the same with every vertex moved along the exact
# normal by N(0, 0.001) (seed 2), as `f7_setup.py`.
_F7_MAJOR, _F7_MINOR, _F7_NOISE = 0.742, 0.371, 0.001
# CGAL isotropic_remeshing's distances to the true torus on the noisy rows
# (ledger M1, the smaller of its two directions), the reference those rows
# may not exceed.
_F7_NOISY_REFERENCE = {1e3: 0.0141, 1e4: 0.00248, 1e5: 0.00241}


def _f7_torus(noisy: bool, /) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = _uv_torus(1000, _F7_MAJOR, _F7_MINOR)
    if not noisy:
        return vertices, faces
    u = 2.0 * np.pi * np.arange(1000) / 1000
    w = 2.0 * np.pi * np.arange(500) / 500
    grid_u, grid_w = np.meshgrid(u, w, indexing="ij")
    normals = np.stack(
        (
            np.cos(grid_w) * np.cos(grid_u),
            np.cos(grid_w) * np.sin(grid_u),
            np.sin(grid_w),
        ),
        axis=-1,
    ).reshape((-1, 3))
    noise = np.random.default_rng(2).normal(0.0, _F7_NOISE, vertices.shape[0])
    return vertices + normals * noise[:, None], faces


def _mesh_remesh_rows() -> tuple[_Row, ...]:
    # The ledger's F7 rows: the dense and the noisy torus remeshed with the
    # input mesh as the only source toward N = 1e3 / 1e4 / 1e5 triangles: edge
    # h of an equilateral triangle of area A/N and tolerance h^2 / (8 r), the
    # sagitta of h on the minor circle; the noisy 1e4 / 1e5 rows allow the
    # noise's 5 sigma more. The row name carries the target (no suffix: 1e4);
    # the size is the input triangle count. The row times the construction;
    # the certificate against the input (mesh_certify, both directions) and
    # the distances to the true torus are measured after it, untimed.
    area = 4.0 * math.pi**2 * _F7_MAJOR * _F7_MINOR
    rows: list[_Row] = []
    for noisy, label in ((False, "mesh-remesh-torus"), (True, "mesh-remesh-noisy-torus")):
        vertices, faces = _f7_torus(noisy)
        for suffix, target in (("-1e3", 1e3), ("", 1e4), ("-1e5", 1e5)):
            edge = math.sqrt(4.0 * area / (math.sqrt(3.0) * target))
            tolerance = edge * edge / (8.0 * _F7_MINOR)
            if noisy and target > 1e3:
                tolerance += 5.0 * _F7_NOISE
            true_tolerance = _F7_NOISY_REFERENCE[target] if noisy else tolerance
            rows.append(
                _Row(
                    label + suffix,
                    faces.shape[0],
                    lambda vertices=vertices, faces=faces, edge=edge, tolerance=tolerance: (
                        _meshcore.mesh_remesh(vertices, faces, tolerance, edge, 30.0)
                    ),
                    _mesh_remeshed_triangles,
                    _mesh_remesh_unmet,
                    quality=lambda result, vertices=vertices, faces=faces, edge=edge, tolerance=tolerance, true_tolerance=true_tolerance: (
                        _mesh_remesh_quality(
                            result, vertices, faces, edge, tolerance, true_tolerance
                        )
                    ),
                )
            )
    return tuple(rows)


def _mesh_remeshed_triangles(result: object, /) -> int:
    if not isinstance(result, _meshcore.MeshRemeshResult):
        raise TypeError("The mesh remesh row returns a MeshRemeshResult.")
    return result.triangles.shape[0]


def _mesh_remesh_unmet(result: object, /) -> int:
    # Triangles below the minimum angle plus triangles sampled beyond the
    # tolerance by the construction itself (counters 4 and 5).
    if not isinstance(result, _meshcore.MeshRemeshResult):
        raise TypeError("The mesh remesh row returns a MeshRemeshResult.")
    return int(result.counters[4] + result.counters[5])


def _mesh_remesh_quality(
    result: object,
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    edge: float,
    tolerance: float,
    true_tolerance: float,
    /,
) -> tuple[int, dict[str, float]]:
    # The certificate against the input (both directions, within the row's
    # tolerance; a refusal is one miss), then angles and distances to the true
    # torus as _surface_quality measures them, against `true_tolerance`, of
    # the certified surface (the constructed one when refused).
    if not isinstance(result, _meshcore.MeshRemeshResult):
        raise TypeError("The mesh remesh row returns a MeshRemeshResult.")
    torus = (
        lambda points: _torus_distance(points, _F7_MAJOR, _F7_MINOR),
        lambda count: _torus_samples(count, _F7_MAJOR, _F7_MINOR),
    )
    try:
        certificate = _meshcore.mesh_certify(
            source_vertices,
            source_faces,
            result.vertices,
            result.triangles,
            tolerance,
            edge,
            30.0,
        )
    except _meshcore.MeshcoreError as error:
        # The refusal is the row's evidence; any other status is a failure.
        if error.status is not _meshcore.MeshcoreStatus.UNRESOLVED:
            raise
        missed, measured = _surface_quality(
            result.vertices, result.triangles, *torus, true_tolerance
        )
        return missed + 1, {
            **measured,
            "certified": 0.0,
            "certified_tolerance": tolerance,
        }
    missed, measured = _surface_quality(
        certificate.vertices, certificate.triangles, *torus, true_tolerance
    )
    return missed, {
        **measured,
        "certified": 1.0,
        "certified_tolerance": tolerance,
        "certified_surface_to_source": float(certificate.distances[0]),
        "certified_source_to_surface": float(certificate.distances[1]),
    }


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


def _sphere_distance(points: np.ndarray, /) -> np.ndarray:
    return np.abs(np.linalg.norm(points, axis=-1) - 1.0)


def _torus_distance(
    points: np.ndarray, major: float = 1.0, minor: float = 0.4, /
) -> np.ndarray:
    ring = np.linalg.norm(points[..., :2], axis=-1) - major
    return np.abs(np.hypot(ring, points[..., 2]) - minor)


def _sphere_samples(count: int, /) -> np.ndarray:
    directions = np.random.default_rng(20261119).normal(size=(count, 3))
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def _torus_samples(count: int, major: float = 1.0, minor: float = 0.4, /) -> np.ndarray:
    u, v = 2.0 * np.pi * np.random.default_rng(20261119).random((2, count))
    ring = major + minor * np.cos(v)
    return np.stack((ring * np.cos(u), ring * np.sin(u), minor * np.sin(v)), axis=1)


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


def _zero_set_quality(
    result: object,
    value: _meshcore.IntervalBranchProgram,
    gradient: _meshcore.IntervalBoxProgram,
    domain: np.ndarray,
    tolerance: float,
    edge: float,
    distance: Callable[[np.ndarray], np.ndarray],
    samples: Callable[[int], np.ndarray],
    /,
) -> tuple[int, dict[str, float]]:
    # The two-way certificate against the zero set in `domain`
    # (zero_set_certify at the tolerance, with no carrier: the rows hold no
    # witness for their input; a refusal is one miss, and so is a certified
    # zero set -> surface bound above the tolerance), then _surface_quality
    # of the certified surface (of the constructed one when refused).
    if not isinstance(result, _meshcore.ZeroSetRemeshResult):
        raise TypeError("The remesh row returns a ZeroSetRemeshResult.")
    try:
        certificate = _meshcore.zero_set_certify(
            result.vertices,
            result.triangles,
            value,
            gradient,
            tolerance,
            result.orientation,
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.int64),
            np.empty((0, 2, 3), dtype=np.float64),
            domain,
            edge,
            30.0,
        )
    except _meshcore.MeshcoreError as error:
        # The refusal is the row's evidence; any other status is a failure.
        if error.status is not _meshcore.MeshcoreStatus.UNRESOLVED:
            raise
        missed, measured = _surface_quality(
            result.vertices, result.triangles, distance, samples, tolerance
        )
        return missed + 1, {**measured, "certified": 0.0}
    missed, measured = _surface_quality(
        certificate.vertices, certificate.triangles, distance, samples, tolerance
    )
    return missed + int(certificate.reverse_distance > tolerance), {
        **measured,
        "certified": 1.0,
        "certified_distance": certificate.distance,
        "certified_reverse_distance": certificate.reverse_distance,
        "certified_reverse_proof": float(certificate.counters[8]),
        "certified_cover_boxes": float(certificate.counters[9]),
    }


def _surface_quality(
    vertices: np.ndarray,
    triangles: np.ndarray,
    distance: Callable[[np.ndarray], np.ndarray],
    samples: Callable[[int], np.ndarray],
    tolerance: float,
    /,
) -> tuple[int, dict[str, float]]:
    # Angles of every triangle; mesh -> surface at 15 barycentric points per
    # triangle and surface -> mesh at 4000 points of the exact surface (exact
    # distance to the triangles whose centroid lies within twice the longest
    # edge), both against the exact distance function.
    from scipy.spatial import cKDTree

    corners = vertices[triangles]
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
        [(i / 4, j / 4, 1 - (i + j) / 4) for i in range(5) for j in range(5 - i)],
        dtype=np.float64,
    )
    # (W, 3) @ (T, 3, 3): every triangle's barycentric sample points.
    mesh_to_surface = float(np.max(distance(weights @ corners)))
    centroids = corners.mean(axis=1)
    reach = 2.0 * float(
        np.max(np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2))
    )
    tree = cKDTree(centroids)
    surface_to_mesh = 0.0
    for point in samples(4000):
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
    from phydrax.meshing._implicit_volume import RestrictedImplicitConstruction

    if not isinstance(result, RestrictedImplicitConstruction):
        raise TypeError("The restricted construction returns its constructed arrays.")
    return result.tetrahedra.shape[0]


# The ledger's F10 surface: parametric torus R = 0.742, r = 0.371 (99,856
# triangles); point location in a Gmsh tetrahedral unit sphere (930,942 cells).
_F10_MAJOR = 0.742
_F10_MINOR = 0.371
# Signs and inside decisions are checked only farther than this from the
# analytic surface (the ledger's band).
_F10_BAND = 0.002
_F10_CHECKED_DISTANCES = 5_000


def _query_count(result: object, /) -> int:
    if not isinstance(result, np.ndarray):
        raise TypeError("A query row returns one host value per query.")
    return result.shape[0]


def _distance_error(
    points: np.ndarray, reported: np.ndarray, corners: np.ndarray, /
) -> float:
    # Exact distances over the triangles whose boxes reach the reported
    # distance: the true nearest is among them unless the report is too small,
    # which the error then exposes anyway.
    lower, upper = np.min(corners, axis=1), np.max(corners, axis=1)
    error = 0.0
    for point, value in zip(points, np.abs(reported), strict=True):
        gap = np.maximum(0.0, np.maximum(lower - point, point - upper))
        near = np.sum(gap * gap, axis=1) <= (value + 1.0e-9) ** 2
        exact = np.min(_point_triangle_distances(point, corners[near]), initial=np.inf)
        error = max(error, abs(value - exact))
    return error


def _distance_quality(
    checked: np.ndarray, corners: np.ndarray, result: object, /
) -> tuple[int, dict[str, float]]:
    error = _distance_error(checked, np.asarray(result)[: checked.shape[0]], corners)
    return int(error > 1.0e-12), {"max_abs_error_first5000": error}


def _signed_quality(
    checked: np.ndarray,
    corners: np.ndarray,
    truth: np.ndarray,
    far: np.ndarray,
    result: object,
    /,
) -> tuple[int, dict[str, float]]:
    values = np.asarray(result)
    error = _distance_error(checked, values[: checked.shape[0]], corners)
    mismatched = int(np.sum((values[far] < 0.0) != (truth[far] < 0.0)))
    return int(error > 1.0e-12) + mismatched, {
        "max_abs_error_first5000": error,
        "sign_mismatch_outside_band": float(mismatched),
    }


def _winding_quality(
    truth: np.ndarray, far: np.ndarray, result: object, /
) -> tuple[int, dict[str, float]]:
    inside = np.asarray(result) > 0.5
    mismatched = int(np.sum(inside[far] != (truth[far] < 0.0)))
    return mismatched, {"inside_mismatch_outside_band": float(mismatched)}


def _locate_quality(
    tets: np.ndarray, queries: np.ndarray, result: object, /
) -> tuple[int, dict[str, float]]:
    cell = np.asarray(result, dtype=np.int64)
    found = cell >= 0
    tet = tets[cell[found]]
    # Independent barycentrics by a dense NumPy solve per found point.
    lam = np.linalg.solve(
        np.stack([tet[:, k] - tet[:, 0] for k in (1, 2, 3)], axis=-1),
        (queries[found] - tet[:, 0])[..., None],
    )[..., 0]
    barycentric = np.concatenate((1.0 - np.sum(lam, axis=1)[:, None], lam), axis=1)
    radius = np.linalg.norm(queries, axis=1)
    measured = {
        "found": float(np.sum(found)),
        "wrong_cell": float(np.sum(np.min(barycentric, axis=1) < -1.0e-9)),
        "missed_inside": float(np.sum((radius < 0.998) & ~found)),
        "false_inside": float(np.sum((radius > 1.002) & found)),
    }
    missed = measured["wrong_cell"] + measured["missed_inside"] + measured["false_inside"]
    return int(missed), measured


def _f10_locator(volume: Mapping[str, np.ndarray], /) -> PreparedSimplicialCellLocator:
    from phydrax.discretization import (
        CellBlock,
        CellMesh,
        FiniteElementFieldSpec,
        FiniteElementPlan,
        lagrange_element,
        PreparedSimplicialCellLocator,
        SimplicialLocationPolicy,
    )
    from phydrax.discretization.fem import prepare_finite_element_cell_map

    mesh = CellMesh(
        jnp.asarray(volume["xyz"], dtype=jnp.float64),
        (CellBlock("tet", "tetrahedron", jnp.asarray(volume["tet"], dtype=jnp.int32)),),
    )
    discretization = FiniteElementPlan(
        mesh, FiniteElementFieldSpec("u", lagrange_element("tetrahedron", 1))
    ).prepare()
    return PreparedSimplicialCellLocator(
        prepare_finite_element_cell_map(discretization, 0),
        discretization.default_runtime.coordinates,
        SimplicialLocationPolicy(16, 8, 4),
    )


def _query_rows(sizes: tuple[int, ...], inputs: Path, /) -> tuple[_Row, ...]:
    # Acceleration structures are built once, outside the timed queries.
    from phydrax.geometry.simplicial import TriangleBVH, TriangleMesh

    surface = np.load(inputs / "torus-1e5.npz")
    vertices = np.asarray(surface["xyz"], dtype=np.float64)
    faces = np.asarray(surface["tri"], dtype=np.int32)
    corners = vertices[faces]
    index = TriangleBVH(TriangleMesh(vertices, faces))
    volume = dict(np.load(inputs / "tet1e6.npz"))
    locator = _f10_locator(volume)
    tets = np.asarray(volume["xyz"], dtype=np.float64)[
        np.asarray(volume["tet"], dtype=np.int64)
    ]
    rows: list[_Row] = []
    for size in sizes:
        if size not in (100_000, 1_000_000):
            continue
        label = f"1e{round(math.log10(size))}"
        points = np.load(inputs / f"q-{label}.npy")
        located = np.load(inputs / f"qloc-{label}.npy")
        radial = np.hypot(
            np.linalg.norm(points[:, :2], axis=1) - _F10_MAJOR, points[:, 2]
        )
        truth = radial - _F10_MINOR
        far = np.abs(truth) > _F10_BAND
        checked = points[:_F10_CHECKED_DISTANCES]
        rows.extend(
            (
                _Row(
                    "query-closest",
                    size,
                    lambda points=points: np.asarray(index.query(points).distance),
                    _query_count,
                    quality=partial(_distance_quality, checked, corners),
                ),
                _Row(
                    "query-signed",
                    size,
                    lambda points=points: np.asarray(index.signed_distance(points)),
                    _query_count,
                    quality=partial(_signed_quality, checked, corners, truth, far),
                ),
                _Row(
                    "query-winding",
                    size,
                    lambda points=points: np.asarray(
                        index.fast_winding_number(points).values
                    ),
                    _query_count,
                    quality=partial(_winding_quality, truth, far),
                ),
                _Row(
                    "query-locate",
                    size,
                    lambda points=located: np.asarray(locator.locate(points).cell_ids),
                    _query_count,
                    quality=partial(_locate_quality, tets, located),
                ),
            )
        )
    return tuple(rows)


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
        samples.append(
            seconds if row.stage_seconds is None else row.stage_seconds(result)
        )
    missed, quality = row.quality(result)
    if isinstance(result, _meshcore.ConstrainedDelaunay2D):
        quality = {**quality, "minimum_angle_degrees": _planar_minimum_angle(result)}
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
    parser.add_argument(
        "--improve-inputs",
        type=Path,
        default=None,
        help="directory from `benchmarks/native_meshing_mmg3d.py prepare` (improve-sphere rows)",
    )
    parser.add_argument(
        "--query-inputs",
        type=Path,
        default=None,
        help="directory of the ledger's F10 inputs (torus-1e5.npz, tet1e6.npz, "
        "q-1eN.npy, qloc-1eN.npy); the query rows run only with it",
    )
    arguments = parser.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise RuntimeError("Run with OMP_NUM_THREADS=1: rows are single-thread rows.")
    ceilings = json.loads(_CEILINGS.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    failed = False
    names = None if arguments.rows is None else frozenset(arguments.rows)
    for row in _rows(
        tuple(arguments.sizes), names, arguments.improve_inputs, arguments.query_inputs
    ):
        measured = _measure(row, arguments.repeats)
        record: dict[str, Any] = {"row": row.name, "size": row.size, **measured}
        if row.domain is not None:
            record["domain"] = row.domain
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
