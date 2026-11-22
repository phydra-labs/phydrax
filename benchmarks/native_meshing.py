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
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from benchmarks._meshing_surfaces import icosphere
from benchmarks._runtime import capture_environment, measure_host, measure_synchronized
from phydrax import _meshcore
from phydrax._exact_kernels import exact_kernel_parallelism
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import CellBlock, CellMesh, PolyhedralConnectivity
from phydrax.discretization._cell_complex import TetrahedralConnectivity
from phydrax.discretization._cell_mesh import SimplicialConnectivity
from phydrax.discretization._exact_power_geometry import ExactPoint3
from phydrax.discretization._exact_voronoi_geometry import (
    ExactVoronoiCellGeometrySource,
)
from phydrax.discretization._hexahedral import HexahedralConnectivity
from phydrax.geometry._interval_enclosure import prepare_interval_function
from phydrax.meshing import (
    BoundaryLayerCollisionPolicy,
    BoundaryLayerControl,
    BoundaryLayerCornerPolicy,
    BoundaryLayerMesh,
    BoundaryLayerRoute,
    certify_cell_mesh,
    LayerSchedule,
    MeshingEntityKind,
    MeshingScope,
    prepare_boundary_layers,
)
from phydrax.meshing._assembly import MeshAssembly, MeshPart
from phydrax.meshing._overset import (
    _cell_result,
    OversetConnectivity,
    OversetConnectivityError,
    OversetPartSpec,
    OversetPolicy,
    prepare_overset_connectivity,
)


if TYPE_CHECKING:
    from phydrax.discretization import PreparedSimplicialCellLocator
    from phydrax.meshing import CellMeshingCandidate, CellMeshingResult, NativeMeshingPlan
    from phydrax.meshing._adaptation import _RouteOutcome
    from phydrax.meshing._tetra_metric import TetraMetricOutcome


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
                lambda points=spatial: _meshcore.delaunay_3d(
                    points, parallelism=exact_kernel_parallelism()
                ),
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
                    points, weights, parallelism=exact_kernel_parallelism()
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
    overset_inputs: Path | None,
    layer_inputs: Path | None,
    voronoi_inputs: Path | None,
    hexdom_inputs: Path | None,
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
        (
            frozenset({"metric-3d-iso", "metric-3d-aniso", "metric-3d-coarsen"}),
            lambda: _f6_rows(sizes),
        ),
        (frozenset({"cad-quadric", "cad-quadric-p2"}), _cad_rows),
        (frozenset({"improve-sphere"}), lambda: _improve_rows(improve_inputs)),
        (
            frozenset({"overset-hex", "overset-tet"}),
            lambda: _overset_rows(sizes, overset_inputs),
        ),
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
    layer_rows = frozenset(name for name, _ in _F9_MODELS)
    if layer_inputs is not None:
        families += ((layer_rows, lambda: _layer_rows(layer_inputs)),)
    elif names is not None and layer_rows & names:
        raise ValueError(
            "The layer rows read the ledger's F9 surfaces: pass --layer-inputs."
        )
    voronoi_rows = frozenset(
        f"voronoi-{shape}" for shape in ("sphere", "cube", "torus", "steinmetz")
    )
    if voronoi_inputs is not None:
        families += ((voronoi_rows, lambda: _voronoi_rows(voronoi_inputs)),)
    elif names is not None and voronoi_rows & names:
        raise ValueError(
            "The voronoi rows read the ledger's F14 surfaces: pass --voronoi-inputs."
        )
    hexdom_rows = frozenset(f"hexdom-{model}-L{layers}" for model, layers, _ in _F8_CASES)
    if hexdom_inputs is not None:
        families += ((hexdom_rows, lambda: _hexdom_rows(hexdom_inputs)),)
    elif names is not None and hexdom_rows & names:
        raise ValueError(
            "The hexdom rows read the ledger's F8 surfaces: pass --hexdom-inputs."
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


# The ledger's F6 / F6b metric rows: the Gmsh OpenCASCADE unit cube meshed with
# MeshSizeMin = MeshSizeMax = h (about 1e4 / 1e5 / 1e6 tetrahedra; the row
# size is that nominal count), adapted toward targets sampled at its
# vertices: (iso) h = min(0.01 + 0.3 d, 0.1), d the distance to the sphere of
# radius 0.3 about the center; (aniso) normal size min(0.01 + 0.3 |z - 0.5|,
# 0.1) to the plane z = 0.5 and tangential size 0.1; (coarsen) twice the mean
# length of the edges at each vertex. Each row grades its target first (in
# the timed run), as MMG3D does by default (hgrad 1.3: h_q <= h_p + ln(1.3)
# |pq|), anisotropically for the anisotropic target. The references are
# MMG3D's, whose share of metric lengths in [0.7, 1.4] and mean quality a row
# may not fall below.
# The ceiling covers grading and construction of the topology edit. The
# published target is still certified; certification and transfer are timed
# separately from the construction samples.
_F6_SIZES = {10_000: 0.0782, 100_000: 0.0363, 1_000_000: 0.01685}
_F6_REFERENCE = {
    ("iso", 10_000): (0.774, 0.845),
    ("iso", 100_000): (0.870, 0.845),
    ("iso", 1_000_000): (0.918, 0.843),
    ("aniso", 10_000): (0.935, 0.847),
    ("aniso", 100_000): (0.906, 0.838),
    ("aniso", 1_000_000): (0.912, 0.839),
    ("coarsen", 10_000): (0.794, 0.827),
    ("coarsen", 100_000): (0.862, 0.839),
    ("coarsen", 1_000_000): (0.893, 0.842),
}


def _gmsh_cube(h: float, /) -> tuple[np.ndarray, np.ndarray]:
    from importlib import import_module

    gmsh = import_module("gmsh")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMin", h)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.model.mesh.generate(3)
    tags, coordinates, _ = gmsh.model.mesh.getNodes()
    types, _, nodes = gmsh.model.mesh.getElements(3)
    gmsh.finalize()
    order = np.argsort(tags)
    tags = np.asarray(tags)[order]
    points = np.asarray(coordinates, dtype=np.float64).reshape((-1, 3))[order]
    cells = np.searchsorted(
        tags,
        np.concatenate(
            [np.asarray(n).reshape((-1, 4)) for t, n in zip(types, nodes) if t == 4]
        ),
    ).astype(np.int32)
    corners = points[cells]
    volume = np.einsum(
        "ij,ij->i",
        corners[:, 1] - corners[:, 0],
        np.cross(corners[:, 2] - corners[:, 0], corners[:, 3] - corners[:, 0]),
    )
    cells[volume < 0.0] = cells[volume < 0.0][:, [0, 2, 1, 3]]
    return points, cells


def _tet_edges(cells: np.ndarray, /) -> np.ndarray:
    pairs = np.concatenate([cells[:, [i, j]] for i in range(4) for j in range(i + 1, 4)])
    return np.unique(np.sort(pairs, axis=1), axis=0)


def _f6_metric(field: str, points: np.ndarray, sizes: np.ndarray | None, /) -> np.ndarray:
    # Tensors 1 / h^2 (iso, coarsen) or diag(1/0.1^2, 1/0.1^2, 1/h_n^2) at points.
    match field:
        case "iso":
            d = np.abs(np.linalg.norm(points - 0.5, axis=1) - 0.3)
            h = np.minimum(0.01 + 0.3 * d, 0.1)
            return np.eye(3)[None] / (h * h)[:, None, None]
        case "aniso":
            normal = np.minimum(0.01 + 0.3 * np.abs(points[:, 2] - 0.5), 0.1)
            metric = np.zeros((points.shape[0], 3, 3), dtype=np.float64)
            metric[:, 0, 0] = metric[:, 1, 1] = 1.0 / 0.01
            metric[:, 2, 2] = 1.0 / (normal * normal)
            return metric
        case "coarsen":
            if sizes is None:
                raise ValueError(
                    "The coarsening target is sampled at the input vertices."
                )
            return np.eye(3)[None] / (sizes * sizes)[:, None, None]
        case _:
            raise ValueError(f"Unknown F6 field {field!r}.")


def _coarsen_sizes(points: np.ndarray, cells: np.ndarray, /) -> np.ndarray:
    edges = _tet_edges(cells)
    length = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    total = np.zeros(points.shape[0], dtype=np.float64)
    count = np.zeros(points.shape[0], dtype=np.float64)
    for end in range(2):
        np.add.at(total, edges[:, end], length)
        np.add.at(count, edges[:, end], 1.0)
    return 2.0 * total / count


class _F6Run(NamedTuple):
    outcome: _RouteOutcome
    construction_seconds: float
    finalization_seconds: float


def _f6_stage_seconds(result: object, /) -> float:
    if not isinstance(result, _F6Run):
        raise TypeError("The F6 row returns construction and finalization evidence.")
    return result.construction_seconds


def _f6_rows(sizes: tuple[int, ...], /) -> tuple[_Row, ...]:
    from phydrax._physical import SpatialCoordinateContract
    from phydrax.discretization._cell_mesh import CellBlock, CellMesh
    from phydrax.meshing import (
        grade_mesh_metric,
        MeshAdaptationPolicy,
        MeshAdaptationRoute,
        MeshingEntityKind,
        MeshingScope,
        MeshMetricField,
        MetricGradationPolicy,
        MetricMeshAdaptation,
        prepare_mesh_adaptation,
        PreparedMeshAdaptation,
    )
    from phydrax.meshing._adaptation import (
        _construct_tetra_metric_edit,
        _finish_tetra_metric_route,
    )
    from phydrax.meshing._canonical import certify_cell_mesh

    rows: list[_Row] = []
    for size, h in _F6_SIZES.items():
        if size not in sizes:
            continue
        points, cells = _gmsh_cube(h)
        block = CellBlock(
            "tetrahedra",
            "tetrahedron",
            cells,
            global_ids=np.arange(cells.shape[0], dtype=np.int64),
        )
        source = certify_cell_mesh(
            CellMesh(
                points,
                (block,),
                vertex_global_ids=np.arange(points.shape[0], dtype=np.int64),
            ),
            SpatialCoordinateContract.si(),
        )
        vertices = source.mesh.entity_set(0)
        scope = MeshingScope(
            source.mesh.mesh_id,
            source.mesh.numeric_version,
            MeshingEntityKind.MESH,
            0,
            vertices.entity_set_id,
            np.asarray(vertices.entity_ids, dtype=np.int64),
        )
        coarse = _coarsen_sizes(points, cells)
        edges = _tet_edges(cells)
        policy = MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_METRIC_3D)
        for field in ("iso", "aniso", "coarsen"):
            target = MeshMetricField(
                scope,
                _f6_metric(field, points, coarse),
                minimum_size=1.0e-3,
                maximum_size=10.0,
                maximum_anisotropy=100.0,
            )
            gradation = MetricGradationPolicy(
                1.0 + math.log(1.3), anisotropic=field == "aniso"
            )

            def run(
                source: CellMeshingResult = source,
                target: MeshMetricField = target,
                gradation: MetricGradationPolicy = gradation,
                edges: np.ndarray = edges,
                points: np.ndarray = points,
            ) -> _F6Run:
                def construct() -> tuple[
                    PreparedMeshAdaptation, MetricMeshAdaptation, TetraMetricOutcome
                ]:
                    graded, _ = grade_mesh_metric(
                        target, policy=gradation, adjacency=edges, coordinates=points
                    )
                    request = MetricMeshAdaptation(graded)
                    prepared = prepare_mesh_adaptation(source, request, policy=policy)
                    return (
                        prepared,
                        request,
                        _construct_tetra_metric_edit(prepared, request),
                    )

                construction, construction_seconds = measure_host(construct)
                outcome, finalization_seconds = measure_synchronized(
                    partial(_finish_tetra_metric_route, *construction)
                )
                return _F6Run(outcome, construction_seconds, finalization_seconds)

            rows.append(
                _Row(
                    f"metric-3d-{field}",
                    size,
                    run,
                    _adapted_cells,
                    quality=lambda result, field=field, size=size: _f6_quality(
                        result, field, _F6_REFERENCE[field, size]
                    ),
                    stage_seconds=_f6_stage_seconds,
                )
            )
    return tuple(rows)


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


# F13 (ledger providers.md): a background on [-3, 3]^3 and a shell r in
# [0.5, 1.5] around a wall at r = 0.5 with its receptor boundary at r = 1.5.
# Hexahedra: the ledger driver's N^3 grid and equi-angular cubed-sphere shell
# (N, n, layers); tetrahedra: the ledger's Kuhn background and Gmsh shell,
# cells sorted by centroid z (f13_tet_setup.py --sorted).
_F13_HEX = {10_000: (17, 9, 9), 100_000: (37, 20, 20), 1_000_000: (79, 44, 44)}
_F13_TET = {10_000: "tet1e4", 100_000: "tet1e5", 1_000_000: "tet1e6"}
_HEX_FACES = (
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
)
_TET_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))


class _OversetCase(NamedTuple):
    vertex_offsets: np.ndarray
    vertices: np.ndarray
    cell_offsets: np.ndarray
    cells: np.ndarray
    face_offsets: np.ndarray
    faces: np.ndarray
    roles: np.ndarray


def _f13_grid(count: int, /) -> tuple[np.ndarray, np.ndarray]:
    side = count + 1
    axis = -3.0 + 6.0 * np.arange(side, dtype=np.float64) / count
    k, j, i = np.meshgrid(*(np.arange(side, dtype=np.int64),) * 3, indexing="ij")
    points = np.stack((axis[i.ravel()], axis[j.ravel()], axis[k.ravel()]), axis=1)
    k, j, i = (
        value.ravel() for value in np.meshgrid(*(np.arange(count),) * 3, indexing="ij")
    )

    def vertex(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        return (c * side + b) * side + a

    cells = np.stack(
        (
            vertex(i, j, k),
            vertex(i + 1, j, k),
            vertex(i + 1, j + 1, k),
            vertex(i, j + 1, k),
            vertex(i, j, k + 1),
            vertex(i + 1, j, k + 1),
            vertex(i + 1, j + 1, k + 1),
            vertex(i, j + 1, k + 1),
        ),
        axis=1,
    )
    return points, cells


def _f13_shell(count: int, layers: int, /) -> tuple[np.ndarray, np.ndarray]:
    # The ledger driver's cubed-sphere shell: node keys rounded at 1e-9
    # de-duplicate face seams, and each hexahedron is turned positive.
    angles = np.tan(np.pi / 4 * (-1 + 2.0 * np.arange(count + 1) / count))
    face, layer, j, i = np.meshgrid(
        np.arange(6),
        np.arange(layers + 1),
        np.arange(count + 1),
        np.arange(count + 1),
        indexing="ij",
    )
    a, b, one = angles[i], angles[j], np.ones(i.shape)
    directions = np.select(
        [face[..., None] == side for side in range(6)],
        [
            np.stack(rows, axis=-1)
            for rows in (
                (one, a, b),
                (-one, -a, b),
                (-a, one, b),
                (a, -one, b),
                (-b, a, one),
                (b, a, -one),
            )
        ],
    )
    scale = 1.0 / np.sqrt(np.sum(directions * directions, axis=-1))
    radius = 0.5 + layer / layers
    points = (directions * scale[..., None] * radius[..., None]).reshape((-1, 3))
    keys = np.rint(points * 1e9).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    ids = rank[inverse.reshape((-1,))].reshape(face.shape)
    vertices = points[first[order]]
    f, l, jj, ii = (
        value.ravel()
        for value in np.meshgrid(
            np.arange(6),
            np.arange(layers),
            np.arange(count),
            np.arange(count),
            indexing="ij",
        )
    )
    cells = np.stack(
        (
            ids[f, l, jj, ii],
            ids[f, l, jj, ii + 1],
            ids[f, l, jj + 1, ii + 1],
            ids[f, l, jj + 1, ii],
            ids[f, l + 1, jj, ii],
            ids[f, l + 1, jj, ii + 1],
            ids[f, l + 1, jj + 1, ii + 1],
            ids[f, l + 1, jj + 1, ii],
        ),
        axis=1,
    )
    corners = vertices[cells]
    turn = (
        np.einsum(
            "ij,ij->i",
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 3] - corners[:, 0]),
            corners[:, 4] - corners[:, 0],
        )
        < 0
    )
    cells[turn] = cells[turn][:, [0, 3, 2, 1, 4, 7, 6, 5]]
    return vertices, cells


def _exterior_faces(cells: np.ndarray, /) -> np.ndarray:
    # Faces used by one cell, as corner cycles padded to four.
    local = _HEX_FACES if cells.shape[1] == 8 else _TET_FACES
    rows = np.concatenate([cells[:, list(face)] for face in local])
    keys = np.sort(rows, axis=1)
    _, first, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
    exterior = rows[np.sort(first[counts == 1])]
    if exterior.shape[1] == 3:
        exterior = np.concatenate(
            (exterior, np.full((exterior.shape[0], 1), -1, dtype=exterior.dtype)), axis=1
        )
    return exterior


def _overset_case(
    background: tuple[np.ndarray, np.ndarray], shell: tuple[np.ndarray, np.ndarray], /
) -> _OversetCase:
    vertex_rows, cell_rows, face_rows, role_rows = [], [], [], []
    base = 0
    for index, (points, cells) in enumerate((background, shell)):
        faces = _exterior_faces(cells)
        radius = np.linalg.norm(points, axis=1)
        roles = np.zeros((faces.shape[0],), dtype=np.uint8)
        if index == 1:
            corner_radius = np.where(faces >= 0, radius[np.maximum(faces, 0)], np.nan)
            on = np.abs(corner_radius - 0.5) < 1e-6
            roles[np.all(on | (faces < 0), axis=1)] = 1
            on = np.abs(corner_radius - 1.5) < 1e-6
            roles[np.all(on | (faces < 0), axis=1)] = 2
        padded = np.full((cells.shape[0], 8), -1, dtype=np.int32)
        padded[:, : cells.shape[1]] = cells + base
        vertex_rows.append(points)
        cell_rows.append(padded)
        face_rows.append(np.where(faces >= 0, faces + base, -1).astype(np.int32))
        role_rows.append(roles)
        base += points.shape[0]

    def offsets(rows: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(([0], np.cumsum([row.shape[0] for row in rows]))).astype(
            np.int64
        )

    return _OversetCase(
        offsets(vertex_rows),
        np.ascontiguousarray(np.concatenate(vertex_rows), dtype=np.float64),
        offsets(cell_rows),
        np.concatenate(cell_rows),
        offsets(face_rows),
        np.concatenate(face_rows),
        np.concatenate(role_rows),
    )


def _overset_kernel(case: _OversetCase, /) -> _meshcore.OversetAssembly:
    # The bare native assembly on the same flat blocks (untimed sub-field).
    cell_count = case.cells.shape[0]
    return _meshcore.overset_assemble(
        case.vertex_offsets,
        case.vertices,
        case.cell_offsets,
        case.cells,
        np.arange(cell_count, dtype=np.int64),
        np.zeros((cell_count,), dtype=np.uint8),
        case.face_offsets,
        case.faces,
        case.roles,
        np.zeros((2,), dtype=np.int32),
        np.arange(2, dtype=np.int32),
        fringe_layers=1,
        donor_priority=0,
        fringe_donors=True,
        prove=False,
    )


class _OversetApiCase(NamedTuple):
    # The consumer's inputs to prepare_overset_connectivity, built untimed
    # (as TIOGA's grids are registered before its timed call), and the same
    # meshes as flat blocks for the independent check.
    flat: _OversetCase
    assembly: MeshAssembly
    specs: tuple[OversetPartSpec, OversetPartSpec]
    policy: OversetPolicy


def _overset_part(name: str, points: np.ndarray, cells: np.ndarray, /) -> MeshPart:
    kind = "hexahedron" if cells.shape[1] == 8 else "tetrahedron"
    mesh = CellMesh(
        jnp.asarray(points, dtype=jnp.float64),
        (CellBlock(name, kind, jnp.asarray(cells, dtype=jnp.int32)),),
    )
    return MeshPart(name, certify_cell_mesh(mesh, SpatialCoordinateContract.si()))


def _overset_api_case(
    background: tuple[np.ndarray, np.ndarray],
    shell: tuple[np.ndarray, np.ndarray],
    policy: OversetPolicy,
    /,
) -> _OversetApiCase:
    parts = (
        _overset_part("background", *background),
        _overset_part("shell", *shell),
    )
    background_mesh, shell_mesh = (_cell_result(part).mesh for part in parts)
    flat = _overset_case(
        (
            np.asarray(background_mesh.coordinates, dtype=np.float64),
            np.asarray(background_mesh.blocks[0].vertices, dtype=np.int64),
        ),
        (
            np.asarray(shell_mesh.coordinates, dtype=np.float64),
            np.asarray(shell_mesh.blocks[0].vertices, dtype=np.int64),
        ),
    )
    # Shell exterior faces: the wall at r = 0.5, the receptor boundary at 1.5.
    entity_set = shell_mesh.entity_set(2)
    identifiers = np.asarray(entity_set.entity_ids, dtype=np.int64)
    boundary = np.asarray(entity_set.subset("boundary").mask, dtype=np.bool_)
    connectivity = shell_mesh.connectivity
    if isinstance(connectivity, (HexahedralConnectivity, TetrahedralConnectivity)):
        faces = np.asarray(connectivity.faces, dtype=np.int64)
    elif isinstance(connectivity, SimplicialConnectivity):
        faces = np.asarray(connectivity.entities[2], dtype=np.int64)
    else:
        raise TypeError("F13 shells are tetrahedral or hexahedral.")
    radius = np.linalg.norm(
        np.asarray(shell_mesh.coordinates, dtype=np.float64)[faces], axis=2
    ).mean(axis=1)
    specs = (
        OversetPartSpec("background"),
        OversetPartSpec(
            "shell",
            wall=parts[1].scope(2, identifiers[boundary & (radius < 1.0)]),
            boundary=parts[1].scope(2, identifiers[boundary & (radius > 1.0)]),
        ),
    )
    return _OversetApiCase(flat, MeshAssembly(parts), specs, policy)


def _overset_run(case: _OversetApiCase, /, *, prove: bool) -> OversetConnectivity:
    return prepare_overset_connectivity(
        case.assembly, case.specs, policy=case.policy, prove=prove
    )


def _overset_cells(result: object, /) -> int:
    if not isinstance(result, OversetConnectivity):
        raise TypeError("Overset rows return OversetConnectivity records.")
    return sum(int(item.cell_status.shape[0]) for item in result.blanking)


def _overset_unmet(result: object, /) -> int:
    # Orphan receptors and unproved ambiguous cells.
    if not isinstance(result, OversetConnectivity):
        raise TypeError("Overset rows return OversetConnectivity records.")
    return result.orphan_count + result.ambiguous_count


def _overset_outcome(
    case: _OversetCase, result: OversetConnectivity, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Vertex status, donor status and flat donor cell per flat vertex, read
    # from the public records (receptor evidence names donors by cell ID).
    vertex_status = np.concatenate(
        [np.asarray(item.vertex_status, dtype=np.int64) for item in result.blanking]
    )
    donor_status = np.full(vertex_status.shape, -1, dtype=np.int64)
    donor_cell = np.full(vertex_status.shape, -1, dtype=np.int64)
    cell_ids = [np.asarray(item.cell_ids, dtype=np.int64) for item in result.geometries]
    for index, evidence in enumerate(result.receptors):
        rows = int(case.vertex_offsets[index]) + np.asarray(
            evidence.receptor_rows, dtype=np.int64
        )
        donor_status[rows] = np.asarray(evidence.status)
        parts = np.asarray(evidence.donor_parts, dtype=np.int64)
        cells = np.asarray(evidence.donor_cells, dtype=np.int64)
        for donor, ids in enumerate(cell_ids):
            selected = parts == donor
            order = np.argsort(ids, kind="stable")
            local = order[np.searchsorted(ids[order], cells[selected])]
            donor_cell[rows[selected]] = int(case.cell_offsets[donor]) + local
    return vertex_status, donor_status, donor_cell


def _trilinear_shapes(xi: np.ndarray, /) -> np.ndarray:
    a, b, c = xi[:, 0], xi[:, 1], xi[:, 2]
    return np.stack(
        (
            (1 - a) * (1 - b) * (1 - c),
            a * (1 - b) * (1 - c),
            a * b * (1 - c),
            (1 - a) * b * (1 - c),
            (1 - a) * (1 - b) * c,
            a * (1 - b) * c,
            a * b * c,
            (1 - a) * b * c,
        ),
        axis=1,
    )


def _independent_reference(corners: np.ndarray, points: np.ndarray, /) -> np.ndarray:
    # Barycentric weights of tetrahedra or Newton trilinear weights of
    # hexahedra, as the ledger's f13_check.py computes them.
    if corners.shape[1] == 4:
        matrix = np.transpose(corners[:, 1:] - corners[:, :1], (0, 2, 1))
        reference = np.linalg.solve(matrix, (points - corners[:, 0])[..., None])[..., 0]
        return np.concatenate(
            (1 - reference.sum(axis=1, keepdims=True), reference), axis=1
        )
    xi = np.full((points.shape[0], 3), 0.5)
    for _ in range(30):
        a, b, c = 1 - xi[:, 0], 1 - xi[:, 1], 1 - xi[:, 2]
        x, y, z = xi[:, 0], xi[:, 1], xi[:, 2]
        derivatives = (
            np.stack(
                (-b * c, b * c, y * c, -y * c, -b * z, b * z, y * z, -y * z), axis=1
            ),
            np.stack(
                (-a * c, -x * c, x * c, a * c, -a * z, -x * z, x * z, a * z), axis=1
            ),
            np.stack(
                (-a * b, -x * b, -x * y, -a * y, a * b, x * b, x * y, a * y), axis=1
            ),
        )
        jacobian = np.stack(
            [np.einsum("ij,ijk->ik", d, corners) for d in derivatives], axis=2
        )
        residual = np.einsum("ij,ijk->ik", _trilinear_shapes(xi), corners) - points
        xi = xi - np.linalg.solve(jacobian, residual[..., None])[..., 0]
    return _trilinear_shapes(xi)


def _linear_field(x: np.ndarray, /) -> np.ndarray:
    return 1 + x[..., 0] + 2 * x[..., 1] + 3 * x[..., 2]


def _overset_donor_check(
    case: _OversetCase, donor_status: np.ndarray, donor_cell: np.ndarray, /
) -> tuple[int, float]:
    # Donors recomputed in NumPy: containment within 1e-9, a mapping residual
    # under 1e-9, and the linear-reproduction error.
    found = np.flatnonzero(donor_status == 0)
    cells = case.cells[donor_cell[found]]
    failures = 0
    linear_error = 0.0
    hexes = cells[:, 4] >= 0
    for selection, width in ((hexes, 8), (~hexes, 4)):
        if not np.any(selection):
            continue
        corners = case.vertices[cells[selection][:, :width]]
        points = case.vertices[found[selection]]
        weights = _independent_reference(corners, points)
        mapped = np.einsum("ij,ijk->ik", weights, corners)
        outside = np.maximum(-weights.min(axis=1), weights.max(axis=1) - 1)
        residual = np.linalg.norm(mapped - points, axis=1)
        failures += int(np.count_nonzero((outside > 1e-9) | (residual > 1e-9)))
        reproduced = np.sum(weights * _linear_field(corners), axis=1)
        linear_error = max(
            linear_error, float(np.max(np.abs(reproduced - _linear_field(points))))
        )
    return failures, linear_error


def _overset_quality(
    api: _OversetApiCase, result: object, /
) -> tuple[int, dict[str, float]]:
    # Independent of the implementation, as the ledger's f13_check.py: every
    # donor of the public records recomputed in NumPy, mandatory receptors
    # left active, and background holes against the wall radius (inside
    # r < 0.45, none beyond r = 0.5: the walls are inscribed in the sphere).
    # Untimed: the certified call (prove=True), which must be complete with
    # the same blanking and donors that pass the same check (certification
    # may pick another containing cell on a shared face), and the bare
    # kernel's own clock as a sub-field.
    if not isinstance(result, OversetConnectivity):
        raise TypeError("Overset rows return OversetConnectivity records.")
    case = api.flat
    status, donor_status, donor_cell = _overset_outcome(case, result)
    found = np.flatnonzero(donor_status == 0)
    failures, linear_error = _overset_donor_check(case, donor_status, donor_cell)
    background = slice(int(case.vertex_offsets[0]), int(case.vertex_offsets[1]))
    radius = np.linalg.norm(case.vertices[background], axis=1)
    # Background vertices of cells touching a hole must not stay active.
    rows = case.cells[int(case.cell_offsets[0]) : int(case.cell_offsets[1])]
    corner_status = np.where(rows >= 0, status[np.maximum(rows, 0)], -1)
    touching = np.any(corner_status == 2, axis=1)
    left_active = np.unique(rows[touching][corner_status[touching] == 0]).size
    shell_faces = case.faces[int(case.face_offsets[1]) :]
    boundary_vertices = np.unique(
        shell_faces[case.roles[int(case.face_offsets[1]) :] == 2]
    )
    boundary_vertices = boundary_vertices[boundary_vertices >= 0]
    missing_receptors = int(np.count_nonzero(status[boundary_vertices] != 1))
    inside_not_hole = int(np.count_nonzero((radius < 0.45) & (status[background] != 2)))
    holes_beyond = int(np.count_nonzero((radius > 0.5) & (status[background] == 2)))
    certified, certified_seconds = measure_host(partial(_overset_run, api, prove=True))
    certified_status, certified_donor_status, certified_cell = _overset_outcome(
        case, certified
    )
    certified_failures = _overset_donor_check(
        case, certified_donor_status, certified_cell
    )[0]
    try:
        certified.require_complete()
        incomplete = 0
    except OversetConnectivityError:
        incomplete = 1
    if not np.array_equal(certified_status, status) or not np.array_equal(
        certified_donor_status, donor_status
    ):
        incomplete = 1
    kernel = _overset_kernel(case)
    # Evidence for the row's explicit policy, untimed: the default policy
    # (two fringe layers, active donors only) on the same parts.
    default = _overset_run(api._replace(policy=OversetPolicy()), prove=True)
    # The row's fringe depth with the default donor rule (active donors only).
    active_only = _overset_run(
        api._replace(
            policy=OversetPolicy(
                fringe_layers=api.policy.fringe_layers, fringe_donors=False
            )
        ),
        prove=True,
    )
    measured = {
        "donor_contains_receptor_failures": float(failures),
        "linear_reproduction_max_abs_error": linear_error,
        "orphans": float(result.orphan_count),
        "ambiguous_cells": float(result.ambiguous_count),
        "receptors": float(found.size + result.orphan_count),
        "background_holes": float(np.count_nonzero(status[background] == 2)),
        "background_receptors": float(np.count_nonzero(status[background] == 1)),
        "shell_receptors": float(
            np.count_nonzero(status[int(case.vertex_offsets[1]) :] == 1)
        ),
        "hole_adjacent_left_active": float(left_active),
        "boundary_not_receptor": float(missing_receptors),
        "background_inside_0.45_not_hole": float(inside_not_hole),
        "background_holes_beyond_0.5": float(holes_beyond),
        # prove=True, outside the ceiling: every donor certified.
        "certified_seconds": certified_seconds,
        "certified_incomplete_or_blanking_differs": float(incomplete),
        "certified_donor_failures": float(certified_failures),
        "certified_donors_changed": float(
            np.count_nonzero(certified_cell[found] != donor_cell[found])
        ),
        "policy_fringe_layers": float(api.policy.fringe_layers),
        "policy_fringe_donors": float(api.policy.fringe_donors),
        "default_policy_orphans": float(default.orphan_count),
        "default_policy_ambiguous_cells": float(default.ambiguous_count),
        "active_donor_only_orphans": float(active_only.orphan_count),
        "kernel_seconds": float(kernel.stats[7]),
        "fringe_donors": float(kernel.stats[14]),
        "shared_face_donors": float(kernel.stats[15]),
    }
    missed = (
        failures
        + left_active
        + missing_receptors
        + inside_not_hole
        + holes_beyond
        + incomplete
        + certified_failures
    )
    return missed, measured


def _overset_rows(sizes: tuple[int, ...], inputs: Path | None, /) -> tuple[_Row, ...]:
    # The rows time the public prepare_overset_connectivity(prove=False) on
    # parts built before timing, under this explicit policy: TIOGA's donors
    # may hold fringe vertices; ours take a fringe donor only where no active
    # cell contains the receptor. The default policy's orphans are reported.
    policy = OversetPolicy(fringe_layers=1, fringe_donors=True)
    rows: list[_Row] = []
    for size in sizes:
        if size not in _F13_HEX:
            continue
        grid, count, layers = _F13_HEX[size]
        cases = [("overset-hex", _f13_grid(grid), _f13_shell(count, layers))]
        if inputs is not None:
            stored = np.load(inputs / f"f13-{_F13_TET[size]}s.npz")
            cases.append(
                (
                    "overset-tet",
                    (
                        np.asarray(stored["xb"], dtype=np.float64),
                        np.asarray(stored["tb"], dtype=np.int64),
                    ),
                    (
                        np.asarray(stored["xs"], dtype=np.float64),
                        np.asarray(stored["ts"], dtype=np.int64),
                    ),
                )
            )
        for name, background, shell in cases:
            case = _overset_api_case(background, shell, policy)
            rows.append(
                _Row(
                    name,
                    size,
                    partial(_overset_run, case, prove=False),
                    _overset_cells,
                    unmet=_overset_unmet,
                    quality=partial(_overset_quality, case),
                )
            )
    return tuple(rows)


def _adapted_arrays(result: _F6Run, /) -> tuple[np.ndarray, np.ndarray]:
    mesh = result.outcome.target.mesh
    cells = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    return np.asarray(mesh.coordinates, dtype=np.float64), cells


def _adapted_cells(result: object, /) -> int:
    if not isinstance(result, _F6Run):
        raise TypeError("The F6 row returns construction and finalization evidence.")
    return sum(block.vertices.shape[0] for block in result.outcome.target.mesh.blocks)


def _f6_quality(
    result: object, field: str, reference: tuple[float, float], /
) -> tuple[int, dict[str, float]]:
    # Ledger F6: midpoint lengths, centroid metric mean ratio, and inverted
    # cells. Coarsening uses the adapted tensors. A stall or a quality miss
    # fails the row even when its timing is within the ceiling.
    from phydrax.meshing import MeshAdaptationStatus

    if not isinstance(result, _F6Run):
        raise TypeError("The F6 row returns construction and finalization evidence.")
    points, cells = _adapted_arrays(result)
    edges = _tet_edges(cells)
    vector = points[edges[:, 1]] - points[edges[:, 0]]
    if field == "coarsen":
        if result.outcome.metric is None:
            raise ValueError("A metric route publishes its target metric.")
        tensors = np.asarray(result.outcome.metric.values, dtype=np.float64)
        midpoint = 0.5 * (tensors[edges[:, 0]] + tensors[edges[:, 1]])
        centroid = tensors[cells].mean(axis=1)
    else:
        midpoint = _f6_metric(
            field, 0.5 * (points[edges[:, 0]] + points[edges[:, 1]]), None
        )
        centroid = _f6_metric(field, points[cells].mean(axis=1), None)
    length = np.sqrt(np.einsum("ni,nij,nj->n", vector, midpoint, vector))
    corners = points[cells]
    volume = (
        np.einsum(
            "ij,ij->i",
            corners[:, 1] - corners[:, 0],
            np.cross(corners[:, 2] - corners[:, 0], corners[:, 3] - corners[:, 0]),
        )
        / 6.0
    )
    squares = np.zeros(cells.shape[0], dtype=np.float64)
    for i in range(4):
        for j in range(i + 1, 4):
            side = corners[:, j] - corners[:, i]
            squares += np.einsum("ni,nij,nj->n", side, centroid, side)
    metric_volume = np.abs(volume) * np.sqrt(np.linalg.det(centroid))
    quality = 12.0 * (3.0 * metric_volume) ** (2.0 / 3.0) / squares
    share = float(np.mean((length >= 0.7) & (length <= 1.4)))
    inverted = int(np.count_nonzero(volume <= 0.0))
    missed = (
        inverted
        + int(share < reference[0])
        + int(float(quality.mean()) < reference[1])
        + int(result.outcome.status is not MeshAdaptationStatus.COMPLETE)
    )
    return missed, {
        "share_0_7_1_4": share,
        "quality_min": float(quality.min()),
        "quality_mean": float(quality.mean()),
        "inverted": float(inverted),
        "certification_and_transfer_seconds": result.finalization_seconds,
        "reference_share": reference[0],
        "reference_quality_mean": reference[1],
    }


# Ledger F9: five prism layers, first height 0.01, growth 1.2 (total 0.074416),
# grown inward from the Gmsh surface meshes the reference tools layer (h 0.1
# and 0.05; the ledger's gmsh-<model>-h<h>-t0.01-n5-sign-1-surf.stl).
_F9_THICKNESSES = 0.01 * 1.2 ** np.arange(5, dtype=np.float64)
_F9_HEIGHTS = np.cumsum(_F9_THICKNESSES)
# The best reference depth band (Netgen on the sphere: 0.0740-0.0744 by true
# distance) bounds the depth error of every unthinned column top.
_F9_DEPTH_TOLERANCE = 4e-4
# Gmsh's classification angle in the reference driver (classifySurfaces 40).
_F9_FEATURE_ANGLE = 40.0
# The certified repair's thinning floor (Netgen thins to 11-17 % at edges).
_F9_MINIMUM_SCALE = 0.1
_F9_MODELS = (("layers-sphere", "sphere"), ("layers-quadric", "quadric"))


class _LayerCase(NamedTuple):
    model: str
    vertices: np.ndarray
    # Outward-oriented wall triangles.
    faces: np.ndarray
    # The core's size target: the mean wall edge.
    size: float
    # The public call's inputs, built before the timer (the references get
    # their surface mesh): the inward-oriented wall and its control.
    wall: CellMesh
    control: BoundaryLayerControl


class _LayerResult(NamedTuple):
    layers: BoundaryLayerMesh
    # Layer mesh vertices, then the core's own vertices.
    points: np.ndarray
    prisms: np.ndarray
    tetrahedra: np.ndarray
    unmet: int
    layer_seconds: float
    core_seconds: float


def _read_ascii_stl(path: Path, /) -> tuple[np.ndarray, np.ndarray]:
    corners = np.asarray(
        [
            line.split()[1:4]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.lstrip().startswith("vertex")
        ],
        dtype=np.float64,
    )
    vertices, inverse = np.unique(corners, axis=0, return_inverse=True)
    return vertices, inverse.reshape((-1, 3)).astype(np.int64)


def _layered_mesh(case: _LayerCase, /) -> _LayerResult:
    # The whole task of Gmsh's timed call (classify, extrude, 3D): the public
    # prepare_boundary_layers on the wall surface (certified prism layers
    # with their certificates and ancestry), then the core meshed by S3 with
    # the layers' cap as fixed boundary; one conforming mixed mesh out.
    start = time.perf_counter()
    layers = prepare_boundary_layers(case.wall, case.control)
    layered = time.perf_counter()
    if layers.cap is None:
        raise RuntimeError("Closed-wall layers must leave a cap.")
    cap_points = np.asarray(layers.cap.coordinates, dtype=np.float64)
    # The cap faces away from the layers (into the core); the core's
    # boundary is oriented outward.
    cap_faces = np.asarray(layers.cap.blocks[0].vertices, dtype=np.int64)[:, ::-1]
    count = cap_points.shape[0]
    recovered = _recover(
        cap_points, cap_faces, np.zeros((cap_faces.shape[0],), dtype=np.int32)
    )
    if recovered.status is not _meshcore.MeshcoreStatus.OK:
        raise RuntimeError(f"Core recovery ended with {recovered.status!r}.")
    mesh = _meshcore.TetMesh3D.from_recovery(
        recovered, boundary_policy=_meshcore.PlcBoundaryPolicy.FIXED
    )
    sizes = np.full((recovered.points.shape[0],), case.size, dtype=np.float64)
    refined = mesh.refine(radius_edge_bound=2.0, sizes=sizes)
    if refined.status not in (
        _meshcore.MeshcoreStatus.OK,
        _meshcore.MeshcoreStatus.UNRESOLVED,
    ):
        raise RuntimeError(f"Core refinement failed with status {refined.status!r}.")
    improved = mesh.improve(radius_edge_bound=2.0, min_dihedral_degrees=15.0)
    core = mesh.arrays()
    if not np.array_equal(core.points[:count], cap_points):
        raise RuntimeError("The core must keep the cap's vertices first.")
    layer_points = np.asarray(layers.mesh.coordinates, dtype=np.float64)
    cap_vertices = np.asarray(layers.cap_vertices, dtype=np.int64)
    cells = core.tetrahedra.astype(np.int64)
    tetrahedra = np.where(
        cells < count,
        cap_vertices[np.minimum(cells, count - 1)],
        cells - count + layer_points.shape[0],
    )
    return _LayerResult(
        layers,
        np.concatenate((layer_points, core.points[count:])),
        np.asarray(layers.mesh.blocks[0].vertices, dtype=np.int64),
        tetrahedra,
        improved.unmet_cells,
        layered - start,
        time.perf_counter() - layered,
    )


def _layer_result(result: object, /) -> _LayerResult:
    if not isinstance(result, _LayerResult):
        raise TypeError("The layer row returns a _LayerResult.")
    return result


def _layer_cells(result: object, /) -> int:
    layered = _layer_result(result)
    return layered.prisms.shape[0] + layered.tetrahedra.shape[0]


def _layer_unmet(result: object, /) -> int:
    return _layer_result(result).unmet


def _solid_depth(model: str, points: np.ndarray, /) -> np.ndarray:
    # Signed true distance to the solid's boundary, positive inside: the unit
    # sphere, and the quadric as the intersection of the z cylinder r 1 and
    # the x cylinder r 0.6 (the distance to the complement of a convex
    # intersection is the least distance to a complement).
    match model:
        case "sphere":
            return 1.0 - np.linalg.norm(points, axis=-1)
        case "quadric":
            return np.minimum(
                1.0 - np.hypot(points[:, 0], points[:, 1]),
                0.6 - np.hypot(points[:, 1], points[:, 2]),
            )
        case _:
            raise ValueError(f"Unknown F9 model {model!r}.")


def _prism_corner_jacobians(points: np.ndarray, prisms: np.ndarray, /) -> np.ndarray:
    # The ledger's six corner Jacobians per prism (meas_bl.py): at each bottom
    # and top corner, det[edge to the next corner, edge to the one after, the
    # corner's own column edge].
    corners = points[prisms]
    values = []
    for k in range(3):
        a, b, c = k, (k + 1) % 3, (k + 2) % 3
        up = corners[:, a + 3] - corners[:, a]
        for level in (0, 3):
            values.append(
                np.linalg.det(
                    np.stack(
                        (
                            corners[:, b + level] - corners[:, a + level],
                            corners[:, c + level] - corners[:, a + level],
                            up,
                        ),
                        axis=2,
                    )
                )
            )
    return np.stack(values, axis=1)


def _layer_quality(case: _LayerCase, result: object, /) -> tuple[int, dict[str, float]]:
    layered = _layer_result(result)
    layers = layered.layers
    # The kernel the public call runs, once more outside the timing: its own
    # seconds and vertex kinds; the public layers must be its levels.
    kernel = _meshcore.prism_layers(
        case.vertices,
        case.faces[:, ::-1],
        _F9_HEIGHTS,
        feature_angle_degrees=_F9_FEATURE_ANGLE,
        minimum_scale=_F9_MINIMUM_SCALE,
        prove=True,
    )
    count = case.vertices.shape[0]
    layer_points = np.asarray(layers.mesh.coordinates, dtype=np.float64)
    levels = layer_points.shape[0]
    scale = np.asarray(layers.column_scale, dtype=np.float64)
    jacobians = _prism_corner_jacobians(layered.points, layered.prisms)
    largest = np.max(np.abs(jacobians), axis=1)
    lowest = np.min(jacobians, axis=1)
    inverted_prisms = int(np.count_nonzero(lowest <= 1e-12 * largest))
    ratios = lowest / largest
    corners = layered.points[layered.tetrahedra]
    volumes = np.linalg.det(
        np.stack(
            (
                corners[:, 1] - corners[:, 0],
                corners[:, 2] - corners[:, 0],
                corners[:, 3] - corners[:, 0],
            ),
            axis=2,
        )
    )
    inverted_tets = int(np.count_nonzero(volumes <= 0.0))
    top_depth = _solid_depth(case.model, layer_points[levels - count :])
    full = scale == 1.0
    total = float(_F9_HEIGHTS[-1])
    depth_error = float(np.max(np.abs(top_depth[full] - total))) if np.any(full) else 0.0
    generated = _solid_depth(case.model, layered.points[count:])
    outside = int(np.count_nonzero(generated < -1e-9))
    ridge = kernel.vertex_kind != 0
    thinned = scale < 1.0
    # Prisms whose wall triangle lies within 1.5 h of a sharp edge (meas_bl).
    bottom = layered.points[layered.prisms[:, :3]].mean(axis=1)
    edge = float(
        np.median(
            np.linalg.norm(
                case.vertices[case.faces[:, 1]] - case.vertices[case.faces[:, 0]], axis=1
            )
        )
    )
    near = (
        (np.abs(np.hypot(bottom[:, 0], bottom[:, 1]) - 1.0) < 1.5 * edge)
        & (np.abs(np.hypot(bottom[:, 1], bottom[:, 2]) - 0.6) < 1.5 * edge)
        if case.model == "quadric"
        else np.zeros((bottom.shape[0],), dtype=np.bool_)
    )
    cell_count = sum(block.cell_count for block in layers.mesh.blocks)
    certified = layers.validity.certified_valid_count == cell_count
    matches_kernel = bool(np.array_equal(layer_points, kernel.points))
    missed = (
        inverted_prisms
        + inverted_tets
        + outside
        + int(depth_error > _F9_DEPTH_TOLERANCE)
        + int(not certified)
        + int(not matches_kernel)
    )
    return missed, {
        "prisms": float(layered.prisms.shape[0]),
        "tetrahedra": float(layered.tetrahedra.shape[0]),
        "inverted_prisms": float(inverted_prisms),
        "inverted_tetrahedra": float(inverted_tets),
        "minimum_corner_ratio": float(np.min(ratios)),
        "minimum_corner_ratio_near_sharp_edges": float(np.min(ratios[near]))
        if np.any(near)
        else math.nan,
        "prisms_near_sharp_edges": float(np.count_nonzero(near)),
        "top_depth_minimum": float(np.min(top_depth)),
        "top_depth_maximum": float(np.max(top_depth)),
        "unthinned_top_depth_error": depth_error,
        "layer_depth_maximum": float(np.max(_solid_depth(case.model, layer_points))),
        "intended_depth": total,
        "generated_nodes_outside": float(outside),
        "wall_nodes_outside_given": float(
            np.count_nonzero(_solid_depth(case.model, case.vertices) < -1e-9)
        ),
        "ridge_vertices": float(np.count_nonzero(ridge)),
        "thinned_vertices": float(np.count_nonzero(thinned)),
        "ridge_thinning_fraction": float(np.count_nonzero(thinned & ridge))
        / max(float(np.count_nonzero(ridge)), 1.0),
        "minimum_scale": float(np.min(scale)),
        "certified": float(certified),
        "certified_cells": float(layers.validity.certified_valid_count),
        "public_matches_kernel": float(matches_kernel),
        "layer_step_seconds": layered.layer_seconds,
        "layer_kernel_seconds": float(kernel.stats[6]),
        "layer_certification_seconds": float(kernel.stats[5]),
        "core_seconds": layered.core_seconds,
    }


def _layer_rows(inputs: Path, /) -> tuple[_Row, ...]:
    # Row sizes are the request h in thousandths.
    rows: list[_Row] = []
    for name, model in _F9_MODELS:
        for thousandths in (100, 50):
            vertices, faces = _read_ascii_stl(
                inputs / f"gmsh-{model}-h{thousandths / 1000:g}-t0.01-n5-sign-1-surf.stl"
            )
            edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
            spacing = float(
                np.mean(
                    np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
                )
            )
            wall = CellMesh(
                vertices,
                (
                    CellBlock(
                        "wall",
                        "triangle",
                        faces[:, ::-1].copy(),
                        global_ids=np.arange(faces.shape[0], dtype=np.int64),
                    ),
                ),
            )
            cells = wall.entity_set(2)
            control = BoundaryLayerControl(
                MeshingScope(
                    wall.mesh_id,
                    wall.numeric_version,
                    MeshingEntityKind.MESH,
                    2,
                    cells.entity_set_id,
                    cells.entity_ids,
                ),
                LayerSchedule(_F9_THICKNESSES),
                route=BoundaryLayerRoute.ADVANCING,
                collision=BoundaryLayerCollisionPolicy.REDUCE_THICKNESS,
                corner=BoundaryLayerCornerPolicy.SMOOTH,
                feature_angle=math.radians(_F9_FEATURE_ANGLE),
                minimum_thickness_fraction=_F9_MINIMUM_SCALE,
            )
            case = _LayerCase(model, vertices, faces, spacing, wall, control)
            rows.append(
                _Row(
                    name,
                    thousandths,
                    partial(_layered_mesh, case),
                    _layer_cells,
                    unmet=_layer_unmet,
                    quality=partial(_layer_quality, case),
                )
            )
    return tuple(rows)


# Ledger F14: conforming Voronoi meshes of four closed surfaces against
# VoroCrust-Meshing (b5e0859, R_MAX calibrated to the cell target, LIP_CONST
# 0.25, VC_ANGLE 60, one thread). Per row: the cell size requested from the
# public route, and VoroCrust's measured quality every row must match or beat
# (largest boundary distance to the closed-form surface, that distance over
# the sizing, |relative volume error|, most faces per cell). VoroCrust's
# sphere stopped at about 5e3 cells within its 60 s cap; its 4.8e3 row is the
# reference of the 1e4 row.
_F14_EXACT_VOLUMES = {
    "sphere": 4.0 / 3.0 * math.pi,
    "cube": 1.0,
    "torus": 2.0 * math.pi**2 * 0.742 * 0.371**2,
    "steinmetz": 16.0 / 3.0 * 0.5**3,
}
_F14_CASES: dict[tuple[str, int], tuple[float, tuple[float, float, float, int]]] = {
    ("sphere", 1_000): (0.18, (1.2e-2, 0.071, 0.0175, 28)),
    ("sphere", 10_000): (0.08, (3.4e-3, 0.038, 0.0051, 26)),
    ("cube", 1_000): (0.1, (5.6e-16, 0.0, 0.0, 28)),
    ("cube", 10_000): (0.045, (1.1e-15, 0.0, 0.0, 35)),
    ("torus", 1_000): (0.15, (2.3e-2, 0.164, 0.0317, 41)),
    ("torus", 10_000): (0.06, (3.8e-3, 0.066, 0.0052, 27)),
    ("steinmetz", 1_000): (0.09, (1.2e-2, 0.205, 0.0127, 29)),
    ("steinmetz", 10_000): (0.045, (9.2e-3, 0.223, 0.0030, 30)),
}


def _f14_distance(shape: str, points: np.ndarray, /) -> np.ndarray:
    """Unsigned distance to the closed-form F14 surface (as the ledger's)."""
    if shape == "sphere":
        return np.abs(np.linalg.norm(points, axis=1) - 1.0)
    if shape == "torus":
        ring = np.hypot(points[:, 0], points[:, 1]) - 0.742
        return np.abs(np.hypot(ring, points[:, 2]) - 0.371)
    if shape == "cube":
        outside = np.abs(points) - 0.5
        return np.where(
            np.any(outside > 0.0, axis=1),
            np.linalg.norm(np.maximum(outside, 0.0), axis=1),
            -np.max(outside, axis=1),
        )
    from scipy.spatial import cKDTree

    # Steinmetz solid x^2 + y^2 <= r^2, y^2 + z^2 <= r^2: two cylinder
    # patches and their intersection ellipses x = +-z.
    radius = 0.5
    around_z = np.hypot(points[:, 0], points[:, 1])
    around_x = np.hypot(points[:, 1], points[:, 2])
    foot_a = points.copy()
    foot_a[:, :2] *= (radius / np.maximum(around_z, 1e-300))[:, None]
    foot_b = points.copy()
    foot_b[:, 1:] *= (radius / np.maximum(around_x, 1e-300))[:, None]
    far = np.full((points.shape[0],), np.inf)
    to_a = np.where(
        foot_a[:, 1] ** 2 + foot_a[:, 2] ** 2 <= radius**2, np.abs(around_z - radius), far
    )
    to_b = np.where(
        foot_b[:, 0] ** 2 + foot_b[:, 1] ** 2 <= radius**2, np.abs(around_x - radius), far
    )
    angle = np.linspace(0.0, 2.0 * np.pi, 400_000, endpoint=False)
    x, y = radius * np.cos(angle), radius * np.sin(angle)
    curves = cKDTree(np.concatenate((np.stack((x, y, x), 1), np.stack((x, y, -x), 1))))
    to_curve, _ = curves.query(points)
    return np.minimum(np.minimum(to_a, to_b), to_curve)


class _VoronoiCase(NamedTuple):
    shape: str
    size: float
    reference: tuple[float, float, float, int]
    plan: NativeMeshingPlan
    results: list[CellMeshingCandidate]


def _voronoi_case(inputs: Path, shape: str, target: int, /) -> _VoronoiCase:
    from phydrax.meshing import (
        CellFamilyPolicy,
        CellMeshingTarget,
        MeshingScope,
        PiecewiseLinearComplex,
        SizeControlStrength,
        UniformSizeControl,
        VolumeFillStrategy,
        VolumeMeshingSpec,
    )
    from phydrax.meshing.providers import (
        NativeMeshingOptions,
        NativeMeshingProvider,
        NativePlcSource,
    )

    size, reference = _F14_CASES[(shape, target)]
    surface = np.load(inputs / f"{shape}.npz")
    vertices = np.asarray(surface["xyz"], dtype=np.float64)
    faces = np.asarray(surface["tri"], dtype=np.int64)
    complex_ = PiecewiseLinearComplex(
        vertices,
        faces,
        np.zeros((faces.shape[0],), dtype=np.int64),
        np.asarray([[-1, 0]], dtype=np.int64),
        ("solid",),
        vertex_ids=tuple(f"v{row}" for row in range(vertices.shape[0])),
        vertex_occurrence_paths=((),) * vertices.shape[0],
        facet_ids=("boundary",),
        facet_occurrence_paths=((),),
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),),
    )
    source = NativePlcSource(complex_, f"f14-{shape}", "r1")
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("polyhedron",))),
        MeshingScope.whole(source, 2),
        VolumeFillStrategy.POLYHEDRAL,
        size_controls=(
            UniformSizeControl(
                MeshingScope.whole(source, 3), size, strength=SizeControlStrength.SOFT
            ),
        ),
    )
    plan = NativeMeshingProvider(NativeMeshingOptions("plc_conforming_voronoi")).plan(
        source, specification, coordinate_contract=SpatialCoordinateContract.si()
    )
    return _VoronoiCase(shape, size, reference, plan, [])


def _voronoi_mesh(case: _VoronoiCase, /) -> CellMeshingCandidate:
    # The public route without acceptance; results are kept for the
    # determinism check, not inspected while timed.
    candidate = case.plan.execute(prove=False)
    case.results.append(candidate)
    return candidate


def _voronoi_candidate(result: object, /) -> CellMeshingCandidate:
    from phydrax.meshing import CellMeshingCandidate

    if not isinstance(result, CellMeshingCandidate):
        raise TypeError("A voronoi row returns CellMeshingCandidate.")
    return result


def _voronoi_cells(result: object, /) -> int:
    return int(_voronoi_candidate(result).mesh.connectivity.cell_count)


def _polyhedral_connectivity(mesh: CellMesh, /) -> PolyhedralConnectivity:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("A voronoi row publishes polyhedral connectivity.")
    return connectivity


def _voronoi_quality(
    case: _VoronoiCase, result: object, /
) -> tuple[int, dict[str, float]]:
    """Measure the published mesh independently and compare with VoroCrust."""
    mesh = _voronoi_candidate(result).mesh
    connectivity = _polyhedral_connectivity(mesh)
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
    cell_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    cell_faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    cell_signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.float64)
    cells = cell_offsets.size - 1
    faces = offsets.size - 1
    sizes = np.diff(offsets)
    rows = np.repeat(np.arange(faces, dtype=np.int64), sizes)
    following = np.arange(1, values.size + 1, dtype=np.int64)
    following[offsets[1:] - 1] = offsets[:-1]
    first, second = points[values], points[values[following]]
    # Area vectors and centroids of the face loops.
    area = np.zeros((faces, 3))
    np.add.at(area, rows, 0.5 * np.cross(first, second))
    centroid = np.zeros((faces, 3))
    np.add.at(centroid, rows, first)
    centroid /= sizes[:, None]
    # Volumes by the divergence theorem over each cell's signed faces; the
    # sign convention (outward or inward) follows from the total.
    incidence_cell = np.repeat(np.arange(cells, dtype=np.int64), np.diff(cell_offsets))
    flux = np.einsum("ij,ij->i", centroid, area)[cell_faces] * cell_signs / 3.0
    volume = np.bincount(incidence_cell, flux, minlength=cells)
    outward = 1.0 if np.sum(volume) > 0.0 else -1.0
    volume *= outward
    exact = _F14_EXACT_VOLUMES[case.shape]
    volume_error = float(volume.sum() / exact - 1.0)
    # Boundary faces (one incident cell): vertices, centroids and edge
    # midpoints, as the ledger samples them.
    boundary = np.bincount(cell_faces, minlength=faces) == 1
    boundary_entries = boundary[rows]
    samples = np.concatenate(
        (
            first[boundary_entries],
            centroid[boundary],
            0.5 * (first + second)[boundary_entries],
        )
    )
    distance = _f14_distance(case.shape, samples)
    largest = float(np.max(distance))
    counts = np.diff(cell_offsets)
    # Closure: every directed edge of a cell's outward loops met exactly once
    # by its reverse.
    vertex_count = points.shape[0]
    entry_cells = np.repeat(incidence_cell, sizes[cell_faces])
    entry_rows = np.concatenate(
        [np.arange(offsets[f], offsets[f + 1]) for f in cell_faces]
    )
    entry_signs = np.repeat(cell_signs * outward, sizes[cell_faces]) > 0.0
    tails = np.where(entry_signs, values[entry_rows], values[following[entry_rows]])
    heads = np.where(entry_signs, values[following[entry_rows]], values[entry_rows])
    keys = (entry_cells * vertex_count + tails) * vertex_count + heads
    reverse = (entry_cells * vertex_count + heads) * vertex_count + tails
    unique_keys, key_counts = np.unique(keys, return_counts=True)
    found = np.searchsorted(unique_keys, reverse)
    paired = (found < unique_keys.size) & (
        unique_keys[np.minimum(found, unique_keys.size - 1)] == reverse
    )
    repeated = key_counts[np.searchsorted(unique_keys, keys)] > 1
    open_cells = np.unique(entry_cells[~paired | repeated]).size
    # Convexity, decided exactly: every vertex of a cell on the inner side of
    # each of its face planes, the plane through the exact vertices with their
    # exact Newell normal. Binary64 only screens: the published vertices are
    # the nearest values of the exact ones, so on a face of area over
    # (1e-3 h)^2 the rounded unit normal is off by far under 1e-12 and a
    # rounded distance below -1e-9 h is a certain inside; every other
    # (vertex, face) pair is decided in rationals.
    source_map = _voronoi_candidate(result).geometry.source_map
    if not isinstance(source_map, ExactVoronoiCellGeometrySource):
        raise TypeError("A voronoi row carries exact Voronoi vertices.")
    exact_points = source_map.source_coordinates()
    scale = case.size
    norm = np.linalg.norm(area, axis=1)
    screened = norm > (1e-3 * scale) ** 2
    unit = area / np.maximum(norm, 1e-300)[:, None]
    exact_planes: dict[int, tuple[ExactPoint3, ExactPoint3]] = {}

    def exact_plane(face: int, /) -> tuple[ExactPoint3, ExactPoint3]:
        if face not in exact_planes:
            loop = [exact_points[v] for v in values[offsets[face] : offsets[face + 1]]]
            normal = [Fraction(0), Fraction(0), Fraction(0)]
            for p, q in zip(loop, loop[1:] + loop[:1]):
                normal[0] += p[1] * q[2] - p[2] * q[1]
                normal[1] += p[2] * q[0] - p[0] * q[2]
                normal[2] += p[0] * q[1] - p[1] * q[0]
            exact_planes[face] = ((normal[0], normal[1], normal[2]), loop[0])
        return exact_planes[face]

    nonconvex = np.zeros((cells,), dtype=bool)
    entry_offsets = np.concatenate(([0], np.cumsum(sizes[cell_faces])))
    for cell in range(cells):
        members = np.unique(
            tails[
                entry_offsets[cell_offsets[cell]] : entry_offsets[cell_offsets[cell + 1]]
            ]
        )
        for slot in range(cell_offsets[cell], cell_offsets[cell + 1]):
            face = int(cell_faces[slot])
            orientation = 1 if cell_signs[slot] * outward > 0.0 else -1
            doubtful = members
            if screened[face]:
                rounded = (points[members] - centroid[face]) @ (orientation * unit[face])
                doubtful = members[rounded > -1e-9 * scale]
            if doubtful.size == 0:
                continue
            normal, origin = exact_plane(face)
            if any(
                orientation
                * sum(
                    (
                        normal[axis] * (exact_points[v][axis] - origin[axis])
                        for axis in range(3)
                    ),
                    Fraction(0),
                )
                > 0
                for v in doubtful.tolist()
            ):
                nonconvex[cell] = True
                break
    reference_distance, reference_ratio, reference_volume, reference_faces = (
        case.reference
    )
    digests = set()
    for candidate in case.results:
        published = _polyhedral_connectivity(candidate.mesh)
        digests.add(
            hash(
                (
                    np.asarray(candidate.mesh.coordinates).tobytes(),
                    np.asarray(published.face_vertex_values).tobytes(),
                    np.asarray(published.face_owner).tobytes(),
                    np.asarray(published.face_neighbor).tobytes(),
                )
            )
        )
    case.results.clear()
    measured = {
        "cells": float(cells),
        "boundary_facets": float(int(boundary.sum())),
        "boundary_distance_max": largest,
        "boundary_distance_mean": float(np.mean(distance)),
        "boundary_distance_over_size": largest / scale,
        "volume_relative_error": volume_error,
        "faces_per_cell_min": float(np.min(counts)),
        "faces_per_cell_mean": float(np.mean(counts)),
        "faces_per_cell_max": float(np.max(counts)),
        "closed_cells": float(cells - open_cells),
        "convex_cells": float(cells - int(nonconvex.sum())),
        "deterministic": float(len(digests) == 1),
    }
    missed = sum(
        (
            largest > reference_distance,
            largest / scale > reference_ratio and reference_ratio > 0.0,
            abs(volume_error) > max(reference_volume, 1e-14),
            int(np.max(counts)) > reference_faces,
            open_cells > 0,
            bool(np.any(nonconvex)),
            len(digests) != 1,
        )
    )
    return missed, measured


def _voronoi_rows(inputs: Path, /) -> tuple[_Row, ...]:
    # Row sizes are the cell targets of the F14 table.
    rows: list[_Row] = []
    for shape in ("sphere", "cube", "torus", "steinmetz"):
        for target in (1_000, 10_000):
            case = _voronoi_case(inputs, shape, target)
            rows.append(
                _Row(
                    f"voronoi-{shape}",
                    target,
                    partial(_voronoi_mesh, case),
                    _voronoi_cells,
                    quality=partial(_voronoi_quality, case),
                )
            )
    return tuple(rows)


# Ledger F8 (cad.md): cfMesh cartesianMesh (maxCellSize h, boundaryCellSize
# h / 2, 45-degree feature edges on the quadric), the best measured reference
# per row: hexahedron fraction (cells of six quadrilaterals), sampled Hausdorff
# distance to the closed-form surface, checkMesh non-orthogonality maximum and
# average (acos of the mean cosine) and skewness maximum.
_F8_REFERENCE: dict[tuple[str, int], dict[str, float]] = {
    ("sphere", 0): {
        "hexahedron_fraction": 0.945,
        "hausdorff": 2.15e-4,
        "nonorthogonality_maximum": 18.7,
        "nonorthogonality_average": 5.53,
        "skewness_maximum": 0.486,
    },
    ("sphere", 2): {
        "hexahedron_fraction": 0.959,
        "hausdorff": 2.73e-4,
        "nonorthogonality_maximum": 26.4,
        "nonorthogonality_average": 7.06,
        "skewness_maximum": 0.737,
    },
    ("quadric", 0): {
        "hexahedron_fraction": 0.911,
        "hausdorff": 2.50e-4,
        "nonorthogonality_maximum": 49.4,
        "nonorthogonality_average": 5.37,
        "skewness_maximum": 0.764,
    },
    ("quadric", 2): {
        "hexahedron_fraction": 0.928,
        "hausdorff": 1.97e-4,
        "nonorthogonality_maximum": 50.5,
        "nonorthogonality_average": 6.32,
        "skewness_maximum": 0.807,
    },
}
# Core size h per row (boundary size h / 2): the cfMesh reference's h on each
# row (ledger F8, ~1e5-cell rung). Two layers grow at 1.2 from a first layer
# of 0.325 boundary sizes, the median first-layer thickness of the ledger's
# cfMesh sphere L2 mesh (cfMesh splits its boundary cells; second 0.39).
_F8_CASES: tuple[tuple[str, int, float], ...] = (
    ("sphere", 0, 0.05464),
    ("sphere", 2, 0.06035),
    ("quadric", 0, 0.04597),
    ("quadric", 2, 0.0513),
)
_F8_FIRST_LAYER = 0.325


class _HexDominantCase(NamedTuple):
    model: str
    layers: int
    plan: NativeMeshingPlan


def _hexdom_case(
    inputs: Path, model: str, layers: int, size: float, /
) -> _HexDominantCase:
    from phydrax.meshing import (
        BoundaryLayerControl,
        BoundaryLayerRoute,
        CellFamilyPolicy,
        CellMeshingTarget,
        LayerSchedule,
        MeshingScope,
        PiecewiseLinearComplex,
        SizeControlStrength,
        UniformSizeControl,
        VolumeFillStrategy,
        VolumeMeshingSpec,
    )
    from phydrax.meshing.providers import (
        NativeMeshingOptions,
        NativeMeshingProvider,
        NativePlcSource,
    )

    surface = np.load(inputs / f"{model}-stl.npz")
    vertices = np.asarray(surface["X"], dtype=np.float64)
    faces = np.asarray(surface["F"], dtype=np.int64)
    complex_ = PiecewiseLinearComplex(
        vertices,
        faces,
        np.zeros((faces.shape[0],), dtype=np.int64),
        np.asarray([[-1, 0]], dtype=np.int64),
        ("solid",),
        vertex_ids=tuple(f"v{row}" for row in range(vertices.shape[0])),
        vertex_occurrence_paths=((),) * vertices.shape[0],
        facet_ids=("boundary",),
        facet_occurrence_paths=((),),
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),),
    )
    source = NativePlcSource(complex_, f"f8-{model}", "r1")
    wall = MeshingScope.whole(source, 2)
    solid = MeshingScope.whole(source, 3)
    boundary_size = size / 2.0
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("polyhedron",))),
        wall,
        VolumeFillStrategy.HEX_DOMINANT_POLYHEDRAL_TRANSITION,
        size_controls=(
            UniformSizeControl(solid, size, strength=SizeControlStrength.SOFT),
            UniformSizeControl(wall, boundary_size, strength=SizeControlStrength.SOFT),
        ),
        layer_controls=(
            (
                BoundaryLayerControl(
                    wall,
                    LayerSchedule.geometric(
                        layers, _F8_FIRST_LAYER * boundary_size, growth_rate=1.2
                    ),
                    route=BoundaryLayerRoute.PROVIDER,
                    volume_scope=solid,
                ),
            )
            if layers
            else ()
        ),
    )
    plan = NativeMeshingProvider(NativeMeshingOptions()).plan(
        source, specification, coordinate_contract=SpatialCoordinateContract.si()
    )
    if plan.route != "plc_octree_hex_dominant":
        raise RuntimeError("The F8 rows run the plc_octree_hex_dominant route.")
    return _HexDominantCase(model, layers, plan)


def _hexdom_mesh(case: _HexDominantCase, /) -> CellMeshingCandidate:
    # The public route without acceptance (as the voronoi rows).
    return case.plan.execute(prove=False)


def _hexdom_candidate(result: object, /) -> CellMeshingCandidate:
    from phydrax.meshing import CellMeshingCandidate

    if not isinstance(result, CellMeshingCandidate):
        raise TypeError("A hexdom row returns CellMeshingCandidate.")
    return result


def _hexdom_cells(result: object, /) -> int:
    return int(_hexdom_candidate(result).mesh.connectivity.cell_count)


def _f8_distance(model: str, points: np.ndarray, /) -> np.ndarray:
    # Unsigned distance to the closed-form surface: the unit sphere, or the
    # boundary of {x^2 + y^2 <= 1} n {y^2 + z^2 <= 0.36}: the radial foot on
    # either cylinder when it lies in the other, else the intersection curve
    # (sampled at 2e6 points, spacing below 1e-5).
    if model == "sphere":
        return np.abs(np.linalg.norm(points, axis=1) - 1.0)
    from scipy.spatial import cKDTree

    around_z = np.hypot(points[:, 0], points[:, 1])
    around_x = np.hypot(points[:, 1], points[:, 2])
    far = np.full((points.shape[0],), np.inf)
    foot_z = points[:, 1] / np.maximum(around_z, 1e-300)
    foot_x = points[:, 1] * 0.6 / np.maximum(around_x, 1e-300)
    to_z = np.where(foot_z**2 + points[:, 2] ** 2 <= 0.36, np.abs(around_z - 1.0), far)
    to_x = np.where(points[:, 0] ** 2 + foot_x**2 <= 1.0, np.abs(around_x - 0.6), far)
    y = 0.6 * np.sin(np.linspace(0.0, 2.0 * np.pi, 500_000, endpoint=False))
    curve = []
    for sx in (-1.0, 1.0):
        z = 0.6 * np.cos(np.linspace(0.0, 2.0 * np.pi, 500_000, endpoint=False))
        curve.append(np.stack((sx * np.sqrt(1.0 - y * y), y, z), axis=1))
    # The curve lies on both cylinders, so it is no nearer than either radial
    # gap; only points where that bound is below the foot distance query it
    # (a nearest-point search far from a sampled curve is slow).
    distance = np.minimum(to_z, to_x)
    closer = np.maximum(np.abs(around_z - 1.0), np.abs(around_x - 0.6)) < distance
    distance[closer] = np.minimum(
        distance[closer], cKDTree(np.concatenate(curve)).query(points[closer])[0]
    )
    return distance


def _f8_samples(model: str, count: int, /) -> np.ndarray:
    # Uniform points of the closed-form surface (the cylinder pieces by
    # rejection, in proportion to their bounding areas).
    generator = np.random.default_rng(8)
    if model == "sphere":
        points = generator.normal(size=(count, 3))
        return points / np.linalg.norm(points, axis=1)[:, None]
    pieces = []
    while sum(piece.shape[0] for piece in pieces) < count:
        angle = generator.uniform(0.0, 2.0 * np.pi, count)
        z = generator.uniform(-0.6, 0.6, count)
        on_z = np.stack((np.cos(angle), np.sin(angle), z), axis=1)
        pieces.append(on_z[on_z[:, 1] ** 2 + z**2 <= 0.36])
        angle = generator.uniform(0.0, 2.0 * np.pi, count)
        x = generator.uniform(-1.0, 1.0, count)
        on_x = np.stack((x, 0.6 * np.cos(angle), 0.6 * np.sin(angle)), axis=1)
        pieces.append(on_x[x**2 + on_x[:, 1] ** 2 <= 1.0])
    return np.concatenate(pieces)[:count]


def _hexdom_quality(
    case: _HexDominantCase, result: object, /
) -> tuple[int, dict[str, float]]:
    """OpenFOAM's checkMesh measures and the ledger's distances, from the
    published mesh alone."""
    from scipy.spatial import cKDTree

    mesh = _hexdom_candidate(result).mesh
    connectivity = _polyhedral_connectivity(mesh)
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
    owner = np.asarray(connectivity.face_owner, dtype=np.int64)
    neighbor = np.asarray(connectivity.face_neighbor, dtype=np.int64)
    cell_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    cell_faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    cell_signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.float64)
    faces, cells = offsets.size - 1, cell_offsets.size - 1
    sizes = np.diff(offsets)
    rows = np.repeat(np.arange(faces, dtype=np.int64), sizes)
    following = np.arange(1, values.size + 1, dtype=np.int64)
    following[offsets[1:] - 1] = offsets[:-1]
    first, second = points[values], points[values[following]]
    # OpenFOAM face centres and area vectors: triangles about the vertex mean.
    average = np.zeros((faces, 3))
    np.add.at(average, rows, first)
    average /= sizes[:, None]
    triangle = 0.5 * np.cross(first - average[rows], second - average[rows])
    magnitude = np.linalg.norm(triangle, axis=1)
    area = np.zeros((faces, 3))
    np.add.at(area, rows, triangle)
    centre = np.zeros((faces, 3))
    np.add.at(centre, rows, (first + second + average[rows]) / 3.0 * magnitude[:, None])
    centre /= np.bincount(rows, magnitude, minlength=faces)[:, None]
    # Orient every area vector out of its owner: the cell-face signs, and the
    # overall convention from the enclosed volume (wall faces only).
    incidence = np.repeat(np.arange(cells, dtype=np.int64), np.diff(cell_offsets))
    owner_entry = incidence == owner[cell_faces]
    sign = np.zeros((faces,))
    sign[cell_faces[owner_entry]] = cell_signs[owner_entry]
    inner = neighbor >= 0
    if np.sum((area * sign[:, None] * centre)[~inner]) < 0.0:
        sign = -sign
    area *= sign[:, None]
    counts = np.bincount(owner, minlength=cells) + np.bincount(
        neighbor[inner], minlength=cells
    )
    estimate = (
        np.stack(
            [
                np.bincount(owner, centre[:, k], cells)
                + np.bincount(neighbor[inner], centre[inner, k], cells)
                for k in range(3)
            ],
            axis=1,
        )
        / counts[:, None]
    )
    pyramid_owner = np.sum(area * (centre - estimate[owner]), axis=1) / 3.0
    pyramid_neighbor = (
        -np.sum(area[inner] * (centre[inner] - estimate[neighbor[inner]]), axis=1) / 3.0
    )
    volume = np.bincount(owner, pyramid_owner, cells) + np.bincount(
        neighbor[inner], pyramid_neighbor, cells
    )
    centroid = (
        np.stack(
            [
                np.bincount(
                    owner,
                    pyramid_owner * (0.75 * centre[:, k] + 0.25 * estimate[owner, k]),
                    cells,
                )
                + np.bincount(
                    neighbor[inner],
                    pyramid_neighbor
                    * (0.75 * centre[inner, k] + 0.25 * estimate[neighbor[inner], k]),
                    cells,
                )
                for k in range(3)
            ],
            axis=1,
        )
        / volume[:, None]
    )
    # Bad cells: a nonpositive decomposition tetrahedron (centroid, face
    # vertex mean, edge) seen from either side.
    spoke = np.cross(first - average[rows], second - average[rows]) * sign[rows, None]
    bad = np.zeros((cells,), dtype=bool)
    from_owner = np.einsum("ij,ij->i", spoke, average[rows] - centroid[owner[rows]])
    bad[owner[rows][from_owner <= 0.0]] = True
    inner_rows = inner[rows]
    from_neighbor = -np.einsum(
        "ij,ij->i",
        spoke[inner_rows],
        average[rows][inner_rows] - centroid[neighbor[rows][inner_rows]],
    )
    bad[neighbor[rows][inner_rows][from_neighbor <= 0.0]] = True
    # Non-orthogonality and skewness (primitiveMeshTools).
    interior = np.flatnonzero(inner)
    join = centroid[neighbor[interior]] - centroid[owner[interior]]
    cosine = np.sum(join * area[interior], axis=1) / (
        np.linalg.norm(join, axis=1) * np.linalg.norm(area[interior], axis=1)
    )
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def skewness(selected: np.ndarray, toward: np.ndarray, /) -> np.ndarray:
        offset = centre[selected] - centroid[owner[selected]]
        sv = (
            offset
            - (
                np.sum(area[selected] * offset, axis=1)
                / np.sum(area[selected] * toward, axis=1)
            )[:, None]
            * toward
        )
        unit = sv / np.maximum(np.linalg.norm(sv, axis=1), 1e-300)[:, None]
        reach = 0.2 * np.linalg.norm(toward, axis=1)
        position = np.full((faces,), -1, dtype=np.int64)
        position[selected] = np.arange(selected.size, dtype=np.int64)
        member = position[rows] >= 0
        spread = np.abs(
            np.sum(
                unit[position[rows][member]] * (first[member] - centre[rows][member]),
                axis=1,
            )
        )
        np.maximum.at(reach, position[rows][member], spread)
        return np.linalg.norm(sv, axis=1) / reach

    wall = np.flatnonzero(~inner)
    normal = area[wall] / np.linalg.norm(area[wall], axis=1)[:, None]
    skew = np.concatenate(
        (
            skewness(interior, join),
            skewness(
                wall,
                normal
                * np.sum(normal * (centre[wall] - centroid[owner[wall]]), axis=1)[
                    :, None
                ],
            ),
        )
    )
    # Hexahedra: six faces, all quadrilateral (checkMesh's shape count).
    quads = np.bincount(owner, sizes == 4, cells) + np.bincount(
        neighbor[inner], (sizes == 4)[inner], cells
    )
    hexahedra = int(np.count_nonzero((counts == 6) & (quads == 6)))
    # Distances (the ledger's meas.py): wall faces fanned about their vertex
    # mean; mesh -> surface at 28 lattice points per triangle, surface ->
    # mesh at 2e5 surface points, exact: the 16 nearest triangles bound the
    # distance, then every triangle whose centroid distance minus its largest
    # vertex distance from the centroid is within the bound is checked.
    wall_rows = np.flatnonzero(~inner[rows])
    fan = np.stack(
        (first[wall_rows], second[wall_rows], average[rows[wall_rows]]), axis=1
    )
    lattice = np.asarray(
        [(i / 6, j / 6, 1 - (i + j) / 6) for i in range(7) for j in range(7 - i)],
        dtype=np.float64,
    )
    mesh_to_surface = 0.0
    for start in range(0, fan.shape[0], 20_000):
        block = lattice @ fan[start : start + 20_000]
        mesh_to_surface = max(
            mesh_to_surface,
            float(np.max(_f8_distance(case.model, block.reshape((-1, 3))))),
        )
    samples = _f8_samples(case.model, 200_000)
    centers = fan.mean(axis=1)
    radius = np.max(np.linalg.norm(fan - centers[:, None, :], axis=2), axis=1)
    tree = cKDTree(centers)
    _, nearest = tree.query(samples, k=16)
    best = np.min(
        _point_triangle_distances(
            np.repeat(samples, 16, axis=0), fan[nearest.reshape(-1)]
        ).reshape((-1, 16)),
        axis=1,
    )
    reach = best + float(np.max(radius))
    for start in range(0, samples.shape[0], 20_000):
        lists = tree.query_ball_point(
            samples[start : start + 20_000], reach[start : start + 20_000]
        )
        sample_rows = np.repeat(
            np.arange(start, start + len(lists)), [len(c) for c in lists]
        )
        candidates = np.fromiter(
            (c for found in lists for c in found),
            dtype=np.int64,
            count=sample_rows.size,
        )
        keep = (
            np.linalg.norm(samples[sample_rows] - centers[candidates], axis=1)
            - radius[candidates]
            <= best[sample_rows]
        )
        sample_rows, candidates = sample_rows[keep], candidates[keep]
        np.minimum.at(
            best,
            sample_rows,
            _point_triangle_distances(samples[sample_rows], fan[candidates]),
        )
    surface_to_mesh = float(np.max(best))
    wall_vertices = np.unique(values[np.flatnonzero(~inner[rows])])
    measured = {
        "cells": float(cells),
        "hexahedron_fraction": hexahedra / cells,
        "hausdorff": max(mesh_to_surface, surface_to_mesh),
        "mesh_to_surface": mesh_to_surface,
        "surface_to_mesh": surface_to_mesh,
        "wall_vertex_distance": float(
            np.max(_f8_distance(case.model, points[wall_vertices]))
        ),
        "nonorthogonality_maximum": float(np.max(angle)),
        "nonorthogonality_average": float(np.degrees(np.arccos(np.mean(cosine)))),
        "nonorthogonality_mean": float(np.mean(angle)),
        "skewness_maximum": float(np.max(skew)),
        "bad_cells": float(np.count_nonzero(bad)),
        "minimum_volume": float(np.min(volume)),
    }
    reference = _F8_REFERENCE[(case.model, case.layers)]
    missed = int(np.count_nonzero(bad)) + int(
        measured["hexahedron_fraction"] < reference["hexahedron_fraction"]
    )
    missed += sum(
        int(measured[name] > reference[name])
        for name in (
            "hausdorff",
            "nonorthogonality_maximum",
            "nonorthogonality_average",
            "skewness_maximum",
        )
    )
    return missed, {
        **measured,
        **{f"reference_{name}": value for name, value in reference.items()},
    }


def _hexdom_rows(inputs: Path, /) -> tuple[_Row, ...]:
    # Row size: the ledger's rung (~1e5 cells).
    rows: list[_Row] = []
    for model, layers, size in _F8_CASES:
        case = _hexdom_case(inputs, model, layers, size)
        rows.append(
            _Row(
                f"hexdom-{model}-L{layers}",
                100_000,
                partial(_hexdom_mesh, case),
                _hexdom_cells,
                quality=partial(_hexdom_quality, case),
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
    parser.add_argument(
        "--overset-inputs",
        type=Path,
        default=None,
        help="directory of the ledger's F13 tetrahedral inputs (f13-tet1eNs.npz); "
        "the overset-tet rows run only with it",
    )
    parser.add_argument(
        "--layer-inputs",
        type=Path,
        default=None,
        help="directory of the ledger's F9 surfaces "
        "(gmsh-<model>-h<h>-t0.01-n5-sign-1-surf.stl); the layer rows run only with it",
    )
    parser.add_argument(
        "--voronoi-inputs",
        type=Path,
        default=None,
        help="directory of the ledger's F14 surfaces (<shape>.npz with xyz and tri); "
        "the voronoi rows run only with it",
    )
    parser.add_argument(
        "--hexdom-inputs",
        type=Path,
        default=None,
        help="directory of the ledger's F8 surfaces (<model>-stl.npz with X and F); "
        "the hexdom rows run only with it",
    )
    arguments = parser.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise RuntimeError("Run with OMP_NUM_THREADS=1: rows are single-thread rows.")
    ceilings = json.loads(_CEILINGS.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    failed = False
    names = None if arguments.rows is None else frozenset(arguments.rows)
    for row in _rows(
        tuple(arguments.sizes),
        names,
        arguments.improve_inputs,
        arguments.query_inputs,
        arguments.overset_inputs,
        arguments.layer_inputs,
        arguments.voronoi_inputs,
        arguments.hexdom_inputs,
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
