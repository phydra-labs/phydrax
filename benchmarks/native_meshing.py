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


@dataclass(frozen=True, slots=True)
class _Row:
    name: str
    size: int
    run: Callable[[], object]
    count: Callable[[object], int]
    # Cells left off the row's quality targets; a row with any does not pass.
    unmet: Callable[[object], int] = _none_unmet
    domain: str | None = None


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


def _rows(sizes: tuple[int, ...], names: frozenset[str] | None, /) -> tuple[_Row, ...]:
    # Families prepare their inputs only when one of their rows is selected.
    families: tuple[tuple[frozenset[str], Callable[[], tuple[_Row, ...]]], ...] = (
        (frozenset({"delaunay-2d", "delaunay-3d"}), lambda: _delaunay_rows(sizes)),
        (frozenset({"cdt-2d", "ruppert-2d"}), lambda: _planar_rows(sizes)),
        (frozenset({"implicit-sphere", "implicit-torus"}), _implicit_rows),
        (frozenset({"recover-sphere", "recover-torus"}), _recovery_rows),
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
    count = 0
    result: object = None
    unmet = 0
    for _ in range(repeats):
        result, seconds = measure_host(row.run)
        samples.append(seconds)
        count = row.count(result)
        unmet = row.unmet(result)
    if isinstance(result, _meshcore.ConstrainedDelaunay2D):
        return {
            "domain": row.domain,
            "elements": count,
            "unmet": unmet,
            "minimum_angle_degrees": _planar_minimum_angle(result),
            "seconds": samples,
            "minimum_seconds": min(samples),
        }
    return {
        "elements": count,
        "unmet": unmet,
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
            record |= {
                "reference_tool": tool,
                "reference_seconds": seconds,
                "ratio": ratio,
                "allowed_ratio": allowed,
                "passed": ratio <= allowed and measured["unmet"] == 0,
            }
            failed = failed or ratio > allowed or measured["unmet"] > 0
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
