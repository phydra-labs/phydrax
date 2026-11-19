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
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array

from benchmarks._runtime import capture_environment, measure_host
from phydrax import _meshcore
from phydrax.geometry._interval_enclosure import prepare_interval_function


_CEILINGS = Path(__file__).with_name("native_meshing_ceilings.json")


@dataclass(frozen=True, slots=True)
class _Row:
    name: str
    size: int
    run: Callable[[], object]
    count: Callable[[object], int]


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
        (frozenset({"refine-improve-sphere", "refine-improve-torus"}), _volume_rows),
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


def _recover(vertices: np.ndarray, faces: np.ndarray, /) -> _meshcore.PlcRecovery3D:
    count = faces.shape[0]
    return _meshcore.recover_plc_3d(
        vertices,
        np.arange(0, 3 * count + 1, 3, dtype=np.int64),
        faces.reshape((-1,)).astype(np.int32),
        np.arange(count, dtype=np.int32),
        np.tile(np.asarray([[-1, 0]], dtype=np.int32), (count, 1)),
        boundary_policy=_meshcore.PlcBoundaryPolicy.FIXED,
    )


def _recovered_cells(result: object, /) -> int:
    if not isinstance(result, _meshcore.PlcRecovery3D):
        raise TypeError("Boundary recovery returns a PlcRecovery3D.")
    return result.tetrahedra.shape[0]


def _recovery_rows() -> tuple[_Row, ...]:
    # Closed surfaces to a constrained tetrahedralization with the boundary
    # fixed; the row size is the triangle count.
    rows: list[_Row] = []
    for level in (4, 5, 6, 7):
        vertices, faces = _icosphere(level)
        rows.append(
            _Row(
                "recover-sphere",
                faces.shape[0],
                lambda vertices=vertices, faces=faces: _recover(vertices, faces),
                _recovered_cells,
            )
        )
    for around in (100, 300):
        vertices, faces = _uv_torus(around)
        rows.append(
            _Row(
                "recover-torus",
                faces.shape[0],
                lambda vertices=vertices, faces=faces: _recover(vertices, faces),
                _recovered_cells,
            )
        )
    return tuple(rows)


def _refine_improve(recovered: _meshcore.PlcRecovery3D, size: float, /) -> int:
    mesh = _meshcore.TetMesh3D(
        recovered.points,
        recovered.tetrahedra,
        recovered.tetrahedron_regions,
        recovered.faces,
        recovered.face_sources,
        recovered.segments,
        recovered.segment_sources,
        boundary_policy=_meshcore.PlcBoundaryPolicy.FIXED,
        protection_radii=recovered.protection_radii,
    )
    sizes = np.full((recovered.points.shape[0],), size, dtype=np.float64)
    refined = mesh.refine(radius_edge_bound=2.0, sizes=sizes)
    improved = mesh.improve(radius_edge_bound=2.0, min_dihedral_degrees=15.0)
    if refined.status != _meshcore.MeshcoreStatus.OK or improved.unmet_cells:
        raise RuntimeError(
            "Refinement and improvement left cells off the quality targets."
        )
    return mesh.arrays().tetrahedra.shape[0]


def _volume_rows() -> tuple[_Row, ...]:
    # State creation, refinement (radius-edge 2) and improvement (radius-edge 2,
    # dihedral 15 degrees) on the recovered surface with its boundary fixed. The
    # circumradius target 1.75 sqrt(3/8) h, h the mean surface edge, matches
    # Gmsh HXT's element counts on the ledger shapes; the row size is the
    # surface triangle count.
    rows: list[_Row] = []
    shapes = (
        ("refine-improve-sphere", _icosphere(5)),
        ("refine-improve-sphere", _icosphere(6)),
        ("refine-improve-torus", _uv_torus(100)),
        ("refine-improve-torus", _uv_torus(300)),
    )
    for name, (vertices, faces) in shapes:
        recovered = _recover(vertices, faces)
        edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        spacing = float(
            np.mean(np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1))
        )
        size = 1.75 * math.sqrt(3.0 / 8.0) * spacing
        rows.append(
            _Row(
                name,
                faces.shape[0],
                lambda recovered=recovered, size=size: _refine_improve(recovered, size),
                _volume_cells,
            )
        )
    return tuple(rows)


def _volume_cells(result: object, /) -> int:
    if not isinstance(result, int):
        raise TypeError("The volume row returns its tetrahedron count.")
    return result


def _implicit_rows() -> tuple[_Row, ...]:
    # Sphere and torus of the external comparison (minor radius 0.371, major
    # 0.742); the row size is the tolerance in thousandths. The lattice spacing
    # follows the chord rule h = 2 sqrt(tolerance * radius).
    radius = 0.371

    def sphere(x: Array) -> Array:
        return jnp.stack((jnp.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) - radius,))

    def torus(x: Array) -> Array:
        ring = jnp.sqrt(x[0] * x[0] + x[1] * x[1]) - 2.0 * radius
        return jnp.stack((jnp.sqrt(ring * ring + x[2] * x[2]) - radius,))

    rows: list[_Row] = []
    for name, field, half_extent in (
        ("implicit-sphere", sphere, (0.4, 0.4, 0.4)),
        ("implicit-torus", torus, (1.15, 1.15, 0.4)),
    ):
        program = prepare_interval_function(
            field, 3, batch_capacity=1, execution="batched"
        ).native_program
        if program is None:
            raise RuntimeError("The implicit rows need the native interval program.")
        upper = np.asarray(half_extent, dtype=np.float64)
        for thousandths in (20, 10):
            spacing = 2.0 * math.sqrt(1e-3 * thousandths * radius)
            rows.append(
                _Row(
                    name,
                    thousandths,
                    lambda program=program, upper=upper, spacing=spacing: (
                        program.stuff_tetrahedra(-upper, upper, spacing)
                    ),
                    _stuffed_cells,
                )
            )
    return tuple(rows)


def _stuffed_cells(result: object, /) -> int:
    if not isinstance(result, _meshcore.StuffedVolume):
        raise TypeError("Isosurface stuffing returns a StuffedVolume.")
    return result.cells.shape[0]


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
    for _ in range(repeats):
        result, seconds = measure_host(row.run)
        samples.append(seconds)
        count = row.count(result)
    return {"elements": count, "seconds": samples, "minimum_seconds": min(samples)}


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
                "passed": ratio <= allowed,
            }
            failed = failed or ratio > allowed
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
