"""MMG3D comparisons for native tetrahedral improvement and metric adaptation.

Runs in a reference environment with the `tetgen` package and an MMG3D
binary (`mmg3d_O3` on PATH or `--mmg3d`), not in the package environment:

  python benchmarks/native_meshing_mmg3d.py prepare DIR
      The rows' inputs: TetGen `pYzQa<h^3/(6 sqrt 2)>` tetrahedra of the icosphere at levels
      3, 4 and 5 (17,274 / 142,911 / 1,163,771 cells), written as
      DIR/improve-sphere-{level}.npz (points, cells) for the rows and
      DIR/improve-sphere-{level}.mesh (Medit) for MMG3D.
  python benchmarks/native_meshing_mmg3d.py time DIR LEVEL
      One MMG3D `-optim -nosurf` run on that input: its reported library time
      (MMG3DLIB: ELAPSED TIME, without file input and output), the process
      time, and the output's measured quality, as one JSON line.

Interleave `time` with the row (reference, ours, repeated; minimum each).

For rapid metric iteration, run in the Phydrax benchmark environment:

  OMP_NUM_THREADS=1 python -m benchmarks.native_meshing_mmg3d metric DIR --field iso

DIR contains the small feature.npz source and feature.FIELD.mbin target. Both
tools read the same mesh and sampled target. This is a single paired
diagnostic, not a pass on the archived size ladder.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks._meshing_surfaces import icosphere


LEVELS = (3, 4, 5)


def _boundary(cells: np.ndarray, /) -> np.ndarray:
    # Faces used once, oriented outward as in their cell.
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


def _write_medit(
    path: Path,
    points: np.ndarray,
    cells: np.ndarray,
    /,
    *,
    boundary: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    if boundary is None:
        triangles = _boundary(cells)
        refs = np.ones(len(triangles), dtype=np.int32)
    else:
        triangles, refs = boundary
    with path.open("w", encoding="ascii") as stream:
        stream.write(f"MeshVersionFormatted 2\nDimension 3\nVertices\n{len(points)}\n")
        np.savetxt(
            stream, np.c_[points, np.zeros(len(points))], fmt="%.17g %.17g %.17g %d"
        )
        stream.write(f"Triangles\n{len(triangles)}\n")
        np.savetxt(stream, np.c_[triangles + 1, refs], fmt="%d")
        stream.write(f"Tetrahedra\n{len(cells)}\n")
        np.savetxt(stream, np.c_[cells + 1, np.zeros(len(cells))], fmt="%d")
        stream.write("End\n")


def _read_medit(path: Path, /) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="ascii").split("\n")
    points = cells = None
    index = 0
    while index < len(lines):
        key = lines[index].strip()
        if key in ("Vertices", "Tetrahedra"):
            count = int(lines[index + 1])
            rows = np.loadtxt(lines[index + 2 : index + 2 + count], ndmin=2)
            if key == "Vertices":
                points = rows[:, :3]
            else:
                cells = rows[:, :4].astype(np.int64) - 1
            index += 2 + count
        else:
            index += 1
    if points is None or cells is None:
        raise RuntimeError(f"{path} has no vertices or tetrahedra.")
    return points, cells


def _quality(points: np.ndarray, cells: np.ndarray, /) -> dict[str, float]:
    corners = points[cells]
    angles = []
    for a, b, c, d in (
        (0, 1, 2, 3),
        (0, 2, 1, 3),
        (0, 3, 1, 2),
        (1, 2, 0, 3),
        (1, 3, 0, 2),
        (2, 3, 0, 1),
    ):
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
    dihedral = np.stack(angles, axis=1)
    edges = corners[:, 1:] - corners[:, :1]
    center = np.linalg.solve(edges, 0.5 * np.sum(edges * edges, axis=2)[..., None])[
        ..., 0
    ]
    shortest = np.min(
        np.stack(
            [np.linalg.norm(corners[:, i] - corners[:, j], axis=1)
             for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))],
            axis=1,
        ),
        axis=1,
    )  # fmt: skip
    volume = np.einsum("ij,ij->i", edges[:, 0], np.cross(edges[:, 1], edges[:, 2]))
    return {
        "cells": float(len(cells)),
        "minimum_dihedral_degrees": float(dihedral.min()),
        "maximum_dihedral_degrees": float(dihedral.max()),
        "maximum_radius_edge": float(np.max(np.linalg.norm(center, axis=1) / shortest)),
        "inverted": float(np.sum(volume <= 0.0)),
    }


def prepare(directory: Path, /) -> None:
    import tetgen

    directory.mkdir(parents=True, exist_ok=True)
    for level in LEVELS:
        vertices, faces = icosphere(level)
        # The ledger's bound: the regular tetrahedron of edge h, h the mean of
        # the triangles' first edges.
        h = float(
            np.mean(np.linalg.norm(vertices[faces[:, 0]] - vertices[faces[:, 1]], axis=1))
        )
        points, cells = tetgen.TetGen(vertices, faces.astype(np.int32)).tetrahedralize(
            switches=f"pYzQa{h**3 / (6.0 * 2.0**0.5):.12f}"
        )[:2]
        cells = np.asarray(cells, dtype=np.int64)
        np.savez(directory / f"improve-sphere-{level}.npz", points=points, cells=cells)
        _write_medit(directory / f"improve-sphere-{level}.mesh", points, cells)
        print(
            json.dumps({"level": level, "cells": len(cells), **_quality(points, cells)})
        )


def measure(directory: Path, level: int, mmg3d: str, /) -> None:
    source = directory / f"improve-sphere-{level}.mesh"
    with tempfile.TemporaryDirectory() as scratch:
        target = Path(scratch) / "out.mesh"
        start = time.perf_counter()
        completed = subprocess.run(
            [mmg3d, "-optim", "-nosurf", "-v", "1", str(source), str(target)],
            check=True,
            capture_output=True,
            text=True,
        )
        process = time.perf_counter() - start
        library = re.search(r"MMG3DLIB: ELAPSED TIME\s+([0-9.]+)s", completed.stdout)
        if library is None:
            raise RuntimeError("MMG3D did not report its library time.")
        points, cells = _read_medit(target)
    print(
        json.dumps(
            {
                "tool": "mmg3d",
                "level": level,
                "seconds": float(library.group(1)),
                "process_seconds": process,
                **_quality(points, cells),
            }
        ),
        flush=True,
    )


def _write_metric(path: Path, tensors: np.ndarray, anisotropic: bool, /) -> None:
    if anisotropic:
        # Medit symmetric tensor order: xx xy yy xz yz zz.
        values = tensors[:, (0, 0, 1, 0, 1, 2), (0, 1, 1, 2, 2, 2)]
    else:
        values = 1.0 / np.sqrt(tensors[:, 0, 0])
    with path.open("w", encoding="ascii") as stream:
        stream.write(
            f"MeshVersionFormatted 2\nDimension 3\nSolAtVertices\n{len(tensors)}\n"
            f"1 {3 if anisotropic else 1}\n"
        )
        np.savetxt(stream, values, fmt="%.17g")
        stream.write("End\n")


def _measure_metric(directory: Path, field: str, mmg3d: str, /) -> None:
    from benchmarks.native_meshing import (
        _F6_FIELDS,
        _f6_quality_values,
        _f6_rows,
        _f6_source,
        _f6_target,
        _F6QualityReference,
        _measure,
    )

    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise RuntimeError("Run the paired metric comparison with OMP_NUM_THREADS=1.")
    if field not in _F6_FIELDS:
        raise ValueError(f"--field must be one of {', '.join(_F6_FIELDS)}.")
    source = directory / "feature.npz"
    with np.load(source, allow_pickle=False) as bank:
        count = bank["el"].shape[0]
        boundary = bank["bnd"], bank["bnd_ref"]
    if not 1 <= count <= 512:
        raise ValueError(
            "The paired metric diagnostic requires at most 512 source tetrahedra."
        )
    points, cells, _ = _f6_source(source, count)
    tensors, _ = _f6_target(
        directory / f"feature.{field}.mbin", len(points), 6 if field == "aniso" else 1
    )
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        mesh, sol, out = root / "source.mesh", root / "target.sol", root / "output.mesh"
        _write_medit(mesh, points, cells, boundary=boundary)
        _write_metric(sol, tensors, field == "aniso")
        completed = subprocess.run(
            [
                mmg3d,
                "-in",
                str(mesh),
                "-sol",
                str(sol),
                "-out",
                str(out),
                "-m",
                "3800",
                "-v",
                "1",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        library = re.search(r"MMG3DLIB: ELAPSED TIME\s+([0-9.]+)s", completed.stdout)
        if library is None:
            raise RuntimeError("MMG3D did not report its library time.")
        output_points, output_cells = _read_medit(out)
    coarsen_sizes = 1.0 / np.sqrt(tensors[:, 0, 0]) if field == "coarsen" else None
    quality = _f6_quality_values(
        field, points, coarsen_sizes, output_points, output_cells
    )
    (row,) = _f6_rows(
        (count,),
        directory,
        frozenset({f"metric-3d-{field}"}),
        sources={count: ("feature", count)},
        references={
            (field, count): _F6QualityReference(
                quality["share_0_7_1_4"], quality["quality_mean"], quality["quality_min"]
            )
        },
    )
    print(
        json.dumps(
            {
                "field": field,
                "source_tetrahedra": count,
                "input_sha256": dict(row.input_sha256),
                "mmg3d": {
                    "elements": len(output_cells),
                    "seconds": float(library.group(1)),
                    "quality": quality,
                },
                "phydrax": _measure(row, 1),
            }
        ),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "time", "metric"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("level", type=int, nargs="?", choices=LEVELS)
    parser.add_argument("--mmg3d", default="mmg3d_O3")
    parser.add_argument(
        "--field", default="coarsen", help="metric target (iso, aniso, coarsen)"
    )
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        prepare(arguments.directory)
    elif arguments.command == "metric":
        _measure_metric(arguments.directory, arguments.field, arguments.mmg3d)
    else:
        if arguments.level is None:
            parser.error("time needs LEVEL")
        measure(arguments.directory, arguments.level, arguments.mmg3d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
