"""Reference driver for the `improve-sphere` rows of `benchmarks/native_meshing.py`.

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
"""

from __future__ import annotations

import argparse
import json
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


def _write_medit(path: Path, points: np.ndarray, cells: np.ndarray, /) -> None:
    triangles = _boundary(cells)
    with path.open("w", encoding="ascii") as stream:
        stream.write(f"MeshVersionFormatted 2\nDimension 3\nVertices\n{len(points)}\n")
        np.savetxt(
            stream, np.c_[points, np.zeros(len(points))], fmt="%.17g %.17g %.17g %d"
        )
        stream.write(f"Triangles\n{len(triangles)}\n")
        np.savetxt(stream, np.c_[triangles + 1, np.ones(len(triangles))], fmt="%d")
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "time"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("level", type=int, nargs="?", choices=LEVELS)
    parser.add_argument("--mmg3d", default="mmg3d_O3")
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        prepare(arguments.directory)
    else:
        if arguments.level is None:
            parser.error("time needs LEVEL")
        measure(arguments.directory, arguments.level, arguments.mmg3d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
