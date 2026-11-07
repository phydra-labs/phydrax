"""Exactly paired periodic tetrahedral unit cells for cell-problem tests.

Coordinates are dyadic, so every lifted image is the exact lattice translate of
its representative and `PeriodicMeshTopology` validates them bitwise.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import permutations

import numpy as np

from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization._periodic_topology import PeriodicMeshTopology


def _periodic_mesh(
    points: np.ndarray,
    representatives: np.ndarray,
    shifts: np.ndarray,
    tetrahedra: np.ndarray,
    phases: np.ndarray,
    names: Sequence[str],
) -> CellMesh:
    corners = points[tetrahedra]
    flip = np.linalg.det(corners[:, 1:] - corners[:, :1]) < 0.0
    tetrahedra = tetrahedra.copy()
    tetrahedra[flip, 1], tetrahedra[flip, 2] = (
        tetrahedra[flip, 2].copy(),
        tetrahedra[flip, 1].copy(),
    )
    blocks = []
    start = 0
    for index, name in enumerate(names):
        selected = tetrahedra[phases == index]
        if not selected.shape[0]:
            continue
        blocks.append(
            CellBlock(
                name,
                "tetrahedron",
                selected.astype(np.int32),
                global_ids=np.arange(start, start + selected.shape[0], dtype=np.int64),
            )
        )
        start += selected.shape[0]
    vertex_ids = np.arange(points.shape[0], dtype=np.int64)
    lifted = CellMesh(points, tuple(blocks), vertex_global_ids=vertex_ids)
    topology = PeriodicMeshTopology(
        lifted,
        PeriodicCell(np.eye(3, dtype=np.float64)),
        representatives.astype(np.int32),
        shifts.astype(np.int32),
    )
    return CellMesh(
        points, tuple(blocks), vertex_global_ids=vertex_ids, periodic_topology=topology
    )


def kuhn_cell(
    axes: Sequence[Sequence[float]],
    phase: Callable[[np.ndarray], np.ndarray] | None = None,
    names: Sequence[str] = ("solid",),
) -> CellMesh:
    """Unit periodic cell of a tensor grid split into six Kuhn tetrahedra per box.

    ``axes`` are three increasing breakpoint lists from exactly 0 to exactly 1.
    ``phase`` maps box centres ``(boxes, 3)`` to block indices; boxes whose
    phase is out of range are left unmeshed (pores).
    """
    grids = [np.asarray(axis, dtype=np.float64) for axis in axes]
    shape = tuple(axis.size for axis in grids)
    assert all(axis[0] == 0.0 and axis[-1] == 1.0 for axis in grids)
    index = np.stack(np.meshgrid(*(np.arange(n) for n in shape), indexing="ij"), -1)
    index = index.reshape((-1, 3))
    points = np.stack([grids[axis][index[:, axis]] for axis in range(3)], axis=1)
    periods = np.asarray(shape) - 1
    wrapped = index % periods
    shifts = index // periods
    rows = np.arange(points.shape[0]).reshape(shape)
    representatives = rows[wrapped[:, 0], wrapped[:, 1], wrapped[:, 2]]
    tetrahedra, centres = [], []
    for box in np.ndindex(*periods):
        base = np.asarray(box)
        centre = np.asarray(
            [0.5 * (grids[a][base[a]] + grids[a][base[a] + 1]) for a in range(3)]
        )
        for order in permutations(range(3)):
            point = base.copy()
            path = [rows[tuple(point)]]
            for axis in order:
                point = point.copy()
                point[axis] += 1
                path.append(rows[tuple(point)])
            tetrahedra.append(path)
            centres.append(centre)
    tets = np.asarray(tetrahedra, dtype=np.int64)
    phases = (
        np.zeros((tets.shape[0],), dtype=np.int64)
        if phase is None
        else np.asarray(phase(np.asarray(centres)), dtype=np.int64)
    )
    return _periodic_mesh(points, representatives, shifts, tets, phases, names)


def octahedral_cell(
    n: int,
    phase: Callable[[np.ndarray], np.ndarray],
    names: Sequence[str],
) -> CellMesh:
    """Unit periodic cell of ``n**3`` boxes, each split into 24 tetrahedra.

    Every box is coned from its centre over its six faces, each face split into
    four triangles at the face centre. The vertex set (box corners, face and box
    centres on the ``1/n`` lattice) is invariant under every coordinate
    permutation and sign change about the origin, so a phase rule with the same
    invariance yields an exactly cubic periodic composite. ``phase`` maps
    tetrahedron centroids to block indices.
    """
    doubled = 2 * n
    candidates = np.stack(
        np.meshgrid(*(np.arange(doubled + 1),) * 3, indexing="ij"), -1
    ).reshape((-1, 3))
    odd = np.sum(candidates % 2, axis=1)
    keep = odd != 1  # corners (0 odd), face centres (2 odd), box centres (3 odd)
    lattice = candidates[keep]
    lookup = {tuple(row): index for index, row in enumerate(lattice.tolist())}
    points = lattice.astype(np.float64) / doubled
    wrapped = lattice % doubled
    shifts = lattice // doubled
    representatives = np.asarray([lookup[tuple(row)] for row in wrapped.tolist()])
    tetrahedra = []
    for box in np.ndindex(n, n, n):
        origin = 2 * np.asarray(box)
        centre = lookup[tuple(origin + 1)]
        for axis in range(3):
            others = [a for a in range(3) if a != axis]
            for side in (0, 2):
                face = origin + 1
                face[axis] = origin[axis] + side
                ring = []
                for first, second in ((0, 0), (2, 0), (2, 2), (0, 2)):
                    corner = origin.copy()
                    corner[axis] += side
                    corner[others[0]] += first
                    corner[others[1]] += second
                    ring.append(lookup[tuple(corner)])
                for index in range(4):
                    tetrahedra.append(
                        (centre, lookup[tuple(face)], ring[index], ring[(index + 1) % 4])
                    )
    tets = np.asarray(tetrahedra, dtype=np.int64)
    centroids = np.mean(points[tets], axis=1)
    phases = np.asarray(phase(centroids), dtype=np.int64)
    return _periodic_mesh(points, representatives, shifts, tets, phases, names)
