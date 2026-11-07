#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Independent cube-boundary and regular graph proof for implicit contours.

The caller has already replayed the dyadic partition, corner enclosures, and
unique simple roots on the complete crossing edges.  This module checks the
remaining topology, not the polygonizer's interpolation decisions.  Strict
monotonicity on each face excludes closed face components; strict monotonicity
in a cell direction makes the entire zero set a graph.  A single boundary
Jordan curve then bounds its only component, a disk.  The recorded triangles
must independently be an embedded, consistently oriented disk with precisely
those boundary arcs.  Face disks with the same edge-root identities glue by
uniqueness of the face arc; no tetrahedron diagonal is a source edge.
"""

from __future__ import annotations

from typing import Literal, Protocol, TYPE_CHECKING

import numpy as np

from ._approximate_source_coverage import (
    _root_edge_side_interval,
    _root_triangle_projected_area,
    _strict_sign,
    SourceCoverageUnavailable,
)
from .implicit._adaptive_discovery import _CUBE_EDGES, _CUBE_FACES


if TYPE_CHECKING:
    from .implicit._adaptive_discovery import AdaptiveImplicitScaffold
    from .simplicial._mesh import TriangleMesh


class _CubeSurface(Protocol):
    @property
    def scaffold(self) -> AdaptiveImplicitScaffold: ...

    @property
    def mesh(self) -> TriangleMesh | None: ...


class _CubeEnclosure(Protocol):
    @property
    def value_lower(self) -> np.ndarray: ...

    @property
    def value_upper(self) -> np.ndarray: ...

    @property
    def gradient_lower(self) -> np.ndarray: ...

    @property
    def gradient_upper(self) -> np.ndarray: ...


class _CubeBounds(Protocol):
    def boxes(self, lower: np.ndarray, upper: np.ndarray) -> _CubeEnclosure: ...


def _fail(message: str, status: Literal["violated", "unresolved"] = "unresolved") -> None:
    raise SourceCoverageUnavailable(message, status=status)


def _records(rows: np.ndarray) -> np.ndarray:
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    return (
        rows.reshape(len(rows), -1)
        .view(
            np.dtype(
                [(str(i), np.int64) for i in range(rows.reshape(len(rows), -1).shape[1])]
            )
        )
        .reshape(-1)
    )


def prove_implicit_cube_topology(surface: _CubeSurface, bounds: _CubeBounds, /) -> None:
    """Prove source cube graph topology against the retained oriented triangles.

    Working arrays are bounded by one fixed-size batch plus existing source
    arrays.  Missing strict interval premises are refusals, never sampled tests.
    """
    scaffold = surface.scaffold
    boxes = np.asarray(scaffold.cell_boxes, dtype=np.float64)
    cell_ids = np.asarray(scaffold.cell_corner_ids, dtype=np.int64)
    corner_ids = np.asarray(scaffold.corner_ids, dtype=np.int64)
    signs = np.asarray(scaffold.corner_signs, dtype=np.int8)
    roots = np.asarray(scaffold.root_edge_ids, dtype=np.int64)
    triangle_cells = np.asarray(scaffold.triangle_cell_rows, dtype=np.int64)
    triangle_roots = np.asarray(scaffold.triangle_root_edge_rows, dtype=np.int64)
    count = len(boxes)
    if (
        boxes.shape != (count, 2, 3)
        or cell_ids.shape != (count, 8, 3)
        or corner_ids.ndim != 2
        or corner_ids.shape[1:] != (3,)
        or signs.shape != (len(corner_ids),)
        or np.any(np.abs(signs) != 1)
        or roots.ndim != 3
        or roots.shape[1:] != (2, 3)
        or triangle_roots.shape != (len(triangle_cells), 3)
        or surface.mesh is None
        or len(triangle_cells) != len(surface.mesh.faces)
        or np.any(triangle_cells < 0)
        or np.any(triangle_cells >= count)
        or np.any(triangle_roots < 0)
        or np.any(triangle_roots >= len(roots))
    ):
        _fail("Implicit cube ancestry has invalid dimensions or indices.", "violated")
    corner_keys = _records(corner_ids)
    root_keys = _records(roots.reshape(-1, 6)) if len(roots) else None
    order = np.argsort(triangle_cells, kind="stable")
    sorted_cells = triangle_cells[order]
    edges = np.asarray(_CUBE_EDGES, dtype=np.int64)
    edge_by_pair = {frozenset(map(int, edge)): i for i, edge in enumerate(edges)}
    face_edges = [
        tuple(edge_by_pair[frozenset((face[i], face[(i + 1) % 4]))] for i in range(4))
        for face in _CUBE_FACES
    ]
    edge_axes = (
        np.asarray(
            [int(np.flatnonzero(cell_ids[0, a] != cell_ids[0, b])[0]) for a, b in edges],
            dtype=np.int64,
        )
        if count
        else np.empty(0, dtype=np.int64)
    )
    face_axes = (
        [
            next(
                axis
                for axis in range(3)
                if len({int(cell_ids[0, c, axis]) for c in face}) == 1
            )
            for face in _CUBE_FACES
        ]
        if count
        else []
    )

    for start in range(0, count, 256):
        stop = min(start + 256, count)
        ids = cell_ids[start:stop]
        positions = np.searchsorted(corner_keys, _records(ids.reshape(-1, 3)))
        if np.any(positions >= len(corner_keys)):
            _fail("Cube references an absent grid corner.", "violated")
        if np.any(corner_keys[positions] != _records(ids.reshape(-1, 3))):
            _fail("Cube references an absent grid corner.", "violated")
        local_signs = signs[positions].reshape(-1, 8)
        # Coordinates and strict signs were independently replayed by the caller.
        coordinates = np.asarray(scaffold.corner_coordinates)[positions].reshape(-1, 8, 3)
        cell_enclosure = bounds.boxes(boxes[start:stop, 0], boxes[start:stop, 1])
        cell_free = (cell_enclosure.value_lower > 0) | (cell_enclosure.value_upper < 0)
        cell_monotone = (cell_enclosure.gradient_lower > 0) | (
            cell_enclosure.gradient_upper < 0
        )
        crossings = local_signs[:, edges[:, 0]] != local_signs[:, edges[:, 1]]
        edge_proven = crossings | cell_free[:, None] | cell_monotone[:, edge_axes]
        missing_cells, missing_edges = np.nonzero(~edge_proven)
        if len(missing_cells):
            endpoints = coordinates[missing_cells[:, None], edges[missing_edges]]
            enclosure = bounds.boxes(np.min(endpoints, axis=1), np.max(endpoints, axis=1))
            axis = edge_axes[missing_edges]
            row = np.arange(len(axis))
            edge_proven[missing_cells, missing_edges] = (
                (enclosure.value_lower > 0)
                | (enclosure.value_upper < 0)
                | (enclosure.gradient_lower[row, axis] > 0)
                | (enclosure.gradient_upper[row, axis] < 0)
            )
        face_crossing = np.any(crossings[:, np.asarray(face_edges)], axis=2)
        tangential = np.asarray(
            [[a for a in range(3) if a != normal] for normal in face_axes], dtype=np.int64
        )
        face_proven = (cell_free[:, None] & ~face_crossing) | np.any(
            cell_monotone[:, tangential], axis=2
        )
        missing_cells, missing_faces = np.nonzero(~face_proven)
        if len(missing_cells):
            points = coordinates[
                missing_cells[:, None], np.asarray(_CUBE_FACES)[missing_faces]
            ]
            enclosure = bounds.boxes(np.min(points, axis=1), np.max(points, axis=1))
            axes = tangential[missing_faces]
            rows = np.arange(len(axes))[:, None]
            monotone = np.any(
                (enclosure.gradient_lower[rows, axes] > 0)
                | (enclosure.gradient_upper[rows, axes] < 0),
                axis=1,
            )
            free = (enclosure.value_lower > 0) | (enclosure.value_upper < 0)
            face_proven[missing_cells, missing_faces] = monotone | (
                free & ~face_crossing[missing_cells, missing_faces]
            )
        for local in range(stop - start):
            cell = start + local
            crossing = crossings[local]
            edge_roots = {}
            for e, (a, b) in enumerate(edges):
                if crossing[e]:
                    key = np.stack(
                        sorted((tuple(ids[local, a]), tuple(ids[local, b])))
                    ).reshape(1, 6)
                    record = _records(key)
                    pos = (
                        int(np.searchsorted(root_keys, record)[0])
                        if root_keys is not None
                        else 0
                    )
                    if (
                        root_keys is None
                        or pos == len(root_keys)
                        or root_keys[pos] != record[0]
                    ):
                        _fail("Cube crossing edge lacks its retained root.", "violated")
                    edge_roots[e] = pos
                elif not edge_proven[local, e]:
                    _fail("Cube edge may contain an untracked zero or tangency.")
            arcs = set()
            for f, boundary in enumerate(face_edges):
                selected = [edge_roots[e] for e in boundary if crossing[e]]
                if len(selected) not in (0, 2):
                    _fail("Cube face has an ambiguous crossing topology.")
                if not face_proven[local, f]:
                    _fail("Cube face has no certified regular arc or zero exclusion.")
                if selected:
                    arcs.add(tuple(sorted(selected)))
            lo = cell_enclosure.gradient_lower[local]
            hi = cell_enclosure.gradient_upper[local]
            axes = np.flatnonzero((lo > 0) | (hi < 0))
            first = int(np.searchsorted(sorted_cells, cell, side="left"))
            last = int(np.searchsorted(sorted_cells, cell, side="right"))
            rows = order[first:last]
            if not len(axes):
                if (
                    not arcs
                    and not len(rows)
                    and (
                        cell_enclosure.value_lower[local] > 0
                        or cell_enclosure.value_upper[local] < 0
                    )
                ):
                    continue
                _fail("Cube has no certified monotone graph direction.")
            if not arcs:
                if len(rows):
                    _fail("Zero-free cube has recorded triangles.", "violated")
                # A compact graph component has boundary on the cube boundary;
                # absent face zeros therefore exclude interior components.
                continue
            if not len(rows):
                _fail("Cube contour omits its source triangles.", "violated")
            triangles = [
                (
                    int(triangle_roots[r, 0]),
                    int(triangle_roots[r, 1]),
                    int(triangle_roots[r, 2]),
                )
                for r in rows
            ]
            vertices = set(edge_roots.values())
            incidences = {}
            for tri in triangles:
                if len(set(tri)) != 3 or not set(tri) <= vertices:
                    _fail("Cube triangle references incorrect edge roots.", "violated")
                for a, b in zip(tri, (tri[1], tri[2], tri[0]), strict=True):
                    key = tuple(sorted((a, b)))
                    incidences.setdefault(key, []).append((a, b))
            boundary = {edge for edge, uses in incidences.items() if len(uses) == 1}
            if (
                boundary != arcs
                or set(v for tri in triangles for v in tri) != vertices
                or any(
                    len(uses) > 2 or (len(uses) == 2 and uses[0] != uses[1][::-1])
                    for uses in incidences.values()
                )
                or len(vertices) - len(incidences) + len(triangles) != 1
            ):
                _fail(
                    "Cube triangles do not form the retained oriented boundary disk.",
                    "violated",
                )
            adjacency = {v: [] for v in vertices}
            for a, b in boundary:
                adjacency[a].append(b)
                adjacency[b].append(a)
            if any(len(n) != 2 for n in adjacency.values()):
                _fail("Cube face arcs do not form a single regular boundary.")
            visited = set()
            stack = [next(iter(vertices))]
            while stack:
                v = stack.pop()
                if v not in visited:
                    visited.add(v)
                    stack.extend(adjacency[v])
            if visited != vertices:
                _fail("Cube has multiple contour loops or hidden sheets.")
            # Find one chart where every bracket-enclosed triangle is oriented
            # with the source gradient and the projected boundary is simple.
            chart = False
            for axis in axes:
                axis = int(axis)
                expected = (1 if lo[axis] > 0 else -1) * (-1 if axis == 1 else 1)
                if any(
                    _strict_sign(_root_triangle_projected_area(scaffold, tri, axis))
                    != expected
                    for tri in triangles
                ):
                    continue
                boundary_list = list(boundary)
                simple = True
                for i, (a, b) in enumerate(boundary_list):
                    for c, d in boundary_list[i + 1 :]:
                        if len({a, b, c, d}) < 4:
                            continue
                        s1 = _strict_sign(
                            _root_edge_side_interval(scaffold, a, b, c, axis)
                        )
                        s2 = _strict_sign(
                            _root_edge_side_interval(scaffold, a, b, d, axis)
                        )
                        s3 = _strict_sign(
                            _root_edge_side_interval(scaffold, c, d, a, axis)
                        )
                        s4 = _strict_sign(
                            _root_edge_side_interval(scaffold, c, d, b, axis)
                        )
                        if not ((s1 and s1 == s2) or (s3 and s3 == s4)):
                            simple = False
                            break
                    if not simple:
                        break
                if simple:
                    chart = True
                    break
            if not chart:
                _fail(
                    "Cube triangles have unresolved orientation, folds, or boundary intersections."
                )
