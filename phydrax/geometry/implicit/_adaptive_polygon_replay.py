"""Cold replay of the retained marching-cubes contour and its ancestry."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from typing import Protocol, TYPE_CHECKING

import numpy as np

from ._adaptive_packed_polygon import (
    _CHUNK,
    _Chunks,
    _records,
    boundary_face_edges,
    clipped_face_keys,
    ClippedBoundary,
)


if TYPE_CHECKING:
    from jax import Array

    from .._contracts import CompiledGeometry
    from ..simplicial._mesh import TriangleMesh
    from ._adaptive_discovery import _FieldEnclosures, _GridEdge, AdaptiveImplicitScaffold
    from ._policy import AdaptiveImplicitSurfacePolicy


class _ReplaySurface(Protocol):
    @property
    def scaffold(self) -> AdaptiveImplicitScaffold | None: ...
    @property
    def mesh(self) -> TriangleMesh | None: ...
    @property
    def policy(self) -> AdaptiveImplicitSurfacePolicy: ...
    @property
    def feature_edges(self) -> np.ndarray: ...
    @property
    def boundary_witness_boxes(self) -> np.ndarray: ...
    @property
    def boundary_witnesses(self) -> np.ndarray: ...
    @property
    def boundary_errors(self) -> np.ndarray: ...


def _same_float_bits(first: np.ndarray | Array, second: np.ndarray | Array) -> bool:
    first = np.asarray(first)
    second = np.asarray(second)
    return (
        first.dtype == np.float64
        and second.dtype == np.float64
        and np.array_equal(first.view(np.uint64), second.view(np.uint64))
    )


def _clipped_roots_match(
    scaffold: AdaptiveImplicitScaffold, clipped: ClippedBoundary, /
) -> bool:
    """Whether every retained domain-face root is the crease bank's record."""
    edges = np.ascontiguousarray(scaffold.root_edge_ids, dtype=np.int64)
    rows = np.flatnonzero(boundary_face_edges(edges, 1 << scaffold.target_level))
    if not rows.size:
        return True
    bank = _records(clipped.root_edge_ids.reshape(-1, 6))
    wanted = _records(np.ascontiguousarray(edges[rows].reshape(-1, 6)))
    found = np.searchsorted(bank, wanted)
    if np.any(found >= len(bank)) or not np.array_equal(bank[found], wanted):
        return False
    return (
        _same_float_bits(
            np.asarray(scaffold.root_points)[rows],
            np.asarray(clipped.root_points, dtype=np.float64)[found],
        )
        and _same_float_bits(
            np.asarray(scaffold.root_errors)[rows],
            np.asarray(clipped.root_errors, dtype=np.float64)[found],
        )
        and _same_float_bits(
            np.asarray(scaffold.root_brackets)[rows],
            np.asarray(clipped.root_brackets, dtype=np.float64)[found],
        )
        and np.array_equal(
            np.asarray(scaffold.root_issues)[rows],
            np.asarray(clipped.root_issues)[found],
        )
    )


def replay_polygon(
    surface: _ReplaySurface,
    bounds: _FieldEnclosures,
    /,
    *,
    clipped: ClippedBoundary | None = None,
) -> bool:
    """Compare generated contours and vertex order without a second mesh.

    The independent lattice verifier must first validate corner coordinates,
    signs, root enclosures and cell metadata.  This replay additionally checks
    every required edge, cell refusal, triangle and mesh vertex in owner order.
    ``clipped`` replays the declared open-boundary mode: the domain-face rule
    is the crease certificate's, and every retained domain-face root must be
    the crease root bank's record bit for bit.
    """
    from ._adaptive_discovery import (
        _Cell,
        _CUBE_EDGES,
        _CUBE_FACES,
        _domain_boundary_faces,
        _edge_key,
        _faces_exclude_zero,
        _prove_constant_signs,
        _tangency_faces,
        _triangulate_cell_band,
        AdaptiveImplicitBoxIssue as Issue,
    )

    s = surface.scaffold
    if s is None:
        # A discovery that retained no polygonization has nothing to replay.
        return False
    unresolved_faces = None if clipped is None else clipped.unresolved_face_keys()
    if clipped is not None and not _clipped_roots_match(s, clipped):
        return False
    count = len(s.cell_indices)
    corner_ids = np.asarray(s.corner_ids)
    corner_records = _records(corner_ids)

    def cell_corners(rows: slice | np.ndarray) -> np.ndarray:
        ids = np.ascontiguousarray(s.cell_corner_ids[rows].reshape(-1, 3))
        return np.searchsorted(corner_records, _records(ids)).reshape(-1, 8)

    signs = np.asarray(s.corner_signs)
    corner_values = np.asarray(s.corner_sampled_values)
    points = np.asarray(s.corner_coordinates)
    root_ids = np.asarray(s.root_edge_ids)
    root_records = _records(root_ids.reshape(-1, 6))
    root_used = np.zeros(len(root_ids), dtype=np.bool_)
    vertex_by_root = np.full(len(root_ids), -1, dtype=np.int64)
    root_by_vertex = np.empty(len(root_ids), dtype=np.int64)
    witness_root = np.full(count, -1, dtype=np.int64)

    # Reproduce the first-occurrence summation order for shared ambiguous faces.
    ambiguous = _Chunks((6,), np.int64)
    for start in range(0, count, _CHUNK):
        stop = min(start + _CHUNK, count)
        cr = cell_corners(slice(start, stop))
        sg = signs[cr]
        valid = np.all(sg != 0, axis=1)
        for face_id, face in enumerate(_CUBE_FACES):
            fs = sg[:, face]
            for local in np.flatnonzero(
                valid & np.all(fs != np.roll(fs, -1, axis=1), axis=1)
            ):
                ambiguous.append(
                    (*np.sort(cr[local, list(face)]), start + int(local), face_id)
                )
    amb = ambiguous.take_array()
    if len(amb):
        _records(amb).sort(kind="quicksort")
        keep = np.ones(len(amb), dtype=np.bool_)
        for start in range(1, len(amb), _CHUNK):
            stop = min(start + _CHUNK, len(amb))
            keep[start:stop] = np.any(
                amb[start:stop, :4] != amb[start - 1 : stop - 1, :4], axis=1
            )
        amb = amb[keep]
    amb_keys = np.ascontiguousarray(amb[:, :4])
    amb_records = _records(amb_keys)
    amb_signs = np.empty(len(amb), dtype=np.int8)
    for start in range(0, len(amb), _CHUNK):
        stop = min(start + _CHUNK, len(amb))
        centers = np.empty((stop - start, 3), dtype=np.float64)
        amb_corners = cell_corners(amb[start:stop, 4])
        for j, entry in enumerate(amb[start:stop]):
            centers[j] = np.mean(
                points[amb_corners[j, list(_CUBE_FACES[entry[5]])]], axis=0
            )
        lo, hi = bounds.point_values(centers)
        amb_signs[start:stop] = np.where(hi < 0, -1, np.where(lo > 0, 1, 0))
    del amb

    from ._adaptive_native_polygon import contours

    vertex_count = 0
    triangle_count = 0
    local_roots = {}

    class Vertices:
        def __len__(self) -> int:
            return vertex_count

        def __getitem__(self, vertex: int) -> np.ndarray:
            if root_by_vertex is None:
                raise RuntimeError("Polygon replay vertex indices have been released.")
            return s.root_points[root_by_vertex[vertex]]

        def append(self, point: np.ndarray) -> None:
            raise ValueError("unassigned replay vertex")

    class VertexKeys:
        def get(self, key: _GridEdge) -> int | None:
            if vertex_by_root is None:
                raise RuntimeError("Polygon replay root indices have been released.")
            value = int(vertex_by_root[local_roots[key]])
            return None if value < 0 else value

        def __setitem__(self, key: _GridEdge, value: int) -> None:
            raise ValueError("unassigned replay root")

    class EdgeRows:
        def append(self, key: _GridEdge) -> None:
            raise ValueError("unassigned replay root")

        def __getitem__(self, vertex: int) -> _GridEdge:
            if root_by_vertex is None:
                raise RuntimeError("Polygon replay vertex indices have been released.")
            endpoints = root_ids[root_by_vertex[vertex]]
            return (
                (int(endpoints[0, 0]), int(endpoints[0, 1]), int(endpoints[0, 2])),
                (int(endpoints[1, 0]), int(endpoints[1, 1]), int(endpoints[1, 2])),
            )

    class IgnoreErrors:
        def append(self, error: float) -> None:
            pass

    vertices, keys, edges = Vertices(), VertexKeys(), EdgeRows()
    face_edge_order = np.asarray((0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 10, 11))
    try:
        for start in range(0, count, _CHUNK):
            stop = min(start + _CHUNK, count)
            cr = cell_corners(slice(start, stop))
            cell_signs = signs[cr]
            singular = np.any(cell_signs == 0, axis=1)
            crossing_signs = np.any(cell_signs < 0, axis=1) & np.any(
                cell_signs > 0, axis=1
            )
            no_crossing = ~singular & ~crossing_signs
            gradient = np.any(
                (s.cell_gradient_lower[start:stop] > 0)
                | (s.cell_gradient_upper[start:stop] < 0),
                axis=1,
            )
            preliminary = np.zeros(stop - start, dtype=np.int64)
            preliminary[singular] = int(Issue.SINGULAR)
            preliminary[no_crossing & gradient] = int(Issue.EDGE)
            preliminary[no_crossing & ~gradient] = int(Issue.SINGULAR)
            empty = np.zeros(stop - start, dtype=np.bool_)
            candidates = np.flatnonzero(no_crossing)
            if len(candidates):
                proved = _prove_constant_signs(
                    bounds,
                    s.cell_boxes[start + candidates, 0],
                    s.cell_boxes[start + candidates, 1],
                    cell_signs[candidates, 0],
                    s.cell_gradient_lower[start + candidates],
                    s.cell_gradient_upper[start + candidates],
                )
                empty[candidates[proved]] = True
                preliminary[candidates[proved]] = 0
                edge = candidates[~proved & gradient[candidates]]
                if len(edge):
                    tangent = np.any(
                        _tangency_faces(
                            bounds,
                            s.cell_boxes[start + edge, 0],
                            s.cell_boxes[start + edge, 1],
                            cell_signs[edge, 0],
                            s.cell_gradient_lower[start + edge],
                            s.cell_gradient_upper[start + edge],
                        ),
                        axis=1,
                    )
                    preliminary[edge[tangent]] = int(Issue.TANGENCY)
            boundary_mask = np.any(
                (s.cell_boxes[start:stop, 0] == s.domain[0])
                | (s.cell_boxes[start:stop, 1] == s.domain[1]),
                axis=1,
            )
            for local in np.flatnonzero((preliminary == 0) & ~empty & boundary_mask):
                row = start + int(local)
                if unresolved_faces is not None:
                    if any(
                        key in unresolved_faces
                        for key in clipped_face_keys(
                            s.cell_indices[row, 1:], int(s.cell_indices[row, 0])
                        )
                    ):
                        preliminary[local] = int(Issue.DOMAIN_BOUNDARY)
                    continue
                cell = _Cell(
                    int(s.cell_indices[row, 0]),
                    (
                        int(s.cell_indices[row, 1]),
                        int(s.cell_indices[row, 2]),
                        int(s.cell_indices[row, 3]),
                    ),
                    s.cell_boxes[row, 0],
                    s.cell_boxes[row, 1],
                    s.cell_value_lower[row],
                    s.cell_value_upper[row],
                    s.cell_gradient_lower[row],
                    s.cell_gradient_upper[row],
                    0,
                )
                boundary = _domain_boundary_faces(cell, s.domain)
                if boundary and not _faces_exclude_zero(bounds, boundary):
                    preliminary[local] = int(Issue.DOMAIN_BOUNDARY)
            rejected = preliminary != 0
            if not np.array_equal(
                np.asarray(s.cell_issues[start:stop])[rejected], preliminary[rejected]
            ) or np.any(np.asarray(s.cell_issues[start:stop])[empty]):
                return False
            selected = np.flatnonzero(~rejected & ~empty)
            if not len(selected):
                continue
            cell_rows = start + selected
            crs = cr[selected]
            active_signs = cell_signs[selected]
            crossing = (
                active_signs[:, _CUBE_EDGES[:, 0]] != active_signs[:, _CUBE_EDGES[:, 1]]
            )
            pair_rows = np.stack(
                (crs[:, _CUBE_EDGES[:, 0]], crs[:, _CUBE_EDGES[:, 1]]), axis=-1
            )[crossing]
            pair_rows.sort(axis=1)
            ids = np.ascontiguousarray(corner_ids[pair_rows].reshape(-1, 6))
            found = np.searchsorted(root_records, _records(ids))
            if np.any(found >= len(root_ids)) or not np.array_equal(
                root_ids[found].reshape(-1, 6), ids
            ):
                return False
            edge_roots = np.full(crossing.shape, -1, dtype=np.int64)
            edge_roots[crossing] = found
            root_used[found] = True
            witness_root[cell_rows] = edge_roots[
                np.arange(len(cell_rows)), np.argmax(crossing, axis=1)
            ]
            occurrences = edge_roots[:, face_edge_order].reshape(-1)
            occurrences = occurrences[occurrences >= 0]
            unique, first = np.unique(occurrences, return_index=True)
            fresh = vertex_by_root[unique] < 0
            new_roots = unique[fresh][np.argsort(first[fresh])]
            if surface.mesh is not None:
                if vertex_count + len(new_roots) > surface.mesh.vertices.shape[0]:
                    return False
                for offset in range(0, len(new_roots), _CHUNK):
                    last = min(offset + _CHUNK, len(new_roots))
                    if not _same_float_bits(
                        surface.mesh.vertices[
                            vertex_count + offset : vertex_count + last
                        ],
                        s.root_points[new_roots[offset:last]],
                    ):
                        return False
            vertex_by_root[new_roots] = np.arange(
                vertex_count, vertex_count + len(new_roots)
            )
            root_by_vertex[vertex_count : vertex_count + len(new_roots)] = new_roots
            vertex_count += len(new_roots)
            fast, tri_counts, fast_triangles, issues = contours(
                active_signs,
                corner_values[crs],
                s.cell_boxes[cell_rows],
                edge_roots,
                vertex_by_root,
                s.root_points,
                s.root_issues,
                s.root_errors,
            )
            if not np.array_equal(s.cell_issues[cell_rows[fast]], issues[fast]):
                return False
            tri_buffer = np.empty((len(cell_rows), 10, 3), dtype=np.int64)
            tri_buffer[fast] = fast_triangles[fast]
            for local in np.flatnonzero(~fast):
                row = int(cell_rows[local])
                corners = tuple(
                    (int(point[0]), int(point[1]), int(point[2]))
                    for point in s.cell_corner_ids[row]
                )
                sign_map = {corners[i]: int(active_signs[local, i]) for i in range(8)}
                value_map = {
                    corners[i]: float(corner_values[crs[local, i]]) for i in range(8)
                }
                local_roots = {}
                root_cache = {}
                for edge_no, (first, second) in enumerate(_CUBE_EDGES):
                    root = int(edge_roots[local, edge_no])
                    if root < 0:
                        continue
                    key = _edge_key(corners[first], corners[second])
                    local_roots[key] = root
                    root_cache[key] = (
                        s.root_points[root],
                        float(s.root_errors[root]),
                        int(s.root_issues[root]),
                        s.root_brackets[root, 0],
                        s.root_brackets[root, 1],
                        0,
                        0,
                    )
                if not local_roots:
                    if int(s.cell_issues[row]) != int(Issue.EDGE):
                        return False
                    continue
                face_signs = {}
                for face in _CUBE_FACES:
                    fs = active_signs[local, list(face)]
                    if np.all(fs != np.roll(fs, -1)):
                        face_key = np.sort(crs[local, list(face)]).reshape(1, 4)
                        pos = int(np.searchsorted(amb_records, _records(face_key))[0])
                        if pos >= len(amb_keys) or not np.array_equal(
                            amb_keys[pos], face_key[0]
                        ):
                            return False
                        face_signs[tuple(sorted(corners[i] for i in face))] = int(
                            amb_signs[pos]
                        )
                cell = _Cell(
                    int(s.cell_indices[row, 0]),
                    (
                        int(s.cell_indices[row, 1]),
                        int(s.cell_indices[row, 2]),
                        int(s.cell_indices[row, 3]),
                    ),
                    s.cell_boxes[row, 0],
                    s.cell_boxes[row, 1],
                    s.cell_value_lower[row],
                    s.cell_value_upper[row],
                    s.cell_gradient_lower[row],
                    s.cell_gradient_upper[row],
                    0,
                )
                local_faces, local_cells = [], []
                issue, _, _ = _triangulate_cell_band(
                    cell,
                    corners,
                    sign_map,
                    value_map,
                    face_signs,
                    root_cache,
                    vertices,
                    IgnoreErrors(),
                    keys,
                    local_faces,
                    edges,
                    local_cells,
                    row,
                    surface.policy,
                )
                if int(s.cell_issues[row]) != issue:
                    return False
                tri_counts[local] = len(local_faces)
                if local_faces:
                    tri_buffer[local, : len(local_faces)] = local_faces
            tri_mask = np.arange(10)[None, :] < tri_counts[:, None]
            expected_faces = tri_buffer[tri_mask]
            expected_cells = np.repeat(cell_rows, tri_counts)
            end = triangle_count + len(expected_faces)
            if end > len(s.triangle_cell_rows):
                return False
            if not np.array_equal(
                s.triangle_cell_rows[triangle_count:end], expected_cells
            ) or not np.array_equal(
                s.triangle_root_edge_rows[triangle_count:end],
                root_by_vertex[expected_faces],
            ):
                return False
            if surface.mesh is not None:
                if end > surface.mesh.faces.shape[0]:
                    return False
                for offset in range(0, len(expected_faces), _CHUNK):
                    last = min(offset + _CHUNK, len(expected_faces))
                    if not np.array_equal(
                        np.asarray(
                            surface.mesh.faces[
                                triangle_count + offset : triangle_count + last
                            ]
                        ),
                        expected_faces[offset:last],
                    ):
                        return False
            triangle_count = end
    except (IndexError, KeyError, ValueError, OverflowError):
        return False
    if not np.all(root_used) or triangle_count != len(s.triangle_cell_rows):
        return False
    if np.asarray(surface.feature_edges).shape != (0, 2):
        return False
    del root_used, amb_keys, amb_records, amb_signs
    vertex_by_root = root_by_vertex = None
    witness_cells = np.flatnonzero(witness_root >= 0)
    if len(witness_cells):
        witness_cells = witness_cells[
            np.lexsort(np.asarray(s.cell_indices)[witness_cells, 1:].T[::-1])
        ]
    if (
        len(surface.boundary_witness_boxes) != len(witness_cells)
        or len(surface.boundary_witnesses) != len(witness_cells)
        or len(surface.boundary_errors) != len(witness_cells)
    ):
        return False
    for start in range(0, len(witness_cells), _CHUNK):
        stop = min(start + _CHUNK, len(witness_cells))
        rows = witness_cells[start:stop]
        roots = witness_root[rows]
        if (
            not _same_float_bits(
                surface.boundary_witness_boxes[start:stop], s.cell_boxes[rows]
            )
            or not _same_float_bits(
                surface.boundary_witnesses[start:stop], s.root_points[roots]
            )
            or not _same_float_bits(
                surface.boundary_errors[start:stop], s.root_errors[roots]
            )
        ):
            return False
    if surface.mesh is not None and (
        vertex_count != surface.mesh.vertices.shape[0]
        or triangle_count != surface.mesh.faces.shape[0]
    ):
        return False
    return True


type _RootRecord = tuple[np.ndarray, float, int, np.ndarray, np.ndarray, int, int]


def _attempt_roots(
    bounds: _FieldEnclosures,
    edge_ids: np.ndarray,
    endpoint_points: np.ndarray,
    endpoint_signs: np.ndarray,
    tolerance: float,
    level: int,
    clipped: ClippedBoundary | None,
    /,
) -> tuple[_RootRecord, ...]:
    """Root records of one cell's crossing edges; clipped face edges from the bank."""
    from ._adaptive_discovery import _root_on_edges

    if clipped is None:
        return _root_on_edges(bounds, endpoint_points, endpoint_signs, tolerance)
    on_face = boundary_face_edges(edge_ids, 1 << level)
    records: list[_RootRecord | None] = [None] * len(edge_ids)
    interior = np.flatnonzero(~on_face)
    if interior.size:
        solved = _root_on_edges(
            bounds, endpoint_points[interior], endpoint_signs[interior], tolerance
        )
        for row, record in zip(interior.tolist(), solved, strict=True):
            records[row] = record
    rows = np.flatnonzero(on_face)
    if rows.size:
        bank = _records(clipped.root_edge_ids.reshape(-1, 6))
        wanted = _records(np.ascontiguousarray(edge_ids[rows].reshape(-1, 6)))
        found = np.searchsorted(bank, wanted)
        if np.any(found >= len(bank)) or not np.array_equal(bank[found], wanted):
            raise ValueError(
                "A clipped domain-face crossing is missing from the crease root bank."
            )
        for row, position in zip(rows.tolist(), found.tolist(), strict=True):
            records[row] = (
                np.asarray(clipped.root_points[position], dtype=np.float64),
                float(clipped.root_errors[position]),
                int(clipped.root_issues[position]),
                np.asarray(clipped.root_brackets[position, 0], dtype=np.float64),
                np.asarray(clipped.root_brackets[position, 1], dtype=np.float64),
                int(clipped.root_signs[position, 0]),
                int(clipped.root_signs[position, 1]),
            )
    return tuple(record for record in records if record is not None)


def replay_history_attempt(
    geometry: CompiledGeometry,
    domain: np.ndarray,
    policy: AdaptiveImplicitSurfacePolicy,
    /,
    *,
    clipped: ClippedBoundary | None = None,
) -> tuple[bool, bool, np.ndarray]:
    """Stream a rejected attempt's retry, lattice-coincidence and tangency predicates.

    Returns whether the attempt refines (`_next_discovery_resolution`), whether
    its undecided vertices are lattice coincidences
    (`_vertex_roots_are_lattice_coincidences`), and its ``TANGENCY`` boxes in
    leaf order. The disk-backed index retains only shared face decisions and
    vertex order; no cover, mesh, scaffold, or prior discovery object is
    materialized. ``clipped`` replays the declared open-boundary mode with that
    attempt's crease certificate: domain-face roots are its bank records.
    """
    from ._adaptive_discovery import (
        _Cell,
        _cell_diagonal,
        _CORNER_OFFSETS,
        _CUBE_EDGES,
        _CUBE_FACES,
        _domain_boundary_faces,
        _edge_key,
        _faces_exclude_zero,
        _FieldEnclosures,
        _GridEdge,
        _prove_constant_signs,
        _ROOT_PROBE_OFFSETS,
        _ROOT_PROBE_SCALE,
        _tangency_faces,
        _target_level,
        _triangulate_cell_band,
        AdaptiveImplicitBoxIssue as Issue,
        AdaptiveImplicitSurfaceStatus as Status,
    )
    from ._adaptive_native_polygon import contours
    from ._adaptive_packed_frontier import _Coordinates, discover_leaves

    level = _target_level(domain, policy.relative_cell_diameter)
    bounds = _FieldEnclosures(geometry, policy.enclosure)
    geometry_bounds = np.asarray(geometry.bounds, dtype=np.float64)
    source_in_domain = (
        geometry_bounds.shape == (2, 3)
        and np.all(np.isfinite(geometry_bounds))
        and np.all(domain[0] <= geometry_bounds[0])
        and np.all(domain[1] >= geometry_bounds[1])
    )
    refine_mask = int(Issue.EDGE | Issue.MULTIPLE_SHEETS)
    tangency_boxes: list[np.ndarray] = []
    unresolved = False
    refinable = True
    max_diagonal = 0.0
    has_undecided = False
    crossing = True
    vertex_count = 0
    boundary_crossing = False
    face_order = (0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 10, 11)
    unresolved_faces = frozenset() if clipped is None else clipped.unresolved_face_keys()

    def key_bytes(key: tuple[tuple[int, int, int], ...]) -> bytes:
        return np.asarray(key, dtype=np.int64).tobytes()

    with closing(sqlite3.connect("")) as database:
        database.execute(
            "CREATE TABLE faces (key BLOB PRIMARY KEY, sign INTEGER NOT NULL)"
        )
        database.execute(
            "CREATE TABLE vertices (key BLOB PRIMARY KEY, row INTEGER NOT NULL)"
        )

        class PointRows(dict[int, np.ndarray]):
            def append(self, point: np.ndarray) -> None:
                raise ValueError("unassigned replay vertex")

        class EdgeRows(dict[int, _GridEdge]):
            def append(self, edge: _GridEdge) -> None:
                raise ValueError("unassigned replay root")

        def consume(
            levels: np.ndarray,
            indices: np.ndarray,
            boxes: np.ndarray,
            lower: np.ndarray,
            upper: np.ndarray,
            grad_lower: np.ndarray,
            grad_upper: np.ndarray,
            issues: np.ndarray,
            mask: np.ndarray,
        ) -> None:
            nonlocal has_undecided, crossing, vertex_count, boundary_crossing
            coords = _Coordinates(domain, level, 0)

            def tally(row: int, issue: int, /) -> None:
                nonlocal unresolved, refinable, max_diagonal
                unresolved = True
                refinable &= not bool(issue & ~refine_mask)
                max_diagonal = max(
                    max_diagonal, _cell_diagonal(boxes[row, 0], boxes[row, 1])
                )

            cell_rows: list[int] = []
            for row in np.flatnonzero(mask).tolist():
                leaf_issue = int(issues[row])
                if leaf_issue or int(levels[row]) != level:
                    tally(row, leaf_issue)
                    continue
                cell_rows.append(row)
            if not cell_rows:
                return
            # Corner and root evaluations are batched per streamed block; every
            # enclosure is row-wise, so the batch never changes a result.
            all_indices = indices[cell_rows][:, None, :] + _CORNER_OFFSETS
            all_points = coords.values(all_indices.reshape(-1, 3), 2).reshape(-1, 8, 3)
            all_lo, all_hi = bounds.point_values(all_points.reshape(-1, 3))
            all_values = bounds.field_values(all_points.reshape(-1, 3)).reshape(-1, 8)
            all_signs = (
                np.where(all_hi < 0, -1, np.where(all_lo > 0, 1, 0))
                .astype(np.int8)
                .reshape(-1, 8)
            )
            pending: list[
                tuple[
                    int,
                    _Cell,
                    np.ndarray,
                    np.ndarray,
                    tuple[tuple[int, int, int], ...],
                    dict[tuple[tuple[int, int, int], ...], int],
                    dict[int, _GridEdge],
                    list[_GridEdge],
                ]
            ] = []
            root_requests: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            for local, row in enumerate(cell_rows):
                cell_indices = all_indices[local]
                points = all_points[local]
                values = all_values[local]
                signs = all_signs[local]
                corner_keys = tuple(
                    (int(point[0]), int(point[1]), int(point[2]))
                    for point in cell_indices
                )
                zero = np.flatnonzero(signs == 0)
                if len(zero):
                    has_undecided = True
                    step = _ROOT_PROBE_SCALE * (domain[1] - domain[0]) / 2.0**level
                    probes = (
                        points[zero, None, :] + _ROOT_PROBE_OFFSETS[None, :, :] * step
                    )
                    probe_lo, probe_hi = bounds.point_values(probes.reshape(-1, 3))
                    probe_lo = probe_lo.reshape(len(zero), -1)
                    probe_hi = probe_hi.reshape(len(zero), -1)
                    crossing &= bool(
                        np.all(
                            np.any(probe_hi < 0, axis=1) & np.any(probe_lo > 0, axis=1)
                        )
                    )
                face_signs = {}
                if not len(zero):
                    for face in _CUBE_FACES:
                        fs = signs[list(face)]
                        if not np.all(fs != np.roll(fs, -1)):
                            continue
                        face_key = tuple(sorted(corner_keys[i] for i in face))
                        encoded = key_bytes(face_key)
                        found = database.execute(
                            "SELECT sign FROM faces WHERE key=?", (encoded,)
                        ).fetchone()
                        if found is None:
                            center = np.mean(points[list(face)], axis=0).reshape(1, 3)
                            center_lo, center_hi = bounds.point_values(center)
                            face_sign = (
                                -1 if center_hi[0] < 0 else 1 if center_lo[0] > 0 else 0
                            )
                            database.execute(
                                "INSERT INTO faces VALUES (?, ?)", (encoded, face_sign)
                            )
                        else:
                            face_sign = found[0]
                        face_signs[face_key] = face_sign
                cell = _Cell(
                    level,
                    (int(indices[row, 0]), int(indices[row, 1]), int(indices[row, 2])),
                    boxes[row, 0],
                    boxes[row, 1],
                    lower[row],
                    upper[row],
                    grad_lower[row],
                    grad_upper[row],
                    0,
                )
                if len(zero):
                    issue = int(Issue.SINGULAR)
                elif not (np.any(signs < 0) and np.any(signs > 0)):
                    if bool(
                        _prove_constant_signs(
                            bounds,
                            boxes[row : row + 1, 0],
                            boxes[row : row + 1, 1],
                            signs[:1],
                            grad_lower[row : row + 1],
                            grad_upper[row : row + 1],
                        )[0]
                    ):
                        # Proved free of the zero set: no contour, no issue.
                        continue
                    gradient = np.any((grad_lower[row] > 0) | (grad_upper[row] < 0))
                    issue = int(Issue.EDGE if gradient else Issue.SINGULAR)
                    if gradient and bool(
                        np.any(
                            _tangency_faces(
                                bounds,
                                boxes[row : row + 1, 0],
                                boxes[row : row + 1, 1],
                                signs[:1],
                                grad_lower[row : row + 1],
                                grad_upper[row : row + 1],
                            )
                        )
                    ):
                        # A lattice-face tangency is not refined; the
                        # lattice is re-anchored instead.
                        tangency_boxes.append(np.array(boxes[row], dtype=np.float64))
                        continue
                else:
                    issue = 0
                if not issue and clipped is not None:
                    if any(
                        key in unresolved_faces
                        for key in clipped_face_keys(indices[row], level)
                    ):
                        issue = int(Issue.DOMAIN_BOUNDARY)
                elif not issue:
                    faces = _domain_boundary_faces(cell, domain)
                    if faces and not _faces_exclude_zero(bounds, faces):
                        issue = int(Issue.DOMAIN_BOUNDARY)
                        boundary_crossing = True
                if issue:
                    tally(row, issue)
                    continue
                edge_keys: dict[int, _GridEdge] = {}
                for edge_no, (a, b) in enumerate(_CUBE_EDGES):
                    if signs[a] != signs[b]:
                        edge_keys[edge_no] = _edge_key(corner_keys[a], corner_keys[b])
                sorted_keys = sorted(edge_keys.values())
                endpoints = np.asarray(sorted_keys, dtype=np.int64)
                sign_by_key = dict(zip(corner_keys, signs, strict=True))
                root_requests.append(
                    (
                        endpoints.reshape(-1, 2, 3),
                        coords.values(endpoints.reshape(-1, 3), 2).reshape(-1, 2, 3),
                        np.asarray(
                            [[sign_by_key[a], sign_by_key[b]] for a, b in sorted_keys],
                            dtype=np.int8,
                        ),
                    )
                )
                pending.append(
                    (
                        row,
                        cell,
                        signs,
                        values,
                        corner_keys,
                        face_signs,
                        edge_keys,
                        sorted_keys,
                    )
                )
            if not pending:
                return
            all_records = _attempt_roots(
                bounds,
                np.concatenate([request[0] for request in root_requests]),
                np.concatenate([request[1] for request in root_requests]),
                np.concatenate([request[2] for request in root_requests]),
                policy.root_tolerance,
                level,
                clipped,
            )
            offset = 0
            for (
                row,
                cell,
                signs,
                values,
                corner_keys,
                face_signs,
                edge_keys,
                sorted_keys,
            ) in pending:
                records = all_records[offset : offset + len(sorted_keys)]
                offset += len(sorted_keys)
                root_cache = dict(zip(sorted_keys, records, strict=True))
                vertices = {}
                for edge_no in face_order:
                    if edge_no not in edge_keys:
                        continue
                    key = edge_keys[edge_no]
                    encoded = key_bytes(key)
                    found = database.execute(
                        "SELECT row FROM vertices WHERE key=?", (encoded,)
                    ).fetchone()
                    if found is None:
                        vertex = vertex_count
                        vertex_count += 1
                        database.execute(
                            "INSERT INTO vertices VALUES (?, ?)", (encoded, vertex)
                        )
                    else:
                        vertex = found[0]
                    vertices[key] = vertex
                root_rows = {key: i for i, key in enumerate(sorted_keys)}
                edge_roots = np.full((1, 12), -1, dtype=np.int64)
                for edge_no, key in edge_keys.items():
                    edge_roots[0, edge_no] = root_rows[key]
                vertex_rows = np.asarray(
                    [vertices[key] for key in sorted_keys], dtype=np.int64
                )
                root_points = np.asarray(
                    [root_cache[key][0] for key in sorted_keys], dtype=np.float64
                )
                root_issues = np.asarray(
                    [root_cache[key][2] for key in sorted_keys], dtype=np.int64
                )
                fast, _, _, native_issues = contours(
                    signs.reshape(1, 8),
                    values.reshape(1, 8),
                    boxes[row : row + 1],
                    edge_roots,
                    vertex_rows,
                    root_points,
                    root_issues,
                    np.asarray(
                        [root_cache[key][1] for key in sorted_keys],
                        dtype=np.float64,
                    ),
                )
                if fast[0]:
                    issue = int(native_issues[0])
                else:
                    point_by_vertex = PointRows(
                        (vertices[key], root_cache[key][0]) for key in sorted_keys
                    )
                    key_by_vertex = EdgeRows((vertices[key], key) for key in sorted_keys)
                    issue, _, _ = _triangulate_cell_band(
                        cell,
                        corner_keys,
                        {
                            key: int(sign)
                            for key, sign in zip(corner_keys, signs, strict=True)
                        },
                        {
                            key: float(value)
                            for key, value in zip(corner_keys, values, strict=True)
                        },
                        face_signs,
                        root_cache,
                        point_by_vertex,
                        [],
                        vertices,
                        [],
                        key_by_vertex,
                        [],
                        0,
                        policy,
                    )
                if issue:
                    tally(row, issue)

        _, status = discover_leaves(bounds, domain, level, sink=consume)
        retryable_status = int(
            Status.UNRESOLVED_BOXES | Status.PHYSICAL_RESOLUTION_REACHED
        )
        retry = (
            unresolved
            and refinable
            and (source_in_domain or clipped is not None)
            and not boundary_crossing
            and not (status & ~retryable_status)
            and max_diagonal > policy.root_tolerance
        )
        next_relative = policy.relative_cell_diameter * 0.5
        if (
            not np.isfinite(next_relative)
            or next_relative <= 0
            or next_relative >= policy.relative_cell_diameter
            or _target_level(domain, next_relative) <= level
        ):
            retry = False
        tangencies = (
            np.stack(tangency_boxes)
            if tangency_boxes
            else np.empty((0, 2, 3), dtype=np.float64)
        )
        return bool(retry), bool(has_undecided and crossing), tangencies
