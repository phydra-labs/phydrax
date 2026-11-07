"""Packed, deterministic polygonization of adaptive implicit leaves.

Only the public scaffold and mesh scale with the extracted surface.  Topology is
sorted in place, while temporary queries and root refinement are chunked.  The
legacy cell contour kernel is reused with dictionaries local to a single cell.
"""

from __future__ import annotations

import math
import mmap
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from jax.typing import DTypeLike


if TYPE_CHECKING:
    from ..simplicial import TriangleMesh
    from ._adaptive_discovery import _FieldEnclosures, _GridEdge
    from ._adaptive_packed_frontier import PackedPolygonLeaves
    from ._policy import AdaptiveImplicitSurfacePolicy

_CHUNK = 16384


@dataclass(slots=True)
class PackedPolygon:
    mesh: TriangleMesh | None
    scaffold_arrays: dict[str, np.ndarray]
    unresolved_boxes: np.ndarray
    unresolved_issues: np.ndarray
    witness_boxes: np.ndarray
    witness_values: np.ndarray
    witness_errors: np.ndarray
    status: int
    max_surface_diagonal: float
    quotient: QuotientContour | None = None
    clipped: ClippedContour | None = None


@dataclass(frozen=True, slots=True)
class QuotientContour:
    """Lifted contour of a quotient discovery over a fully periodic domain box.

    Vertices on opposite domain faces are distinct lifted rows. ``vertex_orbits``
    names each vertex's canonical root edge, and ``vertex_windings`` (0 or 1
    per axis) its translate by whole domain extents: a vertex with winding
    ``w`` is its orbit's canonical root moved by ``w * (upper - lower)``,
    with the shifted coordinates set to the upper domain face exactly.
    """

    points: np.ndarray
    faces: np.ndarray
    vertex_orbits: np.ndarray
    vertex_windings: np.ndarray


def _strictly_increasing_rows(rows: np.ndarray, /) -> bool:
    """Whether integer rows are strictly increasing in lexicographic order."""
    steps = rows[1:] - rows[:-1]
    nonzero = steps != 0
    if not np.all(np.any(nonzero, axis=1)):
        return False
    first = np.argmax(nonzero, axis=1)
    return bool(np.all(steps[np.arange(steps.shape[0]), first] > 0))


@dataclass(frozen=True, slots=True)
class ClippedBoundary:
    """Declared open-boundary polygonization of a clipped discovery domain.

    Clipped mode is explicit and opt-in: the domain box is a clipping set,
    so the zero set may cross its faces. The boundary trace is supplied by
    the parametric-chart crease discovery of every domain face at the same
    target level: ``root_edge_ids`` (sorted corner-id pairs, lexicographic
    rows) with their root records name every boundary-face grid edge with a
    certified sign change, and these records are used verbatim, so contour
    edges in a domain face coincide exactly with the crease segments.
    ``unresolved_faces`` rows ``(axis, side, first, second)`` name the
    target-level face cells whose crease the chart discovery did not
    certify (a tangential or unresolved boundary contact); a band cell on
    such a face keeps the ``DOMAIN_BOUNDARY`` issue.
    """

    root_edge_ids: np.ndarray
    root_points: np.ndarray
    root_errors: np.ndarray
    root_issues: np.ndarray
    root_brackets: np.ndarray
    root_signs: np.ndarray
    unresolved_faces: np.ndarray

    def __post_init__(self) -> None:
        ids = np.ascontiguousarray(self.root_edge_ids, dtype=np.int64)
        count = ids.shape[0] if ids.ndim == 3 else -1
        unresolved = np.ascontiguousarray(self.unresolved_faces, dtype=np.int64)
        if (
            ids.shape != (count, 2, 3)
            or np.asarray(self.root_points).shape != (count, 3)
            or np.asarray(self.root_errors).shape != (count,)
            or np.asarray(self.root_issues).shape != (count,)
            or np.asarray(self.root_brackets).shape != (count, 2, 3)
            or np.asarray(self.root_signs).shape != (count, 2)
            or unresolved.ndim != 2
            or unresolved.shape[1] != 4
        ):
            raise ValueError("Clipped boundary root records must align.")
        flat = ids.reshape(-1, 6)
        if count > 1 and not _strictly_increasing_rows(flat):
            raise ValueError("Clipped boundary root edges must be sorted and unique.")
        object.__setattr__(self, "root_edge_ids", ids)
        object.__setattr__(self, "unresolved_faces", unresolved)

    def unresolved_face_keys(self) -> frozenset[tuple[int, int, int, int]]:
        return frozenset(
            (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
            for row in self.unresolved_faces
        )


@dataclass(frozen=True, slots=True)
class ClippedContour:
    """Open contour of a clipped discovery, bounded by its domain-face trace.

    ``vertex_root_rows`` names each vertex's root-edge row in the scaffold;
    a vertex on a domain face carries the clipped boundary's root record.
    """

    points: np.ndarray
    faces: np.ndarray
    vertex_root_rows: np.ndarray


def clipped_face_keys(
    cell_index: np.ndarray, target_level: int, /
) -> tuple[tuple[int, int, int, int], ...]:
    """``(axis, side, first, second)`` keys of a target-level cell's domain faces."""
    last = (1 << target_level) - 1
    keys: list[tuple[int, int, int, int]] = []
    for axis in range(3):
        first, second = (other for other in range(3) if other != axis)
        for side, index in ((0, 0), (1, last)):
            if int(cell_index[axis]) == index:
                keys.append((axis, side, int(cell_index[first]), int(cell_index[second])))
    return tuple(keys)


def _grid_points(ids: np.ndarray, target_level: int, domain: np.ndarray, /) -> np.ndarray:
    """Exact dyadic grid coordinates of corner ids (rounded once, per axis)."""
    from ._adaptive_discovery import _grid_coordinate

    points = np.empty(ids.shape, dtype=np.float64)
    for axis in range(3):
        axis_ids, inverse = np.unique(ids[:, axis], return_inverse=True)
        coords = np.asarray(
            [
                _grid_coordinate(int(i), target_level, domain[0, axis], domain[1, axis])
                for i in axis_ids
            ],
            dtype=np.float64,
        )
        points[:, axis] = coords[inverse.reshape(-1)]
    return points


def _unique_point_values(
    bounds: _FieldEnclosures, points: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate each distinct point once, so equal points get identical evidence."""
    unique, inverse = np.unique(points, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    lower = np.empty(len(unique), dtype=np.float64)
    upper = np.empty_like(lower)
    values = np.empty_like(lower)
    for start in range(0, len(unique), _CHUNK):
        stop = min(start + _CHUNK, len(unique))
        lower[start:stop], upper[start:stop] = bounds.point_values(unique[start:stop])
        values[start:stop] = bounds.field_values(unique[start:stop])
    return lower[inverse], upper[inverse], values[inverse]


class _PairPoints:
    """Chunk-sliced endpoint coordinates of root pairs without a full copy."""

    def __init__(self, points: np.ndarray, pairs: np.ndarray, /) -> None:
        self._points = points
        self._pairs = pairs

    def __getitem__(self, rows: slice, /) -> np.ndarray:
        return self._points[self._pairs[rows]]


def _records(array: np.ndarray) -> np.ndarray:
    """View contiguous integer rows as lexicographically ordered records."""
    dtype = np.dtype([(f"f{i}", np.int64) for i in range(array.shape[1])])
    return array.view(dtype).reshape(-1)


def _sort_unique_inplace(rows: np.ndarray) -> np.ndarray:
    """Sort rows and compact unique records, with bounded comparison scratch."""
    records = _records(rows)
    records.sort(kind="quicksort")
    size = len(records)
    if not size:
        return rows
    out = 1
    # Compare before writing: compaction always moves toward earlier positions.
    for start in range(1, size, _CHUNK):
        stop = min(start + _CHUNK, size)
        keep = records[start:stop] != records[start - 1 : stop - 1]
        selected = rows[start:stop][keep]
        rows[out : out + len(selected)] = selected
        out += len(selected)
    return rows[:out].copy()


class _Chunks:
    """Append-only numeric rows, retaining no Python object per output row."""

    def __init__(self, shape: tuple[int, ...], dtype: DTypeLike) -> None:
        dtype_ = np.dtype(dtype)
        self.shape, self.dtype = shape, dtype_
        self.blocks: list[np.ndarray] = []
        self.count = 0

    def _new_block(self) -> np.ndarray:
        # Staging buffers must not remain resident in the allocator after coalescing.
        size = _CHUNK * math.prod(self.shape) * self.dtype.itemsize
        if self.dtype.hasobject or size == 0:
            return np.empty((_CHUNK, *self.shape), dtype=self.dtype)
        mapping = mmap.mmap(-1, size)
        return np.ndarray((_CHUNK, *self.shape), dtype=self.dtype, buffer=mapping)

    def append(self, value: object) -> None:
        if self.count % _CHUNK == 0:
            self.blocks.append(self._new_block())
        self.blocks[-1][self.count % _CHUNK] = value
        self.count += 1

    def extend(self, values: np.ndarray) -> None:
        offset = 0
        while offset < len(values):
            if self.count % _CHUNK == 0:
                self.blocks.append(self._new_block())
            available = _CHUNK - self.count % _CHUNK
            take = min(available, len(values) - offset)
            self.blocks[-1][self.count % _CHUNK : self.count % _CHUNK + take] = values[
                offset : offset + take
            ]
            self.count += take
            offset += take

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> np.ndarray:
        return self.blocks[index // _CHUNK][index % _CHUNK]

    def take_array(self) -> np.ndarray:
        """Materialize once, releasing each source block as its rows are copied."""
        result = np.empty((self.count, *self.shape), dtype=self.dtype)
        while self.blocks:
            start = (len(self.blocks) - 1) * _CHUNK
            stop = min(start + _CHUNK, self.count)
            block = self.blocks.pop()
            result[start:stop] = block[: stop - start]
            mapping = block.base
            del block
            if isinstance(mapping, mmap.mmap):
                mapping.close()
        self.count = 0
        return result


type _RootArrays = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def boundary_face_edges(edge_ids: np.ndarray, span: int, /) -> np.ndarray:
    """Whether each ``(n, 2, 3)`` corner-id edge lies in a domain face."""
    ends = np.asarray(edge_ids, dtype=np.int64)
    same = ends[:, 0] == ends[:, 1]
    on_face = same & ((ends[:, 0] == 0) | (ends[:, 0] == span))
    return np.any(on_face, axis=1)


def _clipped_roots(
    clipped: ClippedBoundary,
    corner_ids: np.ndarray,
    pairs: np.ndarray,
    point_signs: np.ndarray,
    span: int,
    solve: Callable[[np.ndarray | _PairPoints, np.ndarray], _RootArrays],
    corner_points: np.ndarray,
    /,
) -> _RootArrays:
    """Roots of crossing edges: boundary-face edges from the crease root bank.

    A domain-face edge takes the declared record verbatim; every other edge is
    solved by the ordinary root kernel. A crossing face edge missing from the
    bank, or a bank record whose endpoint signs differ from the corner signs,
    is an inconsistent declaration and is refused.
    """
    count = len(pairs)
    points = np.empty((count, 3), dtype=np.float64)
    errors = np.empty(count, dtype=np.float64)
    issues = np.empty(count, dtype=np.int64)
    brackets = np.empty((count, 2, 3), dtype=np.float64)
    signs = np.empty((count, 2), dtype=np.int8)
    edges = np.ascontiguousarray(corner_ids[pairs], dtype=np.int64)
    on_boundary = boundary_face_edges(edges, span)
    boundary_rows = np.flatnonzero(on_boundary)
    if boundary_rows.size:
        bank = _records(clipped.root_edge_ids.reshape(-1, 6))
        wanted = _records(np.ascontiguousarray(edges[boundary_rows].reshape(-1, 6)))
        found = np.searchsorted(bank, wanted)
        if np.any(found >= len(bank)) or not np.array_equal(bank[found], wanted):
            raise ValueError(
                "A clipped domain-face crossing is missing from the crease root bank."
            )
        declared_signs = np.asarray(clipped.root_signs, dtype=np.int8)[found]
        if not np.array_equal(declared_signs, point_signs[pairs[boundary_rows]]):
            raise ValueError("A crease root record disagrees with the corner signs.")
        points[boundary_rows] = np.asarray(clipped.root_points)[found]
        errors[boundary_rows] = np.asarray(clipped.root_errors)[found]
        issues[boundary_rows] = np.asarray(clipped.root_issues)[found]
        brackets[boundary_rows] = np.asarray(clipped.root_brackets)[found]
        signs[boundary_rows] = declared_signs
    interior_rows = np.flatnonzero(~on_boundary)
    if interior_rows.size:
        solved = solve(
            _PairPoints(corner_points, pairs[interior_rows]),
            point_signs[pairs[interior_rows]],
        )
        for target, values in zip(
            (points, errors, issues, brackets, signs), solved, strict=True
        ):
            target[interior_rows] = values
    return points, errors, issues, brackets, signs


def polygonize(
    bounds: _FieldEnclosures,
    domain: np.ndarray,
    target_level: int,
    policy: AdaptiveImplicitSurfacePolicy,
    leaves: PackedPolygonLeaves,
    source_id: str,
    source_in_domain: bool | np.bool_,
    /,
    *,
    periodic: bool = False,
    clipped: ClippedBoundary | None = None,
) -> PackedPolygon:
    """Polygonize band leaves; ``periodic`` identifies opposite domain faces.

    In quotient mode the domain box is a fundamental cell of a fully periodic
    field. Every evaluation is made once at the canonical representative of
    its lattice orbit (corners, root edges and ambiguous face centres on an
    upper domain face move to the lower one), so identified entities carry
    bitwise-identical signs and roots, and there is no domain boundary.

    ``clipped`` declares the open-boundary mode (see ``ClippedBoundary``):
    the domain faces clip the zero set, their crossing roots come from the
    declared crease root bank, and only a band cell on an uncertified face
    cell keeps the ``DOMAIN_BOUNDARY`` issue. Without it the closed-domain
    behavior is unchanged.
    """
    if periodic and clipped is not None:
        raise ValueError("A quotient discovery has no clipped domain boundary.")
    # Imports are deferred because the owning discovery module calls this function.
    from ._adaptive_discovery import (
        _Cell,
        _CORNER_OFFSETS,
        _CUBE_EDGES,
        _CUBE_FACES,
        _domain_boundary_faces,
        _edge_key,
        _faces_exclude_zero,
        _grid_coordinate,
        _norm_upper_rows,
        _prove_constant_signs,
        _root_on_edges,
        _tangency_faces,
        _TETRAHEDRA,
        _triangulate_cell_band,
        AdaptiveImplicitBoxIssue as Issue,
        AdaptiveImplicitSurfaceStatus as Status,
        TriangleMesh,
    )

    surface_rows = leaves.surface_rows
    surface_issue = leaves.issue[surface_rows].copy()
    mesh_mask = (surface_issue == 0) & (leaves.level[surface_rows] == target_level)
    mesh_surface_rows = np.flatnonzero(mesh_mask)
    mesh_rows = surface_rows[mesh_surface_rows]
    count = len(mesh_rows)
    cell_indices = np.empty((count, 4), dtype=np.int64)
    cell_indices[:, 0] = leaves.level[mesh_rows]
    cell_indices[:, 1:] = leaves.index[mesh_rows]
    # Legacy corner IDs are exported as int64; do not silently wrap a corner.
    if np.any(cell_indices[:, 1:] == np.iinfo(np.int64).max):
        raise OverflowError("A lattice corner index cannot be represented as int64.")
    cell_boxes = leaves.boxes[mesh_rows]
    cell_corner_ids = np.empty((count, 8, 3), dtype=np.int64)
    for start in range(0, count, _CHUNK):
        stop = min(start + _CHUNK, count)
        cell_corner_ids[start:stop] = cell_indices[start:stop, None, 1:] + _CORNER_OFFSETS
    if cell_corner_ids.size:
        scratch = mmap.mmap(-1, cell_corner_ids.nbytes)
        rows = np.ndarray((count * 8, 3), dtype=np.int64, buffer=scratch)
        try:
            np.copyto(rows, cell_corner_ids.reshape(-1, 3))
            corner_ids = _sort_unique_inplace(rows)
        except BaseException:
            # Do not mask the original failure if its traceback retains a view.
            del rows, scratch
            raise
        del rows
        scratch.close()
    else:
        corner_ids = _sort_unique_inplace(cell_corner_ids.reshape(-1, 3).copy())
    corner_records = _records(corner_ids)
    corner_row_scratch = (
        mmap.mmap(-1, count * 8 * np.dtype(np.int64).itemsize) if count else None
    )
    cell_corner_rows = (
        np.ndarray((count, 8), dtype=np.int64, buffer=corner_row_scratch)
        if corner_row_scratch is not None
        else np.empty((0, 8), dtype=np.int64)
    )
    cr = None  # Later chunk slices must not keep the full inverse map alive.
    for start in range(0, count, _CHUNK):
        stop = min(start + _CHUNK, count)
        cell_corner_rows[start:stop] = np.searchsorted(
            corner_records, _records(cell_corner_ids[start:stop].reshape(-1, 3))
        ).reshape(-1, 8)

    # Deduplicate coordinates within bounded chunks, using the exact Fraction
    # kernel rather than an algebraically equivalent but differently rounded grid.
    corner_points = np.empty((len(corner_ids), 3), dtype=np.float64)
    for start in range(0, len(corner_ids), _CHUNK):
        stop = min(start + _CHUNK, len(corner_ids))
        for axis in range(3):
            axis_ids, inverse = np.unique(
                corner_ids[start:stop, axis], return_inverse=True
            )
            coords = np.asarray(
                [
                    _grid_coordinate(
                        int(i), target_level, domain[0, axis], domain[1, axis]
                    )
                    for i in axis_ids
                ]
            )
            corner_points[start:stop, axis] = coords[inverse]
    span = 1 << target_level
    if periodic:
        point_lower, point_upper, point_values = _unique_point_values(
            bounds, _grid_points(corner_ids % span, target_level, domain)
        )
    else:
        point_lower = np.empty(len(corner_ids), dtype=np.float64)
        point_upper = np.empty_like(point_lower)
        point_values = np.empty_like(point_lower)
        for start in range(0, len(corner_ids), _CHUNK):
            stop = min(start + _CHUNK, len(corner_ids))
            point_lower[start:stop], point_upper[start:stop] = bounds.point_values(
                corner_points[start:stop]
            )
            point_values[start:stop] = bounds.field_values(corner_points[start:stop])
    point_signs = np.zeros(len(corner_ids), dtype=np.int8)
    point_signs[point_lower > 0] = 1
    point_signs[point_upper < 0] = -1

    cell_issue = np.zeros(count, dtype=np.int64)
    # Band cells proved free of the zero set: no contour, no issue.
    empty = np.zeros(count, dtype=np.bool_)
    status = 0
    # Classify cells before collecting any crossing edge, as in the original.
    for start in range(0, count, _CHUNK):
        stop = min(start + _CHUNK, count)
        signs = point_signs[cell_corner_rows[start:stop]]
        singular = np.any(signs == 0, axis=1)
        no_crossing = ~singular & ~(np.any(signs < 0, axis=1) & np.any(signs > 0, axis=1))
        gradient = np.any(
            (leaves.gradient_lower[mesh_rows[start:stop]] > 0)
            | (leaves.gradient_upper[mesh_rows[start:stop]] < 0),
            axis=1,
        )
        cell_issue[start:stop][singular] = int(Issue.SINGULAR)
        cell_issue[start:stop][no_crossing & gradient] = int(Issue.EDGE)
        cell_issue[start:stop][no_crossing & ~gradient] = int(Issue.SINGULAR)
        candidates = np.flatnonzero(no_crossing)
        if len(candidates):
            leaf_rows = mesh_rows[start + candidates]
            proved = _prove_constant_signs(
                bounds,
                cell_boxes[start + candidates, 0],
                cell_boxes[start + candidates, 1],
                signs[candidates, 0],
                leaves.gradient_lower[leaf_rows],
                leaves.gradient_upper[leaf_rows],
            )
            empty[start + candidates[proved]] = True
            cell_issue[start + candidates[proved]] = 0
            edge = ~proved & gradient[candidates]
            if np.any(edge):
                tangent = np.any(
                    _tangency_faces(
                        bounds,
                        cell_boxes[start + candidates[edge], 0],
                        cell_boxes[start + candidates[edge], 1],
                        signs[candidates[edge], 0],
                        leaves.gradient_lower[leaf_rows[edge]],
                        leaves.gradient_upper[leaf_rows[edge]],
                    ),
                    axis=1,
                )
                cell_issue[start + candidates[edge][tangent]] = int(Issue.TANGENCY)
    unresolved_faces = None if clipped is None else clipped.unresolved_face_keys()
    for row in range(count):
        if periodic or cell_issue[row] or empty[row]:
            continue
        if not np.any(
            (cell_boxes[row, 0] == domain[0]) | (cell_boxes[row, 1] == domain[1])
        ):
            continue
        if unresolved_faces is not None:
            if any(
                key in unresolved_faces
                for key in clipped_face_keys(cell_indices[row, 1:], target_level)
            ):
                cell_issue[row] = int(Issue.DOMAIN_BOUNDARY)
            continue
        leaf = mesh_rows[row]
        cell = _Cell(
            target_level,
            tuple(cell_indices[row, 1:]),
            cell_boxes[row, 0],
            cell_boxes[row, 1],
            leaves.value_lower[leaf],
            leaves.value_upper[leaf],
            leaves.gradient_lower[leaf],
            leaves.gradient_upper[leaf],
            0,
        )
        boundary = _domain_boundary_faces(cell, domain)
        if boundary and not _faces_exclude_zero(bounds, boundary):
            cell_issue[row] = int(Issue.DOMAIN_BOUNDARY)
            status |= int(Status.DOMAIN_BOUNDARY_CROSSING)

    # Pairs of sorted corner rows have the same order as the old grid-edge keys.
    crossing_count = 0
    for start in range(0, count, _CHUNK):
        stop = min(start + _CHUNK, count)
        cr = cell_corner_rows[start:stop]
        sg = point_signs[cr]
        crossing_count += int(
            np.count_nonzero(
                (sg[:, _CUBE_EDGES[:, 0]] != sg[:, _CUBE_EDGES[:, 1]])
                & (cell_issue[start:stop, None] == 0)
            )
        )
    pair_scratch = (
        mmap.mmap(-1, crossing_count * 2 * np.dtype(np.int64).itemsize)
        if crossing_count
        else None
    )
    pairs = (
        np.ndarray((crossing_count, 2), dtype=np.int64, buffer=pair_scratch)
        if pair_scratch is not None
        else np.empty((0, 2), dtype=np.int64)
    )
    offset = 0
    for start in range(0, count, _CHUNK):
        stop = min(start + _CHUNK, count)
        cr = cell_corner_rows[start:stop]
        sg = point_signs[cr]
        mask = (sg[:, _CUBE_EDGES[:, 0]] != sg[:, _CUBE_EDGES[:, 1]]) & (
            cell_issue[start:stop, None] == 0
        )
        local = np.stack((cr[:, _CUBE_EDGES[:, 0]], cr[:, _CUBE_EDGES[:, 1]]), axis=-1)[
            mask
        ]
        local.sort(axis=1)
        pairs[offset : offset + len(local)] = local
        offset += len(local)
    unique_pairs = _sort_unique_inplace(pairs)
    del pairs
    if pair_scratch is not None:
        pair_scratch.close()
    pairs = unique_pairs
    pair_records = _records(pairs)
    root_count = len(pairs)

    def solve_roots(
        edge_points: np.ndarray | _PairPoints, edge_signs: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        count = len(edge_signs)
        points = np.empty((count, 3), dtype=np.float64)
        errors = np.empty(count, dtype=np.float64)
        issues = np.empty(count, dtype=np.int64)
        brackets = np.empty((count, 2, 3), dtype=np.float64)
        signs = np.empty((count, 2), dtype=np.int8)
        for start in range(0, count, _CHUNK):
            stop = min(start + _CHUNK, count)
            records = _root_on_edges(
                bounds,
                edge_points[start:stop],
                edge_signs[start:stop],
                policy.root_tolerance,
            )
            for j, record in enumerate(records, start):
                points[j], errors[j], issues[j] = record[:3]
                brackets[j, 0], brackets[j, 1] = record[3:5]
                signs[j] = record[5:7]
        return points, errors, issues, brackets, signs

    root_windings = np.zeros((root_count, 3), dtype=np.int64)
    if periodic:
        # An edge lying in an upper domain face is the translate of its
        # canonical edge in the lower face; solve each canonical edge once.
        ends = corner_ids[pairs]
        root_windings = np.all(ends == span, axis=1).astype(np.int64)
        canonical = ends - span * root_windings[:, None, :]
        edges, first, root_orbits = np.unique(
            canonical.reshape(-1, 6), axis=0, return_index=True, return_inverse=True
        )
        root_orbits = root_orbits.reshape(-1)
        solved = solve_roots(
            _grid_points(edges.reshape(-1, 3), target_level, domain).reshape(-1, 2, 3),
            point_signs[pairs[first]],
        )
        root_points, root_errors, root_issues, root_brackets, root_signs = (
            values[root_orbits] for values in solved
        )
        del solved, ends, canonical, edges, first
        for axis in range(3):
            # A canonical root on a lower-face edge keeps that face coordinate
            # exactly, so its translate is the upper-face coordinate exactly.
            moved = root_windings[:, axis] == 1
            root_points[moved, axis] = domain[1, axis]
            root_brackets[moved, :, axis] = domain[1, axis]
    elif clipped is not None:
        root_orbits = np.arange(root_count, dtype=np.int64)
        root_points, root_errors, root_issues, root_brackets, root_signs = _clipped_roots(
            clipped, corner_ids, pairs, point_signs, span, solve_roots, corner_points
        )
    else:
        root_orbits = np.arange(root_count, dtype=np.int64)
        root_points, root_errors, root_issues, root_brackets, root_signs = solve_roots(
            _PairPoints(corner_points, pairs), point_signs[pairs]
        )

    # Ambiguous faces are sparse. Store sorted corner rows and the earliest
    # (cell, face) occurrence so mean summation uses the original face ordering.
    ambiguous = _Chunks((6,), np.int64)
    for start in range(0, count, _CHUNK):
        stop = min(start + _CHUNK, count)
        cr = cell_corner_rows[start:stop]
        sg = point_signs[cr]
        valid = np.all(sg != 0, axis=1)
        for face_id, face in enumerate(_CUBE_FACES):
            fs = sg[:, face]
            mask = valid & np.all(fs != np.roll(fs, -1, axis=1), axis=1)
            for local in np.flatnonzero(mask):
                key = np.sort(cr[local, list(face)])
                ambiguous.append((*key, start + int(local), face_id))
    amb = ambiguous.take_array()
    del ambiguous
    if len(amb):
        # Full records sort by key, then by original insertion order.
        _records(amb).sort(kind="quicksort")
        keep = np.ones(len(amb), dtype=bool)
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
        if periodic:
            face_ids = np.stack(
                [
                    corner_ids[cell_corner_rows[entry[4], list(_CUBE_FACES[entry[5]])]]
                    for entry in amb[start:stop]
                ]
            )
            # A face lying in an upper domain face moves to the lower one.
            face_ids = face_ids - span * np.all(face_ids == span, axis=1)[:, None, :]
            corners = _grid_points(face_ids.reshape(-1, 3), target_level, domain)
            points = np.mean(corners.reshape(-1, 4, 3), axis=1)
            lo, hi, _ = _unique_point_values(bounds, points)
        else:
            points = np.empty((stop - start, 3), dtype=np.float64)
            for j, entry in enumerate(amb[start:stop]):
                points[j] = np.mean(
                    corner_points[
                        cell_corner_rows[entry[4], list(_CUBE_FACES[entry[5]])]
                    ],
                    axis=0,
                )
            lo, hi = bounds.point_values(points)
        amb_signs[start:stop] = np.where(hi < 0, -1, np.where(lo > 0, 1, 0))
    del amb

    vertex_errors = []
    face_dtype = np.int32 if root_count <= np.iinfo(np.int32).max else np.int64
    faces = _Chunks((3,), face_dtype)
    triangle_cells = _Chunks((), np.int64)
    vertex_root_rows = _Chunks((), np.int64)
    vertex_by_root: np.ndarray | None = np.full(root_count, -1, dtype=np.int64)
    witness_root = np.full(count, -1, dtype=np.int64)

    class RootVertices:
        """Vertex-order view of canonical root points without a second copy."""

        def __len__(self) -> int:
            return len(vertex_root_rows)

        def __getitem__(self, vertex: int) -> np.ndarray:
            return root_points[int(vertex_root_rows[vertex])]

        def append(self, point: np.ndarray) -> None:
            # The scalar helper appends the root row immediately before this.
            root = int(vertex_root_rows[len(vertex_root_rows) - 1])
            if not np.array_equal(root_points[root], point):
                raise RuntimeError("Vertex point differs from its root record.")

    vertices = RootVertices()

    class EdgeRows:
        def append(self, key: _GridEdge) -> None:
            vertex_root_rows.append(local_roots[key])

        def __getitem__(self, vertex: int) -> _GridEdge:
            pair = pairs[int(vertex_root_rows[vertex])]
            first, second = corner_ids[pair[0]], corner_ids[pair[1]]
            return (
                (int(first[0]), int(first[1]), int(first[2])),
                (int(second[0]), int(second[1]), int(second[2])),
            )

    class VertexKeys:
        def get(self, key: _GridEdge) -> int | None:
            if vertex_by_root is None:
                raise RuntimeError("Polygon vertex indexing has been released.")
            vertex = int(vertex_by_root[local_roots[key]])
            return None if vertex < 0 else vertex

        def __setitem__(self, key: _GridEdge, value: int) -> None:
            if vertex_by_root is None:
                raise RuntimeError("Polygon vertex indexing has been released.")
            vertex_by_root[local_roots[key]] = value

    edge_rows = EdgeRows()
    vertex_keys = VertexKeys()
    from ._adaptive_native_polygon import contours

    # First occurrence of each crossing edge in the six face traversals.
    face_edge_order = np.asarray((0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 10, 11))
    for start in range(0, count, _CHUNK):
        stop = min(start + _CHUNK, count)
        selected = np.flatnonzero((cell_issue[start:stop] == 0) & ~empty[start:stop])
        if not len(selected):
            continue
        cell_rows = start + selected
        crs = cell_corner_rows[cell_rows]
        cell_signs = point_signs[crs]
        crossing = cell_signs[:, _CUBE_EDGES[:, 0]] != cell_signs[:, _CUBE_EDGES[:, 1]]
        edge_roots = np.full(crossing.shape, -1, dtype=np.int64)
        pairs_to_find = np.stack(
            (crs[:, _CUBE_EDGES[:, 0]], crs[:, _CUBE_EDGES[:, 1]]), axis=-1
        )[crossing]
        pairs_to_find.sort(axis=1)
        edge_roots[crossing] = np.searchsorted(pair_records, _records(pairs_to_find))
        witness_root[cell_rows] = edge_roots[
            np.arange(len(cell_rows)), np.argmax(crossing, axis=1)
        ]

        # Assign exactly the legacy first-encounter vertex order before running
        # either fast or scalar cells.  A crossing edge is visited on its first
        # incident face regardless of whether that cell later becomes unresolved.
        occurrences = edge_roots[:, face_edge_order].reshape(-1)
        occurrences = occurrences[occurrences >= 0]
        unique, first = np.unique(occurrences, return_index=True)
        fresh = vertex_by_root[unique] < 0
        new_roots = unique[fresh][np.argsort(first[fresh])]
        vertex_by_root[new_roots] = np.arange(
            len(vertices), len(vertices) + len(new_roots), dtype=np.int64
        )
        vertex_root_rows.extend(new_roots)

        fast, tri_counts, fast_triangles, issues = contours(
            cell_signs,
            point_values[crs],
            cell_boxes[cell_rows],
            edge_roots,
            vertex_by_root,
            root_points,
            root_issues,
            root_errors,
        )
        cell_issue[cell_rows[fast]] = issues[fast]
        tri_buffer = np.empty((len(cell_rows), 10, 3), dtype=np.int64)
        tri_buffer[fast] = fast_triangles[fast]
        for local in np.flatnonzero(~fast):
            row = int(cell_rows[local])
            cr = crs[local]
            corners = tuple(
                (int(p[0]), int(p[1]), int(p[2])) for p in cell_corner_ids[row]
            )
            signs = {corners[i]: int(point_signs[cr[i]]) for i in range(8)}
            values = {corners[i]: float(point_values[cr[i]]) for i in range(8)}
            local_roots = {}
            root_cache = {}
            for edge_no, (first, second) in enumerate(_CUBE_EDGES):
                root = int(edge_roots[local, edge_no])
                if root < 0:
                    continue
                key = _edge_key(corners[first], corners[second])
                local_roots[key] = root
                root_cache[key] = (
                    root_points[root],
                    float(root_errors[root]),
                    int(root_issues[root]),
                    root_brackets[root, 0],
                    root_brackets[root, 1],
                    int(root_signs[root, 0]),
                    int(root_signs[root, 1]),
                )
            if not local_roots:
                cell_issue[row] = int(Issue.EDGE)
                continue
            face_signs = {}
            for face in _CUBE_FACES:
                fs = point_signs[cr[list(face)]]
                if np.all(fs != np.roll(fs, -1)):
                    key_rows = np.sort(cr[list(face)]).reshape(1, 4)
                    pos = int(np.searchsorted(amb_records, _records(key_rows))[0])
                    face_signs[tuple(sorted(corners[i] for i in face))] = int(
                        amb_signs[pos]
                    )
            leaf = mesh_rows[row]
            cell = _Cell(
                target_level,
                (cell_indices[row, 1], cell_indices[row, 2], cell_indices[row, 3]),
                cell_boxes[row, 0],
                cell_boxes[row, 1],
                leaves.value_lower[leaf],
                leaves.value_upper[leaf],
                leaves.gradient_lower[leaf],
                leaves.gradient_upper[leaf],
                0,
            )
            local_faces = []
            local_triangle_cells = []
            issue, _, _ = _triangulate_cell_band(
                cell,
                corners,
                signs,
                values,
                face_signs,
                root_cache,
                vertices,
                vertex_errors,
                vertex_keys,
                local_faces,
                edge_rows,
                local_triangle_cells,
                row,
                policy,
            )
            cell_issue[row] = issue
            tri_counts[local] = len(local_faces)
            if local_faces:
                tri_buffer[local, : len(local_faces)] = local_faces
        tri_mask = np.arange(10)[None, :] < tri_counts[:, None]
        faces.extend(tri_buffer[tri_mask])
        triangle_cells.extend(np.repeat(cell_rows, tri_counts))

    surface_issue[mesh_surface_rows] = cell_issue
    unresolved_mask = surface_issue != 0
    unresolved_boxes = leaves.boxes[surface_rows[unresolved_mask]]
    unresolved_issues = surface_issue[unresolved_mask]
    max_diagonal = 0.0
    for start in range(0, len(surface_rows), _CHUNK):
        boxes = leaves.boxes[surface_rows[start : start + _CHUNK]]
        max_diagonal = max(
            max_diagonal,
            float(np.max(_norm_upper_rows(boxes[:, 1] - boxes[:, 0]), initial=0.0)),
        )
    leaves._release_auxiliary()
    del surface_rows, surface_issue, mesh_mask, mesh_surface_rows, unresolved_mask
    # All mesh cells are at one level, so sorting their indices gives legacy key order.
    witness_cells = np.flatnonzero(witness_root >= 0)
    if len(witness_cells):
        witness_cells = witness_cells[np.lexsort(cell_indices[witness_cells, 1:].T[::-1])]
    # The full inverse corner map is needed during contouring, but retaining it
    # through mesh topology construction would overlap two large temporaries.
    # Recompute it in bounded chunks when writing the retained tetrahedra.
    del cr, cell_corner_rows
    if corner_row_scratch is not None:
        corner_row_scratch.close()
    vertex_by_root = None
    del root_signs, amb_keys, amb_records, amb_signs
    face_array = faces.take_array()
    vertex_roots = vertex_root_rows.take_array()
    mesh = None
    quotient = None
    clipped_contour = None
    if periodic:
        if len(face_array) and not len(unresolved_issues):
            quotient = QuotientContour(
                root_points[vertex_roots],
                face_array.astype(np.int64, copy=False),
                root_orbits[vertex_roots],
                root_windings[vertex_roots],
            )
    elif clipped is not None:
        if len(face_array) and not len(unresolved_issues):
            clipped_contour = ClippedContour(
                root_points[vertex_roots],
                face_array.astype(np.int64, copy=False),
                vertex_roots.astype(np.int64, copy=True),
            )
    elif len(face_array) and not len(unresolved_issues) and source_in_domain:
        if len(vertex_roots) > np.iinfo(np.int32).max:
            status |= int(Status.DEGENERATE_EXTRACTION)
        else:
            vertex_array = root_points[vertex_roots]
            try:
                candidate = TriangleMesh(
                    vertex_array,
                    face_array.astype(np.int32, copy=False),
                    source_id=source_id,
                )
                if candidate.topology.watertight:
                    mesh = candidate
                else:
                    status |= int(Status.NONMANIFOLD_EXTRACTION)
                del candidate
            except (ValueError, OverflowError):
                status |= int(Status.DEGENERATE_EXTRACTION)
            del vertex_array
    witness_boxes = cell_boxes[witness_cells]
    witness_values = root_points[witness_root[witness_cells]]
    witness_errors = root_errors[witness_root[witness_cells]]
    del witness_root, witness_cells
    triangle_root_rows = vertex_roots[face_array]
    del vertex_roots, face_array
    triangle_cell_array = triangle_cells.take_array()
    tetrahedra = np.empty((count * len(_TETRAHEDRA), 4), dtype=np.int64)
    for start in range(0, count, _CHUNK):
        stop = min(start + _CHUNK, count)
        chunk_ids = cell_corner_ids[start:stop].reshape(-1, 3)
        chunk_rows = np.searchsorted(corner_records, _records(chunk_ids)).reshape(-1, 8)
        tetrahedra[start * len(_TETRAHEDRA) : stop * len(_TETRAHEDRA)] = chunk_rows[
            :, np.asarray(_TETRAHEDRA)
        ].reshape(-1, 4)
    scaffold = dict(
        cell_indices=cell_indices,
        cell_boxes=cell_boxes,
        cell_issues=cell_issue,
        cell_value_lower=leaves.value_lower[mesh_rows],
        cell_value_upper=leaves.value_upper[mesh_rows],
        cell_gradient_lower=leaves.gradient_lower[mesh_rows],
        cell_gradient_upper=leaves.gradient_upper[mesh_rows],
        cell_corner_ids=cell_corner_ids,
        corner_ids=corner_ids,
        corner_coordinates=corner_points,
        corner_value_lower=point_lower,
        corner_value_upper=point_upper,
        corner_sampled_values=point_values,
        corner_signs=point_signs,
        tetrahedra=tetrahedra,
        tetrahedron_cell_rows=np.repeat(
            np.arange(count, dtype=np.int64), len(_TETRAHEDRA)
        ),
        root_edge_ids=corner_ids[pairs],
        root_brackets=root_brackets,
        root_points=root_points,
        root_errors=root_errors,
        root_issues=root_issues,
        root_endpoint_value_lower=point_lower[pairs],
        root_endpoint_value_upper=point_upper[pairs],
        triangle_cell_rows=triangle_cell_array,
        triangle_root_edge_rows=triangle_root_rows,
    )
    return PackedPolygon(
        mesh,
        scaffold,
        unresolved_boxes,
        unresolved_issues,
        witness_boxes,
        witness_values,
        witness_errors,
        status,
        max_diagonal,
        quotient,
        clipped_contour,
    )
