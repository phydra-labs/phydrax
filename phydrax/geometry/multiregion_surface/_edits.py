#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host working mesh, local guards, volume restoration and CCD of surface events.

This module is the host preparation boundary of topology events: all arrays
are NumPy, every decision is exact or conservative, and nothing here is traced.
A pass edits one `_WorkingMesh` sequentially; each event is drafted by its
builder as an `_Edit` (removed faces, new faces with parents, new and moved
vertices, explicit collision-certification legs), restored in finite-region
volume, screened by `_local_guards`, certified by `_certify_legs` and only then
applied with `_apply_edit`. Events of one pass have disjoint vertex 2-rings, so
no guard of a later event can observe a partially applied earlier event.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from ...discretization.contact._search import ContactCandidateEpoch
    from ...discretization.contact._surface import PreparedCollisionScene

from ..._geometry_predicates import PredicateMode, resolve_host_predicate_mode
from ...linalg import ArraySpace
from ._contracts import (
    DRY_FOAM_EDGE_VALENCE,
    DRY_FOAM_VERTEX_REGIONS,
    MultiRegionSurfaceCapacityPlan,
)
from ._events import SurfaceEventKind, SurfaceEventPolicy, SurfaceEventStatus
from ._state import MultiRegionSurfaceState
from ._topology import _host_incidence, MultiRegionSurfaceTopology
from ._validation import _degenerate_faces, _edge_wedges, _vertex_stars


@dataclass(slots=True)
class _WorkingMesh:
    """Mutable capacity-sized host copy of one surface during one pass."""

    topology: MultiRegionSurfaceTopology
    capacity_plan: MultiRegionSurfaceCapacityPlan
    positions: np.ndarray
    vertex_ids: np.ndarray
    vertex_alive: np.ndarray
    fixed: np.ndarray
    faces: np.ndarray
    labels: np.ndarray
    face_ids: np.ndarray
    face_alive: np.ndarray
    vertex_count: int
    face_count: int
    region_finite: np.ndarray
    region_faces: np.ndarray
    next_vertex_id: int
    next_face_id: int
    vertex_faces: list[set[int]]
    mode: PredicateMode


def _working_mesh(
    topology: MultiRegionSurfaceTopology,
    state: MultiRegionSurfaceState,
    policy: SurfaceEventPolicy,
    /,
) -> _WorkingMesh:
    vertices, faces_count = topology.vertex_count, topology.face_count
    positions = np.zeros((topology.vertex_capacity, 3), dtype=np.float64)
    positions[:vertices] = np.asarray(state.positions[:vertices], dtype=np.float64)
    vertex_ids = np.full((topology.vertex_capacity,), -1, dtype=np.int64)
    vertex_ids[:vertices] = np.asarray(topology.vertex_global_ids[:vertices])
    faces = np.full((topology.face_capacity, 3), -1, dtype=np.int64)
    faces[:faces_count] = topology.host_faces()
    labels = np.full((topology.face_capacity, 2), -1, dtype=np.int64)
    labels[:faces_count] = topology.host_face_labels()
    face_ids = np.full((topology.face_capacity,), -1, dtype=np.int64)
    face_ids[:faces_count] = np.asarray(topology.face_global_ids[:faces_count])
    vertex_faces: list[set[int]] = [set() for _ in range(topology.vertex_capacity)]
    for face, row in enumerate(faces[:faces_count].tolist()):
        for vertex in row:
            vertex_faces[vertex].add(face)
    fixed = np.isin(vertex_ids, np.asarray(policy.fixed_vertex_ids, dtype=np.int64))
    region_faces = np.bincount(
        labels[:faces_count].reshape(-1), minlength=topology.region_capacity
    )
    return _WorkingMesh(
        capacity_plan=topology.plan,
        topology=topology,
        positions=positions,
        vertex_ids=vertex_ids,
        vertex_alive=np.arange(topology.vertex_capacity) < vertices,
        fixed=fixed & (vertex_ids >= 0),
        faces=faces,
        labels=labels,
        face_ids=face_ids,
        face_alive=np.arange(topology.face_capacity) < faces_count,
        vertex_count=vertices,
        face_count=faces_count,
        region_finite=np.asarray(topology.region_finite),
        region_faces=region_faces,
        next_vertex_id=int(np.max(vertex_ids[:vertices])) + 1,
        next_face_id=int(np.max(face_ids[:faces_count])) + 1,
        vertex_faces=vertex_faces,
        mode=resolve_host_predicate_mode(PredicateMode.EXACT),
    )


def _grown_capacity(current: int, required: int, /) -> int:
    return current if required <= current else max(required, 2 * current)


def _grow_surface_capacity(
    plan: MultiRegionSurfaceCapacityPlan,
    /,
    *,
    vertex_capacity: int | None = None,
    edge_capacity: int | None = None,
    face_capacity: int | None = None,
    region_capacity: int | None = None,
    region_pair_capacity: int | None = None,
    maximum_edge_valence: int | None = None,
    maximum_vertex_region_pairs: int | None = None,
) -> MultiRegionSurfaceCapacityPlan:
    required = (
        plan.vertex_capacity if vertex_capacity is None else vertex_capacity,
        plan.edge_capacity if edge_capacity is None else edge_capacity,
        plan.face_capacity if face_capacity is None else face_capacity,
        plan.region_capacity if region_capacity is None else region_capacity,
        plan.region_pair_capacity
        if region_pair_capacity is None
        else region_pair_capacity,
        plan.maximum_edge_valence
        if maximum_edge_valence is None
        else maximum_edge_valence,
        plan.maximum_vertex_region_pairs
        if maximum_vertex_region_pairs is None
        else maximum_vertex_region_pairs,
    )
    current = (
        plan.vertex_capacity,
        plan.edge_capacity,
        plan.face_capacity,
        plan.region_capacity,
        plan.region_pair_capacity,
        plan.maximum_edge_valence,
        plan.maximum_vertex_region_pairs,
    )
    grown = tuple(
        _grown_capacity(before, after)
        for before, after in zip(current, required, strict=True)
    )
    if grown == current:
        return plan
    index_dtype = plan.index_dtype
    if index_dtype == "int32" and max(grown[:3]) * max(grown[5], 3) >= 2**31:
        index_dtype = "int64"
    return MultiRegionSurfaceCapacityPlan(
        vertex_capacity=grown[0],
        edge_capacity=grown[1],
        face_capacity=grown[2],
        region_capacity=grown[3],
        region_pair_capacity=grown[4],
        maximum_edge_valence=grown[5],
        maximum_vertex_region_pairs=grown[6],
        resource_id=plan.resource_id,
        coordinate_dtype=plan.coordinate_dtype,
        index_dtype=index_dtype,
    )


def _ensure_edit_workspace(mesh: _WorkingMesh, edit: _Edit, /) -> None:
    """Grow private array shapes before preparing an edit."""
    plan = _grow_surface_capacity(
        mesh.capacity_plan,
        vertex_capacity=mesh.vertex_count + edit.new_positions.shape[0],
        face_capacity=mesh.face_count + edit.new_faces.shape[0],
    )
    mesh.capacity_plan = plan
    vertex_capacity, face_capacity = plan.vertex_capacity, plan.face_capacity
    if vertex_capacity > mesh.positions.shape[0]:
        old_capacity = mesh.positions.shape[0]
        positions = np.zeros((vertex_capacity, 3), dtype=mesh.positions.dtype)
        positions[:old_capacity] = mesh.positions
        vertex_ids = np.full((vertex_capacity,), -1, dtype=mesh.vertex_ids.dtype)
        vertex_ids[:old_capacity] = mesh.vertex_ids
        vertex_alive = np.zeros((vertex_capacity,), dtype=np.bool_)
        vertex_alive[:old_capacity] = mesh.vertex_alive
        fixed = np.zeros((vertex_capacity,), dtype=np.bool_)
        fixed[:old_capacity] = mesh.fixed
        mesh.positions = positions
        mesh.vertex_ids = vertex_ids
        mesh.vertex_alive = vertex_alive
        mesh.fixed = fixed
        mesh.vertex_faces.extend(set() for _ in range(vertex_capacity - old_capacity))
    if face_capacity > mesh.faces.shape[0]:
        old_capacity = mesh.faces.shape[0]
        faces = np.full((face_capacity, 3), -1, dtype=mesh.faces.dtype)
        faces[:old_capacity] = mesh.faces
        labels = np.full((face_capacity, 2), -1, dtype=mesh.labels.dtype)
        labels[:old_capacity] = mesh.labels
        face_ids = np.full((face_capacity,), -1, dtype=mesh.face_ids.dtype)
        face_ids[:old_capacity] = mesh.face_ids
        face_alive = np.zeros((face_capacity,), dtype=np.bool_)
        face_alive[:old_capacity] = mesh.face_alive
        mesh.faces = faces
        mesh.labels = labels
        mesh.face_ids = face_ids
        mesh.face_alive = face_alive


# ------------------------------------------------------------------ adjacency


def _star(mesh: _WorkingMesh, vertices: Iterable[int], /) -> np.ndarray:
    """Alive faces incident to any of ``vertices`` (sorted)."""
    faces: set[int] = set()
    for vertex in vertices:
        if 0 <= vertex < mesh.vertex_count:
            faces |= mesh.vertex_faces[vertex]
    return np.asarray(sorted(faces), dtype=np.int64)


def _neighbors(mesh: _WorkingMesh, vertex: int, /) -> set[int]:
    star = _star(mesh, (vertex,))
    return set(mesh.faces[star].reshape(-1).tolist()) - {vertex}


def _ring(mesh: _WorkingMesh, core: Iterable[int], depth: int, /) -> set[int]:
    reached = set(core)
    frontier = set(reached)
    for _ in range(depth):
        star = _star(mesh, frontier)
        grown = set(mesh.faces[star].reshape(-1).tolist())
        frontier = grown - reached
        reached |= grown
    return reached


def _edge_faces(mesh: _WorkingMesh, first: int, second: int, /) -> list[int]:
    return sorted(mesh.vertex_faces[first] & mesh.vertex_faces[second])


def _pair(mesh: _WorkingMesh, face: int, /) -> tuple[int, int]:
    left, right = (int(value) for value in mesh.labels[face])
    return (left, right) if left < right else (right, left)


def _is_feature_edge(mesh: _WorkingMesh, first: int, second: int, /) -> bool:
    """A junction, border or label-changing edge (not the interior of one sheet)."""
    faces = _edge_faces(mesh, first, second)
    return len(faces) != 2 or _pair(mesh, faces[0]) != _pair(mesh, faces[1])


def _vertex_rank(mesh: _WorkingMesh, vertex: int, /) -> int:
    """0 sheet interior, 1 on one feature curve, 2 feature corner, 3 fixed."""
    if bool(mesh.fixed[vertex]):
        return 3
    features = sum(
        _is_feature_edge(mesh, vertex, other) for other in _neighbors(mesh, vertex)
    )
    return 0 if features == 0 else 1 if features == 2 else 2


def _slots_of(mesh: _WorkingMesh, global_ids: Sequence[int], /) -> tuple[int, ...] | None:
    """Working slots of alive vertices with the given global ids (``None`` if any is absent)."""
    ids = mesh.vertex_ids[: mesh.vertex_count]
    alive = mesh.vertex_alive[: mesh.vertex_count]
    slots = []
    for value in global_ids:
        match = np.flatnonzero((ids == int(value)) & alive)
        if match.size != 1:
            return None
        slots.append(int(match[0]))
    return tuple(slots)


def _oriented_faces(mesh: _WorkingMesh, face: int, first: int, second: int, /) -> bool:
    """Whether ``face`` traverses the directed edge ``first -> second``."""
    row = mesh.faces[face].tolist()
    position = row.index(first)
    return row[(position + 1) % 3] == second


def _face_normal(points: np.ndarray, /) -> np.ndarray:
    return np.cross(
        points[..., 1, :] - points[..., 0, :], points[..., 2, :] - points[..., 0, :]
    )


# ----------------------------------------------------------------- the edit


@dataclass(frozen=True, slots=True)
class _CCDLeg:
    """One linear motion certified by inclusion CCD.

    ``keys`` name the leg vertices: working slots (``>= 0`` and below the
    working vertex count) are shared with the static environment when they do
    not move; provisional or virtual vertices use other keys. ``faces`` index
    ``keys``; ``exclusions`` are key pairs whose contact is the intent of the
    event (merging vertices).
    """

    keys: np.ndarray
    faces: np.ndarray
    start: np.ndarray
    end: np.ndarray
    exclusions: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class _Edit:
    """One drafted local topology change in working-slot space.

    New vertices occupy provisional slots ``vertex_count + i`` in order;
    ``new_faces`` may reference them. ``moved`` lists existing slots with their
    drafted positions. ``survivors`` records existing vertices that absorb
    removed ones (collapse), as ``(slot, parent slots)``. ``groups`` optionally
    overrides the default single transfer group as ``(removed or moved source
    face slots, new-face row indices)``.
    """

    kind: SurfaceEventKind
    removed_faces: tuple[int, ...]
    new_faces: np.ndarray
    new_labels: np.ndarray
    new_face_parents: tuple[tuple[int, ...], ...]
    removed_vertices: tuple[int, ...]
    new_positions: np.ndarray
    new_vertex_parents: tuple[tuple[int, ...], ...]
    moved: tuple[tuple[int, np.ndarray], ...]
    survivors: tuple[tuple[int, tuple[int, ...]], ...]
    legs: tuple[_CCDLeg, ...]
    groups: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] | None


def _new_edit(
    kind: SurfaceEventKind,
    /,
    *,
    removed_faces: Iterable[int],
    new_faces: Sequence[Sequence[int]],
    new_labels: Sequence[Sequence[int]],
    new_face_parents: Sequence[Sequence[int]],
    removed_vertices: Iterable[int] = (),
    new_positions: Sequence[np.ndarray] = (),
    new_vertex_parents: Sequence[Sequence[int]] = (),
    moved: Mapping[int, np.ndarray] | None = None,
    survivors: Sequence[tuple[int, Sequence[int]]] = (),
    legs: Sequence[_CCDLeg] = (),
    groups: Sequence[tuple[Sequence[int], Sequence[int]]] | None = None,
) -> _Edit:
    return _Edit(
        kind=kind,
        removed_faces=tuple(sorted({int(face) for face in removed_faces})),
        new_faces=np.asarray(new_faces, dtype=np.int64).reshape((-1, 3)),
        new_labels=np.asarray(new_labels, dtype=np.int64).reshape((-1, 2)),
        new_face_parents=tuple(
            tuple(int(p) for p in parents) for parents in new_face_parents
        ),
        removed_vertices=tuple(sorted({int(vertex) for vertex in removed_vertices})),
        new_positions=np.asarray(new_positions, dtype=np.float64).reshape((-1, 3)),
        new_vertex_parents=tuple(
            tuple(int(p) for p in parents) for parents in new_vertex_parents
        ),
        moved=tuple(
            (int(slot), np.asarray(position, dtype=np.float64))
            for slot, position in sorted((moved or {}).items())
        ),
        survivors=tuple(
            (int(slot), tuple(int(p) for p in parents)) for slot, parents in survivors
        ),
        legs=tuple(legs),
        groups=None
        if groups is None
        else tuple(
            (tuple(int(f) for f in source), tuple(int(t) for t in target))
            for source, target in groups
        ),
    )


def _with_positions(
    edit: _Edit, new_positions: np.ndarray, moved: Mapping[int, np.ndarray], /
) -> _Edit:
    return _Edit(
        kind=edit.kind,
        removed_faces=edit.removed_faces,
        new_faces=edit.new_faces,
        new_labels=edit.new_labels,
        new_face_parents=edit.new_face_parents,
        removed_vertices=edit.removed_vertices,
        new_positions=new_positions,
        new_vertex_parents=edit.new_vertex_parents,
        moved=tuple((int(slot), position) for slot, position in sorted(moved.items())),
        survivors=edit.survivors,
        legs=edit.legs,
        groups=edit.groups,
    )


def _edit_positions(mesh: _WorkingMesh, edit: _Edit, /) -> np.ndarray:
    """Working positions with the edit's moved and provisional vertices applied."""
    points = np.concatenate((mesh.positions[: mesh.vertex_count], edit.new_positions))
    for slot, position in edit.moved:
        points[slot] = position
    return points


@dataclass(frozen=True, slots=True)
class _Patch:
    """Faces of the event star before and after the edit (working-slot space)."""

    old_faces: np.ndarray
    old_labels: np.ndarray
    new_faces: np.ndarray
    new_labels: np.ndarray
    touched: np.ndarray
    changed_rows: np.ndarray
    kept_slots: np.ndarray


def _patch(mesh: _WorkingMesh, edit: _Edit, /) -> _Patch:
    """The complete stars of every touched vertex before and after the edit."""
    moved = [slot for slot, _ in edit.moved]
    touched = sorted(
        (set(edit.new_faces.reshape(-1).tolist()) | set(moved))
        - set(edit.removed_vertices)
    )
    removed = set(edit.removed_faces)
    star = _star(mesh, [slot for slot in touched if slot < mesh.vertex_count])
    kept = np.asarray(
        [face for face in star.tolist() if face not in removed], dtype=np.int64
    )
    old = np.asarray(sorted(set(star.tolist()) | removed), dtype=np.int64)
    new_faces = np.concatenate((mesh.faces[kept], edit.new_faces)).reshape((-1, 3))
    new_labels = np.concatenate((mesh.labels[kept], edit.new_labels)).reshape((-1, 2))
    moving = np.isin(mesh.faces[kept], np.asarray(moved, dtype=np.int64)).any(axis=1)
    changed = np.concatenate(
        (np.flatnonzero(moving), kept.size + np.arange(edit.new_faces.shape[0]))
    )
    return _Patch(
        old_faces=mesh.faces[old],
        old_labels=mesh.labels[old],
        new_faces=new_faces,
        new_labels=new_labels,
        touched=np.asarray(touched, dtype=np.int64),
        changed_rows=changed.astype(np.int64),
        kept_slots=kept,
    )


# ------------------------------------------------------------ volume restore


def _signed_contributions(
    points: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    reference: np.ndarray,
    rows: int,
    /,
) -> np.ndarray:
    corners = points[faces] - reference
    volume = np.sum(corners[:, 0] * np.cross(corners[:, 1], corners[:, 2]), axis=1) / 6.0
    result = np.zeros((rows,), dtype=np.float64)
    np.add.at(result, labels[:, 0], volume)
    np.add.at(result, labels[:, 1], -volume)
    return result


def _volume_jacobian(
    points: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    reference: np.ndarray,
    regions: np.ndarray,
    free: np.ndarray,
    /,
) -> np.ndarray:
    """``d V_r / d x_v`` of the given regions with respect to ``free`` vertices."""
    column = np.full((points.shape[0],), -1, dtype=np.int64)
    column[free] = np.arange(free.size)
    row = np.full((int(np.max(labels)) + 1,), -1, dtype=np.int64)
    row[regions] = np.arange(regions.size)
    jacobian = np.zeros((regions.size, free.size, 3), dtype=np.float64)
    corners = points[faces] - reference
    for corner in range(3):
        gradient = (
            np.cross(corners[:, (corner + 1) % 3], corners[:, (corner + 2) % 3]) / 6.0
        )
        target = column[faces[:, corner]]
        for side, sign in ((0, 1.0), (1, -1.0)):
            region = row[labels[:, side]]
            use = (target >= 0) & (region >= 0)
            np.add.at(jacobian, (region[use], target[use]), sign * gradient[use])
    return jacobian.reshape((regions.size, 3 * free.size))


def _restore_volumes(
    mesh: _WorkingMesh, edit: _Edit, policy: SurfaceEventPolicy, /
) -> tuple[_Edit, float, bool]:
    """Restore affected finite-region volumes until tolerance or nonprogress.

    The minimum-norm Newton solve starts on the event movers and widens to
    their free touched one-ring only when the core cannot span every affected
    volume constraint. A monotone residual line search prevents oscillation;
    failure reports the best finite residual reached, never a capped iterate.
    """
    maximum_residual = float(np.finfo(np.float64).max)
    patch = _patch(mesh, edit)
    points = _edit_positions(mesh, edit)
    reference = np.mean(points[patch.touched], axis=0)
    rows = mesh.region_finite.size
    base = _signed_contributions(
        mesh.positions, patch.old_faces, patch.old_labels, reference, rows
    )
    corners = points[patch.new_faces]
    scale = float(np.mean(np.linalg.norm(corners[:, (1, 2, 0)] - corners, axis=2)))
    volume_scale = scale * scale * scale
    if (
        not math.isfinite(scale)
        or scale <= 0.0
        or not math.isfinite(volume_scale)
        or volume_scale <= 0.0
    ):
        return edit, maximum_residual, False
    tolerance = policy.volume_tolerance * volume_scale

    def relative_residual(size: float, /) -> float:
        value = size / volume_scale
        return value if math.isfinite(value) else maximum_residual

    def defect(values: np.ndarray, /) -> np.ndarray:
        change = _signed_contributions(
            values, patch.new_faces, patch.new_labels, reference, rows
        )
        return np.where(mesh.region_finite, change - base, 0.0)

    def residual_size(values: np.ndarray, /) -> float:
        current = defect(values)
        if not np.all(np.isfinite(current)):
            return math.inf
        return float(np.max(np.abs(current)))

    initial_size = residual_size(points)
    if not math.isfinite(initial_size):
        return edit, maximum_residual, False
    if not policy.restore_region_volumes or initial_size <= tolerance:
        return edit, relative_residual(initial_size), True
    regions = np.flatnonzero(
        mesh.region_finite & np.isin(np.arange(rows), patch.new_labels.reshape(-1))
    )
    movers = sorted(
        {slot for slot, _ in edit.moved}
        | set(range(mesh.vertex_count, mesh.vertex_count + edit.new_positions.shape[0]))
    )
    core = np.asarray(
        [slot for slot in movers if slot >= mesh.vertex_count or not mesh.fixed[slot]],
        dtype=np.int64,
    )
    ring = np.asarray(
        sorted(
            set(core.tolist())
            | {
                int(slot)
                for slot in patch.touched.tolist()
                if slot < mesh.vertex_count and not mesh.fixed[slot]
            }
        ),
        dtype=np.int64,
    )
    best_values = points.copy()
    best_size = initial_size
    for free in (core, ring):
        if free.size == 0:
            continue
        values = best_values.copy()
        while True:
            current_size = residual_size(values)
            if current_size <= tolerance:
                moved = {slot: position for slot, position in edit.moved}
                for slot in free.tolist():
                    if slot < mesh.vertex_count:
                        moved[slot] = values[slot]
                restored = _with_positions(edit, values[mesh.vertex_count :], moved)
                return restored, relative_residual(current_size), True
            jacobian = _volume_jacobian(
                values, patch.new_faces, patch.new_labels, reference, regions, free
            )
            step, _, rank, _ = np.linalg.lstsq(
                jacobian, -defect(values)[regions], rcond=None
            )
            if rank < regions.size or not np.all(np.isfinite(step)):
                break
            delta = step.reshape((-1, 3))
            if not np.any(delta):
                break
            fraction = 1.0
            improved = False
            while True:
                moved_values = values[free] + fraction * delta
                if np.array_equal(moved_values, values[free]):
                    break
                if np.all(np.isfinite(moved_values)):
                    trial = values.copy()
                    trial[free] = moved_values
                    trial_size = residual_size(trial)
                    if trial_size < current_size:
                        values = trial
                        improved = True
                        if trial_size < best_size:
                            best_values = trial.copy()
                            best_size = trial_size
                        break
                fraction *= 0.5
                if fraction == 0.0:
                    break
            if not improved:
                break
    return edit, relative_residual(best_size), False


# ------------------------------------------------------------ local guards


def _local_intersections(
    points: np.ndarray, faces: np.ndarray, changed: np.ndarray, others: np.ndarray, /
) -> bool:
    """Exact welded intersection of changed faces against the given faces."""
    from ...meshing._audit_topology import _triangle_pairs_intersect

    if changed.size == 0 or others.size == 0:
        return False
    first = faces[changed]
    second = faces[others]
    low_a, high_a = np.min(points[first], axis=1), np.max(points[first], axis=1)
    low_b, high_b = np.min(points[second], axis=1), np.max(points[second], axis=1)
    overlap = np.all(
        (low_a[:, None, :] <= high_b[None, :, :])
        & (low_b[None, :, :] <= high_a[:, None, :]),
        axis=2,
    )
    left, right = np.nonzero(overlap)
    keep = changed[left] != others[right]
    left, right = changed[left[keep]], others[right[keep]]
    pairs = np.unique(np.sort(np.stack((left, right), axis=1), axis=1), axis=0)
    if pairs.size == 0:
        return False
    hit, certain = _triangle_pairs_intersect(
        points, faces[pairs[:, 0]], faces[pairs[:, 1]]
    )
    return bool(np.any(hit | ~certain))


def _duplicate_faces(mesh: _WorkingMesh, edit: _Edit, /) -> bool:
    """Whether the edited stars would contain the same triangle twice."""
    faces = _patch(mesh, edit).new_faces
    return np.unique(np.sort(faces, axis=1), axis=0).shape[0] != faces.shape[0]


def _local_guards(
    mesh: _WorkingMesh, edit: _Edit, policy: SurfaceEventPolicy, /
) -> SurfaceEventStatus:
    """Exact combinatorial and geometric guards of one drafted edit."""
    patch = _patch(mesh, edit)
    points = _edit_positions(mesh, edit)
    faces = patch.new_faces
    labels = patch.new_labels
    if _duplicate_faces(mesh, edit):
        return SurfaceEventStatus.DUPLICATE_FACE
    removed_regions = (
        np.bincount(
            mesh.labels[list(edit.removed_faces)].reshape(-1),
            minlength=mesh.region_faces.size,
        )
        if edit.removed_faces
        else np.zeros_like(mesh.region_faces)
    )
    added_regions = np.bincount(
        edit.new_labels.reshape(-1), minlength=mesh.region_faces.size
    )
    if np.any(
        (mesh.region_faces > 0)
        & (mesh.region_faces - removed_regions + added_regions <= 0)
    ):
        return SurfaceEventStatus.REGION_EXTINCTION
    changed = patch.changed_rows
    degenerate, certain = _degenerate_faces(points[faces[changed]], mesh.mode)
    if np.any(degenerate | ~certain):
        return SurfaceEventStatus.DEGENERATE_FACE
    local, inverse = np.unique(faces, return_inverse=True)
    local_faces = inverse.reshape(faces.shape)
    incidence = _host_incidence(local_faces, labels, local.size)
    touched = np.isin(local, patch.touched)
    rows = np.flatnonzero(np.any(touched[incidence.edges], axis=1))
    edges = incidence.edges[rows]
    edge_faces = incidence.edge_faces[rows]
    valence = np.sum(edge_faces >= 0, axis=1)
    mesh.capacity_plan = _grow_surface_capacity(
        mesh.capacity_plan,
        maximum_edge_valence=int(np.max(valence, initial=1)),
        maximum_vertex_region_pairs=int(
            np.max(incidence.vertex_slot_count[touched], initial=1)
        ),
    )
    remaining = set(faces.reshape(-1).tolist())
    if (
        remaining & set(edit.removed_vertices)
        or not {s for s, _ in edit.moved} <= remaining
    ):
        return SurfaceEventStatus.SUPPORT_INVALID
    wedges = _edge_wedges(
        edges,
        edge_faces,
        incidence.edge_face_signs[rows],
        mesh.region_finite,
        points[local],
        local_faces,
        labels,
        mesh.mode,
    )
    if not wedges.consistent:
        return SurfaceEventStatus.LABEL_ORIENTATION_INCONSISTENT
    match policy.validation.profile:
        case "general":
            dry, manifold = False, False
        case "dry_foam":
            dry, manifold = True, False
        case "manifold_two_region":
            dry, manifold = False, True
        case profile:
            raise ValueError(f"Unknown validation profile {profile!r}.")
    if dry and np.any(
        ~wedges.border & (valence != 2) & (valence != DRY_FOAM_EDGE_VALENCE)
    ):
        return SurfaceEventStatus.NONPHYSICAL_VALENCE
    if manifold and np.any(wedges.border | (valence != 2)):
        return SurfaceEventStatus.NONPHYSICAL_VALENCE
    border = np.zeros((incidence.edges.shape[0],), dtype=np.bool_)
    border[rows] = wedges.border
    stars = _vertex_stars(
        local_faces,
        labels,
        incidence.edge_faces,
        incidence.edges,
        border,
        local.size,
        mesh.region_finite.size,
    )
    if np.any(stars.singular[touched]):
        return SurfaceEventStatus.FAN_STRUCTURE_INVALID
    if dry or manifold:
        regions = np.bincount(
            np.unique(
                np.repeat(local_faces.reshape(-1), 2) * mesh.region_finite.size
                + np.repeat(labels, 3, axis=0).reshape(-1)
            )
            // mesh.region_finite.size,
            minlength=local.size,
        )
        interior = np.ones((local.size,), dtype=np.bool_)
        interior[incidence.edges[border].reshape(-1)] = False
        if dry and np.any(stars.incomplete[touched]):
            return SurfaceEventStatus.REGION_GRAPH_INCOMPLETE
        if dry and np.any(touched & interior & (regions > DRY_FOAM_VERTEX_REGIONS)):
            return SurfaceEventStatus.NONPHYSICAL_VALENCE
        if manifold and np.any(touched & (regions != 2)):
            return SurfaceEventStatus.NONPHYSICAL_VALENCE
    if _normal_rotation_exceeded(mesh, edit, patch, points, policy):
        return SurfaceEventStatus.NORMAL_INVERSION
    checked_environment = False
    for environment in _environment_faces(mesh, patch, edit, points, faces[changed]):
        checked_environment = True
        all_faces = np.concatenate((faces, mesh.faces[environment]))
        others = np.arange(all_faces.shape[0])
        if _local_intersections(points, all_faces, changed, others):
            return SurfaceEventStatus.LOCAL_INTERSECTION
    if not checked_environment and _local_intersections(
        points, faces, changed, np.arange(faces.shape[0])
    ):
        return SurfaceEventStatus.LOCAL_INTERSECTION
    return SurfaceEventStatus.ACCEPTED


def _normal_rotation_exceeded(
    mesh: _WorkingMesh,
    edit: _Edit,
    patch: _Patch,
    points: np.ndarray,
    policy: SurfaceEventPolicy,
    /,
) -> bool:
    """Surviving moved faces and single-parent reconnected faces may not fold."""
    limit = math.cos(policy.maximum_normal_rotation)
    moving = patch.changed_rows[patch.changed_rows < patch.kept_slots.size]
    before = [_face_normal(mesh.positions[mesh.faces[patch.kept_slots[moving]]])]
    after = [_face_normal(points[patch.new_faces[moving]])]
    for row, parents in enumerate(edit.new_face_parents):
        if len(parents) == 1 and np.array_equal(
            mesh.labels[parents[0]], edit.new_labels[row]
        ):
            before.append(_face_normal(mesh.positions[mesh.faces[parents[0]]])[None, :])
            after.append(_face_normal(points[edit.new_faces[row]])[None, :])
    old = np.concatenate(before)
    new = np.concatenate(after)
    if old.size == 0:
        return False
    norms = np.linalg.norm(old, axis=1) * np.linalg.norm(new, axis=1)
    cosine = np.sum(old * new, axis=1) / np.where(norms > 0.0, norms, 1.0)
    return bool(np.any((norms <= 0.0) | (cosine < limit)))


_CCD_FACE_SCAN_CHUNK = 8192


def _face_aabb_chunks(
    mesh: _WorkingMesh,
    low: np.ndarray,
    high: np.ndarray,
    excluded_faces: Iterable[int],
    /,
) -> Iterable[np.ndarray]:
    """Stream every alive face whose outward-rounded AABB meets the sweep."""
    lower = np.nextafter(np.asarray(low, dtype=np.float64), -np.inf)
    upper = np.nextafter(np.asarray(high, dtype=np.float64), np.inf)
    excluded = np.asarray(
        sorted({int(value) for value in excluded_faces}), dtype=np.int64
    )
    alive = np.asarray(mesh.face_alive[: mesh.face_count], dtype=np.bool_)
    finite_sweep = np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    for start in range(0, mesh.face_count, _CCD_FACE_SCAN_CHUNK):
        stop = min(start + _CCD_FACE_SCAN_CHUNK, mesh.face_count)
        rows = np.arange(start, stop, dtype=np.int64)
        rows = rows[alive[start:stop]]
        if not rows.size:
            continue
        corners = mesh.positions[mesh.faces[rows]]
        finite_faces = np.all(np.isfinite(corners), axis=(1, 2))
        face_lower = np.nextafter(np.min(corners, axis=1), -np.inf)
        face_upper = np.nextafter(np.max(corners, axis=1), np.inf)
        overlap = (
            (~finite_faces)
            | (~finite_sweep)
            | np.all((face_lower <= upper) & (face_upper >= lower), axis=1)
        )
        rows = rows[overlap]
        if excluded.size and rows.size:
            rows = rows[~np.isin(rows, excluded)]
        if rows.size:
            yield rows


def _environment_faces(
    mesh: _WorkingMesh,
    patch: _Patch,
    edit: _Edit,
    points: np.ndarray,
    moving: np.ndarray,
    /,
) -> Iterable[np.ndarray]:
    """Stream alive environment faces overlapping moved-face sweep bounds."""
    if moving.size == 0:
        return
    swept = points[moving].reshape((-1, 3))
    low = np.min(swept, axis=0)
    high = np.max(swept, axis=0)
    excluded = set(edit.removed_faces) | set(patch.kept_slots.tolist())
    yield from _face_aabb_chunks(mesh, low, high, excluded)


# --------------------------------------------------------------------- CCD


def _leg_environment(
    mesh: _WorkingMesh, edit: _Edit, leg: _CCDLeg, /
) -> Iterable[np.ndarray]:
    """Stream static faces in an outward-rounded swept box for one CCD leg."""
    swept = np.concatenate((leg.start, leg.end))
    low, high = np.min(swept, axis=0), np.max(swept, axis=0)
    moving_keys = leg.keys[np.any(leg.start != leg.end, axis=1)]
    moving = {int(key) for key in moving_keys.tolist() if 0 <= key < mesh.vertex_count}
    moving |= set(edit.removed_vertices)
    excluded = set(edit.removed_faces) | set(_star(mesh, moving).tolist())
    yield from _face_aabb_chunks(mesh, low, high, excluded)


_CCD_INITIAL_CANDIDATE_CAPACITY = 64


def _complete_ccd_epoch(
    scene: PreparedCollisionScene,
    start: np.ndarray,
    end: np.ndarray,
    activation_distance: float,
    /,
) -> ContactCandidateEpoch:
    """Reprepare a private route bucket until the full sweep is searchable."""
    from ...discretization.contact import SweepAndPruneContactSearchPlan
    from ...discretization.contact._search import ContactSearchStatus

    capacity = _CCD_INITIAL_CANDIDATE_CAPACITY
    while True:
        search = SweepAndPruneContactSearchPlan(
            edge_vertex_capacity=0,
            edge_edge_capacity=capacity,
            face_vertex_capacity=capacity,
            activation_distance=activation_distance,
        )
        epoch = search.build(scene, start, end_positions=end)
        if bool(epoch.successful):
            return epoch
        if int(epoch.status) != int(ContactSearchStatus.CANDIDATE_OVERFLOW):
            return epoch
        required = max(
            int(epoch.edge_edge.actual_count), int(epoch.face_vertex.actual_count)
        )
        if required <= capacity:
            return epoch
        capacity = required


def _certify_leg_environment(
    mesh: _WorkingMesh,
    leg: _CCDLeg,
    environment: np.ndarray,
    policy: SurfaceEventPolicy,
    /,
) -> tuple[bool, float]:
    """Certify one complete leg against one streamed static-face block."""
    from ...discretization.contact import (
        collision_free_step_limit,
        CollisionSurfacePlan,
        ContactPairPolicy,
        PreparedCollisionScene,
        PreparedCollisionSurface,
        selection_collision_operator,
    )

    static_leg = np.all(leg.start == leg.end, axis=1)
    shared = {
        int(key): index
        for index, key in enumerate(leg.keys.tolist())
        if 0 <= key < mesh.vertex_count and static_leg[index]
    }
    env_faces_global = np.asarray(environment, dtype=np.int64)
    env_vertices = sorted(
        set(mesh.faces[env_faces_global].reshape(-1).tolist()) - set(shared)
    )
    local = dict(shared)
    for offset, slot in enumerate(env_vertices):
        local[slot] = leg.keys.size + offset
    start = np.concatenate((leg.start, mesh.positions[env_vertices].reshape((-1, 3))))
    end = np.concatenate((leg.end, mesh.positions[env_vertices].reshape((-1, 3))))
    env_faces = (
        np.vectorize(local.__getitem__, otypes=[np.int64])(mesh.faces[env_faces_global])
        if env_faces_global.size
        else np.zeros((0, 3), dtype=np.int64)
    )
    faces = np.concatenate((leg.faces, env_faces.reshape((-1, 3))))
    count = start.shape[0]
    corners = start[faces]
    area = np.linalg.norm(_face_normal(corners), axis=1)
    edge_lengths = np.linalg.norm(corners[:, (1, 2, 0)] - corners, axis=2)
    if np.any(area <= 0.0) or np.any(edge_lengths <= 0.0):
        return False, 0.0
    key_index = {int(key): index for index, key in enumerate(leg.keys.tolist())}
    exclusions = np.asarray(
        [(key_index[a], key_index[b]) for a, b in leg.exclusions], dtype=np.int64
    ).reshape((-1, 2))
    static = np.all(start == end, axis=1)
    plan = CollisionSurfacePlan(
        np.arange(count, dtype=np.int64),
        ambient_dimension=3,
        faces=faces,
        static_mask=static,
        pair_policy=ContactPairPolicy(count, excluded_vertex_pairs=exclusions),
    )
    surface = PreparedCollisionSurface(
        plan,
        start,
        selection_collision_operator(
            ArraySpace((count, 3), dtype=np.float64), np.arange(count, dtype=np.int64)
        ),
    )
    scene = PreparedCollisionScene((surface,))
    swept = np.concatenate((leg.start, leg.end))
    span = float(np.max(np.max(swept, axis=0) - np.min(swept, axis=0)))
    epoch = _complete_ccd_epoch(
        scene,
        start,
        end,
        max(1.0e-12 * max(span, 1.0), 1.0e-300),
    )
    safety = collision_free_step_limit(policy.ccd, scene, epoch, start, end)
    certified = (
        bool(epoch.successful)
        and bool(safety.successful)
        and float(safety.step_size) >= 1.0
    )
    return certified, float(safety.minimum_time_of_impact)


def _certify_leg(
    mesh: _WorkingMesh, edit: _Edit, leg: _CCDLeg, policy: SurfaceEventPolicy, /
) -> tuple[bool, float]:
    """Stream all swept-environment blocks and certify the complete motion leg."""
    impact = 1.0
    saw_environment = False
    for environment in _leg_environment(mesh, edit, leg):
        saw_environment = True
        certified, time = _certify_leg_environment(mesh, leg, environment, policy)
        if np.isfinite(time):
            impact = min(impact, float(time))
        elif not certified:
            return False, 0.0
        if not certified:
            return False, impact
    if not saw_environment:
        return _certify_leg_environment(mesh, leg, np.zeros((0,), dtype=np.int64), policy)
    return True, impact


def _restoration_leg(
    mesh: _WorkingMesh, draft: _Edit, restored: _Edit, /
) -> _CCDLeg | None:
    """Candidate-topology motion from drafted to volume-restored positions."""
    before = _edit_positions(mesh, draft)
    after = _edit_positions(mesh, restored)
    moving = np.flatnonzero(np.any(before != after, axis=1))
    if moving.size == 0:
        return None
    patch = _patch(mesh, restored)
    new_rows = np.arange(patch.kept_slots.size, patch.new_faces.shape[0])
    moving_rows = np.flatnonzero(np.isin(patch.new_faces, moving).any(axis=1))
    faces = patch.new_faces[np.union1d(new_rows, moving_rows)]
    keys = np.unique(faces)
    return _CCDLeg(
        keys=keys,
        faces=np.searchsorted(keys, faces),
        start=before[keys],
        end=after[keys],
        exclusions=(),
    )


def _certify_legs(
    mesh: _WorkingMesh,
    edit: _Edit,
    legs: Sequence[_CCDLeg],
    policy: SurfaceEventPolicy,
    /,
) -> tuple[bool, float]:
    """All legs certified; returns the smallest time of impact (``1`` if none)."""
    impact = 1.0
    for leg in legs:
        certified, time = _certify_leg(mesh, edit, leg, policy)
        impact = min(impact, time)
        if not certified:
            return False, impact
    return True, impact


# ----------------------------------------------------------------- apply


def _apply_edit(
    mesh: _WorkingMesh, edit: _Edit, /
) -> tuple[list[int], list[int], list[int]]:
    """Commit an accepted edit; returns created vertex ids, created face ids and slots."""
    created_vertices = []
    base = mesh.vertex_count
    for offset, position in enumerate(edit.new_positions):
        slot = base + offset
        mesh.positions[slot] = position
        mesh.vertex_ids[slot] = mesh.next_vertex_id
        mesh.vertex_alive[slot] = True
        mesh.fixed[slot] = False
        created_vertices.append(mesh.next_vertex_id)
        mesh.next_vertex_id += 1
    mesh.vertex_count = base + edit.new_positions.shape[0]
    for slot, position in edit.moved:
        mesh.positions[slot] = position
    for face in edit.removed_faces:
        mesh.face_alive[face] = False
        for vertex in mesh.faces[face].tolist():
            mesh.vertex_faces[vertex].discard(face)
        mesh.region_faces[mesh.labels[face]] -= 1
    created_faces, created_slots = [], []
    for row, label in zip(edit.new_faces, edit.new_labels, strict=True):
        slot = mesh.face_count
        mesh.faces[slot] = row
        mesh.labels[slot] = label
        mesh.face_ids[slot] = mesh.next_face_id
        mesh.face_alive[slot] = True
        for vertex in row.tolist():
            mesh.vertex_faces[vertex].add(slot)
        mesh.region_faces[label] += 1
        created_faces.append(mesh.next_face_id)
        created_slots.append(slot)
        mesh.next_face_id += 1
        mesh.face_count += 1
    for vertex in edit.removed_vertices:
        mesh.vertex_alive[vertex] = False
    return created_vertices, created_faces, created_slots
