#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded host triangulation of padded planar facet loops using predicates."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .._geometry_predicates import (
    exact_charge,
    orient2d,
    orient3d,
    polygon_simplicity_2d,
    PolygonSimplicityStatus,
    PredicateMode,
    PredicateSign,
    resolve_host_predicate_mode,
)
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from . import (
    _exact_polyhedral_geometry as exact_polyhedral_geometry_owner,
    _planar_primitives as planar_primitives_owner,
)


_EAR_WORKING_ENTRIES = 1 << 20


def _ear_blocked(
    loops: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    alive: np.ndarray,
    predecessor: np.ndarray,
    successor: np.ndarray,
    sign: np.ndarray,
    mode: PredicateMode,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> np.ndarray:
    """Candidate ears containing another live vertex, or not decided exactly."""

    count, corner_count, _ = loops.shape
    apex = loops[:, :, None]
    left = before[:, :, None]
    right = after[:, :, None]
    other = loops[:, None, :]
    inside = np.ones((count, corner_count, corner_count), dtype=np.bool_)
    known = np.ones((count, corner_count, corner_count), dtype=np.bool_)
    for start, stop in ((left, apex), (apex, right), (right, left)):
        side = orient2d(start, stop, other, mode=mode, budget=budget)
        inside &= np.asarray(side.signs, dtype=np.int16) * sign[:, :, None] >= 0
        known &= np.asarray(side.certain)
    local = np.arange(corner_count)
    candidate = (
        alive[:, None, :]
        & (local[None, None, :] != local[None, :, None])
        & (local[None, None, :] != predecessor[:, :, None])
        & (local[None, None, :] != successor[:, :, None])
    )
    return np.any(candidate & (inside | ~known), axis=2)


def _ear_clip(
    loops: np.ndarray,
    orientation: np.ndarray,
    mode: PredicateMode,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate simple planar loops ``(count, n, 2)`` using exact predicates.

    An ear is a strictly convex live vertex whose closed triangle holds no other
    live vertex; simple loops always have one (Meisters' two-ears theorem).
    Returns local triangles ``(count, n - 2, 3)`` and whether every ear decision
    of a loop was certified; uncertified loops must be discarded.
    """

    count, corner_count, _ = loops.shape
    rows = np.arange(count)
    local = np.arange(corner_count)
    successor = np.tile((local + 1) % corner_count, (count, 1))
    predecessor = np.tile((local - 1) % corner_count, (count, 1))
    alive = np.ones((count, corner_count), dtype=np.bool_)
    certified = np.ones((count,), dtype=np.bool_)
    triangles = np.empty((count, corner_count - 2, 3), dtype=np.int64)
    sign = orientation.astype(np.int16)[:, None]
    for step in range(corner_count - 3):
        before = loops[rows[:, None], predecessor]
        after = loops[rows[:, None], successor]
        turn = orient2d(before, loops, after, mode=mode, budget=budget)
        convex = (
            alive
            & np.asarray(turn.certain)
            & (np.asarray(turn.signs, dtype=np.int16) * sign > 0)
        )
        ear = convex & ~_ear_blocked(
            loops,
            before,
            after,
            alive,
            predecessor,
            successor,
            sign,
            mode,
            budget=budget,
        )
        found = np.any(ear, axis=1)
        certified &= found
        choice = np.argmax(np.where(found[:, None], ear, alive), axis=1)
        left = predecessor[rows, choice]
        right = successor[rows, choice]
        triangles[:, step] = np.stack((left, choice, right), axis=1)
        alive[rows, choice] = False
        successor[rows, left] = right
        predecessor[rows, right] = left
    last = np.argmax(alive, axis=1)
    middle = successor[rows, last]
    triangles[:, -1] = np.stack((last, middle, successor[rows, middle]), axis=1)
    return triangles, certified


def polygon_loop_triangles(
    points: np.ndarray,
    loops: np.ndarray,
    candidate_capacity: int,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, bool]:
    """Return triangles and bounded simplicity evidence using exact predicates.

    Returns the triangles in global vertex rows, the loop row containing each
    triangle, masks of self-intersecting and undecided loops, the polygon edge
    candidates spent and whether that capacity was exhausted.
    """

    corner_count = loops.shape[1]
    mode = resolve_host_predicate_mode(PredicateMode.EXACT)
    corners = points[loops]
    centered = corners - np.mean(corners, axis=1, keepdims=True)
    normal = np.sum(np.cross(centered, np.roll(centered, -1, axis=1)), axis=1)
    axes = np.asarray(((1, 2), (0, 2), (0, 1)))[np.argmax(np.abs(normal), axis=1)]
    planar = np.take_along_axis(corners, axes[:, None, :], axis=2)
    simplicity = polygon_simplicity_2d(
        planar, mode=mode, maximum_candidate_pairs=candidate_capacity, budget=budget
    )
    status = np.asarray(simplicity.status)
    orientation = np.asarray(simplicity.orientation)
    ready = orientation != PredicateSign.UNCERTAIN
    triangles = []
    owners = []
    undecided = (status == PolygonSimplicityStatus.UNCERTAIN) | (
        (status == PolygonSimplicityStatus.SIMPLE) & ~ready
    )
    selected = np.flatnonzero(ready)
    chunk = max(1, _EAR_WORKING_ENTRIES // (corner_count * corner_count))
    for start in range(0, selected.size, chunk):
        rows = selected[start : start + chunk]
        local, certified = _ear_clip(planar[rows], orientation[rows], mode, budget=budget)
        undecided[rows[~certified]] = True
        global_rows = loops[rows[certified]]
        cells = np.arange(global_rows.shape[0])[:, None, None]
        triangles.append(global_rows[cells, local[certified]].reshape((-1, 3)))
        owners.append(np.repeat(rows[certified], corner_count - 2))
    return (
        np.concatenate(triangles) if triangles else np.empty((0, 3), dtype=np.int64),
        np.concatenate(owners) if owners else np.empty((0,), dtype=np.int64),
        status == PolygonSimplicityStatus.SELF_INTERSECTING,
        undecided,
        simplicity.candidate_pair_count,
        simplicity.candidate_capacity_exceeded,
    )


def _planar_loops(
    points: np.ndarray,
    loops: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    """Check padded 3-D loop coplanarity using exact predicates."""

    lengths = np.sum(loops >= 0, axis=1)
    nonplanar = np.zeros((loops.shape[0],), dtype=np.bool_)
    undecided = np.zeros((loops.shape[0],), dtype=np.bool_)
    for length in np.unique(lengths[lengths >= 4]):
        selected = np.flatnonzero(lengths == length)
        corners = points[loops[selected, :length]]
        # All tetrahedra with apex at corner zero are flat iff the loop is planar.
        for first, second, third in combinations(range(1, length), 3):
            mode = resolve_host_predicate_mode(PredicateMode.EXACT)
            predicate = orient3d(
                corners[:, 0],
                corners[:, first],
                corners[:, second],
                corners[:, third],
                mode=mode,
                budget=budget,
            )
            sign = np.asarray(predicate.signs, dtype=np.int16)
            certain = np.asarray(predicate.certain, dtype=np.bool_)
            nonplanar[selected] |= certain & (sign != 0)
            undecided[selected] |= ~certain
    return nonplanar, undecided & ~nonplanar


def loop_triangles(
    points: np.ndarray,
    loops: np.ndarray,
    capacity: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return planar 3-D loop triangles, owner rows, and failure masks.

    Returns ``(triangles, owners, nonplanar, undecided)``; nonplanar loops have a
    curved (mapped) boundary and self-intersecting loops count as nonplanar
    faces with no admissible affine triangulation.
    """

    if points.dtype == object:
        exact_triangles, exact_owners = [], []
        failed = np.zeros(loops.shape[0], dtype=np.bool_)
        exhausted = np.zeros_like(failed)
        used = 0
        for owner, row in enumerate(loops):
            indices = tuple(row[row >= 0].tolist())
            used += len(indices) ** 3
            if used > capacity:
                exhausted[owner:] = True
                break
            if len(indices) == 3:
                # A nondegenerate triangle already is its oriented
                # triangulation; retain the rational plane/rank check without
                # repeating it or preparing projected polygon ear clipping.

                corners = tuple(
                    tuple(value for value in points[index]) for index in indices
                )
                if planar_primitives_owner.plane_key(corners, budget=budget) is None:
                    failed[owner] = True
                    continue
                exact_triangles.append(indices)
                exact_owners.append(owner)
                continue
            # Charged outside the refusal handler: exhausting the request
            # ledger is a resource decision, never a failed facet.
            with exact_charge(
                *exact_polyhedral_geometry_owner.triangulation_charge(
                    points, indices, budget=budget
                ),
                budget=budget,
            ):
                try:
                    pieces = exact_polyhedral_geometry_owner.triangulate_loop(
                        points, indices, budget=budget
                    )
                except ValueError:
                    failed[owner] = True
                    continue
            exact_triangles.extend(pieces)
            exact_owners.extend((owner,) * len(pieces))
        return (
            np.asarray(exact_triangles, dtype=np.int64).reshape((-1, 3)),
            np.asarray(exact_owners, dtype=np.int64),
            failed,
            exhausted,
        )
    lengths = np.sum(loops >= 0, axis=1)
    nonplanar, undecided = _planar_loops(points, loops, budget=budget)
    triangles = [np.empty((0, 3), dtype=np.int64)]
    owners = [np.empty((0,), dtype=np.int64)]
    for length in np.unique(lengths):
        selected = np.flatnonzero((lengths == length) & ~nonplanar & ~undecided)
        rows = loops[selected, :length]
        if length == 3:
            triangles.append(rows)
            owners.append(selected)
            continue
        if not selected.size:
            continue
        clipped, local, crossing, unknown, _, exceeded = polygon_loop_triangles(
            points, rows, capacity, budget=budget
        )
        triangles.append(clipped)
        owners.append(selected[local])
        nonplanar[selected[crossing]] = True
        undecided[selected[unknown]] = True
        if exceeded:
            undecided[selected] = True
    return np.concatenate(triangles), np.concatenate(owners), nonplanar, undecided
