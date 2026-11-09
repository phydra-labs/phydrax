#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Certified common refinement (supermesh) of two cell meshes.

The common refinement of a source and a target mesh is the set of
positive-measure intersections of their cells.  Preparation is host-only NumPy
interchange in three certified phases:

1. Decomposition.  Triangles, tetrahedra, and convex polygons are single convex
   pieces.  Every other cell is the cone of its canonically triangulated
   boundary from a vertex or a certified interior apex.  Polyhedral faces are
   triangulated exactly as polyhedral finite-volume geometry integrates them
   (:func:`~phydrax.discretization.finite_volume._polyhedral.polyhedral_face_fan`:
   planar faces fan from their first vertex, warped faces from their vertex
   centroid), so decomposition measures and finite-volume cell volumes describe
   one geometry; other faces fan from their smallest mesh vertex.  Neighbors
   agree on shared faces either way.  A cone is accepted only when every
   nondegenerate cone simplex has one exact orientation: the signed cone
   indicators sum to the winding number of the boundary, so equal signs certify
   a partition of the cell.  Apexes are tried in ascending vertex order, then
   a rounded vertex centroid is tested if no vertex cone was certified.
2. Broad phase.  Packed float64 BVHs over the cell boxes enumerate every cell
   pair whose boxes overlap with positive extent, which any positive-measure
   intersection requires.
3. Narrow phase.  meshcore clips every piece pair with exactly classified
   vertices, in batches with a bounded working set, and the moments (optionally
   simplex partitions) are summed per cell pair.

Nothing is repaired: uncertified cells, unresolved predicates, native item
failures, coverage gaps, and double coverage are reported through
:class:`CommonRefinementStatus` with evidence.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Any, NamedTuple, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from .. import _meshcore
from .._bvh import bvh_overlap_pair_blocks, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._geometry_predicates import _orient3d_kernel, orient2d, orient3d, PredicateMode
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..ein import contract
from ._exact_polyhedral_geometry import determinant3
from ._rational import rational_vector


if TYPE_CHECKING:
    from ..discretization._cell_complex import PolyhedralConnectivity
    from ..discretization._cell_mesh import CellMesh


# Narrow-phase working set per native batch (inputs plus outputs).
_BATCH_BYTES = 64 * 1024 * 1024
# Simplex capacity of one tetrahedron-pair intersection (2 V - 4 with V <= 12).
_TETRAHEDRON_SIMPLICES = 20
# Coverage roundoff floor in units of eps * diameter**dimension of the cell box:
# measures are computed relative to local origins, so their absolute rounding
# scales with the cell size, not with its (possibly sliver) measure.
_COVERAGE_ROUNDOFF_ULPS = 256.0


class CommonRefinementStatus(IntEnum):
    """Fail-closed outcome of common-refinement preparation."""

    SUCCESS = 0
    INVALID_GEOMETRY = 1
    PREDICATE_UNCERTAIN = 2
    INTERSECTION_FAILURE = 3
    DOUBLE_COVERAGE = 4
    COVERAGE_GAP = 5


class CommonRefinementCoverage(StrEnum):
    """Which cell sets must be covered completely by the other mesh.

    Over-coverage (double coverage) fails for every requirement.
    """

    COMPLETE = "complete"
    TARGET = "target"
    SOURCE = "source"
    PARTIAL = "partial"


class CommonRefinementPolicy(StrictModule):
    """Predicate and physical-coverage requirements of a common refinement.

    ``coverage_tolerance`` is relative to the certified cell measure, with a
    roundoff floor proportional to the cell-box diameter raised to the dimension.
    Batch size is private scheduling and cannot change accepted problem sizes.
    For 3D second moments alone, the symbolic intersections are integrated
    directly. When overlap simplices are requested, all moments instead belong
    to their rounded carriers, and unrepresentable carriers are refused.
    """

    predicate_mode: PredicateMode = eqx.field(static=True)
    coverage: CommonRefinementCoverage = eqx.field(static=True)
    coverage_tolerance: float = eqx.field(static=True)
    second_moments: bool = eqx.field(static=True)
    overlap_simplices: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        predicate_mode: PredicateMode = PredicateMode.EXACT,
        coverage: CommonRefinementCoverage = CommonRefinementCoverage.COMPLETE,
        coverage_tolerance: float = 1.0e-10,
        second_moments: bool = False,
        overlap_simplices: bool = False,
    ) -> None:
        if not isinstance(predicate_mode, PredicateMode):
            raise TypeError("predicate_mode must be a PredicateMode.")
        if predicate_mode is PredicateMode.FILTERED_DEVICE:
            raise ValueError(
                "Common refinement certifies on the host: use FILTERED or EXACT."
            )
        if not isinstance(coverage, CommonRefinementCoverage):
            raise TypeError("coverage must be a CommonRefinementCoverage.")
        if not isinstance(second_moments, bool) or not isinstance(
            overlap_simplices, bool
        ):
            raise TypeError("second_moments and overlap_simplices must be bools.")
        tolerance = float(coverage_tolerance)
        if not np.isfinite(tolerance) or not 0.0 <= tolerance < 1.0:
            raise ValueError("coverage_tolerance must be finite and in [0, 1).")
        self.predicate_mode = predicate_mode
        self.coverage = coverage
        self.coverage_tolerance = tolerance
        self.second_moments = second_moments
        self.overlap_simplices = overlap_simplices
        self.policy_id = canonical_fingerprint(
            {
                "kind": "common-refinement-policy",
                "predicate_mode": predicate_mode.value,
                "coverage": coverage.value,
                "coverage_tolerance": tolerance,
                "second_moments": second_moments,
                "overlap_simplices": overlap_simplices,
            }
        )


class CommonRefinementEvidence(StrictModule, NonTrainableState):
    """Coverage defects, required tolerances, and unresolved geometry findings."""

    status: CommonRefinementStatus = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    source_coverage_defects: Array
    target_coverage_defects: Array
    source_coverage_tolerances: Array
    target_coverage_tolerances: Array
    source_gap_count: int = eqx.field(static=True)
    target_gap_count: int = eqx.field(static=True)
    source_double_count: int = eqx.field(static=True)
    target_double_count: int = eqx.field(static=True)
    maximum_relative_source_defect: float = eqx.field(static=True)
    maximum_relative_target_defect: float = eqx.field(static=True)
    uncertain_predicate_count: int = eqx.field(static=True)
    invalid_cell_count: int = eqx.field(static=True)
    intersection_failure_count: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        status: CommonRefinementStatus,
        reason: str,
        source_coverage_defects: np.ndarray,
        target_coverage_defects: np.ndarray,
        source_measures: np.ndarray,
        target_measures: np.ndarray,
        source_tolerances: np.ndarray,
        target_tolerances: np.ndarray,
        counts: _Counts,
    ) -> None:
        if not isinstance(status, CommonRefinementStatus):
            raise TypeError("status must be a CommonRefinementStatus.")
        source_defects = np.asarray(source_coverage_defects, dtype=np.float64)
        target_defects = np.asarray(target_coverage_defects, dtype=np.float64)
        source_tolerance = np.asarray(source_tolerances, dtype=np.float64)
        target_tolerance = np.asarray(target_tolerances, dtype=np.float64)
        if (
            source_defects.shape != source_measures.shape
            or target_defects.shape != target_measures.shape
            or source_tolerance.shape != source_measures.shape
            or target_tolerance.shape != target_measures.shape
        ):
            raise ValueError("Coverage defects must align with cell measures.")
        self.status = status
        self.reason = str(reason)
        self.source_coverage_defects = jnp.asarray(source_defects)
        self.target_coverage_defects = jnp.asarray(target_defects)
        self.source_coverage_tolerances = jnp.asarray(source_tolerance)
        self.target_coverage_tolerances = jnp.asarray(target_tolerance)
        self.source_gap_count = int(np.sum(-source_defects > source_tolerance))
        self.target_gap_count = int(np.sum(-target_defects > target_tolerance))
        self.source_double_count = int(np.sum(source_defects > source_tolerance))
        self.target_double_count = int(np.sum(target_defects > target_tolerance))
        self.maximum_relative_source_defect = _relative_maximum(
            source_defects, source_measures
        )
        self.maximum_relative_target_defect = _relative_maximum(
            target_defects, target_measures
        )
        self.uncertain_predicate_count = counts.uncertain
        self.invalid_cell_count = counts.invalid
        self.intersection_failure_count = counts.failures
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "common-refinement-evidence",
                "status": int(status),
                "reason": self.reason,
                "source_defects": array_tree_fingerprint(source_defects),
                "target_defects": array_tree_fingerprint(target_defects),
                "source_tolerances": array_tree_fingerprint(source_tolerance),
                "target_tolerances": array_tree_fingerprint(target_tolerance),
                "unresolved": (
                    self.uncertain_predicate_count,
                    self.invalid_cell_count,
                    self.intersection_failure_count,
                ),
            }
        )


def _relative_maximum(defects: np.ndarray, measures: np.ndarray, /) -> float:
    # Uncertified cells carry no pieces and hence no measure.
    measured = measures > 0.0
    if not np.any(measured):
        return 0.0
    return float(np.max(np.abs(defects[measured]) / measures[measured]))


class PreparedCommonRefinement(StrictModule, NonTrainableState):
    """CSR common refinement grouped by target cell.

    Entry ``k`` of row ``t`` (``target_offsets[t] <= k < target_offsets[t + 1]``)
    is the positive-measure overlap of source cell ``source_cells[k]`` with
    target cell ``t``, with measure ``volumes[k]``, first moment
    ``first_moments[k]`` (the integral of ``x``, i.e. centroid times measure),
    optional second moment ``second_moments[k]`` (the integral of ``x x^T``), and
    optional simplex partition ``simplices[simplex_offsets[k]:simplex_offsets[k +
    1]]`` whose signed measures sum to ``volumes[k]``.  Rows list source cells in
    ascending order.  Cells are indexed by their position in the concatenated
    mesh blocks.  Entries exist whenever the narrow phase ran; consumers must
    check :attr:`status`.
    """

    status: CommonRefinementStatus = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    source_cell_count: int = eqx.field(static=True)
    target_cell_count: int = eqx.field(static=True)
    target_offsets: Array
    target_cells: Array
    source_cells: Array
    volumes: Array
    first_moments: Array
    second_moments: Array | None
    simplex_offsets: Array | None
    simplices: Array | None
    source_measures: Array
    source_first_moments: Array
    target_measures: Array
    target_first_moments: Array
    source_cell_global_ids: Array
    target_cell_global_ids: Array
    source_mesh_id: str = eqx.field(static=True)
    target_mesh_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    policy: CommonRefinementPolicy
    evidence: CommonRefinementEvidence
    refinement_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        source: _Decomposition,
        target: _Decomposition,
        entries: _Entries,
        identities: tuple[str, str, str, str],
        policy: CommonRefinementPolicy,
        evidence: CommonRefinementEvidence,
    ) -> None:
        if not isinstance(policy, CommonRefinementPolicy) or not isinstance(
            evidence, CommonRefinementEvidence
        ):
            raise TypeError("policy and evidence must be common-refinement types.")
        dimension = source.first_moments.shape[1]
        count = entries.volumes.shape[0]
        offsets = np.zeros((target.cell_count + 1,), dtype=np.int32)
        np.cumsum(
            np.bincount(entries.target_cells, minlength=target.cell_count),
            out=offsets[1:],
        )
        if (
            entries.source_cells.shape != (count,)
            or entries.target_cells.shape != (count,)
            or entries.first_moments.shape != (count, dimension)
            or np.any(np.diff(entries.target_cells) < 0)
        ):
            raise ValueError("Common-refinement entries must be target-grouped CSR.")
        self.status = evidence.status
        self.dimension = dimension
        self.source_cell_count = source.cell_count
        self.target_cell_count = target.cell_count
        self.target_offsets = jnp.asarray(offsets)
        self.target_cells = jnp.asarray(entries.target_cells, dtype=jnp.int32)
        self.source_cells = jnp.asarray(entries.source_cells, dtype=jnp.int32)
        self.volumes = jnp.asarray(entries.volumes, dtype=jnp.float64)
        self.first_moments = jnp.asarray(entries.first_moments, dtype=jnp.float64)
        self.second_moments = (
            None
            if entries.second_moments is None
            else jnp.asarray(entries.second_moments, dtype=jnp.float64)
        )
        self.simplex_offsets = (
            None
            if entries.simplex_offsets is None
            else jnp.asarray(entries.simplex_offsets, dtype=jnp.int32)
        )
        self.simplices = (
            None
            if entries.simplices is None
            else jnp.asarray(entries.simplices, dtype=jnp.float64)
        )
        self.source_measures = jnp.asarray(source.measures)
        self.source_first_moments = jnp.asarray(source.first_moments)
        self.target_measures = jnp.asarray(target.measures)
        self.target_first_moments = jnp.asarray(target.first_moments)
        self.source_cell_global_ids = jnp.asarray(source.global_ids, dtype=jnp.int64)
        self.target_cell_global_ids = jnp.asarray(target.global_ids, dtype=jnp.int64)
        (
            self.source_mesh_id,
            self.target_mesh_id,
            self.source_topology_id,
            self.target_topology_id,
        ) = identities
        self.policy = policy
        self.evidence = evidence
        self.refinement_id = canonical_fingerprint(
            {
                "kind": "prepared-common-refinement",
                "source_mesh": identities[0],
                "target_mesh": identities[1],
                "policy": policy.policy_id,
                "evidence": evidence.evidence_id,
                "target_offsets": array_tree_fingerprint(offsets),
                "source_cells": array_tree_fingerprint(entries.source_cells),
                "volumes": array_tree_fingerprint(entries.volumes),
                "first_moments": array_tree_fingerprint(entries.first_moments),
            }
        )

    @property
    def succeeded(self) -> bool:
        return self.status is CommonRefinementStatus.SUCCESS

    @property
    def entry_count(self) -> int:
        return self.volumes.shape[0]


# ---------------------------------------------------------------- host records


class _Counts(NamedTuple):
    candidates: int
    accepted: int
    piece_pairs: int
    uncertain: int
    invalid: int
    failures: int


class _Decomposition(NamedTuple):
    """Certified convex pieces of every cell of one mesh (host arrays)."""

    cell_count: int
    global_ids: np.ndarray
    # 2-D: (P, K, 2) padded convex polygons with `piece_counts`; 3-D: (P, 4, 3).
    pieces: np.ndarray
    piece_counts: np.ndarray | None
    piece_offsets: np.ndarray
    measures: np.ndarray
    first_moments: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    invalid: int
    uncertain: int


class _Entries(NamedTuple):
    target_cells: np.ndarray
    source_cells: np.ndarray
    volumes: np.ndarray
    first_moments: np.ndarray
    second_moments: np.ndarray | None
    simplex_offsets: np.ndarray | None
    simplices: np.ndarray | None


class _PieceSigns(NamedTuple):
    signs: np.ndarray
    certain: np.ndarray


# ---------------------------------------------------------------- decomposition


def _orientation(points: tuple[np.ndarray, ...], mode: PredicateMode, /) -> _PieceSigns:
    match len(points):
        case 3:
            result = orient2d(*points, mode=mode)
        case 4:
            result = orient3d(*points, mode=mode)
        case _:
            raise ValueError("Orientation needs three or four points.")
    return _PieceSigns(np.asarray(result.signs), np.asarray(result.certain))


def _cone_decomposition(
    coordinates: np.ndarray,
    facets: np.ndarray,
    facet_valid: np.ndarray,
    candidates: np.ndarray,
    candidate_valid: np.ndarray,
    mode: PredicateMode,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Certify cones of oriented boundary facets ``(n, F, d)`` from candidate apexes.

    Returns the chosen apex per cell, the kept (nondegenerate) facet mask, and
    the invalid and uncertain cell masks.  Candidates are tried in column order;
    facets containing the apex are degenerate and skipped.
    """

    cells = facets.shape[0]
    apex = np.full((cells,), -1, dtype=np.int64)
    keep = np.zeros(facets.shape[:2], dtype=np.bool_)
    uncertain_seen = np.zeros((cells,), dtype=np.bool_)
    unresolved = np.arange(cells)
    # Bounded static loop over the candidate width (cell vertex count).
    for column in range(candidates.shape[1]):
        if unresolved.size == 0:
            break
        tip = candidates[unresolved, column]
        local = facets[unresolved]
        live = facet_valid[unresolved] & ~np.any(local == tip[:, None, None], axis=2)
        rows, slots = np.nonzero(live)
        signs = np.zeros(live.shape, dtype=np.int8)
        certain = np.ones(live.shape, dtype=np.bool_)
        if rows.size:
            evaluated = _orientation(
                (coordinates[tip[rows]],)
                + tuple(
                    coordinates[local[rows, slots, k]] for k in range(facets.shape[2])
                ),
                mode,
            )
            signs[rows, slots] = evaluated.signs
            certain[rows, slots] = evaluated.certain
        known = live & certain
        positive = np.any(known & (signs == 1), axis=1)
        negative = np.any(known & (signs == -1), axis=1)
        undecided = np.any(live & ~certain, axis=1)
        conflict = positive & negative
        accepted = (
            candidate_valid[unresolved, column]
            & ~conflict
            & ~undecided
            & (positive | negative)
        )
        uncertain_seen[unresolved] |= undecided & ~conflict
        chosen = unresolved[accepted]
        apex[chosen] = tip[accepted]
        keep[chosen] = known[accepted] & (signs[accepted] != 0)
        unresolved = unresolved[~accepted]
    failed = np.zeros((cells,), dtype=np.bool_)
    failed[unresolved] = True
    return apex, keep, failed & ~uncertain_seen, failed & uncertain_seen


def _polygon_moments(
    vertices: np.ndarray, counts: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Unsigned area and first moment of padded convex polygons ``(P, K, 2)``."""

    width = vertices.shape[1]
    slots = np.arange(width)
    present = slots[None, :] < counts[:, None]
    origin = np.sum(np.where(present[..., None], vertices, 0.0), axis=1) / counts[
        :, None
    ].astype(np.float64)
    relative = vertices - origin[:, None, :]
    following = np.take_along_axis(
        relative, ((slots[None, :] + 1) % counts[:, None])[..., None], axis=1
    )
    cross = np.where(
        present,
        relative[..., 0] * following[..., 1] - following[..., 0] * relative[..., 1],
        0.0,
    )
    area = 0.5 * np.sum(cross, axis=1)
    moment = (
        np.sum((relative + following) * cross[..., None], axis=1) / 6.0
        + origin * area[:, None]
    )
    sign = np.sign(area)
    return np.abs(area), moment * sign[:, None]


def _tetrahedron_moments(tetrahedra: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    edges = tetrahedra[:, 1:] - tetrahedra[:, :1]
    six = contract("ij,ij->i", edges[:, 0], np.cross(edges[:, 1], edges[:, 2]))
    volume = np.abs(six) / 6.0
    return volume, volume[:, None] * np.mean(tetrahedra, axis=1)


def _cell_bounds(
    coordinates: np.ndarray, cells: np.ndarray, valid: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    points = coordinates[np.where(valid, cells, cells[:, :1])]
    return np.min(points, axis=1), np.max(points, axis=1)


class _BlockPieces(NamedTuple):
    cells: np.ndarray
    vertices: np.ndarray
    counts: np.ndarray | None
    invalid: np.ndarray
    uncertain: np.ndarray


def _polygon_block_pieces(
    coordinates: np.ndarray, cells: np.ndarray, mode: PredicateMode, /
) -> _BlockPieces:
    """Convex cells stay whole; other polygons become certified triangle cones."""

    count, arity = cells.shape
    previous = np.roll(cells, 1, axis=1)
    following = np.roll(cells, -1, axis=1)
    turns = _orientation(
        (coordinates[previous], coordinates[cells], coordinates[following]), mode
    )
    known = turns.certain
    positive = np.any(known & (turns.signs == 1), axis=1)
    negative = np.any(known & (turns.signs == -1), axis=1)
    undecided = np.any(~known, axis=1)
    convex = (positive ^ negative) & ~undecided
    rows = np.arange(count)
    whole = rows[convex]
    concave = rows[~convex]
    edges = np.stack((cells[concave], np.roll(cells[concave], -1, axis=1)), axis=2)
    apex, keep, invalid, uncertain = _cone_decomposition(
        coordinates,
        edges,
        np.ones(edges.shape[:2], dtype=np.bool_),
        np.sort(cells[concave], axis=1),
        np.ones((concave.size, arity), dtype=np.bool_),
        mode,
    )
    cone_rows, cone_slots = np.nonzero(keep)
    triangles = np.stack(
        (
            apex[cone_rows],
            edges[cone_rows, cone_slots, 0],
            edges[cone_rows, cone_slots, 1],
        ),
        axis=1,
    )
    width = max(arity, 3)
    padded = np.zeros((whole.size + triangles.shape[0], width, 2), dtype=np.float64)
    padded[: whole.size, :arity] = coordinates[cells[whole]]
    padded[whole.size :, :3] = coordinates[triangles]
    piece_cells = np.concatenate((whole, concave[cone_rows]))
    piece_counts = np.concatenate(
        (
            np.full((whole.size,), arity, dtype=np.int32),
            np.full((triangles.shape[0],), 3, dtype=np.int32),
        )
    )
    failed_invalid = np.zeros((count,), dtype=np.bool_)
    failed_uncertain = np.zeros((count,), dtype=np.bool_)
    failed_invalid[concave] = invalid
    failed_uncertain[concave] = uncertain
    return _BlockPieces(
        piece_cells, padded, piece_counts, failed_invalid, failed_uncertain
    )


def _standard_face_triangles(cells: np.ndarray, kind: str, /) -> np.ndarray:
    """Oriented face triangles ``(n, T, 3)`` of standard volume cells."""

    from ..discretization._cell_complex import _POLYHEDRAL_FACE_ROUTES

    triangles = []
    for route in _POLYHEDRAL_FACE_ROUTES[kind]:
        loops = cells[:, route]
        size = len(route)
        start = np.argmin(loops, axis=1)
        rotated = np.take_along_axis(
            loops, (start[:, None] + np.arange(size)) % size, axis=1
        )
        triangles.append(
            np.stack(
                (
                    np.repeat(rotated[:, :1], size - 2, axis=1),
                    rotated[:, 1:-1],
                    rotated[:, 2:],
                ),
                axis=2,
            )
        )
    return np.concatenate(triangles, axis=1)


class _PolyhedralSurface(NamedTuple):
    """Finite-volume face triangles of a polyhedral cell complex.

    ``coordinates`` are the mesh vertices followed by the vertex centroids of
    warped faces; ``triangles[offsets[f]:offsets[f + 1]]`` index them in the
    orientation of face loop ``f`` of ``connectivity``.  ``failed`` marks faces
    without admissible finite-volume triangles.
    """

    connectivity: PolyhedralConnectivity
    coordinates: np.ndarray
    triangles: np.ndarray
    offsets: np.ndarray
    failed: np.ndarray


def _polyhedral_surface(
    connectivity: PolyhedralConnectivity, coordinates: np.ndarray, /
) -> _PolyhedralSurface:
    from ..discretization.finite_volume._polyhedral import (
        polyhedral_face_fan,
        POLYHEDRAL_PLANARITY_TOLERANCE,
    )

    face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    face_values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
    face_count = face_offsets.size - 1
    vertex_count = coordinates.shape[0]
    apexes: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    counts = np.zeros((face_count,), dtype=np.int64)
    failed = np.zeros((face_count,), dtype=np.bool_)
    for face in range(face_count):
        loop = face_values[face_offsets[face] : face_offsets[face + 1]]
        fan = polyhedral_face_fan(coordinates[loop], POLYHEDRAL_PLANARITY_TOLERANCE)
        if fan.failure:
            failed[face] = True
            continue
        if fan.apex is not None:
            # Corner -1 of a warped fan names this appended vertex centroid.
            loop = np.append(loop, vertex_count + len(apexes))
            apexes.append(fan.apex)
        rows.append(loop[np.asarray(fan.corners, dtype=np.int64)])
        counts[face] = len(fan.corners)
    return _PolyhedralSurface(
        connectivity,
        np.concatenate(
            (coordinates, np.asarray(apexes, dtype=np.float64).reshape((-1, 3)))
        ),
        np.concatenate(rows) if rows else np.zeros((0, 3), dtype=np.int64),
        _offsets(counts),
        failed,
    )


def _polyhedral_face_triangles(
    surface: _PolyhedralSurface, first_cell: int, cell_count: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell-oriented face triangles, their mask, and cells with failed faces."""

    connectivity = surface.connectivity
    cell_face_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    begin = cell_face_offsets[first_cell]
    end = cell_face_offsets[first_cell + cell_count]
    faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)[begin:end]
    signs = np.asarray(connectivity.cell_face_sign_values)[begin:end]
    owner = np.repeat(
        np.arange(cell_count),
        np.diff(cell_face_offsets[first_cell : first_cell + cell_count + 1]),
    )
    sizes = surface.offsets[faces + 1] - surface.offsets[faces]
    incidence = np.repeat(np.arange(faces.size), sizes)
    local = np.arange(incidence.size) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    triangles = surface.triangles[surface.offsets[faces][incidence] + local]
    reverse = signs[incidence] < 0
    triangles[reverse] = triangles[reverse][:, [0, 2, 1]]
    triangle_cells = owner[incidence]
    per_cell = np.bincount(triangle_cells, minlength=cell_count)
    width = int(np.max(per_cell, initial=0))
    slot = np.arange(triangle_cells.size) - np.repeat(
        np.cumsum(per_cell) - per_cell, per_cell
    )
    padded = np.zeros((cell_count, width, 3), dtype=np.int64)
    valid = np.zeros((cell_count, width), dtype=np.bool_)
    padded[triangle_cells, slot] = triangles
    valid[triangle_cells, slot] = True
    failed = np.zeros((cell_count,), dtype=np.bool_)
    failed[owner[surface.failed[faces]]] = True
    return padded, valid, failed


def _volume_block_pieces(
    coordinates: np.ndarray,
    cells: np.ndarray,
    vertex_valid: np.ndarray,
    triangles: np.ndarray,
    triangle_valid: np.ndarray,
    mode: PredicateMode,
    /,
) -> _BlockPieces:
    ordered = np.sort(np.where(vertex_valid, cells, np.iinfo(np.int64).max), axis=1)
    candidate_valid = ordered != np.iinfo(np.int64).max
    apex, keep, invalid, uncertain = _cone_decomposition(
        coordinates,
        triangles,
        triangle_valid,
        np.where(candidate_valid, ordered, 0),
        candidate_valid,
        mode,
    )
    unresolved = np.flatnonzero(invalid | uncertain)
    if unresolved.size:
        # Warped face fans can have a nonempty kernel containing no mesh vertex.
        # A centroid is only a candidate: the same orientation proof must certify
        # its cones. Existing vertex cones and their reduction order are retained.
        local_valid = candidate_valid[unresolved]
        local = coordinates[np.where(local_valid, ordered[unresolved], 0)]
        origin = local[:, 0]
        centers = (
            origin
            + np.sum(
                np.where(local_valid[..., None], local - origin[:, None, :], 0.0),
                axis=1,
            )
            / np.sum(local_valid, axis=1)[:, None]
        )
        center_ids = coordinates.shape[0] + np.arange(unresolved.size)
        coordinates = np.concatenate((coordinates, centers), axis=0)
        center_apex, center_keep, center_invalid, center_uncertain = _cone_decomposition(
            coordinates,
            triangles[unresolved],
            triangle_valid[unresolved],
            center_ids[:, None],
            np.ones((unresolved.size, 1), dtype=np.bool_),
            mode,
        )
        previous_uncertain = uncertain[unresolved]
        apex[unresolved] = center_apex
        keep[unresolved] = center_keep
        invalid[unresolved] = center_invalid & ~previous_uncertain
        uncertain[unresolved] = center_uncertain | (center_invalid & previous_uncertain)
    rows, slots = np.nonzero(keep)
    tetrahedra = np.concatenate((apex[rows, None], triangles[rows, slots]), axis=1)
    return _BlockPieces(rows, coordinates[tetrahedra], None, invalid, uncertain)


def _block_family(kind: str, /) -> str:
    """Name the standard family whose canonical reference order ``kind`` shares.

    Dimension-qualified kinds (for example ``simplex:3``) are admitted only when
    their reference vertices and entities are exactly those of the family that
    the decompositions below assume.
    """
    if kind in ("polygon", "polyhedron"):
        return kind
    from ..discretization._reference_cell import reference_cell_topology

    topology = reference_cell_topology(kind)
    family = topology.family
    standard = reference_cell_topology(family)
    if (topology.vertices, topology.entities) != (standard.vertices, standard.entities):
        raise ValueError(
            f"Unsupported cell kind {kind!r}: its reference order differs from "
            f"{family!r}."
        )
    return family


def _block_pieces(
    block: Any,
    first_cell: int,
    coordinates: np.ndarray,
    surface: _PolyhedralSurface | None,
    mode: Any,
    /,
) -> _BlockPieces:
    cells = np.asarray(block.vertices, dtype=np.int64)
    family = _block_family(block.cell_kind)
    failed = np.zeros((block.cell_count,), dtype=np.bool_)
    match family:
        case "triangle" | "quadrilateral" | "polygon":
            return _polygon_block_pieces(coordinates, cells, mode)
        case "tetrahedron" | "hexahedron" | "prism" | "pyramid" | "polyhedron":
            if surface is None:
                if family == "polyhedron":
                    raise ValueError("Polyhedral blocks require polyhedral connectivity.")
                triangles = _standard_face_triangles(cells, family)
                triangle_valid = np.ones(triangles.shape[:2], dtype=np.bool_)
            else:
                identifiers = np.asarray(
                    surface.connectivity.cell_global_ids, dtype=np.int64
                )
                if not np.array_equal(
                    identifiers[first_cell : first_cell + block.cell_count],
                    np.asarray(block.global_ids, dtype=np.int64),
                ):
                    raise ValueError(
                        "Polyhedral connectivity cells must follow the mesh block order."
                    )
                triangles, triangle_valid, failed = _polyhedral_face_triangles(
                    surface, first_cell, block.cell_count
                )
        case _:
            raise ValueError(f"Unsupported cell kind {block.cell_kind!r}.")
    pieces = _volume_block_pieces(
        coordinates,
        cells,
        np.asarray(block.vertex_valid, dtype=np.bool_),
        triangles,
        triangle_valid,
        mode,
    )
    # A cell with a face outside the finite-volume geometry is uncertified,
    # whatever its remaining cone certifies, and carries no pieces.
    kept = ~failed[pieces.cells]
    return _BlockPieces(
        pieces.cells[kept],
        pieces.vertices[kept],
        None,
        pieces.invalid | failed,
        pieces.uncertain & ~failed,
    )


def _decompose(mesh: CellMesh, mode: PredicateMode, /) -> _Decomposition:
    from ..discretization._cell_complex import PolyhedralConnectivity

    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    dimension = mesh.topological_dimension
    surface = (
        _polyhedral_surface(mesh.connectivity, coordinates)
        if dimension == 3 and isinstance(mesh.connectivity, PolyhedralConnectivity)
        else None
    )
    if surface is not None:
        coordinates = surface.coordinates
    cell_parts, vertex_parts, count_parts = [], [], []
    minima, maxima, invalid, uncertain, identifiers = [], [], [], [], []
    first_cell = 0
    for block in mesh.blocks:
        pieces = _block_pieces(block, first_cell, coordinates, surface, mode)
        cells = np.asarray(block.vertices, dtype=np.int64)
        lower, upper = _cell_bounds(
            coordinates, cells, np.asarray(block.vertex_valid, dtype=np.bool_)
        )
        if surface is not None:
            # Warped-face vertex centroids are piece vertices beyond the cell's
            # own vertices; the broad phase must box them too.
            np.minimum.at(lower, pieces.cells, np.min(pieces.vertices, axis=1))
            np.maximum.at(upper, pieces.cells, np.max(pieces.vertices, axis=1))
        cell_parts.append(pieces.cells + first_cell)
        vertex_parts.append(pieces.vertices)
        count_parts.append(pieces.counts)
        minima.append(lower)
        maxima.append(upper)
        invalid.append(pieces.invalid)
        uncertain.append(pieces.uncertain)
        identifiers.append(np.asarray(block.global_ids, dtype=np.int64))
        first_cell += block.cell_count
    piece_cells = np.concatenate(cell_parts)
    order = np.argsort(piece_cells, kind="stable")
    piece_cells = piece_cells[order]
    if dimension == 2:
        width = max(part.shape[1] for part in vertex_parts)
        padded = np.zeros((piece_cells.size, width, 2), dtype=np.float64)
        start = 0
        for part in vertex_parts:
            padded[start : start + part.shape[0], : part.shape[1]] = part
            start += part.shape[0]
        pieces = padded[order]
        piece_counts = np.concatenate(count_parts)[order]
        piece_measures, piece_moments = _polygon_moments(pieces, piece_counts)
    else:
        pieces = np.concatenate(vertex_parts)[order]
        piece_counts = None
        piece_measures, piece_moments = _tetrahedron_moments(pieces)
    cell_count = first_cell
    offsets = np.zeros((cell_count + 1,), dtype=np.int64)
    np.cumsum(np.bincount(piece_cells, minlength=cell_count), out=offsets[1:])
    measures = np.bincount(piece_cells, weights=piece_measures, minlength=cell_count)
    moments = np.stack(
        [
            np.bincount(piece_cells, weights=piece_moments[:, k], minlength=cell_count)
            for k in range(dimension)
        ],
        axis=1,
    )
    return _Decomposition(
        cell_count=cell_count,
        global_ids=np.concatenate(identifiers),
        pieces=pieces,
        piece_counts=piece_counts,
        piece_offsets=offsets,
        measures=measures,
        first_moments=moments,
        bbox_min=np.concatenate(minima),
        bbox_max=np.concatenate(maxima),
        invalid=int(np.sum(np.concatenate(invalid))),
        uncertain=int(np.sum(np.concatenate(uncertain))),
    )


def _outside_exact_domain(mesh: CellMesh, /) -> bool:
    minimum, maximum = _meshcore.meshcore_exact_domain()
    magnitude = np.abs(np.asarray(mesh.coordinates, dtype=np.float64))
    nonzero = magnitude[magnitude != 0.0]
    return bool(
        np.any(nonzero < np.ldexp(1.0, minimum))
        or np.any(nonzero > np.ldexp(1.0, maximum))
    )


# ---------------------------------------------------------------- broad phase


def _candidate_pairs(
    source: _Decomposition, target: _Decomposition, /
) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate positive-extent box overlaps sorted by local (target, source) cell."""

    target_bvh = prepare_bvh(target.bbox_min, target.bbox_max, dtype=jnp.float64)
    source_bvh = prepare_bvh(source.bbox_min, source.bbox_max, dtype=jnp.float64)
    target_parts, source_parts = [], []
    for target_items, source_items in bvh_overlap_pair_blocks(target_bvh, source_bvh):
        target_parts.append(target_items)
        source_parts.append(source_items)
    if not target_parts:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty.copy()
    target_cells = np.concatenate(target_parts)
    source_cells = np.concatenate(source_parts)
    # The CSR rows are local target cells (`target_offsets` is a bincount of
    # `target_cells`) listing local source cells in ascending order.
    order = np.lexsort((source_cells, target_cells))
    return target_cells[order], source_cells[order]


# ---------------------------------------------------------------- narrow phase


class _Batch(NamedTuple):
    pair: np.ndarray
    volume: np.ndarray
    moment: np.ndarray
    status: np.ndarray
    simplices: np.ndarray | None
    simplex_counts: np.ndarray | None
    second: np.ndarray | None


def _clip_batch(
    source: _Decomposition,
    target: _Decomposition,
    target_cells: np.ndarray,
    source_cells: np.ndarray,
    simplices: bool,
    quadratic: bool,
    /,
) -> _Batch:
    """Clip every piece pair of the given cell pairs with meshcore."""

    source_counts = np.diff(source.piece_offsets)[source_cells]
    target_counts = np.diff(target.piece_offsets)[target_cells]
    per_pair = source_counts * target_counts
    pair = np.repeat(np.arange(target_cells.size), per_pair)
    local = np.arange(pair.size) - np.repeat(np.cumsum(per_pair) - per_pair, per_pair)
    source_piece = source.piece_offsets[source_cells][pair] + local // target_counts[pair]
    target_piece = target.piece_offsets[target_cells][pair] + local % target_counts[pair]
    if source.piece_counts is None:
        first = source.pieces[source_piece]
        second = target.pieces[target_piece]
        if simplices:
            parts, counts, volume, moment, status = (
                _meshcore.tetrahedron_intersection_simplices(first, second)
            )
            return _Batch(pair, volume, moment, status, parts, counts, None)
        if quadratic:
            volume, moment, second_moment, status = (
                _meshcore.tetrahedron_intersection_quadratic_moments(first, second)
            )
            return _Batch(pair, volume, moment, status, None, None, second_moment)
        volume, moment, status = _meshcore.tetrahedron_intersection_moments(first, second)
        return _Batch(pair, volume, moment, status, None, None, None)
    arguments = (
        source.pieces[source_piece],
        source.piece_counts[source_piece],
        target.pieces[target_piece],
        # ty: ignore[not-subscriptable]
        target.piece_counts[target_piece],
    )
    if simplices:
        parts, counts, volume, moment, status = _meshcore.polygon_intersection_simplices(
            *arguments
        )
        return _Batch(pair, volume, moment, status, parts, counts, None)
    volume, moment, status = _meshcore.polygon_intersection_moments(*arguments)
    return _Batch(pair, volume, moment, status, None, None, None)


def _simplex_second_moments(simplices: np.ndarray, /) -> np.ndarray:
    """Integral of ``x x^T`` over signed simplices ``(S, d + 1, d)``.

    A three-dimensional carrier can be so thin that its ordinary cross product
    loses most of the signed determinant. Reuse the orientation filter's
    determinant and permanent, and integrate uncertain carriers over their
    represented binary64 vertices with exact rational arithmetic.
    """
    dimension = simplices.shape[2]
    match dimension:
        case 2:
            edges = simplices[:, 1:] - simplices[:, :1]
            measure = 0.5 * (
                edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0]
            )
            scale = measure / 12.0
        case 3:
            # The predicate kernel uses the last point as origin; rotating the
            # corners by one position reverses the package orientation.
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                determinant, permanent, _, largest = _orient3d_kernel(
                    np, simplices[:, 1], simplices[:, 2], simplices[:, 3], simplices[:, 0]
                )
                determinant = -determinant
                info = np.finfo(np.float64)
                unit = float(info.eps) / 2.0
                bound = (7.0 + 56.0 * unit) * unit * permanent + (
                    144.0 * float(info.tiny) * (1.0 + largest) ** 2
                )
            scale = determinant / 120.0
        case _:
            raise ValueError("Simplex second moments need dimension 2 or 3.")
    total = np.sum(simplices, axis=1)
    products = contract("skd,ske->sde", simplices, simplices) + contract(
        "sd,se->sde", total, total
    )
    result = scale[:, None, None] * products
    if dimension == 3:
        # Bound cancellation in the quadratic carrier polynomial as well as
        # the determinant. These error budgets do not alter coverage tolerances.
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            absolute = np.abs(simplices)
            absolute_total = np.sum(absolute, axis=1)
            product_permanent = contract("skd,ske->sde", absolute, absolute) + contract(
                "sd,se->sde", absolute_total, absolute_total
            )
            largest_coordinate = np.max(absolute, axis=(1, 2))
            product_bound = (
                32.0 * float(info.eps) * product_permanent
                + 128.0
                * float(info.tiny)
                * (1.0 + largest_coordinate[:, None, None]) ** 2
            )
            identically_zero = np.all(simplices == 0.0, axis=1)
            structural_zero = identically_zero[:, :, None] | identically_zero[:, None, :]
            reliable = (
                np.isfinite(determinant)
                & np.isfinite(bound)
                & (bound <= 2.0**-46 * np.abs(determinant))
                & np.all(
                    np.isfinite(products)
                    & np.isfinite(product_bound)
                    & (structural_zero | (product_bound <= 2.0**-45 * np.abs(products))),
                    axis=(1, 2),
                )
            )
        for row in np.flatnonzero(~reliable):
            points = tuple(
                rational_vector((float(point[0]), float(point[1]), float(point[2])))
                for point in simplices[row]
            )
            first, second, third, fourth = points
            exact_determinant = determinant3(first, second, third, fourth)
            exact_total = tuple(sum(point[k] for point in points) for k in range(3))
            for j in range(3):
                for k in range(j, 3):
                    polynomial = sum(point[j] * point[k] for point in points) + (
                        exact_total[j] * exact_total[k]
                    )
                    result[row, j, k] = float(exact_determinant * polynomial / 120)
                    result[row, k, j] = result[row, j, k]
    return result


class _BatchResult(NamedTuple):
    keep: np.ndarray
    volumes: np.ndarray
    moments: np.ndarray
    second: np.ndarray | None
    simplex_counts: np.ndarray | None
    simplices: np.ndarray | None
    uncertain: int
    failures: int


def _reduce_batch(batch: _Batch, pair_count: int, policy: Any, dimension: int, /) -> Any:
    failed = batch.status != int(_meshcore.MeshcoreStatus.OK)
    outside = (batch.status == int(_meshcore.MeshcoreStatus.RANGE_ERROR)) | (
        batch.status == int(_meshcore.MeshcoreStatus.NONFINITE_INPUT)
    )
    volumes = np.bincount(batch.pair, weights=batch.volume, minlength=pair_count)
    moments = np.stack(
        [
            np.bincount(batch.pair, weights=batch.moment[:, k], minlength=pair_count)
            for k in range(dimension)
        ],
        axis=1,
    )
    keep = volumes > 0.0
    second = None
    kept_simplices = None
    kept_counts = None
    if batch.second is not None:
        products = batch.second.reshape((-1, dimension * dimension))
        second = np.stack(
            [
                np.bincount(batch.pair, weights=products[:, k], minlength=pair_count)
                for k in range(products.shape[1])
            ],
            axis=1,
        ).reshape((pair_count, dimension, dimension))[keep]
    if batch.simplices is not None:
        slots = np.arange(batch.simplices.shape[1])
        # ty: ignore[not-subscriptable]
        present = slots[None, :] < batch.simplex_counts[:, None]
        simplex_pair = np.broadcast_to(batch.pair[:, None], present.shape)[present]
        flat = batch.simplices[present]
        if policy.second_moments:
            products = _simplex_second_moments(flat).reshape(
                (flat.shape[0], dimension * dimension)
            )
            second = np.stack(
                [
                    np.bincount(
                        simplex_pair, weights=products[:, k], minlength=pair_count
                    )
                    for k in range(products.shape[1])
                ],
                axis=1,
            ).reshape((pair_count, dimension, dimension))[keep]
        if policy.overlap_simplices:
            retained = keep[simplex_pair]
            kept_simplices = flat[retained]
            kept_counts = np.bincount(simplex_pair, minlength=pair_count)[keep]
    return _BatchResult(
        keep,
        volumes[keep],
        moments[keep],
        second,
        kept_counts,
        kept_simplices,
        int(np.sum(outside)),
        int(np.sum(failed & ~outside)),
    )


def _batch_bounds(
    source: _Decomposition,
    target: _Decomposition,
    target_cells: np.ndarray,
    source_cells: np.ndarray,
    pair_bytes: int,
    /,
) -> np.ndarray:
    """Candidate boundaries with at most `_BATCH_BYTES` of piece pairs per batch."""

    per_pair = (
        np.diff(source.piece_offsets)[source_cells]
        * np.diff(target.piece_offsets)[target_cells]
    )
    cumulative = np.cumsum(per_pair * pair_bytes)
    capacity = max(_BATCH_BYTES, pair_bytes)
    marks = np.arange(capacity, cumulative[-1] if cumulative.size else 0, capacity)
    bounds = np.searchsorted(cumulative, marks, side="right")
    return np.unique(np.concatenate(([0], bounds, [target_cells.size])))


def _piece_pair_bytes(
    source: _Decomposition, target: _Decomposition, policy: Any, /
) -> Any:
    if source.piece_counts is None:
        inputs = 2 * 12 * 8
        outputs = 8 + 3 * 8 + 4
        if policy.second_moments and not policy.overlap_simplices:
            outputs += 9 * 8
        partition = _TETRAHEDRON_SIMPLICES * 12 * 8 + 4 if policy.overlap_simplices else 0
    else:
        simplices = policy.second_moments or policy.overlap_simplices
        width = source.pieces.shape[1] + target.pieces.shape[1]
        inputs = width * 2 * 8 + 8
        outputs = 8 + 2 * 8 + 4
        partition = width * 6 * 8 + 4 if simplices else 0
    # Index expansion (pair, local, piece ids) accompanies every piece pair.
    return inputs + outputs + partition + 5 * 8


class _NarrowResult(NamedTuple):
    entries: _Entries
    counts: _Counts


def _narrow_phase(
    source: _Decomposition,
    target: _Decomposition,
    target_cells: np.ndarray,
    source_cells: np.ndarray,
    policy: CommonRefinementPolicy,
    /,
) -> _NarrowResult:
    dimension = source.first_moments.shape[1]
    pair_bytes = _piece_pair_bytes(source, target, policy)
    simplices = policy.overlap_simplices or (dimension == 2 and policy.second_moments)
    quadratic = dimension == 3 and policy.second_moments and not policy.overlap_simplices
    bounds = _batch_bounds(source, target, target_cells, source_cells, pair_bytes)
    parts: list[_BatchResult] = []
    kept_pairs: list[np.ndarray] = []
    accepted = piece_pairs = uncertain = failures = 0
    for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
        batch = _clip_batch(
            source,
            target,
            target_cells[start:stop],
            source_cells[start:stop],
            simplices,
            quadratic,
        )
        piece_pairs += batch.pair.size
        result = _reduce_batch(batch, stop - start, policy, dimension)
        parts.append(result)
        kept_pairs.append(np.flatnonzero(result.keep) + start)
        accepted += result.volumes.size
        uncertain += result.uncertain
        failures += result.failures
    counts = _Counts(target_cells.size, accepted, piece_pairs, uncertain, 0, failures)
    selected = np.concatenate(kept_pairs) if kept_pairs else np.empty((0,), np.int64)
    entries = _Entries(
        target_cells=target_cells[selected],
        source_cells=source_cells[selected],
        volumes=_joined([part.volumes for part in parts], (0,)),
        first_moments=_joined([part.moments for part in parts], (0, dimension)),
        second_moments=(
            _joined_required(
                [part.second for part in parts],
                (0, dimension, dimension),
                "second moments",
            )
            if policy.second_moments
            else None
        ),
        simplex_offsets=(
            _offsets(
                _joined_required(
                    [part.simplex_counts for part in parts], (0,), "simplex counts"
                )
            )
            if policy.overlap_simplices
            else None
        ),
        simplices=(
            _joined_required(
                [part.simplices for part in parts],
                (0, dimension + 1, dimension),
                "overlap simplices",
            )
            if policy.overlap_simplices
            else None
        ),
    )
    return _NarrowResult(entries, counts)


def _joined(parts: list[np.ndarray], empty_shape: tuple[int, ...], /) -> np.ndarray:
    if not parts:
        return np.zeros(empty_shape, dtype=np.float64)
    return np.concatenate(parts)


def _joined_required(
    parts: list[np.ndarray | None],
    empty_shape: tuple[int, ...],
    role: str,
    /,
) -> np.ndarray:
    if any(part is None for part in parts):
        raise RuntimeError(f"Narrow phase omitted requested {role}.")
    return _joined([part for part in parts if part is not None], empty_shape)


def _offsets(counts: np.ndarray, /) -> np.ndarray:
    offsets = np.zeros((counts.size + 1,), dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return offsets


# ---------------------------------------------------------------- assembly


def _empty_entries(dimension: int, policy: CommonRefinementPolicy, /) -> _Entries:
    index = np.empty((0,), dtype=np.int64)
    return _Entries(
        target_cells=index,
        source_cells=index.copy(),
        volumes=np.empty((0,), dtype=np.float64),
        first_moments=np.empty((0, dimension), dtype=np.float64),
        second_moments=(
            np.empty((0, dimension, dimension), dtype=np.float64)
            if policy.second_moments
            else None
        ),
        simplex_offsets=(
            np.zeros((1,), dtype=np.int64) if policy.overlap_simplices else None
        ),
        simplices=(
            np.empty((0, dimension + 1, dimension), dtype=np.float64)
            if policy.overlap_simplices
            else None
        ),
    )


def _coverage_tolerances(
    part: _Decomposition, policy: CommonRefinementPolicy, /
) -> np.ndarray:
    diameter = np.sqrt(np.sum((part.bbox_max - part.bbox_min) ** 2, axis=1))
    dimension = part.bbox_min.shape[1]
    floor = _COVERAGE_ROUNDOFF_ULPS * np.finfo(np.float64).eps * diameter**dimension
    return policy.coverage_tolerance * part.measures + floor


def _coverage_status(
    source_defects: np.ndarray,
    target_defects: np.ndarray,
    source_tolerances: np.ndarray,
    target_tolerances: np.ndarray,
    policy: CommonRefinementPolicy,
    /,
) -> tuple[CommonRefinementStatus, str]:
    source_gap = bool(np.any(-source_defects > source_tolerances))
    target_gap = bool(np.any(-target_defects > target_tolerances))
    if np.any(source_defects > source_tolerances) or np.any(
        target_defects > target_tolerances
    ):
        return (
            CommonRefinementStatus.DOUBLE_COVERAGE,
            "cells are covered more than once",
        )
    match policy.coverage:
        case CommonRefinementCoverage.COMPLETE:
            gap = source_gap or target_gap
        case CommonRefinementCoverage.TARGET:
            gap = target_gap
        case CommonRefinementCoverage.SOURCE:
            gap = source_gap
        case CommonRefinementCoverage.PARTIAL:
            gap = False
        case _:
            raise ValueError(f"Unsupported coverage requirement {policy.coverage!r}.")
    if gap:
        return (
            CommonRefinementStatus.COVERAGE_GAP,
            f"{policy.coverage.value} coverage requirement is not met",
        )
    return CommonRefinementStatus.SUCCESS, "certified common refinement"


def _validate_meshes(source: Any, target: Any, policy: Any, /) -> None:
    from ..discretization._cell_mesh import CellMesh

    if not isinstance(source, CellMesh) or not isinstance(target, CellMesh):
        raise TypeError("source and target must be CellMesh instances.")
    if not isinstance(policy, CommonRefinementPolicy):
        raise TypeError("policy must be a CommonRefinementPolicy.")
    dimension = source.topological_dimension
    if dimension not in (2, 3):
        raise ValueError("Common refinement supports two- and three-dimensional cells.")
    if (
        target.topological_dimension != dimension
        or source.ambient_dimension != dimension
        or target.ambient_dimension != dimension
    ):
        raise ValueError(
            "Source and target must share one topological dimension equal to their "
            "ambient dimension."
        )


def _decomposition_failure(
    outside: bool, counts: _Counts, /
) -> tuple[CommonRefinementStatus, str] | None:
    if outside:
        return (
            CommonRefinementStatus.PREDICATE_UNCERTAIN,
            "coordinates lie outside the meshcore exact domain",
        )
    if counts.invalid:
        return (
            CommonRefinementStatus.INVALID_GEOMETRY,
            "cells without a certified convex decomposition",
        )
    if counts.uncertain:
        return (
            CommonRefinementStatus.PREDICATE_UNCERTAIN,
            "unresolved decomposition predicates",
        )
    return None


def prepare_common_refinement(
    source: CellMesh,
    target: CellMesh,
    /,
    *,
    policy: CommonRefinementPolicy = CommonRefinementPolicy(),
) -> PreparedCommonRefinement:
    """Exhaustively prepare physical cell-overlap CSR and coverage evidence."""

    _validate_meshes(source, target, policy)
    _meshcore.load_meshcore()
    identities = (
        source.mesh_id,
        target.mesh_id,
        source.topology_id,
        target.topology_id,
    )
    outside = _outside_exact_domain(source) or _outside_exact_domain(target)
    mode = PredicateMode.FILTERED if outside else policy.predicate_mode
    first = _decompose(source, mode)
    second = _decompose(target, mode)
    counts = _Counts(
        0, 0, 0, first.uncertain + second.uncertain, first.invalid + second.invalid, 0
    )
    failure = _decomposition_failure(outside, counts)
    if failure is not None:
        return _prepared(first, second, None, counts, *failure, identities, policy)
    target_cells, source_cells = _candidate_pairs(first, second)
    counts = counts._replace(candidates=target_cells.size)
    narrow = _narrow_phase(first, second, target_cells, source_cells, policy)
    return _prepared(
        first,
        second,
        narrow.entries,
        narrow.counts,
        None,
        "",
        identities,
        policy,
    )


def _prepared(
    source: _Decomposition,
    target: _Decomposition,
    entries: _Entries | None,
    counts: _Counts,
    status: CommonRefinementStatus | None,
    reason: str,
    identities: tuple[str, str, str, str],
    policy: CommonRefinementPolicy,
    /,
) -> PreparedCommonRefinement:
    dimension = source.first_moments.shape[1]
    records = _empty_entries(dimension, policy) if entries is None else entries
    source_defects = (
        np.bincount(
            records.source_cells, weights=records.volumes, minlength=source.cell_count
        )
        - source.measures
    )
    target_defects = (
        np.bincount(
            records.target_cells, weights=records.volumes, minlength=target.cell_count
        )
        - target.measures
    )
    source_tolerances = _coverage_tolerances(source, policy)
    target_tolerances = _coverage_tolerances(target, policy)
    if status is None:
        if counts.uncertain:
            status, reason = (
                CommonRefinementStatus.PREDICATE_UNCERTAIN,
                "native clipping could not certify piece pairs",
            )
        elif counts.failures:
            status, reason = (
                CommonRefinementStatus.INTERSECTION_FAILURE,
                "native clipping failed for certified piece pairs",
            )
        else:
            status, reason = _coverage_status(
                source_defects,
                target_defects,
                source_tolerances,
                target_tolerances,
                policy,
            )
    evidence = CommonRefinementEvidence(
        status=status,
        reason=reason,
        source_coverage_defects=source_defects,
        target_coverage_defects=target_defects,
        source_measures=source.measures,
        target_measures=target.measures,
        source_tolerances=source_tolerances,
        target_tolerances=target_tolerances,
        counts=counts,
    )
    return PreparedCommonRefinement(
        source=source,
        target=target,
        entries=records,
        identities=identities,
        policy=policy,
        evidence=evidence,
    )


def prepare_compact_cell_common_refinement(
    source_cell_vertices: Any,
    source_cell_ids: Any,
    target_cell_vertices: Any,
    target_cell_ids: Any,
    /,
    *,
    source_cell_kind: str,
    target_cell_kind: str,
    source_cell_faces: Any | None = None,
    target_cell_faces: Any | None = None,
    policy: CommonRefinementPolicy = CommonRefinementPolicy(),
) -> PreparedCommonRefinement:
    """Clip owner-routed straight-sided compact cells with the native supermesh.

    Every input cell has a separate vertex bank. Coordinates are never pooled;
    stable cell IDs and, for polyhedra, authored face loops preserve identity.
    Source cells must already have been routed by a complete AABB query.
    """
    from ..discretization._cell_complex import polyhedral_connectivity
    from ..discretization._cell_mesh import CellBlock, CellMesh

    if not isinstance(policy, CommonRefinementPolicy):
        raise TypeError("policy must be a CommonRefinementPolicy.")
    dimensions = {
        "triangle": 2,
        "quadrilateral": 2,
        "polygon": 2,
        "tetrahedron": 3,
        "hexahedron": 3,
        "prism": 3,
        "pyramid": 3,
        "polyhedron": 3,
    }
    source_kind, target_kind = str(source_cell_kind), str(target_cell_kind)
    if (
        source_kind not in dimensions
        or target_kind not in dimensions
        or dimensions[source_kind] != dimensions[target_kind]
    ):
        raise ValueError(
            "Compact common refinement requires matching 2D/3D cell dimensions."
        )
    dimension = dimensions[source_kind]

    def normalize(
        raw_vertices: Any,
        raw_ids: Any,
        kind: str,
        raw_faces: Any | None,
        name: str,
        /,
    ) -> tuple[
        tuple[np.ndarray, ...], np.ndarray, tuple[tuple[np.ndarray, ...], ...] | None
    ]:
        if isinstance(raw_vertices, np.ndarray) and raw_vertices.ndim == 3:
            vertices = tuple(np.asarray(row, dtype=np.float64) for row in raw_vertices)
        else:
            vertices = tuple(np.asarray(row, dtype=np.float64) for row in raw_vertices)
        identifiers = np.asarray(raw_ids)
        if (
            identifiers.ndim != 1
            or identifiers.dtype.kind not in "iu"
            or identifiers.shape != (len(vertices),)
            or np.any(identifiers < 0)
            or np.unique(identifiers).size != identifiers.size
        ):
            raise ValueError(f"{name} compact cells need unique nonnegative stable IDs.")
        if (
            identifiers.dtype.kind == "u"
            and identifiers.size
            and int(np.max(identifiers)) > np.iinfo(np.int64).max
        ):
            raise ValueError(f"{name} stable cell IDs must fit signed int64.")
        minimum = {
            "triangle": 3,
            "quadrilateral": 4,
            "polygon": 5,
            "tetrahedron": 4,
            "hexahedron": 8,
            "prism": 6,
            "pyramid": 5,
            "polyhedron": 4,
        }[kind]
        for cell in vertices:
            if (
                cell.ndim != 2
                or cell.shape[1] != dimension
                or cell.shape[0] < minimum
                or not np.all(np.isfinite(cell))
            ):
                raise ValueError(f"{name} compact cell vertices are malformed.")
        if kind != "polyhedron":
            if raw_faces is not None:
                raise ValueError("Only compact polyhedra may carry authored face loops.")
            faces = None
        else:
            if raw_faces is None or len(raw_faces) != len(vertices):
                raise ValueError(
                    "Compact polyhedra require one authored face set per cell."
                )
            normalized_faces = []
            for cell, cell_faces in zip(vertices, raw_faces, strict=True):
                loops = tuple(np.asarray(face, dtype=np.int32) for face in cell_faces)
                if not loops or any(
                    loop.ndim != 1
                    or loop.size < 3
                    or np.any(loop < 0)
                    or np.any(loop >= cell.shape[0])
                    or np.unique(loop).size != loop.size
                    for loop in loops
                ):
                    raise ValueError("Compact polyhedron face loops are malformed.")
                normalized_faces.append(loops)
            faces = tuple(normalized_faces)
        return (
            vertices,
            identifiers.astype(np.int64, copy=False),
            faces,
        )

    source_vertices, source_ids, source_faces = normalize(
        source_cell_vertices,
        source_cell_ids,
        source_kind,
        source_cell_faces,
        "Source",
    )
    target_vertices, target_ids, target_faces = normalize(
        target_cell_vertices,
        target_cell_ids,
        target_kind,
        target_cell_faces,
        "Target",
    )

    def compact_mesh(
        vertices: tuple[np.ndarray, ...],
        identifiers: np.ndarray,
        kind: str,
        faces: tuple[tuple[np.ndarray, ...], ...] | None,
        side: str,
        /,
    ) -> tuple[CellMesh | None, str, str]:
        identity = canonical_fingerprint(
            {
                "kind": "compact-owner-local-cell-mesh",
                "side": side,
                "cell_kind": kind,
                "cell_ids": array_tree_fingerprint(identifiers),
                "cell_vertices": tuple(array_tree_fingerprint(row) for row in vertices),
                "cell_faces": None
                if faces is None
                else tuple(
                    tuple(array_tree_fingerprint(loop) for loop in cell) for cell in faces
                ),
            }
        )
        if not identifiers.size:
            return None, identity, identity
        offsets = np.concatenate(
            (
                np.zeros((1,), dtype=np.int64),
                np.cumsum([row.shape[0] for row in vertices], dtype=np.int64),
            )
        )
        coordinates = np.concatenate(vertices, axis=0)
        if kind == "polyhedron":
            if faces is None:
                raise ValueError(
                    "Compact polyhedra require one authored face set per cell."
                )
            width = max(row.shape[0] for row in vertices)
            connectivity = np.full((len(vertices), width), -1, dtype=np.int32)
            valid = np.zeros(connectivity.shape, dtype=np.bool_)
            definitions = []
            for cell, (row, cell_faces) in enumerate(zip(vertices, faces, strict=True)):
                count = row.shape[0]
                base = int(offsets[cell])
                connectivity[cell, :count] = np.arange(base, base + count, dtype=np.int32)
                valid[cell, :count] = True
                definitions.append(
                    tuple(np.asarray(loop, dtype=np.int32) + base for loop in cell_faces)
                )
            block = CellBlock(
                f"{side}-compact-polyhedra",
                "polyhedron",
                connectivity,
                vertex_valid=valid,
                global_ids=identifiers,
            )
            polyhedra = polyhedral_connectivity(
                definitions,
                coordinates.shape[0],
                cell_global_ids=identifiers,
            )
            local_mesh = CellMesh(
                coordinates,
                (block,),
                polyhedral_connectivity=polyhedra,
            )
            return local_mesh, local_mesh.mesh_id, local_mesh.topology_id
        groups: dict[int, list[int]] = {}
        for row, cell in enumerate(vertices):
            groups.setdefault(cell.shape[0], []).append(row)
        blocks = []
        for arity, rows in sorted(groups.items()):
            connectivity = np.stack(
                [
                    np.arange(offsets[row], offsets[row + 1], dtype=np.int32)
                    for row in rows
                ]
            )
            blocks.append(
                CellBlock(
                    f"{side}-compact-{arity}",
                    kind,
                    connectivity,
                    global_ids=identifiers[np.asarray(rows, dtype=np.int64)],
                )
            )
        local_mesh = CellMesh(coordinates, tuple(blocks))
        return local_mesh, local_mesh.mesh_id, local_mesh.topology_id

    source_mesh, source_mesh_id, source_topology_id = compact_mesh(
        source_vertices, source_ids, source_kind, source_faces, "source"
    )
    target_mesh, target_mesh_id, target_topology_id = compact_mesh(
        target_vertices, target_ids, target_kind, target_faces, "target"
    )
    identities = (
        source_mesh_id,
        target_mesh_id,
        source_topology_id,
        target_topology_id,
    )
    if source_mesh is not None and target_mesh is not None:
        return prepare_common_refinement(source_mesh, target_mesh, policy=policy)

    def empty_decomposition(kind: str, ids: np.ndarray, /) -> _Decomposition:
        if dimension == 2:
            pieces = np.empty((0, 3, 2), dtype=np.float64)
            piece_counts: np.ndarray | None = np.empty((0,), dtype=np.int32)
        else:
            pieces = np.empty((0, 4, 3), dtype=np.float64)
            piece_counts = None
        return _Decomposition(
            cell_count=0,
            global_ids=ids,
            pieces=pieces,
            piece_counts=piece_counts,
            piece_offsets=np.zeros((1,), dtype=np.int64),
            measures=np.empty((0,), dtype=np.float64),
            first_moments=np.empty((0, dimension), dtype=np.float64),
            bbox_min=np.empty((0, dimension), dtype=np.float64),
            bbox_max=np.empty((0, dimension), dtype=np.float64),
            invalid=0,
            uncertain=0,
        )

    outside = any(
        _outside_exact_domain(local_mesh)
        for local_mesh in (source_mesh, target_mesh)
        if local_mesh is not None
    )
    mode = PredicateMode.FILTERED if outside else policy.predicate_mode
    source_decomposition = (
        empty_decomposition(source_kind, source_ids)
        if source_mesh is None
        else _decompose(source_mesh, mode)
    )
    target_decomposition = (
        empty_decomposition(target_kind, target_ids)
        if target_mesh is None
        else _decompose(target_mesh, mode)
    )
    counts = _Counts(0, 0, 0, 0, 0, 0)
    failure = _decomposition_failure(
        outside,
        counts._replace(
            uncertain=source_decomposition.uncertain + target_decomposition.uncertain,
            invalid=source_decomposition.invalid + target_decomposition.invalid,
        ),
    )
    return _prepared(
        source_decomposition,
        target_decomposition,
        _empty_entries(dimension, policy),
        counts,
        None if failure is None else failure[0],
        "" if failure is None else failure[1],
        identities,
        policy,
    )


__all__ = [
    "CommonRefinementCoverage",
    "CommonRefinementEvidence",
    "CommonRefinementPolicy",
    "CommonRefinementStatus",
    "PreparedCommonRefinement",
    "prepare_common_refinement",
    "prepare_compact_cell_common_refinement",
]
