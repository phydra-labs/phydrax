#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-independent arrangements of affine triangle surfaces.

The arrangement is built from the supplied triangle meshes, not from CAD
features or source curves. Native meshcore predicates retain construction
identity while splitting every AABB candidate pair; binary64 coordinates are
published only after the native arrangement is complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from ..._bvh import bvh_overlap_pair_blocks, prepare_bvh
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._meshcore import (
    arrange_triangles,
    ArrangementClassification,
    ArrangementFailure,
    exact_orient2d,
    IntersectionClass,
    meshcore_exact_domain,
    MeshcoreStatus,
    NativeArrangement,
    triangle_intersection_classes,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import parse


SurfaceArrangementStatus: TypeAlias = Literal[
    "invalid_coordinates",
    "degenerate_triangle",
    "self_intersecting_surface",
    "unrepresentable_publication",
]

_CONTACT_BATCH = 1 << 15
_PAIR_INDEX_MAX = np.iinfo(np.intp).max // (2 * np.dtype(np.int64).itemsize)
_LEGAL_ADJACENCY = (
    IntersectionClass.DISJOINT,
    IntersectionClass.SHARED_VERTEX,
    IntersectionClass.SHARED_EDGE,
)


class SurfaceArrangementError(ValueError):
    """Fail-closed arrangement refusal with source operand and face identity."""

    def __init__(
        self,
        status: SurfaceArrangementStatus,
        message: str,
        /,
        *,
        surface: int | None = None,
        faces: tuple[int, ...] = (),
    ) -> None:
        status = parse(status, SurfaceArrangementStatus, "status")
        self.status = status
        self.surface = surface
        self.faces = tuple(int(face) for face in faces)
        super().__init__(f"{message} (status {status})")


class SurfaceArrangementEvidence(StrictModule, NonTrainableState):
    """Geometric facts about the published arrangement."""

    constructed_vertices: int = eqx.field(static=True)
    coincident_vertices: int = eqx.field(static=True)
    maximum_construction_bound: float = eqx.field(static=True)
    split_faces: tuple[int, ...] = eqx.field(static=True)
    fragments: tuple[int, ...] = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        constructed_vertices: int,
        coincident_vertices: int,
        maximum_construction_bound: float,
        split_faces: tuple[int, ...],
        fragments: tuple[int, ...],
    ) -> None:
        split = tuple(int(count) for count in split_faces)
        fragment_counts = tuple(int(count) for count in fragments)
        constructed = int(constructed_vertices)
        coincident = int(coincident_vertices)
        bound = float(maximum_construction_bound)
        if min((constructed, coincident, *split, *fragment_counts), default=0) < 0:
            raise ValueError("Surface arrangement evidence counts must be nonnegative.")
        if len(split) < 2 or len(split) != len(fragment_counts):
            raise ValueError("split_faces and fragments must name every operand.")
        if not np.isfinite(bound) or bound < 0.0:
            raise ValueError("maximum_construction_bound must be finite and nonnegative.")
        self.constructed_vertices = constructed
        self.coincident_vertices = coincident
        self.maximum_construction_bound = bound
        self.split_faces = split
        self.fragments = fragment_counts
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "surface-arrangement-evidence",
                "constructed_vertices": constructed,
                "coincident_vertices": coincident,
                "maximum_construction_bound": bound,
                "split_faces": split,
                "fragments": fragment_counts,
            }
        )


class SurfaceArrangement(StrictModule, NonTrainableState):
    """Conforming oriented fragments and exact source-face ancestry.

    ``vertex_features[v, i]`` is the sorted, ``-1``-padded input vertex
    simplex of operand ``i`` carrying vertex ``v``. ``vertex_locations``
    retains every source-face incidence, including incidences on an internal
    triangulation edge. ``coincident_face[f, i]`` is a covering coplanar face
    of operand ``i`` and ``coincident_orientation`` is its orientation relative
    to fragment ``f``; the fragment's source operand has no coincidence entry.
    """

    vertices: np.ndarray
    vertex_bounds: np.ndarray
    vertex_features: np.ndarray
    construction_types: np.ndarray
    vertex_locations: np.ndarray
    triangles: np.ndarray
    source_surface: np.ndarray
    source_face: np.ndarray
    coincident_face: np.ndarray
    coincident_orientation: np.ndarray
    intersection_edges: np.ndarray
    input_face_offsets: tuple[int, ...] = eqx.field(static=True)
    evidence: SurfaceArrangementEvidence
    arrangement_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        vertices: ArrayLike,
        vertex_bounds: ArrayLike,
        vertex_features: ArrayLike,
        construction_types: ArrayLike,
        vertex_locations: ArrayLike,
        triangles: ArrayLike,
        source_surface: ArrayLike,
        source_face: ArrayLike,
        coincident_face: ArrayLike,
        coincident_orientation: ArrayLike,
        intersection_edges: ArrayLike,
        input_face_offsets: tuple[int, ...],
        evidence: SurfaceArrangementEvidence,
    ) -> None:
        points = np.asarray(vertices, dtype=np.float64)
        bounds = np.asarray(vertex_bounds, dtype=np.float64)
        features = np.asarray(vertex_features, dtype=np.int64)
        constructions = np.asarray(construction_types, dtype=np.int32)
        locations = np.asarray(vertex_locations, dtype=np.int64)
        faces = np.asarray(triangles, dtype=np.int64)
        surfaces = np.asarray(source_surface, dtype=np.int64)
        sources = np.asarray(source_face, dtype=np.int64)
        coincident = np.asarray(coincident_face, dtype=np.int64)
        orientations = np.asarray(coincident_orientation, dtype=np.int64)
        edges = np.asarray(intersection_edges, dtype=np.int64)
        offsets = tuple(int(value) for value in input_face_offsets)
        if points.ndim != 2 or points.shape[1:] != (3,) or points.shape[0] == 0:
            raise ValueError("SurfaceArrangement.vertices must have shape (n > 0, 3).")
        vertex_count = points.shape[0]
        operand_count = features.shape[1] if features.ndim == 3 else 0
        fragment_count = faces.shape[0] if faces.ndim == 2 else -1
        if (
            bounds.shape != (vertex_count,)
            or features.shape != (vertex_count, operand_count, 3)
            or operand_count < 2
            or constructions.shape != (vertex_count,)
            or locations.ndim != 2
            or locations.shape[1:] != (3,)
            or faces.ndim != 2
            or faces.shape[1:] != (3,)
            or fragment_count == 0
            or surfaces.shape != (fragment_count,)
            or sources.shape != (fragment_count,)
            or coincident.shape != (fragment_count, operand_count)
            or orientations.shape != coincident.shape
            or edges.ndim != 2
            or edges.shape[1:] != (2,)
        ):
            raise ValueError("SurfaceArrangement arrays have inconsistent shapes.")
        if (
            len(offsets) != operand_count + 1
            or offsets[0] != 0
            or any(first >= second for first, second in zip(offsets, offsets[1:]))
        ):
            raise ValueError("input_face_offsets must delimit every nonempty operand.")
        if (
            not np.all(np.isfinite(points))
            or not np.all(np.isfinite(bounds))
            or np.any(bounds < 0.0)
            or np.any(faces < 0)
            or np.any(faces >= vertex_count)
            or np.any(edges < 0)
            or np.any(edges >= vertex_count)
            or np.any(surfaces < 0)
            or np.any(surfaces >= operand_count)
            or np.any(sources < 0)
            or np.any(features < -1)
            or np.any(constructions < 0)
            or np.any(constructions > 3)
            or np.any(sources >= np.diff(offsets)[surfaces])
            or np.any(coincident < -1)
            or np.any((orientations < -1) | (orientations > 1))
            or np.any((coincident < 0) != (orientations == 0))
            or np.any(coincident >= np.diff(offsets)[None, :])
            or np.any(locations[:, 0] < 0)
            or np.any(locations[:, 0] >= vertex_count)
            or np.any(locations[:, 1] < 0)
            or np.any(locations[:, 1] >= offsets[-1])
            or np.any(locations[:, 2] < 0)
            or np.any(locations[:, 2] > 6)
        ):
            raise ValueError("SurfaceArrangement contains invalid geometry or ancestry.")
        if not isinstance(evidence, SurfaceArrangementEvidence):
            raise TypeError("evidence must be SurfaceArrangementEvidence.")
        self.vertices = points
        self.vertex_bounds = bounds
        self.vertex_features = features
        self.construction_types = constructions
        self.vertex_locations = locations
        self.triangles = faces
        self.source_surface = surfaces
        self.source_face = sources
        self.coincident_face = coincident
        self.coincident_orientation = orientations
        self.intersection_edges = edges
        self.input_face_offsets = offsets
        self.evidence = evidence
        self.arrangement_id = canonical_fingerprint(
            {
                "kind": "surface-arrangement",
                "arrays": array_tree_fingerprint(
                    (
                        points,
                        bounds,
                        features,
                        constructions,
                        locations,
                        faces,
                        surfaces,
                        sources,
                        coincident,
                        orientations,
                        edges,
                    )
                ),
                "input_face_offsets": offsets,
                "evidence_id": evidence.evidence_id,
            }
        )


@dataclass(frozen=True, slots=True)
class _SurfaceInput:
    points: np.ndarray
    faces: np.ndarray
    source_faces: np.ndarray


@dataclass(frozen=True, slots=True)
class _PreparedArrangement:
    surfaces: tuple[_SurfaceInput, ...]
    vertex_offsets: np.ndarray
    face_offsets: np.ndarray


def _prepared_surface(
    vertices: ArrayLike, triangles: ArrayLike, index: int, /
) -> _SurfaceInput:
    raw_points = np.asarray(vertices)
    raw_faces = np.asarray(triangles)
    if raw_points.ndim != 2 or raw_points.shape[1:] != (3,) or raw_points.shape[0] == 0:
        raise ValueError("surface vertices must have shape (n > 0, 3).")
    if (
        np.issubdtype(raw_points.dtype, np.bool_)
        or not np.issubdtype(raw_points.dtype, np.number)
        or np.iscomplexobj(raw_points)
    ):
        raise TypeError("surface vertices must be real numeric coordinates.")

    if np.issubdtype(raw_points.dtype, np.integer):
        if np.issubdtype(raw_points.dtype, np.unsignedinteger):
            inexact_integer = np.any(raw_points > 2**53)
        else:
            inexact_integer = np.any((raw_points > 2**53) | (raw_points < -(2**53)))
        if inexact_integer:
            raise ValueError(
                "Integer surface coordinates must be exactly representable as float64."
            )
    if raw_faces.ndim != 2 or raw_faces.shape[1:] != (3,) or raw_faces.shape[0] == 0:
        raise ValueError("surface triangles must have shape (m > 0, 3).")
    if not np.issubdtype(raw_faces.dtype, np.integer) or np.issubdtype(
        raw_faces.dtype, np.bool_
    ):
        raise TypeError("surface triangle indices must be integers.")
    points = np.ascontiguousarray(raw_points, dtype=np.float64)
    if (
        np.issubdtype(raw_points.dtype, np.floating)
        and raw_points.dtype.itemsize > np.dtype(np.float64).itemsize
        and np.all(np.isfinite(raw_points))
        and not np.array_equal(points.astype(raw_points.dtype), raw_points)
    ):
        raise ValueError("Surface coordinates must be exactly representable as float64.")
    if not np.all(np.isfinite(points)):
        raise SurfaceArrangementError(
            "invalid_coordinates",
            f"surface {index} contains nonfinite coordinates",
            surface=index,
        )
    if np.any(raw_faces < 0) or np.any(raw_faces >= points.shape[0]):
        raise ValueError(f"surface {index} triangles index undeclared vertices.")
    source_faces = np.ascontiguousarray(raw_faces, dtype=np.int64)
    repeated_mask = (
        (source_faces[:, 0] == source_faces[:, 1])
        | (source_faces[:, 1] == source_faces[:, 2])
        | (source_faces[:, 2] == source_faces[:, 0])
    )
    if np.any(repeated_mask):
        raise SurfaceArrangementError(
            "degenerate_triangle",
            f"surface {index} contains repeated triangle vertices",
            surface=index,
            faces=tuple(np.flatnonzero(repeated_mask).tolist()),
        )
    minimum, maximum = meshcore_exact_domain()
    magnitude = np.abs(points[source_faces])
    smallest = np.ldexp(np.float64(1.0), minimum)
    largest = np.ldexp(np.float64(1.0), maximum)
    outside = np.any(
        (magnitude != 0.0) & ((magnitude < smallest) | (magnitude > largest)),
        axis=(1, 2),
    )
    if np.any(outside):
        bad_faces = np.flatnonzero(outside)
        raise SurfaceArrangementError(
            "invalid_coordinates",
            f"surface {index} coordinates are outside the native exact-predicate domain",
            surface=index,
            faces=tuple(bad_faces.tolist()),
        )
    corners = points[source_faces]
    collinear = np.ones((source_faces.shape[0],), dtype=np.bool_)
    for axes in ((0, 1), (1, 2), (2, 0)):
        projection = corners[..., axes]
        collinear &= (
            exact_orient2d(projection[:, 0], projection[:, 1], projection[:, 2]) == 0
        )
    degenerate = np.flatnonzero(collinear)
    if degenerate.size:
        raise SurfaceArrangementError(
            "degenerate_triangle",
            f"surface {index} contains collinear triangles",
            surface=index,
            faces=tuple(degenerate.tolist()),
        )
    original_vertices, compact = np.unique(source_faces, return_inverse=True)
    return _SurfaceInput(
        points[original_vertices],
        compact.reshape(source_faces.shape).astype(np.int64, copy=False),
        source_faces,
    )


def _candidate_pairs(
    first: _SurfaceInput, second: _SurfaceInput, /, *, self_pairs: bool
) -> np.ndarray:
    """Enumerate every closed-box overlap in deterministic local face order."""
    first_corners = first.points[first.faces]
    first_bvh = prepare_bvh(
        np.min(first_corners, axis=1), np.max(first_corners, axis=1), dtype=np.float64
    )
    if self_pairs:
        second_bvh = first_bvh
    else:
        second_corners = second.points[second.faces]
        second_bvh = prepare_bvh(
            np.min(second_corners, axis=1),
            np.max(second_corners, axis=1),
            dtype=np.float64,
        )
    blocks: list[np.ndarray] = []
    pair_count = 0
    for first_ids, second_ids in bvh_overlap_pair_blocks(
        first_bvh, second_bvh, include_touching=True
    ):
        if self_pairs:
            keep = first_ids < second_ids
            first_ids, second_ids = first_ids[keep], second_ids[keep]
        if first_ids.size:
            pair_count += int(first_ids.size)
            if pair_count > _PAIR_INDEX_MAX:
                raise OverflowError(
                    "Triangle candidate pairs exceed host index capacity."
                )
            blocks.append(
                np.column_stack((first_ids, second_ids)).astype(np.int64, copy=False)
            )
    if not blocks:
        return np.empty((0, 2), dtype=np.int64)
    pairs = np.concatenate(blocks, axis=0)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return pairs[order]


def _feature_simplices(faces: np.ndarray, features: np.ndarray, /) -> np.ndarray:
    """Sorted, ``-1``-padded input vertex IDs for exact triangle features."""
    rows = np.arange(features.shape[0])
    simplices = np.full((features.shape[0], 3), -1, dtype=np.int64)
    vertex = features <= 2
    simplices[vertex, 0] = faces[rows[vertex], features[vertex]]
    edge = (features >= 3) & (features <= 5)
    opposite = features[edge] - 3
    ends = np.stack(
        (
            faces[rows[edge], (opposite + 1) % 3],
            faces[rows[edge], (opposite + 2) % 3],
        ),
        axis=1,
    )
    simplices[edge, :2] = np.sort(ends, axis=1)
    interior = features == 6
    simplices[interior] = np.sort(faces[rows[interior]], axis=1)
    return simplices


def _embedding_violations(
    surface: _SurfaceInput,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Return illegal or refused self-contact pairs for one triangle mesh."""
    pairs = _candidate_pairs(surface, surface, self_pairs=True)
    illegal: list[np.ndarray] = []
    refused: list[np.ndarray] = []
    for start in range(0, pairs.shape[0], _CONTACT_BATCH):
        batch = pairs[start : start + _CONTACT_BATCH]
        first, second = batch[:, 0], batch[:, 1]
        classes, status = triangle_intersection_classes(
            surface.points[surface.faces[first]],
            surface.points[surface.faces[second]],
            first_ids=surface.faces[first],
            second_ids=surface.faces[second],
        )
        failed = status != MeshcoreStatus.OK
        crossing = ~failed & ~np.isin(classes, _LEGAL_ADJACENCY)
        if np.any(crossing):
            illegal.append(batch[crossing].reshape(-1))
        if np.any(failed):
            refused.append(batch[failed].reshape(-1))
    return (
        np.unique(np.concatenate(illegal)) if illegal else np.empty((0,), dtype=np.int64),
        np.unique(np.concatenate(refused)) if refused else np.empty((0,), dtype=np.int64),
    )


def _require_embedded(surface: _SurfaceInput, index: int, /) -> None:
    illegal, refused = _embedding_violations(surface)
    if refused.size:
        raise SurfaceArrangementError(
            "invalid_coordinates",
            f"surface {index} contains triangles refused by the native contact predicate",
            surface=index,
            faces=tuple(refused.tolist()),
        )
    if illegal.size:
        raise SurfaceArrangementError(
            "self_intersecting_surface",
            f"surface {index} has triangle contacts beyond shared vertices or edges",
            surface=index,
            faces=tuple(illegal.tolist()),
        )


def require_published_embedded(
    vertices: np.ndarray,
    triangles: np.ndarray,
    reported_faces: np.ndarray,
    /,
    *,
    surface: int | None,
) -> None:
    """Refuse rounded output whose triangle embedding no longer conforms."""
    points = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int64)
    reported = np.asarray(reported_faces, dtype=np.int64)
    if faces.shape[0] == 0:
        return
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or reported.shape != (faces.shape[0],)
    ):
        raise ValueError("Published embedding inputs have inconsistent shapes.")
    published = _SurfaceInput(points, faces, faces)
    illegal, refused = _embedding_violations(published)
    offending = np.union1d(illegal, refused)
    if offending.size:
        source_faces = np.unique(reported[offending])
        raise SurfaceArrangementError(
            "unrepresentable_publication",
            "binary64 publication does not preserve the arrangement embedding",
            surface=surface,
            faces=tuple(source_faces.tolist()),
        )


def _prepare_arrangement(
    inputs: tuple[tuple[ArrayLike, ArrayLike], ...], /
) -> _PreparedArrangement:
    if len(inputs) < 2:
        raise ValueError("At least two triangle surfaces are required.")
    surfaces = tuple(
        _prepared_surface(vertices, faces, i)
        for i, (vertices, faces) in enumerate(inputs)
    )
    vertex_counts = [int(surface.points.shape[0]) for surface in surfaces]
    face_counts = [int(surface.faces.shape[0]) for surface in surfaces]
    native_index_max = int(np.iinfo(np.int32).max) - 1
    if sum(vertex_counts) > native_index_max or sum(face_counts) > native_index_max:
        raise OverflowError("Surface arrangement exceeds native signed-index capacity.")
    vertex_offsets = np.r_[0, np.cumsum(vertex_counts, dtype=np.int64)]
    face_offsets = np.r_[0, np.cumsum(face_counts, dtype=np.int64)]
    return _PreparedArrangement(surfaces, vertex_offsets, face_offsets)


def _contacts(prepared: _PreparedArrangement, /) -> np.ndarray:
    """Validate input embeddings and retain every cross-operand AABB pair."""
    surfaces = prepared.surfaces
    offsets = prepared.face_offsets
    for index, surface in enumerate(surfaces):
        _require_embedded(surface, index)
    candidate_count = 0
    candidate_blocks: list[np.ndarray] = []
    for first_index, first_surface in enumerate(surfaces):
        for second_index in range(first_index + 1, len(surfaces)):
            local_pairs = _candidate_pairs(
                first_surface, surfaces[second_index], self_pairs=False
            )
            if local_pairs.size:
                candidate_count += int(local_pairs.shape[0])
                if candidate_count > _PAIR_INDEX_MAX:
                    raise OverflowError(
                        "Arrangement candidate pairs exceed host index capacity."
                    )
                candidate_blocks.append(
                    local_pairs
                    + np.asarray(
                        (offsets[first_index], offsets[second_index]), dtype=np.int64
                    )
                )
    return (
        np.concatenate(candidate_blocks, axis=0)
        if candidate_blocks
        else np.empty((0, 2), dtype=np.int64)
    )


def _publish(
    prepared: _PreparedArrangement,
    native: NativeArrangement,
    /,
) -> SurfaceArrangement:
    surfaces = prepared.surfaces
    face_offsets = prepared.face_offsets
    owners = np.repeat(np.arange(len(surfaces), dtype=np.int32), np.diff(face_offsets))
    source_surface = owners[native.fragment_faces].astype(np.int64, copy=False)
    source_face = native.fragment_faces - face_offsets[source_surface]
    for operand in range(len(surfaces)):
        selected = np.flatnonzero(source_surface == operand)
        published = _SurfaceInput(
            native.vertices,
            native.fragments[selected],
            native.fragments[selected],
        )
        illegal, refused = _embedding_violations(published)
        offending = np.union1d(illegal, refused)
        if offending.size:
            source_rows = np.unique(source_face[selected[offending]])
            raise SurfaceArrangementError(
                "unrepresentable_publication",
                "rounded fragments do not preserve their source-surface embedding",
                surface=operand,
                faces=tuple(source_rows.tolist()),
            )
    max_int64_elements = np.iinfo(np.intp).max // np.dtype(np.int64).itemsize
    if native.vertices.shape[0] > max_int64_elements // (
        3 * len(surfaces)
    ) or native.fragments.shape[0] > max_int64_elements // len(surfaces):
        raise OverflowError("Arrangement provenance exceeds host index capacity.")
    features = np.full((native.vertices.shape[0], len(surfaces), 3), -1, dtype=np.int64)
    locations = native.locations
    for operand, surface in enumerate(surfaces):
        selected = (locations[:, 1] >= face_offsets[operand]) & (
            locations[:, 1] < face_offsets[operand + 1]
        )
        records = locations[selected]
        if records.size == 0:
            continue
        local_faces = records[:, 1] - face_offsets[operand]
        simplices = _feature_simplices(surface.source_faces[local_faces], records[:, 2])
        order = np.lexsort(
            (
                simplices[:, 2],
                simplices[:, 1],
                simplices[:, 0],
                np.sum(simplices >= 0, axis=1),
                records[:, 0],
            )
        )
        _, first = np.unique(records[order, 0], return_index=True)
        features[records[order[first], 0], operand] = simplices[order[first]]
    coincident = np.full((native.fragments.shape[0], len(surfaces)), -1, dtype=np.int64)
    orientation = np.zeros_like(coincident)
    for fragment, other_face, sign in native.coplanar:
        owner = int(owners[other_face])
        local_face = int(other_face - face_offsets[owner])
        prior = coincident[fragment, owner]
        if prior >= 0 and orientation[fragment, owner] != sign:
            raise RuntimeError(
                "Native coplanar fragments report conflicting orientations."
            )
        if prior < 0 or local_face < prior:
            coincident[fragment, owner] = local_face
            orientation[fragment, owner] = sign
    fragment_counts = np.bincount(source_surface, minlength=len(surfaces))
    per_face = np.bincount(native.fragment_faces, minlength=face_offsets[-1])
    split_faces = tuple(
        int(np.count_nonzero(per_face[face_offsets[i] : face_offsets[i + 1]] > 1))
        for i in range(len(surfaces))
    )
    feature_owners = np.sum(np.any(features >= 0, axis=-1), axis=1)
    evidence = SurfaceArrangementEvidence(
        constructed_vertices=int(np.count_nonzero(native.constructions)),
        coincident_vertices=int(np.count_nonzero(feature_owners > 1)),
        maximum_construction_bound=float(np.max(native.bounds, initial=0.0)),
        split_faces=split_faces,
        fragments=tuple(int(value) for value in fragment_counts),
    )
    return SurfaceArrangement(
        vertices=native.vertices,
        vertex_bounds=native.bounds,
        vertex_features=features,
        construction_types=native.constructions,
        vertex_locations=native.locations,
        triangles=native.fragments,
        source_surface=source_surface,
        source_face=source_face,
        coincident_face=coincident,
        coincident_orientation=orientation,
        intersection_edges=native.contact_edges,
        input_face_offsets=tuple(int(value) for value in face_offsets),
        evidence=evidence,
    )


def _arrange(
    inputs: tuple[tuple[ArrayLike, ArrayLike], ...], /, *, classify: bool
) -> tuple[SurfaceArrangement, ArrangementClassification | None]:
    prepared = _prepare_arrangement(inputs)
    pairs = _contacts(prepared)
    all_points = np.concatenate([surface.points for surface in prepared.surfaces], axis=0)
    all_faces = np.concatenate(
        [
            surface.faces + prepared.vertex_offsets[index]
            for index, surface in enumerate(prepared.surfaces)
        ],
        axis=0,
    )
    owners = np.repeat(
        np.arange(len(prepared.surfaces), dtype=np.int32),
        np.diff(prepared.face_offsets),
    )
    try:
        native = arrange_triangles(
            all_points,
            all_faces,
            owners,
            pairs,
            classification_operands=len(prepared.surfaces) if classify else None,
        )
    except ArrangementFailure as error:
        if error.status is MeshcoreStatus.RANGE_ERROR:
            owner = None if error.failed_face < 0 else int(owners[error.failed_face])
            local_face = (
                ()
                if owner is None
                else (int(error.failed_face - prepared.face_offsets[owner]),)
            )
            raise SurfaceArrangementError(
                "unrepresentable_publication",
                str(error),
                surface=owner,
                faces=local_face,
            ) from error
        raise
    arrangement = _publish(prepared, native)
    classification = native.classification
    if classify and classification is None:
        raise RuntimeError("Native arrangement did not return requested classification.")
    if not classify and classification is not None:
        raise RuntimeError("Native arrangement returned unrequested classification.")
    return arrangement, classification


def arrange_triangle_surfaces(
    first_vertices: ArrayLike,
    first_triangles: ArrayLike,
    second_vertices: ArrayLike,
    second_triangles: ArrayLike,
    /,
    *,
    operands: tuple[tuple[ArrayLike, ArrayLike], ...] = (),
) -> SurfaceArrangement:
    """Split two or more embedded triangle surfaces into a joint arrangement.

    Open sheets are supported. A surface may meet another operand at arbitrary
    triangle contacts; within each operand, contacts must be conforming shared
    vertices or edges. Every closed-AABB candidate is processed. This operates
    on the supplied tessellation only and does not recover CAD source features.
    """
    if not isinstance(operands, tuple):
        raise TypeError("operands must be an immutable tuple of vertex/triangle pairs.")
    inputs = (
        (first_vertices, first_triangles),
        (second_vertices, second_triangles),
        *operands,
    )
    if any(not isinstance(item, tuple) or len(item) != 2 for item in inputs):
        raise TypeError("Each arrangement operand must be a (vertices, triangles) tuple.")
    arrangement, _ = _arrange(inputs, classify=False)
    return arrangement


def arrange_classified_surfaces(
    inputs: tuple[tuple[ArrayLike, ArrayLike], ...], /
) -> tuple[SurfaceArrangement, ArrangementClassification]:
    """Build an arrangement and classify each fragment component natively."""
    if not isinstance(inputs, tuple) or len(inputs) < 2:
        raise ValueError("inputs must be a tuple containing at least two surfaces.")
    arrangement, classification = _arrange(inputs, classify=True)
    if classification is None:
        raise RuntimeError("Classified arrangement is missing native membership.")
    return arrangement, classification


__all__ = [
    "arrange_triangle_surfaces",
    "SurfaceArrangement",
    "SurfaceArrangementError",
    "SurfaceArrangementEvidence",
    "SurfaceArrangementStatus",
]
