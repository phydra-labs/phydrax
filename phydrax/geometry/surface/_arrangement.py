#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact n-operand triangle arrangements over original binary64 source features.

Native indirect predicates retain input-plane identities throughout insertion,
including coplanar edge intersections and triple-plane points, so every
construction is resolved exactly. Coordinates are rounded only on publication;
their outward max-norm bounds are not predicates. Binary64 coordinates that
cannot represent the exact arrangement without inverting or colliding
fragments are refused as ``unrepresentable_publication``, never repaired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import final, Literal, TypeAlias

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
    triangle_intersection_classes,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import positive_integer
from ...typing import Dim, HostFloat64, HostInt32, HostInt64, parse, Scope


SurfaceArrangementStatus: TypeAlias = Literal[
    "invalid_coordinates",
    "degenerate_triangle",
    "self_intersecting_surface",
    "limit_exceeded",
    "unrepresentable_publication",
]
_CONTACT_BATCH = 1 << 15
_LEGAL_ADJACENCY = (
    IntersectionClass.DISJOINT,
    IntersectionClass.SHARED_VERTEX,
    IntersectionClass.SHARED_EDGE,
)


class _SurfaceVertexDim(Dim, minimum=1):
    """Referenced input vertices."""


class _SurfaceTriangleDim(Dim, minimum=1):
    """Input faces."""


class _ArrangementVertexDim(Dim, minimum=3):
    """Exactly welded arrangement vertices."""


class _FragmentDim(Dim, minimum=1):
    """Oriented fragments."""


class _OperandDim(Dim, minimum=2):
    """Explicit arrangement operands."""


class _IntersectionEdgeDim(Dim):
    """Intersection edges."""


class _LocationDim(Dim):
    """Exact vertex/source-face feature incidences."""


class SurfaceArrangementError(ValueError):
    """Fail-closed refusal preserving operand and offending face identity."""

    def __init__(
        self,
        status: SurfaceArrangementStatus,
        message: str,
        /,
        *,
        surface: int | None = None,
        faces: tuple[int, ...] = (),
    ) -> None:
        self.status = parse(status, SurfaceArrangementStatus, "status")
        self.surface = surface
        self.faces = faces
        super().__init__(f"{message} (status {self.status})")


@final
class SurfaceArrangementLimits(StrictModule, NonTrainableState):
    """Shared budgets, counted across every operand, never reset per pair.

    ``maximum_incidence_records`` bounds vertex/operand simplex slots,
    fragment/operand coplanar slots and exact source-face location records.
    This prevents n-ary ancestry storage escaping the geometric work budgets.
    """

    maximum_candidate_pairs: int = eqx.field(static=True)
    maximum_vertices: int = eqx.field(static=True)
    maximum_fragments: int = eqx.field(static=True)
    maximum_winding_evaluations: int = eqx.field(static=True)
    maximum_incidence_records: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_candidate_pairs: int = 1 << 22,
        maximum_vertices: int = 1 << 22,
        maximum_fragments: int = 1 << 23,
        maximum_winding_evaluations: int = 1 << 30,
        maximum_incidence_records: int = 1 << 24,
    ) -> None:
        self.maximum_candidate_pairs = positive_integer(
            maximum_candidate_pairs, "maximum_candidate_pairs"
        )
        self.maximum_vertices = positive_integer(maximum_vertices, "maximum_vertices")
        self.maximum_fragments = positive_integer(maximum_fragments, "maximum_fragments")
        self.maximum_winding_evaluations = positive_integer(
            maximum_winding_evaluations, "maximum_winding_evaluations"
        )
        self.maximum_incidence_records = positive_integer(
            maximum_incidence_records, "maximum_incidence_records"
        )


@final
class SurfaceArrangementEvidence(StrictModule, NonTrainableState):
    """Work, original-feature constructions and publication-error evidence."""

    self_candidate_pairs: int = eqx.field(static=True)
    candidate_pairs: int = eqx.field(static=True)
    publication_candidate_pairs: int = eqx.field(static=True)
    contact_counts: tuple[int, ...] = eqx.field(static=True)
    constructed_vertices: int = eqx.field(static=True)
    coincident_vertices: int = eqx.field(static=True)
    maximum_construction_bound: float = eqx.field(static=True)
    split_faces: tuple[int, ...] = eqx.field(static=True)
    fragments: tuple[int, ...] = eqx.field(static=True)
    triple_plane_vertices: int = eqx.field(static=True)
    exact_predicates: int = eqx.field(static=True)
    filtered_predicates: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        self_candidate_pairs: int,
        candidate_pairs: int,
        contact_counts: tuple[int, ...],
        constructed_vertices: int,
        coincident_vertices: int,
        maximum_construction_bound: float,
        split_faces: tuple[int, ...],
        fragments: tuple[int, ...],
        triple_plane_vertices: int,
        exact_predicates: int,
        filtered_predicates: int,
        publication_candidate_pairs: int,
    ) -> None:
        if len(contact_counts) != len(IntersectionClass) or min(contact_counts) < 0:
            raise ValueError("contact_counts needs one count per IntersectionClass.")
        if len(split_faces) != len(fragments) or len(fragments) < 2:
            raise ValueError("Work counts need one entry per operand.")
        if (
            not np.isfinite(maximum_construction_bound)
            or maximum_construction_bound < 0.0
        ):
            raise ValueError("Construction bound must be finite and nonnegative.")
        self.self_candidate_pairs = self_candidate_pairs
        self.candidate_pairs = candidate_pairs
        self.publication_candidate_pairs = publication_candidate_pairs
        self.contact_counts = contact_counts
        self.constructed_vertices = constructed_vertices
        self.coincident_vertices = coincident_vertices
        self.maximum_construction_bound = maximum_construction_bound
        self.split_faces = split_faces
        self.fragments = fragments
        self.triple_plane_vertices = triple_plane_vertices
        self.exact_predicates = exact_predicates
        self.filtered_predicates = filtered_predicates
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "surface-arrangement-evidence",
                "self_pairs": self_candidate_pairs,
                "pairs": candidate_pairs,
                "contacts": contact_counts,
                "publication_pairs": publication_candidate_pairs,
                "constructed": constructed_vertices,
                "coincident": coincident_vertices,
                "bound": maximum_construction_bound,
                "split_faces": split_faces,
                "fragments": fragments,
                "triple_planes": triple_plane_vertices,
                "exact_predicates": exact_predicates,
                "filtered_predicates": filtered_predicates,
            }
        )


@final
class SurfaceArrangement(StrictModule, NonTrainableState):
    """Conforming n-operand fragments and exact original feature incidences.

    ``vertex_features[v,i]`` is the sorted, -1-padded original vertex simplex
    of operand i carrying v. ``vertex_locations`` retains *all* source-face
    incidences, including a point on an internal triangulation edge.
    ``coincident_face[f,i]`` names a covering coplanar face of operand i,
    with orientation relative to fragment f; the owning operand is -1/0.
    """

    __strict_contract__ = True
    vertices: HostFloat64[_ArrangementVertexDim, Literal[3]]
    vertex_bounds: HostFloat64[_ArrangementVertexDim]
    vertex_features: HostInt64[_ArrangementVertexDim, _OperandDim, Literal[3]]
    construction_types: HostInt32[_ArrangementVertexDim]
    vertex_locations: HostInt64[_LocationDim, Literal[3]]
    triangles: HostInt64[_FragmentDim, Literal[3]]
    source_surface: HostInt64[_FragmentDim]
    source_face: HostInt64[_FragmentDim]
    coincident_face: HostInt64[_FragmentDim, _OperandDim]
    coincident_orientation: HostInt64[_FragmentDim, _OperandDim]
    intersection_edges: HostInt64[_IntersectionEdgeDim, Literal[2]]
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
        scope = Scope()
        points = parse(
            np.asarray(vertices, dtype=np.float64),
            HostFloat64[_ArrangementVertexDim, Literal[3]],
            "vertices",
            scope=scope,
        )
        bounds = parse(
            np.asarray(vertex_bounds, dtype=np.float64),
            HostFloat64[_ArrangementVertexDim],
            "vertex_bounds",
            scope=scope,
        )
        features = parse(
            np.asarray(vertex_features, dtype=np.int64),
            HostInt64[_ArrangementVertexDim, _OperandDim, Literal[3]],
            "vertex_features",
            scope=scope,
        )
        constructions = parse(
            np.asarray(construction_types, dtype=np.int32),
            HostInt32[_ArrangementVertexDim],
            "construction_types",
            scope=scope,
        )
        locations = parse(
            np.asarray(vertex_locations, dtype=np.int64),
            HostInt64[_LocationDim, Literal[3]],
            "vertex_locations",
            scope=scope,
        )
        faces = parse(
            np.asarray(triangles, dtype=np.int64),
            HostInt64[_FragmentDim, Literal[3]],
            "triangles",
            scope=scope,
        )
        surfaces = parse(
            np.asarray(source_surface, dtype=np.int64),
            HostInt64[_FragmentDim],
            "source_surface",
            scope=scope,
        )
        sources = parse(
            np.asarray(source_face, dtype=np.int64),
            HostInt64[_FragmentDim],
            "source_face",
            scope=scope,
        )
        coincident = parse(
            np.asarray(coincident_face, dtype=np.int64),
            HostInt64[_FragmentDim, _OperandDim],
            "coincident_face",
            scope=scope,
        )
        orientation = parse(
            np.asarray(coincident_orientation, dtype=np.int64),
            HostInt64[_FragmentDim, _OperandDim],
            "coincident_orientation",
            scope=scope,
        )
        edges = parse(
            np.asarray(intersection_edges, dtype=np.int64),
            HostInt64[_IntersectionEdgeDim, Literal[2]],
            "intersection_edges",
            scope=scope,
        )
        count = features.shape[1]
        if not isinstance(evidence, SurfaceArrangementEvidence):
            raise TypeError("evidence must be SurfaceArrangementEvidence.")
        if (
            len(input_face_offsets) != count + 1
            or input_face_offsets[0] != 0
            or any(a >= b for a, b in zip(input_face_offsets, input_face_offsets[1:]))
        ):
            raise ValueError("input_face_offsets must delimit every nonempty operand.")
        if (
            not np.all(np.isfinite(points))
            or not np.all(np.isfinite(bounds))
            or np.any(bounds < 0.0)
        ):
            raise ValueError("Arrangement publication must have finite outward bounds.")
        if (
            np.any(faces < 0)
            or np.any(faces >= points.shape[0])
            or np.any(edges < 0)
            or np.any(edges >= points.shape[0])
        ):
            raise ValueError("Arrangement connectivity indexes missing vertices.")
        if (
            np.any(surfaces < 0)
            or np.any(surfaces >= count)
            or np.any(sources < 0)
            or np.any(features < -1)
        ):
            raise ValueError("Arrangement source feature identity is invalid.")
        offsets = np.asarray(input_face_offsets, dtype=np.int64)
        if np.any(sources >= np.diff(offsets)[surfaces]):
            raise ValueError("Arrangement ancestry indexes a missing original face.")
        if (
            np.any(coincident < -1)
            or np.any(np.abs(orientation) > 1)
            or np.any((coincident < 0) != (orientation == 0))
            or np.any(coincident >= np.diff(offsets)[None, :])
        ):
            raise ValueError("Coplanar coverage and orientation disagree.")
        if np.any(constructions < 0) or np.any(constructions > 3):
            raise ValueError("Unknown native point construction family.")
        if (
            np.any(locations[:, 0] < 0)
            or np.any(locations[:, 0] >= points.shape[0])
            or np.any(locations[:, 1] < 0)
            or np.any(locations[:, 1] >= offsets[-1])
            or np.any(locations[:, 2] < 0)
            or np.any(locations[:, 2] > 6)
        ):
            raise ValueError("Exact feature incidences index missing source features.")
        self.vertices = points
        self.vertex_bounds = bounds
        self.vertex_features = features
        self.construction_types = constructions
        self.vertex_locations = locations
        self.triangles = faces
        self.source_surface = surfaces
        self.source_face = sources
        self.coincident_face = coincident
        self.coincident_orientation = orientation
        self.intersection_edges = edges
        self.input_face_offsets = input_face_offsets
        self.evidence = evidence
        self.arrangement_id = canonical_fingerprint(
            {
                "kind": "surface-arrangement",
                "arrays": array_tree_fingerprint(
                    (
                        self.vertices,
                        self.vertex_bounds,
                        self.vertex_features,
                        self.construction_types,
                        self.vertex_locations,
                        self.triangles,
                        self.source_surface,
                        self.source_face,
                        self.coincident_face,
                        self.coincident_orientation,
                        self.intersection_edges,
                    )
                ),
                "face_offsets": input_face_offsets,
                "evidence": evidence.evidence_id,
            }
        )


@dataclass(frozen=True, slots=True)
class _Surface:
    """One validated input surface over its referenced vertices only."""

    points: np.ndarray  # referenced input vertices, shape (n, 3)
    faces: np.ndarray  # faces over referenced vertices, shape (m, 3)
    original: np.ndarray  # input id of each referenced vertex, shape (n,)


def _prepared_surface(
    vertices: ArrayLike, triangles: ArrayLike, index: int, /
) -> _Surface:
    scope = Scope()
    points = parse(
        np.asarray(vertices, dtype=np.float64),
        HostFloat64[_SurfaceVertexDim, Literal[3]],
        "vertices",
        scope=scope,
    )
    faces = parse(
        np.asarray(triangles, dtype=np.int64),
        HostInt64[_SurfaceTriangleDim, Literal[3]],
        "triangles",
        scope=scope,
    )
    if not np.all(np.isfinite(points)):
        raise SurfaceArrangementError(
            "invalid_coordinates", "Surface vertices must be finite.", surface=index
        )
    if np.any(faces < 0) or np.any(faces >= points.shape[0]):
        raise ValueError("Surface triangles index missing vertices.")
    minimum, maximum = meshcore_exact_domain()
    smallest = np.ldexp(np.float64(1.0), minimum)
    largest = np.ldexp(np.float64(1.0), maximum)
    magnitude = np.abs(points[faces])
    outside = np.any(
        (magnitude != 0.0) & ((magnitude < smallest) | (magnitude > largest)),
        axis=(1, 2),
    )
    if np.any(outside):
        raise SurfaceArrangementError(
            "invalid_coordinates",
            "Surface coordinates must be zero or have magnitude in "
            f"[2**{minimum}, 2**{maximum}] for exact contact predicates.",
            surface=index,
            faces=tuple(np.flatnonzero(outside).tolist()),
        )
    repeated = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    )
    corners = points[faces]
    collinear = np.ones((faces.shape[0],), dtype=np.bool_)
    for first, second in ((0, 1), (1, 2), (2, 0)):
        plane = corners[..., (first, second)]
        collinear &= exact_orient2d(plane[:, 0], plane[:, 1], plane[:, 2]) == 0
    degenerate = np.flatnonzero(repeated | collinear)
    if degenerate.size:
        raise SurfaceArrangementError(
            "degenerate_triangle",
            "Arrangement triangles must have three non-collinear corners.",
            surface=index,
            faces=tuple(degenerate.tolist()),
        )
    original, compact = np.unique(faces, return_inverse=True)
    return _Surface(
        points[original], compact.reshape(faces.shape).astype(np.int64), original
    )


def _candidate_pairs(
    first: _Surface, second: _Surface, capacity: int, /, *, self_pairs: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-box candidate pairs in (first, second) order, within ``capacity``."""

    first_triangles = first.points[first.faces]
    first_bvh = prepare_bvh(
        np.min(first_triangles, axis=1), np.max(first_triangles, axis=1), dtype=np.float64
    )
    if self_pairs:
        second_bvh = first_bvh
    else:
        second_triangles = second.points[second.faces]
        second_bvh = prepare_bvh(
            np.min(second_triangles, axis=1),
            np.max(second_triangles, axis=1),
            dtype=np.float64,
        )
    firsts = [np.empty((0,), dtype=np.int64)]
    seconds = [np.empty((0,), dtype=np.int64)]
    total = 0
    for block_first, block_second in bvh_overlap_pair_blocks(
        first_bvh, second_bvh, include_touching=True
    ):
        if self_pairs:
            keep = block_first < block_second
            block_first, block_second = block_first[keep], block_second[keep]
        total += block_first.size
        if total > capacity:
            raise SurfaceArrangementError(
                "limit_exceeded",
                f"Arrangement candidate pairs exceed maximum_candidate_pairs={capacity}.",
            )
        firsts.append(block_first.astype(np.int64))
        seconds.append(block_second.astype(np.int64))
    pairs_first = np.concatenate(firsts)
    pairs_second = np.concatenate(seconds)
    order = np.lexsort((pairs_second, pairs_first))
    return pairs_first[order], pairs_second[order]


def _embedding_violations(
    surface: _Surface, capacity: int, /
) -> tuple[int, np.ndarray, np.ndarray]:
    """Candidate pairs, faces in illegal contact and faces the predicates refused.

    Illegal contacts are anything beyond shared vertices and edges; refused
    faces lie outside the exact predicate domain of the contact classifier.
    """

    first, second = _candidate_pairs(surface, surface, capacity, self_pairs=True)
    illegal = [np.empty((0,), dtype=np.int64)]
    refused = [np.empty((0,), dtype=np.int64)]
    for start in range(0, first.size, _CONTACT_BATCH):
        rows = slice(start, start + _CONTACT_BATCH)
        classes, status = triangle_intersection_classes(
            surface.points[surface.faces[first[rows]]],
            surface.points[surface.faces[second[rows]]],
            first_ids=surface.faces[first[rows]],
            second_ids=surface.faces[second[rows]],
        )
        failed = status != MeshcoreStatus.OK
        crossing = ~failed & ~np.isin(classes, _LEGAL_ADJACENCY)
        illegal += [first[rows][crossing], second[rows][crossing]]
        refused += [first[rows][failed], second[rows][failed]]
    return (
        first.size,
        np.unique(np.concatenate(illegal)),
        np.unique(np.concatenate(refused)),
    )


def _require_embedded(surface: _Surface, index: int, capacity: int, /) -> int:
    """Refuse contacts between triangles of one surface beyond shared features."""

    pairs, illegal, refused = _embedding_violations(surface, capacity)
    if refused.size:
        raise SurfaceArrangementError(
            "invalid_coordinates",
            "Surface coordinates lie outside the exact contact-predicate domain.",
            surface=index,
            faces=tuple(refused.tolist()),
        )
    if illegal.size:
        raise SurfaceArrangementError(
            "self_intersecting_surface",
            "Arrangement surfaces must be embedded: triangles may meet only "
            "in shared vertices and edges.",
            surface=index,
            faces=tuple(illegal.tolist()),
        )
    return pairs


def _require_published_embedded(
    vertices: np.ndarray,
    triangles: np.ndarray,
    reported_faces: np.ndarray,
    capacity: int,
    /,
    *,
    surface: int | None,
) -> int:
    """Certify that rounded publication of exact fragments is still embedded.

    ``reported_faces[k]`` names the face reported for triangle ``k`` when the
    binary64 coordinates cannot represent the exact embedding.
    """

    published = _Surface(
        vertices, triangles, np.arange(vertices.shape[0], dtype=np.int64)
    )
    pairs, illegal, refused = _embedding_violations(published, capacity)
    offending = np.union1d(illegal, refused)
    if offending.size:
        raise SurfaceArrangementError(
            "unrepresentable_publication",
            "Binary64 publication of the exact arrangement is not embedded.",
            surface=surface,
            faces=tuple(np.unique(reported_faces[offending]).tolist()),
        )
    return pairs


def _feature_simplices(faces: np.ndarray, features: np.ndarray, /) -> np.ndarray:
    """Sorted ``-1``-padded vertex ids of the simplex carrying each triangle feature.

    Features are 0-2 for vertex ``j``, 3-5 for the edge opposite vertex ``j - 3``
    and 6 for the relative interior.
    """

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


def arrange_triangle_surfaces(
    first_vertices: ArrayLike,
    first_triangles: ArrayLike,
    second_vertices: ArrayLike,
    second_triangles: ArrayLike,
    /,
    *,
    operands: tuple[tuple[ArrayLike, ArrayLike], ...] = (),
    limits: SurfaceArrangementLimits | None = None,
) -> SurfaceArrangement:
    """Split embedded surfaces in one exact n-ary arrangement.

    Extra operands join the *same* construction pool, never a sequential
    Boolean of rounded intermediate meshes. Open embedded sheets are admitted.
    """
    limits_ = SurfaceArrangementLimits() if limits is None else limits
    if not isinstance(limits_, SurfaceArrangementLimits):
        raise TypeError("limits must be SurfaceArrangementLimits or None.")
    inputs = (
        (first_vertices, first_triangles),
        (second_vertices, second_triangles),
        *operands,
    )
    return _arrange(inputs, limits_, None)[0]


def _arrange(
    inputs: tuple[tuple[ArrayLike, ArrayLike], ...],
    limits_: SurfaceArrangementLimits,
    classification_work_limit: int | None,
    /,
) -> tuple[SurfaceArrangement, ArrangementClassification | None]:
    """Exact arrangement and, with a work limit, exact fragment membership.

    The membership is decided natively on the implicit construction points
    (see :class:`ArrangementClassification`), never on rounded coordinates.
    """
    surfaces = tuple(_prepared_surface(v, t, i) for i, (v, t) in enumerate(inputs))
    vertex_offsets = np.r_[
        0, np.cumsum([s.points.shape[0] for s in surfaces], dtype=np.int64)
    ]
    face_offsets = np.r_[
        0, np.cumsum([s.faces.shape[0] for s in surfaces], dtype=np.int64)
    ]
    if vertex_offsets[-1] > limits_.maximum_vertices:
        raise SurfaceArrangementError(
            "limit_exceeded", "Input vertices exceed maximum_vertices."
        )
    total_faces = sum(surface.faces.shape[0] for surface in surfaces)
    minimum_records = len(surfaces) * (total_faces + 3) + 3 * total_faces
    if minimum_records > limits_.maximum_incidence_records:
        raise SurfaceArrangementError(
            "limit_exceeded", "Original ancestry exceeds maximum_incidence_records."
        )
    self_pairs = 0
    for index, surface in enumerate(surfaces):
        self_pairs += _require_embedded(
            surface, index, limits_.maximum_candidate_pairs - self_pairs
        )
    pairs = [np.empty((0, 2), dtype=np.int64)]
    counts = np.zeros((len(IntersectionClass),), dtype=np.int64)
    spent = self_pairs
    for first, a in enumerate(surfaces):
        for second in range(first + 1, len(surfaces)):
            b = surfaces[second]
            left, right = _candidate_pairs(
                a, b, limits_.maximum_candidate_pairs - spent, self_pairs=False
            )
            spent += left.size
            pairs.append(
                np.stack(
                    (left + face_offsets[first], right + face_offsets[second]), axis=1
                )
            )
            for start in range(0, left.size, _CONTACT_BATCH):
                rows = slice(start, start + _CONTACT_BATCH)
                classes, status = triangle_intersection_classes(
                    a.points[a.faces[left[rows]]], b.points[b.faces[right[rows]]]
                )
                refused = np.flatnonzero(status != MeshcoreStatus.OK)
                if refused.size:
                    raise SurfaceArrangementError(
                        "invalid_coordinates",
                        "Contact coordinates lie outside the exact predicate domain.",
                        surface=first,
                        faces=tuple(np.unique(left[rows][refused]).tolist()),
                    )
                counts += np.bincount(classes, minlength=len(IntersectionClass))
    points = np.concatenate([s.points for s in surfaces])
    faces = np.concatenate([s.faces + vertex_offsets[i] for i, s in enumerate(surfaces)])
    owners = np.repeat(np.arange(len(surfaces), dtype=np.int32), np.diff(face_offsets))
    try:
        native = arrange_triangles(
            points,
            faces,
            owners,
            np.concatenate(pairs),
            maximum_vertices=limits_.maximum_vertices,
            maximum_fragments=limits_.maximum_fragments,
            classification_operands=(
                None if classification_work_limit is None else len(surfaces)
            ),
            classification_work_limit=classification_work_limit,
        )
    except ArrangementFailure as error:
        # Only declared scientific refusals become arrangement statuses; any
        # other native status is a kernel defect and propagates unchanged.
        match error.status:
            case MeshcoreStatus.CAPACITY_EXCEEDED:
                status: SurfaceArrangementStatus = "limit_exceeded"
            case MeshcoreStatus.RANGE_ERROR:
                status = "unrepresentable_publication"
            case _:
                raise
        operand = None if error.failed_face < 0 else int(owners[error.failed_face])
        local = (
            () if operand is None else (int(error.failed_face - face_offsets[operand]),)
        )
        raise SurfaceArrangementError(
            status, str(error), surface=operand, faces=local
        ) from error
    records = (native.vertices.shape[0] + native.fragments.shape[0]) * len(
        surfaces
    ) + native.locations.shape[0]
    if records > limits_.maximum_incidence_records:
        raise SurfaceArrangementError(
            "limit_exceeded", "Exact ancestry exceeds maximum_incidence_records."
        )
    source_surface = owners[native.fragment_faces].astype(np.int64)
    source_face = native.fragment_faces - face_offsets[source_surface]
    publication_pairs = 0
    for operand in range(len(surfaces)):
        selected = np.flatnonzero(source_surface == operand)
        publication_pairs += _require_published_embedded(
            native.vertices,
            native.fragments[selected],
            source_face[selected],
            limits_.maximum_candidate_pairs - spent - publication_pairs,
            surface=operand,
        )
    features = np.full((native.vertices.shape[0], len(surfaces), 3), -1, dtype=np.int64)
    # Exact native location codes refer to original input faces, not rounded geometry.
    for operand, surface in enumerate(surfaces):
        loc = native.locations
        selected = (loc[:, 1] >= face_offsets[operand]) & (
            loc[:, 1] < face_offsets[operand + 1]
        )
        records = loc[selected]
        simplices = _feature_simplices(
            surface.original[surface.faces[records[:, 1] - face_offsets[operand]]],
            records[:, 2],
        )
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
        owner = owners[other_face]
        local_face = other_face - face_offsets[owner]
        prior = coincident[fragment, owner]
        if prior >= 0 and orientation[fragment, owner] != sign:
            raise RuntimeError(
                "Native coplanar coverage gave one fragment two orientations "
                "against a single operand."
            )
        if prior < 0 or local_face < prior:
            coincident[fragment, owner] = local_face
            orientation[fragment, owner] = sign
    fragment_counts = np.bincount(source_surface, minlength=len(surfaces))
    per_face = np.bincount(native.fragment_faces, minlength=faces.shape[0])
    split = tuple(
        int(np.count_nonzero(per_face[face_offsets[i] : face_offsets[i + 1]] > 1))
        for i in range(len(surfaces))
    )
    evidence = SurfaceArrangementEvidence(
        self_candidate_pairs=self_pairs,
        candidate_pairs=spent - self_pairs,
        publication_candidate_pairs=publication_pairs,
        contact_counts=tuple(counts.tolist()),
        constructed_vertices=int(np.count_nonzero(native.constructions)),
        coincident_vertices=int(native.counters[3]),
        maximum_construction_bound=float(np.max(native.bounds, initial=0.0)),
        split_faces=split,
        fragments=tuple(fragment_counts.tolist()),
        triple_plane_vertices=int(native.counters[6]),
        exact_predicates=int(native.counters[10]),
        filtered_predicates=int(native.counters[9]),
    )
    arrangement = SurfaceArrangement(
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
        input_face_offsets=tuple(face_offsets.tolist()),
        evidence=evidence,
    )
    return arrangement, native.classification


__all__ = [
    "arrange_triangle_surfaces",
    "SurfaceArrangement",
    "SurfaceArrangementError",
    "SurfaceArrangementEvidence",
    "SurfaceArrangementLimits",
    "SurfaceArrangementStatus",
]
