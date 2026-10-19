#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Build joint n-operand triangle arrangements from input binary64 source features.

Native indirect predicates retain input-plane identities throughout insertion,
including coplanar edge intersections and triple-plane points, so constructions
are resolved in rational arithmetic. Coordinates are rounded only on publication;
their outward max-norm bounds are excluded from predicate inputs. Binary64
publication that would invert or collide fragments is refused as
``unrepresentable_publication``."""

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
    NativeArrangement,
    triangle_intersection_classes,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import positive_integer
from ...typing import checked, Dim, HostFloat64, HostInt32, HostInt64, parse, Scope


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
    """Arrangement vertices welded by their source-feature identities."""


class _FragmentDim(Dim, minimum=1):
    """Oriented fragments."""


class _OperandDim(Dim, minimum=2):
    """Explicit arrangement operands."""


class _IntersectionEdgeDim(Dim):
    """Intersection edges."""


class _LocationDim(Dim):
    """Vertex/source-face feature incidences determined by exact predicates."""


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
    """Shared budgets counted across all operands without per-pair resets.

    ``maximum_incidence_records`` bounds vertex/operand simplex slots,
    fragment/operand coplanar slots and rational source-face location records.
    This bounds n-ary ancestry storage within the geometric work budgets."""

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
    """Work, input-feature constructions and publication-error evidence."""

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
            raise ValueError(
                f"SurfaceArrangementEvidence.contact_counts must contain one nonnegative count per IntersectionClass; got {contact_counts!r}"
            )
        if len(split_faces) != len(fragments) or len(fragments) < 2:
            raise ValueError(
                f"SurfaceArrangementEvidence.fragments must match split_faces and include at least two operands; got fragments={len(fragments)}, split_faces={len(split_faces)}"
            )
        if (
            not np.isfinite(maximum_construction_bound)
            or maximum_construction_bound < 0.0
        ):
            raise ValueError(
                f"SurfaceArrangementEvidence.maximum_construction_bound must be finite and nonnegative; got {maximum_construction_bound!r}"
            )
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
    """Conforming n-operand fragments with rational input-feature incidences.

    ``vertex_features[v,i]`` is the sorted, -1-padded input vertex simplex
    of operand i carrying v. ``vertex_locations`` retains all source-face
    incidences, including a point on an internal triangulation edge.
    ``coincident_face[f,i]`` names a covering coplanar face of operand i,
    with orientation relative to fragment f. The fragment's own source-operand
    column has coincidence -1 and orientation 0."""

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

    @checked
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
        if (
            len(input_face_offsets) != count + 1
            or input_face_offsets[0] != 0
            or any(a >= b for a, b in zip(input_face_offsets, input_face_offsets[1:]))
        ):
            raise ValueError(
                f"SurfaceArrangement.input_face_offsets must start at zero and delimit every nonempty operand; got {input_face_offsets!r} for {count} operands"
            )
        if (
            not np.all(np.isfinite(points))
            or not np.all(np.isfinite(bounds))
            or np.any(bounds < 0.0)
        ):
            raise ValueError(
                f"SurfaceArrangement.vertices must be finite with finite nonnegative publication bounds; got points_shape={points.shape}, bounds_shape={bounds.shape}"
            )
        if (
            np.any(faces < 0)
            or np.any(faces >= points.shape[0])
            or np.any(edges < 0)
            or np.any(edges >= points.shape[0])
        ):
            raise ValueError(
                f"SurfaceArrangement.connectivity must index declared vertices; got faces_shape={faces.shape}, edges_shape={edges.shape}, vertices={points.shape[0]}"
            )
        if (
            np.any(surfaces < 0)
            or np.any(surfaces >= count)
            or np.any(sources < 0)
            or np.any(features < -1)
        ):
            raise ValueError(
                f"SurfaceArrangement.source_surface must name a declared operand with nonnegative source faces and vertex features at least -1; got operands={count}, source_shape={surfaces.shape}, feature_shape={features.shape}"
            )
        offsets = np.asarray(input_face_offsets, dtype=np.int64)
        if np.any(sources >= np.diff(offsets)[surfaces]):
            raise ValueError(
                f"SurfaceArrangement.source_face must index a declared input face of its operand; got source_shape={sources.shape}, offsets={input_face_offsets!r}"
            )
        if (
            np.any(coincident < -1)
            or np.any(np.abs(orientation) > 1)
            or np.any((coincident < 0) != (orientation == 0))
            or np.any(coincident >= np.diff(offsets)[None, :])
        ):
            raise ValueError(
                f"SurfaceArrangement.coincident_face must name input faces or -1 and agree with orientations in -1,0,1; got coverage_shape={coincident.shape}, orientation_shape={orientation.shape}"
            )
        if np.any(constructions < 0) or np.any(constructions > 3):
            raise ValueError(
                f"SurfaceArrangement.construction_types must use native construction codes from 0 through 3; got {np.unique(constructions).tolist()!r}"
            )
        if (
            np.any(locations[:, 0] < 0)
            or np.any(locations[:, 0] >= points.shape[0])
            or np.any(locations[:, 1] < 0)
            or np.any(locations[:, 1] >= offsets[-1])
            or np.any(locations[:, 2] < 0)
            or np.any(locations[:, 2] > 6)
        ):
            raise ValueError(
                f"SurfaceArrangement.vertex_locations must index declared vertices and input faces with feature codes from 0 through 6; got shape={locations.shape}, vertices={points.shape[0]}, input_faces={offsets[-1]}"
            )
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
            "invalid_coordinates",
            f"arrange_triangle_surfaces.vertices must be finite; got nonfinite coordinates for surface={index}",
            surface=index,
        )
    if np.any(faces < 0) or np.any(faces >= points.shape[0]):
        raise ValueError(
            f"arrange_triangle_surfaces.triangles must index declared vertices; got surface={index}, vertices={points.shape[0]}, faces_shape={faces.shape}"
        )
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
            f"arrange_triangle_surfaces.vertices must be zero or have magnitude in [2**{minimum},2**{maximum}] for exact predicates; got surface={index}, out_of_range_mask_shape={outside.shape}",
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
            f"arrange_triangle_surfaces.triangles must have three noncollinear corners; got surface={index}, degenerate_faces={degenerate.tolist()!r}",
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
                f"arrange_triangle_surfaces.maximum_candidate_pairs must admit all closed-box candidates; got requested={total}, limit={capacity}",
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
            f"arrange_triangle_surfaces.vertices must lie within the contact classifier domain; got surface={index}, refused_faces={refused.tolist()!r}",
            surface=index,
            faces=tuple(refused.tolist()),
        )
    if illegal.size:
        raise SurfaceArrangementError(
            "self_intersecting_surface",
            f"arrange_triangle_surfaces.embedding must allow triangle contacts only on shared vertices and edges; got surface={index}, illegal_faces={illegal.tolist()!r}",
            surface=index,
            faces=tuple(illegal.tolist()),
        )
    return pairs


def require_published_embedded(
    vertices: np.ndarray,
    triangles: np.ndarray,
    reported_faces: np.ndarray,
    capacity: int,
    /,
    *,
    surface: int | None,
) -> int:
    """Certify that rounded publication of rational fragments remains embedded.

    ``reported_faces[k]`` names the face reported for triangle ``k`` when
    binary64 publication would violate the source arrangement's embedding."""

    published = _Surface(
        vertices, triangles, np.arange(vertices.shape[0], dtype=np.int64)
    )
    pairs, illegal, refused = _embedding_violations(published, capacity)
    offending = np.union1d(illegal, refused)
    if offending.size:
        raise SurfaceArrangementError(
            "unrepresentable_publication",
            f"require_published_embedded.vertices must preserve fragment embedding after binary64 rounding; got surface={surface}, offending_triangles={offending.tolist()!r}",
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


@checked
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
    """Split embedded surfaces in one joint rational n-ary arrangement.

    All operands join the same construction pool; coordinates are rounded only
    when that joint arrangement is published. Open embedded sheets are admitted."""
    limits_ = SurfaceArrangementLimits() if limits is None else limits
    inputs = (
        (first_vertices, first_triangles),
        (second_vertices, second_triangles),
        *operands,
    )
    prepared = _prepare_arrangement(inputs, limits_)
    contacts = _arrangement_contacts(prepared, limits_)
    native = _execute_arrangement(prepared, contacts, limits_, None)
    return _publish_arrangement(prepared, contacts, native, limits_)


@final
class _ArrangementPreparation(StrictModule):
    """Referenced surfaces and offsets preserving operand and face order."""

    surfaces: tuple[_Surface, ...]
    vertex_offsets: np.ndarray
    face_offsets: np.ndarray


@final
class _ArrangementContacts(StrictModule):
    """Validated input embedding and cross-operand contact work."""

    pairs: np.ndarray
    counts: np.ndarray
    self_pairs: int
    spent: int


@final
class _ArrangementExecution(StrictModule):
    """Native fragments paired with their input operand identities."""

    native: NativeArrangement
    owners: np.ndarray


@final
class ClassifiedSurfaceArrangement(StrictModule):
    """Published fragments and the requested native membership result."""

    arrangement: SurfaceArrangement
    classification: ArrangementClassification


@checked
def arrange_classified_surfaces(
    inputs: tuple[tuple[ArrayLike, ArrayLike], ...],
    limits_: SurfaceArrangementLimits,
    classification_work_limit: int,
    /,
) -> ClassifiedSurfaceArrangement:
    """Arrange surfaces and retain membership, including capacity refusal evidence."""
    prepared = _prepare_arrangement(inputs, limits_)
    contacts = _arrangement_contacts(prepared, limits_)
    native = _execute_arrangement(prepared, contacts, limits_, classification_work_limit)
    arrangement = _publish_arrangement(prepared, contacts, native, limits_)
    match native.native.classification:
        case ArrangementClassification() as classification:
            return ClassifiedSurfaceArrangement(arrangement, classification)
        case None:
            raise RuntimeError(
                "arrange_classified_surfaces.classification must include the requested native membership result; got no classification"
            )


def _prepare_arrangement(
    inputs: tuple[tuple[ArrayLike, ArrayLike], ...],
    limits_: SurfaceArrangementLimits,
    /,
) -> _ArrangementPreparation:
    """Validate source carriers and admit their minimum ancestry storage."""
    surfaces = tuple(_prepared_surface(v, t, i) for i, (v, t) in enumerate(inputs))
    vertex_offsets = np.r_[
        0, np.cumsum([s.points.shape[0] for s in surfaces], dtype=np.int64)
    ]
    face_offsets = np.r_[
        0, np.cumsum([s.faces.shape[0] for s in surfaces], dtype=np.int64)
    ]
    if vertex_offsets[-1] > limits_.maximum_vertices:
        raise SurfaceArrangementError(
            "limit_exceeded",
            f"arrange_triangle_surfaces.maximum_vertices must admit the input vertex count; got requested={vertex_offsets[-1]}, limit={limits_.maximum_vertices}",
        )
    total_faces = sum(surface.faces.shape[0] for surface in surfaces)
    minimum_records = len(surfaces) * (total_faces + 3) + 3 * total_faces
    if minimum_records > limits_.maximum_incidence_records:
        raise SurfaceArrangementError(
            "limit_exceeded",
            f"arrange_triangle_surfaces.maximum_incidence_records must admit input ancestry; got requested={minimum_records}, limit={limits_.maximum_incidence_records}",
        )
    return _ArrangementPreparation(surfaces, vertex_offsets, face_offsets)


def _arrangement_contacts(
    prepared: _ArrangementPreparation, limits_: SurfaceArrangementLimits, /
) -> _ArrangementContacts:
    """Certify input embedding before evaluating cross-operand contact classes."""
    surfaces, face_offsets = prepared.surfaces, prepared.face_offsets
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
                        f"arrange_triangle_surfaces.vertices must lie within the contact classifier domain; got surface={first}, refusal_mask_shape={refused.shape}",
                        surface=first,
                        faces=tuple(np.unique(left[rows][refused]).tolist()),
                    )
                counts += np.bincount(classes, minlength=len(IntersectionClass))
    return _ArrangementContacts(np.concatenate(pairs), counts, self_pairs, spent)


def _execute_arrangement(
    prepared: _ArrangementPreparation,
    contacts: _ArrangementContacts,
    limits_: SurfaceArrangementLimits,
    classification_work_limit: int | None,
    /,
) -> _ArrangementExecution:
    """Execute native construction and map declared refusals to source faces."""
    surfaces = prepared.surfaces
    vertex_offsets, face_offsets = prepared.vertex_offsets, prepared.face_offsets
    points = np.concatenate([s.points for s in surfaces])
    faces = np.concatenate([s.faces + vertex_offsets[i] for i, s in enumerate(surfaces)])
    owners = np.repeat(np.arange(len(surfaces), dtype=np.int32), np.diff(face_offsets))
    try:
        native = arrange_triangles(
            points,
            faces,
            owners,
            contacts.pairs,
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
    return _ArrangementExecution(native, owners)


def _publish_arrangement(
    prepared: _ArrangementPreparation,
    contacts: _ArrangementContacts,
    execution: _ArrangementExecution,
    limits_: SurfaceArrangementLimits,
    /,
) -> SurfaceArrangement:
    """Certify rounded embedding and publish ancestry with measured work."""
    surfaces, face_offsets = prepared.surfaces, prepared.face_offsets
    native, owners = execution.native, execution.owners
    self_pairs, spent, counts = contacts.self_pairs, contacts.spent, contacts.counts
    records = (native.vertices.shape[0] + native.fragments.shape[0]) * len(
        surfaces
    ) + native.locations.shape[0]
    if records > limits_.maximum_incidence_records:
        raise SurfaceArrangementError(
            "limit_exceeded",
            f"arrange_triangle_surfaces.maximum_incidence_records must admit constructed ancestry; got requested={records}, limit={limits_.maximum_incidence_records}",
        )
    source_surface = owners[native.fragment_faces].astype(np.int64)
    source_face = native.fragment_faces - face_offsets[source_surface]
    publication_pairs = 0
    for operand in range(len(surfaces)):
        selected = np.flatnonzero(source_surface == operand)
        publication_pairs += require_published_embedded(
            native.vertices,
            native.fragments[selected],
            source_face[selected],
            limits_.maximum_candidate_pairs - spent - publication_pairs,
            surface=operand,
        )
    features = np.full((native.vertices.shape[0], len(surfaces), 3), -1, dtype=np.int64)
    # Native location codes refer to source faces in the rational arrangement,
    # before coordinate rounding.
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
                f"arrange_triangle_surfaces.coincident_orientation must assign one orientation per fragment and operand; got fragment={fragment}, operand={owner}, prior={orientation[fragment, owner]}, reported={sign}"
            )
        if prior < 0 or local_face < prior:
            coincident[fragment, owner] = local_face
            orientation[fragment, owner] = sign
    fragment_counts = np.bincount(source_surface, minlength=len(surfaces))
    per_face = np.bincount(native.fragment_faces, minlength=face_offsets[-1])
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
        coincident_vertices=native.counters.coincident_vertices,
        maximum_construction_bound=float(np.max(native.bounds, initial=0.0)),
        split_faces=split,
        fragments=tuple(fragment_counts.tolist()),
        triple_plane_vertices=native.counters.triple_plane_vertices,
        exact_predicates=native.counters.exact_predicates,
        filtered_predicates=native.counters.filtered_predicates,
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
    return arrangement


__all__ = [
    "arrange_triangle_surfaces",
    "SurfaceArrangement",
    "SurfaceArrangementError",
    "SurfaceArrangementEvidence",
    "SurfaceArrangementLimits",
    "SurfaceArrangementStatus",
]
