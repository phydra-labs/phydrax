#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Certify open-cell material labels and exhaustive solid ancestry.

An interior point is a proposal, never an absence proof. A non-boundary label
requires a physical enclosure whose radius is strictly smaller than the native
boundary-distance lower bound. The exhaustive face-pair/DCEL coverage then extends
that local decision across its connected open cell. Coincident material sides
come from chart correspondence and authored solid-face coorientation.

Overlap is an open-volume question. Every source/target pair is scheduled, and
absence requires checking both exhaustive boundaries. Opposed coincident faces
and isolated contacts do not create regularized material ancestry. This also
finds contributors whose entire boundary was buried before boundary retention.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from fractions import Fraction
from itertools import combinations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax.geometry.brep._boolean_arrangement as boolean_arrangement_owner
import phydrax.geometry.brep._boolean_cells as boolean_cells_owner

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...typing import checked
from .._interval_enclosure import (
    interval_add,
    interval_multiply,
    interval_subtract,
    prepare_interval_function,
    PreparedIntervalFunction,
)
from .._rational import fraction_endpoint
from .._source_identity import source_entity_id
from ._boolean_arrangement import Arrangement, FaceOwner, Piece
from ._boolean_cells import FaceArrangementCell, FaceArrangementCells, scalar_phase
from ._boolean_coincidence import (
    CoincidentFaceOverlay,
    CoincidentSurfaceCorrespondence,
    prove_surface_correspondence,
    transport_coincident_pcurve,
)
from ._boolean_core import (
    AffinePCurve,
    BRepBooleanFailure,
    material_owner,
    pcurve_affine_correspondence,
    pcurve_surface_period_shifts,
    PeriodicPCurve,
    selection_payload,
    side_payload,
    trim_source_join,
)
from ._carrier_pieces import (
    carrier_payload,
    child_coefficient_enclosures,
    SurfacePiece,
)
from ._intersection import ConicParameterEndpoint, SurfaceIntersectionResult
from ._intersection_curve import CurveTrimSegment
from ._interval_source import evaluate_interval
from ._model import BRepEntityId, BRepModel
from ._patches import AbstractCurve, AbstractSurfacePatch
from ._projection import NativeBRepProjection
from ._projection_contracts import BRepEntityDimension, BRepProjectionStatus


type FacePair = tuple[FaceOwner, FaceOwner]
type ParameterBox = tuple[tuple[float, float], tuple[float, float]]
type PhysicalBox = tuple[tuple[float, float, float], tuple[float, float, float]]


class MaterialLabelRoute(Enum):
    COORIENTATION = "exact-coorientation"
    ENCLOSURE = "physical-enclosure-separation"
    BOOLEAN_SELECTION = "complete-boolean-material-selection"


@dataclass(frozen=True, slots=True)
class MaterialSolid:
    model_index: int
    entity: BRepEntityId

    @property
    def source_id(self) -> str:
        entity = self.entity
        return canonical_fingerprint(
            {
                "kind": "native-boolean-material-source",
                "operand_index": self.model_index,
                "entity": source_entity_id(
                    entity.source_id,
                    entity.source_revision,
                    entity.kind,
                    entity.index,
                    occurrence_path=entity.occurrence_path,
                ),
            }
        )


def _entity_occurrences(model: BRepModel, kind: str, /) -> tuple[BRepEntityId, ...]:
    return tuple(
        sorted(
            {
                record.member
                for record in model.geometry.qualified_entity_incidence(
                    model.source_id, model.source_revision
                )
                if record.container == record.member and record.member.kind == kind
            }
        )
    )


@dataclass(frozen=True, slots=True)
class MaterialEnclosureEvidence:
    parameter_box: ParameterBox
    physical_box: PhysicalBox
    center: tuple[float, float, float]
    radius_upper: float
    distance_lower: float | None
    query_id: str | None
    containment: BRepProjectionStatus | None = None


@dataclass(frozen=True, slots=True)
class SolidMaterialLabel:
    solid: MaterialSolid
    negative_inside: bool
    positive_inside: bool
    route: MaterialLabelRoute
    support_faces: tuple[BRepEntityId, ...]
    enclosure: MaterialEnclosureEvidence
    certificate_id: str


@dataclass(frozen=True, slots=True)
class FaceMaterialLabel:
    """Material owner on each side of one exhaustive source cell.

    The negative side lies against the source patch normal. Equal owners bury
    the cell; otherwise ``orientation`` is +1 when the negative side is owned.
    """

    classification_key: str
    owner: FaceOwner
    solids: tuple[SolidMaterialLabel, ...]
    negative_owner: str | None
    positive_owner: str | None
    orientation: int
    certificate_id: str

    @property
    def negative_selected(self) -> bool:
        return self.negative_owner is not None

    @property
    def positive_selected(self) -> bool:
        return self.positive_owner is not None


@dataclass(frozen=True, slots=True)
class FacePairMaterialEvidence:
    pair: FacePair
    certificate_id: str
    isolated_contacts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MaterialOverlayCompleteness:
    source_revisions: tuple[str, ...]
    face_coverage: tuple[tuple[FaceOwner, str], ...]
    face_pairs: tuple[FacePairMaterialEvidence, ...]
    cell_keys: tuple[str, ...]
    certificate_id: str


@dataclass(frozen=True, slots=True)
class FaceMaterialClassification:
    cells: tuple[FaceArrangementCell, ...]
    labels: tuple[FaceMaterialLabel, ...]
    completeness: MaterialOverlayCompleteness
    certificate_id: str


@dataclass(frozen=True, slots=True)
class TargetCellMaterialLabel:
    classification_key: str
    solids: tuple[SolidMaterialLabel, ...]


@dataclass(frozen=True, slots=True)
class SolidMaterialOverlap:
    source: MaterialSolid
    target: BRepEntityId
    overlaps: bool
    witness_cells: tuple[str, ...]
    checked_source_cells: tuple[str, ...]
    checked_target_cells: tuple[str, ...]
    certificate_id: str


@dataclass(frozen=True, slots=True)
class MaterialOverlapSchedule:
    overlaps: tuple[SolidMaterialOverlap, ...]
    target_labels: tuple[TargetCellMaterialLabel, ...]
    entity_pairs: tuple[tuple[MaterialSolid, BRepEntityId], ...]
    deleted_sources: tuple[MaterialSolid, ...]
    certificate_id: str


class _SurfaceBox(StrictModule):
    lower: np.ndarray
    upper: np.ndarray
    value: PreparedIntervalFunction
    normal: PreparedIntervalFunction


@dataclass
class _Context:
    state: Arrangement
    completeness_id: str
    surface_boxes: dict[FaceOwner, tuple[_SurfaceBox, ...]] = field(default_factory=dict)
    correspondences: dict[
        tuple[FaceOwner, str, int], CoincidentSurfaceCorrespondence | None
    ] = field(default_factory=dict)


def _coverage_id(coverage: FaceArrangementCells) -> str:
    return canonical_fingerprint(
        {
            "kind": "complete-native-face-graph",
            "owner": coverage.owner,
            "halfedges": coverage.covered_halfedges,
            "excluded_cycles": coverage.excluded_cycles,
            "roots": coverage.covered_roots,
            "quotient_sources": tuple(sorted(coverage.quotient_sources.items())),
            "cells": tuple(cell.classification_key for cell in coverage.cells),
        }
    )


def _pair_evidence(
    state: Arrangement,
    pair: FacePair,
    surface_results: Mapping[FacePair, SurfaceIntersectionResult],
    coincident_overlays: Mapping[FacePair, CoincidentFaceOverlay],
) -> FacePairMaterialEvidence:
    first, second = pair
    models = state.models
    a, b = models[first[0]], models[second[0]]
    box_a = a.patches[first[1]].bounding_box(a.parameter_bounds[first[1]])
    box_b = b.patches[second[1]].bounding_box(b.parameter_bounds[second[1]])
    base = {"pair": pair, "revisions": (a.source_revision, b.source_revision)}
    if np.any(box_a[1] < box_b[0]) or np.any(box_b[1] < box_a[0]):
        return FacePairMaterialEvidence(
            pair,
            canonical_fingerprint(
                {
                    **base,
                    "kind": "native-disjoint-face-enclosures",
                    "boxes": (box_a, box_b),
                }
            ),
            (),
        )
    if pair in state.source_adjacencies:
        if first[0] != second[0]:
            raise BRepBooleanFailure(
                "BRepBoolean.source_adjacency must remain within one operand definition; got different operands",
                (str(pair),),
            )
        edges = state.source_adjacencies[pair]
        model = models[first[0]]
        return FacePairMaterialEvidence(
            pair,
            canonical_fingerprint(
                {
                    **base,
                    "kind": "native-authoritative-source-adjacency",
                    "geometry": (
                        None if model.geometry is None else model.geometry.geometry_id
                    ),
                    "faces": (
                        asdict(model.face_ids[first[1]]),
                        asdict(model.face_ids[second[1]]),
                    ),
                    "edges": tuple(asdict(model.edge_ids[edge]) for edge in edges),
                }
            ),
            (),
        )
    if pair in coincident_overlays:
        overlay = coincident_overlays[pair]
        if (
            overlay.first_face_id != a.face_ids[first[1]]
            or overlay.second_face_id != b.face_ids[second[1]]
        ):
            raise BRepBooleanFailure(
                "BRepBoolean.material.coincident_overlay must bind current source-face IDs; got stale IDs",
                (str(pair),),
            )
        return FacePairMaterialEvidence(pair, overlay.certificate_id, ())
    if pair in state.source_pair_coverage:
        proof = state.source_pair_coverage[pair]
        if proof.faces != (a.face_ids[first[1]], b.face_ids[second[1]]):
            raise BRepBooleanFailure(
                "BRepBoolean.material.source_edge_coverage must bind current source-face IDs; got stale IDs",
                (str(pair),),
            )
        return FacePairMaterialEvidence(
            pair,
            canonical_fingerprint(
                {
                    **base,
                    "kind": "native-complete-source-edge-face-pair",
                    "source_edge": asdict(proof.edge_id),
                    "coverage": proof.certificate_id,
                }
            ),
            (),
        )
    result = surface_results.get(pair)
    if result is None or not result.complete:
        domains = (
            np.asarray(a.parameter_bounds)[first[1]],
            np.asarray(b.parameter_bounds)[second[1]],
        )
        details = (
            ()
            if result is None
            else tuple(
                f"{region.reason}:lower={tuple(region.lower)}:upper={tuple(region.upper)}"
                for region in result.unresolved
            )
        )
        if not details:
            details = (
                f"unresolved-pair-parameter-cell:{domains[0].tolist()}:{domains[1].tolist()}",
            )
        raise BRepBooleanFailure(
            "BRepBoolean.material.face_pairs must have resolved discovery evidence; got missing or unresolved overlay",
            (
                f"{a.source_revision}:face:{first[1]}",
                f"{b.source_revision}:face:{second[1]}",
                *details,
            ),
        )
    if result.coincident:
        raise BRepBooleanFailure(
            "BRepBoolean.material.coincident_faces must have a source-defined full-trim overlay; got no overlay",
            (str(pair),),
        )
    contacts = tuple(
        canonical_fingerprint(
            {
                "kind": point.kind,
                "certificate": point.certificate,
                "parameter_lower": point.parameter_lower,
                "parameter_upper": point.parameter_upper,
                "gap_bound": point.gap_bound,
            }
        )
        for point in result.points
    )
    return FacePairMaterialEvidence(
        pair,
        canonical_fingerprint(
            {
                **base,
                "kind": "native-complete-face-intersection",
                "branches": tuple(curve.branch_id for curve in result.curves),
                "contacts": contacts,
            }
        ),
        contacts,
    )


def _verify_pair_graph(
    state: Arrangement,
    pair: FacePair,
    coverage: Mapping[FaceOwner, FaceArrangementCells],
    result: SurfaceIntersectionResult | None,
    overlay: CoincidentFaceOverlay | None,
) -> None:
    if overlay is not None:
        _verify_overlay_graph(pair, coverage, overlay)
        return
    if pair in state.source_adjacencies:
        _verify_source_adjacency_graph(state, pair, coverage)
        return
    if pair in state.source_pair_coverage:
        proof = state.source_pair_coverage[pair]
        for owner in pair:
            graph = coverage[owner]
            tokens = _graph_tokens(graph)
            if not _source_edge_covered(state, graph, tokens, owner, proof.source_edge):
                raise BRepBooleanFailure(
                    "BRepBoolean.cell_graph must cover its certified source-edge pair; got an omitted edge",
                    (str(owner), proof.certificate_id),
                )
        return
    if result is None:
        return
    _verify_branch_graph(state, pair, coverage, result)


def _graph_tokens(graph: FaceArrangementCells) -> set[str]:
    tokens = {edge[0] for edge in graph.covered_halfedges}
    tokens.update(
        token for sources in graph.quotient_sources.values() for token in sources
    )
    return tokens


def _verify_overlay_graph(
    pair: FacePair,
    coverage: Mapping[FaceOwner, FaceArrangementCells],
    overlay: CoincidentFaceOverlay,
) -> None:
    required = {
        (source.face_id, source.coedge)
        for loops in (overlay.first_loops, overlay.second_loops)
        for loop in loops
        for source in loop
    }
    for owner in pair:
        actual = {
            (source.face_id, source.coedge)
            for source in coverage[owner].overlay_sources.values()
        }
        if not required.issubset(actual):
            raise BRepBooleanFailure(
                "BRepBoolean.cell_graph must cover every authored overlay coedge; got omitted coedges",
                (str(owner),),
            )


def _verify_source_adjacency_graph(
    state: Arrangement,
    pair: FacePair,
    coverage: Mapping[FaceOwner, FaceArrangementCells],
) -> None:
    if pair[0][0] != pair[1][0]:
        raise BRepBooleanFailure(
            "BRepBoolean.source_adjacency must remain within one operand definition; got different operands",
            (str(pair),),
        )
    model_index = pair[0][0]
    for owner in pair:
        graph = coverage[owner]
        tokens = _graph_tokens(graph)
        for edge in state.source_adjacencies[pair]:
            if not _source_edge_covered(state, graph, tokens, owner, (model_index, edge)):
                raise BRepBooleanFailure(
                    "BRepBoolean.cell_graph must cover every source adjacency; got an omitted edge",
                    (str(owner), f"edge:{edge}"),
                )


def _verify_branch_graph(
    state: Arrangement,
    pair: FacePair,
    coverage: Mapping[FaceOwner, FaceArrangementCells],
    result: SurfaceIntersectionResult,
) -> None:
    for curve in result.curves:
        for owner in pair:
            graph = coverage[owner]
            tokens = _graph_tokens(graph)
            arcs = tuple(
                arc
                for arc in state.arcs[owner]
                if arc.branch is not None and arc.branch.branch_id == curve.branch_id
            )
            if arcs:
                covered = all(arc.token in tokens for arc in arcs)
            else:
                # A branch may already be represented by a proved source edge.
                covered = _source_edge_covered(
                    state,
                    graph,
                    tokens,
                    owner,
                    state.branch_source_edges.get(curve.branch_id, {}).get(owner),
                )
            if not covered:
                raise BRepBooleanFailure(
                    "BRepBoolean.cell_graph must cover every certified boundary branch; got an omitted branch",
                    (str(owner), curve.branch_id),
                )


def _source_edge_covered(
    state: Arrangement,
    graph: FaceArrangementCells,
    tokens: set[str],
    owner: FaceOwner,
    edge: tuple[int, int] | None,
) -> bool:
    if edge is None:
        return False
    model_index, index = edge
    revision = state.models[model_index].source_revision
    candidates = [
        arc.token
        for arc in state.arcs[owner]
        if arc.branch is None and model_index == owner[0] and arc.source_edge == index
    ]
    candidates += [
        token
        for token, source in graph.overlay_sources.items()
        if graph.parent_tokens.get(token, token) == token
        and source.edge_id.source_revision == revision
        and source.edge_id.index == index
    ]
    return any(
        graph.parent_tokens.get(token, token) == candidate
        for candidate in candidates
        for token in tokens
    )


def _certify_overlay(
    state: Arrangement,
    coverages: tuple[FaceArrangementCells, ...],
    surface_results: Mapping[FacePair, SurfaceIntersectionResult],
    coincident_overlays: Mapping[FacePair, CoincidentFaceOverlay],
) -> tuple[tuple[FaceArrangementCell, ...], MaterialOverlayCompleteness]:
    expected = tuple(
        (operand, face)
        for operand, model in enumerate(state.models)
        for face, solids in enumerate(model.topology.face_solids)
        if solids
    )
    supplied = tuple(sorted(coverage.owner for coverage in coverages))
    if supplied != expected:
        missing = tuple(str(owner) for owner in expected if owner not in supplied)
        raise BRepBooleanFailure(
            "BRepBoolean.material.overlay must cover every source-solid boundary face once; got missing or duplicate coverage",
            missing,
        )
    cells = []
    coverage_ids = []
    for coverage in sorted(coverages, key=lambda item: item.owner):
        if (
            not coverage.complete
            or coverage.unresolved_roots
            or coverage.unresolved_cells
        ):
            details = tuple(item.reason for item in coverage.unresolved_cells)
            raise BRepBooleanFailure(
                "BRepBoolean.material.overlay must resolve open cells and roots; got unresolved evidence",
                (
                    str(coverage.owner),
                    *coverage.unresolved_roots,
                    *details,
                ),
            )
        if not coverage.cells:
            raise BRepBooleanFailure(
                "BRepBoolean.material.source_face must have open-cell coverage; got no cells",
                (str(coverage.owner),),
            )
        if any(cell.owner != coverage.owner for cell in coverage.cells):
            raise BRepBooleanFailure(
                "BRepBoolean.material.face_coverage must retain its source owner; got a different owner",
                (str(coverage.owner),),
            )
        coverage_ids.append((coverage.owner, _coverage_id(coverage)))
        cells.extend(coverage.cells)
    cells_tuple = tuple(
        sorted(cells, key=lambda cell: (cell.owner, cell.classification_key))
    )
    keys = tuple(cell.classification_key for cell in cells_tuple)
    if len(set(keys)) != len(keys):
        raise BRepBooleanFailure(
            "BRepBoolean.material.cells must have unique classification identities; got duplicate identities",
            keys,
        )
    pairs = tuple(
        _pair_evidence(
            state,
            pair,
            surface_results,
            coincident_overlays,
        )
        for pair in combinations(expected, 2)
    )
    by_owner = {coverage.owner: coverage for coverage in coverages}
    for evidence in pairs:
        _verify_pair_graph(
            state,
            evidence.pair,
            by_owner,
            surface_results.get(evidence.pair),
            coincident_overlays.get(evidence.pair),
        )
    revisions = tuple(model.source_revision for model in state.models)
    certificate = canonical_fingerprint(
        {
            "kind": "complete-all-source-material-overlay",
            "revisions": revisions,
            "coverage": tuple(coverage_ids),
            "pairs": tuple(item.certificate_id for item in pairs),
            "cells": keys,
        }
    )
    return cells_tuple, MaterialOverlayCompleteness(
        revisions, tuple(coverage_ids), pairs, keys, certificate
    )


def _normal_function(evaluator: AbstractSurfacePatch, parameters: Array) -> Array:
    derivatives = jax.jacfwd(evaluator.evaluate)(parameters)
    return jnp.cross(derivatives[:, 0], derivatives[:, 1])


def _surface_boxes(context: _Context, owner: FaceOwner) -> tuple[_SurfaceBox, ...]:
    if owner in context.surface_boxes:
        return context.surface_boxes[owner]
    model = context.state.models[owner[0]]
    chart = context.state.face_chart(owner)
    pieces = (SurfacePiece(model.patches[owner[1]], chart[0], chart[1], 0),)
    boxes = []
    for piece in pieces:
        coefficients = child_coefficient_enclosures(piece.evaluator)

        def normal(
            parameters: Array, evaluator: AbstractSurfacePatch = piece.evaluator
        ) -> Array:
            return _normal_function(evaluator, parameters)

        boxes.append(
            _SurfaceBox(
                piece.lower,
                piece.upper,
                prepare_interval_function(
                    piece.evaluator.evaluate,
                    2,
                    batch_capacity=1,
                    constant_bounds=coefficients,
                ),
                prepare_interval_function(
                    normal, 2, batch_capacity=1, constant_bounds=coefficients
                ),
            )
        )
    result = tuple(boxes)
    context.surface_boxes[owner] = result
    return result


def _parameter_tuple(box: np.ndarray) -> ParameterBox:
    return (float(box[0, 0]), float(box[0, 1])), (float(box[1, 0]), float(box[1, 1]))


def _physical_tuple(box: np.ndarray) -> PhysicalBox:
    return (
        (float(box[0, 0]), float(box[0, 1]), float(box[0, 2])),
        (float(box[1, 0]), float(box[1, 1]), float(box[1, 2])),
    )


def _physical_enclosure(
    prepared: _SurfaceBox,
    parameter_box: np.ndarray,
) -> MaterialEnclosureEvidence | None:
    lower, upper = evaluate_interval(
        prepared.value,
        parameter_box[0][None],
        parameter_box[1][None],
    )
    normal_lower, normal_upper = evaluate_interval(
        prepared.normal,
        parameter_box[0][None],
        parameter_box[1][None],
    )
    if not np.all(np.isfinite((lower, upper, normal_lower, normal_upper))) or not np.any(
        (normal_lower[0] > 0.0) | (normal_upper[0] < 0.0)
    ):
        return None
    physical = np.stack((lower[0], upper[0]))
    center = 0.5 * physical[0] + 0.5 * physical[1]
    delta = interval_subtract((physical[0], physical[1]), (center, center))
    delta_lower = np.asarray(delta[0], dtype=np.float64)
    delta_upper = np.asarray(delta[1], dtype=np.float64)
    radii = np.maximum(np.abs(delta_lower), np.abs(delta_upper))
    squares = interval_multiply((radii, radii), (radii, radii))
    squares_lower = np.asarray(squares[0], dtype=np.float64)
    squares_upper = np.asarray(squares[1], dtype=np.float64)
    total = np.asarray(0.0, dtype=np.float64), np.asarray(0.0, dtype=np.float64)
    for axis in range(3):
        total = interval_add(total, (squares_lower[axis], squares_upper[axis]))
    total_upper = np.asarray(total[1], dtype=np.float64)
    radius = float(np.nextafter(np.sqrt(total_upper), np.inf))
    if not np.isfinite(radius):
        return None
    return MaterialEnclosureEvidence(
        _parameter_tuple(parameter_box),
        _physical_tuple(physical),
        (float(center[0]), float(center[1]), float(center[2])),
        radius,
        None,
        None,
    )


def _transport_box(proof: CoincidentSurfaceCorrespondence, box: np.ndarray) -> np.ndarray:
    result = np.empty((2, 2), dtype=np.float64)
    for row in range(2):
        lower = upper = proof.offset[row]
        for axis in range(2):
            first = proof.matrix[row][axis] * Fraction(float(box[0, axis]))
            last = proof.matrix[row][axis] * Fraction(float(box[1, axis]))
            lower += min(first, last)
            upper += max(first, last)
        result[0, row] = fraction_endpoint(lower, upper=False)
        result[1, row] = fraction_endpoint(upper, upper=True)
    return result


def _correspondence(
    context: _Context,
    owner: FaceOwner,
    model: BRepModel,
    face: int,
) -> CoincidentSurfaceCorrespondence | None:
    key = owner, model.source_revision, face
    if key not in context.correspondences:
        original = context.state.models[owner[0]].patches[owner[1]]
        context.correspondences[key] = prove_surface_correspondence(
            original, model.patches[face]
        )
    return context.correspondences[key]


def _coorientations(
    context: _Context,
    cell: FaceArrangementCell,
    model: BRepModel,
    query: NativeBRepProjection,
    solid: int,
    box: np.ndarray,
    known_faces: tuple[int, ...],
) -> tuple[tuple[bool, bool], tuple[BRepEntityId, ...]] | None:

    signs = []
    supports = []
    for face, incidence in zip(
        model.topology.solid_faces[solid],
        model.topology.solid_face_orientations[solid],
        strict=True,
    ):
        proof = _correspondence(context, cell.owner, model, face)
        if proof is None:
            continue
        if face not in known_faces:
            trim_domain = boolean_arrangement_owner.native_chart_trim_domain(model, face)
            if trim_domain is None:
                return None
            decision = boolean_arrangement_owner.trim_box_status(
                trim_domain,
                _transport_box(proof, box),
            )
            if decision == 0:
                return None
            if decision < 0:
                continue
        signs.append(incidence * int(float(model.orientation[face])) * proof.orientation)
        supports.append(model.face_ids[face])
    if not signs:
        return None
    return (any(sign > 0 for sign in signs), any(sign < 0 for sign in signs)), tuple(
        supports
    )


def _validate_projection(model: BRepModel, projection: NativeBRepProjection) -> None:
    if (
        projection.model_id != model.model_id
        or projection.source_revision != model.source_revision
    ):
        raise BRepBooleanFailure(
            "BRepBoolean.material.projection must bind model.source_revision; got a different revision",
            (model.source_revision,),
        )
    present = {occurrence.solid for occurrence in projection.occurrences}
    missing = tuple(
        f"{model.source_revision}:solid:{solid}"
        for solid in range(model.topology.num_solids)
        if solid not in present
    )
    if missing:
        raise BRepBooleanFailure(
            "BRepBoolean.material.solid must have an occurrence projection; got none",
            missing,
        )


def _query_separation(
    context: _Context,
    cell: FaceArrangementCell,
    reference: MaterialSolid,
    query: NativeBRepProjection,
    enclosure: MaterialEnclosureEvidence,
) -> tuple[bool, MaterialEnclosureEvidence] | None:
    faces = query.model.topology.solid_faces[reference.entity.index]
    if not faces:
        return None
    center = np.asarray(enclosure.center, dtype=np.float64)
    points = np.repeat(center[None, :], len(faces), axis=0)
    dimensions = np.full((len(faces),), int(BRepEntityDimension.FACE), dtype=np.int32)
    projected = query.project(points, dimensions, np.asarray(faces, dtype=np.int32))
    statuses = np.asarray(projected.status)
    if np.any(statuses == int(BRepProjectionStatus.FAILED)):
        return None
    lower_bounds = np.asarray(projected.distance_lower_bounds, dtype=np.float64)
    if not np.all(np.isfinite(lower_bounds)):
        return None
    distance = float(np.min(lower_bounds))
    if not distance > enclosure.radius_upper:
        return None
    membership = query.locate_solids(center[None])
    status = BRepProjectionStatus(int(np.asarray(membership.status)[0]))
    if status not in (BRepProjectionStatus.UNIQUE, BRepProjectionStatus.FAILED):
        return None
    inside = (
        status == BRepProjectionStatus.UNIQUE
        and int(np.asarray(membership.indices)[0]) == reference.entity.index
    )
    separated = MaterialEnclosureEvidence(
        enclosure.parameter_box,
        enclosure.physical_box,
        enclosure.center,
        enclosure.radius_upper,
        distance,
        query.projection_id,
        status,
    )
    return inside, separated


def _make_label(
    context: _Context,
    cell: FaceArrangementCell,
    reference: MaterialSolid,
    sides: tuple[bool, bool],
    route: MaterialLabelRoute,
    supports: tuple[BRepEntityId, ...],
    enclosure: MaterialEnclosureEvidence,
) -> SolidMaterialLabel:
    certificate = canonical_fingerprint(
        {
            "kind": "native-complete-cell-material-label",
            "cell": cell.classification_key,
            "completeness": context.completeness_id,
            "solid": asdict(reference),
            "sides": sides,
            "route": route.value,
            "supports": tuple(asdict(face) for face in supports),
            "enclosure": asdict(enclosure),
        }
    )
    return SolidMaterialLabel(
        reference, sides[0], sides[1], route, supports, enclosure, certificate
    )


def _solid_label(
    context: _Context,
    cell: FaceArrangementCell,
    model_index: int,
    model: BRepModel,
    query: NativeBRepProjection,
    solid: int,
    known_faces: tuple[int, ...] = (),
) -> SolidMaterialLabel:
    reference = MaterialSolid(model_index, model.solid_ids[solid])
    seed = np.asarray(cell.seed_box, dtype=np.float64)
    if (
        seed.shape != (2, 2)
        or np.any(seed[0] >= seed[1])
        or boolean_arrangement_owner.trim_box_status(cell.domain, seed) != 1
    ):
        raise BRepBooleanFailure(
            "BRepBoolean.material.cell must have a certified strict interior box; got no box",
            (cell.classification_key,),
        )

    def label_from(seed: np.ndarray, /) -> SolidMaterialLabel | None:
        pending: deque[tuple[_SurfaceBox, np.ndarray]] = deque()
        for prepared in _surface_boxes(context, cell.owner):
            lower, upper = (
                np.maximum(seed[0], prepared.lower),
                np.minimum(seed[1], prepared.upper),
            )
            if np.all(lower < upper):
                pending.append((prepared, np.stack((lower, upper))))
        while pending:
            prepared, box = pending.popleft()
            enclosure = _physical_enclosure(prepared, box)
            if enclosure is not None:
                coorientation = _coorientations(
                    context, cell, model, query, solid, box, known_faces
                )
                if coorientation is not None:
                    sides, supports = coorientation
                    return _make_label(
                        context,
                        cell,
                        reference,
                        sides,
                        MaterialLabelRoute.COORIENTATION,
                        supports,
                        enclosure,
                    )
                separated = _query_separation(context, cell, reference, query, enclosure)
                if separated is not None:
                    inside, proof = separated
                    return _make_label(
                        context,
                        cell,
                        reference,
                        (inside, inside),
                        MaterialLabelRoute.ENCLOSURE,
                        (),
                        proof,
                    )
            axis = int(np.argmax(box[1] - box[0]))
            middle = box[0, axis] + 0.5 * (box[1, axis] - box[0, axis])
            if not box[0, axis] < middle < box[1, axis]:
                continue
            first, last = box.copy(), box.copy()
            first[1, axis], last[0, axis] = middle, middle
            pending.extend(((prepared, first), (prepared, last)))
        return None

    label = label_from(seed)
    if label is not None:
        return label
    # A seed on a tangent contact with the other solid (the loop's centre on a
    # tangent generator) never separates; any other strict-interior seed of
    # the same cell labels it equally.
    off_centre = boolean_cells_owner._seed_box(
        cell.domain,
        tuple(boolean_arrangement_owner.piece_trim(piece) for piece in cell.loops[0]),
        0.381966,
    )
    label = label_from(off_centre)
    if label is not None:
        return label
    raise BRepBooleanFailure(
        "BRepBoolean.material.cell remains unresolved at parameter representability",
        (cell.classification_key, reference.source_id, model.source_revision),
    )


@checked
def classify_face_material_cells(
    state: Arrangement,
    coverages: tuple[FaceArrangementCells, ...],
    /,
    *,
    surface_results: Mapping[FacePair, SurfaceIntersectionResult],
    coincident_overlays: Mapping[FacePair, CoincidentFaceOverlay],
) -> FaceMaterialClassification:
    """Label every exhaustive open source-face cell against each source solid."""
    for model, query in zip(state.models, state.projections, strict=True):
        _validate_projection(model, query)
    cells, completeness = _certify_overlay(
        state,
        coverages,
        surface_results,
        coincident_overlays,
    )
    context = _Context(state, completeness.certificate_id)
    labels = []
    for cell in cells:
        individual = tuple(
            _solid_label(
                context,
                cell,
                operand,
                model,
                state.projections[operand],
                solid,
                (cell.owner[1],) if operand == cell.owner[0] else (),
            )
            for operand, model in enumerate(state.models)
            for solid in range(model.topology.num_solids)
        )
        negative = material_owner(
            state.selection,
            tuple(
                (item.solid.model_index, item.solid.entity.index)
                for item in individual
                if item.negative_inside
            ),
        )
        positive = material_owner(
            state.selection,
            tuple(
                (item.solid.model_index, item.solid.entity.index)
                for item in individual
                if item.positive_inside
            ),
        )
        orientation = (1 if negative is not None else -1) if negative != positive else 0
        certificate = canonical_fingerprint(
            {
                "cell": cell.classification_key,
                "labels": tuple(item.certificate_id for item in individual),
                "operation": selection_payload(state.selection),
                "selected": side_payload(state.selection, negative, positive),
            }
        )
        labels.append(
            FaceMaterialLabel(
                cell.classification_key,
                cell.owner,
                individual,
                negative,
                positive,
                orientation,
                certificate,
            )
        )
    certificate = canonical_fingerprint(
        {
            "kind": "native-complete-source-material-classification",
            "completeness": completeness.certificate_id,
            "labels": tuple(item.certificate_id for item in labels),
        }
    )
    return FaceMaterialClassification(cells, tuple(labels), completeness, certificate)


def _exact_cut_parameter(piece: Piece, start: bool) -> Fraction | None:
    """Return the exact arc parameter of a piece end, if its binding proves one.

    Unbound cuts carry an exact binary parameter; a conic root bound to the
    arc's own p-curve carries its exact carrier parameter. Other roots refuse.
    """
    cut = piece.first if start else piece.last
    endpoint = cut.endpoint
    if endpoint is None:
        return Fraction(cut.parameter)
    arc = piece.arc
    if (
        not isinstance(endpoint, ConicParameterEndpoint)
        or endpoint.parameter_exact is None
        or carrier_payload(endpoint.carrier) != carrier_payload(arc.pcurve)
        or arc.scale == 0.0
    ):
        return None
    return (endpoint.parameter_exact - Fraction(arc.offset)) / Fraction(arc.scale)


def _fixed_phase_piece(
    first: Piece,
    second: Piece,
    proof: CoincidentSurfaceCorrespondence,
) -> bool:

    transported = replace(
        first.arc,
        pcurve=transport_coincident_pcurve(proof, first.arc.pcurve),
        trim=AffinePCurve(
            first.arc.trim,
            np.asarray(proof.matrix, dtype=np.object_),
            np.asarray(proof.offset, dtype=np.object_),
        ),
    )
    phase = scalar_phase(transported, second.arc)
    if phase is None:
        return False
    a = (True, False) if first.sense > 0 else (False, True)
    b = (True, False) if second.sense > 0 else (False, True)
    for left_start, right_start in zip(a, b, strict=True):
        left_cut = first.first if left_start else first.last
        right_cut = second.first if right_start else second.last
        left_root, right_root = left_cut.endpoint, right_cut.endpoint
        # Root-bound ends must name one exact point; the phase then proves the
        # exact carrier parameters correspond.
        if (left_root is None) != (right_root is None):
            return False
        if left_root is not None and (
            not isinstance(left_root, ConicParameterEndpoint)
            or not isinstance(right_root, ConicParameterEndpoint)
            or left_root.point_exact != right_root.point_exact
        ):
            return False
        left = _exact_cut_parameter(first, left_start)
        right = _exact_cut_parameter(second, right_start)
        if left is None or right is None or phase[0] * left + phase[1] != right:
            return False
    return True


def _equivalent_piece(
    first: Piece,
    second: Piece,
    proof: CoincidentSurfaceCorrespondence,
) -> bool:

    shifts = pcurve_surface_period_shifts(
        first.arc.pcurve,
        second.arc.pcurve,
        proof.matrix,
        proof.offset,
        proof.second,
    )
    if shifts is None:
        return _fixed_phase_piece(first, second, proof)
    if (first.arc.scale * first.sense > 0.0) != (second.arc.scale * second.sense > 0.0):
        return False
    cuts = first.first, first.last, second.first, second.last
    if all(cut.endpoint is None for cut in cuts):
        first_cuts = (
            (first.first, first.last) if first.sense > 0 else (first.last, first.first)
        )
        second_cuts = (
            (second.first, second.last)
            if second.sense > 0
            else (second.last, second.first)
        )
        parameters_a = tuple(
            Fraction(first.arc.offset)
            + Fraction(first.arc.scale) * Fraction(cut.parameter)
            for cut in first_cuts
        )
        parameters_b = tuple(
            Fraction(second.arc.offset)
            + Fraction(second.arc.scale) * Fraction(cut.parameter)
            for cut in second_cuts
        )
        return parameters_a == parameters_b
    if not all(cut.endpoint is not None for cut in cuts):
        return False
    # Root-bearing joins cannot fall through to the native affine-coordinate
    # endpoint shortcut: both root/source transports are mandatory.
    matrix = np.asarray(proof.matrix, dtype=np.object_)
    offset = np.asarray(proof.offset, dtype=np.object_)
    transported = AffinePCurve(
        boolean_arrangement_owner.piece_trim(first),
        matrix,
        offset,
    )
    reversed_transport = AffinePCurve(
        boolean_arrangement_owner.piece_trim(replace(first, sense=-first.sense)),
        matrix,
        offset,
    )
    a: AbstractCurve = (
        PeriodicPCurve(transported, proof.second, shifts) if any(shifts) else transported
    )
    reversed_a: AbstractCurve = (
        PeriodicPCurve(reversed_transport, proof.second, shifts)
        if any(shifts)
        else reversed_transport
    )
    b = boolean_arrangement_owner.piece_trim(second)
    reversed_b = boolean_arrangement_owner.piece_trim(
        replace(second, sense=-second.sense)
    )
    return trim_source_join(a, reversed_b) and trim_source_join(reversed_a, b)


@checked
def equivalent_cell(
    state: Arrangement,
    first: FaceArrangementCell,
    second: FaceArrangementCell,
) -> bool:
    first_patch = state.models[first.owner[0]].patches[first.owner[1]]
    second_patch = state.models[second.owner[0]].patches[second.owner[1]]
    proof = prove_surface_correspondence(first_patch, second_patch)
    if proof is None or len(first.loops) != len(second.loops):
        return False
    unmatched = list(second.loops)
    for loop in first.loops:
        match = None
        for index, candidate in enumerate(unmatched):
            if len(loop) != len(candidate):
                continue
            if proof.orientation < 0:
                candidate = tuple(
                    replace(piece, sense=-piece.sense) for piece in reversed(candidate)
                )
            if any(
                all(
                    _equivalent_piece(
                        piece,
                        candidate[(offset + ordinal) % len(candidate)],
                        proof,
                    )
                    for ordinal, piece in enumerate(loop)
                )
                for offset in range(len(candidate))
            ):
                match = index
                break
        if match is None:
            return False
        unmatched.pop(match)
    return True


def _same_trim_segment(
    first: CurveTrimSegment,
    second: CurveTrimSegment,
) -> bool:

    identity = (
        ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))),
        (Fraction(0), Fraction(0)),
    )
    if (
        first.reversed != second.reversed
        or pcurve_affine_correspondence(first.curve, second.curve) != identity
    ):
        return False
    if first.first_root is None or second.first_root is None:
        if first.first_root is not second.first_root or Fraction(first.first) != Fraction(
            second.first
        ):
            return False
    if first.last_root is None or second.last_root is None:
        if first.last_root is not second.last_root or Fraction(first.last) != Fraction(
            second.last
        ):
            return False
    if first.first_root is not None and second.first_root is not None:
        if not trim_source_join(
            replace(first, reversed=True),
            replace(second, reversed=False),
        ):
            return False
    if first.last_root is not None and second.last_root is not None:
        if not trim_source_join(
            replace(first, reversed=False),
            replace(second, reversed=True),
        ):
            return False
    return True


def _target_trim_copy(
    cell: FaceArrangementCell,
    model: BRepModel,
    face: int,
) -> bool:

    geometry = model.geometry
    if geometry is None or len(geometry.face_loops[face]) != len(cell.loops):
        return False
    for coedges, pieces in zip(geometry.face_loops[face], cell.loops, strict=True):
        if len(coedges) != len(pieces):
            return False
        for coedge, piece in zip(coedges, pieces, strict=True):
            edge = geometry.coedge_edges[coedge]
            first, last = (
                float(value) for value in np.asarray(geometry.edge_ranges)[edge]
            )
            roots = geometry.coedge_endpoint_roots[coedge]
            target = CurveTrimSegment(
                geometry.pcurves[coedge],
                first,
                last,
                reversed=geometry.coedge_senses[coedge] < 0,
                first_root=roots[0],
                last_root=roots[1],
            )
            if not _same_trim_segment(
                boolean_arrangement_owner.piece_trim(piece),
                target,
            ):
                return False
    return True


def _target_coverage(
    state: Arrangement,
    classification: FaceMaterialClassification,
    model: BRepModel,
    face_cells: tuple[str, ...],
    representatives: Mapping[str, str],
) -> tuple[dict[str, tuple[int, ...]], str]:
    if len(face_cells) != model.topology.num_faces:
        raise BRepBooleanFailure(
            "BRepBoolean.material.target_face_cells must bind every target face once; got a cardinality mismatch"
        )
    cells = {cell.classification_key: cell for cell in classification.cells}
    labels = {label.classification_key: label for label in classification.labels}
    bound: dict[str, list[int]] = {}
    for face, key in enumerate(face_cells):
        if key not in cells:
            raise BRepBooleanFailure(
                "BRepBoolean.material.target_face must retain its source-cell identity; got an unknown cell",
                (key,),
            )
        cell = cells[key]
        original = state.models[cell.owner[0]].patches[cell.owner[1]]
        proof = prove_surface_correspondence(original, model.patches[face])
        if (
            proof is None
            or proof.matrix != ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
            or proof.offset != (Fraction(0), Fraction(0))
            or not _target_trim_copy(cell, model, face)
        ):
            raise BRepBooleanFailure(
                "BRepBoolean.material.target_face must preserve its source chart and full trim loops; got failed correspondence",
                (key,),
            )
        if int(float(model.orientation[face])) != labels[key].orientation:
            raise BRepBooleanFailure(
                "BRepBoolean.material.target_face.orientation must match selected material sides; got a different orientation",
                (key,),
            )
        if not model.topology.face_solids[face]:
            raise BRepBooleanFailure(
                "BRepBoolean.material.target_face must have a material solid owner; got no incident solid",
                (key,),
            )
        bound.setdefault(key, []).append(face)
    expected = {
        label.classification_key
        for label in classification.labels
        if label.orientation != 0
    }
    if not set(bound).issubset(expected):
        raise BRepBooleanFailure(
            "BRepBoolean.material.target_boundary must retain selected cells only; got an unselected cell",
            tuple(sorted(set(bound) - expected)),
        )
    missing = expected - set(bound)
    if set(representatives) != missing:
        raise BRepBooleanFailure(
            "BRepBoolean.material.target_boundary must bind every omitted selected cell to a representative; got missing or extra representatives",
            tuple(sorted(missing)),
        )
    for key, representative in sorted(representatives.items()):
        if representative not in bound or not equivalent_cell(
            state,
            cells[key],
            cells[representative],
        ):
            raise BRepBooleanFailure(
                "BRepBoolean.material.representative must preserve all source trim loops; got failed loop equality",
                (key, representative),
            )
        first_label, second_label = labels[key], labels[representative]
        proof = prove_surface_correspondence(
            state.models[cells[key].owner[0]].patches[cells[key].owner[1]],
            state.models[cells[representative].owner[0]].patches[
                cells[representative].owner[1]
            ],
        )
        # A reversed chart exchanges the negative and positive material sides.
        second_sides = (second_label.negative_owner, second_label.positive_owner)
        if proof is None or (first_label.negative_owner, first_label.positive_owner) != (
            second_sides if proof.orientation > 0 else second_sides[::-1]
        ):
            raise BRepBooleanFailure(
                "BRepBoolean.material.representative must preserve material sides under chart orientation; got different side owners",
                (key, representative),
            )
    result = {key: tuple(faces) for key, faces in bound.items()}
    certificate = canonical_fingerprint(
        {
            "kind": "exact-target-source-cell-boundary",
            "source": classification.certificate_id,
            "target_revision": model.source_revision,
            "faces": face_cells,
            "representatives": tuple(sorted(representatives.items())),
        }
    )
    return result, certificate


def _joint_inside(first: SolidMaterialLabel, second: SolidMaterialLabel) -> bool:
    return (first.negative_inside and second.negative_inside) or (
        first.positive_inside and second.positive_inside
    )


def _pair_overlap(
    source: MaterialSolid,
    target: BRepEntityId,
    state: Arrangement,
    classification: FaceMaterialClassification,
    target_labels: Mapping[str, tuple[SolidMaterialLabel, ...]],
    target_face_cells: tuple[str, ...],
    target_model: BRepModel,
    completion_id: str,
) -> SolidMaterialOverlap:
    source_faces = state.models[source.model_index].topology.solid_faces[
        source.entity.index
    ]
    source_keys = tuple(
        cell.classification_key
        for cell in classification.cells
        if cell.owner[0] == source.model_index and cell.owner[1] in source_faces
    )
    target_keys = tuple(
        sorted(
            {
                target_face_cells[face]
                for face in target_model.topology.solid_faces[target.index]
            }
        )
    )
    source_labels = {label.classification_key: label for label in classification.labels}
    witnesses = []
    for key in sorted(set(source_keys) | set(target_keys)):
        # Cell labels name the model-level solid; a source occurrence of that
        # solid shares its material label.
        individual = next(
            (
                item
                for item in source_labels[key].solids
                if item.solid == source
                or (
                    item.solid.model_index == source.model_index
                    and item.solid.entity.occurrence_path == ()
                    and replace(
                        item.solid.entity, occurrence_path=source.entity.occurrence_path
                    )
                    == source.entity
                )
            ),
            None,
        )
        if individual is None:
            raise BRepBooleanFailure(
                "BRepBoolean.material.label must classify every source solid occurrence",
                (key, str(source.entity)),
            )
        target_label = target_labels[key][target.index]
        if _joint_inside(individual, target_label):
            witnesses.append(key)
    certificate = canonical_fingerprint(
        {
            "kind": "exhaustive-regularized-source-target-material-overlap",
            "completion": completion_id,
            "source": asdict(source),
            "target": asdict(target),
            "source_boundary": source_keys,
            "target_boundary": target_keys,
            "checked_labels": tuple(
                (
                    key,
                    source_labels[key].certificate_id,
                    target_labels[key][target.index].certificate_id,
                )
                for key in sorted(set(source_keys) | set(target_keys))
            ),
            "overlaps": bool(witnesses),
            "witnesses": tuple(witnesses),
        }
    )
    return SolidMaterialOverlap(
        source,
        target,
        bool(witnesses),
        tuple(witnesses),
        source_keys,
        target_keys,
        certificate,
    )


class _TargetLabelPhase(StrictModule):
    labels: tuple[TargetCellMaterialLabel, ...] = eqx.field(static=True)


class _TargetCellLabelPhase(StrictModule):
    label: TargetCellMaterialLabel = eqx.field(static=True)


class _SelectedSourceMaterial(StrictModule):
    source: FaceMaterialLabel = eqx.field(static=True)
    owner: SolidMaterialLabel = eqx.field(static=True)


def _selected_source_material(
    state: Arrangement,
    classification: FaceMaterialClassification,
    cell: FaceArrangementCell,
) -> _SelectedSourceMaterial:
    source = next(
        label
        for label in classification.labels
        if label.classification_key == cell.classification_key
    )
    owner = next(
        label
        for label in source.solids
        if label.solid.model_index == cell.owner[0]
        and cell.owner[1]
        in state.models[cell.owner[0]].topology.solid_faces[label.solid.entity.index]
    )
    return _SelectedSourceMaterial(source=source, owner=owner)


def _label_target_cell(
    context: _Context,
    classification: FaceMaterialClassification,
    cell: FaceArrangementCell,
    target_model: BRepModel,
    target_query: NativeBRepProjection,
    bound: Mapping[str, tuple[int, ...]],
    exact_regions: tuple[str, ...] | None,
) -> _TargetCellLabelPhase:
    state = context.state
    if exact_regions is not None:
        source = _selected_source_material(state, classification, cell)
        labels = tuple(
            _make_label(
                context,
                cell,
                MaterialSolid(len(state.models), target_model.solid_ids[solid]),
                (
                    source.source.negative_owner == region,
                    source.source.positive_owner == region,
                ),
                MaterialLabelRoute.BOOLEAN_SELECTION,
                tuple(
                    target_model.face_ids[face]
                    for face in bound.get(cell.classification_key, ())
                    if solid in target_model.topology.face_solids[face]
                ),
                source.owner.enclosure,
            )
            for solid, region in enumerate(exact_regions)
        )
    elif target_model.topology.num_solids == 1:
        source = _selected_source_material(state, classification, cell)
        # A bounded component inherits selected sides after oriented boundary
        # equality has been proved, including buried source cells.
        labels = (
            _make_label(
                context,
                cell,
                MaterialSolid(len(state.models), target_model.solid_ids[0]),
                (
                    source.source.negative_selected,
                    source.source.positive_selected,
                ),
                MaterialLabelRoute.BOOLEAN_SELECTION,
                tuple(
                    target_model.face_ids[face]
                    for face in bound.get(cell.classification_key, ())
                ),
                source.owner.enclosure,
            ),
        )
    else:
        labels = tuple(
            _solid_label(
                context,
                cell,
                len(state.models),
                target_model,
                target_query,
                solid,
                bound.get(cell.classification_key, ()),
            )
            for solid in range(target_model.topology.num_solids)
        )
    return _TargetCellLabelPhase(TargetCellMaterialLabel(cell.classification_key, labels))


def _label_target_cells(
    context: _Context,
    classification: FaceMaterialClassification,
    target_model: BRepModel,
    target_query: NativeBRepProjection | None,
    bound: Mapping[str, tuple[int, ...]],
    target_solid_regions: tuple[str, ...] | None,
) -> _TargetLabelPhase:
    target_labels = []
    exact_regions = (
        target_solid_regions
        if target_solid_regions is not None
        and len(set(target_solid_regions)) == len(target_solid_regions)
        else None
    )
    if target_query is not None and target_model.topology.num_solids:
        for cell in classification.cells:
            phase = _label_target_cell(
                context,
                classification,
                cell,
                target_model,
                target_query,
                bound,
                exact_regions,
            )
            target_labels.append(phase.label)
    return _TargetLabelPhase(tuple(target_labels))


@checked
def prove_material_overlaps(
    state: Arrangement,
    classification: FaceMaterialClassification,
    target_model: BRepModel,
    target_query: NativeBRepProjection | None,
    target_face_cells: tuple[str, ...],
    /,
    *,
    target_cell_representatives: Mapping[str, str],
    target_solid_regions: tuple[str, ...] | None = None,
) -> MaterialOverlapSchedule:
    """Schedule every individual source-solid/target-solid pair independently.

    Each target face must be copied from one full cell and retain that cell's
    classification identity. ``target_query`` is unnecessary only for an empty
    target. pairs contain solid ancestry only; callers append face lineage separately, including retained difference-subtractor faces.
    ``target_cell_representatives`` must bind exactly the omitted selected
    coincident cells to emitted cells; exhaustive loop equality is proved here.
    Supply an empty mapping when no physical boundary copies are suppressed.
    A bijective ``target_solid_regions`` tuple binds partition side owners
    directly to target solids, avoiding an inexact distance query at root-adjacent source cells.
    """
    revisions = tuple(model.source_revision for model in state.models)
    if classification.completeness.source_revisions != revisions:
        raise BRepBooleanFailure(
            "BRepBoolean.material.classification must bind the current source revisions; got stale source revisions",
            revisions,
        )
    if target_model.topology.num_solids:
        if target_query is None:
            raise BRepBooleanFailure(
                "BRepBoolean.material.target_query must be defined for nonempty targets; got None"
            )
        _validate_projection(target_model, target_query)
    elif target_model.topology.num_faces:
        raise BRepBooleanFailure(
            "BRepBoolean.material.target must have no faces when it has no solids; got unowned boundary faces"
        )
    if (
        target_solid_regions is not None
        and len(target_solid_regions) != target_model.topology.num_solids
    ):
        raise BRepBooleanFailure(
            "BRepBoolean.material.target_solid_regions must bind every target solid once; got a cardinality mismatch"
        )
    bound, target_completion = _target_coverage(
        state,
        classification,
        target_model,
        target_face_cells,
        target_cell_representatives,
    )
    context = _Context(
        state,
        canonical_fingerprint(
            {
                "source_overlay": classification.completeness.certificate_id,
                "source_classification": classification.certificate_id,
                "target_boundary": target_completion,
            }
        ),
    )
    target_labels = _label_target_cells(
        context,
        classification,
        target_model,
        target_query,
        bound,
        target_solid_regions,
    ).labels
    by_cell = {item.classification_key: item.solids for item in target_labels}
    sources = tuple(
        MaterialSolid(operand, entity)
        for operand, model in enumerate(state.models)
        for entity in _entity_occurrences(model, "solid")
    )
    target_solids = _entity_occurrences(target_model, "solid")
    overlaps = tuple(
        _pair_overlap(
            source,
            target,
            state,
            classification,
            by_cell,
            target_face_cells,
            target_model,
            context.completeness_id,
        )
        for source in sources
        for target in target_solids
    )
    if state.selection == "difference" and any(
        item.overlaps and item.source.model_index == 1 for item in overlaps
    ):
        raise BRepBooleanFailure(
            "BRepBoolean.material.difference must exclude subtractor interiors; got an overlapping target",
            tuple(
                key
                for item in overlaps
                if item.overlaps and item.source.model_index == 1
                for key in item.witness_cells
            ),
        )
    entity_pairs = tuple(
        sorted(
            ((item.source, item.target) for item in overlaps if item.overlaps),
            key=lambda pair: (
                pair[0].model_index,
                pair[0].entity,
                pair[1],
            ),
        )
    )
    mapped_sources = {source for source, _ in entity_pairs}
    deleted_sources = tuple(source for source in sources if source not in mapped_sources)
    certificate = canonical_fingerprint(
        {
            "kind": "native-exhaustive-solid-material-ancestry",
            "source": classification.certificate_id,
            "target": target_completion,
            "overlaps": tuple(item.certificate_id for item in overlaps),
            "pairs": tuple(
                (asdict(source), asdict(target)) for source, target in entity_pairs
            ),
            "deleted_sources": tuple(asdict(source) for source in deleted_sources),
        }
    )
    return MaterialOverlapSchedule(
        overlaps,
        tuple(target_labels),
        entity_pairs,
        deleted_sources,
        certificate,
    )
