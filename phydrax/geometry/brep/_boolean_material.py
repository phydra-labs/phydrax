#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Enclosure-certified open-cell material labels and exhaustive solid ancestry.

An interior point is a proposal, never an absence proof. A non-boundary label
requires a physical enclosure whose radius is strictly smaller than the native
boundary-distance lower bound. The complete face-pair/DCEL premise then extends
that local decision across its connected open cell. Coincident material sides
come from exact chart correspondence and authored solid-face coorientation.

Overlap is an open-volume question. Every source/target pair is scheduled, and
absence requires checking both complete boundaries. Opposed coincident faces
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
from typing import Protocol, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._fingerprint import canonical_fingerprint
from .._atlas import AbstractTrimCurve
from .._interval_enclosure import (
    interval_add,
    interval_multiply,
    interval_subtract,
    prepare_interval_function,
    PreparedIntervalFunction,
)
from ._boolean import (
    _material_owner,
    _selection_payload,
    _side_payload,
    BRepBooleanFailure,
)
from ._boolean_cells import FaceArrangementCell, FaceArrangementCells
from ._boolean_coincidence import (
    CoincidentFaceOverlay,
    CoincidentSurfaceCorrespondence,
    prove_surface_correspondence,
    transport_coincident_pcurve,
)
from ._containment import certify_solid_containment, SolidContainmentCertificate
from ._intersection import SurfaceIntersectionResult
from ._intersection_curve import (
    AffinePCurve,
    coefficient_enclosures,
    CurveTrimSegment,
    pcurve_affine_correspondence,
    pcurve_surface_period_shifts,
    PeriodicPCurve,
    surface_pieces_for_box,
    SurfaceEvaluator,
)
from ._model import BRepEntityId, BRepModel
from ._query import BRepQueryBudget, PreparedBRepQuery


if TYPE_CHECKING:
    from ._boolean_arrangement import _Arrangement, _Piece


type FaceOwner = tuple[int, int]
type FacePair = tuple[FaceOwner, FaceOwner]
type ParameterBox = tuple[tuple[float, float], tuple[float, float]]
type PhysicalBox = tuple[tuple[float, float, float], tuple[float, float, float]]


class BooleanMaterialBudget(Protocol):
    """The same counter used by discovery, cell extraction, and publication."""

    def consume(self, boxes: int) -> None: ...

    @property
    def remaining_boxes(self) -> int: ...


class MaterialLabelRoute(Enum):
    COORIENTATION = "exact-coorientation"
    ENCLOSURE = "physical-enclosure-separation"
    OUTSIDE_BOUNDS = "conservative-solid-bounds"
    BOOLEAN_SELECTION = "complete-boolean-material-selection"


@dataclass(frozen=True, slots=True)
class MaterialSolid:
    model_index: int
    entity: BRepEntityId

    @property
    def source_id(self) -> str:
        return f"operand:{self.model_index}/solid:{self.entity.index}"


@dataclass(frozen=True, slots=True)
class MaterialEnclosureEvidence:
    parameter_box: ParameterBox
    physical_box: PhysicalBox
    center: tuple[float, float, float]
    radius_upper: float
    distance_lower: float | None
    query_id: str | None
    containment: SolidContainmentCertificate | None = None


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
    """Material owner on each side of one complete source cell.

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
class MaterialWork:
    classification_work_units: int
    query_box_upper_bound: int
    solid_labels: int


@dataclass(frozen=True, slots=True)
class FaceMaterialClassification:
    cells: tuple[FaceArrangementCell, ...]
    labels: tuple[FaceMaterialLabel, ...]
    completeness: MaterialOverlayCompleteness
    work: MaterialWork
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
    canonical_pairs: tuple[tuple[str, str], ...]
    deleted_source_ids: tuple[str, ...]
    work: MaterialWork
    certificate_id: str


@dataclass
class _Work:
    budget: BooleanMaterialBudget
    classification_work_units: int = 0
    query_box_upper_bound: int = 0
    solid_labels: int = 0

    def consume(self, count: int, cell: str, solid: str, *, query: bool = False) -> None:
        try:
            self.budget.consume(count)
        except BRepBooleanFailure as error:
            raise BRepBooleanFailure(
                f"material classification shared budget exhausted ({error.reason})",
                (cell, solid),
            ) from error
        if query:
            self.query_box_upper_bound += count
        else:
            self.classification_work_units += count

    def evidence(self) -> MaterialWork:
        return MaterialWork(
            self.classification_work_units, self.query_box_upper_bound, self.solid_labels
        )


@dataclass(frozen=True, slots=True)
class _SurfaceBox:
    lower: np.ndarray
    upper: np.ndarray
    value: PreparedIntervalFunction
    normal: PreparedIntervalFunction


@dataclass
class _Context:
    state: _Arrangement
    work: _Work
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
    state: _Arrangement,
    pair: FacePair,
    surface_results: Mapping[FacePair, SurfaceIntersectionResult],
    coincident_overlays: Mapping[FacePair, CoincidentFaceOverlay],
    work: _Work,
) -> FacePairMaterialEvidence:
    first, second = pair
    models = state.models
    a, b = models[first[0]], models[second[0]]
    work.consume(1, f"face:{first}", f"face:{second}")
    box_a = a.patches[first[1]].bounding_box(np.asarray(a.parameter_bounds)[first[1]])
    box_b = b.patches[second[1]].bounding_box(np.asarray(b.parameter_bounds)[second[1]])
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
                "Source adjacency cannot cross operand definitions.",
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
                "coincident completion has stale source-face identity", (str(pair),)
            )
        return FacePairMaterialEvidence(pair, overlay.certificate_id, ())
    if pair in state.source_pair_coverage:
        proof = state.source_pair_coverage[pair]
        if proof.faces != (a.face_ids[first[1]], b.face_ids[second[1]]):
            raise BRepBooleanFailure(
                "source-edge completion has stale source-face identity", (str(pair),)
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
            "complete all-source face-pair overlay is missing or unresolved",
            (
                f"{a.source_revision}:face:{first[1]}",
                f"{b.source_revision}:face:{second[1]}",
                *details,
            ),
        )
    if result.coincident:
        raise BRepBooleanFailure(
            "coincident face-pair completion needs exact full-trim overlay", (str(pair),)
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
    state: _Arrangement,
    pair: FacePair,
    coverage: Mapping[FaceOwner, FaceArrangementCells],
    result: SurfaceIntersectionResult | None,
    overlay: CoincidentFaceOverlay | None,
) -> None:
    if overlay is not None:
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
                    "coincident cell graph omits authored overlay coedges", (str(owner),)
                )
        return
    if pair in state.source_adjacencies:
        if pair[0][0] != pair[1][0]:
            raise BRepBooleanFailure(
                "Source adjacency cannot cross operand definitions.",
                (str(pair),),
            )
        model_index = pair[0][0]
        for owner in pair:
            graph = coverage[owner]
            tokens = {edge[0] for edge in graph.covered_halfedges}
            tokens.update(
                token for sources in graph.quotient_sources.values() for token in sources
            )
            for edge in state.source_adjacencies[pair]:
                if not _source_edge_covered(
                    state, graph, tokens, owner, (model_index, edge)
                ):
                    raise BRepBooleanFailure(
                        "Complete cell graph omits an authoritative source adjacency.",
                        (str(owner), f"edge:{edge}"),
                    )
        return
    if pair in state.source_pair_coverage:
        proof = state.source_pair_coverage[pair]
        for owner in pair:
            graph = coverage[owner]
            tokens = {edge[0] for edge in graph.covered_halfedges}
            tokens.update(
                token for sources in graph.quotient_sources.values() for token in sources
            )
            if not _source_edge_covered(state, graph, tokens, owner, proof.source_edge):
                raise BRepBooleanFailure(
                    "complete cell graph omits its exact source-edge pair coverage",
                    (str(owner), proof.certificate_id),
                )
        return
    if result is None:
        return
    for curve in result.curves:
        for owner in pair:
            graph = coverage[owner]
            tokens = {edge[0] for edge in graph.covered_halfedges}
            tokens.update(
                token for sources in graph.quotient_sources.values() for token in sources
            )
            arcs = tuple(
                arc
                for arc in state.arcs[owner]
                if arc.branch is not None and arc.branch.branch_id == curve.branch_id
            )
            if arcs:
                covered = all(arc.token in tokens for arc in arcs)
            else:
                # A side proved to BE a present source edge (the face's own or
                # a coincident partner's overlay arc) is covered by that edge's
                # arcs; anything else is an omitted branch.
                covered = _source_edge_covered(
                    state,
                    graph,
                    tokens,
                    owner,
                    state.branch_source_edges.get(curve.branch_id, {}).get(owner),
                )
            if not covered:
                raise BRepBooleanFailure(
                    "complete cell graph omits a certified boundary branch",
                    (
                        str(owner),
                        curve.branch_id,
                    ),
                )


def _source_edge_covered(
    state: _Arrangement,
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
        if ":atom:" not in token
        and source.edge_id.source_revision == revision
        and source.edge_id.index == index
    ]
    return any(
        token == candidate or token.startswith(f"{candidate}:atom:")
        for candidate in candidates
        for token in tokens
    )


def _certify_overlay(
    state: _Arrangement,
    coverages: tuple[FaceArrangementCells, ...],
    surface_results: Mapping[FacePair, SurfaceIntersectionResult],
    coincident_overlays: Mapping[FacePair, CoincidentFaceOverlay],
    work: _Work,
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
            "full material overlay must cover every source-solid boundary face once",
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
                "material overlay contains unresolved open cells or roots",
                (
                    str(coverage.owner),
                    *coverage.unresolved_roots,
                    *details,
                ),
            )
        if not coverage.cells:
            raise BRepBooleanFailure(
                "nonempty source face has no complete open material cell",
                (str(coverage.owner),),
            )
        if any(cell.owner != coverage.owner for cell in coverage.cells):
            raise BRepBooleanFailure(
                "face-cell coverage changed its authoritative owner",
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
            "complete material cells have duplicate classification identity", keys
        )
    pairs = tuple(
        _pair_evidence(state, pair, surface_results, coincident_overlays, work)
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


def _normal_function(evaluator: SurfaceEvaluator, parameters: Array) -> Array:
    derivatives = jax.jacfwd(evaluator.evaluate)(parameters)
    return jnp.cross(derivatives[:, 0], derivatives[:, 1])


def _surface_boxes(context: _Context, owner: FaceOwner) -> tuple[_SurfaceBox, ...]:
    if owner in context.surface_boxes:
        return context.surface_boxes[owner]
    model = context.state.models[owner[0]]
    pieces = surface_pieces_for_box(
        model.patches[owner[1]], np.asarray(model.parameter_bounds)[owner[1]]
    )
    boxes = []
    for piece in pieces:
        coefficients = coefficient_enclosures(piece.evaluator)

        def normal(
            parameters: Array, evaluator: SurfaceEvaluator = piece.evaluator
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
    lower, upper = prepared.value.evaluate(parameter_box[0][None], parameter_box[1][None])
    normal_lower, normal_upper = prepared.normal.evaluate(
        parameter_box[0][None], parameter_box[1][None]
    )
    if not np.all(np.isfinite((lower, upper, normal_lower, normal_upper))) or not np.any(
        (normal_lower[0] > 0.0) | (normal_upper[0] < 0.0)
    ):
        return None
    physical = np.stack((lower[0], upper[0]))
    center = 0.5 * physical[0] + 0.5 * physical[1]
    delta = interval_subtract((physical[0], physical[1]), (center, center))
    radii = np.maximum(np.abs(delta[0]), np.abs(delta[1]))
    squares = interval_multiply((radii, radii), (radii, radii))
    total = np.asarray(0.0, dtype=np.float64), np.asarray(0.0, dtype=np.float64)
    for axis in range(3):
        total = interval_add(total, (squares[0][axis], squares[1][axis]))
    radius = float(np.nextafter(np.sqrt(total[1]), np.inf))
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


def _fraction_endpoint(value: Fraction, *, upper: bool) -> float:
    result = float(value)
    represented = Fraction(result)
    if (upper and represented < value) or (not upper and represented > value):
        result = float(np.nextafter(result, np.inf if upper else -np.inf))
    return result


def _transport_box(proof: CoincidentSurfaceCorrespondence, box: np.ndarray) -> np.ndarray:
    result = np.empty((2, 2), dtype=np.float64)
    for row in range(2):
        lower = upper = proof.offset[row]
        for axis in range(2):
            first = proof.matrix[row][axis] * Fraction(float(box[0, axis]))
            last = proof.matrix[row][axis] * Fraction(float(box[1, axis]))
            lower += min(first, last)
            upper += max(first, last)
        result[0, row] = _fraction_endpoint(lower, upper=False)
        result[1, row] = _fraction_endpoint(upper, upper=True)
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
    query: PreparedBRepQuery,
    solid: int,
    box: np.ndarray,
    known_faces: tuple[int, ...],
) -> tuple[tuple[bool, bool], tuple[BRepEntityId, ...]] | None:
    from ._boolean_arrangement import _trim_box_status

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
            decision = _trim_box_status(
                query.faces[face].trim_domain, _transport_box(proof, box)
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


def _validate_query(model: BRepModel, query: PreparedBRepQuery) -> None:
    if query.model_id != model.model_id or query.source_revision != model.source_revision:
        raise BRepBooleanFailure(
            "material query is not bound to the exact model revision",
            (model.source_revision,),
        )
    if not query.certify_on_host:
        raise BRepBooleanFailure(
            "material classification requires native host-certified query evidence",
            (model.source_revision,),
        )
    present = {occurrence.solid for occurrence in query.occurrences}
    missing = tuple(
        f"{model.source_revision}:solid:{solid}"
        for solid in range(model.topology.num_solids)
        if solid not in present
    )
    if missing:
        raise BRepBooleanFailure(
            "material solid has no authoritative occurrence query", missing
        )
    for occurrence in query.occurrences:
        if not np.array_equal(
            np.asarray(occurrence.rotation), np.eye(3, dtype=np.float64)
        ) or np.any(np.asarray(occurrence.translation) != 0.0):
            raise BRepBooleanFailure(
                "material face overlay needs occurrence-lowered exact charts",
                (
                    model.source_revision,
                    "/".join(occurrence.path),
                ),
            )


def _outside_bounds(
    query: PreparedBRepQuery, solid: int, enclosure: MaterialEnclosureEvidence
) -> bool:
    faces = np.asarray(query.solid_faces[solid], dtype=np.int32)
    boxes = np.asarray(query.face_boxes)[faces]
    physical = np.asarray(enclosure.physical_box, dtype=np.float64)
    return bool(
        np.any(physical[1] < np.min(boxes[:, 0], axis=0))
        or np.any(physical[0] > np.max(boxes[:, 1], axis=0))
    )


def _query_separation(
    context: _Context,
    cell: FaceArrangementCell,
    reference: MaterialSolid,
    query: PreparedBRepQuery,
    enclosure: MaterialEnclosureEvidence,
) -> tuple[bool, MaterialEnclosureEvidence] | None:
    faces = query.solid_faces[reference.entity.index]
    # Allocate a true owner-work allowance while retaining an independently
    # tightened subdivision cap. Identity placements were proved above.
    owner_allowance = (context.work.budget.remaining_boxes - 1) // 2
    available = min(query.policy.maximum_subdivisions, owner_allowance)
    if available < 1:
        raise BRepBooleanFailure(
            "native distance query has no remaining shared material budget",
            (
                cell.classification_key,
                reference.source_id,
            ),
        )
    points = np.asarray(enclosure.center, dtype=np.float64)[None]
    closest = query._closest_definition(
        jnp.asarray(points),
        faces,
        maximum_boxes=available,
        budget=BRepQueryBudget(owner_allowance),
    )
    context.work.consume(
        int(np.asarray(closest.query_operations)[0]),
        cell.classification_key,
        reference.source_id,
        query=True,
    )
    distance = float(np.asarray(closest.distance_lower_bounds)[0])
    if not np.isfinite(distance) or distance <= enclosure.radius_upper:
        return None
    available = min(
        query.policy.maximum_subdivisions, context.work.budget.remaining_boxes
    )
    if available < 1:
        raise BRepBooleanFailure(
            "native containment has no remaining shared material budget",
            (
                cell.classification_key,
                reference.source_id,
            ),
        )
    # This is the same complete ray-parity owner used by PreparedBRepQuery.
    # Calling it directly avoids contains' second hidden closest-point query
    # and retains the actual bounded work and unresolved-face evidence.
    membership = certify_solid_containment(
        query.model,
        points[0],
        reference.entity.index,
        maximum_boxes=available,
        maximum_depth=query.policy.maximum_depth,
        maximum_operations=context.work.budget.remaining_boxes,
    )
    context.work.consume(
        membership.owner_operations,
        cell.classification_key,
        reference.source_id,
        query=True,
    )
    if not membership.complete:
        return None
    separated = MaterialEnclosureEvidence(
        enclosure.parameter_box,
        enclosure.physical_box,
        enclosure.center,
        enclosure.radius_upper,
        distance,
        query.query_id,
        membership,
    )
    return membership.inside, separated


def _make_label(
    context: _Context,
    cell: FaceArrangementCell,
    reference: MaterialSolid,
    sides: tuple[bool, bool],
    route: MaterialLabelRoute,
    supports: tuple[BRepEntityId, ...],
    enclosure: MaterialEnclosureEvidence,
) -> SolidMaterialLabel:
    context.work.solid_labels += 1
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
    query: PreparedBRepQuery,
    solid: int,
    known_faces: tuple[int, ...] = (),
) -> SolidMaterialLabel:
    from ._boolean_arrangement import _trim_box_status

    reference = MaterialSolid(model_index, model.solid_ids[solid])
    seed = np.asarray(cell.seed_box, dtype=np.float64)
    if (
        seed.shape != (2, 2)
        or np.any(seed[0] >= seed[1])
        or _trim_box_status(cell.domain, seed) != 1
    ):
        raise BRepBooleanFailure(
            "open material cell has no certified strict interior box",
            (cell.classification_key,),
        )
    pending: deque[tuple[_SurfaceBox, np.ndarray, int]] = deque()
    for prepared in _surface_boxes(context, cell.owner):
        lower, upper = (
            np.maximum(seed[0], prepared.lower),
            np.minimum(seed[1], prepared.upper),
        )
        if np.all(lower < upper):
            pending.append((prepared, np.stack((lower, upper)), 0))
    while pending:
        prepared, box, depth = pending.popleft()
        context.work.consume(1, cell.classification_key, reference.source_id)
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
            if _outside_bounds(query, solid, enclosure):
                return _make_label(
                    context,
                    cell,
                    reference,
                    (False, False),
                    MaterialLabelRoute.OUTSIDE_BOUNDS,
                    (),
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
        middle = 0.5 * box[0, axis] + 0.5 * box[1, axis]
        if (
            depth >= query.policy.maximum_depth
            or not box[0, axis] < middle < box[1, axis]
        ):
            continue
        first, last = box.copy(), box.copy()
        first[1, axis], last[0, axis] = middle, middle
        pending.extend(((prepared, first, depth + 1), (prepared, last, depth + 1)))
    raise BRepBooleanFailure(
        "material cell has no regular enclosure-separated or exact-cooriented label",
        (
            cell.classification_key,
            reference.source_id,
            model.source_revision,
        ),
    )


def classify_face_material_cells(
    state: _Arrangement,
    coverages: tuple[FaceArrangementCells, ...],
    /,
    *,
    surface_results: Mapping[FacePair, SurfaceIntersectionResult],
    coincident_overlays: Mapping[FacePair, CoincidentFaceOverlay],
    budget: BooleanMaterialBudget,
) -> FaceMaterialClassification:
    """Label every full source cell against every individual source solid.

    ``surface_results`` and ``coincident_overlays`` use lexicographically ordered
    face pairs, including same-operand pairs. Discovery/extraction work must
    already be charged to ``budget``; this consumer charges its own work only.
    """
    for model, query in zip(state.models, state.queries, strict=True):
        _validate_query(model, query)
    work = _Work(budget)
    cells, completeness = _certify_overlay(
        state, coverages, surface_results, coincident_overlays, work
    )
    context = _Context(state, work, completeness.certificate_id)
    labels = []
    for cell in cells:
        individual = tuple(
            _solid_label(
                context,
                cell,
                operand,
                model,
                state.queries[operand],
                solid,
                (cell.owner[1],) if operand == cell.owner[0] else (),
            )
            for operand, model in enumerate(state.models)
            for solid in range(model.topology.num_solids)
        )
        negative = _material_owner(
            state.selection,
            tuple(
                (item.solid.model_index, item.solid.entity.index)
                for item in individual
                if item.negative_inside
            ),
        )
        positive = _material_owner(
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
                "operation": _selection_payload(state.selection),
                "selected": _side_payload(state.selection, negative, positive),
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
    return FaceMaterialClassification(
        cells, tuple(labels), completeness, work.evidence(), certificate
    )


def _fixed_phase_piece(
    first: _Piece,
    second: _Piece,
    proof: CoincidentSurfaceCorrespondence,
) -> bool:
    from ._boolean_cells import _scalar_phase

    if any(
        cut.endpoint is not None
        for cut in (first.first, first.last, second.first, second.last)
    ):
        return False
    transported = replace(
        first.arc,
        pcurve=transport_coincident_pcurve(proof, first.arc.pcurve),
        trim=AffinePCurve(
            first.arc.trim,
            np.asarray(proof.matrix, dtype=np.object_),
            np.asarray(proof.offset, dtype=np.object_),
        ),
    )
    phase = _scalar_phase(transported, second.arc)
    if phase is None:
        return False
    a = (first.first, first.last) if first.sense > 0 else (first.last, first.first)
    b = (second.first, second.last) if second.sense > 0 else (second.last, second.first)
    return all(
        phase[0] * Fraction(left.parameter) + phase[1] == Fraction(right.parameter)
        for left, right in zip(a, b, strict=True)
    )


def _equivalent_piece(
    first: _Piece,
    second: _Piece,
    proof: CoincidentSurfaceCorrespondence,
    work: _Work,
) -> bool:
    from ._boolean_arrangement import _piece_trim
    from ._intersection_curve import _trim_source_join

    work.consume(1, first.arc.token, second.arc.token)
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
    # endpoint shortcut: both complete root/source transports are mandatory.
    matrix = np.asarray(proof.matrix, dtype=np.object_)
    offset = np.asarray(proof.offset, dtype=np.object_)
    transported = AffinePCurve(_piece_trim(first), matrix, offset)
    reversed_transport = AffinePCurve(
        _piece_trim(replace(first, sense=-first.sense)), matrix, offset
    )
    a: AbstractTrimCurve = (
        PeriodicPCurve(transported, proof.second, shifts) if any(shifts) else transported
    )
    reversed_a: AbstractTrimCurve = (
        PeriodicPCurve(reversed_transport, proof.second, shifts)
        if any(shifts)
        else reversed_transport
    )
    b = _piece_trim(second)
    reversed_b = _piece_trim(replace(second, sense=-second.sense))
    return _trim_source_join(a, reversed_b) and _trim_source_join(reversed_a, b)


def _equivalent_cell(
    state: _Arrangement,
    first: FaceArrangementCell,
    second: FaceArrangementCell,
    work: _Work,
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
                        piece, candidate[(offset + ordinal) % len(candidate)], proof, work
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


def _same_trim_segment(first: CurveTrimSegment, second: CurveTrimSegment) -> bool:
    from ._intersection_curve import _trim_source_join

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
        if not _trim_source_join(
            replace(first, reversed=True), replace(second, reversed=False)
        ):
            return False
    if first.last_root is not None and second.last_root is not None:
        if not _trim_source_join(
            replace(first, reversed=False), replace(second, reversed=True)
        ):
            return False
    return True


def _target_trim_copy(
    cell: FaceArrangementCell, model: BRepModel, face: int, work: _Work
) -> bool:
    from ._boolean_arrangement import _piece_trim

    geometry = model.geometry
    if geometry is None or len(geometry.face_loops[face]) != len(cell.loops):
        return False
    for coedges, pieces in zip(geometry.face_loops[face], cell.loops, strict=True):
        if len(coedges) != len(pieces):
            return False
        for coedge, piece in zip(coedges, pieces, strict=True):
            work.consume(
                1, cell.classification_key, f"{model.source_revision}:coedge:{coedge}"
            )
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
            if not _same_trim_segment(_piece_trim(piece), target):
                return False
    return True


def _target_coverage(
    state: _Arrangement,
    classification: FaceMaterialClassification,
    model: BRepModel,
    face_cells: tuple[str, ...],
    representatives: Mapping[str, str],
    work: _Work,
) -> tuple[dict[str, tuple[int, ...]], str]:
    if len(face_cells) != model.topology.num_faces:
        raise BRepBooleanFailure(
            "target material coverage must bind every target face to a complete source cell"
        )
    cells = {cell.classification_key: cell for cell in classification.cells}
    labels = {label.classification_key: label for label in classification.labels}
    bound: dict[str, list[int]] = {}
    for face, key in enumerate(face_cells):
        work.consume(1, key, f"{model.source_revision}:face:{face}")
        if key not in cells:
            raise BRepBooleanFailure(
                "target face has no authoritative complete source-cell identity", (key,)
            )
        cell = cells[key]
        original = state.models[cell.owner[0]].patches[cell.owner[1]]
        proof = prove_surface_correspondence(original, model.patches[face])
        if (
            proof is None
            or proof.matrix != ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
            or proof.offset != (Fraction(0), Fraction(0))
            or not _target_trim_copy(cell, model, face, work)
        ):
            raise BRepBooleanFailure(
                "target face is not an exact chart-and-trim copy of its complete source cell",
                (key,),
            )
        if int(float(model.orientation[face])) != labels[key].orientation:
            raise BRepBooleanFailure(
                "target face orientation contradicts complete Boolean material sides",
                (key,),
            )
        if not model.topology.face_solids[face]:
            raise BRepBooleanFailure(
                "published target face has no material solid owner", (key,)
            )
        bound.setdefault(key, []).append(face)
    expected = {
        label.classification_key
        for label in classification.labels
        if label.orientation != 0
    }
    if not set(bound).issubset(expected):
        raise BRepBooleanFailure(
            "target boundary retained an unselected material cell",
            tuple(sorted(set(bound) - expected)),
        )
    missing = expected - set(bound)
    if set(representatives) != missing:
        raise BRepBooleanFailure(
            "target boundary omits selected cells without exhaustive exact representatives",
            tuple(sorted(missing)),
        )
    for key, representative in sorted(representatives.items()):
        if representative not in bound or not _equivalent_cell(
            state, cells[key], cells[representative], work
        ):
            raise BRepBooleanFailure(
                "coincident target-cell representative has no exact complete-loop proof",
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
                "coincident representative contradicts material coorientation",
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
    state: _Arrangement,
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
        individual = next(
            item for item in source_labels[key].solids if item.solid == source
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


def prove_material_overlaps(
    state: _Arrangement,
    classification: FaceMaterialClassification,
    target_model: BRepModel,
    target_query: PreparedBRepQuery | None,
    target_face_cells: tuple[str, ...],
    /,
    *,
    budget: BooleanMaterialBudget,
    target_cell_representatives: Mapping[str, str],
    target_solid_regions: tuple[str, ...] | None = None,
) -> MaterialOverlapSchedule:
    """Schedule every individual source-solid/target-solid pair independently.

    Each target face must be copied from one full cell and retain that cell's
    classification identity. ``target_query`` is unnecessary only for an empty
    target. Canonical pairs contain solid ancestry only; callers append exact
    face lineage separately, including retained difference-subtractor faces.
    ``target_cell_representatives`` must bind exactly the omitted selected
    coincident cells to emitted cells; complete loop equality is proved here.
    Supply an empty mapping when no physical boundary copies are suppressed.
    A bijective ``target_solid_regions`` tuple binds partition side owners
    directly to target solids, avoiding an inexact distance query at exact
    root-adjacent source cells.
    """
    revisions = tuple(model.source_revision for model in state.models)
    if classification.completeness.source_revisions != revisions:
        raise BRepBooleanFailure(
            "material classification belongs to stale source revisions", revisions
        )
    if target_model.topology.num_solids:
        if target_query is None:
            raise BRepBooleanFailure(
                "nonempty target material overlap requires its native certified query"
            )
        _validate_query(target_model, target_query)
    elif target_model.topology.num_faces:
        raise BRepBooleanFailure(
            "empty regularized target still has unowned boundary faces"
        )
    if (
        target_solid_regions is not None
        and len(target_solid_regions) != target_model.topology.num_solids
    ):
        raise BRepBooleanFailure(
            "target solid-region authority must bind every target solid exactly once"
        )
    work = _Work(budget)
    bound, target_completion = _target_coverage(
        state,
        classification,
        target_model,
        target_face_cells,
        target_cell_representatives,
        work,
    )
    context = _Context(
        state,
        work,
        canonical_fingerprint(
            {
                "source_overlay": classification.completeness.certificate_id,
                "source_classification": classification.certificate_id,
                "target_boundary": target_completion,
            }
        ),
    )
    target_labels = []
    exact_regions = (
        target_solid_regions
        if target_solid_regions is not None
        and len(set(target_solid_regions)) == len(target_solid_regions)
        else None
    )
    if target_query is not None and target_model.topology.num_solids:
        for cell in classification.cells:
            if exact_regions is not None or target_model.topology.num_solids == 1:
                source_label = next(
                    label
                    for label in classification.labels
                    if label.classification_key == cell.classification_key
                )
                owner_label = next(
                    label
                    for label in source_label.solids
                    if label.solid.model_index == cell.owner[0]
                    and cell.owner[1]
                    in state.models[cell.owner[0]].topology.solid_faces[
                        label.solid.entity.index
                    ]
                )
            if exact_regions is not None:
                labels = tuple(
                    _make_label(
                        context,
                        cell,
                        MaterialSolid(len(state.models), target_model.solid_ids[solid]),
                        (
                            source_label.negative_owner == region,
                            source_label.positive_owner == region,
                        ),
                        MaterialLabelRoute.BOOLEAN_SELECTION,
                        tuple(
                            target_model.face_ids[face]
                            for face in bound.get(cell.classification_key, ())
                            if solid in target_model.topology.face_solids[face]
                        ),
                        owner_label.enclosure,
                    )
                    for solid, region in enumerate(exact_regions)
                )
                for label in labels:
                    work.consume(1, cell.classification_key, label.solid.source_id)
            elif target_model.topology.num_solids == 1:
                # Exhaustive exact oriented boundary equality identifies the
                # bounded regularized target with the selected material set.
                # A single target component therefore inherits both Boolean
                # sides, including buried source cells; no distance query on
                # their coincident but excluded supporting faces is needed.
                work.consume(
                    1, cell.classification_key, f"operand:{len(state.models)}/solid:0"
                )
                labels = (
                    _make_label(
                        context,
                        cell,
                        MaterialSolid(len(state.models), target_model.solid_ids[0]),
                        (source_label.negative_selected, source_label.positive_selected),
                        MaterialLabelRoute.BOOLEAN_SELECTION,
                        tuple(
                            target_model.face_ids[face]
                            for face in bound.get(cell.classification_key, ())
                        ),
                        owner_label.enclosure,
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
            target_labels.append(TargetCellMaterialLabel(cell.classification_key, labels))
    by_cell = {item.classification_key: item.solids for item in target_labels}
    sources = tuple(
        MaterialSolid(operand, model.solid_ids[solid])
        for operand, model in enumerate(state.models)
        for solid in range(model.topology.num_solids)
    )
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
        for target in target_model.solid_ids
    )
    if state.selection == "difference" and any(
        item.overlaps and item.source.model_index == 1 for item in overlaps
    ):
        raise BRepBooleanFailure(
            "published difference target overlaps excluded subtractor material",
            tuple(
                key
                for item in overlaps
                if item.overlaps and item.source.model_index == 1
                for key in item.witness_cells
            ),
        )
    pairs = tuple(
        sorted(
            (item.source.source_id, f"solid:{item.target.index}")
            for item in overlaps
            if item.overlaps
        )
    )
    mapped = {first for first, _ in pairs}
    deleted = tuple(
        source.source_id for source in sources if source.source_id not in mapped
    )
    certificate = canonical_fingerprint(
        {
            "kind": "native-exhaustive-solid-material-ancestry",
            "source": classification.certificate_id,
            "target": target_completion,
            "overlaps": tuple(item.certificate_id for item in overlaps),
            "pairs": pairs,
        }
    )
    return MaterialOverlapSchedule(
        overlaps, tuple(target_labels), pairs, deleted, work.evidence(), certificate
    )
