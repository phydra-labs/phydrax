#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Construct exhaustive face overlays, regularized material, and Boolean lineage.

Discovery includes all source face pairs, not just the operands' visible
boundaries. The exhaustive source graphs survive until solid ancestry has been
proved. Construction consumes selected cells, while face provenance and
open-volume material provenance remain separate scientific relations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from itertools import combinations
from types import MappingProxyType
from typing import assert_never

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked
from .._cad_revision import (
    AssociationCoverageEvidence,
    AssociationGraph,
    OccurrenceCorrespondence,
    OccurrenceCorrespondenceTransaction,
)
from ._boolean_arrangement import (
    add_root_cut,
    Arc,
    Arrangement,
    boundary_spatial_events,
    branch_arc,
    build_boundary,
    BuiltBoundary,
    check_surface_result,
    event_vertex,
    FaceFragment,
    FaceOwner,
    intersect_arc_pair,
    overlap,
    periodic_transition,
    propagate_branch_cuts,
    publish_curved,
    reconstruct_region_solids,
    reconstruct_solids,
    remaining_intersection_policy,
    source_arcs,
)
from ._boolean_cells import (
    extract_face_arrangement_cells,
    FaceArrangementCell,
    FaceArrangementCells,
    prepare_face_cell_workset,
)
from ._boolean_coincidence import (
    CoincidentFaceArc,
    CoincidentFaceOverlay,
    prepare_coincident_face_overlay,
    prove_source_ring_pair_coverage,
    prove_surface_correspondence,
)
from ._boolean_core import (
    BRepBooleanFailure,
    BRepBooleanOperation,
    BRepBooleanPolicy,
    BRepBooleanResult,
    MaterialSelection,
    RegionSelection,
    selection_payload,
    source_revision,
)
from ._boolean_material import (
    classify_face_material_cells,
    equivalent_cell,
    FaceMaterialClassification,
    FacePair,
    MaterialOverlapSchedule,
    prove_material_overlaps,
    Work,
)
from ._coordinate_request import coordinate_source_request
from ._intersection import (
    exact_source_lift,
    intersect_surface_regions,
    joint_root_parameter_image,
    native_edge_pcurve,
    prepare,
    same_geometry,
    SurfaceIntersectionResult,
    TrimIntersectionRoot,
    TripleSurfaceIntersectionRoot,
    TripleSurfaceSystem,
)
from ._intersection_curve import (
    IntersectionCurve,
    original_trim_intersection_preparation,
    surface_pieces,
    SurfaceRegion,
)
from ._intersection_kernel import krawczyk
from ._model import BRepCurve, BRepModel
from ._patches import AbstractCurve, AbstractSurfacePatch
from ._projection_contracts import brep_entity_id
from ._query import prepare_brep_query
from ._request_preparations import TrimRequestPreparations
from ._revision import cad_revision_from_brep_model
from ._root_bindings import BRepCurveSurfaceLift, BRepRootSupport, BRepVertexRoot


type JointOwners = tuple[FaceOwner, FaceOwner, FaceOwner]


class _Discovery(StrictModule):
    surface_results: Mapping[FacePair, SurfaceIntersectionResult]
    coincident_overlays: Mapping[FacePair, CoincidentFaceOverlay]
    owner_overlays: Mapping[FaceOwner, tuple[CoincidentFaceOverlay, ...]]


@dataclass
class _Joint:
    root: TripleSurfaceIntersectionRoot
    vertex: str
    supports: list[BRepRootSupport]


@dataclass
class _RootClosure:
    state: Arrangement
    joints: dict[JointOwners, list[_Joint]] = field(default_factory=dict)

    def bind(
        self,
        owners: JointOwners,
        record: _Joint,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        """Merge existing source vertices only through the joint source system."""
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    owners,
                    record,
                )
            ) as budget:
                return self.bind(owners, record, preparations=preparations, budget=budget)
        state = self.state
        joint = record.root
        matches = []
        lifts_by_id: dict[str, BRepCurveSurfaceLift] = {}
        for vertex, spatial in tuple(state.vertex_spatial.items()):
            state.consume(1)
            lifts = {lift.lift_id: lift for lift in state.vertex_lifts.get(vertex, ())}
            for operand, face in owners:
                model = state.models[operand]
                geometry = model.geometry
                if geometry is None:
                    raise BRepBooleanFailure(
                        "BRepBoolean.joint_lift.geometry must be defined; got None"
                    )
                region = _source_region(
                    state, (operand, face), preparations=preparations, budget=budget
                )
                for loop in geometry.face_loops[face]:
                    for coedge in loop:
                        edge = geometry.coedge_edges[coedge]
                        curve = geometry.curves[geometry.edge_curves[edge]]
                        pcurve = geometry.pcurves[coedge]
                        first, last = (
                            float(value)
                            for value in np.asarray(geometry.edge_ranges)[edge]
                        )
                        if (
                            isinstance(curve, AbstractCurve)
                            and isinstance(pcurve, AbstractCurve)
                            and same_geometry(curve, spatial.curve)
                            and exact_source_lift(curve, pcurve, region, first, last)
                        ):
                            lift = BRepCurveSurfaceLift(
                                curve,
                                pcurve,
                                region,
                                first,
                                last,
                                budget=budget,
                                preparations=preparations,
                            )
                            lifts[lift.lift_id] = lift
            if joint.certifies_root(
                spatial,
                source_edge_lifts=tuple(lifts.values()),
                preparations=preparations,
                budget=budget,
            ):
                matches.append((vertex, spatial))
                lifts_by_id.update(lifts)
        old_vertices = {vertex for vertex, _ in matches}
        supports = {support.root_id: support for support in record.supports}
        for vertex in old_vertices:
            for patch, root in state.vertex_aliases.get(vertex, ()):
                supports.setdefault(
                    root.root_id,
                    BRepRootSupport(
                        patch, root, budget=budget, preparations=preparations
                    ),
                )
        record.supports = list(supports.values())
        spatial = matches[0][1] if matches else None
        binding = BRepVertexRoot(
            joint,
            aliases=tuple(record.supports),
            spatial_root=spatial,
            source_edge_lifts=tuple(lifts_by_id.values()),
            budget=budget,
            preparations=preparations,
        )
        for token, cuts in state.cuts.items():
            state.cuts[token] = [
                replace(cut, vertex=record.vertex, vertex_root=binding)
                if cut.vertex in old_vertices or cut.vertex == record.vertex
                else cut
                for cut in cuts
            ]
        for vertex in old_vertices - {record.vertex}:
            state.vertex_spatial.pop(vertex, None)
            state.vertex_joint.pop(vertex, None)
            state.vertex_bindings.pop(vertex, None)
            state.vertex_aliases.pop(vertex, None)
            state.vertex_lifts.pop(vertex, None)
        if spatial is not None:
            state.vertex_spatial[record.vertex] = spatial
        state.vertex_joint[record.vertex] = joint
        state.vertex_bindings[record.vertex] = binding
        state.vertex_aliases[record.vertex] = [
            (support.patch, support.root) for support in record.supports
        ]
        state.vertex_lifts[record.vertex] = tuple(lifts_by_id.values())


class _SelectedCells(StrictModule):
    cells: tuple[FaceArrangementCell, ...]
    fragments: tuple[FaceFragment, ...]
    representatives: Mapping[str, str] = eqx.field(static=True)
    face_parents: tuple[tuple[FaceOwner, ...], ...] = eqx.field(static=True)


class _BoundaryFacts(StrictModule):
    sources: Mapping[str, CoincidentFaceArc]
    vertex_aliases: Mapping[str, str] = eqx.field(static=True)
    bindings: Mapping[str, BRepVertexRoot | None]


def _source_region(
    state: Arrangement,
    owner: FaceOwner,
    *,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
) -> SurfaceRegion:
    return state.face_region(owner, preparations=preparations, budget=budget)


def _register_branches(
    state: Arrangement,
    result: SurfaceIntersectionResult,
    pair: FacePair,
) -> None:
    for curve in result.curves:
        state.branches[curve.branch_id] = curve, pair[0], pair[1]
        if len(state.branches) > state.policy.maximum_cells:
            raise BRepBooleanFailure(
                "BRepBoolean.overlay.branches must fit maximum_cells; got branch capacity exhausted",
                (curve.branch_id,),
            )
        # On a side where the branch is a proved source edge, that edge's
        # existing arc is the branch; no duplicate branch arc is created.
        edges = state.branch_source_edges.get(curve.branch_id, {})
        boundaries = (
            0,
            *(
                chart + 1
                for chart in range(curve.num_charts - 1)
                if periodic_transition(curve, chart)
            ),
            curve.num_charts,
        )
        for first, last in zip(boundaries[:-1], boundaries[1:], strict=True):
            if pair[0] not in edges:
                branch_arc(state, curve, pair[0], "first", first, last, owners=pair)
            if pair[1] not in edges:
                branch_arc(state, curve, pair[1], "second", first, last, owners=pair)


class _EdgeLift(StrictModule):
    edge: tuple[int, int] = eqx.field(static=True)
    first_pcurve: AbstractCurve
    second_pcurve: AbstractCurve
    first: float = eqx.field(static=True)
    last: float = eqx.field(static=True)


def _source_lift(
    curve: BRepCurve,
    patch: AbstractSurfacePatch,
    bounds: np.ndarray,
    first: float,
    last: float,
    /,
    *,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
) -> AbstractCurve | None:
    """Return a native source-edge lift onto another surface, or None."""
    match curve:
        case AbstractCurve():
            if curve.ambient_dimension != 3:
                return None
        case IntersectionCurve():
            return None
        case _:
            assert_never(curve)
    region = SurfaceRegion(patch, bounds, preparations=preparations, budget=budget)
    pcurve = native_edge_pcurve(curve, region)
    if pcurve is None or not exact_source_lift(curve, pcurve, region, first, last):
        return None
    return pcurve


def _face_edge_pcurve(
    state: Arrangement,
    face: FaceOwner,
    model_index: int,
    edge: int,
    *,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
) -> AbstractCurve | None:
    """The edge's p-curve on ``face``: its authored coedge or a native lift."""
    geometry = state.models[model_index].geometry
    target = state.models[face[0]]
    target_geometry = target.geometry
    if geometry is None or target_geometry is None:
        raise BRepBooleanFailure(
            "BRepBoolean.source_edge.geometry must be defined for identification; got None",
            (str(face),),
        )
    if model_index == face[0]:
        for loop in target_geometry.face_loops[face[1]]:
            for coedge in loop:
                if target_geometry.coedge_edges[coedge] == edge:
                    pcurve = target_geometry.pcurves[coedge]
                    return pcurve if isinstance(pcurve, AbstractCurve) else None
    first, last = (float(value) for value in np.asarray(geometry.edge_ranges)[edge])
    return _source_lift(
        geometry.curves[geometry.edge_curves[edge]],
        target.patches[face[1]],
        state.face_chart(face, preparations=preparations, budget=budget),
        first,
        last,
        preparations=preparations,
        budget=budget,
    )


def _edge_lifts(
    state: Arrangement,
    owner: FaceOwner,
    other: FaceOwner,
    partners: tuple[FaceOwner, ...],
    /,
    *,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
) -> tuple[_EdgeLift, ...]:
    """Source edges present on ``owner`` with p-curves on ``owner`` and ``other``.

    Present edges are the owner's own edges and those of faces proved
    coincident with it (their overlay arcs lie in the owner's chart). Authored
    coedges are used where they exist; any other p-curve is a native lift.
    """
    lifts: list[_EdgeLift] = []
    for face in (owner, *partners):
        geometry = state.models[face[0]].geometry
        if geometry is None:
            raise BRepBooleanFailure(
                "BRepBoolean.source_edge.geometry must be defined for identification; got None",
                (str(face),),
            )
        for loop in geometry.face_loops[face[1]]:
            for coedge in loop:
                edge = geometry.coedge_edges[coedge]
                if geometry.edge_curves[edge] < 0:
                    continue
                here = _face_edge_pcurve(
                    state,
                    owner,
                    face[0],
                    edge,
                    preparations=preparations,
                    budget=budget,
                )
                there = (
                    None
                    if here is None
                    else _face_edge_pcurve(
                        state,
                        other,
                        face[0],
                        edge,
                        preparations=preparations,
                        budget=budget,
                    )
                )
                if here is not None and there is not None:
                    first, last = (
                        float(value) for value in np.asarray(geometry.edge_ranges)[edge]
                    )
                    lifts.append(
                        _EdgeLift(
                            edge=(face[0], edge),
                            first_pcurve=here,
                            second_pcurve=there,
                            first=first,
                            last=last,
                        )
                    )
    return tuple(lifts)


def _subarc_in_box(
    first_pcurve: AbstractCurve,
    second_pcurve: AbstractCurve,
    first: float,
    last: float,
    boxes: tuple[np.ndarray, ...],
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Some coupled source-edge sub-arc lies inside one certified chart box."""
    pending = [(first, last, 0)]
    while pending:
        lower, upper, depth = pending.pop()
        image = np.concatenate(
            (
                np.asarray(
                    first_pcurve.bounding_box(
                        lower, upper, preparations=preparations, budget=budget
                    )
                ),
                np.asarray(
                    second_pcurve.bounding_box(
                        lower, upper, preparations=preparations, budget=budget
                    )
                ),
            ),
            axis=1,
        )
        meets = False
        for box in boxes:
            if np.all(image[0] >= box[0]) and np.all(image[1] <= box[1]):
                return True
            meets |= bool(np.all(image[0] <= box[1]) and np.all(box[0] <= image[1]))
        middle = 0.5 * (lower + upper)
        if meets and depth < 16 and lower < middle < upper:
            pending.extend(((lower, middle, depth + 1), (middle, upper, depth + 1)))
    return False


def _chart_boxes(curve: IntersectionCurve, chart: int, /) -> tuple[np.ndarray, ...]:
    """The chart's coupled box and its inward-rounded integer period images."""
    box = np.stack((curve.box_lower[chart], curve.box_upper[chart]))
    periods = [
        region.upper[axis] - region.lower[axis] if region.periodic[axis] else 0.0
        for region in (curve.first, curve.second)
        for axis in range(2)
    ]
    boxes = [box]
    for axis, period in enumerate(periods):
        if not period:
            continue
        for shift in (-1.0, 1.0):
            moved = box.copy()
            moved[:, axis] += shift * float(period)
            # Inward rounding keeps the moved box inside its symbolic period image.
            moved[0, axis] = np.nextafter(np.nextafter(moved[0, axis], np.inf), np.inf)
            moved[1, axis] = np.nextafter(np.nextafter(moved[1, axis], -np.inf), -np.inf)
            boxes.append(moved)
    return tuple(boxes)


def _branch_is_edge(
    curve: IntersectionCurve,
    lift: _EdgeLift,
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Every certified chart's unique curve piece contains an edge sub-arc.

    The lifted edge solves the pair system for every parameter, and each
    certified chart box holds exactly one solution piece, so a contained
    sub-arc identifies that piece with the edge. No numerical proximity is used.
    """
    return bool(curve.num_charts) and all(
        bool(curve.certified[chart])
        and _subarc_in_box(
            lift.first_pcurve,
            lift.second_pcurve,
            lift.first,
            lift.last,
            _chart_boxes(curve, chart),
            preparations=preparations,
            budget=budget,
        )
        for chart in range(curve.num_charts)
    )


def _without_source_edge_branches(
    state: Arrangement,
    pair: FacePair,
    result: SurfaceIntersectionResult,
    partners: Mapping[FaceOwner, tuple[FaceOwner, ...]],
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> SurfaceIntersectionResult:
    """Identify branches that are source edges present on one or both faces.

    A branch that is a present source edge of BOTH faces (shared by adjacent
    faces of one operand, coincident edges of both operands, or an edge carried
    by a coincident-face overlay) is an existing edge and cuts neither face: it
    is dropped. A branch that is such an edge on one face only is recorded as
    that edge on that side.
    """
    if not result.curves:
        return result
    a, b = pair
    first_edges = _edge_lifts(
        state,
        a,
        b,
        partners.get(a, ()),
        preparations=preparations,
        budget=budget,
    )
    second_edges = tuple(
        _EdgeLift(
            edge=lift.edge,
            first_pcurve=lift.second_pcurve,
            second_pcurve=lift.first_pcurve,
            first=lift.first,
            last=lift.last,
        )
        for lift in _edge_lifts(
            state,
            b,
            a,
            partners.get(b, ()),
            preparations=preparations,
            budget=budget,
        )
    )
    kept = []
    for curve in result.curves:
        on_first = next(
            (
                lift.edge
                for lift in first_edges
                if _branch_is_edge(curve, lift, preparations=preparations, budget=budget)
            ),
            None,
        )
        on_second = next(
            (
                lift.edge
                for lift in second_edges
                if _branch_is_edge(curve, lift, preparations=preparations, budget=budget)
            ),
            None,
        )
        if on_first is not None and on_second is not None:
            continue
        kept.append(curve)
        for owner, edge in ((a, on_first), (b, on_second)):
            if edge is not None:
                state.branch_source_edges.setdefault(curve.branch_id, {})[owner] = edge
    if len(kept) == len(result.curves):
        return result
    return SurfaceIntersectionResult(
        tuple(kept), result.points, result.coincident, result.unresolved, result.work
    )


def _represented_source_adjacency(state: Arrangement, pair: FacePair, /) -> bool:
    """Retain a same-operand edge already in its arrangement."""
    first, second = pair
    if first[0] != second[0]:
        return False
    topology = state.models[first[0]].topology
    edges = tuple(
        sorted(set(topology.face_edges[first[1]]) & set(topology.face_edges[second[1]]))
    )
    if edges:
        state.source_adjacencies[pair] = edges
    return bool(edges)


def _cover_source_edge_pair(
    state: Arrangement,
    pair: FacePair,
    partners: Mapping[FaceOwner, tuple[FaceOwner, ...]],
    *,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Prove face-pair coverage using a represented source edge."""
    first, second = pair
    first_lifts = _edge_lifts(
        state,
        first,
        second,
        partners.get(first, ()),
        preparations=preparations,
        budget=budget,
    )
    second_lifts = _edge_lifts(
        state,
        second,
        first,
        partners.get(second, ()),
        preparations=preparations,
        budget=budget,
    )
    for first_lift in first_lifts:
        for second_lift in second_lifts:
            state.consume(1)
            if first_lift.edge != second_lift.edge:
                continue
            proof = prove_source_ring_pair_coverage(
                state.models,
                pair,
                first_lift.edge,
                (first_lift.first_pcurve, second_lift.first_pcurve),
            )
            if proof is not None:
                state.source_pair_coverage[pair] = proof
                return True
    return False


def _discover_all_source_pairs(
    state: Arrangement,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> _Discovery:
    surface_results: dict[FacePair, SurfaceIntersectionResult] = {}
    coincident: dict[FacePair, CoincidentFaceOverlay] = {}
    overlays: dict[FaceOwner, list[CoincidentFaceOverlay]] = {}
    regions = {
        owner: _source_region(state, owner, preparations=preparations, budget=budget)
        for owner in sorted(state.arcs)
    }
    boxes = {
        owner: region.patch.bounding_box(
            region.parameter_box, preparations=preparations, budget=budget
        )
        for owner, region in regions.items()
    }
    candidates = []
    for pair in combinations(tuple(regions), 2):
        if _represented_source_adjacency(state, pair):
            continue
        state.consume(1)
        if not overlap(boxes[pair[0]], boxes[pair[1]]):
            continue
        a, b = pair
        overlay = prepare_coincident_face_overlay(
            state.models[a[0]],
            a[1],
            state.models[b[0]],
            b[1],
            policy=state.policy,
        )
        if overlay is None:
            candidates.append(pair)
            continue
        reverse = prepare_coincident_face_overlay(
            state.models[b[0]],
            b[1],
            state.models[a[0]],
            a[1],
            policy=state.policy,
        )
        if reverse is None:
            raise BRepBooleanFailure(
                "BRepBoolean.coincident_correspondence must have an inverse chart map; got no inverse",
                (str(pair),),
            )
        coincident[pair] = overlay
        overlays.setdefault(a, []).append(overlay)
        overlays.setdefault(b, []).append(reverse)
    # Coincidences are proved first: their overlay arcs are present edges of
    # both faces when identifying branches below.
    partners: dict[FaceOwner, tuple[FaceOwner, ...]] = {}
    for first, second in coincident:
        partners[first] = (*partners.get(first, ()), second)
        partners[second] = (*partners.get(second, ()), first)
    for pair in candidates:
        a, b = pair
        if _cover_source_edge_pair(
            state, pair, partners, preparations=preparations, budget=budget
        ):
            continue
        result = intersect_surface_regions(
            regions[a],
            regions[b],
            policy=remaining_intersection_policy(state),
            preparations=preparations,
            coordinate_budget=budget,
        )
        check_surface_result(state, result, a, b)
        result = _without_source_edge_branches(
            state, pair, result, partners, preparations=preparations, budget=budget
        )
        surface_results[pair] = result
        _register_branches(state, result, pair)
    return _Discovery(
        MappingProxyType(surface_results),
        MappingProxyType(coincident),
        MappingProxyType({owner: tuple(values) for owner, values in overlays.items()}),
    )


def _joint_candidate(
    state: Arrangement,
    owners: JointOwners,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
) -> TripleSurfaceIntersectionRoot | None:
    state.consume(1)
    regions = (
        SurfaceRegion(
            state.models[owners[0][0]].patches[owners[0][1]],
            np.stack((lower[:2], upper[:2])),
            budget=budget,
            preparations=preparations,
        ),
        SurfaceRegion(
            state.models[owners[1][0]].patches[owners[1][1]],
            np.stack((lower[2:4], upper[2:4])),
            budget=budget,
            preparations=preparations,
        ),
        SurfaceRegion(
            state.models[owners[2][0]].patches[owners[2][1]],
            np.stack((lower[4:], upper[4:])),
            budget=budget,
            preparations=preparations,
        ),
    )
    pieces = tuple(surface_pieces(region, budget=budget) for region in regions)
    if any(len(values) != 1 for values in pieces):
        return None
    prepared = prepare(
        TripleSurfaceSystem(
            pieces[0][0].evaluator, pieces[1][0].evaluator, pieces[2][0].evaluator
        ),
        6,
        1,
        preparations=preparations,
    )
    inclusion = krawczyk(
        prepared, lower[None], upper[None], budget=budget, preparations=preparations
    )
    if not bool(inclusion.certified[0]):
        return None
    return TripleSurfaceIntersectionRoot(
        *regions,
        parameter_lower=lower,
        parameter_upper=upper,
        budget=budget,
        preparations=preparations,
    )


def _joint_vertex(
    closure: _RootClosure,
    root: TrimIntersectionRoot,
    first: Arc,
    second: Arc,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> str:
    state = closure.state
    first_pair = state.branch_owners[first.token]
    second_pair = state.branch_owners[second.token]
    all_owners = tuple(sorted(set((*first_pair, *second_pair))))
    if len(all_owners) != 3:
        raise BRepBooleanFailure(
            "BRepBoolean.branch_junction.owners must contain three source faces; got another owner count",
            (root.root_id,),
        )
    owners = all_owners[0], all_owners[1], all_owners[2]
    regions = (
        _source_region(state, owners[0], preparations=preparations, budget=budget),
        _source_region(state, owners[1], preparations=preparations, budget=budget),
        _source_region(state, owners[2], preparations=preparations, budget=budget),
    )
    image = joint_root_parameter_image(
        root, regions, preparations=preparations, budget=budget
    )
    if image is None:
        raise BRepBooleanFailure(
            "BRepBoolean.branch_junction must have all joint parameter images; got a missing image",
            (root.root_id, str(owners)),
        )
    support = BRepRootSupport(
        state.models[first.owner[0]].patches[first.owner[1]],
        root,
        budget=budget,
        preparations=preparations,
    )
    existing = closure.joints.setdefault(owners, [])
    for record in existing:
        state.consume(1)
        candidate = record.root
        if not candidate.certifies_root(root, preparations=preparations, budget=budget):
            lower = np.minimum(image[0], candidate.parameter_lower)
            upper = np.maximum(image[1], candidate.parameter_upper)
            candidate_or_none = _joint_candidate(
                state, owners, lower, upper, budget=budget, preparations=preparations
            )
            if (
                candidate_or_none is None
                or not candidate_or_none.certifies_root(
                    root, preparations=preparations, budget=budget
                )
                or not candidate_or_none.certifies_root(
                    record.root, preparations=preparations, budget=budget
                )
            ):
                continue
            candidate = candidate_or_none
        record.root = candidate
        if all(item.root_id != support.root_id for item in record.supports):
            record.supports.append(support)
        closure.bind(owners, record, preparations=preparations, budget=budget)
        return record.vertex
    margin = (
        256.0
        * np.finfo(np.float64).eps
        * (1.0 + np.maximum(np.abs(image[0]), np.abs(image[1])))
    )
    lower, upper = (
        np.nextafter(image[0] - margin, -np.inf),
        np.nextafter(image[1] + margin, np.inf),
    )
    joint = _joint_candidate(
        state, owners, lower, upper, budget=budget, preparations=preparations
    )
    if joint is None or not joint.certifies_root(
        root, preparations=preparations, budget=budget
    ):
        raise BRepBooleanFailure(
            "BRepBoolean.triple_junction must have a native unique-root certificate; got failed certification",
            (
                root.root_id,
                str(owners),
                f"lower={tuple(lower)}:upper={tuple(upper)}",
            ),
        )
    vertex = f"joint-root:{joint.root_id}"
    record = _Joint(joint, vertex, [support])
    existing.append(record)
    closure.bind(owners, record, preparations=preparations, budget=budget)
    return vertex


def _full_trim_events(
    state: Arrangement,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> None:
    closure = _RootClosure(state)
    for owner, arcs in sorted(state.arcs.items()):
        for first, second in combinations(arcs, 2):
            if first.source_edge is not None and second.source_edge is not None:
                continue
            if (
                first.branch is not None
                and second.branch is not None
                and first.branch.branch_id == second.branch.branch_id
            ):
                continue
            if tuple(sorted((first.token, second.token))) in state.spatial_pairs:
                continue
            result = intersect_arc_pair(
                first,
                second,
                state,
                preparations=preparations,
                budget=budget,
            ).result
            if result is None:
                continue
            state.consume(result.work.boxes_processed)
            if not result.complete or result.coincident:
                raise BRepBooleanFailure(
                    "BRepBoolean.trim_intersection must resolve isolated roots or a scalar quotient; got unresolved or coincident curves",
                    (first.token, second.token),
                )
            for point in result.points:
                if point.kind != "crossing":
                    raise BRepBooleanFailure(
                        "BRepBoolean.trim_junction must have a transversal root; got a nontransversal contact",
                        (first.token, second.token, point.kind),
                    )
                root = TrimIntersectionRoot(
                    first.trim,
                    second.trim,
                    parameter_lower=point.parameter_lower,
                    parameter_upper=point.parameter_upper,
                )
                state.roots[root.root_id] = root
                if len(state.roots) > state.policy.maximum_cells:
                    raise BRepBooleanFailure(
                        "BRepBoolean.trim_roots must fit maximum_cells; got root capacity exhausted",
                        (root.root_id,),
                    )
                if first.branch is not None and second.branch is not None:
                    vertex = _joint_vertex(
                        closure,
                        root,
                        first,
                        second,
                        preparations=preparations,
                        budget=budget,
                    )
                else:
                    vertex = event_vertex(
                        state,
                        root,
                        first,
                        second,
                        preparations=preparations,
                        budget=budget,
                    )
                add_root_cut(state, first, root, 0, vertex)
                add_root_cut(state, second, root, 1, vertex)


def _complete_cells(
    state: Arrangement,
    discovery: _Discovery,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> tuple[FaceArrangementCells, ...]:
    coverages = []
    for owner in sorted(state.arcs):
        workset = prepare_face_cell_workset(
            state,
            owner,
            overlays=discovery.owner_overlays.get(owner, ()),
            preparations=preparations,
            budget=budget,
        )
        coverage = extract_face_arrangement_cells(
            state, owner, workset=workset, preparations=preparations, budget=budget
        )
        if not coverage.complete:
            raise BRepBooleanFailure(
                "BRepBoolean.source_face.cell_graph must be resolved; got unresolved cells or roots",
                (
                    str(owner),
                    *coverage.unresolved_roots,
                    *(item.reason for item in coverage.unresolved_cells),
                ),
            )
        coverages.append(coverage)
    return tuple(coverages)


def _face_parents(
    classification: FaceMaterialClassification, key: str
) -> tuple[FaceOwner, ...]:
    label = next(item for item in classification.labels if item.classification_key == key)
    parents = {label.owner}
    for individual in label.solids:
        for face in individual.support_faces:
            parents.add((individual.solid.model_index, face.index))
    return tuple(sorted(parents))


def _select_cells(
    state: Arrangement,
    discovery: _Discovery,
    classification: FaceMaterialClassification,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None,
) -> _SelectedCells:
    labels = {item.classification_key: item for item in classification.labels}
    emitted: list[FaceArrangementCell] = []
    representatives: dict[str, str] = {}
    work = Work(state)
    for cell in classification.cells:
        label = labels[cell.classification_key]
        if label.orientation == 0:
            continue
        representative = None
        for candidate in emitted:
            pair = tuple(sorted((cell.owner, candidate.owner)))
            if (
                cell.owner != candidate.owner
                and pair not in discovery.coincident_overlays
            ):
                continue
            if equivalent_cell(
                state, cell, candidate, work, preparations=preparations, budget=budget
            ):
                representative = candidate
                break
        if representative is None:
            if len(emitted) >= state.policy.maximum_faces:
                raise BRepBooleanFailure(
                    "BRepBoolean.selected_faces must fit maximum_faces; got selected face capacity exhausted",
                    (cell.classification_key,),
                )
            emitted.append(cell)
        else:
            representatives[cell.classification_key] = representative.classification_key
    fragments = tuple(
        FaceFragment(
            cell.owner,
            cell.loops,
            cell.domain,
            labels[cell.classification_key].orientation,
        )
        for cell in emitted
    )
    parents = tuple(
        _face_parents(classification, cell.classification_key) for cell in emitted
    )
    return _SelectedCells(
        tuple(emitted), fragments, MappingProxyType(representatives), parents
    )


def _binding_add(
    bindings: dict[str, BRepVertexRoot | None],
    key: str,
    binding: BRepVertexRoot | None,
) -> None:
    if key in bindings and bindings[key] is not None and binding is not None:
        existing = bindings[key]
        if existing is not None and existing.root_id != binding.root_id:
            raise BRepBooleanFailure(
                "BRepBoolean.source_vertex.roots must agree on vertex identity; got contradictory root definitions",
                (key,),
            )
    if key not in bindings or bindings[key] is None:
        bindings[key] = binding


def _boundary_facts(
    state: Arrangement,
    coverages: tuple[FaceArrangementCells, ...],
    *,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
) -> _BoundaryFacts:
    sources: dict[str, CoincidentFaceArc] = {}
    aliases: dict[str, str] = {}
    bindings: dict[str, BRepVertexRoot | None] = {}
    for owner in sorted(state.arcs):
        model = state.models[owner[0]]
        geometry = model.geometry
        if geometry is None:
            raise BRepBooleanFailure(
                "BRepBoolean.source_boundary.geometry must define source coedges; got None",
                (model.source_revision,),
            )
        identity = prove_surface_correspondence(
            model.patches[owner[1]], model.patches[owner[1]]
        )
        if identity is None:
            raise BRepBooleanFailure(
                "BRepBoolean.source_chart must have an identity chart correspondence; got no correspondence",
                (str(owner),),
            )
        for loop in geometry.face_loops[owner[1]]:
            for coedge in loop:
                source = CoincidentFaceArc(model, owner[1], coedge, identity)
                sources[f"source:{owner[0]}:coedge:{coedge}"] = source
                for entity, binding in zip(
                    source.vertex_ids, source.vertex_roots, strict=True
                ):
                    key = brep_entity_id(
                        entity.source_revision,
                        0,
                        entity.index,
                        occurrence_path=entity.occurrence_path,
                    )
                    aliases[f"source:{owner[0]}:vertex:{entity.index}"] = key
                    aliases[str(entity)] = key
                    _binding_add(bindings, key, binding)
    for coverage in coverages:
        sources.update(coverage.overlay_sources)
        for source in coverage.overlay_sources.values():
            for entity, binding in zip(
                source.vertex_ids, source.vertex_roots, strict=True
            ):
                key = brep_entity_id(
                    entity.source_revision,
                    0,
                    entity.index,
                    occurrence_path=entity.occurrence_path,
                )
                aliases[str(entity)] = key
                _binding_add(bindings, key, binding)
        patch = state.models[coverage.owner[0]].patches[coverage.owner[1]]
        for root in coverage.intersection_roots.values():
            vertex = f"overlay-root:{root.root_id}:face:{coverage.owner}"
            support = BRepRootSupport(
                patch, root, preparations=preparations, budget=budget
            )
            binding = BRepVertexRoot(support, preparations=preparations, budget=budget)
            _binding_add(bindings, vertex, binding)
    for vertex, binding in state.vertex_bindings.items():
        _binding_add(bindings, aliases.get(vertex, vertex), binding)
    return _BoundaryFacts(
        MappingProxyType(sources), MappingProxyType(aliases), MappingProxyType(bindings)
    )


def _canonical_fragments(
    selected: _SelectedCells, facts: _BoundaryFacts
) -> tuple[FaceFragment, ...]:
    fragments = []
    for fragment in selected.fragments:
        loops = tuple(
            tuple(
                replace(
                    piece,
                    first=replace(
                        piece.first,
                        vertex=facts.vertex_aliases.get(
                            piece.first.vertex, piece.first.vertex
                        ),
                    ),
                    last=replace(
                        piece.last,
                        vertex=facts.vertex_aliases.get(
                            piece.last.vertex, piece.last.vertex
                        ),
                    ),
                )
                for piece in loop
            )
            for loop in fragment.loops
        )
        fragments.append(replace(fragment, loops=loops))
    return tuple(fragments)


def _association_graph(
    state: Arrangement,
    model: BRepModel,
    selected: _SelectedCells,
    schedule: MaterialOverlapSchedule,
    certificate: str,
) -> AssociationGraph:

    source = source_revision(state.models, certificate)
    target = cad_revision_from_brep_model(model)
    pairs = set(schedule.canonical_pairs)
    for target_solid, faces in enumerate(model.topology.solid_faces):
        for face in faces:
            for operand, parent_face in selected.face_parents[face]:
                parent = state.models[operand]
                for parent_solid in parent.topology.face_solids[parent_face]:
                    pairs.add(
                        (
                            f"operand:{operand}/solid:{parent_solid}/face:{parent_face}",
                            f"solid:{target_solid}/face:{face}",
                        )
                    )
    transaction = OccurrenceCorrespondenceTransaction(
        certificate,
        source.revision_id,
        target.revision_id,
        tuple(
            OccurrenceCorrespondence(first, second, certificate)
            for first, second in sorted(pairs)
        ),
        frozenset(),
        frozenset(),
        AssociationCoverageEvidence(
            True,
            True,
            schedule.certificate_id,
            "native-complete-face-cell-material-overlay",
        ),
    )
    return AssociationGraph(source, target, transaction)


@dataclass(frozen=True, slots=True)
class _MaterialOverlay:
    state: Arrangement
    classification: FaceMaterialClassification
    selected: _SelectedCells
    built: BuiltBoundary


def _material_overlay(
    models: tuple[BRepModel, ...],
    selection: MaterialSelection,
    policy: BRepBooleanPolicy,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> _MaterialOverlay:
    """exhaustive source discovery, cells, side owners, and emitted boundary.

    Call inside the invocation's branch preparation scope, which keeps every
    branch's programs and identical-query results through the caller's
    publication and target proof.
    """
    state = Arrangement(
        models,
        selection,
        policy,
        tuple(
            prepare_brep_query(model, budget=budget, preparations=preparations)
            for model in models
        ),
    )
    source_arcs(state, preparations=preparations, budget=budget)
    discovery = _discover_all_source_pairs(
        state, preparations=preparations, budget=budget
    )
    boundary_spatial_events(state, preparations=preparations, budget=budget)
    _full_trim_events(state, preparations=preparations, budget=budget)
    propagate_branch_cuts(state)
    coverages = _complete_cells(
        state, discovery, preparations=preparations, budget=budget
    )
    classification = classify_face_material_cells(
        state,
        coverages,
        surface_results=discovery.surface_results,
        coincident_overlays=discovery.coincident_overlays,
        budget=state,
        preparations=preparations,
        coordinate_budget=budget,
    )
    selected = _select_cells(
        state, discovery, classification, preparations=preparations, budget=budget
    )
    facts = _boundary_facts(state, coverages, preparations=preparations, budget=budget)
    built = build_boundary(
        state,
        _canonical_fragments(selected, facts),
        overlay_sources=facts.sources,
        vertex_bindings=facts.bindings,
        preparations=preparations,
        budget=budget,
    )
    return _MaterialOverlay(state, classification, selected, built)


def _target_schedule(
    overlay: _MaterialOverlay,
    model: BRepModel,
    *,
    solid_regions: tuple[str, ...] | None = None,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> MaterialOverlapSchedule:
    return prove_material_overlaps(
        overlay.state,
        overlay.classification,
        model,
        (
            prepare_brep_query(model, budget=budget, preparations=preparations)
            if model.topology.num_solids
            else None
        ),
        tuple(cell.classification_key for cell in overlay.selected.cells),
        budget=overlay.state,
        target_cell_representatives=overlay.selected.representatives,
        target_solid_regions=solid_regions,
        preparations=preparations,
        coordinate_budget=budget,
    )


@checked
def full_overlay_boolean_brep(
    models: tuple[BRepModel, BRepModel],
    operation: BRepBooleanOperation,
    policy: BRepBooleanPolicy,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> BRepBooleanResult:
    """Compose exhaustive source discovery, cells, material, and lineage."""
    if preparations is None:
        preparations = TrimRequestPreparations()
    if budget is None:
        with coordinate_source_request(
            (
                models,
                operation,
                policy,
            )
        ) as budget:
            return full_overlay_boolean_brep(
                models, operation, policy, preparations=preparations, budget=budget
            )
    with original_trim_intersection_preparation(
        (), preparations=preparations
    ) as preparations:
        return _full_overlay_boolean(
            models, operation, policy, preparations=preparations, budget=budget
        )


def _full_overlay_boolean(
    models: tuple[BRepModel, BRepModel],
    operation: BRepBooleanOperation,
    policy: BRepBooleanPolicy,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> BRepBooleanResult:
    overlay = _material_overlay(
        models, operation, policy, preparations=preparations, budget=budget
    )
    state, selected = overlay.state, overlay.selected
    geometry = reconstruct_solids(
        state, overlay.built, preparations=preparations, budget=budget
    )
    geometry_certificate = canonical_fingerprint(
        {
            "kind": "native-complete-face-overlay-geometry",
            "sources": tuple(model.model_id for model in models),
            "operation": operation,
            "material": overlay.classification.certificate_id,
            "geometry": geometry.geometry_id,
            "selected_cells": tuple(cell.classification_key for cell in selected.cells),
            "representatives": tuple(sorted(selected.representatives.items())),
        }
    )
    model = publish_curved(
        state,
        overlay.built,
        geometry,
        geometry_certificate,
        preparations=preparations,
        budget=budget,
    )
    schedule = _target_schedule(overlay, model, preparations=preparations, budget=budget)
    certificate = canonical_fingerprint(
        {
            "kind": "native-complete-face-cell-boolean",
            "geometry": geometry_certificate,
            "material_overlap": schedule.certificate_id,
            "face_parents": selected.face_parents,
            "shared_work_units": state.boxes_used,
            "shared_work_definition": "native discovery/containment, conservative projection visits, exact cell/quotient/identity operations",
        }
    )
    graph = _association_graph(state, model, selected, schedule, certificate)
    mapped = {item.source_occurrence_id for item in graph.transaction.correspondences}
    deleted = tuple(
        item.occurrence_id
        for item in graph.source_revision.occurrences
        if item.occurrence_id not in mapped
    )
    return BRepBooleanResult(model, graph, operation, certificate, deleted)


@dataclass(frozen=True, slots=True)
class FullOverlayPartition:
    """Retain shared-face cells, region owners, and native-index lineage.

    ``solid_pairs`` and ``face_pairs`` join source occurrence IDs of selected
    solids to target solid and ``(solid, face)`` indices of ``model``.
    """

    model: BRepModel
    solid_regions: tuple[str, ...]
    solid_pairs: tuple[tuple[str, int], ...]
    face_pairs: tuple[tuple[str, int, int], ...]
    certificate_id: str


def _verify_region_owners(
    selection: RegionSelection,
    schedule: MaterialOverlapSchedule,
    solid_regions: tuple[str, ...],
) -> None:
    """Verify every target solid descends from its region and excludes targeted voids.

    Independent of the shared-face construction, open-volume overlap proves
    each published solid's owner against the declared selection.
    """
    selected = set(selection.solids)
    void_targets = dict(selection.voids)
    owned: set[int] = set()
    for item in schedule.overlaps:
        operand = item.source.model_index
        if not item.overlaps or (operand, item.source.entity.index) not in selected:
            continue
        operand_id = selection.operand_ids[operand]
        region = solid_regions[item.target.index]
        if region in void_targets.get(operand_id, ()):
            raise BRepBooleanFailure(
                "BRepPartition.region must exclude targeted void interiors; got an overlap with a removing void",
                (
                    item.source.source_id,
                    f"solid:{item.target.index}",
                    *item.witness_cells,
                ),
            )
        if operand_id == region:
            owned.add(item.target.index)
    if owned != set(range(len(solid_regions))):
        raise BRepBooleanFailure(
            "BRepPartition.solid must have material ancestry in its region; got no regional ancestor",
            tuple(
                f"solid:{index}"
                for index in range(len(solid_regions))
                if index not in owned
            ),
        )


@checked
def full_overlay_partition_brep(
    models: tuple[BRepModel, ...],
    selection: RegionSelection,
    policy: BRepBooleanPolicy,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> FullOverlayPartition:
    """Partition curved source solids on one exhaustive face-cell overlay.

    Cells whose two material owners differ become faces shared by both owners'
    solids. Region solids, cavities, ancestry and face lineage reuse the
    Boolean overlay certificates; no source is tessellated or refitted.
    """
    if preparations is None:
        preparations = TrimRequestPreparations()
    if budget is None:
        with coordinate_source_request(
            (
                models,
                selection,
                policy,
            )
        ) as budget:
            return full_overlay_partition_brep(
                models, selection, policy, preparations=preparations, budget=budget
            )
    with original_trim_intersection_preparation(
        (), preparations=preparations
    ) as preparations:
        return _full_overlay_partition(
            models, selection, policy, preparations=preparations, budget=budget
        )


def _full_overlay_partition(
    models: tuple[BRepModel, ...],
    selection: RegionSelection,
    policy: BRepBooleanPolicy,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> FullOverlayPartition:
    overlay = _material_overlay(
        models, selection, policy, preparations=preparations, budget=budget
    )
    state, selected = overlay.state, overlay.selected
    labels = {item.classification_key: item for item in overlay.classification.labels}
    sides = tuple(
        (
            labels[cell.classification_key].negative_owner,
            labels[cell.classification_key].positive_owner,
        )
        for cell in selected.cells
    )
    geometry, solid_regions = reconstruct_region_solids(
        state,
        overlay.built,
        sides,
        selection.precedence,
        preparations=preparations,
        budget=budget,
    )
    if not solid_regions:
        raise BRepBooleanFailure(
            "BRepPartition.regions must contain surviving material; got no surviving material region"
        )
    geometry_certificate = canonical_fingerprint(
        {
            "kind": "native-complete-face-overlay-partition-geometry",
            "sources": tuple(model.model_id for model in models),
            "selection": selection_payload(selection),
            "material": overlay.classification.certificate_id,
            "geometry": geometry.geometry_id,
            "selected_cells": tuple(cell.classification_key for cell in selected.cells),
            "representatives": tuple(sorted(selected.representatives.items())),
            "solid_regions": solid_regions,
        }
    )
    model = publish_curved(
        state,
        overlay.built,
        geometry,
        geometry_certificate,
        preparations=preparations,
        budget=budget,
    )
    schedule = _target_schedule(
        overlay,
        model,
        solid_regions=solid_regions,
        preparations=preparations,
        budget=budget,
    )
    _verify_region_owners(selection, schedule, solid_regions)
    certificate = canonical_fingerprint(
        {
            "kind": "native-complete-face-cell-partition",
            "geometry": geometry_certificate,
            "material_overlap": schedule.certificate_id,
            "face_parents": selected.face_parents,
            "shared_work_units": state.boxes_used,
        }
    )
    participating = set(selection.solids)
    solid_pairs = tuple(
        sorted(
            (item.source.source_id, item.target.index)
            for item in schedule.overlaps
            if item.overlaps
            and (item.source.model_index, item.source.entity.index) in participating
        )
    )
    face_pairs: set[tuple[str, int, int]] = set()
    for target_solid, faces in enumerate(model.topology.solid_faces):
        for face in faces:
            for operand, parent_face in selected.face_parents[face]:
                for parent_solid in state.models[operand].topology.face_solids[
                    parent_face
                ]:
                    if (operand, parent_solid) in participating:
                        face_pairs.add(
                            (
                                f"operand:{operand}/solid:{parent_solid}/face:{parent_face}",
                                target_solid,
                                face,
                            )
                        )
    return FullOverlayPartition(
        model, solid_regions, solid_pairs, tuple(sorted(face_pairs)), certificate
    )
