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
    joint_root_parameter_image,
    overlap,
    periodic_transition,
    propagate_branch_cuts,
    publish_curved,
    reconstruct_region_solids,
    reconstruct_solids,
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
    entity_occurrence_index,
    MaterialSelection,
    RegionSelection,
    selection_payload,
    source_assembly_occurrence_id,
    source_lineage,
    target_assembly_occurrence_id,
)
from ._boolean_material import (
    classify_face_material_cells,
    equivalent_cell,
    FaceMaterialClassification,
    FacePair,
    MaterialOverlapSchedule,
    prove_material_overlaps,
)
from ._carrier_pieces import carrier_payload
from ._correspondence import exact_source_lift, native_edge_pcurve
from ._intersection import (
    intersect_surface_regions,
    SurfaceIntersectionResult,
    TrimIntersectionRoot,
    TripleSurfaceIntersectionRoot,
)
from ._intersection_curve import (
    IntersectionCurve,
    orthogonal_cylinder_branches_disjoint,
    SurfaceRegion,
)
from ._model import BRepCurve, BRepEntityId, BRepModel
from ._partition import cad_revision_from_brep_model
from ._patches import AbstractCurve, AbstractSurfacePatch
from ._projection import prepare_brep_projection
from ._projection_contracts import brep_entity_id
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
    ) -> None:
        """Merge existing source vertices only through the joint source system."""
        state = self.state
        joint = record.root
        matches = []
        lifts_by_id: dict[str, BRepCurveSurfaceLift] = {}
        for vertex, spatial in tuple(state.vertex_spatial.items()):
            lifts = {lift.root_id: lift for lift in state.vertex_lifts.get(vertex, ())}
            for operand, face in owners:
                model = state.models[operand]
                geometry = model.geometry
                if geometry is None:
                    raise BRepBooleanFailure(
                        "BRepBoolean.joint_lift.geometry must be defined; got None"
                    )
                region = _source_region(state, (operand, face))
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
                            and type(curve) is type(spatial.curve)
                            and carrier_payload(curve) == carrier_payload(spatial.curve)
                            and exact_source_lift(curve, pcurve, region, first, last)
                        ):
                            lift = BRepCurveSurfaceLift(
                                curve,
                                region.patch,
                                pcurve,
                                first,
                                last,
                                source_region=region,
                            )
                            lifts[lift.root_id] = lift
            if joint.certifies_root(
                spatial,
                source_edge_lifts=tuple(lifts.values()),
            ):
                matches.append((vertex, spatial))
                lifts_by_id.update(lifts)
        old_vertices = {vertex for vertex, _ in matches}
        supports = {support.root_id: support for support in record.supports}
        for vertex in old_vertices:
            for patch, root in state.vertex_aliases.get(vertex, ()):
                supports.setdefault(
                    root.root_id,
                    BRepRootSupport(patch, root),
                )
        record.supports = list(supports.values())
        spatial = matches[0][1] if matches else None
        # A merged vertex keeps any branch-node witness: the joint root carries
        # it through the witnessed spatial root it certifies, never dropped.
        witnesses = {
            binding.branch_node.root_id: binding.branch_node
            for vertex in (*old_vertices, record.vertex)
            if (binding := state.vertex_bindings.get(vertex)) is not None
            and binding.branch_node is not None
        }
        if len(witnesses) > 1:
            raise BRepBooleanFailure(
                "BRepBoolean.joint_vertex must carry at most one branch-node witness; got several",
                (record.vertex,),
            )
        branch_node = next(iter(witnesses.values()), None)
        if branch_node is not None:
            spatial = next((root for _, root in matches if branch_node.binds(root)), None)
            if spatial is None:
                raise BRepBooleanFailure(
                    "BRepBoolean.joint_vertex must certify a witnessed branch-node root; got none",
                    (record.vertex,),
                )
        binding = BRepVertexRoot(
            joint,
            aliases=tuple(record.supports),
            spatial_root=spatial,
            source_edge_lifts=tuple(lifts_by_id.values()),
            branch_node=branch_node,
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
    source_vertices: Mapping[str, tuple[tuple[int, BRepEntityId], ...]]
    bindings: Mapping[str, BRepVertexRoot | None]


def _source_region(
    state: Arrangement,
    owner: FaceOwner,
) -> SurfaceRegion:
    return state.face_region(owner)


def _register_branches(
    state: Arrangement,
    result: SurfaceIntersectionResult,
    pair: FacePair,
) -> None:
    for curve in result.curves:
        state.branches[curve.branch_id] = curve, pair[0], pair[1]
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
    region = SurfaceRegion(patch, bounds)
    pcurve = native_edge_pcurve(curve, region.patch)
    if pcurve is None or not exact_source_lift(curve, pcurve, region, first, last):
        return None
    return pcurve


def _face_edge_pcurve(
    state: Arrangement,
    face: FaceOwner,
    model_index: int,
    edge: int,
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
        state.face_chart(face),
        first,
        last,
    )


def _edge_lifts(
    state: Arrangement,
    owner: FaceOwner,
    other: FaceOwner,
    partners: tuple[FaceOwner, ...],
    /,
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
                )
                there = (
                    None
                    if here is None
                    else _face_edge_pcurve(
                        state,
                        other,
                        face[0],
                        edge,
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
    relative_resolution: float,
    /,
) -> bool:
    """Find a source-edge subarc inside a chart box or refuse at resolution."""
    pending = [(first, last)]
    scale = max(1.0, abs(last - first))
    resolution = max(
        relative_resolution * scale,
        64.0 * np.finfo(np.float64).eps * max(1.0, abs(first), abs(last)),
    )
    while pending:
        lower, upper = pending.pop()
        image = np.concatenate(
            (
                np.asarray(first_pcurve.bounding_box(lower, upper)),
                np.asarray(second_pcurve.bounding_box(lower, upper)),
            ),
            axis=1,
        )
        meets = False
        for box in boxes:
            if np.all(image[0] >= box[0]) and np.all(image[1] <= box[1]):
                return True
            meets |= bool(np.all(image[0] <= box[1]) and np.all(box[0] <= image[1]))
        if not meets:
            continue
        middle = lower + 0.5 * (upper - lower)
        if upper - lower <= resolution or not lower < middle < upper:
            raise BRepBooleanFailure(
                "BRepBoolean.source_edge_branch remains unresolved at parameter resolution"
            )
        pending.extend(((lower, middle), (middle, upper)))
    return False


def _chart_boxes(curve: IntersectionCurve, chart: int, /) -> tuple[np.ndarray, ...]:
    """The chart's coupled box and its inward-rounded integer period images."""
    box = np.stack((curve.chart_lower[chart], curve.chart_upper[chart]))
    periods = [
        0.0 if period is None else float(period)
        for region in (curve.first, curve.second)
        for period in region.patch.periods
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
    relative_resolution: float,
    /,
) -> bool:
    """Certify one source edge as a whole branch using chart uniqueness.

    A branch has a source-edge preimage only when every certified chart's
    unique curve piece contains an edge subarc. No coordinate proximity is used.
    """
    return (
        bool(curve.num_charts)
        and curve.fully_certified
        and all(
            _subarc_in_box(
                lift.first_pcurve,
                lift.second_pcurve,
                lift.first,
                lift.last,
                _chart_boxes(curve, chart),
                relative_resolution,
            )
            for chart in range(curve.num_charts)
        )
    )


def _without_source_edge_branches(
    state: Arrangement,
    pair: FacePair,
    result: SurfaceIntersectionResult,
    partners: Mapping[FaceOwner, tuple[FaceOwner, ...]],
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
        )
    )
    kept = []
    for curve in result.curves:
        on_first = next(
            (
                lift.edge
                for lift in first_edges
                if _branch_is_edge(
                    curve, lift, state.policy.intersection.relative_resolution
                )
            ),
            None,
        )
        on_second = next(
            (
                lift.edge
                for lift in second_edges
                if _branch_is_edge(
                    curve, lift, state.policy.intersection.relative_resolution
                )
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
        tuple(kept), result.points, result.coincident, result.unresolved
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
) -> bool:
    """Prove face-pair coverage using a represented source edge."""
    first, second = pair
    first_lifts = _edge_lifts(
        state,
        first,
        second,
        partners.get(first, ()),
    )
    second_lifts = _edge_lifts(
        state,
        second,
        first,
        partners.get(second, ()),
    )
    for first_lift in first_lifts:
        for second_lift in second_lifts:
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
) -> _Discovery:
    surface_results: dict[FacePair, SurfaceIntersectionResult] = {}
    coincident: dict[FacePair, CoincidentFaceOverlay] = {}
    overlays: dict[FaceOwner, list[CoincidentFaceOverlay]] = {}
    regions = {owner: _source_region(state, owner) for owner in sorted(state.arcs)}
    boxes = {
        owner: region.patch.bounding_box(region.parameter_box)
        for owner, region in regions.items()
    }
    candidates = []
    for pair in combinations(tuple(regions), 2):
        if _represented_source_adjacency(state, pair):
            continue
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
        if _cover_source_edge_pair(state, pair, partners):
            continue
        result = intersect_surface_regions(
            regions[a],
            regions[b],
            policy=state.policy.intersection,
        )
        check_surface_result(state, result, a, b)
        result = _without_source_edge_branches(state, pair, result, partners)
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
) -> TripleSurfaceIntersectionRoot | None:
    regions = tuple(_source_region(state, owner) for owner in owners)
    try:
        return TripleSurfaceIntersectionRoot(
            *regions,
            parameter_lower=lower,
            parameter_upper=upper,
        )
    except (ValueError, np.linalg.LinAlgError):
        return None


def _joint_vertex(
    closure: _RootClosure,
    root: TrimIntersectionRoot,
    first: Arc,
    second: Arc,
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
        _source_region(state, owners[0]),
        _source_region(state, owners[1]),
        _source_region(state, owners[2]),
    )
    image = joint_root_parameter_image(root, regions)
    if image is None:
        raise BRepBooleanFailure(
            "BRepBoolean.branch_junction must have all joint parameter images; got a missing image",
            (root.root_id, str(owners)),
        )
    support = BRepRootSupport(
        state.models[first.owner[0]].patches[first.owner[1]],
        root,
    )
    existing = closure.joints.setdefault(owners, [])
    for record in existing:
        candidate = record.root
        if not candidate.certifies_root(root):
            lower = np.minimum(image[0], candidate.parameter_lower)
            upper = np.maximum(image[1], candidate.parameter_upper)
            candidate_or_none = _joint_candidate(state, owners, lower, upper)
            if (
                candidate_or_none is None
                or not candidate_or_none.certifies_root(root)
                or not candidate_or_none.certifies_root(record.root)
            ):
                continue
            candidate = candidate_or_none
        record.root = candidate
        if all(item.root_id != support.root_id for item in record.supports):
            record.supports.append(support)
        closure.bind(owners, record)
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
    joint = _joint_candidate(state, owners, lower, upper)
    if joint is None or not joint.certifies_root(root):
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
    closure.bind(owners, record)
    return vertex


def _full_trim_events(
    state: Arrangement,
) -> None:
    closure = _RootClosure(state)
    for owner, arcs in sorted(state.arcs.items()):
        for first, second in combinations(arcs, 2):
            if first.source_edge is not None and second.source_edge is not None:
                continue
            if first.branch is not None and second.branch is not None:
                if first.branch.branch_id == second.branch.branch_id:
                    continue
                if orthogonal_cylinder_branches_disjoint(
                    first.branch,
                    second.branch,
                ):
                    continue
            if tuple(sorted((first.token, second.token))) in state.spatial_pairs:
                continue
            result = intersect_arc_pair(
                first,
                second,
                state,
            ).result
            if result is None:
                continue
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
                if first.branch is not None and second.branch is not None:
                    vertex = _joint_vertex(
                        closure,
                        root,
                        first,
                        second,
                    )
                else:
                    vertex = event_vertex(
                        state,
                        root,
                        first,
                        second,
                    )
                add_root_cut(state, first, root, 0, vertex)
                add_root_cut(state, second, root, 1, vertex)


def _complete_cells(
    state: Arrangement,
    discovery: _Discovery,
) -> tuple[FaceArrangementCells, ...]:
    coverages = []
    for owner in sorted(state.arcs):
        workset = prepare_face_cell_workset(
            state,
            owner,
            overlays=discovery.owner_overlays.get(owner, ()),
        )
        coverage = extract_face_arrangement_cells(state, owner, workset=workset)
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
) -> _SelectedCells:
    labels = {item.classification_key: item for item in classification.labels}
    emitted: list[FaceArrangementCell] = []
    representatives: dict[str, str] = {}
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
            if equivalent_cell(state, cell, candidate):
                representative = candidate
                break
        if representative is None:
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
) -> _BoundaryFacts:
    sources: dict[str, CoincidentFaceArc] = {}
    aliases: dict[str, str] = {}
    source_vertices: dict[str, set[tuple[int, BRepEntityId]]] = {}
    bindings: dict[str, BRepVertexRoot | None] = {}

    def add_source_vertex(
        operand: int, entity: BRepEntityId, binding: BRepVertexRoot | None
    ) -> None:
        key = brep_entity_id(
            entity.source_id,
            entity.source_revision,
            0,
            entity.index,
            occurrence_path=entity.occurrence_path,
        )
        token = f"source:{operand}:vertex:{entity.index}"
        aliases[token] = key
        aliases[str(entity)] = key
        source_ref = (operand, entity)
        for alias in (token, key, str(entity)):
            source_vertices.setdefault(alias, set()).add(source_ref)
        _binding_add(bindings, key, binding)

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
                    add_source_vertex(owner[0], entity, binding)
    for coverage in coverages:
        sources.update(coverage.overlay_sources)
        for source in coverage.overlay_sources.values():
            source_operands = tuple(
                operand
                for operand, model in enumerate(state.models)
                if model.source_id == source.face_id.source_id
                and model.source_revision == source.face_id.source_revision
            )
            if not source_operands:
                raise BRepBooleanFailure(
                    "BRepBoolean.lineage.overlay_vertex must retain an exact source operand",
                    (str(source.face_id),),
                )
            for source_operand in source_operands:
                for entity, binding in zip(
                    source.vertex_ids, source.vertex_roots, strict=True
                ):
                    add_source_vertex(source_operand, entity, binding)
        patch = state.models[coverage.owner[0]].patches[coverage.owner[1]]
        for root in coverage.intersection_roots.values():
            vertex = f"overlay-root:{root.root_id}:face:{coverage.owner}"
            support = BRepRootSupport(patch, root)
            _binding_add(bindings, vertex, BRepVertexRoot(support))
    for vertex, binding in state.vertex_bindings.items():
        _binding_add(bindings, aliases.get(vertex, vertex), binding)
    return _BoundaryFacts(
        MappingProxyType(sources),
        MappingProxyType(aliases),
        MappingProxyType(
            {key: tuple(sorted(entities)) for key, entities in source_vertices.items()}
        ),
        MappingProxyType(bindings),
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
    built: BuiltBoundary,
    schedule: MaterialOverlapSchedule,
    certificate: str,
) -> AssociationGraph:
    source = source_lineage(state.models, certificate)
    target = cad_revision_from_brep_model(model)
    target_index = entity_occurrence_index(model, target)
    pairs: set[tuple[str, str]] = set()
    for material_solid, target_solid in schedule.entity_pairs:
        source_id = source.occurrence_id(
            material_solid.model_index, material_solid.entity
        )
        for target_entity, target_id in target_index.get(
            ("solid", target_solid.index), ()
        ):
            if target_entity != target_solid:
                continue
            pairs.add((source_id, target_id))
            pairs.add(
                (
                    source_assembly_occurrence_id(
                        source,
                        material_solid.model_index,
                        state.models[material_solid.model_index],
                        material_solid.entity.occurrence_path,
                    ),
                    target_assembly_occurrence_id(
                        target, model, target_entity.occurrence_path
                    ),
                )
            )
    for target_solid, faces in enumerate(model.topology.solid_faces):
        for target_solid_entity, _ in target_index.get(("solid", target_solid), ()):
            for face in faces:
                for operand, parent_face in selected.face_parents[face]:
                    parent = state.models[operand]
                    for parent_solid in parent.topology.face_solids[parent_face]:
                        for source_face in source.entities_for_solid(
                            operand, "face", parent_face, parent_solid
                        ):
                            for target_face, target_face_id in target_index.get(
                                ("face", face), ()
                            ):
                                if (
                                    target_face.occurrence_path
                                    == target_solid_entity.occurrence_path
                                ):
                                    pairs.add(
                                        (
                                            source.occurrence_id(operand, source_face),
                                            target_face_id,
                                        )
                                    )
    for target_edge, references in enumerate(built.edge_sources):
        for operand, source_edge in references:
            source_id = source.occurrence_id(operand, source_edge)
            for _, target_id in target_index.get(("edge", target_edge), ()):
                pairs.add((source_id, target_id))
    for target_vertex, references in enumerate(built.vertex_sources):
        for operand, source_vertex in references:
            source_id = source.occurrence_id(operand, source_vertex)
            for _, target_id in target_index.get(("vertex", target_vertex), ()):
                pairs.add((source_id, target_id))
    transaction = OccurrenceCorrespondenceTransaction(
        certificate,
        source.revision.revision_id,
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
    return AssociationGraph(source.revision, target, transaction)


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
        tuple(prepare_brep_projection(model) for model in models),
    )
    source_arcs(state)
    discovery = _discover_all_source_pairs(state)
    boundary_spatial_events(state)
    _full_trim_events(state)
    propagate_branch_cuts(state)
    coverages = _complete_cells(state, discovery)
    classification = classify_face_material_cells(
        state,
        coverages,
        surface_results=discovery.surface_results,
        coincident_overlays=discovery.coincident_overlays,
    )
    selected = _select_cells(state, discovery, classification)
    facts = _boundary_facts(state, coverages)
    built = build_boundary(
        state,
        _canonical_fragments(selected, facts),
        source_vertex_entities=facts.source_vertices,
        overlay_sources=facts.sources,
        vertex_bindings=facts.bindings,
    )
    return _MaterialOverlay(state, classification, selected, built)


def _target_schedule(
    overlay: _MaterialOverlay,
    model: BRepModel,
    *,
    solid_regions: tuple[str, ...] | None = None,
) -> MaterialOverlapSchedule:
    return prove_material_overlaps(
        overlay.state,
        overlay.classification,
        model,
        (prepare_brep_projection(model) if model.topology.num_solids else None),
        tuple(cell.classification_key for cell in overlay.selected.cells),
        target_cell_representatives=overlay.selected.representatives,
        target_solid_regions=solid_regions,
    )


@checked
def full_overlay_boolean_brep(
    models: tuple[BRepModel, BRepModel],
    operation: BRepBooleanOperation,
    policy: BRepBooleanPolicy,
) -> BRepBooleanResult:
    """Compose exhaustive source discovery, cells, material, and lineage."""
    return _full_overlay_boolean(models, operation, policy)


def _full_overlay_boolean(
    models: tuple[BRepModel, BRepModel],
    operation: BRepBooleanOperation,
    policy: BRepBooleanPolicy,
) -> BRepBooleanResult:
    overlay = _material_overlay(models, operation, policy)
    state, selected = overlay.state, overlay.selected
    geometry = reconstruct_solids(state, overlay.built)
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
    )
    schedule = _target_schedule(overlay, model)
    certificate = canonical_fingerprint(
        {
            "kind": "native-complete-face-cell-boolean",
            "geometry": geometry_certificate,
            "material_overlap": schedule.certificate_id,
            "face_parents": selected.face_parents,
        }
    )
    graph = _association_graph(
        state, model, selected, overlay.built, schedule, certificate
    )
    mapped = {item.source_occurrence_id for item in graph.transaction.correspondences}
    deleted = tuple(
        item.occurrence_id
        for item in graph.source_revision.occurrences
        if item.occurrence_id not in mapped
    )
    return BRepBooleanResult(model, graph, operation, certificate, deleted)


@dataclass(frozen=True, slots=True)
class FullOverlayPartition:
    """Retain shared-face cells, region owners, and typed source ancestry.

    Each pair carries the exact source definition entity and occurrence path
    alongside the target entity key.
    """

    model: BRepModel
    solid_regions: tuple[str, ...]
    solid_pairs: tuple[tuple[int, BRepEntityId, BRepEntityId], ...]
    face_pairs: tuple[tuple[int, BRepEntityId, BRepEntityId], ...]
    edge_pairs: tuple[tuple[int, BRepEntityId, BRepEntityId], ...]
    vertex_pairs: tuple[tuple[int, BRepEntityId, BRepEntityId], ...]
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
) -> FullOverlayPartition:
    """Partition curved source solids on one exhaustive face-cell overlay."""
    return _full_overlay_partition(models, selection, policy)


def _full_overlay_partition(
    models: tuple[BRepModel, ...],
    selection: RegionSelection,
    policy: BRepBooleanPolicy,
) -> FullOverlayPartition:
    overlay = _material_overlay(models, selection, policy)
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
    )
    schedule = _target_schedule(
        overlay,
        model,
        solid_regions=solid_regions,
    )
    _verify_region_owners(selection, schedule, solid_regions)
    certificate = canonical_fingerprint(
        {
            "kind": "native-complete-face-cell-partition",
            "geometry": geometry_certificate,
            "material_overlap": schedule.certificate_id,
            "face_parents": selected.face_parents,
        }
    )
    participating = set(selection.solids)
    source = source_lineage(state.models, certificate)
    target = cad_revision_from_brep_model(model)
    target_index = entity_occurrence_index(model, target)
    solid_pairs = tuple(
        sorted(
            (item.model_index, item.entity, target_entity)
            for item, target_entity in schedule.entity_pairs
            if (item.model_index, item.entity.index) in participating
        )
    )
    face_pairs: set[tuple[int, BRepEntityId, BRepEntityId]] = set()
    for target_solid, faces in enumerate(model.topology.solid_faces):
        for target_solid_entity, _ in target_index.get(("solid", target_solid), ()):
            for face in faces:
                for operand, parent_face in selected.face_parents[face]:
                    for parent_solid in state.models[operand].topology.face_solids[
                        parent_face
                    ]:
                        for source_face in source.entities_for_solid(
                            operand, "face", parent_face, parent_solid
                        ):
                            for target_face, _ in target_index.get(("face", face), ()):
                                if (
                                    target_face.occurrence_path
                                    == target_solid_entity.occurrence_path
                                ):
                                    face_pairs.add((operand, source_face, target_face))
    edge_pairs = {
        (operand, source_edge, target_edge)
        for target_index_value, references in enumerate(overlay.built.edge_sources)
        for operand, source_edge in references
        for target_edge, _ in target_index.get(("edge", target_index_value), ())
    }
    vertex_pairs = {
        (operand, source_vertex, target_vertex)
        for target_index_value, references in enumerate(overlay.built.vertex_sources)
        for operand, source_vertex in references
        for target_vertex, _ in target_index.get(("vertex", target_index_value), ())
    }
    return FullOverlayPartition(
        model,
        solid_regions,
        solid_pairs,
        tuple(sorted(face_pairs)),
        tuple(sorted(edge_pairs)),
        tuple(sorted(vertex_pairs)),
        certificate,
    )
