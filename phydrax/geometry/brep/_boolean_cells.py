#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Complete source-face DCELs, independently of Boolean boundary retention.

The input graph owns exact intersections and chart-sheet node identities. Every
retained geometric atom has two halfedges, and its successor is fixed before any
walk starts. Chords are only the enclosure-backed winding acceleration owned by
CurveTrimLoop; they never supply intersections, nodes, or tangent order here.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from fractions import Fraction
from functools import cmp_to_key
from itertools import combinations
from typing import assert_never, Literal

import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from .._atlas import AbstractTrimCurve, CurveTrimLoop, TrimDomain
from .._interval_enclosure import interval_add, interval_multiply, interval_subtract
from ._boolean import BRepBooleanFailure
from ._boolean_arrangement import (
    _Arc,
    _arc_area_bounds,
    _Arrangement,
    _Cut,
    _Face,
    _ordered_cuts,
    _Piece,
    _piece_nodes,
    _piece_trim,
)
from ._boolean_coincidence import CoincidentFaceArc, CoincidentFaceOverlay
from ._intersection import (
    BranchRootEndpoint,
    CurveIntersectionResult,
    intersect_trim_curves,
    IntersectionCurvePointRoot,
    NativePeriodEndpoint,
    original_curve_surface_root_preparation,
    TrimIntersectionRoot,
    TrimRootEndpoint,
)
from ._intersection_curve import (
    _PeriodOffset,
    _root_parameter_in_source,
    AffinePCurve,
    CurveTrimSegment,
    IntersectionPCurve,
    original_trim_intersection_preparation,
    pcurve_affine_correspondence,
    PeriodicPCurve,
)
from ._patches import AbstractCurve, LineCurve
from ._root_bindings import BRepRootSupport


type HalfedgeKey = tuple[str, str, str, int]
type _ChartScalar = Fraction | _PeriodOffset


def _halfedge_key(piece: _Piece) -> HalfedgeKey:
    return piece.arc.token, piece.first.node, piece.last.node, piece.sense


@dataclass(frozen=True, slots=True)
class FaceCellUnresolved:
    reason: str
    arcs: tuple[str, ...] = ()
    nodes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FaceArrangementCell:
    owner: _Face
    loops: tuple[tuple[_Piece, ...], ...]
    domain: TrimDomain
    classification_key: str
    seed_box: np.ndarray
    covered_halfedges: tuple[HalfedgeKey, ...]
    root_nodes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FaceCellWorkset:
    """Full source/branch/overlay graph, with exact quotient ancestry."""

    arcs: tuple[_Arc, ...]
    cuts: Mapping[str, tuple[_Cut, ...]]
    aliases: Mapping[str, str] = field(default_factory=dict)
    quotient_sources: Mapping[tuple[str, str, str], tuple[str, ...]] = field(
        default_factory=dict
    )
    unresolved: tuple[FaceCellUnresolved, ...] = ()
    intersection_boxes: int = 0
    overlay_sources: Mapping[str, CoincidentFaceArc] = field(default_factory=dict)
    source_root_ids: tuple[str, ...] = ()
    intersection_roots: Mapping[str, TrimIntersectionRoot] = field(default_factory=dict)
    work: int = 0
    branch_owners: Mapping[str, tuple[_Face, _Face]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FaceArrangementCells:
    owner: _Face
    cells: tuple[FaceArrangementCell, ...]
    covered_halfedges: tuple[HalfedgeKey, ...]
    excluded_cycles: tuple[tuple[HalfedgeKey, ...], ...]
    covered_roots: tuple[str, ...]
    unresolved_roots: tuple[str, ...]
    unresolved_cells: tuple[FaceCellUnresolved, ...]
    work: int
    complete: bool
    quotient_sources: Mapping[tuple[str, str, str], tuple[str, ...]]
    overlay_sources: Mapping[str, CoincidentFaceArc]
    intersection_roots: Mapping[str, TrimIntersectionRoot]
    branch_owners: Mapping[str, tuple[_Face, _Face]]


@dataclass
class _Work:
    maximum: int
    used: int = 0
    charge: Callable[[int], None] | None = None

    def consume(self, count: int = 1) -> None:
        if count > self.maximum - self.used:
            raise BRepBooleanFailure("full face-cell work budget exhausted")
        self.used += count
        if self.charge is not None:
            self.charge(count)


def _root_ids(cuts: Mapping[str, tuple[_Cut, ...]]) -> tuple[str, ...]:
    roots: set[str] = set()
    for values in cuts.values():
        for cut in values:
            if cut.endpoint is not None:
                roots.add(cut.endpoint.root_id)
            binding = cut.vertex_root
            if binding is None:
                continue
            primary = binding.primary
            roots.add(
                primary.root.root_id
                if isinstance(primary, BRepRootSupport)
                else primary.root_id
            )
            roots.update(support.root.root_id for support in binding.aliases)
            if binding.spatial_root is not None:
                roots.add(binding.spatial_root.root_id)
            if binding.joint_root is not None:
                roots.add(binding.joint_root.root_id)
    return tuple(sorted(roots))


def _cross_bounds(
    a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    return interval_subtract(
        interval_multiply((a[0][0], a[1][0]), (b[0][1], b[1][1])),
        interval_multiply((a[0][1], a[1][1]), (b[0][0], b[1][0])),
    )


def _outward_tangent(piece: _Piece) -> tuple[np.ndarray, np.ndarray]:
    cut = piece.first if piece.sense > 0 else piece.last
    lower, upper = piece.arc.trim.parameter_interval
    # The entire root enclosure participates, not a nominal root or secant.
    first, last = max(lower, cut.lower), min(upper, cut.upper)
    if first > last:
        raise BRepBooleanFailure(
            "root enclosure leaves its trim carrier", (piece.arc.token,)
        )
    lo, hi = piece.arc.trim.derivative_bounds(first, last)
    if not np.all(np.isfinite((lo, hi))):
        raise BRepBooleanFailure(
            "junction tangent enclosure is unbounded", (piece.arc.token,)
        )
    return (lo, hi) if piece.sense > 0 else (-hi, -lo)


def _cyclic_order(pieces: tuple[_Piece, ...], work: _Work) -> tuple[_Piece, ...]:
    """Prove a total ray order using interval halfplanes and determinants.

    Midpoint angles propose a separating diameter only. An accepted diameter
    strictly separates every full tangent cone from its axis; each comparison
    within a halfplane then requires a strictly positive/negative determinant.
    Opposite rays reside in opposite halfplanes and need no false transversality.
    """
    tangents = tuple(_outward_tangent(piece) for piece in pieces)
    centers = np.asarray([0.5 * (lo + hi) for lo, hi in tangents])
    angles = np.arctan2(centers[:, 1], centers[:, 0]) % np.pi
    candidates = sorted(set(float(angle) for angle in angles))
    for index, angle in enumerate(candidates):
        following = candidates[(index + 1) % len(candidates)]
        if index == len(candidates) - 1:
            following += np.pi
        reference = 0.5 * (angle + following)
        axis = np.asarray((np.cos(reference), np.sin(reference)), dtype=np.float64)
        axis_bounds = (axis, axis)
        sides = []
        for tangent in tangents:
            work.consume()
            lo, hi = _cross_bounds(axis_bounds, tangent)
            sides.append(0 if lo > 0 else 1 if hi < 0 else -1)
        if -1 in sides:
            continue

        def compare(first: int, second: int) -> int:
            work.consume()
            if first == second:
                return 0
            if sides[first] != sides[second]:
                return -1 if sides[first] < sides[second] else 1
            lo, hi = _cross_bounds(tangents[first], tangents[second])
            if lo > 0:
                return -1
            if hi < 0:
                return 1
            raise BRepBooleanFailure(
                "junction tangent cyclic order is unresolved",
                (pieces[first].arc.token, pieces[second].arc.token),
            )

        order = sorted(range(len(pieces)), key=cmp_to_key(compare))
        # Certify every pair, rather than relying on comparisons requested by sort.
        for first, second in combinations(order, 2):
            if compare(first, second) >= 0:
                raise BRepBooleanFailure(
                    "junction tangent cones have no total cyclic order"
                )
        return tuple(pieces[index] for index in order)
    raise BRepBooleanFailure("junction tangent cones have no separating diameter")


def _box_status(domain: TrimDomain, box: np.ndarray, work: _Work) -> int:
    work.consume()
    decision = domain.classify_boxes(
        box[None], maximum_refinements=work.maximum - work.used
    )
    work.consume(int(decision.refinements[0]))
    if not decision.resolved[0]:
        return 0
    return 1 if decision.inside[0] else -1


def _fixed_point_nodes(
    owner: _Face, workset: FaceCellWorkset, work: _Work
) -> Mapping[str, str]:
    """Alias exact fixed branch point IDs only in a common authored chart sheet.

    Closed 0/N roots keep distinct parameter root IDs and one native point ID.
    Closed representative sheets use the native certified integer period shifts,
    not a test on floating continuation residuals. Interior seams stay split.
    """
    aliases = dict(workset.aliases)
    points: dict[tuple[str, str, tuple[int, ...]], str] = {}
    source_points: dict[tuple[str, str, tuple[int, ...]], tuple[_Arc, _Cut]] = {}
    for arc in workset.arcs:
        branch = arc.branch
        if branch is None:
            continue
        first_owner, second_owner = workset.branch_owners[arc.token]
        if owner == first_owner:
            region, columns = branch.first, slice(0, 2)
        elif owner == second_owner:
            region, columns = branch.second, slice(2, 4)
        else:
            raise BRepBooleanFailure(
                "branch point has no declared source chart owner", (arc.token,)
            )
        for cut in workset.cuts[arc.token]:
            work.consume()
            endpoint = cut.endpoint
            match endpoint:
                case None | NativePeriodEndpoint():
                    # Scalar period endpoints have no spatial fixed-point root.
                    continue
                case TrimRootEndpoint() | BranchRootEndpoint():
                    root = endpoint.root
                case _:
                    assert_never(endpoint)
            if isinstance(endpoint, BranchRootEndpoint):
                if endpoint.curve.branch_id != branch.branch_id:
                    raise BRepBooleanFailure(
                        "source point root has a different branch owner", (arc.token,)
                    )
                gauges = tuple(int(value) for value in endpoint._source_gauges()[columns])
                source_key = branch.branch_id, root.root_id, gauges
                previous_source = source_points.get(source_key)
                if previous_source is None:
                    source_points[source_key] = arc, cut
                else:
                    previous_arc, previous_cut = previous_source
                    previous_endpoint = previous_cut.endpoint
                    if (
                        cut.vertex_root is None
                        or not isinstance(previous_endpoint, BranchRootEndpoint)
                        or not isinstance(arc.pcurve, (AbstractCurve, IntersectionPCurve))
                        or not isinstance(
                            previous_arc.pcurve, (AbstractCurve, IntersectionPCurve)
                        )
                        or not cut.vertex_root.same_uv_endpoint(
                            region.patch,
                            arc.pcurve,
                            endpoint,
                            previous_arc.pcurve,
                            previous_endpoint,
                        )
                    ):
                        raise BRepBooleanFailure(
                            "source point nodes lack an exact same-sheet UV alias",
                            (arc.token, previous_arc.token),
                        )
                    node = _resolve_node(cut.node, aliases)
                    previous_node = _resolve_node(previous_cut.node, aliases)
                    if node != previous_node:
                        aliases[max(node, previous_node)] = min(node, previous_node)
                continue
            if not isinstance(root, IntersectionCurvePointRoot):
                continue
            if root.curve.branch_id != branch.branch_id:
                raise BRepBooleanFailure(
                    "fixed point root has a different branch owner", (arc.token,)
                )
            parameter = root.parameter
            if parameter != np.floor(parameter):
                continue
            index = int(parameter)
            shifts = [int(value) for value in branch.node_period_shifts[index, columns]]
            if index > 0 and cut.parameter == arc.trim.parameter_interval[0]:
                # Same gauge rule as native BranchRootEndpoint's node witness:
                # a chart start includes its preceding integer seam transition.
                transition = branch.transition_shifts[index - 1, columns]
                for axis, period in enumerate(region.patch.periods):
                    if period is not None:
                        shifts[axis] += int(round(float(transition[axis]) / period))
            node = _resolve_node(cut.node, aliases)
            key = branch.branch_id, root.point_id, tuple(shifts)
            previous = points.get(key)
            if previous is None:
                points[key] = node
            else:
                previous = _resolve_node(previous, aliases)
                if previous != node:
                    aliases[max(previous, node)] = min(previous, node)
    return {node: _resolve_node(node, aliases) for node in aliases}


def _full_pieces(
    state: _Arrangement, owner: _Face, workset: FaceCellWorkset, work: _Work
) -> tuple[_Piece, ...]:
    pieces = []
    cuts_by_arc = {
        key: [
            replace(
                cut, vertex_root=state.vertex_bindings.get(cut.vertex, cut.vertex_root)
            )
            for cut in values
        ]
        for key, values in workset.cuts.items()
    }
    local = replace(state, arcs={owner: list(workset.arcs)}, cuts=cuts_by_arc)
    bound_workset = replace(
        workset, cuts={key: tuple(values) for key, values in cuts_by_arc.items()}
    )
    aliases = dict(_fixed_point_nodes(owner, bound_workset, work))
    root_nodes: dict[str, list[tuple[_Arc, _Cut]]] = {}
    patch = state.models[owner[0]].patches[owner[1]]
    for arc in workset.arcs:
        if not isinstance(arc.pcurve, (AbstractCurve, IntersectionPCurve)):
            continue
        for cut in cuts_by_arc[arc.token]:
            endpoint = cut.endpoint
            binding = cut.vertex_root
            if binding is None or not isinstance(
                endpoint, (TrimRootEndpoint, BranchRootEndpoint)
            ):
                continue
            previous_nodes = root_nodes.setdefault(binding.root_id, [])
            for previous_arc, previous_cut in previous_nodes:
                work.consume()
                previous_endpoint = previous_cut.endpoint
                if (
                    isinstance(previous_endpoint, (TrimRootEndpoint, BranchRootEndpoint))
                    and isinstance(
                        previous_arc.pcurve, (AbstractCurve, IntersectionPCurve)
                    )
                    and binding.same_uv_endpoint(
                        patch,
                        arc.pcurve,
                        endpoint,
                        previous_arc.pcurve,
                        previous_endpoint,
                    )
                ):
                    node = _resolve_node(cut.node, aliases)
                    previous_node = _resolve_node(previous_cut.node, aliases)
                    if node != previous_node:
                        aliases[max(node, previous_node)] = min(node, previous_node)
                    break
            else:
                previous_nodes.append((arc, cut))
    aliases = {node: _resolve_node(node, aliases) for node in aliases}
    for arc in workset.arcs:
        cuts = _ordered_cuts(local, arc)
        opposite_domain = None
        opposite_pcurve = None
        if arc.branch is not None:
            first_owner, second_owner = workset.branch_owners[arc.token]
            if owner == first_owner:
                opposite_domain = state.domains[second_owner]
                opposite_pcurve = arc.branch.p_curve("second")
            elif owner == second_owner:
                opposite_domain = state.domains[first_owner]
                opposite_pcurve = arc.branch.p_curve("first")
            else:
                raise BRepBooleanFailure(
                    "branch atom has no declared generating face", (arc.token,)
                )
        for first, last in zip(cuts[:-1], cuts[1:], strict=True):
            work.consume()
            if not first.upper < last.lower:
                raise BRepBooleanFailure(
                    "full face atom has unresolved root ordering", (arc.token,)
                )
            if arc.branch is not None:
                middle = 0.5 * (first.upper + last.lower)
                status = _box_status(
                    state.domains[owner], arc.trim.enclosure(middle, middle), work
                )
                if status == 0:
                    raise BRepBooleanFailure(
                        "branch atom original trim membership is unresolved", (arc.token,)
                    )
                if status < 0:
                    continue
                if opposite_domain is None or opposite_pcurve is None:
                    raise RuntimeError(
                        "A source branch lost its opposite trim authority."
                    )
                # The chart rectangle is only a discovery cover. An exact
                # intersection atom exists only inside BOTH generating trims.
                status = _box_status(
                    opposite_domain, opposite_pcurve.enclosure(middle, middle), work
                )
                if status == 0:
                    raise BRepBooleanFailure(
                        "branch atom opposite trim membership is unresolved", (arc.token,)
                    )
                if status < 0:
                    continue
            first = replace(first, node=aliases.get(first.node, first.node))
            last = replace(last, node=aliases.get(last.node, last.node))
            pieces.extend((_Piece(arc, first, last, 1), _Piece(arc, first, last, -1)))
            if len(pieces) > state.policy.sewing.maximum_coedges:
                raise BRepBooleanFailure(
                    "full directed face graph exceeds coedge capacity"
                )
    return tuple(pieces)


def _cycles(
    pieces: tuple[_Piece, ...], work: _Work
) -> tuple[tuple[tuple[_Piece, ...], ...], tuple[int, ...]]:
    outgoing: dict[str, list[int]] = {}
    for index, piece in enumerate(pieces):
        outgoing.setdefault(_piece_nodes(piece)[0], []).append(index)
    by_key = {_halfedge_key(piece): index for index, piece in enumerate(pieces)}
    if len(by_key) != len(pieces):
        raise BRepBooleanFailure("full face graph contains duplicate unquotiented atoms")
    successor: dict[int, int] = {}
    for node, indices in outgoing.items():
        ordered = _cyclic_order(tuple(pieces[index] for index in indices), work)
        for position, ray in enumerate(ordered):
            twin_key = (*_halfedge_key(ray)[:3], -ray.sense)
            twin = by_key[twin_key]
            # Clockwise predecessor of the incoming twin keeps the face on left.
            successor[twin] = by_key[_halfedge_key(ordered[position - 1])]
    if len(successor) != len(pieces) or len(set(successor.values())) != len(pieces):
        raise BRepBooleanFailure("full face successor is not a halfedge permutation")
    node_components: dict[str, int] = {}
    for node in sorted(outgoing):
        if node in node_components:
            continue
        component = len(node_components)
        pending = [node]
        while pending:
            current = pending.pop()
            if current in node_components:
                continue
            node_components[current] = component
            work.consume()
            pending.extend(_piece_nodes(pieces[index])[1] for index in outgoing[current])
    remaining = set(range(len(pieces)))
    loops = []
    components = []
    while remaining:
        start = min(remaining, key=lambda index: _halfedge_key(pieces[index]))
        current = start
        loop = []
        while current in remaining:
            work.consume()
            remaining.remove(current)
            loop.append(pieces[current])
            current = successor[current]
        if current != start:
            raise BRepBooleanFailure("DCEL walk enters a previously consumed orbit")
        orbit_keys = {_halfedge_key(piece) for piece in loop}
        if any((*key[:3], -key[3]) in orbit_keys for key in orbit_keys):
            raise BRepBooleanFailure(
                "face graph contains a bridge without a complete endpoint/root closure",
                tuple(sorted({piece.arc.token for piece in loop})),
            )
        loops.append(tuple(loop))
        components.append(node_components[_piece_nodes(pieces[start])[0]])
    return tuple(loops), tuple(components)


def _exact_loop(
    loop: tuple[_Piece, ...], state: _Arrangement, work: _Work
) -> CurveTrimLoop:
    remaining = work.maximum - work.used
    if len(loop) > remaining:
        raise BRepBooleanFailure("exact face loop cover exceeds remaining work")
    exact = CurveTrimLoop(
        tuple(_piece_trim(piece) for piece in loop),
        tolerance=state.policy.trim_resolution,
        maximum_arcs=remaining,
    )
    work.consume(exact.chords.shape[0])
    return exact


def _area_sign(loop: CurveTrimLoop, work: _Work) -> int:
    cells = [
        (loop.curves[int(index)], float(first), float(last))
        for index, first, last in zip(
            loop.arc_curves, loop.arc_first, loop.arc_last, strict=True
        )
    ]
    while cells:
        lower, upper = (
            np.asarray(0.0, dtype=np.float64),
            np.asarray(0.0, dtype=np.float64),
        )
        for curve, first, last in cells:
            work.consume()
            a, b = _arc_area_bounds(curve, first, last)
            lower, upper = interval_add((lower, upper), (np.asarray(a), np.asarray(b)))
        if lower > 0:
            return 1
        if upper < 0:
            return -1
        if 2 * len(cells) > work.maximum - work.used:
            raise BRepBooleanFailure("exact face loop area-sign work exhausted")
        following = []
        for curve, first, last in cells:
            middle = 0.5 * (first + last)
            if not first < middle < last:
                raise BRepBooleanFailure("exact face loop area sign is unresolvable")
            following.extend(((curve, first, middle), (curve, middle, last)))
        cells = following
    raise BRepBooleanFailure("exact face loop has no arcs")


def _loop_signs(
    loops: tuple[tuple[_Piece, ...], ...],
    exact: tuple[CurveTrimLoop, ...],
    work: _Work,
) -> tuple[int, ...]:
    """Certified loop area signs, sharing each sign with its reversed cycle.

    A closed cycle's signed area is the sum of its directed atoms' Green
    integrals. The cycle of the same atoms with opposite senses traverses each
    exact trim range backwards, so its area is exactly the negation.
    """
    signed: dict[frozenset[tuple[str, float, float, int]], int] = {}
    signs = []
    for loop, curve_loop in zip(loops, exact, strict=True):
        atoms = frozenset(
            (piece.arc.token, piece.first.parameter, piece.last.parameter, piece.sense)
            for piece in loop
        )
        reverse = frozenset((*atom[:3], -atom[3]) for atom in atoms)
        sign = -signed[reverse] if reverse in signed else _area_sign(curve_loop, work)
        signed[atoms] = sign
        signs.append(sign)
    return tuple(signs)


def _reference_box(loop: CurveTrimLoop) -> np.ndarray:
    curve = loop.curves[0]
    first, last = curve.parameter_interval
    middle = 0.5 * (first + last)
    return curve.enclosure(middle, middle)


def _holes(
    exact: tuple[CurveTrimLoop, ...],
    signs: tuple[int, ...],
    components: tuple[int, ...],
    work: _Work,
) -> tuple[dict[int, list[int]], tuple[int, ...]]:
    outers = tuple(index for index, sign in enumerate(signs) if sign > 0)
    holes = {index: [] for index in outers}
    exterior = []
    for index, sign in enumerate(signs):
        if sign > 0:
            continue
        matches = []
        for outer in outers:
            if components[outer] == components[index]:
                continue
            status = _box_status(
                TrimDomain(exact[outer]), _reference_box(exact[index]), work
            )
            if status == 0:
                raise BRepBooleanFailure(
                    "disconnected face loop containment is unresolved"
                )
            if status > 0:
                matches.append(outer)
        if not matches:
            exterior.append(index)
            continue
        immediate = []
        for candidate in matches:
            statuses = [
                _box_status(
                    TrimDomain(exact[parent]), _reference_box(exact[candidate]), work
                )
                for parent in matches
                if parent != candidate
            ]
            if 0 in statuses:
                raise BRepBooleanFailure("face loop immediate ancestry is unresolved")
            if all(status > 0 for status in statuses):
                immediate.append(candidate)
        if len(immediate) != 1:
            raise BRepBooleanFailure("face hole has no unique immediate ancestor")
        holes[immediate[0]].append(index)
    return holes, tuple(exterior)


def _seed_box(domain: TrimDomain, outer: CurveTrimLoop, work: _Work) -> np.ndarray:
    bounds = np.stack((np.min(outer.arc_lower, axis=0), np.max(outer.arc_upper, axis=0)))
    pending = deque((bounds,))
    while pending:
        box = pending.popleft()
        center = 0.5 * (box[0] + box[1])
        probe = np.stack((np.nextafter(center, -np.inf), np.nextafter(center, np.inf)))
        if _box_status(domain, probe, work) > 0:
            return probe
        axis = int(np.argmax(box[1] - box[0]))
        middle = center[axis]
        if not box[0, axis] < middle < box[1, axis]:
            continue
        lower, upper = box.copy(), box.copy()
        lower[1, axis], upper[0, axis] = middle, middle
        pending.extend((lower, upper))
    raise BRepBooleanFailure("open face cell has no resolved interior seed")


def extract_face_arrangement_cells(
    state: _Arrangement,
    owner: _Face,
    /,
    *,
    workset: FaceCellWorkset | None = None,
) -> FaceArrangementCells:
    """Enumerate every open region inside the original source trim.

    Call after complete source/branch intersection discovery and root propagation.
    A false ``complete`` invalidates a material classification over this face;
    callers must not turn its partial cells into an exhaustive boundary claim.
    """
    if owner not in state.arcs or owner not in state.domains:
        raise ValueError("Face-cell extraction requires the full prepared source face.")
    if workset is None:
        workset = prepare_face_cell_workset(state, owner)
    roots = tuple(
        sorted(
            set(
                (
                    *_root_ids(workset.cuts),
                    *workset.source_root_ids,
                    *workset.intersection_roots,
                )
            )
        )
    )
    work = _Work(state.policy.maximum_cells, charge=state.consume)
    cells: list[FaceArrangementCell] = []
    covered: tuple[HalfedgeKey, ...] = ()
    excluded: list[tuple[HalfedgeKey, ...]] = []
    unresolved = list(workset.unresolved)
    if not unresolved:
        try:
            with (
                original_trim_intersection_preparation(
                    tuple(arc.trim for arc in workset.arcs)
                ),
                original_curve_surface_root_preparation(
                    tuple(cut.endpoint for cuts in workset.cuts.values() for cut in cuts)
                ),
            ):
                pieces = _full_pieces(state, owner, workset, work)
                loops, components = _cycles(pieces, work)
                covered = tuple(sorted(_halfedge_key(piece) for piece in pieces))
                exact = tuple(_exact_loop(loop, state, work) for loop in loops)
                signs = _loop_signs(loops, exact, work)
                holes, exterior = _holes(exact, signs, components, work)
                excluded.extend(
                    tuple(_halfedge_key(piece) for piece in loops[index])
                    for index in exterior
                )
                for outer, children in holes.items():
                    owned = (outer, *children)
                    domain = TrimDomain(
                        exact[outer], tuple(exact[index] for index in children)
                    )
                    seed = _seed_box(domain, exact[outer], work)
                    status = _box_status(state.domains[owner], seed, work)
                    if status == 0:
                        raise BRepBooleanFailure(
                            "open cell original trim membership is unresolved"
                        )
                    edges = tuple(
                        _halfedge_key(piece) for index in owned for piece in loops[index]
                    )
                    if status < 0:
                        excluded.append(edges)
                        continue
                    if len(cells) >= state.policy.maximum_faces:
                        raise BRepBooleanFailure(
                            "full face-cell allocation capacity exhausted"
                        )
                    key = canonical_fingerprint(
                        {
                            "kind": "native-open-source-face-cell",
                            "source": state.models[owner[0]].source_revision,
                            "face": owner[1],
                            "boundaries": edges,
                        }
                    )
                    cells.append(
                        FaceArrangementCell(
                            owner,
                            tuple(loops[index] for index in owned),
                            domain,
                            key,
                            seed,
                            edges,
                            tuple(sorted({node for edge in edges for node in edge[1:3]})),
                        )
                    )
        except BRepBooleanFailure as failure:
            unresolved.append(
                FaceCellUnresolved(str(failure), tuple(arc.token for arc in workset.arcs))
            )
    complete = not unresolved
    return FaceArrangementCells(
        owner,
        tuple(cells),
        covered,
        tuple(excluded),
        roots if complete else (),
        () if complete else roots,
        tuple(unresolved),
        workset.work + work.used,
        complete,
        workset.quotient_sources,
        workset.overlay_sources,
        workset.intersection_roots,
        workset.branch_owners,
    )


def _chart_add(first: _ChartScalar, second: _ChartScalar) -> _ChartScalar:
    left = (
        first if isinstance(first, _PeriodOffset) else _PeriodOffset(first, Fraction(0))
    )
    right = (
        second
        if isinstance(second, _PeriodOffset)
        else _PeriodOffset(second, Fraction(0))
    )
    value = left + right
    if not isinstance(value, (Fraction, _PeriodOffset)):
        raise TypeError(
            "Native chart addition must retain an exact rational or period offset."
        )
    return value


def _chart_scale(value: _ChartScalar, factor: Fraction) -> _ChartScalar:
    source = (
        value if isinstance(value, _PeriodOffset) else _PeriodOffset(value, Fraction(0))
    )
    result = source * factor
    if not isinstance(result, (Fraction, _PeriodOffset)):
        raise TypeError(
            "Native chart scaling must retain an exact rational or period offset."
        )
    return result


def _chart_parts(value: _ChartScalar) -> tuple[Fraction, Fraction]:
    return (
        (value, Fraction(0))
        if isinstance(value, Fraction)
        else (value.rational, value.turns)
    )


def _chart_ratio(first: _ChartScalar, second: _ChartScalar) -> Fraction | None:
    """A rational quotient only when both native-period coefficients agree."""
    a, b = _chart_parts(first), _chart_parts(second)
    pivot = next((index for index, value in enumerate(b) if value), None)
    if pivot is None:
        return None
    ratio = a[pivot] / b[pivot]
    return ratio if all(x == ratio * y for x, y in zip(a, b, strict=True)) else None


def _line_coefficients(
    curve: AbstractCurve | AbstractTrimCurve,
) -> tuple[tuple[_ChartScalar, ...], tuple[Fraction, ...]] | None:
    if isinstance(curve, LineCurve):
        return (
            tuple(Fraction(float(value)) for value in np.asarray(curve.origin)),
            tuple(Fraction(float(value)) for value in np.asarray(curve.direction)),
        )
    if isinstance(curve, PeriodicPCurve):
        child = _line_coefficients(curve.source_curve)
        if child is None:
            return None
        origin, direction = child
        return (
            tuple(
                _chart_add(value, _PeriodOffset(Fraction(0), Fraction(shift)))
                for value, shift in zip(origin, curve.period_shifts, strict=True)
            ),
            direction,
        )
    if isinstance(curve, AffinePCurve):
        child = _line_coefficients(curve.curve)
        if child is None:
            return None
        origin, direction = child
        transformed = []
        for row in range(2):
            value: _ChartScalar = curve.offset[row]
            for column in range(2):
                value = _chart_add(
                    value, _chart_scale(origin[column], curve.matrix[row][column])
                )
            transformed.append(value)
        return (
            tuple(transformed),
            tuple(
                sum((curve.matrix[i][j] * direction[j] for j in range(2)), Fraction(0))
                for i in range(2)
            ),
        )
    return None


def _arc_scalar_map(arc: _Arc) -> tuple[_ChartScalar, _ChartScalar] | None:
    """Exact raw-carrier scalar as an affine expression in the local trim."""
    if not isinstance(arc.trim, CurveTrimSegment):
        return Fraction(arc.offset), Fraction(arc.scale)
    endpoints: list[_ChartScalar] = []
    for value, root in (
        (arc.trim.first, arc.trim.first_root),
        (arc.trim.last, arc.trim.last_root),
    ):
        if root is None:
            endpoints.append(Fraction(value))
        elif isinstance(root, NativePeriodEndpoint):
            endpoints.append(root.exact_parameter)
        else:
            return None
    first, last = endpoints if not arc.trim.reversed else tuple(reversed(endpoints))
    return first, _chart_add(last, _chart_scale(first, Fraction(-1)))


def _fixed_chart_cut(arc: _Arc, cut: _Cut) -> bool:
    if cut.endpoint is None:
        return True
    if not isinstance(cut.endpoint, NativePeriodEndpoint):
        return False
    scalar = _arc_scalar_map(arc)
    return scalar is not None and cut.endpoint.exact_parameter == _chart_add(
        scalar[0], _chart_scale(scalar[1], Fraction(cut.parameter))
    )


def _normalized_line(
    arc: _Arc,
) -> tuple[tuple[_ChartScalar, ...], tuple[_ChartScalar, ...]] | None:
    coefficients, scalar = _line_coefficients(arc.pcurve), _arc_scalar_map(arc)
    if coefficients is None or scalar is None:
        return None
    origin, direction = coefficients
    offset, scale = scalar
    return (
        tuple(
            _chart_add(value, _chart_scale(offset, derivative))
            for value, derivative in zip(origin, direction, strict=True)
        ),
        tuple(_chart_scale(scale, value) for value in direction),
    )


def _resolve_node(node: str, aliases: Mapping[str, str]) -> str:
    seen = set()
    while node in aliases:
        if node in seen:
            raise BRepBooleanFailure("exact node alias graph is cyclic")
        seen.add(node)
        node = aliases[node]
    return node


def _line_corner_node(
    root: TrimIntersectionRoot, source: _Arc, target: _Arc, operand: int, endpoint: float
) -> str | None:
    """Use source line coefficients and root uniqueness to prove a corner alias."""
    a, b = _normalized_line(source), _normalized_line(target)
    if a is None or b is None:
        return None
    point = tuple(
        _chart_add(origin, Fraction(endpoint) * direction)
        for origin, direction in zip(*a, strict=True)
    )
    pivot = next((index for index, value in enumerate(b[1]) if value), None)
    if pivot is None:
        return None
    parameter = _chart_ratio(
        _chart_add(point[pivot], _chart_scale(b[0][pivot], Fraction(-1))), b[1][pivot]
    )
    if parameter is None:
        return None
    if any(
        value != _chart_add(origin, parameter * direction)
        for value, origin, direction in zip(point, *b, strict=True)
    ):
        return None
    other = 1 - operand
    if (
        not Fraction(float(root.parameter_lower[other]))
        <= parameter
        <= Fraction(float(root.parameter_upper[other]))
    ):
        return None
    if not root.parameter_lower[operand] <= endpoint <= root.parameter_upper[operand]:
        return None
    return (
        source.first_node
        if endpoint == source.trim.parameter_interval[0]
        else source.last_node
    )


def _scalar_phase(first: _Arc, second: _Arc) -> tuple[Fraction, Fraction] | None:
    """Prove first.trim(s) == second.trim(scale*s+offset) for all s."""
    a, b = _normalized_line(first), _normalized_line(second)
    if a is not None and b is not None:
        origin_a, direction_a = a
        origin_b, direction_b = b
        pivot = next(
            (index for index, value in enumerate(direction_b) if value != 0), None
        )
        if pivot is None:
            return None
        scale = _chart_ratio(direction_a[pivot], direction_b[pivot])
        offset = _chart_ratio(
            _chart_add(origin_a[pivot], _chart_scale(origin_b[pivot], Fraction(-1))),
            direction_b[pivot],
        )
        if scale is None or offset is None or not scale:
            return None
        if any(
            x != _chart_scale(y, scale)
            for x, y in zip(direction_a, direction_b, strict=True)
        ) or any(
            x != _chart_add(y, _chart_scale(d, offset))
            for x, y, d in zip(origin_a, origin_b, direction_b, strict=True)
        ):
            return None
        return scale, offset
    identity = (
        ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))),
        (Fraction(0), Fraction(0)),
    )
    correspondence = pcurve_affine_correspondence(first.pcurve, second.pcurve)
    rooted = tuple(
        arc.trim
        for arc in (first, second)
        if isinstance(arc.trim, CurveTrimSegment)
        and (arc.trim.first_root is not None or arc.trim.last_root is not None)
    )
    if rooted:
        if (
            correspondence != identity
            or not isinstance(first.trim, CurveTrimSegment)
            or not isinstance(second.trim, CurveTrimSegment)
        ):
            return None
        for a_root, b_root in (
            (first.trim.first_root, second.trim.first_root),
            (first.trim.last_root, second.trim.last_root),
        ):
            if a_root is None or b_root is None:
                if a_root is not b_root:
                    return None
            elif isinstance(a_root, NativePeriodEndpoint) and isinstance(
                b_root, NativePeriodEndpoint
            ):
                # An exact common p-curve atom already proves the carrier map.
                # Authored period scalars compare expressions, not wrapper IDs.
                if a_root.exact_parameter != b_root.exact_parameter or (
                    _root_parameter_in_source(first.pcurve, a_root)
                    != _root_parameter_in_source(second.pcurve, b_root)
                ):
                    return None
            elif a_root.root_id != b_root.root_id or _root_parameter_in_source(
                first.pcurve, a_root
            ) != _root_parameter_in_source(second.pcurve, b_root):
                return None
        if first.trim.first != second.trim.first or first.trim.last != second.trim.last:
            return None
    if correspondence == identity:
        return Fraction(first.scale) / Fraction(second.scale), (
            Fraction(first.offset) - Fraction(second.offset)
        ) / Fraction(second.scale)
    return None


def _overlay_arcs(
    state: _Arrangement, owner: _Face, overlay: CoincidentFaceOverlay
) -> tuple[_Arc, ...]:
    model = state.models[owner[0]]
    if model.face_ids[owner[1]] != overlay.first_face_id:
        raise ValueError(
            "Coincident worksets must use the proved common first source chart."
        )
    result = []
    for loops in (overlay.first_loops, overlay.second_loops):
        for loop_index, loop in enumerate(loops):
            for ordinal, source in enumerate(loop):
                first, last = source.parameter_range
                roots = source.coedge_roots
                trim = CurveTrimSegment(
                    source.pcurve,
                    first,
                    last,
                    reversed=source.sense < 0,
                    first_root=roots[0],
                    last_root=roots[1],
                )
                vertices = (
                    source.vertex_ids
                    if source.sense > 0
                    else tuple(reversed(source.vertex_ids))
                )
                prefix = f"overlay-chart:{model.face_ids[owner[1]]}:source:{source.face_id}:loop:{loop_index}"
                result.append(
                    _Arc(
                        f"overlay:{source.certificate_id}",
                        owner,
                        trim,
                        source.curve,
                        source.pcurve,
                        first if source.sense > 0 else last,
                        (last - first) * source.sense,
                        f"{prefix}:corner:{ordinal}",
                        f"{prefix}:corner:{(ordinal + 1) % len(loop)}",
                        str(vertices[0]),
                        str(vertices[1]),
                        source.edge_id.index,
                        None,
                    )
                )
    return tuple(result)


def prepare_face_cell_workset(
    state: _Arrangement,
    owner: _Face,
    /,
    *,
    overlays: tuple[CoincidentFaceOverlay, ...] = (),
    intersection: Callable[
        [AbstractTrimCurve, AbstractTrimCurve], CurveIntersectionResult
    ]
    | None = None,
) -> FaceCellWorkset:
    """Prepare one full graph from all common-chart counterparts and branches.

    Distinct parameter families without a scalar-phase proof remain unresolved.
    The source edge/vertex IDs and endpoint roots are retained; no geometric
    closeness or sampled point equality creates a quotient or a graph node.
    """
    if not overlays:
        return FaceCellWorkset(
            tuple(state.arcs[owner]),
            {arc.token: tuple(state.cuts[arc.token]) for arc in state.arcs[owner]},
            branch_owners={
                arc.token: state.branch_owners[arc.token]
                for arc in state.arcs[owner]
                if arc.branch is not None
            },
        )
    unique_arcs: dict[str, _Arc] = {}
    sources: dict[str, CoincidentFaceArc] = {}
    for overlay in overlays:
        for arc in _overlay_arcs(state, owner, overlay):
            unique_arcs.setdefault(arc.token, arc)
        for loops in (overlay.first_loops, overlay.second_loops):
            for loop in loops:
                for source in loop:
                    sources.setdefault(f"overlay:{source.certificate_id}", source)
    arcs = tuple(unique_arcs.values())
    cuts: dict[str, list[_Cut]] = {}
    prepared_native: set[str] = set()
    parent_arcs = {arc.token: arc for arc in state.arcs[owner]}
    for arc in arcs:
        source = sources[arc.token]
        parent_token = f"source:{owner[0]}:coedge:{source.coedge}"
        if (
            source.face_id == state.models[owner[0]].face_ids[owner[1]]
            and parent_token in parent_arcs
        ):
            original = parent_arcs[parent_token]
            unique_arcs[arc.token] = replace(
                arc,
                first_node=original.first_node,
                last_node=original.last_node,
                first_vertex=original.first_vertex,
                last_vertex=original.last_vertex,
            )
            cuts[arc.token] = list(state.cuts[parent_token])
            prepared_native.add(arc.token)
            continue
        roots = (
            source.coedge_roots
            if source.sense > 0
            else tuple(reversed(source.coedge_roots))
        )
        vertex_roots = (
            source.vertex_roots
            if source.sense > 0
            else tuple(reversed(source.vertex_roots))
        )
        points = np.asarray(arc.trim.evaluate(jnp.asarray((0.0, 1.0), dtype=jnp.float64)))
        # Points are payload only. Topology uses authored IDs and proved roots.
        patch = state.models[owner[0]].patches[owner[1]]
        xyz = np.asarray(patch.evaluate(jnp.asarray(points, dtype=jnp.float64)))
        cuts[arc.token] = [
            _Cut(
                0.0,
                0.0,
                0.0,
                arc.first_node,
                arc.first_vertex,
                roots[0],
                vertex_roots[0],
                xyz[0],
            ),
            _Cut(
                1.0,
                1.0,
                1.0,
                arc.last_node,
                arc.last_vertex,
                roots[1],
                vertex_roots[1],
                xyz[1],
            ),
        ]
    arcs = tuple(unique_arcs.values())
    # Branches already have complete coupled root discovery in the parent.
    for arc in state.arcs[owner]:
        if arc.branch is not None:
            arcs += (arc,)
            cuts[arc.token] = list(state.cuts[arc.token])
            prepared_native.add(arc.token)
    if len(arcs) > state.policy.sewing.maximum_coedges:
        return FaceCellWorkset(
            arcs,
            {key: tuple(values) for key, values in cuts.items()},
            unresolved=(FaceCellUnresolved("full overlay coedge capacity exhausted"),),
            overlay_sources=sources,
        )
    aliases: dict[str, str] = {}
    phases: list[tuple[_Arc, _Arc, Fraction, Fraction]] = []
    unresolved: list[FaceCellUnresolved] = []
    boxes = 0
    event_roots: dict[str, TrimIntersectionRoot] = {}
    preparation_work = _Work(state.policy.maximum_cells, charge=state.consume)
    try:
        for first, second in combinations(arcs, 2):
            preparation_work.consume()
            phase = _scalar_phase(first, second)
            if phase is not None:
                phases.append((first, second, *phase))
        # Fixed scalar endpoint identities are established before cross-family
        # events, so identical authored corners need no new isolated root.
        for first, second, scale, offset in phases:
            for cut in cuts[first.token]:
                value = scale * Fraction(cut.parameter) + offset
                for other in cuts[second.token]:
                    preparation_work.consume()
                    fixed = _fixed_chart_cut(first, cut) and _fixed_chart_cut(
                        second, other
                    )
                    same_root = (
                        cut.endpoint is not None
                        and other.endpoint is not None
                        and cut.endpoint.root_id == other.endpoint.root_id
                        and _root_parameter_in_source(first.pcurve, cut.endpoint)
                        == _root_parameter_in_source(second.pcurve, other.endpoint)
                    )
                    if value == Fraction(other.parameter) and (fixed or same_root):
                        a_node, b_node = (
                            _resolve_node(cut.node, aliases),
                            _resolve_node(other.node, aliases),
                        )
                        if a_node != b_node:
                            aliases[max(a_node, b_node)] = min(a_node, b_node)
    except BRepBooleanFailure as failure:
        unresolved.append(FaceCellUnresolved(str(failure)))
    phase_pairs = {
        frozenset((first.token, second.token)) for first, second, _, _ in phases
    }
    for first, second in combinations(arcs, 2):
        if unresolved:
            break
        try:
            preparation_work.consume()
        except BRepBooleanFailure as failure:
            unresolved.append(FaceCellUnresolved(str(failure)))
            break
        if frozenset((first.token, second.token)) in phase_pairs:
            continue
        if first.token in prepared_native and second.token in prepared_native:
            # These are the parent's already-certified source/branch events in
            # the unchanged first chart, with original SourceRoot cuts retained.
            continue
        shared_nodes = {_resolve_node(cut.node, aliases) for cut in cuts[first.token]} & {
            _resolve_node(cut.node, aliases) for cut in cuts[second.token]
        }
        if (
            shared_nodes
            and _normalized_line(first) is not None
            and _normalized_line(second) is not None
        ):
            # Noncoincident exact lines sharing an authored/proved node have
            # exactly one intersection; its source identity is already owned.
            continue
        crossing = _exact_line_crossing(first, second)
        if crossing is False:
            continue
        if crossing is not None and _add_exact_line_crossing(
            state, owner, first, second, crossing, cuts, aliases
        ):
            continue
        a, b = (
            first.trim.enclosure(*first.trim.parameter_interval),
            second.trim.enclosure(*second.trim.parameter_interval),
        )
        if np.any(a[1] < b[0]) or np.any(b[1] < a[0]):
            continue
        result = (
            intersect_trim_curves(
                first.trim, second.trim, policy=state.policy.intersection
            )
            if intersection is None
            else intersection(first.trim, second.trim)
        )
        boxes += result.work.boxes_processed
        try:
            preparation_work.consume(result.work.boxes_processed)
        except BRepBooleanFailure as failure:
            unresolved.append(
                FaceCellUnresolved(str(failure), (first.token, second.token))
            )
            break
        if not result.complete or result.coincident:
            unresolved.append(
                FaceCellUnresolved(
                    "overlay intersections or scalar-phase coincidence unresolved",
                    (first.token, second.token),
                )
            )
            continue
        for event in result.points:
            if event.kind != "transversal":
                unresolved.append(
                    FaceCellUnresolved(
                        "overlay singular junction lacks tangent order proof",
                        (first.token, second.token),
                    )
                )
                continue
            root = TrimIntersectionRoot(
                first.trim,
                second.trim,
                parameter_lower=event.parameter_lower,
                parameter_upper=event.parameter_upper,
            )
            event_roots[root.root_id] = root
            node = f"overlay-root:{root.root_id}:face:{owner}"
            for operand, arc in enumerate((first, second)):
                parameter = float(root.parameters[operand])
                lower, upper = (
                    float(root.parameter_lower[operand]),
                    float(root.parameter_upper[operand]),
                )
                local_node = node
                domain_first, domain_last = arc.trim.parameter_interval
                if lower <= domain_first <= upper or lower <= domain_last <= upper:
                    corner = (
                        domain_first if lower <= domain_first <= upper else domain_last
                    )
                    other_arc = second if operand == 0 else first
                    proved_node = _line_corner_node(root, arc, other_arc, operand, corner)
                    if proved_node is None:
                        unresolved.append(
                            FaceCellUnresolved(
                                "endpoint event lacks an exact source-node alias",
                                (first.token, second.token),
                                (
                                    arc.first_node
                                    if corner == domain_first
                                    else arc.last_node,
                                ),
                            )
                        )
                        continue
                    local_node = proved_node
                    old_node, root_node = (
                        _resolve_node(local_node, aliases),
                        _resolve_node(node, aliases),
                    )
                    if old_node != root_node:
                        aliases[max(old_node, root_node)] = min(old_node, root_node)
                endpoint = TrimRootEndpoint(root, "first" if operand == 0 else "second")
                patch = state.models[owner[0]].patches[owner[1]]
                xyz = np.asarray(
                    patch.evaluate(jnp.asarray(event.point, dtype=jnp.float64))
                )
                if local_node != node:
                    values = cuts[arc.token]
                    index = 0 if local_node == arc.first_node else 1
                    values[index] = replace(values[index], node=node)
                else:
                    cuts[arc.token].append(
                        _Cut(parameter, lower, upper, node, node, endpoint, None, xyz)
                    )
    try:
        result = _quotient_workset(
            arcs, cuts, aliases, phases, unresolved, boxes, preparation_work
        )
    except BRepBooleanFailure as failure:
        unresolved.append(FaceCellUnresolved(str(failure)))
        result = FaceCellWorkset(
            arcs,
            {key: tuple(values) for key, values in cuts.items()},
            unresolved=tuple(unresolved),
            intersection_boxes=boxes,
        )
    atom_sources = {
        arc.token: sources[arc.token.split(":atom:", 1)[0]]
        for arc in result.arcs
        if arc.token.split(":atom:", 1)[0] in sources
    }
    return replace(
        result,
        overlay_sources={**sources, **atom_sources},
        source_root_ids=_root_ids({key: tuple(values) for key, values in cuts.items()}),
        intersection_roots=event_roots,
        work=preparation_work.used,
        branch_owners={
            arc.token: state.branch_owners[arc.token.split(":atom:", 1)[0]]
            for arc in result.arcs
            if arc.branch is not None
        },
    )


def _exact_line_crossing(
    first: _Arc, second: _Arc
) -> tuple[Fraction, Fraction] | Literal[False] | None:
    """Exact rational local crossing of affine lines in Q + Q*(2*pi).

    Native 2*pi is transcendental, so a rational crossing must satisfy both
    coefficients of each chart coordinate. Non-rational crossings remain with
    the root owner; mathematical period expressions never become binary64 atoms.
    """
    a, b = _normalized_line(first), _normalized_line(second)
    if a is None or b is None:
        return None
    (origin_a, direction_a), (origin_b, direction_b) = a, b
    rows = []
    for oa, ob, da, db in zip(origin_a, origin_b, direction_a, direction_b, strict=True):
        delta = _chart_add(ob, _chart_scale(oa, Fraction(-1)))
        rows.extend(
            zip(_chart_parts(da), _chart_parts(db), _chart_parts(delta), strict=True)
        )
    for left, right in combinations(rows, 2):
        ax, bx, rx = left
        ay, by, ry = right
        determinant = bx * ay - ax * by
        if not determinant:
            continue
        s = (bx * ry - by * rx) / determinant
        t = (ax * ry - ay * rx) / determinant
        if any(x * s - y * t != value for x, y, value in rows):
            return None
        a_first, a_last = first.trim.parameter_interval
        b_first, b_last = second.trim.parameter_interval
        if not (
            Fraction(a_first) <= s <= Fraction(a_last)
            and Fraction(b_first) <= t <= Fraction(b_last)
        ):
            return False
        return s, t
    return None


def _add_exact_line_crossing(
    state: _Arrangement,
    owner: _Face,
    first: _Arc,
    second: _Arc,
    crossing: tuple[Fraction, Fraction],
    cuts: dict[str, list[_Cut]],
    aliases: dict[str, str],
) -> bool:
    """Record an exact rational line crossing as fixed cuts sharing one node.

    An existing fixed cut at the exact parameter (an authored corner or an
    exactly transported endpoint) supplies the node; otherwise both arcs get a
    new fixed node. An isolated root cut whose enclosure meets the crossing, an
    unrepresentable local parameter, or a new native-period world point remains
    with its root owner. Authored period scalars retain their exact expressions.
    """
    if any(Fraction(float(value)) != value for value in crossing):
        return False
    existing: list[_Cut | None] = []
    for arc, value in ((first, crossing[0]), (second, crossing[1])):
        if any(
            cut.endpoint is not None
            and not _fixed_chart_cut(arc, cut)
            and Fraction(cut.lower) <= value <= Fraction(cut.upper)
            for cut in cuts[arc.token]
        ):
            return False
        existing.append(
            next(
                (cut for cut in cuts[arc.token] if Fraction(cut.parameter) == value), None
            )
        )
    origin, direction = _normalized_line(first) or ((), ())
    point = tuple(
        _chart_add(value, _chart_scale(slope, crossing[0]))
        for value, slope in zip(origin, direction, strict=True)
    )
    uv = np.asarray(
        [
            float(value)
            if isinstance(value, Fraction)
            else float(value.rational) + float(value.turns) * float(np.pi * 2.0)
            for value in point
        ],
        dtype=np.float64,
    )
    if uv.shape != (2,):
        return False
    if existing[0] is not None and existing[1] is not None:
        a, b = (
            _resolve_node(existing[0].node, aliases),
            _resolve_node(existing[1].node, aliases),
        )
        if a != b:
            aliases[max(a, b)] = min(a, b)
        return True
    known = existing[0] if existing[0] is not None else existing[1]
    if known is None and any(isinstance(value, _PeriodOffset) for value in point):
        # A nonliteral world vertex needs its owning isolated source root.
        return False
    if known is None:
        node = f"line-crossing:{owner}:{first.token}:{crossing[0]}:{second.token}:{crossing[1]}"
        xyz = np.asarray(
            state.models[owner[0]]
            .patches[owner[1]]
            .evaluate(jnp.asarray(uv, dtype=jnp.float64))
        )
        template = _Cut(0.0, 0.0, 0.0, node, node, None, None, xyz)
    else:
        template = replace(known, endpoint=None)
    for arc, value, cut in (
        (first, crossing[0], existing[0]),
        (second, crossing[1], existing[1]),
    ):
        if cut is None:
            parameter = float(value)
            endpoint = None
            if isinstance(arc.trim, CurveTrimSegment):
                native = next(
                    (
                        root
                        for root in (arc.trim.first_root, arc.trim.last_root)
                        if isinstance(root, NativePeriodEndpoint)
                    ),
                    None,
                )
                scalar = _arc_scalar_map(arc)
                if native is not None and scalar is not None:
                    raw = _chart_add(scalar[0], _chart_scale(scalar[1], value))
                    rational, turns = _chart_parts(raw)
                    endpoint = NativePeriodEndpoint(
                        arc.pcurve,
                        native.patch,
                        native.axis,
                        rational=rational,
                        turns=turns,
                    )
            cuts[arc.token].append(
                replace(
                    template,
                    parameter=parameter,
                    lower=parameter,
                    upper=parameter,
                    endpoint=endpoint,
                )
            )
    return True


def _transport_phase_cut(
    source: _Arc, target: _Arc, cut: _Cut, factor: Fraction, shift: Fraction
) -> _Cut | FaceCellUnresolved | None:
    value = factor * Fraction(cut.parameter) + shift
    first, last = target.trim.parameter_interval
    if not Fraction(first) <= value <= Fraction(last):
        return None
    representative = float(value)
    endpoint = cut.endpoint
    if endpoint is None:
        if Fraction(representative) != value:
            return FaceCellUnresolved(
                "overlap scalar endpoint has no exact represented parameter",
                (source.token, target.token),
                (cut.node,),
            )
    elif isinstance(endpoint, NativePeriodEndpoint):
        scalar = _arc_scalar_map(target)
        if scalar is None or not _fixed_chart_cut(source, cut):
            return FaceCellUnresolved(
                "native overlap endpoint lacks an exact local scalar map",
                (source.token, target.token),
                (cut.node,),
            )
        raw = _chart_add(scalar[0], _chart_scale(scalar[1], value))
        rational, turns = _chart_parts(raw)
        endpoint = NativePeriodEndpoint(
            target.pcurve, endpoint.patch, endpoint.axis, rational=rational, turns=turns
        )
    else:
        if pcurve_affine_correspondence(source.pcurve, target.pcurve) is None:
            return FaceCellUnresolved(
                "distinct-carrier overlap root requires owning scalar-phase root transport",
                (source.token, target.token),
                (cut.node,),
            )
        raw_scale = Fraction(target.scale) * factor / Fraction(source.scale)
        raw_offset = (
            Fraction(target.offset)
            + Fraction(target.scale) * shift
            - raw_scale * Fraction(source.offset)
        )
        scale_float, offset_float = float(raw_scale), float(raw_offset)
        if Fraction(scale_float) != raw_scale or Fraction(offset_float) != raw_offset:
            return FaceCellUnresolved(
                "overlap root affine expression is not exactly representable",
                (source.token, target.token),
                (cut.node,),
            )
        if raw_scale != 1 or raw_offset != 0:
            endpoint = endpoint.affine(scale_float, offset_float)
    lower = factor * Fraction(cut.lower) + shift
    upper = factor * Fraction(cut.upper) + shift
    lower, upper = min(lower, upper), max(lower, upper)
    lo, hi = float(lower), float(upper)
    if Fraction(lo) > lower:
        lo = np.nextafter(lo, -np.inf)
    if Fraction(hi) < upper:
        hi = np.nextafter(hi, np.inf)
    return replace(cut, parameter=representative, lower=lo, upper=hi, endpoint=endpoint)


def _quotient_workset(
    arcs: tuple[_Arc, ...],
    cuts: dict[str, list[_Cut]],
    aliases: dict[str, str],
    phases: list[tuple[_Arc, _Arc, Fraction, Fraction]],
    unresolved: list[FaceCellUnresolved],
    boxes: int,
    work: _Work,
) -> FaceCellWorkset:
    def canonical(node: str) -> str:
        return _resolve_node(node, aliases)

    for first, second, scale, offset in phases:
        for source, target, factor, shift in (
            (first, second, scale, offset),
            (second, first, 1 / scale, -offset / scale),
        ):
            for cut in tuple(cuts[source.token]):
                work.consume()
                same_node = [
                    other
                    for other in cuts[target.token]
                    if canonical(other.node) == canonical(cut.node)
                ]
                if same_node:
                    continue
                transported = _transport_phase_cut(source, target, cut, factor, shift)
                if transported is None:
                    continue
                if isinstance(transported, FaceCellUnresolved):
                    unresolved.append(transported)
                    continue
                matches = [
                    other
                    for other in cuts[target.token]
                    if other.parameter == transported.parameter
                ]
                if matches:
                    for other in matches:
                        fixed = _fixed_chart_cut(
                            target, transported
                        ) and _fixed_chart_cut(target, other)
                        same_root = (
                            transported.endpoint is not None
                            and other.endpoint is not None
                            and transported.endpoint.root_id == other.endpoint.root_id
                            and _root_parameter_in_source(
                                target.pcurve, transported.endpoint
                            )
                            == _root_parameter_in_source(target.pcurve, other.endpoint)
                        )
                        if not fixed and not same_root:
                            unresolved.append(
                                FaceCellUnresolved(
                                    "overlap endpoints lack an exact common source root",
                                    (source.token, target.token),
                                    (cut.node, other.node),
                                )
                            )
                            continue
                        a, b = canonical(other.node), canonical(cut.node)
                        if a != b:
                            aliases[max(a, b)] = min(a, b)
                    continue
                cuts[target.token].append(transported)
    aliases = {node: canonical(node) for node in tuple(aliases)}
    quotient: dict[tuple[str, str, str], tuple[str, ...]] = {}
    retained: list[_Arc] = []
    retained_cuts: dict[str, tuple[_Cut, ...]] = {}
    atoms: dict[tuple[str, str], tuple[_Arc, _Cut, _Cut]] = {}
    atom_sources: dict[str, str] = {}
    phase_pairs = {frozenset((a.token, b.token)) for a, b, _, _ in phases}
    for arc in arcs:
        ordered = sorted(cuts[arc.token], key=lambda cut: cut.parameter)
        unique: list[_Cut] = []
        for cut in ordered:
            work.consume()
            cut = replace(cut, node=aliases.get(cut.node, cut.node))
            if unique and cut.parameter == unique[-1].parameter:
                if cut.node != unique[-1].node:
                    unresolved.append(
                        FaceCellUnresolved(
                            "overlap endpoint root alias is unresolved",
                            (arc.token,),
                            (cut.node, unique[-1].node),
                        )
                    )
                if unique[-1].endpoint is None and cut.endpoint is not None:
                    unique[-1] = cut
                continue
            unique.append(cut)
        for first, last in zip(unique[:-1], unique[1:], strict=True):
            work.consume()
            nodes = (min(first.node, last.node), max(first.node, last.node))
            existing = atoms.get(nodes)
            if (
                existing is not None
                and frozenset((atom_sources[existing[0].token], arc.token)) in phase_pairs
            ):
                key = (existing[0].token, existing[1].node, existing[2].node)
                quotient[key] = (*quotient.get(key, (existing[0].token,)), arc.token)
                continue
            # A private atom keeps the original carrier and source edge, but uses
            # its own workset token to prevent recombining a removed overlap.
            atom = replace(
                arc,
                token=f"{arc.token}:atom:{float(first.parameter).hex()}:{float(last.parameter).hex()}",
            )
            retained.append(atom)
            retained_cuts[atom.token] = (first, last)
            atoms[nodes] = atom, first, last
            atom_sources[atom.token] = arc.token
    return FaceCellWorkset(
        tuple(retained), retained_cuts, aliases, quotient, tuple(unresolved), boxes
    )
