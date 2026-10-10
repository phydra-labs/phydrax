#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact oriented shell sewing from identity incidence or proved carrier maps.

Already identity-conforming coedges are assembled unchanged. Otherwise distinct
edges are identified only through an exact source-coefficient carrier
correspondence and exact parameter fragmentation; proximity, sampled points and
parameterization equality never identify entities, and a refusal publishes
nothing.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cmp_to_key
from typing import Literal, TypeAlias

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ._correspondence import (
    curve_support_key,
    CurveParameterMap,
    exact_parameter_affine,
    exact_parameter_parts,
    exact_parameter_sign,
    ExactParameter,
    prove_curve_correspondence,
    reparameterize_pcurve,
)
from ._intersection import (
    intersect_curve_ranges,
    NativePeriodEndpoint,
    ParametricIntersectionPolicy,
    RootEndpoint,
)
from ._intersection_curve import _native_curve_period_symbol, _period_offset, CurveRange
from ._model import BRepCurve, BRepGeometry, BRepPCurve
from ._patches import AbstractCurve, AbstractSurfacePatch


BRepSewingContactRelation: TypeAlias = Literal[
    "transversal",
    "tangent",
    "singular",
    "coincident-within-bound",
    "unresolved",
    "separated",
]

_OPEN_SHELL = "open, nonmanifold, or inconsistently oriented shell"


class BRepSewingPolicy(StrictModule):
    """Exact sewing budgets; never proximity repair.

    A nonzero tolerance is not permission to merge distinct scientific entities.
    Distinct coincident edges are identified only through a proved exact carrier
    parameter correspondence and exact fragmentation. ``maximum_contact_pairs``
    and ``contact_intersection`` bound the canonical intersection evidence
    attached to a refused open shell; that evidence never decides a pairing.
    """

    tolerance: float = eqx.field(static=True)
    maximum_coedges: int = eqx.field(static=True)
    maximum_contact_pairs: int = eqx.field(static=True)
    contact_intersection: ParametricIntersectionPolicy

    def __init__(
        self,
        *,
        tolerance: float = 0.0,
        maximum_coedges: int = 1_000_000,
        maximum_contact_pairs: int = 64,
        contact_intersection: ParametricIntersectionPolicy | None = None,
    ) -> None:
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and non-negative.")
        for name, value in (
            ("maximum_coedges", maximum_coedges),
            ("maximum_contact_pairs", maximum_contact_pairs),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
        if maximum_coedges <= 0:
            raise ValueError("maximum_coedges must be positive.")
        if maximum_contact_pairs < 0:
            raise ValueError("maximum_contact_pairs must be non-negative.")
        if tolerance != 0.0:
            raise ValueError("Native identity sewing does not admit proximity repair.")
        intersection = (
            ParametricIntersectionPolicy()
            if contact_intersection is None
            else contact_intersection
        )
        if not isinstance(intersection, ParametricIntersectionPolicy):
            raise TypeError(
                "contact_intersection must be a ParametricIntersectionPolicy."
            )
        self.tolerance = float(tolerance)
        self.maximum_coedges = maximum_coedges
        self.maximum_contact_pairs = maximum_contact_pairs
        self.contact_intersection = intersection


@dataclass(frozen=True, slots=True)
class BRepSewingContact:
    """Canonical intersection evidence for an open source edge and a candidate.

    ``relations`` lists the event classes reported by the native curve
    intersection owner (``separated`` when it reports none); ``gap_bound`` is
    the smallest reported operand gap or coincidence distance bound. This is
    diagnostic evidence only: exact carrier algebra refused the pairing.
    """

    open_edge: int
    candidate_edge: int
    relations: tuple[BRepSewingContactRelation, ...]
    gap_bound: float
    boxes_processed: int


class BRepSewingFailure(RuntimeError):
    """A shell could not be certified; the source geometry remains unchanged."""

    def __init__(
        self,
        reason: str,
        edges: tuple[int, ...] = (),
        *,
        contacts: tuple[BRepSewingContact, ...] = (),
        contacts_complete: bool = True,
    ) -> None:
        self.reason = reason
        self.edges = edges
        self.contacts = contacts
        self.contacts_complete = contacts_complete
        evidence = (
            ""
            if not contacts
            else "; contacts="
            + repr(
                tuple(
                    (contact.open_edge, contact.candidate_edge, contact.relations)
                    for contact in contacts
                )
            )
        )
        truncated = (
            "" if contacts_complete else "; contact diagnostics truncated by budget"
        )
        super().__init__(
            f"Native B-Rep sewing failed: {reason}; edges={edges}{evidence}{truncated}."
        )


@dataclass(frozen=True, slots=True)
class BRepSewingEdgeImage:
    """Exact source-edge parameter on one target edge: ``s = scale * t + offset``.

    ``offset = offset_rational + offset_turns * 2*pi`` exactly. The sign of
    ``scale`` is the source orientation relative to the authoritative target.
    """

    source_edge: int
    target_edge: int
    scale: Fraction
    offset_rational: Fraction
    offset_turns: Fraction

    @property
    def orientation(self) -> int:
        return 1 if self.scale > 0 else -1

    @property
    def parameter_map(self) -> CurveParameterMap:
        """``source(scale * t + offset) == target(t)``."""
        return CurveParameterMap(
            self.scale, _period_offset(self.offset_rational, self.offset_turns)
        )


@dataclass(frozen=True, slots=True)
class BRepSewingLineage:
    """Exhaustive many-source lineage of one sewing publication.

    Faces keep their indices. ``vertex_sources[v]`` lists every source vertex
    identified as target vertex ``v``; ``edge_images`` covers every fragment of
    every source edge (degenerate edges map identically); ``coedge_sources[c]``
    is the source coedge whose ordered fragment is target coedge ``c``.
    """

    vertex_sources: tuple[tuple[int, ...], ...]
    edge_images: tuple[BRepSewingEdgeImage, ...]
    coedge_sources: tuple[int, ...]
    lineage_id: str = field(init=False)

    def __post_init__(self) -> None:
        images = tuple(
            sorted(
                self.edge_images, key=lambda image: (image.source_edge, image.target_edge)
            )
        )
        object.__setattr__(self, "edge_images", images)
        object.__setattr__(
            self,
            "lineage_id",
            canonical_fingerprint(
                {
                    "kind": "brep-sewing-lineage",
                    "vertex_sources": [list(sources) for sources in self.vertex_sources],
                    "edge_images": [
                        [
                            image.source_edge,
                            image.target_edge,
                            str(image.scale),
                            str(image.offset_rational),
                            str(image.offset_turns),
                        ]
                        for image in images
                    ],
                    "coedge_sources": list(self.coedge_sources),
                }
            ),
        )

    def source_images(self, edge: int, /) -> tuple[BRepSewingEdgeImage, ...]:
        """Every target fragment carrying source ``edge``."""
        return tuple(image for image in self.edge_images if image.source_edge == edge)

    def target_sources(self, edge: int, /) -> tuple[BRepSewingEdgeImage, ...]:
        """Every source edge identified on target ``edge``."""
        return tuple(image for image in self.edge_images if image.target_edge == edge)


class BRepSewingResult(StrictModule):
    geometry: BRepGeometry
    certificate_id: str = eqx.field(static=True)
    lineage: BRepSewingLineage = eqx.field(static=True)

    def __init__(
        self, geometry: BRepGeometry, certificate_id: str, lineage: BRepSewingLineage
    ) -> None:
        if not isinstance(geometry, BRepGeometry):
            raise TypeError("geometry must be a BRepGeometry.")
        if not certificate_id:
            raise ValueError("certificate_id must be non-empty.")
        if not isinstance(lineage, BRepSewingLineage):
            raise TypeError("lineage must be a BRepSewingLineage.")
        self.geometry = geometry
        self.certificate_id = certificate_id
        self.lineage = lineage


# ------------------------------------------------------------ shell incidence


def _edge_uses(
    geometry: BRepGeometry, faces: tuple[int, ...], signs: np.ndarray
) -> dict[int, list[tuple[int, int]]]:
    uses: dict[int, list[tuple[int, int]]] = {}
    for face in faces:
        for loop in geometry.face_loops[face]:
            for coedge in loop:
                edge = geometry.coedge_edges[coedge]
                if geometry.edge_curves[edge] == -1:
                    continue
                uses.setdefault(edge, []).append(
                    (face, geometry.coedge_senses[coedge] * int(signs[face]))
                )
    return uses


def _invalid_edges(uses: dict[int, list[tuple[int, int]]], /) -> tuple[int, ...]:
    return tuple(
        edge
        for edge, values in sorted(uses.items())
        if len(values) != 2 or sum(sign for _, sign in values) != 0
    )


def _shells(
    geometry: BRepGeometry, faces: tuple[int, ...], signs: np.ndarray
) -> tuple[tuple[int, ...], ...]:
    uses = _edge_uses(geometry, faces, signs)
    invalid = _invalid_edges(uses)
    if invalid:
        raise BRepSewingFailure(_OPEN_SHELL, invalid)
    adjacent: dict[int, set[int]] = {face: set() for face in faces}
    for values in uses.values():
        first, second = values[0][0], values[1][0]
        adjacent[first].add(second)
        adjacent[second].add(first)
    remaining = set(faces)
    shells = []
    while remaining:
        seed = min(remaining)
        queue = deque((seed,))
        remaining.remove(seed)
        component = []
        while queue:
            face = queue.popleft()
            component.append(face)
            following = sorted(adjacent[face] & remaining)
            remaining.difference_update(following)
            queue.extend(following)
        shells.append(tuple(sorted(component)))
    return tuple(shells)


def _region_invalid_edges(
    geometry: BRepGeometry,
    regions: tuple[tuple[tuple[int, ...], np.ndarray], ...],
    /,
) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                edge
                for faces, outward in regions
                for edge in _invalid_edges(_edge_uses(geometry, faces, outward))
            }
        )
    )


# ---------------------------------------------------- exact parameter algebra


def _turns(count: int, /) -> ExactParameter:
    return _period_offset(Fraction(0), Fraction(count))


def _difference(first: ExactParameter, second: ExactParameter, /) -> ExactParameter:
    return exact_parameter_affine(second, Fraction(-1), first)


def _compare(first: ExactParameter, second: ExactParameter, /) -> int:
    sign = exact_parameter_sign(_difference(first, second))
    if sign is None:
        raise BRepSewingFailure("exact carrier parameter order is unresolved")
    return sign


_ORDER = cmp_to_key(_compare)


def _turn_floor(value: ExactParameter, /) -> int:
    rational, turns = exact_parameter_parts(value)
    count = math.floor(float(rational) / math.tau + float(turns))
    while _compare(value, _turns(count)) < 0:
        count -= 1
    while _compare(value, _turns(count + 1)) >= 0:
        count += 1
    return count


def _normalized(value: ExactParameter, periodic: bool, /) -> ExactParameter:
    return _difference(value, _turns(_turn_floor(value))) if periodic else value


# ------------------------------------------------------------ correspondence


@dataclass(frozen=True, slots=True)
class _Member:
    """One source edge on a common support, in reference coordinates."""

    edge: int
    frame: CurveParameterMap
    lower: ExactParameter
    upper: ExactParameter
    lower_vertex: int
    upper_vertex: int


@dataclass(frozen=True, slots=True)
class _Piece:
    """One fragment of a member between consecutive exact stations."""

    member: _Member
    position: int
    lower: ExactParameter
    upper: ExactParameter

    @property
    def address(self) -> tuple[int, int]:
        return self.member.edge, self.position


type _GroupKey = tuple[int, ExactParameter, ExactParameter]
type _Points = dict[ExactParameter, list[int]]


@dataclass(frozen=True, slots=True)
class _Coedge:
    pcurve: BRepPCurve
    sense: int
    roots: tuple[RootEndpoint | None, RootEndpoint | None]


@dataclass(frozen=True, slots=True)
class _TargetEdge:
    curve: int
    lower: float
    upper: float
    vertices: tuple[int, int]
    roots: tuple[RootEndpoint | None, RootEndpoint | None]
    closed: bool


@dataclass
class _Staging:
    """Host-only staging of one atomic correspondence publication."""

    geometry: BRepGeometry
    owned: dict[int, tuple[int, ...]]
    parents: dict[int, int] = field(default_factory=dict)
    curves: list[BRepCurve] = field(default_factory=list)
    curve_ids: dict[int, int] = field(default_factory=dict)
    edges: list[_TargetEdge] = field(default_factory=list)
    images: list[BRepSewingEdgeImage] = field(default_factory=list)
    verbatim: dict[int, int] = field(default_factory=dict)
    pieces: dict[int, tuple[_Piece, ...]] = field(default_factory=dict)
    piece_groups: dict[tuple[int, int], _GroupKey] = field(default_factory=dict)
    groups: dict[_GroupKey, list[_Piece]] = field(default_factory=dict)
    group_edges: dict[_GroupKey, int] = field(default_factory=dict)
    fragments: dict[tuple[_GroupKey, int], _Coedge] = field(default_factory=dict)
    merged: list[tuple[int, tuple[int, ...]]] = field(default_factory=list)

    def find(self, vertex: int, /) -> int:
        root = vertex
        while self.parents.get(root, root) != root:
            root = self.parents[root]
        while vertex != root:
            parent = self.parents[vertex]
            self.parents[vertex] = root
            vertex = parent
        return root

    def union(self, first: int, second: int, /) -> None:
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parents[max(a, b)] = min(a, b)

    def curve(self, source_curve: int, /) -> int:
        if source_curve not in self.curve_ids:
            self.curve_ids[source_curve] = len(self.curves)
            self.curves.append(self.geometry.curves[source_curve])
        return self.curve_ids[source_curve]

    def coedges(self, edge: int, /) -> tuple[int, ...]:
        return self.owned.get(edge, ())


def _exact_range(
    geometry: BRepGeometry, edge: int, coedges: tuple[int, ...], /
) -> tuple[ExactParameter, ExactParameter] | None:
    """Authored exact endpoints; isolated-root endpoints need an alias proof."""
    if any(
        root is not None and not isinstance(root, NativePeriodEndpoint)
        for coedge in coedges
        for root in geometry.coedge_endpoint_roots[coedge]
    ):
        return None
    values = np.asarray(geometry.edge_ranges)[edge]
    result: list[ExactParameter] = []
    for endpoint, root in enumerate(geometry.edge_endpoint_roots[edge]):
        if root is None:
            result.append(Fraction(float(values[endpoint])))
        elif isinstance(root, NativePeriodEndpoint):
            result.append(root.exact_parameter)
        else:
            return None
    return result[0], result[1]


def _member(
    staging: _Staging, reference: BRepCurve, edge: int, periodic: bool, /
) -> _Member:
    geometry = staging.geometry
    frame = prove_curve_correspondence(
        reference, geometry.curves[geometry.edge_curves[edge]]
    )
    if frame is None:
        raise BRepSewingFailure(
            "coincident carrier support has no exactly representable parameter correspondence",
            (edge,),
        )
    bounds = _exact_range(geometry, edge, staging.coedges(edge))
    if bounds is None:
        raise BRepSewingFailure(
            "root-valued edge endpoints require a source alias proof for identification",
            (edge,),
        )
    start, end = geometry.edge_vertices[edge]
    length = _difference(bounds[1], bounds[0])
    if _compare(length, Fraction(0)) <= 0 or (
        periodic and _compare(length, _turns(1)) > 0
    ):
        raise BRepSewingFailure(
            "edge range is not an exact positive carrier interval", (edge,)
        )
    if start == end and (not periodic or length != _turns(1)):
        raise BRepSewingFailure("closed edge lacks an exact native period", (edge,))
    images = frame.apply(bounds[0]), frame.apply(bounds[1])
    lower, upper = images if frame.scale > 0 else (images[1], images[0])
    first_vertex, last_vertex = (start, end) if frame.scale > 0 else (end, start)
    if periodic:
        shift = _turns(_turn_floor(lower))
        frame = CurveParameterMap(frame.scale, _difference(frame.offset, shift))
        lower, upper = _difference(lower, shift), _difference(upper, shift)
    return _Member(edge, frame, lower, upper, first_vertex, last_vertex)


def _bucket_pieces(
    members: tuple[_Member, ...], periodic: bool, /
) -> tuple[dict[int, tuple[_Piece, ...]], _Points]:
    """Cut every member at every interior member endpoint of its support."""
    points: _Points = {}
    for member in members:
        points.setdefault(_normalized(member.lower, periodic), []).append(
            member.lower_vertex
        )
        points.setdefault(_normalized(member.upper, periodic), []).append(
            member.upper_vertex
        )
    ordered = sorted(points, key=_ORDER)
    stations = ordered + (
        [exact_parameter_affine(value, Fraction(1), _turns(1)) for value in ordered]
        if periodic
        else []
    )
    keys = [_ORDER(value) for value in stations]
    pieces: dict[int, tuple[_Piece, ...]] = {}
    for member in members:
        inner = stations[
            bisect_right(keys, _ORDER(member.lower)) : bisect_left(
                keys, _ORDER(member.upper)
            )
        ]
        cuts = (member.lower, *inner, member.upper)
        pieces[member.edge] = tuple(
            _Piece(member, position, a, b)
            for position, (a, b) in enumerate(zip(cuts[:-1], cuts[1:], strict=True))
        )
    return pieces, points


def _authority(
    roots: tuple[RootEndpoint | None, ...], /
) -> tuple[AbstractSurfacePatch | None, Literal[0, 1] | None] | None:
    native = next(
        (root for root in roots if isinstance(root, NativePeriodEndpoint)), None
    )
    return None if native is None else (native.patch, native.axis)


def _endpoint(
    value: ExactParameter,
    carrier: BRepCurve | BRepPCurve,
    periodic: bool,
    authority: tuple[AbstractSurfacePatch | None, Literal[0, 1] | None] | None,
    edges: tuple[int, ...],
    /,
) -> tuple[float, NativePeriodEndpoint | None]:
    """Literal binary endpoint when exact, else an authored native-period scalar."""
    rational, turns = exact_parameter_parts(value)
    if not periodic or (not turns and authority is None):
        number = float(rational)
        if turns or Fraction(number) != rational:
            raise BRepSewingFailure(
                "fragment endpoint is not exactly representable on its carrier", edges
            )
        return number, None
    patch, axis = (None, None) if authority is None else authority
    root = NativePeriodEndpoint(carrier, patch, axis, rational=rational, turns=turns)
    lower, upper = root.parameter_enclosure()
    return (
        root.parameter if lower <= root.parameter <= upper else 0.5 * (lower + upper)
    ), root


def _transport(
    staging: _Staging,
    carrier: _Piece,
    piece: _Piece,
    target: tuple[ExactParameter, ExactParameter],
    periodic: bool,
    /,
) -> tuple[CurveParameterMap, dict[int, _Coedge]] | None:
    """Exact target-to-source map and transported p-curves of one covering member."""
    geometry = staging.geometry
    source, own = carrier.member.frame, piece.member.frame
    shift = _difference(piece.lower, carrier.lower)
    offset = exact_parameter_affine(
        _difference(
            exact_parameter_affine(source.offset, Fraction(1), shift), own.offset
        ),
        1 / own.scale,
        Fraction(0),
    )
    mapping = CurveParameterMap(source.scale / own.scale, offset)
    result: dict[int, _Coedge] = {}
    for coedge in staging.coedges(piece.member.edge):
        pcurve = reparameterize_pcurve(geometry.pcurves[coedge], mapping)
        if pcurve is None:
            return None
        authority = _authority(geometry.coedge_endpoint_roots[coedge])
        roots: tuple[RootEndpoint | None, RootEndpoint | None] = (None, None)
        if authority is not None:
            edges = (piece.member.edge,)
            roots = (
                _endpoint(target[0], pcurve, periodic, authority, edges)[1],
                _endpoint(target[1], pcurve, periodic, authority, edges)[1],
            )
        result[coedge] = _Coedge(
            pcurve,
            geometry.coedge_senses[coedge] * (1 if mapping.scale > 0 else -1),
            roots,
        )
    return mapping, result


def _select_carrier(
    staging: _Staging,
    covering: tuple[_Piece, ...],
    periodic: bool,
    /,
) -> tuple[
    _Piece,
    tuple[ExactParameter, ExactParameter],
    tuple[tuple[CurveParameterMap, dict[int, _Coedge]], ...],
]:
    """First source carrier, in edge order, transporting every incident p-curve exactly."""
    for carrier in covering:
        frame = carrier.member.frame
        inverse = frame.inverse()
        ends = inverse.apply(carrier.lower), inverse.apply(carrier.upper)
        target = ends if frame.scale > 0 else (ends[1], ends[0])
        plans = tuple(
            _transport(staging, carrier, piece, target, periodic) for piece in covering
        )
        complete = tuple(plan for plan in plans if plan is not None)
        if len(complete) == len(plans):
            return carrier, target, complete
    raise BRepSewingFailure(
        "coincident edges have no exactly representable common parameterization",
        tuple(piece.member.edge for piece in covering),
    )


def _build_group(
    staging: _Staging, key: _GroupKey, points: _Points, periodic: bool, /
) -> None:
    """Publish one authoritative target edge from the first exactly transportable carrier."""
    geometry = staging.geometry
    covering = tuple(sorted(staging.groups[key], key=lambda value: value.member.edge))
    sources = tuple(piece.member.edge for piece in covering)
    carrier, target, plans = _select_carrier(staging, covering, periodic)
    frame = carrier.member.frame
    source_curve = geometry.edge_curves[carrier.member.edge]
    curve = geometry.curves[source_curve]
    authority = _authority(geometry.edge_endpoint_roots[carrier.member.edge])
    if (
        periodic
        and authority is None
        and any(exact_parameter_parts(value)[1] for value in target)
    ):
        authority = (None, None)
    lower, lower_root = _endpoint(target[0], curve, periodic, authority, sources)
    upper, upper_root = _endpoint(target[1], curve, periodic, authority, sources)
    member = carrier.member

    def station_vertex(station: ExactParameter, /) -> int:
        # A member's own endpoint keeps its source vertex; an interior cut is
        # the endpoint of a shared fragment, whose station vertices are unioned.
        if station == member.lower:
            return member.lower_vertex
        if station == member.upper:
            return member.upper_vertex
        return points[_normalized(station, periodic)][0]

    first, last = (
        (carrier.lower, carrier.upper)
        if frame.scale > 0
        else (carrier.upper, carrier.lower)
    )
    index = len(staging.edges)
    staging.edges.append(
        _TargetEdge(
            staging.curve(source_curve),
            lower,
            upper,
            (station_vertex(first), station_vertex(last)),
            (lower_root, upper_root),
            periodic and _difference(carrier.upper, carrier.lower) == _turns(1),
        )
    )
    staging.group_edges[key] = index
    for piece, (mapping, coedges) in zip(covering, plans, strict=True):
        rational, turns = exact_parameter_parts(mapping.offset)
        staging.images.append(
            BRepSewingEdgeImage(piece.member.edge, index, mapping.scale, rational, turns)
        )
        for coedge, transported in coedges.items():
            staging.fragments[(key, coedge)] = transported
    if len(covering) > 1:
        staging.merged.append((index, sources))


def _verbatim(staging: _Staging, edge: int, /) -> None:
    geometry = staging.geometry
    source_curve = geometry.edge_curves[edge]
    first, last = (float(value) for value in np.asarray(geometry.edge_ranges)[edge])
    start, end = geometry.edge_vertices[edge]
    index = len(staging.edges)
    staging.edges.append(
        _TargetEdge(
            -1 if source_curve == -1 else staging.curve(source_curve),
            first,
            last,
            (start, end),
            geometry.edge_endpoint_roots[edge],
            start == end,
        )
    )
    staging.verbatim[edge] = index
    staging.images.append(
        BRepSewingEdgeImage(edge, index, Fraction(1), Fraction(0), Fraction(0))
    )


def _target_vertices(
    staging: _Staging, /
) -> tuple[dict[int, int], tuple[tuple[int, ...], ...]]:
    geometry = staging.geometry
    classes: dict[int, list[int]] = {}
    for vertex in range(geometry.vertex_points.shape[0]):
        classes.setdefault(staging.find(vertex), []).append(vertex)
    ordered = sorted(classes.values(), key=min)
    for members in ordered:
        if len(members) > 1 and any(
            geometry.vertex_roots[vertex] is not None for vertex in members
        ):
            raise BRepSewingFailure(
                "root-valued vertex identification requires a source alias proof",
                tuple(
                    edge
                    for edge, ends in enumerate(geometry.edge_vertices)
                    if set(ends) & set(members)
                ),
            )
    target = {
        vertex: index for index, members in enumerate(ordered) for vertex in members
    }
    return target, tuple(tuple(members) for members in ordered)


def _stage_buckets(staging: _Staging, /) -> dict[_GroupKey, tuple[_Points, bool]]:
    """Exact members, fragments and fragment groups of every shared support."""
    geometry = staging.geometry
    buckets: dict[tuple[object, ...], list[int]] = {}
    for edge, curve in enumerate(geometry.edge_curves):
        if curve != -1:
            buckets.setdefault(curve_support_key(geometry.curves[curve]), []).append(edge)
    contexts: dict[_GroupKey, tuple[_Points, bool]] = {}
    for bucket, edges in enumerate(edges for edges in buckets.values() if len(edges) > 1):
        reference = geometry.curves[geometry.edge_curves[edges[0]]]
        periodic = (
            isinstance(reference, AbstractCurve)
            and _native_curve_period_symbol(reference) == "two_pi"
        )
        members = tuple(_member(staging, reference, edge, periodic) for edge in edges)
        pieces, points = _bucket_pieces(members, periodic)
        staging.pieces.update(pieces)
        for member_pieces in pieces.values():
            for piece in member_pieces:
                key = (
                    bucket,
                    _normalized(piece.lower, periodic),
                    _difference(piece.upper, piece.lower),
                )
                staging.groups.setdefault(key, []).append(piece)
                staging.piece_groups[piece.address] = key
                contexts[key] = (points, periodic)
    for key, values in staging.groups.items():
        if len(values) > 1:
            # Only a shared fragment identifies the source vertices at its ends.
            points, periodic = contexts[key]
            for station in (values[0].lower, values[0].upper):
                vertices = points[_normalized(station, periodic)]
                for vertex in vertices[1:]:
                    staging.union(vertices[0], vertex)
    return contexts


def _sew_by_correspondence(
    geometry: BRepGeometry,
    policy: BRepSewingPolicy,
    /,
) -> tuple[BRepGeometry, BRepSewingLineage, _Staging] | None:
    """Fragment and identify exactly corresponding edges; None when none exist."""
    owned: dict[int, list[int]] = {}
    for coedge, edge in enumerate(geometry.coedge_edges):
        owned.setdefault(edge, []).append(coedge)
    staging = _Staging(geometry, {edge: tuple(values) for edge, values in owned.items()})
    contexts = _stage_buckets(staging)
    if all(
        len(values) == 1 and len(staging.pieces[values[0].member.edge]) == 1
        for values in staging.groups.values()
    ):
        return None
    for edge in range(len(geometry.edge_curves)):
        member_pieces = staging.pieces.get(edge)
        if member_pieces is None or (
            len(member_pieces) == 1
            and len(staging.groups[staging.piece_groups[member_pieces[0].address]]) == 1
        ):
            _verbatim(staging, edge)
            continue
        for piece in member_pieces:
            key = staging.piece_groups[piece.address]
            if key not in staging.group_edges:
                _build_group(staging, key, *contexts[key])
    target, lineage = _publish_correspondence(staging, policy)
    return target, lineage, staging


def _target_uses(staging: _Staging, coedge: int, /) -> list[tuple[int, _Coedge]]:
    """Ordered target fragments of one source coedge in its traversal sense."""
    geometry = staging.geometry
    edge, sense = geometry.coedge_edges[coedge], geometry.coedge_senses[coedge]
    if edge in staging.verbatim:
        return [
            (
                staging.verbatim[edge],
                _Coedge(
                    geometry.pcurves[coedge],
                    sense,
                    geometry.coedge_endpoint_roots[coedge],
                ),
            )
        ]
    member_pieces = staging.pieces[edge]
    ascending = (member_pieces[0].member.frame.scale > 0) == (sense > 0)
    keys = [
        staging.piece_groups[piece.address]
        for piece in (member_pieces if ascending else reversed(member_pieces))
    ]
    return [(staging.group_edges[key], staging.fragments[(key, coedge)]) for key in keys]


def _publish_correspondence(
    staging: _Staging, policy: BRepSewingPolicy, /
) -> tuple[BRepGeometry, BRepSewingLineage]:
    geometry = staging.geometry
    vertex_ids, vertex_sources = _target_vertices(staging)
    uses: list[tuple[int, _Coedge]] = []
    coedge_sources: list[int] = []
    face_loops = []
    for loops in geometry.face_loops:
        rebuilt = []
        for loop in loops:
            start = len(uses)
            for coedge in loop:
                fragments = _target_uses(staging, coedge)
                uses.extend(fragments)
                coedge_sources.extend([coedge] * len(fragments))
            rebuilt.append(tuple(range(start, len(uses))))
        face_loops.append(tuple(rebuilt))
    if len(uses) > policy.maximum_coedges:
        raise BRepSewingFailure("coedge budget exhausted")
    edge_vertices = []
    for index, edge in enumerate(staging.edges):
        ends = (vertex_ids[edge.vertices[0]], vertex_ids[edge.vertices[1]])
        if ends[0] == ends[1] and not edge.closed:
            raise BRepSewingFailure(
                "exact vertex identification collapses an edge",
                tuple(
                    image.source_edge
                    for image in staging.images
                    if image.target_edge == index
                ),
            )
        edge_vertices.append(ends)
    points = np.asarray(geometry.vertex_points)
    target = BRepGeometry(
        vertex_points=np.asarray(
            [points[sources[0]] for sources in vertex_sources], dtype=np.float64
        ).reshape((-1, 3)),
        curves=tuple(staging.curves),
        edge_curves=tuple(edge.curve for edge in staging.edges),
        edge_ranges=np.asarray(
            [(edge.lower, edge.upper) for edge in staging.edges], dtype=np.float64
        ).reshape((-1, 2)),
        edge_vertices=tuple(edge_vertices),
        pcurves=tuple(use.pcurve for _, use in uses),
        coedge_edges=tuple(edge for edge, _ in uses),
        coedge_senses=tuple(use.sense for _, use in uses),
        face_loops=tuple(face_loops),
        shell_faces=(),
        shell_orientations=(),
        solid_shells=(),
        vertex_roots=tuple(
            geometry.vertex_roots[sources[0]] for sources in vertex_sources
        ),
        edge_endpoint_roots=tuple(edge.roots for edge in staging.edges),
        coedge_endpoint_roots=tuple(use.roots for _, use in uses),
    )
    lineage = BRepSewingLineage(
        vertex_sources, tuple(staging.images), tuple(coedge_sources)
    )
    _require_exhaustive(geometry, lineage)
    return target, lineage


def _require_exhaustive(geometry: BRepGeometry, lineage: BRepSewingLineage, /) -> None:
    vertices = sorted(vertex for sources in lineage.vertex_sources for vertex in sources)
    if vertices != list(range(geometry.vertex_points.shape[0])):
        raise RuntimeError("Sewing lineage lost or repeated a source vertex.")
    if {image.source_edge for image in lineage.edge_images} != set(
        range(len(geometry.edge_curves))
    ):
        raise RuntimeError("Sewing lineage lost a source edge.")
    if set(lineage.coedge_sources) != set(range(len(geometry.coedge_edges))):
        raise RuntimeError("Sewing lineage lost a source coedge.")


def _require_region_connectivity(
    target: BRepGeometry,
    merged: tuple[tuple[int, tuple[int, ...]], ...],
    solid_faces: tuple[tuple[int, ...], ...],
    /,
) -> None:
    """An identified edge may join regions only through a declared shared face."""
    regions: dict[int, set[int]] = {}
    for region, faces in enumerate(solid_faces):
        for face in faces:
            regions.setdefault(face, set()).add(region)
    owners = target.coedge_faces
    users: dict[int, set[int]] = {}
    for coedge, edge in enumerate(target.coedge_edges):
        users.setdefault(edge, set()).add(owners[coedge])
    for edge, sources in merged:
        faces = sorted(users.get(edge, set()))
        if len(faces) < 2:
            continue
        reached, queue = {faces[0]}, deque((faces[0],))
        while queue:
            face = queue.popleft()
            for other in faces:
                if other not in reached and regions[face] & regions[other]:
                    reached.add(other)
                    queue.append(other)
        if len(reached) != len(faces):
            raise BRepSewingFailure(
                "coincident edges of distinct regions require a declared shared face",
                sources,
            )


# ------------------------------------------------------------- diagnostics


def _contacts(
    geometry: BRepGeometry,
    edges: tuple[int, ...],
    vertex_class: dict[int, int],
    policy: BRepSewingPolicy,
    /,
) -> tuple[tuple[BRepSewingContact, ...], bool]:
    """Canonical intersection evidence among open edges without a common vertex.

    Candidates are prefiltered by enclosures inflated by the intersection
    owner's relative tangency scale, so near-coincident and tangent carriers
    reach the canonical curve intersection rather than a local proximity test.
    """
    candidates = tuple(edge for edge in edges if geometry.edge_curves[edge] != -1)
    ranges = np.asarray(geometry.edge_ranges)
    boxes = {
        edge: np.asarray(
            geometry.curves[geometry.edge_curves[edge]].bounding_box(
                float(ranges[edge, 0]), float(ranges[edge, 1])
            )
        )
        for edge in candidates
    }
    contacts: list[BRepSewingContact] = []
    for position, first in enumerate(candidates):
        for second in candidates[position + 1 :]:
            if {vertex_class[v] for v in geometry.edge_vertices[first]} & {
                vertex_class[v] for v in geometry.edge_vertices[second]
            }:
                continue
            margin = policy.contact_intersection.relative_tangency * float(
                np.linalg.norm(boxes[first][1] - boxes[first][0])
                + np.linalg.norm(boxes[second][1] - boxes[second][0])
            )
            if np.any(boxes[first][0] - margin > boxes[second][1]) or np.any(
                boxes[second][0] - margin > boxes[first][1]
            ):
                continue
            if len(contacts) == policy.maximum_contact_pairs:
                return tuple(contacts), False
            contacts.append(_contact(geometry, first, second, policy))
    return tuple(contacts), True


def _contact(
    geometry: BRepGeometry, first: int, second: int, policy: BRepSewingPolicy, /
) -> BRepSewingContact:
    ranges = np.asarray(geometry.edge_ranges)
    operands = tuple(
        CurveRange(
            geometry.curves[geometry.edge_curves[edge]],
            float(ranges[edge, 0]),
            float(ranges[edge, 1]),
        )
        for edge in (first, second)
    )
    result = intersect_curve_ranges(*operands, policy=policy.contact_intersection)
    relations: set[BRepSewingContactRelation] = set()
    relations.update(point.kind for point in result.points)
    if result.coincident:
        relations.add("coincident-within-bound")
    if result.unresolved or result.work.budget_exhausted:
        relations.add("unresolved")
    gaps = [point.gap_bound for point in result.points] + [
        region.distance_bound for region in result.coincident
    ]
    return BRepSewingContact(
        first,
        second,
        tuple(sorted(relations)) or ("separated",),
        min(gaps, default=math.inf),
        result.work.boxes_processed,
    )


def _open_failure(
    geometry: BRepGeometry,
    reason: str,
    edges: tuple[int, ...],
    vertex_class: dict[int, int],
    policy: BRepSewingPolicy,
    /,
) -> BRepSewingFailure:
    contacts, complete = _contacts(geometry, edges, vertex_class, policy)
    return BRepSewingFailure(reason, edges, contacts=contacts, contacts_complete=complete)


# ------------------------------------------------------------- publication


def _identity_lineage(geometry: BRepGeometry, /) -> BRepSewingLineage:
    return BRepSewingLineage(
        tuple((vertex,) for vertex in range(geometry.vertex_points.shape[0])),
        tuple(
            BRepSewingEdgeImage(edge, edge, Fraction(1), Fraction(0), Fraction(0))
            for edge in range(len(geometry.edge_curves))
        ),
        tuple(range(len(geometry.coedge_edges))),
    )


def sew_brep(
    geometry: BRepGeometry,
    orientation: ArrayLike,
    solid_faces: tuple[tuple[int, ...], ...],
    /,
    *,
    policy: BRepSewingPolicy | None = None,
    solid_orientations: tuple[tuple[int, ...], ...] | None = None,
) -> BRepSewingResult:
    """Assemble closed shells from authoritative or exactly proved edge incidence.

    ``solid_faces`` declares region ownership, including cavity shell faces.
    Sewing neither classifies regions nor guesses cavity ownership from location.
    Each supplied face orientation is its outward normal for that region.

    Identity-conforming coedges are assembled unchanged. Otherwise every pair of
    distinct edges whose carriers have an exact source-coefficient parameter
    correspondence is fragmented at every member endpoint and published as one
    authoritative target edge per common fragment; the result lineage records
    every source vertex/edge/coedge image with its exact orientation and
    parameter map. Unproved, tangent, one-ulp or cross-region contacts refuse
    atomically with native intersection evidence.
    """
    if not isinstance(geometry, BRepGeometry):
        raise TypeError("geometry must be a BRepGeometry.")
    policy_ = BRepSewingPolicy() if policy is None else policy
    if not isinstance(policy_, BRepSewingPolicy):
        raise TypeError("policy must be a BRepSewingPolicy.")
    if len(geometry.coedge_edges) > policy_.maximum_coedges:
        raise BRepSewingFailure("coedge budget exhausted")
    signs = np.asarray(orientation, dtype=np.float64)
    if signs.shape != (len(geometry.face_loops),) or not np.all(
        np.isin(signs, (-1.0, 1.0))
    ):
        raise ValueError("orientation must have one -1 or +1 per face.")
    owned = [face for faces in solid_faces for face in faces]
    if set(owned) != set(range(len(geometry.face_loops))):
        raise ValueError("solid_faces must exhaust the geometry faces.")
    if any(len(faces) != len(set(faces)) for faces in solid_faces):
        raise ValueError("A solid cannot repeat a face.")
    uses = {face: owned.count(face) for face in set(owned)}
    if any(count > 2 for count in uses.values()):
        raise ValueError("A face may belong to at most two regions.")
    region_signs = (
        tuple(tuple(int(signs[face]) for face in faces) for faces in solid_faces)
        if solid_orientations is None
        else solid_orientations
    )
    if len(region_signs) != len(solid_faces) or any(
        len(values) != len(faces) or any(value not in (-1, 1) for value in values)
        for faces, values in zip(solid_faces, region_signs, strict=True)
    ):
        raise ValueError("solid_orientations must align with faces and contain signs.")
    regions = []
    for faces, values in zip(solid_faces, region_signs, strict=True):
        if not faces:
            raise ValueError("A solid must own at least one face.")
        outward = signs.copy()
        for face, value in zip(faces, values, strict=True):
            outward[face] = value
        regions.append((tuple(faces), outward))
    regions_ = tuple(regions)
    invalid = _region_invalid_edges(geometry, regions_)
    if not invalid:
        return _assemble(
            geometry,
            geometry,
            regions_,
            signs,
            solid_faces,
            policy_,
            _identity_lineage(geometry),
        )
    sewn = _sew_by_correspondence(geometry, policy_)
    if sewn is None:
        identity = {vertex: vertex for vertex in range(geometry.vertex_points.shape[0])}
        raise _open_failure(geometry, _OPEN_SHELL, invalid, identity, policy_)
    target, lineage, staging = sewn
    _require_region_connectivity(target, tuple(staging.merged), solid_faces)
    open_target = set(_region_invalid_edges(target, regions_))
    if open_target:
        sources = tuple(
            sorted(
                {
                    image.source_edge
                    for image in lineage.edge_images
                    if image.target_edge in open_target
                }
            )
        )
        classes = {
            vertex: staging.find(vertex)
            for vertex in range(geometry.vertex_points.shape[0])
        }
        raise _open_failure(
            geometry,
            f"{_OPEN_SHELL} after exact correspondence",
            sources,
            classes,
            policy_,
        )
    return _assemble(geometry, target, regions_, signs, solid_faces, policy_, lineage)


def _assemble(
    source: BRepGeometry,
    geometry: BRepGeometry,
    regions: tuple[tuple[tuple[int, ...], np.ndarray], ...],
    signs: np.ndarray,
    solid_faces: tuple[tuple[int, ...], ...],
    policy: BRepSewingPolicy,
    lineage: BRepSewingLineage,
    /,
) -> BRepSewingResult:
    shell_faces = []
    shell_orientations = []
    solid_shells = []
    for faces, outward in regions:
        components = _shells(geometry, faces, outward)
        solid_shells.append(
            tuple(range(len(shell_faces), len(shell_faces) + len(components)))
        )
        shell_faces.extend(components)
        shell_orientations.extend(
            tuple(int(outward[face] * signs[face]) for face in component)
            for component in components
        )
    inherited = source.topology().solid_faces == solid_faces
    if not inherited and (source.occurrences or source.assembly_containers):
        raise ValueError(
            "Regrouping existing solids requires an explicit occurrence remap."
        )
    assembled = BRepGeometry(
        vertex_points=geometry.vertex_points,
        curves=geometry.curves,
        edge_curves=geometry.edge_curves,
        edge_ranges=geometry.edge_ranges,
        edge_vertices=geometry.edge_vertices,
        pcurves=geometry.pcurves,
        coedge_edges=geometry.coedge_edges,
        coedge_senses=geometry.coedge_senses,
        face_loops=geometry.face_loops,
        shell_faces=tuple(shell_faces),
        shell_orientations=tuple(shell_orientations),
        solid_shells=tuple(solid_shells),
        vertex_roots=geometry.vertex_roots,
        edge_endpoint_roots=geometry.edge_endpoint_roots,
        coedge_endpoint_roots=geometry.coedge_endpoint_roots,
        occurrences=source.occurrences if inherited else None,
        assembly_containers=source.assembly_containers if inherited else None,
    )
    payload: dict[str, object] = {
        "kind": "native-oriented-sewing",
        "source_geometry": source.geometry_id,
        "target_geometry": assembled.geometry_id,
        "orientation": signs,
        "tolerance": policy.tolerance,
    }
    if geometry is not source:
        payload |= {
            "kind": "native-exact-correspondence-sewing",
            "lineage": lineage.lineage_id,
        }
    return BRepSewingResult(assembled, canonical_fingerprint(payload), lineage)


__all__ = [
    "BRepSewingContact",
    "BRepSewingContactRelation",
    "BRepSewingEdgeImage",
    "BRepSewingFailure",
    "BRepSewingLineage",
    "BRepSewingPolicy",
    "BRepSewingResult",
    "sew_brep",
]
