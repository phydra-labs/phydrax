#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact native sewing of source-authoritative B-Rep shells.

Distinct source edges are joined only when their analytic carrier coefficients
prove one parameter map. No coordinate proximity or sampled evaluation
establishes topology.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cmp_to_key
from typing import Literal, TypeAlias, final

import equinox as eqx
import jax
import numpy as np
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...typing import checked
from ._intersection import (
    ConicParameterEndpoint,
    CurveRange,
    NativePeriodEndpoint,
    ParametricIntersectionPolicy,
    RootEndpoint,
    endpoint_root_payload,
    intersect_curve_ranges,
)
from ._intersection_curve import CurveTrimSegment
from ._root_bindings import BRepVertexRoot
from ._model import BRepGeometry
from ._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    BSplineCurve,
    CircleCurve,
    ExactLineCurve,
    LineCurve,
)



BRepSewingContactRelation: TypeAlias = Literal[
    "crossing",
    "tangent_point",
    "tangent_curve",
    "singular_point",
    "coincident-within-bound",
    "unresolved",
    "separated",
]

_OPEN_SHELL = "open, nonmanifold, or inconsistently oriented shell"


class BRepSewingPolicy(StrictModule):
    """Exact sewing and source-intersection diagnostic numerical schedule."""

    tolerance: float = eqx.field(static=True)
    contact_intersection: ParametricIntersectionPolicy

    @checked
    def __init__(
        self,
        *,
        tolerance: float = 0.0,
        contact_intersection: ParametricIntersectionPolicy | None = None,
    ) -> None:
        if isinstance(tolerance, bool) or not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("BRepSewingPolicy.tolerance must be finite and nonnegative.")
        if tolerance != 0.0:
            raise ValueError("BRepSewingPolicy.tolerance must forbid proximity repair.")
        intersection = (
            ParametricIntersectionPolicy()
            if contact_intersection is None
            else contact_intersection
        )
        if not isinstance(intersection, ParametricIntersectionPolicy):
            raise TypeError("contact_intersection must be a ParametricIntersectionPolicy.")
        self.tolerance = 0.0
        self.contact_intersection = intersection


@dataclass(frozen=True, slots=True)
class BRepSewingContact:
    """Exact native intersection diagnostics for an open edge pair."""

    open_edge: int
    candidate_edge: int
    relations: tuple[BRepSewingContactRelation, ...]
    gap_bound: float


class BRepSewingFailure(RuntimeError):
    """An exact shell sewing could not be established; source geometry is unchanged."""

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
            + repr(tuple((value.open_edge, value.candidate_edge, value.relations) for value in contacts))
        )
        incomplete = "" if contacts_complete else "; contact diagnostics incomplete"
        super().__init__(f"Native B-Rep sewing failed: {reason}; edges={edges}{evidence}{incomplete}.")


@dataclass(frozen=True, slots=True)
class BRepSewingEdgeImage:
    """Map a target-edge parameter to a source parameter exactly."""

    source_edge: int
    target_edge: int
    scale: Fraction
    offset_rational: Fraction
    offset_turns: Fraction

    @property
    def orientation(self) -> int:
        return 1 if self.scale > 0 else -1

    @property
    def parameter_map(self) -> _ParameterMap:
        """Exact target-to-source affine map, including native-period turns."""
        return _ParameterMap(
            self.scale,
            _Parameter(self.offset_rational, self.offset_turns),
        )


@final
class BRepSewingLineage(StrictModule):
    """Exhaustive source-to-target maps for vertices, edge fragments and coedges."""

    vertex_sources: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    edge_images: tuple[BRepSewingEdgeImage, ...] = eqx.field(static=True)
    coedge_sources: tuple[int, ...] = eqx.field(static=True)
    lineage_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertex_sources: tuple[tuple[int, ...], ...],
        edge_images: tuple[BRepSewingEdgeImage, ...],
        coedge_sources: tuple[int, ...],
    ) -> None:
        images = tuple(sorted(edge_images, key=lambda image: (image.source_edge, image.target_edge)))
        if any(image.source_edge < 0 or image.target_edge < 0 or image.scale == 0 for image in images):
            raise ValueError("BRepSewingLineage.edge_images contains an invalid entity map.")
        self.vertex_sources = tuple(tuple(values) for values in vertex_sources)
        self.edge_images = images
        self.coedge_sources = tuple(coedge_sources)
        self.lineage_id = canonical_fingerprint({
            "kind": "brep-sewing-lineage",
            "vertex_sources": self.vertex_sources,
            "edge_images": tuple(
                (
                    image.source_edge,
                    image.target_edge,
                    str(image.scale),
                    str(image.offset_rational),
                    str(image.offset_turns),
                )
                for image in images
            ),
            "coedge_sources": self.coedge_sources,
        })

    def source_images(self, edge: int, /) -> tuple[BRepSewingEdgeImage, ...]:
        return tuple(image for image in self.edge_images if image.source_edge == edge)

    def target_sources(self, edge: int, /) -> tuple[BRepSewingEdgeImage, ...]:
        return tuple(image for image in self.edge_images if image.target_edge == edge)


class BRepSewingResult(StrictModule):
    geometry: BRepGeometry
    certificate_id: str = eqx.field(static=True)
    lineage: BRepSewingLineage = eqx.field(static=True)

    @checked
    def __init__(self, geometry: BRepGeometry, certificate_id: str, lineage: BRepSewingLineage) -> None:
        if not certificate_id:
            raise ValueError("BRepSewingResult.certificate_id must be nonempty.")
        self.geometry = geometry
        self.certificate_id = certificate_id
        self.lineage = lineage


@dataclass(frozen=True, slots=True)
class _Parameter:
    rational: Fraction
    turns: Fraction = Fraction(0)


def _parameter(value: object, /) -> _Parameter:
    if isinstance(value, Fraction):
        return _Parameter(value)
    rational, turns = getattr(value, "rational", None), getattr(value, "turns", None)
    if isinstance(rational, Fraction) and isinstance(turns, Fraction):
        return _Parameter(rational, turns)
    raise TypeError(f"Unsupported exact sewing parameter {value!r}.")


def _affine(value: _Parameter, scale: Fraction, offset: _Parameter, /) -> _Parameter:
    return _Parameter(value.rational * scale + offset.rational, value.turns * scale + offset.turns)


def _add(first: _Parameter, second: _Parameter, /) -> _Parameter:
    return _Parameter(first.rational + second.rational, first.turns + second.turns)


def _difference(first: _Parameter, second: _Parameter, /) -> _Parameter:
    return _Parameter(first.rational - second.rational, first.turns - second.turns)


def _turns(count: int | Fraction, /) -> _Parameter:
    return _Parameter(Fraction(0), Fraction(count))


def _period_bounds(value: _Parameter, /) -> tuple[Fraction, Fraction]:
    if not value.turns:
        return value.rational, value.rational
    lower_tau = Fraction.from_float(math.nextafter(math.tau, -math.inf))
    upper_tau = Fraction.from_float(math.nextafter(math.tau, math.inf))
    if value.turns > 0:
        return value.rational + value.turns * lower_tau, value.rational + value.turns * upper_tau
    return value.rational + value.turns * upper_tau, value.rational + value.turns * lower_tau


def _compare(first: _Parameter, second: _Parameter, /) -> int:
    delta = _difference(first, second)
    if not delta.turns:
        return (delta.rational > 0) - (delta.rational < 0)
    lower, upper = _period_bounds(delta)
    if lower > 0:
        return 1
    if upper < 0:
        return -1
    raise BRepSewingFailure("sew_brep.parameter_order is numerically unresolved at binary64 resolution")


_ORDER = cmp_to_key(_compare)


def _turn_floor(value: _Parameter, /) -> int:
    try:
        estimate = math.floor(float(value.rational) / math.tau + float(value.turns))
    except (OverflowError, ValueError) as error:
        raise BRepSewingFailure("sew_brep.period_station is outside finite numerical range") from error
    while _compare(value, _turns(estimate)) < 0:
        estimate -= 1
    while _compare(value, _turns(estimate + 1)) >= 0:
        estimate += 1
    return estimate


def _normalized(value: _Parameter, periodic: bool, /) -> _Parameter:
    return _difference(value, _turns(_turn_floor(value))) if periodic else value


@dataclass(frozen=True, slots=True)
class _ParameterMap:
    scale: Fraction
    offset: _Parameter

    def apply(self, value: _Parameter, /) -> _Parameter:
        return _affine(value, self.scale, self.offset)

    def inverse(self) -> _ParameterMap:
        return _ParameterMap(1 / self.scale, _affine(self.offset, -1 / self.scale, _Parameter(Fraction(0))))


def _rational_vector(values: ArrayLike, /) -> tuple[Fraction, ...]:
    return tuple(Fraction.from_float(float(value)) for value in np.asarray(values).reshape(-1))


def _line_coefficients(curve: AbstractCurve, /) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]] | None:
    if isinstance(curve, ExactLineCurve):
        direction = tuple(
            b - a for a, b in zip(curve.first_exact, curve.last_exact, strict=True)
        )
        return curve.first_exact, direction
    if isinstance(curve, LineCurve):
        return _rational_vector(curve.origin), _rational_vector(curve.direction)
    return None


def _projective(values: tuple[Fraction, ...], /) -> tuple[Fraction, ...] | None:
    pivot = next((value for value in values if value), None)
    return None if pivot is None else tuple(value / pivot for value in values)
def _circle_frame(
    curve: CircleCurve, /
) -> tuple[
    tuple[Fraction, ...],
    Fraction,
    tuple[Fraction, ...],
    tuple[Fraction, ...],
]:
    return (
        _rational_vector(curve.center),
        Fraction.from_float(float(np.asarray(curve.radius))),
        _rational_vector(curve.first_axis),
        _rational_vector(curve.second_axis),
    )


def _curve_payload(curve: AbstractCurve, /) -> str:
    identity = getattr(curve, "identity_payload", None)
    return canonical_fingerprint({
        "type": f"{type(curve).__module__}.{type(curve).__qualname__}",
        "structure": str(jax.tree_util.tree_structure(curve)),
        "arrays": array_tree_fingerprint(curve),
        "source": identity() if callable(identity) else None,
        "degree": getattr(curve, "degree", None),
    })


def _support_key(curve: AbstractCurve, /) -> tuple[object, ...]:
    line = _line_coefficients(curve)
    if line is not None:
        origin, direction = line
        pivot = next((index for index, value in enumerate(direction) if value), None)
        if pivot is not None:
            step = origin[pivot] / direction[pivot]
            return (
                "line",
                _projective(direction),
                tuple(point - step * slope for point, slope in zip(origin, direction, strict=True)),
            )
    if isinstance(curve, CircleCurve):
        center, radius, first, second = _circle_frame(curve)
        if len(center) == 3:
            normal = (
                first[1] * second[2] - first[2] * second[1],
                first[2] * second[0] - first[0] * second[2],
                first[0] * second[1] - first[1] * second[0],
            )
            return ("circle", center, radius, _projective(normal))
        return ("circle", center, radius)
    return ("carrier", _curve_payload(curve))


def _line_map(
    first: tuple[tuple[Fraction, ...], tuple[Fraction, ...]],
    second: tuple[tuple[Fraction, ...], tuple[Fraction, ...]],
    /,
) -> _ParameterMap | None:
    (origin_a, direction_a), (origin_b, direction_b) = first, second
    pivot = next((index for index, value in enumerate(direction_a) if value), None)
    if pivot is None:
        return None
    scale = direction_b[pivot] / direction_a[pivot]
    offset = (origin_b[pivot] - origin_a[pivot]) / direction_a[pivot]
    if not scale or any(b != scale * a for a, b in zip(direction_a, direction_b, strict=True)):
        return None
    if any(b != a + offset * d for a, b, d in zip(origin_a, origin_b, direction_a, strict=True)):
        return None
    return _ParameterMap(scale, _Parameter(offset))


def _circle_map(first: CircleCurve, second: CircleCurve, /) -> _ParameterMap | None:
    center_a, radius_a, cosine_a, sine_a = _circle_frame(first)
    center_b, radius_b, cosine_b, sine_b = _circle_frame(second)
    if center_a != center_b or radius_a != radius_b:
        return None
    quarter_turns = (
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
        (Fraction(-1), Fraction(0)),
        (Fraction(0), Fraction(-1)),
    )
    for quarter, (cosine, sine) in enumerate(quarter_turns):
        if cosine_b != tuple(cosine * a + sine * b for a, b in zip(cosine_a, sine_a, strict=True)):
            continue
        rotated = tuple(cosine * b - sine * a for a, b in zip(cosine_a, sine_a, strict=True))
        for scale in (Fraction(1), Fraction(-1)):
            if sine_b == tuple(scale * value for value in rotated):
                return _ParameterMap(scale, _turns(Fraction(quarter, 4)))
    return None


def _prove_curve_map(first: AbstractCurve, second: AbstractCurve, /) -> _ParameterMap | None:
    if first.ambient_dimension != second.ambient_dimension:
        return None
    if _curve_payload(first) == _curve_payload(second):
        return _ParameterMap(Fraction(1), _Parameter(Fraction(0)))
    first_line, second_line = _line_coefficients(first), _line_coefficients(second)
    if first_line is not None and second_line is not None:
        return _line_map(first_line, second_line)
    if isinstance(first, CircleCurve) and isinstance(second, CircleCurve):
        return _circle_map(first, second)
    return None


def _binary_vector(values: tuple[Fraction, ...], /) -> np.ndarray | None:
    numbers = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    if not np.all(np.isfinite(numbers)) or any(
        Fraction.from_float(float(number)) != value
        for number, value in zip(numbers, values, strict=True)
    ):
        return None
    return numbers


def _trim_segment_endpoint(
    segment: CurveTrimSegment,
    local: _Parameter,
    source_parameter: _Parameter,
    /,
) -> tuple[float, RootEndpoint | None] | None:
    zero, one = _Parameter(Fraction(0)), _Parameter(Fraction(1))
    root: RootEndpoint | None = None
    if local == zero:
        root = segment.last_root if segment.reversed else segment.first_root
    elif local == one:
        root = segment.first_root if segment.reversed else segment.last_root
    if root is not None:
        number = float(source_parameter.rational)
        if source_parameter.turns:
            number += float(source_parameter.turns) * math.tau
        lower, upper = (float(value) for value in np.asarray(root.parameter_enclosure()).reshape(-1))
        if not lower <= number <= upper:
            return None
        return number, root
    if not source_parameter.turns:
        number = float(source_parameter.rational)
        if np.isfinite(number) and Fraction.from_float(number) == source_parameter.rational:
            return number, None
    authority = _authority((segment.first_root, segment.last_root))
    if authority is None and not _periodic(segment.curve):
        return None
    patch, axis = (None, None) if authority is None else authority
    root = NativePeriodEndpoint(
        segment.curve,
        patch,
        axis,
        rational=source_parameter.rational,
        turns=source_parameter.turns,
    )
    lower, upper = root.parameter_enclosure()
    number = root.parameter
    if not lower <= number <= upper:
        number = 0.5 * (lower + upper)
    if not np.isfinite(number):
        return None
    return number, root


def _reparameterize_trim_segment(
    segment: CurveTrimSegment,
    parameter_map: _ParameterMap,
    /,
) -> CurveTrimSegment | None:
    zero, one = _Parameter(Fraction(0)), _Parameter(Fraction(1))
    local_first = parameter_map.offset
    local_last = _affine(one, parameter_map.scale, parameter_map.offset)
    if (
        _compare(local_first, zero) < 0
        or _compare(local_first, one) > 0
        or _compare(local_last, zero) < 0
        or _compare(local_last, one) > 0
    ):
        return None
    first = Fraction.from_float(float(segment.first))
    last = Fraction.from_float(float(segment.last))
    origin = last if segment.reversed else first
    slope = first - last if segment.reversed else last - first
    source_first = _affine(local_first, slope, _Parameter(origin))
    source_last = _affine(local_last, slope, _Parameter(origin))
    first_image = _trim_segment_endpoint(segment, local_first, source_first)
    last_image = _trim_segment_endpoint(segment, local_last, source_last)
    if first_image is None or last_image is None:
        return None
    if _compare(source_first, source_last) < 0:
        lower, upper, reversed_ = first_image, last_image, False
    elif _compare(source_first, source_last) > 0:
        lower, upper, reversed_ = last_image, first_image, True
    else:
        return None
    return CurveTrimSegment(
        segment.curve,
        lower[0],
        upper[0],
        reversed=reversed_,
        first_root=lower[1],
        last_root=upper[1],
    )


def _reparameterize_pcurve(curve: AbstractCurve, parameter_map: _ParameterMap, /) -> AbstractCurve | None:
    scale, offset = parameter_map.scale, parameter_map.offset
    if scale == 1 and offset == _Parameter(Fraction(0)):
        return curve
    if isinstance(curve, CurveTrimSegment):
        return _reparameterize_trim_segment(curve, parameter_map)
    if isinstance(curve, ExactLineCurve):
        if offset.turns:
            return None
        return ExactLineCurve(
            curve.evaluate_exact(offset.rational),
            curve.evaluate_exact(scale + offset.rational),
        )
    if isinstance(curve, LineCurve):
        if offset.turns:
            return None
        origin, direction = _rational_vector(curve.origin), _rational_vector(curve.direction)
        moved = _binary_vector(tuple(a + offset.rational * d for a, d in zip(origin, direction, strict=True)))
        scaled = _binary_vector(tuple(scale * value for value in direction))
        return None if moved is None or scaled is None else LineCurve(moved, scaled)
    if isinstance(curve, CircleCurve):
        if offset.rational or abs(scale) != 1 or (4 * offset.turns).denominator != 1:
            return None
        quarter_turns = ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)),
                         (Fraction(-1), Fraction(0)), (Fraction(0), Fraction(-1)))
        cosine, sine = quarter_turns[int(4 * offset.turns) % 4]
        _, _, first, second = _circle_frame(curve)
        cosine_axis = _binary_vector(
            tuple(cosine * a + sine * b for a, b in zip(first, second, strict=True))
        )
        sine_axis = _binary_vector(
            tuple(scale * (cosine * b - sine * a) for a, b in zip(first, second, strict=True))
        )
        if cosine_axis is None or sine_axis is None:
            return None
        return CircleCurve(np.asarray(curve.center), cosine_axis, sine_axis, curve.radius)
    if isinstance(curve, BSplineCurve):
        if offset.turns:
            return None
        knots = _binary_vector(tuple(
            (Fraction.from_float(float(knot)) - offset.rational) / scale
            for knot in np.asarray(curve.knots).reshape(-1)
        ))
        if knots is None:
            return None
        controls, weights = np.asarray(curve.control_points), np.asarray(curve.weights)
        if scale < 0:
            knots, controls, weights = knots[::-1], controls[::-1], weights[::-1]
        return BSplineCurve(controls, weights, knots, curve.degree)
    return None


@dataclass(frozen=True, slots=True)
class _Member:
    edge: int
    frame: _ParameterMap
    lower: _Parameter
    upper: _Parameter
    lower_vertex: int
    upper_vertex: int


@dataclass(frozen=True, slots=True)
class _Piece:
    member: _Member
    position: int
    lower: _Parameter
    upper: _Parameter

    @property
    def address(self) -> tuple[int, int]:
        return self.member.edge, self.position


type _GroupKey = tuple[int, _Parameter, _Parameter]
type _Points = dict[_Parameter, list[int]]


@dataclass(frozen=True, slots=True)
class _Coedge:
    pcurve: AbstractCurve
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
    geometry: BRepGeometry
    owned: dict[int, tuple[int, ...]]
    parents: dict[int, int] = field(default_factory=dict)
    curves: list[AbstractCurve] = field(default_factory=list)
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


def _periodic(curve: AbstractCurve, /) -> bool:
    symbol = getattr(curve, "native_period_symbol", None)
    return symbol() == "two_pi" if callable(symbol) else isinstance(curve, CircleCurve)


def _root_identity(root: object | None, /) -> str | None:
    if root is None:
        return None
    root_id = getattr(root, "root_id", None)
    if isinstance(root_id, str):
        return root_id
    if isinstance(root, RootEndpoint):
        return canonical_fingerprint(endpoint_root_payload(root))
    raise TypeError(f"Unsupported B-Rep sewing root identity: {type(root)!r}.")


def _edge_parameter_range(
    geometry: BRepGeometry, edge: int, /
) -> tuple[_Parameter, _Parameter] | None:
    values = np.asarray(geometry.edge_ranges, dtype=np.float64)[edge]
    result: list[_Parameter] = []
    for endpoint, root in enumerate(geometry.edge_endpoint_roots[edge]):
        if root is None:
            result.append(_Parameter(Fraction.from_float(float(values[endpoint]))))
        elif isinstance(root, NativePeriodEndpoint):
            result.append(_parameter(root.exact_parameter))
        elif isinstance(root, ConicParameterEndpoint) and root.parameter_exact is not None:
            result.append(_Parameter(root.parameter_exact))
        else:
            return None
    return result[0], result[1]


def _exact_range(
    geometry: BRepGeometry, edge: int, coedges: tuple[int, ...], /
) -> tuple[_Parameter, _Parameter] | None:
    if any(
        root is not None
        and not (
            isinstance(root, NativePeriodEndpoint)
            or isinstance(root, ConicParameterEndpoint) and root.parameter_exact is not None
        )
        for coedge in coedges
        for root in geometry.coedge_endpoint_roots[coedge]
    ):
        return None
    return _edge_parameter_range(geometry, edge)


def _member(
    staging: _Staging, reference: AbstractCurve, edge: int, periodic: bool, /
) -> _Member:
    geometry = staging.geometry
    curve_index = geometry.edge_curves[edge]
    curve = geometry.curves[curve_index]
    frame = _prove_curve_map(reference, curve)
    if frame is None:
        raise BRepSewingFailure(
            "sew_brep.carriers have a shared support but no exact representable parameter map",
            (edge,),
        )
    bounds = _exact_range(geometry, edge, staging.coedges(edge))
    if bounds is None:
        raise BRepSewingFailure(
            "sew_brep.endpoint_roots do not provide an exact source parameter binding",
            (edge,),
        )
    lower_source, upper_source = bounds
    length = _difference(upper_source, lower_source)
    if _compare(length, _Parameter(Fraction(0))) <= 0 or (
        periodic and _compare(length, _turns(1)) > 0
    ):
        raise BRepSewingFailure("sew_brep.edge_ranges must be positive source intervals", (edge,))
    start, end = geometry.edge_vertices[edge]
    if start == end and (not periodic or _compare(length, _turns(1)) != 0):
        raise BRepSewingFailure("sew_brep.closed_edges require one exact native period", (edge,))
    images = frame.apply(lower_source), frame.apply(upper_source)
    lower, upper = images if frame.scale > 0 else (images[1], images[0])
    lower_vertex, upper_vertex = (start, end) if frame.scale > 0 else (end, start)
    if periodic:
        shift = _turns(_turn_floor(lower))
        frame = _ParameterMap(frame.scale, _difference(frame.offset, shift))
        lower, upper = _difference(lower, shift), _difference(upper, shift)
    return _Member(edge, frame, lower, upper, lower_vertex, upper_vertex)


def _bucket_pieces(
    members: tuple[_Member, ...], periodic: bool, /
) -> tuple[dict[int, tuple[_Piece, ...]], _Points]:
    points: _Points = {}
    for member in members:
        points.setdefault(_normalized(member.lower, periodic), []).append(member.lower_vertex)
        points.setdefault(_normalized(member.upper, periodic), []).append(member.upper_vertex)
    ordered = sorted(points, key=_ORDER)
    stations = ordered + ([_add(value, _turns(1)) for value in ordered] if periodic else [])
    keys = [_ORDER(value) for value in stations]
    pieces: dict[int, tuple[_Piece, ...]] = {}
    for member in members:
        inner = stations[
            bisect_right(keys, _ORDER(member.lower)) : bisect_left(keys, _ORDER(member.upper))
        ]
        cuts = (member.lower, *inner, member.upper)
        pieces[member.edge] = tuple(
            _Piece(member, position, first, last)
            for position, (first, last) in enumerate(zip(cuts[:-1], cuts[1:], strict=True))
        )
    return pieces, points


def _authority(
    roots: tuple[RootEndpoint | None, ...], /
) -> tuple[AbstractSurfacePatch | None, Literal[0, 1] | None] | None:
    root = next((value for value in roots if isinstance(value, NativePeriodEndpoint)), None)
    if root is None:
        return None
    if root.axis is None:
        return root.patch, None
    if root.axis == 0:
        return root.patch, 0
    if root.axis == 1:
        return root.patch, 1
    raise BRepSewingFailure("Native-period endpoint axis must be 0 or 1", ())


def _endpoint(
    value: _Parameter,
    carrier: AbstractCurve,
    authority: tuple[AbstractSurfacePatch | None, Literal[0, 1] | None] | None,
    edges: tuple[int, ...],
    /,
) -> tuple[float, NativePeriodEndpoint | None]:
    if authority is None and not value.turns:
        number = float(value.rational)
        if np.isfinite(number) and Fraction.from_float(number) == value.rational:
            return number, None
    if authority is None and not _periodic(carrier):
        raise BRepSewingFailure(
            "sew_brep.fragment_endpoint is not representable and has no native-period authority",
            edges,
        )
    patch, axis = (None, None) if authority is None else authority
    root = NativePeriodEndpoint(carrier, patch, axis, rational=value.rational, turns=value.turns)
    lower, upper = root.parameter_enclosure()
    parameter = root.parameter
    if not lower <= parameter <= upper:
        parameter = 0.5 * (lower + upper)
    if not np.isfinite(parameter):
        raise BRepSewingFailure("sew_brep.fragment_endpoint has no finite enclosure", edges)
    return parameter, root


def _mapped_conic_endpoint(
    source_root: ConicParameterEndpoint,
    carrier: AbstractCurve,
    mapping: _ParameterMap,
    /,
) -> tuple[float, ConicParameterEndpoint] | None:
    if source_root.parameter_exact is None:
        return None
    target_parameter = mapping.inverse().apply(_Parameter(source_root.parameter_exact))
    if target_parameter.turns:
        return None
    root = ConicParameterEndpoint(
        source_root.root,
        carrier,
        target_parameter.rational,
        source_root.point_exact,
    )
    lower, upper = root.parameter_enclosure()
    parameter = float(target_parameter.rational)
    if not lower <= parameter <= upper:
        parameter = 0.5 * (lower + upper)
    if not np.isfinite(parameter):
        return None
    return parameter, root


def _target_edge_endpoint(
    value: _Parameter,
    carrier: AbstractCurve,
    authority: tuple[AbstractSurfacePatch | None, Literal[0, 1] | None] | None,
    edges: tuple[int, ...],
    covering: tuple[_Piece, ...],
    plans: tuple[tuple[_ParameterMap, dict[int, _Coedge]], ...],
    geometry: BRepGeometry,
    /,
) -> tuple[float, RootEndpoint | None]:
    candidates: list[tuple[float, ConicParameterEndpoint]] = []
    for piece, (mapping, _) in zip(covering, plans, strict=True):
        source_parameter = mapping.apply(value)
        for root in geometry.edge_endpoint_roots[piece.member.edge]:
            if (
                isinstance(root, ConicParameterEndpoint)
                and root.parameter_exact is not None
                and source_parameter == _Parameter(root.parameter_exact)
            ):
                image = _mapped_conic_endpoint(root, carrier, mapping)
                if image is None:
                    raise BRepSewingFailure(
                        "sew_brep.conic_endpoint has no exact target scalar map", edges
                    )
                candidates.append(image)
    if candidates:
        first = candidates[0]
        if any(
            root.point_exact != first[1].point_exact
            or canonical_fingerprint(root.root.identity_payload())
            != canonical_fingerprint(first[1].root.identity_payload())
            for _, root in candidates[1:]
        ):
            raise BRepSewingFailure(
                "sew_brep.conic_endpoint roots disagree at one exact source station", edges
            )
        return first
    return _endpoint(value, carrier, authority, edges)


def _transport_coedge_roots(
    source_roots: tuple[RootEndpoint | None, RootEndpoint | None],
    carrier: AbstractCurve,
    mapping: _ParameterMap,
    target: tuple[_Parameter, _Parameter],
    periodic: bool,
    piece_edge: int,
    /,
) -> tuple[RootEndpoint | None, RootEndpoint | None] | None:
    authority = _authority(source_roots)
    roots: list[RootEndpoint | None] = []
    for value in target:
        source_parameter = mapping.apply(value)
        conic_roots = tuple(
            root
            for root in source_roots
            if isinstance(root, ConicParameterEndpoint)
            and root.parameter_exact is not None
            and source_parameter == _Parameter(root.parameter_exact)
        )
        if conic_roots:
            images = tuple(
                _mapped_conic_endpoint(root, carrier, mapping) for root in conic_roots
            )
            if any(image is None for image in images):
                return None
            first = images[0]
            assert first is not None
            if any(
                image is None
                or image[1].point_exact != first[1].point_exact
                or canonical_fingerprint(image[1].root.identity_payload())
                != canonical_fingerprint(first[1].root.identity_payload())
                for image in images[1:]
            ):
                raise BRepSewingFailure(
                    "sew_brep.conic coedge roots disagree at one exact source station",
                    (piece_edge,),
                )
            roots.append(first[1])
        elif authority is not None or periodic or _periodic(carrier):
            roots.append(_endpoint(value, carrier, authority, (piece_edge,))[1])
        else:
            roots.append(None)
    return roots[0], roots[1]


def _transport(
    staging: _Staging,
    carrier: _Piece,
    piece: _Piece,
    target: tuple[_Parameter, _Parameter],
    periodic: bool,
    /,
) -> tuple[_ParameterMap, dict[int, _Coedge]] | None:
    geometry = staging.geometry
    carrier_frame, own_frame = carrier.member.frame, piece.member.frame
    shift = _difference(piece.lower, carrier.lower)
    offset = _affine(
        _difference(_add(carrier_frame.offset, shift), own_frame.offset),
        1 / own_frame.scale,
        _Parameter(Fraction(0)),
    )
    mapping = _ParameterMap(carrier_frame.scale / own_frame.scale, offset)
    result: dict[int, _Coedge] = {}
    for coedge in staging.coedges(piece.member.edge):
        pcurve = _reparameterize_pcurve(geometry.pcurves[coedge], mapping)
        if pcurve is None:
            return None
        roots = _transport_coedge_roots(
            geometry.coedge_endpoint_roots[coedge],
            pcurve,
            mapping,
            target,
            periodic,
            piece.member.edge,
        )
        if roots is None:
            return None
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
) -> tuple[_Piece, tuple[_Parameter, _Parameter], tuple[tuple[_ParameterMap, dict[int, _Coedge]], ...]]:
    for carrier in covering:
        inverse = carrier.member.frame.inverse()
        ends = inverse.apply(carrier.lower), inverse.apply(carrier.upper)
        target = ends if carrier.member.frame.scale > 0 else (ends[1], ends[0])
        plans = tuple(_transport(staging, carrier, piece, target, periodic) for piece in covering)
        complete = tuple(plan for plan in plans if plan is not None)
        if len(complete) == len(plans):
            return carrier, target, complete
    raise BRepSewingFailure(
        "sew_brep.pcurves have no exact common carrier parameterization",
        tuple(piece.member.edge for piece in covering),
    )


def _build_group(
    staging: _Staging,
    key: _GroupKey,
    points: _Points,
    periodic: bool,
    /,
) -> None:
    geometry = staging.geometry
    covering = tuple(sorted(staging.groups[key], key=lambda piece: piece.member.edge))
    sources = tuple(piece.member.edge for piece in covering)
    carrier, target, plans = _select_carrier(staging, covering, periodic)
    source_curve_index = geometry.edge_curves[carrier.member.edge]
    curve = geometry.curves[source_curve_index]
    authority = _authority(geometry.edge_endpoint_roots[carrier.member.edge])
    lower, lower_root = _target_edge_endpoint(
        target[0], curve, authority, sources, covering, plans, geometry
    )
    upper, upper_root = _target_edge_endpoint(
        target[1], curve, authority, sources, covering, plans, geometry
    )

    def station_vertex(station: _Parameter, /) -> int:
        if station == carrier.member.lower:
            return carrier.member.lower_vertex
        if station == carrier.member.upper:
            return carrier.member.upper_vertex
        return points[_normalized(station, periodic)][0]

    first_station, last_station = (
        (carrier.lower, carrier.upper)
        if carrier.member.frame.scale > 0
        else (carrier.upper, carrier.lower)
    )
    target_index = len(staging.edges)
    closed = periodic and _compare(_difference(carrier.upper, carrier.lower), _turns(1)) == 0
    staging.edges.append(
        _TargetEdge(
            staging.curve(source_curve_index),
            lower,
            upper,
            (station_vertex(first_station), station_vertex(last_station)),
            (lower_root, upper_root),
            closed,
        )
    )
    staging.group_edges[key] = target_index
    for piece, (mapping, coedges) in zip(covering, plans, strict=True):
        staging.images.append(
            BRepSewingEdgeImage(
                piece.member.edge,
                target_index,
                mapping.scale,
                mapping.offset.rational,
                mapping.offset.turns,
            )
        )
        for coedge, transported in coedges.items():
            staging.fragments[(key, coedge)] = transported
    if len(covering) > 1:
        staging.merged.append((target_index, sources))


def _verbatim(staging: _Staging, edge: int, /) -> None:
    geometry = staging.geometry
    source_curve = geometry.edge_curves[edge]
    first, last = (float(value) for value in np.asarray(geometry.edge_ranges)[edge])
    start, end = geometry.edge_vertices[edge]
    target_index = len(staging.edges)
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
    staging.verbatim[edge] = target_index
    staging.images.append(
        BRepSewingEdgeImage(edge, target_index, Fraction(1), Fraction(0), Fraction(0))
    )


def _target_vertices(
    staging: _Staging, /
) -> tuple[
    dict[int, int],
    tuple[tuple[int, ...], ...],
    np.ndarray,
    tuple[tuple[Fraction, Fraction, Fraction] | None, ...],
    tuple[BRepVertexRoot | None, ...],
]:
    geometry = staging.geometry
    points = np.asarray(geometry.vertex_points, dtype=np.float64)
    classes: dict[int, list[int]] = {}
    for vertex in range(points.shape[0]):
        classes.setdefault(staging.find(vertex), []).append(vertex)
    ordered = sorted(classes.values(), key=min)
    roots_out: list[BRepVertexRoot | None] = []
    exact_out: list[tuple[Fraction, Fraction, Fraction] | None] = []
    points_out: list[np.ndarray] = []
    for members in ordered:
        roots = tuple(geometry.vertex_roots[vertex] for vertex in members)
        bound_roots = tuple(root for root in roots if root is not None)
        if bound_roots and (
            len(bound_roots) != len(members)
            or len({_root_identity(root) for root in bound_roots}) != 1
        ):
            raise BRepSewingFailure(
                "sew_brep.vertex_roots do not prove an exact source alias",
                tuple(
                    edge
                    for edge, ends in enumerate(geometry.edge_vertices)
                    if any(vertex in members for vertex in ends)
                ),
            )
        bindings = tuple(
            exact
            for vertex in members
            if (exact := geometry.vertex_exact_points[vertex]) is not None
        )
        if bindings and len(bindings) != len(members) and not bound_roots:
            raise BRepSewingFailure(
                "sew_brep.vertex_exact_points do not bind every source vertex in an alias class"
            )
        if bindings and any(binding != bindings[0] for binding in bindings[1:]):
            raise BRepSewingFailure("sew_brep.vertex_exact_points disagree within a source alias class")
        point_vertex = next(
            (vertex for vertex in members if geometry.vertex_exact_points[vertex] is not None),
            members[0],
        )
        points_out.append(points[point_vertex])
        exact_out.append(bindings[0] if bindings else None)
        roots_out.append(bound_roots[0] if bound_roots else None)
    target = {vertex: index for index, members in enumerate(ordered) for vertex in members}
    return (
        target,
        tuple(tuple(members) for members in ordered),
        np.asarray(points_out, dtype=np.float64).reshape((-1, 3)),
        tuple(exact_out),
        tuple(roots_out),
    )


def _stage_buckets(staging: _Staging) -> dict[_GroupKey, tuple[_Points, bool]]:
    geometry = staging.geometry
    buckets: dict[tuple[object, ...], list[int]] = {}
    for edge, curve_index in enumerate(geometry.edge_curves):
        if curve_index >= 0:
            buckets.setdefault(_support_key(geometry.curves[curve_index]), []).append(edge)
    contexts: dict[_GroupKey, tuple[_Points, bool]] = {}
    for bucket, edges in enumerate(values for values in buckets.values() if len(values) > 1):
        reference = geometry.curves[geometry.edge_curves[edges[0]]]
        periodic = _periodic(reference)
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
            points, periodic = contexts[key]
            for station in (values[0].lower, values[0].upper):
                vertices = points[_normalized(station, periodic)]
                for vertex in vertices[1:]:
                    staging.union(vertices[0], vertex)
    return contexts


def _sew_by_correspondence(
    geometry: BRepGeometry, /
) -> tuple[BRepGeometry, BRepSewingLineage, _Staging] | None:
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
                points, periodic = contexts[key]
                _build_group(staging, key, points, periodic)
    target, lineage = _publish_correspondence(staging)
    return target, lineage, staging


def _target_uses(staging: _Staging, coedge: int, /) -> list[tuple[int, _Coedge]]:
    geometry = staging.geometry
    edge, sense = geometry.coedge_edges[coedge], geometry.coedge_senses[coedge]
    if edge in staging.verbatim:
        return [(
            staging.verbatim[edge],
            _Coedge(geometry.pcurves[coedge], sense, geometry.coedge_endpoint_roots[coedge]),
        )]
    member_pieces = staging.pieces[edge]
    ascending = (member_pieces[0].member.frame.scale > 0) == (sense > 0)
    ordered = member_pieces if ascending else tuple(reversed(member_pieces))
    result = []
    for piece in ordered:
        key = staging.piece_groups[piece.address]
        result.append((staging.group_edges[key], staging.fragments[(key, coedge)]))
    return result


def _publish_correspondence(staging: _Staging, /) -> tuple[BRepGeometry, BRepSewingLineage]:
    geometry = staging.geometry
    vertex_ids, vertex_sources, target_points, exact_points, vertex_roots = _target_vertices(staging)
    uses: list[tuple[int, _Coedge]] = []
    coedge_sources: list[int] = []
    face_loops: list[tuple[tuple[int, ...], ...]] = []
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
    edge_vertices: list[tuple[int, int]] = []
    for index, edge in enumerate(staging.edges):
        ends = vertex_ids[edge.vertices[0]], vertex_ids[edge.vertices[1]]
        if ends[0] == ends[1] and not edge.closed:
            sources = tuple(image.source_edge for image in staging.images if image.target_edge == index)
            raise BRepSewingFailure("sew_brep.vertex_classes collapse a nonclosed edge", sources)
        edge_vertices.append(ends)
    graph = None
    face_gap_bounds = (
        None
        if geometry.edge_face_gap_bounds is None
        else tuple(geometry.edge_face_gap_bounds[coedge] for coedge in coedge_sources)
    )
    target = BRepGeometry(
        vertex_points=target_points,
        vertex_exact_points=exact_points,
        vertex_roots=vertex_roots,
        curves=tuple(staging.curves),
        edge_curves=tuple(edge.curve for edge in staging.edges),
        edge_ranges=np.asarray(
            [(edge.lower, edge.upper) for edge in staging.edges], dtype=np.float64
        ).reshape((-1, 2)),
        edge_vertices=tuple(edge_vertices),
        edge_endpoint_roots=tuple(edge.roots for edge in staging.edges),
        pcurves=tuple(use.pcurve for _, use in uses),
        coedge_edges=tuple(edge for edge, _ in uses),
        coedge_senses=tuple(use.sense for _, use in uses),
        coedge_endpoint_roots=tuple(use.roots for _, use in uses),
        edge_face_gap_bounds=face_gap_bounds,
        face_loops=tuple(face_loops),
        shell_faces=(),
        shell_orientations=(),
        solid_shells=(),
        assembly_graph=graph,
    )
    lineage = BRepSewingLineage(vertex_sources, tuple(staging.images), tuple(coedge_sources))
    _require_exhaustive(geometry, target, lineage)
    return target, lineage


def _exact_range_for_lineage(
    geometry: BRepGeometry, edge: int, /
) -> tuple[_Parameter, _Parameter] | None:
    return _edge_parameter_range(geometry, edge)


def _map_matches_proof(
    source: AbstractCurve,
    target: AbstractCurve,
    image: BRepSewingEdgeImage,
    /,
) -> bool:
    proof = _prove_curve_map(source, target)
    if proof is None or proof.scale != image.scale:
        return False
    image_offset = _Parameter(image.offset_rational, image.offset_turns)
    delta = _difference(image_offset, proof.offset)
    if delta == _Parameter(Fraction(0)):
        return True
    return _periodic(source) and not delta.rational and delta.turns.denominator == 1


def _require_exhaustive(
    source: BRepGeometry, target: BRepGeometry, lineage: BRepSewingLineage, /
) -> None:
    vertices = sorted(vertex for sources in lineage.vertex_sources for vertex in sources)
    if vertices != list(range(source.vertex_points.shape[0])):
        raise BRepSewingFailure("BRepSewingLineage.vertex_sources are not exhaustive")
    images_by_source: dict[int, list[BRepSewingEdgeImage]] = {}
    images_by_target: dict[int, dict[int, BRepSewingEdgeImage]] = {}
    for image in lineage.edge_images:
        images_by_source.setdefault(image.source_edge, []).append(image)
        images_by_target.setdefault(image.target_edge, {})[image.source_edge] = image
        source_curve_index = source.edge_curves[image.source_edge]
        target_curve_index = target.edge_curves[image.target_edge]
        if source_curve_index < 0:
            if (
                target_curve_index >= 0
                or image.scale != 1
                or image.offset_rational
                or image.offset_turns
            ):
                raise BRepSewingFailure(
                    "Degenerate-edge lineage is not identity-preserving",
                    (image.source_edge,),
                )
            continue
        if target_curve_index < 0 or not _map_matches_proof(
            source.curves[source_curve_index], target.curves[target_curve_index], image
        ):
            raise BRepSewingFailure(
                "BRepSewingLineage contains an unproved carrier map",
                (image.source_edge,),
            )
    if set(images_by_source) != set(range(len(source.edge_curves))):
        raise BRepSewingFailure("BRepSewingLineage.edge_images omit a source edge")
    if (
        len(lineage.coedge_sources) != len(target.coedge_edges)
        or any(
            source_coedge < 0 or source_coedge >= len(source.coedge_edges)
            for source_coedge in lineage.coedge_sources
        )
        or set(lineage.coedge_sources) != set(range(len(source.coedge_edges)))
    ):
        raise BRepSewingFailure(
            "BRepSewingLineage.coedge_sources omit or misalign a source coedge"
        )
    for target_coedge, source_coedge in enumerate(lineage.coedge_sources):
        source_edge = source.coedge_edges[source_coedge]
        target_edge = target.coedge_edges[target_coedge]
        image = images_by_target.get(target_edge, {}).get(source_edge)
        if (
            image is None
            or target.coedge_senses[target_coedge]
            != source.coedge_senses[source_coedge] * image.orientation
        ):
            raise BRepSewingFailure(
                "BRepSewingLineage.coedge map disagrees with edge orientation",
                (source_edge,),
            )
        mapping = _ParameterMap(image.scale, _Parameter(image.offset_rational, image.offset_turns))
        expected = _reparameterize_pcurve(source.pcurves[source_coedge], mapping)
        if expected is None or _curve_payload(expected) != _curve_payload(target.pcurves[target_coedge]):
            raise BRepSewingFailure(
                "BRepSewingLineage contains an unproved p-curve transport",
                (source_edge,),
            )
        source_roots = source.coedge_endpoint_roots[source_coedge]
        target_roots = target.coedge_endpoint_roots[target_coedge]
        unsupported_roots = any(
            root is not None
            and not isinstance(root, NativePeriodEndpoint)
            and not (
                isinstance(root, ConicParameterEndpoint)
                and root.parameter_exact is not None
            )
            for root in source_roots
        )
        if unsupported_roots:
            source_ids = tuple(_root_identity(root) for root in source_roots)
            target_ids = tuple(_root_identity(root) for root in target_roots)
            if mapping.scale != 1 or mapping.offset != _Parameter(Fraction(0)) or source_ids != target_ids:
                raise BRepSewingFailure(
                    "A non-scalar source coedge root was not retained verbatim",
                    (source_edge,),
                )
        else:
            ordered_roots = source_roots if mapping.scale > 0 else tuple(reversed(source_roots))
            inverse = mapping.inverse()
            for source_root, target_root in zip(ordered_roots, target_roots, strict=True):
                if source_root is None:
                    continue
                if isinstance(source_root, NativePeriodEndpoint):
                    if (
                        not isinstance(target_root, NativePeriodEndpoint)
                        or _parameter(target_root.exact_parameter)
                        != inverse.apply(_parameter(source_root.exact_parameter))
                    ):
                        raise BRepSewingFailure(
                            "BRepSewingLineage has an invalid native-period coedge root map",
                            (source_edge,),
                        )
                elif isinstance(source_root, ConicParameterEndpoint):
                    if source_root.parameter_exact is None:
                        raise BRepSewingFailure(
                            "An inexact conic coedge root cannot be transported",
                            (source_edge,),
                        )
                    expected = inverse.apply(_Parameter(source_root.parameter_exact))
                    if (
                        expected.turns
                        or not isinstance(target_root, ConicParameterEndpoint)
                        or target_root.parameter_exact != expected.rational
                        or target_root.point_exact != source_root.point_exact
                        or target_root.root.identity_payload() != source_root.root.identity_payload()
                    ):
                        raise BRepSewingFailure(
                            "BRepSewingLineage has an invalid conic coedge root map",
                            (source_edge,),
                        )
                else:
                    raise BRepSewingFailure(
                        "An unsupported coedge root escaped exact lineage admission",
                        (source_edge,),
                    )
    for source_edge, images in images_by_source.items():
        source_range = _exact_range_for_lineage(source, source_edge)
        if source_range is None:
            if (
                len(images) != 1
                or images[0].scale != 1
                or images[0].offset_rational
                or images[0].offset_turns
            ):
                raise BRepSewingFailure(
                    "A root-bounded edge was fragmented without an exact scalar map",
                    (source_edge,),
                )
            target_edge = images[0].target_edge
            source_root_ids = tuple(_root_identity(root) for root in source.edge_endpoint_roots[source_edge])
            target_root_ids = tuple(_root_identity(root) for root in target.edge_endpoint_roots[target_edge])
            if source_root_ids != target_root_ids or not np.array_equal(
                np.asarray(source.edge_ranges)[source_edge], np.asarray(target.edge_ranges)[target_edge]
            ):
                raise BRepSewingFailure(
                    "A root-bounded source edge was not retained verbatim",
                    (source_edge,),
                )
            for source_vertex, target_vertex in zip(
                source.edge_vertices[source_edge], target.edge_vertices[target_edge], strict=True
            ):
                if source_vertex not in lineage.vertex_sources[target_vertex]:
                    raise BRepSewingFailure(
                        "BRepSewingLineage omits a root-bounded endpoint vertex",
                        (source_edge,),
                    )
            continue
        intervals: list[tuple[_Parameter, _Parameter]] = []
        for image in images:
            target_range = _exact_range_for_lineage(target, image.target_edge)
            if target_range is None:
                raise BRepSewingFailure("A target edge has an unresolved scalar range", (source_edge,))
            mapping = _ParameterMap(image.scale, _Parameter(image.offset_rational, image.offset_turns))
            first, last = mapping.apply(target_range[0]), mapping.apply(target_range[1])
            low, high = (first, last) if _compare(first, last) <= 0 else (last, first)
            intervals.append((low, high))
            start, end = source.edge_vertices[source_edge]
            for target_vertex, parameter in zip(
                target.edge_vertices[image.target_edge], target_range, strict=True
            ):
                source_parameter = mapping.apply(parameter)
                source_vertex = (
                    start if _compare(source_parameter, source_range[0]) == 0
                    else end if _compare(source_parameter, source_range[1]) == 0
                    else None
                )
                if source_vertex is not None and source_vertex not in lineage.vertex_sources[target_vertex]:
                    raise BRepSewingFailure(
                        "BRepSewingLineage omits a source endpoint vertex",
                        (source_edge,),
                    )
        intervals.sort(key=lambda row: _ORDER(row[0]))
        if (
            not intervals
            or _compare(intervals[0][0], source_range[0]) != 0
            or _compare(intervals[-1][1], source_range[1]) != 0
        ):
            raise BRepSewingFailure("BRepSewingLineage does not cover a source edge range", (source_edge,))
        if any(_compare(first[1], second[0]) != 0 for first, second in zip(intervals[:-1], intervals[1:])):
            raise BRepSewingFailure("BRepSewingLineage has a gap or overlap in a source edge", (source_edge,))


def _edge_uses(
    geometry: BRepGeometry, faces: tuple[int, ...], signs: np.ndarray
) -> dict[int, list[tuple[int, int]]]:
    uses: dict[int, list[tuple[int, int]]] = {}
    for face in faces:
        for loop in geometry.face_loops[face]:
            for coedge in loop:
                edge = geometry.coedge_edges[coedge]
                if geometry.edge_curves[edge] >= 0:
                    uses.setdefault(edge, []).append(
                        (face, geometry.coedge_senses[coedge] * int(signs[face]))
                    )
    return uses


def _invalid_edges(uses: dict[int, list[tuple[int, int]]], /) -> tuple[int, ...]:
    return tuple(
        edge for edge, values in sorted(uses.items())
        if len(values) != 2 or sum(sign for _, sign in values) != 0
    )


@checked
def shells(
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
    components: list[tuple[int, ...]] = []
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
        components.append(tuple(sorted(component)))
    return tuple(components)


def _region_invalid_edges(
    geometry: BRepGeometry,
    regions: tuple[tuple[tuple[int, ...], np.ndarray], ...],
    /,
) -> tuple[int, ...]:
    return tuple(sorted({
        edge
        for faces, outward in regions
        for edge in _invalid_edges(_edge_uses(geometry, faces, outward))
    }))


def _contact(
    geometry: BRepGeometry,
    first: int,
    second: int,
    policy: BRepSewingPolicy,
    /,
) -> BRepSewingContact:
    ranges = np.asarray(geometry.edge_ranges, dtype=np.float64)
    operands = tuple(
        CurveRange(
            geometry.curves[geometry.edge_curves[edge]],
            float(ranges[edge, 0]),
            float(ranges[edge, 1]),
        )
        for edge in (first, second)
    )
    result = intersect_curve_ranges(*operands, policy=policy.contact_intersection)
    relations: set[BRepSewingContactRelation] = {point.kind for point in result.points}
    if result.coincident:
        relations.add("coincident-within-bound")
    if result.unresolved:
        relations.add("unresolved")
    gaps = [point.gap_bound for point in result.points]
    gaps.extend(region.distance_bound for region in result.coincident)
    return BRepSewingContact(
        first,
        second,
        tuple(sorted(relations)) or ("separated",),
        min(gaps, default=math.inf),
    )


def _contacts(
    geometry: BRepGeometry,
    edges: tuple[int, ...],
    vertex_class: dict[int, int],
    policy: BRepSewingPolicy,
    /,
) -> tuple[BRepSewingContact, ...]:
    candidates = tuple(
        edge for edge in sorted(set(edges)) if geometry.edge_curves[edge] >= 0
    )
    contacts: list[BRepSewingContact] = []
    for position, first in enumerate(candidates):
        first_vertices = {vertex_class[vertex] for vertex in geometry.edge_vertices[first]}
        for second in candidates[position + 1 :]:
            if first_vertices & {vertex_class[vertex] for vertex in geometry.edge_vertices[second]}:
                continue
            contact = _contact(geometry, first, second, policy)
            contacts.append(contact)
    return tuple(contacts)


def _open_failure(
    geometry: BRepGeometry,
    reason: str,
    edges: tuple[int, ...],
    vertex_class: dict[int, int],
    policy: BRepSewingPolicy,
    /,
) -> BRepSewingFailure:
    return BRepSewingFailure(
        reason,
        edges,
        contacts=_contacts(geometry, edges, vertex_class, policy),
    )


def _require_region_connectivity(
    target: BRepGeometry,
    merged: tuple[tuple[int, tuple[int, ...]], ...],
    solid_faces: tuple[tuple[int, ...], ...],
    /,
) -> None:
    regions: dict[int, set[int]] = {}
    for region, faces in enumerate(solid_faces):
        for face in faces:
            regions.setdefault(face, set()).add(region)
    coedge_faces: dict[int, int] = {}
    for face, loops in enumerate(target.face_loops):
        for loop in loops:
            for coedge in loop:
                coedge_faces[coedge] = face
    users: dict[int, set[int]] = {}
    for coedge, edge in enumerate(target.coedge_edges):
        users.setdefault(edge, set()).add(coedge_faces[coedge])
    for edge, sources in merged:
        faces = sorted(users.get(edge, set()))
        if len(faces) < 2:
            continue
        reached, queue = {faces[0]}, deque((faces[0],))
        while queue:
            face = queue.popleft()
            for other in faces:
                if other not in reached and regions.get(face, set()) & regions.get(other, set()):
                    reached.add(other)
                    queue.append(other)
        if len(reached) != len(faces):
            raise BRepSewingFailure(
                "sew_brep.solid_faces must declare a shared face for coincident edges",
                sources,
            )


def _identity_lineage(geometry: BRepGeometry, /) -> BRepSewingLineage:
    return BRepSewingLineage(
        tuple((vertex,) for vertex in range(geometry.vertex_points.shape[0])),
        tuple(
            BRepSewingEdgeImage(edge, edge, Fraction(1), Fraction(0), Fraction(0))
            for edge in range(len(geometry.edge_curves))
        ),
        tuple(range(len(geometry.coedge_edges))),
    )


@dataclass(frozen=True, slots=True)
class _Plan:
    source: BRepGeometry
    regions: tuple[tuple[tuple[int, ...], np.ndarray], ...]
    signs: np.ndarray
    solid_faces: tuple[tuple[int, ...], ...]
    solid_orientations: tuple[tuple[int, ...], ...]
    policy: BRepSewingPolicy


def _assemble(
    plan: _Plan,
    geometry: BRepGeometry,
    lineage: BRepSewingLineage,
    /,
) -> BRepSewingResult:
    shell_faces: list[tuple[int, ...]] = []
    shell_orientations: list[tuple[int, ...]] = []
    solid_shells: list[tuple[int, ...]] = []
    for faces, outward in plan.regions:
        components = shells(geometry, faces, outward)
        solid_shells.append(tuple(range(len(shell_faces), len(shell_faces) + len(components))))
        shell_faces.extend(components)
        shell_orientations.extend(
            tuple(int(outward[face] * plan.signs[face]) for face in component)
            for component in components
        )
    graph = plan.source.assembly_graph if plan.source.assembly_graph.solid_count else None
    assembled = BRepGeometry(
        vertex_points=geometry.vertex_points,
        vertex_exact_points=geometry.vertex_exact_points,
        vertex_roots=geometry.vertex_roots,
        curves=geometry.curves,
        edge_curves=geometry.edge_curves,
        edge_ranges=geometry.edge_ranges,
        edge_vertices=geometry.edge_vertices,
        edge_endpoint_roots=geometry.edge_endpoint_roots,
        pcurves=geometry.pcurves,
        coedge_edges=geometry.coedge_edges,
        coedge_senses=geometry.coedge_senses,
        coedge_endpoint_roots=geometry.coedge_endpoint_roots,
        edge_face_gap_bounds=geometry.edge_face_gap_bounds,
        face_loops=geometry.face_loops,
        shell_faces=tuple(shell_faces),
        shell_orientations=tuple(shell_orientations),
        solid_shells=tuple(solid_shells),
        assembly_graph=graph,
    )
    certificate = canonical_fingerprint({
        "kind": "native-exact-brep-sewing",
        "source_geometry": plan.source.geometry_id,
        "target_geometry": assembled.geometry_id,
        "orientation": plan.signs,
        "tolerance": plan.policy.tolerance,
        "solid_faces": plan.solid_faces,
        "solid_orientations": plan.solid_orientations,
        "lineage": lineage.lineage_id,
    })
    return BRepSewingResult(assembled, certificate, lineage)


def _prepare_plan(
    geometry: BRepGeometry,
    orientation: ArrayLike,
    solid_faces: tuple[tuple[int, ...], ...],
    policy: BRepSewingPolicy,
    solid_orientations: tuple[tuple[int, ...], ...] | None,
    /,
) -> _Plan:
    signs = np.asarray(orientation, dtype=np.float64)
    if signs.shape != (len(geometry.face_loops),) or not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError("sew_brep.orientation must contain one -1 or +1 per face.")
    if not isinstance(solid_faces, tuple) or any(not isinstance(faces, tuple) for faces in solid_faces):
        raise TypeError("sew_brep.solid_faces must be a tuple of face-index tuples.")
    owned = [face for faces in solid_faces for face in faces]
    if any(isinstance(face, bool) or not isinstance(face, int) for face in owned):
        raise TypeError("sew_brep.solid_faces entries must be integer face IDs.")
    if set(owned) != set(range(len(geometry.face_loops))):
        raise ValueError("sew_brep.solid_faces must exhaust the geometry face inventory.")
    if any(len(faces) != len(set(faces)) for faces in solid_faces):
        raise ValueError("sew_brep.solid_faces cannot repeat a face within one solid.")
    owner_counts: dict[int, int] = {}
    for face in owned:
        owner_counts[face] = owner_counts.get(face, 0) + 1
    if any(count > 2 for count in owner_counts.values()):
        raise ValueError("sew_brep.solid_faces may assign a face to at most two regions.")
    region_signs = (
        tuple(tuple(int(signs[face]) for face in faces) for faces in solid_faces)
        if solid_orientations is None
        else solid_orientations
    )
    if len(region_signs) != len(solid_faces) or any(
        len(values) != len(faces)
        or any(isinstance(value, bool) or value not in (-1, 1) for value in values)
        for faces, values in zip(solid_faces, region_signs, strict=True)
    ):
        raise ValueError("sew_brep.solid_orientations must align with faces and contain signs.")
    face_owners: dict[int, list[int]] = {}
    for region, faces in enumerate(solid_faces):
        if not faces:
            raise ValueError("sew_brep.solid_faces cannot contain an empty solid.")
        for face in faces:
            face_owners.setdefault(face, []).append(region)
    for face, owners in face_owners.items():
        if len(owners) == 2:
            first, second = owners
            first_sign = region_signs[first][solid_faces[first].index(face)]
            second_sign = region_signs[second][solid_faces[second].index(face)]
            if first_sign == second_sign:
                raise ValueError("A shared face must have opposite region orientations.")
    if geometry.assembly_graph.solid_count and tuple(solid_faces) != geometry.topology.solid_faces:
        raise BRepSewingFailure("sew_brep.assembly_graph cannot be remapped across changed solid ownership")
    regions: list[tuple[tuple[int, ...], np.ndarray]] = []
    for faces, values in zip(solid_faces, region_signs, strict=True):
        outward = signs.copy()
        for face, value in zip(faces, values, strict=True):
            outward[face] = value
        regions.append((faces, outward))
    return _Plan(
        geometry,
        tuple(regions),
        signs,
        tuple(tuple(faces) for faces in solid_faces),
        tuple(tuple(values) for values in region_signs),
        policy,
    )


@checked
def sew_brep(
    geometry: BRepGeometry,
    orientation: ArrayLike,
    solid_faces: tuple[tuple[int, ...], ...],
    /,
    *,
    policy: BRepSewingPolicy | None = None,
    solid_orientations: tuple[tuple[int, ...], ...] | None = None,
) -> BRepSewingResult:
    """Sew exact source shells; proximity, fitting and sampled identity are forbidden."""
    policy_ = BRepSewingPolicy() if policy is None else policy
    plan = _prepare_plan(geometry, orientation, solid_faces, policy_, solid_orientations)
    invalid = _region_invalid_edges(geometry, plan.regions)
    if not invalid:
        lineage = _identity_lineage(geometry)
        _require_exhaustive(geometry, geometry, lineage)
        return _assemble(plan, geometry, lineage)
    sewn = _sew_by_correspondence(geometry)
    if sewn is None:
        identity = {vertex: vertex for vertex in range(geometry.vertex_points.shape[0])}
        raise _open_failure(geometry, _OPEN_SHELL, invalid, identity, policy_)
    target, lineage, staging = sewn
    _require_region_connectivity(target, tuple(staging.merged), plan.solid_faces)
    open_target = set(_region_invalid_edges(target, plan.regions))
    if open_target:
        sources = tuple(sorted({
            image.source_edge for image in lineage.edge_images if image.target_edge in open_target
        }))
        vertex_class = {
            vertex: staging.find(vertex)
            for vertex in range(geometry.vertex_points.shape[0])
        }
        reason = f"sew_brep.shells must have closed balanced incidence after correspondence: {_OPEN_SHELL}"
        raise _open_failure(geometry, reason, sources, vertex_class, policy_)
    return _assemble(plan, target, lineage)


__all__ = [
    "BRepSewingEdgeImage",
    "BRepSewingFailure",
    "BRepSewingLineage",
    "BRepSewingPolicy",
    "BRepSewingResult",
    "BRepSewingContact",
    "BRepSewingContactRelation",
    "sew_brep",
    "shells",
]
