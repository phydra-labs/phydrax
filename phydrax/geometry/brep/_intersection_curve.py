#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Evaluate surface intersections and oriented source trim curves.

Analytic and rational surface families are not closed under intersection: the
smooth intersection of two quadrics is in general a genus-one algebraic curve
with no finite rational representation. An `IntersectionCurve` is therefore
defined implicitly by its generating surface pair and one branch of their common
zero set. Its definition is a certified continuation atlas: each chart is a
coupled parameter box on both surfaces in which the branch is the graph of the
three remaining coordinates over one parameter axis, established by a parametric
Krawczyk inclusion. Points, jets and both p-curves are evaluated from the same
chart solution, and every evaluation reports its numerical parameter bound.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal, overload, Self, TypeAlias
from weakref import ref, ReferenceType

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array
from jax.core import Tracer
from jax.typing import ArrayLike

import phydrax.geometry.brep._intersection as intersection_owner
import phydrax.geometry.brep._placed as placed_owner
from phydrax.discretization._coordinate_budget import reserve_polynomial

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._host_storage import (
    declared_object_storage_bytes,
    immutable_host_array,
    immutable_host_array_layout,
    immutable_host_array_layout_bytes,
)
from ..._strict import StrictModule
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from ...nonlinear import VectorLocalRootPlan
from ...typing import (
    checked,
    ConvertibleToArray,
    Dim,
    HostBool,
    HostFloat64,
    HostInt32,
    parse,
    Scope,
)
from .._atlas import AbstractTrimCurve
from .._interval_enclosure import (
    interval_add,
    interval_multiply,
    prepare_interval_function,
)
from .._rational import rational_vector
from ._carrier_pieces import (
    CoefficientEnclosureSource,
    CurveCoordinateResult,
    CurvePiece,
    RootCurveCoordinateSource,
    SurfacePiece,
)
from ._coordinate_request import coordinate_source_request
from ._intersection_kernel import interval_times_interval, krawczyk
from ._interval_source import evaluate_interval
from ._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    BernsteinCurvePiece,
    BernsteinRestrictionBank,
    BernsteinSurfacePiece,
    BSplineCurve,
    BSplineSpanCurve,
    BSplineSpanSurface,
    BSplineSurfacePatch,
    CircleCurve,
    coefficient_enclosures,
    ConePatch,
    CylinderPatch,
    EllipseCurve,
    ExtrusionSurface,
    HyperbolaCurve,
    LineCurve,
    MeridianOverlaps,
    OffsetCurve,
    OffsetSurface,
    ParabolaCurve,
    PlanePatch,
    RevolutionSurface,
    RuledSurface,
    SpherePatch,
    SurfaceIsoparametricCurve,
    TorusPatch,
)
from ._request_preparations import TrimRequestPreparations
from ._trim_helpers import (
    fraction_bounds,
    period_offset,
    PeriodOffset,
)


type RootEndpoint = intersection_owner.RootEndpoint


IntersectionEndpointKind: TypeAlias = Literal[
    "boundary", "singular", "uncertified", "closed"
]
IntersectionCurveSide: TypeAlias = Literal["first", "second"]


class ChartDim(Dim):
    """Charts of one intersection-curve continuation atlas."""


class NodeDim(Dim):
    """Chart nodes (chart count plus one) of an intersection curve."""


_PERIOD_TOLERANCE = 1.0e-12


# ------------------------------------------------------------ generating regions


def _host_box(parameter_box: ConvertibleToArray, /) -> np.ndarray:
    box = np.asarray(parameter_box, dtype=np.float64)
    if box.shape != (2, 2) or not np.all(np.isfinite(box)):
        raise ValueError(
            f"SurfaceRegion.parameter_box must be finite with shape (2, 2); got {box!r}"
        )
    if np.any(box[1] <= box[0]):
        raise ValueError(
            f"SurfaceRegion.parameter_box must be nonempty along both axes; got {box!r}"
        )
    return box


class SurfaceRegion(StrictModule):
    """A generating surface restricted to one closed parameter box.

    ``parameter_box`` is ``[[u_lower, v_lower], [u_upper, v_upper]]``. An axis is
    periodic when the surface declares a period and the box spans exactly one
    period; its two faces are then identified.
    """

    patch: AbstractSurfacePatch
    lower: tuple[float, float] = eqx.field(static=True)
    upper: tuple[float, float] = eqx.field(static=True)
    periodic: tuple[bool, bool] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        patch: AbstractSurfacePatch,
        parameter_box: ConvertibleToArray,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> None:
        if preparations is None:
            preparations = TrimRequestPreparations()
        with coordinate_source_request(
            (
                patch,
                parameter_box,
            ),
            budget=budget,
        ) as budget:
            box = patch.validate_parameter_box(_host_box(parameter_box), budget=budget)
            periodic = tuple(
                period is not None
                and abs((box[1, axis] - box[0, axis]) - period)
                <= _PERIOD_TOLERANCE * max(1.0, period)
                for axis, period in enumerate(patch.periods)
            )
            self.patch = patch
            self.lower = (float(box[0, 0]), float(box[0, 1]))
            self.upper = (float(box[1, 0]), float(box[1, 1]))
            self.periodic = (bool(periodic[0]), bool(periodic[1]))

    @property
    def parameter_box(self) -> np.ndarray:
        return np.asarray((self.lower, self.upper), dtype=np.float64)

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[Self],
        value: SurfaceRegion,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> Self:
        """Revalidate the retained surface and parameter-box projection."""
        return cls(value.patch, value.parameter_box, budget=budget)


class CurveRange(StrictModule):
    """A generating curve restricted to one closed parameter interval."""

    curve: AbstractCurve | AbstractTrimCurve | IntersectionCurve
    first: float = eqx.field(static=True)
    last: float = eqx.field(static=True)
    periodic: bool = eqx.field(static=True)

    @checked
    def __init__(
        self,
        curve: AbstractCurve | AbstractTrimCurve | IntersectionCurve,
        first: float | None = None,
        last: float | None = None,
    ) -> None:
        domain = curve.parameter_domain
        if first is None or last is None:
            if domain is None:
                period = curve.period
                if period is None:
                    raise ValueError(
                        f"CurveRange.interval must be explicit for an unbounded curve; got {(first, last)!r}"
                    )
                domain = (0.0, period)
            first = domain[0] if first is None else first
            last = domain[1] if last is None else last
        first_, last_ = curve.validate_range(first, last)
        period = curve.period
        self.curve = curve
        self.first = first_
        self.last = last_
        self.periodic = period is not None and abs(
            (last_ - first_) - period
        ) <= _PERIOD_TOLERANCE * max(1.0, period)

    @property
    def ambient_dimension(self) -> int:
        return self.curve.ambient_dimension

    @classmethod
    @checked
    def from_stored_fields(cls: type[Self], value: CurveRange, /) -> Self:
        """Revalidate the retained curve and closed parameter interval."""
        return cls(value.curve, value.first, value.last)


# ------------------------------------------------------- piece evaluators


CurveEvaluator: TypeAlias = AbstractCurve | AbstractTrimCurve


def surface_pieces(
    region: SurfaceRegion, /, *, budget: CoordinateEnclosureBudget | None = None
) -> tuple[SurfacePiece[AbstractSurfacePatch], ...]:
    """Split a region into pieces whose evaluators are gather-free arithmetic.

    Rational spline surfaces use Bernstein span pieces with source coefficient
    enclosures; analytic and swept families evaluate directly.
    """
    if budget is None:
        with coordinate_source_request((region,)) as budget:
            return surface_pieces(region, budget=budget)
    return surface_pieces_for_box(region.patch, region.parameter_box, budget=budget)


def surface_pieces_for_box(
    patch: AbstractSurfacePatch,
    parameter_box: ConvertibleToArray,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[SurfacePiece[AbstractSurfacePatch], ...]:
    """Gather-free source pieces for a closed query box, including isolines."""
    if budget is None:
        with coordinate_source_request(
            (
                patch,
                parameter_box,
            )
        ) as budget:
            return surface_pieces_for_box(patch, parameter_box, budget=budget)

    return patch.source_pieces(parameter_box, budget=budget)


def curve_pieces(
    curve_range: CurveRange, /, *, budget: CoordinateEnclosureBudget | None = None
) -> tuple[CurvePiece[CurveEvaluator | IntersectionCurve], ...]:
    """Split a curve range into gather-free arithmetic pieces."""
    if budget is None:
        with coordinate_source_request((curve_range,)) as budget:
            return curve_pieces(curve_range, budget=budget)
    return curve_pieces_for_interval(
        curve_range.curve, curve_range.first, curve_range.last, coordinate_budget=budget
    )


@overload
def curve_pieces_for_interval(
    curve: AbstractCurve,
    first: float,
    last: float,
    /,
    *,
    coordinate_budget: CoordinateEnclosureBudget,
) -> tuple[CurvePiece[AbstractCurve], ...]: ...


@overload
def curve_pieces_for_interval(
    curve: AbstractTrimCurve,
    first: float,
    last: float,
    /,
    *,
    coordinate_budget: CoordinateEnclosureBudget,
) -> tuple[CurvePiece[CurveEvaluator], ...]: ...


@overload
def curve_pieces_for_interval(
    curve: IntersectionCurve,
    first: float,
    last: float,
    /,
    *,
    coordinate_budget: CoordinateEnclosureBudget,
) -> tuple[CurvePiece[IntersectionCurve], ...]: ...


@overload
def curve_pieces_for_interval(
    curve: CurveEvaluator,
    first: float,
    last: float,
    /,
    *,
    coordinate_budget: CoordinateEnclosureBudget,
) -> tuple[CurvePiece[CurveEvaluator], ...]: ...


@overload
def curve_pieces_for_interval(
    curve: CurveEvaluator | IntersectionCurve,
    first: float,
    last: float,
    /,
    *,
    coordinate_budget: CoordinateEnclosureBudget,
) -> tuple[CurvePiece[CurveEvaluator | IntersectionCurve], ...]: ...


def curve_pieces_for_interval(
    curve: CurveEvaluator | IntersectionCurve,
    first: float,
    last: float,
    /,
    *,
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> tuple[CurvePiece[CurveEvaluator | IntersectionCurve], ...]:
    """Deterministic source span restriction of a closed interval (points allowed)."""
    if coordinate_budget is None:
        with coordinate_source_request(
            (
                curve,
                first,
                last,
            )
        ) as coordinate_budget:
            return curve_pieces_for_interval(
                curve, first, last, coordinate_budget=coordinate_budget
            )

    match curve:
        case AbstractCurve() | IntersectionCurve():
            return curve.source_pieces(first, last, coordinate_budget=coordinate_budget)
        case AbstractTrimCurve():
            if not (math.isfinite(first) and math.isfinite(last) and first <= last):
                raise ValueError(
                    f"CurveRange.interval must be finite and ordered; got {(first, last)!r}"
                )
            if first < last:
                curve.validate_range(first, last)
            else:
                domain = curve.parameter_domain
                if domain is not None and not domain[0] <= first <= domain[1]:
                    raise ValueError(
                        f"CurveRange.parameter must lie inside the source curve domain; got {first!r}"
                    )
            return (CurvePiece(curve, first, last, 0),)
        case _:
            raise TypeError(
                f"CurveRange.curve must provide a source carrier; got {type(curve).__name__}"
            )


# --------------------------------------------------------- defining equations


class SurfacePairSystem(StrictModule):
    """``S1(u1, v1) - S2(u2, v2)`` over the coupled parameters ``(u1, v1, u2, v2)``."""

    first: AbstractSurfacePatch
    second: AbstractSurfacePatch

    def residual(self, parameters: Array, /) -> Array:
        return self.first.evaluate(parameters[:2]) - self.second.evaluate(parameters[2:])


# One source-tree address (attribute names / positions) of a B-spline node,
# its active knot span(s) and the knot-break pattern (``diff(knots) > 0`` per
# axis) those spans were addressed in: immutable host integer/boolean
# structure of one surface piece. Not part of the payload or any fingerprint.
_PinPath: TypeAlias = tuple[tuple[str, str | int], ...]
_SpanPins: TypeAlias = tuple[
    tuple[_PinPath, tuple[int, ...], tuple[tuple[bool, ...], ...]], ...
]


def _pin_path(path: tuple[Any, ...], /) -> _PinPath:
    address: list[tuple[str, str | int]] = []
    for key in path:
        match key:
            case jax.tree_util.GetAttrKey():
                address.append(("attr", key.name))
            case jax.tree_util.SequenceKey():
                address.append(("index", key.idx))
            case _:
                raise ValueError(
                    f"IntersectionCurve.source_address must use attributes or positions; got {key!r}"
                )
    return tuple(address)


def _source_child(node: Any, name: str, /) -> Any:
    match node, name:
        case ((placed_owner.PlacedSurface() | placed_owner.PlacedCurve()), "definition"):
            return node.definition
        case ((OffsetSurface() | OffsetCurve()), "base"):
            return node.base
        case SurfaceIsoparametricCurve(), "surface":
            return node.surface
        case ((ExtrusionSurface() | RevolutionSurface() | AffinePCurve()), "curve"):
            return node.curve
        case PeriodicPCurve(), "source_curve":
            return node.source_curve
        case RuledSurface(), "first":
            return node.first
        case RuledSurface(), "second":
            return node.second
        case _:
            raise ValueError(
                "IntersectionCurve.source_address must mirror its source structure; got a source-address mismatch"
            )


def _node_at(tree: Any, address: _PinPath, /) -> Any:
    node = tree
    for kind, value in address:
        match kind, value:
            case "attr", str():
                node = _source_child(node, value)
            case "attr", _:
                raise TypeError(
                    f"IntersectionCurve.source_address must use a string attribute; got {value!r}"
                )
            case "index", int():
                if not isinstance(node, (tuple, list)) or not -len(node) <= value < len(
                    node
                ):
                    raise ValueError(
                        "IntersectionCurve.source_address must mirror its source structure; got a source-address mismatch"
                    )
                node = node[value]
            case "index", _:
                raise TypeError(
                    f"IntersectionCurve.source_address must use an integer position; got {value!r}"
                )
            case _:
                raise ValueError(
                    f"IntersectionCurve.source_address must name an attribute or position; got {(kind, value)!r}"
                )
    return node


def _knot_span(knots: np.ndarray, lower: float, upper: float, /) -> int:
    span = int(np.searchsorted(knots, lower, side="right")) - 1
    if not (
        0 <= span < knots.size - 1 and knots[span] == lower and knots[span + 1] == upper
    ):
        raise ValueError(
            f"IntersectionCurve.span must coincide with one knot span; got {(lower, upper)!r}"
        )
    return span


def _span_pins(
    source: AbstractSurfacePatch, evaluator: AbstractSurfacePatch, /
) -> _SpanPins:
    """Host knot-span addresses of every Bernstein node in one source piece.

    Piece evaluators mirror their source structure (placement, offset, sweep
    and ruling wrappers keep their field names); each Bernstein node is the
    deterministic extraction of the B-spline at the same address.
    """
    kinds = (BernsteinSurfacePiece, BernsteinCurvePiece)
    flat, _ = jax.tree_util.tree_flatten_with_path(
        evaluator, is_leaf=lambda node: isinstance(node, kinds)
    )
    pins = []
    for path, node in flat:
        if not isinstance(node, kinds):
            continue
        address = _pin_path(path)
        target = _node_at(source, address)
        lower = np.asarray(node.lower, dtype=np.float64).reshape(-1)
        upper = np.asarray(node.upper, dtype=np.float64).reshape(-1)
        match node, target:
            case BernsteinSurfacePiece(), BSplineSurfacePatch():
                vectors = tuple(
                    np.asarray(knots, dtype=np.float64)
                    for knots in (target.u_knots, target.v_knots)
                )
            case BernsteinCurvePiece(), BSplineCurve():
                vectors = (np.asarray(target.knots, dtype=np.float64),)
            case _:
                raise ValueError(
                    "IntersectionCurve.source_address must address a B-spline source node; got a non-B-spline source node"
                )
        spans = tuple(
            _knot_span(knots, float(lower[axis]), float(upper[axis]))
            for axis, knots in enumerate(vectors)
        )
        breaks = tuple(
            tuple(bool(value) for value in np.diff(knots) > 0) for knots in vectors
        )
        pins.append((address, spans, breaks))
    return tuple(pins)


def _knot_vectors(node: Any, /) -> tuple[Array, ...]:
    match node:
        case BSplineSurfacePatch():
            return node.u_knots, node.v_knots
        case BSplineCurve():
            return (node.knots,)
        case _:
            raise ValueError(
                f"IntersectionCurve.source_address must name a B-spline source node; got {type(node).__name__!r}"
            )


def _replace_knot_vector(node: Any, index: int, knots: Array, /) -> Any:
    match node, index:
        case BSplineSurfacePatch(), 0:
            return eqx.tree_at(lambda item: item.u_knots, node, knots)
        case BSplineSurfacePatch(), 1:
            return eqx.tree_at(lambda item: item.v_knots, node, knots)
        case BSplineCurve(), 0:
            return eqx.tree_at(lambda item: item.knots, node, knots)
        case _:
            raise ValueError(
                f"IntersectionCurve.source_address must name a B-spline source node; got {type(node).__name__!r}"
            )


_TOPOLOGY_CHANGED = (
    "IntersectionCurve.knots must preserve knot-span topology at certified chart "
    "addresses; got merged or split live knot spans"
)


def _check_span_topology(source: AbstractSurfacePatch, pins: _SpanPins, /) -> None:
    """Host refusal when concrete live knots no longer match the pinned spans."""
    for address, _, breaks in pins:
        node = _node_at(source, address)
        for vector, expected in zip(_knot_vectors(node), breaks, strict=True):
            knots = np.asarray(vector, dtype=np.float64)
            if (
                knots.shape != (len(expected) + 1,)
                or tuple(bool(value) for value in np.diff(knots) > 0) != expected
            ):
                raise ValueError(_TOPOLOGY_CHANGED)


def _bind_span_pins(
    source: AbstractSurfacePatch, pins: _SpanPins, /
) -> AbstractSurfacePatch:
    """The live source with each pinned B-spline node restricted to its span.

    Spans index the current live knots. A concrete topology change is refused
    on the host; traced knots carry the same check as a runtime error, so a
    knot motion that merges or splits spans never reuses stale addresses.
    """
    bound: AbstractSurfacePatch = source
    for address, spans, breaks in pins:
        node = _node_at(bound, address)
        for index, (knots, expected) in enumerate(
            zip(_knot_vectors(node), breaks, strict=True)
        ):
            if knots.shape != (len(expected) + 1,):
                raise ValueError(_TOPOLOGY_CHANGED)
            if not isinstance(knots, Tracer):
                if (
                    tuple(bool(value) for value in np.diff(np.asarray(knots)) > 0)
                    != expected
                ):
                    raise ValueError(_TOPOLOGY_CHANGED)
                continue
            checked = eqx.error_if(
                knots,
                jnp.any((jnp.diff(knots) > 0) != jnp.asarray(expected)),
                _TOPOLOGY_CHANGED,
            )
            node = _replace_knot_vector(node, index, checked)
        pinned = (
            BSplineSpanSurface(node, (spans[0], spans[1]))
            if isinstance(node, BSplineSurfacePatch)
            else BSplineSpanCurve(node, spans[0])
        )
        if not address:
            if not isinstance(pinned, AbstractSurfacePatch):
                raise ValueError(
                    f"IntersectionCurve.source must remain a surface patch; got {type(pinned).__name__!r}"
                )
            bound = pinned
        else:
            bound = eqx.tree_at(
                lambda tree, address=address: _node_at(tree, address), bound, pinned
            )
    return bound


# Row ``d`` maps ``(u1, v1, u2, v2)`` to ``(x_d, remaining coordinates)``.
_CHART_PERMUTATIONS = np.stack(
    [
        np.eye(4, dtype=np.float64)[[axis] + [k for k in range(4) if k != axis]]
        for axis in range(4)
    ]
)


# -------------------------------------------------------------- serialization


# These records persist construction state rather than derived certificates or
# numerical caches. SurfaceRegion retains its deterministic parameter-box projection.
_CONSTRUCTION_FIELDS: Mapping[str, tuple[str, ...]] = {
    "OffsetSurface": ("base", "distance"),
    "SurfaceRegion": ("patch", "parameter_box"),
    "IntersectionCurve": (
        "first",
        "second",
        "chart_axes",
        "chart_start",
        "chart_end",
        "chart_pieces",
        "box_lower",
        "box_upper",
        "preconditioners",
        "contraction",
        "certified",
        "transition_shifts",
        "closed",
        "start_kind",
        "end_kind",
        "node_axes",
        "node_values",
        "node_pieces",
        "node_references",
        "node_period_shifts",
    ),
    "AffinePCurve": ("curve", "matrix", "offset"),
    "PeriodicPCurve": ("source_curve", "patch", "period_shifts"),
    "TrimIntersectionRoot": ("first", "second", "parameter_lower", "parameter_upper"),
    "CurveSurfaceIntersectionRoot": (
        "curve",
        "surface",
        "parameter_lower",
        "parameter_upper",
    ),
    "TripleSurfaceIntersectionRoot": (
        "first",
        "second",
        "third",
        "parameter_lower",
        "parameter_upper",
    ),
    "IntersectionCurvePointRoot": ("curve", "parameter"),
    "BRepPlacedVertex": (
        "source_point",
        "rotation",
        "translation",
        "source_entity_id",
        "source_root",
    ),
    "BRepVertexRoot": (
        "primary",
        "aliases",
        "spatial_root",
        "joint_root",
        "source_edge_lifts",
    ),
    "BRepRootSupport": ("patch", "root"),
    "PlanePatch": ("origin", "first_axis", "second_axis"),
    "CylinderPatch": ("origin", "first_axis", "second_axis", "axis", "radius"),
    "ConePatch": (
        "origin",
        "first_axis",
        "second_axis",
        "axis",
        "reference_radius",
        "semi_angle",
    ),
    "SpherePatch": ("center", "first_axis", "second_axis", "axis", "radius"),
    "TorusPatch": (
        "center",
        "first_axis",
        "second_axis",
        "axis",
        "major_radius",
        "minor_radius",
    ),
    "SurfaceIsoparametricCurve": (
        "surface",
        "fixed_axis",
        "fixed_value",
        "parameter_range",
    ),
    "BSplineSurfacePatch": (
        "control_points",
        "weights",
        "u_knots",
        "v_knots",
        "u_degree",
        "v_degree",
    ),
    "ExtrusionSurface": ("curve", "direction"),
    "RevolutionSurface": ("curve", "axis_origin", "axis_direction"),
    "RuledSurface": ("first", "second"),
    "LineCurve": ("origin", "direction"),
    "CircleCurve": ("center", "first_axis", "second_axis", "radius"),
    "EllipseCurve": (
        "center",
        "first_axis",
        "second_axis",
        "first_radius",
        "second_radius",
    ),
    "ParabolaCurve": ("vertex", "first_axis", "second_axis", "focal_length"),
    "HyperbolaCurve": (
        "center",
        "first_axis",
        "second_axis",
        "first_radius",
        "second_radius",
    ),
    "OffsetCurve": ("base", "distance", "direction"),
    "BSplineCurve": ("control_points", "weights", "knots", "degree"),
    "PlacedSurface": ("definition", "rotation", "translation"),
    "PlacedCurve": ("definition", "rotation", "translation"),
    "TrimRootEndpoint": (
        "root",
        "operand",
        "affine_parameter_offset",
        "affine_parameter_scale",
        "affine_transforms",
        "source_pcurve",
        "source_surface",
    ),
    "BranchRootEndpoint": (
        "root",
        "curve",
        "chart",
        "source_pcurve",
        "source_side",
        "source_first",
        "source_last",
        "periodic_shifts",
        "affine_transforms",
    ),
    "NativePeriodEndpoint": ("curve", "patch", "axis", "rational", "turns"),
    "IntersectionPCurve": ("curve", "side", "first", "last", "reversed"),
    "CurveTrimSegment": ("curve", "first", "last", "reversed", "first_root", "last_root"),
}

_POSITIONAL_FIELDS: Mapping[str, tuple[str, ...]] = {
    "SurfaceIsoparametricCurve": ("surface", "fixed_axis", "fixed_value"),
    "TrimIntersectionRoot": ("first", "second"),
    "CurveSurfaceIntersectionRoot": ("curve", "surface"),
    "TripleSurfaceIntersectionRoot": ("first", "second", "third"),
    "IntersectionCurvePointRoot": ("curve", "parameter"),
    "TrimRootEndpoint": ("root", "operand"),
    "BranchRootEndpoint": ("root", "curve", "chart"),
    "NativePeriodEndpoint": ("curve", "patch", "axis"),
    "BRepVertexRoot": ("primary",),
    "BRepRootSupport": ("patch", "root"),
    "BRepPlacedVertex": ("source_point", "rotation", "translation", "source_entity_id"),
}


def _registered_geometry_class(name: str, /) -> type[StrictModule] | None:
    if name in _MODULE_TYPES:
        return _MODULE_TYPES[name]
    match name:
        case "PlacedSurface":
            return placed_owner.PlacedSurface
        case "PlacedCurve":
            return placed_owner.PlacedCurve
        case "TrimIntersectionRoot":
            return intersection_owner.TrimIntersectionRoot
        case "CurveSurfaceIntersectionRoot":
            return intersection_owner.CurveSurfaceIntersectionRoot
        case "TripleSurfaceIntersectionRoot":
            return intersection_owner.TripleSurfaceIntersectionRoot
        case "IntersectionCurvePointRoot":
            return intersection_owner.IntersectionCurvePointRoot
        case "TrimRootEndpoint":
            return intersection_owner.TrimRootEndpoint
        case "BranchRootEndpoint":
            return intersection_owner.BranchRootEndpoint
        case "NativePeriodEndpoint":
            return intersection_owner.NativePeriodEndpoint
        case "BRepPlacedVertex" | "BRepVertexRoot" | "BRepRootSupport":
            # root_bindings↔correspondence↔intersection_curve requires resolution
            # at codec use, after the participating modules have initialized.
            from . import _root_bindings

            match name:
                case "BRepPlacedVertex":
                    return _root_bindings.BRepPlacedVertex
                case "BRepVertexRoot":
                    return _root_bindings.BRepVertexRoot
                case "BRepRootSupport":
                    return _root_bindings.BRepRootSupport
                case _:
                    return None
        case _:
            return None


def encode_geometry(value: Any, /) -> Any:
    """Encode a curve or surface construction without changing its representation."""
    if value is None:
        return None
    if isinstance(value, Fraction):
        return {"fraction": [value.numerator, value.denominator]}
    if isinstance(value, tuple):
        return {"tuple": [encode_geometry(item) for item in value]}
    if isinstance(value, StrictModule):
        name = type(value).__name__
        if _registered_geometry_class(name) is not type(value):
            raise TypeError(f"geometry.type must have a registered codec; got {name!r}")
        return {
            "type": name,
            "fields": {
                field: encode_geometry(getattr(value, field))
                for field in _CONSTRUCTION_FIELDS[name]
            },
        }
    if isinstance(value, (jax.Array, np.ndarray)):
        host = np.asarray(value)
        if host.dtype not in (
            np.dtype("float64"),
            np.dtype("int32"),
            np.dtype("bool"),
        ) or not np.all(np.isfinite(host)):
            raise ValueError(
                f"geometry.array must contain finite float64, int32 or bool values; got {host.dtype!r}"
            )
        return {
            "dtype": str(host.dtype),
            "shape": list(host.shape),
            "hex": [float(item).hex() for item in host.reshape((-1,))]
            if host.dtype == np.float64
            else host.reshape((-1,)).tolist(),
        }
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(
        f"geometry.value must use a supported payload kind; got {type(value).__name__}"
    )


def _decode_fraction(values: Any, /) -> Fraction:
    if (
        not isinstance(values, list)
        or len(values) != 2
        or any(type(value) is not int for value in values)
        or values[1] <= 0
    ):
        raise ValueError(
            f"geometry.fraction must contain an integer numerator and positive denominator; got {values!r}"
        )
    restored = Fraction(*values)
    if [restored.numerator, restored.denominator] != values:
        raise ValueError(f"geometry.fraction must be reduced; got {values!r}")
    return restored


def _decode_geometry_record(payload: Mapping[str, Any], /) -> StrictModule:
    if set(payload) != {"type", "fields"} or not isinstance(payload["type"], str):
        raise ValueError(
            f"geometry.record must contain only a string type and fields; got {tuple(payload)!r}"
        )
    cls = _registered_geometry_class(payload["type"])
    if cls is None:
        raise ValueError(
            f"geometry.type must name a registered construction; got {payload['type']!r}"
        )
    fields_payload = payload["fields"]
    expected = _CONSTRUCTION_FIELDS[payload["type"]]
    if not isinstance(fields_payload, Mapping) or set(fields_payload) != set(expected):
        raise ValueError(
            f"geometry.fields must match the registered construction field set; got {fields_payload!r}"
        )
    fields = {name: decode_geometry(fields_payload[name]) for name in expected}
    positional_names = _POSITIONAL_FIELDS.get(payload["type"], ())
    positional = [fields[name] for name in positional_names]
    keywords = {
        name: value for name, value in fields.items() if name not in positional_names
    }
    restored = cls(*positional, **keywords)
    for name, value in fields.items():
        if isinstance(value, np.ndarray):
            target = np.asarray(getattr(restored, name))
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError(
                    f"{'geometry.' + name} must retain its dtype and shape; got {(value.dtype, value.shape)!r}"
                )
    if canonical_fingerprint(encode_geometry(restored)) != canonical_fingerprint(payload):
        raise ValueError(
            "geometry.fields must round-trip without changing their representation; got a changed reconstruction payload"
        )
    return restored


def _decode_geometry_array(payload: Mapping[str, Any], /) -> np.ndarray:
    if set(payload) != {"dtype", "shape", "hex"} or payload["dtype"] not in (
        "float64",
        "int32",
        "bool",
    ):
        raise ValueError(
            f"geometry.array must contain a registered dtype, shape and values; got {tuple(payload)!r}"
        )
    shape = payload["shape"]
    encoded = payload["hex"]
    if (
        not isinstance(shape, list)
        or any(type(size) is not int or size < 0 for size in shape)
        or not isinstance(encoded, list)
        or len(encoded) != math.prod(shape)
    ):
        raise ValueError(
            f"geometry.shape must agree with the encoded value count; got {shape!r}"
        )
    dtype = payload["dtype"]
    if dtype == "float64":
        if any(not isinstance(item, str) for item in encoded):
            raise ValueError(
                f"geometry.values must contain hexadecimal strings for float64; got {encoded!r}"
            )
        values = np.asarray([float.fromhex(item) for item in encoded], dtype=np.float64)
    else:
        expected_type = bool if dtype == "bool" else int
        if any(type(item) is not expected_type for item in encoded):
            raise ValueError(
                f"geometry.values must match their declared scalar dtype; got {encoded!r}"
            )
        if dtype == "int32" and any(not -(2**31) <= item < 2**31 for item in encoded):
            raise ValueError(f"geometry.values must lie inside int32; got {encoded!r}")
        values = np.asarray(encoded, dtype=dtype)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"geometry.values must be finite; got {values!r}")
    return values.reshape(tuple(shape))


def decode_geometry(payload: Any, /) -> Any:
    """Restore registered construction records and typed arrays losslessly."""
    match payload:
        case None:
            return None
        case Mapping() if set(payload) == {"fraction"}:
            return _decode_fraction(payload["fraction"])
        case Mapping() if set(payload) == {"tuple"}:
            if not isinstance(payload["tuple"], list):
                raise ValueError(
                    f"geometry.tuple must contain an item list; got {payload['tuple']!r}"
                )
            return tuple(decode_geometry(item) for item in payload["tuple"])
        case Mapping() if "type" in payload:
            return _decode_geometry_record(payload)
        case Mapping():
            return _decode_geometry_array(payload)
        case bool() | int() | str():
            return payload
        case float() if math.isfinite(payload):
            return payload
        case _:
            raise ValueError(
                f"geometry.payload must use a registered record or scalar representation; got {type(payload).__name__!r}"
            )


# ------------------------------------------------------------ intersection curve


class _AtlasInput(StrictModule):
    axes: np.ndarray
    start: np.ndarray
    end: np.ndarray
    pieces: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    preconditioners: np.ndarray
    contraction: np.ndarray
    certified: np.ndarray
    shifts: np.ndarray
    start_kind: IntersectionEndpointKind = eqx.field(static=True)
    end_kind: IntersectionEndpointKind = eqx.field(static=True)


class _AtlasAddresses(StrictModule):
    addresses: tuple[tuple[int, int], ...] = eqx.field(static=True)
    pins: tuple[tuple[_SpanPins, _SpanPins], ...] = eqx.field(static=True)
    selector: tuple[int, ...] = eqx.field(static=True)


class _AtlasNodes(StrictModule):
    axes: np.ndarray
    values: np.ndarray
    pieces: np.ndarray
    references: np.ndarray
    shifts: np.ndarray


def _prepare_atlas_input(
    chart_axes: ArrayLike,
    chart_start: ArrayLike,
    chart_end: ArrayLike,
    chart_pieces: ArrayLike,
    box_lower: ArrayLike,
    box_upper: ArrayLike,
    preconditioners: ArrayLike,
    contraction: ArrayLike,
    certified: ArrayLike,
    transition_shifts: ArrayLike,
    closed: bool,
    start_kind: IntersectionEndpointKind,
    end_kind: IntersectionEndpointKind,
) -> _AtlasInput:
    scope = Scope()
    axes = parse(
        np.asarray(chart_axes, dtype=np.int32),
        HostInt32[ChartDim],
        "chart_axes",
        scope=scope,
    )
    arrays = {
        name: parse(
            np.asarray(value, dtype=np.float64),
            HostFloat64[ChartDim, Literal[4]],
            name,
            scope=scope,
        )
        for name, value in (
            ("chart_start", chart_start),
            ("chart_end", chart_end),
            ("box_lower", box_lower),
            ("box_upper", box_upper),
            ("transition_shifts", transition_shifts),
        )
    }
    pieces = parse(
        np.asarray(chart_pieces, dtype=np.int32),
        HostInt32[ChartDim, Literal[2]],
        "chart_pieces",
        scope=scope,
    )
    preconditioners_ = parse(
        np.asarray(preconditioners, dtype=np.float64),
        HostFloat64[ChartDim, Literal[3], Literal[3]],
        "preconditioners",
        scope=scope,
    )
    contraction_ = parse(
        np.asarray(contraction, dtype=np.float64),
        HostFloat64[ChartDim],
        "contraction",
        scope=scope,
    )
    certified_ = parse(
        np.asarray(certified, dtype=np.bool_),
        HostBool[ChartDim],
        "certified",
        scope=scope,
    )
    if axes.shape[0] < 1:
        raise ValueError(
            f"IntersectionCurve.chart_axes must contain at least one chart; got {axes.shape[0]!r}"
        )
    if np.any((axes < 0) | (axes > 3)):
        raise ValueError(f"IntersectionCurve.chart_axes must lie in [0, 3]; got {axes!r}")
    start, end = arrays["chart_start"], arrays["chart_end"]
    rows = np.arange(axes.shape[0])
    if np.any(start[rows, axes] == end[rows, axes]):
        raise ValueError(
            "IntersectionCurve.chart_nodes must advance along each chart axis; got equal chart-axis endpoints"
        )
    if np.any(arrays["box_lower"] > arrays["box_upper"]):
        raise ValueError(
            "IntersectionCurve.chart_boxes must satisfy lower <= upper; got an inverted chart box"
        )
    for name, value in arrays.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(
                f"{'IntersectionCurve.' + name} must be finite; got {value!r}"
            )
    if not np.all(np.isfinite(preconditioners_)) or not np.all(np.isfinite(contraction_)):
        raise ValueError(
            "IntersectionCurve.chart_bounds must have finite preconditioners and contraction bounds; got nonfinite chart bounds"
        )
    if np.any(contraction_ < 0.0) or np.any(certified_ & (contraction_ >= 1.0)):
        raise ValueError(
            f"IntersectionCurve.contraction must be nonnegative and below one for certified charts; got {contraction_!r}"
        )
    for node in (start, end):
        if np.any(node < arrays["box_lower"]) or np.any(node > arrays["box_upper"]):
            raise ValueError(
                f"IntersectionCurve.chart_nodes must lie inside their continuation boxes; got {node!r}"
            )
    start_kind = parse(start_kind, IntersectionEndpointKind, "endpoint kind")
    end_kind = parse(end_kind, IntersectionEndpointKind, "endpoint kind")
    if bool(closed) != (start_kind == "closed" and end_kind == "closed"):
        raise ValueError(
            f"IntersectionCurve.endpoint_kinds must agree with closed state; got {(closed, start_kind, end_kind)!r}"
        )
    return _AtlasInput(
        axes=axes,
        start=start,
        end=end,
        pieces=pieces,
        lower=arrays["box_lower"],
        upper=arrays["box_upper"],
        preconditioners=preconditioners_,
        contraction=contraction_,
        certified=certified_,
        shifts=arrays["transition_shifts"],
        start_kind=start_kind,
        end_kind=end_kind,
    )


def _prepare_atlas_addresses(
    first: SurfaceRegion,
    second: SurfaceRegion,
    pieces: np.ndarray,
    *,
    budget: CoordinateEnclosureBudget,
) -> _AtlasAddresses:
    addresses = tuple(sorted({(int(a), int(b)) for a, b in pieces}))
    first_workset = {piece.index: piece for piece in surface_pieces(first, budget=budget)}
    second_workset = {
        piece.index: piece for piece in surface_pieces(second, budget=budget)
    }
    if any(a not in first_workset or b not in second_workset for a, b in addresses):
        raise ValueError(
            f"IntersectionCurve.chart_pieces must name admitted source surface pieces; got {addresses!r}"
        )
    pins = tuple(
        (
            _span_pins(first.patch, first_workset[a].evaluator),
            _span_pins(second.patch, second_workset[b].evaluator),
        )
        for a, b in addresses
    )
    pair_slots = {pair: slot for slot, pair in enumerate(addresses)}
    selector = tuple(pair_slots[(int(a), int(b))] for a, b in pieces)
    return _AtlasAddresses(addresses=addresses, pins=pins, selector=selector)


def _prepare_atlas_nodes(
    atlas: _AtlasInput,
    first: SurfaceRegion,
    second: SurfaceRegion,
    closed: bool,
    node_axes: ArrayLike | None,
    node_values: ArrayLike | None,
    node_pieces: ArrayLike | None,
    node_references: ArrayLike | None,
    node_period_shifts: ArrayLike | None,
) -> _AtlasNodes:
    count = atlas.axes.shape[0]
    axes, start, end, pieces = atlas.axes, atlas.start, atlas.end, atlas.pieces
    rows = np.arange(count)
    default_axes = np.concatenate((axes[:1], axes))
    default_values = np.concatenate((start[0, axes[0]][None], end[rows, axes]))
    default_pieces = np.concatenate((pieces[:1], pieces), axis=0)
    references = np.arange(count + 1, dtype=np.int32)
    shifts = np.zeros((count + 1, 4), dtype=np.int32)
    if closed:
        references[-1] = 0
        periods = (*first.patch.periods, *second.patch.periods)
        for column, period in enumerate(periods):
            if period is not None:
                shifts[-1, column] = int(
                    round((end[-1, column] - start[0, column]) / period)
                )
    axes_ = parse(
        np.asarray(default_axes if node_axes is None else node_axes, dtype=np.int32),
        HostInt32[NodeDim],
        "node_axes",
    )
    values_ = parse(
        np.asarray(
            default_values if node_values is None else node_values,
            dtype=np.float64,
        ),
        HostFloat64[NodeDim],
        "node_values",
    )
    pieces_ = parse(
        np.asarray(
            default_pieces if node_pieces is None else node_pieces, dtype=np.int32
        ),
        HostInt32[NodeDim, Literal[2]],
        "node_pieces",
    )
    references_ = parse(
        np.asarray(
            references if node_references is None else node_references,
            dtype=np.int32,
        ),
        HostInt32[NodeDim],
        "node_references",
    )
    shifts_ = parse(
        np.asarray(
            shifts if node_period_shifts is None else node_period_shifts,
            dtype=np.int32,
        ),
        HostInt32[NodeDim, Literal[4]],
        "node_period_shifts",
    )
    if (
        axes_.shape != (count + 1,)
        or values_.shape != (count + 1,)
        or pieces_.shape != (count + 1, 2)
        or references_.shape != (count + 1,)
        or shifts_.shape != (count + 1, 4)
        or np.any((axes_ < 0) | (axes_ > 3))
        or not np.all(np.isfinite(values_))
        or np.any(references_ < 0)
        or np.any(references_ > np.arange(count + 1))
    ):
        raise ValueError(
            "IntersectionCurve.node_constraints must agree with the atlas dimensions and ordering; got inconsistent node dimensions or ordering"
        )
    if closed and references_[-1] != 0:
        raise ValueError(
            f"IntersectionCurve.node_references must reference node zero at the closed end; got {references_[-1]!r}"
        )
    return _AtlasNodes(
        axes=axes_, values=values_, pieces=pieces_, references=references_, shifts=shifts_
    )


type GeometryDefinition = (
    AbstractCurve
    | AbstractSurfacePatch
    | AbstractTrimCurve
    | IntersectionCurve
    | SurfaceRegion
)


class IntersectionCurvePoint(CurveCoordinateResult):
    """Coupled evaluation of an intersection curve from one chart solution."""

    point: Array
    first_parameters: Array
    second_parameters: Array
    tangent: Array
    parameter_bound: Array
    gap: Array
    chart: Array


_INTERSECTION_HOST_BANKS = (
    "chart_axes", "chart_start", "chart_end", "chart_pieces", "box_lower", "box_upper",
    "preconditioners", "contraction", "certified", "transition_shifts", "node_axes",
    "node_values", "node_pieces", "node_references", "node_period_shifts", "node_lower",
    "node_upper", "nodes_certified", "chart_start_lower", "chart_start_upper",
    "chart_end_lower", "chart_end_upper", "transitions_certified",
)


class IntersectionCurve(
    StrictModule, CoefficientEnclosureSource, RootCurveCoordinateSource
):
    """Represent one branch of the intersection of two generating surfaces.

    Chart ``k`` covers ``tau`` in ``[k, k + 1]``. Its graph coordinate
    interpolates shared source nodes; ``chart_start`` and ``chart_end`` are
    bounded numerical representatives. Node constraints fix one source
    coordinate of the pair and certify the other three. Node references retain
    symbolic periodic transport. ``fully_certified`` requires chart, shared-node,
    and transition certificates. A closed final node references the first
    source root.
    """

    __strict_contract__ = True

    first: SurfaceRegion
    second: SurfaceRegion
    chart_axes: HostInt32[ChartDim]
    chart_start: HostFloat64[ChartDim, Literal[4]]
    chart_end: HostFloat64[ChartDim, Literal[4]]
    chart_pieces: HostInt32[ChartDim, Literal[2]]
    _pair_addresses: tuple[tuple[int, int], ...] = eqx.field(static=True)
    _pair_pins: tuple[tuple[_SpanPins, _SpanPins], ...] = eqx.field(static=True)
    _chart_selector: tuple[int, ...] = eqx.field(static=True)
    box_lower: HostFloat64[ChartDim, Literal[4]]
    box_upper: HostFloat64[ChartDim, Literal[4]]
    preconditioners: HostFloat64[ChartDim, Literal[3], Literal[3]]
    contraction: HostFloat64[ChartDim]
    certified: HostBool[ChartDim]
    transition_shifts: HostFloat64[ChartDim, Literal[4]]
    node_axes: HostInt32[NodeDim]
    node_values: HostFloat64[NodeDim]
    node_pieces: HostInt32[NodeDim, Literal[2]]
    node_references: HostInt32[NodeDim]
    node_period_shifts: HostInt32[NodeDim, Literal[4]]
    node_lower: HostFloat64[NodeDim, Literal[4]]
    node_upper: HostFloat64[NodeDim, Literal[4]]
    nodes_certified: HostBool[NodeDim]
    chart_start_lower: HostFloat64[ChartDim, Literal[4]]
    chart_start_upper: HostFloat64[ChartDim, Literal[4]]
    chart_end_lower: HostFloat64[ChartDim, Literal[4]]
    chart_end_upper: HostFloat64[ChartDim, Literal[4]]
    transitions_certified: HostBool[ChartDim]
    _qualified: bool = eqx.field(static=True)
    closed: bool = eqx.field(static=True)
    start_kind: IntersectionEndpointKind = eqx.field(static=True)
    end_kind: IntersectionEndpointKind = eqx.field(static=True)
    branch_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        first: SurfaceRegion,
        second: SurfaceRegion,
        *,
        chart_axes: ArrayLike,
        chart_start: ArrayLike,
        chart_end: ArrayLike,
        chart_pieces: ArrayLike,
        box_lower: ArrayLike,
        box_upper: ArrayLike,
        preconditioners: ArrayLike,
        contraction: ArrayLike,
        certified: ArrayLike,
        transition_shifts: ArrayLike,
        closed: bool,
        start_kind: IntersectionEndpointKind,
        end_kind: IntersectionEndpointKind,
        node_axes: ArrayLike | None = None,
        node_values: ArrayLike | None = None,
        node_pieces: ArrayLike | None = None,
        node_references: ArrayLike | None = None,
        node_period_shifts: ArrayLike | None = None,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> None:
        if preparations is None:
            preparations = TrimRequestPreparations()
        with coordinate_source_request(
            (
                first,
                second,
                chart_axes,
                chart_start,
                chart_end,
                chart_pieces,
                box_lower,
                box_upper,
                preconditioners,
                contraction,
                certified,
                transition_shifts,
                closed,
                start_kind,
                end_kind,
                node_axes,
                node_values,
                node_pieces,
                node_references,
                node_period_shifts,
            ),
            budget=budget,
        ) as budget:
            atlas = _prepare_atlas_input(
                chart_axes,
                chart_start,
                chart_end,
                chart_pieces,
                box_lower,
                box_upper,
                preconditioners,
                contraction,
                certified,
                transition_shifts,
                closed,
                start_kind,
                end_kind,
            )
            self.first = first
            self.second = second
            self.chart_axes = atlas.axes
            self.chart_start = atlas.start
            self.chart_end = atlas.end
            self.chart_pieces = atlas.pieces
            self.box_lower = atlas.lower
            self.box_upper = atlas.upper
            self.preconditioners = atlas.preconditioners
            self.contraction = atlas.contraction
            self.certified = atlas.certified
            self.transition_shifts = atlas.shifts
            self.closed = bool(closed)
            self.start_kind = atlas.start_kind
            self.end_kind = atlas.end_kind
            addresses = _prepare_atlas_addresses(
                first, second, atlas.pieces, budget=budget
            )
            self._pair_addresses = addresses.addresses
            self._pair_pins = addresses.pins
            self._chart_selector = addresses.selector
            nodes = _prepare_atlas_nodes(
                atlas,
                first,
                second,
                closed,
                node_axes,
                node_values,
                node_pieces,
                node_references,
                node_period_shifts,
            )
            self.node_axes = nodes.axes
            self.node_values = nodes.values
            self.node_pieces = nodes.pieces
            self.node_references = nodes.references
            self.node_period_shifts = nodes.shifts
            self._certify_nodes(budget=budget)
            self._verify_chart_witnesses(budget=budget)
            # Retained numeric banks become immutable bytes with unchanged
            # values, dtype and shape, so carrier identities of prepared source
            # maps can be memoized instead of rehashed on every source query.
            # Each snapshot copy is admitted before it is allocated.
            budget.reserve(
                len(_INTERSECTION_HOST_BANKS),
                sum(getattr(self, name).nbytes for name in _INTERSECTION_HOST_BANKS),
            )
            for name in _INTERSECTION_HOST_BANKS:
                setattr(self, name, immutable_host_array(getattr(self, name)))
            self._qualified = bool(
                np.all(self.certified)
                and np.all(self.nodes_certified)
                and np.all(self.transitions_certified)
            )
            self.branch_id = self._branch_fingerprint()

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[Self],
        value: IntersectionCurve,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> Self:
        """Rebuild chart and node certificates from retained construction inputs."""
        return cls(
            value.first,
            value.second,
            chart_axes=value.chart_axes,
            chart_start=value.chart_start,
            chart_end=value.chart_end,
            chart_pieces=value.chart_pieces,
            box_lower=value.box_lower,
            box_upper=value.box_upper,
            preconditioners=value.preconditioners,
            contraction=value.contraction,
            certified=value.certified,
            transition_shifts=value.transition_shifts,
            closed=value.closed,
            start_kind=value.start_kind,
            end_kind=value.end_kind,
            node_axes=value.node_axes,
            node_values=value.node_values,
            node_pieces=value.node_pieces,
            node_references=value.node_references,
            node_period_shifts=value.node_period_shifts,
            budget=budget,
        )

    def _branch_fingerprint(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "intersection-curve",
                "first": encode_geometry(self.first.patch),
                "first_box": [list(self.first.lower), list(self.first.upper)],
                "second": encode_geometry(self.second.patch),
                "second_box": [list(self.second.lower), list(self.second.upper)],
                "atlas": array_tree_fingerprint(
                    (
                        self.chart_axes,
                        self.chart_start,
                        self.chart_end,
                        self.chart_pieces,
                        self.box_lower,
                        self.box_upper,
                        self.preconditioners,
                        self.contraction,
                        self.certified,
                        self.transition_shifts,
                        self.node_axes,
                        self.node_values,
                        self.node_pieces,
                        self.node_references,
                        self.node_period_shifts,
                    )
                ),
                "closed": self.closed,
                "endpoints": [self.start_kind, self.end_kind],
            }
        )

    @property
    def num_charts(self) -> int:
        return self.chart_axes.shape[0]

    @property
    def parameter_interval(self) -> tuple[float, float]:
        return 0.0, float(self.num_charts)

    @property
    def parameter_domain(self) -> tuple[float, float]:
        return self.parameter_interval

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def period(self) -> float | None:
        return float(self.num_charts) if self.closed else None

    def native_period_symbol(self) -> str | None:
        """Return no analytic period symbol for the continuation-chart scalar."""
        return None

    def coefficient_enclosures(
        self,
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
        """Return coefficient intervals from both generating surface definitions."""
        return (*coefficient_enclosures(self.first), *coefficient_enclosures(self.second))

    def source_pieces(
        self,
        first: float,
        last: float,
        /,
        *,
        coordinate_budget: CoordinateEnclosureBudget,
    ) -> tuple[CurvePiece[IntersectionCurve], ...]:
        """Retain the coupled branch on a closed source parameter interval."""
        if not (
            math.isfinite(first)
            and math.isfinite(last)
            and 0 <= first <= last <= self.num_charts
        ):
            raise ValueError(
                f"IntersectionCurve.interval must lie inside the source parameter domain; got {(first, last)!r}"
            )
        return (CurvePiece(self, first, last, 0),)

    def validate_range(self, first: float, last: float, /) -> tuple[float, float]:
        first_, last_ = float(first), float(last)
        if not (
            math.isfinite(first_)
            and math.isfinite(last_)
            and 0 <= first_ < last_ <= self.num_charts
        ):
            raise ValueError(
                f"IntersectionCurve.interval must lie inside the parameter domain; got {(first_, last_)!r}"
            )
        return first_, last_

    def is_c1_on(
        self,
        first: float,
        last: float,
        /,
        *,
        endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None] | None = None,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> bool:
        """Only one smooth source chart proves a jet; C0 nodes do not prove C1."""
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.is_c1_on(
                    first,
                    last,
                    endpoint_roots=endpoint_roots,
                    preparations=preparations,
                    budget=budget,
                )
        first_, last_, extended = self._rooted_query_range(
            first, last, endpoint_roots, preparations=preparations, budget=budget
        )
        if extended:
            boxes = self.parameter_enclosures(
                first_,
                last_,
                source_extension=True,
                preparations=preparations,
                budget=budget,
            )
            if not np.all(np.isfinite(boxes)):
                raise ValueError(
                    "IntersectionCurve.rooted_range must remain inside certified chart support; got an unsupported rooted range"
                )
            if any(
                not self.first.patch.is_c1_on(
                    box[:, :2], preparations=preparations, budget=budget
                )
                or not self.second.patch.is_c1_on(
                    box[:, 2:], preparations=preparations, budget=budget
                )
                for box in boxes
            ):
                return False
        return _intersection_is_c1(
            self,
            max(0.0, min(first_, self.num_charts)),
            max(0.0, min(last_, self.num_charts)),
            preparations=preparations,
            budget=budget,
        )

    def derivative_bounds(
        self,
        first: float,
        last: float,
        /,
        *,
        order: int = 1,
        endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None] | None = None,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
        restriction_bank: BernsteinRestrictionBank | None = None,
        meridian_jets: list[MeridianOverlaps] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interval implicit graph jets; rooted extensions retain source definition."""
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.derivative_bounds(
                    first,
                    last,
                    order=order,
                    endpoint_roots=endpoint_roots,
                    preparations=preparations,
                    budget=budget,
                    restriction_bank=restriction_bank,
                    meridian_jets=meridian_jets,
                )
        first_, last_, extended = self._rooted_query_range(
            first, last, endpoint_roots, preparations=preparations, budget=budget
        )
        result = intersection_derivative_bounds(
            self,
            first_,
            last_,
            order,
            None,
            source_extension=extended,
            preparations=preparations,
            budget=budget,
        )
        if extended and not np.all(
            np.isfinite(
                self.parameter_enclosures(
                    first_,
                    last_,
                    source_extension=True,
                    preparations=preparations,
                    budget=budget,
                )
            )
        ):
            raise ValueError(
                "IntersectionCurve.rooted_range must remain inside certified chart support; got an unsupported rooted range"
            )
        return result

    def bounding_box(
        self,
        first: float,
        last: float,
        /,
        *,
        endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None] | None = None,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        """Enclose a branch range; outside-domain uncertainty needs its source root."""
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.bounding_box(
                    first,
                    last,
                    endpoint_roots=endpoint_roots,
                    preparations=preparations,
                    budget=budget,
                )
        first_, last_, extended = self._rooted_query_range(
            first, last, endpoint_roots, preparations=preparations, budget=budget
        )
        boxes = self.parameter_enclosures(
            first_,
            last_,
            source_extension=extended,
            preparations=preparations,
            budget=budget,
        )
        if not np.all(np.isfinite(boxes)):
            raise ValueError(
                "IntersectionCurve.rooted_range must remain inside certified chart support; got an unsupported rooted range"
            )
        preparation = _original_trim_intersection_preparation(
            self, preparations=preparations, budget=budget
        )
        systems, selector = (
            self._pairs(budget=budget) if preparation is None else preparation.pairs
        )
        head = max(0, min(int(math.floor(first_)), self.num_charts - 1))
        images = []

        for offset, box in enumerate(boxes):
            system = systems[int(selector[head + offset])]
            prepared = (
                prepare_interval_function(
                    system.first.evaluate,
                    2,
                    batch_capacity=1,
                    constant_bounds=coefficient_enclosures(system),
                )
                if preparation is None
                else preparation.program(
                    head + offset, "spatial-value", system.first.evaluate, 2
                )
            )
            lower, upper = prepared.evaluate(box[None, 0, :2], box[None, 1, :2])
            images.append(np.stack((lower[0], upper[0])))
        images_ = np.asarray(images)
        return np.stack((np.min(images_[:, 0], axis=0), np.max(images_[:, 1], axis=0)))

    def _rooted_query_range(
        self,
        first: float,
        last: float,
        endpoints: tuple[RootEndpoint | None, RootEndpoint | None] | None,
        /,
        *,
        domain_lower: float = 0.0,
        domain_upper: float | None = None,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[float, float, bool]:
        """Authorize only numeric endpoint uncertainty on the source scalar source."""
        first_, last_ = float(first), float(last)
        if not (math.isfinite(first_) and math.isfinite(last_) and first_ <= last_):
            raise ValueError(
                f"IntersectionCurve.interval must be finite and ordered; got {(first_, last_)!r}"
            )
        domain_upper = float(self.num_charts) if domain_upper is None else domain_upper
        if not 0.0 <= domain_lower < domain_upper <= self.num_charts:
            raise ValueError(
                f"IntersectionCurve.chart_range must remain inside the branch domain; got {(domain_lower, domain_upper)!r}"
            )
        if domain_lower <= first_ <= last_ <= domain_upper:
            return first_, last_, False

        if endpoints is None or len(endpoints) != 2:
            raise ValueError(
                "IntersectionCurve.endpoints must retain roots for extensions outside the closed domain; got missing extension roots"
            )
        for index, outside in (
            (0, first_ < domain_lower),
            (1, last_ > domain_upper),
        ):
            if not outside:
                continue
            endpoint = endpoints[index]
            if not isinstance(
                endpoint,
                (
                    intersection_owner.BranchRootEndpoint,
                    intersection_owner.TrimRootEndpoint,
                ),
            ):
                raise ValueError(
                    f"IntersectionCurve.endpoints must retain a source root for each extension; got {type(endpoint).__name__!r}"
                )
            if _root_parameter_affine(endpoint) != (Fraction(1), Fraction(0)):
                raise ValueError(
                    "IntersectionCurve.endpoints must preserve the source scalar atom; got a changed scalar atom"
                )
            carrier = endpoint.carrier
            if isinstance(carrier, IntersectionCurve):
                branch = carrier
            else:
                source, _, _ = _pcurve_affine_source(carrier)
                branch = (
                    source.curve
                    if isinstance(source, IntersectionPCurve) and not source.reversed
                    else None
                )
            if branch is None or branch.branch_id != self.branch_id:
                raise ValueError(
                    "IntersectionCurve.endpoints must bind the source branch; got a different endpoint branch"
                )
            lower, upper = endpoint.parameter_enclosure(
                maximum_steps=0, preparations=preparations, budget=budget
            )
            interval_lower, interval_upper = (
                (first_, min(last_, domain_lower))
                if index == 0
                else (max(first_, domain_upper), last_)
            )
            if not lower <= interval_lower <= interval_upper <= upper:
                raise ValueError(
                    f"IntersectionCurve.rooted_range must remain inside the endpoint scalar enclosure; got {(interval_lower, interval_upper)!r}"
                )
        return first_, last_, True

    @property
    def fully_certified(self) -> bool:
        return self._qualified

    def shift_parameter_bounds(
        self, shifts: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        """Enclose integer deck shifts in the four source parameter coordinates."""
        lower, upper = np.zeros(4), np.zeros(4)
        for axis, (shift, period) in enumerate(
            zip(
                shifts,
                (*self.first.patch.periods, *self.second.patch.periods),
                strict=True,
            )
        ):
            if not shift:
                continue
            if period is None:
                raise ValueError(
                    f"IntersectionCurve.period_shifts must use a declared source period; got {axis!r}"
                )
            if period == 2 * math.pi:
                bounds = (
                    np.nextafter(2 * math.pi, -np.inf),
                    np.nextafter(2 * math.pi, np.inf),
                )
            else:
                bounds = (period, period)
            values = (int(shift) * bounds[0], int(shift) * bounds[1])
            lower[axis], upper[axis] = (
                np.nextafter(min(values), -np.inf),
                np.nextafter(max(values), np.inf),
            )
        return lower, upper

    def transition_turns(self) -> np.ndarray:
        """Integer native-period deck shift certified at each chart transition.

        Row ``k`` maps chart ``k`` coordinates to chart ``k + 1`` coordinates
        (the final row closes a closed branch). Representative shifts round to
        whole periods on periodic axes; nonperiodic axes have no deck shift.
        """
        turns = np.zeros((self.num_charts, 4), dtype=np.int32)
        for axis, period in enumerate(
            (*self.first.patch.periods, *self.second.patch.periods)
        ):
            if period is not None:
                turns[:, axis] = np.rint(
                    np.asarray(self.transition_shifts[:, axis], dtype=np.float64)
                    / period
                ).astype(np.int32)
        return turns

    def _verify_chart_witnesses(
        self,
        *,
        budget: CoordinateEnclosureBudget,
        preparations: TrimRequestPreparations | None = None,
    ) -> None:
        """Reprove every claimed chart over the source coefficient box."""

        systems, selector = self._pairs(budget=budget)
        for chart in range(self.num_charts):
            if not bool(self.certified[chart]):
                continue
            prepared = intersection_owner.prepare(
                systems[int(selector[chart])], 4, 1, preparations=preparations
            )
            result = krawczyk(
                prepared,
                self.box_lower[chart][None],
                self.box_upper[chart][None],
                parameter_axis=int(self.chart_axes[chart]),
                budget=budget,
                preparations=preparations,
            )
            if not bool(result.certified[0]):
                raise ValueError(
                    f"IntersectionCurve.chart_certificate must prove graph inclusion; got uncertified chart {chart}"
                )
            # Bounds evaluated later with this stored preconditioner must also
            # be covered by the revalidated certificate, not an arbitrary payload.
            if (
                not np.array_equal(self.preconditioners[chart], result.preconditioner[0])
                or self.contraction[chart] < result.contraction[0]
            ):
                raise ValueError(
                    f"IntersectionCurve.chart_bounds must match the certified preconditioner and cover contraction; got inconsistent chart {chart}"
                )

    def _certify_nodes(
        self,
        *,
        budget: CoordinateEnclosureBudget,
        preparations: TrimRequestPreparations | None = None,
    ) -> None:
        """Use shared source root constraints, never equality of floating nodes."""

        first = {
            piece.index: piece for piece in surface_pieces(self.first, budget=budget)
        }
        second = {
            piece.index: piece for piece in surface_pieces(self.second, budget=budget)
        }
        count = self.num_charts
        lower, upper = np.empty((count + 1, 4)), np.empty((count + 1, 4))
        certified = np.zeros(count + 1, dtype=np.bool_)
        for node in range(count + 1):
            reference = int(self.node_references[node])
            if reference != node:
                delta_lower, delta_upper = self.shift_parameter_bounds(
                    self.node_period_shifts[node]
                )
                lower[node] = np.nextafter(lower[reference] + delta_lower, -np.inf)
                upper[node] = np.nextafter(upper[reference] + delta_upper, np.inf)
                certified[node] = certified[reference]
                continue
            chart = max(0, node - 1)
            a, b = (int(index) for index in self.node_pieces[node])
            if a not in first or b not in second:
                raise ValueError(
                    f"IntersectionCurve.node_pieces must reference admitted source spans; got {(a, b)!r}"
                )
            prepared = intersection_owner.prepare(
                SurfacePairSystem(first[a].evaluator, second[b].evaluator),
                4,
                1,
                preparations=preparations,
            )
            # A boundary node can sit near the end of the marching chart's
            # graph-coordinate box. Re-preconditioning that whole asymmetric
            # box for the boundary coordinate is unnecessarily inconclusive.
            # Isolate near the already corrected representative, wholly inside
            # the existing support; the new Krawczyk test is still the proof.
            center = self.chart_start[0] if node == 0 else self.chart_end[node - 1]
            radius = np.maximum(
                0.01 * (self.box_upper[chart] - self.box_lower[chart]),
                256 * np.finfo(np.float64).eps * (1 + np.abs(center)),
            )
            lo = np.maximum(self.box_lower[chart], np.nextafter(center - radius, -np.inf))
            hi = np.minimum(self.box_upper[chart], np.nextafter(center + radius, np.inf))
            axis = int(self.node_axes[node])
            lo[axis] = hi[axis] = self.node_values[node]
            result = krawczyk(
                prepared,
                lo[None],
                hi[None],
                parameter_axis=axis,
                budget=budget,
                preparations=preparations,
            )
            certified[node] = bool(result.certified[0])
            if certified[node]:
                free = [index for index in range(4) if index != axis]
                for _ in range(16):
                    result = krawczyk(
                        prepared,
                        lo[None],
                        hi[None],
                        parameter_axis=axis,
                        budget=budget,
                        preparations=preparations,
                    )
                    new_lo, new_hi = lo.copy(), hi.copy()
                    new_lo[free] = np.maximum(lo[free], result.lower[0])
                    new_hi[free] = np.minimum(hi[free], result.upper[0])
                    if np.any(new_lo > new_hi):
                        certified[node] = False
                        break
                    unchanged = np.array_equal(new_lo, lo) and np.array_equal(new_hi, hi)
                    lo, hi = new_lo, new_hi
                    if unchanged:
                        break
            lower[node], upper[node] = lo, hi
        self.node_lower, self.node_upper, self.nodes_certified = lower, upper, certified
        start_lower, start_upper = lower[:-1].copy(), upper[:-1].copy()
        turns = self.transition_turns()
        for chart in range(1, count):
            shifts = turns[chart - 1]
            piece_transition = not np.array_equal(
                self.chart_pieces[chart - 1],
                self.chart_pieces[chart],
            )
            if piece_transition and not np.any(shifts):
                continue
            delta_lower, delta_upper = self.shift_parameter_bounds(shifts)
            start_lower[chart] = np.nextafter(
                start_lower[chart] + delta_lower,
                -np.inf,
            )
            start_upper[chart] = np.nextafter(
                start_upper[chart] + delta_upper,
                np.inf,
            )
        self.chart_start_lower, self.chart_start_upper = start_lower, start_upper
        self.chart_end_lower, self.chart_end_upper = lower[1:].copy(), upper[1:].copy()
        valid = np.ones(count, dtype=np.bool_)
        for chart in range(count):
            axis = int(self.chart_axes[chart])
            advance_lower = lower[chart + 1, axis] - start_upper[chart, axis]
            advance_upper = upper[chart + 1, axis] - start_lower[chart, axis]
            valid[chart] = bool(
                certified[chart]
                and certified[chart + 1]
                and np.all(start_lower[chart] >= self.box_lower[chart])
                and np.all(start_upper[chart] <= self.box_upper[chart])
                and np.all(lower[chart + 1] >= self.box_lower[chart])
                and np.all(upper[chart + 1] <= self.box_upper[chart])
                and (advance_lower > 0 or advance_upper < 0)
            )
        self.transitions_certified = valid

    def parameter_enclosures(
        self,
        first: float,
        last: float,
        /,
        *,
        minimum_chart: int = 0,
        maximum_chart: int | None = None,
        source_extension: bool = False,
        _preparation: IntersectionJetPreparation | None = None,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        """Source-faithful coupled chart boxes for a closed parameter subrange.

        ``source_extension`` is an endpoint-root query of the defining graph,
        not a larger represented curve domain. It requires the extended chart
        coordinate to remain in its certified atlas box; otherwise the result
        is unbounded and cannot certify an event.
        """
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.parameter_enclosures(
                    first,
                    last,
                    minimum_chart=minimum_chart,
                    maximum_chart=maximum_chart,
                    source_extension=source_extension,
                    _preparation=_preparation,
                    preparations=preparations,
                    budget=budget,
                )
        if not self.fully_certified:
            raise ValueError(
                "IntersectionCurve.atlas must have certified charts, nodes and transitions; got an uncertified atlas"
            )
        if _preparation is None:
            _preparation = _original_trim_intersection_preparation(
                self, preparations=preparations, budget=budget
            )
        elif (
            _preparation.curve is not self
            and _preparation
            is not _original_trim_intersection_preparation(
                self, preparations=preparations, budget=budget
            )
        ):
            raise ValueError(
                "IntersectionCurve.preparation must retain the source curve; got a different prepared source"
            )
        return _chart_enclosures(
            self,
            ((first, last, minimum_chart, maximum_chart, source_extension),),
            _preparation,
            preparations=preparations,
            budget=budget,
        )[0]

    def parameter_enclosure_batch(
        self,
        requests: Sequence[tuple[float, float, int, int | None, bool]],
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> list[np.ndarray]:
        """``parameter_enclosures`` of independent requests in one chart pass.

        Each request is ``(first, last, minimum_chart, maximum_chart,
        source_extension)`` with the single-query meaning and result. Shared
        chart queries contract once; distinct ones share batched Krawczyk
        evaluations of the same chart program, row by row.
        """
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    requests,
                )
            ) as budget:
                return self.parameter_enclosure_batch(
                    requests, preparations=preparations, budget=budget
                )
        if not self.fully_certified:
            raise ValueError(
                "IntersectionCurve.atlas must have certified charts, nodes and transitions; got an uncertified atlas"
            )
        return _chart_enclosures(
            self,
            requests,
            _original_trim_intersection_preparation(
                self, preparations=preparations, budget=budget
            ),
            preparations=preparations,
            budget=budget,
        )

    def _pairs(
        self, *, budget: CoordinateEnclosureBudget
    ) -> tuple[tuple[SurfacePairSystem, ...], np.ndarray]:
        """Host certification pieces bound to the immutable chart addressing.

        Interval programs need gather-free Bernstein pieces with source
        coefficient enclosures, extracted on the host from the current concrete
        ``first``/``second`` leaves at every use (never cached).
        """
        for first_pins, second_pins in self._pair_pins:
            _check_span_topology(self.first.patch, first_pins)
            _check_span_topology(self.second.patch, second_pins)
        first = {
            piece.index: piece.evaluator
            for piece in surface_pieces(self.first, budget=budget)
        }
        second = {
            piece.index: piece.evaluator
            for piece in surface_pieces(self.second, budget=budget)
        }
        if any(a not in first or b not in second for a, b in self._pair_addresses):
            raise ValueError(
                f"IntersectionCurve.chart_pieces must remain admitted by the source pieces; got {self._pair_addresses!r}"
            )
        systems = tuple(
            SurfacePairSystem(first[a], second[b]) for a, b in self._pair_addresses
        )
        return systems, np.asarray(self._chart_selector, dtype=np.int32)

    def _evaluation_pairs(self) -> tuple[tuple[SurfacePairSystem, ...], np.ndarray]:
        """Live source equations bound to immutable host chart/span addresses.

        Each system evaluates the live ``first``/``second`` source pytrees, with
        every Bernstein piece replaced by its pinned deterministic knot span, so
        traced control points, weights, knots and analytic parameters reach the
        point, both p-curves and the jet without host extraction or copies.
        """
        systems = tuple(
            SurfacePairSystem(
                _bind_span_pins(self.first.patch, first_pins),
                _bind_span_pins(self.second.patch, second_pins),
            )
            for first_pins, second_pins in self._pair_pins
        )
        return systems, np.asarray(self._chart_selector, dtype=np.int32)

    def evaluate(
        self,
        parameters: ArrayLike,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> IntersectionCurvePoint:
        """Evaluate points, both p-curves, the unit tangent and numerical bounds.

        ``parameter_bound`` is a conservative max-norm enclosure distance from
        the returned parameters to the branch point on the same chart line.
        It uses the chart box, not a floating residual as a proof of accuracy;
        ``gap`` is the numerical value of ``|S1 - S2|``.
        """
        with coordinate_source_request(
            (
                self,
                parameters,
            ),
            budget=budget,
        ) as budget:
            return self._evaluate_in_chart_range(
                parameters,
                0,
                self.num_charts - 1,
                preparations=preparations,
                budget=budget,
            )

    def _evaluate_in_chart_range(
        self,
        parameters: ArrayLike,
        minimum_chart: int,
        maximum_chart: int,
        /,
        *,
        certify: bool = True,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None,
    ) -> IntersectionCurvePoint:
        tau = jnp.asarray(parameters, dtype=jnp.float64)
        # Concrete queries in an explicit request reuse its prepared chart
        # rows; traced parameters and unprepared curves build them live.
        atlas = (
            None
            if isinstance(tau, Tracer)
            else _request_chart_atlas(
                self,
                minimum_chart,
                maximum_chart,
                preparations=preparations,
                budget=budget,
            )
        )
        if atlas is None:
            systems, selector = self._evaluation_pairs()
            atlas = _ChartAtlas(
                systems,
                *_chart_atlas_rows(self, selector),
                *_chart_atlas_range(minimum_chart, maximum_chart, selector.shape[0]),
            )
        if isinstance(tau, Tracer):
            result = _evaluate_atlas(atlas, tau.reshape((-1,)))
            result = jax.tree_util.tree_map(
                lambda leaf: leaf.reshape(tau.shape + leaf.shape[1:]), result
            )
        else:
            # Concrete queries share power-of-four compiled lane counts; the
            # padding repeats the last independent lane and is discarded on
            # the host, so no per-count slicing program is compiled.
            flat = np.asarray(tau, dtype=np.float64).reshape(-1)
            count = flat.size
            lanes = _query_capacity(count)
            padded = _evaluate_atlas(
                atlas,
                jnp.asarray(np.concatenate((flat, np.repeat(flat[-1:], lanes - count)))),
            )

            def unpad(leaf: Array) -> Array:
                shape = tau.shape + leaf.shape[1:]
                if isinstance(leaf, Tracer):
                    return leaf[:count].reshape(shape)
                return jnp.asarray(np.asarray(leaf)[:count].reshape(shape))

            result = jax.tree_util.tree_map(unpad, padded)
        if certify and not isinstance(tau, Tracer) and self.fully_certified:
            values = np.asarray(tau).reshape(-1)
            if (
                np.any(~np.isfinite(values))
                or np.any(values < 0)
                or np.any(values > self.num_charts)
            ):
                raise ValueError(
                    "IntersectionCurve.parameters must lie inside the curve domain; got parameters outside the domain"
                )
        # Host box certification needs concrete source leaves; a traced source
        # (or parameter) keeps the atlas-box bound computed on the device.
        if (
            certify
            and not isinstance(result.first_parameters, Tracer)
            and self.fully_certified
        ):
            if preparations is None:
                preparations = TrimRequestPreparations()
            values = np.asarray(tau).reshape(-1)
            coupled = np.concatenate(
                (
                    np.asarray(result.first_parameters).reshape((-1, 2)),
                    np.asarray(result.second_parameters).reshape((-1, 2)),
                ),
                axis=1,
            )
            bounds = []
            for start in range(0, values.size, _CHART_BATCH):
                batch = values[start : start + _CHART_BATCH]
                plans = self.parameter_enclosure_batch(
                    [
                        (
                            float(value),
                            float(value),
                            minimum_chart,
                            maximum_chart,
                            False,
                        )
                        for value in batch
                    ],
                    preparations=preparations,
                    budget=budget,
                )
                for boxes, point in zip(
                    plans, coupled[start : start + _CHART_BATCH], strict=True
                ):
                    distances = np.nextafter(np.abs(boxes - point), np.inf)
                    bounds.append(np.nextafter(np.max(distances), np.inf))
            result = eqx.tree_at(
                lambda item: item.parameter_bound,
                result,
                jnp.asarray(bounds, dtype=jnp.float64).reshape(tau.shape),
            )
        return result

    def payload(self) -> dict[str, Any]:
        """Lossless deterministic record: generating surfaces, boxes and the atlas."""
        return {
            "kind": "intersection-curve",
            "first": encode_geometry(self.first.patch),
            "first_box": [list(self.first.lower), list(self.first.upper)],
            "second": encode_geometry(self.second.patch),
            "second_box": [list(self.second.lower), list(self.second.upper)],
            "chart_axes": self.chart_axes.tolist(),
            "chart_start": encode_geometry(self.chart_start),
            "chart_end": encode_geometry(self.chart_end),
            "chart_pieces": self.chart_pieces.tolist(),
            "box_lower": encode_geometry(self.box_lower),
            "box_upper": encode_geometry(self.box_upper),
            "preconditioners": encode_geometry(self.preconditioners),
            "contraction": encode_geometry(self.contraction),
            "certified": self.certified.tolist(),
            "transition_shifts": encode_geometry(self.transition_shifts),
            "closed": self.closed,
            "start_kind": self.start_kind,
            "end_kind": self.end_kind,
            "branch_id": self.branch_id,
            "node_axes": self.node_axes.tolist(),
            "node_values": encode_geometry(self.node_values),
            "node_pieces": self.node_pieces.tolist(),
            "node_references": self.node_references.tolist(),
            "node_period_shifts": self.node_period_shifts.tolist(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], /) -> IntersectionCurve:
        """Restore a curve and verify its branch identity."""
        if payload.get("kind") != "intersection-curve":
            raise ValueError(
                "IntersectionCurve.payload must be an intersection-curve record; got a different record kind"
            )
        expected = {
            "kind",
            "first",
            "first_box",
            "second",
            "second_box",
            "chart_axes",
            "chart_start",
            "chart_end",
            "chart_pieces",
            "box_lower",
            "box_upper",
            "preconditioners",
            "contraction",
            "certified",
            "transition_shifts",
            "closed",
            "start_kind",
            "end_kind",
            "branch_id",
            "node_axes",
            "node_values",
            "node_pieces",
            "node_references",
            "node_period_shifts",
        }
        if set(payload) != expected or type(payload["closed"]) is not bool:
            raise ValueError(
                "IntersectionCurve.payload must match the declared field set and types; got a field-set or type mismatch"
            )
        for name in ("chart_axes", "node_axes", "node_references"):
            if not isinstance(payload[name], list) or any(
                type(value) is not int for value in payload[name]
            ):
                raise ValueError(
                    f"{'IntersectionCurve.' + name} must contain an integer list; got {payload[name]!r}"
                )
        for name, width in (
            ("chart_pieces", 2),
            ("node_pieces", 2),
            ("node_period_shifts", 4),
        ):
            if not isinstance(payload[name], list) or any(
                not isinstance(row, list)
                or len(row) != width
                or any(type(value) is not int for value in row)
                for row in payload[name]
            ):
                raise ValueError(
                    f"{'IntersectionCurve.' + name} must have integer rows of width {width}; got {payload[name]!r}"
                )
        if not isinstance(payload["certified"], list) or any(
            type(value) is not bool for value in payload["certified"]
        ):
            raise ValueError(
                f"IntersectionCurve.certified must contain a boolean list; got {payload['certified']!r}"
            )
        curve = cls(
            SurfaceRegion(
                decode_geometry(payload["first"]), np.asarray(payload["first_box"])
            ),
            SurfaceRegion(
                decode_geometry(payload["second"]), np.asarray(payload["second_box"])
            ),
            chart_axes=np.asarray(payload["chart_axes"], dtype=np.int32),
            chart_start=decode_geometry(payload["chart_start"]),
            chart_end=decode_geometry(payload["chart_end"]),
            chart_pieces=np.asarray(payload["chart_pieces"], dtype=np.int32),
            box_lower=decode_geometry(payload["box_lower"]),
            box_upper=decode_geometry(payload["box_upper"]),
            preconditioners=decode_geometry(payload["preconditioners"]),
            contraction=decode_geometry(payload["contraction"]),
            certified=np.asarray(payload["certified"], dtype=np.bool_),
            transition_shifts=decode_geometry(payload["transition_shifts"]),
            closed=bool(payload["closed"]),
            start_kind=payload["start_kind"],
            end_kind=payload["end_kind"],
            node_axes=np.asarray(payload["node_axes"], dtype=np.int32),
            node_values=decode_geometry(payload["node_values"]),
            node_pieces=np.asarray(payload["node_pieces"], dtype=np.int32),
            node_references=np.asarray(payload["node_references"], dtype=np.int32),
            node_period_shifts=np.asarray(payload["node_period_shifts"], dtype=np.int32),
        )
        if curve.branch_id != payload["branch_id"]:
            raise ValueError(
                "IntersectionCurve.branch_id must match the restored construction record; got a different restored branch ID"
            )
        return curve

    def p_curve(self, side: IntersectionCurveSide, /) -> IntersectionPCurve:
        """Oriented trim curve of this branch in one generating surface's chart."""
        return IntersectionPCurve(self, side)


type _SurfaceResidualOperand = tuple[Array, tuple[SurfacePairSystem, ...]]


def _surface_residual_branch(
    index: int,
    /,
) -> Callable[[_SurfaceResidualOperand], Array]:
    """Select one dynamic surface-pair system without static array-bearing methods."""

    def residual(operand: _SurfaceResidualOperand, /) -> Array:
        parameters, systems = operand
        return systems[index].residual(parameters)

    return residual


def _chart_capacity(count: int, /) -> int:
    """Power-of-two atlas row capacity, so branches share compiled programs."""
    return max(8, 1 << max(count - 1, 0).bit_length())


def _query_capacity(count: int, /) -> int:
    """Power-of-four concrete query lane capacity (zero stays empty)."""
    lanes = 0 if count == 0 else 4
    while lanes < count:
        lanes *= 4
    return lanes


def _pad_rows(values: ArrayLike, rows: int, /) -> Array | np.ndarray:
    """Repeat the last per-chart row up to ``rows``; padded rows are never selected."""
    if isinstance(values, Tracer):
        array = jnp.asarray(values)
        extra = rows - array.shape[0]
        if extra == 0:
            return array
        return jnp.concatenate((array, jnp.repeat(array[-1:], extra, axis=0)))
    host = np.asarray(values)
    extra = rows - host.shape[0]
    if extra == 0:
        return host
    return np.concatenate((host, np.repeat(host[-1:], extra, axis=0)))


class _ChartAtlas(StrictModule):
    """Device view of one continuation atlas and its defining equations.

    Per-chart rows are padded to a shared capacity; ``charts`` is the true
    chart count and the dynamic chart range never selects a padded row.
    """

    systems: tuple[SurfacePairSystem, ...]
    selector: Array
    axes: Array
    start: Array
    end: Array
    preconditioners: Array
    contraction: Array
    certified: Array
    lower: Array
    upper: Array
    minimum_chart: Array
    maximum_chart: Array
    charts: Array

    def residual(self, chart: Array, parameters: Array, /) -> Array:
        if len(self.systems) == 1:
            return self.systems[0].residual(parameters)
        return jax.lax.switch(
            self.selector[chart],
            tuple(_surface_residual_branch(index) for index in range(len(self.systems))),
            (parameters, self.systems),
        )


# Per-chart banks of ``_ChartAtlas`` in field order after the padded selector.
_CHART_ATLAS_BANKS = (
    "chart_axes", "chart_start", "chart_end", "preconditioners", "contraction",
    "certified", "box_lower", "box_upper",
)
# Declared bound of one retained request atlas record: its field slots, the
# padded-row and bound-layout tuples and its request-table slot.
_PREPARED_ATLAS_RECORD_BYTES = 4096
# Declared bound of one retained chart range: three scalar device leaves, the
# retained ``_ChartAtlas`` shell and the range-table slot.
_PREPARED_ATLAS_RANGE_BYTES = 2048


type _ChartAtlasRows = tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]


def _chart_atlas_rows(
    curve: IntersectionCurve, selector: np.ndarray, /
) -> _ChartAtlasRows:
    """Padded device chart rows in ``_ChartAtlas`` field order after the systems."""
    rows = _chart_capacity(selector.shape[0])
    return (
        jnp.asarray(_pad_rows(selector, rows)),
        jnp.asarray(_pad_rows(curve.chart_axes, rows)),
        jnp.asarray(_pad_rows(curve.chart_start, rows)),
        jnp.asarray(_pad_rows(curve.chart_end, rows)),
        jnp.asarray(_pad_rows(curve.preconditioners, rows)),
        jnp.asarray(_pad_rows(curve.contraction, rows)),
        jnp.asarray(_pad_rows(curve.certified, rows)),
        jnp.asarray(_pad_rows(curve.box_lower, rows)),
        jnp.asarray(_pad_rows(curve.box_upper, rows)),
    )


def _chart_atlas_range(
    minimum_chart: int, maximum_chart: int, charts: int, /
) -> tuple[Array, Array, Array]:
    """Dynamic chart range and true chart count of one ``_ChartAtlas``."""
    return (
        jnp.asarray(minimum_chart, dtype=jnp.int32),
        jnp.asarray(maximum_chart, dtype=jnp.int32),
        jnp.asarray(charts, dtype=jnp.float64),
    )


def _immutable_layouts(
    leaves: Sequence[object], /
) -> tuple[tuple[np.ndarray, tuple[object, ...]], ...] | None:
    """Byte-backed layouts of concrete immutable host leaves, else ``None``.

    Device arrays and scalars are immutable values. A traced leaf, an unknown
    owner, or a caller-mutable host bank (including a read-only view of
    mutable storage) has no reusable layout.
    """
    bound = []
    for leaf in leaves:
        if isinstance(leaf, Tracer):
            return None
        if isinstance(leaf, np.ndarray):
            layout = immutable_host_array_layout(leaf)
            if layout is None:
                return None
            bound.append((leaf, layout))
        elif not isinstance(leaf, (jax.Array, np.generic, bool, int, float, complex)):
            return None
    return tuple(bound)


@dataclass(init=False, eq=False, repr=False)
class PreparedChartAtlas:
    """Request-owned device chart rows of one unchanged concrete curve.

    Without it, ``_evaluate_in_chart_range`` rebuilds these exact padded rows,
    their device copies and the span-pinned source systems on every concrete
    query. They do not depend on the queried parameters: every query still
    solves its own chart roots through ``_evaluate_atlas``. Reuse is bound to
    the retained curve object, its unchanged bank objects and their immutable
    byte-backed layouts, and to the budget that admitted its retained storage.
    Systems are retained only when every source leaf is immutable;
    caller-mutable source leaves rebuild the live systems on every query, and
    mutable or traced banks are never prepared.
    """

    curve: IntersectionCurve
    budget: ReferenceType[CoordinateEnclosureBudget]
    banks: tuple[object, ...]
    bank_layouts: tuple[tuple[np.ndarray, tuple[object, ...]], ...]
    source_layouts: tuple[tuple[np.ndarray, tuple[object, ...]], ...]
    systems: tuple[SurfacePairSystem, ...] | None
    rows: _ChartAtlasRows
    charts: int
    ranges: dict[
        tuple[int, int], tuple[tuple[Array, Array, Array], _ChartAtlas | None]
    ]

    def __init__(
        self,
        curve: IntersectionCurve,
        budget: CoordinateEnclosureBudget,
        banks: tuple[object, ...],
        bank_layouts: tuple[tuple[np.ndarray, tuple[object, ...]], ...],
        source_layouts: tuple[tuple[np.ndarray, tuple[object, ...]], ...],
        systems: tuple[SurfacePairSystem, ...] | None,
        rows: _ChartAtlasRows,
        charts: int,
    ) -> None:
        self.curve = curve
        # A weak edge: the request record never extends its budget's lifetime.
        self.budget = ref(budget)
        self.banks = banks
        self.bank_layouts = bank_layouts
        self.source_layouts = source_layouts
        self.systems = systems
        self.rows = rows
        self.charts = charts
        self.ranges = {}

    @classmethod
    def admit(
        cls, curve: IntersectionCurve, /, *, budget: CoordinateEnclosureBudget
    ) -> PreparedChartAtlas | None:
        """Prepare a concrete curve's rows after admitting their storage.

        Returns ``None`` (the live path) for traced source leaves and for
        traced or caller-mutable banks.
        """
        banks = tuple(getattr(curve, name) for name in _CHART_ATLAS_BANKS)
        bank_layouts = _immutable_layouts(banks)
        if bank_layouts is None:
            return None
        leaves = jax.tree_util.tree_leaves((curve.first.patch, curve.second.patch))
        if any(isinstance(leaf, Tracer) for leaf in leaves):
            return None
        source_layouts = _immutable_layouts(leaves)
        systems = None if source_layouts is None else curve._evaluation_pairs()[0]
        selector = np.asarray(curve._chart_selector, dtype=np.int32)
        rows = _chart_capacity(selector.shape[0])
        row_bytes = rows * (
            selector.dtype.itemsize
            + sum(
                np.dtype(bank.dtype).itemsize * math.prod(bank.shape[1:])
                for bank in banks
            )
        )
        layouts = bank_layouts + (source_layouts or ())
        # Retained rows and record are admitted before any copy exists.
        budget.retain_operand_metadata(
            _PREPARED_ATLAS_RECORD_BYTES
            + row_bytes
            + sum(immutable_host_array_layout_bytes(leaf.ndim) for leaf, _ in layouts)
            + (
                0
                if systems is None
                else declared_object_storage_bytes(jax.tree_util.tree_structure(systems))
            )
        )
        with budget.temporary_scope():
            # Padded host rows, plus host copies of device banks, are released
            # once their device rows exist.
            budget.reserve(
                0,
                row_bytes
                + sum(
                    np.dtype(bank.dtype).itemsize * math.prod(bank.shape)
                    for bank in banks
                    if not isinstance(bank, np.ndarray)
                ),
            )
            device = _chart_atlas_rows(curve, selector)
        if any(isinstance(row, Tracer) for row in device):
            return None
        return cls(
            curve,
            budget,
            banks,
            bank_layouts,
            source_layouts or (),
            systems,
            device,
            selector.shape[0],
        )

    def current(
        self, curve: IntersectionCurve, budget: CoordinateEnclosureBudget, /
    ) -> bool:
        """Same curve, budget, bank objects and immutable layouts; no rescans."""
        return (
            self.curve is curve
            and self.budget() is budget
            and all(
                getattr(curve, name) is bank
                for name, bank in zip(_CHART_ATLAS_BANKS, self.banks, strict=True)
            )
            and all(
                immutable_host_array_layout(leaf) == layout
                for leaf, layout in self.bank_layouts + self.source_layouts
            )
        )

    def atlas(
        self,
        minimum_chart: int,
        maximum_chart: int,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> _ChartAtlas:
        """The exact live ``_ChartAtlas`` of one chart range."""
        key = (minimum_chart, maximum_chart)
        entry = self.ranges.get(key)
        if entry is None:
            budget.retain_operand_metadata(_PREPARED_ATLAS_RANGE_BYTES)
            bounds = _chart_atlas_range(minimum_chart, maximum_chart, self.charts)
            if any(isinstance(bound, Tracer) for bound in bounds):
                systems = (
                    self.curve._evaluation_pairs()[0]
                    if self.systems is None
                    else self.systems
                )
                return _ChartAtlas(systems, *self.rows, *bounds)
            entry = (
                bounds,
                None
                if self.systems is None
                else _ChartAtlas(self.systems, *self.rows, *bounds),
            )
            self.ranges[key] = entry
        bounds, atlas = entry
        if atlas is not None:
            return atlas
        # Caller-mutable source leaves: bind the live source equations now.
        return _ChartAtlas(self.curve._evaluation_pairs()[0], *self.rows, *bounds)


def _request_chart_atlas(
    curve: IntersectionCurve,
    minimum_chart: int,
    maximum_chart: int,
    /,
    *,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget | None,
) -> _ChartAtlas | None:
    """The request's prepared atlas of a concrete curve, or ``None`` (live path)."""
    if (
        preparations is None
        or budget is None
        or type(minimum_chart) is not int
        or type(maximum_chart) is not int
    ):
        return None
    atlases = preparations.chart_atlases
    prepared = atlases.get(id(curve))
    if prepared is None or not prepared.current(curve, budget):
        atlases.pop(id(curve), None)
        prepared = PreparedChartAtlas.admit(curve, budget=budget)
        if prepared is None:
            return None
        atlases[id(curve)] = prepared
    return prepared.atlas(minimum_chart, maximum_chart, budget=budget)


_CHART_ROOT = VectorLocalRootPlan(
    3, maximum_steps=40, tolerance=1.0e-13, plan_id="intersection-curve-chart"
)


@eqx.filter_jit
def _evaluate_atlas(atlas: _ChartAtlas, tau: Array) -> IntersectionCurvePoint:
    """Solve each chart's graph equations and derive the coupled jet and bounds."""
    count = atlas.charts
    permutations = jnp.asarray(_CHART_PERMUTATIONS)

    def one(value: Array) -> IntersectionCurvePoint:
        chart = jnp.clip(
            jnp.floor(value), atlas.minimum_chart, atlas.maximum_chart
        ).astype(jnp.int32)
        local = value - chart
        axis = atlas.axes[chart]
        first_node = atlas.start[chart]
        last_node = atlas.end[chart]
        guess = first_node + local * (last_node - first_node)
        parameter = jnp.take(guess, axis)

        def graph_residual(rest: Array) -> Array:
            return atlas.residual(chart, _assemble(axis, parameter, rest))

        rest, diagnostics = _CHART_ROOT.solve_with_diagnostics(
            graph_residual, _rest(axis, guess)
        )
        rest = eqx.error_if(
            rest,
            ~diagnostics.converged,
            "IntersectionCurve.chart_root must converge under the native root plan; got native convergence=False",
        )
        coupled = _assemble(axis, parameter, rest)
        jacobian = jax.jacfwd(lambda x: atlas.residual(chart, x))(coupled)
        permuted = jacobian @ permutations[axis].T
        derivative = solve_small_linear(
            SmallLinearSolvePlan(3), permuted[:, 1:], -permuted[:, 0]
        )
        direction = permutations[axis].T @ jnp.concatenate(
            (jnp.ones((1,), dtype=coupled.dtype), derivative.value)
        )
        direction = direction * jnp.sign(jnp.take(last_node - first_node, axis))
        point, tangent = jax.jvp(
            lambda p: _evaluate_first(atlas.systems, atlas.selector, chart, p),
            (coupled[:2],),
            (direction[:2],),
        )
        distance = jnp.maximum(
            jnp.abs(coupled - atlas.lower[chart]),
            jnp.abs(coupled - atlas.upper[chart]),
        )
        valid = atlas.certified[chart] & (value >= 0.0) & (value <= count)
        # Outward-rounded error metadata is not a source coordinate or jet.
        # Detach only this diagnostic; coupled UV and residual AD stay live.
        bound = jnp.where(
            valid,
            jnp.nextafter(jax.lax.stop_gradient(jnp.max(distance)), jnp.inf),
            jnp.inf,
        )
        return IntersectionCurvePoint(
            point,
            coupled[:2],
            coupled[2:],
            tangent / jnp.linalg.norm(tangent),
            bound,
            diagnostics.residual_norm,
            chart,
        )

    return jax.vmap(one)(tau)


def _assemble(axis: Array, parameter: Array, rest: Array, /) -> Array:
    permutation = jnp.asarray(_CHART_PERMUTATIONS)[axis]
    return permutation.T @ jnp.concatenate((parameter[None], rest))


def _rest(axis: Array, parameters: Array, /) -> Array:
    permutation = jnp.asarray(_CHART_PERMUTATIONS)[axis]
    return (permutation @ parameters)[1:]


def _evaluate_first(
    systems: tuple[SurfacePairSystem, ...],
    selector: Array,
    chart: Array,
    parameters: Array,
    /,
) -> Array:
    if len(systems) == 1:
        return systems[0].first.evaluate(parameters)
    return jax.lax.switch(
        selector[chart],
        tuple((lambda p, system=system: system.first.evaluate(p)) for system in systems),
        parameters,
    )


def _intersection_is_c1(
    curve: IntersectionCurve,
    first: float,
    last: float,
    /,
    *,
    minimum_chart: int = 0,
    maximum_chart: int | None = None,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> bool:
    if not (
        math.isfinite(first)
        and math.isfinite(last)
        and 0 <= first <= last <= curve.num_charts
    ):
        raise ValueError(
            f"IntersectionCurve.interval must be a closed source subinterval; got {(first, last)!r}"
        )
    maximum_chart = curve.num_charts - 1 if maximum_chart is None else maximum_chart
    # Shared-node certification proves C0 only. Independent graph coordinates
    # and chart advances need not have equal one-sided velocities.
    for node in range(minimum_chart + 1, maximum_chart + 1):
        if first <= node <= last:
            return False
    head = max(minimum_chart, min(int(math.floor(first)), maximum_chart))
    tail = max(head, min(int(math.floor(last)), maximum_chart))
    for chart in range(head, tail + 1):
        box = np.stack((curve.box_lower[chart], curve.box_upper[chart]))
        if not curve.first.patch.is_c1_on(
            box[:, :2], preparations=preparations, budget=budget
        ) or not curve.second.patch.is_c1_on(
            box[:, 2:], preparations=preparations, budget=budget
        ):
            return False
    return True


# Inverse Jacobian bounds and coupled first jet of one chart box.
type _ChartFirstJet = tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass(init=False, eq=False, repr=False)
class IntersectionJetPreparation:
    """Reusable deterministic graph programs, optionally resource-observed.

    The preparation/wrapping hooks change only how the existing interval
    programs are retained and evaluated; they do not change source equations,
    inverse enclosures, root definition, chart ownership or derivative formulas.
    An unobserved preparation also retains the results of identical
    deterministic queries; an observed one charges every evaluation and retains
    results only inside an explicit ``retained_queries`` scope. Declared fields
    are the complete owned state, so retained-storage profiling sizes the
    programs, roots and memo banks a retained source map keeps alive.
    """

    curve: IntersectionCurve
    pairs: Any
    prepare: Any
    _wrap: Any
    _observe_point_jet: Any
    _programs: dict[tuple[int, str, int], Any]
    _roots: dict[int, Any]
    _batch_roots: dict[int, Any]
    _observed: bool
    chart_boxes: dict[tuple[int, float, float], np.ndarray] | None
    jets: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray]] | None
    chart_jets: dict[tuple[int, bytes], _ChartFirstJet | None] | None

    def __init__(
        self,
        curve: IntersectionCurve,
        *,
        prepare: Any = None,
        wrap: Any = None,
        observe_point_jet: Any = None,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> None:
        if preparations is None:
            preparations = TrimRequestPreparations()
        with coordinate_source_request(
            (
                curve,
                prepare,
                wrap,
                observe_point_jet,
            ),
            budget=budget,
        ) as budget:
            self.curve = curve
            self.pairs = curve._pairs(budget=budget)
            self.prepare = prepare
            self._wrap = wrap
            self._observe_point_jet = observe_point_jet
            self._programs: dict[tuple[int, str, int], Any] = {}
            self._roots: dict[int, Any] = {}
            self._batch_roots: dict[int, Any] = {}
            memoize = prepare is None and wrap is None
            self._observed = not memoize
            self.chart_boxes: dict[tuple[int, float, float], np.ndarray] | None = (
                {} if memoize else None
            )
            self.jets: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray]] | None = (
                {} if memoize else None
            )
            self.chart_jets: dict[tuple[int, bytes], _ChartFirstJet | None] | None = (
                {} if memoize else None
            )

    def program(
        self, chart: int, kind: str, function: Any, dimension: int, capacity: int = 1
    ) -> Any:
        """One chart program per source piece, kind and interval batch capacity."""
        index = int(self.pairs[1][chart])
        key = (index, kind, capacity)
        if key not in self._programs:
            coefficients = coefficient_enclosures(self.pairs[0][index])
            self._programs[key] = (
                prepare_interval_function(
                    function,
                    dimension,
                    batch_capacity=capacity,
                    constant_bounds=coefficients,
                )
                if self.prepare is None
                else self.prepare(function, dimension, coefficients, capacity)
            )
        return self._programs[key]

    @contextmanager
    def retained_queries(self) -> Iterator[None]:
        """Reuse identical deterministic query results within one caller scope.

        Observed preparations still charge every evaluation they perform; the
        retained boxes and jets are released when the scope exits.
        """
        if not self._observed:
            yield
            return
        self.chart_boxes, self.jets, self.chart_jets = {}, {}, {}
        try:
            yield
        finally:
            self.chart_boxes = self.jets = self.chart_jets = None

    def root(self, chart: int) -> Any:

        index = int(self.pairs[1][chart])
        if index not in self._roots:
            prepared = intersection_owner.prepare(self.pairs[0][index], 4, 1)
            self._roots[index] = (
                prepared
                if self._wrap is None
                else _ObservedIntersectionRoot(
                    prepared, self._wrap, self._observe_point_jet
                )
            )
        return self._roots[index]

    def batch_root(self, chart: int) -> Any:
        """The same chart system at a multi-box interval batch capacity.

        Rows are independent interval extensions, so a batch returns exactly
        the per-row results. Observed preparations charge every evaluated row
        and the batch-capacity interpreter workspace through their hooks.
        """
        index = int(self.pairs[1][chart])
        if index not in self._batch_roots:
            prepared = intersection_owner.prepare(self.pairs[0][index], 4, _CHART_BATCH)
            self._batch_roots[index] = (
                prepared
                if self._wrap is None
                else _ObservedIntersectionRoot(
                    prepared, self._wrap, self._observe_point_jet
                )
            )
        return self._batch_roots[index]


@contextmanager
def original_trim_intersection_preparation(
    curves: tuple[AbstractTrimCurve, ...],
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
) -> Iterator[TrimRequestPreparations]:
    """Register trim roots and borrow branch programs for one explicit request."""
    if preparations is None:
        preparations = TrimRequestPreparations()

    request = TrimRequestPreparations() if preparations is None else preparations
    with intersection_owner.original_trim_root_preparation_scope(
        curves, preparations=request
    ):
        yield request


def _original_trim_intersection_preparation(
    curve: IntersectionCurve,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
) -> IntersectionJetPreparation | None:
    if preparations is None:
        return None
    objects, identities = preparations.branch_objects, preparations.branch_identities
    entry = objects.get(id(curve))
    if entry is not None:
        return entry[1]
    if any(isinstance(leaf, Tracer) for leaf in jax.tree_util.tree_leaves(curve)):
        return None
    identity = (
        str(jax.tree_util.tree_structure(curve)),
        canonical_fingerprint(array_tree_fingerprint(curve)),
    )
    preparation = identities.get(identity)
    if preparation is None:
        preparation = IntersectionJetPreparation(
            curve, preparations=preparations, budget=budget
        )
        identities[identity] = preparation
    objects[id(curve)] = (curve, preparation)
    return preparation


class _ObservedIntersectionRoot:
    """Resource hooks around the deterministic Krawczyk evaluator, not a new solver."""

    def __init__(self, prepared: Any, wrap: Any, observe_point_jet: Any) -> None:
        self._prepared = prepared
        self.value = wrap(prepared.value)
        self.jacobian = wrap(prepared.jacobian)
        self._observe_point_jet = observe_point_jet

    def point_jacobians(
        self,
        points: np.ndarray,
        /,
        *,
        budget: CoordinateEnclosureBudget,
        preparations: TrimRequestPreparations | None = None,
    ) -> np.ndarray:
        if self._observe_point_jet is not None:
            self._observe_point_jet(points.shape[0])
        return self._prepared.point_jacobians(
            points, budget=budget, preparations=preparations
        )


def intersection_derivative_bounds(
    curve: IntersectionCurve,
    first: float,
    last: float,
    order: int,
    side: IntersectionCurveSide | Literal["all"] | None,
    /,
    *,
    minimum_chart: int = 0,
    maximum_chart: int | None = None,
    source_extension: bool = False,
    _preparation: IntersectionJetPreparation | None = None,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    """Enclose graph inverse/implicit derivatives, never sampled root jets."""
    if _preparation is None:
        _preparation = _original_trim_intersection_preparation(
            curve, preparations=preparations, budget=budget
        )
    elif _preparation.curve is not curve:
        raise ValueError(
            "IntersectionCurve.preparation must retain the source curve; got a different prepared source"
        )
    if order not in (1, 2) or isinstance(order, bool):
        raise ValueError(
            f"IntersectionCurve.derivative_order must be one or two; got {order!r}"
        )
    memo = None if _preparation is None else _preparation.jets
    key = (first, last, order, side, minimum_chart, maximum_chart, source_extension)
    result = None if memo is None else memo.get(key)
    if result is None:
        result = _intersection_jet_bounds(
            curve,
            first,
            last,
            order,
            side,
            minimum_chart,
            maximum_chart,
            source_extension,
            _preparation,
            preparations=preparations,
            budget=budget,
        )
        if memo is not None:
            memo[key] = result
    return result[0].copy(), result[1].copy()


class _ChartQuery(StrictModule):
    lower: np.ndarray
    upper: np.ndarray
    fractions: np.ndarray
    axis: int = eqx.field(static=True)


def _prepare_chart_query(
    curve: IntersectionCurve,
    chart: int,
    first: float,
    last: float,
    head: int,
    tail: int,
    source_extension: bool,
) -> _ChartQuery | None:
    lower, upper = curve.box_lower[chart].copy(), curve.box_upper[chart].copy()
    axis = int(curve.chart_axes[chart])

    fractions = np.asarray((first, last)) - chart
    if source_extension:
        fractions[0] = max(0.0, fractions[0]) if chart > head else fractions[0]
        fractions[1] = min(1.0, fractions[1]) if chart < tail else fractions[1]
    else:
        fractions = np.clip(fractions, 0.0, 1.0)
    endpoints = []
    for fraction in fractions:
        if fraction == 0:
            endpoints.append(
                (
                    curve.chart_start_lower[chart, axis],
                    curve.chart_start_upper[chart, axis],
                )
            )
        elif fraction == 1:
            endpoints.append(
                (
                    curve.chart_end_lower[chart, axis],
                    curve.chart_end_upper[chart, axis],
                )
            )
        else:
            complement = (
                np.nextafter(1 - fraction, -np.inf),
                np.nextafter(1 - fraction, np.inf),
            )
            endpoints.append(
                interval_add(
                    interval_multiply(
                        (
                            curve.chart_start_lower[chart, axis],
                            curve.chart_start_upper[chart, axis],
                        ),
                        complement,
                    ),
                    interval_multiply(
                        (
                            curve.chart_end_lower[chart, axis],
                            curve.chart_end_upper[chart, axis],
                        ),
                        (fraction, fraction),
                    ),
                )
            )
    if source_extension and (
        min(value[0] for value in endpoints) < lower[axis]
        or max(value[1] for value in endpoints) > upper[axis]
    ):
        return None
    lower[axis] = max(lower[axis], min(value[0] for value in endpoints))
    upper[axis] = min(upper[axis], max(value[1] for value in endpoints))
    return _ChartQuery(lower=lower, upper=upper, fractions=fractions, axis=axis)


_CHART_BATCH = 64


def _contract_chart_queries(
    queries: Sequence[_ChartQuery],
    prepared: Any,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None,
) -> list[np.ndarray]:
    """Contract chart queries sharing one chart program and graph axis.

    Krawczyk images are row-independent interval extensions; each row keeps
    its own iterate, emptiness refusal and five-percent stopping rule, so a
    batch returns exactly the boxes of separate contractions.
    """
    axis = queries[0].axis
    if any(query.axis != axis for query in queries):
        raise ValueError(
            "IntersectionCurve.chart_queries must share one graph axis; got mixed axes"
        )
    free = [index for index in range(4) if index != axis]
    lower = np.stack([query.lower for query in queries])
    upper = np.stack([query.upper for query in queries])
    active = np.arange(len(queries))
    for _ in range(24):
        if active.size == 0:
            break
        budget.ticket.checkpoint()
        current_lower, current_upper = lower[active], upper[active]
        image = krawczyk(
            prepared,
            current_lower,
            current_upper,
            parameter_axis=axis,
            budget=budget,
            preparations=preparations,
        )
        new_lower, new_upper = current_lower.copy(), current_upper.copy()
        new_lower[:, free] = np.maximum(current_lower[:, free], image.lower)
        new_upper[:, free] = np.minimum(current_upper[:, free], image.upper)
        if np.any(new_lower > new_upper):
            raise ValueError(
                "IntersectionCurve.chart_box must retain a nonempty branch enclosure; got an empty contracted chart box"
            )
        old_width = np.max(current_upper[:, free] - current_lower[:, free], axis=1)
        new_width = np.max(new_upper[:, free] - new_lower[:, free], axis=1)
        lower[active], upper[active] = new_lower, new_upper
        active = active[new_width < 0.95 * old_width]
    return [np.stack((lower[row], upper[row])) for row in range(len(queries))]


type _ChartKey = tuple[int, float, float]


def _chart_enclosures(
    curve: IntersectionCurve,
    requests: Sequence[tuple[float, float, int, int | None, bool]],
    preparation: IntersectionJetPreparation | None,
    /,
    *,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
) -> list[np.ndarray]:
    """Coupled chart boxes of each ``parameter_enclosures`` request.

    A chart box is a deterministic function of its chart and fractions;
    identical chart queries share one contraction. Preparations retaining
    query results (memoizing, or observed inside ``retained_queries``)
    contract distinct queries of one chart program together.
    """
    count = curve.num_charts
    systems, selector = (
        curve._pairs(budget=budget) if preparation is None else preparation.pairs
    )
    memo = None if preparation is None else preparation.chart_boxes
    pending: dict[_ChartKey, _ChartQuery] = {}
    plans: list[tuple[_ChartKey, ...] | None] = []
    for first, last, minimum_chart, maximum_chart, source_extension in requests:
        if not (math.isfinite(first) and math.isfinite(last) and first <= last):
            raise ValueError(
                f"IntersectionCurve.interval must be finite and ordered; got {(first, last)!r}"
            )
        if not source_extension and not 0 <= first <= last <= count:
            raise ValueError(
                f"IntersectionCurve.interval must lie inside the curve parameter interval; got {(first, last)!r}"
            )
        last_chart = count - 1 if maximum_chart is None else maximum_chart
        if not 0 <= minimum_chart <= last_chart < count:
            raise ValueError(
                f"IntersectionCurve.chart_range must lie inside the atlas; got {(minimum_chart, last_chart)!r}"
            )
        head = max(minimum_chart, min(int(math.floor(first)), last_chart))
        tail = max(head, min(int(math.ceil(last)) - 1, last_chart))
        keys: list[_ChartKey] = []
        for chart in range(head, tail + 1):
            query = _prepare_chart_query(
                curve, chart, first, last, head, tail, source_extension
            )
            if query is None:
                break
            key = (chart, float(query.fractions[0]), float(query.fractions[1]))
            keys.append(key)
            if (memo is None or key not in memo) and key not in pending:
                pending[key] = query
        else:
            plans.append(tuple(keys))
            continue
        plans.append(None)
    groups: dict[tuple[int, int], list[_ChartKey]] = {}
    for key, query in pending.items():
        groups.setdefault((int(selector[key[0]]), query.axis), []).append(key)
    contracted: dict[_ChartKey, np.ndarray] = {}
    for (index, _), keys in groups.items():
        if preparation is not None and memo is not None and len(keys) > 1:
            batches = [keys]
            prepared = preparation.batch_root(keys[0][0])
        else:
            batches = [[key] for key in keys]
            prepared = (
                intersection_owner.prepare(
                    systems[index], 4, 1, preparations=preparations
                )
                if preparation is None
                else preparation.root(keys[0][0])
            )
        for batch in batches:
            boxes = _contract_chart_queries(
                [pending[key] for key in batch],
                prepared,
                budget=budget,
                preparations=preparations,
            )
            for key, box in zip(batch, boxes, strict=True):
                contracted[key] = box
                if memo is not None:
                    memo[key] = box
    results = []
    available = contracted if memo is None else memo
    for plan in plans:
        if plan is None:
            results.append(np.asarray([[np.full(4, -np.inf), np.full(4, np.inf)]]))
            continue
        results.append(np.asarray([available[key] for key in plan]))
    return results


class _JetBounds(StrictModule):
    lower: np.ndarray
    upper: np.ndarray


def _interval_matrix_product(
    a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = interval_times_interval(a[0][None], a[1][None], b[0][None], b[1][None])
    return lower[0], upper[0]


def _interval_quadratic(
    tensor: tuple[np.ndarray, np.ndarray], vector: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    result = (np.zeros(tensor[0].shape[0]), np.zeros(tensor[0].shape[0]))
    for i in range(vector[0].shape[0]):
        for j in range(vector[0].shape[0]):
            product = interval_multiply(
                (vector[0][i], vector[1][i]), (vector[0][j], vector[1][j])
            )
            result = interval_add(
                result,
                interval_multiply((tensor[0][:, i, j], tensor[1][:, i, j]), product),
            )
    return np.asarray(result[0], dtype=np.float64), np.asarray(
        result[1], dtype=np.float64
    )


def _second_jet_from_hessian(
    hessian: tuple[np.ndarray, np.ndarray],
    free: list[int],
    inverse_bounds: tuple[np.ndarray, np.ndarray],
    coupled: tuple[np.ndarray, np.ndarray],
) -> _JetBounds:
    """Implicit second jet from one enclosed residual-Hessian row."""
    curvature = _interval_quadratic(hessian, coupled)
    second = _interval_matrix_product(
        inverse_bounds, (-curvature[1][:, None], -curvature[0][:, None])
    )
    second_lo, second_hi = np.zeros(4), np.zeros(4)
    second_lo[free], second_hi[free] = second[0][:, 0], second[1][:, 0]
    return _JetBounds(lower=second_lo, upper=second_hi)


def _spatial_jet_from_bounds(
    jacobian: tuple[np.ndarray, np.ndarray],
    parameters: tuple[np.ndarray, np.ndarray],
    hessian: tuple[np.ndarray, np.ndarray] | None = None,
    coupled: tuple[np.ndarray, np.ndarray] | None = None,
) -> _JetBounds:
    """Physical chain-rule jet from enclosed spatial Jacobian/Hessian rows."""
    physical = _interval_matrix_product(
        jacobian,
        (
            parameters[0][:2, None],
            parameters[1][:2, None],
        ),
    )
    result = (physical[0][:, 0], physical[1][:, 0])
    if hessian is not None and coupled is not None:
        result = interval_add(
            result,
            _interval_quadratic(hessian, (coupled[0][:2], coupled[1][:2])),
        )
    return _JetBounds(
        lower=np.asarray(result[0], dtype=np.float64),
        upper=np.asarray(result[1], dtype=np.float64),
    )


def _chart_program(
    system: SurfacePairSystem,
    chart: int,
    kind: str,
    preparation: IntersectionJetPreparation | None,
    capacity: int = 1,
) -> Any:
    match kind:
        case "residual-hessian":
            function, dimension = jax.jacfwd(jax.jacfwd(system.residual)), 4
        case "spatial-value":
            function, dimension = system.first.evaluate, 2
        case "spatial-jacobian":
            function, dimension = jax.jacfwd(system.first.evaluate), 2
        case "spatial-hessian":
            function, dimension = jax.jacfwd(jax.jacfwd(system.first.evaluate)), 2
        case _:
            raise ValueError(
                f"IntersectionCurve.chart_program must name a graph program; got {kind!r}"
            )
    if preparation is None:
        return prepare_interval_function(
            function,
            dimension,
            batch_capacity=capacity,
            constant_bounds=coefficient_enclosures(system),
        )
    return preparation.program(chart, kind, function, dimension, capacity)


def _chart_second_jet(
    system: SurfacePairSystem,
    chart: int,
    box: np.ndarray,
    free: list[int],
    inverse_bounds: tuple[np.ndarray, np.ndarray],
    coupled: tuple[np.ndarray, np.ndarray],
    preparation: IntersectionJetPreparation | None,
) -> _JetBounds:
    hessian = _chart_program(system, chart, "residual-hessian", preparation)
    h_lo, h_hi = hessian.evaluate(box[0][None], box[1][None])
    return _second_jet_from_hessian((h_lo[0], h_hi[0]), free, inverse_bounds, coupled)


def _chart_spatial_jet(
    system: SurfacePairSystem,
    chart: int,
    box: np.ndarray,
    order: int,
    coupled: tuple[np.ndarray, np.ndarray],
    parameters: tuple[np.ndarray, np.ndarray],
    preparation: IntersectionJetPreparation | None,
) -> _JetBounds:
    spatial_jacobian = _chart_program(system, chart, "spatial-jacobian", preparation)
    s_lo, s_hi = spatial_jacobian.evaluate(box[None, 0, :2], box[None, 1, :2])
    if order != 2:
        return _spatial_jet_from_bounds((s_lo[0], s_hi[0]), parameters)
    spatial_hessian = _chart_program(system, chart, "spatial-hessian", preparation)
    sh_lo, sh_hi = spatial_hessian.evaluate(box[None, 0, :2], box[None, 1, :2])
    return _spatial_jet_from_bounds(
        (s_lo[0], s_hi[0]), parameters, (sh_lo[0], sh_hi[0]), coupled
    )


def _intersection_jet_bounds(
    curve: IntersectionCurve,
    first: float,
    last: float,
    order: int,
    side: IntersectionCurveSide | Literal["all"] | None,
    minimum_chart: int,
    maximum_chart: int | None,
    source_extension: bool,
    _preparation: IntersectionJetPreparation | None,
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    dimension = 7 if side == "all" else (3 if side is None else 2)
    if order == 2 and not _intersection_is_c1(
        curve,
        max(0.0, min(first, curve.num_charts)) if source_extension else first,
        max(0.0, min(last, curve.num_charts)) if source_extension else last,
        minimum_chart=minimum_chart,
        maximum_chart=maximum_chart,
        preparations=preparations,
        budget=budget,
    ):
        return np.full(dimension, -np.inf), np.full(dimension, np.inf)

    systems, selector = (
        type(curve)._pairs(curve, budget=budget) if _preparation is None else _preparation.pairs
    )
    last_chart = curve.num_charts - 1 if maximum_chart is None else maximum_chart
    head = max(minimum_chart, min(int(math.floor(first)), last_chart))
    lower_bounds, upper_bounds = [], []
    for offset, box in enumerate(
        type(curve).parameter_enclosures(
            curve, first,
            last,
            minimum_chart=minimum_chart,
            maximum_chart=last_chart,
            source_extension=source_extension,
            _preparation=_preparation,
            preparations=preparations,
            budget=budget,
        )
    ):
        chart = head + offset
        if not np.all(np.isfinite(box)):
            return np.full(dimension, -np.inf), np.full(dimension, np.inf)
        system = systems[int(selector[chart])]
        prepared = (
            intersection_owner.prepare(system, 4, 1, preparations=preparations)
            if _preparation is None
            else _preparation.root(chart)
        )
        axis = int(curve.chart_axes[chart])
        free = [index for index in range(4) if index != axis]
        # The inverse and first jet are functions of the chart and its source-defined
        # box; every order and side of one range shares a single inclusion.
        memo = None if _preparation is None else _preparation.chart_jets
        key = (chart, box.tobytes())
        if memo is not None and key in memo:
            basis = memo[key]
        else:
            basis = _chart_first_jet(
                curve, prepared, chart, box, budget=budget, preparations=preparations
            )
            if memo is not None:
                memo[key] = basis
        if basis is None:
            return np.full(dimension, -np.inf), np.full(dimension, np.inf)
        inverse_bounds, coupled = basis
        parameters = coupled
        if order == 2:
            second = _chart_second_jet(
                system, chart, box, free, inverse_bounds, coupled, _preparation
            )
            parameters = (second.lower, second.upper)
        if side is not None and side != "all":
            columns = slice(0, 2) if side == "first" else slice(2, 4)
            result = parameters
            lower_bounds.append(result[0][columns])
            upper_bounds.append(result[1][columns])
            continue
        physical = _chart_spatial_jet(
            system, chart, box, order, coupled, parameters, _preparation
        )
        result = (physical.lower, physical.upper)
        if side == "all":
            result = (
                np.concatenate((result[0], parameters[0])),
                np.concatenate((result[1], parameters[1])),
            )
        lower_bounds.append(result[0])
        upper_bounds.append(result[1])
    return np.min(lower_bounds, axis=0), np.max(upper_bounds, axis=0)


def _chart_first_jet(
    curve: IntersectionCurve,
    prepared: Any,
    chart: int,
    box: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
) -> _ChartFirstJet | None:
    """Inverse Jacobian bounds and coupled first jet over one chart box.

    ``None`` means the box has no contracting parametric inclusion, so no
    implicit derivative is enclosed there.
    """

    axis = int(curve.chart_axes[chart])
    inclusion = krawczyk(
        prepared,
        box[0][None],
        box[1][None],
        parameter_axis=axis,
        budget=budget,
        preparations=preparations,
    )
    contraction = float(inclusion.contraction[0])
    if not math.isfinite(contraction) or contraction >= 1.0:
        return None
    jac_lo, jac_hi = evaluate_interval(
        prepared.jacobian,
        box[0][None],
        box[1][None],
        budget=budget,
        preparations=preparations,
    )
    return _chart_first_jet_from_bounds(
        curve, chart, contraction, inclusion.preconditioner[0], jac_lo[0], jac_hi[0]
    )


def _chart_first_jet_from_bounds(
    curve: IntersectionCurve,
    chart: int,
    contraction: float,
    inverse: np.ndarray,
    jac_lo: np.ndarray,
    jac_hi: np.ndarray,
    /,
) -> _ChartFirstJet:
    """Apply the scalar first-jet formula to one independently enclosed row."""
    axis = int(curve.chart_axes[chart])
    free = [index for index in range(4) if index != axis]
    inverse_norm = np.nextafter(
        np.max(np.sum(np.abs(inverse), axis=1)) * (1 + 16 * np.finfo(np.float64).eps),
        np.inf,
    )
    radius = np.nextafter(
        contraction
        / (1 - contraction)
        * inverse_norm
        * (1 + 16 * np.finfo(np.float64).eps),
        np.inf,
    )
    inverse_bounds = (
        np.nextafter(inverse - radius, -np.inf),
        np.nextafter(inverse + radius, np.inf),
    )
    rest_lo, rest_hi = _interval_matrix_product(
        inverse_bounds, (-jac_hi[:, axis, None], -jac_lo[:, axis, None])
    )
    advance_bounds = (
        np.nextafter(
            curve.chart_end_lower[chart, axis] - curve.chart_start_upper[chart, axis],
            -np.inf,
        ),
        np.nextafter(
            curve.chart_end_upper[chart, axis] - curve.chart_start_lower[chart, axis],
            np.inf,
        ),
    )
    rest = interval_multiply((rest_lo[:, 0], rest_hi[:, 0]), advance_bounds)
    first_lo, first_hi = np.zeros(4), np.zeros(4)
    first_lo[axis], first_hi[axis] = advance_bounds
    first_lo[free], first_hi[free] = rest
    return inverse_bounds, (first_lo, first_hi)


def _prepare_chart_first_jets(
    preparation: IntersectionJetPreparation,
    queries: Sequence[tuple[int, np.ndarray]],
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations,
) -> None:
    """Batch independent first-jet inclusions into the existing request bank."""
    memo = preparation.chart_jets
    if memo is None:
        return
    curve = preparation.curve
    pending: dict[tuple[int, bytes], np.ndarray] = {}
    for chart, box in queries:
        key = (chart, box.tobytes())
        if key not in memo and np.all(np.isfinite(box)):
            pending.setdefault(key, box)
    groups: dict[tuple[int, int], list[tuple[int, bytes]]] = {}
    for key in pending:
        chart = key[0]
        groups.setdefault(
            (int(preparation.pairs[1][chart]), int(curve.chart_axes[chart])), []
        ).append(key)
    for (_, axis), keys in groups.items():
        prepared = preparation.batch_root(keys[0][0])
        for start in range(0, len(keys), _CHART_BATCH):
            budget.ticket.checkpoint()
            batch = keys[start : start + _CHART_BATCH]
            boxes = np.stack([pending[key] for key in batch])
            inclusion = krawczyk(
                prepared,
                boxes[:, 0],
                boxes[:, 1],
                parameter_axis=axis,
                budget=budget,
                preparations=preparations,
            )
            valid_mask = np.isfinite(inclusion.contraction) & (
                inclusion.contraction < 1.0
            )
            valid = np.flatnonzero(valid_mask)
            for row, key in enumerate(batch):
                if not valid_mask[row]:
                    memo[key] = None
            if not valid.size:
                continue
            jac_lo, jac_hi = evaluate_interval(
                prepared.jacobian,
                boxes[valid, 0],
                boxes[valid, 1],
                budget=budget,
                preparations=preparations,
            )
            for row, lower, upper in zip(valid, jac_lo, jac_hi, strict=True):
                key = batch[int(row)]
                memo[key] = _chart_first_jet_from_bounds(
                    curve,
                    key[0],
                    float(inclusion.contraction[row]),
                    inclusion.preconditioner[row],
                    lower,
                    upper,
                )


class CoupledCellSourceJets(StrictModule):
    """Coupled source enclosures of one approximation cell inside one chart.

    ``center_box`` is the chart's coupled box at the cell center and
    ``center_point`` its physical image; ``first``, ``second`` and
    ``center_first`` are the seven-column ``"all"`` jets of
    ``intersection_derivative_bounds`` restricted to the cell's chart.
    """

    center_box: np.ndarray
    center_point: _JetBounds
    first: _JetBounds
    second: _JetBounds
    center_first: _JetBounds


def coupled_cell_source_jets(
    preparation: IntersectionJetPreparation,
    cells: Sequence[tuple[int, float, float, float]],
    /,
    *,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> list[CoupledCellSourceJets]:
    """Batched single-chart cell jets ``(chart, first, last, center)``.

    Every row equals the one-cell path through ``parameter_enclosures``,
    ``intersection_derivative_bounds`` and the spatial value program: chart
    contractions, first-jet inclusions and graph programs are row-independent
    interval extensions evaluated at batch capacity. An observed preparation
    charges every evaluated row; call inside its ``retained_queries`` scope.
    """
    curve = preparation.curve
    systems, selector = preparation.pairs
    unbounded = _JetBounds(lower=np.full(7, -np.inf), upper=np.full(7, np.inf))
    requests: list[tuple[float, float, int, int | None, bool]] = []
    for chart, first, last, center in cells:
        requests.append((center, center, chart, chart, False))
        requests.append((first, last, chart, chart, False))
    boxes = _chart_enclosures(
        curve, requests, preparation, preparations=preparations, budget=budget
    )
    centers = [boxes[2 * row][0] for row in range(len(cells))]
    intervals = [boxes[2 * row + 1][0] for row in range(len(cells))]
    _prepare_chart_first_jets(
        preparation,
        [
            (chart, box)
            for (chart, *_), pair in zip(cells, zip(intervals, centers), strict=True)
            for box in pair
        ],
        budget=budget,
        preparations=preparations,
    )
    memo = preparation.chart_jets
    if memo is None:
        raise ValueError(
            "coupled_cell_source_jets.preparation must retain chart jets; got a preparation outside retained_queries"
        )

    def basis(chart: int, box: np.ndarray) -> _ChartFirstJet | None:
        if not np.all(np.isfinite(box)):
            return None
        return memo[(chart, box.tobytes())]

    def evaluate_rows(
        kind: str, rows: list[tuple[int, np.ndarray]], columns: slice
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        results: list[tuple[np.ndarray, np.ndarray]] = [
            (np.empty(0), np.empty(0)) for _ in rows
        ]
        groups: dict[int, list[int]] = {}
        for position, (chart, _) in enumerate(rows):
            groups.setdefault(int(selector[chart]), []).append(position)
        for index, positions in groups.items():
            program = _chart_program(
                systems[index], rows[positions[0]][0], kind, preparation, _CHART_BATCH
            )
            budget.ticket.checkpoint()
            lower, upper = program.evaluate(
                np.stack([rows[position][1][0, columns] for position in positions]),
                np.stack([rows[position][1][1, columns] for position in positions]),
            )
            for row, position in enumerate(positions):
                results[position] = (lower[row], upper[row])
        return results

    charts = [int(chart) for chart, *_ in cells]
    points = evaluate_rows(
        "spatial-value", list(zip(charts, centers, strict=True)), slice(0, 2)
    )
    first_bases = [basis(chart, box) for chart, box in zip(charts, intervals, strict=True)]
    center_bases = [basis(chart, box) for chart, box in zip(charts, centers, strict=True)]
    smooth = [
        base is not None
        and _intersection_is_c1(
            curve,
            first,
            last,
            minimum_chart=chart,
            maximum_chart=chart,
            preparations=preparations,
            budget=budget,
        )
        for (chart, first, last, _), base in zip(cells, first_bases, strict=True)
    ]
    interval_rows = [row for row, base in enumerate(first_bases) if base is not None]
    center_rows = [row for row, base in enumerate(center_bases) if base is not None]
    jacobians = evaluate_rows(
        "spatial-jacobian",
        [(charts[row], intervals[row]) for row in interval_rows]
        + [(charts[row], centers[row]) for row in center_rows],
        slice(0, 2),
    )
    interval_jacobians = dict(
        zip(interval_rows, jacobians[: len(interval_rows)], strict=True)
    )
    center_jacobians = dict(
        zip(center_rows, jacobians[len(interval_rows) :], strict=True)
    )
    curved_rows = [
        (chart, box)
        for chart, box, flag in zip(charts, intervals, smooth, strict=True)
        if flag
    ]
    residual_hessians = iter(evaluate_rows("residual-hessian", curved_rows, slice(None)))
    spatial_hessians = iter(evaluate_rows("spatial-hessian", curved_rows, slice(0, 2)))

    def jet(
        spatial: _JetBounds, parameters: tuple[np.ndarray, np.ndarray]
    ) -> _JetBounds:
        return _JetBounds(
            lower=np.concatenate((spatial.lower, parameters[0])),
            upper=np.concatenate((spatial.upper, parameters[1])),
        )

    results = []
    for row, chart in enumerate(charts):
        axis = int(curve.chart_axes[chart])
        free = [index for index in range(4) if index != axis]
        first_base, center_base = first_bases[row], center_bases[row]
        first_jet = second_jet = center_jet = unbounded
        if first_base is not None:
            inverse_bounds, coupled = first_base
            spatial = interval_jacobians[row]
            first_jet = jet(_spatial_jet_from_bounds(spatial, coupled), coupled)
            if smooth[row]:
                second = _second_jet_from_hessian(
                    next(residual_hessians), free, inverse_bounds, coupled
                )
                parameters = (second.lower, second.upper)
                second_jet = jet(
                    _spatial_jet_from_bounds(
                        spatial, parameters, next(spatial_hessians), coupled
                    ),
                    parameters,
                )
        if center_base is not None:
            center_spatial = center_jacobians[row]
            center_coupled = center_base[1]
            center_jet = jet(
                _spatial_jet_from_bounds(center_spatial, center_coupled), center_coupled
            )
        point_lower, point_upper = points[row]
        results.append(
            CoupledCellSourceJets(
                center_box=centers[row],
                center_point=_JetBounds(lower=point_lower, upper=point_upper),
                first=first_jet,
                second=second_jet,
                center_first=center_jet,
            )
        )
    return results


class IntersectionPCurve(AbstractTrimCurve):
    """Parameter-space trace of an intersection branch on one generating surface.

    Enclosures come from the certified chart boxes, so trim classification and
    the 3D curve use the same coupled definition.
    """

    curve: IntersectionCurve
    side: IntersectionCurveSide = eqx.field(static=True)
    first: float = eqx.field(static=True)
    last: float = eqx.field(static=True)
    reversed: bool = eqx.field(static=True)

    @checked
    def __init__(
        self,
        curve: IntersectionCurve,
        side: IntersectionCurveSide,
        *,
        first: float | None = None,
        last: float | None = None,
        reversed: bool = False,
    ) -> None:
        first_ = 0.0 if first is None else float(first)
        last_ = float(curve.num_charts) if last is None else float(last)
        self.first, self.last = curve.validate_range(first_, last_)
        self.reversed = bool(reversed)
        self.curve = curve
        self.side = parse(side, IntersectionCurveSide, "side")

    def _columns(self) -> slice:
        match self.side:
            case "first":
                return slice(0, 2)
            case "second":
                return slice(2, 4)
            case _:
                raise ValueError(
                    f"IntersectionPCurve.side must name a generating side; got {self.side!r}"
                )

    @property
    def parameter_interval(self) -> tuple[float, float]:
        return self.first, self.last

    @property
    def parameter_domain(self) -> tuple[float, float]:
        return self.parameter_interval

    @property
    def ambient_dimension(self) -> int:
        return 2

    @property
    def period(self) -> float | None:
        return (
            self.last - self.first
            if self.curve.closed
            and self.first == 0.0
            and self.last == self.curve.num_charts
            else None
        )

    def validate_range(self, first: float, last: float, /) -> tuple[float, float]:
        if not (
            math.isfinite(first)
            and math.isfinite(last)
            and self.first <= first < last <= self.last
        ):
            raise ValueError(
                f"IntersectionPCurve.interval must lie inside its parameter interval; got {(first, last)!r}"
            )
        return float(first), float(last)

    def _carrier(self, parameter: Array | float, /) -> Array | float:
        return self.first + self.last - parameter if self.reversed else parameter

    def bounding_box(
        self,
        first: float,
        last: float,
        /,
        *,
        endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None] | None = None,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.bounding_box(
                    first,
                    last,
                    endpoint_roots=endpoint_roots,
                    preparations=preparations,
                    budget=budget,
                )
        return self.enclosure(
            first,
            last,
            endpoint_roots=endpoint_roots,
            preparations=preparations,
            budget=budget,
        )

    def _rooted_query_range(
        self,
        first: float,
        last: float,
        endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None] | None,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[list[float], bool]:
        first_, last_, extended = self.curve._rooted_query_range(
            first,
            last,
            endpoint_roots,
            domain_lower=self.first,
            domain_upper=self.last,
            preparations=preparations,
            budget=budget,
        )
        if extended and self.reversed:
            raise ValueError(
                "IntersectionPCurve.endpoints must retain the scalar source for reversed extension; got a changed reversed scalar source"
            )
        return sorted(
            (float(self._carrier(first_)), float(self._carrier(last_)))
        ), extended

    def is_c1_on(
        self,
        first: float,
        last: float,
        /,
        *,
        endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None] | None = None,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> bool:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.is_c1_on(
                    first,
                    last,
                    endpoint_roots=endpoint_roots,
                    preparations=preparations,
                    budget=budget,
                )
        endpoints, extended = self._rooted_query_range(
            first, last, endpoint_roots, preparations=preparations, budget=budget
        )
        if extended and not np.all(
            np.isfinite(
                self.curve.parameter_enclosures(
                    *endpoints,
                    source_extension=True,
                    minimum_chart=min(
                        int(math.floor(self.first)), self.curve.num_charts - 1
                    ),
                    maximum_chart=min(
                        int(math.ceil(self.last)) - 1, self.curve.num_charts - 1
                    ),
                    preparations=preparations,
                    budget=budget,
                )
            )
        ):
            raise ValueError(
                "IntersectionPCurve.rooted_range must remain inside certified chart support; got an unsupported rooted range"
            )
        return _intersection_is_c1(
            self.curve,
            max(self.first, endpoints[0]),
            min(self.last, endpoints[1]),
            minimum_chart=min(int(math.floor(self.first)), self.curve.num_charts - 1),
            maximum_chart=min(int(math.ceil(self.last)) - 1, self.curve.num_charts - 1),
            preparations=preparations,
            budget=budget,
        )

    def derivative_bounds(
        self,
        first: float,
        last: float,
        /,
        *,
        order: int = 1,
        endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None] | None = None,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
        restriction_bank: BernsteinRestrictionBank | None = None,
        meridian_jets: list[MeridianOverlaps] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.derivative_bounds(
                    first,
                    last,
                    order=order,
                    endpoint_roots=endpoint_roots,
                    preparations=preparations,
                    budget=budget,
                    restriction_bank=restriction_bank,
                    meridian_jets=meridian_jets,
                )
        endpoints, extended = self._rooted_query_range(
            first, last, endpoint_roots, preparations=preparations, budget=budget
        )
        lower, upper = intersection_derivative_bounds(
            self.curve,
            endpoints[0],
            endpoints[1],
            order,
            self.side,
            minimum_chart=min(int(math.floor(self.first)), self.curve.num_charts - 1),
            maximum_chart=min(int(math.ceil(self.last)) - 1, self.curve.num_charts - 1),
            source_extension=extended,
            preparations=preparations,
            budget=budget,
        )
        return (-upper, -lower) if self.reversed and order == 1 else (lower, upper)

    def evaluate(
        self,
        parameters: Array,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> Array:
        with coordinate_source_request(
            (
                self,
                parameters,
            ),
            budget=budget,
        ) as budget:
            first_chart = min(int(math.floor(self.first)), self.curve.num_charts - 1)
            last_chart = min(int(math.ceil(self.last)) - 1, self.curve.num_charts - 1)
            result = self.curve._evaluate_in_chart_range(
                self._carrier(parameters),
                first_chart,
                last_chart,
                certify=False,
                preparations=preparations,
                budget=budget,
            )
            match self.side:
                case "first":
                    return result.first_parameters
                case "second":
                    return result.second_parameters
                case _:
                    raise ValueError(
                        f"IntersectionPCurve.side must name a generating side; got {self.side!r}"
                    )

    def host_points(
        self,
        parameters: ArrayLike,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        """Concrete rows through the shared bucketed atlas program.

        An explicit request reuses its prepared chart rows of this branch.
        """
        first_chart = min(int(math.floor(self.first)), self.curve.num_charts - 1)
        last_chart = min(int(math.ceil(self.last)) - 1, self.curve.num_charts - 1)
        result = self.curve._evaluate_in_chart_range(
            _host_carrier(self, np.asarray(parameters, dtype=np.float64)),
            first_chart,
            last_chart,
            certify=False,
            preparations=preparations,
            budget=budget,
        )
        return np.asarray(
            result.first_parameters if self.side == "first" else result.second_parameters
        )

    def shares_endpoint(
        self,
        other: AbstractTrimCurve,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> bool:
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    other,
                )
            ) as budget:
                return self.shares_endpoint(other, budget=budget)
        return trim_source_join(self, other, budget=budget)

    def enclosure(
        self,
        first: float,
        last: float,
        /,
        *,
        endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None] | None = None,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.enclosure(
                    first,
                    last,
                    endpoint_roots=endpoint_roots,
                    preparations=preparations,
                    budget=budget,
                )
        endpoints, extended = self._rooted_query_range(
            first, last, endpoint_roots, preparations=preparations, budget=budget
        )
        boxes = self.curve.parameter_enclosures(
            *endpoints,
            minimum_chart=min(int(math.floor(self.first)), self.curve.num_charts - 1),
            maximum_chart=min(int(math.ceil(self.last)) - 1, self.curve.num_charts - 1),
            source_extension=extended,
            preparations=preparations,
            budget=budget,
        )
        if not np.all(np.isfinite(boxes)):
            raise ValueError(
                "IntersectionPCurve.rooted_range must remain inside certified chart support; got an unsupported rooted range"
            )
        columns = self._columns()
        return np.stack(
            (
                np.min(boxes[:, 0, columns], axis=0),
                np.max(boxes[:, 1, columns], axis=0),
            )
        )


def _fraction_array_bounds(values: Any, /) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(values, dtype=object)
    pairs = [fraction_bounds(value) for value in source.reshape(-1)]
    return tuple(
        np.asarray([pair[index] for pair in pairs]).reshape(source.shape)
        for index in (0, 1)
    )


class AffinePCurve(AbstractCurve, AbstractTrimCurve):
    """Apply ``matrix @ curve(t) + offset`` without changing source parameters.

    No conic or rational control net is refitted. UV transport never changes
    the parameter atom, root expression, orientation or intrinsic domain.
    """

    curve: AbstractCurve | AbstractTrimCurve
    matrix: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]] = eqx.field(
        static=True
    )
    offset: tuple[Fraction, Fraction] = eqx.field(static=True)
    matrix_value: Array
    offset_value: Array

    @checked
    def __init__(
        self,
        curve: AbstractCurve | AbstractTrimCurve,
        matrix: ConvertibleToArray | _ExactUVMatrix,
        offset: ConvertibleToArray | tuple[Fraction, Fraction],
    ) -> None:
        if curve.ambient_dimension != 2:
            raise TypeError(
                f"AffinePCurve.curve must have ambient dimension two; got {curve.ambient_dimension!r}"
            )
        matrix_ = np.asarray(matrix, dtype=object)
        offset_ = np.asarray(offset, dtype=object)
        if matrix_.shape != (2, 2) or offset_.shape != (2,):
            raise ValueError(
                f"AffinePCurve.transform must contain a 2x2 matrix and two-vector; got {(matrix_.shape, offset_.shape)!r}"
            )

        def rational(value: object) -> Fraction:
            if isinstance(value, Fraction):
                return value
            if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
                return Fraction(float(value))
            if isinstance(value, (int, np.integer)) and not isinstance(
                value, (bool, np.bool_)
            ):
                return Fraction(int(value))
            raise ValueError(
                f"AffinePCurve.coefficients must be finite rational or binary numeric values; got {value!r}"
            )

        exact_matrix = (
            (rational(matrix_[0, 0]), rational(matrix_[0, 1])),
            (rational(matrix_[1, 0]), rational(matrix_[1, 1])),
        )
        exact_offset = (rational(offset_[0]), rational(offset_[1]))
        a, b = exact_matrix[0]
        c, d = exact_matrix[1]
        if a * d == b * c:
            raise ValueError(
                f"AffinePCurve.matrix must be invertible in rational arithmetic; got {exact_matrix!r}"
            )
        nominal_matrix = np.asarray(exact_matrix, dtype=np.float64)
        nominal_offset = np.asarray(exact_offset, dtype=np.float64)
        if not np.all(np.isfinite(nominal_matrix)) or not np.all(
            np.isfinite(nominal_offset)
        ):
            raise ValueError(
                "AffinePCurve.coefficients must have finite float64 representatives; got nonfinite coefficient representatives"
            )
        self.curve, self.matrix, self.offset = curve, exact_matrix, exact_offset
        self.matrix_value, self.offset_value = (
            jnp.asarray(nominal_matrix),
            jnp.asarray(nominal_offset),
        )

    @property
    def ambient_dimension(self) -> int:
        return 2

    @property
    def parameter_domain(self) -> tuple[float, float] | None:
        return self.curve.parameter_domain

    @property
    def parameter_interval(self) -> tuple[float, float]:
        return (
            (-math.inf, math.inf)
            if self.parameter_domain is None
            else self.parameter_domain
        )

    @property
    def period(self) -> float | None:
        return self.curve.period

    @property
    def source_curve(self) -> AbstractCurve | AbstractTrimCurve:
        source = self.curve
        while isinstance(source, (AffinePCurve, PeriodicPCurve)):
            source = (
                source.curve if isinstance(source, AffinePCurve) else source.source_curve
            )
        return source

    @property
    def source_definition(self) -> dict[str, Any]:
        """Constructor-only closure, including the coupled branch when present."""
        return encode_geometry(self)

    @property
    def source_id(self) -> str:
        return canonical_fingerprint(self.source_definition)

    def coefficient_enclosures(
        self,
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
        """Return source intervals and rational UV coefficient enclosures."""
        return (
            *coefficient_enclosures(self.curve),
            (np.asarray(self.matrix_value), *_fraction_array_bounds(self.matrix)),
            (np.asarray(self.offset_value), *_fraction_array_bounds(self.offset)),
        )

    def native_period_symbol(self) -> str | None:
        """Return the source curve's proved mathematical period."""
        match self.curve:
            case AbstractCurve():
                return self.curve.native_period_symbol()
            case _:
                return None

    def source_pieces(
        self,
        first: float,
        last: float,
        /,
        *,
        coordinate_budget: CoordinateEnclosureBudget,
    ) -> tuple[CurvePiece[AbstractCurve], ...]:
        """Transport source span restrictions through the rational UV map."""
        return tuple(
            CurvePiece(
                AffinePCurve(piece.evaluator, self.matrix, self.offset),
                piece.lower,
                piece.upper,
                piece.index,
            )
            for piece in curve_pieces_for_interval(
                self.curve, first, last, coordinate_budget=coordinate_budget
            )
        )

    def exact_affine_map(self) -> _ExactUVMap:
        """Composed UV map, including symbolic native period offsets."""
        _, matrix, offset = _pcurve_affine_source(self)
        return matrix, offset

    def validate_range(self, first: float, last: float, /) -> tuple[float, float]:
        curve = self.curve
        if isinstance(curve, AbstractTrimCurve):
            return type(curve).validate_range(curve, first, last)
        return type(curve).validate_range(curve, first, last)

    def validate_query_range(self, first: float, last: float, /) -> tuple[float, float]:
        if not (math.isfinite(first) and math.isfinite(last) and first <= last):
            raise ValueError(
                f"AffinePCurve.interval must be a finite closed subinterval; got {(first, last)!r}"
            )
        if first < last:
            return type(self).validate_range(self, first, last)
        domain = self.parameter_domain
        if domain is not None and not domain[0] <= first <= domain[1]:
            raise ValueError(
                f"AffinePCurve.interval must lie inside the source domain; got {(first, last)!r}"
            )
        return float(first), float(last)

    def evaluate(
        self,
        parameters: Array,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> Array:
        match self.curve:
            case AbstractTrimCurve() as curve:
                value = curve.evaluate(
                    parameters, budget=budget, preparations=preparations
                )
            case AbstractCurve() as curve:
                value = curve.evaluate(parameters)
            case _:
                raise TypeError(
                    f"AffinePCurve.curve must provide a native curve or trim curve; got {type(self.curve).__name__}"
                )
        return value @ self.matrix_value.T + self.offset_value

    def _transport_bounds(self, bounds: np.ndarray, /, *, translate: bool) -> np.ndarray:

        result = []
        for row, offset in zip(self.matrix, self.offset, strict=True):
            value = fraction_bounds(offset if translate else Fraction(0))
            for coefficient, lower, upper in zip(row, bounds[0], bounds[1], strict=True):
                if coefficient:
                    value = interval_add(
                        value,
                        interval_multiply(fraction_bounds(coefficient), (lower, upper)),
                    )
            result.append(value)
        return np.asarray(result).T

    def enclosure(
        self,
        first: float,
        last: float,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return type(self).enclosure(
                    self, first, last, preparations=preparations, budget=budget
                )
        type(self).validate_query_range(self, first, last)
        match self.curve:
            case AbstractTrimCurve() as curve:
                bounds = type(curve).enclosure(
                    curve, first, last, preparations=preparations, budget=budget
                )
            case AbstractCurve() as curve if first < last:
                bounds = type(curve).bounding_box(
                    curve, first, last, preparations=preparations, budget=budget
                )
            case AbstractCurve() as curve:
                images = []
                for piece in curve_pieces_for_interval(
                    curve, first, last, coordinate_budget=budget
                ):
                    evaluator = piece.evaluator
                    prepared = prepare_interval_function(
                        lambda parameter: evaluator.evaluate(parameter[0]),
                        1,
                        batch_capacity=1,
                        constant_bounds=coefficient_enclosures(evaluator),
                    )
                    lower, upper = prepared.evaluate(
                        np.asarray([[first]]), np.asarray([[last]])
                    )
                    images.append(np.stack((lower[0], upper[0])))
                boxes = np.asarray(images)
                bounds = np.stack(
                    (np.min(boxes[:, 0], axis=0), np.max(boxes[:, 1], axis=0))
                )
            case _:
                raise TypeError(
                    f"AffinePCurve.curve must provide a native curve or trim curve; got {type(self.curve).__name__}"
                )
        return type(self)._transport_bounds(self, np.asarray(bounds), translate=True)

    def bounding_box(
        self,
        first: float,
        last: float,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return type(self).bounding_box(
                    self, first, last, preparations=preparations, budget=budget
                )
        return type(self).enclosure(
            self, first, last, preparations=preparations, budget=budget
        )

    def is_c1_on(
        self,
        first: float,
        last: float,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> bool:
        self.validate_query_range(first, last)
        return self.curve.is_c1_on(first, last, budget=budget, preparations=preparations)

    def derivative_bounds(
        self,
        first: float,
        last: float,
        /,
        *,
        order: int = 1,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
        restriction_bank: BernsteinRestrictionBank | None = None,
        meridian_jets: list[MeridianOverlaps] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.derivative_bounds(
                    first,
                    last,
                    order=order,
                    preparations=preparations,
                    budget=budget,
                    restriction_bank=restriction_bank,
                    meridian_jets=meridian_jets,
                )
        self.validate_query_range(first, last)
        if order not in (1, 2) or isinstance(order, bool):
            raise ValueError(
                f"AffinePCurve.derivative_order must be one or two; got {order!r}"
            )
        if order == 2 and not self.is_c1_on(
            first, last, preparations=preparations, budget=budget
        ):
            return np.full(2, -np.inf), np.full(2, np.inf)
        bounds = np.stack(
            self.curve.derivative_bounds(
                first,
                last,
                order=order,
                preparations=preparations,
                budget=budget,
                restriction_bank=restriction_bank,
                meridian_jets=meridian_jets,
            )
        )
        lower, upper = self._transport_bounds(bounds, translate=False)
        return lower, upper

    def shares_endpoint(
        self,
        other: AbstractTrimCurve,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> bool:
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    other,
                )
            ) as budget:
                return self.shares_endpoint(other, budget=budget)
        return trim_source_join(self, other, budget=budget)


def exact_line_pcurve(
    origin: tuple[Fraction, Fraction], direction: tuple[float, float], /
) -> LineCurve | AffinePCurve:
    """Deterministic UV line through an origin.

    A binary64-representable origin is the plain ``LineCurve``; otherwise the
    origin is retained as an identity ``AffinePCurve`` offset of the line
    through zero. Constructors and readers share this one representation, so
    the same p-curve has one geometry identity.
    """
    rounded = tuple(float(value) for value in origin)
    signless = tuple(0.0 if value == 0.0 else value for value in direction)
    if all(
        Fraction(value) == exact for value, exact in zip(rounded, origin, strict=True)
    ):
        return LineCurve(
            tuple(0.0 if value == 0.0 else value for value in rounded), signless
        )
    return AffinePCurve(
        LineCurve((0.0, 0.0), signless),
        ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))),
        origin,
    )


_ExactUVMatrix: TypeAlias = tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]
_ExactUVOffset: TypeAlias = tuple[Fraction | PeriodOffset, Fraction | PeriodOffset]
_ExactUVMap: TypeAlias = tuple[_ExactUVMatrix, _ExactUVOffset]
_EndpointTransportTag: TypeAlias = (
    tuple[Literal["branch"], str, IntersectionCurveSide]
    | tuple[Literal["branch-source"], str, IntersectionCurveSide | None, str]
    | None
)
_EndpointTransport: TypeAlias = tuple[
    _ExactUVMatrix, _ExactUVOffset, _EndpointTransportTag
]


def affine_curve_coefficients(
    curve: AbstractCurve | None, point: np.ndarray, /
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]] | None:
    """Affine source coefficients, including deterministic rational UV maps."""
    if curve is None:
        return (
            rational_vector(point),
            (Fraction(0),) * point.size,
        )
    if isinstance(curve, AffinePCurve):
        if not isinstance(curve.curve, AbstractCurve):
            return None
        source = affine_curve_coefficients(curve.curve, point)
        if source is None:
            return None
        constant = tuple(
            offset
            + sum(
                (
                    coefficient * value
                    for coefficient, value in zip(row, source[0], strict=True)
                ),
                Fraction(0),
            )
            for row, offset in zip(curve.matrix, curve.offset, strict=True)
        )
        derivative = tuple(
            sum(
                (
                    coefficient * value
                    for coefficient, value in zip(row, source[1], strict=True)
                ),
                Fraction(0),
            )
            for row in curve.matrix
        )
        return constant, derivative
    if isinstance(curve, LineCurve):
        return (
            rational_vector(np.asarray(curve.origin)),
            rational_vector(np.asarray(curve.direction)),
        )
    if (
        not isinstance(curve, BSplineCurve)
        or curve.degree != 1
        or curve.control_points.shape[0] != 2
    ):
        return None
    weights = np.asarray(curve.weights)
    knots = np.asarray(curve.knots)
    controls = np.asarray(curve.control_points)
    source_weights = rational_vector(weights)
    if (
        source_weights[0] <= 0
        or source_weights[0] != source_weights[1]
        or knots.shape != (4,)
    ):
        return None
    source_knots = rational_vector(knots)
    if source_knots[0] != source_knots[1] or source_knots[2] != source_knots[3]:
        return None
    first = source_knots[1]
    span = source_knots[2] - first
    if span <= 0:
        return None
    derivative = tuple(
        (Fraction(float(last)) - Fraction(float(start))) / span
        for start, last in zip(controls[0], controls[1], strict=True)
    )
    origin = tuple(
        Fraction(float(start)) - first * slope
        for start, slope in zip(controls[0], derivative, strict=True)
    )
    return origin, derivative


class PeriodicPCurve(AbstractCurve, AbstractTrimCurve):
    """UV source plus integer multiples of mathematical native periods.

    The patch definition and integer gauge are construction data. Neither source
    coefficients nor parameter/root atoms are replaced. Numerical coordinates
    are representatives only; enclosures include the irrational period.
    Distinct gauges are distinct UV sheets even when their 3D images coincide.
    """

    source_curve: AbstractCurve | AbstractTrimCurve
    patch: AbstractSurfacePatch
    period_shifts: tuple[int, int] = eqx.field(static=True)
    offset_value: Array

    @checked
    def __init__(
        self,
        source_curve: AbstractCurve | AbstractTrimCurve,
        patch: AbstractSurfacePatch,
        period_shifts: tuple[int, int] = (0, 0),
    ) -> None:
        if source_curve.ambient_dimension != 2:
            raise TypeError(
                f"PeriodicPCurve.source_curve must have ambient dimension two; got {source_curve.ambient_dimension!r}"
            )
        shifts = tuple(period_shifts)
        if len(shifts) != 2 or any(
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            for value in shifts
        ):
            raise ValueError(
                f"PeriodicPCurve.period_shifts must contain two integers; got {shifts!r}"
            )
        symbols = patch.native_period_symbols()
        if any(
            shift and symbol is None
            for shift, symbol in zip(shifts, symbols, strict=True)
        ):
            raise ValueError(
                f"PeriodicPCurve.period_shifts must use a proved mathematical period; got {shifts!r}"
            )
        nominal = np.asarray(
            [int(shift) * (2 * math.pi) for shift in shifts], dtype=np.float64
        )
        if not np.all(np.isfinite(nominal)):
            raise ValueError(
                f"PeriodicPCurve.period_shifts must have finite numerical representatives; got {shifts!r}"
            )
        self.source_curve, self.patch = source_curve, patch
        self.period_shifts = (int(shifts[0]), int(shifts[1]))
        self.offset_value = jnp.asarray(nominal)

    @property
    def ambient_dimension(self) -> int:
        return 2

    @property
    def parameter_domain(self) -> tuple[float, float] | None:
        return self.source_curve.parameter_domain

    @property
    def parameter_interval(self) -> tuple[float, float]:
        return (
            (-math.inf, math.inf)
            if self.parameter_domain is None
            else self.parameter_domain
        )

    @property
    def period(self) -> float | None:
        return self.source_curve.period

    @property
    def exact_periods(self) -> tuple[str | None, str | None]:
        return self.patch.native_period_symbols()

    @property
    def source_definition(self) -> dict[str, object]:
        return encode_geometry(self)

    @property
    def source_id(self) -> str:
        return canonical_fingerprint(self.source_definition)

    @classmethod
    @checked
    def from_stored_fields(cls: type[Self], value: PeriodicPCurve, /) -> Self:
        """Rebuild the period transport from the retained source and sheet shifts."""
        return cls(value.source_curve, value.patch, value.period_shifts)

    def coefficient_enclosures(
        self,
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
        """Return source intervals and symbolic period-offset enclosures."""
        source = coefficient_enclosures(self.source_curve)
        lower, upper = self.period_offset_bounds()
        return (
            *source,
            (np.asarray(self.offset_value), lower, upper),
        )

    def native_period_symbol(self) -> str | None:
        """Return the source curve's proved mathematical period."""
        match self.source_curve:
            case AbstractCurve():
                return self.source_curve.native_period_symbol()
            case _:
                return None

    def source_pieces(
        self,
        first: float,
        last: float,
        /,
        *,
        coordinate_budget: CoordinateEnclosureBudget,
    ) -> tuple[CurvePiece[AbstractCurve], ...]:
        """Retain source span restrictions on this integer-period sheet."""
        return tuple(
            CurvePiece(
                PeriodicPCurve(piece.evaluator, self.patch, self.period_shifts),
                piece.lower,
                piece.upper,
                piece.index,
            )
            for piece in curve_pieces_for_interval(
                self.source_curve, first, last, coordinate_budget=coordinate_budget
            )
        )

    @property
    def gauge_definition(self) -> dict[str, object]:
        return {
            "patch": encode_geometry(self.patch),
            "periods": self.exact_periods,
            "period_shifts": self.period_shifts,
        }

    def exact_affine_map(self) -> _ExactUVMap:
        _, matrix, offset = _pcurve_affine_source(self)
        return matrix, offset

    def period_offset_bounds(self) -> np.ndarray:

        period = (
            np.asarray(np.nextafter(2 * math.pi, -np.inf)),
            np.asarray(np.nextafter(2 * math.pi, np.inf)),
        )
        return np.asarray(
            [
                interval_multiply(fraction_bounds(Fraction(shift)), period)
                if shift
                else (0.0, 0.0)
                for shift in self.period_shifts
            ]
        ).T

    def validate_range(self, first: float, last: float, /) -> tuple[float, float]:
        return self.source_curve.validate_range(first, last)

    def validate_query_range(self, first: float, last: float, /) -> tuple[float, float]:
        return AbstractCurve.validate_query_range(self, first, last)

    def evaluate(
        self,
        parameters: Array,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> Array:
        match self.source_curve:
            case AbstractTrimCurve() as curve:
                value = curve.evaluate(
                    parameters, budget=budget, preparations=preparations
                )
            case AbstractCurve() as curve:
                value = curve.evaluate(parameters)
            case _:
                raise TypeError(
                    f"PeriodicPCurve.source_curve must provide a native curve or trim curve; got {type(self.source_curve).__name__}"
                )
        return value + self.offset_value

    def _transport_bounds(self, bounds: np.ndarray, /) -> np.ndarray:

        offsets = type(self).period_offset_bounds(self)
        return np.asarray(
            [
                interval_add(
                    (np.asarray(bounds[0, axis]), np.asarray(bounds[1, axis])),
                    (np.asarray(offsets[0, axis]), np.asarray(offsets[1, axis])),
                )
                if self.period_shifts[axis]
                else (bounds[0, axis], bounds[1, axis])
                for axis in range(2)
            ]
        ).T

    def enclosure(
        self,
        first: float,
        last: float,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.enclosure(
                    first, last, preparations=preparations, budget=budget
                )
        self.validate_query_range(first, last)
        return self._transport_bounds(
            _trim_carrier_enclosure(
                self.source_curve,
                first,
                last,
                first,
                last,
                preparations=preparations,
                budget=budget,
            )
        )

    def bounding_box(
        self,
        first: float,
        last: float,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.bounding_box(
                    first, last, preparations=preparations, budget=budget
                )
        return self.enclosure(first, last, preparations=preparations, budget=budget)

    def is_c1_on(
        self,
        first: float,
        last: float,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> bool:
        self.validate_query_range(first, last)
        return self.source_curve.is_c1_on(
            first, last, budget=budget, preparations=preparations
        )

    def derivative_bounds(
        self,
        first: float,
        last: float,
        /,
        *,
        order: int = 1,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
        restriction_bank: BernsteinRestrictionBank | None = None,
        meridian_jets: list[MeridianOverlaps] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.derivative_bounds(
                    first,
                    last,
                    order=order,
                    preparations=preparations,
                    budget=budget,
                    restriction_bank=restriction_bank,
                    meridian_jets=meridian_jets,
                )
        self.validate_query_range(first, last)
        if order not in (1, 2) or isinstance(order, bool):
            raise ValueError(
                f"PeriodicPCurve.derivative_order must be one or two; got {order!r}"
            )
        if order == 2 and not self.is_c1_on(
            first, last, preparations=preparations, budget=budget
        ):
            return np.full(2, -np.inf), np.full(2, np.inf)
        return self.source_curve.derivative_bounds(
            first,
            last,
            order=order,
            preparations=preparations,
            budget=budget,
            restriction_bank=restriction_bank,
            meridian_jets=meridian_jets,
        )

    def shares_endpoint(
        self,
        other: AbstractTrimCurve,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> bool:
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    other,
                )
            ) as budget:
                return self.shares_endpoint(other, budget=budget)
        return trim_source_join(self, other, budget=budget)


def pcurve_periodic_source(
    curve: CurveEvaluator,
    patch: AbstractSurfacePatch,
    /,
) -> tuple[CurveEvaluator, tuple[int, int]] | None:
    """Extract a proved integer sheet gauge on this patch definition.

    Callers can cancel these integers against coupled atlas node/root shifts
    BEFORE enclosing periods. An affine transport with an incompatible gauge
    cannot be stripped, and a floating seam coordinate supplies no such proof.
    Physical placement retains its definition's UV chart and native periods;
    admitting that retained ancestor does not cancel or identify poses.
    """

    definitions = [encode_geometry(patch)]
    definition = patch
    while isinstance(definition, placed_owner.PlacedSurface):
        definition = definition.definition
        definitions.append(encode_geometry(definition))
    shifts = [0, 0]
    while isinstance(curve, PeriodicPCurve):
        if encode_geometry(curve.patch) not in definitions:
            return None
        shifts = [a + b for a, b in zip(shifts, curve.period_shifts, strict=True)]
        curve = curve.source_curve
    if isinstance(curve, AffinePCurve):
        source = pcurve_periodic_source(curve.curve, patch)
        if source is None:
            return None
        base, inner = source
        if any(inner):
            transported = (
                curve.matrix[0][0] * inner[0] + curve.matrix[0][1] * inner[1],
                curve.matrix[1][0] * inner[0] + curve.matrix[1][1] * inner[1],
            )
            periods = patch.native_period_symbols()
            if any(
                value.denominator != 1 or (value and period is None)
                for value, period in zip(transported, periods, strict=True)
            ):
                return None
            inner = (int(transported[0]), int(transported[1]))
        curve = (
            curve
            if base is curve.curve
            else AffinePCurve(base, curve.matrix, curve.offset)
        )
        shifts = [a + b for a, b in zip(shifts, inner, strict=True)]
    return curve, (shifts[0], shifts[1])


class CurveTrimSegment(AbstractTrimCurve):
    """Restrict and orient a two-dimensional source curve for a trim edge."""

    curve: AbstractCurve | AbstractTrimCurve
    first: float = eqx.field(static=True)
    last: float = eqx.field(static=True)
    reversed: bool = eqx.field(static=True)
    first_root: RootEndpoint | None
    last_root: RootEndpoint | None

    @checked
    def __init__(
        self,
        curve: AbstractCurve | AbstractTrimCurve,
        first: float,
        last: float,
        *,
        reversed: bool = False,
        first_root: RootEndpoint | None = None,
        last_root: RootEndpoint | None = None,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> None:
        if preparations is None:
            preparations = TrimRequestPreparations()
        with coordinate_source_request(
            (
                curve,
                first,
                last,
                reversed,
                first_root,
                last_root,
            ),
            budget=budget,
        ) as budget:
            if isinstance(curve, AbstractCurve):
                if curve.ambient_dimension != 2:
                    raise ValueError(
                        f"CurveTrimSegment.curve must have ambient dimension two; got {curve.ambient_dimension!r}"
                    )
                first_, last_ = curve.validate_range(first, last)
            else:
                first_, last_ = float(first), float(last)
                lower, upper = curve.parameter_interval
                if not (
                    math.isfinite(first_) and math.isfinite(last_) and first_ < last_
                ):
                    raise ValueError(
                        f"CurveTrimSegment.interval must be finite and ordered; got {(first_, last_)!r}"
                    )
                if not lower <= first_ < last_ <= upper:
                    if not isinstance(curve, IntersectionPCurve) or curve.reversed:
                        raise ValueError(
                            f"CurveTrimSegment.interval must lie inside the curve domain; got {(first_, last_)!r}"
                        )
                    curve.curve._rooted_query_range(
                        first_,
                        last_,
                        (first_root, last_root),
                        domain_lower=lower,
                        domain_upper=upper,
                        budget=budget,
                        preparations=preparations,
                    )
            self.curve = curve
            self.first = first_
            self.last = last_
            self.reversed = bool(reversed)

            for endpoint, parameter in ((first_root, first_), (last_root, last_)):
                if endpoint is not None:
                    if not isinstance(
                        endpoint,
                        (
                            intersection_owner.TrimRootEndpoint,
                            intersection_owner.BranchRootEndpoint,
                            intersection_owner.NativePeriodEndpoint,
                        ),
                    ):
                        raise TypeError(
                            f"CurveTrimSegment.endpoint_root must be a source root expression; got {type(endpoint).__name__!r}"
                        )
                    if _endpoint_source_transport(curve, endpoint) is None:
                        raise ValueError(
                            "CurveTrimSegment.endpoint_root must bind the source carrier through its UV transport; got an unproved source-carrier transport"
                        )
                    lower, upper = endpoint.parameter_enclosure(
                        budget=budget, preparations=preparations
                    )
                    if not lower <= parameter <= upper:
                        raise ValueError(
                            f"CurveTrimSegment.endpoint must lie inside its root enclosure; got {parameter!r}"
                        )
            self.first_root, self.last_root = first_root, last_root
            first_box = (
                (first_, first_)
                if first_root is None
                else first_root.parameter_enclosure(
                    budget=budget, preparations=preparations
                )
            )
            last_box = (
                (last_, last_)
                if last_root is None
                else last_root.parameter_enclosure(
                    budget=budget, preparations=preparations
                )
            )
            if first_box[1] >= last_box[0]:
                raise ValueError(
                    "CurveTrimSegment.endpoints must have proved parameter ordering; got overlapping endpoint parameter bounds"
                )

    @property
    def parameter_interval(self) -> tuple[float, float]:
        return 0.0, 1.0

    @overload
    def source_parameters(self, local: Array, /) -> Array: ...

    @overload
    def source_parameters(self, local: float, /) -> float: ...

    def source_parameters(self, local: Array | float, /) -> Array | float:
        fraction = 1.0 - local if self.reversed else local
        return self.first + fraction * (self.last - self.first)

    def carrier_parameter_enclosure(
        self,
        first: float,
        last: float,
        /,
        *,
        source_extension: bool = False,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.carrier_parameter_enclosure(
                    first,
                    last,
                    source_extension=source_extension,
                    preparations=preparations,
                    budget=budget,
                )
        if not (math.isfinite(first) and math.isfinite(last) and first <= last) or (
            not source_extension and not 0 <= first <= last <= 1
        ):
            raise ValueError(
                f"CurveTrimSegment.interval must be an ordered source subinterval; got {(first, last)!r}"
            )
        intersection_owner.admit_segment_endpoint_roots(
            (self.first_root, self.last_root), preparations, budget=budget
        )

        # type(root) selects this same root's method; ty loses that union correlation.
        a = (
            (self.first, self.first)
            if self.first_root is None
            else type(self.first_root).parameter_enclosure(
                self.first_root, preparations=preparations, budget=budget  # ty: ignore[invalid-argument-type]
            )
        )
        b = (
            (self.last, self.last)
            if self.last_root is None
            else type(self.last_root).parameter_enclosure(
                self.last_root, preparations=preparations, budget=budget  # ty: ignore[invalid-argument-type]
            )
        )
        endpoints = []
        for local in (first, last):
            fraction = 1.0 - local if self.reversed else local
            if fraction == 0.0:
                endpoints.append(a)
            elif fraction == 1.0:
                endpoints.append(b)
            else:
                complement = (
                    np.nextafter(1.0 - fraction, -np.inf),
                    np.nextafter(1.0 - fraction, np.inf),
                )
                lo, hi = interval_add(
                    interval_multiply(
                        (np.asarray(a[0]), np.asarray(a[1])),
                        (np.asarray(complement[0]), np.asarray(complement[1])),
                    ),
                    interval_multiply(
                        (np.asarray(b[0]), np.asarray(b[1])),
                        (np.asarray(fraction), np.asarray(fraction)),
                    ),
                )
                endpoints.append((float(lo), float(hi)))
        lower, upper = (
            min(value[0] for value in endpoints),
            max(value[1] for value in endpoints),
        )
        domain = self.curve.parameter_domain
        if (
            not source_extension
            and domain is not None
            and a[0] >= domain[0]
            and b[1] <= domain[1]
        ):
            lower, upper = max(lower, domain[0]), min(upper, domain[1])
        return np.asarray(lower), np.asarray(upper)

    def evaluate(
        self,
        parameters: Array,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> Array:
        with coordinate_source_request(
            (
                self,
                parameters,
            ),
            budget=budget,
        ) as budget:
            local = jnp.asarray(parameters, dtype=jnp.float64)
            return _trim_carrier_evaluate(
                self.curve,
                self.source_parameters(local),
                self.first,
                self.last,
                preparations=preparations,
                budget=budget,
            )

    def host_points(
        self,
        parameters: ArrayLike,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        """Concrete rows through the shared atlas program for intersection carriers."""
        values = _trim_carrier_host(
            self.curve,
            _host_source_parameters(self, np.asarray(parameters, dtype=np.float64)),
            self.first,
            self.last,
            preparations=preparations,
            budget=budget,
        )
        if values is None:
            return AbstractTrimCurve.host_points(
                self, parameters, preparations=preparations, budget=budget
            )
        return values

    def is_c1_on(
        self,
        first: float,
        last: float,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> bool:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.is_c1_on(
                    first, last, preparations=preparations, budget=budget
                )
        endpoints = self.carrier_parameter_enclosure(
            first, last, preparations=preparations, budget=budget
        )
        return _trim_carrier_is_c1(
            self.curve,
            float(endpoints[0]),
            float(endpoints[1]),
            self.first,
            self.last,
            preparations=preparations,
            budget=budget,
        )

    def derivative_bounds(
        self,
        first: float,
        last: float,
        /,
        *,
        order: int = 1,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
        restriction_bank: BernsteinRestrictionBank | None = None,
        meridian_jets: list[MeridianOverlaps] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.derivative_bounds(
                    first,
                    last,
                    order=order,
                    preparations=preparations,
                    budget=budget,
                    restriction_bank=restriction_bank,
                    meridian_jets=meridian_jets,
                )
        if order not in (1, 2) or isinstance(order, bool):
            raise ValueError(
                f"CurveTrimSegment.derivative_order must be one or two; got {order!r}"
            )
        if not (0 <= first <= last <= 1):
            raise ValueError(
                f"CurveTrimSegment.interval must lie inside [0, 1]; got {(first, last)!r}"
            )
        endpoints = type(self).carrier_parameter_enclosure(
            self, first, last, preparations=preparations, budget=budget
        )
        if order == 2 and not type(self).is_c1_on(
            self, first, last, preparations=preparations, budget=budget
        ):
            return np.full(2, -np.inf), np.full(2, np.inf)
        lower, upper = _trim_carrier_derivatives(
            self.curve,
            float(endpoints[0]),
            float(endpoints[1]),
            self.first,
            self.last,
            order,
            preparations=preparations,
            budget=budget,
        )
        # type(root) selects this same root's method; ty loses that union correlation.
        a = (
            (self.first, self.first)
            if self.first_root is None
            else type(self.first_root).parameter_enclosure(
                self.first_root, preparations=preparations, budget=budget  # ty: ignore[invalid-argument-type]
            )
        )
        b = (
            (self.last, self.last)
            if self.last_root is None
            else type(self.last_root).parameter_enclosure(
                self.last_root, preparations=preparations, budget=budget  # ty: ignore[invalid-argument-type]
            )
        )
        delta = (np.nextafter(b[0] - a[1], -np.inf), np.nextafter(b[1] - a[0], np.inf))
        scale = (
            delta
            if order == 1
            else (
                np.nextafter(delta[0] ** 2, -np.inf),
                np.nextafter(delta[1] ** 2, np.inf),
            )
        )

        lower, upper = interval_multiply((lower, upper), scale)
        lower_ = np.asarray(lower, dtype=np.float64)
        upper_ = np.asarray(upper, dtype=np.float64)
        return (-upper_, -lower_) if self.reversed and order == 1 else (lower_, upper_)

    def shares_endpoint(
        self,
        other: AbstractTrimCurve,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> bool:
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    other,
                )
            ) as budget:
                return self.shares_endpoint(other, budget=budget)
        return trim_source_join(self, other, budget=budget)

    def enclosure(
        self,
        first: float,
        last: float,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.enclosure(
                    first, last, preparations=preparations, budget=budget
                )
        endpoints = type(self).carrier_parameter_enclosure(
            self, first, last, preparations=preparations, budget=budget
        )
        lower, upper = float(endpoints[0]), float(endpoints[1])
        return _trim_carrier_enclosure(
            self.curve,
            lower,
            upper,
            self.first,
            self.last,
            preparations=preparations,
            budget=budget,
        )

    def enclosures(
        self,
        intervals: npt.ArrayLike,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        """Row-wise ``enclosure`` of independent ``(first, last)`` intervals."""
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    intervals,
                )
            ) as budget:
                return self.enclosures(
                    intervals, preparations=preparations, budget=budget
                )
        rows = np.asarray(intervals, dtype=np.float64).reshape(-1, 2)
        carriers = np.asarray(
            [
                self.carrier_parameter_enclosure(
                    float(first), float(last), preparations=preparations, budget=budget
                )
                for first, last in rows
            ],
            dtype=np.float64,
        ).reshape(-1, 2)
        return _trim_carrier_enclosures(
            self.curve,
            carriers,
            self.first,
            self.last,
            preparations=preparations,
            budget=budget,
        )

    def arc_partition_binding(self) -> tuple[tuple[object, ...], bool]:
        """Carrier, carrier range and endpoint roots; ``reversed`` maps s to 1 - s."""
        return (
            (self.curve, self.first, self.last, self.first_root, self.last_root),
            self.reversed,
        )


def pcurve_chart_support(
    curve: IntersectionPCurve, first: float, last: float, /
) -> tuple[int, int]:
    carrier = type(curve)._carrier
    lower, upper = sorted((float(carrier(curve, first)), float(carrier(curve, last))))
    source_lower, source_upper = sorted(
        (float(carrier(curve, curve.first)), float(carrier(curve, curve.last)))
    )
    # Select the source supporting charts, not a neighboring chart reached only
    # by outward endpoint uncertainty. Query coordinates themselves are not
    # clipped: their source enclosures still require certified chart support.
    lower = max(source_lower, min(lower, source_upper))
    upper = max(source_lower, min(upper, source_upper))
    head = max(0, min(int(math.floor(lower)), curve.curve.num_charts - 1))
    return head, max(head, min(int(math.ceil(upper)) - 1, curve.curve.num_charts - 1))


def _trim_carrier_evaluate(
    curve: CurveEvaluator,
    parameters: Array,
    first: float,
    last: float,
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> Array:
    if isinstance(curve, PeriodicPCurve):
        return (
            _trim_carrier_evaluate(
                curve.source_curve,
                parameters,
                first,
                last,
                preparations=preparations,
                budget=budget,
            )
            + curve.offset_value
        )
    if isinstance(curve, AffinePCurve):
        return (
            _trim_carrier_evaluate(
                curve.curve,
                parameters,
                first,
                last,
                preparations=preparations,
                budget=budget,
            )
            @ curve.matrix_value.T
            + curve.offset_value
        )
    if isinstance(curve, IntersectionPCurve):
        head, tail = pcurve_chart_support(curve, first, last)
        result = curve.curve._evaluate_in_chart_range(
            curve._carrier(parameters),
            head,
            tail,
            certify=False,
            preparations=preparations,
            budget=budget,
        )
        return (
            result.first_parameters if curve.side == "first" else result.second_parameters
        )
    match curve:
        case AbstractTrimCurve():
            return curve.evaluate(parameters, budget=budget, preparations=preparations)
        case AbstractCurve():
            return curve.evaluate(parameters)
        case _:
            raise TypeError(
                f"CurveTrimSegment.curve must provide a native curve or trim curve; got {type(curve).__name__}"
            )


def _host_carrier(curve: IntersectionPCurve, parameters: np.ndarray, /) -> np.ndarray:
    """``IntersectionPCurve._carrier`` on concrete host parameters."""
    return curve.first + curve.last - parameters if curve.reversed else parameters


def _host_source_parameters(segment: CurveTrimSegment, local: np.ndarray, /) -> np.ndarray:
    """``CurveTrimSegment.source_parameters`` on concrete host parameters."""
    fraction = 1.0 - local if segment.reversed else local
    return segment.first + fraction * (segment.last - segment.first)


def _trim_carrier_host(
    curve: CurveEvaluator,
    parameters: np.ndarray,
    first: float,
    last: float,
    /,
    *,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget | None,
) -> np.ndarray | None:
    """``_trim_carrier_evaluate`` on concrete parameters through the bucketed atlas.

    Returns ``None`` unless the carrier reduces to an intersection p-curve,
    whose atlas program is shared across branches and ranges.
    """
    match curve:
        case PeriodicPCurve():
            inner = _trim_carrier_host(
                curve.source_curve,
                parameters,
                first,
                last,
                preparations=preparations,
                budget=budget,
            )
            return None if inner is None else inner + np.asarray(curve.offset_value)
        case AffinePCurve():
            inner = _trim_carrier_host(
                curve.curve,
                parameters,
                first,
                last,
                preparations=preparations,
                budget=budget,
            )
            return (
                None
                if inner is None
                else inner @ np.asarray(curve.matrix_value).T
                + np.asarray(curve.offset_value)
            )
        case IntersectionPCurve():
            head, tail = pcurve_chart_support(curve, first, last)
            result = curve.curve._evaluate_in_chart_range(
                _host_carrier(curve, parameters),
                head,
                tail,
                certify=False,
                preparations=preparations,
                budget=budget,
            )
            return np.asarray(
                result.first_parameters if curve.side == "first" else result.second_parameters
            )
        case _:
            return None


def _trim_carrier_is_c1(
    curve: CurveEvaluator,
    lower: float,
    upper: float,
    first: float,
    last: float,
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> bool:
    if isinstance(curve, PeriodicPCurve):
        return _trim_carrier_is_c1(
            curve.source_curve,
            lower,
            upper,
            first,
            last,
            preparations=preparations,
            budget=budget,
        )
    if isinstance(curve, AffinePCurve):
        return _trim_carrier_is_c1(
            curve.curve,
            lower,
            upper,
            first,
            last,
            preparations=preparations,
            budget=budget,
        )
    if isinstance(curve, IntersectionPCurve):
        head, tail = pcurve_chart_support(curve, first, last)
        values = sorted((float(curve._carrier(lower)), float(curve._carrier(upper))))
        boxes = curve.curve.parameter_enclosures(
            *values,
            minimum_chart=head,
            maximum_chart=tail,
            source_extension=True,
            preparations=preparations,
            budget=budget,
        )
        if not np.all(np.isfinite(boxes)):
            return False
        return _intersection_is_c1(
            curve.curve,
            max(0.0, values[0]),
            min(curve.curve.num_charts, values[1]),
            minimum_chart=head,
            maximum_chart=tail,
            preparations=preparations,
            budget=budget,
        )
    return curve.is_c1_on(lower, upper)


def _trim_carrier_derivatives(
    curve: CurveEvaluator,
    lower: float,
    upper: float,
    first: float,
    last: float,
    order: int,
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(curve, PeriodicPCurve):
        return _trim_carrier_derivatives(
            curve.source_curve,
            lower,
            upper,
            first,
            last,
            order,
            preparations=preparations,
            budget=budget,
        )
    if isinstance(curve, AffinePCurve):
        bounds = np.stack(
            _trim_carrier_derivatives(
                curve.curve,
                lower,
                upper,
                first,
                last,
                order,
                preparations=preparations,
                budget=budget,
            )
        )
        lo, hi = type(curve)._transport_bounds(curve, bounds, translate=False)
        return lo, hi
    if isinstance(curve, IntersectionPCurve):
        head, tail = pcurve_chart_support(curve, first, last)
        carrier = type(curve)._carrier
        values = sorted((float(carrier(curve, lower)), float(carrier(curve, upper))))
        lo, hi = intersection_derivative_bounds(
            curve.curve,
            values[0],
            values[1],
            order,
            curve.side,
            minimum_chart=head,
            maximum_chart=tail,
            source_extension=True,
            preparations=preparations,
            budget=budget,
        )
        return (-hi, -lo) if curve.reversed and order == 1 else (lo, hi)
    # type(curve) keeps the receiver's class; ty loses the curve/trim union correlation.
    return type(curve).derivative_bounds(
        curve, lower, upper, order=order, preparations=preparations, budget=budget  # ty: ignore[invalid-argument-type]
    )


def _trim_carrier_enclosure(
    curve: CurveEvaluator,
    lower: float,
    upper: float,
    first: float,
    last: float,
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    if isinstance(curve, PeriodicPCurve):
        return type(curve)._transport_bounds(
            curve,
            _trim_carrier_enclosure(
                curve.source_curve,
                lower,
                upper,
                first,
                last,
                preparations=preparations,
                budget=budget,
            )
        )
    if isinstance(curve, AffinePCurve):
        return type(curve)._transport_bounds(
            curve,
            _trim_carrier_enclosure(
                curve.curve,
                lower,
                upper,
                first,
                last,
                preparations=preparations,
                budget=budget,
            ),
            translate=True,
        )
    if isinstance(curve, IntersectionPCurve):
        head, tail = pcurve_chart_support(curve, first, last)
        values = sorted((
            float(type(curve)._carrier(curve, lower)),
            float(type(curve)._carrier(curve, upper)),
        ))
        boxes = type(curve.curve).parameter_enclosures(
            curve.curve, *values,
            minimum_chart=head,
            maximum_chart=tail,
            source_extension=True,
            preparations=preparations,
            budget=budget,
        )
        columns = type(curve)._columns(curve)
        return np.stack(
            (np.min(boxes[:, 0, columns], axis=0), np.max(boxes[:, 1, columns], axis=0))
        )
    if isinstance(curve, AbstractTrimCurve):
        return type(curve).enclosure(
            curve, lower, upper, preparations=preparations, budget=budget
        )
    if lower < upper:
        return np.asarray(
            type(curve).bounding_box(
                curve, lower, upper, preparations=preparations, budget=budget
            ),
            dtype=np.float64,
        )

    images = []
    for piece in curve_pieces_for_interval(curve, lower, upper, coordinate_budget=budget):
        evaluator = piece.evaluator
        prepared = prepare_interval_function(
            lambda parameter: evaluator.evaluate(parameter[0]),
            1,
            batch_capacity=1,
            constant_bounds=coefficient_enclosures(evaluator),
        )
        lo, hi = prepared.evaluate(np.asarray([[lower]]), np.asarray([[upper]]))
        images.append(np.stack((lo[0], hi[0])))
    boxes = np.asarray(images)
    return np.stack((np.min(boxes[:, 0], axis=0), np.max(boxes[:, 1], axis=0)))


def _trim_carrier_enclosures(
    curve: CurveEvaluator,
    intervals: np.ndarray,
    first: float,
    last: float,
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Row-wise ``_trim_carrier_enclosure``; branch carriers share chart passes."""
    if intervals.shape[0] == 0:
        return np.empty((0, 2, 2), dtype=np.float64)
    if isinstance(curve, PeriodicPCurve):
        return np.stack(
            [
                curve._transport_bounds(box)
                for box in _trim_carrier_enclosures(
                    curve.source_curve,
                    intervals,
                    first,
                    last,
                    preparations=preparations,
                    budget=budget,
                )
            ]
        )
    if isinstance(curve, AffinePCurve):
        return np.stack(
            [
                curve._transport_bounds(box, translate=True)
                for box in _trim_carrier_enclosures(
                    curve.curve,
                    intervals,
                    first,
                    last,
                    preparations=preparations,
                    budget=budget,
                )
            ]
        )
    if isinstance(curve, IntersectionPCurve):
        head, tail = pcurve_chart_support(curve, first, last)
        requests = []
        for lower, upper in intervals.tolist():
            values = sorted(
                (float(curve._carrier(lower)), float(curve._carrier(upper)))
            )
            requests.append((values[0], values[1], head, tail, True))
        columns = curve._columns()
        return np.stack(
            [
                np.stack(
                    (
                        np.min(boxes[:, 0, columns], axis=0),
                        np.max(boxes[:, 1, columns], axis=0),
                    )
                )
                for boxes in curve.curve.parameter_enclosure_batch(
                    requests, preparations=preparations, budget=budget
                )
            ]
        )
    return np.stack(
        [
            np.asarray(
                _trim_carrier_enclosure(
                    curve,
                    float(lower),
                    float(upper),
                    first,
                    last,
                    preparations=preparations,
                    budget=budget,
                ),
                dtype=np.float64,
            )
            for lower, upper in intervals
        ]
    )


def prepare_trim_first_jets(
    cells: Sequence[tuple[AbstractTrimCurve, float, float]],
    /,
    *,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Prepare implicit trim derivatives without changing their scalar formulas.

    The ordinary derivative methods still own range validation, UV transport,
    scaling, orientation and the returned enclosures. This batches only the
    shared chart inclusions that those methods would otherwise compute one by
    one; non-intersection carriers keep their existing derivative implementation.
    """
    groups: dict[
        int,
        tuple[
            IntersectionJetPreparation,
            list[tuple[float, float, int, int | None, bool]],
        ],
    ] = {}
    for trim, lower, upper in cells:
        if not isinstance(trim, CurveTrimSegment):
            continue
        source: CurveEvaluator = trim
        first, last = trim.parameter_interval
        while True:
            if isinstance(source, CurveTrimSegment):
                bounds = source.carrier_parameter_enclosure(
                    lower, upper, preparations=preparations, budget=budget
                )
                lower, upper = float(bounds[0]), float(bounds[1])
                first, last = source.first, source.last
                source = source.curve
            elif isinstance(source, PeriodicPCurve):
                source = source.source_curve
            elif isinstance(source, AffinePCurve):
                source = source.curve
            else:
                break
        if not isinstance(source, IntersectionPCurve):
            continue
        preparation = _original_trim_intersection_preparation(
            source.curve, preparations=preparations, budget=budget
        )
        if preparation is None or preparation.chart_jets is None:
            continue
        head, tail = pcurve_chart_support(source, first, last)
        values = sorted(
            (float(source._carrier(lower)), float(source._carrier(upper)))
        )
        entry = groups.setdefault(id(preparation), (preparation, []))
        entry[1].append((values[0], values[1], head, tail, True))
    for preparation, requests in groups.values():
        curve = preparation.curve
        plans = _chart_enclosures(
            curve, requests, preparation, preparations=preparations, budget=budget
        )
        queries = []
        for request, boxes in zip(requests, plans, strict=True):
            first, _, minimum_chart, maximum_chart, _ = request
            last_chart = curve.num_charts - 1 if maximum_chart is None else maximum_chart
            head = max(minimum_chart, min(int(math.floor(first)), last_chart))
            queries.extend((head + offset, box) for offset, box in enumerate(boxes))
        _prepare_chart_first_jets(
            preparation, queries, preparations=preparations, budget=budget
        )


def trim_source_jet(
    curve: CurveEvaluator,
    lower: float,
    upper: float,
    order: int,
    /,
    *,
    first: float | None = None,
    last: float | None = None,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    """Jet of the source, extending only inside certified chart boxes.

    Range walls restrict the represented trim, not its defining equations.
    Regular endpoint roots need an open uniqueness box; this source query
    never changes the trim domain or admits an uncertified graph extension.
    """

    if isinstance(curve, CurveTrimSegment):
        if preparations is not None:
            intersection_owner.admit_segment_endpoint_roots(
                (curve.first_root, curve.last_root), preparations, budget=budget
            )
        a = (
            (curve.first, curve.first)
            if curve.first_root is None
            else curve.first_root.parameter_enclosure(
                preparations=preparations, budget=budget
            )
        )
        b = (
            (curve.last, curve.last)
            if curve.last_root is None
            else curve.last_root.parameter_enclosure(
                preparations=preparations, budget=budget
            )
        )
        lo, hi = curve.carrier_parameter_enclosure(
            lower, upper, source_extension=True, preparations=preparations, budget=budget
        )
        result = trim_source_jet(
            curve.curve,
            float(lo),
            float(hi),
            order,
            first=curve.first,
            last=curve.last,
            preparations=preparations,
            budget=budget,
        )
        if order:
            delta = (
                np.nextafter(b[0] - a[1], -np.inf),
                np.nextafter(b[1] - a[0], np.inf),
            )
            scale = delta if order == 1 else interval_multiply(delta, delta)
            result = interval_multiply(result, scale)
            if curve.reversed and order == 1:
                return (
                    -np.asarray(result[1], dtype=np.float64),
                    -np.asarray(result[0], dtype=np.float64),
                )
        return np.asarray(result[0], dtype=np.float64), np.asarray(
            result[1], dtype=np.float64
        )
    if isinstance(curve, PeriodicPCurve):
        result = trim_source_jet(
            curve.source_curve,
            lower,
            upper,
            order,
            first=first,
            last=last,
            preparations=preparations,
            budget=budget,
        )
        box = (
            curve._transport_bounds(np.stack(result)) if order == 0 else np.stack(result)
        )
        return box[0], box[1]
    if isinstance(curve, AffinePCurve):
        result = trim_source_jet(
            curve.curve,
            lower,
            upper,
            order,
            first=first,
            last=last,
            preparations=preparations,
            budget=budget,
        )
        box = curve._transport_bounds(np.stack(result), translate=order == 0)
        return box[0], box[1]
    if isinstance(curve, IntersectionPCurve):
        head, tail = pcurve_chart_support(
            curve,
            curve.first if first is None else first,
            curve.last if last is None else last,
        )
        values = sorted((float(curve._carrier(lower)), float(curve._carrier(upper))))
        if order:
            result = intersection_derivative_bounds(
                curve.curve,
                values[0],
                values[1],
                order,
                curve.side,
                minimum_chart=head,
                maximum_chart=tail,
                source_extension=True,
                preparations=preparations,
                budget=budget,
            )
            return (-result[1], -result[0]) if curve.reversed and order == 1 else result
        boxes = curve.curve.parameter_enclosures(
            *values,
            minimum_chart=head,
            maximum_chart=tail,
            source_extension=True,
            preparations=preparations,
            budget=budget,
        )
        columns = curve._columns()
        return np.min(boxes[:, 0, columns], axis=0), np.max(boxes[:, 1, columns], axis=0)
    if isinstance(curve, AbstractTrimCurve):
        domain = curve.parameter_interval
        if lower < domain[0] or upper > domain[1]:
            return np.full(2, -np.inf), np.full(2, np.inf)
        if order == 0:
            box = curve.enclosure(lower, upper, preparations=preparations, budget=budget)
            return box[0], box[1]
        return curve.derivative_bounds(
            lower, upper, order=order, preparations=preparations, budget=budget
        )
    derivative = curve.evaluate
    for _ in range(order):
        derivative = jax.jacfwd(derivative)

    prepared = prepare_interval_function(
        lambda parameter: derivative(parameter[0]),
        1,
        batch_capacity=1,
        constant_bounds=coefficient_enclosures(curve),
    )
    lo, hi = prepared.evaluate(np.asarray([[lower]]), np.asarray([[upper]]))
    return lo[0], hi[0]


_IDENTITY_UV = ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
_ZERO_UV = (Fraction(0), Fraction(0))


def _compose_uv(outer: _ExactUVMap, inner: _ExactUVMap, /) -> _ExactUVMap:
    a, b = outer
    c, d = inner
    return (
        (
            (
                a[0][0] * c[0][0] + a[0][1] * c[1][0],
                a[0][0] * c[0][1] + a[0][1] * c[1][1],
            ),
            (
                a[1][0] * c[0][0] + a[1][1] * c[1][0],
                a[1][0] * c[0][1] + a[1][1] * c[1][1],
            ),
        ),
        (b[0] + a[0][0] * d[0] + a[0][1] * d[1], b[1] + a[1][0] * d[0] + a[1][1] * d[1]),
    )


def _pcurve_affine_source(
    curve: CurveEvaluator | IntersectionCurve,
    /,
) -> tuple[CurveEvaluator | IntersectionCurve, _ExactUVMatrix, _ExactUVOffset]:
    if isinstance(curve, PeriodicPCurve):
        source, matrix, offset = _pcurve_affine_source(curve.source_curve)
        return (
            source,
            matrix,
            (
                offset[0] + period_offset(Fraction(0), Fraction(curve.period_shifts[0])),
                offset[1] + period_offset(Fraction(0), Fraction(curve.period_shifts[1])),
            ),
        )
    if isinstance(curve, AffinePCurve):
        source, matrix, offset = _pcurve_affine_source(curve.curve)
        matrix, offset = _compose_uv((curve.matrix, curve.offset), (matrix, offset))
        return source, matrix, offset
    return curve, _IDENTITY_UV, _ZERO_UV


def _same_pcurve_parameter(
    first: CurveEvaluator | IntersectionCurve,
    second: CurveEvaluator | IntersectionCurve,
    /,
) -> bool:
    if first is second:
        return True
    if isinstance(first, IntersectionPCurve) and isinstance(second, IntersectionPCurve):
        return (
            first.curve.branch_id == second.curve.branch_id
            and first.side == second.side
            and first.reversed == second.reversed
            and (
                not first.reversed
                or Fraction(first.first) + Fraction(first.last)
                == Fraction(second.first) + Fraction(second.last)
            )
        )
    if (
        type(first).__name__ in _CONSTRUCTION_FIELDS
        and type(second).__name__ in _CONSTRUCTION_FIELDS
    ):
        return encode_geometry(first) == encode_geometry(second)
    return False


def pcurve_affine_correspondence(
    first: CurveEvaluator, second: CurveEvaluator, /
) -> _ExactUVMap | None:
    """UV map from first to second iff source/parameter atoms agree.

    Consumers must compare this map (rational coefficients and symbolic
    native period offsets) to their independently proved coincident-surface
    chart map before concluding physical-source equality. This operation alone
    NEVER asserts that distinct charts are one surface.
    """
    source_a, matrix_a, offset_a = _pcurve_affine_source(first)
    source_b, matrix_b, offset_b = _pcurve_affine_source(second)
    if not _same_pcurve_parameter(source_a, source_b):
        return None
    a, b = matrix_a[0]
    c, d = matrix_a[1]
    determinant = a * d - b * c
    inverse = ((d / determinant, -b / determinant), (-c / determinant, a / determinant))
    inverse_offset = (
        -(inverse[0][0] * offset_a[0] + inverse[0][1] * offset_a[1]),
        -(inverse[1][0] * offset_a[0] + inverse[1][1] * offset_a[1]),
    )
    return _compose_uv((matrix_b, offset_b), (inverse, inverse_offset))


def pcurve_surface_period_shifts(
    first: CurveEvaluator,
    second: CurveEvaluator,
    expected_matrix: _ExactUVMatrix,
    expected_offset: tuple[Fraction, Fraction],
    patch: AbstractSurfacePatch,
    /,
) -> tuple[int, int] | None:
    """Compare a source map to a proved surface map modulo native deck shifts.

    ``second = expected_matrix @ first + expected_offset + shifts*periods``.
    The supplied surface map must already have an independent surface-equality
    proof. A nonzero rational discrepancy (including a float 2*pi seam), a
    fractional turn, or an unsupported destination period is UNPROVED.
    This is a 3D support correspondence, not equality of the two UV sheets.
    """
    if pcurve_periodic_source(second, patch) is None:
        return None
    correspondence = pcurve_affine_correspondence(first, second)
    if correspondence is None or correspondence[0] != expected_matrix:
        return None
    shifts = []
    for actual, expected, period in zip(
        correspondence[1],
        expected_offset,
        patch.native_period_symbols(),
        strict=True,
    ):
        delta = actual - expected
        match delta:
            case PeriodOffset():
                if delta.rational or delta.turns.denominator != 1 or period != "two_pi":
                    return None
                shifts.append(int(delta.turns))
            case _ if delta:
                return None
            case _:
                shifts.append(0)
    return shifts[0], shifts[1]


def _root_parameter_affine(endpoint: RootEndpoint, /) -> tuple[Fraction, Fraction]:

    scale = (
        Fraction(endpoint.affine_parameter_scale)
        if isinstance(endpoint, intersection_owner.TrimRootEndpoint)
        else Fraction(1)
    )
    offset = (
        Fraction(endpoint.affine_parameter_offset)
        if isinstance(endpoint, intersection_owner.TrimRootEndpoint)
        else Fraction(0)
    )
    for next_scale, next_offset in endpoint.affine_transforms:
        scale, offset = (
            Fraction(next_scale) * scale,
            Fraction(next_scale) * offset + Fraction(next_offset),
        )
    return scale, offset


def root_parameter_in_source(
    carrier: CurveEvaluator, endpoint: RootEndpoint, /
) -> tuple[Fraction, Fraction]:
    scale, offset = _root_parameter_affine(endpoint)
    source, _, _ = _pcurve_affine_source(carrier)
    if isinstance(source, IntersectionPCurve) and source.reversed:
        return -scale, Fraction(source.first) + Fraction(source.last) - offset
    return scale, offset


def _endpoint_source_transport(
    carrier: CurveEvaluator, endpoint: RootEndpoint, /
) -> _EndpointTransport | None:
    """Prove the parameter's source carrier and retain the UV operation."""
    source, matrix, offset = _pcurve_affine_source(carrier)
    root_source, root_matrix, root_offset = _pcurve_affine_source(endpoint.carrier)
    if isinstance(root_source, IntersectionCurve):
        if (
            isinstance(source, IntersectionPCurve)
            and source.curve.branch_id == root_source.branch_id
        ):
            return matrix, offset, ("branch", root_source.branch_id, source.side)

        if (
            not isinstance(endpoint, intersection_owner.BranchRootEndpoint)
            or endpoint.source_pcurve is None
        ):
            return None
        branch = root_source
        root_source, _, _ = _pcurve_affine_source(endpoint.source_pcurve)
        if not _same_pcurve_parameter(source, root_source):
            return None
        # This is an absolute source-sheet operation, not a map relative to
        # the endpoint's own gauge: equal 3D roots do not identify UV sheets.
        return (
            matrix,
            offset,
            (
                "branch-source",
                branch.branch_id,
                endpoint.source_side,
                canonical_fingerprint(encode_geometry(source)),
            ),
        )
    same_parameter = _same_pcurve_parameter(source, root_source)
    if (
        not same_parameter
        and isinstance(source, IntersectionPCurve)
        and isinstance(root_source, IntersectionPCurve)
    ):
        same_parameter = (
            source.curve.branch_id == root_source.curve.branch_id
            and source.side == root_source.side
            and not root_source.reversed
            and root_parameter_in_source(carrier, endpoint) == (Fraction(1), Fraction(0))
        )
    if not same_parameter:
        return None
    a, b = root_matrix[0]
    c, d = root_matrix[1]
    determinant = a * d - b * c
    inverse = ((d / determinant, -b / determinant), (-c / determinant, a / determinant))
    inverse_offset = (
        -(inverse[0][0] * root_offset[0] + inverse[0][1] * root_offset[1]),
        -(inverse[1][0] * root_offset[0] + inverse[1][1] * root_offset[1]),
    )
    matrix, offset = _compose_uv((matrix, offset), (inverse, inverse_offset))
    return matrix, offset, None


def _trim_endpoint_root(
    curve: CurveEvaluator,
    start: bool,
    /,
) -> tuple[RootEndpoint, CurveEvaluator, _ExactUVMap] | None:
    if isinstance(curve, PeriodicPCurve):
        value = _trim_endpoint_root(curve.source_curve, start)
        if value is None:
            return None
        endpoint, carrier, transform = value
        offset = (
            period_offset(Fraction(0), Fraction(curve.period_shifts[0])),
            period_offset(Fraction(0), Fraction(curve.period_shifts[1])),
        )
        return endpoint, carrier, _compose_uv((_IDENTITY_UV, offset), transform)
    if isinstance(curve, AffinePCurve):
        value = _trim_endpoint_root(curve.curve, start)
        if value is None:
            return None
        endpoint, carrier, transform = value
        return endpoint, carrier, _compose_uv((curve.matrix, curve.offset), transform)
    if isinstance(curve, CurveTrimSegment):
        endpoint = curve.first_root if start != curve.reversed else curve.last_root
        if endpoint is not None:
            return endpoint, curve.curve, (_IDENTITY_UV, _ZERO_UV)
    return None


def _trim_source_endpoint(
    curve: CurveEvaluator, start: bool, /
) -> tuple[CurveEvaluator, float] | None:
    if isinstance(curve, PeriodicPCurve):
        endpoint = _trim_source_endpoint(curve.source_curve, start)
        return (
            None
            if endpoint is None
            else (
                PeriodicPCurve(endpoint[0], curve.patch, curve.period_shifts),
                endpoint[1],
            )
        )
    if isinstance(curve, AffinePCurve):
        endpoint = _trim_source_endpoint(curve.curve, start)
        return (
            None
            if endpoint is None
            else (AffinePCurve(endpoint[0], curve.matrix, curve.offset), endpoint[1])
        )
    if isinstance(curve, IntersectionPCurve):
        return curve, curve.first if start else curve.last
    if isinstance(curve, CurveTrimSegment):
        parameter = curve.first if start != curve.reversed else curve.last
        return curve.curve, parameter
    if isinstance(curve, AbstractCurve) and curve.parameter_domain is not None:
        return curve, curve.parameter_domain[0 if start else 1]
    return None


def _analytic_curve_parameter_point(
    carrier: LineCurve | CircleCurve | EllipseCurve,
    parameter: Fraction | PeriodOffset,
) -> tuple[Fraction | PeriodOffset, ...] | None:
    match carrier:
        case LineCurve():
            return tuple(
                Fraction(float(origin)) + parameter * Fraction(float(direction))
                for origin, direction in zip(
                    np.asarray(carrier.origin), np.asarray(carrier.direction), strict=True
                )
            )
        case CircleCurve() | EllipseCurve():
            match parameter:
                case PeriodOffset():
                    quarter = parameter.turns * 4
                    if parameter.rational or quarter.denominator != 1:
                        return None
                    index = int(quarter)
                case Fraction() if parameter == 0:
                    index = 0
                case _:
                    # Binary64 multiples of float(pi) retain their authored real
                    # parameters; they do not prove mathematical quarter-turns.
                    return None
            cosine, sine = ((1, 0), (0, 1), (-1, 0), (0, -1))[index % 4]
            match carrier:
                case CircleCurve():
                    a, b = float(carrier.radius), float(carrier.radius)
                case EllipseCurve():
                    a, b = float(carrier.first_radius), float(carrier.second_radius)
            return tuple(
                Fraction(float(origin))
                + cosine * Fraction(a) * Fraction(float(x))
                + sine * Fraction(b) * Fraction(float(y))
                for origin, x, y in zip(
                    np.asarray(carrier.center),
                    np.asarray(carrier.first_axis),
                    np.asarray(carrier.second_axis),
                    strict=True,
                )
            )
        case _:
            return None


def _bspline_endpoint_image(
    carrier: BSplineCurve,
    parameter: Fraction,
    *,
    coordinate_budget: CoordinateEnclosureBudget,
) -> tuple[Fraction, ...] | None:
    reserve_polynomial(2, 2, 0, 2200, budget=coordinate_budget)
    coordinate_budget.reserve(carrier.knots.size, 128 + 32 * carrier.knots.size)
    knots = np.asarray(carrier.knots)
    lower, upper = float(knots[carrier.degree]), float(knots[-carrier.degree - 1])
    if parameter in (Fraction(lower), Fraction(upper)):
        # Multiplicity at least degree makes the active one-sided image
        # a source control. Its index follows the source knot run, not
        # the outer control-bank endpoints: exterior zero-width spans
        # can leave other controls outside the finite active domain.
        knot = lower if parameter == lower else upper
        matches = np.flatnonzero(knots == knot)
        if matches.size >= carrier.degree:
            coordinate_budget.reserve(
                carrier.control_points.size, 128 + 8 * carrier.control_points.size
            )
            reserve_polynomial(
                carrier.ambient_dimension,
                carrier.ambient_dimension,
                0,
                2200,
                budget=coordinate_budget,
            )
            index = (
                int(matches[-1]) - carrier.degree
                if parameter == lower
                else int(matches[0]) - 1
            )
            control = np.asarray(carrier.control_points)[index]
            return rational_vector(control)
    return None


@checked
def exact_curve_parameter_point(
    carrier: CurveEvaluator | IntersectionCurve,
    parameter: Fraction | PeriodOffset,
    /,
    *,
    coordinate_budget: CoordinateEnclosureBudget,
) -> tuple[Fraction | PeriodOffset, ...] | None:
    """Source image when the source expression has a provable finite form."""

    match carrier:
        case placed_owner.PlacedCurve():
            point = exact_curve_parameter_point(
                carrier.definition, parameter, coordinate_budget=coordinate_budget
            )
            if point is None:
                return None
            matrix, offset = np.asarray(carrier.rotation), np.asarray(carrier.translation)
            return tuple(
                Fraction(float(offset[i]))
                + sum(
                    (Fraction(float(matrix[i, j])) * point[j] for j in range(3)),
                    Fraction(0),
                )
                for i in range(3)
            )
        case PeriodicPCurve():
            point = exact_curve_parameter_point(
                carrier.source_curve, parameter, coordinate_budget=coordinate_budget
            )
            if point is None:
                return None
            return tuple(
                value + period_offset(Fraction(0), Fraction(shift))
                for value, shift in zip(point, carrier.period_shifts, strict=True)
            )
        case AffinePCurve():
            point = exact_curve_parameter_point(
                carrier.curve, parameter, coordinate_budget=coordinate_budget
            )
            if point is None:
                return None
            return tuple(
                carrier.offset[i]
                + sum((carrier.matrix[i][j] * point[j] for j in range(2)), Fraction(0))
                for i in range(2)
            )
        case LineCurve() | CircleCurve() | EllipseCurve():
            return _analytic_curve_parameter_point(carrier, parameter)
        case BSplineCurve() if isinstance(parameter, Fraction):
            return _bspline_endpoint_image(
                carrier, parameter, coordinate_budget=coordinate_budget
            )
        case _:
            return None


def _trim_exact_parameter_endpoint(
    curve: CurveEvaluator,
    start: bool,
    /,
) -> tuple[CurveEvaluator, Fraction | PeriodOffset] | None:

    if isinstance(curve, PeriodicPCurve):
        value = _trim_exact_parameter_endpoint(curve.source_curve, start)
        return (
            None
            if value is None
            else (PeriodicPCurve(value[0], curve.patch, curve.period_shifts), value[1])
        )
    if isinstance(curve, AffinePCurve):
        value = _trim_exact_parameter_endpoint(curve.curve, start)
        return (
            None
            if value is None
            else (AffinePCurve(value[0], curve.matrix, curve.offset), value[1])
        )
    if isinstance(curve, CurveTrimSegment):
        endpoint = curve.first_root if start != curve.reversed else curve.last_root
        if endpoint is not None:
            if (
                not isinstance(endpoint, intersection_owner.NativePeriodEndpoint)
                or _endpoint_source_transport(curve.curve, endpoint) is None
            ):
                return None
            return curve.curve, endpoint.exact_parameter
    endpoint = _trim_source_endpoint(curve, start)
    return None if endpoint is None else (endpoint[0], Fraction(endpoint[1]))


def _trim_exact_endpoint_image(
    curve: CurveEvaluator, start: bool, /, *, budget: CoordinateEnclosureBudget
) -> tuple[Fraction | PeriodOffset, ...] | None:
    endpoint = _trim_exact_parameter_endpoint(curve, start)
    return (
        None
        if endpoint is None
        else exact_curve_parameter_point(*endpoint, coordinate_budget=budget)
    )


def _period_endpoint_join(
    first: AbstractTrimCurve,
    second: AbstractTrimCurve,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    left = _trim_exact_parameter_endpoint(first, False)
    right = _trim_exact_parameter_endpoint(second, True)
    if left is not None and right is not None:
        source_a, matrix_a, offset_a = _pcurve_affine_source(left[0])
        source_b, matrix_b, offset_b = _pcurve_affine_source(right[0])
        if (matrix_a, offset_a) == (matrix_b, offset_b) and _same_pcurve_parameter(
            source_a, source_b
        ):
            difference = left[1] - right[1]
            if difference == 0:
                return True
            if (
                isinstance(source_a, AbstractCurve)
                and source_a.native_period_symbol() == "two_pi"
                and isinstance(difference, PeriodOffset)
                and not difference.rational
                and difference.turns.denominator == 1
            ):
                return True
    point_a = _trim_exact_endpoint_image(first, False, budget=budget)
    point_b = _trim_exact_endpoint_image(second, True, budget=budget)
    return point_a is not None and point_a == point_b


def _root_parameter_join(
    a: intersection_owner.TrimRootEndpoint | intersection_owner.BranchRootEndpoint,
    b: intersection_owner.TrimRootEndpoint | intersection_owner.BranchRootEndpoint,
    carrier_a: CurveEvaluator,
    carrier_b: CurveEvaluator,
) -> bool:
    match a, b:
        case (
            intersection_owner.BranchRootEndpoint(),
            intersection_owner.BranchRootEndpoint(),
        ):
            if (
                root_parameter_in_source(carrier_a, a)
                == root_parameter_in_source(carrier_b, b)
                == (Fraction(1), Fraction(0))
            ):
                return True
            return encode_geometry(a) == encode_geometry(b)
        case intersection_owner.TrimRootEndpoint(), intersection_owner.TrimRootEndpoint():
            expression_a = root_parameter_in_source(carrier_a, a)
            expression_b = root_parameter_in_source(carrier_b, b)
            if expression_a == expression_b == (Fraction(1), Fraction(0)):
                return True
            return (
                a.operand == b.operand
                and expression_a == expression_b
                and _same_pcurve_parameter(carrier_a, carrier_b)
            )
        case _:
            return False


def _root_endpoint_join(
    left_root: tuple[RootEndpoint, CurveEvaluator, _ExactUVMap] | None,
    right_root: tuple[RootEndpoint, CurveEvaluator, _ExactUVMap] | None,
) -> bool:
    # A failed root/source binding cannot fall through to floating endpoints.
    if left_root is None or right_root is None:
        return False
    a, carrier_a, transform_a = left_root
    b, carrier_b, transform_b = right_root
    if not isinstance(
        a,
        (intersection_owner.TrimRootEndpoint, intersection_owner.BranchRootEndpoint),
    ) or not isinstance(
        b,
        (intersection_owner.TrimRootEndpoint, intersection_owner.BranchRootEndpoint),
    ):
        return False
    same_point = a.root_id == b.root_id and (
        a.root is b.root or encode_geometry(a.root) == encode_geometry(b.root)
    )

    if isinstance(a.root, intersection_owner.IntersectionCurvePointRoot) and isinstance(
        b.root, intersection_owner.IntersectionCurvePointRoot
    ):
        same_point |= (
            a.root.point_id == b.root.point_id
            and a.root.curve.branch_id == b.root.curve.branch_id
        )
    if not same_point:
        return False
    map_a = _endpoint_source_transport(carrier_a, a)
    map_b = _endpoint_source_transport(carrier_b, b)
    if map_a is None or map_b is None:
        return False
    map_a = (*_compose_uv(transform_a, map_a[:2]), map_a[2])
    map_b = (*_compose_uv(transform_b, map_b[:2]), map_b[2])
    if map_a != map_b:
        return False

    return _root_parameter_join(a, b, carrier_a, carrier_b)


def _literal_endpoint_join(
    first: AbstractTrimCurve,
    second: AbstractTrimCurve,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    left, right = _trim_source_endpoint(first, False), _trim_source_endpoint(second, True)
    if left is None or right is None:
        return False
    carrier_a, parameter_a = left
    carrier_b, parameter_b = right
    source_a, matrix_a, offset_a = _pcurve_affine_source(carrier_a)
    source_b, matrix_b, offset_b = _pcurve_affine_source(carrier_b)
    if (matrix_a, offset_a) == (matrix_b, offset_b):
        if isinstance(source_a, IntersectionPCurve) and isinstance(
            source_b, IntersectionPCurve
        ):
            atom_a = (
                Fraction(source_a.first) + Fraction(source_a.last) - Fraction(parameter_a)
                if source_a.reversed
                else Fraction(parameter_a)
            )
            atom_b = (
                Fraction(source_b.first) + Fraction(source_b.last) - Fraction(parameter_b)
                if source_b.reversed
                else Fraction(parameter_b)
            )
            return (
                source_a.curve.branch_id == source_b.curve.branch_id
                and source_a.side == source_b.side
                and (
                    atom_a == atom_b
                    or (
                        source_a.curve.closed
                        and abs(atom_a - atom_b) == source_a.curve.num_charts
                    )
                )
            )
        if _same_pcurve_parameter(source_a, source_b):
            if parameter_a == parameter_b:
                return True

    point_a = exact_curve_parameter_point(
        carrier_a, Fraction(parameter_a), coordinate_budget=budget
    )
    point_b = exact_curve_parameter_point(
        carrier_b, Fraction(parameter_b), coordinate_budget=budget
    )
    return point_a is not None and point_a == point_b


@checked
def trim_source_join(
    first: AbstractTrimCurve,
    second: AbstractTrimCurve,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Prove endpoint identity from source atoms and their UV transport."""
    left_root = _trim_endpoint_root(first, False)
    right_root = _trim_endpoint_root(second, True)
    if (
        left_root is not None
        and isinstance(left_root[0], intersection_owner.NativePeriodEndpoint)
    ) or (
        right_root is not None
        and isinstance(right_root[0], intersection_owner.NativePeriodEndpoint)
    ):
        return _period_endpoint_join(first, second, budget=budget)
    if left_root is not None or right_root is not None:
        return _root_endpoint_join(left_root, right_root)
    return _literal_endpoint_join(first, second, budget=budget)


_MODULE_TYPES: Mapping[str, type[StrictModule]] = {
    "PlanePatch": PlanePatch,
    "CylinderPatch": CylinderPatch,
    "ConePatch": ConePatch,
    "SpherePatch": SpherePatch,
    "TorusPatch": TorusPatch,
    "OffsetSurface": OffsetSurface,
    "SurfaceIsoparametricCurve": SurfaceIsoparametricCurve,
    "BSplineSurfacePatch": BSplineSurfacePatch,
    "ExtrusionSurface": ExtrusionSurface,
    "RevolutionSurface": RevolutionSurface,
    "RuledSurface": RuledSurface,
    "LineCurve": LineCurve,
    "CircleCurve": CircleCurve,
    "EllipseCurve": EllipseCurve,
    "ParabolaCurve": ParabolaCurve,
    "HyperbolaCurve": HyperbolaCurve,
    "OffsetCurve": OffsetCurve,
    "BSplineCurve": BSplineCurve,
    "SurfaceRegion": SurfaceRegion,
    "IntersectionCurve": IntersectionCurve,
    "IntersectionPCurve": IntersectionPCurve,
    "AffinePCurve": AffinePCurve,
    "PeriodicPCurve": PeriodicPCurve,
    "CurveTrimSegment": CurveTrimSegment,
}


__all__ = [
    "AffinePCurve",
    "CurveRange",
    "CurveTrimSegment",
    "GeometryDefinition",
    "IntersectionCurve",
    "IntersectionCurvePoint",
    "IntersectionCurveSide",
    "IntersectionEndpointKind",
    "IntersectionPCurve",
    "PeriodicPCurve",
    "SurfacePairSystem",
    "SurfaceRegion",
    "curve_pieces",
    "decode_geometry",
    "encode_geometry",
    "exact_curve_parameter_point",
    "surface_pieces",
    "pcurve_affine_correspondence",
    "pcurve_periodic_source",
    "pcurve_surface_period_shifts",
]
