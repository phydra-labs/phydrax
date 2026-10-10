#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Authoritative surface/surface intersection curves and exact oriented trim curves.

Analytic and rational surface families are not closed under intersection: the
smooth intersection of two quadrics is in general a genus-one algebraic curve
with no finite rational representation. An `IntersectionCurve` is therefore
defined implicitly by its generating surface pair and one branch of their common
zero set. Its authority is a certified continuation atlas: each chart is a
coupled parameter box on both surfaces in which the branch is the graph of the
three remaining coordinates over one parameter axis, established by a parametric
Krawczyk inclusion. Points, jets and both p-curves are evaluated from the same
chart solution, and every evaluation reports its numerical parameter bound.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from fractions import Fraction
from inspect import signature
from types import NotImplementedType
from typing import Any, Generic, Literal, overload, TYPE_CHECKING, TypeAlias, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from ...nonlinear import VectorLocalRootPlan
from ...typing import (
    ConvertibleToArray,
    Dim,
    HostBool,
    HostFloat64,
    HostInt32,
    parse,
    Scope,
)
from .._atlas import AbstractTrimCurve
from ._patches import (
    _piece_hull,
    _prepare_bspline_curve_spans,
    AbstractCurve,
    AbstractSurfacePatch,
    BSplineCurve,
    BSplineSpanCurve,
    BSplineSpanSurface,
    BSplineSurfacePatch,
    CircleCurve,
    ConePatch,
    CylinderPatch,
    EllipseCurve,
    ExtrusionSurface,
    HyperbolaCurve,
    LineCurve,
    OffsetCurve,
    OffsetSurface,
    ParabolaCurve,
    PlanePatch,
    RationalBezierPiece,
    RevolutionSurface,
    RuledSurface,
    SpherePatch,
    SurfaceIsoparametricCurve,
    TorusPatch,
)


if TYPE_CHECKING:
    from ._intersection import RootEndpoint


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
        raise ValueError("parameter_box must be a finite (2, 2) array [lower, upper].")
    if np.any(box[1] <= box[0]):
        raise ValueError("parameter_box must be nonempty along both axes.")
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

    def __init__(
        self, patch: AbstractSurfacePatch, parameter_box: ConvertibleToArray
    ) -> None:
        if not isinstance(patch, AbstractSurfacePatch):
            raise TypeError("patch must be an AbstractSurfacePatch.")
        box = patch.validate_parameter_box(_host_box(parameter_box))
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


class CurveRange(StrictModule):
    """A generating curve restricted to one closed parameter interval."""

    curve: AbstractCurve | AbstractTrimCurve | IntersectionCurve
    first: float = eqx.field(static=True)
    last: float = eqx.field(static=True)
    periodic: bool = eqx.field(static=True)

    def __init__(
        self,
        curve: AbstractCurve | AbstractTrimCurve | IntersectionCurve,
        first: float | None = None,
        last: float | None = None,
    ) -> None:
        if not isinstance(curve, (AbstractCurve, AbstractTrimCurve, IntersectionCurve)):
            raise TypeError(
                "curve must provide a native curve, trim curve or coupled intersection capability."
            )
        domain = curve.parameter_domain
        if first is None or last is None:
            if domain is None:
                period = curve.period
                if period is None:
                    raise ValueError(
                        "An unbounded curve requires an explicit parameter range."
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


# ------------------------------------------------------- exact piece evaluators


def _bernstein(parameter: Array, degree: int, /) -> Array:
    binomial = jnp.asarray(
        [math.comb(degree, index) for index in range(degree + 1)], dtype=jnp.float64
    )
    return binomial * jnp.stack(
        [
            parameter**index * (1.0 - parameter) ** (degree - index)
            for index in range(degree + 1)
        ],
        axis=-1,
    )


class BernsteinSurfacePiece(AbstractSurfacePatch):
    """Rational Bernstein restriction carrying source construction enclosures."""

    controls: Array
    lower: Array
    upper: Array
    controls_lower: Array
    controls_upper: Array

    def __init__(
        self,
        controls: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        *,
        controls_lower: ArrayLike,
        controls_upper: ArrayLike,
    ) -> None:
        controls_ = jnp.asarray(controls, dtype=jnp.float64)
        if controls_.ndim != 3 or controls_.shape[-1] != 4:
            raise ValueError("Surface piece controls must have shape (du+1, dv+1, 4).")
        self.controls = controls_
        self.lower = jnp.asarray(lower, dtype=jnp.float64).reshape((2,))
        self.upper = jnp.asarray(upper, dtype=jnp.float64).reshape((2,))
        self.controls_lower = jnp.asarray(controls_lower, dtype=jnp.float64)
        self.controls_upper = jnp.asarray(controls_upper, dtype=jnp.float64)

    def bounding_box(self, parameter_box: ConvertibleToArray, /) -> np.ndarray:
        box = self.validate_parameter_box(parameter_box)
        piece = RationalBezierPiece(
            np.asarray(self.controls),
            tuple(zip(np.asarray(self.lower), np.asarray(self.upper), strict=True)),
            (0, 0),
            np.asarray(self.controls_lower),
            np.asarray(self.controls_upper),
        )
        return _piece_hull(piece, tuple(box[0]), tuple(box[1]))

    def evaluate(self, parameters: Array, /) -> Array:
        local = (parameters - self.lower) / (self.upper - self.lower)
        first = _bernstein(local[0], self.controls.shape[0] - 1)
        second = _bernstein(local[1], self.controls.shape[1] - 1)
        rows = first @ self.controls.reshape((self.controls.shape[0], -1))
        homogeneous = second @ rows.reshape((self.controls.shape[1], 4))
        return homogeneous[:3] / homogeneous[3]


class BernsteinCurvePiece(AbstractCurve):
    """Rational Bernstein restriction carrying source construction enclosures."""

    controls: Array
    lower: float = eqx.field(static=True)
    upper: float = eqx.field(static=True)
    controls_lower: Array
    controls_upper: Array

    def __init__(
        self,
        controls: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        *,
        controls_lower: ArrayLike,
        controls_upper: ArrayLike,
    ) -> None:
        controls_ = jnp.asarray(controls, dtype=jnp.float64)
        if controls_.ndim != 2 or controls_.shape[-1] not in (3, 4):
            raise ValueError("Curve piece controls must have shape (d+1, dim+1).")
        self.controls = controls_
        lower_, upper_ = float(np.asarray(lower)), float(np.asarray(upper))
        if not (math.isfinite(lower_) and math.isfinite(upper_) and lower_ < upper_):
            raise ValueError("Curve piece bounds must be finite and ordered.")
        self.lower, self.upper = lower_, upper_
        self.controls_lower = jnp.asarray(controls_lower, dtype=jnp.float64)
        self.controls_upper = jnp.asarray(controls_upper, dtype=jnp.float64)

    @property
    def ambient_dimension(self) -> int:
        return self.controls.shape[-1] - 1

    @property
    def parameter_domain(self) -> tuple[float, float]:
        return float(self.lower), float(self.upper)

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        first_, last_ = self.validate_range(first, last)
        piece = RationalBezierPiece(
            np.asarray(self.controls),
            (self.parameter_domain,),
            (0,),
            np.asarray(self.controls_lower),
            np.asarray(self.controls_upper),
        )
        return _piece_hull(piece, (first_,), (last_,))

    def evaluate(self, parameters: Array, /) -> Array:
        local = (parameters - self.lower) / (self.upper - self.lower)
        homogeneous = _bernstein(local, self.controls.shape[0] - 1) @ self.controls
        return homogeneous[..., :-1] / homogeneous[..., -1, None]


SurfaceEvaluator: TypeAlias = AbstractSurfacePatch | BernsteinSurfacePiece
CurveEvaluator: TypeAlias = AbstractCurve | AbstractTrimCurve
_CurvePieceT = TypeVar("_CurvePieceT", bound=StrictModule, covariant=True)


@dataclasses.dataclass(frozen=True, slots=True)
class SurfacePiece:
    """One interval-evaluable piece of a surface region."""

    evaluator: SurfaceEvaluator
    lower: np.ndarray
    upper: np.ndarray
    index: int


@dataclasses.dataclass(frozen=True, slots=True)
class CurvePiece(Generic[_CurvePieceT]):
    """One interval-evaluable piece of a curve range."""

    evaluator: _CurvePieceT
    lower: float
    upper: float
    index: int


def surface_pieces(region: SurfaceRegion, /) -> tuple[SurfacePiece, ...]:
    """Split a region into pieces whose evaluators are gather-free arithmetic.

    Rational spline surfaces use Bernstein span pieces with source coefficient
    enclosures; analytic and swept families evaluate directly.
    """
    return surface_pieces_for_box(region.patch, region.parameter_box)


def surface_pieces_for_box(
    patch: AbstractSurfacePatch,
    parameter_box: ConvertibleToArray,
    /,
) -> tuple[SurfacePiece, ...]:
    """Gather-free source pieces for a closed query box, including isolines."""
    from ._placed import PlacedSurface

    if isinstance(patch, PlacedSurface):
        return tuple(
            SurfacePiece(
                PlacedSurface(piece.evaluator, patch.rotation, patch.translation),
                piece.lower,
                piece.upper,
                piece.index,
            )
            for piece in surface_pieces_for_box(patch.definition, parameter_box)
        )
    box = np.asarray(patch.validate_parameter_box(parameter_box), dtype=np.float64)
    lower, upper = box
    degenerate = bool(np.any(lower == upper))
    if isinstance(patch, OffsetSurface):
        return tuple(
            SurfacePiece(
                OffsetSurface(piece.evaluator, patch.distance),
                piece.lower,
                piece.upper,
                piece.index,
            )
            for piece in surface_pieces_for_box(patch.base, box)
        )
    if isinstance(patch, (ExtrusionSurface, RevolutionSurface)):
        axis = 0 if isinstance(patch, ExtrusionSurface) else 1
        pieces = []
        for index, piece in enumerate(
            curve_pieces_for_interval(patch.curve, lower[axis], upper[axis])
        ):
            if isinstance(patch, ExtrusionSurface):
                evaluator = ExtrusionSurface(piece.evaluator, patch.direction)
            else:
                evaluator = RevolutionSurface(
                    piece.evaluator, patch.axis_origin, patch.axis_direction
                )
            piece_lower, piece_upper = lower.copy(), upper.copy()
            piece_lower[axis], piece_upper[axis] = piece.lower, piece.upper
            pieces.append(SurfacePiece(evaluator, piece_lower, piece_upper, index))
        return tuple(pieces)
    if isinstance(patch, RuledSurface):
        first = curve_pieces_for_interval(patch.first, lower[0], upper[0])
        second = curve_pieces_for_interval(patch.second, lower[0], upper[0])
        if lower[0] == upper[0]:
            return tuple(
                SurfacePiece(RuledSurface(a.evaluator, b.evaluator), lower, upper, index)
                for index, (a, b) in enumerate((a, b) for a in first for b in second)
            )
        breaks = sorted(
            {
                lower[0],
                upper[0],
                *(piece.lower for piece in (*first, *second)),
                *(piece.upper for piece in (*first, *second)),
            }
        )
        pieces = []
        for index, (start, end) in enumerate(zip(breaks[:-1], breaks[1:], strict=True)):
            midpoint = 0.5 * (start + end)
            a = next(piece for piece in first if piece.lower <= midpoint <= piece.upper)
            b = next(piece for piece in second if piece.lower <= midpoint <= piece.upper)
            evaluator = RuledSurface(a.evaluator, b.evaluator)
            piece_lower, piece_upper = lower.copy(), upper.copy()
            piece_lower[0], piece_upper[0] = start, end
            pieces.append(SurfacePiece(evaluator, piece_lower, piece_upper, index))
        return tuple(pieces)
    if not isinstance(patch, BSplineSurfacePatch):
        return (SurfacePiece(patch, lower, upper, 0),)
    pieces = []
    for index, piece in enumerate(patch.bezier_pieces()):
        bounds = np.asarray(piece.parameter_bounds, dtype=np.float64)
        piece_lower = np.maximum(bounds[:, 0], lower)
        piece_upper = np.minimum(bounds[:, 1], upper)
        if np.all(piece_upper >= piece_lower) and (
            degenerate or np.all(piece_upper > piece_lower)
        ):
            evaluator = BernsteinSurfacePiece(
                piece.homogeneous_controls,
                bounds[:, 0],
                bounds[:, 1],
                controls_lower=piece.homogeneous_lower,
                controls_upper=piece.homogeneous_upper,
            )
            pieces.append(SurfacePiece(evaluator, piece_lower, piece_upper, index))
    return tuple(pieces)


def curve_pieces(
    curve_range: CurveRange, /
) -> tuple[CurvePiece[CurveEvaluator | IntersectionCurve], ...]:
    """Split a curve range into gather-free arithmetic pieces."""
    return curve_pieces_for_interval(
        curve_range.curve, curve_range.first, curve_range.last
    )


@overload
def curve_pieces_for_interval(
    curve: AbstractCurve,
    first: float,
    last: float,
    /,
) -> tuple[CurvePiece[AbstractCurve], ...]: ...


@overload
def curve_pieces_for_interval(
    curve: AbstractTrimCurve,
    first: float,
    last: float,
    /,
) -> tuple[CurvePiece[CurveEvaluator], ...]: ...


@overload
def curve_pieces_for_interval(
    curve: IntersectionCurve,
    first: float,
    last: float,
    /,
) -> tuple[CurvePiece[IntersectionCurve], ...]: ...


@overload
def curve_pieces_for_interval(
    curve: CurveEvaluator,
    first: float,
    last: float,
    /,
) -> tuple[CurvePiece[CurveEvaluator], ...]: ...


@overload
def curve_pieces_for_interval(
    curve: CurveEvaluator | IntersectionCurve,
    first: float,
    last: float,
    /,
) -> tuple[CurvePiece[CurveEvaluator | IntersectionCurve], ...]: ...


def curve_pieces_for_interval(
    curve: CurveEvaluator | IntersectionCurve,
    first: float,
    last: float,
    /,
) -> tuple[CurvePiece[CurveEvaluator | IntersectionCurve], ...]:
    """Canonical source span restriction of a closed interval (points allowed)."""
    from ._placed import PlacedCurve

    if isinstance(curve, PlacedCurve):
        definition = curve.definition
        pieces: tuple[CurvePiece[AbstractCurve | IntersectionCurve], ...]
        if isinstance(definition, IntersectionCurve):
            pieces = curve_pieces_for_interval(definition, first, last)
        else:
            pieces = curve_pieces_for_interval(definition, first, last)
        return tuple(
            CurvePiece(
                PlacedCurve(piece.evaluator, curve.rotation, curve.translation),
                piece.lower,
                piece.upper,
                piece.index,
            )
            for piece in pieces
        )
    if isinstance(curve, IntersectionCurve):
        if not (
            math.isfinite(first)
            and math.isfinite(last)
            and 0 <= first <= last <= curve.num_charts
        ):
            raise ValueError(
                "A source branch query must lie inside its parameter domain."
            )
        return (CurvePiece(curve, first, last, 0),)
    if isinstance(curve, PeriodicPCurve):
        return tuple(
            CurvePiece(
                PeriodicPCurve(piece.evaluator, curve.patch, curve.period_shifts),
                piece.lower,
                piece.upper,
                piece.index,
            )
            for piece in curve_pieces_for_interval(curve.source_curve, first, last)
        )
    if isinstance(curve, AffinePCurve):
        return tuple(
            CurvePiece(
                AffinePCurve(piece.evaluator, curve.matrix, curve.offset),
                piece.lower,
                piece.upper,
                piece.index,
            )
            for piece in curve_pieces_for_interval(curve.curve, first, last)
        )
    if isinstance(curve, SurfaceIsoparametricCurve):
        return tuple(
            CurvePiece(
                SurfaceIsoparametricCurve(
                    piece.evaluator,
                    curve.fixed_axis,
                    curve.fixed_value,
                    parameter_range=(first, last) if first < last else None,
                ),
                float(piece.lower[1 - curve.fixed_axis]),
                float(piece.upper[1 - curve.fixed_axis]),
                piece.index,
            )
            for piece in surface_pieces_for_box(
                curve.surface, curve.parameter_box(first, last)
            )
        )
    if isinstance(curve, OffsetCurve):
        return tuple(
            CurvePiece(
                OffsetCurve(piece.evaluator, curve.distance, curve.direction),
                piece.lower,
                piece.upper,
                piece.index,
            )
            for piece in curve_pieces_for_interval(curve.base, first, last)
        )
    if not (math.isfinite(first) and math.isfinite(last) and first <= last):
        raise ValueError("Curve query intervals must be finite and ordered.")
    if first < last:
        curve.validate_range(first, last)
    else:
        domain = curve.parameter_domain
        if domain is not None and not domain[0] <= first <= domain[1]:
            raise ValueError("A point query lies outside its source curve domain.")
    if not isinstance(curve, BSplineCurve):
        return (CurvePiece(curve, first, last, 0),)
    from ...discretization._coordinate_enclosure import _COORDINATE_BUDGET

    budget = _COORDINATE_BUDGET.get()
    source_pieces = (
        curve.bezier_pieces()
        if budget is None
        else _prepare_bspline_curve_spans(curve, budget)
    )
    pieces = []
    for index, piece in enumerate(source_pieces):
        ((lower, upper),) = piece.parameter_bounds
        piece_lower = max(lower, first)
        piece_upper = min(upper, last)
        if piece_upper > piece_lower or (first == last and piece_upper == piece_lower):
            if budget is not None:
                terms = piece.homogeneous_controls.size
                budget.reserve(3 * terms + 2, 1024 + 8 * (3 * terms + 2))
            evaluator = BernsteinCurvePiece(
                piece.homogeneous_controls,
                lower,
                upper,
                controls_lower=piece.homogeneous_lower,
                controls_upper=piece.homogeneous_upper,
            )
            pieces.append(CurvePiece(evaluator, piece_lower, piece_upper, index))
    return tuple(pieces)


def coefficient_enclosures(
    system: StrictModule, /
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    """Source coefficient intervals to propagate through value and AD jaxprs."""
    types = (BernsteinCurvePiece, BernsteinSurfacePiece, AffinePCurve, PeriodicPCurve)
    pieces = jax.tree_util.tree_leaves(
        system, is_leaf=lambda item: isinstance(item, types)
    )
    bounds = []
    for piece in pieces:
        if isinstance(piece, PeriodicPCurve):
            bounds.extend(coefficient_enclosures(piece.source_curve))
            lower, upper = piece.period_offset_bounds()
            bounds.append((np.asarray(piece.offset_value), lower, upper))
            continue
        if isinstance(piece, AffinePCurve):
            bounds.extend(coefficient_enclosures(piece.curve))
            bounds.extend(
                (
                    (
                        np.asarray(piece.matrix_value),
                        *_fraction_array_bounds(piece.matrix),
                    ),
                    (
                        np.asarray(piece.offset_value),
                        *_fraction_array_bounds(piece.offset),
                    ),
                )
            )
        elif isinstance(piece, (BernsteinCurvePiece, BernsteinSurfacePiece)):
            bounds.append(
                (
                    np.asarray(piece.controls),
                    np.asarray(piece.controls_lower),
                    np.asarray(piece.controls_upper),
                )
            )
    # The interval interpreter associates jaxpr constants by nominal value.
    # Distinct exact rationals can round to the same constant: enclose ALL
    # matching definitions rather than selecting the first source's interval.
    merged = {}
    for nominal, lower, upper in bounds:
        key = (nominal.dtype.str, nominal.shape, nominal.tobytes())
        if key in merged:
            old_nominal, old_lower, old_upper = merged[key]
            merged[key] = (
                old_nominal,
                np.minimum(old_lower, lower),
                np.maximum(old_upper, upper),
            )
        else:
            merged[key] = nominal, lower, upper
    return tuple(merged.values())


# --------------------------------------------------------- defining equations


class SurfacePairSystem(StrictModule):
    """``S1(u1, v1) - S2(u2, v2)`` over the coupled parameters ``(u1, v1, u2, v2)``."""

    first: SurfaceEvaluator
    second: SurfaceEvaluator

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
        if isinstance(key, jax.tree_util.GetAttrKey):
            address.append(("attr", key.name))
        elif isinstance(key, jax.tree_util.SequenceKey):
            address.append(("index", key.idx))
        else:
            raise ValueError(
                "Source pieces must be addressed by attributes or positions."
            )
    return tuple(address)


def _node_at(tree: Any, address: _PinPath, /) -> Any:
    node = tree
    for kind, value in address:
        if kind == "attr":
            if not isinstance(value, str):
                raise TypeError("A source attribute address must be a string.")
            node = getattr(node, value)
        elif kind == "index":
            if not isinstance(value, int):
                raise TypeError("A source sequence address must be an integer.")
            node = node[value]
        else:
            raise ValueError(
                "A source address must name an attribute or sequence position."
            )
    return node


def _knot_span(knots: np.ndarray, lower: float, upper: float, /) -> int:
    span = int(np.searchsorted(knots, lower, side="right")) - 1
    if not (
        0 <= span < knots.size - 1 and knots[span] == lower and knots[span + 1] == upper
    ):
        raise ValueError("A Bernstein source piece must coincide with one knot span.")
    return span


def _span_pins(source: AbstractSurfacePatch, evaluator: SurfaceEvaluator, /) -> _SpanPins:
    """Host knot-span addresses of every Bernstein node in one source piece.

    Piece evaluators mirror their source structure (placement, offset, sweep
    and ruling wrappers keep their field names); each Bernstein node is the
    canonical extraction of the B-spline at the same address.
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
        try:
            target = _node_at(source, address)
        except (AttributeError, IndexError, TypeError) as error:
            raise ValueError(
                "A source piece does not mirror its source structure."
            ) from error
        lower = np.asarray(node.lower, dtype=np.float64).reshape(-1)
        upper = np.asarray(node.upper, dtype=np.float64).reshape(-1)
        if isinstance(node, BernsteinSurfacePiece) and isinstance(
            target, BSplineSurfacePatch
        ):
            vectors = tuple(
                np.asarray(knots, dtype=np.float64)
                for knots in (target.u_knots, target.v_knots)
            )
        elif isinstance(node, BernsteinCurvePiece) and isinstance(target, BSplineCurve):
            vectors = (np.asarray(target.knots, dtype=np.float64),)
        else:
            raise ValueError(
                "A Bernstein source piece must address a B-spline source node."
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


def _knot_vectors(node: Any, /) -> tuple[str, ...]:
    if isinstance(node, BSplineSurfacePatch):
        return ("u_knots", "v_knots")
    if isinstance(node, BSplineCurve):
        return ("knots",)
    raise ValueError("A pinned source address no longer names a B-spline source node.")


_TOPOLOGY_CHANGED = (
    "The live knot-span topology differs from the certified chart addresses; "
    "construct a new intersection for this source revision."
)


def _check_span_topology(source: AbstractSurfacePatch, pins: _SpanPins, /) -> None:
    """Host refusal when concrete live knots no longer match the pinned spans."""
    for address, _, breaks in pins:
        node = _node_at(source, address)
        for name, expected in zip(_knot_vectors(node), breaks, strict=True):
            knots = np.asarray(getattr(node, name), dtype=np.float64)
            if (
                knots.shape != (len(expected) + 1,)
                or tuple(bool(value) for value in np.diff(knots) > 0) != expected
            ):
                raise ValueError(_TOPOLOGY_CHANGED)


def _bind_span_pins(source: AbstractSurfacePatch, pins: _SpanPins, /) -> SurfaceEvaluator:
    """The live source with each pinned B-spline node restricted to its span.

    Spans index the current live knots. A concrete topology change is refused
    on the host; traced knots carry the same check as a runtime error, so a
    knot motion that merges or splits spans never reuses stale addresses.
    """
    bound: AbstractSurfacePatch = source
    for address, spans, breaks in pins:
        node = _node_at(bound, address)
        names = _knot_vectors(node)
        for name, expected in zip(names, breaks, strict=True):
            knots = getattr(node, name)
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
            node = eqx.tree_at(lambda item, name=name: getattr(item, name), node, checked)
        pinned = (
            BSplineSpanSurface(node, (spans[0], spans[1]))
            if isinstance(node, BSplineSurfacePatch)
            else BSplineSpanCurve(node, spans[0])
        )
        if not address:
            if not isinstance(pinned, AbstractSurfacePatch):
                raise ValueError("A pinned surface root must remain a surface patch.")
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


_MODULE_TYPES: dict[str, type[StrictModule]] = {
    cls.__name__: cls
    for cls in (
        PlanePatch,
        CylinderPatch,
        ConePatch,
        SpherePatch,
        TorusPatch,
        OffsetSurface,
        SurfaceIsoparametricCurve,
        BSplineSurfacePatch,
        ExtrusionSurface,
        RevolutionSurface,
        RuledSurface,
        LineCurve,
        CircleCurve,
        EllipseCurve,
        ParabolaCurve,
        HyperbolaCurve,
        OffsetCurve,
        BSplineCurve,
    )
}

_CONSTRUCTION_FIELDS = {
    name: tuple(
        parameter
        for parameter in signature(cls.__init__).parameters
        if parameter != "self" and not parameter.startswith("_")
    )
    for name, cls in _MODULE_TYPES.items()
}

_ROOT_GEOMETRY_NAMES = frozenset(
    {
        "TrimIntersectionRoot",
        "CurveSurfaceIntersectionRoot",
        "TripleSurfaceIntersectionRoot",
        "IntersectionCurvePointRoot",
        "TrimRootEndpoint",
        "BranchRootEndpoint",
        "NativePeriodEndpoint",
    }
)


def _registered_geometry_class(name: str, /) -> type[StrictModule] | None:
    if name in _MODULE_TYPES:
        return _MODULE_TYPES[name]
    if name in ("PlacedSurface", "PlacedCurve"):
        from . import _placed

        cls = getattr(_placed, name)
    elif name in ("BRepPlacedVertex", "BRepVertexRoot", "BRepRootSupport"):
        from . import _root_bindings

        cls = getattr(_root_bindings, name)
    elif name in _ROOT_GEOMETRY_NAMES:
        # Root constructors import this owner. Resolve schemas only after a real
        # construction record reaches the codec and the module cycle has closed.
        from . import _intersection

        cls = getattr(_intersection, name)
    else:
        return None
    _MODULE_TYPES[name] = cls
    _CONSTRUCTION_FIELDS[name] = tuple(
        field
        for field in signature(cls.__init__).parameters
        if field != "self" and not field.startswith("_")
    )
    return cls


def encode_geometry(value: Any, /) -> Any:
    """Lossless canonical payload of an exact curve/surface definition."""
    if value is None:
        return None
    if isinstance(value, Fraction):
        return {"fraction": [value.numerator, value.denominator]}
    if isinstance(value, tuple):
        return {"tuple": [encode_geometry(item) for item in value]}
    if isinstance(value, StrictModule):
        name = type(value).__name__
        if _registered_geometry_class(name) is not type(value):
            raise TypeError(f"{name} has no native intersection-curve encoding.")
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
                "Exact geometry arrays require finite float64, int32 or bool values."
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
    raise TypeError(f"Unsupported geometry payload value {type(value).__name__}.")


def decode_geometry(payload: Any, /) -> Any:
    """Restore only complete registered construction records and typed arrays."""
    if payload is None:
        return None
    if isinstance(payload, Mapping) and set(payload) == {"fraction"}:
        values = payload["fraction"]
        if (
            not isinstance(values, list)
            or len(values) != 2
            or any(type(value) is not int for value in values)
            or values[1] <= 0
        ):
            raise ValueError(
                "Exact rational geometry fields require an integer numerator and positive denominator."
            )
        restored = Fraction(*values)
        if [restored.numerator, restored.denominator] != values:
            raise ValueError(
                "Exact rational geometry fields must be in reduced canonical form."
            )
        return restored
    if isinstance(payload, Mapping) and set(payload) == {"tuple"}:
        if not isinstance(payload["tuple"], list):
            raise ValueError("A tuple geometry field requires an item list.")
        return tuple(decode_geometry(item) for item in payload["tuple"])
    if isinstance(payload, Mapping):
        if "type" in payload:
            if set(payload) != {"type", "fields"} or not isinstance(payload["type"], str):
                raise ValueError("A geometry record requires only type and fields.")
            cls = _registered_geometry_class(payload["type"])
            if cls is None:
                raise ValueError(f"Unknown geometry type {payload['type']!r}.")
            fields_payload = payload["fields"]
            expected = _CONSTRUCTION_FIELDS[payload["type"]]
            if not isinstance(fields_payload, Mapping) or set(fields_payload) != set(
                expected
            ):
                raise ValueError("Geometry construction fields are missing or unknown.")
            fields = {name: decode_geometry(fields_payload[name]) for name in expected}
            parameters = signature(cls.__init__).parameters
            positional = [
                fields[name]
                for name in expected
                if parameters[name].kind == parameters[name].POSITIONAL_ONLY
            ]
            keywords = {
                name: value
                for name, value in fields.items()
                if parameters[name].kind != parameters[name].POSITIONAL_ONLY
            }
            restored = cls(*positional, **keywords)
            for name, value in fields.items():
                if isinstance(value, np.ndarray):
                    target = np.asarray(getattr(restored, name))
                    if value.shape != target.shape or value.dtype != target.dtype:
                        raise ValueError(
                            f"Geometry field {name!r} has a noncanonical dtype or shape."
                        )
            if canonical_fingerprint(encode_geometry(restored)) != canonical_fingerprint(
                payload
            ):
                raise ValueError(
                    "Geometry construction fields are not losslessly canonical."
                )
            return restored
        if set(payload) != {"dtype", "shape", "hex"} or payload["dtype"] not in (
            "float64",
            "int32",
            "bool",
        ):
            raise ValueError(
                "A geometry array requires a registered dtype, shape and values."
            )
        shape = payload["shape"]
        encoded = payload["hex"]
        if (
            not isinstance(shape, list)
            or any(type(size) is not int or size < 0 for size in shape)
            or not isinstance(encoded, list)
            or len(encoded) != math.prod(shape)
        ):
            raise ValueError("Geometry array shape and value count are inconsistent.")
        dtype = payload["dtype"]
        if dtype == "float64":
            if any(not isinstance(item, str) for item in encoded):
                raise ValueError("Float64 geometry arrays require hexadecimal strings.")
            values = np.asarray(
                [float.fromhex(item) for item in encoded], dtype=np.float64
            )
        else:
            expected_type = bool if dtype == "bool" else int
            if any(type(item) is not expected_type for item in encoded):
                raise ValueError(
                    "Integer and boolean geometry array values have the wrong type."
                )
            if dtype == "int32" and any(not -(2**31) <= item < 2**31 for item in encoded):
                raise ValueError("Geometry integer values lie outside int32.")
            values = np.asarray(encoded, dtype=dtype)
        if not np.all(np.isfinite(values)):
            raise ValueError("Geometry array values must be finite.")
        return values.reshape(tuple(shape))
    if isinstance(payload, (bool, int, str)):
        return payload
    if isinstance(payload, float) and math.isfinite(payload):
        return payload
    raise ValueError("Unrecognized exact geometry payload.")


# ------------------------------------------------------------ intersection curve


class IntersectionCurvePoint(StrictModule):
    """Coupled evaluation of an intersection curve from one chart solution."""

    point: Array
    first_parameters: Array
    second_parameters: Array
    tangent: Array
    parameter_bound: Array
    gap: Array
    chart: Array


class IntersectionCurve(StrictModule):
    """One branch of the intersection of two exact surfaces.

    The authoritative definition is the generating pair, the branch identity and
    the certified continuation atlas. Chart ``k`` covers the curve parameter
    ``tau`` in ``[k, k + 1]``. Its graph coordinate interpolates the exact shared
    source nodes; ``chart_start`` and ``chart_end`` are bounded numerical
    representatives, not exact constructions. Node constraints fix one source
    coordinate of the generating pair and certify the other three; node
    references retain exact periodic transport. ``fully_certified`` requires
    both chart and shared-node witnesses. A closed final node references the
    first source root, rather than welding two near numerical points.
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
    ) -> None:
        if not isinstance(first, SurfaceRegion) or not isinstance(second, SurfaceRegion):
            raise TypeError("Intersection curves require two SurfaceRegion values.")
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
            raise ValueError("An intersection curve requires at least one chart.")
        if np.any((axes < 0) | (axes > 3)):
            raise ValueError("chart_axes entries must lie in [0, 3].")
        start, end = arrays["chart_start"], arrays["chart_end"]
        rows = np.arange(axes.shape[0])
        if np.any(start[rows, axes] == end[rows, axes]):
            raise ValueError("Every chart must advance along its chart axis.")
        if np.any(arrays["box_lower"] > arrays["box_upper"]):
            raise ValueError("Chart boxes require lower <= upper.")
        for name, value in arrays.items():
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite.")
        if not np.all(np.isfinite(preconditioners_)) or not np.all(
            np.isfinite(contraction_)
        ):
            raise ValueError(
                "Chart preconditioners and contraction bounds must be finite."
            )
        if np.any(contraction_ < 0.0) or np.any(certified_ & (contraction_ >= 1.0)):
            raise ValueError("Certified charts require a contraction bound in [0, 1).")
        for node in (start, end):
            if np.any(node < arrays["box_lower"]) or np.any(node > arrays["box_upper"]):
                raise ValueError("Chart nodes must lie in their continuation boxes.")
        start_kind = parse(start_kind, IntersectionEndpointKind, "endpoint kind")
        end_kind = parse(end_kind, IntersectionEndpointKind, "endpoint kind")
        endpoint_kinds = (start_kind, end_kind)
        if bool(closed) != (start_kind == "closed" and end_kind == "closed"):
            raise ValueError("Closed curves use 'closed' endpoints on both ends.")
        self.first = first
        self.second = second
        self.chart_axes = axes
        self.chart_start = start
        self.chart_end = end
        self.chart_pieces = pieces
        self.box_lower = arrays["box_lower"]
        self.box_upper = arrays["box_upper"]
        self.preconditioners = preconditioners_
        self.contraction = contraction_
        self.certified = certified_
        self.transition_shifts = arrays["transition_shifts"]
        self.closed = bool(closed)
        self.start_kind = start_kind
        self.end_kind = end_kind
        count = axes.shape[0]
        # Immutable host chart addressing only: the sorted piece incidences,
        # each chart's pair slot and every addressed piece's knot-span pins
        # come from the concrete source records before any tracing. Numerical
        # evaluators are never cached; `_pairs` (host certification) and
        # `_evaluation_pairs` (live equations) bind the current source leaves.
        self._pair_addresses = tuple(sorted({(int(a), int(b)) for a, b in pieces}))
        first_workset = {piece.index: piece for piece in surface_pieces(first)}
        second_workset = {piece.index: piece for piece in surface_pieces(second)}
        if any(
            a not in first_workset or b not in second_workset
            for a, b in self._pair_addresses
        ):
            raise ValueError("Chart addresses must name admitted source surface pieces.")
        self._pair_pins = tuple(
            (
                _span_pins(first.patch, first_workset[a].evaluator),
                _span_pins(second.patch, second_workset[b].evaluator),
            )
            for a, b in self._pair_addresses
        )
        pair_slots = {pair: slot for slot, pair in enumerate(self._pair_addresses)}
        self._chart_selector = tuple(pair_slots[(int(a), int(b))] for a, b in pieces)
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
        self.node_axes = parse(
            np.asarray(default_axes if node_axes is None else node_axes, dtype=np.int32),
            HostInt32[NodeDim],
            "node_axes",
        )
        self.node_values = parse(
            np.asarray(
                default_values if node_values is None else node_values, dtype=np.float64
            ),
            HostFloat64[NodeDim],
            "node_values",
        )
        self.node_pieces = parse(
            np.asarray(
                default_pieces if node_pieces is None else node_pieces, dtype=np.int32
            ),
            HostInt32[NodeDim, Literal[2]],
            "node_pieces",
        )
        self.node_references = parse(
            np.asarray(
                references if node_references is None else node_references, dtype=np.int32
            ),
            HostInt32[NodeDim],
            "node_references",
        )
        self.node_period_shifts = parse(
            np.asarray(
                shifts if node_period_shifts is None else node_period_shifts,
                dtype=np.int32,
            ),
            HostInt32[NodeDim, Literal[4]],
            "node_period_shifts",
        )
        if (
            self.node_axes.shape != (count + 1,)
            or self.node_values.shape != (count + 1,)
            or self.node_pieces.shape != (count + 1, 2)
            or self.node_references.shape != (count + 1,)
            or self.node_period_shifts.shape != (count + 1, 4)
            or np.any((self.node_axes < 0) | (self.node_axes > 3))
            or not np.all(np.isfinite(self.node_values))
            or np.any(self.node_references < 0)
            or np.any(self.node_references > np.arange(count + 1))
        ):
            raise ValueError(
                "Intersection node constraint records are inconsistent with the atlas."
            )
        if closed and self.node_references[-1] != 0:
            raise ValueError("A closed atlas must reference its first exact source node.")
        self._certify_nodes()
        self._verify_chart_witnesses()
        self._qualified = bool(
            np.all(self.certified)
            and np.all(self.nodes_certified)
            and np.all(self.transitions_certified)
        )
        self.branch_id = canonical_fingerprint(
            {
                "kind": "intersection-curve",
                "first": encode_geometry(first.patch),
                "first_box": [list(first.lower), list(first.upper)],
                "second": encode_geometry(second.patch),
                "second_box": [list(second.lower), list(second.upper)],
                "atlas": array_tree_fingerprint(
                    (
                        axes,
                        start,
                        end,
                        pieces,
                        arrays["box_lower"],
                        arrays["box_upper"],
                        preconditioners_,
                        contraction_,
                        certified_,
                        arrays["transition_shifts"],
                        self.node_axes,
                        self.node_values,
                        self.node_pieces,
                        self.node_references,
                        self.node_period_shifts,
                    )
                ),
                "closed": bool(closed),
                "endpoints": list(endpoint_kinds),
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

    def validate_range(self, first: float, last: float, /) -> tuple[float, float]:
        first_, last_ = float(first), float(last)
        if not (
            math.isfinite(first_)
            and math.isfinite(last_)
            and 0 <= first_ < last_ <= self.num_charts
        ):
            raise ValueError(
                "An intersection range must lie inside its parameter domain."
            )
        return first_, last_

    def is_c1_on(
        self,
        first: float,
        last: float,
        /,
        *,
        endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None] | None = None,
    ) -> bool:
        """Only one smooth source chart proves a jet; C0 nodes do not prove C1."""
        first_, last_, extended = self._rooted_query_range(first, last, endpoint_roots)
        if extended:
            boxes = self.parameter_enclosures(first_, last_, source_extension=True)
            if not np.all(np.isfinite(boxes)):
                raise ValueError(
                    "The rooted range leaves its certified source chart support."
                )
            if any(
                not self.first.patch.is_c1_on(box[:, :2])
                or not self.second.patch.is_c1_on(box[:, 2:])
                for box in boxes
            ):
                return False
        return _intersection_is_c1(
            self,
            max(0.0, min(first_, self.num_charts)),
            max(0.0, min(last_, self.num_charts)),
        )

    def derivative_bounds(
        self,
        first: float,
        last: float,
        /,
        *,
        order: int = 1,
        endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interval implicit graph jets; rooted extensions retain source authority."""
        first_, last_, extended = self._rooted_query_range(first, last, endpoint_roots)
        result = _intersection_derivative_bounds(
            self, first_, last_, order, None, source_extension=extended
        )
        if extended and not np.all(
            np.isfinite(self.parameter_enclosures(first_, last_, source_extension=True))
        ):
            raise ValueError(
                "The rooted range leaves its certified source chart support."
            )
        return result

    def bounding_box(
        self,
        first: float,
        last: float,
        /,
        *,
        endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None] | None = None,
    ) -> np.ndarray:
        """Enclose a branch range; outside-domain uncertainty needs its source root."""
        first_, last_, extended = self._rooted_query_range(first, last, endpoint_roots)
        boxes = self.parameter_enclosures(first_, last_, source_extension=extended)
        if not np.all(np.isfinite(boxes)):
            raise ValueError(
                "The rooted range leaves its certified source chart support."
            )
        preparation = _original_trim_intersection_preparation(self)
        systems, selector = self._pairs() if preparation is None else preparation.pairs
        head = max(0, min(int(math.floor(first_)), self.num_charts - 1))
        images = []
        from .._interval_enclosure import prepare_interval_function

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
    ) -> tuple[float, float, bool]:
        """Authorize only numeric endpoint uncertainty on the ORIGINAL scalar source."""
        first_, last_ = float(first), float(last)
        if not (math.isfinite(first_) and math.isfinite(last_) and first_ <= last_):
            raise ValueError("A branch query requires a finite ordered parameter range.")
        domain_upper = float(self.num_charts) if domain_upper is None else domain_upper
        if not 0.0 <= domain_lower < domain_upper <= self.num_charts:
            raise ValueError(
                "Rooted chart ownership must remain inside the original branch domain."
            )
        if domain_lower <= first_ <= last_ <= domain_upper:
            return first_, last_, False
        from ._intersection import BranchRootEndpoint, TrimRootEndpoint

        if endpoints is None or len(endpoints) != 2:
            raise ValueError(
                "A branch query must lie inside its closed domain or retain endpoint roots."
            )
        for index, outside in (
            (0, first_ < domain_lower),
            (1, last_ > domain_upper),
        ):
            if not outside:
                continue
            endpoint = endpoints[index]
            if not isinstance(endpoint, (BranchRootEndpoint, TrimRootEndpoint)):
                raise ValueError(
                    "An extended branch endpoint requires its original source root."
                )
            if _root_parameter_affine(endpoint) != (Fraction(1), Fraction(0)):
                raise ValueError(
                    "An extended branch endpoint cannot change its original scalar atom."
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
                    "The extended endpoint does not bind the original source branch."
                )
            lower, upper = endpoint.parameter_enclosure(maximum_steps=0)
            interval_lower, interval_upper = (
                (first_, min(last_, domain_lower))
                if index == 0
                else (max(first_, domain_upper), last_)
            )
            if not lower <= interval_lower <= interval_upper <= upper:
                raise ValueError(
                    "The extended range leaves its source endpoint's scalar enclosure."
                )
        return first_, last_, True

    @property
    def fully_certified(self) -> bool:
        return self._qualified

    def _shift_bounds(self, shifts: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
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
                raise ValueError("A node period shift requires a declared source period.")
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

    def _verify_chart_witnesses(self) -> None:
        """Reprove every claimed chart over the actual source coefficient box."""
        from ._intersection import _krawczyk, _prepare

        systems, selector = self._pairs()
        for chart in range(self.num_charts):
            if not bool(self.certified[chart]):
                continue
            prepared = _prepare(systems[int(selector[chart])], 4, 1)
            result = _krawczyk(
                prepared,
                self.box_lower[chart][None],
                self.box_upper[chart][None],
                parameter_axis=int(self.chart_axes[chart]),
            )
            if not bool(result.certified[0]):
                raise ValueError(
                    f"Chart {chart} has no actual source graph inclusion certificate."
                )
            # Bounds evaluated later with this stored preconditioner must also
            # be covered by the revalidated witness, not an arbitrary payload.
            if (
                not np.array_equal(self.preconditioners[chart], result.preconditioner[0])
                or self.contraction[chart] < result.contraction[0]
            ):
                raise ValueError(
                    f"Chart {chart} has an inconsistent source preconditioner or contraction witness."
                )

    def _certify_nodes(self) -> None:
        """Use shared source root constraints, never equality of floating nodes."""
        from ._intersection import _krawczyk, _prepare

        first = {piece.index: piece for piece in surface_pieces(self.first)}
        second = {piece.index: piece for piece in surface_pieces(self.second)}
        count = self.num_charts
        lower, upper = np.empty((count + 1, 4)), np.empty((count + 1, 4))
        certified = np.zeros(count + 1, dtype=np.bool_)
        for node in range(count + 1):
            reference = int(self.node_references[node])
            if reference != node:
                delta_lower, delta_upper = self._shift_bounds(
                    self.node_period_shifts[node]
                )
                lower[node] = np.nextafter(lower[reference] + delta_lower, -np.inf)
                upper[node] = np.nextafter(upper[reference] + delta_upper, np.inf)
                certified[node] = certified[reference]
                continue
            chart = max(0, node - 1)
            a, b = (int(index) for index in self.node_pieces[node])
            if a not in first or b not in second:
                raise ValueError("A node references an absent source span.")
            prepared = _prepare(
                SurfacePairSystem(first[a].evaluator, second[b].evaluator), 4, 1
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
            result = _krawczyk(prepared, lo[None], hi[None], parameter_axis=axis)
            certified[node] = bool(result.certified[0])
            if certified[node]:
                free = [index for index in range(4) if index != axis]
                for _ in range(16):
                    result = _krawczyk(prepared, lo[None], hi[None], parameter_axis=axis)
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
        for chart in range(1, count):
            shifts = np.zeros(4, dtype=np.int32)
            for axis, period in enumerate(
                (*self.first.patch.periods, *self.second.patch.periods)
            ):
                if period is not None:
                    shifts[axis] = int(
                        round(self.transition_shifts[chart - 1, axis] / period)
                    )
            piece_transition = not np.array_equal(
                self.chart_pieces[chart - 1],
                self.chart_pieces[chart],
            )
            if piece_transition and not np.any(shifts):
                continue
            delta_lower, delta_upper = self._shift_bounds(shifts)
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
        _preparation: _IntersectionJetPreparation | None = None,
    ) -> np.ndarray:
        """Source-faithful coupled chart boxes for a closed parameter subrange.

        ``source_extension`` is an endpoint-root query of the defining graph,
        not a larger represented curve domain. It requires the extended chart
        coordinate to remain in its certified atlas box; otherwise the result
        is unbounded and cannot certify an event.
        """
        if not self.fully_certified:
            raise ValueError(
                "Branch enclosures require a fully certified continuation atlas."
            )
        if _preparation is None:
            _preparation = _original_trim_intersection_preparation(self)
        elif (
            _preparation.curve is not self
            and _preparation is not _original_trim_intersection_preparation(self)
        ):
            raise ValueError(
                "Prepared branch enclosures must retain their original source curve."
            )
        count = self.num_charts
        if not (math.isfinite(first) and math.isfinite(last) and first <= last):
            raise ValueError("An enclosure requires a finite ordered parameter interval.")
        if not source_extension and not 0 <= first <= last <= count:
            raise ValueError(
                "An enclosure must lie inside the curve's parameter interval."
            )
        from ._intersection import _krawczyk, _prepare

        last_chart = count - 1 if maximum_chart is None else maximum_chart
        if not 0 <= minimum_chart <= last_chart < count:
            raise ValueError("Chart ownership limits must lie inside the atlas.")
        systems, selector = self._pairs() if _preparation is None else _preparation.pairs
        head = max(minimum_chart, min(int(math.floor(first)), last_chart))
        tail = max(head, min(int(math.ceil(last)) - 1, last_chart))
        boxes = []
        for chart in range(head, tail + 1):
            lower, upper = self.box_lower[chart].copy(), self.box_upper[chart].copy()
            axis = int(self.chart_axes[chart])
            from .._interval_enclosure import interval_add, interval_multiply

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
                            self.chart_start_lower[chart, axis],
                            self.chart_start_upper[chart, axis],
                        )
                    )
                elif fraction == 1:
                    endpoints.append(
                        (
                            self.chart_end_lower[chart, axis],
                            self.chart_end_upper[chart, axis],
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
                                    self.chart_start_lower[chart, axis],
                                    self.chart_start_upper[chart, axis],
                                ),
                                complement,
                            ),
                            interval_multiply(
                                (
                                    self.chart_end_lower[chart, axis],
                                    self.chart_end_upper[chart, axis],
                                ),
                                (fraction, fraction),
                            ),
                        )
                    )
            if source_extension and (
                min(value[0] for value in endpoints) < lower[axis]
                or max(value[1] for value in endpoints) > upper[axis]
            ):
                unbounded = np.asarray([[np.full(4, -np.inf), np.full(4, np.inf)]])
                return unbounded
            lower[axis] = max(lower[axis], min(value[0] for value in endpoints))
            upper[axis] = min(upper[axis], max(value[1] for value in endpoints))
            # A chart box is a deterministic function of its chart and
            # fractions; identical chart queries share one contraction.
            memo = None if _preparation is None else _preparation.chart_boxes
            key = (chart, float(fractions[0]), float(fractions[1]))
            if memo is not None and key in memo:
                boxes.append(memo[key])
                continue
            prepared = (
                _prepare(systems[int(selector[chart])], 4, 1)
                if _preparation is None
                else _preparation.root(chart)
            )
            free = [index for index in range(4) if index != axis]
            for _ in range(24):
                image = _krawczyk(prepared, lower[None], upper[None], parameter_axis=axis)
                new_lower, new_upper = lower.copy(), upper.copy()
                new_lower[free] = np.maximum(lower[free], image.lower[0])
                new_upper[free] = np.minimum(upper[free], image.upper[0])
                if np.any(new_lower > new_upper):
                    raise ValueError("The intersection chart lost its branch enclosure.")
                old_width = float(np.max(upper[free] - lower[free]))
                new_width = float(np.max(new_upper[free] - new_lower[free]))
                lower, upper = new_lower, new_upper
                if new_width >= 0.95 * old_width:
                    break
            box = np.stack((lower, upper))
            if memo is not None:
                memo[key] = box
            boxes.append(box)
        return np.asarray(boxes)

    def _pairs(self) -> tuple[tuple[SurfacePairSystem, ...], np.ndarray]:
        """Host certification pieces bound to the immutable chart addressing.

        Interval programs need gather-free Bernstein pieces with source
        coefficient enclosures, extracted on the host from the current concrete
        ``first``/``second`` leaves at every use (never cached).
        """
        for first_pins, second_pins in self._pair_pins:
            _check_span_topology(self.first.patch, first_pins)
            _check_span_topology(self.second.patch, second_pins)
        first = {piece.index: piece.evaluator for piece in surface_pieces(self.first)}
        second = {piece.index: piece.evaluator for piece in surface_pieces(self.second)}
        try:
            systems = tuple(
                SurfacePairSystem(first[a], second[b]) for a, b in self._pair_addresses
            )
        except KeyError as error:
            raise ValueError(
                "The current source pieces no longer admit the chart addresses."
            ) from error
        return systems, np.asarray(self._chart_selector, dtype=np.int32)

    def _evaluation_pairs(self) -> tuple[tuple[SurfacePairSystem, ...], np.ndarray]:
        """Live source equations bound to immutable host chart/span addresses.

        Each system evaluates the live ``first``/``second`` source pytrees, with
        every Bernstein piece replaced by its pinned canonical knot span, so
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

    def evaluate(self, parameters: ArrayLike, /) -> IntersectionCurvePoint:
        """Evaluate points, both p-curves, the unit tangent and numerical bounds.

        ``parameter_bound`` is a conservative max-norm enclosure distance from
        the returned parameters to the branch point on the same chart line.
        It uses the chart box, not a floating residual as a proof of accuracy;
        ``gap`` is the numerical value of ``|S1 - S2|``.
        """
        return self._evaluate_in_chart_range(parameters, 0, self.num_charts - 1)

    def _evaluate_in_chart_range(
        self,
        parameters: ArrayLike,
        minimum_chart: int,
        maximum_chart: int,
        /,
        *,
        certify: bool = True,
    ) -> IntersectionCurvePoint:
        tau = jnp.asarray(parameters, dtype=jnp.float64)
        systems, selector = self._evaluation_pairs()
        atlas = _ChartAtlas(
            systems,
            jnp.asarray(selector),
            jnp.asarray(self.chart_axes),
            jnp.asarray(self.chart_start),
            jnp.asarray(self.chart_end),
            jnp.asarray(self.preconditioners),
            jnp.asarray(self.contraction),
            jnp.asarray(self.certified),
            jnp.asarray(self.box_lower),
            jnp.asarray(self.box_upper),
            minimum_chart,
            maximum_chart,
        )
        result = _evaluate_atlas(atlas, tau.reshape((-1,)))
        result = jax.tree_util.tree_map(
            lambda leaf: leaf.reshape(tau.shape + leaf.shape[1:]), result
        )
        if certify and not isinstance(tau, Tracer) and self.fully_certified:
            values = np.asarray(tau).reshape(-1)
            if (
                np.any(~np.isfinite(values))
                or np.any(values < 0)
                or np.any(values > self.num_charts)
            ):
                raise ValueError(
                    "Evaluation parameters must lie in the intersection curve domain."
                )
        # Host box certification needs concrete source leaves; a traced source
        # (or parameter) keeps the atlas-box bound computed on the device.
        if (
            certify
            and not isinstance(result.first_parameters, Tracer)
            and self.fully_certified
        ):
            values = np.asarray(tau).reshape(-1)
            coupled = np.concatenate(
                (
                    np.asarray(result.first_parameters).reshape((-1, 2)),
                    np.asarray(result.second_parameters).reshape((-1, 2)),
                ),
                axis=1,
            )
            bounds = []
            for value, point in zip(values, coupled, strict=True):
                boxes = self.parameter_enclosures(
                    float(value),
                    float(value),
                    minimum_chart=minimum_chart,
                    maximum_chart=maximum_chart,
                )
                distances = np.nextafter(np.abs(boxes - point), np.inf)
                bounds.append(np.nextafter(np.max(distances), np.inf))
            result = eqx.tree_at(
                lambda item: item.parameter_bound,
                result,
                jnp.asarray(bounds, dtype=jnp.float64).reshape(tau.shape),
            )
        return result

    def payload(self) -> dict[str, Any]:
        """Lossless canonical record: generating surfaces, boxes and the atlas."""
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
            raise ValueError("Payload is not an intersection-curve record.")
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
                "Intersection payload fields are incomplete, unknown or ill-typed."
            )
        for name in ("chart_axes", "node_axes", "node_references"):
            if not isinstance(payload[name], list) or any(
                type(value) is not int for value in payload[name]
            ):
                raise ValueError(f"{name} must be an integer list.")
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
                raise ValueError(f"{name} has an invalid integer matrix shape.")
        if not isinstance(payload["certified"], list) or any(
            type(value) is not bool for value in payload["certified"]
        ):
            raise ValueError("certified must be a boolean list.")
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
            raise ValueError("Restored intersection curve identity does not match.")
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


class _ChartAtlas(StrictModule):
    """Device view of one continuation atlas and its defining equations."""

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
    minimum_chart: int = eqx.field(static=True)
    maximum_chart: int = eqx.field(static=True)

    def residual(self, chart: Array, parameters: Array, /) -> Array:
        if len(self.systems) == 1:
            return self.systems[0].residual(parameters)
        return jax.lax.switch(
            self.selector[chart],
            tuple(_surface_residual_branch(index) for index in range(len(self.systems))),
            (parameters, self.systems),
        )


_CHART_ROOT = VectorLocalRootPlan(
    3, maximum_steps=40, tolerance=1.0e-13, plan_id="intersection-curve-chart"
)


@eqx.filter_jit
def _evaluate_atlas(atlas: _ChartAtlas, tau: Array) -> IntersectionCurvePoint:
    """Solve each chart's graph equations and derive the coupled jet and bounds."""
    count = atlas.axes.shape[0]
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

        rest = _CHART_ROOT.solve(graph_residual, _rest(axis, guess))
        coupled = _assemble(axis, parameter, rest)
        value_residual = atlas.residual(chart, coupled)
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
            jnp.linalg.norm(value_residual),
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
) -> bool:
    if not (
        math.isfinite(first)
        and math.isfinite(last)
        and 0 <= first <= last <= curve.num_charts
    ):
        raise ValueError(
            "Intersection continuity queries require a closed source subinterval."
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
        if not curve.first.patch.is_c1_on(box[:, :2]) or not curve.second.patch.is_c1_on(
            box[:, 2:]
        ):
            return False
    return True


# Inverse Jacobian bounds and coupled first jet of one exact chart box.
type _ChartFirstJet = tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


class _IntersectionJetPreparation:
    """Reusable canonical graph programs, optionally resource-observed.

    The preparation/wrapping hooks change only how the existing interval
    programs are retained and evaluated; they do not change source equations,
    inverse enclosures, root authority, chart ownership or derivative formulas.
    An unobserved preparation also retains the results of identical
    deterministic queries; an observed one charges every evaluation.
    """

    def __init__(
        self,
        curve: IntersectionCurve,
        *,
        prepare: Any = None,
        wrap: Any = None,
        observe_point_jet: Any = None,
    ) -> None:
        self.curve = curve
        self.pairs = curve._pairs()
        self._prepare = prepare
        self._wrap = wrap
        self._observe_point_jet = observe_point_jet
        self._programs: dict[tuple[int, str], Any] = {}
        self._roots: dict[int, Any] = {}
        memoize = prepare is None and wrap is None
        self.chart_boxes: dict[tuple[int, float, float], np.ndarray] | None = (
            {} if memoize else None
        )
        self.jets: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray]] | None = (
            {} if memoize else None
        )
        self.chart_jets: dict[tuple[int, bytes], _ChartFirstJet | None] | None = (
            {} if memoize else None
        )

    def program(self, chart: int, kind: str, function: Any, dimension: int) -> Any:
        from .._interval_enclosure import prepare_interval_function

        index = int(self.pairs[1][chart])
        key = (index, kind)
        if key not in self._programs:
            coefficients = coefficient_enclosures(self.pairs[0][index])
            self._programs[key] = (
                prepare_interval_function(
                    function, dimension, batch_capacity=1, constant_bounds=coefficients
                )
                if self._prepare is None
                else self._prepare(function, dimension, coefficients)
            )
        return self._programs[key]

    def root(self, chart: int) -> Any:
        from ._intersection import _prepare

        index = int(self.pairs[1][chart])
        if index not in self._roots:
            prepared = _prepare(self.pairs[0][index], 4, 1)
            self._roots[index] = (
                prepared
                if self._wrap is None
                else _ObservedIntersectionRoot(
                    prepared, self._wrap, self._observe_point_jet
                )
            )
        return self._roots[index]


# Per invocation: source objects (held alive, so ids stay unique) and the one
# preparation of each scientific branch identity (structure plus every leaf).
type _BranchPreparations = tuple[
    dict[int, tuple[IntersectionCurve, _IntersectionJetPreparation]],
    dict[tuple[str, str], _IntersectionJetPreparation],
]
_TRIM_INTERSECTION_PREPARATIONS: ContextVar[_BranchPreparations | None] = ContextVar(
    "original_trim_intersection_preparations", default=None
)


@contextmanager
def original_trim_intersection_preparation(
    curves: tuple[AbstractTrimCurve, ...],
    /,
) -> Iterator[None]:
    """Retain branch preparations for this invocation's lifetime.

    Every concrete IntersectionCurve queried inside the outermost scope shares
    one preparation with all copies of the same scientific branch: identical
    tree structure and canonical fingerprint of every numerical leaf. Sources
    stay alive until the scope ends; traced branches are never prepared.
    Nested scopes join the active invocation and register their trim roots.
    """
    from ._intersection import _original_trim_root_preparation_scope

    with _original_trim_root_preparation_scope(curves):
        if _TRIM_INTERSECTION_PREPARATIONS.get() is not None:
            yield
            return
        token = _TRIM_INTERSECTION_PREPARATIONS.set(({}, {}))
        try:
            yield
        finally:
            _TRIM_INTERSECTION_PREPARATIONS.reset(token)


def _original_trim_intersection_preparation(
    curve: IntersectionCurve,
    /,
) -> _IntersectionJetPreparation | None:
    active = _TRIM_INTERSECTION_PREPARATIONS.get()
    if active is None:
        return None
    objects, identities = active
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
        preparation = _IntersectionJetPreparation(curve)
        identities[identity] = preparation
    objects[id(curve)] = (curve, preparation)
    return preparation


class _ObservedIntersectionRoot:
    """Resource hooks around the canonical Krawczyk evaluator, not a new solver."""

    def __init__(self, prepared: Any, wrap: Any, observe_point_jet: Any) -> None:
        self._prepared = prepared
        self.value = wrap(prepared.value)
        self.jacobian = wrap(prepared.jacobian)
        self._observe_point_jet = observe_point_jet

    def point_jacobians(self, points: np.ndarray, /) -> np.ndarray:
        if self._observe_point_jet is not None:
            self._observe_point_jet()
        return self._prepared.point_jacobians(points)


def _intersection_derivative_bounds(
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
    _preparation: _IntersectionJetPreparation | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Enclose graph inverse/implicit derivatives, never sampled root jets."""
    if _preparation is None:
        _preparation = _original_trim_intersection_preparation(curve)
    elif _preparation.curve is not curve:
        raise ValueError("Prepared branch jets must retain their original source curve.")
    if order not in (1, 2) or isinstance(order, bool):
        raise ValueError("Intersection derivative bounds support orders one and two.")
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
        )
        if memo is not None:
            memo[key] = result
    return result[0].copy(), result[1].copy()


def _intersection_jet_bounds(
    curve: IntersectionCurve,
    first: float,
    last: float,
    order: int,
    side: IntersectionCurveSide | Literal["all"] | None,
    minimum_chart: int,
    maximum_chart: int | None,
    source_extension: bool,
    _preparation: _IntersectionJetPreparation | None,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    if order == 2 and not _intersection_is_c1(
        curve,
        max(0.0, min(first, curve.num_charts)) if source_extension else first,
        max(0.0, min(last, curve.num_charts)) if source_extension else last,
        minimum_chart=minimum_chart,
        maximum_chart=maximum_chart,
    ):
        dimension = 7 if side == "all" else (3 if side is None else 2)
        return np.full(dimension, -np.inf), np.full(dimension, np.inf)
    from .._interval_enclosure import (
        interval_add,
        interval_multiply,
        prepare_interval_function,
    )
    from ._intersection import _interval_times_interval, _prepare

    def multiply_matrix(
        a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        lower, upper = _interval_times_interval(
            a[0][None], a[1][None], b[0][None], b[1][None]
        )
        return lower[0], upper[0]

    def quadratic(
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
        return result

    systems, selector = curve._pairs() if _preparation is None else _preparation.pairs
    last_chart = curve.num_charts - 1 if maximum_chart is None else maximum_chart
    head = max(minimum_chart, min(int(math.floor(first)), last_chart))
    lower_bounds, upper_bounds = [], []
    for offset, box in enumerate(
        curve.parameter_enclosures(
            first,
            last,
            minimum_chart=minimum_chart,
            maximum_chart=last_chart,
            source_extension=source_extension,
            _preparation=_preparation,
        )
    ):
        chart = head + offset
        if not np.all(np.isfinite(box)):
            dimension = 7 if side == "all" else (3 if side is None else 2)
            return np.full(dimension, -np.inf), np.full(dimension, np.inf)
        system = systems[int(selector[chart])]
        prepared = (
            _prepare(system, 4, 1) if _preparation is None else _preparation.root(chart)
        )
        axis = int(curve.chart_axes[chart])
        free = [index for index in range(4) if index != axis]
        # The inverse and first jet are functions of the chart and its exact
        # box; every order and side of one range shares a single inclusion.
        memo = None if _preparation is None else _preparation.chart_jets
        key = (chart, box.tobytes())
        if memo is not None and key in memo:
            basis = memo[key]
        else:
            basis = _chart_first_jet(curve, prepared, chart, box)
            if memo is not None:
                memo[key] = basis
        if basis is None:
            dimension = 7 if side == "all" else (3 if side is None else 2)
            return np.full(dimension, -np.inf), np.full(dimension, np.inf)
        inverse_bounds, coupled = basis
        if order == 2:
            hessian = (
                prepare_interval_function(
                    jax.jacfwd(jax.jacfwd(system.residual)),
                    4,
                    batch_capacity=1,
                    constant_bounds=coefficient_enclosures(system),
                )
                if _preparation is None
                else _preparation.program(
                    chart, "residual-hessian", jax.jacfwd(jax.jacfwd(system.residual)), 4
                )
            )
            h_lo, h_hi = hessian.evaluate(box[0][None], box[1][None])
            curvature = quadratic((h_lo[0], h_hi[0]), coupled)
            second = multiply_matrix(
                inverse_bounds, (-curvature[1][:, None], -curvature[0][:, None])
            )
            second_lo, second_hi = np.zeros(4), np.zeros(4)
            second_lo[free], second_hi[free] = second[0][:, 0], second[1][:, 0]
            coupled_second = (second_lo, second_hi)
        if side is not None and side != "all":
            columns = slice(0, 2) if side == "first" else slice(2, 4)
            result = coupled if order == 1 else coupled_second
            lower_bounds.append(result[0][columns])
            upper_bounds.append(result[1][columns])
            continue
        spatial_jacobian = (
            prepare_interval_function(
                jax.jacfwd(system.first.evaluate),
                2,
                batch_capacity=1,
                constant_bounds=coefficient_enclosures(system),
            )
            if _preparation is None
            else _preparation.program(
                chart, "spatial-jacobian", jax.jacfwd(system.first.evaluate), 2
            )
        )
        s_lo, s_hi = spatial_jacobian.evaluate(box[None, 0, :2], box[None, 1, :2])
        physical = multiply_matrix(
            (s_lo[0], s_hi[0]),
            (
                (coupled if order == 1 else coupled_second)[0][:2, None],
                (coupled if order == 1 else coupled_second)[1][:2, None],
            ),
        )
        result = (physical[0][:, 0], physical[1][:, 0])
        if order == 2:
            spatial_hessian = (
                prepare_interval_function(
                    jax.jacfwd(jax.jacfwd(system.first.evaluate)),
                    2,
                    batch_capacity=1,
                    constant_bounds=coefficient_enclosures(system),
                )
                if _preparation is None
                else _preparation.program(
                    chart,
                    "spatial-hessian",
                    jax.jacfwd(jax.jacfwd(system.first.evaluate)),
                    2,
                )
            )
            sh_lo, sh_hi = spatial_hessian.evaluate(box[None, 0, :2], box[None, 1, :2])
            result = interval_add(
                result, quadratic((sh_lo[0], sh_hi[0]), (coupled[0][:2], coupled[1][:2]))
            )
        if side == "all":
            parameters = coupled if order == 1 else coupled_second
            result = (
                np.concatenate((result[0], parameters[0])),
                np.concatenate((result[1], parameters[1])),
            )
        lower_bounds.append(result[0])
        upper_bounds.append(result[1])
    return np.min(lower_bounds, axis=0), np.max(upper_bounds, axis=0)


def _chart_first_jet(
    curve: IntersectionCurve, prepared: Any, chart: int, box: np.ndarray, /
) -> _ChartFirstJet | None:
    """Inverse Jacobian bounds and coupled first jet over one exact chart box.

    ``None`` means the box has no contracting parametric inclusion, so no
    implicit derivative is enclosed there.
    """
    from .._interval_enclosure import interval_multiply
    from ._intersection import _interval_times_interval, _krawczyk

    axis = int(curve.chart_axes[chart])
    free = [index for index in range(4) if index != axis]
    inclusion = _krawczyk(prepared, box[0][None], box[1][None], parameter_axis=axis)
    contraction = float(inclusion.contraction[0])
    if not math.isfinite(contraction) or contraction >= 1.0:
        return None
    inverse = inclusion.preconditioner[0]
    inverse_norm = np.nextafter(
        np.max(np.sum(np.abs(inverse), axis=1)) * (1 + 16 * np.finfo(float).eps),
        np.inf,
    )
    radius = np.nextafter(
        contraction / (1 - contraction) * inverse_norm * (1 + 16 * np.finfo(float).eps),
        np.inf,
    )
    inverse_bounds = (
        np.nextafter(inverse - radius, -np.inf),
        np.nextafter(inverse + radius, np.inf),
    )
    jac_lo, jac_hi = prepared.jacobian.evaluate(box[0][None], box[1][None])
    rest_lo, rest_hi = _interval_times_interval(
        inverse_bounds[0][None],
        inverse_bounds[1][None],
        -jac_hi[0, :, axis, None][None],
        -jac_lo[0, :, axis, None][None],
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
    rest = interval_multiply((rest_lo[0][:, 0], rest_hi[0][:, 0]), advance_bounds)
    first_lo, first_hi = np.zeros(4), np.zeros(4)
    first_lo[axis], first_hi[axis] = advance_bounds
    first_lo[free], first_hi[free] = rest
    return inverse_bounds, (first_lo, first_hi)


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

    def __init__(
        self,
        curve: IntersectionCurve,
        side: IntersectionCurveSide,
        *,
        first: float | None = None,
        last: float | None = None,
        reversed: bool = False,
    ) -> None:
        if not isinstance(curve, IntersectionCurve):
            raise TypeError("curve must be an IntersectionCurve.")
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
                raise ValueError(f"Unknown p-curve side {self.side!r}.")

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
            raise ValueError("A p-curve range must lie inside its parameter interval.")
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
    ) -> np.ndarray:
        return self.enclosure(first, last, endpoint_roots=endpoint_roots)

    def _rooted_query_range(
        self,
        first: float,
        last: float,
        endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None] | None,
        /,
    ) -> tuple[list[float], bool]:
        first_, last_, extended = self.curve._rooted_query_range(
            first,
            last,
            endpoint_roots,
            domain_lower=self.first,
            domain_upper=self.last,
        )
        if extended and self.reversed:
            raise ValueError(
                "A reversed p-curve extension requires its original scalar source."
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
    ) -> bool:
        endpoints, extended = self._rooted_query_range(first, last, endpoint_roots)
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
                )
            )
        ):
            raise ValueError(
                "The rooted p-curve leaves its certified source chart support."
            )
        return _intersection_is_c1(
            self.curve,
            max(self.first, endpoints[0]),
            min(self.last, endpoints[1]),
            minimum_chart=min(int(math.floor(self.first)), self.curve.num_charts - 1),
            maximum_chart=min(int(math.ceil(self.last)) - 1, self.curve.num_charts - 1),
        )

    def derivative_bounds(
        self,
        first: float,
        last: float,
        /,
        *,
        order: int = 1,
        endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        endpoints, extended = self._rooted_query_range(first, last, endpoint_roots)
        lower, upper = _intersection_derivative_bounds(
            self.curve,
            endpoints[0],
            endpoints[1],
            order,
            self.side,
            minimum_chart=min(int(math.floor(self.first)), self.curve.num_charts - 1),
            maximum_chart=min(int(math.ceil(self.last)) - 1, self.curve.num_charts - 1),
            source_extension=extended,
        )
        return (-upper, -lower) if self.reversed and order == 1 else (lower, upper)

    def evaluate(self, parameters: Array, /) -> Array:
        first_chart = min(int(math.floor(self.first)), self.curve.num_charts - 1)
        last_chart = min(int(math.ceil(self.last)) - 1, self.curve.num_charts - 1)
        result = self.curve._evaluate_in_chart_range(
            self._carrier(parameters),
            first_chart,
            last_chart,
            certify=False,
        )
        match self.side:
            case "first":
                return result.first_parameters
            case "second":
                return result.second_parameters
            case _:
                raise ValueError(f"Unknown p-curve side {self.side!r}.")

    def shares_endpoint(self, other: AbstractTrimCurve, /) -> bool:
        return _trim_source_join(self, other)

    def enclosure(
        self,
        first: float,
        last: float,
        /,
        *,
        endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None] | None = None,
    ) -> np.ndarray:
        endpoints, extended = self._rooted_query_range(first, last, endpoint_roots)
        boxes = self.curve.parameter_enclosures(
            *endpoints,
            minimum_chart=min(int(math.floor(self.first)), self.curve.num_charts - 1),
            maximum_chart=min(int(math.ceil(self.last)) - 1, self.curve.num_charts - 1),
            source_extension=extended,
        )
        if not np.all(np.isfinite(boxes)):
            raise ValueError(
                "The rooted p-curve leaves its certified source chart support."
            )
        columns = self._columns()
        return np.stack(
            (
                np.min(boxes[:, 0, columns], axis=0),
                np.max(boxes[:, 1, columns], axis=0),
            )
        )


def _fraction_bounds(value: Fraction, /) -> tuple[np.ndarray, np.ndarray]:
    nominal = float(value)
    represented = Fraction(nominal)
    return (
        np.asarray(nominal if represented <= value else np.nextafter(nominal, -np.inf)),
        np.asarray(nominal if represented >= value else np.nextafter(nominal, np.inf)),
    )


def _fraction_array_bounds(values: Any, /) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(values, dtype=object)
    pairs = [_fraction_bounds(value) for value in source.reshape(-1)]
    return tuple(
        np.asarray([pair[index] for pair in pairs]).reshape(source.shape)
        for index in (0, 1)
    )


class AffinePCurve(AbstractCurve, AbstractTrimCurve):
    """Exact UV operation ``matrix @ curve(t) + offset`` on the ORIGINAL source.

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

    def __init__(
        self,
        curve: AbstractCurve | AbstractTrimCurve,
        matrix: ConvertibleToArray | _ExactUVMatrix,
        offset: ConvertibleToArray | tuple[Fraction, Fraction],
    ) -> None:
        if (
            not isinstance(curve, (AbstractCurve, AbstractTrimCurve))
            or curve.ambient_dimension != 2
        ):
            raise TypeError(
                "An affine p-curve requires a two-dimensional exact source curve."
            )
        matrix_ = np.asarray(matrix, dtype=object)
        offset_ = np.asarray(offset, dtype=object)
        if matrix_.shape != (2, 2) or offset_.shape != (2,):
            raise ValueError("Affine UV transport requires a 2x2 matrix and two-vector.")

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
                "Affine UV coefficients must be finite exact rationals or binary floats."
            )

        exact_matrix = (
            (rational(matrix_[0, 0]), rational(matrix_[0, 1])),
            (rational(matrix_[1, 0]), rational(matrix_[1, 1])),
        )
        exact_offset = (rational(offset_[0]), rational(offset_[1]))
        a, b = exact_matrix[0]
        c, d = exact_matrix[1]
        if a * d == b * c:
            raise ValueError("Affine UV transport requires an exactly invertible matrix.")
        nominal_matrix = np.asarray(exact_matrix, dtype=np.float64)
        nominal_offset = np.asarray(exact_offset, dtype=np.float64)
        if not np.all(np.isfinite(nominal_matrix)) or not np.all(
            np.isfinite(nominal_offset)
        ):
            raise ValueError(
                "Affine UV coefficients require finite float64 numerical representatives."
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

    def exact_affine_map(self) -> _ExactUVMap:
        """Composed exact UV map, including symbolic native period offsets."""
        _, matrix, offset = _pcurve_affine_source(self)
        return matrix, offset

    def validate_range(self, first: float, last: float, /) -> tuple[float, float]:
        return self.curve.validate_range(first, last)

    def validate_query_range(self, first: float, last: float, /) -> tuple[float, float]:
        if not (math.isfinite(first) and math.isfinite(last) and first <= last):
            raise ValueError(
                "Affine p-curve queries require a finite closed subinterval."
            )
        if first < last:
            return self.validate_range(first, last)
        domain = self.parameter_domain
        if domain is not None and not domain[0] <= first <= domain[1]:
            raise ValueError("Affine p-curve query lies outside its source domain.")
        return float(first), float(last)

    def evaluate(self, parameters: Array, /) -> Array:
        return self.curve.evaluate(parameters) @ self.matrix_value.T + self.offset_value

    def _transport_bounds(self, bounds: np.ndarray, /, *, translate: bool) -> np.ndarray:
        from .._interval_enclosure import interval_add, interval_multiply

        result = []
        for row, offset in zip(self.matrix, self.offset, strict=True):
            value = _fraction_bounds(offset if translate else Fraction(0))
            for coefficient, lower, upper in zip(row, bounds[0], bounds[1], strict=True):
                if coefficient:
                    value = interval_add(
                        value,
                        interval_multiply(_fraction_bounds(coefficient), (lower, upper)),
                    )
            result.append(value)
        return np.asarray(result).T

    def enclosure(self, first: float, last: float, /) -> np.ndarray:
        self.validate_query_range(first, last)
        if isinstance(self.curve, AbstractTrimCurve):
            bounds = self.curve.enclosure(first, last)
        elif first < last:
            bounds = self.curve.bounding_box(first, last)
        else:
            from .._interval_enclosure import prepare_interval_function

            images = []
            for piece in curve_pieces_for_interval(self.curve, first, last):
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
            bounds = np.stack((np.min(boxes[:, 0], axis=0), np.max(boxes[:, 1], axis=0)))
        return self._transport_bounds(np.asarray(bounds), translate=True)

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        return self.enclosure(first, last)

    def is_c1_on(self, first: float, last: float, /) -> bool:
        self.validate_query_range(first, last)
        capability = getattr(self.curve, "is_c1_on", None)
        return capability is not None and capability(first, last)

    def derivative_bounds(
        self,
        first: float,
        last: float,
        /,
        *,
        order: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.validate_query_range(first, last)
        if order not in (1, 2) or isinstance(order, bool):
            raise ValueError("Affine p-curve derivatives support orders one and two.")
        if order == 2 and not self.is_c1_on(first, last):
            return np.full(2, -np.inf), np.full(2, np.inf)
        bounds = np.stack(self.curve.derivative_bounds(first, last, order=order))
        lower, upper = self._transport_bounds(bounds, translate=False)
        return lower, upper

    def shares_endpoint(self, other: AbstractTrimCurve, /) -> bool:
        return _trim_source_join(self, other)


def exact_line_pcurve(
    origin: tuple[Fraction, Fraction], direction: tuple[float, float], /
) -> LineCurve | AffinePCurve:
    """Canonical UV line through an exact origin.

    A binary64-representable origin is the plain ``LineCurve``; otherwise the
    exact origin is retained as an identity ``AffinePCurve`` offset of the line
    through zero. Constructors and readers share this one representation, so
    the same exact p-curve has one geometry identity.
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


@dataclasses.dataclass(frozen=True)
class _PeriodOffset:
    """Exact scalar in Q + Q*(2*pi); never a rationalized float period."""

    rational: Fraction
    turns: Fraction

    @overload
    def __add__(
        self, other: Fraction | int | _PeriodOffset
    ) -> Fraction | _PeriodOffset: ...

    @overload
    def __add__(self, other: object) -> Fraction | _PeriodOffset | NotImplementedType: ...

    def __add__(self, other: object) -> Fraction | _PeriodOffset | NotImplementedType:
        if isinstance(other, _PeriodOffset):
            return _period_offset(
                self.rational + other.rational, self.turns + other.turns
            )
        if isinstance(other, (Fraction, int)):
            return _period_offset(self.rational + other, self.turns)
        return NotImplemented

    __radd__ = __add__

    def __neg__(self) -> Fraction | _PeriodOffset:
        return _period_offset(-self.rational, -self.turns)

    @overload
    def __sub__(
        self, other: Fraction | int | _PeriodOffset
    ) -> Fraction | _PeriodOffset: ...

    @overload
    def __sub__(self, other: object) -> Fraction | _PeriodOffset | NotImplementedType: ...

    def __sub__(self, other: object) -> Fraction | _PeriodOffset | NotImplementedType:
        if isinstance(other, (Fraction, int, _PeriodOffset)):
            return self + -other
        return NotImplemented

    @overload
    def __rsub__(
        self, other: Fraction | int | _PeriodOffset
    ) -> Fraction | _PeriodOffset: ...

    @overload
    def __rsub__(
        self, other: object
    ) -> Fraction | _PeriodOffset | NotImplementedType: ...

    def __rsub__(self, other: object) -> Fraction | _PeriodOffset | NotImplementedType:
        if isinstance(other, (Fraction, int, _PeriodOffset)):
            return -self + other
        return NotImplemented

    @overload
    def __mul__(self, other: Fraction | int) -> Fraction | _PeriodOffset: ...

    @overload
    def __mul__(self, other: object) -> Fraction | _PeriodOffset | NotImplementedType: ...

    def __mul__(self, other: object) -> Fraction | _PeriodOffset | NotImplementedType:
        if isinstance(other, (Fraction, int)):
            return _period_offset(self.rational * other, self.turns * other)
        return NotImplemented

    __rmul__ = __mul__


_ExactUVMatrix: TypeAlias = tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]
_ExactUVOffset: TypeAlias = tuple[Fraction | _PeriodOffset, Fraction | _PeriodOffset]
_ExactUVMap: TypeAlias = tuple[_ExactUVMatrix, _ExactUVOffset]
_EndpointTransportTag: TypeAlias = (
    tuple[Literal["branch"], str, IntersectionCurveSide]
    | tuple[Literal["branch-source"], str, IntersectionCurveSide | None, str]
    | None
)
_EndpointTransport: TypeAlias = tuple[
    _ExactUVMatrix, _ExactUVOffset, _EndpointTransportTag
]


def _period_offset(rational: Fraction, turns: Fraction, /) -> Fraction | _PeriodOffset:
    return (
        Fraction(rational)
        if not turns
        else _PeriodOffset(Fraction(rational), Fraction(turns))
    )


def _period_scalar_bounds(
    value: Fraction | _PeriodOffset, /
) -> tuple[np.ndarray, np.ndarray]:
    """Directed enclosure of the authored Q + Q*(2*pi) expression."""
    from .._interval_enclosure import interval_add, interval_multiply

    if isinstance(value, Fraction):
        return _fraction_bounds(value)
    period = (
        np.asarray(np.nextafter(math.tau, -np.inf)),
        np.asarray(np.nextafter(math.tau, np.inf)),
    )
    return interval_add(
        _fraction_bounds(value.rational),
        interval_multiply(_fraction_bounds(value.turns), period),
    )


def _affine_curve_coefficients(
    curve: AbstractCurve | None, point: np.ndarray, /
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]] | None:
    """Exact affine source coefficients, including canonical rational UV maps."""
    if curve is None:
        return (
            tuple(Fraction(float(value)) for value in point),
            (Fraction(0),) * point.size,
        )
    if isinstance(curve, AffinePCurve):
        if not isinstance(curve.curve, AbstractCurve):
            return None
        source = _affine_curve_coefficients(curve.curve, point)
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
            tuple(Fraction(float(value)) for value in np.asarray(curve.origin)),
            tuple(Fraction(float(value)) for value in np.asarray(curve.direction)),
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
    if (
        weights[0] <= 0.0
        or weights[0] != weights[1]
        or knots.shape != (4,)
        or knots[0] != knots[1]
        or knots[2] != knots[3]
    ):
        return None
    first = Fraction(float(knots[1]))
    span = Fraction(float(knots[2])) - first
    derivative = tuple(
        (Fraction(float(last)) - Fraction(float(start))) / span
        for start, last in zip(controls[0], controls[1], strict=True)
    )
    origin = tuple(
        Fraction(float(start)) - first * slope
        for start, slope in zip(controls[0], derivative, strict=True)
    )
    return origin, derivative


def _native_curve_period_symbol(
    curve: AbstractCurve | IntersectionCurve, /
) -> str | None:
    from ._placed import PlacedCurve

    if isinstance(curve, PlacedCurve):
        return _native_curve_period_symbol(curve.definition)
    if isinstance(curve, (AffinePCurve, PeriodicPCurve)):
        source = curve.curve if isinstance(curve, AffinePCurve) else curve.source_curve
        return (
            _native_curve_period_symbol(source)
            if isinstance(source, AbstractCurve)
            else None
        )
    if isinstance(curve, SurfaceIsoparametricCurve) and curve.fixed_axis in (0, 1):
        # The isoline runs along the surface's other parameter axis.
        return _native_period_symbols(curve.surface)[1 - curve.fixed_axis]
    return "two_pi" if type(curve) in (CircleCurve, EllipseCurve) else None


def _native_period_symbols(
    patch: AbstractSurfacePatch, /
) -> tuple[str | None, str | None]:
    """Proved native periods, not a claim inferred from a numerical declaration."""
    from ._placed import PlacedSurface

    if isinstance(patch, PlacedSurface):
        return _native_period_symbols(patch.definition)
    if type(patch) in (CylinderPatch, ConePatch, SpherePatch):
        return "two_pi", None
    if type(patch) is TorusPatch:
        return "two_pi", "two_pi"
    if type(patch) is OffsetSurface:
        return _native_period_symbols(patch.base)
    if type(patch) is ExtrusionSurface:
        return _native_curve_period_symbol(patch.curve), None
    if type(patch) is RevolutionSurface:
        return "two_pi", _native_curve_period_symbol(patch.curve)
    if type(patch) is RuledSurface:
        first, second = (
            _native_curve_period_symbol(patch.first),
            _native_curve_period_symbol(patch.second),
        )
        return (first if first == second else None), None
    return None, None


class PeriodicPCurve(AbstractCurve, AbstractTrimCurve):
    """Original UV source plus integer multiples of mathematical native periods.

    The patch definition and integer gauge are construction data. Neither source
    coefficients nor parameter/root atoms are replaced. Numerical coordinates
    are representatives only; enclosures include the exact irrational period.
    Distinct gauges are distinct UV sheets even when their 3D images coincide.
    """

    source_curve: AbstractCurve | AbstractTrimCurve
    patch: AbstractSurfacePatch
    period_shifts: tuple[int, int] = eqx.field(static=True)
    offset_value: Array

    def __init__(
        self,
        source_curve: AbstractCurve | AbstractTrimCurve,
        patch: AbstractSurfacePatch,
        period_shifts: tuple[int, int] = (0, 0),
    ) -> None:
        if (
            not isinstance(source_curve, (AbstractCurve, AbstractTrimCurve))
            or source_curve.ambient_dimension != 2
        ):
            raise TypeError(
                "A periodic p-curve requires a two-dimensional exact source curve."
            )
        if not isinstance(patch, AbstractSurfacePatch):
            raise TypeError("A periodic p-curve requires a native surface patch.")
        shifts = tuple(period_shifts)
        if len(shifts) != 2 or any(
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            for value in shifts
        ):
            raise ValueError("Periodic UV shifts must be two integers.")
        symbols = _native_period_symbols(patch)
        if any(
            shift and symbol is None
            for shift, symbol in zip(shifts, symbols, strict=True)
        ):
            raise ValueError(
                "A UV period shift requires a proved mathematical native period."
            )
        nominal = np.asarray(
            [int(shift) * (2 * math.pi) for shift in shifts], dtype=np.float64
        )
        if not np.all(np.isfinite(nominal)):
            raise ValueError(
                "Periodic UV shifts require finite numerical representatives."
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
        return _native_period_symbols(self.patch)

    @property
    def source_definition(self) -> dict[str, object]:
        return encode_geometry(self)

    @property
    def source_id(self) -> str:
        return canonical_fingerprint(self.source_definition)

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
        from .._interval_enclosure import interval_multiply

        period = (
            np.asarray(np.nextafter(2 * math.pi, -np.inf)),
            np.asarray(np.nextafter(2 * math.pi, np.inf)),
        )
        return np.asarray(
            [
                interval_multiply(_fraction_bounds(Fraction(shift)), period)
                if shift
                else (0.0, 0.0)
                for shift in self.period_shifts
            ]
        ).T

    def validate_range(self, first: float, last: float, /) -> tuple[float, float]:
        return self.source_curve.validate_range(first, last)

    def validate_query_range(self, first: float, last: float, /) -> tuple[float, float]:
        return AbstractCurve.validate_query_range(self, first, last)

    def evaluate(self, parameters: Array, /) -> Array:
        return self.source_curve.evaluate(parameters) + self.offset_value

    def _transport_bounds(self, bounds: np.ndarray, /) -> np.ndarray:
        from .._interval_enclosure import interval_add

        offsets = self.period_offset_bounds()
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

    def enclosure(self, first: float, last: float, /) -> np.ndarray:
        self.validate_query_range(first, last)
        return self._transport_bounds(
            _trim_carrier_enclosure(self.source_curve, first, last, first, last)
        )

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        return self.enclosure(first, last)

    def is_c1_on(self, first: float, last: float, /) -> bool:
        self.validate_query_range(first, last)
        capability = getattr(self.source_curve, "is_c1_on", None)
        return capability is not None and capability(first, last)

    def derivative_bounds(
        self,
        first: float,
        last: float,
        /,
        *,
        order: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.validate_query_range(first, last)
        if order not in (1, 2) or isinstance(order, bool):
            raise ValueError("Periodic p-curve derivatives support orders one and two.")
        if order == 2 and not self.is_c1_on(first, last):
            return np.full(2, -np.inf), np.full(2, np.inf)
        return self.source_curve.derivative_bounds(first, last, order=order)

    def shares_endpoint(self, other: AbstractTrimCurve, /) -> bool:
        return _trim_source_join(self, other)


def pcurve_periodic_source(
    curve: CurveEvaluator,
    patch: AbstractSurfacePatch,
    /,
) -> tuple[CurveEvaluator, tuple[int, int]] | None:
    """Extract a proved integer sheet gauge on this exact patch definition.

    Callers can cancel these integers against coupled atlas node/root shifts
    BEFORE enclosing periods. An affine transport with an incompatible gauge
    cannot be stripped, and a floating seam coordinate supplies no such proof.
    Physical placement retains its definition's UV chart and native periods;
    admitting that exact retained ancestor does not cancel or identify poses.
    """
    from ._placed import PlacedSurface

    definitions = [encode_geometry(patch)]
    definition = patch
    while isinstance(definition, PlacedSurface):
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
            periods = _native_period_symbols(patch)
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
    """Oriented range of an exact two-dimensional curve used as a trim edge."""

    curve: AbstractCurve | AbstractTrimCurve
    first: float = eqx.field(static=True)
    last: float = eqx.field(static=True)
    reversed: bool = eqx.field(static=True)
    first_root: RootEndpoint | None
    last_root: RootEndpoint | None

    def __init__(
        self,
        curve: AbstractCurve | AbstractTrimCurve,
        first: float,
        last: float,
        *,
        reversed: bool = False,
        first_root: RootEndpoint | None = None,
        last_root: RootEndpoint | None = None,
    ) -> None:
        if not isinstance(curve, (AbstractCurve, AbstractTrimCurve)):
            raise TypeError("curve must provide a native curve or trim-curve capability.")
        if isinstance(curve, AbstractCurve):
            if curve.ambient_dimension != 2:
                raise ValueError("Trim curves must be two-dimensional p-curves.")
            first_, last_ = curve.validate_range(first, last)
        else:
            first_, last_ = float(first), float(last)
            lower, upper = curve.parameter_interval
            if not (math.isfinite(first_) and math.isfinite(last_) and first_ < last_):
                raise ValueError("Trim segment range must be finite and ordered.")
            if not lower <= first_ < last_ <= upper:
                if not isinstance(curve, IntersectionPCurve) or curve.reversed:
                    raise ValueError("Trim segment range lies outside its curve.")
                curve.curve._rooted_query_range(
                    first_,
                    last_,
                    (first_root, last_root),
                    domain_lower=lower,
                    domain_upper=upper,
                )
        self.curve = curve
        self.first = first_
        self.last = last_
        self.reversed = bool(reversed)
        from ._intersection import (
            BranchRootEndpoint,
            NativePeriodEndpoint,
            TrimRootEndpoint,
        )

        for endpoint, parameter in ((first_root, first_), (last_root, last_)):
            if endpoint is not None:
                if not isinstance(
                    endpoint, (TrimRootEndpoint, BranchRootEndpoint, NativePeriodEndpoint)
                ):
                    raise TypeError(
                        "Trim endpoint roots must be canonical source root expressions."
                    )
                if _endpoint_source_transport(curve, endpoint) is None:
                    raise ValueError(
                        "A trim root parameter must bind this original source carrier or its exact affine UV operation."
                    )
                lower, upper = endpoint.parameter_enclosure()
                if not lower <= parameter <= upper:
                    raise ValueError(
                        "A numerical trim endpoint lies outside its root expression enclosure."
                    )
        self.first_root, self.last_root = first_root, last_root
        first_box = (
            (first_, first_) if first_root is None else first_root.parameter_enclosure()
        )
        last_box = (
            (last_, last_) if last_root is None else last_root.parameter_enclosure()
        )
        if first_box[1] >= last_box[0]:
            raise ValueError(
                "Root-valued trim endpoints must have a proved parameter ordering."
            )

    @property
    def parameter_interval(self) -> tuple[float, float]:
        return 0.0, 1.0

    @overload
    def _carrier(self, local: Array, /) -> Array: ...

    @overload
    def _carrier(self, local: float, /) -> float: ...

    def _carrier(self, local: Array | float, /) -> Array | float:
        fraction = 1.0 - local if self.reversed else local
        return self.first + fraction * (self.last - self.first)

    def carrier_parameter_enclosure(
        self,
        first: float,
        last: float,
        /,
        *,
        source_extension: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not (math.isfinite(first) and math.isfinite(last) and first <= last) or (
            not source_extension and not 0 <= first <= last <= 1
        ):
            raise ValueError(
                "Trim carrier enclosure requires an ordered source subinterval."
            )
        from .._interval_enclosure import interval_add, interval_multiply

        a = (
            (self.first, self.first)
            if self.first_root is None
            else self.first_root.parameter_enclosure()
        )
        b = (
            (self.last, self.last)
            if self.last_root is None
            else self.last_root.parameter_enclosure()
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

    def evaluate(self, parameters: Array, /) -> Array:
        local = jnp.asarray(parameters, dtype=jnp.float64)
        return _trim_carrier_evaluate(
            self.curve, self._carrier(local), self.first, self.last
        )

    def is_c1_on(self, first: float, last: float, /) -> bool:
        endpoints = self.carrier_parameter_enclosure(first, last)
        return _trim_carrier_is_c1(
            self.curve,
            float(endpoints[0]),
            float(endpoints[1]),
            self.first,
            self.last,
        )

    def derivative_bounds(
        self,
        first: float,
        last: float,
        /,
        *,
        order: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        if order not in (1, 2) or isinstance(order, bool):
            raise ValueError("Trim derivatives support orders one and two.")
        if not (0 <= first <= last <= 1):
            raise ValueError("Trim derivative range must lie inside [0, 1].")
        endpoints = self.carrier_parameter_enclosure(first, last)
        if order == 2 and not self.is_c1_on(first, last):
            return np.full(2, -np.inf), np.full(2, np.inf)
        lower, upper = _trim_carrier_derivatives(
            self.curve,
            float(endpoints[0]),
            float(endpoints[1]),
            self.first,
            self.last,
            order,
        )
        a = (
            (self.first, self.first)
            if self.first_root is None
            else self.first_root.parameter_enclosure()
        )
        b = (
            (self.last, self.last)
            if self.last_root is None
            else self.last_root.parameter_enclosure()
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
        from .._interval_enclosure import interval_multiply

        lower, upper = interval_multiply((lower, upper), scale)
        return (-upper, -lower) if self.reversed and order == 1 else (lower, upper)

    def shares_endpoint(self, other: AbstractTrimCurve, /) -> bool:
        return _trim_source_join(self, other)

    def enclosure(self, first: float, last: float, /) -> np.ndarray:
        endpoints = self.carrier_parameter_enclosure(first, last)
        lower, upper = float(endpoints[0]), float(endpoints[1])
        return _trim_carrier_enclosure(self.curve, lower, upper, self.first, self.last)


def _pcurve_chart_support(
    curve: IntersectionPCurve, first: float, last: float, /
) -> tuple[int, int]:
    lower, upper = sorted((float(curve._carrier(first)), float(curve._carrier(last))))
    source_lower, source_upper = sorted(
        (float(curve._carrier(curve.first)), float(curve._carrier(curve.last)))
    )
    # Select the original owning charts, not a neighboring chart reached only
    # by outward endpoint uncertainty. Query coordinates themselves are not
    # clipped: their source enclosures still require certified chart support.
    lower = max(source_lower, min(lower, source_upper))
    upper = max(source_lower, min(upper, source_upper))
    head = max(0, min(int(math.floor(lower)), curve.curve.num_charts - 1))
    return head, max(head, min(int(math.ceil(upper)) - 1, curve.curve.num_charts - 1))


def _trim_carrier_evaluate(
    curve: CurveEvaluator, parameters: Array, first: float, last: float, /
) -> Array:
    if isinstance(curve, PeriodicPCurve):
        return (
            _trim_carrier_evaluate(curve.source_curve, parameters, first, last)
            + curve.offset_value
        )
    if isinstance(curve, AffinePCurve):
        return (
            _trim_carrier_evaluate(curve.curve, parameters, first, last)
            @ curve.matrix_value.T
            + curve.offset_value
        )
    if isinstance(curve, IntersectionPCurve):
        head, tail = _pcurve_chart_support(curve, first, last)
        result = curve.curve._evaluate_in_chart_range(
            curve._carrier(parameters), head, tail, certify=False
        )
        return (
            result.first_parameters if curve.side == "first" else result.second_parameters
        )
    return curve.evaluate(parameters)


def _trim_carrier_is_c1(
    curve: CurveEvaluator, lower: float, upper: float, first: float, last: float, /
) -> bool:
    if isinstance(curve, PeriodicPCurve):
        return _trim_carrier_is_c1(curve.source_curve, lower, upper, first, last)
    if isinstance(curve, AffinePCurve):
        return _trim_carrier_is_c1(curve.curve, lower, upper, first, last)
    if isinstance(curve, IntersectionPCurve):
        head, tail = _pcurve_chart_support(curve, first, last)
        values = sorted((float(curve._carrier(lower)), float(curve._carrier(upper))))
        boxes = curve.curve.parameter_enclosures(
            *values, minimum_chart=head, maximum_chart=tail, source_extension=True
        )
        if not np.all(np.isfinite(boxes)):
            return False
        return _intersection_is_c1(
            curve.curve,
            max(0.0, values[0]),
            min(curve.curve.num_charts, values[1]),
            minimum_chart=head,
            maximum_chart=tail,
        )
    capability = getattr(curve, "is_c1_on", None)
    return capability is not None and capability(lower, upper)


def _trim_carrier_derivatives(
    curve: CurveEvaluator,
    lower: float,
    upper: float,
    first: float,
    last: float,
    order: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(curve, PeriodicPCurve):
        return _trim_carrier_derivatives(
            curve.source_curve, lower, upper, first, last, order
        )
    if isinstance(curve, AffinePCurve):
        bounds = np.stack(
            _trim_carrier_derivatives(curve.curve, lower, upper, first, last, order)
        )
        lo, hi = curve._transport_bounds(bounds, translate=False)
        return lo, hi
    if isinstance(curve, IntersectionPCurve):
        head, tail = _pcurve_chart_support(curve, first, last)
        values = sorted((float(curve._carrier(lower)), float(curve._carrier(upper))))
        lo, hi = _intersection_derivative_bounds(
            curve.curve,
            values[0],
            values[1],
            order,
            curve.side,
            minimum_chart=head,
            maximum_chart=tail,
            source_extension=True,
        )
        return (-hi, -lo) if curve.reversed and order == 1 else (lo, hi)
    return curve.derivative_bounds(lower, upper, order=order)


def _trim_carrier_enclosure(
    curve: CurveEvaluator, lower: float, upper: float, first: float, last: float, /
) -> np.ndarray:
    if isinstance(curve, PeriodicPCurve):
        return curve._transport_bounds(
            _trim_carrier_enclosure(curve.source_curve, lower, upper, first, last)
        )
    if isinstance(curve, AffinePCurve):
        return curve._transport_bounds(
            _trim_carrier_enclosure(curve.curve, lower, upper, first, last),
            translate=True,
        )
    if isinstance(curve, IntersectionPCurve):
        head, tail = _pcurve_chart_support(curve, first, last)
        values = sorted((float(curve._carrier(lower)), float(curve._carrier(upper))))
        boxes = curve.curve.parameter_enclosures(
            *values,
            minimum_chart=head,
            maximum_chart=tail,
            source_extension=True,
        )
        columns = curve._columns()
        return np.stack(
            (np.min(boxes[:, 0, columns], axis=0), np.max(boxes[:, 1, columns], axis=0))
        )
    if isinstance(curve, AbstractTrimCurve):
        return curve.enclosure(lower, upper)
    if lower < upper:
        return np.asarray(curve.bounding_box(lower, upper), dtype=np.float64)
    from .._interval_enclosure import prepare_interval_function

    images = []
    for piece in curve_pieces_for_interval(curve, lower, upper):
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


def _trim_source_jet(
    curve: CurveEvaluator,
    lower: float,
    upper: float,
    order: int,
    /,
    *,
    first: float | None = None,
    last: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Jet of the original source, extending only inside certified chart boxes.

    Range walls restrict the represented trim, not its defining equations.
    Regular endpoint roots need an open uniqueness box; this source query
    never changes the trim domain or admits an uncertified graph extension.
    """
    from .._interval_enclosure import interval_multiply

    if isinstance(curve, CurveTrimSegment):
        a = (
            (curve.first, curve.first)
            if curve.first_root is None
            else curve.first_root.parameter_enclosure()
        )
        b = (
            (curve.last, curve.last)
            if curve.last_root is None
            else curve.last_root.parameter_enclosure()
        )
        lo, hi = curve.carrier_parameter_enclosure(lower, upper, source_extension=True)
        result = _trim_source_jet(
            curve.curve, float(lo), float(hi), order, first=curve.first, last=curve.last
        )
        if order:
            delta = (
                np.nextafter(b[0] - a[1], -np.inf),
                np.nextafter(b[1] - a[0], np.inf),
            )
            scale = delta if order == 1 else interval_multiply(delta, delta)
            result = interval_multiply(result, scale)
            if curve.reversed and order == 1:
                result = -result[1], -result[0]
        return result
    if isinstance(curve, PeriodicPCurve):
        result = _trim_source_jet(
            curve.source_curve, lower, upper, order, first=first, last=last
        )
        box = (
            curve._transport_bounds(np.stack(result)) if order == 0 else np.stack(result)
        )
        return box[0], box[1]
    if isinstance(curve, AffinePCurve):
        result = _trim_source_jet(
            curve.curve, lower, upper, order, first=first, last=last
        )
        box = curve._transport_bounds(np.stack(result), translate=order == 0)
        return box[0], box[1]
    if isinstance(curve, IntersectionPCurve):
        head, tail = _pcurve_chart_support(
            curve,
            curve.first if first is None else first,
            curve.last if last is None else last,
        )
        values = sorted((float(curve._carrier(lower)), float(curve._carrier(upper))))
        if order:
            result = _intersection_derivative_bounds(
                curve.curve,
                values[0],
                values[1],
                order,
                curve.side,
                minimum_chart=head,
                maximum_chart=tail,
                source_extension=True,
            )
            return (-result[1], -result[0]) if curve.reversed and order == 1 else result
        boxes = curve.curve.parameter_enclosures(
            *values,
            minimum_chart=head,
            maximum_chart=tail,
            source_extension=True,
        )
        columns = curve._columns()
        return np.min(boxes[:, 0, columns], axis=0), np.max(boxes[:, 1, columns], axis=0)
    if isinstance(curve, AbstractTrimCurve):
        domain = curve.parameter_interval
        if lower < domain[0] or upper > domain[1]:
            return np.full(2, -np.inf), np.full(2, np.inf)
        if order == 0:
            box = curve.enclosure(lower, upper)
            return box[0], box[1]
        return curve.derivative_bounds(lower, upper, order=order)
    derivative = curve.evaluate
    for _ in range(order):
        derivative = jax.jacfwd(derivative)
    from .._interval_enclosure import prepare_interval_function

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
                offset[0] + _period_offset(Fraction(0), Fraction(curve.period_shifts[0])),
                offset[1] + _period_offset(Fraction(0), Fraction(curve.period_shifts[1])),
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
    if type(first).__name__ in _MODULE_TYPES and type(second).__name__ in _MODULE_TYPES:
        return encode_geometry(first) == encode_geometry(second)
    return False


def pcurve_affine_correspondence(
    first: CurveEvaluator, second: CurveEvaluator, /
) -> _ExactUVMap | None:
    """Exact UV map from first to second iff original source/parameter atoms agree.

    Consumers must compare this exact map (rational coefficients and symbolic
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
    if (
        not isinstance(patch, AbstractSurfacePatch)
        or pcurve_periodic_source(second, patch) is None
    ):
        return None
    correspondence = pcurve_affine_correspondence(first, second)
    if correspondence is None or correspondence[0] != expected_matrix:
        return None
    shifts = []
    for actual, expected, period in zip(
        correspondence[1],
        expected_offset,
        _native_period_symbols(patch),
        strict=True,
    ):
        delta = actual - expected
        if isinstance(delta, _PeriodOffset):
            if delta.rational or delta.turns.denominator != 1 or period != "two_pi":
                return None
            shifts.append(int(delta.turns))
        elif delta:
            return None
        else:
            shifts.append(0)
    return shifts[0], shifts[1]


def _root_parameter_affine(endpoint: RootEndpoint, /) -> tuple[Fraction, Fraction]:
    from ._intersection import TrimRootEndpoint

    scale = (
        Fraction(endpoint.affine_parameter_scale)
        if isinstance(endpoint, TrimRootEndpoint)
        else Fraction(1)
    )
    offset = (
        Fraction(endpoint.affine_parameter_offset)
        if isinstance(endpoint, TrimRootEndpoint)
        else Fraction(0)
    )
    for next_scale, next_offset in endpoint.affine_transforms:
        scale, offset = (
            Fraction(next_scale) * scale,
            Fraction(next_scale) * offset + Fraction(next_offset),
        )
    return scale, offset


def _root_parameter_in_source(
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
    """Prove the parameter's original carrier and retain the exact UV operation."""
    source, matrix, offset = _pcurve_affine_source(carrier)
    root_source, root_matrix, root_offset = _pcurve_affine_source(endpoint.carrier)
    if isinstance(root_source, IntersectionCurve):
        if (
            isinstance(source, IntersectionPCurve)
            and source.curve.branch_id == root_source.branch_id
        ):
            return matrix, offset, ("branch", root_source.branch_id, source.side)
        from ._intersection import BranchRootEndpoint

        if not isinstance(endpoint, BranchRootEndpoint) or endpoint.source_pcurve is None:
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
            and _root_parameter_in_source(carrier, endpoint) == (Fraction(1), Fraction(0))
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
            _period_offset(Fraction(0), Fraction(curve.period_shifts[0])),
            _period_offset(Fraction(0), Fraction(curve.period_shifts[1])),
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


def _exact_curve_parameter_point(
    carrier: CurveEvaluator | IntersectionCurve,
    parameter: Fraction | _PeriodOffset,
    /,
) -> tuple[Fraction | _PeriodOffset, ...] | None:
    """Exact source image when the source expression has a provable finite form."""
    from ._placed import PlacedCurve

    if isinstance(carrier, PlacedCurve):
        point = _exact_curve_parameter_point(carrier.definition, parameter)
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
    if isinstance(carrier, PeriodicPCurve):
        point = _exact_curve_parameter_point(carrier.source_curve, parameter)
        if point is None:
            return None
        return tuple(
            value + _period_offset(Fraction(0), Fraction(shift))
            for value, shift in zip(point, carrier.period_shifts, strict=True)
        )
    if isinstance(carrier, AffinePCurve):
        point = _exact_curve_parameter_point(carrier.curve, parameter)
        if point is None:
            return None
        return tuple(
            carrier.offset[i]
            + sum((carrier.matrix[i][j] * point[j] for j in range(2)), Fraction(0))
            for i in range(2)
        )
    if isinstance(carrier, LineCurve):
        return tuple(
            Fraction(float(origin)) + parameter * Fraction(float(direction))
            for origin, direction in zip(
                np.asarray(carrier.origin), np.asarray(carrier.direction), strict=True
            )
        )
    if isinstance(carrier, BSplineCurve) and isinstance(parameter, Fraction):
        from ...discretization._coordinate_enclosure import _COORDINATE_BUDGET

        budget = _COORDINATE_BUDGET.get()
        if budget is not None:
            from ...discretization._coordinate_enclosure import _reserve_polynomial

            _reserve_polynomial(2, 2, 0, 2200)
            budget.reserve(carrier.knots.size, 128 + 32 * carrier.knots.size)
        knots = np.asarray(carrier.knots)
        lower, upper = float(knots[carrier.degree]), float(knots[-carrier.degree - 1])
        if parameter in (Fraction(lower), Fraction(upper)):
            # Multiplicity at least degree makes the active one-sided image
            # a source control. Its index follows the actual knot run, not
            # the outer control-bank endpoints: exterior zero-width spans
            # can leave other controls outside the finite active domain.
            knot = lower if parameter == lower else upper
            matches = np.flatnonzero(knots == knot)
            if matches.size >= carrier.degree:
                if budget is not None:
                    budget.reserve(
                        carrier.control_points.size, 128 + 8 * carrier.control_points.size
                    )
                    _reserve_polynomial(
                        carrier.ambient_dimension, carrier.ambient_dimension, 0, 2200
                    )
                index = (
                    int(matches[-1]) - carrier.degree
                    if parameter == lower
                    else int(matches[0]) - 1
                )
                control = np.asarray(carrier.control_points)[index]
                return tuple(Fraction(float(value)) for value in control)
    if isinstance(carrier, (CircleCurve, EllipseCurve)):
        if isinstance(parameter, _PeriodOffset):
            quarter = parameter.turns * 4
            if parameter.rational or quarter.denominator != 1:
                return None
            index = int(quarter)
        elif parameter == 0:
            index = 0
        else:
            # A nonzero binary64 multiple of float(pi) is not a mathematical
            # quarter-turn. Preserve its authored real parameter meaning.
            return None
        cosine, sine = ((1, 0), (0, 1), (-1, 0), (0, -1))[index % 4]
        a = (
            float(carrier.radius)
            if isinstance(carrier, CircleCurve)
            else float(carrier.first_radius)
        )
        b = (
            float(carrier.radius)
            if isinstance(carrier, CircleCurve)
            else float(carrier.second_radius)
        )
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
    return None


def _trim_exact_parameter_endpoint(
    curve: CurveEvaluator,
    start: bool,
    /,
) -> tuple[CurveEvaluator, Fraction | _PeriodOffset] | None:
    from ._intersection import NativePeriodEndpoint

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
                not isinstance(endpoint, NativePeriodEndpoint)
                or _endpoint_source_transport(curve.curve, endpoint) is None
            ):
                return None
            return curve.curve, endpoint.exact_parameter
    endpoint = _trim_source_endpoint(curve, start)
    return None if endpoint is None else (endpoint[0], Fraction(endpoint[1]))


def _trim_exact_endpoint_image(
    curve: CurveEvaluator,
    start: bool,
    /,
) -> tuple[Fraction | _PeriodOffset, ...] | None:
    endpoint = _trim_exact_parameter_endpoint(curve, start)
    return None if endpoint is None else _exact_curve_parameter_point(*endpoint)


def _trim_source_join(first: AbstractTrimCurve, second: AbstractTrimCurve, /) -> bool:
    """Structural source endpoint identity, not a sampled gap test."""
    left_root, right_root = (
        _trim_endpoint_root(first, False),
        _trim_endpoint_root(second, True),
    )
    from ._intersection import BranchRootEndpoint, NativePeriodEndpoint, TrimRootEndpoint

    if (
        left_root is not None
        and isinstance(left_root[0], NativePeriodEndpoint)
        or right_root is not None
        and isinstance(right_root[0], NativePeriodEndpoint)
    ):
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
                    and _native_curve_period_symbol(source_a) == "two_pi"
                    and isinstance(difference, _PeriodOffset)
                    and not difference.rational
                    and difference.turns.denominator == 1
                ):
                    return True
        point_a = _trim_exact_endpoint_image(first, False)
        point_b = _trim_exact_endpoint_image(second, True)
        return point_a is not None and point_a == point_b
    if left_root is not None or right_root is not None:
        # A failed root/source binding cannot fall through to floating endpoints.
        if left_root is None or right_root is None:
            return False
        a, carrier_a, transform_a = left_root
        b, carrier_b, transform_b = right_root
        if not isinstance(a, (TrimRootEndpoint, BranchRootEndpoint)) or not isinstance(
            b, (TrimRootEndpoint, BranchRootEndpoint)
        ):
            return False
        same_point = a.root_id == b.root_id and (
            a.root is b.root or encode_geometry(a.root) == encode_geometry(b.root)
        )
        from ._intersection import IntersectionCurvePointRoot

        if isinstance(a.root, IntersectionCurvePointRoot) and isinstance(
            b.root, IntersectionCurvePointRoot
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
        from ._intersection import BranchRootEndpoint, TrimRootEndpoint

        if isinstance(a, BranchRootEndpoint) or isinstance(b, BranchRootEndpoint):
            if not isinstance(a, BranchRootEndpoint) or not isinstance(
                b, BranchRootEndpoint
            ):
                return False
            if (
                _root_parameter_in_source(carrier_a, a)
                == _root_parameter_in_source(carrier_b, b)
                == (Fraction(1), Fraction(0))
            ):
                return True
            return encode_geometry(a) == encode_geometry(b)
        if isinstance(a, TrimRootEndpoint) and isinstance(b, TrimRootEndpoint):
            expression_a, expression_b = (
                _root_parameter_in_source(carrier_a, a),
                _root_parameter_in_source(carrier_b, b),
            )
            if expression_a == expression_b == (Fraction(1), Fraction(0)):
                return True
            return (
                a.operand == b.operand
                and expression_a == expression_b
                and _same_pcurve_parameter(carrier_a, carrier_b)
            )
        return False
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

    point_a = _exact_curve_parameter_point(carrier_a, Fraction(parameter_a))
    point_b = _exact_curve_parameter_point(carrier_b, Fraction(parameter_b))
    return point_a is not None and point_a == point_b


for _geometry_class in (
    SurfaceRegion,
    IntersectionCurve,
    IntersectionPCurve,
    AffinePCurve,
    PeriodicPCurve,
    CurveTrimSegment,
):
    _MODULE_TYPES[_geometry_class.__name__] = _geometry_class
    _CONSTRUCTION_FIELDS[_geometry_class.__name__] = tuple(
        name
        for name in signature(_geometry_class.__init__).parameters
        if name != "self" and not name.startswith("_")
    )


__all__ = [
    "AffinePCurve",
    "BernsteinCurvePiece",
    "BernsteinSurfacePiece",
    "CurvePiece",
    "CurveRange",
    "CurveTrimSegment",
    "IntersectionCurve",
    "IntersectionCurvePoint",
    "IntersectionCurveSide",
    "IntersectionEndpointKind",
    "IntersectionPCurve",
    "PeriodicPCurve",
    "SurfacePairSystem",
    "SurfacePiece",
    "SurfaceRegion",
    "curve_pieces",
    "decode_geometry",
    "encode_geometry",
    "surface_pieces",
    "pcurve_affine_correspondence",
    "pcurve_periodic_source",
    "pcurve_surface_period_shifts",
]
