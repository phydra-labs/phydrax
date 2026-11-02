#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-native enclosed B-Rep volume measures.

Every measure is the divergence-theorem flux of ``x / 3`` through the source
faces, enclosed from source carriers; tessellation triangles never define it.
Direct smoke recipe::

    from phydrax._physical import SpatialCoordinateContract
    from phydrax.geometry.brep._constructors import brep_sphere
    from phydrax.geometry.brep._measure import brep_mass_properties

    model = brep_sphere(1.0, coordinate_contract=SpatialCoordinateContract.si())
    measure = brep_mass_properties(model, relative_tolerance=1e-6)
    assert measure.status == ("certified",)
    lower, upper = measure.volume_enclosures[0]
    assert lower <= 4.18879020478639 <= upper
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import comb, fsum, isfinite
from numbers import Integral, Real
from typing import final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._jax_config import x64_enabled
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._exact_algebraic_geometry import coefficient_interval, ExactScalar
from ...typing import as_array, Bool, checked, Dim, Float64, parse, Scope, Size
from .._atlas import TrimDomain, TrimEnclosures
from .._interval_enclosure import (
    interval_add,
    interval_divide,
    interval_multiply,
    interval_subtract,
    IntervalExtensionError,
    prepare_interval_function,
    PreparedIntervalFunction,
)
from .._rational import (
    fraction_down,
    fraction_up,
    rational_cross,
    rational_dot,
    rational_vector,
)
from ._assembly import BRepAssemblyOccurrence
from ._boolean_core import AffinePCurve, PeriodicPCurve
from ._exact import rational_determinant
from ._intersection import _straight_revolution
from ._intersection_curve import IntersectionPCurve, orthogonal_cylinder_family
from ._model import BRepEntityId, BRepGeometry, BRepModel
from ._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    BSplineCurve,
    BSplineSurfacePatch,
    CircleCurve,
    ConePatch,
    ConicRootLinePCurve,
    CylinderPatch,
    EllipseCurve,
    ExactLineCurve,
    ExtrusionSurface,
    HyperbolaCurve,
    LineCurve,
    OffsetCurve,
    OffsetSurface,
    ParabolaCurve,
    PlaneCylinderSectionPCurve,
    PlanePatch,
    RevolutionSurface,
    RuledSurface,
    SpherePatch,
    TorusPatch,
    TorusRootedCirclePCurve,
)
from ._placed import PlacedCurve, PlacedSurface
from ._trim_helpers import period_scalar_bounds, PeriodOffset
from ._tspline import TSplineSurfacePatch


type BRepMassMeasureStatus = Literal["certified", "unsupported", "unresolved"]
type _Interval = tuple[np.ndarray, np.ndarray]
type _ScalarInterval = tuple[float, float]
type _Vector = tuple[Fraction, Fraction, Fraction]

# Scheduling widths of one interval batch; they never limit refinement. The
# native executor runs small batches fastest, the host interpreter amortizes
# its per-equation dispatch over wide ones.
_NATIVE_BATCH = 64
_HOST_BATCH = 512
_NO_BREAKS = np.empty((0,), dtype=np.float64)
# Relative headroom for the few binary64 operations of the closure bounds,
# each rounded at most once to nearest.
_BOUND_HEADROOM = 1.0 + 64.0 * float(np.finfo(np.float64).eps)


class _MeasureOccurrenceDim(Dim, minimum=1):
    """Solid occurrences represented by one mass measure."""


class _MeasureFailure(Exception):
    def __init__(
        self,
        status: BRepMassMeasureStatus,
        reason: str,
        /,
        *,
        enclosure: _ScalarInterval = (-np.inf, np.inf),
    ) -> None:
        self.status = status
        self.reason = reason
        self.enclosure = enclosure
        super().__init__(reason)


@final
class BRepMassProperties(StrictModule, NonTrainableState):
    """Certified source-volume intervals in assembly occurrence order.

    ``volume_enclosures`` meet ``relative_tolerance``: a certified occurrence's
    half-width is at most ``relative_tolerance`` times its lower bound. ``volume``
    is the interval midpoint for numerical consumers; scientific decisions must
    use ``volume_enclosures`` and ``status``. ``exact`` marks occurrences whose
    enclosure is the outward binary64 rounding of an exact rational volume.
    Unsupported and unresolved occurrences carry ``[-inf, inf]`` and NaN, never a
    sampled finite estimate; their ``reached_enclosures`` row keeps the tightest
    certified enclosure the refinement reached before refusing (infinite when
    none formed) and ``reasons`` states why. ``evaluations`` counts interval
    cell enclosures per occurrence. ``solid_ids`` retain source owner, revision,
    and assembly occurrence identity.
    """

    __strict_contract__ = True

    model_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    solid_indices: tuple[int, ...] = eqx.field(static=True)
    occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    solid_ids: tuple[BRepEntityId, ...] = eqx.field(static=True)
    status: tuple[BRepMassMeasureStatus, ...] = eqx.field(static=True)
    reasons: tuple[str, ...] = eqx.field(static=True)
    exact: tuple[bool, ...] = eqx.field(static=True)
    evaluations: tuple[int, ...] = eqx.field(static=True)
    volume: Float64[_MeasureOccurrenceDim]
    volume_enclosures: Float64[_MeasureOccurrenceDim, Literal[2]]
    reached_enclosures: Float64[_MeasureOccurrenceDim, Literal[2]]

    def __init__(
        self,
        *,
        model_id: str,
        source_id: str,
        relative_tolerance: float,
        solid_indices: tuple[int, ...],
        occurrence_paths: tuple[tuple[str, ...], ...],
        solid_ids: tuple[BRepEntityId, ...],
        status: tuple[BRepMassMeasureStatus, ...],
        reasons: tuple[str, ...],
        exact: tuple[bool, ...],
        evaluations: tuple[int, ...],
        reached_enclosures: np.ndarray,
    ) -> None:
        count = len(solid_indices)
        scope = Scope()
        parse(
            count,
            Size[_MeasureOccurrenceDim],
            "BRepMassProperties.occurrence_count",
            scope=scope,
        )
        tolerance = _relative_tolerance(relative_tolerance)
        status_ = tuple(
            parse(value, BRepMassMeasureStatus, "BRepMassProperties.status")
            for value in status
        )
        reasons_ = tuple(reasons)
        exact_ = tuple(bool(value) for value in exact)
        evaluations_ = tuple(int(value) for value in evaluations)
        reached = np.asarray(reached_enclosures, dtype=np.float64)
        if (
            len(occurrence_paths) != count
            or len(solid_ids) != count
            or len(status_) != count
            or len(reasons_) != count
            or len(exact_) != count
            or len(evaluations_) != count
            or any(value < 0 for value in evaluations_)
            or reached.shape != (count, 2)
            or np.any(np.isnan(reached))
            or np.any(reached[:, 0] > reached[:, 1])
        ):
            raise ValueError("BRepMassProperties occurrence data must align.")
        certified = tuple(state == "certified" for state in status_)
        if any(
            ok != (bool(np.all(np.isfinite(reached[index]))) and not reasons_[index])
            for index, ok in enumerate(certified)
        ) or any(flag and not ok for flag, ok in zip(exact_, certified, strict=True)):
            raise ValueError(
                "BRepMassProperties finite enclosures and failure reasons must match status."
            )
        bounds = np.where(
            np.asarray(certified, dtype=np.bool_)[:, None],
            reached,
            np.asarray((-np.inf, np.inf), dtype=np.float64),
        )
        midpoint = np.asarray(
            [
                bounds[index, 0] * 0.5 + bounds[index, 1] * 0.5 if ok else np.nan
                for index, ok in enumerate(certified)
            ],
            dtype=np.float64,
        )
        self.model_id = model_id
        self.source_id = source_id
        self.relative_tolerance = tolerance
        self.solid_indices = solid_indices
        self.occurrence_paths = occurrence_paths
        self.solid_ids = solid_ids
        self.status = status_
        self.reasons = reasons_
        self.exact = exact_
        self.evaluations = evaluations_
        self.volume = as_array(
            midpoint,
            Float64[_MeasureOccurrenceDim],
            "BRepMassProperties.volume",
            scope=scope,
        )
        self.volume_enclosures = as_array(
            bounds,
            Float64[_MeasureOccurrenceDim, Literal[2]],
            "BRepMassProperties.volume_enclosures",
            scope=scope,
        )
        self.reached_enclosures = as_array(
            reached,
            Float64[_MeasureOccurrenceDim, Literal[2]],
            "BRepMassProperties.reached_enclosures",
            scope=scope,
        )

    @property
    def successful(self) -> Bool[_MeasureOccurrenceDim]:
        return jnp.asarray(
            tuple(value == "certified" for value in self.status), dtype=jnp.bool_
        )


class BRepMeasureRefusal(ValueError):
    """A source volume decision was refused; ``status``/``reason`` carry the evidence.

    ``enclosure`` is the tightest certified enclosure reached before refusing
    (infinite when none formed). Interior-mass refusals also carry the full
    per-occurrence ``properties``.
    """

    def __init__(
        self,
        status: BRepMassMeasureStatus,
        reason: str,
        /,
        *,
        enclosure: _ScalarInterval = (-np.inf, np.inf),
        properties: BRepMassProperties | None = None,
    ) -> None:
        self.status = status
        self.reason = reason
        self.enclosure = enclosure
        self.properties = properties
        super().__init__(f"{status}: {reason}")


def _relative_tolerance(value: object, /) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("relative_tolerance must be a real number.")
    tolerance = float(value)
    if not (isfinite(tolerance) and tolerance > 0.0):
        raise ValueError("relative_tolerance must be finite and positive.")
    return tolerance


# ---------------------------------------------------------------------------
# Smooth source pieces


def _curve_breaks(curve: AbstractCurve, /) -> np.ndarray | None:
    """Parameters where a source curve may lose C2 smoothness; None if unknown."""
    match curve:
        case (
            LineCurve()
            | ExactLineCurve()
            | CircleCurve()
            | EllipseCurve()
            | ParabolaCurve()
            | HyperbolaCurve()
            | ConicRootLinePCurve()
            | PlaneCylinderSectionPCurve()
            | TorusRootedCirclePCurve()
        ):
            return _NO_BREAKS
        case BSplineCurve():
            return np.unique(np.asarray(curve.knots, dtype=np.float64))
        case (
            OffsetCurve(base=base)
            | PeriodicPCurve(source_curve=base)
            | AffinePCurve(curve=base)
            | PlacedCurve(definition=base)
        ):
            return _curve_breaks(base)
        case _:
            return None


def _surface_breaks(
    patch: AbstractSurfacePatch, /
) -> tuple[np.ndarray, np.ndarray] | None:
    """Parameter lines where a source surface may lose C2 smoothness."""
    match patch:
        case PlanePatch() | CylinderPatch() | ConePatch() | SpherePatch() | TorusPatch():
            return _NO_BREAKS, _NO_BREAKS
        case BSplineSurfacePatch():
            return (
                np.unique(np.asarray(patch.u_knots, dtype=np.float64)),
                np.unique(np.asarray(patch.v_knots, dtype=np.float64)),
            )
        case TSplineSurfacePatch():
            pieces = np.asarray(patch.realization.piece_bounds, dtype=np.float64)
            return np.unique(pieces[:, :2]), np.unique(pieces[:, 2:])
        case ExtrusionSurface(curve=curve):
            breaks = _curve_breaks(curve)
            return None if breaks is None else (breaks, _NO_BREAKS)
        case RevolutionSurface(curve=curve):
            breaks = _curve_breaks(curve)
            return None if breaks is None else (_NO_BREAKS, breaks)
        case RuledSurface(first=first, second=second):
            first_breaks, second_breaks = _curve_breaks(first), _curve_breaks(second)
            if first_breaks is None or second_breaks is None:
                return None
            return np.union1d(first_breaks, second_breaks), _NO_BREAKS
        case OffsetSurface(base=base) | PlacedSurface(definition=base):
            return _surface_breaks(base)
        case _:
            return None


def _interior_breaks(breaks: np.ndarray, first: float, last: float, /) -> np.ndarray:
    return breaks[(breaks > first) & (breaks < last)]


def _break_grid(
    lower: np.ndarray, upper: np.ndarray, breaks: Sequence[np.ndarray]
) -> np.ndarray:
    """Boxes of the product grid that splits ``[lower, upper]`` at every break."""
    axes = [
        np.concatenate(
            (
                lower[axis : axis + 1],
                _interior_breaks(breaks[axis], lower[axis], upper[axis]),
                upper[axis : axis + 1],
            )
        )
        for axis in range(lower.size)
    ]
    starts = np.stack(
        np.meshgrid(*(values[:-1] for values in axes), indexing="ij"), axis=-1
    ).reshape((-1, lower.size))
    stops = np.stack(
        np.meshgrid(*(values[1:] for values in axes), indexing="ij"), axis=-1
    ).reshape((-1, lower.size))
    return np.stack((starts, stops), axis=1)


# ---------------------------------------------------------------------------
# Interval programs


def _surface_flux(patch: AbstractSurfacePatch, parameters: Array, /) -> Array:
    """The divergence-theorem integrand ``x . (S_u x S_v) / 3``."""
    point = patch.evaluate(parameters)
    frame = jax.jacfwd(patch.evaluate)(parameters)
    return jnp.dot(point, jnp.cross(frame[:, 0], frame[:, 1])) / 3.0


def _with_hessian(
    function: Callable[[Array], Array], /, *, fourth: bool = False
) -> Callable[[Array], Array]:
    fourth_derivative = (
        jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(function)))) if fourth else None
    )

    def jet(parameters: Array, /) -> Array:
        hessian = jax.jacfwd(jax.jacfwd(function))(parameters)
        values = (function(parameters)[None], hessian.reshape((-1,)))
        if fourth_derivative is not None:
            values += (fourth_derivative(parameters).reshape((-1,)),)
        return jnp.concatenate(values)

    return jet


def _cell_program(
    function: Callable[[Array], Array], dimension: int, /
) -> PreparedIntervalFunction:
    """Prepare a cell integrand on the native executor when it lowers completely.

    Both executions return outward-rounded enclosures of the same jaxpr; the
    host interpreter is used only for programs with an operation the native
    executor does not advertise.
    """
    try:
        return prepare_interval_function(
            function, dimension, batch_capacity=_NATIVE_BATCH, execution="native"
        )
    except IntervalExtensionError:
        return prepare_interval_function(function, dimension, batch_capacity=_HOST_BATCH)


def _surface_flux_program(patch: AbstractSurfacePatch, /) -> PreparedIntervalFunction:
    """The chart integrand and its parameter Hessian."""
    return _cell_program(
        _with_hessian(lambda parameters: _surface_flux(patch, parameters)), 2
    )


def _surface_frame_program(patch: AbstractSurfacePatch, /) -> _IntervalProgram:
    """The chart integrand followed by ``S_u`` and ``S_v``."""
    if isinstance(patch, CylinderPatch):
        coefficients = (
            *patch.origin_exact,
            *patch.first_axis_exact,
            *patch.second_axis_exact,
            *patch.axis_exact,
            patch.radius_exact,
        )
        bounds = tuple(coefficient_interval(value, bits=128) for value in coefficients)
        lower = np.asarray([fraction_down(value[0]) for value in bounds])
        upper = np.asarray([fraction_up(value[1]) for value in bounds])

        def source_frame(parameters: Array, /) -> Array:
            uv, values = parameters[:2], parameters[2:]
            origin, first_axis, second_axis, axis = (
                values[:3],
                values[3:6],
                values[6:9],
                values[9:12],
            )
            radius = values[12]

            def point(angles: Array, /) -> Array:
                return (
                    origin
                    + radius
                    * (jnp.cos(angles[0]) * first_axis + jnp.sin(angles[0]) * second_axis)
                    + angles[1] * axis
                )

            frame = jax.jacfwd(point)(uv)
            flux = jnp.dot(point(uv), jnp.cross(frame[:, 0], frame[:, 1])) / 3
            return jnp.concatenate((flux[None], frame.T.reshape((-1,))))

        return _BoundProgram(_cell_program(source_frame, 15), lower, upper)
    straight = (
        _straight_revolution(patch) if isinstance(patch, RevolutionSurface) else None
    )
    if straight is not None and straight.slope == 0:
        coefficients = (
            *straight.origin,
            *straight.radial,
            *straight.tangent,
            *straight.axis,
            straight.axial_origin,
            straight.axial_rate,
            straight.scale_origin,
        )
        bounds = tuple(coefficient_interval(value, bits=128) for value in coefficients)
        lower = np.asarray([fraction_down(value[0]) for value in bounds])
        upper = np.asarray([fraction_up(value[1]) for value in bounds])

        def revolution_frame(parameters: Array, /) -> Array:
            uv, values = parameters[:2], parameters[2:]
            origin, radial, tangent, axis = (
                values[:3],
                values[3:6],
                values[6:9],
                values[9:12],
            )
            axial_origin, axial_rate, scale = values[12], values[13], values[14]

            def point(angles: Array, /) -> Array:
                return (
                    origin
                    + (axial_origin + axial_rate * angles[1]) * axis
                    + scale * (jnp.cos(angles[0]) * radial + jnp.sin(angles[0]) * tangent)
                )

            frame = jax.jacfwd(point)(uv)
            flux = jnp.dot(point(uv), jnp.cross(frame[:, 0], frame[:, 1])) / 3
            return jnp.concatenate((flux[None], frame.T.reshape((-1,))))

        return _BoundProgram(_cell_program(revolution_frame, 17), lower, upper)
    return prepare_interval_function(
        lambda parameters: jnp.concatenate(
            (
                _surface_flux(patch, parameters)[None],
                jax.jacfwd(patch.evaluate)(parameters).T.reshape((-1,)),
            )
        ),
        2,
        batch_capacity=1,
    )


def _pcurve_point(pcurve: AbstractCurve, parameter: Array, /) -> Array:
    return jnp.reshape(pcurve.evaluate(jnp.reshape(parameter, ())), (2,))


def _section_source_bounds(curve: PlaneCylinderSectionPCurve, /) -> _Interval:
    coefficients = (
        *curve.offset_exact,
        *curve.linear_exact,
        *curve.cosine_exact,
        *curve.sine_exact,
    )
    bounds = tuple(coefficient_interval(value, bits=128) for value in coefficients)
    lower = [fraction_down(value[0]) for value in bounds]
    upper = [fraction_up(value[1]) for value in bounds]
    root = curve.angle_offset_root
    lower.append(0.0 if root is None else root.parameter_lower)
    upper.append(0.0 if root is None else root.parameter_upper)
    return np.asarray(lower), np.asarray(upper)


def _section_point(parameter: Array, coefficients: Array, /) -> Array:
    return (
        coefficients[:2]
        + parameter * coefficients[2:4]
        + jnp.cos(parameter) * coefficients[4:6]
        + jnp.sin(parameter) * coefficients[6:8]
        + jnp.stack((coefficients[8], jnp.zeros_like(parameter)))
    )


def _parameter_area_program(pcurve: AbstractCurve, /) -> _IntervalProgram:
    """Green density ``(u v' - v u') / 2`` of one p-curve and its derivatives."""
    section = pcurve
    shifts = (0, 0)
    while isinstance(section, PeriodicPCurve):
        shifts = tuple(a + b for a, b in zip(shifts, section.period_shifts, strict=True))
        section = section.source_curve
    if isinstance(section, PlaneCylinderSectionPCurve):
        lower, upper = _section_source_bounds(section)
        if any(shifts):
            bounds = [
                period_scalar_bounds(PeriodOffset(Fraction(0), Fraction(shift)))
                for shift in shifts
            ]
            lower = np.concatenate((lower, [float(bound[0]) for bound in bounds]))
            upper = np.concatenate((upper, [float(bound[1]) for bound in bounds]))

        def jet(parameters: Array, /) -> Array:
            coefficients = parameters[1:10]
            shift = parameters[10:12] if any(shifts) else jnp.zeros((2,))

            def point(value: Array, /) -> Array:
                return _section_point(value, coefficients) + shift

            def density(value: Array, /) -> Array:
                position = point(value[0])
                tangent = jax.jacfwd(point)(value[0])
                return 0.5 * (position[0] * tangent[1] - position[1] * tangent[0])

            return _with_hessian(density, fourth=True)(parameters[:1])

        return _BoundProgram(_cell_program(jet, 1 + lower.size), lower, upper)

    def density(parameters: Array, /) -> Array:
        point = _pcurve_point(pcurve, parameters[0])
        tangent = jax.jacfwd(lambda value: _pcurve_point(pcurve, value))(parameters[0])
        return 0.5 * (point[0] * tangent[1] - point[1] * tangent[0])

    return _cell_program(_with_hessian(density), 1)


@dataclass(frozen=True, slots=True)
class _BoundProgram:
    """Bind exact source scalars as directed non-integrated interval inputs."""

    program: PreparedIntervalFunction
    lower: np.ndarray
    upper: np.ndarray

    def evaluate(self, lower: np.ndarray, upper: np.ndarray, /) -> _Interval:
        count = lower.shape[0]
        return self.program.evaluate(
            np.concatenate(
                (lower, np.broadcast_to(self.lower, (count, self.lower.size))), axis=1
            ),
            np.concatenate(
                (upper, np.broadcast_to(self.upper, (count, self.upper.size))), axis=1
            ),
        )


@dataclass(frozen=True, slots=True)
class _PeriodBoundProgram:
    """Supply the directed exact native turn as a non-integrated parameter."""

    program: PreparedIntervalFunction

    def evaluate(self, lower: np.ndarray, upper: np.ndarray, /) -> _Interval:
        period_lower, period_upper = period_scalar_bounds(
            PeriodOffset(Fraction(0), Fraction(1))
        )
        count = lower.shape[0]
        return self.program.evaluate(
            np.concatenate((lower, np.full((count, 1), period_lower)), axis=1),
            np.concatenate((upper, np.full((count, 1), period_upper)), axis=1),
        )


type _IntervalProgram = PreparedIntervalFunction | _PeriodBoundProgram | _BoundProgram


def _green_strip_program(
    patch: AbstractSurfacePatch,
    pcurve: AbstractCurve,
    origin: float,
    /,
    *,
    chart: int | None = None,
    cylinder: bool = False,
) -> _IntervalProgram:
    """``g(u0 + s (u(t) - u0), v(t)) (u(t) - u0) v'(t)`` on the strip ``(t, s)``.

    Its integral over ``s`` in ``[0, 1]`` is ``G(u(t), v(t)) v'(t)`` with
    ``G(u, v) = int_u0^u g(r, v) dr``, so summing the p-curves' strip integrals
    gives the trimmed face integral by Green's theorem.
    """

    lower: list[float] = []
    upper: list[float] = []
    chart_evaluator = None
    chart_slice = None
    if chart is not None and isinstance(pcurve, IntersectionPCurve):
        chart_evaluator, chart_lower, chart_upper = (
            pcurve._chart_evaluator_with_source_bounds(chart)
        )
        chart_slice = slice(len(lower), len(lower) + chart_lower.size)
        lower.extend(chart_lower.tolist())
        upper.extend(chart_upper.tolist())

    def bind(values: Sequence[ExactScalar], /) -> slice:
        start = len(lower)
        for value in values:
            lo, hi = coefficient_interval(value, bits=128)
            lower.append(fraction_down(lo))
            upper.append(fraction_up(hi))
        return slice(start, len(lower))

    shifts = (0, 0)
    line = pcurve
    while isinstance(line, PeriodicPCurve):
        shifts = tuple(a + b for a, b in zip(shifts, line.period_shifts, strict=True))
        line = line.source_curve
    section_slice = None
    if isinstance(line, PlaneCylinderSectionPCurve):
        section_slice = slice(len(lower), len(lower) + 9)
        section_low, section_high = _section_source_bounds(line)
        lower.extend(section_low.tolist())
        upper.extend(section_high.tolist())
    line_slice = None
    root_slice = None
    if isinstance(line, LineCurve):
        line_slice = bind((*line.origin_exact, *line.direction_exact))
    elif isinstance(line, ExactLineCurve):
        line_slice = bind(
            (
                *line.first_exact,
                *(
                    last - first
                    for first, last in zip(line.first_exact, line.last_exact, strict=True)
                ),
            )
        )
    elif isinstance(line, ConicRootLinePCurve):
        root_slice = slice(len(lower), len(lower) + 3)
        domain = line.parameter_domain
        parameter = 0.0 if domain is None else domain[0]
        box = line.bounding_box(parameter, parameter)
        lower.append(float(box[0, 0]))
        upper.append(float(box[1, 0]))
        bind((line.height_origin_exact, line.height_direction_exact))
    elif section_slice is None:
        shifts = (0, 0)
    period_index = None
    if chart_evaluator is not None:
        period_index = len(lower)
        period_low, period_high = period_scalar_bounds(
            PeriodOffset(Fraction(0), Fraction(1))
        )
        lower.append(float(period_low))
        upper.append(float(period_high))
    shift_slice = None
    if any(shifts):
        shift_slice = slice(len(lower), len(lower) + 2)
        for shift in shifts:
            lo, hi = period_scalar_bounds(PeriodOffset(Fraction(0), Fraction(shift)))
            lower.append(float(lo))
            upper.append(float(hi))
    cylinder_slice = None
    if cylinder and isinstance(patch, CylinderPatch):
        cylinder_slice = bind(
            (
                *patch.origin_exact,
                *patch.first_axis_exact,
                *patch.second_axis_exact,
                *patch.axis_exact,
                patch.radius_exact,
            )
        )
    revolution_slice = None
    straight = (
        _straight_revolution(patch) if isinstance(patch, RevolutionSurface) else None
    )
    if cylinder and straight is not None and straight.slope == 0:
        revolution_slice = bind(
            (
                *straight.origin,
                *straight.radial,
                *straight.tangent,
                *straight.axis,
                straight.axial_rate,
                straight.scale_origin,
            )
        )

    def evaluator(bound: Array) -> Callable[[Array], Array]:
        if chart_evaluator is not None:
            assert period_index is not None and chart_slice is not None
            return lambda parameter: chart_evaluator(
                parameter, bound[period_index], bound[chart_slice]
            )
        if section_slice is not None:
            shift = bound[shift_slice] if shift_slice is not None else jnp.zeros((2,))
            return lambda parameter: (
                _section_point(parameter, bound[section_slice]) + shift
            )
        if root_slice is not None:
            root = bound[root_slice]
            shift = bound[shift_slice] if shift_slice is not None else jnp.zeros((2,))
            return lambda parameter: (
                jnp.stack((root[0], root[1] + parameter * root[2])) + shift
            )
        if line_slice is not None:
            line = bound[line_slice]
            shift = bound[shift_slice] if shift_slice is not None else jnp.zeros((2,))
            return lambda parameter: line[:2] + parameter * line[2:4] + shift
        return lambda parameter: _pcurve_point(pcurve, parameter)

    if cylinder:

        def cylinder_density(parameters: Array, bound: Array) -> Array:
            evaluate = evaluator(bound)
            if cylinder_slice is not None:
                values = bound[cylinder_slice]
                center, first_axis, second_axis, axis = (
                    values[:3],
                    values[3:6],
                    values[6:9],
                    values[9:12],
                )
                radius = values[12]
                determinant = jnp.dot(first_axis, jnp.cross(second_axis, axis))
                cosine = jnp.dot(center, jnp.cross(second_axis, axis))
                sine = -jnp.dot(center, jnp.cross(first_axis, axis))

                def primitive(angle: Array, /) -> Array:
                    return (
                        radius
                        / 3
                        * (
                            radius * determinant * angle
                            + cosine * jnp.sin(angle)
                            + sine * (1 - jnp.cos(angle))
                        )
                    )
            elif revolution_slice is not None:
                values = bound[revolution_slice]
                center, radial, tangent_axis, axis = (
                    values[:3],
                    values[3:6],
                    values[6:9],
                    values[9:12],
                )
                rate, scale = values[12], values[13]
                determinant = jnp.dot(radial, jnp.cross(tangent_axis, axis))
                cosine = jnp.dot(center, jnp.cross(tangent_axis, axis))
                sine = -jnp.dot(center, jnp.cross(radial, axis))

                def primitive(angle: Array, /) -> Array:
                    return (
                        rate
                        * scale
                        / 3
                        * (
                            scale * determinant * angle
                            + cosine * jnp.sin(angle)
                            + sine * (1 - jnp.cos(angle))
                        )
                    )
            else:

                def flux_at(angle: Array, /) -> Array:
                    return _surface_flux(patch, jnp.stack((angle, jnp.zeros_like(angle))))

                zero = parameters[0] * 0.0
                value = flux_at(zero)
                first = jax.grad(flux_at)(zero)
                second = jax.grad(jax.grad(flux_at))(zero)

                def primitive(angle: Array, /) -> Array:
                    return (
                        (value + second) * angle
                        + first * (1.0 - jnp.cos(angle))
                        - second * jnp.sin(angle)
                    )

            point = evaluate(parameters[0])
            tangent = jax.jacfwd(evaluate)(parameters[0])
            return (primitive(point[0]) - primitive(jnp.asarray(origin))) * tangent[1]

        dimension = 1
        integrand = cylinder_density
    else:
        dimension = 2

        def density(parameters: Array, bound: Array) -> Array:
            evaluate = evaluator(bound)
            point = evaluate(parameters[0])
            tangent = jax.jacfwd(evaluate)(parameters[0])
            width = point[0] - origin
            chart_point = jnp.stack((origin + parameters[1] * width, point[1]))
            return _surface_flux(patch, chart_point) * width * tangent[1]

        integrand = density

    fourth = cylinder and (
        isinstance(
            pcurve,
            (IntersectionPCurve, PlaneCylinderSectionPCurve, LineCurve, ExactLineCurve),
        )
        or line_slice is not None
        or section_slice is not None
    )
    if not lower:
        return _cell_program(
            _with_hessian(lambda x: integrand(x, x[:0]), fourth=fourth),
            dimension,
        )

    def jet_with_bounds(parameters: Array, /) -> Array:
        bound = parameters[dimension:]
        return _with_hessian(lambda x: integrand(x, bound), fourth=fourth)(
            parameters[:dimension]
        )

    return _BoundProgram(
        _cell_program(jet_with_bounds, dimension + len(lower)),
        np.asarray(lower),
        np.asarray(upper),
    )


def _pcurve_point_program(pcurve: AbstractCurve, /) -> PreparedIntervalFunction:
    return prepare_interval_function(
        lambda parameters: _pcurve_point(pcurve, parameters[0]), 1, batch_capacity=2
    )


def _prepare(label: str, build: Callable[[], _IntervalProgram], /) -> _IntervalProgram:
    try:
        return build()
    except IntervalExtensionError as error:
        raise _MeasureFailure(
            "unsupported", f"{label} has no sound interval extension: {error}"
        ) from error
    except (NotImplementedError, TypeError, ValueError) as error:
        raise _MeasureFailure(
            "unsupported", f"{label} cannot prepare an interval program: {error}"
        ) from error


def _arrays(interval: tuple[object, object], /) -> _Interval:
    """An interval result as a pair of float arrays."""
    return (
        np.asarray(interval[0], dtype=np.float64),
        np.asarray(interval[1], dtype=np.float64),
    )


def _evaluate(
    label: str, program: _IntervalProgram, lower: np.ndarray, upper: np.ndarray
) -> _Interval:
    try:
        low, high = program.evaluate(lower, upper)
    except IntervalExtensionError as error:
        raise _MeasureFailure(
            "unsupported", f"{label} has no sound interval extension: {error}"
        ) from error
    except (NotImplementedError, ValueError) as error:
        raise _MeasureFailure(
            "unresolved", f"{label} interval evaluation did not resolve: {error}"
        ) from error
    invalid = np.isnan(low) | np.isnan(high)
    return np.where(invalid, -np.inf, low), np.where(invalid, np.inf, high)


# ---------------------------------------------------------------------------
# Exact affine faces


def _fraction_vector(values: Sequence[object], /) -> _Vector | None:
    if len(values) != 3:
        return None
    first, second, third = values
    if not (
        isinstance(first, Fraction)
        and isinstance(second, Fraction)
        and isinstance(third, Fraction)
    ):
        return None
    return first, second, third


def _rational_point(values: object, /) -> _Vector:
    vector = _fraction_vector(rational_vector(np.asarray(values)))
    if vector is None:
        raise ValueError("An affine face frame must be a three-dimensional vector.")
    return vector


def _affine_frame(
    patch: AbstractSurfacePatch, /
) -> tuple[_Vector, _Vector, _Vector] | None:
    """Exact ``(origin, S_u, S_v)`` of an affine source face, if rational."""
    match patch:
        case PlanePatch():
            return (
                _rational_point(patch.origin),
                _rational_point(patch.first_axis),
                _rational_point(patch.second_axis),
            )
        case ExtrusionSurface(curve=LineCurve() as curve):
            origin = _fraction_vector(curve.origin_exact)
            first = _fraction_vector(curve.direction_exact)
        case ExtrusionSurface(curve=ExactLineCurve() as curve):
            origin = _fraction_vector(curve.first_exact)
            last = _fraction_vector(curve.last_exact)
            first = (
                None
                if origin is None or last is None
                else (
                    last[0] - origin[0],
                    last[1] - origin[1],
                    last[2] - origin[2],
                )
            )
        case _:
            return None
    if origin is None or first is None:
        return None
    return origin, first, _rational_point(patch.direction)


def _affine_flux(frame: tuple[_Vector, _Vector, _Vector], /) -> Fraction:
    """The constant integrand ``x . (S_u x S_v) / 3`` of an affine face."""
    origin, first, second = frame
    return rational_dot(origin, rational_cross(first, second)) / 3


def _fraction_uv(value: object, /) -> tuple[Fraction, Fraction] | None:
    if not isinstance(value, tuple) or len(value) != 2:
        return None
    first, second = value
    if not isinstance(first, Fraction) or not isinstance(second, Fraction):
        return None
    return first, second


def _traversal_range(geometry: BRepGeometry, coedge: int, /) -> tuple[float, float]:
    first, last = (
        float(value)
        for value in np.asarray(geometry.edge_ranges)[geometry.coedge_edges[coedge]]
    )
    return (first, last) if geometry.coedge_senses[coedge] > 0 else (last, first)


type _Point = tuple[Fraction, Fraction]


def _bernstein_products(*degrees: int) -> np.ndarray:
    """Exact ``int_0^1 prod_k B_{i_k}^{n_k}`` for every index tuple.

    A Bernstein product is ``prod C(n_k, i_k) / C(N, sum i_k) B^N_{sum i_k}``
    with ``N = sum n_k``, and every degree-``N`` Bernstein polynomial
    integrates to ``1 / (N + 1)``.
    """
    total = sum(degrees)
    table = np.empty(tuple(degree + 1 for degree in degrees), dtype=object)
    for index in np.ndindex(table.shape):
        numerator = 1
        for degree, value in zip(degrees, index, strict=True):
            numerator *= comb(degree, value)
        table[index] = Fraction(numerator, comb(total, sum(index)) * (total + 1))
    return table


def _polynomial_curve_spans(
    pcurve: AbstractCurve, first: float, last: float, /
) -> list[np.ndarray] | None:
    """Exact Bernstein nets of a polynomial B-spline p-curve covering ``[first, last]``."""
    if not isinstance(pcurve, BSplineCurve):
        return None
    nets = []
    reached = Fraction.from_float(first)
    for span in pcurve.prepare_spans().spans:
        lower, upper = span.parameter_bounds
        if upper <= reached or lower >= Fraction.from_float(last):
            continue
        coefficients = span.homogeneous_bernstein_coefficients
        if lower != reached or any(
            not all(isinstance(value, Fraction) for value in row) or row[2] != 1
            for row in coefficients
        ):
            return None
        nets.append(np.asarray([row[:2] for row in coefficients], dtype=object))
        reached = upper
    return nets if nets and reached == Fraction.from_float(last) else None


def _exact_coedge(
    geometry: BRepGeometry, coedge: int, /
) -> tuple[Fraction, _Point, _Point] | None:
    """Exact Green term ``int (u dv - v du) / 2`` and traversal endpoints.

    Line p-curves use their exact endpoints; polynomial B-spline p-curves
    integrate their Bernstein nets exactly (scale-free on each span).
    """
    pcurve = geometry.pcurves[coedge]
    first, last = _traversal_range(geometry, coedge)
    if isinstance(pcurve, (LineCurve, ExactLineCurve)):
        start, end = (
            _fraction_uv(pcurve.evaluate_exact(Fraction.from_float(value)))
            for value in (first, last)
        )
        if start is None or end is None:
            return None
        return (start[0] * end[1] - end[0] * start[1]) / 2, start, end
    nets = _polynomial_curve_spans(pcurve, min(first, last), max(first, last))
    if nets is None:
        return None
    term = Fraction(0)
    for net in nets:
        degree = net.shape[0] - 1
        if degree == 0:
            continue
        derivative = degree * (net[1:] - net[:-1])
        weights = _bernstein_products(degree, degree - 1)
        term += (
            np.sum(weights * np.multiply.outer(net[:, 0], derivative[:, 1]))
            - np.sum(weights * np.multiply.outer(net[:, 1], derivative[:, 0]))
        ) / 2
    start = (nets[0][0, 0], nets[0][0, 1])
    end = (nets[-1][-1, 0], nets[-1][-1, 1])
    if geometry.coedge_senses[coedge] < 0:
        return -term, end, start
    return term, start, end


def _exact_parameter_area(geometry: BRepGeometry, face: int, /) -> Fraction | None:
    """Exact Green area of the p-curve loops, closed by exact parameter chords."""
    area = Fraction(0)
    for loop in geometry.face_loops[face]:
        terms = [_exact_coedge(geometry, coedge) for coedge in loop]
        if any(term is None for term in terms):
            return None
        for index, term in enumerate(terms):
            assert term is not None
            following = terms[(index + 1) % len(terms)]
            assert following is not None
            end, start = term[2], following[1]
            area += term[0] + (end[0] * start[1] - start[0] * end[1]) / 2
    return area


_PERMUTATIONS = (
    ((0, 1, 2), 1),
    ((1, 2, 0), 1),
    ((2, 0, 1), 1),
    ((0, 2, 1), -1),
    ((2, 1, 0), -1),
    ((1, 0, 2), -1),
)


def _polynomial_span_flux(net: np.ndarray, /) -> Fraction:
    """Exact ``int det(X, X_s, X_t) / 3`` over one polynomial Bernstein span.

    The chart scale cancels: ``X_u du`` and ``X_v dv`` are ``X_s ds`` and
    ``X_t dt``. Each determinant term separates into exact one-dimensional
    Bernstein triple-product integrals contracted against the nets.
    """
    first, second = net.shape[0] - 1, net.shape[1] - 1
    if first == 0 or second == 0:
        return Fraction(0)
    along_first = first * (net[1:] - net[:-1])
    along_second = second * (net[:, 1:] - net[:, :-1])
    first_weights = _bernstein_products(first, first - 1, first)
    second_weights = _bernstein_products(second, second, second - 1)
    total = Fraction(0)
    for (a, b, c), sign in _PERMUTATIONS:
        # sum over (i, j, k, l, m, n) of W1[i, k, m] W2[j, l, n] P_a[i, j]
        # D1_b[k, l] D2_c[m, n], contracted one operand at a time.
        stage = np.tensordot(net[..., a], second_weights, axes=([1], [0]))
        stage = np.einsum("ilm,kl->ikm", stage, along_first[..., b])
        stage = np.einsum("ikm,qm->ikq", stage, along_second[..., c])
        total += sign * np.sum(stage * first_weights)
    return total / 3


def _exact_spline_flux(
    patch: AbstractSurfacePatch, bounds: np.ndarray, /
) -> Fraction | None:
    """Exact chart integral of a polynomial B-spline face whose trim is its box."""
    if not isinstance(patch, BSplineSurfacePatch):
        return None
    low = tuple(Fraction.from_float(float(value)) for value in bounds[0])
    high = tuple(Fraction.from_float(float(value)) for value in bounds[1])
    total = Fraction(0)
    covered = Fraction(0)
    for span in patch.prepare_spans().spans:
        box = span.parameter_bounds
        if any(
            box[axis][1] <= low[axis] or box[axis][0] >= high[axis] for axis in (0, 1)
        ):
            continue
        if any(box[axis][0] < low[axis] or box[axis][1] > high[axis] for axis in (0, 1)):
            return None
        coefficients = span.homogeneous_bernstein_coefficients
        if any(
            not all(isinstance(value, Fraction) for value in point) or point[3] != 1
            for row in coefficients
            for point in row
        ):
            return None
        net = np.asarray(
            [[point[:3] for point in row] for row in coefficients], dtype=object
        )
        total += _polynomial_span_flux(net)
        covered += (box[0][1] - box[0][0]) * (box[1][1] - box[1][0])
    if covered != (high[0] - low[0]) * (high[1] - low[1]):
        return None
    return total


def _spans_chart(geometry: BRepGeometry, face: int, bounds: np.ndarray, /) -> bool:
    """Whether exact line p-curves on the chart sides trim the whole chart box."""
    low = tuple(Fraction.from_float(float(value)) for value in bounds[0])
    high = tuple(Fraction.from_float(float(value)) for value in bounds[1])
    for loop in geometry.face_loops[face]:
        for coedge in loop:
            pcurve = geometry.pcurves[coedge]
            if not isinstance(pcurve, (LineCurve, ExactLineCurve)):
                return False
            points = tuple(
                _fraction_uv(pcurve.evaluate_exact(Fraction.from_float(value)))
                for value in _traversal_range(geometry, coedge)
            )
            if any(point is None for point in points) or not any(
                all(point is not None and point[axis] == side for point in points)
                for axis in range(2)
                for side in (low[axis], high[axis])
            ):
                return False
    area = _exact_parameter_area(geometry, face)
    return area is not None and area == (high[0] - low[0]) * (high[1] - low[1])


# ---------------------------------------------------------------------------
# Cell families


@dataclass(slots=True, eq=False)
class _Family:
    """Adaptive parameter cells of one integrand times an exact constant factor."""

    label: str
    program: _IntervalProgram
    factor: _Interval
    dimension: int
    domain: TrimDomain | None
    enclosures: TrimEnclosures | None
    normalization: np.ndarray | None
    boxes: np.ndarray
    status: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    axis: np.ndarray
    refinable: np.ndarray
    rounding: np.ndarray


# Per-cell arrays of a family, in the order `_enclose` returns them.
_CELL_FIELDS = ("status", "lower", "upper", "axis", "refinable", "rounding")


def _classify(family: _Family, boxes: np.ndarray, status: np.ndarray) -> np.ndarray:
    """Trim side of each undecided cell's open interior (decided sides persist)."""
    if family.domain is None:
        return np.ones_like(status)
    undecided = np.flatnonzero(status == 0)
    if not undecided.size:
        return status
    selected = boxes[undecided]
    if family.normalization is not None:
        origin, last = family.normalization
        span = interval_subtract((last, last), (origin, origin))
        lower = interval_divide(
            interval_subtract((selected[:, 0], selected[:, 0]), (origin, origin)), span
        )[0]
        upper = interval_divide(
            interval_subtract((selected[:, 1], selected[:, 1]), (origin, origin)), span
        )[1]
        selected = np.stack((lower, upper), axis=1)
    try:
        decision = family.domain.classify_boxes(
            selected, interior=True, enclosures=family.enclosures
        )
    except (NotImplementedError, ValueError) as error:
        raise _MeasureFailure(
            "unresolved", f"{family.label} trim classification failed: {error}"
        ) from error
    resolved = np.asarray(decision.resolved) & ~np.asarray(decision.boundary)
    inside = np.asarray(decision.inside)
    result = status.copy()
    result[undecided] = np.where(resolved, np.where(inside, 1, -1), 0)
    return result


def _splittable(boxes: np.ndarray) -> np.ndarray:
    middle = boxes[:, 0] * 0.5 + boxes[:, 1] * 0.5
    return (boxes[:, 0] < middle) & (middle < boxes[:, 1])


def _enclose(
    family: _Family, boxes: np.ndarray, status: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Certified cell integrals, split axes, refinability and rounding widths."""
    count, dimension = boxes.shape[0], family.dimension
    status = _classify(family, boxes, status)
    lower = np.zeros((count,), dtype=np.float64)
    upper = np.zeros((count,), dtype=np.float64)
    axis = np.full((count,), -1, dtype=np.int64)
    refinable = np.zeros((count,), dtype=np.bool_)
    rounding_width = np.zeros((count,), dtype=np.float64)
    active = np.flatnonzero(status >= 0)
    if not active.size:
        return status, lower, upper, axis, refinable, rounding_width
    cells = boxes[active]
    low_corner, high_corner = cells[:, 0], cells[:, 1]
    middle = low_corner * 0.5 + high_corner * 0.5
    # The binary64 midpoint is within half an ulp of the exact cell center.
    center_low = np.maximum(np.nextafter(middle, -np.inf), low_corner)
    center_high = np.minimum(np.nextafter(middle, np.inf), high_corner)
    values_low, values_high = _evaluate(
        family.label,
        family.program,
        np.concatenate((low_corner, center_low)),
        np.concatenate((high_corner, center_high)),
    )
    size = active.size
    box_values = (values_low[:size], values_high[:size])
    center_values = (values_low[size:, 0], values_high[size:, 0])
    widths = _arrays(
        interval_subtract((high_corner, high_corner), (low_corner, low_corner))
    )
    area = (widths[0][:, 0], widths[1][:, 0])
    for index in range(1, dimension):
        area = _arrays(
            interval_multiply(area, (widths[0][:, index], widths[1][:, index]))
        )
    value_range = (box_values[0][:, 0], box_values[1][:, 0])
    inside = status[active] == 1
    # A trim-crossing cell holds some unknown part of its area.
    zeroth = interval_multiply(
        (
            np.where(inside, value_range[0], np.minimum(value_range[0], 0.0)),
            np.where(inside, value_range[1], np.maximum(value_range[1], 0.0)),
        ),
        area,
    )
    # Inside a smooth cell, Taylor's theorem about the exact center c gives
    # f(c) A + sum_i f_ii(xi) A h_i^2 / 24 + sum_{i<j} (f_ij(xi) - m) s_i s_j,
    # because the linear and the constant-Hessian cross moments vanish.
    rounding = _arrays(interval_multiply(center_values, area))
    second = rounding
    terms = []
    for index in range(dimension):
        width = (widths[0][:, index], widths[1][:, index])
        diagonal = 1 + index * dimension + index
        term = _arrays(
            interval_divide(
                interval_multiply(
                    interval_multiply(
                        (box_values[0][:, diagonal], box_values[1][:, diagonal]),
                        area,
                    ),
                    interval_multiply(width, width),
                ),
                (np.float64(24.0), np.float64(24.0)),
            )
        )
        terms.append(term[1] - term[0])
        second = _arrays(interval_add(second, term))
    for first_axis in range(dimension):
        for second_axis in range(first_axis + 1, dimension):
            forward = 1 + first_axis * dimension + second_axis
            backward = 1 + second_axis * dimension + first_axis
            mixed_low = np.maximum(box_values[0][:, forward], box_values[0][:, backward])
            mixed_high = np.minimum(box_values[1][:, forward], box_values[1][:, backward])
            radius = _arrays(
                interval_multiply(
                    interval_subtract((mixed_high, mixed_high), (mixed_low, mixed_low)),
                    (np.float64(0.5), np.float64(0.5)),
                )
            )[1]
            moment = _arrays(
                interval_divide(
                    interval_multiply(
                        interval_multiply(area, (radius, radius)),
                        interval_multiply(
                            (widths[0][:, first_axis], widths[1][:, first_axis]),
                            (widths[0][:, second_axis], widths[1][:, second_axis]),
                        ),
                    ),
                    (np.float64(16.0), np.float64(16.0)),
                )
            )[1]
            second = _arrays(interval_add(second, (-moment, moment)))
            for index in (first_axis, second_axis):
                terms[index] = terms[index] + moment
    if dimension == 1 and box_values[0].shape[1] == 3:
        # Simpson's rule has remainder -f''''(xi) h**5 / 2880.  All
        # nodes, derivatives, widths and arithmetic are interval-enclosed;
        # intersect with the independent midpoint enclosure above.
        endpoint_low, endpoint_high = _evaluate(
            family.label,
            family.program,
            np.concatenate((low_corner, high_corner)),
            np.concatenate((low_corner, high_corner)),
        )
        endpoints = (
            endpoint_low[:size, 0],
            endpoint_high[:size, 0],
        )
        endpoints = _arrays(
            interval_add(endpoints, (endpoint_low[size:, 0], endpoint_high[size:, 0]))
        )
        weighted = _arrays(
            interval_add(
                endpoints,
                interval_multiply(center_values, (np.float64(4), np.float64(4))),
            )
        )
        simpson = _arrays(
            interval_divide(
                interval_multiply(weighted, area),
                (np.float64(6), np.float64(6)),
            )
        )
        width = (widths[0][:, 0], widths[1][:, 0])
        fifth = width
        for _ in range(4):
            fifth = _arrays(interval_multiply(fifth, width))
        remainder = _arrays(
            interval_divide(
                interval_multiply((box_values[0][:, 2], box_values[1][:, 2]), fifth),
                (np.float64(2880), np.float64(2880)),
            )
        )
        simpson = _arrays(interval_subtract(simpson, remainder))
        second = (np.maximum(second[0], simpson[0]), np.minimum(second[1], simpson[1]))
        rounding = _arrays(
            interval_divide(
                interval_multiply(weighted, area),
                (np.float64(6), np.float64(6)),
            )
        )
    enclosure = (
        np.where(inside, np.maximum(second[0], zeroth[0]), zeroth[0]),
        np.where(inside, np.minimum(second[1], zeroth[1]), zeroth[1]),
    )
    contribution = _arrays(interval_multiply(enclosure, family.factor))
    floor = _arrays(interval_multiply(rounding, family.factor))
    lower[active], upper[active] = contribution
    rounding_width[active] = np.where(inside, floor[1] - floor[0], 0.0)
    split = _splittable(cells)
    preferred = np.argmax(np.where(split, np.stack(terms, axis=1), -np.inf), axis=1)
    cell_width = contribution[1] - contribution[0]
    refinable[active] = np.where(
        inside,
        np.any(split, axis=1) & (cell_width > 2.0 * (floor[1] - floor[0])),
        np.any(split, axis=1),
    )
    axis[active] = np.where(inside, preferred, -1)
    return status, lower, upper, axis, refinable, rounding_width


def _new_family(
    label: str,
    program: _IntervalProgram,
    factor: _Interval,
    boxes: np.ndarray,
    *,
    domain: TrimDomain | None = None,
    normalization: np.ndarray | None = None,
) -> _Family:
    dimension = boxes.shape[2]
    family = _Family(
        label=label,
        program=program,
        factor=factor,
        dimension=dimension,
        domain=domain,
        enclosures=None if domain is None else domain.prepare_enclosures(),
        normalization=normalization,
        boxes=boxes,
        status=np.zeros((boxes.shape[0],), dtype=np.int8),
        lower=np.empty((0,)),
        upper=np.empty((0,)),
        axis=np.empty((0,), dtype=np.int64),
        refinable=np.empty((0,), dtype=np.bool_),
        rounding=np.empty((0,)),
    )
    for name, values in zip(
        _CELL_FIELDS, _enclose(family, boxes, family.status), strict=True
    ):
        setattr(family, name, values)
    return family


def _children(box: np.ndarray, axis: int, /) -> list[np.ndarray]:
    axes = [axis] if axis >= 0 else list(range(box.shape[1]))
    pieces = [box]
    for index in axes:
        first, last = box[0, index], box[1, index]
        middle = first * 0.5 + last * 0.5
        if not first < middle < last:
            continue
        split = []
        for piece in pieces:
            low, high = piece.copy(), piece.copy()
            low[1, index] = middle
            high[0, index] = middle
            split.extend((low, high))
        pieces = split
    return pieces


def _refine(family: _Family, rows: np.ndarray, /) -> int:
    """Split ``rows`` of one family; return the number of new cell enclosures."""
    boxes: list[np.ndarray] = []
    inherited: list[int] = []
    for row in rows.tolist():
        pieces = _children(family.boxes[row], int(family.axis[row]))
        if len(pieces) == 1:
            pieces = _children(family.boxes[row], -1)
        boxes.extend(pieces)
        inherited.extend([int(family.status[row])] * len(pieces))
    keep = np.ones((family.boxes.shape[0],), dtype=np.bool_)
    keep[rows] = False
    new_boxes = np.stack(boxes)
    computed = _enclose(family, new_boxes, np.asarray(inherited, dtype=np.int8))
    family.boxes = np.concatenate((family.boxes[keep], new_boxes))
    for name, values in zip(_CELL_FIELDS, computed, strict=True):
        setattr(family, name, np.concatenate((getattr(family, name)[keep], values)))
    return new_boxes.shape[0]


def _certified_sum(lower: np.ndarray, upper: np.ndarray, /) -> _ScalarInterval:
    if np.any(np.isinf(lower)):
        low = -np.inf
    else:
        total = fsum(lower.tolist())
        low = float(np.nextafter(total, -np.inf)) if lower.size > 1 else total
    if np.any(np.isinf(upper)):
        high = np.inf
    else:
        total = fsum(upper.tolist())
        high = float(np.nextafter(total, np.inf)) if upper.size > 1 else total
    return low, high


@dataclass(frozen=True, slots=True)
class _Integral:
    enclosure: _ScalarInterval
    evaluations: int
    exact: Fraction | None


def _integrate(
    families: list[_Family],
    fixed: tuple[np.ndarray, np.ndarray],
    exact: Fraction,
    *,
    relative_tolerance: float | None,
    label: str,
) -> _Integral:
    """Refine batched cells until the enclosure decides, or refuse with evidence.

    With ``relative_tolerance`` the decision is a positive enclosure whose
    half-width is at most that fraction of its lower bound; without it, a
    strict sign. Each round splits the widest refinable cells that together
    cover half of the refinable width and, once the enclosure is positive,
    every refinable cell wider than its equal share of the requested width.
    A cell is not refinable once its midpoint is no longer representable or
    its width is rounding-dominated.
    """
    exact_low, exact_high = fraction_down(exact), fraction_up(exact)
    evaluations = sum(family.boxes.shape[0] for family in families)
    if not families and not fixed[0].size:
        return _Integral((exact_low, exact_high), 0, exact)
    while True:
        lower = np.concatenate(
            (np.asarray((exact_low,)), fixed[0], *(family.lower for family in families))
        )
        upper = np.concatenate(
            (np.asarray((exact_high,)), fixed[1], *(family.upper for family in families))
        )
        total = _certified_sum(lower, upper)
        half_width = float(np.nextafter((total[1] - total[0]) * 0.5, np.inf))
        if relative_tolerance is None:
            if total[0] > 0.0 or total[1] < 0.0:
                return _Integral(total, evaluations, None)
            if total[0] == total[1] == 0.0:
                raise _MeasureFailure(
                    "unresolved", f"{label} has an exact zero volume", enclosure=total
                )
        else:
            if total[1] <= 0.0:
                raise _MeasureFailure(
                    "unresolved",
                    f"{label} volume enclosure [{total[0]!r}, {total[1]!r}] proves "
                    "nonpositive material volume",
                    enclosure=total,
                )
            if total[0] > 0.0 and half_width <= relative_tolerance * total[0]:
                return _Integral(total, evaluations, None)
        empty = [np.empty((0,))]
        widths = np.concatenate(
            [family.upper - family.lower for family in families] or empty
        )
        owners = np.concatenate(
            [np.full(family.lower.shape, index) for index, family in enumerate(families)]
            or empty
        )
        rows = np.concatenate(
            [np.arange(family.lower.size) for family in families] or empty
        ).astype(np.int64)
        refinable = np.concatenate(
            [family.refinable for family in families] or [np.empty((0,), np.bool_)]
        )
        candidates = np.flatnonzero(refinable & (widths > 0.0))
        refinable_width = fsum(widths[candidates].tolist())
        rounding = np.concatenate([family.rounding for family in families] or empty)
        # Splitting leaves the rounding widths of refinable cells: they enclose
        # the binary64 evaluation of the integrand at each center.
        floor = (
            fsum(np.concatenate((fixed[1] - fixed[0], widths)).tolist())
            - refinable_width
            + fsum(rounding[candidates].tolist())
        )
        if (
            not candidates.size
            or not refinable_width > 0.0
            or (
                relative_tolerance is not None
                and floor * 0.5 > relative_tolerance * total[1]
            )
        ):
            target = (
                "a strict sign"
                if relative_tolerance is None
                else f"relative half-width {relative_tolerance:g}"
            )
            raise _MeasureFailure(
                "unresolved",
                f"{label} enclosure [{total[0]!r}, {total[1]!r}] (half-width "
                f"{half_width:.3e}) cannot reach {target}: binary64 parameter and "
                f"rounding resolution leave width {floor:.3e} irreducible after "
                f"{evaluations} cell enclosures",
                enclosure=total,
            )
        order = candidates[np.argsort(-widths[candidates], kind="stable")]
        covered = np.cumsum(widths[order])
        count = int(np.searchsorted(covered, 0.5 * refinable_width)) + 1
        chosen = np.zeros(widths.shape, dtype=np.bool_)
        chosen[order[:count]] = True
        if relative_tolerance is not None and total[0] > 0.0:
            share = 2.0 * relative_tolerance * total[0] / widths.size
            chosen[candidates[widths[candidates] > share]] = True
        for index, family in enumerate(families):
            selected = rows[chosen & (owners == index)]
            if selected.size:
                evaluations += _refine(family, selected)


# ---------------------------------------------------------------------------
# Source faces and closure


@dataclass(frozen=True, slots=True)
class _VolumeSource:
    geometry: BRepGeometry
    patches: tuple[AbstractSurfacePatch, ...]
    parameter_bounds: np.ndarray
    trim_domains: tuple[TrimDomain | None, ...]


def _factor(value: Fraction, /) -> _Interval:
    return (
        np.asarray(fraction_down(value), dtype=np.float64),
        np.asarray(fraction_up(value), dtype=np.float64),
    )


def _norm_bound(lower: np.ndarray, upper: np.ndarray, /) -> float:
    magnitude = np.maximum(np.abs(np.asarray(lower)), np.abs(np.asarray(upper)))
    return float(np.sqrt(np.sum(magnitude * magnitude))) * _BOUND_HEADROOM


@dataclass(frozen=True, slots=True)
class _FacePlan:
    families: tuple[_Family, ...]
    exact: Fraction
    fixed: tuple[float, float]
    jacobian: tuple[float, float]


@dataclass(frozen=True, slots=True)
class _End:
    """A coedge endpoint in the chart: its outward box and exact value if known."""

    box: np.ndarray
    exact: _Point | None


def _exact_end(point: _Point, /) -> _End:
    return _End(
        np.asarray(
            (
                [fraction_down(value) for value in point],
                [fraction_up(value) for value in point],
            ),
            dtype=np.float64,
        ),
        point,
    )


type _LoopEnds = tuple[tuple[tuple[_End, _End], ...], ...]


def _loop_ends(source: _VolumeSource, face: int, /) -> _LoopEnds:
    """Start and end parameter points of each coedge in traversal order.

    Line and polynomial B-spline p-curves give exact endpoints; other
    p-curves are enclosed by interval evaluation.
    """
    geometry = source.geometry
    loops = []
    for loop in geometry.face_loops[face]:
        ends = []
        for coedge in loop:
            exact = _exact_coedge(geometry, coedge)
            if exact is not None:
                ends.append((_exact_end(exact[1]), _exact_end(exact[2])))
                continue
            pcurve = geometry.pcurves[coedge]
            if isinstance(
                pcurve,
                (
                    IntersectionPCurve,
                    TorusRootedCirclePCurve,
                    ConicRootLinePCurve,
                    PlaneCylinderSectionPCurve,
                    PeriodicPCurve,
                    AffinePCurve,
                ),
            ):
                first, last = _traversal_range(geometry, coedge)
                ends.append(
                    (
                        _End(pcurve.bounding_box(first, first), None),
                        _End(pcurve.bounding_box(last, last), None),
                    )
                )
                continue
            label = f"source coedge {coedge} p-curve"
            program = _prepare(label, lambda: _pcurve_point_program(pcurve))
            values = np.asarray(_traversal_range(geometry, coedge))[:, None]
            low, high = _evaluate(label, program, values, values)
            ends.append(
                (
                    _End(np.stack((low[0], high[0])), None),
                    _End(np.stack((low[1], high[1])), None),
                )
            )
        loops.append(tuple(ends))
    return tuple(loops)


def _chord_distance(end: _End, start: _End, /) -> float:
    """Upper bound of the parameter distance closing ``end`` to ``start``."""
    if end.exact is not None and start.exact is not None:
        return fraction_up(
            max(abs(end.exact[axis] - start.exact[axis]) for axis in (0, 1))
        )
    gap = np.maximum(np.abs(end.box[1] - start.box[0]), np.abs(start.box[1] - end.box[0]))
    largest = float(np.max(gap))
    return float(np.nextafter(largest, np.inf)) if largest else 0.0


def _green_curve_breaks(curve: AbstractCurve, /) -> np.ndarray | None:
    """Admit analytic intersection p-curves only one certified chart at a time."""
    if isinstance(curve, IntersectionPCurve):
        if orthogonal_cylinder_family(curve.curve.first, curve.curve.second) is None:
            return None
        breaks = np.arange(curve.curve.num_charts + 1, dtype=np.float64)
        if curve.reversed:
            breaks = curve.first + curve.last - breaks[::-1]
        return breaks
    return _curve_breaks(curve)


def _torus_periodic_face(
    source: _VolumeSource, face: int, sign: int, ends: _LoopEnds, /
) -> _FacePlan | None:
    """Green integral for latitude loops joined by exact native-turn meridians."""
    geometry, patch = source.geometry, source.patches[face]
    if not isinstance(patch, TorusPatch):
        return None
    loops = geometry.face_loops[face]
    if not any(
        isinstance(geometry.pcurves[coedge], TorusRootedCirclePCurve)
        for loop in loops
        for coedge in loop
    ):
        return None

    def refuse() -> None:
        raise _MeasureFailure(
            "unsupported", f"source face {face} torus trim is not proved once-wound"
        )

    if len(loops) != 1:
        refuse()
    loop = loops[0]
    seams: list[tuple[int, int]] = []
    positions: list[int] = []
    for position, coedge in enumerate(loop):
        pcurve = geometry.pcurves[coedge]
        if isinstance(pcurve, TorusRootedCirclePCurve):
            continue
        shifts = (0, 0)
        if isinstance(pcurve, PeriodicPCurve):
            shifts, pcurve = pcurve.period_shifts, pcurve.source_curve
        if (
            not isinstance(pcurve, LineCurve)
            or pcurve.origin_exact != (Fraction(0), Fraction(0))
            or pcurve.direction_exact != (Fraction(0), Fraction(1))
            or shifts[1] != 0
        ):
            refuse()
        seams.append((coedge, shifts[0]))
        positions.append(position)
    if len(seams) != 2:
        refuse()
    first_seam, second_seam = seams
    if (
        geometry.coedge_edges[first_seam[0]] != geometry.coedge_edges[second_seam[0]]
        or geometry.coedge_senses[first_seam[0]]
        != -geometry.coedge_senses[second_seam[0]]
        or abs(first_seam[1] - second_seam[1]) != 1
    ):
        refuse()

    # Between the two uses of the same seam each chain must traverse a
    # single rooted latitude monotonically. This rules out repeated or
    # backtracking windings even when the seam gauges differ by one turn.
    first_position, second_position = positions
    chains = (
        loop[first_position + 1 : second_position],
        (*loop[second_position + 1 :], *loop[:first_position]),
    )
    directions: list[int] = []
    for chain in chains:
        if not chain:
            refuse()
        first_curve = geometry.pcurves[chain[0]]
        assert isinstance(first_curve, TorusRootedCirclePCurve)
        first_root = first_curve.root_line.root
        root_key = (first_root.cosine_exact, first_root.sine_exact, first_root.turns)
        direction = first_curve.orientation * geometry.coedge_senses[chain[0]]
        for coedge in chain:
            curve = geometry.pcurves[coedge]
            assert isinstance(curve, TorusRootedCirclePCurve)
            root = curve.root_line.root
            if (
                root.cosine_exact,
                root.sine_exact,
                root.turns,
            ) != root_key or curve.orientation * geometry.coedge_senses[
                coedge
            ] != direction:
                refuse()
        directions.append(direction)
    if directions[0] != -directions[1]:
        refuse()

    coefficients = (
        *patch.center_exact,
        *patch.first_axis_exact,
        *patch.second_axis_exact,
        *patch.axis_exact,
        patch.major_radius_exact,
        patch.minor_radius_exact,
    )
    exact_bounds = tuple(coefficient_interval(value, bits=128) for value in coefficients)
    coefficient_low = np.asarray([fraction_down(item[0]) for item in exact_bounds])
    coefficient_high = np.asarray([fraction_up(item[1]) for item in exact_bounds])

    def norm(first: int, last: int, /) -> float:
        return _norm_bound(coefficient_low[first:last], coefficient_high[first:last])

    major = max(abs(coefficient_low[12]), abs(coefficient_high[12]))
    minor = max(abs(coefficient_low[13]), abs(coefficient_high[13]))
    radial = norm(3, 6) + norm(6, 9)
    axis = norm(9, 12)
    position = norm(0, 3) + (major + minor) * radial + minor * axis
    jacobian = (
        (major + minor) * radial * _BOUND_HEADROOM,
        minor * (radial + axis) * _BOUND_HEADROOM,
    )
    magnitude = position * jacobian[0] * jacobian[1] / 3.0 * _BOUND_HEADROOM

    def full_turn(parameters: Array, /) -> Array:
        center = parameters[:3]
        first_axis = parameters[3:6]
        second_axis = parameters[6:9]
        axis_vector = parameters[9:12]
        major_radius, minor_radius = parameters[12], parameters[13]
        first, last, period = parameters[14], parameters[15], parameters[16]
        normal = jnp.cross(first_axis, second_axis)
        center_det = jnp.dot(center, normal)
        axis_det = jnp.dot(axis_vector, normal)

        def primitive(angle: Array, /) -> Array:
            sine, cosine = jnp.sin(angle), jnp.cos(angle)
            return center_det * (
                -major_radius * cosine + minor_radius * sine**2 / 2
            ) + axis_det * (
                (major_radius**2 + minor_radius**2) * sine
                + major_radius * minor_radius * (3 * angle / 2 + jnp.sin(2 * angle) / 4)
            )

        return jnp.reshape(
            period * minor_radius * (primitive(last) - primitive(first)) / 3, (1,)
        )

    label = f"source face {face} torus Green primitive"
    program = _prepare(label, lambda: _cell_program(full_turn, 17))
    period_low, period_high = period_scalar_bounds(PeriodOffset(Fraction(0), Fraction(1)))
    fixed: _Interval = (np.zeros(()), np.zeros(()))
    for coedge, turns in seams:
        if not turns:
            continue
        first, last = (
            float(value)
            for value in np.asarray(geometry.edge_ranges)[geometry.coedge_edges[coedge]]
        )
        lower = np.concatenate((coefficient_low, [first, last, float(period_low)]))[None]
        upper = np.concatenate((coefficient_high, [first, last, float(period_high)]))[
            None
        ]
        low, high = _evaluate(label, program, lower, upper)
        factor = _factor(Fraction(sign * geometry.coedge_senses[coedge] * turns))
        fixed = _arrays(
            interval_add(fixed, interval_multiply((low[0, 0], high[0, 0]), factor))
        )

    chords = 0.0
    longitude = 0.0
    for loop_ends in ends:
        for index, (start, end) in enumerate(loop_ends):
            chords += _chord_distance(end, loop_ends[(index + 1) % len(loop_ends)][0])
            longitude = max(
                longitude,
                *(abs(float(value)) for value in start.box[:, 0]),
                *(abs(float(value)) for value in end.box[:, 0]),
            )
    bound = magnitude * longitude * chords * _BOUND_HEADROOM if chords else 0.0
    fixed = _arrays(interval_add(fixed, (np.asarray(-bound), np.asarray(bound))))
    return _FacePlan((), Fraction(0), (float(fixed[0]), float(fixed[1])), jacobian)


def _plan_face(
    source: _VolumeSource, face: int, sign: int, ends: _LoopEnds, /
) -> _FacePlan:
    geometry, patch = source.geometry, source.patches[face]
    label = f"source face {face}"
    frame = _affine_frame(patch)
    if frame is not None:
        flux = _affine_flux(frame) * sign
        jacobian = (
            _norm_bound(np.asarray(frame[1], dtype=np.float64), np.zeros(3)),
            _norm_bound(np.asarray(frame[2], dtype=np.float64), np.zeros(3)),
        )
        area = _exact_parameter_area(geometry, face)
        if area is not None:
            return _FacePlan((), flux * area, (0.0, 0.0), jacobian)
    domain = source.trim_domains[face]
    if domain is None or domain.numeric_only:
        raise _MeasureFailure(
            "unsupported", f"{label} has no source-authoritative trim domain"
        )
    torus_plan = _torus_periodic_face(source, face, sign, ends)
    if torus_plan is not None:
        return torus_plan
    if frame is not None:
        # Affine face: the constant flux times the Green area of its p-curve
        # loops, each loop closed by straight parameter chords.
        families = []
        chord = (np.zeros(()), np.zeros(()))
        for loop, loop_ends in zip(geometry.face_loops[face], ends, strict=True):
            for position, coedge in enumerate(loop):
                pcurve = geometry.pcurves[coedge]
                breaks = _curve_breaks(pcurve)
                if breaks is None:
                    raise _MeasureFailure(
                        "unsupported",
                        f"{label} p-curve {type(pcurve).__name__} declares no smooth pieces",
                    )
                first, last = (
                    float(value)
                    for value in np.asarray(geometry.edge_ranges)[
                        geometry.coedge_edges[coedge]
                    ]
                )
                coedge_label = f"{label} coedge {coedge}"
                families.append(
                    _new_family(
                        coedge_label,
                        _prepare(coedge_label, lambda: _parameter_area_program(pcurve)),
                        _factor(flux * geometry.coedge_senses[coedge]),
                        _break_grid(np.asarray((first,)), np.asarray((last,)), (breaks,)),
                    )
                )
                end = loop_ends[position][1].box
                start = loop_ends[(position + 1) % len(loop)][0].box
                cross = interval_subtract(
                    interval_multiply((end[0, 0], end[1, 0]), (start[0, 1], start[1, 1])),
                    interval_multiply((start[0, 0], start[1, 0]), (end[0, 1], end[1, 1])),
                )
                chord = interval_add(
                    chord,
                    interval_multiply(cross, (np.float64(0.5), np.float64(0.5))),
                )
        fixed = interval_multiply(chord, _factor(flux))
        return _FacePlan(
            tuple(families), Fraction(0), (float(fixed[0]), float(fixed[1])), jacobian
        )
    breaks = _surface_breaks(patch)
    if breaks is None:
        raise _MeasureFailure(
            "unsupported",
            f"{label} surface {type(patch).__name__} declares no smooth pieces",
        )
    bounds = np.asarray(source.parameter_bounds[face], dtype=np.float64)
    spanned = _spans_chart(geometry, face, bounds)
    exact_flux = _exact_spline_flux(patch, bounds) if spanned else None
    if exact_flux is not None:
        # The source's exact span derivative bounds bound the chord images.
        try:
            derivative_low, derivative_high = patch.derivative_bounds(bounds)
        except (NotImplementedError, ValueError) as error:
            raise _MeasureFailure(
                "unsupported", f"{label} has no chart derivative bound: {error}"
            ) from error
        jacobian = (
            _norm_bound(derivative_low[:, 0], derivative_high[:, 0]),
            _norm_bound(derivative_low[:, 1], derivative_high[:, 1]),
        )
        return _FacePlan((), exact_flux * sign, (0.0, 0.0), jacobian)
    frame_low, frame_high = _evaluate(
        label,
        _prepare(label, lambda: _surface_frame_program(patch)),
        bounds[0][None],
        bounds[1][None],
    )
    jacobian = (
        _norm_bound(frame_low[0, 1:4], frame_high[0, 1:4]),
        _norm_bound(frame_low[0, 4:], frame_high[0, 4:]),
    )
    pcurve_breaks = tuple(
        _green_curve_breaks(geometry.pcurves[coedge])
        for loop in geometry.face_loops[face]
        for coedge in loop
    )
    smooth_surface = not any(
        _interior_breaks(values, bounds[0, axis], bounds[1, axis]).size
        for axis, values in enumerate(breaks)
    )
    if (
        not spanned
        and smooth_surface
        and all(values is not None for values in pcurve_breaks)
    ):
        # Green's theorem in the chart: the face integral is the sum over its
        # p-curves of a smooth integrand on parameter strips (tau, sigma) that
        # sweep from the chart's lower u edge, so no cell crosses the trim.
        families = []
        straight = _straight_revolution(patch)
        cylinder = straight is not None and straight.slope == 0

        def grid(first: float, last: float, breaks: np.ndarray, /) -> np.ndarray:
            if cylinder:
                return _break_grid(np.asarray((first,)), np.asarray((last,)), (breaks,))
            return _break_grid(
                np.asarray((first, 0.0)),
                np.asarray((last, 1.0)),
                (breaks, _NO_BREAKS),
            )

        magnitude = max(abs(float(frame_low[0, 0])), abs(float(frame_high[0, 0])))
        sweep = float(bounds[1, 0] - bounds[0, 0])
        chords = 0.0
        for loop, loop_ends in zip(geometry.face_loops[face], ends, strict=True):
            for position, coedge in enumerate(loop):
                pcurve = geometry.pcurves[coedge]
                first, last = (
                    float(value)
                    for value in np.asarray(geometry.edge_ranges)[
                        geometry.coedge_edges[coedge]
                    ]
                )
                piece_breaks = _green_curve_breaks(pcurve)
                assert piece_breaks is not None
                coedge_label = f"{label} coedge {coedge}"
                if isinstance(pcurve, TorusRootedCirclePCurve):
                    # The root is a constant latitude, so dv is identically
                    # zero even when the longitude carries an exact turn.
                    pass
                elif isinstance(pcurve, IntersectionPCurve):
                    cuts = np.concatenate(
                        ([first], _interior_breaks(piece_breaks, first, last), [last])
                    )
                    for start, stop in zip(cuts[:-1], cuts[1:], strict=True):
                        midpoint = float(start + 0.5 * (stop - start))
                        carrier = (
                            pcurve.first + pcurve.last - midpoint
                            if pcurve.reversed
                            else midpoint
                        )
                        chart = min(int(carrier), pcurve.curve.num_charts - 1)
                        families.append(
                            _new_family(
                                coedge_label,
                                _prepare(
                                    coedge_label,
                                    lambda: _green_strip_program(
                                        patch,
                                        pcurve,
                                        float(bounds[0, 0]),
                                        chart=chart,
                                        cylinder=cylinder,
                                    ),
                                ),
                                _factor(Fraction(sign * geometry.coedge_senses[coedge])),
                                grid(float(start), float(stop), _NO_BREAKS),
                            )
                        )
                else:
                    families.append(
                        _new_family(
                            coedge_label,
                            _prepare(
                                coedge_label,
                                lambda: _green_strip_program(
                                    patch, pcurve, float(bounds[0, 0]), cylinder=cylinder
                                ),
                            ),
                            _factor(Fraction(sign * geometry.coedge_senses[coedge])),
                            grid(first, last, piece_breaks),
                        )
                    )
                chords += _chord_distance(
                    loop_ends[position][1], loop_ends[(position + 1) % len(loop)][0]
                )
        # A closing chord's |G dv| is at most max|g| * (chart u width) * length.
        bound = magnitude * sweep * chords * _BOUND_HEADROOM if chords else 0.0
        return _FacePlan(tuple(families), Fraction(0), (-bound, bound), jacobian)
    family = _new_family(
        label,
        _prepare(label, lambda: _surface_flux_program(patch)),
        _factor(Fraction(sign)),
        _break_grid(bounds[0], bounds[1], breaks),
        domain=None if spanned else domain,
        normalization=None if domain.curve_loops else bounds,
    )
    return _FacePlan((family,), Fraction(0), (0.0, 0.0), jacobian)


def _closure_bound(
    source: _VolumeSource,
    faces: Sequence[int],
    ends: dict[int, _LoopEnds],
    jacobians: dict[int, tuple[float, float]],
    /,
) -> float:
    """Flux bound of the strips and vertex fans that close certified source gaps.

    A face meets its edge curve within the certified edge-face gap ``delta``;
    the ruled strip between them has flux at most
    ``(R + delta) / 3 * delta * L``. At a vertex, every hole-boundary point lies
    within ``rho`` of the vertex (vertex gaps, face gaps and the image of the
    parameter chord), so a fan of area ``2 n rho^2`` closes the hole.
    """
    geometry = source.geometry
    face_gaps = geometry.edge_face_gap_bounds
    vertex_gaps = geometry.edge_vertex_gap_bounds
    ranges = np.asarray(geometry.edge_ranges, dtype=np.float64)
    points = np.asarray(geometry.vertex_points, dtype=np.float64)
    total = 0.0
    radii: dict[int, list[float]] = {}
    for face in faces:
        jacobian = jacobians[face]
        for loop, loop_ends in zip(geometry.face_loops[face], ends[face], strict=True):
            for position, coedge in enumerate(loop):
                edge = geometry.coedge_edges[coedge]
                delta = 0.0 if face_gaps is None else float(face_gaps[coedge])
                curve_index = geometry.edge_curves[edge]
                if delta > 0.0 and curve_index >= 0:
                    curve = geometry.curves[curve_index]
                    pcurve = geometry.pcurves[coedge]
                    first, last = (float(value) for value in ranges[edge])
                    try:
                        box = np.asarray(curve.bounding_box(first, last))
                        tangent = curve.derivative_bounds(first, last)
                        chart = pcurve.derivative_bounds(first, last)
                    except (NotImplementedError, ValueError) as error:
                        raise _MeasureFailure(
                            "unsupported",
                            f"source coedge {coedge} gap strip has no derivative bound: {error}",
                        ) from error
                    speed = max(
                        _norm_bound(*tangent),
                        jacobian[0] * _norm_bound(chart[0][:1], chart[1][:1])
                        + jacobian[1] * _norm_bound(chart[0][1:], chart[1][1:]),
                    )
                    radius = _norm_bound(box[0], box[1])
                    total += (
                        (radius + delta) / 3.0 * delta * (last - first) * speed
                    ) * _BOUND_HEADROOM
                following = loop[(position + 1) % len(loop)]
                chord = _chord_distance(
                    loop_ends[position][1], loop_ends[(position + 1) % len(loop)][0]
                )
                next_delta = 0.0 if face_gaps is None else float(face_gaps[following])
                senses = geometry.coedge_senses
                end_vertex = geometry.edge_vertices[edge][1 if senses[coedge] > 0 else 0]
                next_edge = geometry.coedge_edges[following]
                end_gap = 0.0
                start_gap = 0.0
                if vertex_gaps is not None:
                    end_gap = float(vertex_gaps[edge][1 if senses[coedge] > 0 else 0])
                    start_gap = float(
                        vertex_gaps[next_edge][0 if senses[following] > 0 else 1]
                    )
                rho = (
                    end_gap
                    + start_gap
                    + delta
                    + next_delta
                    + (jacobian[0] + jacobian[1]) * chord
                ) * _BOUND_HEADROOM
                if rho > 0.0:
                    radii.setdefault(int(end_vertex), []).append(rho)
    for vertex, values in radii.items():
        rho = max(values)
        reach = float(np.linalg.norm(points[vertex])) + rho
        total += reach / 3.0 * 2.0 * len(values) * rho * rho * _BOUND_HEADROOM
    if not isfinite(total):
        raise _MeasureFailure("unresolved", "source gap closure has no finite flux bound")
    return float(np.nextafter(total, np.inf)) if total else 0.0


def _volume_integral(
    source: _VolumeSource,
    faces: Sequence[int],
    signs: Sequence[int],
    *,
    relative_tolerance: float | None,
    label: str,
) -> _Integral:
    families: list[_Family] = []
    exact = Fraction(0)
    fixed_lower: list[float] = []
    fixed_upper: list[float] = []
    jacobians: dict[int, tuple[float, float]] = {}
    ends: dict[int, _LoopEnds] = {}
    for face, sign in zip(faces, signs, strict=True):
        if sign not in (-1, 1):
            raise ValueError("Source face volume signs must be -1 or 1.")
        ends[face] = _loop_ends(source, face)
        plan = _plan_face(source, face, sign, ends[face])
        families.extend(plan.families)
        exact += plan.exact
        if plan.fixed != (0.0, 0.0):
            fixed_lower.append(plan.fixed[0])
            fixed_upper.append(plan.fixed[1])
        jacobians[face] = plan.jacobian
    closure = _closure_bound(source, faces, ends, jacobians)
    if closure:
        fixed_lower.append(-closure)
        fixed_upper.append(closure)
    integral = _integrate(
        families,
        (np.asarray(fixed_lower), np.asarray(fixed_upper)),
        exact,
        relative_tolerance=relative_tolerance,
        label=label,
    )
    if integral.exact is not None and relative_tolerance is not None and exact <= 0:
        raise _MeasureFailure(
            "unresolved",
            f"{label} exact volume {exact} is not a positive material volume",
            enclosure=integral.enclosure,
        )
    if integral.exact is not None:
        return integral
    # Each face integral and closure bound is one certified evaluation beyond
    # the cell enclosures, so a non-exact enclosure always reports evidence.
    return _Integral(
        integral.enclosure,
        integral.evaluations + len(faces) + (1 if closure else 0),
        None,
    )


def _closed_solid(model: BRepModel, solid: int, /) -> str | None:
    geometry = model.geometry
    shell_indices = geometry.solid_shells[solid]
    if not shell_indices:
        return f"solid {solid} has no source boundary shells"
    for shell in shell_indices:
        counts = [0] * geometry.topology.num_edges
        balance = [0] * geometry.topology.num_edges
        for face, shell_sign in zip(
            geometry.shell_faces[shell],
            geometry.shell_orientations[shell],
            strict=True,
        ):
            face_sign = int(model.orientation[face]) * shell_sign
            for loop in geometry.face_loops[face]:
                for coedge in loop:
                    edge = geometry.coedge_edges[coedge]
                    if geometry.edge_curves[edge] < 0:
                        continue
                    counts[edge] += 1
                    balance[edge] += geometry.coedge_senses[coedge] * face_sign
        for edge, curve_index in enumerate(geometry.edge_curves):
            if (
                counts[edge]
                and curve_index >= 0
                and (counts[edge] != 2 or balance[edge] != 0)
            ):
                return f"solid {solid} shell {shell} is not closed at source edge {edge}"
    return None


def _model_source(model: BRepModel, /) -> _VolumeSource:
    return _VolumeSource(
        model.geometry,
        model.patches,
        np.asarray(model.parameter_bounds, dtype=np.float64),
        model.trim_domains,
    )


def _solid_volume(
    model: BRepModel, solid: int, relative_tolerance: float, /
) -> _Integral:
    reason = _closed_solid(model, solid)
    if reason is not None:
        raise _MeasureFailure("unresolved", reason)
    faces = model.topology.solid_faces[solid]
    if not faces:
        raise _MeasureFailure(
            "unresolved", f"solid {solid} has no source faces to integrate"
        )
    signs = tuple(
        int(sign) * int(model.orientation[face])
        for face, sign in zip(faces, model.topology.solid_face_orientations[solid])
    )
    return _volume_integral(
        _model_source(model),
        faces,
        signs,
        relative_tolerance=relative_tolerance,
        label=f"solid {solid}",
    )


def _placement_volume_factor(occurrence: BRepAssemblyOccurrence, /) -> Fraction:
    matrix = occurrence.placement.matrix
    rows = tuple(tuple(matrix[row][column] for column in range(3)) for row in range(3))
    return abs(rational_determinant(*rows))


@checked
def brep_mass_properties(
    model: BRepModel,
    /,
    *,
    relative_tolerance: float,
) -> BRepMassProperties:
    """Enclose the positive material volume of every source solid occurrence.

    The volume is the flux of ``x / 3`` through the solid's source faces.
    Affine faces with line p-curves integrate exactly in rationals; other affine
    faces integrate their constant flux over the Green area of their p-curve
    loops; curved faces integrate over source parameter cells split at every
    smoothness break and classified against the source trim. Inside a smooth
    cell the second-order midpoint/Hessian enclosure applies; a trim-crossing
    cell holds the zero-extended range of its integrand. Certified edge-face and
    vertex gaps add the flux of their closing strips and fans. Batched rounds
    refine the widest cells until the half-width is at most
    ``relative_tolerance`` times the positive lower bound. Unsupported carriers
    or trims, and enclosures that binary64 parameter or rounding resolution
    stops short of the request, return an explicit per-occurrence status with
    the reached enclosure as evidence; no tessellation or sample-derived
    estimate is published.
    """
    if not x64_enabled():
        raise ValueError(
            "BRepMassProperties.precision must retain binary64 enclosures; got JAX x64 disabled."
        )
    tolerance = _relative_tolerance(relative_tolerance)
    if model.topology.num_solids == 0:
        raise ValueError("BRepMassProperties requires at least one closed source solid.")
    occurrences = tuple(model.geometry.iter_occurrences())
    if not occurrences:
        raise ValueError("BRepMassProperties requires at least one solid occurrence.")

    reached = np.tile(
        np.asarray((-np.inf, np.inf), dtype=np.float64), (len(occurrences), 1)
    )
    statuses: list[BRepMassMeasureStatus] = []
    reasons: list[str] = []
    exact_flags: list[bool] = []
    evaluations: list[int] = []
    solids: dict[int, _Integral | _MeasureFailure] = {}
    indices = tuple(int(occurrence.solid_index) for occurrence in occurrences)
    paths = tuple(occurrence.path for occurrence in occurrences)
    ids = tuple(
        BRepEntityId(
            model.source_id,
            model.source_revision,
            "solid",
            index,
            occurrence_path=path,
        )
        for index, path in zip(indices, paths, strict=True)
    )
    for row, occurrence in enumerate(occurrences):
        solid = indices[row]
        if solid not in solids:
            try:
                solids[solid] = _solid_volume(model, solid, tolerance)
            except _MeasureFailure as error:
                solids[solid] = error
            except (FloatingPointError, OverflowError) as error:
                solids[solid] = _MeasureFailure(
                    "unresolved",
                    f"source integration could not be represented: {error}",
                )
        local = solids[solid]
        if isinstance(local, _MeasureFailure):
            statuses.append(local.status)
            reasons.append(local.reason)
            exact_flags.append(False)
            evaluations.append(0)
            reached[row] = local.enclosure
            continue
        scale = _placement_volume_factor(occurrence)
        if local.exact is not None:
            volume = local.exact * scale
            placed = (fraction_down(volume), fraction_up(volume))
        else:
            placed = interval_multiply(
                (np.float64(local.enclosure[0]), np.float64(local.enclosure[1])),
                (np.float64(fraction_down(scale)), np.float64(fraction_up(scale))),
            )
        placed = (float(placed[0]), float(placed[1]))
        half_width = float(np.nextafter((placed[1] - placed[0]) * 0.5, np.inf))
        reached[row] = placed
        evaluations.append(local.evaluations)
        if placed[0] > 0.0 and (
            local.exact is not None or half_width <= tolerance * placed[0]
        ):
            statuses.append("certified")
            reasons.append("")
            exact_flags.append(local.exact is not None)
        else:
            statuses.append("unresolved")
            reasons.append(
                f"placed solid volume enclosure [{placed[0]!r}, {placed[1]!r}] does "
                f"not meet relative half-width {tolerance:g} with positive volume"
            )
            exact_flags.append(False)

    return BRepMassProperties(
        model_id=model.model_id,
        source_id=model.source_id,
        relative_tolerance=tolerance,
        solid_indices=indices,
        occurrence_paths=paths,
        solid_ids=ids,
        status=tuple(statuses),
        reasons=tuple(reasons),
        exact=tuple(exact_flags),
        evaluations=tuple(evaluations),
        reached_enclosures=reached,
    )


def source_shell_volume_sign(
    geometry: BRepGeometry,
    patches: tuple[AbstractSurfacePatch, ...],
    parameter_bounds: np.ndarray,
    trim_domains: tuple[TrimDomain | None, ...],
    signs: np.ndarray,
    faces: Sequence[int],
    /,
) -> int:
    """Certify the oriented volume sign of one closed source shell.

    ``signs`` holds each face's orientation relative to its patch normal. The
    shell's flux of ``x / 3`` is refined until its enclosure excludes zero;
    otherwise `BRepMeasureRefusal` reports the reached enclosure.
    """
    source = _VolumeSource(
        geometry,
        tuple(patches),
        np.asarray(parameter_bounds, dtype=np.float64),
        tuple(trim_domains),
    )
    face_signs = tuple(1 if float(signs[face]) > 0.0 else -1 for face in faces)
    try:
        integral = _volume_integral(
            source, faces, face_signs, relative_tolerance=None, label="shell"
        )
    except _MeasureFailure as error:
        raise BRepMeasureRefusal(
            error.status, error.reason, enclosure=error.enclosure
        ) from error
    if integral.exact is not None:
        if integral.exact == 0:
            raise BRepMeasureRefusal(
                "unresolved", "shell has an exact zero volume", enclosure=(0.0, 0.0)
            )
        return 1 if integral.exact > 0 else -1
    return 1 if integral.enclosure[0] > 0.0 else -1


def brep_shell_volume_sign(model: BRepModel, shell: int, /) -> int:
    """Return a certified oriented-volume sign for one exact source shell.

    This is intentionally distinct from ``brep_mass_properties``: it determines
    one shell's sign without requiring a positive solid-volume enclosure.
    Unsupported carriers, trims, or interval extensions raise
    `BRepMeasureRefusal` with the reached enclosure.
    """
    if not isinstance(model, BRepModel):
        raise TypeError("model must be a BRepModel.")
    if isinstance(shell, bool) or not isinstance(shell, Integral):
        raise TypeError("shell must be an integer source shell index.")
    geometry = model.geometry
    shell_index = int(shell)
    if shell_index < 0 or shell_index >= len(geometry.shell_faces):
        raise ValueError("shell must select an existing source shell.")
    faces = geometry.shell_faces[shell_index]
    if not faces:
        raise ValueError("A signed shell volume requires at least one source face.")
    signs = [float(value) for value in np.asarray(model.orientation)]
    for face, shell_orientation in zip(
        faces, geometry.shell_orientations[shell_index], strict=True
    ):
        signs[face] = float(model.orientation[face]) * shell_orientation
    return source_shell_volume_sign(
        geometry,
        model.patches,
        np.asarray(model.parameter_bounds, dtype=np.float64),
        model.trim_domains,
        np.asarray(signs, dtype=np.float64),
        faces,
    )


__all__ = [
    "BRepMassMeasureStatus",
    "BRepMassProperties",
    "BRepMeasureRefusal",
    "brep_mass_properties",
    "brep_shell_volume_sign",
    "source_shell_volume_sign",
]
