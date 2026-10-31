#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable, Sequence
from fractions import Fraction
from typing import Generic, TYPE_CHECKING, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.tree_util import PyTreeDef
from jax.typing import ArrayLike

from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ...typing import checked
from .._atlas import BoundaryAtlas, TrimDomain
from .._capabilities import GeometryCapability
from .._certificate import (
    DistanceSemantics,
    FieldCertificate,
    FieldRegularity,
    SignReliability,
    ZeroSetAccuracy,
)
from .._closest_point import represented_mesh_closest_point, triangle_query_evidence
from .._contracts import (
    ClosestPointResult,
    GeometryKernel,
    GeometryKind,
    GeometrySource,
)
from .._interval_enclosure import Interval
from .._sampling import (
    bounded_rejection_sample,
    RejectionSamplingPlan,
    sample_boundary_atlas,
    SamplingResult,
)
from ..design._schema import (
    _ParameterCollector,
    DesignState,
    ParameterBinding,
    ParameterId,
)
from ..simplicial import MeshQueryResult, TriangleBVH, TriangleMesh
from ..simplicial._mesh import _closest_points_on_triangles
from ._model import BRepBoundaryMap, BRepGeometry, BRepModel
from ._patches import AbstractCurve, AbstractSurfacePatch


if TYPE_CHECKING:
    from ..._interpolation import (
        ASTSTransferPlan,
        LocalExtractedBernsteinRealization,
        TMesh2D,
    )
    from ._intersection_curve import IntersectionCurve, IntersectionCurvePoint
    from ._tspline import TSplineSurfacePatch

type _SurfaceDomain = tuple[tuple[float, float], tuple[float, float]] | None
_BoundValue = TypeVar("_BoundValue")


class _PatchBinding(StrictModule, Generic[_BoundValue]):
    bindings: tuple[ParameterBinding | None, ...] = eqx.field(static=True)
    tree_definition: PyTreeDef = eqx.field(static=True)
    defaults: tuple[Array, ...]

    def __init__(
        self,
        source_value: _BoundValue,
        bindings: Iterable[ParameterBinding | None],
        tree_definition: PyTreeDef,
        defaults: Iterable[Array],
    ) -> None:
        if jax.tree_util.tree_structure(source_value) != tree_definition:
            raise ValueError(
                "A BRep parameter binding must retain its declared PyTree structure."
            )
        self.bindings = tuple(bindings)
        self.tree_definition = tree_definition
        self.defaults = tuple(defaults)

    def realize(self, state: DesignState, /) -> _BoundValue:
        values = tuple(
            default if binding is None else binding.read(state)
            for binding, default in zip(self.bindings, self.defaults, strict=True)
        )
        return jax.tree_util.tree_unflatten(self.tree_definition, values)


_SurfaceCarrier = TypeVar("_SurfaceCarrier", bound=AbstractSurfacePatch)
_CurveCarrier = TypeVar("_CurveCarrier", bound=AbstractCurve)


def _require_surface_realization(
    value: AbstractSurfacePatch, carrier_type: type[_SurfaceCarrier], /
) -> _SurfaceCarrier:
    if not isinstance(value, carrier_type):
        raise TypeError(
            "A dynamic surface realization must preserve its source carrier type."
        )
    return value


def _require_curve_realization(
    value: AbstractCurve, carrier_type: type[_CurveCarrier], /
) -> _CurveCarrier:
    if not isinstance(value, carrier_type):
        raise TypeError(
            "A dynamic curve realization must preserve its source carrier type."
        )
    return value


class _RealizedTrimEdge(AbstractCurve):
    """Dynamic 3D edge realization from one surface and its native p-curve."""

    patch: AbstractSurfacePatch
    pcurve: AbstractCurve

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def period(self) -> float | None:
        return self.pcurve.period

    def native_period_symbol(self, /) -> str | None:
        return self.pcurve.native_period_symbol()

    @property
    def parameter_domain(self) -> tuple[float, float] | None:
        return self.pcurve.parameter_domain

    def evaluate(self, parameters: Array, /) -> Array:
        return self.patch.evaluate(self.pcurve.evaluate(parameters))


class _RealizedIsoparametricCurve(AbstractCurve):
    """Numeric-only fixed-coordinate curve on a realized source patch."""

    surface: AbstractSurfacePatch
    fixed_axis: int = eqx.field(static=True)
    fixed_value: Array
    parameter_range: tuple[float, float] | None = eqx.field(static=True)

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def period(self) -> float | None:
        return self.surface.periods[1 - self.fixed_axis]

    def native_period_symbol(self, /) -> str | None:
        return self.surface.native_period_symbols()[1 - self.fixed_axis]

    @property
    def parameter_domain(self) -> tuple[float, float] | None:
        if self.parameter_range is not None:
            return self.parameter_range
        domain = self.surface.parameter_domain
        if domain is None:
            return None
        return domain[0][1 - self.fixed_axis], domain[1][1 - self.fixed_axis]

    def evaluate(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=self.fixed_value.dtype)
        fixed = jnp.broadcast_to(self.fixed_value, values.shape)
        coordinates = (
            jnp.stack((fixed, values), axis=-1)
            if self.fixed_axis == 0
            else jnp.stack((values, fixed), axis=-1)
        )
        return self.surface.evaluate(coordinates)


class _RealizedNormalizedCurve(AbstractCurve):
    """Chart-normalized numeric p-curve with no exact source identity."""

    curve: AbstractCurve
    lower: Array
    extent: Array

    def __init__(
        self, curve: AbstractCurve, lower: ArrayLike, extent: ArrayLike, /
    ) -> None:
        if not isinstance(curve, AbstractCurve) or curve.ambient_dimension != 2:
            raise TypeError("A normalized p-curve requires a 2D source carrier.")
        lower_ = jnp.asarray(lower, dtype=jnp.float64)
        extent_ = jnp.asarray(extent, dtype=jnp.float64)
        if lower_.shape != (2,) or extent_.shape != (2,):
            raise ValueError("A normalized p-curve chart requires two coordinate bounds.")
        extent_ = eqx.error_if(
            extent_,
            jnp.any(~jnp.isfinite(extent_) | (extent_ <= 0.0)),
            "A normalized p-curve chart extent must be finite and positive.",
        )
        self.curve, self.lower, self.extent = curve, lower_, extent_

    @property
    def ambient_dimension(self) -> int:
        return 2

    @property
    def parameter_domain(self) -> tuple[float, float] | None:
        return self.curve.parameter_domain

    def evaluate(self, parameters: Array, /) -> Array:
        return (self.curve.evaluate(parameters) - self.lower) / self.extent

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        from .._interval_enclosure import interval_divide, interval_subtract

        bounds = np.asarray(self.curve.bounding_box(first, last), dtype=np.float64)
        shifted = interval_subtract(
            (bounds[0], bounds[1]),
            (np.asarray(self.lower), np.asarray(self.lower)),
        )
        normalized = interval_divide(
            shifted,
            (np.asarray(self.extent), np.asarray(self.extent)),
        )
        return np.stack(normalized)


class _RealizedPlacedSurface(AbstractSurfacePatch):
    """Numeric affine image without the source placement's exact identity payload."""

    definition: AbstractSurfacePatch
    linear: Array
    translation: Array
    source_parameter_domain: _SurfaceDomain = eqx.field(static=True)

    @property
    def parameter_domain(self) -> _SurfaceDomain:
        return self.source_parameter_domain

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return self.definition.periods

    def native_period_symbols(self, /) -> tuple[str | None, str | None]:
        return self.definition.native_period_symbols()

    def evaluate(self, parameters: Array, /) -> Array:
        return self.definition.evaluate(parameters) @ self.linear.T + self.translation


class _RealizedPlacedCurve(AbstractCurve):
    """Numeric affine image of an edge without exact occurrence payload."""

    definition: AbstractCurve
    linear: Array
    translation: Array

    @property
    def ambient_dimension(self) -> int:
        return self.definition.ambient_dimension

    @property
    def period(self) -> float | None:
        return self.definition.period

    def native_period_symbol(self, /) -> str | None:
        return self.definition.native_period_symbol()

    @property
    def parameter_domain(self) -> tuple[float, float] | None:
        return self.definition.parameter_domain

    def evaluate(self, parameters: Array, /) -> Array:
        return self.definition.evaluate(parameters) @ self.linear.T + self.translation


class _RealizedSurfaceRegion(StrictModule):
    patch: AbstractSurfacePatch
    parameter_box: Array


class _RealizedIntersectionCurve(AbstractCurve):
    """JAX branch realization separated from its exact source-root snapshot."""

    first: _RealizedSurfaceRegion
    second: _RealizedSurfaceRegion
    chart_axes: Array
    chart_start: Array
    chart_end: Array
    chart_lower: Array
    chart_upper: Array
    preconditioners: Array
    closed: bool = eqx.field(static=True)

    def __init__(
        self,
        source: IntersectionCurve,
        first: AbstractSurfacePatch,
        second: AbstractSurfacePatch,
        numeric: IntersectionCurve | None = None,
    ) -> None:
        value = source if numeric is None else numeric
        self.first = _RealizedSurfaceRegion(first, value.first.parameter_box)
        self.second = _RealizedSurfaceRegion(second, value.second.parameter_box)
        self.chart_axes = value.chart_axes
        self.chart_start = value.chart_start
        self.chart_end = value.chart_end
        self.chart_lower = value.chart_lower
        self.chart_upper = value.chart_upper
        self.preconditioners = value.preconditioners
        self.closed = source.closed

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def num_charts(self) -> int:
        return self.chart_axes.shape[0]

    @property
    def parameter_domain(self) -> tuple[float, float]:
        return 0.0, float(self.num_charts)

    @property
    def period(self) -> float | None:
        return float(self.num_charts) if self.closed else None

    def evaluate_coupled(self, parameters: Array, /) -> IntersectionCurvePoint:
        from ._intersection_curve import _evaluate_curve_point, IntersectionCurvePoint

        values = jnp.asarray(parameters, dtype=jnp.float64)
        if values.ndim > 0 and values.shape[-1:] == (1,):
            values = values[..., 0]
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values) | (values < 0.0) | (values > self.num_charts)),
            "A realized intersection parameter must lie inside its fixed branch interval.",
        )
        scalar = values.ndim == 0
        flat = values.reshape((-1,))
        result = jax.vmap(_evaluate_curve_point, in_axes=(None, 0))(self, flat)
        shape = values.shape
        restored = tuple(value.reshape((*shape, *value.shape[1:])) for value in result)
        if scalar:
            restored = tuple(
                value.reshape(value.shape[len(shape) :]) for value in restored
            )
        return IntersectionCurvePoint(*restored)

    def evaluate(self, parameters: Array, /) -> Array:
        return self.evaluate_coupled(parameters).point


class _RealizedIntersectionPCurve(AbstractCurve):
    curve: _RealizedIntersectionCurve
    side: str = eqx.field(static=True)
    first: float = eqx.field(static=True)
    last: float = eqx.field(static=True)
    reversed: bool = eqx.field(static=True)

    @property
    def ambient_dimension(self) -> int:
        return 2

    @property
    def parameter_domain(self) -> tuple[float, float]:
        return self.first, self.last

    @property
    def period(self) -> float | None:
        if self.curve.closed and self.first == 0.0 and self.last == self.curve.num_charts:
            return self.last - self.first
        return None

    def evaluate(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=jnp.float64)
        carrier = self.first + self.last - values if self.reversed else values
        coupled = self.curve.evaluate_coupled(carrier)
        return (
            coupled.first_parameters
            if self.side == "first"
            else coupled.second_parameters
        )


class _RealizedLineCurve(AbstractCurve):
    """Numeric-only affine curve carrier for traced fixed-topology realizations."""

    origin: Array
    direction: Array
    source_parameter_domain: tuple[float, float] | None = eqx.field(static=True)

    def __init__(
        self,
        origin: ArrayLike,
        direction: ArrayLike,
        /,
        *,
        parameter_domain: tuple[float, float] | None = None,
    ) -> None:
        origin_ = jnp.asarray(origin, dtype=jnp.float64)
        direction_ = jnp.asarray(direction, dtype=jnp.float64)
        if (
            origin_.ndim != 1
            or origin_.shape[0] not in (2, 3)
            or direction_.shape != origin_.shape
        ):
            raise ValueError("A realized affine curve requires aligned 2D or 3D vectors.")
        origin_ = eqx.error_if(
            origin_,
            jnp.any(~jnp.isfinite(origin_)),
            "A realized affine curve origin must be finite.",
        )
        direction_ = eqx.error_if(
            direction_,
            jnp.any(~jnp.isfinite(direction_)),
            "A realized affine curve direction must be finite.",
        )
        direction_ = eqx.error_if(
            direction_,
            jnp.all(direction_ == 0.0),
            "A realized affine curve direction must be nonzero.",
        )
        if parameter_domain is not None:
            first, last = parameter_domain
            if not (np.isfinite(first) and np.isfinite(last) and first < last):
                raise ValueError(
                    "A realized affine curve domain must be finite and increasing."
                )
            parameter_domain = float(first), float(last)
        self.origin = origin_
        self.direction = direction_
        self.source_parameter_domain = parameter_domain

    @property
    def ambient_dimension(self) -> int:
        return self.origin.shape[0]

    @property
    def parameter_domain(self) -> tuple[float, float] | None:
        return self.source_parameter_domain

    def evaluate(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=self.origin.dtype)
        return self.origin + values[..., None] * self.direction

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        from .._rational import fraction_down, fraction_up

        origin = tuple(
            Fraction.from_float(float(value)) for value in np.asarray(self.origin)
        )
        direction = tuple(
            Fraction.from_float(float(value)) for value in np.asarray(self.direction)
        )
        parameters = (Fraction.from_float(float(first)), Fraction.from_float(float(last)))
        points = tuple(
            tuple(
                origin[axis] + parameter * direction[axis] for axis in range(len(origin))
            )
            for parameter in parameters
        )
        return np.stack(
            (
                tuple(
                    fraction_down(min(point[axis] for point in points))
                    for axis in range(len(origin))
                ),
                tuple(
                    fraction_up(max(point[axis] for point in points))
                    for axis in range(len(origin))
                ),
            )
        )


class _RealizedConicCurve(AbstractCurve):
    """Numeric-only circle, ellipse, parabola, or hyperbola realization."""

    kind: str = eqx.field(static=True)
    center: Array
    first_axis: Array
    second_axis: Array
    first_parameter: Array
    second_parameter: Array
    source_parameter_domain: tuple[float, float] | None = eqx.field(static=True)
    source_period: float | None = eqx.field(static=True)
    source_period_symbol: str | None = eqx.field(static=True)

    def __init__(
        self,
        kind: str,
        center: ArrayLike,
        first_axis: ArrayLike,
        second_axis: ArrayLike,
        first_parameter: ArrayLike,
        second_parameter: ArrayLike = 0.0,
        /,
        *,
        parameter_domain: tuple[float, float] | None = None,
        period: float | None = None,
        period_symbol: str | None = None,
    ) -> None:
        if kind not in ("circle", "ellipse", "parabola", "hyperbola"):
            raise ValueError(
                "A realized conic kind must be circle, ellipse, parabola, or hyperbola."
            )
        center_ = jnp.asarray(center, dtype=jnp.float64)
        first_ = jnp.asarray(first_axis, dtype=jnp.float64)
        second_ = jnp.asarray(second_axis, dtype=jnp.float64)
        first_parameter_ = jnp.asarray(first_parameter, dtype=jnp.float64).reshape(())
        second_parameter_ = jnp.asarray(second_parameter, dtype=jnp.float64).reshape(())
        if (
            center_.ndim != 1
            or center_.shape[0] not in (2, 3)
            or first_.shape != center_.shape
            or second_.shape != center_.shape
        ):
            raise ValueError("A realized conic requires aligned 2D or 3D frame vectors.")
        for name, value in (
            ("center", center_),
            ("first axis", first_),
            ("second axis", second_),
            ("first parameter", first_parameter_),
            ("second parameter", second_parameter_),
        ):
            value = eqx.error_if(
                value,
                jnp.any(~jnp.isfinite(value)),
                f"A realized conic {name} must be finite.",
            )
            if name == "center":
                center_ = value
            elif name == "first axis":
                first_ = value
            elif name == "second axis":
                second_ = value
            elif name == "first parameter":
                first_parameter_ = value
            else:
                second_parameter_ = value
        if kind in ("circle", "ellipse", "hyperbola"):
            first_parameter_ = eqx.error_if(
                first_parameter_,
                first_parameter_ <= 0.0,
                "Realized conic radii must be positive.",
            )
        if kind in ("ellipse", "hyperbola"):
            second_parameter_ = eqx.error_if(
                second_parameter_,
                second_parameter_ <= 0.0,
                "Realized conic radii must be positive.",
            )
        if kind == "parabola":
            first_parameter_ = eqx.error_if(
                first_parameter_,
                first_parameter_ <= 0.0,
                "A realized parabola focal length must be positive.",
            )
        if parameter_domain is not None:
            lower, upper = parameter_domain
            if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
                raise ValueError(
                    "A realized conic parameter domain must be finite and increasing."
                )
            parameter_domain = float(lower), float(upper)
        if period is not None and (not np.isfinite(period) or period <= 0.0):
            raise ValueError("A realized conic period must be finite and positive.")
        if period_symbol is not None and (
            not isinstance(period_symbol, str) or not period_symbol
        ):
            raise ValueError(
                "A realized conic period symbol must be nonempty when supplied."
            )
        self.kind = kind
        self.center = center_
        self.first_axis = first_
        self.second_axis = second_
        self.first_parameter = first_parameter_
        self.second_parameter = second_parameter_
        self.source_parameter_domain = parameter_domain
        self.source_period = None if period is None else float(period)
        self.source_period_symbol = period_symbol

    @property
    def ambient_dimension(self) -> int:
        return self.center.shape[0]

    @property
    def period(self) -> float | None:
        return self.source_period

    def native_period_symbol(self, /) -> str | None:
        return self.source_period_symbol

    @property
    def parameter_domain(self) -> tuple[float, float] | None:
        return self.source_parameter_domain

    def evaluate(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=self.center.dtype)[..., None]
        if self.kind == "circle":
            return self.center + self.first_parameter * (
                jnp.cos(values) * self.first_axis + jnp.sin(values) * self.second_axis
            )
        if self.kind == "ellipse":
            return (
                self.center
                + self.first_parameter * jnp.cos(values) * self.first_axis
                + self.second_parameter * jnp.sin(values) * self.second_axis
            )
        if self.kind == "parabola":
            return (
                self.center
                + values**2 / (4.0 * self.first_parameter) * self.first_axis
                + values * self.second_axis
            )
        return (
            self.center
            + self.first_parameter * jnp.cosh(values) * self.first_axis
            + self.second_parameter * jnp.sinh(values) * self.second_axis
        )

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        from .._interval_enclosure import (
            interval_add,
            interval_cos,
            interval_divide,
            interval_multiply,
            interval_sin,
            interval_square,
        )

        start, end = float(first), float(last)
        if not (np.isfinite(start) and np.isfinite(end) and start <= end):
            raise ValueError("A realized conic bound requires a finite ordered interval.")
        domain = self.parameter_domain
        if domain is not None and (start < domain[0] or end > domain[1]):
            raise ValueError("A realized conic bound must lie in its source interval.")
        center = np.asarray(self.center, dtype=np.float64)
        if self.kind in ("circle", "ellipse"):
            angle: Interval = (
                np.asarray(start, dtype=np.float64),
                np.asarray(end, dtype=np.float64),
            )
            cosine, sine = interval_cos(angle), interval_sin(angle)
            first_scale = np.asarray(self.first_parameter, dtype=np.float64)
            second_scale = (
                first_scale
                if self.kind == "circle"
                else np.asarray(self.second_parameter, dtype=np.float64)
            )
            first_term = interval_multiply(
                interval_multiply(
                    (first_scale, first_scale),
                    (np.asarray(self.first_axis), np.asarray(self.first_axis)),
                ),
                cosine,
            )
            second_term = interval_multiply(
                interval_multiply(
                    (second_scale, second_scale),
                    (np.asarray(self.second_axis), np.asarray(self.second_axis)),
                ),
                sine,
            )
            lower, upper = interval_add(
                interval_add((center, center), first_term), second_term
            )
            return np.stack((lower, upper))
        if self.kind == "parabola":
            parameter: Interval = (
                np.asarray(start, dtype=np.float64),
                np.asarray(end, dtype=np.float64),
            )
            squared = interval_square(parameter)
            quadratic = interval_divide(
                interval_multiply(
                    squared,
                    (np.asarray(self.first_axis), np.asarray(self.first_axis)),
                ),
                (
                    4.0 * np.asarray(self.first_parameter),
                    4.0 * np.asarray(self.first_parameter),
                ),
            )
            linear = interval_multiply(
                parameter,
                (np.asarray(self.second_axis), np.asarray(self.second_axis)),
            )
            lower, upper = interval_add(interval_add((center, center), quadratic), linear)
            return np.stack((lower, upper))
        raise NotImplementedError(
            "A sound interval extension for the hyperbola realization is unavailable."
        )


class _RealizedPlanePatch(AbstractSurfacePatch):
    origin: Array
    first_axis: Array
    second_axis: Array
    source_parameter_domain: _SurfaceDomain = eqx.field(static=True)
    source_periods: tuple[float | None, float | None] = eqx.field(static=True)
    source_period_symbols: tuple[str | None, str | None] = eqx.field(static=True)

    def __init__(
        self,
        origin: ArrayLike,
        first_axis: ArrayLike,
        second_axis: ArrayLike,
        /,
        *,
        parameter_domain: _SurfaceDomain = None,
        periods: tuple[float | None, float | None] = (None, None),
        period_symbols: tuple[str | None, str | None] = (None, None),
    ) -> None:
        self.origin = jnp.asarray(origin, dtype=jnp.float64)
        self.first_axis = jnp.asarray(first_axis, dtype=jnp.float64)
        self.second_axis = jnp.asarray(second_axis, dtype=jnp.float64)
        self.source_parameter_domain = parameter_domain
        self.source_periods = periods
        self.source_period_symbols = period_symbols

    @property
    def parameter_domain(self) -> _SurfaceDomain:
        return self.source_parameter_domain

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return self.source_periods

    def native_period_symbols(self, /) -> tuple[str | None, str | None]:
        return self.source_period_symbols

    def evaluate(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=self.origin.dtype)
        return (
            self.origin
            + values[..., :1] * self.first_axis
            + values[..., 1:2] * self.second_axis
        )


class _RealizedCylinderPatch(AbstractSurfacePatch):
    origin: Array
    first_axis: Array
    second_axis: Array
    axis: Array
    radius: Array
    source_parameter_domain: _SurfaceDomain = eqx.field(static=True)
    source_periods: tuple[float | None, float | None] = eqx.field(static=True)
    source_period_symbols: tuple[str | None, str | None] = eqx.field(static=True)

    def __init__(
        self,
        origin: ArrayLike,
        first_axis: ArrayLike,
        second_axis: ArrayLike,
        axis: ArrayLike,
        radius: ArrayLike,
        /,
        *,
        parameter_domain: _SurfaceDomain = None,
        periods: tuple[float | None, float | None] = (float(2.0 * np.pi), None),
        period_symbols: tuple[str | None, str | None] = ("two_pi", None),
    ) -> None:
        origin_ = jnp.asarray(origin, dtype=jnp.float64)
        first_ = jnp.asarray(first_axis, dtype=jnp.float64)
        second_ = jnp.asarray(second_axis, dtype=jnp.float64)
        axis_ = jnp.asarray(axis, dtype=jnp.float64)
        radius_ = jnp.asarray(radius, dtype=jnp.float64).reshape(())
        if any(value.shape != (3,) for value in (origin_, first_, second_, axis_)):
            raise ValueError("A realized cylinder requires four aligned 3D vectors.")
        origin_ = eqx.error_if(
            origin_,
            jnp.any(~jnp.isfinite(origin_)),
            "Realized cylinder origin must be finite.",
        )
        first_ = eqx.error_if(
            first_,
            jnp.any(~jnp.isfinite(first_)),
            "Realized cylinder first axis must be finite.",
        )
        second_ = eqx.error_if(
            second_,
            jnp.any(~jnp.isfinite(second_)),
            "Realized cylinder second axis must be finite.",
        )
        axis_ = eqx.error_if(
            axis_, jnp.any(~jnp.isfinite(axis_)), "Realized cylinder axis must be finite."
        )
        radius_ = eqx.error_if(
            radius_,
            jnp.any(~jnp.isfinite(radius_)),
            "Realized cylinder radius must be finite.",
        )
        radius_ = eqx.error_if(
            radius_, radius_ <= 0.0, "A realized cylinder radius must be positive."
        )
        self.origin, self.first_axis, self.second_axis, self.axis = (
            origin_,
            first_,
            second_,
            axis_,
        )
        self.radius = radius_
        self.source_parameter_domain = parameter_domain
        self.source_periods = periods
        self.source_period_symbols = period_symbols

    @property
    def parameter_domain(self) -> _SurfaceDomain:
        return self.source_parameter_domain

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return self.source_periods

    def native_period_symbols(self, /) -> tuple[str | None, str | None]:
        return self.source_period_symbols

    def evaluate(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=self.origin.dtype)
        angle, height = values[..., 0], values[..., 1]
        radial = (
            jnp.cos(angle)[..., None] * self.first_axis
            + jnp.sin(angle)[..., None] * self.second_axis
        )
        return self.origin + self.radius * radial + height[..., None] * self.axis

    def is_c1_along(
        self, fixed_axis: int, fixed_value: float, first: float, last: float, /
    ) -> bool:
        if isinstance(fixed_axis, bool) or fixed_axis not in (0, 1):
            raise ValueError("fixed_axis must be 0 or 1.")
        if (
            not all(np.isfinite(value) for value in (fixed_value, first, last))
            or first > last
        ):
            raise ValueError(
                "Realized cylinder isoline bounds must be finite and ordered."
            )
        return True


class _RealizedTorusPatch(AbstractSurfacePatch):
    center: Array
    first_axis: Array
    second_axis: Array
    axis: Array
    major_radius: Array
    minor_radius: Array
    source_parameter_domain: _SurfaceDomain = eqx.field(static=True)
    source_periods: tuple[float | None, float | None] = eqx.field(static=True)
    source_period_symbols: tuple[str | None, str | None] = eqx.field(static=True)

    def __init__(
        self,
        center: ArrayLike,
        first_axis: ArrayLike,
        second_axis: ArrayLike,
        axis: ArrayLike,
        major_radius: ArrayLike,
        minor_radius: ArrayLike,
        /,
        *,
        parameter_domain: _SurfaceDomain = None,
        periods: tuple[float | None, float | None] = (
            float(2.0 * np.pi),
            float(2.0 * np.pi),
        ),
        period_symbols: tuple[str | None, str | None] = ("two_pi", "two_pi"),
    ) -> None:
        center_ = jnp.asarray(center, dtype=jnp.float64)
        first_ = jnp.asarray(first_axis, dtype=jnp.float64)
        second_ = jnp.asarray(second_axis, dtype=jnp.float64)
        axis_ = jnp.asarray(axis, dtype=jnp.float64)
        major_ = jnp.asarray(major_radius, dtype=jnp.float64).reshape(())
        minor_ = jnp.asarray(minor_radius, dtype=jnp.float64).reshape(())
        if any(value.shape != (3,) for value in (center_, first_, second_, axis_)):
            raise ValueError("A realized torus requires four aligned 3D vectors.")
        center_ = eqx.error_if(
            center_,
            jnp.any(~jnp.isfinite(center_)),
            "Realized torus center must be finite.",
        )
        first_ = eqx.error_if(
            first_,
            jnp.any(~jnp.isfinite(first_)),
            "Realized torus first axis must be finite.",
        )
        second_ = eqx.error_if(
            second_,
            jnp.any(~jnp.isfinite(second_)),
            "Realized torus second axis must be finite.",
        )
        axis_ = eqx.error_if(
            axis_, jnp.any(~jnp.isfinite(axis_)), "Realized torus axis must be finite."
        )
        major_ = eqx.error_if(
            major_,
            jnp.any(~jnp.isfinite(major_)),
            "Realized torus major radius must be finite.",
        )
        minor_ = eqx.error_if(
            minor_,
            jnp.any(~jnp.isfinite(minor_)),
            "Realized torus minor radius must be finite.",
        )
        major_ = eqx.error_if(
            major_, major_ <= 0.0, "A realized torus major radius must be positive."
        )
        minor_ = eqx.error_if(
            minor_, minor_ <= 0.0, "A realized torus minor radius must be positive."
        )
        self.center, self.first_axis, self.second_axis, self.axis = (
            center_,
            first_,
            second_,
            axis_,
        )
        self.major_radius, self.minor_radius = major_, minor_
        self.source_parameter_domain = parameter_domain
        self.source_periods = periods
        self.source_period_symbols = period_symbols

    @property
    def parameter_domain(self) -> _SurfaceDomain:
        return self.source_parameter_domain

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return self.source_periods

    def native_period_symbols(self, /) -> tuple[str | None, str | None]:
        return self.source_period_symbols

    def evaluate(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=self.center.dtype)
        longitude, tube = values[..., 0], values[..., 1]
        radial = (
            jnp.cos(longitude)[..., None] * self.first_axis
            + jnp.sin(longitude)[..., None] * self.second_axis
        )
        return (
            self.center
            + (self.major_radius + self.minor_radius * jnp.cos(tube))[..., None] * radial
            + (self.minor_radius * jnp.sin(tube))[..., None] * self.axis
        )


class _RealizedConePatch(AbstractSurfacePatch):
    origin: Array
    first_axis: Array
    second_axis: Array
    axis: Array
    reference_radius: Array
    semi_angle: Array
    source_periods: tuple[float | None, float | None] = eqx.field(static=True)
    source_period_symbols: tuple[str | None, str | None] = eqx.field(static=True)

    def __init__(
        self,
        origin: ArrayLike,
        first_axis: ArrayLike,
        second_axis: ArrayLike,
        axis: ArrayLike,
        reference_radius: ArrayLike,
        semi_angle: ArrayLike,
        /,
    ) -> None:
        self.origin = jnp.asarray(origin, dtype=jnp.float64)
        self.first_axis = jnp.asarray(first_axis, dtype=jnp.float64)
        self.second_axis = jnp.asarray(second_axis, dtype=jnp.float64)
        self.axis = jnp.asarray(axis, dtype=jnp.float64)
        self.reference_radius = jnp.asarray(reference_radius, dtype=jnp.float64).reshape(
            ()
        )
        self.semi_angle = jnp.asarray(semi_angle, dtype=jnp.float64).reshape(())
        self.source_periods = (float(2.0 * np.pi), None)
        self.source_period_symbols = ("two_pi", None)

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return self.source_periods

    def native_period_symbols(self, /) -> tuple[str | None, str | None]:
        return self.source_period_symbols

    def evaluate(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=self.origin.dtype)
        angle, axial = values[..., 0], values[..., 1]
        radial = (
            jnp.cos(angle)[..., None] * self.first_axis
            + jnp.sin(angle)[..., None] * self.second_axis
        )
        radius = self.reference_radius + axial * jnp.tan(self.semi_angle)
        return self.origin + axial[..., None] * self.axis + radius[..., None] * radial


class _RealizedSpherePatch(AbstractSurfacePatch):
    center: Array
    first_axis: Array
    second_axis: Array
    axis: Array
    radius: Array
    source_periods: tuple[float | None, float | None] = eqx.field(static=True)
    source_period_symbols: tuple[str | None, str | None] = eqx.field(static=True)

    def __init__(
        self,
        center: ArrayLike,
        first_axis: ArrayLike,
        second_axis: ArrayLike,
        axis: ArrayLike,
        radius: ArrayLike,
        /,
    ) -> None:
        self.center = jnp.asarray(center, dtype=jnp.float64)
        self.first_axis = jnp.asarray(first_axis, dtype=jnp.float64)
        self.second_axis = jnp.asarray(second_axis, dtype=jnp.float64)
        self.axis = jnp.asarray(axis, dtype=jnp.float64)
        self.radius = jnp.asarray(radius, dtype=jnp.float64).reshape(())
        self.source_periods = (float(2.0 * np.pi), float(np.pi))
        self.source_period_symbols = ("two_pi", None)

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return self.source_periods

    def native_period_symbols(self, /) -> tuple[str | None, str | None]:
        return self.source_period_symbols

    def evaluate(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=self.center.dtype)
        longitude, latitude = values[..., 0], values[..., 1]
        equatorial = (
            jnp.cos(longitude)[..., None] * self.first_axis
            + jnp.sin(longitude)[..., None] * self.second_axis
        )
        direction = (
            jnp.cos(latitude)[..., None] * equatorial
            + jnp.sin(latitude)[..., None] * self.axis
        )
        return self.center + self.radius * direction


def _realized_bspline_basis(
    knots: Array, degree: int, count: int, parameter: Array, /
) -> Array:
    basis = ((parameter >= knots[:-1]) & (parameter < knots[1:])).astype(knots.dtype)
    for order in range(1, degree + 1):
        width = basis.shape[0] - 1
        left_denominator = knots[order : order + width] - knots[:width]
        right_denominator = knots[order + 1 : order + 1 + width] - knots[1 : 1 + width]
        left = (parameter - knots[:width]) / jnp.where(
            left_denominator != 0.0, left_denominator, 1.0
        )
        right = (knots[order + 1 : order + 1 + width] - parameter) / jnp.where(
            right_denominator != 0.0, right_denominator, 1.0
        )
        basis = left * basis[:-1] + right * basis[1:]
    endpoint = parameter == knots[count]
    endpoint_basis = jax.nn.one_hot(count - 1, count, dtype=knots.dtype)
    return jnp.where(endpoint, endpoint_basis, basis)


class _RealizedBSplineSurfacePatch(AbstractSurfacePatch):
    """Rational B-spline numeric realization without exact source banks."""

    control_points: Array
    weights: Array
    u_knots: Array
    v_knots: Array
    u_degree: int = eqx.field(static=True)
    v_degree: int = eqx.field(static=True)
    source_parameter_domain: _SurfaceDomain = eqx.field(static=True)
    source_periods: tuple[float | None, float | None] = eqx.field(static=True)
    source_period_symbols: tuple[str | None, str | None] = eqx.field(static=True)

    def __init__(
        self,
        control_points: ArrayLike,
        weights: ArrayLike,
        u_knots: ArrayLike,
        v_knots: ArrayLike,
        u_degree: int,
        v_degree: int,
        /,
        *,
        parameter_domain: _SurfaceDomain = None,
        periods: tuple[float | None, float | None] = (None, None),
        period_symbols: tuple[str | None, str | None] = (None, None),
    ) -> None:
        control_points_ = jnp.asarray(control_points, dtype=jnp.float64)
        weights_ = jnp.asarray(weights, dtype=jnp.float64)
        u_knots_ = jnp.asarray(u_knots, dtype=jnp.float64)
        v_knots_ = jnp.asarray(v_knots, dtype=jnp.float64)
        if (
            control_points_.ndim != 3
            or control_points_.shape[-1] != 3
            or weights_.shape != control_points_.shape[:2]
            or u_knots_.shape != (control_points_.shape[0] + u_degree + 1,)
            or v_knots_.shape != (control_points_.shape[1] + v_degree + 1,)
            or type(u_degree) is not int
            or type(v_degree) is not int
            or u_degree < 1
            or v_degree < 1
            or control_points_.shape[0] <= u_degree
            or control_points_.shape[1] <= v_degree
        ):
            raise ValueError(
                "Realized B-spline surface arrays must preserve their fixed tensor topology."
            )
        control_points_ = eqx.error_if(
            control_points_,
            jnp.any(~jnp.isfinite(control_points_)),
            "B-spline controls must be finite.",
        )
        weights_ = eqx.error_if(
            weights_,
            jnp.any(~jnp.isfinite(weights_)) | jnp.any(weights_ <= 0.0),
            "B-spline weights must be finite and positive.",
        )
        u_knots_ = eqx.error_if(
            u_knots_,
            jnp.any(~jnp.isfinite(u_knots_)) | jnp.any(jnp.diff(u_knots_) < 0.0),
            "B-spline u-knots must be finite and nondecreasing.",
        )
        v_knots_ = eqx.error_if(
            v_knots_,
            jnp.any(~jnp.isfinite(v_knots_)) | jnp.any(jnp.diff(v_knots_) < 0.0),
            "B-spline v-knots must be finite and nondecreasing.",
        )
        self.control_points, self.weights = control_points_, weights_
        self.u_knots, self.v_knots = u_knots_, v_knots_
        self.u_degree, self.v_degree = u_degree, v_degree
        self.source_parameter_domain = parameter_domain
        self.source_periods = periods
        self.source_period_symbols = period_symbols

    @property
    def parameter_domain(self) -> _SurfaceDomain:
        return self.source_parameter_domain

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return self.source_periods

    def native_period_symbols(self, /) -> tuple[str | None, str | None]:
        return self.source_period_symbols

    def _evaluate_one(self, uv: Array, /) -> Array:
        from ...ein import contract

        basis_u = _realized_bspline_basis(
            self.u_knots, self.u_degree, self.control_points.shape[0], uv[0]
        )
        basis_v = _realized_bspline_basis(
            self.v_knots, self.v_degree, self.control_points.shape[1], uv[1]
        )
        weighted_basis = basis_u[:, None] * basis_v[None, :] * self.weights
        denominator = jnp.sum(weighted_basis)
        return contract("uv,uvc->c", weighted_basis, self.control_points) / denominator

    def evaluate(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=self.control_points.dtype)
        if values.ndim == 1:
            return self._evaluate_one(values)
        leading = values.shape[:-1]
        return jax.vmap(self._evaluate_one)(values.reshape((-1, 2))).reshape(
            (*leading, 3)
        )


class _RealizedOperationSurface(AbstractSurfacePatch):
    """Numeric-only offset, extrusion, or revolution surface realization."""

    kind: str = eqx.field(static=True)
    frame_sign: int = eqx.field(static=True)
    base: AbstractSurfacePatch | None
    curve: AbstractCurve | None
    distance: Array | None
    direction: Array | None
    axis_origin: Array | None
    axis_direction: Array | None
    source_parameter_domain: _SurfaceDomain = eqx.field(static=True)
    source_periods: tuple[float | None, float | None] = eqx.field(static=True)
    source_period_symbols: tuple[str | None, str | None] = eqx.field(static=True)

    def __init__(
        self,
        kind: str,
        *,
        base: AbstractSurfacePatch | None = None,
        curve: AbstractCurve | None = None,
        distance: ArrayLike | None = None,
        frame_sign: int = 0,
        direction: ArrayLike | None = None,
        axis_origin: ArrayLike | None = None,
        axis_direction: ArrayLike | None = None,
        parameter_domain: _SurfaceDomain = None,
        periods: tuple[float | None, float | None] = (None, None),
        period_symbols: tuple[str | None, str | None] = (None, None),
    ) -> None:
        if kind not in ("offset", "extrusion", "revolution"):
            raise ValueError(
                "A realized operation surface kind must be offset, extrusion, or revolution."
            )
        if frame_sign not in (-1, 0, 1):
            raise ValueError("A realized offset frame sign must be -1, 0, or 1.")
        self.kind, self.frame_sign = kind, frame_sign
        self.base = base
        self.curve = curve
        self.distance = (
            None
            if distance is None
            else jnp.asarray(distance, dtype=jnp.float64).reshape(())
        )
        self.direction = (
            None
            if direction is None
            else jnp.asarray(direction, dtype=jnp.float64).reshape((3,))
        )
        self.axis_origin = (
            None
            if axis_origin is None
            else jnp.asarray(axis_origin, dtype=jnp.float64).reshape((3,))
        )
        self.axis_direction = (
            None
            if axis_direction is None
            else jnp.asarray(axis_direction, dtype=jnp.float64).reshape((3,))
        )
        self.source_parameter_domain = parameter_domain
        self.source_periods = periods
        self.source_period_symbols = period_symbols

    @property
    def parameter_domain(self) -> _SurfaceDomain:
        return self.source_parameter_domain

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return self.source_periods

    def native_period_symbols(self, /) -> tuple[str | None, str | None]:
        return self.source_period_symbols

    def _evaluate_one(self, uv: Array, /) -> Array:
        if self.kind == "extrusion":
            assert self.curve is not None and self.direction is not None
            return self.curve.evaluate(uv[0]) + uv[1] * self.direction
        if self.kind == "revolution":
            assert (
                self.curve is not None
                and self.axis_origin is not None
                and self.axis_direction is not None
            )
            angle = uv[0]
            relative = self.curve.evaluate(uv[1]) - self.axis_origin
            axial = jnp.sum(relative * self.axis_direction)
            rotated = (
                relative * jnp.cos(angle)
                + jnp.cross(self.axis_direction, relative) * jnp.sin(angle)
                + self.axis_direction * axial * (1.0 - jnp.cos(angle))
            )
            return self.axis_origin + rotated
        assert self.base is not None and self.distance is not None
        if self.frame_sign != 0 and isinstance(self.base, _RealizedSpherePatch):
            longitude, latitude = uv[0], uv[1]
            equatorial = (
                jnp.cos(longitude) * self.base.first_axis
                + jnp.sin(longitude) * self.base.second_axis
            )
            direction = (
                jnp.cos(latitude) * equatorial + jnp.sin(latitude) * self.base.axis
            )
            return (
                self.base.center
                + (self.base.radius + self.frame_sign * self.distance) * direction
            )
        if self.frame_sign != 0 and isinstance(self.base, _RealizedCylinderPatch):
            angle, height = uv[0], uv[1]
            radial = (
                jnp.cos(angle) * self.base.first_axis
                + jnp.sin(angle) * self.base.second_axis
            )
            return (
                self.base.origin
                + (self.base.radius + self.frame_sign * self.distance) * radial
                + height * self.base.axis
            )
        if self.frame_sign != 0 and isinstance(self.base, _RealizedTorusPatch):
            longitude, tube = uv[0], uv[1]
            radial = (
                jnp.cos(longitude) * self.base.first_axis
                + jnp.sin(longitude) * self.base.second_axis
            )
            normal = (
                jnp.cos(tube) * radial + jnp.sin(tube) * self.base.axis
            ) * self.frame_sign
            return self.base.evaluate(uv) + self.distance * normal
        if self.frame_sign != 0 and isinstance(self.base, _RealizedConePatch):
            angle = uv[0]
            radial = (
                jnp.cos(angle) * self.base.first_axis
                + jnp.sin(angle) * self.base.second_axis
            )
            tangent = jnp.tan(self.base.semi_angle)
            normal = (radial - tangent * self.base.axis) / jnp.sqrt(
                1.0 + tangent * tangent
            )
            return self.base.evaluate(uv) + self.frame_sign * self.distance * normal
        point = self.base.evaluate(uv)
        differential = jax.jacfwd(self.base.evaluate)(uv)
        normal = jnp.cross(differential[:, 0], differential[:, 1])
        normal = normal / jnp.linalg.norm(normal)
        sign = self.frame_sign if self.frame_sign != 0 else 1
        return point + sign * self.distance * normal

    def evaluate(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=jnp.float64)
        if values.ndim == 1:
            return self._evaluate_one(values)
        leading = values.shape[:-1]
        return jax.vmap(self._evaluate_one)(values.reshape((-1, 2))).reshape(
            (*leading, 3)
        )


class _RealizedBSplineCurve(AbstractCurve):
    control_points: Array
    weights: Array
    knots: Array
    degree: int = eqx.field(static=True)
    source_parameter_domain: tuple[float, float] | None = eqx.field(static=True)
    source_period: float | None = eqx.field(static=True)
    source_period_symbol: str | None = eqx.field(static=True)

    def __init__(
        self,
        control_points: ArrayLike,
        weights: ArrayLike,
        knots: ArrayLike,
        degree: int,
        /,
        *,
        parameter_domain: tuple[float, float] | None = None,
        period: float | None = None,
        period_symbol: str | None = None,
    ) -> None:
        controls = jnp.asarray(control_points, dtype=jnp.float64)
        weights_ = jnp.asarray(weights, dtype=jnp.float64)
        knots_ = jnp.asarray(knots, dtype=jnp.float64)
        if (
            controls.ndim != 2
            or controls.shape[1] not in (2, 3)
            or weights_.shape != (controls.shape[0],)
            or knots_.shape != (controls.shape[0] + degree + 1,)
            or type(degree) is not int
            or degree < 1
            or controls.shape[0] <= degree
        ):
            raise ValueError(
                "Realized B-spline curve arrays must preserve fixed curve topology."
            )
        controls = eqx.error_if(
            controls,
            jnp.any(~jnp.isfinite(controls)),
            "B-spline controls must be finite.",
        )
        weights_ = eqx.error_if(
            weights_,
            jnp.any(~jnp.isfinite(weights_)) | jnp.any(weights_ <= 0.0),
            "B-spline weights must be finite and positive.",
        )
        knots_ = eqx.error_if(
            knots_,
            jnp.any(~jnp.isfinite(knots_)) | jnp.any(jnp.diff(knots_) < 0.0),
            "B-spline knots must be finite and nondecreasing.",
        )
        if parameter_domain is not None:
            lower, upper = parameter_domain
            if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
                raise ValueError(
                    "A realized B-spline curve domain must be finite and increasing."
                )
            parameter_domain = float(lower), float(upper)
        self.control_points, self.weights, self.knots = controls, weights_, knots_
        self.degree = degree
        self.source_parameter_domain = parameter_domain
        self.source_period = period
        self.source_period_symbol = period_symbol

    @property
    def ambient_dimension(self) -> int:
        return self.control_points.shape[1]

    @property
    def period(self) -> float | None:
        return self.source_period

    def native_period_symbol(self, /) -> str | None:
        return self.source_period_symbol

    @property
    def parameter_domain(self) -> tuple[float, float] | None:
        return self.source_parameter_domain

    def _evaluate_one(self, parameter: Array, /) -> Array:
        basis = _realized_bspline_basis(
            self.knots, self.degree, self.control_points.shape[0], parameter
        )
        weighted_basis = basis * self.weights
        return jnp.sum(weighted_basis[:, None] * self.control_points, axis=0) / jnp.sum(
            weighted_basis
        )

    def evaluate(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=self.control_points.dtype)
        if values.ndim == 0:
            return self._evaluate_one(values)
        return jax.vmap(self._evaluate_one)(values.reshape((-1,))).reshape(
            (*values.shape, self.ambient_dimension)
        )

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        if not (np.isfinite(first) and np.isfinite(last) and first <= last):
            raise ValueError(
                "A realized B-spline bound requires a finite ordered interval."
            )
        domain = self.parameter_domain
        if domain is not None and (first < domain[0] or last > domain[1]):
            raise ValueError("A realized B-spline bound must lie in its source interval.")
        controls = np.asarray(self.control_points, dtype=np.float64)
        return np.stack((np.min(controls, axis=0), np.max(controls, axis=0)))


class _RealizedTrimSegment(AbstractCurve):
    """Numeric-only normalized interval over one realized planar curve."""

    curve: AbstractCurve
    first: Array
    last: Array
    reversed: bool = eqx.field(static=True)

    def __init__(
        self,
        curve: AbstractCurve,
        first: ArrayLike,
        last: ArrayLike,
        /,
        *,
        reversed: bool = False,
    ) -> None:
        if not isinstance(curve, AbstractCurve) or curve.ambient_dimension != 2:
            raise TypeError("A realized planar trim requires a 2D curve carrier.")
        if not isinstance(reversed, bool):
            raise TypeError("A realized trim orientation must be bool.")
        first_ = jnp.asarray(first, dtype=jnp.float64).reshape(())
        last_ = jnp.asarray(last, dtype=jnp.float64).reshape(())
        first_ = eqx.error_if(
            first_, ~jnp.isfinite(first_), "A realized trim start must be finite."
        )
        last_ = eqx.error_if(
            last_,
            ~jnp.isfinite(last_) | (last_ <= first_),
            "A realized trim range must be finite and increasing.",
        )
        self.curve, self.first, self.last, self.reversed = curve, first_, last_, reversed

    @property
    def ambient_dimension(self) -> int:
        return 2

    def evaluate(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=self.first.dtype)
        start, end = (self.last, self.first) if self.reversed else (self.first, self.last)
        return self.curve.evaluate(start + values * (end - start))

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        start, end = (self.last, self.first) if self.reversed else (self.first, self.last)
        first_value = float(np.asarray(start + first * (end - start)))
        last_value = float(np.asarray(start + last * (end - start)))
        lower, upper = sorted((first_value, last_value))
        return self.curve.bounding_box(lower, upper)

    def identity_payload(self, /) -> dict[str, object]:
        raise NotImplementedError(
            "A numeric trim realization has no exact source identity."
        )


class _RealizedTSplineSurfacePatch(AbstractSurfacePatch):
    """T-spline evaluation over a dynamic control bank, without exact bank fields."""

    mesh: TMesh2D
    realization: LocalExtractedBernsteinRealization
    transfer_plans: tuple[ASTSTransferPlan, ...]
    control_points: Array
    weights: Array
    source_parameter_domain: _SurfaceDomain = eqx.field(static=True)
    source_periods: tuple[float | None, float | None] = eqx.field(static=True)
    source_period_symbols: tuple[str | None, str | None] = eqx.field(static=True)

    def __init__(
        self, source: TSplineSurfacePatch, control_points: ArrayLike, weights: ArrayLike
    ) -> None:
        self.mesh = source.mesh
        self.realization = source.realization
        self.transfer_plans = tuple(source.transfer_plans)
        self.control_points = jnp.asarray(control_points, dtype=jnp.float64)
        self.weights = jnp.asarray(weights, dtype=jnp.float64)
        self.source_parameter_domain = source.parameter_domain
        self.source_periods = source.periods
        self.source_period_symbols = source.native_period_symbols()

    @property
    def parameter_domain(self) -> _SurfaceDomain:
        return self.source_parameter_domain

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return self.source_periods

    def native_period_symbols(self, /) -> tuple[str | None, str | None]:
        return self.source_period_symbols

    def evaluate(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=jnp.float64)
        control_points, weights = self.control_points, self.weights
        for plan in self.transfer_plans:
            control_points, weights = plan.apply_homogeneous_geometry(
                control_points, weights
            )
        leading = values.shape[:-1]
        jet = self.realization.evaluate_rational(
            values.reshape((-1, 2)), control_points, weights, maximum_order=0
        )
        return jet.values.reshape((*leading, 3))


def _realize_surface_patch(
    patch: AbstractSurfacePatch,
    patch_lookup: dict[int, AbstractSurfacePatch] | None = None,
    /,
    *,
    numeric: AbstractSurfacePatch | None = None,
) -> AbstractSurfacePatch:
    value = patch if numeric is None else numeric
    realized_types = (
        _RealizedPlanePatch,
        _RealizedCylinderPatch,
        _RealizedConePatch,
        _RealizedSpherePatch,
        _RealizedTorusPatch,
        _RealizedBSplineSurfacePatch,
        _RealizedOperationSurface,
        _RealizedPlacedSurface,
        _RealizedTSplineSurfacePatch,
    )
    if isinstance(value, realized_types):
        return value
    if isinstance(patch, realized_types):
        return patch
    from ._patches import (
        BSplineSurfacePatch,
        ConePatch,
        CylinderPatch,
        ExtrusionSurface,
        OffsetSurface,
        PlanePatch,
        RevolutionSurface,
        SpherePatch,
        TorusPatch,
    )
    from ._placed import PlacedSurface

    if isinstance(patch, PlanePatch):
        value = _require_surface_realization(value, PlanePatch)
        return _RealizedPlanePatch(
            value.origin,
            value.first_axis,
            value.second_axis,
            parameter_domain=patch.parameter_domain,
            periods=patch.periods,
            period_symbols=patch.native_period_symbols(),
        )
    if isinstance(patch, CylinderPatch):
        value = _require_surface_realization(value, CylinderPatch)
        return _RealizedCylinderPatch(
            value.origin,
            value.first_axis,
            value.second_axis,
            value.axis,
            value.radius,
            parameter_domain=patch.parameter_domain,
            periods=patch.periods,
            period_symbols=patch.native_period_symbols(),
        )
    if isinstance(patch, ConePatch):
        value = _require_surface_realization(value, ConePatch)
        return _RealizedConePatch(
            value.origin,
            value.first_axis,
            value.second_axis,
            value.axis,
            value.reference_radius,
            value.semi_angle,
        )
    if isinstance(patch, SpherePatch):
        value = _require_surface_realization(value, SpherePatch)
        return _RealizedSpherePatch(
            value.center,
            value.first_axis,
            value.second_axis,
            value.axis,
            value.radius,
        )
    if isinstance(patch, TorusPatch):
        value = _require_surface_realization(value, TorusPatch)
        return _RealizedTorusPatch(
            value.center,
            value.first_axis,
            value.second_axis,
            value.axis,
            value.major_radius,
            value.minor_radius,
            parameter_domain=patch.parameter_domain,
            periods=patch.periods,
            period_symbols=patch.native_period_symbols(),
        )
    if isinstance(patch, BSplineSurfacePatch):
        value = _require_surface_realization(value, BSplineSurfacePatch)
        return _RealizedBSplineSurfacePatch(
            value.control_points,
            value.weights,
            value.u_knots,
            value.v_knots,
            patch.u_degree,
            patch.v_degree,
            parameter_domain=patch.parameter_domain,
            periods=patch.periods,
            period_symbols=patch.native_period_symbols(),
        )
    if isinstance(patch, OffsetSurface):
        value = _require_surface_realization(value, OffsetSurface)
        base = _realize_surface_patch(patch.base, patch_lookup, numeric=value.base)
        return _RealizedOperationSurface(
            "offset",
            base=base,
            distance=value.distance,
            frame_sign=patch.frame_sign,
            parameter_domain=patch.parameter_domain,
            periods=patch.periods,
            period_symbols=patch.native_period_symbols(),
        )
    if isinstance(patch, ExtrusionSurface):
        value = _require_surface_realization(value, ExtrusionSurface)
        return _RealizedOperationSurface(
            "extrusion",
            curve=_realize_curve(patch.curve, patch_lookup, numeric=value.curve),
            direction=value.direction,
            parameter_domain=patch.parameter_domain,
            periods=patch.periods,
            period_symbols=patch.native_period_symbols(),
        )
    if isinstance(patch, RevolutionSurface):
        value = _require_surface_realization(value, RevolutionSurface)
        return _RealizedOperationSurface(
            "revolution",
            curve=_realize_curve(patch.curve, patch_lookup, numeric=value.curve),
            axis_origin=value.axis_origin,
            axis_direction=value.axis_direction,
            parameter_domain=patch.parameter_domain,
            periods=patch.periods,
            period_symbols=patch.native_period_symbols(),
        )
    if isinstance(patch, PlacedSurface):
        value = _require_surface_realization(value, PlacedSurface)
        definition = (
            patch_lookup.get(id(patch.definition)) if patch_lookup is not None else None
        )
        if definition is None:
            definition = _realize_surface_patch(
                patch.definition, patch_lookup, numeric=value.definition
            )
        return _RealizedPlacedSurface(
            definition,
            value.rotation,
            value.translation,
            patch.definition.parameter_domain,
        )
    from ._tspline import TSplineSurfacePatch

    if isinstance(patch, TSplineSurfacePatch):
        value = _require_surface_realization(value, TSplineSurfacePatch)
        return _RealizedTSplineSurfacePatch(patch, value.control_points, value.weights)
    raise NotImplementedError(
        f"A numeric-only CAD realization for {type(patch).__name__} is unavailable."
    )


def _realize_curve(
    curve: AbstractCurve,
    patch_lookup: dict[int, AbstractSurfacePatch] | None = None,
    curve_lookup: dict[int, AbstractCurve] | None = None,
    /,
    *,
    numeric: AbstractCurve | None = None,
) -> AbstractCurve:
    value = curve if numeric is None else numeric
    realized_types = (
        _RealizedTrimEdge,
        _RealizedLineCurve,
        _RealizedConicCurve,
        _RealizedBSplineCurve,
        _RealizedTrimSegment,
        _RealizedIsoparametricCurve,
        _RealizedPlacedCurve,
        _RealizedIntersectionCurve,
        _RealizedIntersectionPCurve,
    )
    if isinstance(value, realized_types):
        return value
    if isinstance(curve, realized_types):
        return curve
    if curve_lookup is not None and id(curve) in curve_lookup:
        return curve_lookup[id(curve)]
    from ._boolean_core import AffinePCurve, PeriodicPCurve
    from ._intersection_curve import IntersectionCurve, IntersectionPCurve
    from ._patches import (
        BSplineCurve,
        CircleCurve,
        ConicRootLinePCurve,
        EllipseCurve,
        ExactLineCurve,
        HyperbolaCurve,
        LineCurve,
        ParabolaCurve,
        SurfaceIsoparametricCurve,
        SurfaceMappedCurve,
    )
    from ._placed import PlacedCurve

    if isinstance(curve, AffinePCurve):
        value = _require_curve_realization(value, AffinePCurve)
        return _RealizedPlacedCurve(
            _realize_curve(curve.curve, patch_lookup, curve_lookup, numeric=value.curve),
            value.matrix_value,
            value.offset_value,
        )
    if isinstance(curve, PeriodicPCurve):
        value = _require_curve_realization(value, PeriodicPCurve)
        return _RealizedPlacedCurve(
            _realize_curve(
                curve.source_curve,
                patch_lookup,
                curve_lookup,
                numeric=value.source_curve,
            ),
            jnp.eye(2, dtype=value.offset_value.dtype),
            value.offset_value,
        )
    if isinstance(curve, LineCurve):
        value = _require_curve_realization(value, LineCurve)
        return _RealizedLineCurve(
            value.origin, value.direction, parameter_domain=curve.parameter_domain
        )
    if isinstance(curve, ExactLineCurve):
        value = _require_curve_realization(value, ExactLineCurve)
        return _RealizedLineCurve(
            value.first,
            value.last - value.first,
            parameter_domain=curve.parameter_domain,
        )
    if isinstance(curve, CircleCurve):
        value = _require_curve_realization(value, CircleCurve)
        return _RealizedConicCurve(
            "circle",
            value.center,
            value.first_axis,
            value.second_axis,
            value.radius,
            parameter_domain=curve.parameter_domain,
            period=curve.period,
            period_symbol=curve.native_period_symbol(),
        )
    if isinstance(curve, EllipseCurve):
        value = _require_curve_realization(value, EllipseCurve)
        return _RealizedConicCurve(
            "ellipse",
            value.center,
            value.first_axis,
            value.second_axis,
            value.first_radius,
            value.second_radius,
            parameter_domain=curve.parameter_domain,
            period=curve.period,
            period_symbol=curve.native_period_symbol(),
        )
    if isinstance(curve, ParabolaCurve):
        value = _require_curve_realization(value, ParabolaCurve)
        return _RealizedConicCurve(
            "parabola",
            value.vertex,
            value.first_axis,
            value.second_axis,
            value.focal_length,
            parameter_domain=curve.parameter_domain,
        )
    if isinstance(curve, HyperbolaCurve):
        value = _require_curve_realization(value, HyperbolaCurve)
        return _RealizedConicCurve(
            "hyperbola",
            value.center,
            value.first_axis,
            value.second_axis,
            value.first_radius,
            value.second_radius,
            parameter_domain=curve.parameter_domain,
        )
    if isinstance(curve, BSplineCurve):
        value = _require_curve_realization(value, BSplineCurve)
        return _RealizedBSplineCurve(
            value.control_points,
            value.weights,
            value.knots,
            curve.degree,
            parameter_domain=curve.parameter_domain,
            period=curve.period,
            period_symbol=curve.native_period_symbol(),
        )
    if isinstance(curve, ConicRootLinePCurve):
        value = _require_curve_realization(value, ConicRootLinePCurve)
        origin = jnp.stack((value.angle, value.height_origin))
        direction = jnp.stack((jnp.zeros_like(value.angle), value.height_direction))
        return _RealizedLineCurve(
            origin, direction, parameter_domain=curve.parameter_domain
        )
    if isinstance(curve, SurfaceMappedCurve):
        value = _require_curve_realization(value, SurfaceMappedCurve)
        surface = (
            patch_lookup.get(id(curve.surface)) if patch_lookup is not None else None
        )
        if surface is None:
            surface = _realize_surface_patch(
                curve.surface, patch_lookup, numeric=value.surface
            )
        return _RealizedTrimEdge(
            surface,
            _realize_curve(
                curve.pcurve, patch_lookup, curve_lookup, numeric=value.pcurve
            ),
        )
    if isinstance(curve, SurfaceIsoparametricCurve):
        value = _require_curve_realization(value, SurfaceIsoparametricCurve)
        surface = (
            patch_lookup.get(id(curve.surface)) if patch_lookup is not None else None
        )
        if surface is None:
            surface = _realize_surface_patch(
                curve.surface, patch_lookup, numeric=value.surface
            )
        return _RealizedIsoparametricCurve(
            surface,
            curve.fixed_axis,
            jnp.asarray(value.fixed_value, dtype=jnp.float64),
            curve.parameter_range,
        )
    if isinstance(curve, PlacedCurve):
        value = _require_curve_realization(value, PlacedCurve)
        definition = (
            curve_lookup.get(id(curve.definition)) if curve_lookup is not None else None
        )
        if definition is None:
            definition = _realize_curve(
                curve.definition,
                patch_lookup,
                curve_lookup,
                numeric=value.definition,
            )
        return _RealizedPlacedCurve(definition, value.rotation, value.translation)
    if isinstance(curve, IntersectionCurve):
        value = _require_curve_realization(value, IntersectionCurve)
        first = (
            patch_lookup.get(id(curve.first.patch)) if patch_lookup is not None else None
        )
        second = (
            patch_lookup.get(id(curve.second.patch)) if patch_lookup is not None else None
        )
        if first is None:
            first = _realize_surface_patch(
                curve.first.patch, patch_lookup, numeric=value.first.patch
            )
        if second is None:
            second = _realize_surface_patch(
                curve.second.patch, patch_lookup, numeric=value.second.patch
            )
        return _RealizedIntersectionCurve(curve, first, second, value)
    if isinstance(curve, IntersectionPCurve):
        value = _require_curve_realization(value, IntersectionPCurve)
        realized = curve_lookup.get(id(curve.curve)) if curve_lookup is not None else None
        if realized is None:
            realized = _realize_curve(
                curve.curve,
                patch_lookup,
                curve_lookup,
                numeric=value.curve,
            )
        if not isinstance(realized, _RealizedIntersectionCurve):
            raise TypeError(
                "An intersection p-curve requires its realized branch carrier."
            )
        return _RealizedIntersectionPCurve(
            realized, curve.side, curve.first, curve.last, curve.reversed
        )
    raise NotImplementedError(
        f"A numeric-only CAD realization for {type(curve).__name__} is unavailable."
    )


def _realize_trim_domains(
    source_domains: Sequence[TrimDomain | None],
    geometry: BRepGeometry,
    pcurves: tuple[AbstractCurve, ...],
    edge_ranges: Array,
    parameter_bounds: Array,
    /,
) -> tuple[TrimDomain, ...]:
    """Build runtime trims from realized coedge carriers, never query surrogates."""
    if len(source_domains) != len(geometry.face_loops):
        raise ValueError("Source trim domains must preserve the source face table.")
    ranges = edge_ranges
    realized_domains: list[TrimDomain] = []
    for face, source in enumerate(source_domains):
        if source is not None and not isinstance(source, TrimDomain):
            raise TypeError(
                "BRep parameter source trims must be TrimDomain values or None."
            )
        face_loops = geometry.face_loops[face]
        lower = parameter_bounds[face, 0]
        extent = parameter_bounds[face, 1] - lower
        if all(
            isinstance(pcurves[coedge], _RealizedLineCurve)
            for coedge_loop in face_loops
            for coedge in coedge_loop
        ):
            polygon_loops = []
            for coedge_loop in face_loops:
                points = []
                for coedge in coedge_loop:
                    edge = geometry.coedge_edges[coedge]
                    parameter = (
                        ranges[edge, 0]
                        if geometry.coedge_senses[coedge] > 0
                        else ranges[edge, 1]
                    )
                    points.append((pcurves[coedge].evaluate(parameter) - lower) / extent)
                polygon_loops.append(jnp.stack(points))
            realized_domains.append(
                TrimDomain(
                    polygon_loops[0],
                    tuple(polygon_loops[1:]),
                    _numeric_only=True,
                )
            )
            continue
        loops: list[tuple[_RealizedTrimSegment, ...]] = []
        for coedge_loop in face_loops:
            segments = []
            for coedge in coedge_loop:
                edge = geometry.coedge_edges[coedge]
                normalized = _RealizedNormalizedCurve(pcurves[coedge], lower, extent)
                segments.append(
                    _RealizedTrimSegment(
                        normalized,
                        ranges[edge, 0],
                        ranges[edge, 1],
                        reversed=geometry.coedge_senses[coedge] < 0,
                    )
                )
            loops.append(tuple(segments))
        realized_domains.append(TrimDomain.from_curve_loops(tuple(loops)))
    return tuple(realized_domains)


class BRepParameterGeometry(StrictModule):
    """Numeric CAD realization paired with an immutable exact source snapshot.

    The ``trim_domains`` input is the exact source snapshot; ``source_trim_domains``
    retains it while ``trim_domains`` on the constructed value contains rebuilt
    numeric trims from the current realized coedge carriers.
    """

    patches: tuple[AbstractSurfacePatch, ...]
    source_patches: tuple[AbstractSurfacePatch, ...]
    source_geometry: BRepGeometry
    vertex_points: Array
    curves: tuple[AbstractCurve, ...]
    pcurves: tuple[AbstractCurve, ...]
    edge_ranges: Array
    parameter_bounds: Array
    orientation: Array
    source_trim_domains: tuple[TrimDomain | None, ...]
    trim_domains: tuple[TrimDomain, ...]
    triangle_parameters: Array | None

    def __init__(
        self,
        *,
        patches: Iterable[AbstractSurfacePatch],
        source_geometry: BRepGeometry,
        parameter_bounds: ArrayLike,
        orientation: ArrayLike,
        trim_domains: Sequence[TrimDomain | None],
        source_patches: Iterable[AbstractSurfacePatch],
        vertex_points: ArrayLike | None = None,
        curves: Iterable[AbstractCurve] | None = None,
        pcurves: Iterable[AbstractCurve] | None = None,
        edge_ranges: ArrayLike | None = None,
        triangle_parameters: ArrayLike | None = None,
    ) -> None:
        patch_values = tuple(patches)
        if not isinstance(source_geometry, BRepGeometry):
            raise TypeError(
                "BRepParameterGeometry.source_geometry must be a BRepGeometry snapshot."
            )
        source_patches_ = tuple(source_patches)
        if (
            len(patch_values) != len(source_geometry.face_loops)
            or len(patch_values) != len(source_patches_)
            or any(
                not isinstance(patch, AbstractSurfacePatch)
                for patch in (*patch_values, *source_patches_)
            )
        ):
            raise ValueError(
                "BRepParameterGeometry patches must preserve source face incidence."
            )
        patches_ = tuple(
            _realize_surface_patch(source, numeric=value)
            for source, value in zip(source_patches_, patch_values, strict=True)
        )
        patch_lookup = {
            id(source): realized
            for source, realized in zip(source_patches_, patches_, strict=True)
        }
        bounds_ = jnp.asarray(parameter_bounds, dtype=jnp.float64)
        orientation_ = jnp.asarray(orientation, dtype=jnp.float64)
        if bounds_.shape != (len(patches_), 2, 2):
            raise ValueError(
                "BRepParameterGeometry.parameter_bounds must have shape (num_faces, 2, 2)."
            )
        if orientation_.shape != (len(patches_),):
            raise ValueError(
                "BRepParameterGeometry.orientation must have one entry per source face."
            )
        bounds_ = eqx.error_if(
            bounds_,
            jnp.any(~jnp.isfinite(bounds_))
            | jnp.any(bounds_[:, 1, :] <= bounds_[:, 0, :]),
            "BRepParameterGeometry parameter bounds must be finite and nonempty.",
        )
        orientation_ = eqx.error_if(
            orientation_,
            jnp.any(~jnp.isfinite(orientation_)),
            "BRepParameterGeometry orientation values must be finite.",
        )
        source_trims = tuple(trim_domains)
        if len(source_trims) != len(patches_):
            raise ValueError(
                "BRepParameterGeometry.source_trim_domains must have one entry per source face."
            )

        if any(
            trim is not None and not isinstance(trim, TrimDomain) for trim in source_trims
        ):
            raise TypeError(
                "BRepParameterGeometry source trims must be TrimDomain values or None."
            )
        points_ = jnp.asarray(
            source_geometry.vertex_points if vertex_points is None else vertex_points,
            dtype=jnp.float64,
        )
        if points_.shape != source_geometry.vertex_points.shape:
            raise ValueError(
                "Dynamic B-Rep vertex points must preserve source vertex rows."
            )
        points_ = eqx.error_if(
            points_,
            jnp.any(~jnp.isfinite(points_)),
            "Dynamic B-Rep vertex points must be finite.",
        )
        curve_values = tuple(source_geometry.curves if curves is None else curves)
        pcurve_values = tuple(source_geometry.pcurves if pcurves is None else pcurves)
        if len(curve_values) != len(source_geometry.curves) or any(
            not isinstance(curve, AbstractCurve) for curve in curve_values
        ):
            raise ValueError(
                "Dynamic B-Rep curves must preserve source edge-carrier rows."
            )
        if len(pcurve_values) != len(source_geometry.pcurves) or any(
            not isinstance(curve, AbstractCurve) for curve in pcurve_values
        ):
            raise ValueError("Dynamic B-Rep p-curves must preserve source coedge rows.")
        curve_values_ = tuple(
            _realize_curve(source, patch_lookup, numeric=value)
            for source, value in zip(source_geometry.curves, curve_values, strict=True)
        )
        curve_lookup = {
            id(source): realized
            for source, realized in zip(
                source_geometry.curves, curve_values_, strict=True
            )
        }
        pcurves_ = tuple(
            _realize_curve(source, patch_lookup, curve_lookup, numeric=value)
            for source, value in zip(source_geometry.pcurves, pcurve_values, strict=True)
        )
        curve_lookup.update(
            {
                id(source): realized
                for source, realized in zip(
                    source_geometry.pcurves, pcurves_, strict=True
                )
            }
        )
        curves_ = tuple(
            _realize_curve(source, patch_lookup, curve_lookup, numeric=value)
            for source, value in zip(source_geometry.curves, curve_values, strict=True)
        )
        ranges_ = jnp.asarray(
            source_geometry.edge_ranges if edge_ranges is None else edge_ranges,
            dtype=jnp.float64,
        )
        if ranges_.shape != source_geometry.edge_ranges.shape:
            raise ValueError("Dynamic B-Rep edge ranges must preserve source edge rows.")
        nondegenerate_edges = tuple(
            index
            for index, curve_index in enumerate(source_geometry.edge_curves)
            if curve_index >= 0
        )
        checked_ranges = ranges_[jnp.asarray(nondegenerate_edges, dtype=jnp.int32)]
        ranges_ = eqx.error_if(
            ranges_,
            jnp.any(~jnp.isfinite(ranges_))
            | jnp.any(checked_ranges[:, 0] >= checked_ranges[:, 1]),
            "Dynamic B-Rep edge ranges must be finite and increasing.",
        )
        trims = _realize_trim_domains(
            source_trims,
            source_geometry,
            pcurves_,
            ranges_,
            bounds_,
        )
        triangle_parameters_ = (
            None
            if triangle_parameters is None
            else jnp.asarray(triangle_parameters, dtype=jnp.float64)
        )
        if triangle_parameters_ is not None:
            if triangle_parameters_.ndim != 3 or triangle_parameters_.shape[1:] != (3, 2):
                raise ValueError(
                    "Dynamic triangle parameters must have shape (num_triangles, 3, 2)."
                )
            triangle_parameters_ = eqx.error_if(
                triangle_parameters_,
                jnp.any(~jnp.isfinite(triangle_parameters_)),
                "Dynamic triangle parameters must be finite.",
            )
        self.patches = patches_
        self.source_patches = source_patches_
        self.source_geometry = source_geometry
        self.vertex_points = points_
        self.curves = curves_
        self.pcurves = pcurves_
        self.edge_ranges = ranges_
        self.parameter_bounds = bounds_
        self.orientation = orientation_
        self.source_trim_domains = source_trims
        self.trim_domains = trims
        self.triangle_parameters = triangle_parameters_


class AbstractBRepParameterMap(StrictModule):
    """Declare correlated continuous CAD parameters before numeric realization."""

    @abstractmethod
    def compile(
        self, context: _ParameterCollector, model: BRepModel, /
    ) -> AbstractBoundBRepParameterMap:
        raise NotImplementedError


class AbstractBoundBRepParameterMap(StrictModule):
    """Bound CAD parameter declaration evaluated only from its dynamic state."""

    @abstractmethod
    def realize(self, state: DesignState, /) -> BRepParameterGeometry:
        raise NotImplementedError


_NUMERIC_SOURCE_ARRAY_FIELDS = frozenset(
    {
        "angle",
        "axis_direction",
        "axis_origin",
        "center",
        "chart_end",
        "chart_lower",
        "chart_start",
        "chart_upper",
        "control_points",
        "direction",
        "distance",
        "edge_ranges",
        "extent",
        "first",
        "first_axis",
        "first_parameter",
        "first_radius",
        "focal_length",
        "height_direction",
        "height_origin",
        "knots",
        "last",
        "major_radius",
        "minor_radius",
        "origin",
        "parameters",
        "parameter_bounds",
        "preconditioners",
        "radius",
        "reference_radius",
        "rotation",
        "second_axis",
        "second_parameter",
        "second_radius",
        "semi_angle",
        "translation",
        "u_knots",
        "v_knots",
        "vertex_parameters",
        "vertex_points",
        "vertices",
        "weights",
    }
)


def _bind_numeric_tree(
    context: _ParameterCollector,
    value: _BoundValue,
    feature_id: str,
    role_prefix: str,
    trainable_fields: frozenset[str],
    /,
    *,
    root_field: str | None = None,
) -> _PatchBinding[_BoundValue]:
    path_leaves, tree_definition = jax.tree_util.tree_flatten_with_path(value)
    bindings: list[ParameterBinding | None] = []
    defaults: list[Array] = []
    for path, leaf in path_leaves:
        array = jnp.asarray(leaf)
        field = jax.tree_util.keystr(path).removeprefix(".")
        field = field or ("value" if root_field is None else root_field)
        leaf_name = field.rsplit(".", 1)[-1]
        defaults.append(array)
        if (
            not jnp.issubdtype(array.dtype, jnp.floating)
            or leaf_name not in _NUMERIC_SOURCE_ARRAY_FIELDS
        ):
            bindings.append(None)
            continue
        host = np.asarray(array)
        scale = float(max(np.max(np.abs(host), initial=0.0), 1.0))
        lower = (
            float(np.finfo(np.float64).eps)
            if leaf_name
            in {
                "first_radius",
                "focal_length",
                "major_radius",
                "minor_radius",
                "radius",
                "reference_radius",
                "second_radius",
                "weights",
            }
            else None
        )
        bindings.append(
            context.bind(
                ParameterId(feature_id, field),
                array,
                role=f"{role_prefix}_{field}",
                physical_scale=scale,
                bounds=(lower, None),
                trainable=leaf_name in trainable_fields,
            )
        )
    return _PatchBinding(value, bindings, tree_definition, defaults)


class _PatchCoefficientMap(AbstractBRepParameterMap):
    """Declare patch and incidence arrays for one fixed-topology source."""

    model: BRepModel
    trainable_fields: frozenset[str] = eqx.field(static=True)

    def __init__(self, model: BRepModel, trainable_fields: Iterable[str]) -> None:
        self.model = model
        self.trainable_fields = frozenset(trainable_fields)

    def compile(
        self, context: _ParameterCollector, model: BRepModel, /
    ) -> AbstractBoundBRepParameterMap:
        if model.source_revision != self.model.source_revision:
            raise ValueError(
                "A patch coefficient map must compile against its declared source."
            )
        geometry = model.geometry
        revision = model.source_revision
        patch_bindings = tuple(
            _bind_numeric_tree(
                context,
                patch,
                f"{revision}:face:{face}",
                "brep_face",
                self.trainable_fields,
            )
            for face, patch in enumerate(model.patches)
        )
        curve_bindings = tuple(
            _bind_numeric_tree(
                context,
                curve,
                f"{revision}:edge_curve:{index}",
                "brep_edge_curve",
                self.trainable_fields,
            )
            for index, curve in enumerate(geometry.curves)
        )
        pcurve_bindings = tuple(
            _bind_numeric_tree(
                context,
                curve,
                f"{revision}:coedge_pcurve:{index}",
                "brep_pcurve",
                self.trainable_fields,
            )
            for index, curve in enumerate(geometry.pcurves)
        )
        vertex_points = _bind_numeric_tree(
            context,
            geometry.vertex_points,
            f"{revision}:vertices",
            "brep_vertex",
            self.trainable_fields,
            root_field="vertex_points",
        )
        edge_ranges = _bind_numeric_tree(
            context,
            geometry.edge_ranges,
            f"{revision}:edge_ranges",
            "brep_edge_range",
            self.trainable_fields,
            root_field="edge_ranges",
        )
        parameter_bounds = _bind_numeric_tree(
            context,
            model.parameter_bounds,
            f"{revision}:parameter_bounds",
            "brep_parameter_bounds",
            self.trainable_fields,
            root_field="parameter_bounds",
        )
        triangle_parameters_binding = (
            None
            if model.tessellation is None
            else _bind_numeric_tree(
                context,
                model.tessellation.parameters,
                f"{revision}:tessellation_parameters",
                "brep_tessellation",
                self.trainable_fields,
                root_field="parameters",
            )
        )
        return _BoundPatchCoefficientMap(
            model,
            patch_bindings,
            curve_bindings,
            pcurve_bindings,
            vertex_points,
            edge_ranges,
            parameter_bounds,
            triangle_parameters_binding,
        )


class _BoundPatchCoefficientMap(AbstractBoundBRepParameterMap):
    model: BRepModel
    patch_bindings: tuple[_PatchBinding[AbstractSurfacePatch], ...]
    curve_bindings: tuple[_PatchBinding[AbstractCurve], ...]
    pcurve_bindings: tuple[_PatchBinding[AbstractCurve], ...]
    vertex_points_binding: _PatchBinding[Array]
    edge_ranges_binding: _PatchBinding[Array]
    parameter_bounds_binding: _PatchBinding[Array]
    triangle_parameters_binding: _PatchBinding[Array] | None

    def __init__(
        self,
        model: BRepModel,
        patch_bindings: Iterable[_PatchBinding[AbstractSurfacePatch]],
        curve_bindings: Iterable[_PatchBinding[AbstractCurve]],
        pcurve_bindings: Iterable[_PatchBinding[AbstractCurve]],
        vertex_points_binding: _PatchBinding[Array],
        edge_ranges_binding: _PatchBinding[Array],
        parameter_bounds_binding: _PatchBinding[Array],
        triangle_parameters_binding: _PatchBinding[Array] | None,
    ) -> None:
        self.model = model
        self.patch_bindings = tuple(patch_bindings)
        self.curve_bindings = tuple(curve_bindings)
        self.pcurve_bindings = tuple(pcurve_bindings)
        self.vertex_points_binding = vertex_points_binding
        self.edge_ranges_binding = edge_ranges_binding
        self.parameter_bounds_binding = parameter_bounds_binding
        self.triangle_parameters_binding = triangle_parameters_binding

    def realize(self, state: DesignState, /) -> BRepParameterGeometry:
        return BRepParameterGeometry(
            patches=tuple(binding.realize(state) for binding in self.patch_bindings),
            source_patches=self.model.patches,
            source_geometry=self.model.geometry,
            parameter_bounds=self.parameter_bounds_binding.realize(state),
            orientation=self.model.orientation,
            trim_domains=self.model.trim_domains,
            vertex_points=self.vertex_points_binding.realize(state),
            curves=tuple(binding.realize(state) for binding in self.curve_bindings),
            pcurves=tuple(binding.realize(state) for binding in self.pcurve_bindings),
            edge_ranges=self.edge_ranges_binding.realize(state),
            triangle_parameters=(
                None
                if self.triangle_parameters_binding is None
                else self.triangle_parameters_binding.realize(state)
            ),
        )


class FixedTopologyBRepRealization(StrictModule):
    """Differentiable face and welded-mesh realization at one design state."""

    parameter_geometry: BRepParameterGeometry
    patches: tuple[AbstractSurfacePatch, ...]
    vertices: Array | None
    faces: Array | None
    atlas: BoundaryAtlas
    seam_residual: Array | None

    def __init__(
        self,
        *,
        parameter_geometry: BRepParameterGeometry,
        patches: Iterable[AbstractSurfacePatch],
        vertices: ArrayLike | None,
        faces: ArrayLike | None,
        atlas: BoundaryAtlas,
        seam_residual: ArrayLike | None,
    ) -> None:
        self.parameter_geometry = parameter_geometry
        self.patches = tuple(patches)
        self.vertices = (
            None if vertices is None else jnp.asarray(vertices, dtype=jnp.float64)
        )
        self.faces = None if faces is None else jnp.asarray(faces, dtype=jnp.int32)
        self.atlas = atlas
        self.seam_residual = (
            None
            if seam_residual is None
            else jnp.asarray(seam_residual, dtype=jnp.float64).reshape(())
        )


def _corner_weld_weights(model: BRepModel) -> np.ndarray:
    if model.tessellation is None:
        return np.empty((0,), dtype=np.float64)
    vertex_indices = np.asarray(model.mesh_faces, dtype=np.int32).reshape((-1,))
    face_indices = np.repeat(np.asarray(model.triangle_face_ids, dtype=np.int32), 3)
    pair_counts: dict[tuple[int, int], int] = {}
    incident_faces: dict[int, set[int]] = {}
    for vertex, face in zip(vertex_indices.tolist(), face_indices.tolist(), strict=True):
        pair = (vertex, face)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        incident_faces.setdefault(vertex, set()).add(face)
    return np.asarray(
        [
            1.0 / (pair_counts[(vertex, face)] * len(incident_faces[vertex]))
            for vertex, face in zip(
                vertex_indices.tolist(), face_indices.tolist(), strict=True
            )
        ],
        dtype=np.float64,
    )


def _evaluate_corners(
    patches: tuple[AbstractSurfacePatch, ...],
    model: BRepModel,
    parameter_bounds: Array | None = None,
    triangle_parameters: Array | None = None,
) -> Array:
    bounds_ = model.parameter_bounds if parameter_bounds is None else parameter_bounds
    parameters_ = (
        model.triangle_parameters
        if triangle_parameters is None
        else jnp.asarray(triangle_parameters, dtype=jnp.float64)
    )
    if parameters_.shape != model.triangle_parameters.shape:
        raise ValueError(
            "Dynamic triangle parameters must preserve the source corner rows."
        )
    chart_indices = jnp.repeat(model.triangle_face_ids, 3)
    parameters = parameters_.reshape((-1, 2))
    bounds = bounds_[chart_indices]
    reference = (parameters - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    return BRepBoundaryMap(patches, bounds_).map(chart_indices, reference)


def evaluate_fixed_topology_mesh(
    patches: tuple[AbstractSurfacePatch, ...],
    model: BRepModel,
    /,
    *,
    corner_weights: Array | None = None,
    parameter_bounds: Array | None = None,
    triangle_parameters: Array | None = None,
) -> tuple[Array, Array]:
    """Evaluate CAD faces, then weld shared mesh vertices by face-balanced averaging."""

    if len(patches) != len(model.patches):
        raise ValueError("patches must preserve the imported face count and ordering.")
    weights = (
        jnp.asarray(_corner_weld_weights(model), dtype=jnp.float64)
        if corner_weights is None
        else jnp.asarray(corner_weights, dtype=jnp.float64).reshape((-1,))
    )
    corners = _evaluate_corners(patches, model, parameter_bounds, triangle_parameters)
    vertex_indices = model.mesh_faces.reshape((-1,))
    if weights.shape != (vertex_indices.shape[0],):
        raise ValueError("corner_weights must contain one entry per triangle corner.")
    vertices = (
        jnp.zeros_like(model.mesh_vertices)
        .at[vertex_indices]
        .add(corners * weights[:, None])
    )
    return vertices, model.mesh_faces


def _seam_residual(
    patches: tuple[AbstractSurfacePatch, ...],
    model: BRepModel,
    parameter_bounds: Array | None = None,
    triangle_parameters: Array | None = None,
) -> Array:
    corners = _evaluate_corners(patches, model, parameter_bounds, triangle_parameters)
    vertex_indices = model.mesh_faces.reshape((-1,))
    reference = jnp.zeros_like(model.mesh_vertices)
    reference = reference.at[vertex_indices].add(corners)
    counts = jnp.zeros((model.mesh_vertices.shape[0],), dtype=corners.dtype)
    counts = counts.at[vertex_indices].add(1.0)
    reference = reference / counts[:, None]
    difference = corners - reference[vertex_indices]
    tiny = jnp.finfo(difference.dtype).tiny
    distances = jnp.sqrt(jnp.sum(difference * difference, axis=-1) + tiny) - jnp.sqrt(
        tiny
    )
    return jnp.max(distances)


class FixedTopologyBRepSource(GeometrySource):
    """Differentiable B-Rep source with immutable incidence and optional tessellation."""

    model: BRepModel
    parameter_map: AbstractBRepParameterMap | None
    trainable_fields: frozenset[str] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        model: BRepModel,
        *,
        parameter_map: AbstractBRepParameterMap | None = None,
        trainable_fields: Sequence[str] = (
            "origin",
            "center",
            "radius",
            "reference_radius",
            "semi_angle",
            "major_radius",
            "minor_radius",
            "control_points",
            "weights",
        ),
    ) -> None:
        if parameter_map is not None and not isinstance(
            parameter_map, AbstractBRepParameterMap
        ):
            raise TypeError("parameter_map must be an AbstractBRepParameterMap or None.")
        self.model = model
        self.parameter_map = parameter_map
        self.trainable_fields = frozenset(trainable_fields)

    @property
    def coordinate_contract(self) -> SpatialCoordinateContract:
        return self.model.coordinate_contract

    def _compile(self, context: _ParameterCollector, /) -> GeometryKernel:
        declaration = self.parameter_map or _PatchCoefficientMap(
            self.model, self.trainable_fields
        )
        bound_map = declaration.compile(context, self.model)
        if not isinstance(bound_map, AbstractBoundBRepParameterMap):
            raise TypeError(
                "A BRep parameter map must compile to an AbstractBoundBRepParameterMap."
            )
        return _FixedTopologyBRepKernel(
            self.model,
            bound_map,
            _corner_weld_weights(self.model),
        )


class _FixedTopologyBRepKernel(GeometryKernel):
    model: BRepModel
    bound_parameter_map: AbstractBoundBRepParameterMap
    corner_weights: Array
    query_bvh: TriangleBVH | None

    def __init__(
        self,
        model: BRepModel,
        bound_parameter_map: AbstractBoundBRepParameterMap,
        corner_weights: ArrayLike,
    ) -> None:
        self.model = model
        self.bound_parameter_map = bound_parameter_map
        self.corner_weights = jnp.asarray(corner_weights, dtype=jnp.float64)
        self.query_bvh = (
            None
            if model.tessellation is None
            else TriangleBVH(
                TriangleMesh(
                    model.mesh_vertices, model.mesh_faces, source_id=model.model_id
                )
            )
        )

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def intrinsic_dimension(self) -> int:
        return 3

    @property
    def kind(self) -> GeometryKind:
        return GeometryKind.REGION

    @property
    def capabilities(self) -> frozenset[GeometryCapability]:
        from ._patches import ExactLineCurve, LineCurve

        boundary_sampling = bool(self.model.patches) and all(
            isinstance(curve, (LineCurve, ExactLineCurve))
            for curve in self.model.geometry.pcurves
        )
        capabilities = {GeometryCapability.BOUNDARY_ATLAS}
        if self.query_bvh is not None:
            capabilities.update(
                {
                    GeometryCapability.REGION_QUERY,
                    GeometryCapability.SIGNED_DISTANCE,
                    GeometryCapability.CLOSEST_POINT,
                    GeometryCapability.BOUNDARY_NORMAL,
                    GeometryCapability.INTERIOR_MEASURE,
                    GeometryCapability.BOUNDARY_MEASURE,
                    GeometryCapability.INTERIOR_SAMPLING,
                    GeometryCapability.SEAM_DIAGNOSTICS,
                }
            )
        if boundary_sampling:
            capabilities.add(GeometryCapability.BOUNDARY_SAMPLING)
        return frozenset(capabilities)

    @property
    def field_certificate(self) -> FieldCertificate:
        return FieldCertificate(
            zero_set_accuracy=ZeroSetAccuracy.TOLERANCE_BOUND,
            sign_reliability=SignReliability.LOCAL,
            distance_semantics=DistanceSemantics.APPROXIMATE,
            regularity=FieldRegularity.PIECEWISE_SMOOTH,
            safe_step_factor=1.0,
            validity_region="fixed topology; requires positive Jacobians and compatible seams",
            parameter_differentiable=True,
            provenance=("native_brep", "fixed_topology_realization"),
        )

    def _parameter_geometry(self, state: DesignState, /) -> BRepParameterGeometry:
        value = self.bound_parameter_map.realize(state)
        if not isinstance(value, BRepParameterGeometry):
            raise TypeError(
                "A bound BRep parameter map must realize BRepParameterGeometry."
            )
        from .._mesh_certificates import _same_tree_content

        # Leaf restoration rebuilds the kernel's and the map's model nodes
        # separately, so the snapshot is compared by structure and every leaf.
        source = self.model.geometry
        if (
            not _same_tree_content(value.source_geometry, source)
            or len(value.source_patches) != len(self.model.patches)
            or any(
                not _same_tree_content(actual, expected)
                for actual, expected in zip(
                    value.source_patches, self.model.patches, strict=True
                )
            )
            or len(value.source_trim_domains) != len(self.model.trim_domains)
            or any(
                not _same_tree_content(actual, expected)
                for actual, expected in zip(
                    value.source_trim_domains, self.model.trim_domains, strict=True
                )
            )
        ):
            raise ValueError(
                "A fixed-topology parameter map must retain the exact source snapshot."
            )
        return value

    def _patches(self, state: DesignState) -> tuple[AbstractSurfacePatch, ...]:
        return self._parameter_geometry(state).patches

    def realize(self, state: DesignState, /) -> FixedTopologyBRepRealization:
        parameter_geometry = self._parameter_geometry(state)
        patches = parameter_geometry.patches
        vertices: Array | None = None
        faces: Array | None = None
        seam_residual: Array | None = None
        if self.query_bvh is not None:
            vertices, faces = evaluate_fixed_topology_mesh(
                patches,
                self.model,
                corner_weights=self.corner_weights,
                parameter_bounds=parameter_geometry.parameter_bounds,
                triangle_parameters=parameter_geometry.triangle_parameters,
            )
            seam_residual = _seam_residual(
                patches,
                self.model,
                parameter_geometry.parameter_bounds,
                parameter_geometry.triangle_parameters,
            )
        atlas = BoundaryAtlas(
            BRepBoundaryMap(patches, parameter_geometry.parameter_bounds),
            source_entity_ids=jnp.arange(len(patches), dtype=jnp.int32),
            source_id=self.model.source_id,
            physical_tags=self.model.physical_tags,
            orientation=parameter_geometry.orientation,
            trim_domains=parameter_geometry.trim_domains,
        )
        return FixedTopologyBRepRealization(
            parameter_geometry=parameter_geometry,
            patches=patches,
            vertices=vertices,
            faces=faces,
            atlas=atlas,
            seam_residual=seam_residual,
        )

    def seam_residual(self, state: DesignState, /) -> Array:
        residual = self.realize(state).seam_residual
        if residual is None:
            raise NotImplementedError(
                "A source-only B-Rep has no tessellated seam residual."
            )
        return residual

    def _triangles(self, state: DesignState, /) -> Array:
        realization = self.realize(state)
        if realization.vertices is None or realization.faces is None:
            raise NotImplementedError(
                "This source-only B-Rep has no triangle realization."
            )
        return realization.vertices[realization.faces]

    def _nearest_query(
        self, state: DesignState, points: Array, *, k: int
    ) -> tuple[MeshQueryResult, Array, Array]:
        if self.query_bvh is None:
            raise NotImplementedError(
                "This source-only B-Rep has no tessellated query backend."
            )
        points_ = jnp.asarray(points, dtype=jnp.float64)
        leading = points_.shape[:-1]
        flat = points_.reshape((-1, 3))
        realization = self.realize(state)
        if realization.vertices is None or realization.faces is None:
            raise RuntimeError("A tessellated query lost its triangle realization.")
        tree = self.query_bvh.refit(realization.vertices)
        selected = jax.lax.stop_gradient(tree.nearest_faces(flat, k=k).items)
        selected_triangles = realization.vertices[realization.faces[selected]]
        closest_by_selected = jax.vmap(_closest_points_on_triangles)(
            flat, selected_triangles
        )
        distance_squared = jnp.sum((closest_by_selected - flat[:, None, :]) ** 2, axis=-1)
        closest = closest_by_selected[:, 0]
        triangle = selected_triangles[:, 0]
        normal = jnp.cross(
            triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]
        )
        normal = normal / jnp.linalg.norm(normal, axis=-1, keepdims=True)
        query = MeshQueryResult(
            closest_point=closest.reshape((*leading, 3)),
            distance=jnp.sqrt(distance_squared[:, 0]).reshape(leading),
            face_index=selected[:, 0].reshape(leading),
            normal=normal.reshape((*leading, 3)),
        )
        # Distances are differentiated only after discrete traversal selected IDs.
        return query, triangle, distance_squared

    def _query(self, state: DesignState, points: Array) -> MeshQueryResult:
        return self._nearest_query(state, points, k=1)[0]

    def contains(self, state: DesignState, points: Array, /) -> Array:
        points_ = jnp.asarray(points, dtype=jnp.float64)
        triangles = self._triangles(state)
        first = triangles[:, 0] - points_[..., None, :]
        second = triangles[:, 1] - points_[..., None, :]
        third = triangles[:, 2] - points_[..., None, :]
        numerator = jnp.sum(first * jnp.cross(second, third), axis=-1)
        denominator = (
            jnp.linalg.norm(first, axis=-1)
            * jnp.linalg.norm(second, axis=-1)
            * jnp.linalg.norm(third, axis=-1)
            + jnp.sum(first * second, axis=-1) * jnp.linalg.norm(third, axis=-1)
            + jnp.sum(second * third, axis=-1) * jnp.linalg.norm(first, axis=-1)
            + jnp.sum(third * first, axis=-1) * jnp.linalg.norm(second, axis=-1)
        )
        winding = jnp.sum(2.0 * jnp.arctan2(numerator, denominator), axis=-1)
        return jnp.abs(winding / (4.0 * jnp.pi)) > 0.5

    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        query = self._query(state, points)
        points_ = jnp.asarray(points, dtype=query.closest_point.dtype)
        difference = points_ - query.closest_point
        squared_distance = jnp.sum(difference * difference, axis=-1)
        away_from_boundary = squared_distance > 0.0
        distance = jnp.sqrt(jnp.where(away_from_boundary, squared_distance, 1.0))
        signed_distance = jnp.where(
            self.contains(state, points_),
            -distance,
            distance,
        )
        boundary_linearization = jnp.sum(difference * query.normal, axis=-1)
        return jnp.where(
            away_from_boundary,
            signed_distance,
            boundary_linearization,
        )

    def boundary_normal(self, state: DesignState, points: Array, /) -> Array:
        return self._query(state, points).normal

    def closest_point(self, state: DesignState, points: Array, /) -> ClosestPointResult:
        points_ = jnp.asarray(points, dtype=jnp.float64)
        leading = points_.shape[:-1]
        flat = points_.reshape((-1, 3))
        query, triangle, selected_distance_squared = self._nearest_query(state, flat, k=2)
        unique, regular, margin = triangle_query_evidence(
            flat,
            triangle,
            query.closest_point,
            selected_distance_squared[:, 1],
        )
        return represented_mesh_closest_point(
            points_,
            closest_point=query.closest_point.reshape((*leading, 3)),
            distance=query.distance.reshape(leading),
            normal=query.normal.reshape((*leading, 3)),
            source_entity_id=query.face_index.reshape(leading),
            inside=self.contains(state, points_),
            unique=unique.reshape(leading),
            regular=regular.reshape(leading),
            margin=margin.reshape(leading),
            represented_geometry_id=f"{self.model.model_id}:fixed-topology-query",
            physical_geometry_id=self.model.source_revision,
            exact_to_physical=False,
        )

    def bounds(self, state: DesignState, /) -> Array:
        vertices = self.realize(state).vertices
        if vertices is None:
            raise NotImplementedError(
                "Fixed-topology bounds require a tessellation realization."
            )
        return jnp.stack((jnp.min(vertices, axis=0), jnp.max(vertices, axis=0)))

    def measure(self, state: DesignState, /) -> Array:
        triangles = self._triangles(state)
        return jnp.abs(
            jnp.sum(
                jnp.sum(
                    triangles[:, 0] * jnp.cross(triangles[:, 1], triangles[:, 2]),
                    axis=-1,
                )
            )
            / 6.0
        )

    def boundary_measure(self, state: DesignState, /) -> Array:
        triangles = self._triangles(state)
        return jnp.sum(
            0.5
            * jnp.linalg.norm(
                jnp.cross(
                    triangles[:, 1] - triangles[:, 0],
                    triangles[:, 2] - triangles[:, 0],
                ),
                axis=-1,
            )
        )

    def sample_interior(
        self,
        state: DesignState,
        num_points: int,
        /,
        *,
        key: Array,
        plan: RejectionSamplingPlan | None = None,
    ) -> SamplingResult:
        bounds = self.bounds(state)
        plan_ = RejectionSamplingPlan() if plan is None else plan
        return bounded_rejection_sample(
            lambda proposal_key, count: jax.random.uniform(
                proposal_key,
                (count, 3),
                minval=bounds[0],
                maxval=bounds[1],
                dtype=bounds.dtype,
            ),
            lambda values: self.contains(state, values),
            num_points=num_points,
            point_dimension=3,
            key=key,
            plan=plan_,
            dtype=bounds.dtype,
        )

    def sample_boundary(
        self,
        state: DesignState,
        num_points: int,
        /,
        *,
        key: Array,
    ) -> SamplingResult:
        realization = self.realize(state)
        if not self.model.patches or not all(
            trim is not None and trim.outer is not None and not trim.curve_loops
            for trim in realization.atlas.trim_domains
        ):
            raise NotImplementedError(
                "Fixed-topology B-Rep boundary sampling requires line-bounded chart trims."
            )
        return sample_boundary_atlas(
            realization.atlas,
            num_points,
            key=key,
        )

    def boundary_atlas(self, state: DesignState, /) -> BoundaryAtlas:
        return self.realize(state).atlas


__all__ = [
    "AbstractBRepParameterMap",
    "AbstractBoundBRepParameterMap",
    "BRepParameterGeometry",
    "FixedTopologyBRepRealization",
    "FixedTopologyBRepSource",
    "evaluate_fixed_topology_mesh",
]
