#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Immutable, explicitly parameterized deterministic financial curves."""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..core import Currency, FinanceDate


class CurveRepresentation(str, Enum):
    """Stored node quantity for a deterministic term structure."""

    LOG_DISCOUNT = "log_discount"
    ZERO_RATE = "zero_rate"
    FORWARD_RATE = "forward_rate"
    LOG_SURVIVAL = "log_survival"
    HAZARD_RATE = "hazard_rate"


class InterpolationMethod(str, Enum):
    """Interpolation between adjacent curve nodes."""

    LINEAR = "linear"
    STEP_LEFT = "step_left"
    STEP_RIGHT = "step_right"


class ExtrapolationMode(str, Enum):
    """Behavior outside a curve's node interval."""

    FORBID = "forbid"
    FLAT = "flat"
    LINEAR = "linear"
    FLAT_FORWARD = "flat_forward"


CurveQuantity: TypeAlias = Literal[
    "parameter",
    "log_discount",
    "discount_factor",
    "zero_rate",
    "instantaneous_forward_rate",
    "log_survival",
    "survival_probability",
    "hazard_rate",
]

_YIELD_REPRESENTATIONS = frozenset(
    {
        CurveRepresentation.LOG_DISCOUNT,
        CurveRepresentation.ZERO_RATE,
        CurveRepresentation.FORWARD_RATE,
    }
)
_SURVIVAL_REPRESENTATIONS = frozenset(
    {CurveRepresentation.LOG_SURVIVAL, CurveRepresentation.HAZARD_RATE}
)


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{name} must be a non-empty string.")
    return identifier


def _enum(value: Any, enum_type: type[Enum], name: str, /):
    if isinstance(value, enum_type):
        return value
    allowed_values = tuple(member.value for member in enum_type)
    if value not in allowed_values:
        allowed = ", ".join(allowed_values)
        raise ValueError(f"{name} must be one of: {allowed}.")
    return enum_type(value)


def _real_vector(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    if array.ndim != 1:
        raise ValueError(f"{name} must be a rank-one array.")
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(float)
    return array


def _concrete_numpy(value: Array, /) -> np.ndarray | None:
    if isinstance(value, jax.core.Tracer):
        return None
    return np.asarray(value)


class CurveGrid(StrictModule):
    """Strictly increasing nonnegative year fractions anchored at zero."""

    times: Array
    grid_id: str = eqx.field(static=True)

    def __init__(self, times: ArrayLike, /):
        times_ = _real_vector(times, "times")
        concrete = _concrete_numpy(times_)
        if concrete is None:
            raise TypeError("CurveGrid must be resolved on the host before device use.")
        if concrete.size < 2:
            raise ValueError("CurveGrid requires at least two nodes.")
        if not np.all(np.isfinite(concrete)):
            raise ValueError("CurveGrid times must be finite.")
        if concrete[0] != 0.0:
            raise ValueError("CurveGrid must be anchored exactly at time zero.")
        if np.any(np.diff(concrete) <= 0.0):
            raise ValueError("CurveGrid times must be strictly increasing.")
        payload = concrete.astype(np.float64, copy=False).tobytes()
        self.times = times_
        self.grid_id = hashlib.sha256(payload).hexdigest()

    @property
    def node_count(self) -> int:
        return int(self.times.shape[0])


class InterpolationPolicy(StrictModule):
    """Explicit interpolation and independent left/right extrapolation."""

    method: InterpolationMethod = eqx.field(static=True)
    left_extrapolation: ExtrapolationMode = eqx.field(static=True)
    right_extrapolation: ExtrapolationMode = eqx.field(static=True)

    def __init__(
        self,
        method: InterpolationMethod | str,
        /,
        *,
        left_extrapolation: ExtrapolationMode | str,
        right_extrapolation: ExtrapolationMode | str,
    ):
        self.method = _enum(method, InterpolationMethod, "method")
        self.left_extrapolation = _enum(
            left_extrapolation, ExtrapolationMode, "left_extrapolation"
        )
        self.right_extrapolation = _enum(
            right_extrapolation, ExtrapolationMode, "right_extrapolation"
        )


class CurveDefinition(StrictModule):
    """Static economic identity and resolved numerical topology of one curve."""

    grid: CurveGrid
    interpolation: InterpolationPolicy
    curve_id: str = eqx.field(static=True)
    role: str = eqx.field(static=True)
    valuation_date: FinanceDate = eqx.field(static=True)
    currency: Currency | None = eqx.field(static=True)
    representation: CurveRepresentation = eqx.field(static=True)

    def __init__(
        self,
        *,
        curve_id: str,
        role: str,
        valuation_date: FinanceDate,
        currency: Currency | None,
        representation: CurveRepresentation | str,
        grid: CurveGrid,
        interpolation: InterpolationPolicy,
    ):
        if not isinstance(valuation_date, FinanceDate):
            raise TypeError("valuation_date must be a FinanceDate.")
        if currency is not None and not isinstance(currency, Currency):
            raise TypeError("currency must be a Currency or None.")
        if not isinstance(grid, CurveGrid):
            raise TypeError("grid must be a CurveGrid.")
        if not isinstance(interpolation, InterpolationPolicy):
            raise TypeError("interpolation must be an InterpolationPolicy.")
        self.curve_id = _identifier(curve_id, "curve_id")
        self.role = _identifier(role, "role")
        self.valuation_date = valuation_date
        self.currency = currency
        self.representation = _enum(representation, CurveRepresentation, "representation")
        self.grid = grid
        self.interpolation = interpolation

    @property
    def topology_key(self) -> tuple[Any, ...]:
        return (
            self.curve_id,
            self.role,
            repr(self.valuation_date),
            repr(self.currency),
            self.representation.value,
            self.grid.grid_id,
            self.interpolation.method.value,
            self.interpolation.left_extrapolation.value,
            self.interpolation.right_extrapolation.value,
        )


def _effective_extrapolation(
    mode: ExtrapolationMode,
    representation: CurveRepresentation,
    /,
) -> ExtrapolationMode:
    if mode is not ExtrapolationMode.FLAT_FORWARD:
        return mode
    if representation in (
        CurveRepresentation.LOG_DISCOUNT,
        CurveRepresentation.LOG_SURVIVAL,
    ):
        return ExtrapolationMode.LINEAR
    return ExtrapolationMode.FLAT


def _bounded_query(
    query: Array,
    grid: Array,
    left: ExtrapolationMode,
    right: ExtrapolationMode,
    /,
) -> Array:
    outside_left = query < grid[0]
    outside_right = query > grid[-1]
    if left is ExtrapolationMode.FORBID:
        query = eqx.error_if(
            query, jnp.any(outside_left), "Curve query is below the first node."
        )
    if right is ExtrapolationMode.FORBID:
        query = eqx.error_if(
            query, jnp.any(outside_right), "Curve query is above the last node."
        )
    return query


def _interpolate_inside(
    grid: Array,
    values: Array,
    query: Array,
    method: InterpolationMethod,
    /,
) -> Array:
    count = int(grid.shape[0])
    clipped = jnp.clip(query, grid[0], grid[-1])
    lower = jnp.clip(jnp.searchsorted(grid, clipped, side="right") - 1, 0, count - 2)
    upper = lower + 1
    if method is InterpolationMethod.LINEAR:
        fraction = (clipped - grid[lower]) / (grid[upper] - grid[lower])
        return values[lower] + fraction * (values[upper] - values[lower])
    if method is InterpolationMethod.STEP_LEFT:
        return jnp.where(clipped == grid[-1], values[-1], values[lower])
    return jnp.where(clipped == grid[0], values[0], values[upper])


def _endpoint_slope(
    grid: Array, values: Array, *, left: bool, method: InterpolationMethod
) -> Array:
    if method is not InterpolationMethod.LINEAR:
        return jnp.asarray(0.0, dtype=values.dtype)
    if left:
        return (values[1] - values[0]) / (grid[1] - grid[0])
    return (values[-1] - values[-2]) / (grid[-1] - grid[-2])


def _interpolate_parameter(
    definition: CurveDefinition,
    node_values: Array,
    times: Array,
    /,
) -> Array:
    grid = definition.grid.times
    policy = definition.interpolation
    left = _effective_extrapolation(policy.left_extrapolation, definition.representation)
    right = _effective_extrapolation(
        policy.right_extrapolation, definition.representation
    )
    times = _bounded_query(times, grid, left, right)
    result = _interpolate_inside(grid, node_values, times, policy.method)
    left_distance = times - grid[0]
    right_distance = times - grid[-1]
    if left is ExtrapolationMode.LINEAR:
        result = jnp.where(
            times < grid[0],
            node_values[0]
            + _endpoint_slope(grid, node_values, left=True, method=policy.method)
            * left_distance,
            result,
        )
    if right is ExtrapolationMode.LINEAR:
        result = jnp.where(
            times > grid[-1],
            node_values[-1]
            + _endpoint_slope(grid, node_values, left=False, method=policy.method)
            * right_distance,
            result,
        )
    return result


def _parameter_derivative(
    definition: CurveDefinition,
    node_values: Array,
    times: Array,
    /,
) -> Array:
    grid = definition.grid.times
    policy = definition.interpolation
    left = _effective_extrapolation(policy.left_extrapolation, definition.representation)
    right = _effective_extrapolation(
        policy.right_extrapolation, definition.representation
    )
    times = _bounded_query(times, grid, left, right)
    count = int(grid.shape[0])
    clipped = jnp.clip(times, grid[0], grid[-1])
    lower = jnp.clip(jnp.searchsorted(grid, clipped, side="right") - 1, 0, count - 2)
    if policy.method is InterpolationMethod.LINEAR:
        derivative = (node_values[lower + 1] - node_values[lower]) / (
            grid[lower + 1] - grid[lower]
        )
    else:
        derivative = jnp.zeros_like(clipped)
    left_slope = _endpoint_slope(grid, node_values, left=True, method=policy.method)
    right_slope = _endpoint_slope(grid, node_values, left=False, method=policy.method)
    derivative = jnp.where(
        times < grid[0],
        left_slope if left is ExtrapolationMode.LINEAR else 0.0,
        derivative,
    )
    return jnp.where(
        times > grid[-1],
        right_slope if right is ExtrapolationMode.LINEAR else 0.0,
        derivative,
    )


def _integral_to_time(
    definition: CurveDefinition,
    node_values: Array,
    times: Array,
    /,
) -> Array:
    """Integrate an instantaneous forward/hazard representation from zero."""

    grid = definition.grid.times
    policy = definition.interpolation
    left = _effective_extrapolation(policy.left_extrapolation, definition.representation)
    right = _effective_extrapolation(
        policy.right_extrapolation, definition.representation
    )
    times = _bounded_query(times, grid, left, right)
    widths = jnp.diff(grid)
    used = jnp.clip(times[..., None] - grid[:-1], 0.0, widths)
    if policy.method is InterpolationMethod.LINEAR:
        slopes = jnp.diff(node_values) / widths
        pieces = node_values[:-1] * used + 0.5 * slopes * used**2
    elif policy.method is InterpolationMethod.STEP_LEFT:
        pieces = node_values[:-1] * used
    else:
        pieces = node_values[1:] * used
    result = jnp.sum(pieces, axis=-1)
    beyond = jnp.maximum(times - grid[-1], 0.0)
    if right is ExtrapolationMode.LINEAR:
        slope = _endpoint_slope(grid, node_values, left=False, method=policy.method)
        extension = node_values[-1] * beyond + 0.5 * slope * beyond**2
    else:
        extension = node_values[-1] * beyond
    result = result + extension
    before = jnp.minimum(times - grid[0], 0.0)
    if left is ExtrapolationMode.LINEAR:
        slope = _endpoint_slope(grid, node_values, left=True, method=policy.method)
        result = result + node_values[0] * before + 0.5 * slope * before**2
    else:
        result = result + node_values[0] * before
    return result


class CurveSensitivity(StrictModule):
    """Value and derivative with respect to an explicit ordered input layout."""

    values: Array
    jacobian: Array
    input_ids: tuple[str, ...] = eqx.field(static=True)
    quantity: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        jacobian: ArrayLike,
        /,
        *,
        input_ids: tuple[str, ...],
        quantity: CurveQuantity,
    ):
        values_ = jnp.asarray(values)
        jacobian_ = jnp.asarray(jacobian)
        if jacobian_.shape != values_.shape + (len(input_ids),):
            raise ValueError(
                "Sensitivity Jacobian must append exactly one ordered input axis."
            )
        self.values = values_
        self.jacobian = jacobian_
        self.input_ids = tuple(_identifier(value, "input_id") for value in input_ids)
        self.quantity = str(quantity)


class PreparedCurve(StrictModule):
    """A curve definition paired with one dynamic numeric realization."""

    definition: CurveDefinition
    node_values: Array
    node_quote_jacobian: Array | None
    quote_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        definition: CurveDefinition,
        node_values: ArrayLike,
        /,
        *,
        node_quote_jacobian: ArrayLike | None = None,
        quote_ids: tuple[str, ...] = (),
    ):
        if not isinstance(definition, CurveDefinition):
            raise TypeError("definition must be a CurveDefinition.")
        nodes = _real_vector(node_values, "node_values")
        if nodes.shape != definition.grid.times.shape:
            raise ValueError("node_values shape must match the curve grid.")
        nodes = eqx.error_if(
            nodes,
            jnp.any(~jnp.isfinite(nodes)),
            "Curve node values must be finite.",
        )
        representation = definition.representation
        if representation is CurveRepresentation.LOG_DISCOUNT:
            nodes = eqx.error_if(
                nodes,
                jnp.abs(nodes[0]) > 1e-12,
                "A log-discount curve must have log discount zero at time zero.",
            )
        elif representation is CurveRepresentation.LOG_SURVIVAL:
            nodes = eqx.error_if(
                nodes,
                (jnp.abs(nodes[0]) > 1e-12)
                | jnp.any(nodes > 1e-12)
                | jnp.any(jnp.diff(nodes) > 1e-12),
                "Log survival must start at zero and be nonincreasing and nonpositive.",
            )
        elif representation is CurveRepresentation.HAZARD_RATE:
            nodes = eqx.error_if(
                nodes,
                jnp.any(nodes < 0.0),
                "Hazard-rate nodes must be nonnegative.",
            )
        ids = tuple(_identifier(value, "quote_id") for value in quote_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("quote_ids must be unique.")
        if node_quote_jacobian is None:
            if ids:
                raise ValueError("quote_ids require an explicit node_quote_jacobian.")
            quote_jacobian = None
        else:
            quote_jacobian = jnp.asarray(node_quote_jacobian)
            if quote_jacobian.shape != (definition.grid.node_count, len(ids)):
                raise ValueError(
                    "node_quote_jacobian shape must be (node_count, quote_count)."
                )
            quote_jacobian = eqx.error_if(
                quote_jacobian,
                jnp.any(~jnp.isfinite(quote_jacobian)),
                "node_quote_jacobian must be finite.",
            )
        self.definition = definition
        self.node_values = nodes
        self.node_quote_jacobian = quote_jacobian
        self.quote_ids = ids

    @classmethod
    def from_discount_factors(
        cls,
        definition: CurveDefinition,
        discount_factors: ArrayLike,
        /,
    ) -> PreparedCurve:
        if definition.representation is not CurveRepresentation.LOG_DISCOUNT:
            raise ValueError("from_discount_factors requires a log-discount definition.")
        factors = _real_vector(discount_factors, "discount_factors")
        concrete = _concrete_numpy(factors)
        if concrete is None:
            raise TypeError("Discount-factor conversion must be resolved on the host.")
        if not np.all(np.isfinite(concrete)) or np.any(concrete <= 0.0):
            raise ValueError("Discount factors must be finite and strictly positive.")
        return cls(definition, jnp.log(factors))

    @classmethod
    def from_survival_probabilities(
        cls,
        definition: CurveDefinition,
        survival_probabilities: ArrayLike,
        /,
    ) -> PreparedCurve:
        if definition.representation is not CurveRepresentation.LOG_SURVIVAL:
            raise ValueError(
                "from_survival_probabilities requires a log-survival definition."
            )
        probabilities = _real_vector(survival_probabilities, "survival_probabilities")
        concrete = _concrete_numpy(probabilities)
        if concrete is None:
            raise TypeError("Survival conversion must be resolved on the host.")
        if not np.all(np.isfinite(concrete)) or np.any(concrete <= 0.0):
            raise ValueError(
                "Survival probabilities must be finite and strictly positive."
            )
        if np.any(concrete > 1.0 + 1e-12) or np.any(np.diff(concrete) > 1e-12):
            raise ValueError(
                "Survival probabilities must not exceed one or increase with time."
            )
        return cls(definition, jnp.log(probabilities))

    def refresh_numeric(
        self,
        node_values: ArrayLike,
        /,
        *,
        node_quote_jacobian: ArrayLike | None = None,
        quote_ids: tuple[str, ...] | None = None,
    ) -> PreparedCurve:
        ids = self.quote_ids if quote_ids is None else tuple(quote_ids)
        if ids != self.quote_ids:
            raise ValueError("Numeric refresh must preserve the quote layout.")
        jacobian = (
            self.node_quote_jacobian
            if node_quote_jacobian is None
            else node_quote_jacobian
        )
        if (jacobian is None) != (self.node_quote_jacobian is None):
            raise ValueError(
                "Numeric refresh must preserve quote-sensitivity availability."
            )
        return PreparedCurve(
            self.definition,
            node_values,
            node_quote_jacobian=jacobian,
            quote_ids=self.quote_ids,
        )

    def _query_times(self, times: ArrayLike, /) -> Array:
        query = jnp.asarray(times)
        if jnp.issubdtype(query.dtype, jnp.complexfloating):
            raise TypeError("Curve query times must be real-valued.")
        if not jnp.issubdtype(query.dtype, jnp.inexact):
            query = query.astype(float)
        return eqx.error_if(
            query,
            jnp.any(~jnp.isfinite(query)) | jnp.any(query < 0.0),
            "Curve query times must be finite and nonnegative.",
        )

    def parameter(self, times: ArrayLike, /) -> Array:
        query = self._query_times(times)
        return _interpolate_parameter(self.definition, self.node_values, query)

    def log_discount(self, times: ArrayLike, /) -> Array:
        if self.definition.representation not in _YIELD_REPRESENTATIONS:
            raise ValueError("log_discount requires a yield-curve representation.")
        query = self._query_times(times)
        representation = self.definition.representation
        if representation is CurveRepresentation.LOG_DISCOUNT:
            return _interpolate_parameter(self.definition, self.node_values, query)
        if representation is CurveRepresentation.ZERO_RATE:
            return -query * _interpolate_parameter(
                self.definition, self.node_values, query
            )
        return -_integral_to_time(self.definition, self.node_values, query)

    def discount_factor(self, times: ArrayLike, /) -> Array:
        return jnp.exp(self.log_discount(times))

    def zero_rate(self, times: ArrayLike, /) -> Array:
        query = self._query_times(times)
        log_discount = self.log_discount(query)
        return jnp.where(
            query == 0.0,
            self.instantaneous_forward_rate(query),
            -log_discount / query,
        )

    def instantaneous_forward_rate(self, times: ArrayLike, /) -> Array:
        if self.definition.representation not in _YIELD_REPRESENTATIONS:
            raise ValueError(
                "instantaneous_forward_rate requires a yield-curve representation."
            )
        query = self._query_times(times)
        representation = self.definition.representation
        if representation is CurveRepresentation.FORWARD_RATE:
            return _interpolate_parameter(self.definition, self.node_values, query)
        if representation is CurveRepresentation.LOG_DISCOUNT:
            return -_parameter_derivative(self.definition, self.node_values, query)
        zero = _interpolate_parameter(self.definition, self.node_values, query)
        zero_derivative = _parameter_derivative(self.definition, self.node_values, query)
        return zero + query * zero_derivative

    def forward_rate(
        self,
        start_times: ArrayLike,
        end_times: ArrayLike,
        /,
        *,
        accrual_fractions: ArrayLike | None = None,
    ) -> Array:
        start = self._query_times(start_times)
        end = self._query_times(end_times)
        start, end = jnp.broadcast_arrays(start, end)
        end = eqx.error_if(
            end,
            jnp.any(end <= start),
            "Forward-rate end times must be later than start times.",
        )
        accrual = (
            end - start if accrual_fractions is None else jnp.asarray(accrual_fractions)
        )
        accrual = jnp.broadcast_to(accrual, start.shape)
        accrual = eqx.error_if(
            accrual,
            jnp.any(~jnp.isfinite(accrual)) | jnp.any(accrual <= 0.0),
            "Forward-rate accrual fractions must be finite and positive.",
        )
        return (self.discount_factor(start) / self.discount_factor(end) - 1.0) / accrual

    def log_survival(self, times: ArrayLike, /) -> Array:
        if self.definition.representation not in _SURVIVAL_REPRESENTATIONS:
            raise ValueError("log_survival requires a survival-curve representation.")
        query = self._query_times(times)
        if self.definition.representation is CurveRepresentation.LOG_SURVIVAL:
            return _interpolate_parameter(self.definition, self.node_values, query)
        return -_integral_to_time(self.definition, self.node_values, query)

    def survival_probability(self, times: ArrayLike, /) -> Array:
        return jnp.exp(self.log_survival(times))

    def hazard_rate(self, times: ArrayLike, /) -> Array:
        if self.definition.representation not in _SURVIVAL_REPRESENTATIONS:
            raise ValueError("hazard_rate requires a survival-curve representation.")
        query = self._query_times(times)
        if self.definition.representation is CurveRepresentation.HAZARD_RATE:
            return _interpolate_parameter(self.definition, self.node_values, query)
        return -_parameter_derivative(self.definition, self.node_values, query)

    def evaluate(self, times: ArrayLike, /, *, quantity: CurveQuantity) -> Array:
        if quantity == "parameter":
            return self.parameter(times)
        if quantity == "log_discount":
            return self.log_discount(times)
        if quantity == "discount_factor":
            return self.discount_factor(times)
        if quantity == "zero_rate":
            return self.zero_rate(times)
        if quantity == "instantaneous_forward_rate":
            return self.instantaneous_forward_rate(times)
        if quantity == "log_survival":
            return self.log_survival(times)
        if quantity == "survival_probability":
            return self.survival_probability(times)
        if quantity == "hazard_rate":
            return self.hazard_rate(times)
        raise ValueError(f"Unsupported curve quantity {quantity!r}.")

    def node_sensitivity(
        self, times: ArrayLike, /, *, quantity: CurveQuantity
    ) -> CurveSensitivity:
        query = self._query_times(times)

        def evaluated(nodes):
            refreshed = eqx.tree_at(lambda curve: curve.node_values, self, nodes)
            return refreshed.evaluate(query, quantity=quantity)

        values = evaluated(self.node_values)
        jacobian = jax.jacfwd(evaluated)(self.node_values)
        node_ids = tuple(
            f"{self.definition.curve_id}:node:{index}"
            for index in range(self.definition.grid.node_count)
        )
        return CurveSensitivity(
            values,
            jacobian,
            input_ids=node_ids,
            quantity=quantity,
        )

    def quote_sensitivity(
        self, times: ArrayLike, /, *, quantity: CurveQuantity
    ) -> CurveSensitivity:
        if self.node_quote_jacobian is None:
            raise ValueError("This curve has no calibrated quote sensitivity.")
        node = self.node_sensitivity(times, quantity=quantity)
        jacobian = jnp.tensordot(
            node.jacobian,
            self.node_quote_jacobian,
            axes=((-1,), (0,)),
        )
        return CurveSensitivity(
            node.values,
            jacobian,
            input_ids=self.quote_ids,
            quantity=quantity,
        )


class CurveSet(StrictModule):
    """Ordered, uniquely identified curves with no implicit role selection."""

    curves: tuple[PreparedCurve, ...]
    curve_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(self, curves: tuple[PreparedCurve, ...], /):
        curves_ = tuple(curves)
        if not curves_:
            raise ValueError("CurveSet requires at least one curve.")
        if any(not isinstance(curve, PreparedCurve) for curve in curves_):
            raise TypeError("CurveSet entries must be PreparedCurve instances.")
        ids = tuple(curve.definition.curve_id for curve in curves_)
        if len(set(ids)) != len(ids):
            raise ValueError("CurveSet curve identifiers must be unique.")
        self.curves = curves_
        self.curve_ids = ids

    def curve(self, curve_id: str, /) -> PreparedCurve:
        identifier = _identifier(curve_id, "curve_id")
        if identifier not in self.curve_ids:
            raise KeyError(f"Curve {identifier!r} is not present in this CurveSet.")
        return self.curves[self.curve_ids.index(identifier)]

    def refresh_numeric(self, curves: tuple[PreparedCurve, ...], /) -> CurveSet:
        refreshed = tuple(curves)
        if len(refreshed) != len(self.curves):
            raise ValueError("Numeric refresh must preserve the curve count.")
        for previous, current in zip(self.curves, refreshed, strict=True):
            if previous.definition.topology_key != current.definition.topology_key:
                raise ValueError("Numeric refresh must preserve every curve topology.")
            if previous.quote_ids != current.quote_ids:
                raise ValueError("Numeric refresh must preserve every quote layout.")
            if previous.node_values.shape != current.node_values.shape:
                raise ValueError("Numeric refresh must preserve every node layout.")
        return CurveSet(refreshed)


__all__ = [
    "CurveDefinition",
    "CurveGrid",
    "CurveQuantity",
    "CurveRepresentation",
    "CurveSensitivity",
    "CurveSet",
    "ExtrapolationMode",
    "InterpolationMethod",
    "InterpolationPolicy",
    "PreparedCurve",
]
