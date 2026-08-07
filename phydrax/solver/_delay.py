#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from math import isinf, prod
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jax import core as jax_core
from jaxtyping import Array, ArrayLike

from .._frozendict import frozendict
from .._strict import StrictModule
from ..integration import (
    GaussLegendreRule,
    interval_rule_data,
    IntervalRule,
)
from ..metrix import AbstractStateGeometry
from ._differential import DifferentialInterpretation, NoiseStructure


DelayHistory: TypeAlias = Callable[[Array, Any], ArrayLike]
DelayHistoryDerivative: TypeAlias = Callable[[Array, Any], ArrayLike]
DelayVectorField: TypeAlias = Callable[[Array, Array, "DelayValues", Any], ArrayLike]
StateDependentLag: TypeAlias = Callable[[Array, Array, Any], ArrayLike]
DistributedDelayKernel: TypeAlias = Callable[[Array, Array, Array, Any], ArrayLike]
DistributedDelayReducer: TypeAlias = Callable[
    [Array, Array, Array, Array, Array, Array, Any], ArrayLike
]
HistoryFunctional: TypeAlias = Callable[
    [Array, Array, "DelayHistoryWindow", Any], ArrayLike
]
NeutralFunctional: TypeAlias = Callable[[Array, "DelayValues", Any], ArrayLike]
EndpointNeutralFunctional: TypeAlias = Callable[
    [Array, Array, "DelayValues", Any], ArrayLike
]
NeutralRecoveryGuess: TypeAlias = Callable[
    [Array, Array, "DelayValues", Any], ArrayLike
]


class DelayValues(StrictModule):
    """Named delay observations supplied to a delay vector field."""

    values: tuple[Array, ...]
    names: tuple[str, ...] = eqx.field(static=True)

    def __init__(self, names: Sequence[str], values: Sequence[ArrayLike], /):
        resolved_names = tuple(names)
        resolved_values = tuple(jnp.asarray(value) for value in values)
        if len(resolved_names) != len(resolved_values):
            raise ValueError("DelayValues needs one value per name.")
        if len(set(resolved_names)) != len(resolved_names):
            raise ValueError("DelayValues names must be unique.")
        self.names = resolved_names
        self.values = resolved_values

    def __getitem__(self, key: str | int, /) -> Array:
        if isinstance(key, str):
            if key not in self.names:
                raise KeyError(key)
            return self.values[self.names.index(key)]
        return self.values[key]

    def __len__(self) -> int:
        return len(self.values)

    @property
    def stacked(self) -> Array:
        """Stack values in declaration order."""
        return jnp.stack(self.values, axis=0)


class DelayHistoryWindow(StrictModule):
    """Declared lag-coordinate view supplied to a history functional.

    ``value(lag)`` evaluates ``y(time - lag)``. Vectorized ``values`` and the
    corresponding derivative methods accept arbitrary finite lag-array shapes.
    The upper bound may be positive infinity for an infinite-memory functional.
    """

    history: Any
    time: Array
    minimum_delay: Array
    maximum_delay: Array

    def __init__(
        self,
        history: Any,
        time: ArrayLike,
        minimum_delay: ArrayLike,
        maximum_delay: ArrayLike,
        /,
    ):
        self.history = history
        self.time = jnp.asarray(time)
        self.minimum_delay = jnp.asarray(minimum_delay)
        self.maximum_delay = jnp.asarray(maximum_delay)

    def _checked_lags(self, lags: ArrayLike, /) -> Array:
        values = jnp.asarray(lags)
        if jnp.iscomplexobj(values):
            raise TypeError("Functional delay lags must be real.")
        values = values.astype(jnp.result_type(self.time, float))
        return eqx.error_if(
            values,
            ~jnp.all(jnp.isfinite(values))
            | jnp.any(values < self.minimum_delay)
            | jnp.any(values > self.maximum_delay),
            "A history functional queried outside its declared lag interval.",
        )

    def value(self, lag: ArrayLike, /, *, left: bool = False) -> Array:
        checked = self._checked_lags(lag)
        if checked.shape != ():
            raise ValueError("DelayHistoryWindow.value requires a scalar lag.")
        return self.history.value(self.time - checked, left=left)

    def values(self, lags: ArrayLike, /, *, left: bool = False) -> Array:
        checked = self._checked_lags(lags)
        return self.history.values(self.time - checked, left=left)

    def derivative(self, lag: ArrayLike, /, *, left: bool = False) -> Array:
        checked = self._checked_lags(lag)
        if checked.shape != ():
            raise ValueError("DelayHistoryWindow.derivative requires a scalar lag.")
        return self.history.derivative(self.time - checked, left=left)

    def derivatives(self, lags: ArrayLike, /, *, left: bool = False) -> Array:
        checked = self._checked_lags(lags)
        return self.history.derivatives(self.time - checked, left=left)

    def __call__(self, lag: ArrayLike, /, *, left: bool = False) -> Array:
        return self.value(lag, left=left)

    @property
    def lag_interval(self) -> tuple[Array, Array]:
        return self.minimum_delay, self.maximum_delay


class ConstantDelay(StrictModule):
    """One positive constant point delay."""

    name: str = eqx.field(static=True)
    delay: Array

    def __init__(self, name: str, delay: ArrayLike, /):
        if not isinstance(name, str) or not name:
            raise ValueError("ConstantDelay name must be a non-empty string.")
        value = jnp.asarray(delay, dtype=float)
        if value.shape != ():
            raise ValueError("ConstantDelay delay must be scalar.")
        value = eqx.error_if(
            value,
            ~jnp.isfinite(value) | (value <= 0.0),
            "ConstantDelay delay must be finite and positive.",
        )
        self.name = name
        self.delay = value

    @property
    def minimum_delay(self) -> Array:
        return self.delay

    @property
    def maximum_delay(self) -> Array:
        return self.delay


class StateDependentDelay(StrictModule):
    """Point delay with certified lag bounds and delayed-argument topology.

    A nonmonotone delayed argument must declare ``root_isolation_step``: a
    maximum time interval on which each source crossing is sign-isolating
    (there cannot be two crossings with equal endpoint signs).
    """

    name: str = eqx.field(static=True)
    lag: StateDependentLag
    minimum_delay: Array
    maximum_delay: Array | None
    monotone_argument: bool = eqx.field(static=True)
    root_isolation_step: Array | None

    def __init__(
        self,
        name: str,
        lag: StateDependentLag,
        /,
        *,
        minimum_delay: ArrayLike,
        maximum_delay: ArrayLike | None = None,
        monotone_argument: bool = True,
        root_isolation_step: ArrayLike | None = None,
    ):
        if not isinstance(name, str) or not name:
            raise ValueError("StateDependentDelay name must be a non-empty string.")
        if not callable(lag):
            raise TypeError("StateDependentDelay lag must be callable.")
        if not isinstance(monotone_argument, bool):
            raise TypeError("monotone_argument must be a bool.")
        if not monotone_argument and root_isolation_step is None:
            raise ValueError(
                "Nonmonotone delayed arguments require root_isolation_step."
            )
        lower = jnp.asarray(minimum_delay, dtype=float)
        if lower.shape != ():
            raise ValueError("minimum_delay must be scalar.")
        lower = eqx.error_if(
            lower,
            ~jnp.isfinite(lower) | (lower <= 0.0),
            "minimum_delay must be finite and positive.",
        )
        if maximum_delay is None:
            upper = None
        else:
            upper = jnp.asarray(maximum_delay, dtype=float)
            if upper.shape != ():
                raise ValueError("maximum_delay must be scalar or None.")
            upper = eqx.error_if(
                upper,
                ~jnp.isfinite(upper) | (upper < lower),
                "maximum_delay must be finite and at least minimum_delay.",
            )
        if root_isolation_step is None:
            isolation_step = None
        else:
            isolation_step = jnp.asarray(root_isolation_step, dtype=float)
            if isolation_step.shape != ():
                raise ValueError("root_isolation_step must be scalar or None.")
            isolation_step = eqx.error_if(
                isolation_step,
                ~jnp.isfinite(isolation_step) | (isolation_step <= 0.0),
                "root_isolation_step must be finite and positive.",
            )
        self.name = name
        self.lag = lag
        self.minimum_delay = lower
        self.maximum_delay = upper
        self.monotone_argument = monotone_argument
        self.root_isolation_step = isolation_step

    def value(self, time: Array, state: Array, args: Any, /) -> Array:
        delay = jnp.asarray(self.lag(time, state, args), dtype=float)
        if delay.shape != ():
            raise ValueError("State-dependent lag functions must return a scalar.")
        invalid = ~jnp.isfinite(delay) | (delay < self.minimum_delay)
        if self.maximum_delay is not None:
            invalid = invalid | (delay > self.maximum_delay)
        return eqx.error_if(
            delay,
            invalid,
            f"State-dependent delay {self.name!r} violated its declared bounds.",
        )


class FunctionalDelay(StrictModule):
    """Arbitrary functional of a bounded or infinite lag-coordinate history.

    The callback receives ``(time, state, history, args)``. ``history`` only
    permits queries within the declared positive lag interval. Optional
    ``discontinuity_lags`` declare exact lag translations along which known
    discontinuities should be propagated.
    """

    name: str = eqx.field(static=True)
    functional: HistoryFunctional
    minimum_delay: Array
    maximum_delay: Array
    output_kind: Literal["ambient", "point", "tangent"] = eqx.field(static=True)
    discontinuity_lags: Array
    infinite_memory: bool = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        functional: HistoryFunctional,
        lag_interval: tuple[ArrayLike, ArrayLike],
        /,
        *,
        output_kind: Literal["ambient", "point", "tangent"] = "ambient",
        discontinuity_lags: ArrayLike | None = None,
    ):
        if not isinstance(name, str) or not name:
            raise ValueError("FunctionalDelay name must be a non-empty string.")
        if not callable(functional):
            raise TypeError("FunctionalDelay functional must be callable.")
        if not isinstance(lag_interval, tuple) or len(lag_interval) != 2:
            raise TypeError("FunctionalDelay lag_interval must be a pair.")
        if output_kind not in ("ambient", "point", "tangent"):
            raise ValueError(
                "FunctionalDelay output_kind must be 'ambient', 'point', or 'tangent'."
            )
        lower = jnp.asarray(lag_interval[0], dtype=float)
        upper = jnp.asarray(lag_interval[1], dtype=float)
        if lower.shape != () or upper.shape != ():
            raise ValueError("FunctionalDelay lag interval bounds must be scalar.")
        lower = eqx.error_if(
            lower,
            ~jnp.isfinite(lower) | (lower <= 0.0),
            "FunctionalDelay minimum delay must be finite and positive.",
        )
        raw_upper = lag_interval[1]
        infinite_memory = (
            isinstance(raw_upper, (int, float)) and isinf(float(raw_upper))
        ) or (
            eqx.is_array(raw_upper)
            and not isinstance(raw_upper, jax_core.Tracer)
            and bool(jax.device_get(jnp.isposinf(raw_upper)))
        )
        upper = eqx.error_if(
            upper,
            jnp.isnan(upper) | jnp.isneginf(upper) | (upper < lower),
            "FunctionalDelay maximum delay must be at least its minimum or +inf.",
        )
        if discontinuity_lags is None:
            propagation_lags = jnp.empty((0,), dtype=lower.dtype)
        else:
            propagation_lags = jnp.asarray(discontinuity_lags, dtype=lower.dtype)
            if propagation_lags.ndim != 1:
                raise ValueError("discontinuity_lags must be a rank-1 array or None.")
            propagation_lags = eqx.error_if(
                propagation_lags,
                ~jnp.all(jnp.isfinite(propagation_lags))
                | jnp.any(propagation_lags < lower)
                | jnp.any(propagation_lags > upper),
                "FunctionalDelay discontinuity_lags must lie in its lag interval.",
            )
        self.name = name
        self.functional = functional
        self.minimum_delay = lower
        self.maximum_delay = upper
        self.output_kind = output_kind
        self.discontinuity_lags = propagation_lags
        self.infinite_memory = infinite_memory


class DistributedDelay(StrictModule):
    """Static-quadrature distributed delay over a finite positive lag interval.

    A custom ``reducer`` receives ``(time, state, lags, weights, kernel_values,
    delayed_values, args)``. It is required when the problem state geometry is
    nontrivial, because a Euclidean weighted sum of manifold points is undefined.
    """

    name: str = eqx.field(static=True)
    kernel: DistributedDelayKernel
    reducer: DistributedDelayReducer | None
    lower_lag: Array
    upper_lag: Array
    nodes: Array
    weights: Array
    quadrature: IntervalRule

    def __init__(
        self,
        name: str,
        kernel: DistributedDelayKernel,
        lag_interval: tuple[ArrayLike, ArrayLike],
        /,
        *,
        quadrature: IntervalRule | None = None,
        reducer: DistributedDelayReducer | None = None,
    ):
        if not isinstance(name, str) or not name:
            raise ValueError("DistributedDelay name must be a non-empty string.")
        if not callable(kernel):
            raise TypeError("DistributedDelay kernel must be callable.")
        if reducer is not None and not callable(reducer):
            raise TypeError("DistributedDelay reducer must be callable or None.")
        if not isinstance(lag_interval, tuple) or len(lag_interval) != 2:
            raise TypeError("DistributedDelay lag_interval must be a pair.")
        raw_lower = jnp.asarray(lag_interval[0])
        raw_upper = jnp.asarray(lag_interval[1])
        if jnp.iscomplexobj(raw_lower) or jnp.iscomplexobj(raw_upper):
            raise TypeError("DistributedDelay lag interval bounds must be real.")
        lower = raw_lower.astype(float)
        upper = raw_upper.astype(float)
        if lower.shape != () or upper.shape != ():
            raise ValueError("DistributedDelay lag interval bounds must be scalar.")
        lower = eqx.error_if(
            lower,
            ~jnp.isfinite(lower) | (lower < 0.0),
            "DistributedDelay lower lag must be finite and nonnegative.",
        )
        upper = eqx.error_if(
            upper,
            ~jnp.isfinite(upper) | (upper <= lower),
            "DistributedDelay upper lag must be finite and exceed its lower lag.",
        )

        rule = GaussLegendreRule(16) if quadrature is None else quadrature
        data = interval_rule_data(rule)
        reference_nodes = jnp.asarray(data.nodes)
        reference_weights = jnp.asarray(data.weights)
        if jnp.iscomplexobj(reference_nodes) or jnp.iscomplexobj(reference_weights):
            raise TypeError("DistributedDelay quadrature nodes and weights must be real.")
        if (
            reference_nodes.ndim != 1
            or reference_weights.shape != reference_nodes.shape
            or int(reference_nodes.size) == 0
        ):
            raise ValueError(
                "DistributedDelay quadrature needs equally sized, non-empty "
                "rank-1 nodes and weights."
            )
        reference_nodes = eqx.error_if(
            reference_nodes,
            ~(
                jnp.all(jnp.isfinite(reference_nodes))
                & jnp.all(jnp.isfinite(reference_weights))
            ),
            "DistributedDelay quadrature nodes and weights must be finite.",
        )
        reference_nodes = eqx.error_if(
            reference_nodes,
            jnp.any((reference_nodes < -1.0) | (reference_nodes > 1.0)),
            "DistributedDelay quadrature nodes must lie in the reference interval.",
        )
        span = upper - lower
        nodes = lower + 0.5 * span * (reference_nodes + 1.0)
        weights = 0.5 * span * reference_weights
        nodes = eqx.error_if(
            nodes,
            jnp.any(~jnp.isfinite(nodes) | (nodes <= 0.0)),
            "DistributedDelay quadrature nodes must query finite positive lags.",
        )
        weights = eqx.error_if(
            weights,
            jnp.any(~jnp.isfinite(weights)),
            "DistributedDelay quadrature weights must be finite.",
        )
        self.name = name
        self.kernel = kernel
        self.reducer = reducer
        self.lower_lag = lower
        self.upper_lag = upper
        self.nodes = nodes
        self.weights = weights
        self.quadrature = rule

    @property
    def minimum_delay(self) -> Array:
        return jnp.min(self.nodes)

    @property
    def maximum_delay(self) -> Array:
        return jnp.max(self.nodes)

    @property
    def quadrature_family(self) -> str:
        return type(self.quadrature).__name__

    @property
    def quadrature_order(self) -> int:
        return int(self.quadrature.order)

    @property
    def node_count(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def effective_lag_range(self) -> tuple[Array, Array]:
        return self.minimum_delay, self.maximum_delay


def _distributed_delay_value(
    term: DistributedDelay,
    time: Array,
    state: Array,
    delayed_values: Array,
    args: Any,
    state_shape: tuple[int, ...],
    /,
) -> Array:
    expected_values = (term.node_count,) + state_shape
    if delayed_values.shape != expected_values:
        raise ValueError(
            f"Distributed history values must stack to shape {expected_values}; "
            f"got {delayed_values.shape}."
        )
    kernels = jax.vmap(lambda lag: jnp.asarray(term.kernel(time, lag, state, args)))(
        term.nodes
    )
    scalar_kernel_shape = (term.node_count,)
    scalar_kernel = kernels.shape == scalar_kernel_shape
    state_kernel = kernels.shape == expected_values
    if not scalar_kernel and not state_kernel:
        raise ValueError(
            f"DistributedDelay {term.name!r} kernel must return a scalar or "
            f"the exact state shape {state_shape}."
        )
    if term.reducer is None:
        if scalar_kernel:
            weighted = (
                kernels.reshape(scalar_kernel_shape + (1,) * len(state_shape))
                * delayed_values
            )
        else:
            weighted = kernels * delayed_values
        value = jnp.tensordot(term.weights, weighted, axes=((0,), (0,)))
    else:
        value = jnp.asarray(
            term.reducer(
                time,
                state,
                term.nodes,
                term.weights,
                kernels,
                delayed_values,
                args,
            )
        )
    if value.shape != state_shape:
        raise ValueError(
            f"DistributedDelay {term.name!r} reducer must return state shape "
            f"{state_shape}; got {value.shape}."
        )
    return value


PointDelay: TypeAlias = ConstantDelay | StateDependentDelay


class DerivativeDelay(StrictModule):
    """First derivative of one delayed history value.

    ``transport`` maps ``(delayed_state, current_state, delayed_derivative, args)``
    to a state-shaped tangent at ``current_state``. It is mandatory when the
    problem has non-Euclidean state geometry.
    """

    name: str = eqx.field(static=True)
    delay: PointDelay
    transport: Callable[[Array, Array, Array, Any], ArrayLike] | None

    def __init__(
        self,
        name: str,
        delay: PointDelay,
        /,
        *,
        transport: Callable[[Array, Array, Array, Any], ArrayLike] | None = None,
    ):
        if not isinstance(name, str) or not name:
            raise ValueError("DerivativeDelay name must be a non-empty string.")
        if not isinstance(delay, (ConstantDelay, StateDependentDelay)):
            raise TypeError("DerivativeDelay requires a point delay.")
        if transport is not None and not callable(transport):
            raise TypeError("transport must be callable or None.")
        self.name = name
        self.delay = delay
        self.transport = transport

    @property
    def minimum_delay(self) -> Array:
        return self.delay.minimum_delay

    @property
    def maximum_delay(self) -> Array | None:
        return self.delay.maximum_delay


DelayTerm: TypeAlias = (
    ConstantDelay
    | StateDependentDelay
    | FunctionalDelay
    | DistributedDelay
    | DerivativeDelay
)


class DelayWienerTerm(StrictModule):
    """One named Wiener source whose coefficient may use delayed observations."""

    name: str = eqx.field(static=True)
    coefficient: DelayVectorField
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    structure: NoiseStructure = eqx.field(static=True)
    basis_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        coefficient: DelayVectorField,
        noise_shape: Sequence[int],
        /,
        *,
        structure: NoiseStructure = "general",
        basis_id: str | None = None,
    ):
        if not isinstance(name, str) or not name:
            raise ValueError("DelayWienerTerm name must be a non-empty string.")
        if not callable(coefficient):
            raise TypeError("DelayWienerTerm coefficient must be callable.")
        shape = tuple(int(size) for size in noise_shape)
        if any(size <= 0 for size in shape):
            raise ValueError("DelayWienerTerm noise dimensions must be positive.")
        if structure not in ("additive", "commutative", "general"):
            raise ValueError(
                "DelayWienerTerm structure must be 'additive', 'commutative', or 'general'."
            )
        if basis_id is not None and (not isinstance(basis_id, str) or not basis_id):
            raise ValueError("DelayWienerTerm basis_id must be non-empty or None.")
        self.name = name
        self.coefficient = coefficient
        self.noise_shape = shape
        self.structure = structure
        self.basis_id = basis_id

    @property
    def noise_size(self) -> int:
        return prod(self.noise_shape) if self.noise_shape else 1


def _delay_noise_identity(terms: tuple[DelayWienerTerm, ...], /) -> str | None:
    if not terms or all(term.basis_id is None for term in terms):
        return None
    if len(terms) == 1:
        return terms[0].basis_id
    digest = hashlib.sha256()
    digest.update(b"phydrax-delay-wiener-terms\0")
    for term in terms:
        digest.update(
            repr((term.name, term.noise_shape, term.structure, term.basis_id)).encode(
                "utf-8"
            )
        )
        digest.update(b"\0")
    return digest.hexdigest()


def _point_lag(term: PointDelay, time: Array, state: Array, args: Any, /) -> Array:
    if isinstance(term, ConstantDelay):
        return term.delay
    return term.value(time, state, args)


def _invalid_geometry_tangent(
    geometry: AbstractStateGeometry,
    point: Array,
    tangent: Array,
    state_shape: tuple[int, ...],
    /,
) -> Array:
    projected = jnp.asarray(geometry.project_tangent(point, tangent))
    if projected.shape != state_shape:
        raise ValueError("State geometry tangent projection changed the state shape.")
    comparison_dtype = jnp.result_type(tangent, projected, float)
    tangent_value = tangent.astype(comparison_dtype)
    projected_value = projected.astype(comparison_dtype)
    scale = jnp.maximum(
        1.0,
        jnp.maximum(
            jnp.max(jnp.abs(tangent_value)),
            jnp.max(jnp.abs(projected_value)),
        ),
    )
    tolerance = 256.0 * jnp.finfo(comparison_dtype).eps * scale
    return (
        ~jnp.all(jnp.isfinite(tangent_value))
        | ~jnp.all(jnp.isfinite(projected_value))
        | jnp.any(jnp.abs(tangent_value - projected_value) > tolerance)
    )


def _validated_geometry_point(
    geometry: AbstractStateGeometry,
    point: Array,
    message: str,
    /,
) -> Array:
    membership = jnp.asarray(geometry.contains(point), dtype=bool)
    if membership.shape != ():
        raise ValueError("State geometry contains() must return a scalar boolean.")
    return eqx.error_if(point, ~membership, message)


def _validated_geometry_tangent(
    geometry: AbstractStateGeometry,
    point: Array,
    tangent: Array,
    state_shape: tuple[int, ...],
    message: str,
    /,
) -> Array:
    return eqx.error_if(
        tangent,
        _invalid_geometry_tangent(geometry, point, tangent, state_shape),
        message,
    )


class _InitialDelayHistoryView(eqx.Module):
    history: DelayHistory
    history_derivative: DelayHistoryDerivative | None
    args: Any
    state_shape: tuple[int, ...] = eqx.field(static=True)
    geometry: AbstractStateGeometry | None

    def value(self, time: Array, /, *, left: bool = False) -> Array:
        del left
        value = jnp.asarray(self.history(time, self.args))
        if value.shape != self.state_shape:
            raise ValueError("Delay history changed its declared state shape.")
        if self.geometry is not None:
            value = _validated_geometry_point(
                self.geometry,
                value,
                "A delayed history value lies outside state_geometry.",
            )
        return value

    def values(self, times: Array, /, *, left: bool = False) -> Array:
        query = jnp.asarray(times)
        values = jax.vmap(lambda item: self.value(item, left=left))(query.reshape((-1,)))
        return values.reshape(query.shape + self.state_shape)

    def derivative(self, time: Array, /, *, left: bool = False) -> Array:
        del left
        if self.history_derivative is None:
            raise ValueError("Delay history does not define a derivative callback.")
        value = jnp.asarray(self.history_derivative(time, self.args))
        if value.shape != self.state_shape:
            raise ValueError("Delay history derivative changed its declared shape.")
        if self.geometry is not None and not self.geometry.trivial:
            point = self.value(time)
            value = _validated_geometry_tangent(
                self.geometry,
                point,
                value,
                self.state_shape,
                "A delayed history derivative is not tangent to state_geometry.",
            )
        return value

    def derivatives(self, times: Array, /, *, left: bool = False) -> Array:
        query = jnp.asarray(times)
        values = jax.vmap(lambda item: self.derivative(item, left=left))(
            query.reshape((-1,))
        )
        return values.reshape(query.shape + self.state_shape)


def _initial_term_value(
    term: DelayTerm,
    time: Array,
    state: Array,
    history: DelayHistory,
    history_derivative: DelayHistoryDerivative | None,
    args: Any,
    state_shape: tuple[int, ...],
    state_geometry: AbstractStateGeometry | None,
    /,
) -> Array:
    if isinstance(term, FunctionalDelay):
        initial_view = _InitialDelayHistoryView(
            history=history,
            history_derivative=history_derivative,
            args=args,
            state_shape=state_shape,
            geometry=state_geometry,
        )
        window = DelayHistoryWindow(
            initial_view,
            time,
            term.minimum_delay,
            term.maximum_delay,
        )
        value = jnp.asarray(term.functional(time, state, window, args))
        if value.shape != state_shape:
            raise ValueError(
                f"FunctionalDelay {term.name!r} must return state shape "
                f"{state_shape}; got {value.shape}."
            )
        if state_geometry is not None and term.output_kind == "point":
            value = _validated_geometry_point(
                state_geometry,
                value,
                f"FunctionalDelay {term.name!r} returned a point outside "
                "state_geometry.",
            )
        if (
            state_geometry is not None
            and not state_geometry.trivial
            and term.output_kind == "tangent"
        ):
            value = _validated_geometry_tangent(
                state_geometry,
                state,
                value,
                state_shape,
                f"FunctionalDelay {term.name!r} returned a non-tangent value.",
            )
        return value
    if isinstance(term, DistributedDelay):
        values = jax.vmap(lambda lag: jnp.asarray(history(time - lag, args)))(term.nodes)
        expected_values = (term.node_count,) + state_shape
        if values.shape != expected_values:
            raise ValueError(
                f"Distributed history values must stack to shape {expected_values}; "
                f"got {values.shape}."
            )
        if state_geometry is not None:
            memberships = jax.vmap(
                lambda value: jnp.asarray(state_geometry.contains(value), dtype=bool)
            )(values)
            expected_memberships = (term.node_count,)
            if memberships.shape != expected_memberships:
                raise ValueError(
                    "State geometry contains() must return a scalar boolean."
                )
            values = eqx.error_if(
                values,
                ~jnp.all(memberships),
                f"DistributedDelay {term.name!r} queried initial history outside "
                "state_geometry.",
            )
        return _distributed_delay_value(
            term,
            time,
            state,
            values,
            args,
            state_shape,
        )
    if isinstance(term, DerivativeDelay):
        if history_derivative is None:
            raise ValueError("history_derivative is required by DerivativeDelay terms.")
        lag = _point_lag(term.delay, time, state, args)
        delayed_state = jnp.asarray(history(time - lag, args))
        derivative = jnp.asarray(history_derivative(time - lag, args))
        if delayed_state.shape != state_shape or derivative.shape != state_shape:
            raise ValueError(
                f"DerivativeDelay {term.name!r} history must preserve state shape "
                f"{state_shape}."
            )
        if state_geometry is not None:
            delayed_state = _validated_geometry_point(
                state_geometry,
                delayed_state,
                f"DerivativeDelay {term.name!r} queried initial history outside "
                "state_geometry.",
            )
            if not state_geometry.trivial:
                derivative = _validated_geometry_tangent(
                    state_geometry,
                    delayed_state,
                    derivative,
                    state_shape,
                    f"DerivativeDelay {term.name!r} queried a history derivative "
                    "that is not tangent at the delayed state.",
                )
        if term.transport is not None:
            derivative = jnp.asarray(
                term.transport(delayed_state, state, derivative, args)
            )
        if derivative.shape != state_shape:
            raise ValueError(
                f"DerivativeDelay {term.name!r} transport changed the state shape."
            )
        if state_geometry is not None and not state_geometry.trivial:
            derivative = _validated_geometry_tangent(
                state_geometry,
                state,
                derivative,
                state_shape,
                f"DerivativeDelay {term.name!r} transport must return a tangent at "
                "the current state.",
            )
        return derivative
    lag = _point_lag(term, time, state, args)
    delayed_state = jnp.asarray(history(time - lag, args))
    if delayed_state.shape != state_shape:
        raise ValueError(
            f"Delay term {term.name!r} history must preserve state shape {state_shape}."
        )
    if state_geometry is not None:
        delayed_state = _validated_geometry_point(
            state_geometry,
            delayed_state,
            "A delayed history value lies outside state_geometry.",
        )
    return delayed_state


class DelayDifferentialProblem(StrictModule):
    """Finite-dimensional delay differential problem with declared memory terms."""

    drift: DelayVectorField
    history: DelayHistory
    history_derivative: DelayHistoryDerivative | None
    delay_terms: tuple[DelayTerm, ...]
    initial_state: Array
    t0: Array
    t1: Array
    args: Any
    wiener_terms: tuple[DelayWienerTerm, ...]
    wiener_term_slices: frozendict[str, tuple[int, int]] = eqx.field(static=True)
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    noise_id: str | None = eqx.field(static=True)
    interpretation: DifferentialInterpretation = eqx.field(static=True)
    state_geometry: AbstractStateGeometry | None
    state_geometry_id: str | None = eqx.field(static=True)
    initial_left_derivative: Array | None
    initial_right_derivative: Array
    initial_derivative_jump: Array | None
    initial_derivative_compatible: Array
    state_shape: tuple[int, ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        drift: DelayVectorField,
        history: DelayHistory,
        delay_terms: Sequence[DelayTerm],
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
        args: Any = None,
        history_derivative: DelayHistoryDerivative | None = None,
        wiener_terms: Sequence[DelayWienerTerm] = (),
        interpretation: DifferentialInterpretation = "ito",
        state_geometry: AbstractStateGeometry | None = None,
        problem_id: str = "delay-differential-problem",
    ):
        if not callable(drift) or not callable(history):
            raise TypeError("drift and history must be callable.")
        if history_derivative is not None and not callable(history_derivative):
            raise TypeError("history_derivative must be callable or None.")
        start = jnp.asarray(t0, dtype=float)
        end = jnp.asarray(t1, dtype=float)
        if start.shape != () or end.shape != ():
            raise ValueError("DelayDifferentialProblem t0 and t1 must be scalar.")
        start = eqx.error_if(
            start,
            ~(jnp.isfinite(start) & jnp.isfinite(end)),
            "DelayDifferentialProblem time bounds must be finite.",
        )
        end = eqx.error_if(
            end,
            ~(end > start),
            "DelayDifferentialProblem requires t1 > t0.",
        )
        terms = tuple(delay_terms)
        if not terms or any(
            not isinstance(
                term,
                (
                    ConstantDelay,
                    StateDependentDelay,
                    FunctionalDelay,
                    DistributedDelay,
                    DerivativeDelay,
                ),
            )
            for term in terms
        ):
            raise TypeError("delay_terms must be a non-empty sequence of delay terms.")
        names = tuple(term.name for term in terms)
        if len(set(names)) != len(names):
            raise ValueError("Delay term names must be unique within a problem.")
        if (
            any(isinstance(term, DerivativeDelay) for term in terms)
            and history_derivative is None
        ):
            raise ValueError("history_derivative is required by DerivativeDelay terms.")
        if interpretation not in ("ito", "stratonovich"):
            raise ValueError("interpretation must be 'ito' or 'stratonovich'.")

        state = jnp.asarray(history(start, args))
        state_shape = tuple(int(size) for size in state.shape)
        if state_geometry is not None:
            if not isinstance(state_geometry, AbstractStateGeometry):
                raise TypeError(
                    "state_geometry must be an AbstractStateGeometry or None."
                )
            membership = jnp.asarray(state_geometry.contains(state), dtype=bool)
            if membership.shape != ():
                raise ValueError(
                    "State geometry contains() must return a scalar boolean."
                )
            state = eqx.error_if(
                state,
                ~membership,
                "DelayDifferentialProblem initial state is outside state_geometry.",
            )
            if not state_geometry.trivial and any(
                isinstance(term, DerivativeDelay) and term.transport is None
                for term in terms
            ):
                raise ValueError(
                    "Manifold DerivativeDelay terms require explicit tangent transport."
                )
            if not state_geometry.trivial and any(
                isinstance(term, DistributedDelay) and term.reducer is None
                for term in terms
            ):
                raise ValueError(
                    "Non-Euclidean DistributedDelay terms require an explicit reducer."
                )

        initial_values = tuple(
            _initial_term_value(
                term,
                start,
                state,
                history,
                history_derivative,
                args,
                state_shape,
                state_geometry,
            )
            for term in terms
        )
        checked_initial_values = []
        for term, value in zip(terms, initial_values, strict=True):
            if value.shape != state_shape:
                raise ValueError(
                    f"Delay term {term.name!r} must return state shape {state_shape}; "
                    f"got {value.shape}."
                )
            if state_geometry is not None and isinstance(term, DistributedDelay):
                membership = jnp.asarray(state_geometry.contains(value), dtype=bool)
                if membership.shape != ():
                    raise ValueError(
                        "State geometry contains() must return a scalar boolean."
                    )
                value = eqx.error_if(
                    value,
                    ~membership,
                    f"DistributedDelay {term.name!r} reducer returned a point "
                    "outside state_geometry.",
                )
            checked_initial_values.append(value)
        initial_values = tuple(checked_initial_values)
        memory = DelayValues(names, initial_values)
        drift_value = jnp.asarray(drift(start, state, memory, args))
        if drift_value.shape != state_shape:
            raise ValueError("drift must preserve the history state shape.")
        if state_geometry is not None and not state_geometry.trivial:
            drift_value = _validated_geometry_tangent(
                state_geometry,
                state,
                drift_value,
                state_shape,
                "DelayDifferentialProblem initial drift must be tangent-compatible "
                "with state_geometry.",
            )
        if history_derivative is None:
            initial_left_derivative = None
            initial_derivative_jump = None
            initial_derivative_compatible = jnp.asarray(True)
        else:
            initial_left_derivative = jnp.asarray(history_derivative(start, args))
            if initial_left_derivative.shape != state_shape:
                raise ValueError(
                    "history_derivative must preserve the history state shape."
                )
            if state_geometry is not None and not state_geometry.trivial:
                initial_left_derivative = _validated_geometry_tangent(
                    state_geometry,
                    state,
                    initial_left_derivative,
                    state_shape,
                    "DelayDifferentialProblem history_derivative at t0 must be tangent "
                    "to state_geometry.",
                )
            comparison_dtype = jnp.result_type(
                initial_left_derivative, drift_value, float
            )
            residual = drift_value.astype(
                comparison_dtype
            ) - initial_left_derivative.astype(comparison_dtype)
            scale = jnp.maximum(
                jnp.abs(drift_value.astype(comparison_dtype)),
                jnp.abs(initial_left_derivative.astype(comparison_dtype)),
            )
            tolerance = 32.0 * jnp.finfo(comparison_dtype).eps * (1.0 + scale)
            initial_derivative_jump = residual
            initial_derivative_compatible = jnp.all(jnp.abs(residual) <= tolerance)

        stochastic_terms = tuple(wiener_terms)
        if any(not isinstance(term, DelayWienerTerm) for term in stochastic_terms):
            raise TypeError("wiener_terms must contain only DelayWienerTerm objects.")
        if stochastic_terms and any(isinstance(term, DerivativeDelay) for term in terms):
            raise ValueError("Stochastic neutral delay equations are not supported.")
        stochastic_names = tuple(term.name for term in stochastic_terms)
        if len(set(stochastic_names)) != len(stochastic_names):
            raise ValueError("DelayWienerTerm names must be unique within a problem.")
        offset = 0
        slices: dict[str, tuple[int, int]] = {}
        for term in stochastic_terms:
            coefficient = jnp.asarray(term.coefficient(start, state, memory, args))
            expected = state_shape + term.noise_shape
            if coefficient.shape != expected:
                raise ValueError(
                    f"DelayWienerTerm {term.name!r} coefficient must return shape "
                    f"{expected}; got {coefficient.shape}."
                )
            slices[term.name] = (offset, offset + term.noise_size)
            offset += term.noise_size

        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.drift = drift
        self.history = history
        self.history_derivative = history_derivative
        self.delay_terms = terms
        self.initial_state = state
        self.t0 = start
        self.t1 = end
        self.args = args
        self.wiener_terms = stochastic_terms
        self.wiener_term_slices = frozendict(slices)
        self.noise_shape = (offset,) if stochastic_terms else ()
        self.noise_id = _delay_noise_identity(stochastic_terms)
        self.interpretation = interpretation
        self.state_geometry = state_geometry
        self.state_geometry_id = (
            None if state_geometry is None else state_geometry.geometry_id
        )
        self.state_shape = state_shape
        self.initial_left_derivative = initial_left_derivative
        self.initial_right_derivative = drift_value
        self.initial_derivative_jump = initial_derivative_jump
        self.initial_derivative_compatible = initial_derivative_compatible
        self.problem_id = identifier

    @property
    def stochastic(self) -> bool:
        return bool(self.wiener_terms)

    @property
    def num_delays(self) -> int:
        return len(self.delay_terms)

    @property
    def delay_names(self) -> tuple[str, ...]:
        return tuple(term.name for term in self.delay_terms)

    @property
    def minimum_delay(self) -> Array:
        return jnp.min(
            jnp.stack(tuple(jnp.asarray(term.minimum_delay) for term in self.delay_terms))
        )

    @property
    def maximum_delay(self) -> Array | None:
        if any(
            isinstance(term, FunctionalDelay) and term.infinite_memory
            for term in self.delay_terms
        ):
            return None
        values = tuple(term.maximum_delay for term in self.delay_terms)
        if any(value is None for value in values):
            return None
        return jnp.max(jnp.stack(tuple(jnp.asarray(value) for value in values)))

    @property
    def has_state_dependent_delays(self) -> bool:
        return any(
            isinstance(term, StateDependentDelay)
            or (
                isinstance(term, DerivativeDelay)
                and isinstance(term.delay, StateDependentDelay)
            )
            for term in self.delay_terms
        )

    @property
    def has_functional_delays(self) -> bool:
        return any(isinstance(term, FunctionalDelay) for term in self.delay_terms)


    @property
    def has_distributed_delays(self) -> bool:
        return any(isinstance(term, DistributedDelay) for term in self.delay_terms)

    @property
    def neutral(self) -> bool:
        return any(isinstance(term, DerivativeDelay) for term in self.delay_terms)


class NeutralDelayProblem(StrictModule):
    """Deterministic neutral equation in transformed-state form.

    The solved equation is ``d(y - N_r(t, y_t) - N_e(t, y, y_t))/dt = F``.
    ``neutral_functional`` is retarded. ``endpoint_neutral`` is optional and is
    recovered by a nonlinear solve at every numerical stage and accepted endpoint.
    """

    drift: DelayVectorField
    history: DelayHistory
    history_derivative: DelayHistoryDerivative | None
    delay_terms: tuple[DelayTerm, ...]
    initial_state: Array
    t0: Array
    t1: Array
    args: Any
    wiener_terms: tuple[DelayWienerTerm, ...]
    wiener_term_slices: frozendict[str, tuple[int, int]] = eqx.field(static=True)
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    noise_id: str | None = eqx.field(static=True)
    interpretation: DifferentialInterpretation = eqx.field(static=True)
    state_geometry: AbstractStateGeometry | None
    state_geometry_id: str | None = eqx.field(static=True)
    initial_left_derivative: Array | None
    initial_right_derivative: Array
    initial_derivative_jump: Array | None
    initial_derivative_compatible: Array
    state_shape: tuple[int, ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    neutral_functional: NeutralFunctional
    endpoint_neutral: EndpointNeutralFunctional | None
    transformed_initial_state: Array
    recovery_initial_guess: NeutralRecoveryGuess | None
    recovery_solver: optx.AbstractRootFinder
    recovery_rtol: float = eqx.field(static=True)
    recovery_atol: float = eqx.field(static=True)
    recovery_max_steps: int = eqx.field(static=True)

    def __init__(
        self,
        neutral_functional: NeutralFunctional,
        differential: DelayVectorField,
        history: DelayHistory,
        delay_terms: Sequence[DelayTerm],
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
        args: Any = None,
        endpoint_neutral: EndpointNeutralFunctional | None = None,
        recovery_initial_guess: NeutralRecoveryGuess | None = None,
        recovery_solver: optx.AbstractRootFinder | None = None,
        recovery_rtol: float = 1e-10,
        recovery_atol: float = 1e-12,
        recovery_max_steps: int = 64,
        history_derivative: DelayHistoryDerivative | None = None,
        state_geometry: AbstractStateGeometry | None = None,
        problem_id: str = "neutral-delay-problem",
    ):
        if not callable(neutral_functional):
            raise TypeError("neutral_functional must be callable.")
        if endpoint_neutral is not None and not callable(endpoint_neutral):
            raise TypeError("endpoint_neutral must be callable or None.")
        if recovery_initial_guess is not None and not callable(recovery_initial_guess):
            raise TypeError("recovery_initial_guess must be callable or None.")
        if recovery_rtol < 0.0 or recovery_atol <= 0.0:
            raise ValueError("Recovery tolerances must be nonnegative and positive.")
        if (
            not isinstance(recovery_max_steps, int)
            or isinstance(recovery_max_steps, bool)
            or recovery_max_steps <= 0
        ):
            raise ValueError("recovery_max_steps must be a positive integer.")
        resolved_recovery_solver = (
            optx.Newton(
                rtol=float(recovery_rtol),
                atol=float(recovery_atol),
                norm=optx.rms_norm,
            )
            if recovery_solver is None
            else recovery_solver
        )
        if not isinstance(resolved_recovery_solver, optx.AbstractRootFinder):
            raise TypeError(
                "recovery_solver must be an Optimistix AbstractRootFinder or None."
            )
        constant_terms = []
        for term in delay_terms:
            if not isinstance(term, ConstantDelay):
                raise ValueError(
                    "NeutralDelayProblem currently requires ConstantDelay terms."
                )
            constant_terms.append(term)
        terms = tuple(constant_terms)
        if not terms:
            raise ValueError(
                "NeutralDelayProblem currently requires ConstantDelay terms."
            )
        if state_geometry is not None and not state_geometry.trivial:
            raise ValueError(
                "NeutralDelayProblem requires Euclidean state geometry until a "
                "geometry-specific subtraction contract is supplied."
            )

        base = DelayDifferentialProblem(
            differential,
            history,
            terms,
            t0=t0,
            t1=t1,
            args=args,
            history_derivative=history_derivative,
            state_geometry=state_geometry,
            problem_id=problem_id,
        )
        self.drift = base.drift
        self.history = base.history
        self.history_derivative = base.history_derivative
        self.delay_terms = base.delay_terms
        self.initial_state = base.initial_state
        self.t0 = base.t0
        self.t1 = base.t1
        self.args = base.args
        self.wiener_terms = base.wiener_terms
        self.wiener_term_slices = base.wiener_term_slices
        self.noise_shape = base.noise_shape
        self.noise_id = base.noise_id
        self.interpretation = base.interpretation
        self.state_geometry = base.state_geometry
        self.state_geometry_id = base.state_geometry_id
        self.initial_left_derivative = base.initial_left_derivative
        self.initial_right_derivative = base.initial_right_derivative
        self.initial_derivative_jump = base.initial_derivative_jump
        self.initial_derivative_compatible = base.initial_derivative_compatible
        self.state_shape = base.state_shape
        self.problem_id = base.problem_id
        initial_memory = DelayValues(
            self.delay_names,
            tuple(
                jnp.asarray(self.history(self.t0 - term.delay, self.args))
                for term in terms
            ),
        )
        retarded = jnp.asarray(
            neutral_functional(self.t0, initial_memory, self.args)
        )
        if retarded.shape != self.state_shape:
            raise ValueError("neutral_functional must preserve the state shape.")
        if endpoint_neutral is None:
            endpoint = jnp.zeros_like(self.initial_state)
        else:
            endpoint = jnp.asarray(
                endpoint_neutral(
                    self.t0,
                    self.initial_state,
                    initial_memory,
                    self.args,
                )
            )
            if endpoint.shape != self.state_shape:
                raise ValueError("endpoint_neutral must preserve the state shape.")
        transformed = self.initial_state - retarded - endpoint
        transformed = eqx.error_if(
            transformed,
            ~jnp.all(jnp.isfinite(transformed)),
            "NeutralDelayProblem initial transformed state must be finite.",
        )
        self.neutral_functional = neutral_functional
        self.endpoint_neutral = endpoint_neutral
        self.transformed_initial_state = transformed
        self.recovery_initial_guess = recovery_initial_guess
        self.recovery_rtol = float(recovery_rtol)
        self.recovery_atol = float(recovery_atol)
        self.recovery_solver = resolved_recovery_solver
        self.recovery_max_steps = recovery_max_steps

    @property
    def differential(self) -> DelayVectorField:
        return self.drift

    @property
    def neutral(self) -> bool:
        return True

    @property
    def implicit_recovery(self) -> bool:
        return self.endpoint_neutral is not None


    @property
    def stochastic(self) -> bool:
        return False

    @property
    def num_delays(self) -> int:
        return len(self.delay_terms)

    @property
    def delay_names(self) -> tuple[str, ...]:
        return tuple(term.name for term in self.delay_terms)

    @property
    def minimum_delay(self) -> Array:
        return jnp.min(
            jnp.stack(
                tuple(jnp.asarray(term.minimum_delay) for term in self.delay_terms)
            )
        )

    @property
    def maximum_delay(self) -> Array | None:
        if any(
            isinstance(term, FunctionalDelay) and term.infinite_memory
            for term in self.delay_terms
        ):
            return None
        return jnp.max(
            jnp.stack(
                tuple(jnp.asarray(term.maximum_delay) for term in self.delay_terms)
            )
        )

    @property
    def has_state_dependent_delays(self) -> bool:
        return False

    @property
    def has_functional_delays(self) -> bool:
        return False


    @property
    def has_distributed_delays(self) -> bool:
        return False
__all__ = [
    "ConstantDelay",
    "EndpointNeutralFunctional",
    "DelayDifferentialProblem",
    "DelayHistory",
    "DelayHistoryWindow",
    "DelayHistoryDerivative",
    "DelayTerm",
    "DelayValues",
    "DelayVectorField",
    "DelayWienerTerm",
    "DerivativeDelay",
    "DistributedDelay",
    "FunctionalDelay",
    "HistoryFunctional",
    "DistributedDelayKernel",
    "NeutralDelayProblem",
    "NeutralFunctional",
    "NeutralRecoveryGuess",
    "PointDelay",
    "StateDependentDelay",
    "StateDependentLag",

]
