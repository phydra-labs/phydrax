#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._interpolation import bspline_evaluate, BSplineGrid
from .._strict import StrictModule
from .._trainable import NonTrainableState


DrivingPathSide: TypeAlias = Literal["left", "right"]


def _path_id(value: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("path_id must be a non-empty string.")
    return value


def _side(value: DrivingPathSide, /) -> DrivingPathSide:
    if value not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'.")
    return value


def _support_array(value: ArrayLike, /) -> Array:
    support_raw = jnp.asarray(value)
    if support_raw.shape != (2,):
        raise ValueError("Driving-path support must contain exactly two endpoints.")
    if jnp.issubdtype(support_raw.dtype, jnp.complexfloating):
        raise TypeError("Driving-path support must be real-valued.")
    support_host = np.asarray(support_raw, dtype=float)
    if not np.all(np.isfinite(support_host)) or not support_host[1] > support_host[0]:
        raise ValueError("Driving-path support must be finite and strictly increasing.")
    return support_raw.astype(jnp.result_type(support_raw, float))


def _value_shape(value: tuple[int, ...], /) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if any(size <= 0 for size in shape):
        raise ValueError("Driving-path value_shape dimensions must be positive.")
    return shape


def _checked_time(time: ArrayLike, support: Array, /) -> Array:
    time_ = jnp.asarray(time, dtype=support.dtype).reshape(())
    return eqx.error_if(
        time_,
        ~jnp.isfinite(time_) | (time_ < support[0]) | (time_ > support[1]),
        "Driving-path time is outside its closed support.",
    )


def _breakpoint_schedule(
    breakpoints: ArrayLike,
    breakpoint_mask: ArrayLike,
    support: Array,
    /,
) -> tuple[Array, Array]:
    points_raw = jnp.asarray(breakpoints)
    mask = jnp.asarray(breakpoint_mask, dtype=bool)
    if points_raw.ndim != 1:
        raise ValueError("Driving-path breakpoints must be a rank-one array.")
    if mask.shape != points_raw.shape:
        raise ValueError("breakpoint_mask must have the same shape as breakpoints.")
    if jnp.issubdtype(points_raw.dtype, jnp.complexfloating):
        raise TypeError("Driving-path breakpoints must be real-valued.")
    points = points_raw.astype(jnp.result_type(points_raw, support, float))
    points_host = np.asarray(points, dtype=float)
    mask_host = np.asarray(mask, dtype=bool)
    active = points_host[mask_host]
    support_host = np.asarray(support, dtype=float)
    if not np.all(np.isfinite(active)):
        raise ValueError("Active driving-path breakpoints must be finite.")
    if np.any(active <= support_host[0]) or np.any(active >= support_host[1]):
        raise ValueError("Active driving-path breakpoints must lie inside support.")
    if active.size > 1 and np.any(np.diff(active) <= 0.0):
        raise ValueError("Active driving-path breakpoints must be strictly increasing.")
    return points, mask


def _sample_value_mask(values: Array, value_mask: ArrayLike, /) -> tuple[Array, Array]:
    mask = jnp.asarray(value_mask, dtype=bool)
    if mask.shape == values.shape[:1]:
        return mask, mask
    if mask.shape != values.shape:
        raise ValueError(
            "value_mask must have shape (sample_capacity,) or the complete values shape."
        )
    flat_host = np.asarray(mask, dtype=bool).reshape((values.shape[0], -1))
    row_all = np.all(flat_host, axis=1)
    row_any = np.any(flat_host, axis=1)
    if np.any(row_all != row_any):
        raise ValueError(
            "Every sampled path value must be either wholly valid or wholly invalid."
        )
    return mask, jnp.asarray(row_all, dtype=bool)


def _validated_samples(
    times: ArrayLike,
    values: ArrayLike,
    time_mask: ArrayLike,
    value_mask: ArrayLike,
    /,
    *,
    minimum_samples: int,
) -> tuple[Array, Array, Array, Array, Array, int, tuple[int, ...]]:
    times_raw = jnp.asarray(times)
    values_raw = jnp.asarray(values)
    if times_raw.ndim != 1:
        raise ValueError("Sampled driving-path times must be a rank-one array.")
    if values_raw.ndim < 1 or values_raw.shape[0] != times_raw.shape[0]:
        raise ValueError("Sampled path values must have one leading entry per time.")
    if jnp.issubdtype(times_raw.dtype, jnp.complexfloating):
        raise TypeError("Sampled driving-path times must be real-valued.")
    if not (
        jnp.issubdtype(values_raw.dtype, jnp.number)
        or jnp.issubdtype(values_raw.dtype, jnp.bool_)
    ):
        raise TypeError("Sampled driving-path values must be numeric.")
    value_shape = _value_shape(tuple(int(size) for size in values_raw.shape[1:]))
    capacity = int(times_raw.shape[0])
    if capacity < minimum_samples:
        raise ValueError(
            f"Driving-path fitting requires capacity for at least {minimum_samples} samples."
        )

    times_valid = jnp.asarray(time_mask, dtype=bool)
    if times_valid.shape != times_raw.shape:
        raise ValueError("time_mask must have the same shape as times.")
    values_valid, value_sample_valid = _sample_value_mask(values_raw, value_mask)
    time_host = np.asarray(times_valid, dtype=bool)
    value_host = np.asarray(value_sample_valid, dtype=bool)
    if np.any(np.diff(time_host.astype(np.int8)) > 0):
        raise ValueError("time_mask must be a prefix mask.")
    if np.any(np.diff(value_host.astype(np.int8)) > 0):
        raise ValueError("value_mask must define a prefix of complete values.")

    sample_mask = times_valid & value_sample_valid
    sample_count = int(np.sum(np.asarray(sample_mask, dtype=bool)))
    if sample_count < minimum_samples:
        raise ValueError(
            f"Driving-path fitting requires at least {minimum_samples} valid samples."
        )
    times_host = np.asarray(times_raw[:sample_count], dtype=float)
    values_host = np.asarray(values_raw[:sample_count])
    if not np.all(np.isfinite(times_host)):
        raise ValueError("Valid sampled driving-path times must be finite.")
    if np.any(np.diff(times_host) <= 0.0):
        raise ValueError("Valid sampled driving-path times must be strictly increasing.")
    if not np.all(np.isfinite(values_host)):
        raise ValueError("Valid sampled driving-path values must be finite.")

    times_ = times_raw.astype(jnp.result_type(times_raw, float))
    values_ = values_raw.astype(jnp.result_type(values_raw, float))
    return (
        times_,
        values_,
        times_valid,
        values_valid,
        sample_mask,
        sample_count,
        value_shape,
    )


class DrivingPathFitDiagnostics(StrictModule, NonTrainableState):
    """Explicit validity, residual, method, and approximation provenance for a fit."""

    support: Array
    residual_norm: Array
    maximum_residual: Array
    minimum_spacing: Array
    maximum_spacing: Array
    valid: Array
    status: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    sample_count: int = eqx.field(static=True)
    sample_capacity: int = eqx.field(static=True)
    value_shape: tuple[int, ...] = eqx.field(static=True)
    regularization: float = eqx.field(static=True)


class AbstractDifferentiableDrivingPath(StrictModule):
    """Piecewise-differentiable path on closed support with an explicit jump schedule.

    ``breakpoints`` has JAX-static capacity and ``breakpoint_mask`` identifies exactly
    the interior times where the first derivative can be discontinuous. Values are
    defined at both support endpoints. ``side`` selects the corresponding one-sided
    polynomial or spline derivative at a breakpoint.
    """

    breakpoints: Array
    breakpoint_mask: Array
    value_shape: tuple[int, ...] = eqx.field(static=True)
    path_id: str = eqx.field(static=True)

    @property
    @abstractmethod
    def support(self) -> tuple[Array, Array]:
        """Return the closed lower and upper support endpoints."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, time: ArrayLike, /, side: DrivingPathSide = "right") -> Array:
        raise NotImplementedError

    def increment(
        self, t0: ArrayLike, t1: ArrayLike, /, side: DrivingPathSide = "right"
    ) -> Array:
        """Return the oriented path increment ``X(t1) - X(t0)``."""
        side_ = _side(side)
        return self.evaluate(t1, side_) - self.evaluate(t0, side_)

    @abstractmethod
    def derivative(self, time: ArrayLike, /, side: DrivingPathSide = "right") -> Array:
        raise NotImplementedError


class CallableDrivingPath(AbstractDifferentiableDrivingPath):
    """Declared value and derivative callables with explicit support and jump times."""

    value: Callable[[Array, DrivingPathSide], ArrayLike]
    derivative_value: Callable[[Array, DrivingPathSide], ArrayLike]
    _support: Array

    def __init__(
        self,
        value: Callable[[Array, DrivingPathSide], ArrayLike],
        derivative: Callable[[Array, DrivingPathSide], ArrayLike],
        /,
        *,
        support: ArrayLike,
        value_shape: tuple[int, ...],
        path_id: str,
        breakpoints: ArrayLike,
        breakpoint_mask: ArrayLike,
    ):
        if not callable(value) or not callable(derivative):
            raise TypeError(
                "Declared driving-path value and derivative must be callable."
            )
        support_ = _support_array(support)
        shape = _value_shape(value_shape)
        points, point_mask = _breakpoint_schedule(breakpoints, breakpoint_mask, support_)
        midpoint = jnp.mean(support_)
        declared_value = jnp.asarray(value(midpoint, "right"))
        declared_derivative = jnp.asarray(derivative(midpoint, "right"))
        if declared_value.shape != shape or declared_derivative.shape != shape:
            raise ValueError(
                "Declared value and derivative callbacks must return value_shape."
            )
        self.value = value
        self.derivative_value = derivative
        self._support = support_
        self.breakpoints = points
        self.breakpoint_mask = point_mask
        self.value_shape = shape
        self.path_id = _path_id(path_id)

    @property
    def support(self) -> tuple[Array, Array]:
        return self._support[0], self._support[1]

    def evaluate(self, time: ArrayLike, /, side: DrivingPathSide = "right") -> Array:
        side_ = _side(side)
        time_ = _checked_time(time, self._support)
        value = jnp.asarray(self.value(time_, side_))
        return eqx.error_if(
            value,
            jnp.any(~jnp.isfinite(value)),
            "Declared driving-path value is non-finite.",
        )

    def derivative(self, time: ArrayLike, /, side: DrivingPathSide = "right") -> Array:
        side_ = _side(side)
        time_ = _checked_time(time, self._support)
        value = jnp.asarray(self.derivative_value(time_, side_))
        return eqx.error_if(
            value,
            jnp.any(~jnp.isfinite(value)),
            "Declared driving-path derivative is non-finite.",
        )


class _AbstractSampledDrivingPath(AbstractDifferentiableDrivingPath):
    times: Array
    values: Array
    time_mask: Array
    value_mask: Array
    sample_mask: Array
    num_samples: int = eqx.field(static=True)

    @property
    def support(self) -> tuple[Array, Array]:
        return self.times[0], self.times[self.num_samples - 1]

    def _time(self, time: ArrayLike, /) -> Array:
        support = jnp.stack(self.support)
        return _checked_time(time, support)

    def _segment(self, time: Array, side: DrivingPathSide, /) -> Array:
        index = jnp.searchsorted(self.times[: self.num_samples], time, side=side) - 1
        return jnp.where(
            time == self.times[0],
            0,
            jnp.where(
                time == self.times[self.num_samples - 1],
                self.num_samples - 2,
                index,
            ),
        )


class PiecewiseLinearDrivingPath(_AbstractSampledDrivingPath):
    """Continuous piecewise-linear path through an explicitly masked sample prefix."""

    def __init__(
        self,
        times: ArrayLike,
        values: ArrayLike,
        /,
        *,
        time_mask: ArrayLike,
        value_mask: ArrayLike,
        path_id: str,
    ):
        (
            times_,
            values_,
            time_mask_,
            value_mask_,
            sample_mask,
            count,
            shape,
        ) = _validated_samples(times, values, time_mask, value_mask, minimum_samples=2)
        self.times = times_
        self.values = values_
        self.time_mask = time_mask_
        self.value_mask = value_mask_
        self.sample_mask = sample_mask
        self.num_samples = count
        self.breakpoints = times_[1:-1]
        self.breakpoint_mask = jnp.arange(max(0, times_.shape[0] - 2)) < count - 2
        self.value_shape = shape
        self.path_id = _path_id(path_id)

    @classmethod
    def fit(
        cls,
        times: ArrayLike,
        values: ArrayLike,
        /,
        *,
        time_mask: ArrayLike,
        value_mask: ArrayLike,
        path_id: str,
    ) -> tuple[PiecewiseLinearDrivingPath, DrivingPathFitDiagnostics]:
        path = cls(
            times,
            values,
            time_mask=time_mask,
            value_mask=value_mask,
            path_id=path_id,
        )
        return path, _fit_diagnostics(
            path,
            method_id="sampled-piecewise-linear",
            approximation_id="piecewise-linear-interpolant",
            backend="closed-form",
        )

    def _evaluate_order(
        self,
        time: ArrayLike,
        side: DrivingPathSide,
        derivative_order: int,
        /,
    ) -> Array:
        side_ = _side(side)
        time_ = self._time(time)
        index = self._segment(time_, side_)
        left_time = self.times[index]
        right_time = self.times[index + 1]
        left_value = self.values[index]
        right_value = self.values[index + 1]
        slope = (right_value - left_value) / (right_time - left_time)
        if derivative_order == 1:
            return slope
        fraction = (time_ - left_time) / (right_time - left_time)
        return left_value + fraction * (right_value - left_value)

    def evaluate(self, time: ArrayLike, /, side: DrivingPathSide = "right") -> Array:
        return self._evaluate_order(time, side, 0)

    def derivative(self, time: ArrayLike, /, side: DrivingPathSide = "right") -> Array:
        return self._evaluate_order(time, side, 1)


class CausalBackwardHermiteDrivingPath(_AbstractSampledDrivingPath):
    """Cubic Hermite interpolation using only backward-difference knot slopes.

    The first backward slope is the first secant, so the first interval is exactly
    linear. At every interior knot both sides use the same already-observed backward
    slope; consequently the first derivative is continuous and the derivative-jump
    schedule is empty.
    """

    slopes: Array

    def __init__(
        self,
        times: ArrayLike,
        values: ArrayLike,
        /,
        *,
        time_mask: ArrayLike,
        value_mask: ArrayLike,
        path_id: str,
    ):
        (
            times_,
            values_,
            time_mask_,
            value_mask_,
            sample_mask,
            count,
            shape,
        ) = _validated_samples(times, values, time_mask, value_mask, minimum_samples=2)
        valid_times = times_[:count]
        valid_values = values_[:count]
        widths = valid_times[1:] - valid_times[:-1]
        width_shape = (count - 1,) + (1,) * len(shape)
        secants = (valid_values[1:] - valid_values[:-1]) / widths.reshape(width_shape)
        slopes = jnp.zeros_like(values_)
        slopes = slopes.at[0].set(secants[0])
        slopes = slopes.at[1:count].set(secants)
        self.times = times_
        self.values = values_
        self.time_mask = time_mask_
        self.value_mask = value_mask_
        self.sample_mask = sample_mask
        self.num_samples = count
        self.slopes = slopes
        self.breakpoints = jnp.empty((0,), dtype=times_.dtype)
        self.breakpoint_mask = jnp.empty((0,), dtype=bool)
        self.value_shape = shape
        self.path_id = _path_id(path_id)

    @classmethod
    def fit(
        cls,
        times: ArrayLike,
        values: ArrayLike,
        /,
        *,
        time_mask: ArrayLike,
        value_mask: ArrayLike,
        path_id: str,
    ) -> tuple[CausalBackwardHermiteDrivingPath, DrivingPathFitDiagnostics]:
        path = cls(
            times,
            values,
            time_mask=time_mask,
            value_mask=value_mask,
            path_id=path_id,
        )
        return path, _fit_diagnostics(
            path,
            method_id="sampled-causal-backward-hermite",
            approximation_id="backward-difference-cubic-hermite-interpolant",
            backend="closed-form",
        )

    def _evaluate_order(
        self,
        time: ArrayLike,
        side: DrivingPathSide,
        derivative_order: int,
        /,
    ) -> Array:
        side_ = _side(side)
        time_ = self._time(time)
        index = self._segment(time_, side_)
        left_time = self.times[index]
        width = self.times[index + 1] - left_time
        fraction = (time_ - left_time) / width
        left_value = self.values[index]
        right_value = self.values[index + 1]
        left_slope = self.slopes[index]
        right_slope = self.slopes[index + 1]
        if derivative_order == 1:
            return (
                (6.0 * fraction**2 - 6.0 * fraction) * left_value / width
                + (3.0 * fraction**2 - 4.0 * fraction + 1.0) * left_slope
                + (-6.0 * fraction**2 + 6.0 * fraction) * right_value / width
                + (3.0 * fraction**2 - 2.0 * fraction) * right_slope
            )
        return (
            (2.0 * fraction**3 - 3.0 * fraction**2 + 1.0) * left_value
            + (fraction**3 - 2.0 * fraction**2 + fraction) * width * left_slope
            + (-2.0 * fraction**3 + 3.0 * fraction**2) * right_value
            + (fraction**3 - fraction**2) * width * right_slope
        )

    def evaluate(self, time: ArrayLike, /, side: DrivingPathSide = "right") -> Array:
        return self._evaluate_order(time, side, 0)

    def derivative(self, time: ArrayLike, /, side: DrivingPathSide = "right") -> Array:
        return self._evaluate_order(time, side, 1)


class OfflineCubicDrivingPath(_AbstractSampledDrivingPath):
    """Natural cubic interpolant fitted from the complete valid sample prefix."""

    second_derivatives: Array

    def __init__(
        self,
        times: ArrayLike,
        values: ArrayLike,
        /,
        *,
        time_mask: ArrayLike,
        value_mask: ArrayLike,
        path_id: str,
    ):
        (
            times_,
            values_,
            time_mask_,
            value_mask_,
            sample_mask,
            count,
            shape,
        ) = _validated_samples(times, values, time_mask, value_mask, minimum_samples=4)
        valid_times = times_[:count]
        valid_values = values_[:count]
        widths = valid_times[1:] - valid_times[:-1]
        widths = eqx.error_if(
            widths,
            jnp.any(~jnp.isfinite(widths)) | jnp.any(widths <= 0.0),
            "Natural-cubic fitting requires finite, nonsingular sample spacing.",
        )
        payload_size = int(np.prod(shape, dtype=int)) if shape else 1
        flat_values = valid_values.reshape((count, payload_size))
        linear_dtype = jnp.result_type(times_, values_)
        lower_diagonal = jnp.zeros((count,), dtype=linear_dtype)
        diagonal = jnp.ones((count,), dtype=linear_dtype)
        upper_diagonal = jnp.zeros((count,), dtype=linear_dtype)
        lower_diagonal = lower_diagonal.at[1:-1].set(widths[:-1])
        diagonal = diagonal.at[1:-1].set(2.0 * (widths[:-1] + widths[1:]))
        upper_diagonal = upper_diagonal.at[1:-1].set(widths[1:])
        secants = (flat_values[1:] - flat_values[:-1]) / widths[:, None]
        right_hand_side = jnp.zeros((count, payload_size), dtype=linear_dtype)
        right_hand_side = right_hand_side.at[1:-1].set(6.0 * (secants[1:] - secants[:-1]))
        second_flat = jax.lax.linalg.tridiagonal_solve(
            lower_diagonal,
            diagonal,
            upper_diagonal,
            right_hand_side,
        )
        second = jnp.zeros_like(values_)
        second = second.at[:count].set(second_flat.reshape((count, *shape)))
        self.times = times_
        self.values = values_
        self.time_mask = time_mask_
        self.value_mask = value_mask_
        self.sample_mask = sample_mask
        self.num_samples = count
        self.second_derivatives = second
        self.breakpoints = jnp.empty((0,), dtype=times_.dtype)
        self.breakpoint_mask = jnp.empty((0,), dtype=bool)
        self.value_shape = shape
        self.path_id = _path_id(path_id)

    @classmethod
    def fit(
        cls,
        times: ArrayLike,
        values: ArrayLike,
        /,
        *,
        time_mask: ArrayLike,
        value_mask: ArrayLike,
        path_id: str,
    ) -> tuple[OfflineCubicDrivingPath, DrivingPathFitDiagnostics]:
        path = cls(
            times,
            values,
            time_mask=time_mask,
            value_mask=value_mask,
            path_id=path_id,
        )
        return path, _fit_diagnostics(
            path,
            method_id="offline-natural-cubic",
            approximation_id="global-natural-cubic-interpolant",
            backend="jax-tridiagonal-solve",
        )

    def _evaluate_order(
        self,
        time: ArrayLike,
        side: DrivingPathSide,
        derivative_order: int,
        /,
    ) -> Array:
        side_ = _side(side)
        time_ = self._time(time)
        index = self._segment(time_, side_)
        left_time = self.times[index]
        right_time = self.times[index + 1]
        width = right_time - left_time
        left_delta = right_time - time_
        right_delta = time_ - left_time
        left_value = self.values[index]
        right_value = self.values[index + 1]
        left_second = self.second_derivatives[index]
        right_second = self.second_derivatives[index + 1]
        if derivative_order == 1:
            return (
                -left_second * left_delta**2 / (2.0 * width)
                + right_second * right_delta**2 / (2.0 * width)
                + (right_value - left_value) / width
                - width * (right_second - left_second) / 6.0
            )
        return (
            left_second * left_delta**3 / (6.0 * width)
            + right_second * right_delta**3 / (6.0 * width)
            + (left_value - left_second * width**2 / 6.0) * left_delta / width
            + (right_value - right_second * width**2 / 6.0) * right_delta / width
        )

    def evaluate(self, time: ArrayLike, /, side: DrivingPathSide = "right") -> Array:
        return self._evaluate_order(time, side, 0)

    def derivative(self, time: ArrayLike, /, side: DrivingPathSide = "right") -> Array:
        return self._evaluate_order(time, side, 1)


class FixedBSplineDrivingPath(AbstractDifferentiableDrivingPath):
    """Fixed-grid B-spline path with differentiable coefficient leaves.

    The validated ``BSplineGrid`` is non-trainable configuration, while
    ``coefficients`` remains an ordinary inexact array leaf. Left limits are
    evaluated by reflecting the canonical B-spline representation, so repeated
    knots have exact, explicit one-sided derivative semantics without perturbing
    query times.
    """

    grid: BSplineGrid
    coefficients: Array
    _support: Array

    def __init__(
        self,
        grid: BSplineGrid,
        coefficients: ArrayLike,
        /,
        *,
        path_id: str,
    ):
        if not isinstance(grid, BSplineGrid):
            raise TypeError("grid must be a BSplineGrid.")
        if grid.degree < 1:
            raise ValueError("Differentiable B-spline paths require positive degree.")
        if any(order < 0 for order in grid.continuity_orders):
            raise ValueError("Differentiable B-spline paths must be continuous at knots.")
        coefficients_ = jnp.asarray(coefficients)
        if coefficients_.ndim < 1 or coefficients_.shape[0] != grid.coefficient_count:
            raise ValueError(
                "B-spline path coefficients need one leading entry per basis function."
            )
        if not (
            jnp.issubdtype(coefficients_.dtype, jnp.number)
            or jnp.issubdtype(coefficients_.dtype, jnp.bool_)
        ):
            raise TypeError("B-spline path coefficients must be numeric.")
        value_shape = _value_shape(tuple(int(size) for size in coefficients_.shape[1:]))
        coefficients_ = coefficients_.astype(jnp.result_type(coefficients_, float))
        coefficients_ = eqx.error_if(
            coefficients_,
            jnp.any(~jnp.isfinite(coefficients_)),
            "B-spline path coefficients must be finite.",
        )
        self.grid = grid
        self.coefficients = coefficients_
        self._support = jnp.stack(grid.active_interval)
        self.breakpoints = grid.breakpoints[1:-1]
        self.breakpoint_mask = jnp.asarray(
            tuple(order < 1 for order in grid.continuity_orders), dtype=bool
        )
        self.value_shape = value_shape
        self.path_id = _path_id(path_id)

    @property
    def support(self) -> tuple[Array, Array]:
        return self._support[0], self._support[1]

    def _evaluate_order(
        self,
        time: ArrayLike,
        side: DrivingPathSide,
        derivative_order: int,
        /,
    ) -> Array:
        side_ = _side(side)
        time_ = _checked_time(time, self._support)
        if side_ == "right":
            return bspline_evaluate(
                self.grid.knots,
                self.coefficients,
                time_,
                degree=self.grid.degree,
                derivative_order=derivative_order,
                bounds="error",
            ).values
        return (-1.0) ** derivative_order * bspline_evaluate(
            -self.grid.knots[::-1],
            self.coefficients[::-1],
            -time_,
            degree=self.grid.degree,
            derivative_order=derivative_order,
            bounds="error",
        ).values

    def evaluate(self, time: ArrayLike, /, side: DrivingPathSide = "right") -> Array:
        return self._evaluate_order(time, side, 0)

    def derivative(self, time: ArrayLike, /, side: DrivingPathSide = "right") -> Array:
        return self._evaluate_order(time, side, 1)


def _fit_diagnostics(
    path: _AbstractSampledDrivingPath,
    /,
    *,
    method_id: str,
    approximation_id: str,
    backend: str,
    regularization: float = 0.0,
) -> DrivingPathFitDiagnostics:
    sample_times = path.times[: path.num_samples]
    fitted = jax.vmap(lambda time: path.evaluate(time, "right"))(sample_times)
    residual = fitted - path.values[: path.num_samples]
    residual_norm = jnp.linalg.norm(residual)
    maximum_residual = jnp.max(jnp.abs(residual))
    widths = jnp.diff(sample_times)
    return DrivingPathFitDiagnostics(
        support=jnp.stack(path.support),
        residual_norm=residual_norm,
        maximum_residual=maximum_residual,
        valid=jnp.asarray(
            jnp.all(jnp.isfinite(residual))
            & jnp.isfinite(residual_norm)
            & jnp.isfinite(maximum_residual)
            & jnp.all(jnp.isfinite(widths))
            & jnp.all(widths > 0.0)
        ),
        status="success",
        method_id=method_id,
        approximation_id=approximation_id,
        backend=backend,
        sample_count=path.num_samples,
        sample_capacity=int(path.times.shape[0]),
        minimum_spacing=jnp.min(widths),
        maximum_spacing=jnp.max(widths),
        value_shape=path.value_shape,
        regularization=regularization,
    )


__all__ = [
    "AbstractDifferentiableDrivingPath",
    "CallableDrivingPath",
    "CausalBackwardHermiteDrivingPath",
    "DrivingPathFitDiagnostics",
    "DrivingPathSide",
    "FixedBSplineDrivingPath",
    "OfflineCubicDrivingPath",
    "PiecewiseLinearDrivingPath",
]
