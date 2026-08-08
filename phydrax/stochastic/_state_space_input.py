#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from math import prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._interpolation._bspline import bspline_stencil
from .._interpolation._bspline_grid import BSplineGrid
from .._interpolation._stencil import apply_gather_stencil
from .._strict import StrictModule


SampledInputInterpolation: TypeAlias = Literal["zero-order-hold", "linear"]


def _name(value: str, /, *, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _shape(value: tuple[int, ...], /, *, owner: str) -> tuple[int, ...]:
    resolved = tuple(int(size) for size in value)
    if any(size <= 0 for size in resolved):
        raise ValueError(f"{owner} dimensions must be positive.")
    return resolved


def _guard_case_index(
    value: Array,
    case_index: Array,
    case_count: int,
    /,
) -> Array:
    return eqx.error_if(
        value,
        (case_index < 0) | (case_index >= case_count),
        "State-space input physical case index is out of bounds.",
    )


class InputEvaluation(StrictModule):
    """One exogenous-input value with explicit interpolation support."""

    value: Array
    valid: Array


class AbstractStateSpaceInput(StrictModule):
    """Typed exogenous signal evaluated by physical case and time."""

    input_shape: tuple[int, ...] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    input_id: str = eqx.field(static=True)

    @abstractmethod
    def evaluate(
        self,
        time: ArrayLike,
        case_index: ArrayLike,
        /,
    ) -> InputEvaluation:
        raise NotImplementedError

    @abstractmethod
    def breakpoints(
        self,
        t0: ArrayLike,
        t1: ArrayLike,
        case_index: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        """Return fixed-capacity breakpoint times and an explicit interior mask."""
        raise NotImplementedError


class SampledStateSpaceInput(AbstractStateSpaceInput):
    """Sampled signal with explicit zero-order-hold or linear semantics.

    Valid knots must form a nonempty prefix in every physical case. Evaluation is
    supported on the closed interval from the first through the last valid knot.
    Zero-order hold is right-continuous at interior knots; linear interpolation
    includes both interval endpoints.
    """

    times: Array
    values: Array
    knot_valid: Array
    interpolation: SampledInputInterpolation = eqx.field(static=True)
    num_knots: int = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        values: ArrayLike,
        /,
        *,
        knot_valid: ArrayLike | None = None,
        interpolation: SampledInputInterpolation,
        input_id: str,
    ):
        if interpolation not in ("zero-order-hold", "linear"):
            raise ValueError("interpolation must be 'zero-order-hold' or 'linear'.")
        times_raw = jnp.asarray(times)
        values_raw = jnp.asarray(values)
        if times_raw.ndim < 1:
            raise ValueError("Sampled input times must have a trailing knot axis.")
        if values_raw.ndim < times_raw.ndim or tuple(
            int(size) for size in values_raw.shape[: times_raw.ndim]
        ) != tuple(int(size) for size in times_raw.shape):
            raise ValueError(
                "Sampled input values must begin with the complete times shape."
            )
        if jnp.issubdtype(times_raw.dtype, jnp.complexfloating):
            raise TypeError("Sampled input times must be real-valued.")
        if not (
            jnp.issubdtype(values_raw.dtype, jnp.number)
            or jnp.issubdtype(values_raw.dtype, jnp.bool_)
        ):
            raise TypeError("Sampled input values must be numeric.")

        case_shape = tuple(int(size) for size in times_raw.shape[:-1])
        input_shape = tuple(int(size) for size in values_raw.shape[times_raw.ndim :])
        _shape(case_shape, owner="case_shape")
        _shape(input_shape, owner="input_shape")
        num_knots = int(times_raw.shape[-1])
        minimum_knots = 1 if interpolation == "zero-order-hold" else 2
        if num_knots < minimum_knots:
            raise ValueError(
                f"{interpolation} input requires at least {minimum_knots} knots."
            )

        valid = (
            jnp.ones(times_raw.shape, dtype=bool)
            if knot_valid is None
            else jnp.asarray(knot_valid, dtype=bool)
        )
        if valid.shape != times_raw.shape:
            raise ValueError("knot_valid must have the same shape as times.")

        times_host = np.asarray(times_raw)
        values_host = np.asarray(values_raw)
        valid_host = np.asarray(valid)
        flat_times = times_host.reshape((-1, num_knots))
        flat_valid = valid_host.reshape((-1, num_knots))
        flat_values = values_host.reshape((-1, num_knots) + input_shape)
        valid_counts = np.sum(flat_valid, axis=-1)
        if np.any(valid_counts < minimum_knots):
            raise ValueError(
                f"Every physical case requires at least {minimum_knots} valid knots."
            )
        if np.any(np.diff(flat_valid.astype(np.int8), axis=-1) > 0):
            raise ValueError("Valid sampled-input knots must form a prefix.")
        for case_index, count in enumerate(valid_counts):
            count_ = int(count)
            if not np.all(np.isfinite(flat_times[case_index, :count_])):
                raise ValueError("Valid sampled-input times must be finite.")
            if np.any(np.diff(flat_times[case_index, :count_]) <= 0.0):
                raise ValueError("Valid sampled-input times must be strictly increasing.")
            if not np.all(np.isfinite(flat_values[case_index, :count_])):
                raise ValueError("Valid sampled-input values must be finite.")

        time_dtype = jnp.result_type(times_raw, float)
        value_dtype = jnp.result_type(values_raw, float)
        self.times = times_raw.astype(time_dtype)
        self.values = values_raw.astype(value_dtype)
        self.knot_valid = valid
        self.interpolation = interpolation
        self.num_knots = num_knots
        self.case_shape = case_shape
        self.input_shape = input_shape
        self.input_id = _name(input_id, owner="input_id")

    def evaluate(
        self,
        time: ArrayLike,
        case_index: ArrayLike,
        /,
    ) -> InputEvaluation:
        time_ = jnp.asarray(time, dtype=self.times.dtype).reshape(())
        case_index_ = jnp.asarray(case_index, dtype=jnp.int32).reshape(())
        case_count = prod(self.case_shape) if self.case_shape else 1
        case_index_ = _guard_case_index(case_index_, case_index_, case_count)
        times = self.times.reshape((case_count, self.num_knots))[case_index_]
        values = self.values.reshape((case_count, self.num_knots) + self.input_shape)[
            case_index_
        ]
        knot_valid = self.knot_valid.reshape((case_count, self.num_knots))[case_index_]
        valid_count = jnp.sum(knot_valid, dtype=jnp.int32)
        first_time = times[0]
        last_time = times[valid_count - 1]
        in_support = jnp.isfinite(time_) & (time_ >= first_time) & (time_ <= last_time)
        count_at_or_before = jnp.sum(knot_valid & (times <= time_), dtype=jnp.int32)

        if self.interpolation == "zero-order-hold":
            index = jnp.clip(count_at_or_before - 1, 0, valid_count - 1)
            value = values[index]
        else:
            lower_index = jnp.clip(count_at_or_before - 1, 0, valid_count - 2)
            upper_index = lower_index + 1
            lower_time = times[lower_index]
            upper_time = times[upper_index]
            fraction = (time_ - lower_time) / (upper_time - lower_time)
            fraction = jnp.clip(fraction, 0.0, 1.0)
            payload_axes = (1,) * len(self.input_shape)
            weight = fraction.reshape(payload_axes)
            value = values[lower_index] + weight * (
                values[upper_index] - values[lower_index]
            )

        valid = in_support & jnp.all(jnp.isfinite(value))
        return InputEvaluation(value=value, valid=valid)

    def breakpoints(
        self,
        t0: ArrayLike,
        t1: ArrayLike,
        case_index: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        start = jnp.asarray(t0, dtype=self.times.dtype).reshape(())
        end = jnp.asarray(t1, dtype=self.times.dtype).reshape(())
        case_index_ = jnp.asarray(case_index, dtype=jnp.int32).reshape(())
        case_count = prod(self.case_shape) if self.case_shape else 1
        case_index_ = _guard_case_index(case_index_, case_index_, case_count)
        times = self.times.reshape((case_count, self.num_knots))[case_index_]
        knot_valid = self.knot_valid.reshape((case_count, self.num_knots))[case_index_]
        valid = knot_valid & (times > start) & (times < end)
        return times, valid


class BSplineStateSpaceInput(AbstractStateSpaceInput):
    """Fixed-grid B-spline coefficient signal over a closed active interval."""

    grid: BSplineGrid
    coefficients: Array

    def __init__(
        self,
        grid: BSplineGrid,
        coefficients: ArrayLike,
        /,
        *,
        case_shape: tuple[int, ...] = (),
        input_id: str,
    ):
        if not isinstance(grid, BSplineGrid):
            raise TypeError("grid must be a BSplineGrid.")
        cases = _shape(tuple(case_shape), owner="case_shape")
        coefficients_raw = jnp.asarray(coefficients)
        coefficient_axis = len(cases)
        expected_prefix = cases + (grid.coefficient_count,)
        if (
            coefficients_raw.ndim <= coefficient_axis
            or tuple(int(size) for size in coefficients_raw.shape[: coefficient_axis + 1])
            != expected_prefix
        ):
            raise ValueError(
                "B-spline input coefficients must begin with case_shape and "
                "the grid coefficient count."
            )
        if not (
            jnp.issubdtype(coefficients_raw.dtype, jnp.number)
            or jnp.issubdtype(coefficients_raw.dtype, jnp.bool_)
        ):
            raise TypeError("B-spline input coefficients must be numeric.")

        self.grid = grid
        coefficients_array = coefficients_raw.astype(
            jnp.result_type(coefficients_raw, float)
        )
        self.coefficients = eqx.error_if(
            coefficients_array,
            jnp.any(~jnp.isfinite(coefficients_array)),
            "B-spline input coefficients must be finite.",
        )
        self.case_shape = cases
        self.input_shape = tuple(
            int(size) for size in coefficients_raw.shape[coefficient_axis + 1 :]
        )
        _shape(self.input_shape, owner="input_shape")
        self.input_id = _name(input_id, owner="input_id")

    def evaluate(
        self,
        time: ArrayLike,
        case_index: ArrayLike,
        /,
    ) -> InputEvaluation:
        time_ = jnp.asarray(time, dtype=self.grid.knots.dtype).reshape(())
        case_index_ = jnp.asarray(case_index, dtype=jnp.int32).reshape(())
        case_count = prod(self.case_shape) if self.case_shape else 1
        case_index_ = _guard_case_index(case_index_, case_index_, case_count)
        lower, upper = self.grid.active_interval
        finite = jnp.isfinite(time_)
        safe_time = jnp.where(finite, time_, lower)
        clipped_time = jnp.clip(safe_time, lower, upper)
        stencil = bspline_stencil(
            self.grid.knots,
            clipped_time,
            degree=self.grid.degree,
            bounds="clip",
        )
        coefficients = self.coefficients.reshape(
            (case_count, self.grid.coefficient_count) + self.input_shape
        )[case_index_]
        value = apply_gather_stencil(coefficients, stencil).values
        valid = (
            finite & (time_ >= lower) & (time_ <= upper) & jnp.all(jnp.isfinite(value))
        )
        return InputEvaluation(value=value, valid=valid)

    def breakpoints(
        self,
        t0: ArrayLike,
        t1: ArrayLike,
        case_index: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        case_index_ = jnp.asarray(case_index, dtype=jnp.int32).reshape(())
        case_count = prod(self.case_shape) if self.case_shape else 1
        start = jnp.asarray(t0, dtype=self.grid.knots.dtype).reshape(())
        end = jnp.asarray(t1, dtype=self.grid.knots.dtype).reshape(())
        times = _guard_case_index(self.grid.breakpoints, case_index_, case_count)
        valid = (times > start) & (times < end)
        return times, valid


__all__ = [
    "AbstractStateSpaceInput",
    "BSplineStateSpaceInput",
    "SampledInputInterpolation",
    "SampledStateSpaceInput",
    "InputEvaluation",
]
