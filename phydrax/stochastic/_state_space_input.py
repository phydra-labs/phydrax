#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from math import prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._interpolation._bspline import bspline_stencil
from .._interpolation._bspline_grid import BSplineGrid
from .._interpolation._stencil import apply_gather_stencil
from .._strict import StrictModule
from ..series import SampledSeries, SampledSeriesReconstruction, SeriesSupport


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

    reconstruction: SampledSeriesReconstruction
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
        identifier = _name(input_id, owner="input_id")
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
        times_ = times_raw.astype(jnp.result_type(times_raw, float))
        values_ = values_raw.astype(jnp.result_type(values_raw, float))
        times_ = eqx.error_if(
            times_,
            jnp.any(jnp.sum(valid, axis=-1) < minimum_knots),
            f"Every physical case requires at least {minimum_knots} valid knots.",
        )
        support = SeriesSupport(
            times_,
            node_valid=valid,
            series_shape=case_shape,
            coordinate_name="time",
            coordinate_id=f"{identifier}:time",
        )
        series = SampledSeries(
            support,
            values_,
            series_id=f"{identifier}:values",
        )
        method = "previous" if interpolation == "zero-order-hold" else "linear"
        self.reconstruction = SampledSeriesReconstruction(
            series,
            interpolation=method,
            bounds="fill",
        )
        self.interpolation = interpolation
        self.num_knots = num_knots
        self.case_shape = case_shape
        self.input_shape = input_shape
        self.input_id = identifier

    @property
    def times(self) -> Array:
        return self.reconstruction.series.support.coordinates

    @property
    def values(self) -> Array:
        return self.reconstruction.series.values

    @property
    def knot_valid(self) -> Array:
        return self.reconstruction.series.support.node_valid

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
        evaluation = self.reconstruction.evaluate(time_, case_index_)
        return InputEvaluation(value=evaluation.values, valid=evaluation.support)

    def breakpoints(
        self,
        t0: ArrayLike,
        t1: ArrayLike,
        case_index: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        case_index_ = jnp.asarray(case_index, dtype=jnp.int32).reshape(())
        case_count = prod(self.case_shape) if self.case_shape else 1
        case_index_ = _guard_case_index(case_index_, case_index_, case_count)
        return self.reconstruction.breakpoints(t0, t1, case_index_)


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
