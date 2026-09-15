#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


DerivativeBoundary: TypeAlias = Literal["periodic", "one_sided"]


def _spatial_axis(value: Array, axis: int, /) -> int:
    axis_ = int(axis)
    if value.ndim < 3 or axis_ not in (0, 1, 2):
        raise ValueError("Finite differences require three trailing spatial axes.")
    return value.ndim - 3 + axis_


def _spacing(value: ArrayLike, /) -> Array:
    spacing = jnp.asarray(value)
    if spacing.shape != ():
        raise ValueError("spacing must be scalar.")
    return spacing


def _shift(value: Array, offset: int, axis: int, /) -> Array:
    return jnp.roll(value, shift=-offset, axis=axis)


def _set_index(value: Array, axis: int, index: int, update: Array, /) -> Array:
    selection = [slice(None)] * value.ndim
    selection[axis] = index
    return value.at[tuple(selection)].set(update)


def _take(value: Array, axis: int, index: int, /) -> Array:
    return jnp.take(value, index, axis=axis)


def _one_sided_first_boundaries(value: Array, derivative: Array, axis: int, /) -> Array:
    coefficients = jnp.asarray(
        (
            (-25.0, 48.0, -36.0, 16.0, -3.0),
            (-3.0, -10.0, 18.0, -6.0, 1.0),
        ),
        dtype=value.dtype,
    ) / 12.0
    lower0 = sum(coefficients[0, k] * _take(value, axis, k) for k in range(5))
    lower1 = sum(coefficients[1, k] * _take(value, axis, k) for k in range(5))
    upper0 = -sum(
        coefficients[0, k] * _take(value, axis, -1 - k) for k in range(5)
    )
    upper1 = -sum(
        coefficients[1, k] * _take(value, axis, -1 - k) for k in range(5)
    )
    derivative = _set_index(derivative, axis, 0, lower0)
    derivative = _set_index(derivative, axis, 1, lower1)
    derivative = _set_index(derivative, axis, -1, upper0)
    return _set_index(derivative, axis, -2, upper1)


def centered_first_derivative(
    field: ArrayLike,
    spacing: ArrayLike,
    axis: int,
    /,
    *,
    boundary: DerivativeBoundary = "periodic",
) -> Array:
    """Fourth-order centered first derivative on a uniform Cartesian axis."""

    value = jnp.asarray(field)
    actual_axis = _spatial_axis(value, axis)
    if value.shape[actual_axis] < (5 if boundary == "periodic" else 6):
        raise ValueError("Fourth-order first derivatives need at least five points.")
    h = _spacing(spacing)
    derivative = (
        _shift(value, -2, actual_axis)
        - 8.0 * _shift(value, -1, actual_axis)
        + 8.0 * _shift(value, 1, actual_axis)
        - _shift(value, 2, actual_axis)
    ) / (12.0 * h)
    if boundary == "periodic":
        return derivative
    if boundary != "one_sided":
        raise ValueError("boundary must be 'periodic' or 'one_sided'.")
    return _one_sided_first_boundaries(value, derivative * h, actual_axis) / h


def centered_second_derivative(
    field: ArrayLike,
    spacing: ArrayLike,
    axis: int,
    /,
    *,
    boundary: DerivativeBoundary = "periodic",
) -> Array:
    """Fourth-order centered second derivative on a uniform Cartesian axis."""

    value = jnp.asarray(field)
    actual_axis = _spatial_axis(value, axis)
    if value.shape[actual_axis] < (5 if boundary == "periodic" else 6):
        raise ValueError("Fourth-order second derivatives need at least five points.")
    h = _spacing(spacing)
    derivative = (
        -_shift(value, -2, actual_axis)
        + 16.0 * _shift(value, -1, actual_axis)
        - 30.0 * value
        + 16.0 * _shift(value, 1, actual_axis)
        - _shift(value, 2, actual_axis)
    ) / 12.0
    if boundary == "periodic":
        return derivative / h**2
    if boundary != "one_sided":
        raise ValueError("boundary must be 'periodic' or 'one_sided'.")
    coefficients = jnp.asarray(
        (
            (45.0, -154.0, 214.0, -156.0, 61.0, -10.0),
            (10.0, -15.0, -4.0, 14.0, -6.0, 1.0),
        ),
        dtype=value.dtype,
    ) / 12.0
    lower0 = sum(coefficients[0, k] * _take(value, actual_axis, k) for k in range(6))
    lower1 = sum(coefficients[1, k] * _take(value, actual_axis, k) for k in range(6))
    upper0 = sum(
        coefficients[0, k] * _take(value, actual_axis, -1 - k) for k in range(6)
    )
    upper1 = sum(
        coefficients[1, k] * _take(value, actual_axis, -1 - k) for k in range(6)
    )
    derivative = _set_index(derivative, actual_axis, 0, lower0)
    derivative = _set_index(derivative, actual_axis, 1, lower1)
    derivative = _set_index(derivative, actual_axis, -1, upper0)
    derivative = _set_index(derivative, actual_axis, -2, upper1)
    return derivative / h**2


def upwind_first_derivative(
    field: ArrayLike,
    velocity: ArrayLike,
    spacing: ArrayLike,
    axis: int,
    /,
    *,
    boundary: DerivativeBoundary = "periodic",
) -> Array:
    """Fourth-order upwind-biased derivative selected pointwise by velocity sign."""

    value = jnp.asarray(field)
    speed = jnp.asarray(velocity)
    actual_axis = _spatial_axis(value, axis)
    if value.shape[actual_axis] < 5:
        raise ValueError("Fourth-order upwind derivatives need at least five points.")
    h = _spacing(spacing)
    backward = (
        -_shift(value, -3, actual_axis)
        + 6.0 * _shift(value, -2, actual_axis)
        - 18.0 * _shift(value, -1, actual_axis)
        + 10.0 * value
        + 3.0 * _shift(value, 1, actual_axis)
    ) / (12.0 * h)
    forward = (
        -3.0 * _shift(value, -1, actual_axis)
        - 10.0 * value
        + 18.0 * _shift(value, 1, actual_axis)
        - 6.0 * _shift(value, 2, actual_axis)
        + _shift(value, 3, actual_axis)
    ) / (12.0 * h)
    derivative = jnp.where(speed >= 0.0, backward, forward)
    if boundary == "periodic":
        return derivative
    if boundary != "one_sided":
        raise ValueError("boundary must be 'periodic' or 'one_sided'.")
    centered = centered_first_derivative(value, h, axis, boundary="one_sided")
    indices = jnp.arange(value.shape[actual_axis])
    boundary_indices = (indices < 3) | (indices >= value.shape[actual_axis] - 3)
    broadcast_shape = [1] * value.ndim
    broadcast_shape[actual_axis] = value.shape[actual_axis]
    return jnp.where(boundary_indices.reshape(broadcast_shape), centered, derivative)


def kreiss_oliger_dissipation(
    field: ArrayLike,
    spacing: ArrayLike,
    /,
    *,
    strength: ArrayLike,
    boundary: DerivativeBoundary = "periodic",
) -> Array:
    """Sixth-derivative Kreiss--Oliger damping for fourth-order spatial schemes."""

    value = jnp.asarray(field)
    coefficient = jnp.asarray(strength, dtype=value.dtype)
    if coefficient.shape != ():
        raise ValueError("strength must be scalar.")
    steps = jnp.asarray(spacing, dtype=value.dtype)
    if steps.shape != (3,):
        raise ValueError("spacing must contain three values.")
    result = jnp.zeros_like(value)
    for grid_axis in range(3):
        axis = _spatial_axis(value, grid_axis)
        if value.shape[axis] < 7:
            raise ValueError("Sixth-order dissipation needs at least seven points.")
        stencil = (
            _shift(value, -3, axis)
            - 6.0 * _shift(value, -2, axis)
            + 15.0 * _shift(value, -1, axis)
            - 20.0 * value
            + 15.0 * _shift(value, 1, axis)
            - 6.0 * _shift(value, 2, axis)
            + _shift(value, 3, axis)
        )
        contribution = coefficient * stencil / (64.0 * steps[grid_axis])
        if boundary == "one_sided":
            indices = jnp.arange(value.shape[axis])
            interior = (indices >= 3) & (indices < value.shape[axis] - 3)
            broadcast_shape = [1] * value.ndim
            broadcast_shape[axis] = value.shape[axis]
            contribution = jnp.where(interior.reshape(broadcast_shape), contribution, 0.0)
        elif boundary != "periodic":
            raise ValueError("boundary must be 'periodic' or 'one_sided'.")
        result = result + contribution
    return result


class FourthOrderDerivatives(StrictModule, NonTrainableState):
    """Prepared fixed-shape fourth-order Cartesian finite-difference operators."""

    grid_shape: tuple[int, int, int] = eqx.field(static=True)
    spacing: tuple[float, float, float] = eqx.field(static=True)
    boundary: DerivativeBoundary = eqx.field(static=True)
    dissipation_strength: float = eqx.field(static=True)
    derivative_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        spacing: tuple[float, float, float],
        /,
        *,
        boundary: DerivativeBoundary = "periodic",
        dissipation_strength: float = 0.0,
    ):
        shape = tuple(int(value) for value in grid_shape)
        steps = tuple(float(value) for value in spacing)
        strength = float(dissipation_strength)
        minimum = 7 if strength > 0.0 else (5 if boundary == "periodic" else 6)
        if len(shape) != 3 or any(value < minimum for value in shape):
            raise ValueError(f"Every grid extent must be at least {minimum}.")
        if len(steps) != 3 or any(not isfinite(value) or value <= 0.0 for value in steps):
            raise ValueError("spacing must contain three finite positive values.")
        if boundary not in ("periodic", "one_sided"):
            raise ValueError("boundary must be 'periodic' or 'one_sided'.")
        if not isfinite(strength) or strength < 0.0:
            raise ValueError("dissipation_strength must be finite and non-negative.")
        self.grid_shape = shape
        self.spacing = steps
        self.boundary = boundary
        self.dissipation_strength = strength
        self.derivative_id = canonical_fingerprint(
            {
                "kind": "fourth-order-cartesian-derivatives",
                "shape": list(shape),
                "spacing": list(steps),
                "boundary": boundary,
                "dissipation_strength": strength,
            }
        )

    def _validate(self, field: ArrayLike, /) -> Array:
        value = jnp.asarray(field)
        if value.shape[-3:] != self.grid_shape:
            raise ValueError(
                f"field trailing shape must be {self.grid_shape}; got {value.shape[-3:]}."
            )
        return value

    def first(self, field: ArrayLike, axis: int, /) -> Array:
        value = self._validate(field)
        return centered_first_derivative(
            value, self.spacing[int(axis)], axis, boundary=self.boundary
        )

    def second(self, field: ArrayLike, axis: int, /) -> Array:
        value = self._validate(field)
        return centered_second_derivative(
            value, self.spacing[int(axis)], axis, boundary=self.boundary
        )

    def mixed_second(self, field: ArrayLike, axis1: int, axis2: int, /) -> Array:
        value = self._validate(field)
        if int(axis1) == int(axis2):
            return self.second(value, axis1)
        return self.first(self.first(value, axis1), axis2)

    def gradient(self, field: ArrayLike, /) -> Array:
        value = self._validate(field)
        return jnp.stack(tuple(self.first(value, axis) for axis in range(3)), axis=0)

    def hessian(self, field: ArrayLike, /) -> Array:
        value = self._validate(field)
        return jnp.stack(
            tuple(
                jnp.stack(
                    tuple(self.mixed_second(value, i, j) for j in range(3)), axis=0
                )
                for i in range(3)
            ),
            axis=0,
        )

    def upwind(self, field: ArrayLike, velocity: ArrayLike, axis: int, /) -> Array:
        value = self._validate(field)
        return upwind_first_derivative(
            value,
            velocity,
            self.spacing[int(axis)],
            axis,
            boundary=self.boundary,
        )

    def advect(self, field: ArrayLike, velocity: ArrayLike, /) -> Array:
        value = self._validate(field)
        speed = jnp.asarray(velocity)
        if speed.shape != (3,) + self.grid_shape:
            raise ValueError(
                f"velocity must have shape {(3,) + self.grid_shape}."
            )
        return sum(speed[i] * self.upwind(value, speed[i], i) for i in range(3))

    def dissipation(self, field: ArrayLike, /) -> Array:
        value = self._validate(field)
        if self.dissipation_strength == 0.0:
            return jnp.zeros_like(value)
        return kreiss_oliger_dissipation(
            value,
            jnp.asarray(self.spacing),
            strength=self.dissipation_strength,
            boundary=self.boundary,
        )


__all__ = [
    "DerivativeBoundary",
    "FourthOrderDerivatives",
    "centered_first_derivative",
    "centered_second_derivative",
    "kreiss_oliger_dissipation",
    "upwind_first_derivative",
]
