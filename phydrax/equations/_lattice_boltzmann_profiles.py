#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class ParabolicVelocityParameters(StrictModule):
    center: Array
    radius: Array
    peak_velocity: Array

    def __init__(
        self,
        center: ArrayLike,
        radius: ArrayLike,
        peak_velocity: ArrayLike,
        /,
    ):
        center_ = jnp.asarray(center)
        if center_.ndim != 1 or not jnp.issubdtype(center_.dtype, jnp.inexact):
            raise ValueError("Parabolic profile center must be one inexact vector.")
        center_ = eqx.error_if(
            center_,
            jnp.any(~jnp.isfinite(center_)),
            "Parabolic profile center must be finite.",
        )
        radius_ = jnp.asarray(radius, dtype=center_.dtype)
        peak = jnp.asarray(peak_velocity, dtype=center_.dtype)
        if radius_.shape != () or peak.shape != ():
            raise ValueError("Parabolic profile parameters have incompatible shapes.")
        radius_ = eqx.error_if(
            radius_,
            ~jnp.isfinite(radius_) | (radius_ <= 0.0),
            "Parabolic profile radius must be finite and positive.",
        )
        peak = eqx.error_if(
            peak,
            ~jnp.isfinite(peak),
            "Parabolic peak velocity must be finite.",
        )
        self.center = center_
        self.radius = radius_
        self.peak_velocity = peak


class ParabolicVelocityProfilePlan(StrictModule, NonTrainableState):
    dimension: int = eqx.field(static=True)
    flow_axis: int = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    def __init__(self, dimension: int, flow_axis: int, /):
        dimension_ = int(dimension)
        axis = int(flow_axis)
        if dimension_ not in (2, 3) or not 0 <= axis < dimension_:
            raise ValueError("Parabolic profile dimension or flow axis is invalid.")
        self.dimension = dimension_
        self.flow_axis = axis
        self.profile_id = canonical_fingerprint(
            {
                "kind": "parabolic-velocity-profile",
                "dimension": dimension_,
                "flow_axis": axis,
            }
        )

    def __call__(
        self,
        time: ArrayLike,
        coordinates: ArrayLike,
        parameters: ParabolicVelocityParameters,
        /,
    ) -> Array:
        del time
        if not isinstance(parameters, ParabolicVelocityParameters):
            raise TypeError("parameters must be ParabolicVelocityParameters.")
        points = jnp.asarray(coordinates)
        if (
            points.ndim < 1
            or points.shape[-1] != self.dimension
            or parameters.center.shape != (self.dimension,)
            or not jnp.issubdtype(points.dtype, jnp.inexact)
        ):
            raise ValueError(
                "Parabolic profile coordinates and center must match dimension."
            )
        points = eqx.error_if(
            points,
            jnp.any(~jnp.isfinite(points)),
            "Parabolic profile coordinates must be finite.",
        )
        radius = eqx.error_if(
            parameters.radius,
            ~jnp.isfinite(parameters.radius) | (parameters.radius <= 0.0),
            "Parabolic profile radius must be finite and positive.",
        )
        center = eqx.error_if(
            parameters.center,
            jnp.any(~jnp.isfinite(parameters.center)),
            "Parabolic profile center must be finite.",
        )
        peak = eqx.error_if(
            parameters.peak_velocity,
            ~jnp.isfinite(parameters.peak_velocity),
            "Parabolic peak velocity must be finite.",
        )
        transverse = points - center
        transverse = transverse.at[..., self.flow_axis].set(0.0)
        radial_squared = jnp.sum(transverse * transverse, axis=-1)
        inside = radial_squared <= radius * radius
        speed = peak * (1.0 - radial_squared / (radius * radius))
        velocity = jnp.zeros(points.shape, dtype=points.dtype)
        return velocity.at[..., self.flow_axis].set(jnp.where(inside, speed, 0.0))


class WomersleyVelocityParameters(StrictModule):
    center: Array
    radius: Array
    angular_frequency: Array
    womersley_number: Array
    centerline_amplitude: Array
    phase: Array

    def __init__(
        self,
        center: ArrayLike,
        radius: ArrayLike,
        angular_frequency: ArrayLike,
        womersley_number: ArrayLike,
        centerline_amplitude: ArrayLike,
        /,
        *,
        phase: ArrayLike = 0.0,
    ):
        center_ = jnp.asarray(center)
        if center_.ndim != 1 or not jnp.issubdtype(center_.dtype, jnp.inexact):
            raise ValueError("Womersley profile center must be one inexact vector.")
        center_ = eqx.error_if(
            center_,
            jnp.any(~jnp.isfinite(center_)),
            "Womersley profile center must be finite.",
        )
        values = tuple(
            jnp.asarray(value, dtype=center_.dtype)
            for value in (
                radius,
                angular_frequency,
                womersley_number,
                centerline_amplitude,
                phase,
            )
        )
        if any(value.shape != () for value in values):
            raise ValueError("Womersley profile parameters have incompatible shapes.")
        radius_, frequency, alpha, amplitude, phase_ = values
        radius_ = eqx.error_if(
            radius_,
            ~jnp.isfinite(radius_) | (radius_ <= 0.0),
            "Womersley radius must be finite and positive.",
        )
        frequency = eqx.error_if(
            frequency,
            ~jnp.isfinite(frequency) | (frequency <= 0.0),
            "Womersley angular frequency must be finite and positive.",
        )
        alpha = eqx.error_if(
            alpha,
            ~jnp.isfinite(alpha) | (alpha <= 0.0),
            "Womersley number must be finite and positive.",
        )
        amplitude = eqx.error_if(
            amplitude,
            ~jnp.isfinite(amplitude),
            "Womersley centerline amplitude must be finite.",
        )
        phase_ = eqx.error_if(
            phase_,
            ~jnp.isfinite(phase_),
            "Womersley phase must be finite.",
        )
        self.center = center_
        self.radius = radius_
        self.angular_frequency = frequency
        self.womersley_number = alpha
        self.centerline_amplitude = amplitude
        self.phase = phase_


def _bessel_j0_series(value: Array, terms: int, /) -> Array:
    squared = -0.25 * value * value
    result = jnp.ones_like(value)
    term = jnp.ones_like(value)
    for order in range(1, terms):
        term = term * squared / float(order * order)
        result = result + term
    return result


class WomersleyVelocityProfilePlan(StrictModule, NonTrainableState):
    dimension: int = eqx.field(static=True)
    flow_axis: int = eqx.field(static=True)
    series_terms: int = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        flow_axis: int,
        /,
        *,
        series_terms: int = 32,
    ):
        dimension_ = int(dimension)
        axis = int(flow_axis)
        terms = int(series_terms)
        if dimension_ not in (2, 3) or not 0 <= axis < dimension_ or terms < 8:
            raise ValueError("Womersley profile configuration is invalid.")
        self.dimension = dimension_
        self.flow_axis = axis
        self.series_terms = terms
        self.profile_id = canonical_fingerprint(
            {
                "kind": "womersley-velocity-profile",
                "dimension": dimension_,
                "flow_axis": axis,
                "series_terms": terms,
            }
        )

    def __call__(
        self,
        time: ArrayLike,
        coordinates: ArrayLike,
        parameters: WomersleyVelocityParameters,
        /,
    ) -> Array:
        if not isinstance(parameters, WomersleyVelocityParameters):
            raise TypeError("parameters must be WomersleyVelocityParameters.")
        points = jnp.asarray(coordinates)
        if (
            points.ndim < 1
            or points.shape[-1] != self.dimension
            or parameters.center.shape != (self.dimension,)
            or not jnp.issubdtype(points.dtype, jnp.inexact)
        ):
            raise ValueError("Womersley coordinates and center must match dimension.")
        points = eqx.error_if(
            points,
            jnp.any(~jnp.isfinite(points)),
            "Womersley coordinates must be finite.",
        )
        radius = eqx.error_if(
            parameters.radius,
            ~jnp.isfinite(parameters.radius) | (parameters.radius <= 0.0),
            "Womersley radius must be finite and positive.",
        )
        alpha = eqx.error_if(
            parameters.womersley_number,
            ~jnp.isfinite(parameters.womersley_number)
            | (parameters.womersley_number <= 0.0),
            "Womersley number must be finite and positive.",
        )
        center = eqx.error_if(
            parameters.center,
            jnp.any(~jnp.isfinite(parameters.center)),
            "Womersley profile center must be finite.",
        )
        frequency = eqx.error_if(
            parameters.angular_frequency,
            ~jnp.isfinite(parameters.angular_frequency)
            | (parameters.angular_frequency <= 0.0),
            "Womersley angular frequency must be finite and positive.",
        )
        amplitude = eqx.error_if(
            parameters.centerline_amplitude,
            ~jnp.isfinite(parameters.centerline_amplitude),
            "Womersley centerline amplitude must be finite.",
        )
        phase = eqx.error_if(
            parameters.phase,
            ~jnp.isfinite(parameters.phase),
            "Womersley phase must be finite.",
        )
        time_ = jnp.asarray(time, dtype=points.dtype)
        if time_.shape != ():
            raise ValueError("Womersley time must be scalar.")
        time_ = eqx.error_if(
            time_,
            ~jnp.isfinite(time_),
            "Womersley time must be finite.",
        )
        transverse = points - center
        transverse = transverse.at[..., self.flow_axis].set(0.0)
        radial = jnp.sqrt(jnp.sum(transverse * transverse, axis=-1)) / radius
        complex_dtype = jnp.complex128 if points.dtype == jnp.float64 else jnp.complex64
        lam = alpha.astype(complex_dtype) * jnp.asarray(
            (-1.0 + 1.0j) / jnp.sqrt(2.0), dtype=complex_dtype
        )
        wall_value = _bessel_j0_series(lam, self.series_terms)
        comparison = _bessel_j0_series(lam, self.series_terms - 2)
        series_tolerance = jnp.asarray(
            1.0e-10 if points.dtype == jnp.float64 else 5.0e-5,
            dtype=points.dtype,
        )
        wall_value = eqx.error_if(
            wall_value,
            jnp.abs(wall_value - comparison)
            > series_tolerance * jnp.maximum(jnp.abs(wall_value), 1.0),
            "Womersley Bessel series did not converge.",
        )
        center_shape = 1.0 - 1.0 / wall_value
        center_shape = eqx.error_if(
            center_shape,
            ~jnp.isfinite(center_shape) | (jnp.abs(center_shape) <= 1.0e-10),
            "Womersley normalization is singular for these parameters.",
        )
        shape = (
            1.0 - _bessel_j0_series(lam * radial, self.series_terms) / wall_value
        ) / center_shape
        oscillation = jnp.exp(1.0j * (frequency * time_ + phase))
        speed = amplitude * jnp.real(shape * oscillation)
        speed = jnp.where(radial <= 1.0, speed, 0.0)
        velocity = jnp.zeros(points.shape, dtype=points.dtype)
        return velocity.at[..., self.flow_axis].set(speed.astype(points.dtype))


__all__ = [
    "ParabolicVelocityParameters",
    "ParabolicVelocityProfilePlan",
    "WomersleyVelocityParameters",
    "WomersleyVelocityProfilePlan",
]
