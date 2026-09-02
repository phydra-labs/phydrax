#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..linalg import (
    continuous_lyapunov_equation,
    DenseLinearOperator,
    matrix_exponential_action,
    solve_matrix_equation,
)
from ._base import _as_point, _as_points, AbstractPositiveDefiniteKernel


class SHOKernel(AbstractPositiveDefiniteKernel):
    """Stationary covariance of a damped stochastic harmonic oscillator.

    ``frequency`` is the positive angular frequency ω₀ and ``quality_factor`` is
    Q. The implementation covers under-, critically-, and over-damped systems;
    the name does not assert that every valid parameter choice oscillates.
    """

    frequency: Array
    quality_factor: Array
    variance: Array

    def __init__(
        self,
        *,
        frequency: ArrayLike,
        quality_factor: ArrayLike,
        variance: ArrayLike = 1.0,
    ):
        omega = _positive_scalar(frequency, name="frequency")
        quality = _positive_scalar(quality_factor, name="quality_factor")
        variance_array = _nonnegative_scalar(variance, name="variance")
        self.frequency = omega
        self.quality_factor = quality
        self.variance = variance_array

    @property
    def drift_matrix(self) -> Array:
        omega = self.frequency
        return jnp.asarray(
            ((0.0, 1.0), (-omega * omega, -omega / self.quality_factor)),
            dtype=omega.dtype,
        )

    @property
    def observation_row(self) -> Array:
        return jnp.asarray((1.0, 0.0), dtype=self.frequency.dtype)

    @property
    def stationary_covariance(self) -> Array:
        omega = self.frequency
        return jnp.asarray(
            ((self.variance, 0.0), (0.0, omega * omega * self.variance)),
            dtype=omega.dtype,
        )

    @property
    def diffusion_intensity(self) -> Array:
        omega = self.frequency
        return 2.0 * omega * omega * omega * self.variance / self.quality_factor

    def pairwise(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        lag = jnp.abs(_time(left, name="left") - _time(right, name="right"))
        return _stationary_pairwise(
            self.drift_matrix,
            self.observation_row,
            self.stationary_covariance,
            lag,
        )

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_times = _times(left, name="left")
        right_times = _times(right, name="right")
        return jax.vmap(
            lambda time: jax.vmap(lambda other: self.pairwise(time, other))(right_times)
        )(left_times)

    def diagonal(self, points: ArrayLike, /) -> Array:
        times = _times(points, name="points")
        return jnp.broadcast_to(self.variance, (times.shape[0],))

    @property
    def max_derivative_order(self) -> int:
        return 1

    @property
    def is_unit_diagonal(self) -> bool:
        return False

    @property
    def kernel_id(self) -> str:
        return "SHOKernel"


class CARMAKernel(AbstractPositiveDefiniteKernel):
    """Stationary covariance of a stable finite-order continuous ARMA model.

    ``ar_coefficients=(a₁,…,aₚ)`` defines
    ``s**p + a₁ s**(p-1) + … + aₚ``. ``ma_coefficients`` are stored in
    increasing state-coordinate order and must have length at most ``p``.
    Pole validation and stationary-covariance preparation happen on the host.
    """

    ar_coefficients: Array
    ma_coefficients: Array
    innovation_scale: Array
    drift_matrix: Array
    observation_row: Array
    stationary_covariance: Array
    order: int = eqx.field(static=True)
    moving_average_order: int = eqx.field(static=True)
    stability_margin: float = eqx.field(static=True)

    def __init__(
        self,
        ar_coefficients: Sequence[Real] | ArrayLike,
        ma_coefficients: Sequence[Real] | ArrayLike,
        innovation_scale: ArrayLike,
        /,
        *,
        stability_margin: float = 1e-8,
    ):
        ar = _coefficient_vector(ar_coefficients, name="ar_coefficients")
        ma = _coefficient_vector(ma_coefficients, name="ma_coefficients")
        order = int(ar.shape[0])
        if ma.shape[0] > order:
            raise ValueError("CARMA requires 0 <= q < p.")
        if not isinstance(stability_margin, Real) or isinstance(stability_margin, bool):
            raise TypeError("stability_margin must be a real scalar.")
        margin = float(stability_margin)
        if not np.isfinite(margin) or margin <= 0.0:
            raise ValueError("stability_margin must be finite and positive.")
        poles = np.roots(np.concatenate((np.ones((1,)), np.asarray(ar))))
        if np.any(~np.isfinite(poles)) or np.any(np.real(poles) >= -margin):
            raise ValueError(
                "The continuous-time AR polynomial must be stable beyond "
                "stability_margin."
            )
        innovation = _positive_scalar(innovation_scale, name="innovation_scale")
        dtype = jnp.result_type(ar, ma, innovation)
        drift = jnp.zeros((order, order), dtype=dtype)
        if order > 1:
            drift = drift.at[jnp.arange(order - 1), jnp.arange(1, order)].set(1.0)
        drift = drift.at[-1, :].set(-ar[::-1])
        observation = jnp.pad(ma.astype(dtype), (0, order - int(ma.shape[0])))
        diffusion = (
            jnp.zeros((order, order), dtype=dtype).at[-1, -1].set(innovation * innovation)
        )
        covariance_result = solve_matrix_equation(
            continuous_lyapunov_equation(
                drift,
                diffusion,
                problem_id="carma-stationary-covariance",
            )
        )
        stationary = eqx.error_if(
            covariance_result.value,
            ~covariance_result.successful,
            "CARMA stationary Lyapunov solve failed its numerical certificate.",
        )
        self.ar_coefficients = ar
        self.ma_coefficients = ma
        self.innovation_scale = innovation
        self.drift_matrix = drift
        self.observation_row = observation
        self.stationary_covariance = stationary
        self.order = order
        self.moving_average_order = int(ma.shape[0]) - 1
        self.stability_margin = margin

    def pairwise(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        lag = jnp.abs(_time(left, name="left") - _time(right, name="right"))
        return _stationary_pairwise(
            self.drift_matrix,
            self.observation_row,
            self.stationary_covariance,
            lag,
        )

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_times = _times(left, name="left")
        right_times = _times(right, name="right")
        return jax.vmap(
            lambda time: jax.vmap(lambda other: self.pairwise(time, other))(right_times)
        )(left_times)

    def diagonal(self, points: ArrayLike, /) -> Array:
        times = _times(points, name="points")
        variance = (
            self.observation_row @ self.stationary_covariance @ self.observation_row
        )
        return jnp.broadcast_to(variance, (times.shape[0],))

    @property
    def max_derivative_order(self) -> int:
        return self.order - self.moving_average_order - 1

    @property
    def is_unit_diagonal(self) -> bool:
        return False

    @property
    def kernel_id(self) -> str:
        return f"CARMAKernel[p={self.order},q={self.moving_average_order}]"


def _stationary_pairwise(
    drift: Array,
    observation: Array,
    stationary_covariance: Array,
    lag: Array,
) -> Array:
    operator = DenseLinearOperator(drift, operator_id="temporal-kernel-drift")
    propagated = matrix_exponential_action(
        operator,
        stationary_covariance @ observation,
        lag,
    )
    value = eqx.error_if(
        propagated.value,
        ~propagated.converged,
        "Temporal-kernel matrix exponential action did not converge.",
    )
    return observation @ value


def _positive_scalar(value: ArrayLike, /, *, name: str) -> Array:
    scalar = jnp.asarray(value, dtype=float)
    if scalar.ndim != 0:
        raise ValueError(f"{name} must be scalar.")
    return eqx.error_if(
        scalar,
        ~jnp.isfinite(scalar) | (scalar <= 0.0),
        f"{name} must be finite and strictly positive.",
    )


def _nonnegative_scalar(value: ArrayLike, /, *, name: str) -> Array:
    scalar = jnp.asarray(value, dtype=float)
    if scalar.ndim != 0:
        raise ValueError(f"{name} must be scalar.")
    return eqx.error_if(
        scalar,
        ~jnp.isfinite(scalar) | (scalar < 0.0),
        f"{name} must be finite and nonnegative.",
    )


def _coefficient_vector(value: Sequence[Real] | ArrayLike, /, *, name: str) -> Array:
    vector = jnp.asarray(value, dtype=float)
    if vector.ndim != 1 or vector.shape[0] <= 0:
        raise ValueError(f"{name} must be a nonempty vector.")
    if not bool(jnp.all(jnp.isfinite(vector))):
        raise ValueError(f"{name} must contain only finite coefficients.")
    return vector


def _time(value: ArrayLike, /, *, name: str) -> Array:
    point = _as_point(value, name=name)
    if point.shape != (1,):
        raise ValueError("Temporal kernels require scalar time inputs.")
    return point[0]


def _times(value: ArrayLike, /, *, name: str) -> Array:
    points = _as_points(value, name=name)
    if points.shape[1] != 1:
        raise ValueError("Temporal kernels require scalar time inputs.")
    return points


__all__ = ["CARMAKernel", "SHOKernel"]
