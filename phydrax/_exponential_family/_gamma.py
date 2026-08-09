#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from ._contracts import (
    _mean_domain_result,
    _natural_domain_result,
    AbstractExponentialFamily,
    EXPONENTIAL_FAMILY_NONCONVERGED,
    EXPONENTIAL_FAMILY_NONFINITE,
    ExponentialFamilyConversionResult,
    ExponentialFamilyDomainResult,
    ExponentialFamilySignature,
    MeanCoordinates,
    NaturalCoordinates,
    StatisticBatch,
)


_GAMMA_SIGNATURE = ExponentialFamilySignature(
    "gamma",
    2,
    (),
    "lebesgue",
    "positive-real",
    "log-linear-shape-rate",
)


def _gamma_shape_residual(shape: Array, delta: Array, /) -> Array:
    return jnp.log(shape) - jsp.special.digamma(shape) - delta


def _initial_gamma_shape(delta: Array, /) -> Array:
    discriminant = (delta - 3.0) ** 2 + 24.0 * delta
    return (3.0 - delta + jnp.sqrt(discriminant)) / (12.0 * delta)


def _solve_gamma_shape(
    delta: Array,
    /,
    *,
    atol: float,
    rtol: float,
    max_iterations: int,
) -> tuple[Array, Array, Array]:
    dtype = delta.dtype
    tiny = jnp.finfo(dtype).tiny
    initial = jnp.maximum(_initial_gamma_shape(delta), jnp.sqrt(tiny))
    lower = jnp.maximum(0.5 * initial, tiny)
    upper = jnp.maximum(2.0 * initial, 1.0)

    def bracket_step(_, bounds):
        lower_value, upper_value = bounds
        lower_residual = _gamma_shape_residual(lower_value, delta)
        upper_residual = _gamma_shape_residual(upper_value, delta)
        lower_value = jnp.where(lower_residual <= 0.0, 0.5 * lower_value, lower_value)
        upper_value = jnp.where(upper_residual >= 0.0, 2.0 * upper_value, upper_value)
        return jnp.maximum(lower_value, tiny), upper_value

    lower, upper = jax.lax.fori_loop(0, 64, bracket_step, (lower, upper))
    shape = jnp.clip(initial, lower, upper)
    iterations = jnp.zeros(delta.shape, dtype=jnp.int32)
    converged = jnp.zeros(delta.shape, dtype=bool)

    def condition(state):
        _, _, _, _, converged_values, loop_iteration = state
        return (loop_iteration < max_iterations) & jnp.any(~converged_values)

    def step(state):
        (
            shape_value,
            lower_value,
            upper_value,
            iteration_values,
            converged_values,
            loop_iteration,
        ) = state
        residual = _gamma_shape_residual(shape_value, delta)
        tolerance = jnp.maximum(atol, 64.0 * jnp.finfo(dtype).eps) + rtol * delta
        newly_converged = jnp.abs(residual) <= tolerance
        active = ~(converged_values | newly_converged)
        lower_candidate = jnp.where((residual > 0.0) & active, shape_value, lower_value)
        upper_candidate = jnp.where((residual < 0.0) & active, shape_value, upper_value)
        derivative = 1.0 / shape_value - jsp.special.polygamma(1, shape_value)
        newton = shape_value - residual / derivative
        midpoint = 0.5 * (lower_candidate + upper_candidate)
        acceptable = (
            jnp.isfinite(newton) & (newton > lower_candidate) & (newton < upper_candidate)
        )
        next_shape = jnp.where(acceptable, newton, midpoint)
        shape_value = jnp.where(active, next_shape, shape_value)
        iteration_values = iteration_values + active.astype(jnp.int32)
        return (
            shape_value,
            lower_candidate,
            upper_candidate,
            iteration_values,
            converged_values | newly_converged,
            loop_iteration + 1,
        )

    shape, _, _, iterations, converged, _ = jax.lax.while_loop(
        condition,
        step,
        (
            shape,
            lower,
            upper,
            iterations,
            converged,
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    final_residual = jnp.abs(_gamma_shape_residual(shape, delta))
    effective_atol = jnp.maximum(atol, 64.0 * jnp.finfo(dtype).eps)
    converged = converged | (final_residual <= effective_atol + rtol * delta)
    return shape, iterations, converged


@jax.custom_jvp
def _implicit_gamma_shape(delta: Array, shape: Array, /) -> Array:
    """Attach the implicit inverse-map derivative to a converged shape."""
    del delta
    return shape


@_implicit_gamma_shape.defjvp
def _implicit_gamma_shape_jvp(primals, tangents):
    delta, shape = primals
    delta_tangent, _ = tangents
    del delta
    derivative = 1.0 / shape - jsp.special.polygamma(1, shape)
    return shape, delta_tangent / derivative


class GammaFamily(AbstractExponentialFamily):
    """Gamma laws in full shape-rate natural coordinates."""

    atol: float = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        atol: float = 1e-10,
        rtol: float = 1e-10,
        max_iterations: int = 64,
    ):
        absolute = float(atol)
        relative = float(rtol)
        iterations = int(max_iterations)
        if not jnp.isfinite(absolute) or absolute <= 0.0:
            raise ValueError("atol must be finite and strictly positive.")
        if not jnp.isfinite(relative) or relative < 0.0:
            raise ValueError("rtol must be finite and non-negative.")
        if iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        self.atol = absolute
        self.rtol = relative
        self.max_iterations = iterations

    @property
    def signature(self) -> ExponentialFamilySignature:
        return _GAMMA_SIGNATURE

    def natural_from_shape_rate(
        self, shape: ArrayLike, rate: ArrayLike, /
    ) -> NaturalCoordinates:
        """Return natural coordinates from conventional shape and rate."""
        shape_array, rate_array = jnp.broadcast_arrays(
            jnp.asarray(shape), jnp.asarray(rate)
        )
        dtype = jnp.result_type(shape_array, rate_array, 0.0)
        return self.natural(
            jnp.stack(
                (shape_array.astype(dtype) - 1.0, -rate_array.astype(dtype)),
                axis=-1,
            )
        )

    def law_from_shape_rate(self, shape: ArrayLike, rate: ArrayLike, /):
        """Return a Gamma law from conventional shape and rate."""
        return self.law(self.natural_from_shape_rate(shape, rate))

    def shape_rate_from_natural(
        self, natural: NaturalCoordinates, /
    ) -> tuple[Array, Array]:
        domain = self.natural_domain(natural)
        shape = natural.values[..., 0] + 1.0
        rate = -natural.values[..., 1]
        return (
            jnp.where(domain.valid, shape, jnp.nan),
            jnp.where(domain.valid, rate, jnp.nan),
        )

    def _natural_domain(self, values: Array, /) -> ExponentialFamilyDomainResult:
        shape_coordinate = values[..., 0]
        rate_coordinate = values[..., 1]
        closure = (shape_coordinate >= -1.0) & (rate_coordinate <= 0.0)
        return _natural_domain_result(
            self.signature,
            values,
            interior=(shape_coordinate > -1.0) & (rate_coordinate < 0.0),
            boundary=closure & ((shape_coordinate == -1.0) | (rate_coordinate == 0.0)),
        )

    def _mean_domain(self, values: Array, /) -> ExponentialFamilyDomainResult:
        expected_log = values[..., 0]
        expected_value = values[..., 1]
        safe_value = jnp.where(expected_value > 0.0, expected_value, 1.0)
        delta = jnp.log(safe_value) - expected_log
        scale = jnp.maximum(
            jnp.maximum(jnp.abs(jnp.log(safe_value)), jnp.abs(expected_log)), 1.0
        )
        tolerance = 64.0 * jnp.finfo(values.dtype).eps * scale
        positive = expected_value > 0.0
        return _mean_domain_result(
            self.signature,
            values,
            interior=positive & (delta > tolerance),
            boundary=positive & (jnp.abs(delta) <= tolerance),
        )

    def _sufficient_statistics(self, value: ArrayLike, /) -> StatisticBatch:
        raw = jnp.asarray(value)
        if jnp.issubdtype(raw.dtype, jnp.complexfloating):
            raise TypeError("Gamma observations must be real-valued.")
        observation = raw.astype(jnp.result_type(raw, 0.0))
        valid = jnp.isfinite(observation) & (observation > 0.0)
        safe = jnp.where(valid, observation, 1.0)
        return StatisticBatch(
            jnp.stack((jnp.log(safe), safe), axis=-1),
            valid,
            self.signature,
        )

    def _log_base_density(self, value: ArrayLike, /) -> Array:
        return jnp.zeros_like(jnp.asarray(value, dtype=float))

    def _log_normalizer(self, natural_values: Array, /) -> Array:
        shape = natural_values[..., 0] + 1.0
        rate = -natural_values[..., 1]
        return jsp.special.gammaln(shape) - shape * jnp.log(rate)

    def _mean_values(self, natural_values: Array, /) -> Array:
        shape = natural_values[..., 0] + 1.0
        rate = -natural_values[..., 1]
        return jnp.stack(
            (jsp.special.digamma(shape) - jnp.log(rate), shape / rate), axis=-1
        )

    def _natural_from_mean_values(self, mean_values: Array, /) -> Array:
        expected_log = mean_values[..., 0]
        expected_value = mean_values[..., 1]
        delta = jnp.log(expected_value) - expected_log
        shape, _, _ = _solve_gamma_shape(
            delta,
            atol=self.atol,
            rtol=self.rtol,
            max_iterations=self.max_iterations,
        )
        rate = shape / expected_value
        return jnp.stack((shape - 1.0, -rate), axis=-1)

    def _natural_from_mean_result(
        self,
        mean: MeanCoordinates,
        domain: ExponentialFamilyDomainResult,
        /,
    ) -> ExponentialFamilyConversionResult:
        safe_mean = jnp.where(
            domain.interior[..., None],
            mean.values,
            jnp.asarray([-0.5772156649015329, 1.0], dtype=mean.values.dtype),
        )
        expected_log = safe_mean[..., 0]
        expected_value = safe_mean[..., 1]
        delta = jnp.log(expected_value) - expected_log
        shape, iterations, converged = _solve_gamma_shape(
            delta,
            atol=self.atol,
            rtol=self.rtol,
            max_iterations=self.max_iterations,
        )
        shape = _implicit_gamma_shape(delta, jax.lax.stop_gradient(shape))
        rate = shape / expected_value
        candidate_values = jnp.stack((shape - 1.0, -rate), axis=-1)
        candidate_values = jnp.where(
            domain.interior[..., None], candidate_values, jnp.nan
        )
        natural = NaturalCoordinates(candidate_values, self.signature)
        reconstructed = self._mean_values(candidate_values)
        residual = jnp.linalg.norm(reconstructed - mean.values, axis=-1)
        candidate_finite = jnp.all(jnp.isfinite(candidate_values), axis=-1)
        residual_finite = jnp.isfinite(residual)
        finite = candidate_finite & residual_finite
        valid = domain.valid & finite & converged
        status = jnp.where(
            domain.valid & ~finite,
            EXPONENTIAL_FAMILY_NONFINITE,
            jnp.where(
                domain.valid & finite & ~converged,
                EXPONENTIAL_FAMILY_NONCONVERGED,
                domain.status,
            ),
        )
        return ExponentialFamilyConversionResult(
            mean=mean,
            natural=natural,
            valid=valid,
            status=status,
            residual=jnp.where(domain.valid & residual_finite, residual, jnp.inf),
            iterations=jnp.where(domain.valid, iterations, 0),
            method_id="gamma-safeguarded-newton",
        )

    def _sample(
        self,
        key,
        natural_values: Array,
        sample_shape: tuple[int, ...],
        /,
    ) -> Array:
        shape = natural_values[..., 0] + 1.0
        rate = -natural_values[..., 1]
        standard = jr.gamma(
            key,
            shape,
            shape=sample_shape + shape.shape,
            dtype=natural_values.dtype,
        )
        return standard / rate


__all__ = ["GammaFamily"]
