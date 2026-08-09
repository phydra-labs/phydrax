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


_EULER_MASCHERONI = 0.5772156649015329


def _dirichlet_residual(concentration: Array, mean: Array, /) -> Array:
    total = jnp.sum(concentration, axis=-1, keepdims=True)
    return jsp.special.digamma(concentration) - jsp.special.digamma(total) - mean


def _inverse_digamma(value: Array, /) -> Array:
    initial = jnp.where(
        value >= -2.22,
        jnp.exp(value) + 0.5,
        -1.0 / (value + _EULER_MASCHERONI),
    )
    initial = jnp.maximum(initial, jnp.sqrt(jnp.finfo(value.dtype).tiny))

    def step(_, current):
        update = (jsp.special.digamma(current) - value) / jsp.special.polygamma(
            1, current
        )
        candidate = current - update
        return jnp.where(
            jnp.isfinite(candidate) & (candidate > 0.0),
            candidate,
            0.5 * current,
        )

    return jax.lax.fori_loop(0, 5, step, initial)


def _solve_dirichlet_concentration(
    mean: Array,
    /,
    *,
    atol: float,
    rtol: float,
    max_iterations: int,
) -> tuple[Array, Array, Array]:
    dtype = mean.dtype
    categories = int(mean.shape[-1])
    gap = -jax.nn.logsumexp(mean, axis=-1)
    initial_total = jnp.maximum((categories - 1.0) / (2.0 * gap), 0.01)
    lower = jnp.log(initial_total) - 4.0
    upper = jnp.log(initial_total) + 4.0

    def concentration_from_log_total(log_total):
        total = jnp.exp(log_total)[..., None]
        shifted_mean = mean + jsp.special.digamma(total)
        return _inverse_digamma(shifted_mean)

    def total_residual(log_total):
        total = jnp.exp(log_total)
        concentration = concentration_from_log_total(log_total)
        return jnp.sum(concentration, axis=-1) / total - 1.0

    def bracket_step(_, bounds):
        lower_value, upper_value = bounds
        lower_residual = total_residual(lower_value)
        upper_residual = total_residual(upper_value)
        return (
            jnp.where(lower_residual <= 0.0, lower_value - 4.0, lower_value),
            jnp.where(upper_residual >= 0.0, upper_value + 4.0, upper_value),
        )

    lower, upper = jax.lax.fori_loop(0, 16, bracket_step, (lower, upper))
    concentration = concentration_from_log_total(0.5 * (lower + upper))
    iterations = jnp.zeros(mean.shape[:-1], dtype=jnp.int32)
    converged = jnp.zeros(mean.shape[:-1], dtype=bool)
    coordinate_scale = jnp.maximum(jnp.abs(mean), 1.0)
    coordinate_tolerance = (
        jnp.maximum(atol, 64.0 * jnp.finfo(dtype).eps * coordinate_scale)
        + rtol * coordinate_scale
    )

    def condition(state):
        _, _, _, _, converged_values, loop_iteration = state
        return (loop_iteration < max_iterations) & jnp.any(~converged_values)

    def step(state):
        (
            lower_value,
            upper_value,
            concentration_value,
            iteration_values,
            converged_values,
            loop_iteration,
        ) = state
        midpoint = 0.5 * (lower_value + upper_value)
        candidate = concentration_from_log_total(midpoint)
        root_residual = total_residual(midpoint)
        residual = jnp.max(
            jnp.abs(_dirichlet_residual(candidate, mean)) / coordinate_tolerance,
            axis=-1,
        )
        active = ~converged_values
        newly_converged = active & (residual <= 1.0)
        lower_value = jnp.where(active & (root_residual > 0.0), midpoint, lower_value)
        upper_value = jnp.where(active & (root_residual <= 0.0), midpoint, upper_value)
        concentration_value = jnp.where(active[..., None], candidate, concentration_value)
        iteration_values = iteration_values + active.astype(jnp.int32)
        return (
            lower_value,
            upper_value,
            concentration_value,
            iteration_values,
            converged_values | newly_converged,
            loop_iteration + 1,
        )

    _, _, concentration, iterations, converged, _ = jax.lax.while_loop(
        condition,
        step,
        (
            lower,
            upper,
            concentration,
            iterations,
            converged,
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    final_residual = jnp.max(
        jnp.abs(_dirichlet_residual(concentration, mean)) / coordinate_tolerance,
        axis=-1,
    )
    converged = converged | (final_residual <= 1.0)
    return concentration, iterations, converged


@jax.custom_jvp
def _implicit_dirichlet_concentration(mean: Array, concentration: Array, /) -> Array:
    """Attach the inverse-Fisher derivative to a converged concentration."""
    del mean
    return concentration


@_implicit_dirichlet_concentration.defjvp
def _implicit_dirichlet_concentration_jvp(primals, tangents):
    mean, concentration = primals
    mean_tangent, _ = tangents
    del mean
    diagonal = jsp.special.polygamma(1, concentration)
    inverse_diagonal = 1.0 / diagonal
    total = jnp.sum(concentration, axis=-1)
    shared = jsp.special.polygamma(1, total)
    weighted = mean_tangent * inverse_diagonal
    denominator = 1.0 / shared - jnp.sum(inverse_diagonal, axis=-1)
    correction = jnp.sum(weighted, axis=-1) / denominator
    concentration_tangent = weighted + correction[..., None] * inverse_diagonal
    return concentration, concentration_tangent


class DirichletFamily(AbstractExponentialFamily):
    """Dirichlet laws relative to Hausdorff measure on the unit simplex."""

    num_categories: int = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)
    _signature: ExponentialFamilySignature = eqx.field(static=True)

    def __init__(
        self,
        num_categories: int,
        *,
        atol: float = 1e-12,
        rtol: float = 1e-12,
        max_iterations: int = 100,
    ):
        categories = int(num_categories)
        absolute = float(atol)
        relative = float(rtol)
        iterations = int(max_iterations)
        if categories < 2:
            raise ValueError("num_categories must be at least two.")
        if not jnp.isfinite(absolute) or absolute <= 0.0:
            raise ValueError("atol must be finite and strictly positive.")
        if not jnp.isfinite(relative) or relative < 0.0:
            raise ValueError("rtol must be finite and non-negative.")
        if iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        self.num_categories = categories
        self.atol = absolute
        self.rtol = relative
        self.max_iterations = iterations
        self._signature = ExponentialFamilySignature(
            "dirichlet",
            categories,
            (categories,),
            "hausdorff",
            f"positive-unit-simplex-{categories}",
            f"log-simplex-statistics-{categories}",
        )

    @property
    def signature(self) -> ExponentialFamilySignature:
        return self._signature

    def natural_from_concentration(
        self, concentration: ArrayLike, /
    ) -> NaturalCoordinates:
        values = jnp.asarray(concentration)
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise TypeError("Dirichlet concentrations must be real-valued.")
        if values.ndim == 0 or int(values.shape[-1]) != self.num_categories:
            raise ValueError(
                "Dirichlet concentration must end in num_categories="
                f"{self.num_categories}; got {values.shape}."
            )
        values = values.astype(jnp.result_type(values, 0.0))
        return self.natural(values - 1.0)

    def law_from_concentration(self, concentration: ArrayLike, /):
        """Return a Dirichlet law from conventional concentration parameters."""
        return self.law(self.natural_from_concentration(concentration))

    def concentration_from_natural(self, natural: NaturalCoordinates, /) -> Array:
        domain = self.natural_domain(natural)
        return jnp.where(domain.valid[..., None], natural.values + 1.0, jnp.nan)

    def _natural_domain(self, values: Array, /) -> ExponentialFamilyDomainResult:
        concentration = values + 1.0
        minimum = jnp.min(concentration, axis=-1)
        return _natural_domain_result(
            self.signature,
            values,
            interior=minimum > 0.0,
            boundary=minimum == 0.0,
        )

    def _mean_domain(self, values: Array, /) -> ExponentialFamilyDomainResult:
        log_total = jax.nn.logsumexp(values, axis=-1)
        scale = jnp.maximum(jnp.max(jnp.abs(values), axis=-1), 1.0)
        tolerance = 64.0 * jnp.finfo(values.dtype).eps * scale
        return _mean_domain_result(
            self.signature,
            values,
            interior=log_total < -tolerance,
            boundary=jnp.abs(log_total) <= tolerance,
        )

    def _sufficient_statistics(self, value: ArrayLike, /) -> StatisticBatch:
        raw = jnp.asarray(value)
        if jnp.issubdtype(raw.dtype, jnp.complexfloating):
            raise TypeError("Dirichlet observations must be real-valued.")
        if raw.ndim == 0 or int(raw.shape[-1]) != self.num_categories:
            raise ValueError(
                "Dirichlet observations must end in simplex dimension "
                f"{self.num_categories}; got {raw.shape}."
            )
        observation = raw.astype(jnp.result_type(raw, 0.0))
        tolerance = 64.0 * jnp.finfo(observation.dtype).eps
        valid = (
            jnp.all(jnp.isfinite(observation), axis=-1)
            & jnp.all(observation > 0.0, axis=-1)
            & (jnp.abs(jnp.sum(observation, axis=-1) - 1.0) <= tolerance)
        )
        safe = jnp.where(valid[..., None], observation, 1.0 / self.num_categories)
        return StatisticBatch(jnp.log(safe), valid, self.signature)

    def _log_base_density(self, value: ArrayLike, /) -> Array:
        values = jnp.asarray(value)
        if values.ndim == 0 or int(values.shape[-1]) != self.num_categories:
            raise ValueError("Dirichlet observations have an incompatible event shape.")
        dtype = jnp.result_type(values, 0.0)
        return jnp.full(
            values.shape[:-1],
            -0.5 * jnp.log(jnp.asarray(self.num_categories, dtype=dtype)),
            dtype=dtype,
        )

    def _log_normalizer(self, natural_values: Array, /) -> Array:
        concentration = natural_values + 1.0
        return jnp.sum(jsp.special.gammaln(concentration), axis=-1) - jsp.special.gammaln(
            jnp.sum(concentration, axis=-1)
        )

    def _mean_values(self, natural_values: Array, /) -> Array:
        concentration = natural_values + 1.0
        total = jnp.sum(concentration, axis=-1, keepdims=True)
        return jsp.special.digamma(concentration) - jsp.special.digamma(total)

    def _natural_from_mean_values(self, mean_values: Array, /) -> Array:
        concentration, _, _ = _solve_dirichlet_concentration(
            mean_values,
            atol=self.atol,
            rtol=self.rtol,
            max_iterations=self.max_iterations,
        )
        concentration = _implicit_dirichlet_concentration(
            mean_values, jax.lax.stop_gradient(concentration)
        )
        return concentration - 1.0

    def _natural_from_mean_result(
        self,
        mean: MeanCoordinates,
        domain: ExponentialFamilyDomainResult,
        /,
    ) -> ExponentialFamilyConversionResult:
        unit_concentration_mean = jsp.special.digamma(
            jnp.ones((self.num_categories,), dtype=mean.values.dtype)
        ) - jsp.special.digamma(jnp.asarray(self.num_categories, dtype=mean.values.dtype))
        safe_mean = jnp.where(
            domain.interior[..., None], mean.values, unit_concentration_mean
        )
        concentration, iterations, converged = _solve_dirichlet_concentration(
            safe_mean,
            atol=self.atol,
            rtol=self.rtol,
            max_iterations=self.max_iterations,
        )
        concentration = _implicit_dirichlet_concentration(
            safe_mean, jax.lax.stop_gradient(concentration)
        )
        candidate_values = jnp.where(
            domain.interior[..., None], concentration - 1.0, jnp.nan
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
            method_id="dirichlet-total-concentration-bisection",
        )

    def _sample(
        self,
        key,
        natural_values: Array,
        sample_shape: tuple[int, ...],
        /,
    ) -> Array:
        concentration = natural_values + 1.0
        return jr.dirichlet(
            key,
            concentration,
            shape=sample_shape + concentration.shape[:-1],
            dtype=natural_values.dtype,
        )


__all__ = ["DirichletFamily"]
