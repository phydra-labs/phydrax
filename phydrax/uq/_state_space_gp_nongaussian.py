#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-iteration Laplace sites for bounded state-space GP likelihoods."""

from __future__ import annotations

from numbers import Integral, Real

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._exponential_family import BernoulliFamily, PoissonFamily
from .._likelihoods import AbstractLikelihood, ScalarNaturalExponentialFamilyLikelihood
from .._strict import StrictModule
from ._state_space_gp import (
    _training_marginals,
    fit_state_space_gaussian_process,
    StateSpaceGaussianProcessPlan,
    StateSpaceGaussianProcessResult,
)


STATE_SPACE_GP_LAPLACE_CURVATURE_FAILURE = 10
STATE_SPACE_GP_LAPLACE_SITE_FAILURE = 11
STATE_SPACE_GP_LAPLACE_CONVERGENCE_FAILURE = 12
STATE_SPACE_GP_LAPLACE_GAUSSIAN_FAILURE = 13


class StateSpaceGaussianProcessLaplace(StrictModule):
    """Bounded fixed-iteration Laplace-site policy for scalar log-concave rows."""

    max_iterations: int = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    minimum_curvature: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_iterations: int = 8,
        damping: float = 1.0,
        tolerance: float = 1e-5,
        minimum_curvature: float = 1e-8,
    ):
        if not isinstance(max_iterations, Integral) or isinstance(max_iterations, bool):
            raise TypeError("max_iterations must be an integer.")
        count = int(max_iterations)
        if count <= 0:
            raise ValueError("max_iterations must be positive.")
        self.max_iterations = count
        self.damping = _bounded_real(damping, name="damping", lower=0.0, upper=1.0)
        if self.damping == 0.0:
            raise ValueError("damping must be strictly positive.")
        self.tolerance = _positive_real(tolerance, name="tolerance")
        self.minimum_curvature = _positive_real(
            minimum_curvature, name="minimum_curvature"
        )


class StateSpaceGaussianProcessApproximateResult(StrictModule):
    """Approximate posterior and explicit Laplace-site convergence evidence."""

    gaussian_result: StateSpaceGaussianProcessResult
    mode: Array
    site_curvature: Array
    mode_residual: Array
    minimum_site_curvature: Array
    maximum_site_curvature: Array
    valid: Array
    status: Array
    iterations: int = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    approximation_kind: str = eqx.field(static=True)
    exact: bool = eqx.field(static=True)

    @property
    def posterior_times(self) -> Array:
        return self.gaussian_result.posterior_times

    @property
    def posterior_mean(self) -> Array:
        return self.gaussian_result.posterior_mean

    @property
    def posterior_variance(self) -> Array:
        return self.gaussian_result.posterior_variance

    @property
    def predictive_mean(self) -> Array:
        return self.gaussian_result.predictive_mean

    @property
    def predictive_variance(self) -> Array:
        return self.gaussian_result.predictive_variance

    @property
    def successful(self) -> Array:
        return self.valid


def fit_state_space_approximate_gaussian_process(
    plan: StateSpaceGaussianProcessPlan,
    train_values: ArrayLike,
    likelihood: AbstractLikelihood,
    approximation: StateSpaceGaussianProcessLaplace,
    /,
    *,
    temporal_method: str = "sequential",
    covariance_form: str = "square_root",
) -> StateSpaceGaussianProcessApproximateResult:
    """Fit certified Bernoulli/Poisson natural-parameter Laplace sites.

    This is an approximation. Every iteration runs the exact Gaussian child model;
    nonpositive curvature, nonfinite sites, child failure, and unmet convergence are
    distinct typed non-success states rather than covariance repair or fallback.
    """
    if not isinstance(plan, StateSpaceGaussianProcessPlan):
        raise TypeError("plan must be a StateSpaceGaussianProcessPlan.")
    _validate_likelihood(likelihood)
    if not isinstance(approximation, StateSpaceGaussianProcessLaplace):
        raise TypeError("approximation must be a StateSpaceGaussianProcessLaplace.")
    observations = jnp.asarray(train_values, dtype=plan.schedule_times.dtype)
    if observations.shape != (plan.train_size,):
        raise ValueError("train_values must have shape (plan.train_size,).")
    observations = eqx.error_if(
        observations,
        jnp.any(~jnp.isfinite(observations)),
        "train_values must be finite; missing rows belong in the prepared mask.",
    )
    mode = jnp.zeros_like(observations)
    curvature_valid = jnp.asarray(True)
    sites_valid = jnp.asarray(True)
    gaussian = fit_state_space_gaussian_process(
        plan,
        observations,
        noise_scale=jnp.ones_like(observations),
        temporal_method=temporal_method,
        covariance_form=covariance_form,
    )
    residual = jnp.asarray(jnp.inf, dtype=observations.dtype)
    curvature = jnp.ones_like(observations)

    def row_terms(latent: Array, target: Array) -> tuple[Array, Array]:
        def log_probability(value: Array) -> Array:
            return jnp.asarray(likelihood.log_prob(value, target)).reshape(())

        gradient = jax.grad(log_probability)(latent)
        hessian = jax.grad(jax.grad(log_probability))(latent)
        return gradient, -hessian

    for _ in range(approximation.max_iterations):
        gradient, curvature = jax.vmap(row_terms)(mode, observations)
        row_curvature_valid = (
            jnp.isfinite(curvature)
            & (curvature >= approximation.minimum_curvature)
            & plan.train_mask
        ) | ~plan.train_mask
        curvature_valid = curvature_valid & jnp.all(row_curvature_valid)
        safe_curvature = jnp.where(
            row_curvature_valid, curvature, jnp.ones_like(curvature)
        )
        pseudo_values = mode + gradient / safe_curvature
        row_sites_valid = (
            jnp.isfinite(pseudo_values) & jnp.isfinite(gradient)
        ) | ~plan.train_mask
        sites_valid = sites_valid & jnp.all(row_sites_valid)
        safe_values = jnp.where(row_sites_valid, pseudo_values, jnp.zeros_like(mode))
        gaussian = fit_state_space_gaussian_process(
            plan,
            safe_values,
            noise_scale=jax.lax.rsqrt(safe_curvature),
            temporal_method=temporal_method,
            covariance_form=covariance_form,
        )
        posterior_mode, _ = _training_marginals(plan, gaussian)
        next_mode = mode + approximation.damping * (posterior_mode - mode)
        residual = jnp.max(jnp.where(plan.train_mask, jnp.abs(next_mode - mode), 0.0))
        mode = next_mode

    converged = jnp.isfinite(residual) & (residual <= approximation.tolerance)
    gaussian_valid = gaussian.valid
    valid = curvature_valid & sites_valid & gaussian_valid & converged
    status = jnp.where(
        ~curvature_valid,
        STATE_SPACE_GP_LAPLACE_CURVATURE_FAILURE,
        jnp.where(
            ~sites_valid,
            STATE_SPACE_GP_LAPLACE_SITE_FAILURE,
            jnp.where(
                ~gaussian_valid,
                STATE_SPACE_GP_LAPLACE_GAUSSIAN_FAILURE,
                jnp.where(
                    ~converged,
                    STATE_SPACE_GP_LAPLACE_CONVERGENCE_FAILURE,
                    0,
                ),
            ),
        ),
    ).astype(jnp.int32)
    active_curvature = jnp.where(plan.train_mask, curvature, jnp.inf)
    minimum = jnp.min(active_curvature)
    maximum = jnp.max(jnp.where(plan.train_mask, curvature, -jnp.inf))
    return StateSpaceGaussianProcessApproximateResult(
        gaussian_result=gaussian,
        mode=mode,
        site_curvature=curvature,
        mode_residual=residual,
        minimum_site_curvature=minimum,
        maximum_site_curvature=maximum,
        valid=valid,
        status=status,
        iterations=approximation.max_iterations,
        damping=approximation.damping,
        approximation_kind="fixed-iteration-log-concave-laplace",
        exact=False,
    )


def _validate_likelihood(likelihood: AbstractLikelihood, /) -> None:
    if not isinstance(likelihood, AbstractLikelihood):
        raise TypeError("likelihood must implement AbstractLikelihood.")
    if not isinstance(
        likelihood, ScalarNaturalExponentialFamilyLikelihood
    ) or not isinstance(likelihood.family, (BernoulliFamily, PoissonFamily)):
        raise TypeError(
            "State-space Laplace sites require a certified scalar Bernoulli or "
            "Poisson natural-parameter likelihood."
        )


def _positive_real(value: Real, /, *, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar.")
    result = float(value)
    if not jnp.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _bounded_real(
    value: Real,
    /,
    *,
    name: str,
    lower: float,
    upper: float,
) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar.")
    result = float(value)
    if not jnp.isfinite(result) or result < lower or result > upper:
        raise ValueError(f"{name} must lie in [{lower}, {upper}].")
    return result


__all__ = [
    "STATE_SPACE_GP_LAPLACE_CONVERGENCE_FAILURE",
    "STATE_SPACE_GP_LAPLACE_CURVATURE_FAILURE",
    "STATE_SPACE_GP_LAPLACE_GAUSSIAN_FAILURE",
    "STATE_SPACE_GP_LAPLACE_SITE_FAILURE",
    "StateSpaceGaussianProcessApproximateResult",
    "StateSpaceGaussianProcessLaplace",
    "fit_state_space_approximate_gaussian_process",
]
