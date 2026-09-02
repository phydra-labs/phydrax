#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from ._minibatch_posterior import MinibatchPosteriorProblem, MinibatchSource
from ._posterior_diagnostics import _nonfinite_locations, _tree_allclose


class MinibatchPosteriorCapabilities(StrictModule):
    """Static primitives available to stochastic-gradient inference."""

    factorized_prior: bool = eqx.field(static=True)
    automatic_prior_sampling: bool = eqx.field(static=True)
    prediction: bool = eqx.field(static=True)
    observation_variance: bool = eqx.field(static=True)
    observation_sampling: bool = eqx.field(static=True)
    full_log_density: bool = eqx.field(static=True)
    control_variates: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        factorized_prior: bool,
        prediction: bool,
        observation_variance: bool,
        observation_sampling: bool,
        full_log_density: bool,
    ):
        has_factorized_prior = bool(factorized_prior)
        has_full_density = bool(full_log_density)
        self.factorized_prior = has_factorized_prior
        self.automatic_prior_sampling = has_factorized_prior
        self.prediction = bool(prediction)
        self.observation_variance = bool(observation_variance)
        self.observation_sampling = bool(observation_sampling)
        self.full_log_density = has_full_density
        self.control_variates = True

    def as_dict(self) -> dict[str, bool]:
        return {
            "factorized_prior": self.factorized_prior,
            "automatic_prior_sampling": self.automatic_prior_sampling,
            "prediction": self.prediction,
            "observation_variance": self.observation_variance,
            "observation_sampling": self.observation_sampling,
            "full_log_density": self.full_log_density,
            "control_variates": self.control_variates,
        }


class MinibatchPosteriorDiagnostics(StrictModule):
    """Factor, source, stochastic-gradient, and optional full-density checks."""

    capabilities: MinibatchPosteriorCapabilities
    initial_log_density_estimate: Array
    epoch_log_density: Array
    stochastic_gradient_norm: Array
    epoch_gradient_norm: Array
    epoch_active_factor_count: int = eqx.field(static=True)
    source_fingerprint: str = eqx.field(static=True)
    nonfinite_gradient_locations: tuple[str, ...] = eqx.field(static=True)
    repeated_evaluation_matches: bool = eqx.field(static=True)
    jit_evaluation_matches: bool = eqx.field(static=True)
    source_population_matches: bool = eqx.field(static=True)
    epoch_factor_count_matches: bool = eqx.field(static=True)
    full_log_density_matches: bool | None = eqx.field(static=True)
    full_gradient_matches: bool | None = eqx.field(static=True)
    failures: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        capabilities: MinibatchPosteriorCapabilities,
        initial_log_density_estimate: Array,
        epoch_log_density: Array,
        stochastic_gradient_norm: Array,
        epoch_gradient_norm: Array,
        epoch_active_factor_count: int,
        source_fingerprint: str,
        nonfinite_gradient_locations: tuple[str, ...],
        repeated_evaluation_matches: bool,
        jit_evaluation_matches: bool,
        source_population_matches: bool,
        epoch_factor_count_matches: bool,
        full_log_density_matches: bool | None,
        full_gradient_matches: bool | None,
        failures: tuple[str, ...],
    ):
        self.capabilities = capabilities
        self.initial_log_density_estimate = jnp.asarray(initial_log_density_estimate)
        self.epoch_log_density = jnp.asarray(epoch_log_density)
        self.stochastic_gradient_norm = jnp.asarray(stochastic_gradient_norm)
        self.epoch_gradient_norm = jnp.asarray(epoch_gradient_norm)
        self.epoch_active_factor_count = int(epoch_active_factor_count)
        self.source_fingerprint = str(source_fingerprint)
        self.nonfinite_gradient_locations = tuple(nonfinite_gradient_locations)
        self.repeated_evaluation_matches = bool(repeated_evaluation_matches)
        self.jit_evaluation_matches = bool(jit_evaluation_matches)
        self.source_population_matches = bool(source_population_matches)
        self.epoch_factor_count_matches = bool(epoch_factor_count_matches)
        self.full_log_density_matches = full_log_density_matches
        self.full_gradient_matches = full_gradient_matches
        self.failures = tuple(failures)

    @property
    def passed(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failures": self.failures,
            "initial_log_density_estimate": float(self.initial_log_density_estimate),
            "epoch_log_density": float(self.epoch_log_density),
            "stochastic_gradient_norm": float(self.stochastic_gradient_norm),
            "epoch_gradient_norm": float(self.epoch_gradient_norm),
            "epoch_active_factor_count": self.epoch_active_factor_count,
            "source_fingerprint": self.source_fingerprint,
            "nonfinite_gradient_locations": self.nonfinite_gradient_locations,
            "repeated_evaluation_matches": self.repeated_evaluation_matches,
            "jit_evaluation_matches": self.jit_evaluation_matches,
            "source_population_matches": self.source_population_matches,
            "epoch_factor_count_matches": self.epoch_factor_count_matches,
            "full_log_density_matches": self.full_log_density_matches,
            "full_gradient_matches": self.full_gradient_matches,
            "capabilities": self.capabilities.as_dict(),
        }


def diagnose_minibatch_posterior(
    problem: MinibatchPosteriorProblem,
    source: MinibatchSource,
    /,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-7,
) -> MinibatchPosteriorDiagnostics:
    """Validate factor scaling and source identity before SG-MCMC compilation."""
    if not isinstance(problem, MinibatchPosteriorProblem):
        raise TypeError("problem must be a MinibatchPosteriorProblem.")
    if not isinstance(source, MinibatchSource):
        raise TypeError("source must implement MinibatchSource.")
    relative_tolerance = float(rtol)
    absolute_tolerance = float(atol)
    if relative_tolerance < 0.0 or absolute_tolerance < 0.0:
        raise ValueError("rtol and atol cannot be negative.")
    if not source.fingerprint:
        raise ValueError("source fingerprint must be non-empty.")
    json.dumps(source.configuration(), allow_nan=False, sort_keys=True)

    batches = tuple(source.epoch(0))
    if len(batches) != int(source.batches_per_epoch):
        raise ValueError("source epoch length does not match batches_per_epoch.")
    if not batches:
        raise ValueError("source epochs must contain at least one batch.")
    if any(batch.capacity != int(source.batch_capacity) for batch in batches):
        raise ValueError("source emitted a batch with incompatible capacity.")
    audit_batches = tuple(source.audit_epoch())
    if not audit_batches:
        raise ValueError("source audit epochs must contain at least one batch.")
    if any(batch.capacity != int(source.batch_capacity) for batch in audit_batches):
        raise ValueError("source audit emitted a batch with incompatible capacity.")

    position = problem.initial_position
    first_batch = batches[0]
    value_and_grad = jax.value_and_grad(problem.log_density_estimate)
    initial_value, stochastic_gradient = value_and_grad(position, first_batch)
    repeated_value, repeated_gradient = value_and_grad(position, first_batch)
    compiled_value, compiled_gradient = eqx.filter_jit(
        lambda current, current_batch: jax.value_and_grad(problem.log_density_estimate)(
            current, current_batch
        )
    )(position, first_batch)

    def epoch_log_density(current: PyTree[Any]) -> Array:
        physical = problem.parameter_space.constrain(current)
        likelihood = sum(
            (
                jnp.sum(problem.log_likelihood_factors(physical, batch))
                for batch in audit_batches
            ),
            jnp.zeros((), dtype=float),
        )
        return (
            likelihood
            + problem.parameter_space.log_prior(physical)
            + problem.parameter_space.log_abs_det_jacobian(current)
        )

    epoch_value, epoch_gradient = jax.value_and_grad(epoch_log_density)(position)
    active_count = sum(int(batch.factor_count) for batch in audit_batches)
    repeated_matches = bool(jnp.array_equal(initial_value, repeated_value)) and (
        _tree_allclose(stochastic_gradient, repeated_gradient, rtol=0.0, atol=0.0)
    )
    jit_matches = bool(
        jnp.allclose(
            initial_value,
            compiled_value,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
    ) and _tree_allclose(
        stochastic_gradient,
        compiled_gradient,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
    source_population_matches = int(source.num_factors) == problem.num_factors
    epoch_factor_count_matches = active_count == problem.num_factors

    full_value_matches = None
    full_gradient_matches = None
    if problem.full_log_likelihood_fn is not None:
        full_value, full_gradient = jax.value_and_grad(problem.full_log_density)(position)
        full_value_matches = bool(
            jnp.allclose(
                epoch_value,
                full_value,
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            )
        )
        full_gradient_matches = _tree_allclose(
            epoch_gradient,
            full_gradient,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )

    nonfinite_gradient_locations = _nonfinite_locations(stochastic_gradient)
    stochastic_gradient_norm = _tree_l2_norm(stochastic_gradient)
    epoch_gradient_norm = _tree_l2_norm(epoch_gradient)
    capabilities = MinibatchPosteriorCapabilities(
        factorized_prior=problem.parameter_space.priors is not None,
        prediction=problem.predict_fn is not None,
        observation_variance=problem.observation_variance_fn is not None,
        observation_sampling=problem.sample_observation_fn is not None,
        full_log_density=problem.full_log_likelihood_fn is not None,
    )
    failures: list[str] = []
    if not bool(jnp.isfinite(initial_value)):
        failures.append("initial_log_density_estimate_nonfinite")
    if nonfinite_gradient_locations:
        failures.append("initial_stochastic_gradient_nonfinite")
    if not repeated_matches:
        failures.append("repeated_evaluation_mismatch")
    if not jit_matches:
        failures.append("jit_evaluation_mismatch")
    if not source_population_matches:
        failures.append("source_population_mismatch")
    if not epoch_factor_count_matches:
        failures.append("epoch_factor_count_mismatch")
    if not bool(jnp.isfinite(epoch_value)) or _nonfinite_locations(epoch_gradient):
        failures.append("epoch_density_or_gradient_nonfinite")
    if full_value_matches is False:
        failures.append("full_log_density_mismatch")
    if full_gradient_matches is False:
        failures.append("full_gradient_mismatch")

    return MinibatchPosteriorDiagnostics(
        capabilities=capabilities,
        initial_log_density_estimate=initial_value,
        epoch_log_density=epoch_value,
        stochastic_gradient_norm=stochastic_gradient_norm,
        epoch_gradient_norm=epoch_gradient_norm,
        epoch_active_factor_count=active_count,
        source_fingerprint=source.fingerprint,
        nonfinite_gradient_locations=nonfinite_gradient_locations,
        repeated_evaluation_matches=repeated_matches,
        jit_evaluation_matches=jit_matches,
        source_population_matches=source_population_matches,
        epoch_factor_count_matches=epoch_factor_count_matches,
        full_log_density_matches=full_value_matches,
        full_gradient_matches=full_gradient_matches,
        failures=tuple(failures),
    )


def _tree_l2_norm(tree: PyTree[Any], /) -> Array:
    return jnp.sqrt(
        sum(
            (
                jnp.sum(jnp.asarray(leaf, dtype=float) ** 2)
                for leaf in jax.tree_util.tree_leaves(tree)
            ),
            jnp.zeros((), dtype=float),
        )
    )


__all__ = [
    "diagnose_minibatch_posterior",
    "MinibatchPosteriorCapabilities",
    "MinibatchPosteriorDiagnostics",
]
