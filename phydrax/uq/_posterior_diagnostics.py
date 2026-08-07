#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from ._posterior import PosteriorProblem


class PosteriorCapabilities(StrictModule):
    """Static posterior primitives available to inference methods."""

    factorized_prior: bool = eqx.field(static=True)
    automatic_prior_sampling: bool = eqx.field(static=True)
    prediction: bool = eqx.field(static=True)
    observation_variance: bool = eqx.field(static=True)
    observation_sampling: bool = eqx.field(static=True)
    gauss_newton_residual: bool = eqx.field(static=True)
    automatic_flow_nuts_initialization: bool = eqx.field(static=True)
    automatic_tempered_smc_initialization: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        factorized_prior: bool,
        prediction: bool,
        observation_variance: bool,
        observation_sampling: bool,
        gauss_newton_residual: bool,
    ):
        has_factorized_prior = bool(factorized_prior)
        self.factorized_prior = has_factorized_prior
        self.automatic_prior_sampling = has_factorized_prior
        self.prediction = bool(prediction)
        self.observation_variance = bool(observation_variance)
        self.observation_sampling = bool(observation_sampling)
        self.gauss_newton_residual = bool(gauss_newton_residual)
        self.automatic_flow_nuts_initialization = has_factorized_prior
        self.automatic_tempered_smc_initialization = has_factorized_prior

    def as_dict(self) -> dict[str, bool]:
        """Return machine-readable capability flags."""
        return {
            "factorized_prior": self.factorized_prior,
            "automatic_prior_sampling": self.automatic_prior_sampling,
            "prediction": self.prediction,
            "observation_variance": self.observation_variance,
            "observation_sampling": self.observation_sampling,
            "gauss_newton_residual": self.gauss_newton_residual,
            "automatic_flow_nuts_initialization": (
                self.automatic_flow_nuts_initialization
            ),
            "automatic_tempered_smc_initialization": (
                self.automatic_tempered_smc_initialization
            ),
        }


class PosteriorDiagnostics(StrictModule):
    """Structured eager, compiled, vectorized, and coordinate checks."""

    capabilities: PosteriorCapabilities
    initial_log_density: Array
    gradient_norm: Array
    roundtrip_error: PyTree[Array]
    max_roundtrip_error: Array
    prior_sample_finite_fraction: Array | None
    nonfinite_gradient_locations: tuple[str, ...] = eqx.field(static=True)
    roundtrip_failure_locations: tuple[str, ...] = eqx.field(static=True)
    repeated_evaluation_matches: bool = eqx.field(static=True)
    jit_evaluation_matches: bool = eqx.field(static=True)
    vmap_evaluation_matches: bool = eqx.field(static=True)
    failures: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        capabilities: PosteriorCapabilities,
        initial_log_density: Array,
        gradient_norm: Array,
        roundtrip_error: PyTree[Array],
        max_roundtrip_error: Array,
        prior_sample_finite_fraction: Array | None,
        nonfinite_gradient_locations: tuple[str, ...],
        roundtrip_failure_locations: tuple[str, ...],
        repeated_evaluation_matches: bool,
        jit_evaluation_matches: bool,
        vmap_evaluation_matches: bool,
        failures: tuple[str, ...],
    ):
        self.capabilities = capabilities
        self.initial_log_density = jnp.asarray(initial_log_density)
        self.gradient_norm = jnp.asarray(gradient_norm)
        self.roundtrip_error = roundtrip_error
        self.max_roundtrip_error = jnp.asarray(max_roundtrip_error)
        self.prior_sample_finite_fraction = (
            None
            if prior_sample_finite_fraction is None
            else jnp.asarray(prior_sample_finite_fraction)
        )
        self.nonfinite_gradient_locations = tuple(nonfinite_gradient_locations)
        self.roundtrip_failure_locations = tuple(roundtrip_failure_locations)
        self.repeated_evaluation_matches = bool(repeated_evaluation_matches)
        self.jit_evaluation_matches = bool(jit_evaluation_matches)
        self.vmap_evaluation_matches = bool(vmap_evaluation_matches)
        self.failures = tuple(failures)

    @property
    def passed(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict[str, Any]:
        """Return scalar evidence, exact locations, and capability flags."""
        return {
            "passed": self.passed,
            "failures": self.failures,
            "initial_log_density": float(self.initial_log_density),
            "gradient_norm": float(self.gradient_norm),
            "max_roundtrip_error": float(self.max_roundtrip_error),
            "prior_sample_finite_fraction": (
                None
                if self.prior_sample_finite_fraction is None
                else float(self.prior_sample_finite_fraction)
            ),
            "nonfinite_gradient_locations": self.nonfinite_gradient_locations,
            "roundtrip_failure_locations": self.roundtrip_failure_locations,
            "repeated_evaluation_matches": self.repeated_evaluation_matches,
            "jit_evaluation_matches": self.jit_evaluation_matches,
            "vmap_evaluation_matches": self.vmap_evaluation_matches,
            "capabilities": self.capabilities.as_dict(),
        }


def diagnose_posterior(
    problem: PosteriorProblem,
    /,
    *,
    key: Array | None = None,
    num_prior_samples: int = 8,
    rtol: float = 1e-6,
    atol: float = 1e-7,
) -> PosteriorDiagnostics:
    """Evaluate numerical posterior contracts without selecting an inference backend."""
    if not isinstance(problem, PosteriorProblem):
        raise TypeError("problem must be a PosteriorProblem.")
    count = int(num_prior_samples)
    if count <= 0:
        raise ValueError("num_prior_samples must be positive.")
    relative_tolerance = float(rtol)
    absolute_tolerance = float(atol)
    if relative_tolerance < 0.0 or absolute_tolerance < 0.0:
        raise ValueError("rtol and atol cannot be negative.")

    capabilities = PosteriorCapabilities(
        factorized_prior=problem.parameter_space.priors is not None,
        prediction=problem.predict_fn is not None,
        observation_variance=problem.observation_variance_fn is not None,
        observation_sampling=problem.sample_observation_fn is not None,
        gauss_newton_residual=problem.gauss_newton_residual_fn is not None,
    )
    position = problem.initial_position
    value_and_grad = jax.value_and_grad(problem.log_density)
    initial_log_density, gradient = value_and_grad(position)
    repeated_log_density, repeated_gradient = value_and_grad(position)
    compiled_log_density, compiled_gradient = jax.jit(value_and_grad)(position)
    batched_position = jax.tree_util.tree_map(
        lambda value: jnp.broadcast_to(value, (2, *value.shape)),
        position,
    )
    vectorized_log_density = jax.vmap(problem.log_density)(batched_position)
    physical = problem.parameter_space.constrain(position)
    restored = problem.parameter_space.unconstrain(physical)
    roundtrip_error = jax.tree_util.tree_map(
        lambda left, right: jnp.max(jnp.abs(jnp.asarray(left) - jnp.asarray(right))),
        position,
        restored,
    )
    max_roundtrip_error = _tree_max(roundtrip_error)
    nonfinite_gradient_locations = _nonfinite_locations(gradient)
    roundtrip_failure_locations = _roundtrip_failures(
        position,
        restored,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
    repeated_matches = bool(jnp.array_equal(initial_log_density, repeated_log_density))
    repeated_matches = repeated_matches and _tree_allclose(
        gradient,
        repeated_gradient,
        rtol=0.0,
        atol=0.0,
    )
    jit_matches = bool(
        jnp.allclose(
            initial_log_density,
            compiled_log_density,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
    ) and _tree_allclose(
        gradient,
        compiled_gradient,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
    vmap_matches = bool(
        jnp.allclose(
            vectorized_log_density,
            jnp.broadcast_to(initial_log_density, (2,)),
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
    )

    prior_sample_finite_fraction = None
    if key is not None and capabilities.automatic_prior_sampling:
        prior_positions = problem.parameter_space.sample_prior(
            key,
            num_samples=count,
        )
        prior_log_density = jax.vmap(problem.log_density)(prior_positions)
        prior_sample_finite_fraction = jnp.mean(jnp.isfinite(prior_log_density))

    gradient_norm = jnp.sqrt(
        sum(
            (
                jnp.sum(jnp.asarray(leaf, dtype=float) ** 2)
                for leaf in jax.tree_util.tree_leaves(gradient)
            ),
            jnp.zeros((), dtype=float),
        )
    )
    failures: list[str] = []
    if not bool(jnp.isfinite(initial_log_density)):
        failures.append("initial_log_density_nonfinite")
    if nonfinite_gradient_locations:
        failures.append("initial_gradient_nonfinite")
    if roundtrip_failure_locations:
        failures.append("coordinate_roundtrip_failed")
    if not repeated_matches:
        failures.append("repeated_evaluation_mismatch")
    if not jit_matches:
        failures.append("jit_evaluation_mismatch")
    if not vmap_matches:
        failures.append("vmap_evaluation_mismatch")
    if prior_sample_finite_fraction is not None and not bool(
        prior_sample_finite_fraction == 1.0
    ):
        failures.append("prior_sample_log_density_nonfinite")

    return PosteriorDiagnostics(
        capabilities=capabilities,
        initial_log_density=initial_log_density,
        gradient_norm=gradient_norm,
        roundtrip_error=roundtrip_error,
        max_roundtrip_error=max_roundtrip_error,
        prior_sample_finite_fraction=prior_sample_finite_fraction,
        nonfinite_gradient_locations=nonfinite_gradient_locations,
        roundtrip_failure_locations=roundtrip_failure_locations,
        repeated_evaluation_matches=repeated_matches,
        jit_evaluation_matches=jit_matches,
        vmap_evaluation_matches=vmap_matches,
        failures=tuple(failures),
    )


def _nonfinite_locations(tree: PyTree[Any]) -> tuple[str, ...]:
    locations: list[str] = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        value = np.asarray(leaf)
        base = jax.tree_util.keystr(path) or "<root>"
        for index in np.argwhere(~np.isfinite(value)):
            suffix = ""
            if value.ndim:
                suffix = "[" + ",".join(str(int(item)) for item in index) + "]"
            locations.append(base + suffix)
    return tuple(locations)


def _roundtrip_failures(
    original: PyTree[Any],
    restored: PyTree[Any],
    *,
    rtol: float,
    atol: float,
) -> tuple[str, ...]:
    locations = []
    original_leaves = jax.tree_util.tree_flatten_with_path(original)[0]
    restored_leaves = jax.tree_util.tree_leaves(restored)
    for (path, left), right in zip(original_leaves, restored_leaves, strict=True):
        if not bool(jnp.allclose(left, right, rtol=rtol, atol=atol)):
            locations.append(jax.tree_util.keystr(path) or "<root>")
    return tuple(locations)


def _tree_allclose(left, right, *, rtol: float, atol: float) -> bool:
    comparisons = jax.tree_util.tree_map(
        lambda first, second: jnp.allclose(
            first,
            second,
            rtol=rtol,
            atol=atol,
        ),
        left,
        right,
    )
    return all(bool(value) for value in jax.tree_util.tree_leaves(comparisons))


def _tree_max(tree: PyTree[Any]) -> Array:
    leaves = [jnp.asarray(value) for value in jax.tree_util.tree_leaves(tree)]
    return jnp.max(jnp.stack(leaves))


__all__ = ["PosteriorCapabilities", "PosteriorDiagnostics", "diagnose_posterior"]
