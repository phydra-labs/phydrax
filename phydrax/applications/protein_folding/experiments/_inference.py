# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Native least-squares/MCMC execution and explicitly scoped uncertainty."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ....optim import (
    AbstractLeastSquaresMethod,
    least_squares,
    LeastSquaresResult,
    LevenbergMarquardt,
    OptimizationTermination,
)
from ....uq import MCMCResult, ParameterSpace, PosteriorProblem, sample_nuts
from ._experiment import PreparedProteinExperiments
from ._models import ThermodynamicConvention


@dataclass(frozen=True)
class ExperimentIdentifiability:
    """Local, noise-whitened sensitivity in standardized free coordinates.

    Null vectors are rows in the reported free_names order. Full local rank
    does not establish global identifiability (e.g. parallel-path permutations).
    A prior never changes this likelihood-only identifiability evidence.
    """

    free_names: tuple[str, ...]
    sensitivity: Array
    singular_values: Array
    right_singular_vectors: Array
    rank: int
    threshold: float
    condition_number: float

    @property
    def locally_identifiable(self):
        return self.rank == len(self.free_names)

    @property
    def null_vectors(self):
        return self.right_singular_vectors[self.rank :]


def protein_experiment_identifiability(
    problem, coordinates=None, *, relative_tolerance=None
):
    """Host rank decomposition of the real prepared residual Jacobian."""
    z = problem.initial_coordinates if coordinates is None else jnp.asarray(coordinates)
    if z.shape != problem.initial_coordinates.shape or np.any(
        ~np.isfinite(np.asarray(z))
    ):
        raise ValueError(
            "Identifiability coordinates must be finite and match free parameters."
        )
    sensitivity = jax.jacrev(problem.residual)(z)
    device_values = np.asarray(sensitivity)
    matrix = device_values.astype(float)
    if np.any(~np.isfinite(matrix)):
        raise ValueError("Nonfinite model derivatives cannot establish identifiability.")
    _, singular, vt = np.linalg.svd(
        matrix, full_matrices=matrix.shape[0] < matrix.shape[1]
    )
    tolerance = (
        10 * max(matrix.shape) * np.finfo(device_values.dtype).eps
        if relative_tolerance is None
        else float(relative_tolerance)
    )
    if not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("relative_tolerance must be finite and positive.")
    threshold = tolerance * max(float(np.max(singular, initial=0)), 1.0)
    rank = int(np.sum(singular > threshold))
    count = z.size
    condition = (
        float(singular[0] / singular[-1]) if count and rank == count else float("inf")
    )
    return ExperimentIdentifiability(
        problem.parameters.free_names,
        sensitivity,
        jnp.asarray(singular),
        jnp.asarray(vt),
        rank,
        threshold,
        condition,
    )


@dataclass(frozen=True)
class ProteinExperimentFit:
    """Fit outcome retaining native solver status and likelihood-only evidence.

    covariance is the local inverse Fisher approximation in free physical
    parameter units, only supplied for an accepted full-rank fit. It is NOT a
    posterior credible interval and is never a pseudoinverse with zero error
    bars along an unidentifiable direction.
    """

    problem: PreparedProteinExperiments
    optimization: LeastSquaresResult
    identifiability: ExperimentIdentifiability
    covariance: Array | None

    @property
    def coordinates(self):
        return self.optimization.parameters

    @property
    def named_parameters(self):
        return self.problem.parameters.named_values(self.coordinates)

    def predict(self):
        return self.problem.predict(self.coordinates)


def fit_protein_experiments(
    problem: PreparedProteinExperiments,
    /,
    *,
    initial_coordinates=None,
    method: AbstractLeastSquaresMethod | None = None,
    termination: OptimizationTermination | None = None,
    identifiability_tolerance=None,
):
    """Execute native nonlinear least squares, then compute host rank evidence.

    All named models use this same real fit path; no forward-only specialized
    model is advertised as a fitter. The residual/likelihood are JIT/grad safe;
    this wrapper's host diagnostics and optimizer decisions are not pathwise
    derivatives of a kinetic process or posterior sampler.
    """
    if not isinstance(problem, PreparedProteinExperiments):
        raise TypeError("problem must be PreparedProteinExperiments.")
    if not problem.parameters.free_names:
        raise ValueError("Fitting requires at least one explicitly free parameter.")
    initial = (
        problem.initial_coordinates
        if initial_coordinates is None
        else jnp.asarray(initial_coordinates)
    )
    if initial.shape != problem.initial_coordinates.shape or np.any(
        ~np.isfinite(np.asarray(initial))
    ):
        raise ValueError(
            "Initial fit coordinates must be finite and match the parameter map."
        )
    result = least_squares(
        lambda z, args: problem.residual(z),
        initial,
        method=LevenbergMarquardt() if method is None else method,
        termination=termination,
    )
    evidence = protein_experiment_identifiability(
        problem, result.parameters, relative_tolerance=identifiability_tolerance
    )
    covariance = None
    if bool(result.successful) and evidence.locally_identifiable:
        # Use the rank-revealing host SVD, not a regularized inverse of J.T@J.
        singular = np.asarray(evidence.singular_values)
        vt = np.asarray(evidence.right_singular_vectors)
        scale = np.asarray(problem.parameters.scale[problem.parameters.free_indices])
        physical_inverse = scale[:, None] * vt.T / singular[None, :]
        covariance = jnp.asarray(physical_inverse @ physical_inverse.T)
    return ProteinExperimentFit(problem, result, evidence, covariance)


def protein_experiment_posterior_problem(
    problem: PreparedProteinExperiments,
    /,
    *,
    prior_mean: ArrayLike,
    prior_standard_deviation: ArrayLike,
    initial_coordinates=None,
):
    """Build a native posterior with explicitly specified Gaussian priors on z.

    z is the named map's standardized free coordinate (not a log parameter
    unless that named physical parameter is itself a log rate). Prior scales
    are never inferred from fit covariance or from the held-out observations.
    """
    initial = (
        problem.initial_coordinates
        if initial_coordinates is None
        else jnp.asarray(initial_coordinates)
    )
    shape = problem.initial_coordinates.shape
    mean, sigma = np.broadcast_arrays(
        np.asarray(prior_mean, dtype=float),
        np.asarray(prior_standard_deviation, dtype=float),
    )
    mean, sigma = np.broadcast_to(mean, shape), np.broadcast_to(sigma, shape)
    if (
        not shape[0]
        or initial.shape != shape
        or np.any(~np.isfinite(mean))
        or np.any(~np.isfinite(sigma))
        or np.any(sigma <= 0)
    ):
        raise ValueError(
            "Posterior needs finite explicit prior means and positive scales for all free coordinates."
        )
    mean, sigma = jnp.asarray(mean), jnp.asarray(sigma)

    def log_prior(z):
        return -0.5 * jnp.sum(
            ((z - mean) / sigma) ** 2 + 2 * jnp.log(sigma) + jnp.log(2 * jnp.pi)
        )

    return PosteriorProblem(
        ParameterSpace(initial, log_prior=log_prior),
        problem.log_likelihood,
        predict=problem.predict,
        gauss_newton_residual=problem.residual,
    )


@dataclass(frozen=True)
class ProteinExperimentPosterior:
    """Actual native MCMC output; chain/draw axes and diagnostics are retained."""

    problem: PreparedProteinExperiments
    mcmc: MCMCResult

    def named_samples(self):
        raw = self.mcmc.samples
        flat = raw.reshape((-1, raw.shape[-1]))
        values = jax.vmap(self.problem.parameters.decode)(flat)
        values = values.reshape(raw.shape[:-1] + (len(self.problem.parameters.names),))
        return {
            name: values[..., index]
            for index, name in enumerate(self.problem.parameters.names)
        }

    def predictive_samples(self):
        """Conditional mean draws, not additional measurement-noise draws."""
        raw = self.mcmc.samples
        predictions = jax.vmap(self.problem.predict)(raw.reshape((-1, raw.shape[-1])))
        return tuple(
            value.reshape(raw.shape[:-1] + value.shape[1:]) for value in predictions
        )


def sample_protein_experiments(
    problem,
    /,
    *,
    key,
    prior_mean,
    prior_standard_deviation,
    initial_coordinates=None,
    num_chains=4,
    num_warmup=1000,
    num_samples=1000,
    target_acceptance_rate=0.8,
    max_num_doublings=10,
):
    """Run the existing NUTS owner on the joint normalized likelihood."""
    posterior = protein_experiment_posterior_problem(
        problem,
        prior_mean=prior_mean,
        prior_standard_deviation=prior_standard_deviation,
        initial_coordinates=initial_coordinates,
    )
    result = sample_nuts(
        posterior,
        key=key,
        num_chains=num_chains,
        num_warmup=num_warmup,
        num_samples=num_samples,
        target_acceptance_rate=target_acceptance_rate,
        max_num_doublings=max_num_doublings,
    )
    return ProteinExperimentPosterior(problem, result)


@dataclass(frozen=True)
class PhiPosterior:
    """Paired-draw Phi with undefined near-zero stability changes made explicit."""

    samples: Array
    valid: Array
    credible_interval: Array
    valid_fraction: float
    credible_mass: float
    source_id: str


def phi_posterior(
    named_samples: Mapping[str, ArrayLike],
    /,
    *,
    wild_type_stability: str,
    mutant_stability: str,
    wild_type_log_folding_rate: str,
    mutant_log_folding_rate: str,
    convention: ThermodynamicConvention,
    source_id: str,
    minimum_stability_change: float,
    credible_mass: float = 0.95,
):
    """Derive Phi from paired WT/mutant posterior samples, never fit covariance.

    Phi = RT*(log kf_WT-log kf_mut)/(dG_unfold_WT-dG_unfold_mut).
    Named stability draws must already evaluate the same condition and energy
    convention; log rates are natural logarithms with the same rate unit.
    Identical chain/draw indexing retains WT/mutant correlation. Values outside
    [0,1] are reported, not clipped. Intervals are conditional on denominator
    validity and only returned if every draw is valid (otherwise NaN).
    """
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("Posterior source identity is required.")
    if not isfinite(minimum_stability_change) or minimum_stability_change <= 0:
        raise ValueError(
            "A positive physical denominator-resolution threshold is required."
        )
    if not 0 < credible_mass < 1:
        raise ValueError("credible_mass must lie strictly between zero and one.")
    names = (
        wild_type_stability,
        mutant_stability,
        wild_type_log_folding_rate,
        mutant_log_folding_rate,
    )
    if any(name not in named_samples for name in names):
        raise ValueError("Missing named posterior variable for Phi.")
    arrays = tuple(np.asarray(named_samples[name], dtype=float) for name in names)
    if (
        arrays[0].ndim != 2
        or arrays[0].size < 2
        or any(value.shape != arrays[0].shape for value in arrays)
    ):
        raise ValueError("Phi requires aligned nonempty (chain, draw) samples.")
    wt, mutant, log_wt, log_mutant = arrays
    denominator = wt - mutant
    valid = np.logical_and.reduce(tuple(np.isfinite(value) for value in arrays)) & (
        np.abs(denominator) >= minimum_stability_change
    )
    safe = np.where(valid, denominator, 1.0)
    values = (
        convention.thermal_constant
        * convention.reference_temperature
        * (log_wt - log_mutant)
        / safe
    )
    values = np.where(valid, values, np.nan)
    quantiles = ((1 - credible_mass) / 2, (1 + credible_mass) / 2)
    interval = np.quantile(values, quantiles) if np.all(valid) else np.full(2, np.nan)
    return PhiPosterior(
        jnp.asarray(values),
        jnp.asarray(valid),
        jnp.asarray(interval),
        float(np.mean(valid)),
        credible_mass,
        source_id,
    )
