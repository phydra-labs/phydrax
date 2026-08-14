#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._strict import StrictModule
from ._covariance import _solve_covariance_system
from ._guided_particle import GuidedParticleFilterResult
from ._kalman import KalmanFilterResult
from ._particle import normalize_log_weights, ParticleFilterResult


class FixedLagParticleSmootherResult(StrictModule):
    """Particle marginals conditioned through at most ``lag`` future steps."""

    particles: Array
    log_weights: Array
    means: Array
    lineage_indices: Array
    horizons: Array
    valid: Array
    filter_result: Any
    lag: int = eqx.field(static=True)


class FixedLagKalmanSmootherResult(StrictModule):
    """Gaussian marginals conditioned through at most ``lag`` future steps."""

    means: Array
    covariances: Array
    horizons: Array
    valid: Array
    filter_result: KalmanFilterResult
    lag: int = eqx.field(static=True)


def _lag(value: int, /) -> int:
    lag = int(value)
    if lag < 0:
        raise ValueError("lag must be nonnegative.")
    return lag


def fixed_lag_particle_smoother(
    result: ParticleFilterResult | GuidedParticleFilterResult,
    lag: int,
    /,
) -> FixedLagParticleSmootherResult:
    """Trace descendants to form fixed-lag empirical smoothing marginals."""
    if not isinstance(result, (ParticleFilterResult, GuidedParticleFilterResult)):
        raise TypeError("result must be a bootstrap or guided particle-filter result.")
    resolved_lag = _lag(lag)
    case_shape = result.case_shape
    case_count = prod(case_shape) if case_shape else 1
    num_steps = result.problem.observations.num_steps
    count = result.num_particles
    state_shape = result.state_shape
    particles = result.particles.reshape((case_count, num_steps, count) + state_shape)
    log_weights = result.log_weights.reshape((case_count, num_steps, count))
    ancestors = result.ancestor_indices.reshape((case_count, num_steps, count))
    active = result.step_valid.reshape((case_count, num_steps))
    valid = result.valid.reshape((case_count, num_steps)) & active

    output_weights = []
    output_lineages = []
    output_horizons = []
    output_valid = []
    for case_index in range(case_count):
        last = int(np.sum(np.asarray(jax.device_get(active[case_index])))) - 1
        case_weights = []
        case_lineages = []
        case_horizons = []
        case_validity = []
        for target in range(num_steps):
            target_active = bool(active[case_index, target])
            horizon = min(target + resolved_lag, last) if target_active else target
            lineage = jnp.arange(count, dtype=jnp.int32)
            for step in range(horizon, target, -1):
                lineage = ancestors[case_index, step, lineage]
            horizon_weights = log_weights[case_index, horizon]
            grouped = jax.scipy.special.logsumexp(
                jnp.where(
                    lineage[None, :] == jnp.arange(count, dtype=jnp.int32)[:, None],
                    horizon_weights[None, :],
                    -jnp.inf,
                ),
                axis=-1,
            )
            normalized, _, grouped_valid = normalize_log_weights(grouped)
            window_valid = target_active and bool(
                jnp.all(valid[case_index, target : horizon + 1])
            )
            case_weights.append(
                jnp.where(
                    window_valid & grouped_valid,
                    normalized,
                    log_weights[case_index, target],
                )
            )
            case_lineages.append(lineage)
            case_horizons.append(jnp.asarray(horizon, dtype=jnp.int32))
            case_validity.append(jnp.asarray(window_valid & bool(grouped_valid)))
        output_weights.append(jnp.stack(case_weights))
        output_lineages.append(jnp.stack(case_lineages))
        output_horizons.append(jnp.stack(case_horizons))
        output_valid.append(jnp.stack(case_validity))

    smoothed_weights = jnp.stack(output_weights)
    means = jnp.sum(
        jnp.exp(smoothed_weights)[..., *(None for _ in state_shape)] * particles,
        axis=2,
    )
    return FixedLagParticleSmootherResult(
        particles=result.particles,
        log_weights=smoothed_weights.reshape(case_shape + (num_steps, count)),
        means=means.reshape(case_shape + (num_steps,) + state_shape),
        lineage_indices=jnp.stack(output_lineages).reshape(
            case_shape + (num_steps, count)
        ),
        horizons=jnp.stack(output_horizons).reshape(case_shape + (num_steps,)),
        valid=jnp.stack(output_valid).reshape(case_shape + (num_steps,)),
        filter_result=result,
        lag=resolved_lag,
    )


def fixed_lag_kalman_smoother(
    result: KalmanFilterResult,
    lag: int,
    /,
) -> FixedLagKalmanSmootherResult:
    """Apply an RTS recursion independently over each finite lag window."""
    if not isinstance(result, KalmanFilterResult):
        raise TypeError("result must be a KalmanFilterResult.")
    resolved_lag = _lag(lag)
    case_shape = result.case_shape
    case_count = prod(case_shape) if case_shape else 1
    num_steps = int(result.filtered_means.shape[len(case_shape)])
    state_size = prod(result.state_shape) if result.state_shape else 1
    filtered_means = result.filtered_means.reshape((case_count, num_steps, state_size))
    filtered_covariances = result.filtered_covariances.reshape(
        (case_count, num_steps, state_size, state_size)
    )
    predicted_means = result.predicted_means.reshape((case_count, num_steps, state_size))
    predicted_covariances = result.predicted_covariances.reshape(
        (case_count, num_steps, state_size, state_size)
    )
    transitions = result.transition_matrices.reshape(
        (case_count, num_steps, state_size, state_size)
    )
    active = result.step_valid.reshape((case_count, num_steps))
    valid = result.valid.reshape((case_count, num_steps)) & active

    means = []
    covariances = []
    horizons = []
    output_valid = []
    for case_index in range(case_count):
        last = int(np.sum(np.asarray(jax.device_get(active[case_index])))) - 1
        case_means = []
        case_covariances = []
        case_horizons = []
        case_validity = []
        for target in range(num_steps):
            target_active = bool(active[case_index, target])
            horizon = min(target + resolved_lag, last) if target_active else target
            mean = filtered_means[case_index, horizon]
            covariance = filtered_covariances[case_index, horizon]
            smoother_valid = jnp.asarray(True)
            for step in range(horizon - 1, target - 1, -1):
                cross = (
                    filtered_covariances[case_index, step]
                    @ transitions[case_index, step + 1].T
                )
                solve_result = _solve_covariance_system(
                    predicted_covariances[case_index, step + 1],
                    cross.T,
                )
                gain = solve_result.value.T
                mean = filtered_means[case_index, step] + gain @ (
                    mean - predicted_means[case_index, step + 1]
                )
                covariance = (
                    filtered_covariances[case_index, step]
                    + gain
                    @ (covariance - predicted_covariances[case_index, step + 1])
                    @ gain.T
                )
                covariance = 0.5 * (covariance + covariance.T)
                smoother_valid = (
                    smoother_valid
                    & jnp.all(solve_result.successful)
                    & jnp.all(jnp.isfinite(gain))
                    & jnp.all(jnp.isfinite(mean))
                    & jnp.all(jnp.isfinite(covariance))
                )
            window_valid = target_active and bool(
                smoother_valid & jnp.all(valid[case_index, target : horizon + 1])
            )
            case_means.append(
                jnp.where(window_valid, mean, filtered_means[case_index, target])
            )
            case_covariances.append(
                jnp.where(
                    window_valid,
                    covariance,
                    filtered_covariances[case_index, target],
                )
            )
            case_horizons.append(jnp.asarray(horizon, dtype=jnp.int32))
            case_validity.append(jnp.asarray(window_valid))
        means.append(jnp.stack(case_means))
        covariances.append(jnp.stack(case_covariances))
        horizons.append(jnp.stack(case_horizons))
        output_valid.append(jnp.stack(case_validity))

    return FixedLagKalmanSmootherResult(
        means=jnp.stack(means).reshape(case_shape + (num_steps,) + result.state_shape),
        covariances=jnp.stack(covariances).reshape(
            case_shape + (num_steps, state_size, state_size)
        ),
        horizons=jnp.stack(horizons).reshape(case_shape + (num_steps,)),
        valid=jnp.stack(output_valid).reshape(case_shape + (num_steps,)),
        filter_result=result,
        lag=resolved_lag,
    )


__all__ = [
    "FixedLagKalmanSmootherResult",
    "FixedLagParticleSmootherResult",
    "fixed_lag_kalman_smoother",
    "fixed_lag_particle_smoother",
]
