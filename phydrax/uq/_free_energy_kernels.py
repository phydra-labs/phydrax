#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pure JAX numerical kernels for authenticated free-energy estimators."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array

from phydrax.ein import contract

from ..linalg._dense_pseudoinverse import (
    factor_pseudoinverse,
    materialize_pseudoinverse,
)
from ..linalg._policies import RankPolicy


def masked_logsumexp_kernel(value: Array, mask: Array, axis: int, /) -> Array:
    """Masked log-sum-exp with finite inactive arithmetic."""

    mask_ = jnp.broadcast_to(mask, value.shape)
    masked = jnp.where(mask_, value, -jnp.inf)
    maximum = jnp.max(masked, axis=axis, keepdims=True)
    safe_maximum = jnp.where(jnp.isfinite(maximum), maximum, 0.0)
    safe_value = jnp.where(mask_, value, safe_maximum)
    total = jnp.sum(jnp.exp(safe_value - safe_maximum) * mask_, axis=axis)
    squeezed = jnp.squeeze(safe_maximum, axis=axis)
    return jnp.where(
        total > 0.0,
        squeezed + jnp.log(jnp.maximum(total, jnp.finfo(value.dtype).tiny)),
        -jnp.inf,
    )


def fep_kernel(
    values: Array,
    mask: Array,
    sample_weight: Array,
    /,
) -> tuple[Array, Array, Array]:
    """Return FEP delta, importance ESS, and per-observation influence."""

    active = mask & (sample_weight > 0.0)
    weight = sample_weight * active
    count = jnp.sum(weight)
    log_weight = -values
    maximum = jnp.max(jnp.where(active, log_weight, -jnp.inf))
    maximum = jnp.where(jnp.isfinite(maximum), maximum, 0.0)
    safe_log_weight = jnp.where(active, log_weight, maximum)
    scaled = weight * jnp.exp(safe_log_weight - maximum)
    total = jnp.sum(scaled)
    mean = total / jnp.maximum(count, 1.0)
    delta = jnp.where(
        count > 0.0,
        -(maximum + jnp.log(jnp.maximum(mean, jnp.finfo(values.dtype).tiny))),
        0.0,
    )
    effective_sample_size = total**2 / jnp.maximum(
        jnp.sum(scaled**2), jnp.finfo(values.dtype).tiny
    )
    influence = jnp.where(
        active,
        -(scaled / jnp.maximum(mean, jnp.finfo(values.dtype).tiny) - 1.0)
        / jnp.maximum(count, 1.0),
        0.0,
    )
    return delta, effective_sample_size, influence


def bar_kernel(
    values: Array,
    forward: Array,
    reverse: Array,
    sample_weight: Array,
    /,
    *,
    maximum_iterations: int,
    tolerance: float,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Solve BAR and return estimate, residual, iterations, Fermi weights, and counts."""

    forward_weight = sample_weight * forward
    reverse_weight = sample_weight * reverse
    forward_count = jnp.sum(forward_weight)
    reverse_count = jnp.sum(reverse_weight)
    safe_forward = jnp.maximum(forward_count, 1.0)
    safe_reverse = jnp.maximum(reverse_count, 1.0)
    initial = 0.5 * (
        jnp.sum(forward_weight * values) / safe_forward
        - jnp.sum(reverse_weight * values) / safe_reverse
    )
    log_ratio = jnp.log(safe_forward / safe_reverse)
    tolerance_ = jnp.asarray(tolerance, dtype=values.dtype)

    def body(_, carry):
        estimate, residual, iterations = carry
        forward_probability = jax.nn.sigmoid(-(values - estimate + log_ratio))
        reverse_probability = jax.nn.sigmoid(-(values + estimate - log_ratio))
        function = jnp.sum(forward_weight * forward_probability) - jnp.sum(
            reverse_weight * reverse_probability
        )
        derivative = jnp.sum(
            forward_weight * forward_probability * (1.0 - forward_probability)
        ) + jnp.sum(reverse_weight * reverse_probability * (1.0 - reverse_probability))
        update = function / jnp.maximum(derivative, jnp.finfo(values.dtype).tiny)
        active_iteration = residual > tolerance_
        return (
            jnp.where(active_iteration, estimate - update, estimate),
            jnp.where(active_iteration, jnp.abs(update), residual),
            iterations + active_iteration.astype(jnp.int32),
        )

    estimate, residual, iterations = jax.lax.fori_loop(
        0,
        maximum_iterations,
        body,
        (
            initial,
            jnp.asarray(jnp.inf, dtype=values.dtype),
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    valid = (forward_count > 0.0) & (reverse_count > 0.0)
    forward_probability = jax.nn.sigmoid(-(values - estimate + log_ratio))
    reverse_probability = jax.nn.sigmoid(-(values + estimate - log_ratio))
    return (
        jnp.where(valid, estimate, 0.0),
        residual,
        iterations,
        forward_probability,
        reverse_probability,
        jnp.asarray([forward_count, reverse_count]),
    )


def thermodynamic_integration_kernel(
    values: Array,
    retained: Array,
    path_parameter: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    """Return TI free energies, state means/counts, transform, and influences."""

    counts = jnp.sum(retained, axis=1)
    means = jnp.sum(jnp.where(retained, values, 0.0), axis=1) / jnp.maximum(counts, 1)
    state_count = values.shape[0]
    increments = jnp.diff(path_parameter)
    interval_included = (
        jnp.arange(state_count)[:, None] > jnp.arange(state_count - 1)[None, :]
    ).astype(values.dtype)
    transform = jnp.zeros((state_count, state_count), dtype=values.dtype)
    transform = transform.at[:, :-1].add(0.5 * interval_included * increments[None, :])
    transform = transform.at[:, 1:].add(0.5 * interval_included * increments[None, :])
    free = transform @ means
    centered = jnp.where(
        retained,
        (values - means[:, None]) / jnp.maximum(counts[:, None], 1),
        0.0,
    )
    derivative_influence = (
        jnp.eye(state_count, dtype=values.dtype)[:, :, None] * centered[:, None, :]
    )
    observation_influence = contract(
        "ij,jkn->ikn", transform, derivative_influence
    ).reshape((state_count, -1))
    return free, means, counts, transform, observation_influence


def mbar_kernel(
    values: Array,
    origins: Array,
    sample_weight: Array,
    /,
    *,
    reference_state: int,
    maximum_iterations: int,
    tolerance: float,
) -> tuple[Array, Array, Array, Array, Array]:
    """Solve the dense MBAR fixed point in one declared gauge."""

    state_count = values.shape[0]
    safe_origin = jnp.clip(origins, 0, state_count - 1)
    counts = jnp.bincount(
        safe_origin,
        weights=sample_weight,
        length=state_count,
    )
    log_counts = jnp.where(
        counts > 0.0,
        jnp.log(jnp.maximum(counts, 1.0)),
        -jnp.inf,
    )
    active = sample_weight > 0.0
    log_sample_weight = jnp.where(
        active,
        jnp.log(jnp.maximum(sample_weight, 1.0)),
        -jnp.inf,
    )
    tolerance_ = jnp.asarray(tolerance, dtype=values.dtype)

    def body(_, carry):
        free, residual, iterations = carry
        denominator = masked_logsumexp_kernel(
            log_counts[:, None] + free[:, None] - values,
            jnp.broadcast_to(active[None, :], values.shape),
            0,
        )
        updated = -masked_logsumexp_kernel(
            log_sample_weight[None, :] - values - denominator[None, :],
            jnp.broadcast_to(active[None, :], values.shape),
            1,
        )
        updated = updated - updated[reference_state]
        change = jnp.max(jnp.abs(updated - free))
        continue_iteration = residual > tolerance_
        return (
            jnp.where(continue_iteration, updated, free),
            jnp.where(continue_iteration, change, residual),
            iterations + continue_iteration.astype(jnp.int32),
        )

    free, residual, iterations = jax.lax.fori_loop(
        0,
        maximum_iterations,
        body,
        (
            jnp.zeros((state_count,), dtype=values.dtype),
            jnp.asarray(jnp.inf, dtype=values.dtype),
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    denominator = masked_logsumexp_kernel(
        log_counts[:, None] + free[:, None] - values,
        jnp.broadcast_to(active[None, :], values.shape),
        0,
    )
    weights = jnp.where(
        active[None, :],
        jnp.exp(free[:, None] - values - denominator[None, :]),
        0.0,
    )
    return free, residual, iterations, counts, weights


def mbar_asymptotic_covariance_kernel(
    weights: Array,
    sample_weight: Array,
    counts: Array,
    /,
    *,
    reference_state: int,
    rank_tolerance: float,
) -> Array:
    """Published dense MBAR asymptotic covariance in a fixed gauge."""

    weighted = jnp.sqrt(sample_weight)[:, None] * weights.T
    factors = factor_pseudoinverse(
        weighted,
        RankPolicy(relative_cutoff=rank_tolerance),
    )
    right = jnp.conj(factors.right_adjoint.T)
    singular = factors.singular_values
    projected_counts = contract("ir,i,ij->rj", right, counts, right)
    middle = jnp.eye(singular.size, dtype=weights.dtype) - (
        singular[:, None] * projected_counts * singular[None, :]
    )
    middle_factors = factor_pseudoinverse(
        middle,
        RankPolicy(relative_cutoff=rank_tolerance),
        hermitian=True,
    )
    middle_inverse = materialize_pseudoinverse(middle_factors)
    theta = (
        (right * singular[None, :])
        @ middle_inverse
        @ (singular[:, None] * jnp.conj(right.T))
    )
    gauge = jnp.eye(weights.shape[0], dtype=weights.dtype)
    gauge = gauge.at[:, reference_state].add(-1.0)
    gauge = gauge.at[reference_state, :].set(0.0)
    covariance = gauge @ theta @ gauge.T
    return 0.5 * (covariance + covariance.T)


__all__ = [
    "bar_kernel",
    "fep_kernel",
    "masked_logsumexp_kernel",
    "mbar_asymptotic_covariance_kernel",
    "mbar_kernel",
    "thermodynamic_integration_kernel",
]
