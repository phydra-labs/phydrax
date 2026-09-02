#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


def _logsumexp(value: Array, axis=None):
    maximum = jnp.max(value, axis=axis, keepdims=True)
    result = maximum + jnp.log(
        jnp.sum(jnp.exp(value - maximum), axis=axis, keepdims=True)
    )
    return jnp.squeeze(result, axis=axis)


class ReducedPotentialSamples(StrictModule, NonTrainableState):
    values: Array
    state_counts: Array
    origin_states: Array
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        state_counts: ArrayLike,
        origin_states: ArrayLike,
        /,
        *,
        source_id: str | None = None,
    ):
        potential = jnp.asarray(values, dtype=float)
        counts = jnp.asarray(state_counts, dtype=jnp.int32)
        origin = jnp.asarray(origin_states, dtype=jnp.int32)
        if (
            potential.ndim != 2
            or counts.shape != potential.shape[:1]
            or origin.shape != potential.shape[1:]
        ):
            raise ValueError(
                "Reduced potentials must have shape (states,samples) with aligned counts and origins."
            )
        if (
            int(jnp.sum(counts)) != potential.shape[1]
            or jnp.any(counts < 0)
            or jnp.any(origin < 0)
            or jnp.any(origin >= potential.shape[0])
        ):
            raise ValueError("Reduced-potential state counts or origins are invalid.")
        if not bool(jnp.all(jnp.isfinite(potential))):
            raise ValueError("Reduced potentials must be finite.")
        self.values = potential
        self.state_counts = counts
        self.origin_states = origin
        self.source_id = (
            canonical_fingerprint(
                {
                    "kind": "reduced-potential-samples",
                    "shape": list(potential.shape),
                    "counts": np.asarray(counts).tolist(),
                }
            )
            if source_id is None
            else str(source_id)
        )


class FreeEnergyResult(StrictModule):
    free_energies: Array
    differences: Array
    standard_errors: Array
    effective_sample_size: Array
    overlap: Array
    iterations: Array
    converged: Array
    method: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)


def free_energy_perturbation(
    work: ArrayLike, /, *, source_id: str = "fep"
) -> FreeEnergyResult:
    value = jnp.asarray(work, dtype=float).reshape((-1,))
    if value.size == 0 or not bool(jnp.all(jnp.isfinite(value))):
        raise ValueError("FEP requires non-empty finite work samples.")
    log_weights = -value
    maximum = jnp.max(log_weights)
    scaled_weights = jnp.exp(log_weights - maximum)
    mean_weight = jnp.mean(scaled_weights)
    delta = -(maximum + jnp.log(mean_weight))
    normalized = scaled_weights / jnp.sum(scaled_weights)
    ess = 1.0 / jnp.sum(normalized**2)
    weight_variance = jnp.mean((scaled_weights - mean_weight) ** 2)
    error = jnp.sqrt(weight_variance / value.size) / mean_weight
    matrix = jnp.asarray([[0.0, delta], [-delta, 0.0]])
    return FreeEnergyResult(
        jnp.asarray([0.0, delta]),
        matrix,
        jnp.asarray([[0.0, error], [error, 0.0]]),
        jnp.asarray([value.size, ess]),
        jnp.eye(2),
        jnp.asarray(1),
        jnp.asarray(True),
        "fep",
        source_id,
    )


def thermodynamic_integration(
    lambda_values: ArrayLike,
    derivative_means: ArrayLike,
    derivative_errors: ArrayLike | None = None,
    /,
    *,
    source_id: str = "ti",
) -> FreeEnergyResult:
    lambdas = jnp.asarray(lambda_values, dtype=float).reshape((-1,))
    derivative = jnp.asarray(derivative_means, dtype=float).reshape((-1,))
    if (
        lambdas.shape != derivative.shape
        or lambdas.size < 2
        or bool(jnp.any(jnp.diff(lambdas) <= 0.0))
        or not bool(jnp.all(jnp.isfinite(lambdas)))
        or not bool(jnp.all(jnp.isfinite(derivative)))
    ):
        raise ValueError(
            "Thermodynamic integration requires finite increasing aligned lambda points."
        )
    increments = 0.5 * jnp.diff(lambdas) * (derivative[:-1] + derivative[1:])
    free = jnp.concatenate((jnp.zeros((1,)), jnp.cumsum(increments)))
    error_values = (
        jnp.zeros_like(derivative)
        if derivative_errors is None
        else jnp.asarray(derivative_errors, dtype=float)
    )
    if error_values.shape != derivative.shape or not bool(
        jnp.all(jnp.isfinite(error_values) & (error_values >= 0.0))
    ):
        raise ValueError(
            "Thermodynamic-integration errors must be finite and non-negative."
        )
    increment_variance = (0.5 * jnp.diff(lambdas)) ** 2 * (
        error_values[:-1] ** 2 + error_values[1:] ** 2
    )
    cumulative_error = jnp.concatenate(
        (jnp.zeros((1,)), jnp.sqrt(jnp.cumsum(increment_variance)))
    )
    difference = free[None, :] - free[:, None]
    errors = jnp.sqrt(cumulative_error[:, None] ** 2 + cumulative_error[None, :] ** 2)
    return FreeEnergyResult(
        free,
        difference,
        errors,
        jnp.full(free.shape, jnp.inf),
        jnp.eye(free.size),
        jnp.asarray(1),
        jnp.asarray(True),
        "thermodynamic-integration",
        source_id,
    )


def bennett_acceptance_ratio(
    forward_work: ArrayLike,
    reverse_work: ArrayLike,
    /,
    *,
    maximum_iterations: int = 128,
    tolerance: float = 1e-10,
    source_id: str = "bar",
) -> FreeEnergyResult:
    forward = jnp.asarray(forward_work, dtype=float).reshape((-1,))
    reverse = jnp.asarray(reverse_work, dtype=float).reshape((-1,))
    if (
        forward.size == 0
        or reverse.size == 0
        or not bool(jnp.all(jnp.isfinite(forward)))
        or not bool(jnp.all(jnp.isfinite(reverse)))
        or int(maximum_iterations) <= 0
        or float(tolerance) <= 0.0
    ):
        raise ValueError("BAR requires finite samples and positive solver controls.")
    initial = 0.5 * (jnp.mean(forward) - jnp.mean(reverse))
    log_ratio = jnp.log(forward.size / reverse.size)

    def body(_, carry):
        estimate, _ = carry
        forward_logistic = jax.nn.sigmoid(-(forward - estimate - log_ratio))
        reverse_logistic = jax.nn.sigmoid(-(reverse + estimate + log_ratio))
        function = jnp.mean(forward_logistic) - jnp.mean(reverse_logistic)
        derivative = jnp.mean(forward_logistic * (1.0 - forward_logistic)) + jnp.mean(
            reverse_logistic * (1.0 - reverse_logistic)
        )
        update = function / jnp.maximum(derivative, 1e-30)
        return estimate - update, jnp.abs(update)

    estimate, residual = jax.lax.fori_loop(
        0, int(maximum_iterations), body, (initial, jnp.asarray(jnp.inf))
    )
    overlap_value = 0.5 * (
        jnp.mean(jax.nn.sigmoid(-(forward - estimate)))
        + jnp.mean(jax.nn.sigmoid(-(reverse + estimate)))
    )
    ess = overlap_value * (forward.size + reverse.size)
    error = jnp.sqrt(1.0 / jnp.maximum(ess, 1.0))
    difference = jnp.asarray([[0.0, estimate], [-estimate, 0.0]])
    return FreeEnergyResult(
        jnp.asarray([0.0, estimate]),
        difference,
        jnp.asarray([[0.0, error], [error, 0.0]]),
        jnp.asarray([forward.size, reverse.size]),
        jnp.asarray([[1.0, overlap_value], [overlap_value, 1.0]]),
        jnp.asarray(maximum_iterations),
        residual <= tolerance,
        "bar",
        source_id,
    )


def multistate_bennett_acceptance_ratio(
    samples: ReducedPotentialSamples,
    /,
    *,
    maximum_iterations: int = 10_000,
    tolerance: float = 1e-10,
) -> FreeEnergyResult:
    if not isinstance(samples, ReducedPotentialSamples):
        raise TypeError("samples must be ReducedPotentialSamples.")
    if int(maximum_iterations) <= 0 or float(tolerance) <= 0.0:
        raise ValueError("MBAR solver controls must be positive.")
    u = samples.values
    counts = samples.state_counts.astype(u.dtype)
    count_log = jnp.where(counts > 0, jnp.log(counts), -jnp.inf)

    def body(_, carry):
        free, residual, iterations = carry
        denominator = _logsumexp(count_log[:, None] + free[:, None] - u, axis=0)
        updated = -_logsumexp(-u - denominator[None, :], axis=1)
        updated = updated - updated[0]
        change = jnp.max(jnp.abs(updated - free))
        choose = residual > tolerance
        return (
            jnp.where(choose, updated, free),
            jnp.where(choose, change, residual),
            iterations + choose.astype(jnp.int32),
        )

    free, residual, iterations = jax.lax.fori_loop(
        0,
        int(maximum_iterations),
        body,
        (
            jnp.zeros((u.shape[0],)),
            jnp.asarray(jnp.inf),
            jnp.zeros((), dtype=jnp.int32),
        ),
    )
    denominator = _logsumexp(count_log[:, None] + free[:, None] - u, axis=0)
    weights = jnp.exp(free[:, None] - u - denominator[None, :])
    overlap = contract("in,jn->ij", weights, weights * counts[:, None])
    ess = 1.0 / jnp.sum(weights**2, axis=1)
    errors = jnp.sqrt(
        1.0 / jnp.maximum(ess[:, None], 1.0) + 1.0 / jnp.maximum(ess[None, :], 1.0)
    )
    return FreeEnergyResult(
        free,
        free[None, :] - free[:, None],
        errors,
        ess,
        overlap,
        iterations,
        residual <= tolerance,
        "mbar",
        samples.source_id,
    )


__all__ = [
    "FreeEnergyResult",
    "ReducedPotentialSamples",
    "bennett_acceptance_ratio",
    "free_energy_perturbation",
    "multistate_bennett_acceptance_ratio",
    "thermodynamic_integration",
]
