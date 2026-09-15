#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Controlled phase/sign evidence with fail-closed reweighting abstention."""

from __future__ import annotations

import enum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...uq import correlated_observable_diagnostics, CorrelatedObservablePolicy


class ControlledSignStudyStatus(enum.IntEnum):
    SUCCESS = 0
    NONFINITE = 1
    INSUFFICIENT_DRAWS = 2
    PHASE_CANCELLATION = 3
    INSUFFICIENT_EFFECTIVE_SAMPLES = 4


class ControlledSignStudyPlan(StrictModule, NonTrainableState):
    minimum_chains: int = eqx.field(static=True)
    minimum_draws: int = eqx.field(static=True)
    minimum_average_phase: float = eqx.field(static=True)
    minimum_effective_sample_size: float = eqx.field(static=True)
    maximum_lag: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        minimum_chains: int = 1,
        minimum_draws: int = 16,
        minimum_average_phase: float = 0.05,
        minimum_effective_sample_size: float = 8.0,
        maximum_lag: int = 64,
    ):
        chains, draws, lag = int(minimum_chains), int(minimum_draws), int(maximum_lag)
        phase, effective = (
            float(minimum_average_phase),
            float(minimum_effective_sample_size),
        )
        if (
            chains < 1
            or draws < 4
            or lag < 1
            or not np.isfinite(phase)
            or not 0.0 < phase <= 1.0
            or not np.isfinite(effective)
            or effective <= 1.0
        ):
            raise ValueError("Controlled sign-study thresholds are invalid.")
        self.minimum_chains = chains
        self.minimum_draws = draws
        self.minimum_average_phase = phase
        self.minimum_effective_sample_size = effective
        self.maximum_lag = lag
        self.plan_id = canonical_fingerprint(
            {
                "kind": "controlled-sign-study-plan",
                "minimum_chains": chains,
                "minimum_draws": draws,
                "minimum_average_phase": phase,
                "minimum_effective_sample_size": effective,
                "maximum_lag": lag,
            }
        )


class ControlledSignStudyResult(StrictModule, NonTrainableState):
    raw_weights: Array
    raw_phases: Array
    raw_observables: Array
    average_phase: Array
    average_sign: Array
    phase_covariance: Array
    phase_observable_covariance: Array
    integrated_autocorrelation_time: Array
    effective_sample_size: Array
    reweighted_mean: Array
    reweighted_standard_error: Array
    finite: Array
    abstained: Array
    status: Array
    plan_id: str = eqx.field(static=True)
    study_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(ControlledSignStudyStatus.SUCCESS)


def evaluate_controlled_sign_study(
    plan: ControlledSignStudyPlan,
    complex_weights: ArrayLike,
    observables: ArrayLike,
    /,
    *,
    study_id: str,
) -> ControlledSignStudyResult:
    """Diagnose phase-quenched chains; never infer or claim a sign cure."""
    if not isinstance(plan, ControlledSignStudyPlan):
        raise TypeError("plan must be ControlledSignStudyPlan.")
    weights = jnp.asarray(complex_weights)
    values = jnp.asarray(observables)
    if (
        weights.ndim != 2
        or weights.shape[0] < 1
        or weights.shape[1] < 4
        or values.shape[:2] != weights.shape
        or values.ndim < 2
    ):
        raise ValueError("weights and observables need leading (chain, draw>=4) axes.")
    if not jnp.issubdtype(weights.dtype, jnp.complexfloating):
        weights = weights.astype(complex)
    if not str(study_id):
        raise ValueError("study_id must be non-empty.")
    magnitude = jnp.abs(weights)
    phases = weights / jnp.where(magnitude > 0.0, magnitude, 1.0)
    finite = (
        jnp.all(jnp.isfinite(jnp.real(weights)))
        & jnp.all(jnp.isfinite(jnp.imag(weights)))
        & jnp.all(magnitude > 0.0)
        & jnp.all(jnp.isfinite(values))
    )
    components = jnp.stack((jnp.real(phases), jnp.imag(phases)), axis=-1)
    policy = CorrelatedObservablePolicy(
        max_lag=min(plan.maximum_lag, weights.shape[1] - 1), minimum_draws=4
    )
    diagnostics = correlated_observable_diagnostics(components, policy=policy)
    sample_count = float(weights.size)
    varying = diagnostics.variance > 0.0
    tau_components = jnp.where(varying, diagnostics.integrated_autocorrelation_time, 1.0)
    ess_components = jnp.where(varying, diagnostics.effective_sample_size, sample_count)
    integrated_tau = jnp.max(tau_components)
    effective = jnp.min(ess_components)
    flat_components = components.reshape((-1, 2))
    centered_phase = flat_components - jnp.mean(flat_components, axis=0)
    phase_covariance = centered_phase.T @ centered_phase / max(weights.size - 1, 1)
    value_shape = values.shape[2:]
    flat_values = values.reshape((weights.size, -1))
    centered_values = flat_values - jnp.mean(flat_values, axis=0)
    phase_observable_covariance = (
        centered_phase.T @ centered_values / max(weights.size - 1, 1)
    )
    average_phase = jnp.mean(phases)
    average_sign = jnp.abs(average_phase)
    numerator_samples = phases.reshape(weights.shape + (1,) * len(value_shape)) * values
    numerator = jnp.mean(numerator_samples, axis=(0, 1))
    safe_denominator = jnp.where(average_sign > 0.0, average_phase, 1.0 + 0.0j)
    reweighted = numerator / safe_denominator
    centered_ratio = (
        numerator_samples
        - phases.reshape(weights.shape + (1,) * len(value_shape)) * reweighted
    )
    variance = jnp.sum(jnp.abs(centered_ratio) ** 2, axis=(0, 1)) / max(
        weights.size - 1, 1
    )
    standard_error = jnp.sqrt(variance * integrated_tau / weights.size) / jnp.maximum(
        average_sign, jnp.finfo(magnitude.dtype).tiny
    )
    enough_draws = (
        weights.shape[0] >= plan.minimum_chains and weights.shape[1] >= plan.minimum_draws
    )
    phase_cancelled = jnp.isfinite(average_sign) & (
        average_sign < plan.minimum_average_phase
    )
    status = jnp.where(
        ~finite,
        int(ControlledSignStudyStatus.NONFINITE),
        jnp.where(
            phase_cancelled,
            int(ControlledSignStudyStatus.PHASE_CANCELLATION),
            jnp.where(
                not enough_draws,
                int(ControlledSignStudyStatus.INSUFFICIENT_DRAWS),
                jnp.where(
                    effective < plan.minimum_effective_sample_size,
                    int(ControlledSignStudyStatus.INSUFFICIENT_EFFECTIVE_SAMPLES),
                    int(ControlledSignStudyStatus.SUCCESS),
                ),
            ),
        ),
    ).astype(jnp.int32)
    abstained = status != int(ControlledSignStudyStatus.SUCCESS)
    nan_value = jnp.full_like(reweighted, jnp.nan)
    nan_error = jnp.full_like(standard_error, jnp.nan)
    return ControlledSignStudyResult(
        weights,
        phases,
        values,
        average_phase,
        average_sign,
        phase_covariance,
        phase_observable_covariance.reshape((2,) + value_shape),
        integrated_tau,
        effective,
        jnp.where(abstained, nan_value, reweighted),
        jnp.where(abstained, nan_error, standard_error),
        finite,
        abstained,
        status,
        plan.plan_id,
        str(study_id),
        "candidate controlled sign evidence only; no production or generic sign-cure claim",
    )


__all__ = [
    "ControlledSignStudyPlan",
    "ControlledSignStudyResult",
    "ControlledSignStudyStatus",
    "evaluate_controlled_sign_study",
]
