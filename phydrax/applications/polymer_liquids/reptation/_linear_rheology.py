#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule


class DoiEdwardsTubePlan(StrictModule):
    disengagement_time: float = eqx.field(static=True)
    plateau_modulus: float = eqx.field(static=True)
    odd_mode_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        disengagement_time: float,
        plateau_modulus: float,
        /,
        *,
        odd_mode_count: int = 64,
    ):
        if (
            not math.isfinite(float(disengagement_time))
            or float(disengagement_time) <= 0.0
        ):
            raise ValueError("disengagement_time must be finite and positive.")
        if not math.isfinite(float(plateau_modulus)) or float(plateau_modulus) <= 0.0:
            raise ValueError("plateau_modulus must be finite and positive.")
        if int(odd_mode_count) <= 0:
            raise ValueError("odd_mode_count must be positive.")
        self.disengagement_time = float(disengagement_time)
        self.plateau_modulus = float(plateau_modulus)
        self.odd_mode_count = int(odd_mode_count)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "doi-edwards-tube",
                "tau_d": self.disengagement_time,
                "plateau": self.plateau_modulus,
                "odd_modes": self.odd_mode_count,
            }
        )


class LinearRheologyResult(StrictModule):
    times: Array
    tube_survival: Array
    relaxation_modulus: Array
    angular_frequencies: Array
    storage_modulus: Array
    loss_modulus: Array
    mode_amplitudes: Array
    mode_relaxation_times: Array
    finite: Array
    successful: Array
    model_id: str = eqx.field(static=True)


def _spectrum_response(
    amplitudes: Array, relaxation_times: Array, frequencies: Array
) -> tuple[Array, Array]:
    omega_tau = frequencies[:, None] * relaxation_times[None, :]
    denominator = 1.0 + omega_tau * omega_tau
    storage = jnp.sum(amplitudes[None, :] * omega_tau * omega_tau / denominator, axis=1)
    loss = jnp.sum(amplitudes[None, :] * omega_tau / denominator, axis=1)
    return storage, loss


def doi_edwards_linear_rheology(
    plan: DoiEdwardsTubePlan,
    times: ArrayLike,
    angular_frequencies: ArrayLike,
    /,
) -> LinearRheologyResult:
    if not isinstance(plan, DoiEdwardsTubePlan):
        raise TypeError("plan must be DoiEdwardsTubePlan.")
    t = jnp.asarray(times)
    omega = jnp.asarray(angular_frequencies)
    if t.ndim != 1 or omega.ndim != 1:
        raise ValueError("times and angular_frequencies must be rank-1 arrays.")
    if bool(jnp.any(t < 0.0)) or bool(jnp.any(omega < 0.0)):
        raise ValueError("times and angular_frequencies must be nonnegative.")
    odd = 2 * jnp.arange(plan.odd_mode_count, dtype=t.dtype) + 1
    raw_weight = 8.0 / (jnp.pi * jnp.pi * odd * odd)
    normalized_weight = raw_weight / jnp.sum(raw_weight)
    relaxation = plan.disengagement_time / (odd * odd)
    survival = jnp.sum(
        normalized_weight[None, :] * jnp.exp(-t[:, None] / relaxation[None, :]),
        axis=1,
    )
    amplitudes = plan.plateau_modulus * normalized_weight
    storage, loss = _spectrum_response(amplitudes, relaxation, omega)
    modulus = plan.plateau_modulus * survival
    finite = (
        jnp.all(jnp.isfinite(survival))
        & jnp.all(jnp.isfinite(storage))
        & jnp.all(jnp.isfinite(loss))
    )
    successful = finite & jnp.all(survival >= 0.0) & jnp.all(survival <= 1.0)
    return LinearRheologyResult(
        t,
        survival,
        modulus,
        omega,
        storage,
        loss,
        amplitudes,
        relaxation,
        finite,
        successful,
        plan.plan_id,
    )


class LikhtmanMcLeishPlan(StrictModule):
    entanglement_count: int = eqx.field(static=True)
    entanglement_time: float = eqx.field(static=True)
    plateau_modulus: float = eqx.field(static=True)
    constraint_release_time: float = eqx.field(static=True)
    odd_mode_count: int = eqx.field(static=True)
    model_variant: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        entanglement_count: int,
        entanglement_time: float,
        plateau_modulus: float,
        /,
        *,
        constraint_release_time: float = math.inf,
        odd_mode_count: int = 64,
    ):
        if int(entanglement_count) < 2:
            raise ValueError("entanglement_count must be at least two.")
        if not math.isfinite(float(entanglement_time)) or float(entanglement_time) <= 0.0:
            raise ValueError("entanglement_time must be finite and positive.")
        if not math.isfinite(float(plateau_modulus)) or float(plateau_modulus) <= 0.0:
            raise ValueError("plateau_modulus must be finite and positive.")
        release = float(constraint_release_time)
        if not (math.isinf(release) or (math.isfinite(release) and release > 0.0)):
            raise ValueError("constraint_release_time must be positive or infinity.")
        if int(odd_mode_count) <= 0:
            raise ValueError("odd_mode_count must be positive.")
        self.entanglement_count = int(entanglement_count)
        self.entanglement_time = float(entanglement_time)
        self.plateau_modulus = float(plateau_modulus)
        self.constraint_release_time = release
        self.odd_mode_count = int(odd_mode_count)
        self.model_variant = "single_chain_lm_spectrum"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "likhtman-mcleish-linear",
                "variant": self.model_variant,
                "Z": self.entanglement_count,
                "tau_e": self.entanglement_time,
                "plateau": self.plateau_modulus,
                "tau_cr": "infinity" if math.isinf(release) else release,
                "odd_modes": self.odd_mode_count,
            }
        )


class LikhtmanMcLeishResult(StrictModule):
    rheology: LinearRheologyResult
    rouse_time: Array
    disengagement_time: Array
    zero_time_modulus: Array
    model_variant: str = eqx.field(static=True)
    successful: Array
    plan_id: str = eqx.field(static=True)


def likhtman_mcleish_linear_rheology(
    plan: LikhtmanMcLeishPlan,
    times: ArrayLike,
    angular_frequencies: ArrayLike,
    /,
) -> LikhtmanMcLeishResult:
    if not isinstance(plan, LikhtmanMcLeishPlan):
        raise TypeError("plan must be LikhtmanMcLeishPlan.")
    t = jnp.asarray(times)
    omega = jnp.asarray(angular_frequencies)
    if t.ndim != 1 or omega.ndim != 1:
        raise ValueError("times and angular_frequencies must be rank-1 arrays.")
    if bool(jnp.any(t < 0.0)) or bool(jnp.any(omega < 0.0)):
        raise ValueError("times and angular_frequencies must be nonnegative.")

    dtype = jnp.result_type(t, omega, jnp.asarray(plan.plateau_modulus))
    z = jnp.asarray(plan.entanglement_count, dtype=dtype)
    tau_e = jnp.asarray(plan.entanglement_time, dtype=dtype)
    tau_rouse = tau_e * z * z
    tau_d = 3.0 * z * tau_rouse
    odd = 2 * jnp.arange(plan.odd_mode_count, dtype=dtype) + 1
    tube_weight = 8.0 / (jnp.pi * jnp.pi * odd * odd)
    tube_weight = tube_weight / jnp.sum(tube_weight)
    tube_rate = odd * odd / tau_d
    if math.isfinite(plan.constraint_release_time):
        tube_rate = tube_rate + 1.0 / plan.constraint_release_time
    tube_tau = 1.0 / tube_rate
    tube_amplitude = 0.8 * plan.plateau_modulus * tube_weight

    rouse_mode = jnp.arange(1, plan.entanglement_count, dtype=dtype)
    rouse_tau = tau_rouse / (rouse_mode * rouse_mode)
    rouse_amplitude = jnp.full_like(
        rouse_tau, plan.plateau_modulus / (5.0 * plan.entanglement_count)
    )
    relaxation = jnp.concatenate((tube_tau, rouse_tau))
    amplitudes = jnp.concatenate((tube_amplitude, rouse_amplitude))
    decay = jnp.exp(-t[:, None] / relaxation[None, :])
    modulus = jnp.sum(amplitudes[None, :] * decay, axis=1)
    zero_modulus = jnp.sum(amplitudes)
    survival = modulus / zero_modulus
    storage, loss = _spectrum_response(amplitudes, relaxation, omega)
    finite = (
        jnp.all(jnp.isfinite(modulus))
        & jnp.all(jnp.isfinite(storage))
        & jnp.all(jnp.isfinite(loss))
        & jnp.isfinite(zero_modulus)
    )
    successful = finite & (zero_modulus > 0.0)
    rheology = LinearRheologyResult(
        t,
        survival,
        modulus,
        omega,
        storage,
        loss,
        amplitudes,
        relaxation,
        finite,
        successful,
        plan.plan_id,
    )
    return LikhtmanMcLeishResult(
        rheology,
        tau_rouse,
        tau_d,
        zero_modulus,
        plan.model_variant,
        successful,
        plan.plan_id,
    )


__all__ = [
    "DoiEdwardsTubePlan",
    "LikhtmanMcLeishPlan",
    "LikhtmanMcLeishResult",
    "LinearRheologyResult",
    "doi_edwards_linear_rheology",
    "likhtman_mcleish_linear_rheology",
]
