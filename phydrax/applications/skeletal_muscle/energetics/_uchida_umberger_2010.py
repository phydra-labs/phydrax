#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pinned Uchida--Umberger 2010 phenomenological muscle energetics."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


UCHIDA_UMBERGER_2010_SOURCE_REVISION = "86b30588374650fbaf012a345a836a64f6855522"
UCHIDA_UMBERGER_2010_VALIDATION_DOI = "10.1371/journal.pone.0150378"


def _vector(value: ArrayLike, name: str, /) -> Array:
    result = jnp.asarray(value)
    if result.ndim != 1 or result.shape[0] == 0:
        raise ValueError(f"{name} must be one nonempty muscle vector.")
    if not jnp.issubdtype(result.dtype, jnp.inexact):
        result = result.astype(float)
    return result


class UchidaUmberger2010Parameters(StrictModule):
    muscle_mass_kg: Array
    slow_twitch_fraction: Array
    optimal_fiber_length_m: Array
    maximum_contraction_velocity_per_s: Array
    aerobic_factor: Array
    minimum_heat_rate_W_per_kg: Array

    def __init__(
        self,
        muscle_mass_kg: ArrayLike,
        slow_twitch_fraction: ArrayLike,
        optimal_fiber_length_m: ArrayLike,
        maximum_contraction_velocity_per_s: ArrayLike,
        /,
        *,
        aerobic_factor: ArrayLike = 1.5,
        minimum_heat_rate_W_per_kg: ArrayLike = 1.0,
    ):
        vectors = tuple(
            _vector(value, name)
            for value, name in (
                (muscle_mass_kg, "muscle_mass_kg"),
                (slow_twitch_fraction, "slow_twitch_fraction"),
                (optimal_fiber_length_m, "optimal_fiber_length_m"),
                (
                    maximum_contraction_velocity_per_s,
                    "maximum_contraction_velocity_per_s",
                ),
            )
        )
        if any(value.shape != vectors[0].shape for value in vectors[1:]):
            raise ValueError("All metabolic parameter vectors must agree in shape.")
        aerobic = jnp.asarray(aerobic_factor, dtype=vectors[0].dtype)
        minimum = jnp.asarray(minimum_heat_rate_W_per_kg, dtype=vectors[0].dtype)
        if aerobic.shape != () or minimum.shape != ():
            raise ValueError("Energetics policy coefficients must be scalar.")
        self.muscle_mass_kg = vectors[0]
        self.slow_twitch_fraction = vectors[1]
        self.optimal_fiber_length_m = vectors[2]
        self.maximum_contraction_velocity_per_s = vectors[3]
        self.aerobic_factor = aerobic
        self.minimum_heat_rate_W_per_kg = minimum


class UchidaUmberger2010Evidence(StrictModule, NonTrainableState):
    finite: Array
    parameters_admissible: Array
    inputs_admissible: Array
    heat_floor_active: Array
    negative_power_correction_active: Array
    branch_smooth: Array
    successful: Array
    model_id: str = eqx.field(static=True)
    differentiation_scope: str = eqx.field(
        static=True,
        default=(
            "piecewise local; no derivative across recruitment denominator, "
            "activity, velocity, slow-rate cap, length, active-force clamp, "
            "correction, or floor switches"
        ),
    )


class UchidaUmberger2010Result(StrictModule, NonTrainableState):
    activation_maintenance_heat_W_per_kg: Array
    shortening_lengthening_heat_W_per_kg: Array
    mechanical_work_W_per_kg: Array
    heat_rate_W_per_kg: Array
    muscle_metabolic_power_W: Array
    total_muscle_metabolic_power_W: Array
    slow_recruitment_fraction: Array
    evidence: UchidaUmberger2010Evidence
    model_id: str = eqx.field(static=True)


class UchidaUmberger2010Plan(StrictModule):
    """Algebraic muscle-only metabolic power with pinned OpenSim policy."""

    parameters: UchidaUmberger2010Parameters
    muscle_ids: tuple[str, ...] = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameters: UchidaUmberger2010Parameters,
        muscle_ids: tuple[str, ...],
        /,
    ):
        if not isinstance(parameters, UchidaUmberger2010Parameters):
            raise TypeError("parameters must be UchidaUmberger2010Parameters.")
        ids = tuple(str(value).strip() for value in muscle_ids)
        if (
            len(ids) != parameters.muscle_mass_kg.shape[0]
            or any(not value for value in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("muscle_ids must uniquely match parameter vectors.")
        self.parameters = parameters
        self.muscle_ids = ids
        self.model_id = canonical_fingerprint(
            {
                "kind": "uchida-umberger-2010-muscle-metabolic-power",
                "reference_revision": UCHIDA_UMBERGER_2010_SOURCE_REVISION,
                "validation_doi": UCHIDA_UMBERGER_2010_VALIDATION_DOI,
                "muscle_ids": ids,
                "negative_work": "included-immediately-dissipated",
                "recruitment": "bhargava-orderly",
                "basal": "excluded",
            }
        )

    def evaluate(
        self,
        excitation: ArrayLike,
        activation: ArrayLike,
        active_fiber_force_N: ArrayLike,
        active_force_length_multiplier: ArrayLike,
        fiber_length_m: ArrayLike,
        fiber_velocity_m_per_s: ArrayLike,
        /,
    ) -> UchidaUmberger2010Result:
        p = self.parameters
        values = tuple(
            _vector(value, name)
            for value, name in (
                (excitation, "excitation"),
                (activation, "activation"),
                (active_fiber_force_N, "active_fiber_force_N"),
                (
                    active_force_length_multiplier,
                    "active_force_length_multiplier",
                ),
                (fiber_length_m, "fiber_length_m"),
                (fiber_velocity_m_per_s, "fiber_velocity_m_per_s"),
            )
        )
        expected = p.muscle_mass_kg.shape
        if any(value.shape != expected for value in values):
            raise ValueError(f"Energetics inputs must have shape {expected}.")
        excitation_, activation_, force, force_length, length, velocity = values
        slow_drive = p.slow_twitch_fraction * jnp.sin(0.5 * jnp.pi * excitation_)
        fast_drive = (1.0 - p.slow_twitch_fraction) * (
            1.0 - jnp.cos(0.5 * jnp.pi * excitation_)
        )
        denominator = slow_drive + fast_drive
        safe_denominator = jnp.where(denominator > 0.0, denominator, 1.0)
        slow_fraction = jnp.where(
            denominator > 0.0, slow_drive / safe_denominator, 1.0
        )
        activity = jnp.where(
            excitation_ > activation_, excitation_, 0.5 * (excitation_ + activation_)
        )
        normalized_length = length / p.optimal_fiber_length_m
        normalized_velocity_per_s = velocity / p.optimal_fiber_length_m
        coefficient = 128.0 * (1.0 - slow_fraction) + 25.0
        activation_maintenance = p.aerobic_factor * activity**0.6 * coefficient
        activation_maintenance = jnp.where(
            normalized_length <= 1.0,
            activation_maintenance,
            activation_maintenance * (0.4 + 0.6 * force_length),
        )
        alpha_fast = 153.0 / p.maximum_contraction_velocity_per_s
        alpha_slow = 100.0 / (
            p.maximum_contraction_velocity_per_s / 2.5
        )
        slow_shortening_rate = -alpha_slow * normalized_velocity_per_s
        shortening = p.aerobic_factor * activity**2 * (
            slow_fraction * jnp.minimum(slow_shortening_rate, 100.0)
            - alpha_fast * normalized_velocity_per_s * (1.0 - slow_fraction)
        )
        lengthening = (
            p.aerobic_factor
            * activity
            * (4.0 * alpha_slow)
            * normalized_velocity_per_s
        )
        shortening_lengthening = jnp.where(
            normalized_velocity_per_s <= 0.0, shortening, lengthening
        )
        shortening_lengthening = jnp.where(
            normalized_length > 1.0,
            shortening_lengthening * force_length,
            shortening_lengthening,
        )
        mechanical_work = -jnp.maximum(force, 0.0) * velocity / p.muscle_mass_kg
        raw_total = activation_maintenance + shortening_lengthening + mechanical_work
        correction_active = raw_total < 0.0
        shortening_lengthening = jnp.where(
            correction_active,
            shortening_lengthening - raw_total,
            shortening_lengthening,
        )
        heat_before_floor = activation_maintenance + shortening_lengthening
        heat_floor_active = heat_before_floor < p.minimum_heat_rate_W_per_kg
        heat = jnp.maximum(heat_before_floor, p.minimum_heat_rate_W_per_kg)
        specific_power = heat + mechanical_work
        specific_power = jnp.where(
            correction_active & ~heat_floor_active, 0.0, specific_power
        )
        power = p.muscle_mass_kg * specific_power

        parameter_values = jnp.concatenate(
            (
                p.muscle_mass_kg,
                p.slow_twitch_fraction,
                p.optimal_fiber_length_m,
                p.maximum_contraction_velocity_per_s,
                jnp.asarray((p.aerobic_factor, p.minimum_heat_rate_W_per_kg)),
            )
        )
        parameters_valid = (
            jnp.all(jnp.isfinite(parameter_values))
            & jnp.all(p.muscle_mass_kg > 0.0)
            & jnp.all((p.slow_twitch_fraction >= 0.0) & (p.slow_twitch_fraction <= 1.0))
            & jnp.all(p.optimal_fiber_length_m > 0.0)
            & jnp.all(p.maximum_contraction_velocity_per_s > 0.0)
            & (p.aerobic_factor > 0.0)
            & (p.minimum_heat_rate_W_per_kg >= 0.0)
        )
        input_values = jnp.concatenate(values)
        inputs_valid = (
            jnp.all(jnp.isfinite(input_values))
            & jnp.all((excitation_ >= 0.0) & (excitation_ <= 1.0))
            & jnp.all((activation_ >= 0.0) & (activation_ <= 1.0))
            & jnp.all(force >= 0.0)
            & jnp.all(force_length >= 0.0)
            & jnp.all(length > 0.0)
        )
        finite = (
            jnp.all(jnp.isfinite(activation_maintenance))
            & jnp.all(jnp.isfinite(shortening_lengthening))
            & jnp.all(jnp.isfinite(mechanical_work))
            & jnp.all(jnp.isfinite(power))
        )
        eps = 16.0 * jnp.finfo(power.dtype).eps
        branch_smooth = jnp.all(
            (jnp.abs(excitation_) > eps)
            & (jnp.abs(denominator) > eps)
            & (jnp.abs(excitation_ - activation_) > eps)
            & (jnp.abs(normalized_length - 1.0) > eps)
            & (jnp.abs(normalized_velocity_per_s) > eps)
            & (jnp.abs(force) > eps)
            & (jnp.abs(slow_shortening_rate - 100.0) > eps)
            & (jnp.abs(raw_total) > eps)
            & (jnp.abs(heat_before_floor - p.minimum_heat_rate_W_per_kg) > eps)
        )
        successful = parameters_valid & inputs_valid & finite
        evidence = UchidaUmberger2010Evidence(
            finite,
            parameters_valid,
            inputs_valid,
            heat_floor_active,
            correction_active,
            branch_smooth,
            successful,
            self.model_id,
        )
        return UchidaUmberger2010Result(
            activation_maintenance,
            shortening_lengthening,
            mechanical_work,
            heat,
            power,
            jnp.sum(power),
            slow_fraction,
            evidence,
            self.model_id,
        )


def integrate_metabolic_energy_joule(
    time_s: ArrayLike, muscle_metabolic_power_W: ArrayLike, /
) -> Array:
    """Integrate one time-by-muscle power trace with the trapezoidal rule."""
    time = jnp.asarray(time_s)
    power = jnp.asarray(muscle_metabolic_power_W)
    if time.ndim != 1 or power.ndim != 2 or power.shape[0] != time.shape[0]:
        raise ValueError("time_s and power must have shapes (time,) and (time, muscle).")
    if time.shape[0] < 2:
        raise ValueError("At least two power samples are required.")
    time = eqx.error_if(
        time,
        jnp.any(~jnp.isfinite(time)) | jnp.any(jnp.diff(time) <= 0.0),
        "time_s must be finite and strictly increasing.",
    )
    power = eqx.error_if(
        power,
        jnp.any(~jnp.isfinite(power)) | jnp.any(power < 0.0),
        "muscle_metabolic_power_W must be finite and non-negative.",
    )
    interval = jnp.diff(time)
    return jnp.sum(
        interval[:, None] * 0.5 * (power[:-1] + power[1:]), axis=0
    )


__all__ = [
    "UCHIDA_UMBERGER_2010_SOURCE_REVISION",
    "UCHIDA_UMBERGER_2010_VALIDATION_DOI",
    "UchidaUmberger2010Evidence",
    "UchidaUmberger2010Parameters",
    "UchidaUmberger2010Plan",
    "UchidaUmberger2010Result",
    "integrate_metabolic_energy_joule",
]
