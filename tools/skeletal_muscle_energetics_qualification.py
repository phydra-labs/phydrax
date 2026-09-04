#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.skeletal_muscle.energetics import (
    integrate_metabolic_energy_joule,
    UCHIDA_UMBERGER_2010_SOURCE_REVISION,
    UCHIDA_UMBERGER_2010_VALIDATION_DOI,
    UchidaUmberger2010Parameters,
    UchidaUmberger2010Plan,
)


jax.config.update("jax_enable_x64", True)


def _source_terms(
    mass: np.ndarray,
    slow_twitch_ratio: np.ndarray,
    optimal_length: np.ndarray,
    maximum_velocity: np.ndarray,
    excitation: np.ndarray,
    activation: np.ndarray,
    force: np.ndarray,
    force_length: np.ndarray,
    fiber_length: np.ndarray,
    fiber_velocity: np.ndarray,
    *,
    aerobic_factor: float = 1.5,
    minimum_heat_rate: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the pinned source equations without Phydrax/JAX operations."""
    u_slow = slow_twitch_ratio * np.sin(0.5 * np.pi * excitation)
    u_fast = (1.0 - slow_twitch_ratio) * (
        1.0 - np.cos(0.5 * np.pi * excitation)
    )
    denominator = u_slow + u_fast
    recruited_slow = np.ones_like(denominator)
    np.divide(u_slow, denominator, out=recruited_slow, where=denominator > 0.0)

    activity = np.where(
        excitation > activation, excitation, 0.5 * (excitation + activation)
    )
    normalized_length = fiber_length / optimal_length
    normalized_velocity = fiber_velocity / optimal_length

    unscaled_amdot = 128.0 * (1.0 - recruited_slow) + 25.0
    amdot = aerobic_factor * np.power(activity, 0.6) * unscaled_amdot
    amdot = np.where(
        normalized_length <= 1.0,
        amdot,
        amdot * (0.4 + 0.6 * force_length),
    )

    alpha_fast = 153.0 / maximum_velocity
    alpha_slow = 100.0 / (maximum_velocity / 2.5)
    concentric_sdot = aerobic_factor * np.square(activity) * (
        recruited_slow
        * np.minimum(-alpha_slow * normalized_velocity, 100.0)
        - alpha_fast * normalized_velocity * (1.0 - recruited_slow)
    )
    eccentric_sdot = (
        aerobic_factor * activity * 4.0 * alpha_slow * normalized_velocity
    )
    sdot = np.where(normalized_velocity <= 0.0, concentric_sdot, eccentric_sdot)
    sdot = np.where(normalized_length > 1.0, sdot * force_length, sdot)

    wdot = -np.maximum(force, 0.0) * fiber_velocity / mass
    unclamped_total = amdot + sdot + wdot
    correction_active = unclamped_total < 0.0
    sdot = np.where(correction_active, sdot - unclamped_total, sdot)
    heat_before_floor = amdot + sdot
    floor_active = heat_before_floor < minimum_heat_rate
    specific_total = np.maximum(heat_before_floor, minimum_heat_rate) + wdot
    specific_total = np.where(
        correction_active & ~floor_active, 0.0, specific_total
    )
    total = mass * specific_total
    return amdot, sdot, wdot, total


def qualify() -> dict[str, object]:
    scenario = (
        "concentric",
        "eccentric",
        "zero-drive-floor",
        "negative-power-clamp",
    )
    mass = np.asarray((0.5, 0.8, 0.6, 0.7))
    slow_twitch_ratio = np.asarray((0.5, 0.7, 0.6, 0.4))
    optimal_length = np.asarray((0.1, 0.12, 0.11, 0.09))
    maximum_velocity = np.asarray((10.0, 9.0, 11.0, 8.0))
    excitation = np.asarray((0.8, 0.65, 0.0, 0.75))
    activation = np.asarray((0.7, 0.55, 0.0, 0.6))
    force_length = np.asarray((1.0, 0.85, 1.0, 0.9))
    fiber_length = np.asarray((0.095, 0.132, 0.11, 0.09))
    fiber_velocity = np.asarray((-0.01, 0.015, 0.0, 0.02))
    force = np.asarray((100.0, 120.0, 0.0, 0.0))

    unloaded_amdot, unloaded_sdot, _, _ = _source_terms(
        mass,
        slow_twitch_ratio,
        optimal_length,
        maximum_velocity,
        excitation,
        activation,
        force,
        force_length,
        fiber_length,
        fiber_velocity,
    )
    force[-1] = (
        mass[-1]
        * (unloaded_amdot[-1] + unloaded_sdot[-1] + 20.0)
        / fiber_velocity[-1]
    )
    expected_amdot, expected_sdot, expected_wdot, expected_total = _source_terms(
        mass,
        slow_twitch_ratio,
        optimal_length,
        maximum_velocity,
        excitation,
        activation,
        force,
        force_length,
        fiber_length,
        fiber_velocity,
    )

    plan = UchidaUmberger2010Plan(
        UchidaUmberger2010Parameters(
            jnp.asarray(mass),
            jnp.asarray(slow_twitch_ratio),
            jnp.asarray(optimal_length),
            jnp.asarray(maximum_velocity),
        ),
        scenario,
    )
    result = plan.evaluate(
        jnp.asarray(excitation),
        jnp.asarray(activation),
        jnp.asarray(force),
        jnp.asarray(force_length),
        jnp.asarray(fiber_length),
        jnp.asarray(fiber_velocity),
    )

    observed_terms = {
        "AMdot_W_per_kg": np.asarray(
            result.activation_maintenance_heat_W_per_kg
        ),
        "Sdot_W_per_kg": np.asarray(result.shortening_lengthening_heat_W_per_kg),
        "Wdot_W_per_kg": np.asarray(result.mechanical_work_W_per_kg),
        "total_W": np.asarray(result.muscle_metabolic_power_W),
    }
    expected_terms = {
        "AMdot_W_per_kg": expected_amdot,
        "Sdot_W_per_kg": expected_sdot,
        "Wdot_W_per_kg": expected_wdot,
        "total_W": expected_total,
    }
    maximum_absolute_error = {
        name: float(np.max(np.abs(observed_terms[name] - expected)))
        for name, expected in expected_terms.items()
    }
    tolerance = 1.0e-10

    time = jnp.asarray((0.0, 0.5, 1.0))
    trace = jnp.stack((result.muscle_metabolic_power_W,) * 3)
    energy = integrate_metabolic_energy_joule(time, trace)
    integration_error = float(
        jnp.max(jnp.abs(energy - result.muscle_metabolic_power_W))
    )
    passed = (
        bool(result.evidence.successful)
        and result.mechanical_work_W_per_kg[0] > 0.0
        and result.mechanical_work_W_per_kg[1] < 0.0
        and bool(result.evidence.heat_floor_active[2])
        and bool(result.evidence.negative_power_correction_active[3])
        and bool(jnp.all(result.muscle_metabolic_power_W >= -tolerance))
        and all(error <= tolerance for error in maximum_absolute_error.values())
        and integration_error <= tolerance
    )
    return {
        "qualification": "uchida-umberger-2010-muscle-metabolic-power",
        "reference_revision": UCHIDA_UMBERGER_2010_SOURCE_REVISION,
        "validation_doi": UCHIDA_UMBERGER_2010_VALIDATION_DOI,
        "passed": bool(passed),
        "scenarios": list(scenario),
        "expected_source_terms": {
            name: values.tolist() for name, values in expected_terms.items()
        },
        "observed_terms": {
            name: values.tolist() for name, values in observed_terms.items()
        },
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_allowed_error": tolerance,
        "integrated_energy_J": np.asarray(energy).tolist(),
        "integration_error_J": integration_error,
        "negative_power_correction_active": (
            result.evidence.negative_power_correction_active.tolist()
        ),
        "claim_scope": "phenomenological muscle-only power; no ATP, heat-field, or temperature claim",
        "thermal_status": (
            "not implemented: no source-backed conversion from this total muscle "
            "power to local retained heat density, perfusion, geometry, and boundary data"
        ),
    }


def main() -> None:
    payload = qualify()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
