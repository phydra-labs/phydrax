#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Qualify the Potvin--Fuglevand 2017 motor-unit population model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.skeletal_muscle.motor_units import (
    POTVIN_FUGLEVAND_2017_DOI,
    POTVIN_FUGLEVAND_2017_REFERENCE_SHA,
    PotvinFuglevand2017Plan,
)


def _scalar_reference(runtime, state, excitation: float, step_s: float):
    parameters = runtime.parameters
    threshold = np.asarray(parameters.recruitment_threshold)
    rested = np.asarray(parameters.rested_twitch_force)
    resting_time = np.asarray(parameters.resting_contraction_time_s)
    maximum_rate = np.asarray(parameters.maximum_firing_rate_hz)
    force_loss_rate = np.asarray(parameters.nominal_twitch_force_loss_per_s)
    minimum_rate = float(parameters.minimum_firing_rate_hz)
    gain = float(parameters.firing_rate_gain_hz)
    duration = np.asarray(state.recruitment_duration_s)
    capacity = np.asarray(state.current_twitch_force)

    recruited = excitation >= threshold
    unadapted = np.where(
        recruited,
        np.minimum(gain * (excitation - threshold) + minimum_rate, maximum_rate),
        0.0,
    )
    threshold_fraction = (threshold - threshold[0]) / (threshold[-1] - threshold[0])
    maximum_adaptation = np.maximum(
        float(parameters.adaptation_scale)
        * (unadapted - minimum_rate + float(parameters.derecruitment_delta_hz))
        * threshold_fraction,
        0.0,
    )
    adaptation = maximum_adaptation * (
        1.0
        - np.exp(-duration / float(parameters.adaptation_time_constant_s))
    )
    if not runtime.plan.central_adaptation:
        adaptation = np.zeros_like(adaptation)
    firing_rate = np.maximum(unadapted - adaptation, 0.0)
    capacity_fraction = capacity / rested
    contraction_time = resting_time * (
        1.0
        + float(parameters.contraction_time_change_ratio)
        * (1.0 - capacity_fraction)
    )
    normalized_rate = contraction_time * firing_rate
    switch_force = 1.0 - np.exp(-2.0 * 0.4**3)
    normalized_force = np.where(
        normalized_rate <= 0.4,
        normalized_rate / 0.4 * switch_force,
        1.0 - np.exp(-2.0 * normalized_rate**3),
    )
    normalized_force = np.where(recruited, normalized_force, 0.0)
    unit_force = normalized_force * capacity
    was_recruited = duration > 0.0
    next_duration = np.where(was_recruited | recruited, duration + step_s, 0.0)
    if not runtime.plan.central_adaptation:
        next_duration = np.zeros_like(next_duration)
    next_capacity = capacity
    if runtime.plan.peripheral_fatigue:
        next_capacity = np.clip(
            capacity - force_loss_rate * normalized_force * step_s,
            0.0,
            rested,
        )
    return {
        "unadapted_firing_rate_hz": unadapted,
        "firing_rate_adaptation_hz": adaptation,
        "firing_rate_hz": firing_rate,
        "contraction_time_s": contraction_time,
        "normalized_firing_rate": normalized_rate,
        "normalized_force": normalized_force,
        "motor_unit_force": unit_force,
        "total_force": np.sum(unit_force),
        "recruitment_duration_s": next_duration,
        "current_twitch_force": next_capacity,
    }


def _reference_step_qualification(runtime):
    state = runtime.initialize()
    maximum_error = 0.0
    cases = []
    for excitation in (0.0, 1.0, 20.0, 40.0, 67.0):
        candidate = runtime.candidate(state, excitation, 0.1)
        reference = _scalar_reference(runtime, state, excitation, 0.1)
        comparisons = {
            "unadapted_firing_rate_hz": candidate.output.unadapted_firing_rate_hz,
            "firing_rate_adaptation_hz": candidate.output.firing_rate_adaptation_hz,
            "firing_rate_hz": candidate.output.firing_rate_hz,
            "contraction_time_s": candidate.output.contraction_time_s,
            "normalized_firing_rate": candidate.output.normalized_firing_rate,
            "normalized_force": candidate.output.normalized_force,
            "motor_unit_force": candidate.output.motor_unit_force,
            "total_force": candidate.output.total_force,
            "recruitment_duration_s": candidate.candidate_state.recruitment_duration_s,
            "current_twitch_force": candidate.candidate_state.current_twitch_force,
        }
        errors = {
            name: float(np.max(np.abs(np.asarray(value) - reference[name])))
            for name, value in comparisons.items()
        }
        case_error = max(errors.values())
        maximum_error = max(maximum_error, case_error)
        cases.append(
            {
                "excitation": excitation,
                "maximum_absolute_error": case_error,
                "successful": bool(candidate.evidence.successful),
            }
        )
    tolerance = 5.0e-12
    return {
        "passed": maximum_error <= tolerance and all(case["successful"] for case in cases),
        "maximum_absolute_error": maximum_error,
        "tolerance": tolerance,
        "cases": cases,
    }


def _target_endurance(runtime, target_fraction: float, maximum_time_s: float):
    step_s = 0.1
    step_count = round(maximum_time_s / step_s)
    excitation = (
        jnp.arange(1, 6701, dtype=runtime.parameters.rested_twitch_force.dtype)
        / 100.0
    )
    rested_maximum_force = runtime.rested_maximum_force()
    target_force = target_fraction * rested_maximum_force

    def protocol_step(carry, _):
        state, running = carry
        force_curve = jax.vmap(
            lambda drive: runtime.evaluate(state, drive).total_force
        )(excitation)
        meets = force_curve >= target_force
        can_meet = jnp.any(meets)
        selected_index = jnp.argmax(meets)
        selected_excitation = excitation[selected_index]
        candidate = runtime.candidate(state, selected_excitation, step_s)
        advance = running & can_meet & candidate.evidence.successful
        proposed = candidate.commit()
        next_state = type(state)(
            jnp.where(
                advance,
                proposed.recruitment_duration_s,
                state.recruitment_duration_s,
            ),
            jnp.where(
                advance,
                proposed.current_twitch_force,
                state.current_twitch_force,
            ),
        )
        return (next_state, advance), jnp.stack(
            (
                advance.astype(force_curve.dtype),
                selected_excitation,
                force_curve[selected_index] / rested_maximum_force,
                candidate.output.total_force_capacity_fraction,
            )
        )

    (_, _), history = jax.lax.scan(
        protocol_step,
        (runtime.initialize(), jnp.asarray(True)),
        xs=None,
        length=step_count,
    )
    history = np.asarray(history)
    failed = np.flatnonzero(history[:, 0] == 0.0)
    final_index = (
        len(history) - 1 if failed.size == 0 else max(int(failed[0]) - 1, 0)
    )
    endurance_s = (
        maximum_time_s if failed.size == 0 else float(failed[0]) * step_s
    )
    return {
        "target_fraction": target_fraction,
        "endurance_s": endurance_s,
        "completed_horizon": not bool(failed.size),
        "final_excitation": float(history[final_index, 1]),
        "final_force_fraction": float(history[final_index, 2]),
        "final_capacity_fraction": float(history[final_index, 3]),
    }


def _protocol_qualification(runtime):
    reported = {0.5: 95.5, 0.8: 14.8}
    protocols = []
    for target, horizon in ((0.2, 200.0), (0.5, 150.0), (0.8, 50.0)):
        result = _target_endurance(runtime, target, horizon)
        if target in reported:
            result["reported_endurance_s"] = reported[target]
            result["absolute_endurance_error_s"] = abs(
                result["endurance_s"] - reported[target]
            )
        protocols.append(result)
    compared = [case for case in protocols if "absolute_endurance_error_s" in case]
    tolerance_s = 0.100000001
    passed = all(case["absolute_endurance_error_s"] <= tolerance_s for case in compared)
    return {"passed": passed, "tolerance_s": tolerance_s, "cases": protocols}

def _constant_excitation_endpoint(runtime, step_s: float):
    step_count = round(20.0 / step_s)

    def protocol_step(state, _):
        candidate = runtime.candidate(state, 40.0, step_s)
        return candidate.commit(), candidate.evidence.successful

    final_state, successful = jax.lax.scan(
        protocol_step,
        runtime.initialize(),
        xs=None,
        length=step_count,
    )
    final_output = runtime.evaluate(final_state, 40.0)
    return {
        "step_s": step_s,
        "total_force": float(final_output.total_force),
        "capacity_fraction": float(final_output.total_force_capacity_fraction),
        "all_successful": bool(jnp.all(successful)),
    }


def _time_step_refinement(runtime):
    cases = [
        _constant_excitation_endpoint(runtime, step_s)
        for step_s in (0.1, 0.05, 0.025)
    ]
    ratios = {}
    for field in ("total_force", "capacity_fraction"):
        coarse_medium = abs(cases[0][field] - cases[1][field])
        medium_fine = abs(cases[1][field] - cases[2][field])
        ratios[field] = medium_fine / coarse_medium
    maximum_ratio = max(ratios.values())
    ratio_limit = 0.55
    return {
        "passed": (
            all(case["all_successful"] for case in cases)
            and maximum_ratio <= ratio_limit
        ),
        "cases": cases,
        "successive_error_ratios": ratios,
        "maximum_successive_error_ratio": maximum_ratio,
        "ratio_limit": ratio_limit,
    }


def qualify():
    runtime = PotvinFuglevand2017Plan().prepare()
    reference = _reference_step_qualification(runtime)
    protocols = _protocol_qualification(runtime)
    refinement = _time_step_refinement(runtime)
    maximum_force_error = abs(float(runtime.rested_maximum_force()) - 2215.9811474699964)
    maximum_force_tolerance = 5.0e-10
    return {
        "qualification": "potvin-fuglevand-2017-sustained-isometric",
        "source_doi": POTVIN_FUGLEVAND_2017_DOI,
        "reference_sha": POTVIN_FUGLEVAND_2017_REFERENCE_SHA,
        "model_id": runtime.plan.plan_id,
        "prepared_id": runtime.prepared_id,
        "time_origin_convention": (
            "the initial state is at t=0 and the first candidate advances [0, 0.1] s"
        ),
        "passed": (
            reference["passed"]
            and protocols["passed"]
            and refinement["passed"]
            and maximum_force_error <= maximum_force_tolerance
        ),
        "rested_maximum_force": float(runtime.rested_maximum_force()),
        "rested_maximum_force_error": maximum_force_error,
        "rested_maximum_force_tolerance": maximum_force_tolerance,
        "reference_steps": reference,
        "published_protocols": protocols,
        "time_step_refinement": refinement,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    payload = qualify()
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
