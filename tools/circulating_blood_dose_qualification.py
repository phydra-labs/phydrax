#!/usr/bin/env python3
# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Run synthetic research-only qualification of circulating-blood dose scoring."""

from __future__ import annotations

import json

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from phydrax.applications.radiation_biophysics import circulating_blood as cb
from phydrax.stochastic import PoissonClockRealization


def qualify() -> dict[str, object]:
    prepared = cb.prepare_circulating_blood_model(
        cb.CirculatingBloodModel(
            (
                cb.BloodCompartment("central", 1.0),
                cb.BloodCompartment("peripheral", 2.0),
            ),
            (
                cb.BloodFlow("central", "peripheral", 0.8),
                cb.BloodFlow("peripheral", "central", 0.8),
            ),
        )
    )
    quantity = cb.circulating_blood_dose_rate_quantity(
        "qualification_blood_dose_rate",
        cb.DOSE_TO_WATER_RATE_REFERENCE,
    )
    schedule = cb.PiecewiseConstantDoseRateSchedule(
        (
            cb.DoseRateInterval(
                0.0,
                1.5,
                quantity,
                np.asarray((0.2, 0.9)),
                np.asarray((0.01, 0.03)),
            ),
            cb.DoseRateInterval(
                2.0,
                4.0,
                quantity,
                np.asarray((0.5, 0.1)),
                np.asarray((0.02, 0.01)),
            ),
        )
    )
    deterministic = cb.integrate_circulating_blood_dose(
        prepared,
        schedule,
        prepared.point_distribution("central"),
        t0_s=0.0,
        t1_s=4.0,
    )
    realization = PoissonClockRealization(
        jr.key(16092026),
        prepared.process.num_channels,
        support=(0.0, 4.0),
        max_events_per_channel=64,
        sample_shape=(2048,),
        process_id=prepared.process.process_id,
        label="circulating-blood-dose-qualification",
    )
    stochastic = cb.simulate_circulating_blood_dose(
        prepared,
        schedule,
        realization,
        "central",
        t0_s=0.0,
        t1_s=4.0,
    )
    replay = cb.simulate_circulating_blood_dose(
        prepared,
        schedule,
        realization,
        "central",
        t0_s=0.0,
        t1_s=4.0,
    )
    occupation_error = float(
        jnp.max(jnp.abs(jnp.sum(stochastic.occupation_seconds, axis=-1) - 4.0))
    )
    replay_exact = bool(
        jnp.array_equal(stochastic.solution.events.valid, replay.solution.events.valid)
        & jnp.array_equal(
            stochastic.solution.events.times,
            replay.solution.events.times,
            equal_nan=True,
        )
        & jnp.array_equal(stochastic.total_dose_gy, replay.total_dose_gy)
    )
    stochastic_mean = float(jnp.mean(stochastic.total_dose_gy))
    deterministic_mean = float(deterministic.total_dose_gy)
    mean_error = abs(stochastic_mean - deterministic_mean)
    if (
        not prepared.capacity.successful
        or not bool(jnp.all(stochastic.successful))
        or not replay_exact
        or occupation_error > 2.0e-6
        or mean_error > 0.08
        or deterministic.standard_uncertainty_gy is None
        or stochastic.standard_uncertainty_gy is None
    ):
        raise RuntimeError("Circulating-blood dose qualification failed.")
    return {
        "model_id": prepared.model.model_id,
        "schedule_id": schedule.schedule_id,
        "deterministic_total_dose_gy": deterministic_mean,
        "stochastic_mean_total_dose_gy": stochastic_mean,
        "mean_absolute_error_gy": mean_error,
        "maximum_occupation_conservation_error_s": occupation_error,
        "path_count": realization.num_paths,
        "maximum_events_used": int(jnp.max(stochastic.capacity.events_used)),
        "same_realization_replay_exact": replay_exact,
        "successful": True,
        "scope": (
            "Synthetic CTMC occupation and reward mechanics only; research use, with "
            "no patient, treatment, or clinical dose claim."
        ),
    }


def main() -> None:
    print(json.dumps(qualify(), indent=2))


if __name__ == "__main__":
    main()
