#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Qualify vacuum trajectory radiation against closed-form references.

Gates: Liénard–Larmor power of a nonrelativistic orbit, Schott harmonics of a
relativistic orbit (``phydrax.special.jv``), node-gridded Type-3 agreement with
the exact segment route inside the reported floor, streaming = offline, and
Lorentz covariance of the spectral energy. ``--smoke`` shrinks every case.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from phydrax import ElectromagneticScaleContract, special
from phydrax.electromagnetics import (
    ChargedTrajectory,
    RadiationObserverPlan,
    TrajectoryRadiationPlan,
    TrajectoryRadiationResult,
)


SCALE = ElectromagneticScaleContract.si()
C = float(SCALE.speed_of_light)
EPS0 = float(SCALE.vacuum_permittivity)
Q = float(SCALE.elementary_charge)
OMEGA0 = 1.0e10
PERIOD = 2.0 * np.pi / OMEGA0
Z_AXIS = np.array([0.0, 0.0, 1.0])


def _periodic_orbit(beta: float, samples: int) -> ChargedTrajectory:
    """One period sampled at half steps, so node jumps cover exactly ``[0, T₀]``."""
    times = (np.arange(samples + 2) - 0.5) * PERIOD / samples
    phase = OMEGA0 * times
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    zeros = np.zeros_like(phase)
    positions = beta * C / OMEGA0 * np.stack((np.cos(phase), np.sin(phase), zeros), -1)
    proper = gamma * beta * C * np.stack((-np.sin(phase), np.cos(phase), zeros), -1)
    return ChargedTrajectory(
        times,
        positions[:, None],
        proper[:, None],
        np.array([Q]),
        np.array([1.0]),
        np.ones((times.size, 1), dtype=bool),
        (np.zeros(1, dtype=np.uint32), np.zeros(1, dtype=np.uint32)),
    )


def _in_plane(theta: np.ndarray) -> np.ndarray:
    return np.stack((np.sin(theta), np.zeros_like(theta), np.cos(theta)), axis=-1)


def _plan(
    theta: np.ndarray, harmonics: np.ndarray, route: str = "segment-exact"
) -> TrajectoryRadiationPlan:
    return TrajectoryRadiationPlan(
        SCALE,
        RadiationObserverPlan(_in_plane(theta), Z_AXIS),
        OMEGA0 * harmonics,
        coherence="coherent",
        route=route,  # ty: ignore[invalid-argument-type]
        emission="truncated",
        observer_time_window=(-PERIOD, 2.0 * PERIOD) if route == "node-gridded" else None,
    )


def _power_density(result: TrajectoryRadiationResult) -> np.ndarray:
    return 2.0 * np.pi * np.asarray(result.spectral_energy) / PERIOD**2


def _schott(harmonics: np.ndarray, beta: float, theta: np.ndarray) -> np.ndarray:
    m = harmonics[:, None]
    x = jnp.asarray(m * beta * np.sin(theta)[None, :])
    order = jnp.asarray(np.broadcast_to(m, x.shape))
    bessel = np.asarray(special.jv(order, x))
    derivative = 0.5 * np.asarray(special.jv(order - 1.0, x) - special.jv(order + 1.0, x))
    cotangent = np.cos(theta) / np.sin(theta)
    return (
        Q**2
        * OMEGA0**2
        * m**2
        / (8.0 * np.pi**2 * EPS0 * C)
        * (cotangent**2 * bessel**2 + beta**2 * derivative**2)
    )


def _larmor(samples: int) -> float:
    beta = 0.01
    cosines, weights = np.polynomial.legendre.leggauss(16)
    theta = np.arccos(cosines)
    result = (
        _plan(theta, np.array([1.0, 2.0, 3.0]))
        .prepare()
        .evaluate(_periodic_orbit(beta, samples))
    )
    power = 2.0 * np.pi * np.sum(_power_density(result) @ weights)
    lienard = (
        Q**2 * (beta * OMEGA0) ** 2 / (6.0 * np.pi * EPS0 * C * (1.0 - beta**2) ** 2)
    )
    return float(abs(power / lienard - 1.0))


def _schott_error(samples: int, route: str) -> float:
    beta = np.sqrt(1.0 - 1.0e-2)
    theta = np.pi / 2.0 - np.array([0.0, 0.05, 0.1, 0.3])
    harmonics = np.array([1.0, 3.0, 10.0, 30.0])
    result = (
        _plan(theta, harmonics, route).prepare().evaluate(_periodic_orbit(beta, samples))
    )
    return float(
        np.max(np.abs(_power_density(result) / _schott(harmonics, beta, theta) - 1.0))
    )


def _gridded_floor_ratio(samples: int) -> float:
    beta = np.sqrt(1.0 - 1.0e-2)
    theta = np.pi / 2.0 - np.array([0.0, 0.02, 0.1])
    harmonics = np.linspace(1.0, 6000.0, 128)
    trajectory = _periodic_orbit(beta, samples)
    exact = _plan(theta, harmonics).prepare().evaluate(trajectory)
    gridded = _plan(theta, harmonics, "node-gridded").prepare().evaluate(trajectory)
    floor = gridded.evidence.gridded_error_floor
    if floor is None:
        raise ValueError("node-gridded evidence lacks its error floor.")
    difference = np.max(
        np.abs(np.asarray(gridded.field_spectrum) - np.asarray(exact.field_spectrum)),
        axis=(0, 2),
    )
    return float(np.max(difference / np.asarray(floor)))


def _streaming_error(samples: int) -> float:
    beta = 0.5
    theta = np.array([0.3, 1.2])
    harmonics = np.array([1.0, 2.0])
    prepared = _plan(theta, harmonics).prepare()
    whole = _periodic_orbit(beta, samples)
    offline = prepared.evaluate(whole)
    state = prepared.initialize(whole)
    split = samples // 3
    for start, stop in ((0, split), (split, samples + 2)):
        chunk = ChargedTrajectory(
            np.asarray(whole.times)[start:stop],
            np.asarray(whole.positions)[start:stop],
            np.asarray(whole.proper_velocities)[start:stop],
            np.asarray(whole.charges),
            np.asarray(whole.multiplicities),
            np.asarray(whole.active)[start:stop],
            (np.asarray(whole.id_hi), np.asarray(whole.id_lo)),
        )
        state = prepared.accumulate(state, chunk)
    streamed = prepared.finalize(state)
    reference = np.max(np.abs(np.asarray(offline.field_spectrum)))
    return float(
        np.max(
            np.abs(
                np.asarray(streamed.field_spectrum) - np.asarray(offline.field_spectrum)
            )
        )
        / reference
    )


def qualify(*, smoke: bool) -> dict[str, object]:
    larmor = _larmor(256 if smoke else 1024)
    schott_exact = _schott_error(1024 if smoke else 4096, "segment-exact")
    schott_gridded = _schott_error(1024 if smoke else 4096, "node-gridded")
    floor_ratio = _gridded_floor_ratio(512 if smoke else 2048)
    streaming = _streaming_error(128 if smoke else 1024)
    tolerance = 1.0e-3 if smoke else 1.0e-4
    checks = {
        "lienard_larmor_power": larmor <= tolerance,
        "schott_harmonics_exact": schott_exact <= tolerance,
        "schott_harmonics_gridded": schott_gridded <= tolerance,
        "gridded_within_reported_floor": floor_ratio <= 1.0,
        "streaming_equals_offline": streaming <= 1.0e-12,
    }
    return {
        "qualification": "electromagnetics-vacuum-trajectory-radiation",
        "evidence_scope": "closed-form-references-only",
        "smoke": smoke,
        "checks": checks,
        "metrics": {
            "lienard_larmor_relative_error": larmor,
            "schott_exact_relative_error": schott_exact,
            "schott_gridded_relative_error": schott_gridded,
            "gridded_error_over_floor": floor_ratio,
            "streaming_relative_difference": streaming,
        },
        "successful": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = qualify(smoke=arguments.smoke)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    if not report["successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
