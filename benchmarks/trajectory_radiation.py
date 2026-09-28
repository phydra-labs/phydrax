#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phase-separated far-field trajectory radiation benchmark.

A bunch of relativistic particles on circular orbits with distinct phases
radiates into ``--directions`` observers at ``--frequencies`` angular
frequencies. Each route is lowered, compiled, and executed separately; the
frequency count is the controlling capacity of the scaling campaign.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import numpy as np
from _runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)

from phydrax import ElectromagneticScaleContract
from phydrax.electromagnetics import (
    ChargedTrajectory,
    RadiationObserverPlan,
    TrajectoryRadiationPlan,
    TrajectoryRadiationResources,
)


_ROUTES = ("segment-exact", "node-gridded")


def _trajectory(
    gamma: float, particles: int, samples: int, omega: float, speed_of_light: float
) -> ChargedTrajectory:
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    radius = beta * speed_of_light / omega
    times = np.linspace(0.0, 2.0 * np.pi / omega, samples)
    phase = omega * times[:, None] + np.linspace(0.0, 2.0 * np.pi, particles)[None, :]
    zeros = np.zeros_like(phase)
    positions = radius * np.stack((np.cos(phase), np.sin(phase), zeros), axis=-1)
    proper = (
        gamma
        * beta
        * speed_of_light
        * np.stack((-np.sin(phase), np.cos(phase), zeros), axis=-1)
    )
    return ChargedTrajectory(
        times,
        positions,
        proper,
        np.full(particles, 1.602176634e-19),
        np.ones(particles),
        np.ones((samples, particles), dtype=bool),
        (np.arange(particles, dtype=np.uint32), np.zeros(particles, dtype=np.uint32)),
    )


def _compiler_record(compiled: Any) -> dict[str, object]:
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-executable",
    )
    return {
        "flops": evidence.flops,
        "bytes_accessed": evidence.bytes_accessed,
        "argument_bytes": evidence.argument_bytes,
        "output_bytes": evidence.output_bytes,
        "temporary_bytes": evidence.temporary_bytes,
        "generated_code_bytes": evidence.generated_code_bytes,
    }


def _case(
    route: str,
    frequency_count: int,
    arguments: argparse.Namespace,
    scale: ElectromagneticScaleContract,
) -> dict[str, object]:
    omega = 1.0e10
    period = 2.0 * np.pi / omega
    speed_of_light = float(scale.speed_of_light)
    elevation = np.linspace(0.0, 0.3, arguments.directions)
    directions = np.stack(
        (np.cos(elevation), np.zeros_like(elevation), np.sin(elevation)), axis=-1
    )
    frequencies = omega * np.linspace(1.0, 3.0 * arguments.gamma**3, frequency_count)
    plan = TrajectoryRadiationPlan(
        scale,
        RadiationObserverPlan(directions, np.array([0.0, 0.0, 1.0])),
        frequencies,
        coherence="coherent",
        route=route,  # ty: ignore[invalid-argument-type]
        emission="truncated",
        observer_time_window=(-period, 2.0 * period) if route == "node-gridded" else None,
        resources=TrajectoryRadiationResources(particle_chunk=arguments.particle_chunk),
    )
    prepared = plan.prepare()
    trajectory = _trajectory(
        arguments.gamma, arguments.particles, arguments.samples, omega, speed_of_light
    )
    dynamic, static = eqx.partition((prepared, trajectory), eqx.is_array)

    def evaluate(leaves: Any) -> Any:
        prepared_, trajectory_ = eqx.combine(leaves, static)
        result = prepared_.evaluate(trajectory_)
        return result.spectral_energy, result.evidence.status

    function = jax.jit(evaluate)
    compiled, compilation = measure_lower_and_compile(
        lambda: function.lower(dynamic), lambda lowered: lowered.compile()
    )
    (energy, status), execution = measure_repeated(
        lambda: compiled(dynamic), warmup=arguments.warmup, repeats=arguments.repeats
    )
    estimate = prepared.resource_estimate(arguments.particles, arguments.samples)
    return {
        "route": route,
        "frequencies": frequency_count,
        "identity": plan.plan_id,
        "compilation": {
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
        },
        "compiler": _compiler_record(compiled),
        "execution": execution.to_seconds_dict(),
        "memory": {
            "estimated_working_bytes": estimate.working_bytes,
            "estimated_state_bytes": estimate.state_bytes,
            "input_logical_bytes": logical_array_bytes(dynamic),
            "output_logical_bytes": logical_array_bytes(energy),
        },
        "physics": {
            "finite": bool(np.all(np.isfinite(np.asarray(energy)))),
            "status": int(status),
            "peak_spectral_energy_J_s_per_sr": float(np.max(np.asarray(energy))),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=5.0)
    parser.add_argument("--particles", type=int, default=16)
    parser.add_argument("--samples", type=int, default=2049)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--frequencies", type=str, default="64,256")
    parser.add_argument("--particle-chunk", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    frequency_counts = tuple(int(value) for value in arguments.frequencies.split(","))
    if (
        arguments.gamma < 1.0
        or min(arguments.particles, arguments.directions, arguments.particle_chunk) < 1
        or arguments.samples < 2
        or min(frequency_counts) < 1
        or arguments.warmup < 0
        or arguments.repeats < 1
    ):
        raise ValueError(
            "gamma >= 1, positive counts, samples >= 2, nonnegative warmup and "
            "positive repeats are required"
        )
    scale = ElectromagneticScaleContract.si()
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "gamma": arguments.gamma,
            "particles": arguments.particles,
            "samples": arguments.samples,
            "directions": arguments.directions,
            "particle_chunk": arguments.particle_chunk,
            "warmup": arguments.warmup,
            "repeats": arguments.repeats,
        },
        "cases": [
            _case(route, count, arguments, scale)
            for route in _ROUTES
            for count in frequency_counts
        ],
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
