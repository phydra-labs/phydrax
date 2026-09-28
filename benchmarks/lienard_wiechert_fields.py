#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phase-separated near-zone Liénard–Wiechert field benchmark.

A bunch of relativistic particles on circular orbits with distinct phases is
observed at ``--observers`` events on a sphere around the orbit. Each
interpolation is lowered, compiled, and executed separately; the observer
count is the controlling capacity of the scaling campaign.
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
    LienardWiechertFieldPlan,
    LienardWiechertResources,
)


_INTERPOLATIONS = ("hermite-cubic", "hermite-quintic")


def _trajectory(
    gamma: float, particles: int, samples: int, radius: float, speed_of_light: float
) -> ChargedTrajectory:
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    omega = beta * speed_of_light / radius
    times = np.linspace(-8.0 * np.pi / omega, 2.0 * np.pi / omega, samples)
    phase = omega * times[:, None] + np.linspace(0.0, 2.0 * np.pi, particles)[None, :]
    zeros = np.zeros_like(phase)
    cosine, sine = np.cos(phase), np.sin(phase)
    speed = gamma * beta * speed_of_light
    return ChargedTrajectory(
        times,
        radius * np.stack((cosine, sine, zeros), axis=-1),
        speed * np.stack((-sine, cosine, zeros), axis=-1),
        np.full(particles, 1.602176634e-19),
        np.ones(particles),
        np.ones((samples, particles), dtype=bool),
        (np.arange(particles, dtype=np.uint32), np.zeros(particles, dtype=np.uint32)),
        proper_accelerations=-speed * omega * np.stack((cosine, sine, zeros), axis=-1),
    )


def _observer_events(count: int, radius: float) -> np.ndarray:
    index = np.arange(count) + 0.5
    polar = np.arccos(1.0 - 2.0 * index / count)
    azimuth = np.pi * (1.0 + np.sqrt(5.0)) * index
    points = radius * np.stack(
        (
            np.sin(polar) * np.cos(azimuth),
            np.sin(polar) * np.sin(azimuth),
            np.cos(polar),
        ),
        axis=-1,
    )
    return np.concatenate((np.zeros((count, 1)), points), axis=1)


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
    interpolation: str,
    observer_count: int,
    arguments: argparse.Namespace,
    scale: ElectromagneticScaleContract,
) -> dict[str, object]:
    radius = 0.1
    plan = LienardWiechertFieldPlan(
        scale,
        history="refuse",
        exclusion_radius=1.0e-6,
        interpolation=interpolation,  # ty: ignore[invalid-argument-type]
        resources=LienardWiechertResources(
            observer_chunk=arguments.observer_chunk,
            particle_chunk=arguments.particle_chunk,
        ),
    )
    prepared = plan.prepare()
    trajectory = _trajectory(
        arguments.gamma,
        arguments.particles,
        arguments.samples,
        radius,
        float(scale.speed_of_light),
    )
    events = _observer_events(observer_count, 10.0 * radius)
    dynamic, static = eqx.partition((prepared, trajectory, events), eqx.is_array)

    def evaluate(leaves: Any) -> Any:
        prepared_, trajectory_, events_ = eqx.combine(leaves, static)
        result = prepared_.evaluate(trajectory_, events_)
        return result.electric_field, result.evidence.status

    function = jax.jit(evaluate)
    compiled, compilation = measure_lower_and_compile(
        lambda: function.lower(dynamic), lambda lowered: lowered.compile()
    )
    (field, status), execution = measure_repeated(
        lambda: compiled(dynamic), warmup=arguments.warmup, repeats=arguments.repeats
    )
    estimate = prepared.resource_estimate(
        observer_count, arguments.particles, arguments.samples
    )
    status_ = np.asarray(status)
    return {
        "interpolation": interpolation,
        "observers": observer_count,
        "identity": plan.plan_id,
        "compilation": {
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
        },
        "compiler": _compiler_record(compiled),
        "execution": execution.to_seconds_dict(),
        "memory": {
            "estimated_working_bytes": estimate.working_bytes,
            "estimated_output_bytes": estimate.output_bytes,
            "input_logical_bytes": logical_array_bytes(dynamic),
            "output_logical_bytes": logical_array_bytes(field),
        },
        "physics": {
            "finite": bool(np.all(np.isfinite(np.asarray(field)))),
            "status_union": int(np.bitwise_or.reduce(status_)),
            "peak_field_V_per_m": float(np.max(np.abs(np.asarray(field)))),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=5.0)
    parser.add_argument("--particles", type=int, default=16)
    parser.add_argument("--samples", type=int, default=2049)
    parser.add_argument("--observers", type=str, default="256,1024")
    parser.add_argument("--observer-chunk", type=int, default=128)
    parser.add_argument("--particle-chunk", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    observer_counts = tuple(int(value) for value in arguments.observers.split(","))
    if (
        arguments.gamma < 1.0
        or min(arguments.particles, arguments.observer_chunk, arguments.particle_chunk)
        < 1
        or arguments.samples < 2
        or min(observer_counts) < 1
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
            "observer_chunk": arguments.observer_chunk,
            "particle_chunk": arguments.particle_chunk,
            "warmup": arguments.warmup,
            "repeats": arguments.repeats,
        },
        "cases": [
            _case(interpolation, count, arguments, scale)
            for interpolation in _INTERPOLATIONS
            for count in observer_counts
        ],
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
