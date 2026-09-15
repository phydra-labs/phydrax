#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Benchmark SRHD primitive recovery, fluxes, causal bounds, and derivatives."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
from _runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)

import phydrax as phx
from phydrax.units import KILOGRAM


def _compiler_record(compiled) -> dict[str, object]:
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-executable",
        unavailable_reason="The selected JAX backend did not report compiler analysis.",
    )
    record = asdict(evidence)
    record["estimated_device_memory_bytes"] = evidence.estimated_device_memory_bytes
    return record


def _measure(function, arguments, warmup, repeats):
    compiled, compilation = measure_lower_and_compile(
        lambda: jax.jit(function).lower(*arguments),
        lambda lowered: lowered.compile(),
    )
    result, execution = measure_repeated(
        lambda: compiled(*arguments), warmup=warmup, repeats=repeats
    )
    return result, {
        "compilation": asdict(compilation),
        "execution": execution.to_seconds_dict(),
        "compiler": _compiler_record(compiled),
    }


def _setup(cell_count: int):
    scale = phx.RelativityScaleContract.geometric(KILOGRAM)
    eos = phx.equations.GammaLawEOS(scale, 5.0 / 3.0)
    system = phx.equations.SRHDSystem(eos, 3)
    phase = 2.0 * jnp.pi * (jnp.arange(cell_count) + 0.5) / cell_count
    primitive = jnp.stack(
        (
            1.0 + 0.2 * jnp.sin(phase),
            0.3 + 0.05 * jnp.cos(phase),
            0.20 * jnp.sin(phase),
            0.10 * jnp.cos(phase),
            0.05 * jnp.sin(2.0 * phase),
        ),
        axis=-1,
    )
    direction = jnp.stack(
        (
            0.01 * jnp.cos(phase),
            0.01 * jnp.sin(phase),
            0.005 * jnp.cos(phase),
            -0.005 * jnp.sin(phase),
            0.0025 * jnp.cos(2.0 * phase),
        ),
        axis=-1,
    )
    return scale, eos, system, primitive, direction


def run(cell_count: int, warmup: int, repeats: int) -> dict[str, object]:
    environment = capture_environment().to_dict()
    setup, setup_seconds = measure_synchronized(lambda: _setup(cell_count))
    scale, eos, system, primitive, direction = setup

    def matter_kernel(values):
        evaluation = system.primitive_evaluation(values)
        conserved = system.primitive_to_conserved(values)
        recovered = system.conserved_to_primitive(conserved)
        flux = system.physical_flux(conserved, 0)
        lower, upper = system.signal_bounds(conserved, conserved, 0)
        return (
            conserved,
            recovered,
            flux,
            lower,
            upper,
            system.admissible(conserved),
            evaluation.finite,
            evaluation.physically_valid,
            evaluation.lorentz_factor,
            evaluation.sound_speed_squared,
            evaluation.qualified,
            evaluation.derivative_valid,
        )

    def derivative_kernel(values, tangent):
        def objective(candidate):
            conserved = system.primitive_to_conserved(candidate)
            flux = system.physical_flux(conserved, 0)
            return jnp.mean(conserved[..., -1] + 0.25 * flux[..., -1])

        return jax.jvp(objective, (values,), (tangent,))

    primal, primal_performance = _measure(
        matter_kernel, (primitive,), warmup, repeats
    )
    derivative, derivative_performance = _measure(
        derivative_kernel, (primitive, direction), warmup, repeats
    )
    scale_value = jnp.maximum(jnp.abs(primitive), jnp.asarray(1.0))
    round_trip_error = jnp.max(jnp.abs(primal[1] - primitive) / scale_value)
    tolerance = 1_000.0 * jnp.finfo(primitive.dtype).eps
    flux_finite = bool(jnp.all(jnp.isfinite(primal[2])))
    derivative_finite = bool(
        jnp.isfinite(derivative[0]) & jnp.isfinite(derivative[1])
    )
    successful = bool(
        jnp.all(primal[5])
        & jnp.all(primal[6])
        & jnp.all(primal[7])
        & jnp.all(primal[10])
        & jnp.all(primal[11])
        & flux_finite
        & jnp.all(primal[3] >= -1.0)
        & jnp.all(primal[4] <= 1.0)
        & jnp.all(primal[3] <= primal[4])
        & (round_trip_error <= tolerance)
    ) and derivative_finite
    mean_seconds = primal_performance["execution"]["mean_seconds"]
    return {
        "identities": {
            "benchmark": "relativistic-matter",
            "kernel": "srhd-primitive-recovery-flux-and-characteristics",
            "system": system.system_id,
            "eos": eos.eos_id,
            "scale": scale.scale_id,
            "layout": system.layout.layout_id,
        },
        "configuration": {
            "cell_capacity": cell_count,
            "spatial_dimension": system.dimension,
            "adiabatic_index": 5.0 / 3.0,
            "density_floor": system.density_floor,
            "pressure_floor": system.pressure_floor,
            "warmup": warmup,
            "repeats": repeats,
        },
        "environment": environment,
        "physics": {
            "successful": successful,
            "all_admissible": bool(jnp.all(primal[5])),
            "all_finite": bool(jnp.all(primal[6])),
            "all_physically_valid": bool(jnp.all(primal[7])),
            "all_qualified": bool(jnp.all(primal[10])),
            "all_derivatives_valid": bool(jnp.all(primal[11])),
            "all_flux_components_finite": flux_finite,
            "maximum_primitive_round_trip_relative_error": float(
                round_trip_error
            ),
            "round_trip_tolerance": float(tolerance),
            "signal_bounds_causal": bool(
                jnp.all(primal[3] >= -1.0)
                & jnp.all(primal[4] <= 1.0)
                & jnp.all(primal[3] <= primal[4])
            ),
            "signal_speed_range": [
                float(jnp.min(primal[3])),
                float(jnp.max(primal[4])),
            ],
            "maximum_lorentz_factor": float(jnp.max(primal[8])),
            "sound_speed_squared_range": [
                float(jnp.min(primal[9])),
                float(jnp.max(primal[9])),
            ],
            "directional_objective": float(derivative[0]),
            "directional_derivative": float(derivative[1]),
            "directional_derivative_finite": derivative_finite,
        },
        "performance": {
            "setup_seconds": setup_seconds,
            "primal": primal_performance,
            "directional_derivative": derivative_performance,
            "logical_bytes": {
                "system": logical_array_bytes(system),
                "primitive_input": logical_array_bytes(primitive),
                "direction_input": logical_array_bytes(direction),
                "primal_output": logical_array_bytes(primal),
                "derivative_output": logical_array_bytes(derivative),
            },
            "cell_states_per_second": None
            if mean_seconds in (None, 0.0)
            else cell_count / mean_seconds,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=65_536)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if not 1 <= arguments.cells <= 10_000_000:
        raise ValueError("cells must be between 1 and 10,000,000.")
    if not 0 <= arguments.warmup <= 100:
        raise ValueError("warmup must be between 0 and 100.")
    if not 1 <= arguments.repeats <= 1_000:
        raise ValueError("repeats must be between 1 and 1,000.")
    payload = run(arguments.cells, arguments.warmup, arguments.repeats)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    raise SystemExit(0 if payload["physics"]["successful"] else 1)


if __name__ == "__main__":
    main()
