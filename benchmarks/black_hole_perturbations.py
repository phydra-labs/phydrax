#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Benchmark bounded Schwarzschild radial perturbation matching and its JVP."""

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

from phydrax.applications import compact_objects


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


def _setup(node_count: int):
    mode = compact_objects.SeparatedMode(
        0,
        0,
        0,
        sector="scalar",
        family="qnm",
        background_id="benchmark:schwarzschild",
    )
    plan = compact_objects.SchwarzschildRadialPlan(
        mode,
        1.0,
        node_count=node_count,
        outer_radius=40.0,
        asymptotic_tolerance=0.2,
        maximum_dimension=257,
    )
    frequency = jnp.asarray(0.4 - 0.05j)
    frequency_direction = jnp.asarray(1.0 + 0.0j, dtype=frequency.dtype)
    return mode, plan, frequency, frequency_direction


def run(node_count: int, warmup: int, repeats: int) -> dict[str, object]:
    environment = capture_environment().to_dict()
    setup, setup_seconds = measure_synchronized(lambda: _setup(node_count))
    mode, plan, frequency, frequency_direction = setup

    def radial_kernel(value):
        result = compact_objects.evaluate_schwarzschild_radial(plan, value)
        return (
            result.solution,
            result.logarithmic_derivative,
            result.residual,
            result.residual_evidence.differential_residual,
            result.residual_evidence.residual_norm,
            result.residual_evidence.relative_residual,
            result.asymptotic.horizon_defect,
            result.asymptotic.infinity_defect,
            result.finite,
            result.converged,
            result.physically_valid,
            result.qualified,
            result.derivative_valid,
            result.status,
        )

    def derivative_kernel(value, direction):
        residual, tangent = jax.jvp(
            lambda candidate: compact_objects.schwarzschild_outgoing_residual(
                plan, candidate
            ),
            (value,),
            (direction,),
        )
        return residual, tangent

    primal, primal_performance = _measure(
        radial_kernel, (frequency,), warmup, repeats
    )
    derivative, derivative_performance = _measure(
        derivative_kernel,
        (frequency, frequency_direction),
        warmup,
        repeats,
    )
    mean_seconds = primal_performance["execution"]["mean_seconds"]
    derivative_finite = bool(
        jnp.isfinite(jnp.real(derivative[1]))
        & jnp.isfinite(jnp.imag(derivative[1]))
    )
    successful = bool(primal[8] & primal[10] & primal[12]) and derivative_finite
    return {
        "identities": {
            "benchmark": "black-hole-perturbations",
            "kernel": "schwarzschild-two-sided-radial-matching",
            "mode": mode.mode_id,
            "convention": mode.convention_id,
            "plan": plan.plan_id,
            "background": mode.background_id,
        },
        "configuration": {
            "node_capacity": node_count,
            "mass": float(plan.mass),
            "frequency": {
                "real": float(jnp.real(frequency)),
                "imaginary": float(jnp.imag(frequency)),
            },
            "sector": mode.sector,
            "family": mode.family,
            "outer_radius": float(jnp.max(plan.radial_nodes)),
            "profile_claim": "off-root radial response; not a solved QNM root",
            "warmup": warmup,
            "repeats": repeats,
        },
        "environment": environment,
        "physics": {
            "successful": successful,
            "finite": bool(primal[8]),
            "matching_converged": bool(primal[9]),
            "physically_valid": bool(primal[10]),
            "qualified_qnm_root": bool(primal[11]),
            "derivative_valid": bool(primal[12]),
            "status": int(primal[13]),
            "matching_residual": {
                "real": float(jnp.real(primal[2])),
                "imaginary": float(jnp.imag(primal[2])),
                "magnitude": float(jnp.abs(primal[2])),
            },
            "spectral_residual_norm": float(primal[4]),
            "spectral_relative_residual": float(primal[5]),
            "horizon_asymptotic_defect": float(primal[6]),
            "infinity_asymptotic_defect": float(primal[7]),
            "frequency_directional_residual_derivative": {
                "real": float(jnp.real(derivative[1])),
                "imaginary": float(jnp.imag(derivative[1])),
                "finite": derivative_finite,
            },
        },
        "performance": {
            "setup_seconds": setup_seconds,
            "primal": primal_performance,
            "directional_derivative": derivative_performance,
            "logical_bytes": {
                "plan": logical_array_bytes(plan),
                "input_frequency": logical_array_bytes(frequency),
                "frequency_direction": logical_array_bytes(frequency_direction),
                "primal_output": logical_array_bytes(primal),
                "derivative_output": logical_array_bytes(derivative),
            },
            "radial_nodes_per_second": None
            if mean_seconds in (None, 0.0)
            else node_count / mean_seconds,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=65)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if not 9 <= arguments.nodes <= 257:
        raise ValueError("nodes must be between 9 and 257.")
    if not 0 <= arguments.warmup <= 100:
        raise ValueError("warmup must be between 0 and 100.")
    if not 1 <= arguments.repeats <= 1_000:
        raise ValueError("repeats must be between 1 and 1,000.")
    payload = run(arguments.nodes, arguments.warmup, arguments.repeats)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    raise SystemExit(0 if payload["physics"]["successful"] else 1)


if __name__ == "__main__":
    main()
