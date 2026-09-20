#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
)
from tools.phase_field_production_qualification import (
    _allen_cahn,
    _cahn_hilliard,
    _model,
    _rectangle_mesh,
)


def _compiler_report(executable) -> dict[str, object]:
    cost = executable.compiled.cost_analysis()
    memory = executable.compiled.memory_analysis()
    unavailable = (
        "Compiler cost and memory analysis were unavailable on this backend."
        if not cost and memory is None
        else None
    )
    evidence = compiler_evidence(
        cost,
        memory,
        source="jax-lowered-compiled-executable",
        unavailable_reason=unavailable,
    )
    return {
        **asdict(evidence),
        "estimated_device_memory_bytes": evidence.estimated_device_memory_bytes,
    }


def _initial_values(method, field_index: int, /):
    coordinates = method.discretization.dof_maps[field_index].evaluate_coordinates(
        method.discretization.mesh,
        method.discretization.default_runtime.coordinates,
    )
    return (
        0.2
        * jnp.sin(2.0 * jnp.pi * coordinates[:, 0])
        * jnp.cos(2.0 * jnp.pi * coordinates[:, 1])
    )


def _benchmark_method(
    equation: str,
    resolution: int,
    /,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    mesh = _rectangle_mesh(resolution, resolution)
    model = _model(0.05)
    if equation == "allen-cahn":
        method, preparation_seconds = measure_host(lambda: _allen_cahn(mesh, model))
        values = _initial_values(method, method.field_index)
        state = method.initialize(values)
        step_size = jnp.asarray(0.001, dtype=jnp.float64)
        primary = lambda value: value.phase
    elif equation == "cahn-hilliard":
        termination = phx.nonlinear.NonlinearTermination(
            absolute_residual=1.0e-10,
            relative_residual=1.0e-10,
            maximum_steps=200,
        )
        method, preparation_seconds = measure_host(
            lambda: _cahn_hilliard(mesh, model, termination=termination)
        )
        values = _initial_values(method, method.concentration_index)
        state = method.initialize(values)
        step_size = jnp.asarray(0.0005, dtype=jnp.float64)
        primary = lambda value: value.concentration
    else:
        raise ValueError("Unknown phase-field benchmark equation.")

    arguments = (
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float64),
        state,
        step_size,
        None,
    )
    kernel: Any = eqx.filter_jit(method.step)
    executable, compilation = measure_lower_and_compile(
        lambda: kernel.lower(*arguments),
        lambda lowered: lowered.compile(),
    )
    result, durations = measure_repeated(
        lambda: executable(*arguments),
        warmup=warmup,
        repeats=repeats,
    )
    candidate = primary(result.candidate_state)
    finite = bool(jnp.all(jnp.isfinite(candidate)))
    successful = bool(result.successful) and finite
    return {
        "equation": equation,
        "resolution": resolution,
        "cells": 2 * resolution * resolution,
        "degrees_of_freedom": candidate.size,
        "method_id": method.method_id,
        "preparation_seconds": preparation_seconds,
        "compilation": {
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
        },
        "steady_step": durations.to_milliseconds_dict(),
        "compiler": _compiler_report(executable),
        "logical_argument_bytes": logical_array_bytes(arguments),
        "logical_result_bytes": logical_array_bytes(result),
        "successful": successful,
        "residual": float(np.asarray(result.residual)),
        "nonlinear_iterations": int(np.asarray(result.iterations)),
        "work": int(np.asarray(result.work)),
        "energy_before": float(np.asarray(state.energy)),
        "energy_after": float(np.asarray(result.candidate_state.energy)),
        "mass_before": float(np.asarray(state.mass)),
        "mass_after": float(np.asarray(result.candidate_state.mass)),
        "cells_across_transition": float(
            np.asarray(method.resolution.cells_across_transition)
        ),
    }


def benchmark(*, quick: bool, repeats: int) -> dict[str, object]:
    if not bool(jax.config.read("jax_enable_x64")):
        raise ValueError("Phase-field benchmarks require JAX float64 support.")
    resolutions = (8,) if quick else (8, 16, 24)
    warmup = 0 if quick else 1
    cases = [
        _benchmark_method(
            equation,
            resolution,
            warmup=warmup,
            repeats=repeats,
        )
        for resolution in resolutions
        for equation in ("allen-cahn", "cahn-hilliard")
    ]
    passed = all(case["successful"] for case in cases)
    return {
        "status": "pass" if passed else "fail",
        "environment": capture_environment().to_dict(),
        "settings": {
            "quick": quick,
            "warmup": warmup,
            "repeats": repeats,
            "resolutions": list(resolutions),
        },
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark prepared binary phase-field finite elements."
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/phase_field_production.json"),
    )
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        raise ValueError("repeats must be positive.")
    report = benchmark(quick=arguments.quick, repeats=arguments.repeats)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(payload, encoding="utf-8")
    print(arguments.output)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
