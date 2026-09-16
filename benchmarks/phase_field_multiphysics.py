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

from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)
from tools.phase_field_multiphysics_qualification import build_coupled_case


def _compiler(executable) -> dict[str, object]:
    cost = executable.compiled.cost_analysis()
    memory = executable.compiled.memory_analysis()
    evidence = compiler_evidence(
        cost,
        memory,
        source="jax-lowered-compiled-executable",
        unavailable_reason=(
            "Compiler analysis unavailable." if not cost and memory is None else None
        ),
    )
    return {
        **asdict(evidence),
        "estimated_device_memory_bytes": evidence.estimated_device_memory_bytes,
    }


def _coupled_case(repeats: int) -> dict[str, object]:
    plan, state, inputs = build_coupled_case()
    operation: Any = eqx.filter_jit(plan.step)
    arguments = (state, inputs, jnp.asarray(0.0), jnp.asarray(0.01))
    executable, compilation = measure_lower_and_compile(
        lambda: operation.lower(*arguments), lambda lowered: lowered.compile()
    )
    result, timing = measure_repeated(
        lambda: executable(*arguments), warmup=1, repeats=repeats
    )
    return {
        "name": "electro-elasto-hydrodynamic-thermal-flagship",
        "successful": bool(result.successful),
        "compilation": {
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
        },
        "steady_step": timing.to_milliseconds_dict(),
        "compiler": _compiler(executable),
        "logical_argument_bytes": logical_array_bytes(arguments),
        "logical_result_bytes": logical_array_bytes(result),
        "energy_residual": float(np.asarray(result.evidence.ledger.energy_residual)),
        "exchange_defect": float(np.asarray(result.evidence.ledger.exchange_defect)),
        "conservation_defect": float(
            np.asarray(result.evidence.ledger.maximum_conservation_defect)
        ),
    }


def _subsystem_cases(repeats: int) -> dict[str, dict[str, Any]]:
    plan, state, inputs = build_coupled_case()
    thermal_operation: Any = eqx.filter_jit(plan.thermal.step)
    thermal_arguments = (
        state.thermal,
        inputs.phase_logits,
        inputs.chemical_potential,
    )

    def thermal_call():
        return thermal_operation(
            *thermal_arguments,
            heat_input=inputs.heat_input,
            entropy_flux=inputs.entropy_flux,
            entropy_production=inputs.entropy_production,
        )

    thermal_result, thermal_timing = measure_repeated(
        thermal_call, warmup=1, repeats=repeats
    )
    anti_operation: Any = eqx.filter_jit(plan.anti_trapping.evaluate)
    anti_result, anti_timing = measure_repeated(
        lambda: anti_operation(
            inputs.phase_rate,
            inputs.phase_gradient,
            inputs.scalar_chemical_potential,
        ),
        warmup=1,
        repeats=repeats,
    )
    mechanics_operation: Any = eqx.filter_jit(plan.mechanics.evaluate)
    mechanics_result, mechanics_timing = measure_repeated(
        lambda: mechanics_operation(inputs.phase_logits, inputs.displacement_gradient),
        warmup=1,
        repeats=repeats,
    )
    electro_operation: Any = eqx.filter_jit(plan.electrostatic.evaluate)
    electro_result, electro_timing = measure_repeated(
        lambda: electro_operation(
            inputs.phase_logits,
            inputs.potential_gradient,
            free_charge=inputs.free_charge,
            displacement_divergence=inputs.displacement_divergence,
        ),
        warmup=1,
        repeats=repeats,
    )
    return {
        "thermal_constitutive": {
            "successful": bool(thermal_result[1].successful),
            "timing": thermal_timing.to_milliseconds_dict(),
        },
        "anti_trapping": {
            "successful": bool(anti_result.successful),
            "timing": anti_timing.to_milliseconds_dict(),
        },
        "mechanics": {
            "successful": bool(mechanics_result.successful),
            "timing": mechanics_timing.to_milliseconds_dict(),
        },
        "electrostatic": {
            "successful": bool(electro_result.successful),
            "timing": electro_timing.to_milliseconds_dict(),
        },
    }


def benchmark(*, quick: bool, repeats: int) -> dict[str, object]:
    if not bool(jax.config.read("jax_enable_x64")):
        raise ValueError("Multiphysics benchmark requires float64.")
    repetitions = 1 if quick else repeats
    coupled = _coupled_case(repetitions)
    subsystems = _subsystem_cases(repetitions)
    passed = coupled["successful"] and all(
        result["successful"] for result in subsystems.values()
    )
    return {
        "status": "pass" if passed else "fail",
        "environment": capture_environment().to_dict(),
        "settings": {"quick": quick, "repeats": repetitions},
        "coupled": coupled,
        "subsystems": subsystems,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark coupled phase-field multiphysics."
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/phase_field_multiphysics.json"),
    )
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        raise ValueError("repeats must be positive.")
    report = benchmark(quick=arguments.quick, repeats=arguments.repeats)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(arguments.output)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
