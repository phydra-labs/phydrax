#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
from _runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)

import phydrax as phx


_CRITICAL_BETA = 0.5 * math.log(1.0 + math.sqrt(2.0))
_CRITICAL_LOG_PARTITION = 0.5 * math.log(2.0) + 2.0 * 0.915965594177219 / math.pi


def _case(method_name: str, bond: int, steps: int, repeats: int):
    tn = phx.tensor_network
    spins = jnp.asarray((-1.0, 1.0), dtype=jnp.float64)
    pair_weight = jnp.exp(_CRITICAL_BETA * spins[:, None] * spins[None, :])
    tensor = tn.build_uniform_pair_partition_tensor(pair_weight).tensor
    method = tn.TRGMethod() if method_name == "trg" else tn.HOTRGMethod()
    problem = tn.TensorRenormalizationProblem(tensor)
    policy = tn.TensorRenormalizationPolicy(
        method,
        maximum_bond_dimension=bond,
        steps=steps,
    )
    plan = tn.plan_tensor_renormalization(problem, policy)
    prepared = tn.prepare_tensor_renormalization(problem, plan)
    execute = eqx.filter_jit(tn.run_tensor_renormalization)
    compiled, compilation = measure_lower_and_compile(
        lambda: execute.lower(prepared),
        lambda lowered: lowered.compile(),
    )
    result, execution = measure_repeated(
        lambda: compiled(prepared),
        warmup=1,
        repeats=repeats,
    )
    compiler = compiler_evidence(
        compiled.compiled.cost_analysis(),
        compiled.compiled.memory_analysis(),
        source="jax-compiled-executable",
    )
    computed = float(result.log_partition_density)
    return {
        "method": method_name,
        "maximum_bond_dimension": bond,
        "steps": steps,
        "logical_input_bytes": logical_array_bytes(prepared),
        "logical_output_bytes": logical_array_bytes(result),
        "planned_peak_bytes": plan.cost.peak_workspace_bytes,
        "estimated_flops": plan.cost.estimated_flops,
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "compiler": {
            "flops": compiler.flops,
            "bytes_accessed": compiler.bytes_accessed,
            "argument_bytes": compiler.argument_bytes,
            "output_bytes": compiler.output_bytes,
            "temporary_bytes": compiler.temporary_bytes,
            "generated_code_bytes": compiler.generated_code_bytes,
            "source": compiler.source,
            "unavailable_reason": compiler.unavailable_reason,
        },
        "execution": execution.to_milliseconds_dict(),
        "computed_log_partition_density": computed,
        "reference_log_partition_density": _CRITICAL_LOG_PARTITION,
        "absolute_error": abs(computed - _CRITICAL_LOG_PARTITION),
        "first_discarded_weight_sum": float(
            jnp.sum(result.diagnostics.first_discarded_weight_history)
        ),
        "second_discarded_weight_sum": float(
            jnp.sum(result.diagnostics.second_discarded_weight_history)
        ),
        "status": int(result.diagnostics.status),
        "accepted": bool(result.successful),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods", nargs="+", choices=("trg", "hotrg"), default=("trg", "hotrg")
    )
    parser.add_argument("--bond-dimensions", nargs="+", type=int, default=(4, 8))
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if (
        any(value < 1 for value in arguments.bond_dimensions)
        or arguments.steps < 1
        or arguments.repeats < 1
    ):
        raise ValueError("Benchmark dimensions, steps, and repeats must be positive.")
    cases = [
        _case(method, bond, arguments.steps, arguments.repeats)
        for method in arguments.methods
        for bond in arguments.bond_dimensions
    ]
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": cases,
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
