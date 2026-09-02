#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
from _runtime import (
    capture_environment,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
)

import phydrax as phx


def _case(size: int, repeats: int):
    tn = phx.tensor_network
    structure = tn.ContractionStructure(
        (
            tn.ContractionOperand(
                "left",
                (tn.ContractionLeg("i", size), tn.ContractionLeg("j", size)),
            ),
            tn.ContractionOperand(
                "right",
                (tn.ContractionLeg("j", size), tn.ContractionLeg("k", size)),
            ),
        ),
        ("i", "k"),
    )
    plan, plan_seconds = measure_host(
        lambda: tn.plan_contraction(structure, dtype="float64")
    )
    left = jnp.arange(float(size * size)).reshape((size, size))
    right = jnp.flip(left, axis=0)
    prepared, prepare_seconds = measure_host(
        lambda: tn.prepare_contraction(plan, (left, right))
    )
    execute = eqx.filter_jit(tn.execute_contraction)
    compiled, compilation = measure_lower_and_compile(
        lambda: execute.lower(prepared), lambda lowered: lowered.compile()
    )
    result, execution = measure_repeated(
        lambda: compiled(prepared), warmup=1, repeats=repeats
    )
    refreshed, refresh_seconds = measure_host(
        lambda: tn.refresh_contraction(prepared, (left + 1.0, right))
    )
    return {
        "size": size,
        "plan_seconds": plan_seconds,
        "prepare_seconds": prepare_seconds,
        "refresh_seconds": refresh_seconds,
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "execution": execution.to_milliseconds_dict(),
        "cost": {
            "operand_elements": plan.cost.operand_elements,
            "output_elements": plan.cost.output_elements,
            "largest_intermediate_elements": plan.cost.largest_intermediate_elements,
            "workspace_bytes": plan.cost.workspace_bytes,
            "estimated_flops": plan.cost.estimated_flops,
        },
        "finite": bool(result.evidence.finite),
        "refresh_version": int(refreshed.numeric_version),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[8, 32, 128])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if any(value < 1 for value in arguments.sizes) or arguments.repeats < 1:
        raise ValueError("Benchmark sizes and repeats must be positive.")
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": [_case(size, arguments.repeats) for size in arguments.sizes],
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
