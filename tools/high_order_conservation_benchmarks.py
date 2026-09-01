#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
from high_order_conservation_qualification import (
    _constant_state,
    _tensor_problem,
    _triangle_problem,
)


def _measure(dynamics, state, repeats):
    compiled = jax.jit(lambda value: dynamics(0.0, value))
    start = perf_counter()
    result = compiled(state)
    jax.block_until_ready(result)
    compile_seconds = perf_counter() - start
    start = perf_counter()
    for _ in range(repeats):
        result = compiled(state)
    jax.block_until_ready(result)
    elapsed = perf_counter() - start
    return {
        "dofs": int(state.shape[0]),
        "components": int(state.shape[-1]),
        "compile_seconds": compile_seconds,
        "seconds_per_rhs": elapsed / repeats,
        "dof_updates_per_second": repeats * state.shape[0] / elapsed,
        "finite": bool(jnp.all(jnp.isfinite(result))),
    }


def run(repeats=20):
    tensor, tensor_system, tensor_discretization = _tensor_problem()
    tensor_state = _constant_state(tensor_system, tensor_discretization)
    viscous, viscous_system, viscous_discretization = _tensor_problem(viscous=True)
    viscous_state = _constant_state(viscous_system, viscous_discretization)
    triangle, triangle_system, triangle_discretization = _triangle_problem()
    triangle_state = _constant_state(triangle_system, triangle_discretization)
    result = {
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "dtype": str(tensor_state.dtype),
        "repeats": int(repeats),
        "tensor_dgsem": _measure(tensor, tensor_state, repeats),
        "tensor_ldg": _measure(viscous, viscous_state, repeats),
        "triangle_nodal_dg": _measure(triangle, triangle_state, repeats),
    }
    result["passed"] = all(
        value["finite"]
        for name, value in result.items()
        if name in ("tensor_dgsem", "tensor_ldg", "triangle_nodal_dg")
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/high_order_conservation.json"),
    )
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive.")
    result = run(args.repeats)
    if not result["passed"]:
        raise RuntimeError("High-order conservation benchmark failed.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
