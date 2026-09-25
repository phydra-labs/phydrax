#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native Krylov loop drivers: early exit versus fixed trip.

The early-exit route (every non-algorithmic solve) and the fixed-trip route
(`DifferentiationPolicy("algorithmic")`) run on one matrix-free
convection-diffusion system. Cases vary the controlling capacity, the FGMRES
restart length, at a fixed step budget and record lowering, compilation,
warmed execution, and compiler temporary/code bytes for the forward solve and
for a reverse-mode gradient through the executed iteration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from _runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)

import phydrax as phx


la = phx.linalg


def _action(scale: jax.Array, value: jax.Array) -> jax.Array:
    padded = jnp.pad(value, 1)
    laplacian = 2.0 * value - padded[:-2] - padded[2:]
    return laplacian + 0.8 * (value - padded[:-2]) + (1.0 + scale) * value


def _solve(scale, rhs, *, restart, max_steps, mode):
    size = rhs.shape[0]
    space = la.ArraySpace((size,), dtype=rhs.dtype)
    operator = la.FunctionLinearOperator(
        lambda value: _action(scale, value), source=space, target=space
    )
    policy = la.LinearSolvePolicy(
        la.FGMRES(restart=restart, stagnation_iterations=max_steps),
        tolerance=la.TolerancePolicy(relative=0.0, absolute=0.0, max_steps=max_steps),
        differentiation=la.DifferentiationPolicy(mode),
        failure=la.FailurePolicy("status"),
    )
    return la.solve(la.LinearSystem(operator), rhs, policy=policy)


def _timed(name, function, arguments, *, warmup, repeats) -> dict[str, Any]:
    compiled, compilation = measure_lower_and_compile(
        lambda: function.lower(*arguments),
        lambda lowered: lowered.compile(),
    )
    value, execution = measure_repeated(
        lambda: compiled(*arguments),
        warmup=warmup,
        repeats=repeats,
    )
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-executable",
    )
    return {
        "name": name,
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "execution": execution.to_milliseconds_dict(),
        "logical_output_bytes": logical_array_bytes(value),
        "compiler": {
            "flops": evidence.flops,
            "temporary_bytes": evidence.temporary_bytes,
            "generated_code_bytes": evidence.generated_code_bytes,
            "argument_bytes": evidence.argument_bytes,
            "output_bytes": evidence.output_bytes,
            "unavailable_reason": evidence.unavailable_reason,
        },
    }


def _cases(size, max_steps, restarts, *, warmup, repeats) -> list[dict[str, Any]]:
    rhs = jnp.linspace(1.0, 2.0, size)
    scale = jnp.asarray(0.0)
    cases = []
    for restart in restarts:
        def forward(scale_, rhs_, *, mode, restart_=restart):
            return _solve(
                scale_, rhs_, restart=restart_, max_steps=max_steps, mode=mode
            ).value

        def loss(scale_, rhs_, restart_=restart):
            value = _solve(
                scale_, rhs_, restart=restart_, max_steps=max_steps, mode="algorithmic"
            ).value
            return jnp.sum(value**2)

        early = jax.jit(lambda s, r: forward(s, r, mode="none"))
        fixed = jax.jit(lambda s, r: forward(s, r, mode="algorithmic"))
        gradient = jax.jit(jax.grad(loss))
        records = {
            "early_exit_forward": _timed(
                "early-exit", early, (scale, rhs), warmup=warmup, repeats=repeats
            ),
            "fixed_trip_forward": _timed(
                "fixed-trip", fixed, (scale, rhs), warmup=warmup, repeats=repeats
            ),
            "fixed_trip_reverse": _timed(
                "fixed-trip-grad", gradient, (scale, rhs), warmup=warmup, repeats=repeats
            ),
        }
        parity = float(jnp.max(jnp.abs(early(scale, rhs) - fixed(scale, rhs))))
        cases.append(
            {
                "size": size,
                "max_steps": max_steps,
                "restart": restart,
                "early_fixed_max_abs_difference": parity,
                **records,
            }
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--restarts", nargs="+", type=int, default=[4, 8, 16, 32, 64])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.size < 2 or arguments.max_steps < 1:
        raise ValueError("size must exceed one and max_steps must be positive.")
    if any(value < 1 for value in arguments.restarts):
        raise ValueError("restarts must be positive.")
    cases = _cases(
        arguments.size,
        arguments.max_steps,
        tuple(arguments.restarts),
        warmup=arguments.warmup,
        repeats=arguments.repeats,
    )
    passed = all(case["early_fixed_max_abs_difference"] == 0.0 for case in cases)
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "passed": passed,
    }
    encoded = json.dumps(payload, allow_nan=False, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        from benchmarks._io import write_json_atomic

        write_json_atomic(arguments.output, payload)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
