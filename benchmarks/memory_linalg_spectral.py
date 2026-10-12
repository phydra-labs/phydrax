"""Compile-only probe workspace scaling; smoke defaults stay below 4 GB."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from phydrax import linalg as la

from ._io import write_json_atomic
from ._memory import (
    evaluate_memory_gate,
    fitted_scaling_exponent,
    MemoryGatePolicy,
    process_peak_rss_bytes,
)
from ._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--probes", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    properties = la.OperatorProperties(
        self_adjoint=True, evidence={"self_adjoint": "asserted"}
    )
    operator = la.DenseLinearOperator(
        jnp.diag(jnp.linspace(1.0, 2.0, args.size, dtype=jnp.float64)),
        properties=properties,
    )
    cases: list[dict[str, Any]] = []
    for count in args.probes:

        def entry(key: jax.Array) -> jax.Array:
            if args.batch_size is None:
                return la.stochastic_trace(
                    operator,
                    key=key,
                    num_probes=count,
                    max_dimension=16,
                    execution="parallel",
                ).samples
            return la.stochastic_trace(
                operator,
                key=key,
                num_probes=count,
                max_dimension=16,
                batch_size=args.batch_size,
            ).samples

        compiled, timing = measure_lower_and_compile(
            lambda: jax.jit(entry).lower(jax.random.key(0)),
            lambda lowered: lowered.compile(),
        )
        evidence = compiler_evidence(
            compiled.cost_analysis(),
            compiled.memory_analysis(),
            source="jax-spectral-probe-workspace",
            unavailable_reason="Compiler memory evidence unavailable.",
        )
        if evidence.temporary_bytes is None:
            raise RuntimeError("Compiler temporary-byte evidence is required.")
        cases.append(
            {
                "probes": count,
                "compiler": asdict(evidence),
                "temporary_bytes": evidence.temporary_bytes,
                "lowering_seconds": timing.lowering_seconds,
                "compilation_seconds": timing.compilation_seconds,
            }
        )
    inverse_cases: list[dict[str, Any]] = []
    for count in args.probes:

        def inverse_entry(matrix_: jax.Array, key: jax.Array) -> jax.Array:
            local = la.DenseLinearOperator(
                matrix_, properties=properties, operator_id="memory-inverse-diagonal"
            )
            return la.estimate_inverse_diagonal(
                local,
                key=key,
                num_probes=count,
                batch_size=4 if args.batch_size is None else args.batch_size,
            ).samples

        compiled, timing = measure_lower_and_compile(
            lambda: jax.jit(inverse_entry).lower(operator.matrix, jax.random.key(0)),
            lambda lowered: lowered.compile(),
        )
        value, warm = measure_repeated(
            lambda: compiled(operator.matrix, jax.random.key(0)), warmup=1, repeats=1
        )
        np.testing.assert_allclose(
            value,
            np.tile(1.0 / np.linspace(1.0, 2.0, args.size), (count, 1)),
            rtol=1e-12,
            atol=1e-12,
        )
        evidence = compiler_evidence(
            compiled.cost_analysis(),
            compiled.memory_analysis(),
            source="jax-prepared-inverse-diagonal-probe-chunks",
            unavailable_reason="Compiler memory evidence unavailable.",
        )
        if evidence.temporary_bytes is None:
            raise RuntimeError("Compiler temporary-byte evidence is required.")
        inverse_cases.append(
            {
                "probes": count,
                "compiler": asdict(evidence),
                "temporary_bytes": evidence.temporary_bytes,
                "retained_output_bytes": logical_array_bytes(value),
                "lowering_seconds": timing.lowering_seconds,
                "compilation_seconds": timing.compilation_seconds,
                "warm": warm.to_seconds_dict(),
            }
        )
    resolvent_cases: list[dict[str, Any]] = []
    for size in (args.size, 2 * args.size):
        matrix = jnp.diag(jnp.linspace(1.0, 2.0, size, dtype=jnp.float64))[None]
        rhs = jnp.ones((size, 2), dtype=jnp.float64)
        for route in ("inverse-reference", "rhs-action"):

            def resolvent(matrix_: jax.Array, rhs_: jax.Array) -> jax.Array:
                if route == "inverse-reference":
                    identity = jnp.eye(size, dtype=matrix_.dtype)
                    return jnp.linalg.solve(-identity - matrix_, identity) @ rhs_
                local = la.DenseLinearOperator(matrix_, operator_id="memory-resolvent")
                value = la.matrix_function_action(
                    local,
                    rhs_,
                    kind="resolvent",
                    shift=-1.0,
                    rhs_layout=la.RHSLayout((2,)),
                ).value
                if not isinstance(value, jax.Array):
                    raise TypeError("Dense resolvent returned a non-array value.")
                return value

            compiled, timing = measure_lower_and_compile(
                lambda: jax.jit(resolvent).lower(matrix, rhs),
                lambda lowered: lowered.compile(),
            )
            value, warm = measure_repeated(
                lambda: compiled(matrix, rhs), warmup=1, repeats=1
            )
            expected = (
                -np.ones((size, 2), dtype=np.float64)
                / (1.0 + np.linspace(1.0, 2.0, size))[:, None]
            )
            np.testing.assert_allclose(value[0], expected, rtol=1e-12, atol=1e-12)
            evidence = compiler_evidence(
                compiled.cost_analysis(),
                compiled.memory_analysis(),
                source="jax-resolvent-rhs-action",
                unavailable_reason="Compiler memory evidence unavailable.",
            )
            resolvent_cases.append(
                {
                    "size": size,
                    "rhs_count": 2,
                    "route": route,
                    "compiler": asdict(evidence),
                    "action_rhs_workspace_bytes": (
                        size * size + size * 2
                        if route == "inverse-reference"
                        else 3 * size * 2
                    )
                    * matrix.dtype.itemsize,
                    "factor_storage_bytes": size * size * matrix.dtype.itemsize,
                    "retained_output_bytes": logical_array_bytes(value),
                    "lowering_seconds": timing.lowering_seconds,
                    "compilation_seconds": timing.compilation_seconds,
                    "warm": warm.to_seconds_dict(),
                }
            )
    record = {
        "environment": asdict(capture_environment()),
        "cases": cases,
        "inverse_cases": inverse_cases,
        "inverse_workspace_exponent": fitted_scaling_exponent(
            args.probes, [case["temporary_bytes"] for case in inverse_cases]
        ),
        "inverse_fourfold_workspace_ratio": (
            inverse_cases[-1]["temporary_bytes"] / inverse_cases[0]["temporary_bytes"]
        ),
        "workspace_exponent": fitted_scaling_exponent(
            args.probes, [case["temporary_bytes"] for case in cases]
        ),
        "peak_rss_bytes": process_peak_rss_bytes(),
        "batch_size": args.batch_size,
        "resolvent_cases": resolvent_cases,
        "identity": asdict(
            capture_benchmark_identity(
                Path.cwd(),
                Path(__file__),
                ("cases", "inverse_cases", "resolvent_cases", "workspace_exponent"),
            )
        ),
    }
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text())
        record["gate"] = evaluate_memory_gate(
            args.probes,
            [case["temporary_bytes"] for case in baseline["cases"]],
            [case["temporary_bytes"] for case in cases],
            MemoryGatePolicy(),
        ).to_dict()
        record["fourfold_workspace_ratio"] = (
            cases[-1]["temporary_bytes"] / cases[0]["temporary_bytes"]
        )
    write_json_atomic(args.output, record)


if __name__ == "__main__":
    main()
