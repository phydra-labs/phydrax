"""Measure compiler CSE versus explicit fixed-coordinate reuse."""

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._io import write_json_atomic
from benchmarks._memory import evaluate_memory_gate, MemoryGatePolicy
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    measure_lower_and_compile,
)
from phydrax.integration._lowering import materialize_fixed_component


def run(shared: bool, terms: int) -> dict[str, Any]:
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    target = phx.integration.over(domain.component())
    plan = phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(64))

    def evaluate(scale: jax.Array) -> jax.Array:
        batches = (
            (materialize_fixed_component(target, plan),)
            if shared
            else tuple(materialize_fixed_component(target, plan) for _ in range(terms))
        )
        outputs = []
        for index in range(terms):
            batch = batches[0] if shared else batches[index]
            leaves = [
                leaf
                for leaf in jax.tree.leaves(batch)
                if isinstance(leaf, jax.Array)
                and jnp.issubdtype(leaf.dtype, jnp.floating)
            ]
            outputs.append(sum(jnp.sum(jnp.sin(scale * leaf + index)) for leaf in leaves))
        return jnp.stack(outputs)

    value = jnp.asarray(0.7, dtype=jnp.float64)
    compiled, timing = measure_lower_and_compile(
        lambda: jax.jit(evaluate).lower(value), lambda lowered: lowered.compile()
    )
    evidence = asdict(
        compiler_evidence(
            compiled.cost_analysis(), compiled.memory_analysis(), source="jax-compiled"
        )
    )
    return {
        "terms": terms,
        "shared": shared,
        "compiler": evidence,
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "value": np.asarray(compiled(value)).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    sizes = [1, 2] if args.smoke else [1, 2, 4, 8]
    baseline = [run(False, size) for size in sizes]
    candidate = [run(True, size) for size in sizes]
    for left, right in zip(baseline, candidate, strict=True):
        np.testing.assert_allclose(left["value"], right["value"], atol=1.0e-12)
    gate = evaluate_memory_gate(
        sizes,
        [row["compiler"]["temporary_bytes"] for row in baseline],
        [row["compiler"]["temporary_bytes"] for row in candidate],
        MemoryGatePolicy(),
    )
    root = Path(__file__).resolve().parents[1]
    identity = capture_benchmark_identity(
        root, Path(__file__), ("terms", "compiler", "value")
    ).to_dict()
    common = {
        "identity": identity,
        "environment": capture_environment().to_dict(),
        "gate": gate.to_dict(),
    }
    write_json_atomic(
        args.output_dir / "memory_integration_lowering_baseline.json",
        {**common, "cases": baseline},
    )
    write_json_atomic(
        args.output_dir / "memory_integration_lowering.json",
        {**common, "cases": candidate},
    )
    print(gate.to_dict())


if __name__ == "__main__":
    main()
