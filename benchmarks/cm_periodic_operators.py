#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax.operators.periodic import (
    differentiate_periodic_translation_family,
    periodic_translation_family_from_dense_blocks,
    realize_periodic_translation_family,
)


def _family(dofs: int, rank: int, q_count: int):
    translations = []
    blocks = []
    for axis in range(rank):
        for direction in (-1, 1):
            translation = np.zeros(rank, dtype=np.int32)
            translation[axis] = direction
            translations.append(translation)
            blocks.append(-0.5 * np.eye(dofs).reshape(dofs, 1, dofs, 1))
    translations.append(np.zeros(rank, dtype=np.int32))
    blocks.append(2.0 * rank * np.eye(dofs).reshape(dofs, 1, dofs, 1))
    return periodic_translation_family_from_dense_blocks(
        translations,
        blocks,
        maximum_dense_entries=max(q_count * rank * dofs * dofs, q_count * dofs * dofs),
        maximum_finite_entries=8_000_000,
    )


def benchmark_case(dofs: int, rank: int, q_count: int, repeats: int):
    prepared, prepare_seconds = measure_host(lambda: _family(dofs, rank, q_count))
    points = jnp.linspace(-0.45, 0.45, q_count * rank).reshape(q_count, rank)
    vectors = jnp.ones((q_count, dofs))
    sparse = jax.jit(lambda q, value: prepared.apply(q, value))
    derivative = jax.jit(
        lambda q: differentiate_periodic_translation_family(prepared, q, order=1)
    )
    lowered = sparse.lower(points, vectors)
    executable, compilation = measure_lower_and_compile(
        lambda: sparse.lower(points, vectors), lambda value: value.compile()
    )
    warm_value, warm_seconds = measure_synchronized(lambda: executable(points, vectors))
    steady_value, steady = measure_repeated(
        lambda: executable(points, vectors), warmup=0, repeats=repeats
    )
    derivative_value, derivative_seconds = measure_synchronized(
        lambda: derivative(points)
    )
    finite, finite_seconds = measure_host(
        lambda: realize_periodic_translation_family(
            prepared,
            (3,) * rank,
            periodic_axes=(False,) * rank,
        )
    )
    analysis = compiler_evidence(
        lowered.cost_analysis(),
        executable.memory_analysis(),
        source="jax-compiler-analysis",
    )
    dense = prepared.evaluate(points)
    residual = float(
        jnp.max(jnp.abs(steady_value - (dense @ vectors[..., None])[..., 0]))
    )
    return {
        "axes": {"dofs": dofs, "rank": rank, "q_points": q_count},
        "plan_id": prepared.plan.plan_id,
        "prepared_id": prepared.prepared_id,
        "prepare_seconds": prepare_seconds,
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "warm_seconds": warm_seconds,
        "steady": steady.to_seconds_dict(),
        "derivative_seconds": derivative_seconds,
        "finite_realization_seconds": finite_seconds,
        "logical_host_bytes": logical_array_bytes(prepared),
        "logical_finite_bytes": logical_array_bytes(finite),
        "compiler": asdict(analysis),
        "operator_matvec_count": repeats + 1,
        "scientific_residual": residual,
        "finite_entries": finite.entry_count,
        "successful": bool(
            jnp.all(jnp.isfinite(warm_value)) and jnp.all(jnp.isfinite(derivative_value))
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    cases = [
        benchmark_case(2, 1, 8, args.repeats),
        benchmark_case(4, 2, 16, args.repeats),
        benchmark_case(8, 3, 27, args.repeats),
    ]
    payload = {"environment": capture_environment().to_dict(), "cases": cases}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
