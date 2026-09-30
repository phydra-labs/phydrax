# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax.discretization import (
    FiniteVolumePlan,
    MACOperatorPlan,
    PreparedMACOperators,
    TensorGridPlan,
    UniformCellAxisSpec,
)
from phydrax.linalg import AbstractLinearOperator


@eqx.filter_jit
def _mac_laplacian(operators: PreparedMACOperators, value: Array, /) -> Array:
    return operators.positive_laplacian(value)


@eqx.filter_jit
def _complex_laplacian(divergence: AbstractLinearOperator, value: Array, /) -> Array:
    return divergence.mv(divergence.adjoint_mv(value))


def prepare_mac(count: int, dimension: int, periodic: bool, /) -> PreparedMACOperators:
    grid = TensorGridPlan(
        tuple(UniformCellAxisSpec(count, periodic=periodic) for _ in range(dimension)),
        axis_names=tuple(f"axis{axis}" for axis in range(dimension)),
    ).prepare(jnp.asarray([[0.0] * dimension, [1.0] * dimension], dtype=jnp.float64))
    return MACOperatorPlan(FiniteVolumePlan(grid).prepare()).prepare()


def campaign(
    count: int, dimension: int, periodic: bool, repeats: int, /
) -> dict[str, Any]:
    operators, preparation_seconds = measure_synchronized(
        lambda: prepare_mac(count, dimension, periodic)
    )
    complex_, pairing_preparation_seconds = measure_synchronized(
        operators.hilbert_complex_slice
    )
    divergence = complex_.differential(0)
    pressure = jnp.sin(
        jnp.arange(operators.pressure_space.size, dtype=jnp.float64)
    ).reshape(operators.pressure_space.shape)

    reference = operators.positive_laplacian(pressure)
    actions = []
    for name in ("mac_laplacian", "complex_laplacian"):
        # Keep each prepared operator's numerical PyTree leaves dynamic.
        if name == "mac_laplacian":
            # Equinox annotates filter_jit as Callable, omitting runtime lower().
            compiled_mac, timing = measure_lower_and_compile(
                lambda: _mac_laplacian.lower(  # ty: ignore[unresolved-attribute]
                    operators, pressure
                ),
                lambda lowered: lowered.compile(),
            )
            result, warm = measure_repeated(
                lambda: compiled_mac(operators, pressure), warmup=1, repeats=repeats
            )
            executable = compiled_mac.compiled
        else:
            compiled_complex, timing = measure_lower_and_compile(
                lambda: _complex_laplacian.lower(  # ty: ignore[unresolved-attribute]
                    divergence, pressure
                ),
                lambda lowered: lowered.compile(),
            )
            result, warm = measure_repeated(
                lambda: compiled_complex(divergence, pressure), warmup=1, repeats=repeats
            )
            executable = compiled_complex.compiled
        np.testing.assert_allclose(result, reference, rtol=1e-12, atol=1e-10)
        evidence = compiler_evidence(
            executable.cost_analysis(),
            executable.memory_analysis(),
            source="jax-compiled",
        )
        actions.append(
            {
                "action": name,
                "maximum_error": float(jnp.max(jnp.abs(result - reference))),
                "compilation": asdict(timing),
                "warm": warm.to_seconds_dict(),
                "compiler": asdict(evidence),
            }
        )
    adjoint = divergence.adjoint_mv(pressure)
    gradient = operators.gradient(pressure)
    adjoint_error = max(
        float(jnp.max(jnp.abs(a + g))) for a, g in zip(adjoint, gradient, strict=True)
    )
    if adjoint_error > 1e-10:
        raise ValueError("MAC paired adjoint is not minus the pressure gradient.")
    return {
        "count_per_axis": count,
        "dimension": dimension,
        "periodic": periodic,
        "pressure_coordinates": operators.pressure_space.size,
        "face_coordinates": operators.velocity_space.size,
        "retained_bytes": logical_array_bytes((operators, complex_)),
        "preparation_seconds": preparation_seconds,
        "pairing_preparation_seconds": pairing_preparation_seconds,
        "adjoint_error": adjoint_error,
        "actions": actions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/mac_complex_parity.json")
    )
    options = parser.parse_args()
    rows = [
        campaign(count, options.dimension, periodic, options.repeats)
        for count in options.counts
        for periodic in (False, True)
    ]
    root = Path(__file__).resolve().parents[1]
    identity = capture_benchmark_identity(root, Path(__file__), rows[0].keys())
    write_json_atomic(
        options.output,
        {
            "identity": identity.to_dict(),
            "environment": capture_environment().to_dict(),
            "cases": rows,
        },
    )


if __name__ == "__main__":
    main()
