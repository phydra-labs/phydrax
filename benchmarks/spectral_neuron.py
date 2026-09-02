#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from _runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)
from opt_einsum import contract

import phydrax as phx


def _array_outputs(value: Any, /) -> tuple[jax.Array, ...]:
    return tuple(
        leaf for leaf in jax.tree_util.tree_leaves(value) if isinstance(leaf, jax.Array)
    )


def _timed_case(
    name: str,
    function,
    arguments: tuple[Any, ...],
    /,
    *,
    metadata: dict[str, Any],
    warmup: int,
    repeats: int,
    unwrap_compiled: bool = False,
) -> tuple[dict[str, Any], Any]:
    compiled, compilation = measure_lower_and_compile(
        lambda: function.lower(*arguments),
        lambda lowered: lowered.compile(),
    )
    value, execution = measure_repeated(
        lambda: compiled(*arguments),
        warmup=warmup,
        repeats=repeats,
    )
    compiler_target = compiled.compiled if unwrap_compiled else compiled
    evidence = compiler_evidence(
        compiler_target.cost_analysis(),
        compiler_target.memory_analysis(),
        source="jax-compiled-executable",
    )
    outputs = _array_outputs(value)
    finite = all(bool(jnp.all(jnp.isfinite(output))) for output in outputs)
    maximum_absolute = max(
        (float(jnp.max(jnp.abs(output))) for output in outputs if output.size),
        default=0.0,
    )
    return (
        {
            "name": name,
            **metadata,
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
            "execution": execution.to_milliseconds_dict(),
            "logical_input_bytes": logical_array_bytes(arguments),
            "logical_output_bytes": logical_array_bytes(value),
            "output_all_finite": finite,
            "output_maximum_absolute": maximum_absolute,
            "compiler": {
                "flops": evidence.flops,
                "bytes_accessed": evidence.bytes_accessed,
                "argument_bytes": evidence.argument_bytes,
                "output_bytes": evidence.output_bytes,
                "temporary_bytes": evidence.temporary_bytes,
                "generated_code_bytes": evidence.generated_code_bytes,
                "source": evidence.source,
                "unavailable_reason": evidence.unavailable_reason,
            },
        },
        value,
    )


def _monotonicity(mode: str, feature_count: int, /) -> tuple[str, ...]:
    if mode == "free":
        return ("free",) * feature_count
    pattern = ("free", "increasing", "decreasing")
    return tuple(pattern[index % len(pattern)] for index in range(feature_count))


def _benchmark_configuration(
    *,
    matrix_size: int,
    feature_count: int,
    batch_size: int,
    mode: str,
    key: jax.Array,
    warmup: int,
    repeats: int,
) -> list[dict[str, Any]]:
    model_key, input_key = jax.random.split(key)
    model = phx.nn.layers.SpectralNeuron(
        in_size=feature_count,
        matrix_size=matrix_size,
        eigen_index=matrix_size // 2,
        monotonicity=_monotonicity(mode, feature_count),
        dtype=jnp.float32,
        key=model_key,
    )
    points = jax.random.normal(input_key, (batch_size, feature_count), dtype=jnp.float32)
    base, features = model.materialize_coefficients()
    metadata = {
        "matrix_size": matrix_size,
        "feature_count": feature_count,
        "batch_size": batch_size,
        "mode": mode,
        "parameter_bytes": logical_array_bytes(model),
    }

    forward = jax.jit(model)
    dense_reference = jax.jit(
        lambda value: jnp.linalg.eigvalsh(
            base + contract("...i,ijk->...jk", value, features)
        )[..., model.eigen_index]
    )
    input_gradient = jax.jit(jax.grad(lambda value: jnp.sum(model(value))))
    parameter_gradient = eqx.filter_jit(
        eqx.filter_grad(lambda layer: jnp.sum(layer(points)))
    )

    cases = []
    forward_case, forward_value = _timed_case(
        "spectral_neuron_forward",
        forward,
        (points,),
        metadata=metadata,
        warmup=warmup,
        repeats=repeats,
    )
    cases.append(forward_case)
    reference_case, reference_value = _timed_case(
        "dense_reference_forward",
        dense_reference,
        (points,),
        metadata=metadata,
        warmup=warmup,
        repeats=repeats,
    )
    cases.append(reference_case)
    input_case, _ = _timed_case(
        "spectral_neuron_input_gradient",
        input_gradient,
        (points,),
        metadata=metadata,
        warmup=warmup,
        repeats=repeats,
    )
    cases.append(input_case)
    parameter_case, _ = _timed_case(
        "spectral_neuron_parameter_gradient",
        parameter_gradient,
        (model,),
        metadata=metadata,
        warmup=warmup,
        repeats=repeats,
        unwrap_compiled=True,
    )
    cases.append(parameter_case)

    residual = float(jnp.max(jnp.abs(forward_value - reference_value)))
    for case in (forward_case, reference_case):
        case["reference_maximum_absolute_residual"] = residual
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-sizes", nargs="+", type=int, default=[3, 7, 11, 31])
    parser.add_argument("--feature-counts", nargs="+", type=int, default=[8, 32])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 256])
    parser.add_argument(
        "--modes", nargs="+", choices=("free", "mixed"), default=["free", "mixed"]
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    matrix_sizes = tuple(int(value) for value in arguments.matrix_sizes)
    feature_counts = tuple(int(value) for value in arguments.feature_counts)
    batch_sizes = tuple(int(value) for value in arguments.batch_sizes)
    if any(value <= 0 for value in matrix_sizes + feature_counts + batch_sizes):
        raise ValueError(
            "matrix sizes, feature counts, and batch sizes must be positive."
        )
    if arguments.warmup < 0 or arguments.repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive.")

    key = jax.random.PRNGKey(20260902)
    cases = []
    case_index = 0
    for mode in arguments.modes:
        for batch_size in batch_sizes:
            for feature_count in feature_counts:
                for matrix_size in matrix_sizes:
                    cases.extend(
                        _benchmark_configuration(
                            matrix_size=matrix_size,
                            feature_count=feature_count,
                            batch_size=batch_size,
                            mode=mode,
                            key=jax.random.fold_in(key, case_index),
                            warmup=arguments.warmup,
                            repeats=arguments.repeats,
                        )
                    )
                    case_index += 1
    residuals = [
        case["reference_maximum_absolute_residual"]
        for case in cases
        if "reference_maximum_absolute_residual" in case
    ]
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "maximum_reference_residual": max(residuals, default=0.0),
        "all_outputs_finite": all(case["output_all_finite"] for case in cases),
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
