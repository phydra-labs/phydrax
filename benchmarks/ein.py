#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

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
from phydrax.discretization.iga._basis import TensorSplineBasisSpec
from phydrax.discretization.iga._realization import (
    DirectTensorRealization,
    ExtractedBernsteinRealization,
)
from phydrax.discretization.iga._topology import SplineSpanTopology


_STABLEHLO_OPERATIONS = (
    "stablehlo.broadcast_in_dim",
    "stablehlo.dot_general",
    "stablehlo.reduce",
    "stablehlo.reshape",
    "stablehlo.transpose",
)


def _timed_case(
    name: str,
    function,
    arguments: tuple[jax.Array, ...],
    reference,
    /,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    compiled, compilation = measure_lower_and_compile(
        lambda: function.lower(*arguments),
        lambda lowered: lowered.compile(),
    )
    value, execution = measure_repeated(
        lambda: compiled(*arguments),
        warmup=warmup,
        repeats=repeats,
    )
    expected = reference(*arguments)
    residual = jnp.max(jnp.abs(value - expected), initial=0.0)
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-executable",
    )
    stablehlo = str(function.lower(*arguments).compiler_ir(dialect="stablehlo"))
    return {
        "name": name,
        "input_shapes": [list(argument.shape) for argument in arguments],
        "input_dtypes": [str(argument.dtype) for argument in arguments],
        "output_shape": list(value.shape),
        "output_dtype": str(value.dtype),
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "execution": execution.to_milliseconds_dict(),
        "logical_input_bytes": logical_array_bytes(arguments),
        "logical_output_bytes": logical_array_bytes(value),
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
        "stablehlo_operation_counts": {
            operation: stablehlo.count(operation) for operation in _STABLEHLO_OPERATIONS
        },
        "host_callback_present": "xla_python_cpu_callback" in stablehlo,
        "maximum_residual": float(residual),
    }


def _transform_cases(
    key: jax.Array,
    *,
    batch_size: int,
    feature_size: int,
    payload_size: int,
    warmup: int,
    repeats: int,
) -> list[dict[str, Any]]:
    values = jax.random.normal(
        key,
        (batch_size, 2 * feature_size, payload_size),
        dtype=jnp.float64,
    )

    def native(array):
        heads = phx.ein.rearrange(
            array,
            "batch (head channel) payload -> batch head payload channel",
            head=2,
        )
        reduced = phx.ein.reduce(
            heads,
            "batch head payload channel -> batch head channel",
            "mean",
        )
        return phx.ein.repeat(
            reduced,
            "batch head channel -> batch replica (head channel)",
            replica=2,
        )

    def reference(array):
        heads = array.reshape(batch_size, 2, feature_size, payload_size).transpose(
            0,
            1,
            3,
            2,
        )
        reduced = jnp.mean(heads, axis=2)
        repeated = jnp.broadcast_to(
            reduced[:, None, :, :],
            (batch_size, 2, 2, feature_size),
        )
        return repeated.reshape(batch_size, 2, 2 * feature_size)

    return [
        _timed_case(
            "native_transform_pipeline",
            jax.jit(native),
            (values,),
            reference,
            warmup=warmup,
            repeats=repeats,
        ),
        _timed_case(
            "direct_jax_transform_pipeline",
            jax.jit(reference),
            (values,),
            reference,
            warmup=warmup,
            repeats=repeats,
        ),
    ]


def _iga_cases(
    key: jax.Array,
    *,
    batch_size: int,
    payload_size: int,
    warmup: int,
    repeats: int,
) -> list[dict[str, Any]]:
    grid = phx.discretization.iga.BSplineGrid.open_uniform(1, max(2, batch_size))
    basis = TensorSplineBasisSpec((grid,), axis_names=("xi",))
    direct = DirectTensorRealization(basis, SplineSpanTopology(basis))
    extraction_key, coefficient_key, dual_key = jax.random.split(key, 3)
    extraction = jax.random.normal(
        extraction_key,
        (direct.cell_count, direct.local_width, direct.local_width),
        dtype=jnp.float64,
    ) + 2.0 * jnp.eye(direct.local_width, dtype=jnp.float64)
    realization = ExtractedBernsteinRealization(direct, extraction)
    coefficients = jax.random.normal(
        coefficient_key,
        (basis.coefficient_count, payload_size),
        dtype=jnp.float64,
    )
    dual = jax.random.normal(
        dual_key,
        (direct.cell_count, direct.local_width, payload_size),
        dtype=jnp.float64,
    )
    expanded_extraction = extraction[..., None]

    def realize_reference(values):
        local = direct.gather(values)
        return jnp.sum(expanded_extraction * local[:, None, :, :], axis=2)

    def transpose_reference(values):
        local_dual = jnp.sum(
            expanded_extraction * values[:, :, None, :],
            axis=1,
        )
        return direct.gather_transpose(local_dual)

    return [
        _timed_case(
            "iga_extraction_forward",
            jax.jit(lambda values: realization.realize(values)),
            (coefficients,),
            realize_reference,
            warmup=warmup,
            repeats=repeats,
        ),
        _timed_case(
            "iga_extraction_transpose",
            jax.jit(lambda values: realization.transpose(values)),
            (dual,),
            transpose_reference,
            warmup=warmup,
            repeats=repeats,
        ),
    ]


def _clifford_cases(
    key: jax.Array,
    *,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> list[dict[str, Any]]:
    algebra = phx.metrix.clifford.CliffordAlgebraSpec((1, 1))
    input_representation = phx.nn.operator.representations.CliffordGradeRepresentation(
        algebra,
        (4, 4, 0),
    )
    output_representation = phx.nn.operator.representations.CliffordGradeRepresentation(
        algebra,
        (4, 4, 2),
    )
    layer_key, values_key = jax.random.split(key)
    layer = phx.nn.operator.layers.CliffordGradeLinear(
        input_representation,
        output_representation,
        key=layer_key,
    )
    values = jax.random.normal(
        values_key,
        (batch_size, 2, input_representation.packed_size),
        dtype=jnp.float64,
    )

    def reference(array):
        features = input_representation.split(array)
        leading = array.shape[:-1]
        grades = []
        for grade, (grade_values, weight, output_count, layout) in enumerate(
            zip(
                features.grades,
                layer.weights,
                output_representation.multiplicities,
                output_representation.grade_layouts,
                strict=True,
            )
        ):
            if weight is None:
                mixed = jnp.zeros(
                    leading + (output_count, layout.blade_count),
                    dtype=array.dtype,
                )
            else:
                expanded_weight = weight.reshape(
                    (1,) * len(leading) + weight.shape + (1,)
                )
                mixed = jnp.sum(
                    expanded_weight * grade_values[..., None, :, :],
                    axis=len(leading) + 1,
                )
            if grade == 0 and layer.scalar_bias is not None:
                mixed = mixed + layer.scalar_bias.reshape(
                    (1,) * len(leading) + layer.scalar_bias.shape + (1,)
                )
            grades.append(mixed)
        return output_representation.join(
            phx.nn.operator.representations.CliffordGradeFeatures(tuple(grades))
        )

    return [
        _timed_case(
            "clifford_grade_channel_mixing",
            jax.jit(layer),
            (values,),
            reference,
            warmup=warmup,
            repeats=repeats,
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-size", type=int, default=32)
    parser.add_argument("--payload-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    positive = (
        arguments.batch_size,
        arguments.feature_size,
        arguments.payload_size,
        arguments.repeats,
    )
    if any(value < 1 for value in positive) or arguments.warmup < 0:
        raise ValueError("sizes and repeats must be positive and warmup nonnegative.")

    key = jax.random.PRNGKey(20260902)
    cases = []
    cases.extend(
        _transform_cases(
            key,
            batch_size=arguments.batch_size,
            feature_size=arguments.feature_size,
            payload_size=arguments.payload_size,
            warmup=arguments.warmup,
            repeats=arguments.repeats,
        )
    )
    cases.extend(
        _iga_cases(
            jax.random.fold_in(key, 1),
            batch_size=arguments.batch_size,
            payload_size=arguments.payload_size,
            warmup=arguments.warmup,
            repeats=arguments.repeats,
        )
    )
    cases.extend(
        _clifford_cases(
            jax.random.fold_in(key, 2),
            batch_size=arguments.batch_size,
            warmup=arguments.warmup,
            repeats=arguments.repeats,
        )
    )
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "batch_size": arguments.batch_size,
            "feature_size": arguments.feature_size,
            "payload_size": arguments.payload_size,
            "warmup": arguments.warmup,
            "repeats": arguments.repeats,
        },
        "cases": cases,
        "maximum_residual": max(case["maximum_residual"] for case in cases),
        "host_callbacks_present": any(case["host_callback_present"] for case in cases),
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
