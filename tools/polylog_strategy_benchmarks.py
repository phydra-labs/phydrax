#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import logical_array_bytes, synchronize
from phydrax.special._polylog import (
    _polylog_series_argument_derivative_streaming,
    _polylog_series_argument_derivative_term_axis,
    _polylog_series_order_derivative_streaming,
    _polylog_series_order_derivative_term_axis,
    _polylog_series_value_streaming,
    _polylog_series_value_term_axis,
    _TERM_AXIS_MAX_ELEMENTS,
    _term_count,
)


def _checksum(value: Any, /) -> float:
    return sum(
        float(jnp.sum(jnp.abs(leaf)))
        for leaf in jax.tree_util.tree_leaves(value)
        if isinstance(leaf, jax.Array)
    )


def _measure(
    operation: Callable[..., Any],
    arguments: tuple[Any, ...],
    /,
    *,
    repeats: int,
    reference: Any | None = None,
) -> dict[str, float | int | None]:
    lowered = jax.jit(operation).lower(*arguments)
    started = time.perf_counter()
    executable = lowered.compile()
    compile_ms = 1.0e3 * (time.perf_counter() - started)
    result = synchronize(executable(*arguments))
    started = time.perf_counter()
    for _ in range(repeats):
        result = synchronize(executable(*arguments))
    steady_ms = 1.0e3 * (time.perf_counter() - started) / repeats
    memory = executable.memory_analysis()
    output: dict[str, float | int | None] = {
        "compile_ms": compile_ms,
        "steady_ms": steady_ms,
        "checksum": _checksum(result),
        "output_bytes": logical_array_bytes(result),
        "generated_code_bytes": (
            None if memory is None else int(memory.generated_code_size_in_bytes)
        ),
        "workspace_bytes": None if memory is None else int(memory.temp_size_in_bytes),
    }
    if reference is not None:
        leaves = jax.tree_util.tree_leaves(result)
        expected = jax.tree_util.tree_leaves(reference)
        output["reference_max_abs_error"] = max(
            (
                float(jnp.max(jnp.abs(actual - target)))
                for actual, target in zip(leaves, expected, strict=True)
            ),
            default=0.0,
        )
    return output


def _arguments(batch_size: int):
    if batch_size == 0:
        return jnp.asarray(2.5 + 0.2j), jnp.asarray(0.45 + 0.1j)
    order = jnp.linspace(0.5, 4.0, batch_size) + 0.2j
    phase = jnp.linspace(-0.6, 0.6, batch_size)
    argument = 0.6 * jnp.exp(1j * phase)
    return order, argument


def run(*, batch_sizes: Sequence[int], repeats: int) -> dict[str, Any]:
    output: dict[str, Any] = {
        "configuration": {
            "batch_sizes": ["scalar" if size == 0 else size for size in batch_sizes],
            "repeats": repeats,
            "term_axis_max_elements": _TERM_AXIS_MAX_ELEMENTS,
        },
        "batches": {},
    }
    for batch_size in batch_sizes:
        order, argument = _arguments(batch_size)
        terms = _term_count(jnp.real(order).dtype)
        element_count = (math.prod(argument.shape) if argument.shape else 1) * terms
        reference_value = _polylog_series_value_streaming(order, argument)
        reference_order = _polylog_series_order_derivative_streaming(order, argument)
        reference_argument = _polylog_series_argument_derivative_streaming(
            order, argument
        )
        cases: dict[str, Any] = {
            "elements_if_materialized": element_count,
            "public_value": _measure(
                phx.special.polylog,
                (order, argument),
                repeats=repeats,
                reference=reference_value,
            ),
            "public_order_jvp": _measure(
                lambda s, z: jax.jvp(
                    lambda order_: phx.special.polylog(order_, z),
                    (s,),
                    (jnp.ones_like(s),),
                )[1],
                (order, argument),
                repeats=repeats,
                reference=reference_order,
            ),
            "public_argument_jvp": _measure(
                lambda s, z: jax.jvp(
                    lambda argument_: phx.special.polylog(s, argument_),
                    (z,),
                    (jnp.ones_like(z),),
                )[1],
                (order, argument),
                repeats=repeats,
                reference=reference_argument,
            ),
            "streaming_value": _measure(
                _polylog_series_value_streaming,
                (order, argument),
                repeats=repeats,
                reference=reference_value,
            ),
        }
        if argument.ndim > 0 and element_count <= _TERM_AXIS_MAX_ELEMENTS:
            cases["term_axis_value"] = _measure(
                _polylog_series_value_term_axis,
                (order, argument),
                repeats=repeats,
                reference=reference_value,
            )
            cases["term_axis_order_derivative"] = _measure(
                _polylog_series_order_derivative_term_axis,
                (order, argument),
                repeats=repeats,
                reference=reference_order,
            )
            cases["term_axis_argument_derivative"] = _measure(
                _polylog_series_argument_derivative_term_axis,
                (order, argument),
                repeats=repeats,
                reference=reference_argument,
            )
        else:
            cases["term_axis_skip_reason"] = (
                "scalar route avoids the temporary"
                if argument.ndim == 0
                else "term tensor exceeds the static element budget"
            )
        output["batches"]["scalar" if batch_size == 0 else str(batch_size)] = cases
    return output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark polylog execution strategies."
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[0, 1_024, 65_536],
        help="Use zero to request a scalar input.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    arguments = parser.parse_args(argv)
    if arguments.repeats <= 0 or any(size < 0 for size in arguments.batch_sizes):
        parser.error("batch sizes must be nonnegative and repeats positive")
    print(
        json.dumps(
            run(batch_sizes=arguments.batch_sizes, repeats=arguments.repeats),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
