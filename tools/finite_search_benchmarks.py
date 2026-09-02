#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Benchmark factorized finite search against a bounded dense oracle."""

from __future__ import annotations

import argparse
import json
import platform
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axes", type=int, default=3)
    parser.add_argument("--axis-length", type=int, default=16)
    parser.add_argument("--payload-size", type=int, default=1)
    parser.add_argument("--objective-size", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Candidate batch size; zero selects scalar streaming.",
    )
    parser.add_argument("--work", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--max-dense-bytes", type=int, default=512 * 1024 * 1024)
    return parser


def _validate_arguments(arguments: argparse.Namespace) -> None:
    positive = (
        "axes",
        "axis_length",
        "payload_size",
        "objective_size",
        "work",
        "repeat",
        "max_dense_bytes",
    )
    for name in positive:
        if int(vars(arguments)[name]) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if arguments.batch_size < 0:
        raise ValueError("--batch-size must be nonnegative.")


def _candidate_space(
    num_axes: int,
    axis_length: int,
    payload_size: int,
) -> phx.optim.FiniteProductSpace:
    axes = []
    payload_offsets = jnp.arange(payload_size, dtype=float) / max(payload_size, 1)
    for axis_index in range(num_axes):
        coordinates = jnp.linspace(-1.0, 1.0, axis_length)
        values = coordinates[:, None] + payload_offsets[None, :] + axis_index * 0.01
        if payload_size == 1:
            values = values[:, 0]
        axes.append(phx.optim.FiniteAxis(values))
    return phx.optim.FiniteProductSpace(tuple(axes))


def _evaluator(objective_size: int, work: int):
    def evaluate(point):
        leaves = [jnp.ravel(value) for value in jax.tree_util.tree_leaves(point)]
        state = jnp.concatenate(leaves)
        score = jnp.sum((state - 0.125) ** 2)
        for iteration in range(work):
            scale = 0.001 * (iteration + 1)
            score = score + scale * jnp.sum(jnp.sin(state + scale) ** 2)
        if objective_size == 1:
            return score, jnp.asarray(True)
        values = score + jnp.arange(objective_size, dtype=score.dtype) * 1e-6
        return values, jnp.ones((objective_size,), dtype=bool)

    return evaluate


def _compile_and_time(function, repeat: int):
    started = perf_counter()
    executable = jax.jit(function).lower().compile()
    compilation_seconds = perf_counter() - started
    output = jax.block_until_ready(executable())
    samples = []
    for _ in range(repeat):
        started = perf_counter()
        output = jax.block_until_ready(executable())
        samples.append(perf_counter() - started)
    return executable, output, compilation_seconds, samples


def _memory_report(executable) -> dict[str, int]:
    memory = executable.memory_analysis()
    return {
        "argument_bytes": int(memory.argument_size_in_bytes),
        "output_bytes": int(memory.output_size_in_bytes),
        "temporary_bytes": int(memory.temp_size_in_bytes),
        "alias_bytes": int(memory.alias_size_in_bytes),
        "generated_code_bytes": int(memory.generated_code_size_in_bytes),
    }


def _timing_report(samples: list[float], candidate_count: int) -> dict[str, float]:
    values = np.asarray(samples, dtype=float)
    median = float(np.median(values))
    return {
        "minimum_seconds": float(np.min(values)),
        "median_seconds": median,
        "maximum_seconds": float(np.max(values)),
        "candidate_evaluations_per_second": float(candidate_count / median),
    }


def _json_value(value):
    array = np.asarray(value)
    return array.item() if array.shape == () else array.tolist()


def run_benchmark(arguments: argparse.Namespace) -> dict[str, object]:
    _validate_arguments(arguments)
    space = _candidate_space(
        arguments.axes,
        arguments.axis_length,
        arguments.payload_size,
    )
    evaluator = _evaluator(arguments.objective_size, arguments.work)
    search = phx.optim.FiniteExhaustiveSearch(arguments.batch_size or None)
    candidate_count = space.size
    dtype_bytes = np.dtype(jnp.asarray(0.0).dtype).itemsize
    candidate_storage_bytes = sum(
        int(value.size * value.dtype.itemsize)
        for axis in jax.tree_util.tree_leaves(
            space.axes,
            is_leaf=lambda value: isinstance(value, phx.optim.FiniteAxis),
        )
        for value in jax.tree_util.tree_leaves(axis.values)
    )
    dense_candidate_bytes = (
        candidate_count * arguments.axes * arguments.payload_size * dtype_bytes
    )
    dense_objective_bytes = candidate_count * arguments.objective_size * dtype_bytes
    estimated_dense_bytes = dense_candidate_bytes + dense_objective_bytes

    def streaming_search():
        evidence = phx.optim.search_finite(
            evaluator,
            space,
            phx.optim.FiniteMinimum(),
            search=search,
        )
        return evidence.scores[0], evidence.flat_indices[0]

    streaming_executable, streaming_output, streaming_compile, streaming_samples = (
        _compile_and_time(streaming_search, arguments.repeat)
    )
    report: dict[str, object] = {
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "host": platform.platform(),
        "configuration": {
            "axis_sizes": list(space.product_shape),
            "axis_count": arguments.axes,
            "payload_size": arguments.payload_size,
            "objective_size": arguments.objective_size,
            "candidate_count": candidate_count,
            "requested_batch_size": arguments.batch_size or None,
            "effective_batch_size": search.effective_batch_size(candidate_count),
            "work": arguments.work,
            "repeat": arguments.repeat,
            "max_dense_bytes": arguments.max_dense_bytes,
        },
        "storage": {
            "factorized_candidate_bytes": candidate_storage_bytes,
            "estimated_dense_candidate_bytes": dense_candidate_bytes,
            "estimated_dense_objective_bytes": dense_objective_bytes,
            "estimated_dense_total_bytes": estimated_dense_bytes,
        },
        "streaming": {
            "compilation_seconds": streaming_compile,
            "timing": _timing_report(streaming_samples, candidate_count),
            "compiler_memory": _memory_report(streaming_executable),
            "minimum": _json_value(streaming_output[0]),
            "flat_index": _json_value(streaming_output[1]),
            "attempted_evaluations": candidate_count,
        },
    }

    if estimated_dense_bytes > arguments.max_dense_bytes:
        report["dense"] = {
            "executed": False,
            "reason": "estimated_dense_bytes_exceeds_limit",
        }
        return report

    def dense_search():
        points = space.take(jnp.arange(candidate_count, dtype=jnp.int64))
        scores, valid = jax.vmap(evaluator)(points)
        finite = valid & jnp.isfinite(scores)
        masked = jnp.where(finite, scores, jnp.inf)
        minimum = jnp.min(masked, axis=0)
        flat_index = jnp.argmin(masked, axis=0).astype(jnp.int64)
        return minimum, flat_index

    dense_executable, dense_output, dense_compile, dense_samples = _compile_and_time(
        dense_search,
        arguments.repeat,
    )
    if not np.array_equal(
        np.asarray(streaming_output[1]),
        np.asarray(dense_output[1]),
    ) or not np.allclose(
        np.asarray(streaming_output[0]),
        np.asarray(dense_output[0]),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError("Streaming search disagrees with the dense oracle.")
    report["dense"] = {
        "executed": True,
        "compilation_seconds": dense_compile,
        "timing": _timing_report(dense_samples, candidate_count),
        "compiler_memory": _memory_report(dense_executable),
        "minimum": _json_value(dense_output[0]),
        "flat_index": _json_value(dense_output[1]),
        "attempted_evaluations": candidate_count,
    }
    return report


def main() -> None:
    arguments = _parser().parse_args()
    print(json.dumps(run_benchmark(arguments), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
