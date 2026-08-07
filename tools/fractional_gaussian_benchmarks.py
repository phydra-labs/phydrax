#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _block(tree: Any, /) -> Any:
    return jax.tree.map(jax.block_until_ready, tree)


def _array_bytes(tree: Any, /) -> int:
    return sum(
        int(leaf.size * leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(tree)
        if isinstance(leaf, jax.Array)
    )


def _benchmark_method(
    method: phx.stochastic.FractionalGaussianSamplingMethod,
    /,
    *,
    num_times: int,
    num_paths: int,
    repeats: int,
) -> dict[str, float | int | str]:
    process = phx.stochastic.FractionalGaussianProcess(
        0.7,
        0.8,
        process_id="fractional-gaussian-benchmark",
    )
    grid = jnp.linspace(0.0, 1.0, num_times)

    started = time.perf_counter()
    realization = phx.stochastic.FractionalGaussianRealization(
        process,
        jr.key(2026),
        grid,
        sample_shape=(num_paths,),
        method=method,
    )
    _block(realization)
    setup_ms = 1e3 * (time.perf_counter() - started)

    compiled = eqx.filter_jit(lambda value: value.values)
    started = time.perf_counter()
    values = _block(compiled(realization))
    compile_and_first_ms = 1e3 * (time.perf_counter() - started)

    started = time.perf_counter()
    for _ in range(repeats):
        values = _block(compiled(realization))
    steady_ms = 1e3 * (time.perf_counter() - started) / repeats

    empirical_terminal_variance = jnp.var(values[..., -1, 0], ddof=1)
    exact_terminal_variance = process.scale[0] ** 2
    terminal_variance_relative_error = float(
        jnp.abs(empirical_terminal_variance - exact_terminal_variance)
        / exact_terminal_variance
    )
    return {
        "requested_method": method,
        "sampling_method": realization.sampling_method,
        "sampling_provenance": realization.sampling_provenance,
        "num_times": num_times,
        "num_paths": num_paths,
        "setup_ms": setup_ms,
        "compile_and_first_ms": compile_and_first_ms,
        "steady_ms": steady_ms,
        "realization_bytes": _array_bytes(realization),
        "output_bytes": _array_bytes(values),
        "terminal_variance_relative_error": terminal_variance_relative_error,
    }


def run_benchmarks(
    *,
    sizes: Sequence[int] = (129, 257, 513),
    num_paths: int = 128,
    repeats: int = 5,
) -> dict[str, Any]:
    """Compare dense and Davies–Harte setup, storage, and sampled-path costs."""
    resolved_sizes = tuple(int(size) for size in sizes)
    if not resolved_sizes or any(size < 2 for size in resolved_sizes):
        raise ValueError("sizes must contain integers of at least two.")
    if int(num_paths) < 2:
        raise ValueError("num_paths must be at least two.")
    if int(repeats) < 1:
        raise ValueError("repeats must be at least one.")

    records = []
    for num_times in resolved_sizes:
        for method in ("dense", "davies-harte"):
            records.append(
                _benchmark_method(
                    method,
                    num_times=num_times,
                    num_paths=int(num_paths),
                    repeats=int(repeats),
                )
            )
    return {
        "benchmark": "fractional-gaussian-sampling",
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark dense and Davies-Harte fractional Gaussian sampling."
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[129, 257, 513])
    parser.add_argument("--num-paths", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_benchmarks(
                sizes=arguments.sizes,
                num_paths=arguments.num_paths,
                repeats=arguments.repeats,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["run_benchmarks"]
