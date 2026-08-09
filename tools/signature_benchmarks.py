#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from types import ModuleType
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


_Result = TypeVar("_Result")


@dataclass(frozen=True)
class BenchmarkResult:
    method: str
    length: int
    dimension: int
    order: int
    batch_size: int
    feature_size: int | None
    compile_forward_seconds: float | None
    forward_seconds: float
    compile_reverse_seconds: float | None
    reverse_seconds: float | None
    max_abs_error: float | None


def _integers(value: str, /) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(","))
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers.")
    return values


def _block_until_ready(value: Any, /) -> None:
    for leaf in jax.tree.leaves(value):
        leaf.block_until_ready()


def _measure(
    function: Callable[..., _Result],
    arguments: tuple[Any, ...],
    /,
    *,
    repeats: int,
) -> tuple[float, float, _Result]:
    compiled = jax.jit(function)
    start = time.perf_counter()
    value = compiled(*arguments)
    _block_until_ready(value)
    compile_seconds = time.perf_counter() - start
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = compiled(*arguments)
        _block_until_ready(value)
        samples.append(time.perf_counter() - start)
    return compile_seconds, float(jnp.median(jnp.asarray(samples))), value


def _measure_eager(
    function: Callable[..., _Result],
    arguments: tuple[Any, ...],
    /,
    *,
    repeats: int,
) -> tuple[float, _Result]:
    value = function(*arguments)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = function(*arguments)
        samples.append(time.perf_counter() - start)
    return float(np.median(np.asarray(samples))), value


def _paths(
    key: jax.Array,
    batch_size: int,
    length: int,
    dimension: int,
    /,
) -> jax.Array:
    increments = jax.random.normal(key, (batch_size, length - 1, dimension))
    increments = increments / jnp.sqrt(jnp.asarray(max(length - 1, 1), dtype=float))
    origins = jnp.zeros((batch_size, 1, dimension), dtype=increments.dtype)
    return jnp.concatenate((origins, jnp.cumsum(increments, axis=1)), axis=1)


def _feature_size(dimension: int, depth: int, /) -> int:
    return 1 + sum(dimension**degree for degree in range(1, depth + 1))


def _feature_benchmark(
    paths: jax.Array,
    depth: int,
    /,
    *,
    repeats: int,
) -> BenchmarkResult:
    dimension = int(paths.shape[-1])
    features = phx.stochastic.SignatureFeatures(
        dimension,
        depth,
        include_scalar=True,
    )
    forward = jax.vmap(features)
    reverse = jax.grad(lambda values: jnp.sum(forward(values)))
    compile_forward, forward_seconds, _ = _measure(forward, (paths,), repeats=repeats)
    compile_reverse, reverse_seconds, _ = _measure(reverse, (paths,), repeats=repeats)
    return BenchmarkResult(
        method="tensor-features",
        length=int(paths.shape[1]),
        dimension=dimension,
        order=depth,
        batch_size=int(paths.shape[0]),
        feature_size=features.output_size,
        compile_forward_seconds=compile_forward,
        forward_seconds=forward_seconds,
        compile_reverse_seconds=compile_reverse,
        reverse_seconds=reverse_seconds,
        max_abs_error=None,
    )


def _signax_benchmark(
    paths: jax.Array,
    depth: int,
    signax: ModuleType,
    /,
    *,
    repeats: int,
) -> BenchmarkResult:
    def forward(values):
        return signax.signature(values, depth=depth, flatten=True)

    reverse = jax.grad(lambda values: jnp.sum(forward(values)))
    compile_forward, forward_seconds, values = _measure(
        forward, (paths,), repeats=repeats
    )
    compile_reverse, reverse_seconds, _ = _measure(reverse, (paths,), repeats=repeats)
    exact = jax.vmap(
        phx.stochastic.SignatureFeatures(
            int(paths.shape[-1]),
            depth,
            include_scalar=False,
        )
    )(paths)
    return BenchmarkResult(
        method="signax",
        length=int(paths.shape[1]),
        dimension=int(paths.shape[-1]),
        order=depth,
        batch_size=int(paths.shape[0]),
        feature_size=int(exact.shape[-1]),
        compile_forward_seconds=compile_forward,
        forward_seconds=forward_seconds,
        compile_reverse_seconds=compile_reverse,
        reverse_seconds=reverse_seconds,
        max_abs_error=float(jnp.max(jnp.abs(values - exact))),
    )


def _iisignature_benchmark(
    paths: jax.Array,
    depth: int,
    iisignature: ModuleType,
    /,
    *,
    repeats: int,
) -> BenchmarkResult:
    host_paths = np.asarray(paths)
    forward_seconds, values = _measure_eager(
        lambda array: iisignature.sig(array, depth),
        (host_paths,),
        repeats=repeats,
    )
    exact = np.asarray(
        jax.vmap(
            phx.stochastic.SignatureFeatures(
                int(paths.shape[-1]),
                depth,
                include_scalar=False,
            )
        )(paths)
    )
    return BenchmarkResult(
        method="iisignature",
        length=int(paths.shape[1]),
        dimension=int(paths.shape[-1]),
        order=depth,
        batch_size=int(paths.shape[0]),
        feature_size=int(exact.shape[-1]),
        compile_forward_seconds=None,
        forward_seconds=forward_seconds,
        compile_reverse_seconds=None,
        reverse_seconds=None,
        max_abs_error=float(np.max(np.abs(values - exact))),
    )


def _pde_benchmark(
    left: jax.Array,
    right: jax.Array,
    order: int,
    /,
    *,
    repeats: int,
    max_feature_size: int,
) -> BenchmarkResult:
    dimension = int(left.shape[-1])
    kernel = phx.kernels.SignaturePDEKernel(
        phx.kernels.LinearKernel(),
        polynomial_order=order,
        pair_block_size=1,
    )
    forward = jax.vmap(kernel.pairwise)
    reverse = jax.grad(lambda values: jnp.sum(forward(values, right)))
    compile_forward, forward_seconds, values = _measure(
        forward, (left, right), repeats=repeats
    )
    compile_reverse, reverse_seconds, _ = _measure(reverse, (left,), repeats=repeats)
    feature_size = _feature_size(dimension, order)
    max_abs_error = None
    if feature_size <= max_feature_size:
        features = phx.stochastic.SignatureFeatures(
            dimension,
            order,
            include_scalar=True,
        )
        left_features = jax.vmap(features)(left)
        right_features = jax.vmap(features)(right)
        exact = jnp.sum(left_features * right_features, axis=1)
        max_abs_error = float(jnp.max(jnp.abs(values - exact)))
    return BenchmarkResult(
        method="signature-pde",
        length=int(left.shape[1]),
        dimension=dimension,
        order=order,
        batch_size=int(left.shape[0]),
        feature_size=feature_size,
        compile_forward_seconds=compile_forward,
        forward_seconds=forward_seconds,
        compile_reverse_seconds=compile_reverse,
        reverse_seconds=reverse_seconds,
        max_abs_error=max_abs_error,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark exact tensor features and the signature PDE kernel."
    )
    parser.add_argument("--lengths", type=_integers, default=(16, 64, 256))
    parser.add_argument("--dimensions", type=_integers, default=(2, 4, 8))
    parser.add_argument("--depths", type=_integers, default=(2, 4, 6))
    parser.add_argument("--orders", type=_integers, default=(3, 5, 7))
    parser.add_argument("--batch-sizes", type=_integers, default=(1, 32))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-feature-size", type=int, default=100_000)
    parser.add_argument("--platform", choices=("cpu", "gpu"), default=None)
    parser.add_argument("--quick", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.repeats <= 0:
        raise ValueError("repeats must be positive.")
    if arguments.max_feature_size <= 0:
        raise ValueError("max_feature_size must be positive.")
    if arguments.platform is not None:
        jax.config.update("jax_platform_name", arguments.platform)

    signax = (
        importlib.import_module("signax")
        if importlib.util.find_spec("signax") is not None
        else None
    )
    iisignature = (
        importlib.import_module("iisignature")
        if importlib.util.find_spec("iisignature") is not None
        else None
    )
    lengths = (16, 64) if arguments.quick else arguments.lengths
    dimensions = (2, 4) if arguments.quick else arguments.dimensions
    depths = (2, 4) if arguments.quick else arguments.depths
    orders = (3, 5) if arguments.quick else arguments.orders
    batch_sizes = (1, 4) if arguments.quick else arguments.batch_sizes
    print(
        json.dumps(
            {
                "backend": jax.default_backend(),
                "devices": [str(device) for device in jax.devices()],
                "max_feature_size": arguments.max_feature_size,
                "references": {
                    "iisignature": iisignature is not None,
                    "signax": signax is not None,
                    "sigkax": False,
                },
            },
            sort_keys=True,
        )
    )

    key = jax.random.key(2026)
    for length in lengths:
        for dimension in dimensions:
            for batch_size in batch_sizes:
                key, left_key, right_key = jax.random.split(key, 3)
                left = _paths(left_key, batch_size, length, dimension)
                right = _paths(right_key, batch_size, length, dimension)
                for depth in depths:
                    if _feature_size(dimension, depth) <= arguments.max_feature_size:
                        result = _feature_benchmark(
                            left,
                            depth,
                            repeats=arguments.repeats,
                        )
                        print(json.dumps(asdict(result), sort_keys=True))
                        if signax is not None:
                            print(
                                json.dumps(
                                    asdict(
                                        _signax_benchmark(
                                            left,
                                            depth,
                                            signax,
                                            repeats=arguments.repeats,
                                        )
                                    ),
                                    sort_keys=True,
                                )
                            )
                        if iisignature is not None:
                            print(
                                json.dumps(
                                    asdict(
                                        _iisignature_benchmark(
                                            left,
                                            depth,
                                            iisignature,
                                            repeats=arguments.repeats,
                                        )
                                    ),
                                    sort_keys=True,
                                )
                            )
                for order in orders:
                    result = _pde_benchmark(
                        left,
                        right,
                        order,
                        repeats=arguments.repeats,
                        max_feature_size=arguments.max_feature_size,
                    )
                    print(json.dumps(asdict(result), sort_keys=True))


if __name__ == "__main__":
    main()
