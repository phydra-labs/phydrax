#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp

import phydrax as phx


def _block(tree: Any, /) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def _array_bytes(tree: Any, /) -> int:
    return sum(
        int(leaf.nbytes)
        for leaf in jax.tree_util.tree_leaves(tree)
        if isinstance(leaf, jax.Array)
    )


def _checksum(tree: Any, /) -> float:
    return sum(
        float(jnp.sum(jnp.abs(leaf)))
        for leaf in jax.tree_util.tree_leaves(tree)
        if isinstance(leaf, jax.Array)
    )


def _benchmark(
    operation: Callable[..., Any],
    arguments: tuple[Any, ...],
    /,
    *,
    repeats: int,
) -> dict[str, float | int]:
    lowered = jax.jit(operation).lower(*arguments)
    started = time.perf_counter()
    executable = lowered.compile()
    compile_ms = 1e3 * (time.perf_counter() - started)

    result = executable(*arguments)
    _block(result)
    started = time.perf_counter()
    for _ in range(repeats):
        result = executable(*arguments)
        _block(result)
    execution_ms = 1e3 * (time.perf_counter() - started) / repeats
    return {
        "compile_ms": compile_ms,
        "execution_mean_ms": execution_ms,
        "output_bytes": _array_bytes(result),
        "checksum": _checksum(result),
    }


def run_benchmarks(*, batch_sizes: Sequence[int], repeats: int) -> dict[str, Any]:
    """Benchmark compiled values and first derivatives for every public family."""
    results: dict[str, Any] = {
        "configuration": {
            "batch_sizes": list(batch_sizes),
            "dtype": "float64/complex128",
            "repeats": repeats,
        },
        "batches": {},
    }
    for batch_size in batch_sizes:
        x = jnp.linspace(-8.0, 8.0, batch_size)
        positive = jnp.geomspace(0.1, 30.0, batch_size)
        unit = jnp.linspace(0.1, 0.9, batch_size)
        order = jnp.linspace(0.0, 20.0, batch_size)
        z = jax.lax.complex(x, jnp.full_like(x, 0.5))
        operations = {
            "faddeeva_values": (
                lambda real, complex_: (
                    phx.special.dawsn(real),
                    phx.special.wofz(complex_),
                    phx.special.voigt_profile(real, 0.8, 0.2),
                ),
                (x, z),
            ),
            "faddeeva_derivatives": (
                lambda real, complex_: (
                    jax.jvp(
                        phx.special.dawsn,
                        (real,),
                        (jnp.ones_like(real),),
                    )[1],
                    jax.jvp(
                        phx.special.wofz,
                        (complex_,),
                        (jnp.ones_like(complex_),),
                    )[1],
                    jax.jvp(
                        lambda values: phx.special.voigt_profile(values, 0.8, 0.2),
                        (real,),
                        (jnp.ones_like(real),),
                    )[1],
                ),
                (x, z),
            ),
            "carlson_values": (
                lambda values: (
                    phx.special.elliprc(values, values + 0.5),
                    phx.special.elliprf(values, values + 0.5, values + 1.0),
                    phx.special.elliprd(values, values + 0.5, values + 1.0),
                    phx.special.elliprj(values, values + 0.5, values + 1.0, values + 0.8),
                    phx.special.elliprg(values, values + 0.5, values + 1.0),
                ),
                (unit,),
            ),
            "carlson_derivatives": (
                lambda values: jax.jvp(
                    lambda arguments: (
                        phx.special.elliprc(arguments, arguments + 0.5),
                        phx.special.elliprf(arguments, arguments + 0.5, arguments + 1.0),
                        phx.special.elliprd(arguments, arguments + 0.5, arguments + 1.0),
                        phx.special.elliprj(
                            arguments,
                            arguments + 0.5,
                            arguments + 1.0,
                            arguments + 0.8,
                        ),
                        phx.special.elliprg(arguments, arguments + 0.5, arguments + 1.0),
                    ),
                    (values,),
                    (jnp.ones_like(values),),
                )[1],
                (unit,),
            ),
            "legendre_values": (
                lambda values, amplitude: (
                    phx.special.ellipk(values),
                    phx.special.ellipkm1(1.0 - values),
                    phx.special.ellipe(values),
                    phx.special.ellipkinc(amplitude, values),
                    phx.special.ellipeinc(amplitude, values),
                    phx.special.ellippi(0.2, values),
                    phx.special.ellippiinc(0.2, amplitude, values),
                ),
                (unit, 0.5 * x),
            ),
            "legendre_derivatives": (
                lambda values: jax.jvp(
                    lambda parameters: (
                        phx.special.ellipk(parameters),
                        phx.special.ellipe(parameters),
                        phx.special.ellipkinc(0.5, parameters),
                        phx.special.ellipeinc(0.5, parameters),
                        phx.special.ellippi(0.2, parameters),
                        phx.special.ellippiinc(0.2, 0.5, parameters),
                    ),
                    (values,),
                    (jnp.ones_like(values),),
                )[1],
                (unit,),
            ),
            "jacobi_values": (
                lambda amplitude, values: (
                    phx.special.ellipj(amplitude, values),
                    phx.special.ellipam(amplitude, values),
                ),
                (x, unit),
            ),
            "jacobi_derivatives": (
                lambda amplitude, values: jax.jvp(
                    phx.special.ellipj,
                    (amplitude, values),
                    (jnp.ones_like(amplitude), jnp.ones_like(values)),
                )[1],
                (x, unit),
            ),
            "airy_values": (
                lambda values: (
                    phx.special.airy(values),
                    phx.special.airye(values),
                ),
                (x,),
            ),
            "airy_derivatives": (
                lambda values: (
                    jax.jvp(
                        phx.special.airy,
                        (values,),
                        (jnp.ones_like(values),),
                    )[1],
                    jax.jvp(
                        phx.special.airye,
                        (values,),
                        (jnp.ones_like(values),),
                    )[1],
                ),
                (x,),
            ),
            "modified_bessel_values": (
                lambda orders, arguments: (
                    phx.special.iv(orders, arguments),
                    phx.special.ive(orders, arguments),
                    phx.special.kv(orders, arguments),
                    phx.special.kve(orders, arguments),
                ),
                (order, positive),
            ),
            "modified_bessel_derivatives": (
                lambda orders, arguments: jax.jvp(
                    lambda values: (
                        phx.special.iv(orders, values),
                        phx.special.ive(orders, values),
                        phx.special.kv(orders, values),
                        phx.special.kve(orders, values),
                    ),
                    (arguments,),
                    (jnp.ones_like(arguments),),
                )[1],
                (order, positive),
            ),
            "cylindrical_bessel_values": (
                lambda orders, arguments: (
                    phx.special.jv(orders, arguments),
                    phx.special.yv(orders, arguments),
                    phx.special.hankel1(orders, arguments),
                    phx.special.hankel2(orders, arguments),
                ),
                (order, positive),
            ),
            "cylindrical_bessel_derivatives": (
                lambda orders, arguments: jax.jvp(
                    lambda values: (
                        phx.special.jv(orders, values),
                        phx.special.yv(orders, values),
                        phx.special.hankel1(orders, values),
                        phx.special.hankel2(orders, values),
                    ),
                    (arguments,),
                    (jnp.ones_like(arguments),),
                )[1],
                (order, positive),
            ),
        }
        results["batches"][str(batch_size)] = {
            name: _benchmark(operation, arguments, repeats=repeats)
            for name, (operation, arguments) in operations.items()
        }
    return results


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark JAX compilation and execution of phydrax.special."
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 1_024, 65_536],
    )
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args(argv)
    if min(*args.batch_sizes, args.repeats) <= 0:
        parser.error("batch sizes and repeats must be positive")
    print(
        json.dumps(
            run_benchmarks(batch_sizes=args.batch_sizes, repeats=args.repeats),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
