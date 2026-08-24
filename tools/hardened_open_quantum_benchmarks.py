#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from time import perf_counter

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


def _benchmark(function, argument, repeats):
    compiled = eqx.filter_jit(function)
    start = perf_counter()
    output = jax.block_until_ready(compiled(argument))
    first = perf_counter() - start
    start = perf_counter()
    for _ in range(repeats):
        output = jax.block_until_ready(compiled(argument))
    return {
        "compile_and_first_seconds": first,
        "steady_seconds": (perf_counter() - start) / repeats,
        "output_bytes": int(output.size * output.dtype.itemsize),
    }


def run_benchmarks(*, repeats=3):
    mps = phx.tensor_network.product_mps(
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    )
    expansion = phx.operators.quantum.underdamped_brownian_two_pole(1.0, 0.2, 0.1)
    hierarchy = phx.solver.HEOMHierarchy(expansion.rank, 1)
    scaled = phx.solver.ScaledHEOMTopology(hierarchy, expansion)
    return {
        "backend": jax.default_backend(),
        "records": {
            "mps_norm_environment": _benchmark(lambda state: state.norm(), mps, repeats),
            "scaled_heom_round_trip": _benchmark(
                lambda values: scaled.unscale(scaled.scale(values)),
                jnp.zeros((hierarchy.auxiliary_count, 2, 2), dtype=complex)
                .at[0]
                .set(0.5 * jnp.eye(2)),
                repeats,
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_benchmarks(repeats=1 if arguments.smoke else arguments.repeats),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
