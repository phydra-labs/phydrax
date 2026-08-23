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
    steady = (perf_counter() - start) / repeats
    return {
        "compile_and_first_seconds": first,
        "steady_seconds": steady,
        "output_bytes": int(output.size * output.dtype.itemsize),
    }


def run_benchmarks(*, repeats=5):
    density = jnp.asarray([[0.6 + 0.0j, 0.1], [0.1, 0.4 + 0.0j]])
    tangent = jnp.asarray([[0.1, 0.05j], [-0.05j, -0.1]])
    bures = phx.metrix.BuresDensityManifold(2)
    fermat = phx.geometry.complex.fermat_hypersurface(2)
    point = jnp.asarray([1.0 + 0.0j, -1.0 + 0.0j, 0.0j]) / jnp.sqrt(2.0)
    patch = phx.geometry.complex.HypersurfacePatchGeometry(fermat)
    return {
        "backend": jax.default_backend(),
        "records": {
            "hermitian_spectrum": _benchmark(
                lambda value: phx.linalg.HermitianSpectrum(value).eigenvalues,
                density,
                repeats,
            ),
            "bures_metric": _benchmark(
                lambda value: bures.inner(density, value, value), tangent, repeats
            ),
            "hypersurface_patch": _benchmark(
                lambda value: (
                    patch.evaluate(value, chart_index=0, pivot_index=0).induced_metric
                ),
                point,
                repeats,
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    result = run_benchmarks(repeats=1 if arguments.smoke else arguments.repeats)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
