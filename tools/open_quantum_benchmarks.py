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


def run_benchmarks(*, repeats=5):
    gaussian = phx.solver.damped_thermal_oscillator(0.4, 1.0)
    fock = phx.operators.quantum.BosonicFockSpace((16,))
    state = jnp.zeros((16,), dtype=complex).at[3].set(1.0)
    heom = phx.solver.drude_lorentz_qubit_heom(
        0.05,
        1.0,
        jnp.asarray([[0.6 + 0j, 0j], [0j, 0.4 + 0j]]),
        depth=2,
    )
    return {
        "backend": jax.default_backend(),
        "records": {
            "gaussian_rhs": _benchmark(
                lambda covariance: gaussian.rhs(gaussian.initial_state.mean, covariance)[
                    1
                ],
                gaussian.initial_state.covariance,
                repeats,
            ),
            "fock_annihilation": _benchmark(
                lambda vector: fock.annihilate(vector, 0), state, repeats
            ),
            "heom_rhs": _benchmark(heom.rhs, heom.initial_state, repeats),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
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
