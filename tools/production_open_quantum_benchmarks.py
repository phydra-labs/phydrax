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
    state = phx.tensor_network.product_mps(
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    )
    hamiltonian = phx.tensor_network.NearestNeighborHamiltonian(
        (jnp.zeros((4, 4), dtype=complex),),
        (2, 2),
        hamiltonian_id="benchmark-zero",
    )
    fermionic = phx.solver.damped_fermionic_mode(0.4, 0.25)
    process = phx.tensor_network.markov_process_tensor(
        (jnp.eye(4, dtype=complex),),
        jnp.asarray([[0.7 + 0j, 0j], [0j, 0.3 + 0j]]),
    )
    return {
        "backend": jax.default_backend(),
        "records": {
            "tebd_step": _benchmark(
                lambda value: phx.tensor_network.tebd_step(
                    value,
                    hamiltonian,
                    0.01,
                    maximum_bond_dimension=2,
                )[0].to_dense(),
                state,
                repeats,
            ),
            "fermionic_rhs": _benchmark(
                fermionic.rhs, fermionic.initial_state.covariance, repeats
            ),
            "process_causality": _benchmark(
                lambda value: (
                    phx.tensor_network.validate_process_comb_causality(
                        value
                    ).slot_residuals
                ),
                process,
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
