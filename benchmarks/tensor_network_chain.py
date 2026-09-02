#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from _runtime import (
    capture_environment,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)

import phydrax as phx


def _random_mps(key, sites: int, physical: int, bond: int):
    keys = jax.random.split(key, sites)
    tensors = []
    for site, local_key in enumerate(keys):
        left = 1 if site == 0 else bond
        right = 1 if site == sites - 1 else bond
        tensors.append(
            jax.random.normal(local_key, (left, physical, right), dtype=jnp.float64)
        )
    return phx.tensor_network.MatrixProductState(tuple(tensors)).normalized()


def _random_mpo(key, sites: int, physical: int, bond: int):
    keys = jax.random.split(key, sites)
    tensors = []
    for site, local_key in enumerate(keys):
        left = 1 if site == 0 else bond
        right = 1 if site == sites - 1 else bond
        tensors.append(
            jax.random.normal(
                local_key, (left, physical, physical, right), dtype=jnp.float64
            )
        )
    return phx.tensor_network.MatrixProductOperator(tuple(tensors))


def _case(sites: int, physical: int, bond: int, repeats: int):
    key = jax.random.key(20260901 + sites + bond)
    state = _random_mps(key, sites, physical, bond)
    operator = _random_mpo(jax.random.fold_in(key, 1), sites, physical, bond)
    expectation = eqx.filter_jit(phx.tensor_network.mps_mpo_expectation)
    compiled_expectation, expectation_compilation = measure_lower_and_compile(
        lambda: expectation.lower(state, operator),
        lambda lowered: lowered.compile(),
    )
    value, expectation_execution = measure_repeated(
        lambda: compiled_expectation(state, operator), warmup=1, repeats=repeats
    )

    apply = eqx.filter_jit(
        lambda op, vector: phx.tensor_network.apply_mpo(
            op,
            vector,
            maximum_bond_dimension=bond,
            normalize=False,
        )
    )
    compiled_apply, apply_compilation = measure_lower_and_compile(
        lambda: apply.lower(operator, state),
        lambda lowered: lowered.compile(),
    )
    applied, apply_execution = measure_repeated(
        lambda: compiled_apply(operator, state), warmup=1, repeats=repeats
    )

    return {
        "sites": sites,
        "physical_dimension": physical,
        "bond_dimension": bond,
        "logical_input_bytes": logical_array_bytes((state, operator)),
        "expectation": {
            "lowering_seconds": expectation_compilation.lowering_seconds,
            "compilation_seconds": expectation_compilation.compilation_seconds,
            "execution": expectation_execution.to_milliseconds_dict(),
            "finite": bool(jnp.isfinite(value)),
        },
        "apply": {
            "lowering_seconds": apply_compilation.lowering_seconds,
            "compilation_seconds": apply_compilation.compilation_seconds,
            "execution": apply_execution.to_milliseconds_dict(),
            "output_bytes": logical_array_bytes(applied),
            "discarded_weight": float(applied[1].accumulated_discarded_weight),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--physical-dimension", type=int, default=2)
    parser.add_argument("--bond-dimensions", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if (
        any(value < 2 for value in arguments.sites)
        or arguments.physical_dimension < 1
        or any(value < 1 for value in arguments.bond_dimensions)
        or arguments.repeats < 1
    ):
        raise ValueError("Benchmark sizes and repeats are outside supported bounds.")
    cases = [
        _case(sites, arguments.physical_dimension, bond, arguments.repeats)
        for sites in arguments.sites
        for bond in arguments.bond_dimensions
    ]
    payload = {"environment": capture_environment().to_dict(), "cases": cases}
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
