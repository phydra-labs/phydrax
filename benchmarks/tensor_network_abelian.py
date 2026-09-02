#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
from _runtime import (
    capture_environment,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)

import phydrax as phx


def _state():
    tn = phx.tensor_network
    group = tn.AbelianGroup((None,))
    physical = tn.AbelianLeg(group, ((0,), (1,)), (1, 1), orientation=1)
    left = tn.AbelianLeg(group, ((0,),), (1,), orientation=1)
    middle_out = tn.AbelianLeg(group, ((0,), (1,)), (1, 1), orientation=-1)
    middle_in = middle_out.dual()
    right = tn.AbelianLeg(group, ((1,),), (1,), orientation=-1)
    first_layout = tn.AbelianTensorLayout((left, physical, middle_out))
    second_layout = tn.AbelianTensorLayout((middle_in, physical, right))
    root_two = jnp.sqrt(2.0)
    first = jnp.zeros((1, 2, 2), dtype=jnp.complex128)
    first = first.at[0, 0, 0].set(1.0 / root_two)
    first = first.at[0, 1, 1].set(1.0 / root_two)
    second = jnp.zeros((2, 2, 1), dtype=jnp.complex128)
    second = second.at[0, 1, 0].set(1.0)
    second = second.at[1, 0, 0].set(1.0)
    return tn.AbelianMatrixProductState(
        (
            tn.AbelianTensor.from_dense(first_layout, first),
            tn.AbelianTensor.from_dense(second_layout, second),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        raise ValueError("repeats must be positive.")
    state = _state()
    gate = jnp.eye(4, dtype=jnp.complex128).reshape((2, 2, 2, 2))
    apply = eqx.filter_jit(
        lambda value, operation: phx.tensor_network.apply_abelian_two_site_gate(
            value,
            0,
            operation,
            maximum_bond_dimension=1,
            normalize=True,
        )
    )
    compiled, compilation = measure_lower_and_compile(
        lambda: apply.lower(state, gate), lambda lowered: lowered.compile()
    )
    result, execution = measure_repeated(
        lambda: compiled(state, gate), warmup=1, repeats=arguments.repeats
    )
    payload = {
        "environment": capture_environment().to_dict(),
        "logical_input_bytes": logical_array_bytes((state, gate)),
        "logical_output_bytes": logical_array_bytes(result),
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "execution": execution.to_milliseconds_dict(),
        "discarded_weight": float(result[1].discarded_weight),
        "dense_residual": float(
            jnp.linalg.norm(
                result[0].to_dense()
                - jnp.asarray([0.0, 1.0, 0.0, 0.0], dtype=jnp.complex128)
            )
        ),
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
