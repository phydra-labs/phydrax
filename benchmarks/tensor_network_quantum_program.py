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


def _case(sites: int, bond: int, repeats: int):
    quantum = phx.operators.quantum
    layout = quantum.HilbertRegisterLayout(
        tuple(f"q{site}" for site in range(sites)), (2,) * sites
    )
    pauli_x = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
    program = quantum.QuantumProgram(
        layout,
        (quantum.LocalUnitaryOperation(pauli_x, (f"q{sites - 1}",)),),
        state_kind="state-vector",
    )
    state = phx.tensor_network.product_mps(
        jnp.tile(jnp.asarray([[1.0, 0.0]], dtype=jnp.complex128), (sites, 1))
    )
    policy = phx.solver.MPSQuantumProgramPolicy(maximum_bond_dimension=bond)
    plan = phx.solver.plan_mps_quantum_program(program, state, policy)
    prepared = phx.solver.prepare_mps_quantum_program(program, plan)
    execute = eqx.filter_jit(phx.solver.execute_mps_quantum_program)
    compiled, compilation = measure_lower_and_compile(
        lambda: execute.lower(prepared, state), lambda lowered: lowered.compile()
    )
    result, execution = measure_repeated(
        lambda: compiled(prepared, state), warmup=1, repeats=repeats
    )
    return {
        "sites": sites,
        "maximum_bond_dimension": bond,
        "maximum_window_elements": plan.cost.maximum_window_elements,
        "logical_input_bytes": logical_array_bytes((prepared, state)),
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "execution": execution.to_milliseconds_dict(),
        "status": int(result.diagnostics.status),
        "norm_residual": float(result.diagnostics.final_norm_residual),
        "discarded_weight": float(result.diagnostics.accumulated_discarded_weight),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--bond-dimensions", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if (
        any(value < 1 for value in arguments.sites)
        or any(value < 1 for value in arguments.bond_dimensions)
        or arguments.repeats < 1
    ):
        raise ValueError("Benchmark sizes and repeats must be positive.")
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": [
            _case(sites, bond, arguments.repeats)
            for sites in arguments.sites
            for bond in arguments.bond_dimensions
        ],
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
