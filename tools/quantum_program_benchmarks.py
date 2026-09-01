#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json
import platform

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import measure_repeated, measure_synchronized


Q = phx.operators.quantum
X = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)


def _state_vector_rows():
    rows = []
    for wire_count in (4, 8, 10):
        wire_ids = tuple(f"q{index}" for index in range(wire_count))
        layout = Q.HilbertRegisterLayout(wire_ids, (2,) * wire_count)
        state = jnp.zeros((layout.dimension,), dtype=jnp.complex128).at[0].set(1.0)
        program = Q.QuantumProgram(
            layout,
            (Q.LocalUnitaryOperation(X, (wire_ids[-1],)),),
            state_kind="state-vector",
        )
        prepared, prepare_seconds = measure_synchronized(
            lambda program=program: phx.solver.prepare_dense_quantum_program(program)
        )
        execute = eqx.filter_jit(phx.solver.execute_dense_quantum_program)
        result, first_seconds = measure_synchronized(
            lambda prepared=prepared, state=state, execute=execute: execute(
                prepared, state
            )
        )
        _, warm = measure_repeated(
            lambda prepared=prepared, state=state, execute=execute: execute(
                prepared, state
            ),
            warmup=1,
            repeats=5,
        )
        global_operator = jnp.kron(
            jnp.eye(layout.dimension // 2, dtype=jnp.complex128), X
        )
        expected = global_operator @ state
        rows.append(
            {
                "wire_count": wire_count,
                "dimension": layout.dimension,
                "prepare_seconds": prepare_seconds,
                "first_seconds": first_seconds,
                "warm_seconds": warm.median_seconds,
                "state_bytes": prepared.plan.cost.state_bytes_per_case,
                "operation_bytes": prepared.plan.cost.operation_bytes,
                "workspace_bytes": prepared.plan.cost.workspace_bytes_per_case,
                "promoted_operator_bytes": int(global_operator.nbytes),
                "maximum_error": float(jnp.max(jnp.abs(result.final_state - expected))),
                "successful": bool(result.diagnostics.successful),
            }
        )
    return rows


def _density_row():
    wire_ids = tuple(f"q{index}" for index in range(6))
    layout = Q.HilbertRegisterLayout(wire_ids, (2,) * len(wire_ids))
    gamma = jnp.asarray(0.2)
    kraus = jnp.stack(
        (
            jnp.asarray(
                [[1.0, 0.0], [0.0, jnp.sqrt(1.0 - gamma)]],
                dtype=jnp.complex128,
            ),
            jnp.asarray(
                [[0.0, jnp.sqrt(gamma)], [0.0, 0.0]],
                dtype=jnp.complex128,
            ),
        )
    )
    program = Q.QuantumProgram(
        layout,
        (Q.LocalKrausChannelOperation(kraus, (wire_ids[0],)),),
        state_kind="density-matrix",
    )
    prepared, prepare_seconds = measure_synchronized(
        lambda: phx.solver.prepare_dense_quantum_program(program)
    )
    ket = (
        jnp.zeros((layout.dimension,), dtype=jnp.complex128)
        .at[layout.dimension // 2]
        .set(1.0)
    )
    density = jnp.outer(ket, jnp.conj(ket))
    execute = eqx.filter_jit(phx.solver.execute_dense_quantum_program)
    result, first_seconds = measure_synchronized(lambda: execute(prepared, density))
    _, warm = measure_repeated(lambda: execute(prepared, density), warmup=1, repeats=5)
    return {
        "wire_count": len(wire_ids),
        "dimension": layout.dimension,
        "kraus_capacity": int(kraus.shape[0]),
        "prepare_seconds": prepare_seconds,
        "first_seconds": first_seconds,
        "warm_seconds": warm.median_seconds,
        "state_bytes": prepared.plan.cost.state_bytes_per_case,
        "operation_bytes": prepared.plan.cost.operation_bytes,
        "workspace_bytes": prepared.plan.cost.workspace_bytes_per_case,
        "trace_residual": float(result.diagnostics.final_trace_residual),
        "minimum_eigenvalue": float(result.diagnostics.final_minimum_eigenvalue),
        "successful": bool(result.diagnostics.successful),
    }


def main():
    print(
        json.dumps(
            {
                "kind": "quantum-program-benchmark",
                "python": platform.python_version(),
                "jax": jax.__version__,
                "backend": jax.default_backend(),
                "dtype": "complex128",
                "state_vector": _state_vector_rows(),
                "density": _density_row(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
