#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json
import platform

import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import measure_repeated, measure_synchronized


def _ladder(size: int):
    instances = []
    for index in range(size):
        instances.append(
            phx.circuit.CircuitInstance(
                f"shunt-{index}",
                phx.circuit.Resistor(1_000.0),
                (f"n{index}", "0"),
            )
        )
        if index:
            instances.append(
                phx.circuit.CircuitInstance(
                    f"series-{index - 1}",
                    phx.circuit.Resistor(50.0),
                    (f"n{index - 1}", f"n{index}"),
                )
            )
    reference = phx.circuit.ElectricalWaveReference(50.0)
    return phx.circuit.NodalCircuit(
        tuple(instances),
        (
            phx.circuit.NodalPort("input", "n0", "0", reference),
            phx.circuit.NodalPort("output", f"n{size - 1}", "0", reference),
        ),
        ground="0",
        circuit_id=f"benchmark-ladder-{size}",
    )


def main():
    rows = []
    for assembly in ("dense", "sparse"):
        for size in (8, 32, 128):
            circuit = _ladder(size)
            policy = phx.circuit.MNASolvePolicy(
                assembly=assembly,
                residual_tolerance=1e-8,
            )
            prepared, prepare_seconds = measure_synchronized(
                lambda circuit=circuit, policy=policy: phx.circuit.prepare_mna(
                    circuit,
                    jnp.asarray(1.0),
                    policy,
                )
            )
            incident = jnp.asarray([[1.0], [0.0]], dtype=jnp.complex128)
            result, first_seconds = measure_synchronized(
                lambda prepared=prepared, incident=incident: phx.circuit.solve_mna(
                    prepared, incident
                )
            )
            _, warm_distribution = measure_repeated(
                lambda prepared=prepared, incident=incident: phx.circuit.solve_mna(
                    prepared, incident
                ),
                warmup=1,
                repeats=8,
            )
            rows.append(
                {
                    "assembly": assembly,
                    "sections": size,
                    "nodes": prepared.plan.cost.nodes,
                    "total_unknowns": prepared.plan.cost.total_unknowns,
                    "structural_entries": prepared.plan.cost.structural_entries,
                    "matrix_bytes": prepared.plan.cost.matrix_bytes,
                    "factor_bytes": prepared.plan.cost.factor_bytes,
                    "prepare_seconds": prepare_seconds,
                    "first_solve_seconds": first_seconds,
                    "warm_solve_seconds": warm_distribution.median_seconds,
                    "original_residual": float(result.diagnostics.original_residual),
                    "relative_residual": float(result.diagnostics.relative_residual),
                    "successful": bool(result.diagnostics.successful),
                }
            )
    print(
        json.dumps(
            {
                "kind": "circuit-mna-benchmark",
                "python": platform.python_version(),
                "jax": jax.__version__,
                "backend": jax.default_backend(),
                "dtype": "complex128",
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
