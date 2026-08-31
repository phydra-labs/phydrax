#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json
import platform

import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import measure_repeated, measure_synchronized


def _chain(size: int):
    reference = phx.circuit.ElectricalWaveReference(50.0)
    through = phx.circuit.MatrixScatteringComponent(
        jnp.asarray([[0.0, 0.999], [0.999, 0.0]], dtype=jnp.complex128),
        (
            phx.circuit.WavePort("left", reference),
            phx.circuit.WavePort("right", reference),
        ),
        component_id="benchmark-through",
    )
    instances = tuple(
        phx.circuit.ScatteringInstance(f"section-{index}", through)
        for index in range(size)
    )
    connections = tuple(
        phx.circuit.WaveConnection(
            phx.circuit.InstancePort(f"section-{index}", "right"),
            phx.circuit.InstancePort(f"section-{index + 1}", "left"),
        )
        for index in range(size - 1)
    )
    return phx.circuit.ScatteringNetwork(
        instances,
        connections,
        (
            phx.circuit.InstancePort("section-0", "left"),
            phx.circuit.InstancePort(f"section-{size - 1}", "right"),
        ),
        external_port_ids=("input", "output"),
        network_id=f"benchmark-chain-{size}",
    )


def main():
    rows = []
    for size in (8, 32, 128):
        network = _chain(size)
        prepared, prepare_seconds = measure_synchronized(
            lambda: phx.circuit.prepare_scattering_network(network, jnp.asarray(1.0))
        )
        incident = jnp.asarray([[1.0], [0.0]], dtype=jnp.complex128)
        result, first_seconds = measure_synchronized(
            lambda: phx.circuit.solve_scattering_network(prepared, incident)
        )
        _, warm_distribution = measure_repeated(
            lambda: phx.circuit.solve_scattering_network(prepared, incident),
            warmup=1,
            repeats=8,
        )
        rows.append(
            {
                "instances": size,
                "channels": int(prepared.plan.cost.channels),
                "matrix_bytes": int(prepared.plan.cost.matrix_bytes),
                "factor_bytes": int(prepared.plan.cost.factor_bytes),
                "prepare_seconds": prepare_seconds,
                "first_solve_seconds": first_seconds,
                "warm_solve_seconds": warm_distribution.median_seconds,
                "constitutive_residual": float(result.diagnostics.constitutive_residual),
                "connection_residual": float(result.diagnostics.connection_residual),
                "relative_residual": float(result.diagnostics.relative_residual),
                "successful": bool(result.diagnostics.successful),
            }
        )
    print(
        json.dumps(
            {
                "kind": "circuit-network-benchmark",
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
