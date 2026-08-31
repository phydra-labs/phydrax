#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json
import platform

import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import measure_repeated, measure_synchronized


def _network(size):
    reference = phx.circuit.ElectricalWaveReference(50.0)
    through = phx.circuit.MatrixScatteringComponent(
        jnp.asarray([[0.0, 0.999], [0.999, 0.0]], dtype=jnp.complex128),
        (
            phx.circuit.WavePort("left", reference),
            phx.circuit.WavePort("right", reference),
        ),
        component_id="substrate-through",
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
        network_id=f"substrate-chain-{size}",
    )


def _rc_circuit():
    reference = phx.circuit.ElectricalWaveReference(50.0)
    source = phx.circuit.CircuitElement(
        phx.circuit.IndependentCurrentSourceLaw(1.0), element_id="source"
    )
    return phx.circuit.NodalCircuit(
        (
            phx.circuit.CircuitInstance(
                "resistor", phx.circuit.Resistor(1.0), ("n", "0")
            ),
            phx.circuit.CircuitInstance(
                "capacitor", phx.circuit.Capacitor(1.0), ("n", "0")
            ),
            phx.circuit.CircuitInstance("source", source, ("0", "n")),
        ),
        (phx.circuit.NodalPort("port", "n", "0", reference),),
        ground="0",
        circuit_id="substrate-rc",
    )


def _network_rows():
    rows = []
    for size in (8, 32, 128):
        network = _network(size)
        dense, dense_prepare = measure_synchronized(
            lambda network=network: phx.circuit.prepare_scattering_network(
                network, jnp.asarray(1.0)
            )
        )
        action, action_prepare = measure_synchronized(
            lambda network=network: phx.circuit.prepare_scattering_action(
                network, jnp.asarray(1.0)
            )
        )
        incident = jnp.asarray([[1.0], [0.0]], dtype=jnp.complex128)
        dense_result, dense_first = measure_synchronized(
            lambda dense=dense, incident=incident: phx.circuit.solve_scattering_network(
                dense, incident
            )
        )
        action_result, action_first = measure_synchronized(
            lambda action=action, incident=incident: phx.circuit.solve_scattering_action(
                action, incident
            )
        )
        _, action_warm = measure_repeated(
            lambda action=action, incident=incident: phx.circuit.solve_scattering_action(
                action, incident
            ),
            warmup=1,
            repeats=5,
        )
        rows.append(
            {
                "sections": size,
                "channels": action.plan.cost.channels,
                "dense_retained_bytes": dense.plan.cost.matrix_bytes
                + dense.plan.cost.factor_bytes,
                "action_retained_bytes": action.plan.cost.retained_bytes,
                "dense_prepare_seconds": dense_prepare,
                "action_prepare_seconds": action_prepare,
                "dense_first_seconds": dense_first,
                "action_first_seconds": action_first,
                "action_warm_seconds": action_warm.median_seconds,
                "dense_relative_residual": float(
                    dense_result.diagnostics.relative_residual
                ),
                "action_relative_residual": float(
                    action_result.diagnostics.relative_residual
                ),
                "successful": bool(dense_result.diagnostics.successful)
                and bool(action_result.diagnostics.successful),
            }
        )
    return rows


def _dynamic_row():
    prepared, prepare_seconds = measure_synchronized(
        lambda: phx.circuit.prepare_circuit_dae(_rc_circuit())
    )
    initial = prepared.initialize(node_voltages=jnp.asarray([0.0]))
    grid = phx.dynamics.TimeGrid(jnp.linspace(0.0, 1.0, 21), time_id="benchmark-rc")
    result, solve_seconds = measure_synchronized(
        lambda: phx.circuit.solve_circuit_dae(prepared, initial, grid)
    )
    operating, operating_prepare = measure_synchronized(
        lambda: phx.circuit.prepare_circuit_operating_point(prepared, jnp.asarray([0.5]))
    )
    root, operating_solve = measure_synchronized(
        lambda: phx.circuit.solve_circuit_operating_point(operating)
    )
    return {
        "state_size": prepared.plan.layout.size,
        "prepare_seconds": prepare_seconds,
        "transient_seconds": solve_seconds,
        "operating_prepare_seconds": operating_prepare,
        "operating_solve_seconds": operating_solve,
        "final_state": float(result.solution.states[-1, 0]),
        "operating_state": float(root.state[0]),
        "dae_residual": float(result.final_diagnostics.residual_norm),
        "operating_residual": float(root.circuit_diagnostics.residual_norm),
        "successful": bool(jnp.all(result.solution.valid))
        and bool(root.nonlinear.successful),
    }


def _macromodel_row():
    poles = jnp.asarray([-1.0 + 0.0j, -10.0 + 0.0j])
    model = phx.circuit.RationalMatrixModel(
        poles,
        jnp.asarray([[[2.0 + 0.0j]], [[0.5 + 0.0j]]]),
        jnp.asarray([[0.1 + 0.0j]]),
        jnp.zeros((1, 1), dtype=jnp.complex128),
    )
    frequencies = jnp.geomspace(1e-2, 1e2, 256)
    samples = model.evaluate_frequency(frequencies)
    fit, fit_seconds = measure_synchronized(
        lambda: phx.circuit.fit_rational_matrix(
            frequencies,
            samples,
            poles=poles,
            policy=phx.circuit.RationalFitPolicy(pole_count=2, residual_tolerance=1e-8),
        )
    )
    return {
        "frequencies": int(frequencies.size),
        "poles": int(fit.model.poles.size),
        "fit_seconds": fit_seconds,
        "relative_residual": float(fit.evidence.relative_residual),
        "maximum_error": float(fit.evidence.maximum_error),
        "successful": bool(fit.evidence.accepted),
    }


def main():
    print(
        json.dumps(
            {
                "kind": "circuit-substrate-benchmark",
                "python": platform.python_version(),
                "jax": jax.__version__,
                "backend": jax.default_backend(),
                "dtype": "float64/complex128",
                "network": _network_rows(),
                "dynamic": _dynamic_row(),
                "macromodel": _macromodel_row(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
