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
import jax.scipy as jsp
from _runtime import (
    capture_environment,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)

import phydrax as phx


def _mode_case(cutoff: int, repeats: int) -> dict[str, object]:
    quantum = phx.operators.quantum
    basis = quantum.ChargeBasis(cutoff)
    parameters = quantum.TransmonParameters(0.23, 8.0, 7.0, external_phase=0.2)
    prepared = quantum.prepare_mode_reduction(
        quantum.transmon_mode_problem(parameters, basis),
        policy=quantum.ModeReductionPolicy(4),
    )

    def transition(charging_rate):
        refreshed = quantum.refresh_mode_reduction(
            prepared,
            quantum.transmon_mode_problem(
                quantum.TransmonParameters(
                    charging_rate,
                    8.0,
                    7.0,
                    external_phase=0.2,
                ),
                basis,
            ),
        )
        return refreshed.energies[1] - refreshed.energies[0]

    evaluate = eqx.filter_jit(jax.value_and_grad(transition))
    charging_rate = jnp.asarray(0.23)
    compiled, compilation = measure_lower_and_compile(
        lambda: evaluate.lower(charging_rate),
        lambda lowered: lowered.compile(),
    )
    result, execution = measure_repeated(
        lambda: compiled(charging_rate),
        warmup=1,
        repeats=repeats,
    )
    return {
        "kind": "mode-reduction-gradient",
        "charge_cutoff": cutoff,
        "raw_dimension": basis.dimension,
        "retained_dimension": prepared.plan.cost.retained_dimension,
        "logical_input_bytes": logical_array_bytes((prepared, charging_rate)),
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "execution": execution.to_milliseconds_dict(),
        "transition": float(result[0]),
        "gradient": float(result[1]),
        "eigen_residual": float(prepared.diagnostics.eigen_residual),
        "boundary_gap": float(prepared.diagnostics.boundary_gap),
        "valid": bool(prepared.diagnostics.valid),
    }


def _device(site_count: int):
    quantum = phx.operators.quantum
    solver = phx.solver
    topology = phx.graph.GraphIR(
        n_node=jnp.asarray([site_count]),
        n_edge=jnp.asarray([site_count - 1]),
        senders=jnp.arange(site_count - 1),
        receivers=jnp.arange(1, site_count),
    )
    basis = quantum.OscillatorBasis(4)
    reduction = quantum.ModeReductionPolicy(2)
    placements = tuple(
        solver.CircuitModePlacement(
            f"q{index}",
            "harmonic",
            basis,
            0,
            reduction,
        )
        for index in range(site_count)
    )
    interactions = tuple(
        solver.CircuitInteraction(
            (index, index + 1),
            ("phase", "phase"),
            0,
        )
        for index in range(site_count - 1)
    )
    spec = solver.CircuitQEDDeviceSpec(topology, placements, interactions)
    parameters = solver.CircuitQEDDeviceParameters(
        (quantum.HarmonicModeParameters(2.0),),
        interaction_strengths=jnp.asarray([0.05]),
    )
    return solver.prepare_circuit_qed_device(spec, parameters)


def _evolution_case(
    site_count: int,
    interval_count: int,
    differentiation: str,
    repeats: int,
) -> dict[str, object]:
    solver = phx.solver
    device = _device(site_count)
    grid = jnp.linspace(0.0, 0.5, interval_count + 1)
    coefficients = jnp.ones((interval_count, len(device.drift.terms)))
    schedule = solver.FixedGridLocalHamiltonian(device.drift, grid, coefficients)
    prepared = solver.prepare_local_hamiltonian_evolution(
        schedule,
        policy=solver.LocalHamiltonianEvolutionPolicy(
            order=2,
            differentiation=differentiation,
        ),
    )
    initial = (
        jnp.zeros((device.plan.layout.dimension,), dtype=jnp.complex128).at[0].set(1.0)
    )
    execute = eqx.filter_jit(solver.solve_local_hamiltonian_evolution)
    compiled, compilation = measure_lower_and_compile(
        lambda: execute.lower(prepared, initial),
        lambda lowered: lowered.compile(),
    )
    result, execution = measure_repeated(
        lambda: compiled(prepared, initial),
        warmup=1,
        repeats=repeats,
    )

    def population(scale):
        refreshed_schedule = solver.FixedGridLocalHamiltonian(
            device.drift,
            grid,
            scale * coefficients,
        )
        refreshed = solver.refresh_local_hamiltonian_evolution(
            prepared,
            refreshed_schedule,
        )
        state = solver.solve_local_hamiltonian_evolution(refreshed, initial).final_state
        return jnp.real(state[0] * jnp.conj(state[0]))

    gradient_evaluate = eqx.filter_jit(jax.value_and_grad(population))
    scale = jnp.asarray(1.0)
    gradient_compiled, gradient_compilation = measure_lower_and_compile(
        lambda: gradient_evaluate.lower(scale),
        lambda lowered: lowered.compile(),
    )
    gradient_result, gradient_execution = measure_repeated(
        lambda: gradient_compiled(scale),
        warmup=1,
        repeats=repeats,
    )
    dense = solver.materialize_local_hamiltonian(
        device.drift,
        policy=phx.linalg.MaterializationPolicy(
            max_entries=device.plan.cost.dense_entries,
            max_bytes=device.plan.cost.dense_bytes,
        ),
    )
    reference = jsp.linalg.expm(-0.5j * dense) @ initial
    difference = result.final_state - reference
    error = jnp.sqrt(jnp.sum(jnp.real(difference * jnp.conj(difference))))
    mpo = solver.lower_local_hamiltonian_to_mpo(device.drift)
    mpo_error = jnp.max(jnp.abs(mpo.operator.to_dense() - dense))
    return {
        "kind": "local-product-formula",
        "sites": site_count,
        "hilbert_dimension": device.plan.layout.dimension,
        "intervals": interval_count,
        "terms": len(device.drift.terms),
        "order": prepared.plan.policy.order,
        "differentiation": differentiation,
        "logical_input_bytes": logical_array_bytes((prepared, initial)),
        "state_bytes": prepared.plan.cost.state_bytes,
        "workspace_bytes": prepared.plan.cost.workspace_bytes,
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "execution": execution.to_milliseconds_dict(),
        "gradient_lowering_seconds": gradient_compilation.lowering_seconds,
        "gradient_compilation_seconds": gradient_compilation.compilation_seconds,
        "gradient_execution": gradient_execution.to_milliseconds_dict(),
        "population": float(gradient_result[0]),
        "population_gradient": float(gradient_result[1]),
        "dense_state_error": float(error),
        "mpo_dense_error": float(mpo_error),
        "maximum_mpo_bond_dimension": mpo.evidence.maximum_bond_dimension,
        "norm_residual": float(result.diagnostics.final_norm_residual),
        "maximum_local_unitarity_residual": float(
            result.diagnostics.maximum_local_unitarity_residual
        ),
        "valid": bool(result.successful & mpo.evidence.valid),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--charge-cutoffs", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--sites", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--intervals", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if (
        any(value < 1 for value in arguments.charge_cutoffs)
        or any(value < 2 for value in arguments.sites)
        or arguments.intervals < 1
        or arguments.repeats < 1
    ):
        raise ValueError("Benchmark dimensions, intervals, and repeats are invalid.")
    mode_cases = [
        _mode_case(cutoff, arguments.repeats) for cutoff in arguments.charge_cutoffs
    ]
    evolution_cases = [
        _evolution_case(
            sites,
            arguments.intervals,
            differentiation,
            arguments.repeats,
        )
        for sites in arguments.sites
        for differentiation in ("autodiff", "reversible-product-formula")
    ]
    gradient_comparisons = [
        {
            "sites": evolution_cases[index]["sites"],
            "absolute_gradient_difference": abs(
                evolution_cases[index]["population_gradient"]
                - evolution_cases[index + 1]["population_gradient"]
            ),
            "absolute_population_difference": abs(
                evolution_cases[index]["population"]
                - evolution_cases[index + 1]["population"]
            ),
        }
        for index in range(0, len(evolution_cases), 2)
    ]
    payload = {
        "environment": capture_environment().to_dict(),
        "mode_cases": mode_cases,
        "evolution_cases": evolution_cases,
        "gradient_comparisons": gradient_comparisons,
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
