from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from _runtime import capture_environment, measure_lower_and_compile, measure_repeated

import phydrax as phx


def _geometry(cell_count):
    rho = np.linspace(0.0, 0.95, cell_count + 1)
    theta = 2.0 * np.pi * np.arange(16) / 16
    contours = np.empty((cell_count + 1, 16, 2))
    for index, radius in enumerate(0.8 * rho):
        contours[index, :, 0] = 2.0 + radius * np.cos(theta)
        contours[index, :, 1] = radius * np.sin(theta)
    volume = 2.0 * np.pi**2 * 2.0 * (0.8 * rho) ** 2
    area = 4.0 * np.pi**2 * 2.0 * 0.8 * rho
    return phx.applications.tokamak.FluxSurfaceGeometry(
        rho,
        contours,
        volume,
        area,
        np.full_like(rho, 2.0),
        0.8 * rho,
        1.0 + rho,
        "benchmark-equilibrium",
    )


def _activation(nuclide_count):
    payload = b"synthetic-activation-benchmark"
    reference = phx.qualification.ReferenceArtifactManifest(
        "synthetic-activation-benchmark",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="synthetic",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="public",
        nondimensionalization={"identity": 1.0},
        uncertainty={"analytic": 0.0},
        lineage_ids=("synthetic-generator",),
    )
    data = phx.nuclear.NuclearDataProvenance(
        reference,
        "synthetic://activation-benchmark",
        "synthetic",
        "current",
        "isomer-chain",
    )
    nuclides = tuple(phx.nuclear.NuclideKey(20, 40, i) for i in range(nuclide_count))
    groups = phx.nuclear.EnergyGroupStructure(
        [0.0, 1.0], phx.units.MEGAELECTRONVOLT, source_id="benchmark-groups"
    )
    transitions = tuple(
        phx.nuclear.InventoryTransition(
            f"isomer-{index}",
            nuclides[index],
            ((nuclides[index - 1], 1.0),),
            0.01 + 0.001 * index,
            np.asarray([0.0]),
            1.0e-14,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            data,
        )
        for index in range(1, nuclide_count)
    )
    return phx.nuclear.ActivationNetworkPlan(
        nuclides, groups, transitions, error_tolerance=1.0e-9
    ).prepare()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=64)
    parser.add_argument("--nuclides", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cells < 2 or args.nuclides < 2 or args.warmup < 0 or args.repeats < 1:
        raise ValueError("Benchmark sizes and repetition counts are invalid.")

    transport = phx.applications.tokamak.TokamakCoreTransportPlan(
        _geometry(args.cells), 1.0
    ).prepare()
    state = phx.applications.tokamak.TokamakCoreState(
        jnp.linspace(1.0e20, 5.0e19, args.cells),
        jnp.linspace(2.0e-15, 1.0e-15, args.cells),
        jnp.linspace(2.5e-15, 1.2e-15, args.cells),
    )
    conductance = jnp.zeros((args.cells + 1,)).at[1:-1].set(0.1)
    coefficients = phx.applications.tokamak.TokamakTransportCoefficients(
        conductance, conductance, conductance
    )
    sources = phx.applications.tokamak.TokamakTransportSources(
        jnp.zeros(args.cells), jnp.zeros(args.cells), jnp.zeros(args.cells)
    )

    def transport_step(density_scale):
        scaled = phx.applications.tokamak.TokamakCoreState(
            density_scale * state.electron_density_m3,
            state.electron_thermal_energy_j,
            state.ion_thermal_energy_j,
        )
        result = transport.step(scaled, 0.01, coefficients, sources)
        gradient = jax.grad(
            lambda value: jnp.sum(
                transport.step(
                    phx.applications.tokamak.TokamakCoreState(
                        value * state.electron_density_m3,
                        state.electron_thermal_energy_j,
                        state.ion_thermal_energy_j,
                    ),
                    0.01,
                    coefficients,
                    sources,
                ).accepted_state.electron_density_m3
            )
        )(density_scale)
        relative_balance = jnp.max(
            jnp.abs(result.ledger.closure_residual)
            / jnp.maximum(1.0, jnp.abs(result.ledger.initial_totals))
        )
        return (
            result.accepted_state.electron_density_m3,
            relative_balance,
            gradient,
            result.successful,
        )

    compiled_transport, transport_compilation = measure_lower_and_compile(
        lambda: jax.jit(transport_step).lower(jnp.asarray(1.0)),
        lambda lowered: lowered.compile(),
    )
    transport_result, transport_execution = measure_repeated(
        lambda: compiled_transport(jnp.asarray(1.0)),
        warmup=args.warmup,
        repeats=args.repeats,
    )

    activation = _activation(args.nuclides)
    inventory = activation.inventory(
        jnp.concatenate((jnp.zeros(args.nuclides - 1), jnp.ones(1)))
    )

    def activation_step(duration):
        result = activation.step(inventory, jnp.asarray([0.0]), duration)
        return result.accepted.amounts_mol, result.exponential_error, result.successful

    compiled_activation, activation_compilation = measure_lower_and_compile(
        lambda: jax.jit(activation_step).lower(jnp.asarray(1.0)),
        lambda lowered: lowered.compile(),
    )
    activation_result, activation_execution = measure_repeated(
        lambda: compiled_activation(jnp.asarray(1.0)),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "radial_cells": args.cells,
            "activation_nuclides": args.nuclides,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "transport": {
            "lowering_seconds": transport_compilation.lowering_seconds,
            "compilation_seconds": transport_compilation.compilation_seconds,
            "execution": transport_execution.to_seconds_dict(),
            "maximum_relative_balance_residual": float(transport_result[1]),
            "gradient": float(transport_result[2]),
            "successful": bool(transport_result[3]),
        },
        "activation": {
            "lowering_seconds": activation_compilation.lowering_seconds,
            "compilation_seconds": activation_compilation.compilation_seconds,
            "execution": activation_execution.to_seconds_dict(),
            "matrix_function_error": float(activation_result[1]),
            "successful": bool(activation_result[2]),
        },
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
