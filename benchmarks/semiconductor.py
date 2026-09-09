#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Classical continuum and selected-source coherent-quantum benchmark.

Run from the repository root with PYTHONPATH=.:benchmarks and JAX_ENABLE_X64=1.
Timings separate preparation, physical solves/integration, lowering/compilation,
and warm residual/JVP or spectral-source execution. Failed physical evidence is
never timed as a successful result. No schema or external-reference claim.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from _runtime import capture_environment, measure_lower_and_compile, measure_repeated

from phydrax.applications.semiconductor import (
    bipolar_transistor,
    mos_capacitor,
    pn_junction,
    PreparedSemiconductorDevice,
    quantum,
)


def run_quantum_case(nodes: int, warmup: int, repeats: int) -> dict:
    charge = 1.602176634e-19
    start = time.perf_counter()
    resources = quantum.QuantumResources(
        max_nodes=max(256, nodes),
        max_evaluations=20_000,
        max_intervals=256,
        workspace_bytes=256 * 1024 * 1024,
    )
    hamiltonian = quantum.ChainHamiltonian(
        np.full(nodes, 2 * charge),
        np.full(nodes - 1, -charge),
        np.full(nodes, 1e-27),
        energy_reference="synthetic benchmark datum",
        resources=resources,
    )

    def lead(chemical_potential):
        return quantum.SemiInfiniteLead(
            2 * charge,
            -charge,
            -charge,
            chemical_potential * charge,
            300.0,
            energy_reference="synthetic benchmark datum",
        )

    device = quantum.CoherentDevice(
        hamiltonian,
        lead(2.05),
        lead(1.95),
        transverse=quantum.TransverseModes((0.0,), (2.0,)),
    )
    preparation = time.perf_counter() - start

    def solve():
        result = quantum.integrate_coherent(
            device,
            tolerance=2e-4,
            spectral_tolerance=5e-3,
            initial_panels=8,
            max_refinements=2,
        )
        if not bool(result.successful):
            raise RuntimeError("quantum: coherent integration did not qualify")
        return result

    result, timing = measure_repeated(solve, warmup=warmup, repeats=repeats)
    fixed = jax.jit(lambda energy: device.spectral(energy).transmission)
    compiled, compilation = measure_lower_and_compile(
        lambda: fixed.lower(2 * charge),
        lambda lowered: lowered.compile(),
    )
    transmission, execution = measure_repeated(
        lambda: compiled(2 * charge), warmup=warmup, repeats=repeats
    )
    return {
        "case": "quantum",
        "nodes": nodes,
        "edges": nodes - 1,
        "unknowns": nodes,
        "preparation_seconds": preparation,
        "coherent_integration": timing.to_seconds_dict(),
        "spectral_compilation": {
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
        },
        "spectral_execution": execution.to_seconds_dict(),
        "physics": {
            "successful": bool(result.successful),
            "transmission_at_band_center": float(transmission),
            "terminal_current_A": [float(value) for value in result.terminal_currents],
            "current_conservation_error": float(
                result.evidence.current_conservation_error
            ),
            "spectral_sum_error": float(result.evidence.spectral_sum_error),
            "refinement_error": float(result.evidence.refinement_error),
            "quadrature_evaluations": int(result.evidence.evaluations),
            "empirical_qualified": result.evidence.empirical_qualified,
        },
    }


def run_case(case: str, nodes: int, warmup: int, repeats: int) -> dict:
    if case == "quantum":
        return run_quantum_case(nodes, warmup, repeats)
    start = time.perf_counter()
    if case == "pn":
        plan = pn_junction(nodes)
        voltage = jnp.asarray([0.05, 0.0])
    elif case == "mos":
        plan = mos_capacitor(nodes)
        voltage = jnp.asarray([0.02, 0.0])
    else:
        plan = bipolar_transistor(max(nodes, 13), 5)
        voltage = jnp.asarray([0.0, 0.05, 0.1])
    prepared = PreparedSemiconductorDevice(plan)
    preparation = time.perf_counter() - start
    start = time.perf_counter()
    equilibrium = prepared.equilibrium()
    jax.block_until_ready(equilibrium.coordinates)
    equilibrium_seconds = time.perf_counter() - start
    if not bool(equilibrium.successful):
        raise RuntimeError(f"{case}: equilibrium did not converge")

    def biased_solve():
        sweep = prepared.sweep(
            jnp.stack((jnp.zeros_like(voltage), voltage)), initial=equilibrium.coordinates
        )
        if not bool(sweep.successful):
            raise RuntimeError(f"{case}: bias continuation did not reach its target")
        return sweep.points[-1]

    point, solve_timing = measure_repeated(
        biased_solve,
        warmup=warmup,
        repeats=repeats,
    )
    if not bool(point.successful):
        raise RuntimeError(f"{case}: biased operating point did not converge")

    def residual_and_jvp(u, v, direction):
        return jax.jvp(lambda state: prepared.residual(state, v), (u,), (direction,))

    direction = jnp.ones_like(point.coordinates)
    function = jax.jit(residual_and_jvp)
    compiled, compilation = measure_lower_and_compile(
        lambda: function.lower(point.coordinates, voltage, direction),
        lambda lowered: lowered.compile(),
    )
    (residual, tangent), execution = measure_repeated(
        lambda: compiled(point.coordinates, voltage, direction),
        warmup=warmup,
        repeats=repeats,
    )
    n, p = prepared.densities(point.coordinates)
    physics = {
        "successful": bool(point.successful),
        "scaled_residual_rms": float(point.evidence.scaled_residual_norm),
        "maximum_scaled_residual": float(
            prepared.time_scale * jnp.max(jnp.abs(residual))
        ),
        "terminal_current_A": [float(x) for x in point.terminal_currents],
        "terminal_charge_C": [float(x) for x in point.terminal_charges],
        "terminal_kcl_defect_A": float(jnp.abs(jnp.sum(point.terminal_currents))),
        "minimum_semiconductor_electron_density_m3": float(
            jnp.min(jnp.where(plan.semiconductor_mask, n, jnp.inf))
        ),
        "minimum_semiconductor_hole_density_m3": float(
            jnp.min(jnp.where(plan.semiconductor_mask, p, jnp.inf))
        ),
        "jvp_finite": bool(jnp.all(jnp.isfinite(tangent))),
    }
    if case == "pn":
        vt = float(plan.thermal_voltage)
        neutrality_voltage = vt * (
            float(
                jnp.arcsinh(
                    (plan.donor_density[-1] - plan.acceptor_density[-1])
                    / (2 * plan.intrinsic_density[-1])
                )
            )
            - float(
                jnp.arcsinh(
                    (plan.donor_density[0] - plan.acceptor_density[0])
                    / (2 * plan.intrinsic_density[0])
                )
            )
        )
        solved_voltage = vt * float(
            equilibrium.coordinates[-1, 0] - equilibrium.coordinates[0, 0]
        )
        physics["equilibrium_builtin_voltage_V"] = solved_voltage
        physics["contact_neutrality_builtin_voltage_V"] = neutrality_voltage
    return {
        "case": case,
        "nodes": int(plan.support.positions.shape[0]),
        "edges": int(plan.support.tail.shape[0]),
        "unknowns": int(point.coordinates.size),
        "preparation_seconds": preparation,
        "equilibrium_seconds": equilibrium_seconds,
        "biased_solve": solve_timing.to_seconds_dict(),
        "residual_jvp_compilation": {
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
        },
        "residual_jvp_execution": execution.to_seconds_dict(),
        "physics": physics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("pn", "mos", "bjt", "quantum", "all"),
        default="pn",
    )
    parser.add_argument("--nodes", type=int, nargs="+", default=[21, 41])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.nodes) < 5 or args.warmup < 0 or args.repeats < 1:
        raise ValueError(
            "Use at least five nodes, nonnegative warmup, and positive repeats"
        )
    if not jax.config.x64_enabled:
        raise ValueError(
            "This benchmark requires JAX_ENABLE_X64=1 for matched physical accuracy"
        )
    cases = ("pn", "mos", "bjt", "quantum") if args.case == "all" else (args.case,)
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "nodes": args.nodes,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "runs": [
            run_case(case, nodes, args.warmup, args.repeats)
            for case in cases
            for nodes in args.nodes
        ],
    }
    encoded = json.dumps(payload, indent=2, allow_nan=False)
    if args.output is None:
        print(encoded, flush=True)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
