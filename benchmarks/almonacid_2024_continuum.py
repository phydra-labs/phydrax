# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Benchmark full native Q2/DGPM1/DGPM1 transient solves, not material-only calls."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from phydrax.applications.skeletal_muscle.continuum import almonacid_2024_repository_case


def run(inputs, refinement, steps, repetitions, sensitivity):
    start = time.perf_counter()
    plan, parameters, history, dt, _ = almonacid_2024_repository_case(
        inputs, refinement=refinement
    )
    initial = plan.prepare(parameters)
    prepare_seconds = time.perf_counter() - start
    propose = eqx.filter_jit(lambda model, control: model.propose(control))
    start = time.perf_counter()
    first = propose(initial, history.sample(dt))
    jax.block_until_ready(first)
    first_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(repetitions):
        warm = propose(initial, history.sample(dt))
        jax.block_until_ready(warm)
    steady_seconds = (time.perf_counter() - start) / repetitions
    prepared = initial
    rows = []
    start = time.perf_counter()
    for step in range(1, steps + 1):
        candidate = propose(prepared, history.sample(step * dt))
        jax.block_until_ready(candidate)
        rows.append(
            {
                "step": step,
                "successful": bool(candidate.successful),
                "nonlinear_status": int(candidate.nonlinear_result.status),
                "free_force_residual_N": float(
                    candidate.diagnostics.free_force_residual_N
                ),
                "pressure_weak_residual_m3": float(
                    candidate.diagnostics.mixed_space.pressure_weak_residual_m3
                ),
                "dilation_weak_residual_J": float(
                    candidate.diagnostics.mixed_space.dilation_weak_residual_J
                ),
                "reaction_x_N": float(candidate.diagnostics.reaction_pulling_N[0]),
                "interface_traction_jump_l2_Pa": float(
                    candidate.diagnostics.interface_traction_jump_l2_Pa
                ),
                "interface_power_defect_W": float(
                    candidate.diagnostics.interface_power_defect_W
                ),
                "work_energy_residual_J": float(
                    candidate.diagnostics.work_energy_residual_J
                ),
            }
        )
        prepared = candidate.commit(prepared)
        if not bool(candidate.successful):
            break
    rollout_seconds = time.perf_counter() - start
    derivative = None
    if sensitivity:
        # Actual implicit tangent and adjoint to density at a loaded source step.
        def reaction(log_density):
            varied = eqx.tree_at(
                lambda p: p.parameters.density_kg_per_m3,
                initial,
                initial.parameters.density_kg_per_m3 * jnp.exp(log_density),
            )
            candidate = varied.propose(history.sample(dt))
            return candidate.diagnostics.reaction_pulling_N[0] * jnp.where(
                candidate.successful, 1.0, jnp.nan
            )

        derivative_start = time.perf_counter()
        value, jvp = eqx.filter_jit(
            lambda x: jax.jvp(reaction, (x,), (jnp.ones_like(x),))
        )(jnp.asarray(0.0))
        vjp = eqx.filter_jit(jax.grad(reaction))(jnp.asarray(0.0))
        jax.block_until_ready((value, jvp, vjp))
        derivative = {
            "reaction_x_N": float(value),
            "density_log_jvp_N": float(jvp),
            "density_log_vjp_N": float(vjp),
            "dual_error_N": float(jnp.abs(jvp - vjp)),
            "finite": bool(jnp.isfinite(jvp) & jnp.isfinite(vjp)),
            "elapsed_seconds": time.perf_counter() - derivative_start,
        }
    inventory = json.loads((Path(inputs) / "reference_outputs/manifest.json").read_text())
    qualified_cases = {
        entry["case"]
        for entry in inventory.get("native_qualifications", ())
        if entry["source_equivalence_passed"] and entry["full_101_sample_campaign"]
    }
    source_qualified = qualified_cases == {
        "repository-default-fields",
        "fixed-end-activation",
        "passive-cyclic",
    }
    return {
        "source": "flexodeal0698e3d-repository-default;not-SIAM-table",
        "scope": "idealized-muscle-aponeurosis",
        "plan_id": plan.plan_id,
        "backend": jax.default_backend(),
        "dtype": "float64",
        "refinement": refinement,
        "cells": initial.geometry.cell_count,
        "quadrature_points": initial.geometry.cell_count * 125,
        "displacement_dofs": initial.geometry.displacement_dof_count * 3,
        "pressure_dofs": initial.geometry.cell_count * 4,
        "dilation_dofs": initial.geometry.cell_count * 4,
        "jacobian": "matrix-free-JVP;no-full-state-dense-Jacobian",
        "prepare_seconds": prepare_seconds,
        "first_execution_seconds": first_seconds,
        "steady_first_step_seconds": steady_seconds,
        "rollout_seconds": rollout_seconds,
        "steps": rows,
        "implicit_sensitivity": derivative,
        "all_successful": len(rows) == steps
        and all(row["successful"] for row in rows)
        and bool(first.successful),
        "source_equivalence_qualified": source_qualified,
        "source_qualification_manifest_id": inventory["content_id"],
        "biological_validation": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests/fixtures/flexodeal_0698e3d",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--refinement", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.repetitions < 1:
        parser.error("steps and repetitions must be positive")
    jax.config.update("jax_enable_x64", True)
    result = run(
        args.inputs,
        0 if args.smoke else args.refinement,
        2 if args.smoke else args.steps,
        1 if args.smoke else args.repetitions,
        args.sensitivity,
    )
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    args.output.write_text(payload + "\n")
    print(payload)
    if not result["all_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
