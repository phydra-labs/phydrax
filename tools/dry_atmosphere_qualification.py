"""Executable Cartesian dry-atmosphere qualification, not global validation.

JAX_ENABLE_X64=1 python -m tools.dry_atmosphere_qualification --nx 16 --nz 12 --steps 4
Outputs measured rest, budget, manufactured-shear refinement, continuation,
thermal/gravity-wave/cold-pool onset, and execution-time evidence. These are
inviscid method-qualification cases, not claims of resolved turbulent fronts.
"""

from __future__ import annotations

import argparse
import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.dry_atmosphere import build_case
from phydrax.applications.atmosphere._dry import (
    DryAtmospherePlan,
    DryHydrostaticReference,
)


def _rest(family):
    prepared = DryAtmospherePlan(
        (8,), ((0.0,), (8000.0,)), reference=DryHydrostaticReference(family)
    ).prepare()
    initial = prepared.initial_state()
    evaluate = eqx.filter_jit(prepared.balance.evaluate)
    residual = evaluate(initial.conserved)
    dt = 0.4 * eqx.filter_jit(prepared.stable_step)(initial)
    result = eqx.filter_jit(prepared.rollout)(initial, jnp.full((4,), dt))
    velocity = (
        result.state.conserved[..., prepared.system.momentum_slice]
        / prepared.system.density(result.state.conserved)[..., None]
    )
    return {
        "family": family,
        "successful": bool(result.successful),
        "maximum_rest_residual": float(jnp.max(jnp.abs(residual.residual))),
        "maximum_rest_velocity_m_s": float(jnp.max(jnp.abs(velocity))),
        "relative_energy_closure": float(
            result.budget.closure[-1] / result.budget.total_energy
        ),
    }


def manufactured_shear_error(nz):
    """An exact steady Euler solution: arbitrary u(z), w=0, hydrostatic p/rho.

    Errors are volume-mean absolute residuals normalized by fixed physical
    scales. Refinement tests the actual nonlinear numerical operator, with no
    source chosen to cancel its discrete residual.
    """
    prepared = DryAtmospherePlan((4, nz), ((0.0, 0.0), (4000.0, 8000.0))).prepare()
    z = prepared.balance.discretization.cell_centers[..., 1]
    u = 5.0 * jnp.sin(jnp.pi * z / 8000.0) ** 2
    state = prepared.thermal_state(
        jnp.zeros_like(z), velocity=jnp.stack((u, jnp.zeros_like(u)), axis=-1)
    )
    evaluation = eqx.filter_jit(prepared.balance.evaluate)(state.conserved)
    scales = jnp.asarray((1.0, 1.0, 1.0, 5.0, 5.0, 250000.0))
    error = jnp.mean(jnp.abs(evaluation.residual) / scales)
    return float(error)


def _dynamics(case, nx, nz, steps):
    start = time.perf_counter()
    prepared, initial = build_case(case, nx=nx, nz=nz)
    step_size = 0.35 * eqx.filter_jit(prepared.stable_step)(initial)
    rollout = eqx.filter_jit(prepared.rollout)
    schedule = jnp.full((steps,), step_size)
    first = rollout(initial, schedule)
    jax.block_until_ready(first)
    prepared_and_compiled_s = time.perf_counter() - start
    start = time.perf_counter()
    result = rollout(initial, schedule)
    jax.block_until_ready(result)
    execution_s = time.perf_counter() - start
    system = prepared.system
    velocity = (
        result.state.conserved[..., system.momentum_slice]
        / system.density(result.state.conserved)[..., None]
    )
    warm = case in ("rising_thermal", "gravity_wave")
    directional_velocity = (
        float(jnp.max(velocity[..., -1])) if warm else -float(jnp.min(velocity[..., -1]))
    )
    split = max(1, steps // 2)
    prefix = eqx.filter_jit(prepared.rollout)(initial, schedule[:split])
    restart = prepared.checkpoint(prefix.state)
    restored = prepared.restore(restart)
    if split < steps:
        continued = eqx.filter_jit(prepared.rollout)(restored, schedule[split:]).state
    else:
        continued = restored
    restart_error = float(jnp.max(jnp.abs(continued.conserved - result.state.conserved)))
    # Real acceptance boundary: thermodynamically forbidden density must not
    # be silently clipped into an atmosphere, and rejected steps are atomic.
    near_vacuum_valid = bool(jnp.any(system.admissible(initial.conserved * 1.0e-12)))
    rejected = eqx.filter_jit(prepared.advance)(
        initial, 2.0 * eqx.filter_jit(prepared.stable_step)(initial)
    )
    rejected_state_unchanged = bool(
        jnp.array_equal(rejected.state.conserved, initial.conserved)
    ) and float(rejected.state.time) == float(initial.time)
    return {
        "case": case,
        "successful": bool(result.successful),
        "time_s": float(result.state.time),
        "step_s": float(step_size),
        "vertical_response_in_expected_direction_m_s": directional_velocity,
        "relative_energy_closure": float(
            result.budget.closure[-1] / result.budget.total_energy
        ),
        "maximum_relative_species_closure": float(
            jnp.max(
                jnp.abs(
                    result.budget.closure[: system.species_count]
                    / result.budget.species_mass
                )
            )
        ),
        "restart_maximum_absolute_error": restart_error,
        "near_vacuum_valid": near_vacuum_valid,
        "unstable_step_rejected_atomically": not bool(rejected.accepted)
        and rejected_state_unchanged,
        "prepare_compile_first_run_s": prepared_and_compiled_s,
        "warm_execution_s": execution_s,
        "cell_updates_per_second": nx * nz * steps / execution_s,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--nz", type=int, default=12)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    if args.nx < 8 or args.nz < 8 or args.steps < 2:
        parser.error("qualification requires nx,nz >= 8 and steps >= 2")
    rests = [_rest(family) for family in ("isothermal", "isentropic")]
    counts = (12, 24, 48)
    errors = [manufactured_shear_error(n) for n in counts]
    orders = [float(np.log(errors[i] / errors[i + 1]) / np.log(2.0)) for i in range(2)]
    cases = [
        _dynamics(case, args.nx, args.nz, args.steps)
        for case in ("gravity_wave", "rising_thermal", "density_current")
    ]
    report = {
        "scope": "fixed-Cartesian inviscid dry column and slice; not global atmospheric qualification",
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "rest": rests,
        "manufactured_steady_shear": {
            "vertical_cells": counts,
            "mean_scaled_residual": errors,
            "observed_orders": orders,
        },
        "dynamics": cases,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    rest_ok = all(
        item["successful"] and item["maximum_rest_velocity_m_s"] < 1.0e-10
        for item in rests
    )
    refinement_ok = errors[-1] < errors[0] and all(order > 0.0 for order in orders)
    cases_ok = all(
        item["successful"]
        and item["vertical_response_in_expected_direction_m_s"] > 0.0
        and abs(item["relative_energy_closure"]) < 1.0e-11
        and item["maximum_relative_species_closure"] < 1.0e-11
        and item["restart_maximum_absolute_error"] < 1.0e-9
        and not item["near_vacuum_valid"]
        and item["unstable_step_rejected_atomically"]
        for item in cases
    )
    if not (jax.config.jax_enable_x64 and rest_ok and refinement_ok and cases_ok):
        raise RuntimeError(
            "Dry atmospheric qualification failed; inspect measured evidence above."
        )


if __name__ == "__main__":
    main()
