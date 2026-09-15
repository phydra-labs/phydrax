#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Benchmark a bounded periodic Z4c right-hand side and branch-local JVP."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
from _runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)

import phydrax as phx
from phydrax.applications import numerical_relativity as nr


def _compiler_record(compiled) -> dict[str, object]:
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-executable",
        unavailable_reason="The selected JAX backend did not report compiler analysis.",
    )
    record = asdict(evidence)
    record["estimated_device_memory_bytes"] = evidence.estimated_device_memory_bytes
    return record


def _measure(function, arguments, warmup, repeats):
    compiled, compilation = measure_lower_and_compile(
        lambda: jax.jit(function).lower(*arguments),
        lambda lowered: lowered.compile(),
    )
    result, execution = measure_repeated(
        lambda: compiled(*arguments), warmup=warmup, repeats=repeats
    )
    return result, {
        "compilation": asdict(compilation),
        "execution": execution.to_seconds_dict(),
        "compiler": _compiler_record(compiled),
    }


def _setup(cell_count: int):
    shape = (cell_count, cell_count, cell_count)
    lower = tuple(-0.5 * (cell_count - 1) for _ in range(3))
    spacing = (1.0, 1.0, 1.0)
    grid = nr.FixedGridGeometry(shape, lower, spacing, periodic=True)
    derivatives = nr.FourthOrderDerivatives(grid.shape, grid.spacing)
    system = nr.Z4cSystem(
        phx.RelativityScaleContract.geometric(phx.units.KILOGRAM),
        phx.metrix.RelativityConvention.canonical(),
        chart_id="benchmark-cartesian",
        constraint_tolerance=1.0e-2,
    )
    gauge = nr.HarmonicGauge()
    flat = nr.flat_z4c_state(grid.shape, grid_id=grid.grid_id)
    wave = 1.0e-5 * jnp.sin(
        2.0 * jnp.pi * grid.coordinates[0] / (cell_count * spacing[0])
    )
    conformal_metric = (
        flat.conformal_metric.at[1, 1].add(wave).at[2, 2].add(-wave)
    )
    state = nr.make_z4c_state(
        flat.chi,
        conformal_metric,
        jnp.zeros_like(flat.k_hat),
        flat.conformal_extrinsic_curvature,
        flat.theta,
        flat.conformal_connection,
        flat.lapse,
        flat.shift,
        flat.shift_driver,
        grid_id=grid.grid_id,
    )
    direction = jnp.full_like(state.values, 1.0e-7)
    snapshot_token = nr.z4c_snapshot_token(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    return grid, derivatives, system, gauge, state, direction, snapshot_token


def run(cell_count: int, warmup: int, repeats: int) -> dict[str, object]:
    environment = capture_environment().to_dict()
    setup, setup_seconds = measure_synchronized(lambda: _setup(cell_count))
    grid, derivatives, system, gauge, state, direction, snapshot_token = setup

    def rhs_kernel(values):
        evaluation = nr.evaluate_z4c_rhs(
            system,
            grid,
            derivatives,
            gauge,
            state.with_values(values),
            snapshot_token=snapshot_token,
        )
        return (
            evaluation.rates.values,
            evaluation.constraints.maximum_norm,
            evaluation.constraints.l2_norm,
            evaluation.constraints.qualified,
            evaluation.finite,
            evaluation.physically_valid,
            evaluation.source_valid,
            evaluation.derivative_valid,
        )

    def derivative_kernel(values, tangent):
        def objective(candidate):
            rates = nr.evaluate_z4c_rhs(
                system,
                grid,
                derivatives,
                gauge,
                state.with_values(candidate),
                snapshot_token=snapshot_token,
            ).rates.values
            return jnp.mean(rates * rates)

        return jax.jvp(objective, (values,), (tangent,))

    primal, primal_performance = _measure(
        rhs_kernel, (state.values,), warmup, repeats
    )
    derivative, derivative_performance = _measure(
        derivative_kernel, (state.values, direction), warmup, repeats
    )
    derivative_finite = bool(
        jnp.isfinite(derivative[0]) & jnp.isfinite(derivative[1])
    )
    successful = bool(
        primal[3] & primal[4] & primal[5] & primal[6] & primal[7]
    ) and derivative_finite
    mean_seconds = primal_performance["execution"]["mean_seconds"]
    point_count = cell_count**3
    return {
        "identities": {
            "benchmark": "numerical-relativity",
            "kernel": "periodic-z4c-right-hand-side",
            "system": system.system_id,
            "grid": grid.grid_id,
            "derivatives": derivatives.derivative_id,
            "state_grid": state.grid_id,
            "gauge": gauge.gauge_id,
            "snapshot_token": int(snapshot_token),
        },
        "configuration": {
            "cells_per_axis": cell_count,
            "cell_capacity": point_count,
            "shape": list(grid.shape),
            "spacing": [float(value) for value in grid.spacing],
            "boundary": "periodic",
            "gauge": "harmonic",
            "wave_amplitude": 1.0e-5,
            "constraint_tolerance": system.constraint_tolerance,
            "snapshot_address": {"step_index": 0, "stage_slot": 0},
            "snapshot_token": int(snapshot_token),
            "warmup": warmup,
            "repeats": repeats,
        },
        "environment": environment,
        "physics": {
            "successful": successful,
            "finite": bool(primal[4]),
            "physically_valid": bool(primal[5]),
            "vacuum_source_valid": bool(primal[6]),
            "constraints_qualified": bool(primal[3]),
            "derivative_valid": bool(primal[7]),
            "maximum_constraint_norm": float(primal[1]),
            "constraint_l2_norm": float(primal[2]),
            "maximum_absolute_rate": float(jnp.max(jnp.abs(primal[0]))),
            "squared_rate_objective": float(derivative[0]),
            "state_directional_objective_derivative": float(derivative[1]),
            "state_directional_derivative_finite": derivative_finite,
        },
        "performance": {
            "setup_seconds": setup_seconds,
            "primal": primal_performance,
            "directional_derivative": derivative_performance,
            "logical_bytes": {
                "grid": logical_array_bytes(grid),
                "derivatives": logical_array_bytes(derivatives),
                "state": logical_array_bytes(state),
                "direction": logical_array_bytes(direction),
                "snapshot_token": logical_array_bytes(snapshot_token),
                "primal_output": logical_array_bytes(primal),
                "derivative_output": logical_array_bytes(derivative),
            },
            "grid_points_per_second": None
            if mean_seconds in (None, 0.0)
            else point_count / mean_seconds,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if not 5 <= arguments.cells <= 32:
        raise ValueError("cells must be between 5 and 32 per axis.")
    if not 0 <= arguments.warmup <= 100:
        raise ValueError("warmup must be between 0 and 100.")
    if not 1 <= arguments.repeats <= 1_000:
        raise ValueError("repeats must be between 1 and 1,000.")
    payload = run(arguments.cells, arguments.warmup, arguments.repeats)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    raise SystemExit(0 if payload["physics"]["successful"] else 1)


if __name__ == "__main__":
    main()
