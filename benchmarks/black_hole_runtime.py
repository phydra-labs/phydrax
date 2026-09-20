#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Benchmark the atomic Valencia GRMHD and constrained-transport runtime."""

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
from phydrax.units import KILOGRAM


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


def _setup(cells_per_axis: int):
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(cells_per_axis, periodic=True),
            phx.discretization.UniformCellAxisSpec(cells_per_axis, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    scale = phx.RelativityScaleContract.geometric(KILOGRAM)
    convention = phx.metrix.RelativityConvention.canonical()
    eos = phx.equations.GammaLawEOS(
        scale, 4.0 / 3.0, minimum_density=1.0e-15
    )
    system = phx.equations.IdealValenciaGRMHDSystem(
        eos,
        scale,
        convention=convention,
        pressure_ceiling=1.0e5,
        recovery_iterations=40,
        enthalpy_iterations=40,
    )
    constrained_transport = phx.solver.GRMHDConstrainedTransportPlan(bridge)
    runtime = phx.solver.GRMHDSSPRK3Plan(
        system, constrained_transport, cfl=0.2
    )
    shape = grid.shape
    dtype = jnp.asarray(0.0).dtype
    identity = jnp.broadcast_to(jnp.eye(3, dtype=dtype), shape + (3, 3))
    geometry = phx.metrix.ADMGridGeometry(
        jnp.ones(shape, dtype=dtype),
        jnp.zeros(shape + (3,), dtype=dtype),
        identity,
        identity,
        jnp.ones(shape, dtype=dtype),
        jnp.zeros(shape + (3, 3), dtype=dtype),
        jnp.ones(shape, dtype="bool"),
        jnp.ones(shape, dtype="bool"),
        snapshot_token=jnp.asarray(0, dtype=jnp.int32),
        chart_id="benchmark-minkowski-cartesian",
        convention_id=convention.convention_id,
        scale_id=scale.scale_id,
        topology_id=grid.topology.topology_id,
        geometry_lineage_id="benchmark-minkowski-grid",
    )
    x = grid.points[:, 0].reshape(shape)
    primitive = jnp.zeros(shape + (8,), dtype=dtype)
    primitive = primitive.at[..., 0].set(1.0 + 0.01 * jnp.sin(2.0 * jnp.pi * x))
    primitive = primitive.at[..., 1].set(0.02)
    primitive = primitive.at[..., 4].set(0.05)
    primitive = primitive.at[..., 5].set(0.02)
    primitive = primitive.at[..., 6].set(-0.01)
    conserved = system.primitive_to_conserved(primitive, geometry)
    magnetic_flux = constrained_transport.pack_densitized_face_flux(
        tuple(primitive[..., 5 + axis] for axis in range(2))
    )
    state = runtime.initialize(
        conserved,
        geometry,
        magnetic_flux=magnetic_flux,
        step_size=1.0e-4,
    )
    return (
        grid,
        bridge,
        scale,
        convention,
        eos,
        system,
        constrained_transport,
        runtime,
        geometry,
        state,
    )


def run(
    cells_per_axis: int,
    step_count: int,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    environment = capture_environment().to_dict()
    setup, setup_seconds = measure_synchronized(lambda: _setup(cells_per_axis))
    (
        grid,
        bridge,
        scale,
        convention,
        eos,
        system,
        constrained_transport,
        runtime,
        geometry,
        initial_state,
    ) = setup

    def rollout(state, step_size):
        def advance(current, _):
            result = runtime.advance(
                current, current.time, current.time + step_size, geometry
            )
            evidence = (
                result.accepted,
                result.finite,
                result.converged,
                result.physically_valid,
                result.qualified,
                result.derivative_valid,
                jnp.max(result.stages.maximum_recovery_residuals),
                jnp.max(result.stages.maximum_magnetizations),
                jnp.max(jnp.abs(result.attempted_ledger.material_balance_defect)),
                jnp.max(jnp.abs(result.attempted_ledger.faraday_balance_defect)),
                jnp.max(jnp.abs(result.attempted_ledger.magnetic_divergence_after)),
            )
            return result.state, evidence

        final, evidence = jax.lax.scan(
            advance, state, xs=None, length=step_count
        )
        return (
            final.material_state,
            final.constrained_transport.magnetic_flux,
            final.time,
            final.accepted_step,
            final.status,
            jnp.all(evidence[0]),
            jnp.all(evidence[1]),
            jnp.all(evidence[2]),
            jnp.all(evidence[3]),
            jnp.all(evidence[4]),
            jnp.all(evidence[5]),
            jnp.max(evidence[6]),
            jnp.max(evidence[7]),
            jnp.max(evidence[8]),
            jnp.max(evidence[9]),
            jnp.max(evidence[10]),
        )

    def step_size_derivative(state, step_size, direction):
        def objective(value):
            material = rollout(state, value)[0]
            return jnp.mean(material * material)

        return jax.jvp(objective, (step_size,), (direction,))

    step_size = jnp.asarray(1.0e-4, dtype=initial_state.time.dtype)
    direction = jnp.asarray(1.0, dtype=step_size.dtype)
    primal, primal_performance = _measure(
        rollout, (initial_state, step_size), warmup, repeats
    )
    derivative, derivative_performance = _measure(
        step_size_derivative,
        (initial_state, step_size, direction),
        warmup,
        repeats,
    )
    derivative_finite = bool(
        jnp.isfinite(derivative[0]) & jnp.isfinite(derivative[1])
    )
    successful = bool(
        primal[5] & primal[6] & primal[7] & primal[8] & primal[9] & primal[10]
    ) and derivative_finite
    mean_seconds = primal_performance["execution"]["mean_seconds"]
    cell_count = cells_per_axis**2
    return {
        "identities": {
            "benchmark": "black-hole-runtime",
            "kernel": "atomic-valencia-grmhd-ssprk3-constrained-transport",
            "runtime": runtime.plan_id,
            "system": system.system_id,
            "eos": eos.eos_id,
            "constrained_transport": constrained_transport.plan_id,
            "grid": grid.prepared_id,
            "topology": grid.topology.topology_id,
            "scale": scale.scale_id,
            "convention": convention.convention_id,
            "geometry_lineage": geometry.geometry_lineage_id,
            "geometry_snapshot_token": int(geometry.snapshot_token),
        },
        "configuration": {
            "cells_per_axis": cells_per_axis,
            "cell_capacity": cell_count,
            "step_capacity": step_count,
            "step_size": float(step_size),
            "spatial_dimension": 2,
            "periodic": True,
            "cfl": runtime.cfl,
            "warmup": warmup,
            "repeats": repeats,
        },
        "environment": environment,
        "physics": {
            "successful": successful,
            "all_steps_accepted": bool(primal[5]),
            "all_steps_finite": bool(primal[6]),
            "all_stages_converged": bool(primal[7]),
            "all_steps_physically_valid": bool(primal[8]),
            "all_steps_qualified": bool(primal[9]),
            "all_step_derivatives_valid": bool(primal[10]),
            "final_status": int(primal[4]),
            "accepted_steps": int(primal[3]),
            "final_time": float(primal[2]),
            "maximum_recovery_residual": float(primal[11]),
            "maximum_magnetization": float(primal[12]),
            "maximum_material_balance_defect": float(primal[13]),
            "maximum_faraday_balance_defect": float(primal[14]),
            "maximum_magnetic_divergence": float(primal[15]),
            "step_size_objective": float(derivative[0]),
            "step_size_directional_derivative": float(derivative[1]),
            "step_size_derivative_finite": derivative_finite,
        },
        "performance": {
            "setup_seconds": setup_seconds,
            "primal_rollout": primal_performance,
            "step_size_directional_derivative": derivative_performance,
            "logical_bytes": {
                "grid": logical_array_bytes(grid),
                "bridge": logical_array_bytes(bridge),
                "runtime": logical_array_bytes(runtime),
                "geometry": logical_array_bytes(geometry),
                "initial_state": logical_array_bytes(initial_state),
                "step_size": logical_array_bytes(step_size),
                "step_size_direction": logical_array_bytes(direction),
                "primal_output": logical_array_bytes(primal),
                "derivative_output": logical_array_bytes(derivative),
            },
            "cell_steps_per_second": None
            if mean_seconds in (None, 0.0)
            else cell_count * step_count / mean_seconds,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=16)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if not 2 <= arguments.cells <= 128:
        raise ValueError("cells must be between 2 and 128 per axis.")
    if not 1 <= arguments.steps <= 1_000:
        raise ValueError("steps must be between 1 and 1,000.")
    if not 0 <= arguments.warmup <= 100:
        raise ValueError("warmup must be between 0 and 100.")
    if not 1 <= arguments.repeats <= 1_000:
        raise ValueError("repeats must be between 1 and 1,000.")
    payload = run(
        arguments.cells, arguments.steps, arguments.warmup, arguments.repeats
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    raise SystemExit(0 if payload["physics"]["successful"] else 1)


if __name__ == "__main__":
    main()
