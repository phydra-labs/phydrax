#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Benchmark one bounded stage-exact Z4c--Einstein--Vlasov transaction."""

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
from phydrax._physical import RelativityScaleContract
from phydrax.applications.numerical_relativity._boundaries import PeriodicBoundary
from phydrax.applications.numerical_relativity._derivatives import FourthOrderDerivatives
from phydrax.applications.numerical_relativity._einstein_vlasov import (
    EinsteinVlasovConstraintSolveEvidence,
    EinsteinVlasovConstraintSolveResult,
    EinsteinVlasovMatterPlan,
)
from phydrax.applications.numerical_relativity._enforcement import (
    Z4cAlgebraicEnforcement,
)
from phydrax.applications.numerical_relativity._gauge import GeodesicGauge
from phydrax.applications.numerical_relativity._grid import FixedGridGeometry
from phydrax.applications.numerical_relativity._state import flat_z4c_state
from phydrax.applications.numerical_relativity._z4c import (
    z4c_adm_geometry,
    Z4cSystem,
)
from phydrax.applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from phydrax.discretization.particle._relativistic_stress_transfer import (
    RelativisticStressDepositPlan,
)
from phydrax.metrix import RelativityConvention
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


def _setup(cell_count: int, particle_count: int):
    shape = (cell_count,) * 3
    upper = float(cell_count - 1)
    grid = FixedGridGeometry(shape, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), periodic=True)
    transfer_grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(cell_count) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, 0.0), (upper, upper, upper))))
    support = phx.discretization.ParticleSetPlan(
        jnp.arange(particle_count, dtype=jnp.int64),
        jnp.ones((particle_count,)),
        ambient_dimension=3,
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(transfer_grid).prepare(support)
    scale = RelativityScaleContract.geometric(KILOGRAM)
    convention = RelativityConvention.canonical()
    units = RelativisticUnitContract(scale, convention)
    stress = RelativisticStressDepositPlan(
        splat,
        units,
        jnp.asarray((1,), dtype=jnp.int32),
        jnp.asarray((1.0,)),
        topology_id=grid.grid_id,
        mass_shell_relative_tolerance=1.0e-5,
        conservation_tolerance=1.0e-5,
        frame_momentum_relative_tolerance=1.0e-5,
    )
    system = Z4cSystem(
        scale,
        convention,
        chart_id="benchmark-cartesian",
        constraint_tolerance=1.0e-3,
    )
    spatial = jnp.moveaxis(grid.coordinates, 0, -1)
    coordinates = jnp.concatenate((jnp.zeros(shape + (1,)), spatial), axis=-1)

    def frame_provider(geometry, time, scale_factor):
        return LocalRelativisticFramePlan.from_adm(
            geometry,
            units,
            coordinates.at[..., 0].set(time),
            time,
            scale_factor,
            observer_id="benchmark-eulerian-observer",
            orientation_id="right-handed-future",
        )

    z4c = flat_z4c_state(shape, grid_id=grid.grid_id)
    initial_geometry = z4c_adm_geometry(system, grid, z4c, snapshot_token=jnp.int32(0))
    initial_frame = frame_provider(initial_geometry, jnp.asarray(0.0), jnp.asarray(1.0))
    ordinal = jnp.arange(particle_count, dtype=jnp.float64)
    fraction = (ordinal + 0.5) / particle_count
    positions = jnp.stack(
        (
            0.5 + fraction * (cell_count - 2.0),
            0.5 + jnp.mod(7.0 * fraction, 1.0) * (cell_count - 2.0),
            0.5 + jnp.mod(13.0 * fraction, 1.0) * (cell_count - 2.0),
        ),
        axis=-1,
    )
    signs = jnp.where(
        jnp.arange(particle_count) % 2 == 0,
        jnp.asarray(1.0, dtype=positions.dtype),
        jnp.asarray(-1.0, dtype=positions.dtype),
    )
    momenta = jnp.stack((0.1 * signs, 0.03 * signs, -0.02 * signs), axis=-1).astype(
        positions.dtype
    )
    particles = stress.initialize(
        jnp.full((particle_count,), 1.0e-15, dtype=positions.dtype),
        positions,
        momenta,
        jnp.ones((particle_count,), dtype=jnp.int32),
        initial_frame,
    )
    plan = EinsteinVlasovMatterPlan(
        system,
        grid,
        FourthOrderDerivatives(shape, grid.spacing),
        GeodesicGauge(),
        PeriodicBoundary(),
        Z4cAlgebraicEnforcement(maximum_correction=1.0),
        stress,
        frame_provider,
        frame_provider_id="benchmark-adm-frame-provider",
        time_step=0.01,
        source_tolerance=1.0e-4,
        mass_shell_tolerance=1.0e-4,
        energy_condition_tolerance=1.0e-6,
        maximum_consecutive_failures=2,
    )

    def constraint_solver(state, particle_state, projection, geometry):
        del particle_state, projection, geometry
        return EinsteinVlasovConstraintSolveResult(
            state,
            EinsteinVlasovConstraintSolveEvidence(
                0.0,
                jnp.zeros((3,)),
                0.0,
                jnp.zeros((3,)),
                1,
                True,
                True,
                solver_id="benchmark-manufactured-solve",
            ),
        )

    state = plan.initialize(z4c, particles, constraint_solver)
    return plan, state


def run(
    cell_count: int, particle_count: int, warmup: int, repeats: int
) -> dict[str, object]:
    environment = capture_environment().to_dict()
    setup, setup_seconds = measure_synchronized(
        lambda: _setup(cell_count, particle_count)
    )
    plan, state = setup

    def kernel(candidate):
        result = plan.advance(candidate)
        return (
            result.accepted,
            result.evidence.source_exchange_defect,
            result.evidence.adm_constraint_linf,
            result.evidence.mass_shell_linf,
            result.evidence.dominant_energy_violation,
            result.evidence.metric_condition_number,
            result.evidence.maximum_extrinsic_curvature,
            result.evidence.resource_valid,
            result.evidence.strong_field_supported,
            result.evidence.derivative_valid,
            result.evidence.qualified,
            result.successful,
            result.endpoint_stress.source_integrals.energy,
            jnp.max(
                jnp.abs(
                    result.accepted.particles.positions - candidate.particles.positions
                )
            ),
        )

    output, performance = _measure(kernel, (state,), warmup, repeats)
    accepted = output[0]
    mean_seconds = performance["execution"]["mean_seconds"]
    successful = bool(output[11])
    return {
        "identities": {
            "benchmark": "einstein-vlasov",
            "runtime": plan.runtime_id,
            "stress_plan": plan.stress.plan_id,
            "grid": plan.grid.grid_id,
            "unit_contract": plan.stress.units.contract_id,
            "frame_provider": plan.frame_provider_id,
        },
        "configuration": {
            "cells_per_axis": cell_count,
            "cell_capacity": cell_count**3,
            "particle_capacity": particle_count,
            "route_capacity": plan.stress.transfer.route_count,
            "time_step": plan.time_step,
            "warmup": warmup,
            "repeats": repeats,
        },
        "environment": environment,
        "physics": {
            "successful": successful,
            "source_exchange_defect": float(output[1]),
            "adm_constraint_linf": float(output[2]),
            "mass_shell_linf": float(output[3]),
            "dominant_energy_violation": float(output[4]),
            "metric_condition_number": float(output[5]),
            "maximum_extrinsic_curvature": float(output[6]),
            "resource_valid": bool(output[7]),
            "strong_field_supported": bool(output[8]),
            "derivative_valid": bool(output[9]),
            "qualified": bool(output[10]),
            "endpoint_source_energy": float(output[12]),
            "maximum_geodesic_displacement": float(output[13]),
            "accepted_steps": int(accepted.accepted_steps),
            "rejected_steps": int(accepted.rejected_steps),
        },
        "performance": {
            "setup_seconds": setup_seconds,
            "transaction": performance,
            "logical_bytes": {
                "plan": logical_array_bytes(plan),
                "state": logical_array_bytes(state),
                "accepted_state": logical_array_bytes(accepted),
            },
            "particles_per_second": None
            if mean_seconds in (None, 0.0)
            else particle_count / mean_seconds,
            "grid_points_per_second": None
            if mean_seconds in (None, 0.0)
            else cell_count**3 / mean_seconds,
            "particle_grid_interactions_per_second": None
            if mean_seconds in (None, 0.0)
            else plan.stress.transfer.route_count / mean_seconds,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=5)
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if not 5 <= arguments.cells <= 24:
        raise ValueError("cells must be between 5 and 24 per axis.")
    if not 2 <= arguments.particles <= 4096:
        raise ValueError("particles must be between 2 and 4096.")
    if not 0 <= arguments.warmup <= 100:
        raise ValueError("warmup must be between 0 and 100.")
    if not 1 <= arguments.repeats <= 1_000:
        raise ValueError("repeats must be between 1 and 1,000.")
    payload = run(
        arguments.cells, arguments.particles, arguments.warmup, arguments.repeats
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
