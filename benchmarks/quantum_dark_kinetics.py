#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from benchmarks._runtime import capture_environment
from phydrax.equations._uehling_uhlenbeck import UehlingUhlenbeckPlan


def _case(repetitions: int, spatial_cells: int):
    plan = UehlingUhlenbeckPlan(
        jnp.asarray((-1, -1, -1, -1), dtype=jnp.int8),
        jnp.ones((4, 1)),
        jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
        jnp.zeros((1, 4), dtype=jnp.int32),
        jnp.asarray((2.0,)),
        jnp.asarray(
            (
                ((2.0, 1.0, 0.0, 0.0),),
                ((2.0, -1.0, 0.0, 0.0),),
                ((2.0, 0.0, 1.0, 0.0),),
                ((2.0, 0.0, -1.0, 0.0),),
            )
        ),
        jnp.asarray(((1.0,), (-1.0,), (1.0,), (-1.0,))),
        time_unit_id="dimensionless-benchmark-time",
        invariant_tolerance=1.0e-5,
        entropy_tolerance=1.0e-5,
    )
    occupancy = jnp.broadcast_to(
        jnp.asarray((0.8, 0.7, 0.1, 0.2))[:, None, None],
        (4, spatial_cells, 1),
    )
    compiled_flux = eqx.filter_jit(plan.event_flux)
    compiled_advance = eqx.filter_jit(plan.advance)

    started = time.perf_counter()
    flux = compiled_flux(occupancy)
    first = compiled_advance(occupancy, 0.02)
    jax.block_until_ready(first.accepted_occupancy)
    compile_and_first_ms = 1_000.0 * (time.perf_counter() - started)

    started = time.perf_counter()
    result = first
    for _ in range(repetitions):
        flux = compiled_flux(occupancy)
        result = compiled_advance(occupancy, 0.02)
    jax.block_until_ready(result.accepted_occupancy)
    execution_ms = 1_000.0 * (time.perf_counter() - started) / repetitions
    resources = plan.fixed_stencil_resources
    rate_evaluations = resources["active_events"] * spatial_cells
    resident_bytes = sum(
        value.size * value.dtype.itemsize
        for value in (
            plan.statistics,
            plan.phase_space_weights,
            plan.event_species,
            plan.event_momenta,
            plan.event_kernels,
            plan.event_active,
            plan.four_momenta,
            plan.charges,
            occupancy,
        )
    )
    return {
        "profile": "fixed-support-quantum-2to2",
        "stencil_resources": {
            **resources,
            "spatial_cells": spatial_cells,
            "state_elements": occupancy.size,
            "resident_array_bytes": resident_bytes,
        },
        "rate_resources": {
            "event_cell_fluxes_per_step": rate_evaluations,
            "event_cell_fluxes_per_second": 1_000.0 * rate_evaluations / execution_ms,
            "maximum_absolute_flux": float(jnp.max(jnp.abs(flux))),
            "required_substeps": int(result.evidence.required_substeps),
        },
        "compile_and_first_ms": compile_and_first_ms,
        "execution_ms": execution_ms,
        "steps_per_second": 1_000.0 / execution_ms,
        "successful": bool(result.successful),
        "charge_defect_max": float(jnp.max(jnp.abs(result.evidence.charge_defect))),
        "four_momentum_defect_max": float(
            jnp.max(jnp.abs(result.evidence.four_momentum_defect))
        ),
        "minimum_pointwise_entropy_production": float(
            jnp.min(result.evidence.entropy_production)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--spatial-cells", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/quantum_dark_kinetics.json"),
    )
    arguments = parser.parse_args()
    if arguments.repeats <= 0 or arguments.spatial_cells <= 0:
        raise ValueError("--repeats and --spatial-cells must be positive.")
    case = _case(arguments.repeats, arguments.spatial_cells)
    payload = {
        "environment": capture_environment().to_dict(),
        "case": case,
        "all_successful": case["successful"],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
