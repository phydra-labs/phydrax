# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Intrinsic surface and moving extensive lifecycle point-capacity scaling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from benchmarks._io import write_json_atomic
from benchmarks._runtime import logical_array_bytes, measure_synchronized
from benchmarks.meshfree_scaling import (
    add_config_arguments,
    apply_baseline,
    config_from_arguments,
    execution_evidence,
    make_record,
    measured_phase,
    MeshfreeConfig,
    unavailable_phases,
)


def measure_capacity(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    from examples.meshfree_moving_surface_reaction_diffusion import (
        perform_epoch_transition,
        perform_shift,
        prepare_workflow,
    )
    from examples.meshfree_surface_laplace_beltrami import surface_plan

    if config.dimension != 3:
        raise ValueError("Moving/surface benchmark requires actual ambient dimension=3.")
    reservation = config.check_capacity(capacity)
    plan = surface_plan(
        size=capacity, seed=seed, neighbors=config.neighbors, chunk_rows=config.chunk_rows
    )
    surface, surface_seconds = measure_synchronized(plan.prepare)
    values = surface.points[:, 0]
    surface_execution = execution_evidence(surface.laplace_beltrami.mv, values, config)
    surface_actual = np.asarray(surface_execution.pop("result"))
    expected = -2 * np.asarray(values)
    surface_error = float(
        np.linalg.norm(surface_actual - expected) / np.linalg.norm(expected)
    )
    if not np.isfinite(surface_error) or surface_error > 1:
        raise AssertionError(
            "Surface Laplace--Beltrami failed coordinate eigenfunction oracle."
        )
    # A same-support numeric refresh is a genuine owner operation even when the
    # coordinates are unchanged; no fictitious rebuild/compilation is counted.
    refreshed_surface, surface_refresh_seconds = measure_synchronized(
        lambda: surface.refresh(surface.points)
    )
    if not bool(np.asarray(refreshed_surface.accepted)):
        raise AssertionError("Fixed-support surface numeric refresh refused.")
    prepared_workflow, preparation_seconds = measure_synchronized(
        lambda: prepare_workflow(size=capacity, dimension=3, seed=seed)
    )
    moving_plan, state, pairs = prepared_workflow
    dt = jnp.asarray(0.01, dtype=state.content.dtype)

    def step(content: Array) -> Array:
        candidate = eqx.tree_at(lambda old: old.content, state, content)
        return moving_plan.step(candidate, dt).state.content

    moving_execution = execution_evidence(step, state.content, config)
    moving_actual = np.asarray(moving_execution.pop("result"))
    full_step, status_seconds = measure_synchronized(lambda: moving_plan.step(state, dt))
    if not bool(np.asarray(full_step.evidence.successful)):
        raise AssertionError("Native moving surface step refused.")
    geometry, geometry_seconds = measure_synchronized(
        lambda: moving_plan.geometry_refresh(state, state.time + dt, dt, None)
    )
    if not bool(np.asarray(geometry.successful)):
        raise AssertionError("Native moving numeric geometry refresh refused.")
    epoch, epoch_seconds = measure_synchronized(
        lambda: perform_epoch_transition(moving_plan, full_step.state, seed=seed)
    )
    if not epoch.successful:
        raise AssertionError("Native atomic historical epoch transfer refused.")
    shifted, shift_seconds = measure_synchronized(lambda: perform_shift(full_step.state))
    shifted_state, shift = shifted
    if not bool(np.asarray(shift.successful)):
        raise AssertionError("Native tangential surface shift refused.")
    radius = 1 + 0.2 * float(np.asarray(full_step.state.time))
    analytic = np.exp(-0.1 * float(np.asarray(full_step.state.time))) / radius**2
    concentration_error = float(
        np.max(np.abs(moving_actual / np.asarray(full_step.state.measures) - analytic))
    )
    conservation = float(abs(np.asarray(full_step.evidence.conservation_residual)))
    epoch_conservation = float(np.max(np.abs(np.asarray(epoch.conservation_residuals))))
    if concentration_error > 5e-3 or conservation > 1e-6 or epoch_conservation > 1e-6:
        raise AssertionError(
            "Moving surface disagrees with dilution or native extensive ledger."
        )
    retained_visible = logical_array_bytes(
        (
            surface,
            refreshed_surface,
            prepared_workflow,
            full_step,
            epoch,
            shifted_state,
            shift,
        )
    )
    if retained_visible > config.working_set_bytes:
        raise ValueError(
            "Visible retained surface/lifecycle arrays exceed working-set budget."
        )
    surface_phases = unavailable_phases()
    surface_phases.update(surface_execution.pop("phases"))
    surface_phases["stencil"] = {
        **measured_phase(surface_seconds),
        "scope": "surface-neighbor,geometry,quadrature,intrinsic-stencils",
    }
    surface_phases["neighbor"] = {
        "status": "unavailable",
        "reason": "SurfacePointCloudPlan exposes joint geometry/support admission",
    }
    surface_phases["numeric-refresh"] = measured_phase(surface_refresh_seconds)
    moving_phases = unavailable_phases()
    moving_phases.update(moving_execution.pop("phases"))
    moving_phases["assembly"] = {
        **measured_phase(preparation_seconds),
        "scope": "native-moving-graph-and-initial-state-preparation",
    }
    moving_phases["numeric-refresh"] = measured_phase(geometry_seconds)
    moving_phases["epoch-transition"] = {
        **measured_phase(epoch_seconds),
        "includes_transfer_preparation": True,
    }
    return {
        "capacity": capacity,
        "seed": seed,
        "dimension": 3,
        "domain": "unit-sphere-expanding-to-radius-1+0.2t",
        "reserved_working_set_bytes": reservation,
        "retained_bytes": None,
        "retained_visible_bytes": retained_visible,
        "retained_bytes_unavailable_reason": "Native moving callbacks capture operator arrays outside the runtime object walker",
        "surface": {
            "phases": surface_phases,
            **surface_execution,
            "laplace_relative_error": surface_error,
        },
        "moving": {
            "phases": moving_phases,
            **moving_execution,
            "status_execution_seconds": status_seconds,
            "shift_seconds": shift_seconds,
            "concentration_error": concentration_error,
            "conservation_defect": conservation,
            "epoch_conservation_defect": epoch_conservation,
            "active_points": state.points.shape[0],
        },
        "oracle": "independent-host-NumPy sphere eigenfunction and exp(-0.1t)/(1+0.2t)^2 dilution",
    }


def run(config: MeshfreeConfig = MeshfreeConfig(dimension=3), /) -> dict[str, Any]:
    jax.config.update("jax_enable_x64", config.precision == "float64")
    import examples.meshfree_moving_surface_reaction_diffusion as moving_consumer
    import examples.meshfree_surface_laplace_beltrami as surface_consumer

    return make_record(
        config,
        [
            measure_capacity(size, seed, config)
            for size in config.sizes
            for seed in config.seeds
        ],
        Path(__file__),
        consumers=(Path(surface_consumer.__file__), Path(moving_consumer.__file__)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.set_defaults(dimension=3)
    args = parser.parse_args()
    record = run(config_from_arguments(args))
    apply_baseline(record, args.baseline)
    if args.output is not None:
        write_json_atomic(args.output, record)
    else:
        print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
