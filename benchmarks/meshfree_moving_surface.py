# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Intrinsic surface and moving extensive lifecycle point-capacity scaling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from benchmarks._io import write_json_atomic
from benchmarks._runtime import logical_array_bytes
from benchmarks.meshfree_scaling import (
    add_config_arguments,
    apply_baseline,
    config_from_arguments,
    configure_precision,
    make_record,
    MeshfreeConfig,
    PhaseRecorder,
)
from phydrax.discretization.meshfree import MovingSurfacePlan, MovingSurfaceState
from phydrax.solver.advanced import AdditiveIMEXScheme


_COMPILED_STEP = eqx.filter_jit(MovingSurfacePlan.step)


def _step_concentration(
    plan: MovingSurfacePlan,
    state: MovingSurfaceState,
    dt: Array,
    content: Array,
    /,
) -> Array:
    candidate = eqx.tree_at(lambda old: old.content, state, content)
    return _COMPILED_STEP(plan, candidate, dt).state.concentration


def _method_record(
    method: AdditiveIMEXScheme,
    capacity: int,
    seed: int,
    config: MeshfreeConfig,
    analytic: float,
    /,
) -> dict[str, Any]:
    from examples.meshfree_moving_surface_reaction_diffusion import prepare_workflow

    plan, state = prepare_workflow(size=capacity, dimension=3, seed=seed, method=method)
    dt = jnp.asarray(0.01, dtype=state.content.dtype)

    step = eqx.Partial(_step_concentration, plan, state, dt)

    recorder = PhaseRecorder()
    result, execution = recorder.compiled_action(
        step,
        state.content,
        budget_bytes=config.resource_bytes,
        repeats=config.repeats,
        scope=method,
    )
    actual = np.asarray(result)
    full_step = recorder.run(
        "solve",
        lambda: _COMPILED_STEP(plan, state, dt),
        scope=f"{method}-full-step-evidence",
    )
    if not bool(np.asarray(full_step.evidence.successful)):
        raise AssertionError(f"Native moving surface {method} step refused.")
    return {
        "phases": recorder.record(),
        **execution,
        "stage_count": plan.tableau.stage_count,
        "concentration_error": float(np.max(np.abs(actual - analytic))),
        "successful": bool(np.asarray(full_step.evidence.successful)),
        "conservation_defect": float(
            abs(np.asarray(full_step.evidence.conservation_residual))
        ),
        "gcl_defect": float(np.asarray(full_step.evidence.gcl_defect)),
    }


def measure_capacity(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    from examples.meshfree_moving_surface_reaction_diffusion import (
        perform_epoch_transition,
        perform_shift,
        prepare_shift_plan,
        prepare_workflow,
    )
    from examples.meshfree_surface_laplace_beltrami import surface_plan

    if config.dimension != 3:
        raise ValueError("Moving/surface benchmark requires actual ambient dimension=3.")
    reservation = config.check_capacity(capacity)
    plan = surface_plan(
        size=capacity, seed=seed, neighbors=config.neighbors, chunk_rows=config.chunk_rows
    )
    surface_recorder = PhaseRecorder()
    surface = surface_recorder.run(
        "geometry",
        plan.prepare,
        scope="surface-neighbor,geometry,quadrature,intrinsic-stencils",
    )
    for phase in ("search", "local-fit"):
        surface_recorder.unavailable(
            phase, "SurfacePointCloudPlan exposes joint geometry/support admission"
        )
    values = surface.points[:, 0]
    surface_result, surface_execution = surface_recorder.compiled_action(
        surface.laplace_beltrami.mv,
        values,
        budget_bytes=config.resource_bytes,
        repeats=config.repeats,
        scope="laplace-beltrami",
    )
    surface_actual = np.asarray(surface_result)
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
    refreshed_surface = surface_recorder.run(
        "numeric-refresh", lambda: surface.refresh(surface.points)
    )
    if not bool(np.asarray(refreshed_surface.accepted)):
        raise AssertionError("Fixed-support surface numeric refresh refused.")
    moving_recorder = PhaseRecorder()
    prepared_workflow = moving_recorder.run(
        "assembly",
        lambda: prepare_workflow(size=capacity, dimension=3, seed=seed),
        scope="native-moving-graph-and-initial-state-preparation",
    )
    moving_plan, state = prepared_workflow
    dt = jnp.asarray(0.01, dtype=state.content.dtype)
    time = 0.01
    analytic = np.exp(-0.1 * time) / (1 + 0.2 * time) ** 2
    # Every selected stage rule refreshes geometry at its own stages and binds
    # the one prepared implicit solve template; record each separately.
    methods = {
        method: _method_record(method, capacity, seed, config, analytic)
        for method in ("forward-backward-euler", "ars-222", "ars-443")
    }
    full_step = moving_recorder.run(
        "solve",
        lambda: _COMPILED_STEP(moving_plan, state, dt),
        scope="full step with status",
    )
    if not bool(np.asarray(full_step.evidence.successful)):
        raise AssertionError("Native moving surface step refused.")
    geometry = moving_recorder.run(
        "numeric-refresh",
        lambda: moving_plan.geometry(state.points, state.time + dt, None),
        scope="moving geometry refresh",
    )
    if not bool(np.asarray(geometry.successful)):
        raise AssertionError("Native moving numeric geometry refresh refused.")
    epoch_transition = moving_recorder.run(
        "epoch-commit",
        lambda: perform_epoch_transition(moving_plan, full_step.state, seed=seed),
        scope="transfer preparation and atomic historical commit",
    )
    _next_plan, epoch = epoch_transition
    if not epoch.successful:
        raise AssertionError("Native atomic historical epoch transfer refused.")
    shift_plan = prepare_shift_plan(moving_plan)
    shifted = moving_recorder.run(
        "transfer",
        lambda: perform_shift(shift_plan, full_step.state),
        scope="tangential surface shift through the native stage rule",
    )
    shifted_state, shift = shifted
    if not all(bool(np.asarray(item.successful)) for item in shift):
        raise AssertionError("Native tangential surface shift refused.")
    concentration_error = float(
        np.max(np.abs(np.asarray(full_step.state.concentration) - analytic))
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
    surface_phases = surface_recorder.record()
    moving_phases = moving_recorder.record()
    return {
        "capacity": capacity,
        "seed": seed,
        "dimension": 3,
        "domain": "unit-sphere-expanding-to-radius-1+0.2t",
        "reserved_working_set_bytes": reservation,
        "retained_bytes": None,
        "retained_visible_bytes": retained_visible,
        "retained_bytes_unavailable_reason": (
            "Visible PyTree arrays are measured; static preparation and "
            "compiled/provider cache payloads are not fully traversed"
        ),
        "surface": {
            "phases": surface_phases,
            **surface_execution,
            "laplace_relative_error": surface_error,
        },
        "moving": {
            "phases": moving_phases,
            "methods": methods,
            "concentration_error": concentration_error,
            "conservation_defect": conservation,
            "epoch_conservation_defect": epoch_conservation,
            "active_points": state.points.shape[0],
        },
        "oracle": "independent-host-NumPy sphere eigenfunction and exp(-0.1t)/(1+0.2t)^2 dilution",
    }


def run(config: MeshfreeConfig = MeshfreeConfig(dimension=3), /) -> dict[str, Any]:
    configure_precision(config, supported=("float64",))
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
