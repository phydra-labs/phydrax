#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
import time

import jax
import jax.numpy as jnp

import phydrax as phx


def _case(cell_count: int, steps: int):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(cell_count, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    system = phx.equations.EulerSystem()
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "differentiable-rollout-benchmark",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.MUSCLReconstruction(),
        phx.discretization.HLLCFluxPlan(),
        positivity=phx.discretization.ConvexStateLimiterPlan(),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem, discretization, method
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(cfl=0.3, maximum_retries=0),
    )
    temporal_mesh = phx.discretization.TemporalMesh.uniform(
        0.0, 0.01, steps, role="internal"
    )
    x = grid.structured_axes[0].interval_centers
    primitive = jnp.stack(
        (
            1.0 + 0.02 * jnp.sin(2.0 * jnp.pi * x),
            0.05 * jnp.cos(2.0 * jnp.pi * x),
            jnp.ones_like(x),
        ),
        axis=-1,
    )
    initial = runtime.initialize_state(
        system.primitive_to_conserved(primitive),
        temporal_mesh.t0,
        temporal_mesh.widths[0],
    )
    return runtime, temporal_mesh, initial


def _measure(runtime, mesh, initial, replay):
    plan = phx.solver.ScheduledFiniteVolumeRolloutPlan(
        runtime, mesh, replay=replay, retention="final"
    )

    def objective(content):
        state = phx.solver.FiniteVolumeRuntimeState(
            initial.content_state.with_content(content),
            initial.topology_journal,
            initial.step_size,
        )
        result = plan.rollout(state)
        return jnp.sum(result.final_state.content_state.conservative_content[..., -1])

    value_and_gradient = jax.jit(jax.value_and_grad(objective))
    value, gradient = value_and_gradient(initial.content_state.conservative_content)
    jax.block_until_ready((value, gradient))
    start = time.perf_counter()
    value, gradient = value_and_gradient(initial.content_state.conservative_content)
    jax.block_until_ready((value, gradient))
    duration = time.perf_counter() - start
    result = plan.rollout(initial)
    jax.block_until_ready(result.final_state.content_state.conservative_content)
    return {
        "value": float(value),
        "gradient_norm": float(jnp.linalg.norm(gradient)),
        "duration_seconds": duration,
        "retained_state_values": int(result.retained_states.size),
        "final_state": result.final_state.content_state.conservative_content,
    }


def main() -> None:
    runtime, mesh, initial = _case(64, 8)
    measured = {
        "full": _measure(
            runtime, mesh, initial, phx.solver.FiniteVolumeReplayPolicy("full")
        ),
        "step": _measure(
            runtime, mesh, initial, phx.solver.FiniteVolumeReplayPolicy("step")
        ),
        "block": _measure(
            runtime,
            mesh,
            initial,
            phx.solver.FiniteVolumeReplayPolicy("block", block_size=3),
        ),
    }
    reference = measured["full"]["final_state"]
    report = {
        "cell_count": 64,
        "step_count": 8,
        "dtype": str(reference.dtype),
        "backend": jax.default_backend(),
        "modes": {
            name: {key: value for key, value in result.items() if key != "final_state"}
            for name, result in measured.items()
        },
        "step_final_defect": float(
            jnp.max(jnp.abs(measured["step"]["final_state"] - reference))
        ),
        "block_final_defect": float(
            jnp.max(jnp.abs(measured["block"]["final_state"] - reference))
        ),
        "finite": bool(
            all(
                jnp.isfinite(result["value"]) and jnp.isfinite(result["gradient_norm"])
                for result in measured.values()
            )
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
