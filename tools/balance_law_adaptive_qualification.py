#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _problem():
    count = 32
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(count, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    system = phx.equations.EulerSystem()
    discretization = phx.discretization.FiniteVolumePlan(
        grid,
        component_names=system.component_names,
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "adaptive-balance-law-qualification",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.HLLCFluxPlan(),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem,
        discretization,
        method,
    ).dynamics
    transport = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(cfl=0.3, maximum_retries=0),
    )
    x = grid.structured_axes[0].interval_centers
    density = 1.0 + 0.02 * jnp.sin(2.0 * jnp.pi * x)
    primitive = jnp.stack(
        (density, jnp.zeros_like(density), jnp.ones_like(density)),
        axis=-1,
    )
    transport_state = transport.initialize_state(
        system.primitive_to_conserved(primitive),
        0.0,
        1e-3,
    )
    balance_transport = phx.solver.prepare_balance_law_transport(transport)
    gravity = phx.solver.NewtonianSelfGravityPlan(
        0.1,
        gravity_argument="gravity",
    ).prepare(balance_transport)
    runtime = phx.solver.PreparedBalanceLawRuntime(balance_transport, (gravity,))
    initial = runtime.initialize_state(transport_state)
    adaptive = phx.solver.AdaptiveBalanceLawRolloutPlan(
        runtime,
        8e-3,
        phx.solver.BalanceLawAdaptivePolicy(
            8,
            maximum_retries=2,
            safety_factor=1.0,
            growth_factor=1.0,
        ),
    )
    realized = adaptive.rollout(initial, {"gravity": jnp.asarray(0.1)})
    return runtime, initial, realized


def _state_bytes(state) -> int:
    return sum(
        int(np.asarray(jax.device_get(leaf)).nbytes)
        for leaf in jax.tree.leaves(state)
        if eqx_is_array(leaf)
    )


def eqx_is_array(value) -> bool:
    return isinstance(value, jax.Array | np.ndarray)


def _mode_report(runtime, initial, realized, mode, block_size):
    replay = phx.solver.FiniteVolumeReplayPolicy(mode, block_size=block_size)
    scheduled = phx.solver.ScheduledBalanceLawRolloutPlan.from_realized_mesh(
        runtime,
        realized.realized_mesh,
        replay=replay,
    )

    def loss(gravity):
        result = scheduled.rollout(initial, {"gravity": gravity})
        momentum = result.final_state.transport_state.cell_average()[..., 1]
        return jnp.sum(momentum**2)

    value_and_gradient = jax.jit(jax.value_and_grad(loss))
    parameter = jnp.asarray(0.1)
    compile_started = time.perf_counter()
    compiled = value_and_gradient.lower(parameter).compile()
    compile_seconds = time.perf_counter() - compile_started
    memory = compiled.memory_analysis()
    executed = compiled(parameter)
    jax.block_until_ready(executed)
    samples = []
    for _ in range(3):
        started = time.perf_counter()
        executed = compiled(parameter)
        jax.block_until_ready(executed)
        samples.append(time.perf_counter() - started)
    value, gradient = executed
    epsilon = jnp.asarray(1e-4, dtype=parameter.dtype)
    finite_difference = (loss(parameter + epsilon) - loss(parameter - epsilon)) / (
        2.0 * epsilon
    )
    interval_count = int(np.asarray(realized.realized_mesh.count))
    state_bytes = _state_bytes(initial)
    checkpoint_count = (
        interval_count + 1
        if mode == "full"
        else 1
        if mode == "step"
        else math.ceil(interval_count / int(block_size)) + 1
    )
    return {
        "compile_seconds": compile_seconds,
        "execution_seconds_median": float(np.median(samples)),
        "loss": float(value),
        "gradient": float(gradient),
        "finite_difference_gradient": float(finite_difference),
        "gradient_residual": float(jnp.abs(gradient - finite_difference)),
        "compiled_argument_bytes": memory.argument_size_in_bytes,
        "compiled_output_bytes": memory.output_size_in_bytes,
        "compiled_temporary_bytes": memory.temp_size_in_bytes,
        "compiled_alias_bytes": memory.alias_size_in_bytes,
        "state_bytes": state_bytes,
        "estimated_checkpoint_state_bytes": state_bytes * checkpoint_count,
        "checkpoint_count": checkpoint_count,
    }


def main() -> None:
    runtime, initial, realized = _problem()
    if not bool(realized.completed):
        raise RuntimeError("Adaptive qualification did not reach its final time.")
    modes = {
        "full": _mode_report(runtime, initial, realized, "full", None),
        "step": _mode_report(runtime, initial, realized, "step", None),
        "block": _mode_report(runtime, initial, realized, "block", 3),
    }
    gradients = jnp.asarray([record["gradient"] for record in modes.values()])
    report = {
        "adaptive": {
            "attempt_count": int(realized.journal.attempt_count),
            "accepted_count": int(realized.journal.accepted_count),
            "completed": bool(realized.completed),
            "final_time": float(realized.final_state.time),
            "minimum_stability_margin": float(
                jnp.nanmin(realized.journal.stability_margins)
            ),
        },
        "replay_modes": modes,
        "replay_gradient_spread": float(jnp.max(gradients) - jnp.min(gradients)),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
