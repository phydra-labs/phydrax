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


def _problem():
    count = 3
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True) for _ in range(3)
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    system = phx.equations.IdealMHDSystem(3)
    discretization = phx.discretization.FiniteVolumePlan(
        grid,
        component_names=system.component_names,
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "balance-law-transport-qualification",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x", "y", "z")),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.HLLDFluxPlan(),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem,
        discretization,
        method,
    ).dynamics
    magnetic_flux = bridge.pack_face_flux(
        (
            jnp.full(grid.shape, 0.2),
            jnp.zeros(grid.shape),
            jnp.zeros(grid.shape),
        )
    )
    primitive = jnp.zeros(grid.shape + (8,))
    primitive = primitive.at[..., 0].set(1.0)
    primitive = primitive.at[..., 4].set(1.0)
    primitive = primitive.at[..., 5].set(0.2)
    full = system.primitive_to_conserved(primitive)
    spatial = phx.discretization.UpwindConstrainedTransportPlan(dynamics, bridge)
    integrator = phx.solver.ConstrainedMHDSSPRK3Plan(spatial, cfl=0.2)
    state = integrator.initialize(full, magnetic_flux, step_size=1e-4)
    transport = phx.solver.prepare_balance_law_transport(integrator)
    curve = phx.equations.TabulatedCoolingCurve(
        jnp.asarray([-6.0, 6.0]),
        jnp.asarray([-3.0, -3.0]),
        bounds_policy="power_law_extrapolate",
    )
    cooling = phx.solver.RadiativeCoolingProcessPlan(
        curve,
        accuracy_fraction=1.0,
        tolerance=1e-10,
    ).prepare(transport)
    runtime = phx.solver.PreparedBalanceLawRuntime(transport, (cooling,))
    initial = runtime.initialize_state(state)
    adaptive = phx.solver.AdaptiveBalanceLawRolloutPlan(
        runtime,
        4e-4,
        phx.solver.BalanceLawAdaptivePolicy(
            4,
            maximum_retries=1,
            safety_factor=1.0,
            growth_factor=1.0,
        ),
    )
    return transport, runtime, initial, adaptive


def _replay(runtime, initial, realized, mode, block_size):
    policy = phx.solver.FiniteVolumeReplayPolicy(mode, block_size=block_size)
    plan = phx.solver.ScheduledBalanceLawRolloutPlan.from_realized_mesh(
        runtime,
        realized.realized_mesh,
        replay=policy,
    )
    started = time.perf_counter()
    result = plan.rollout(initial)
    jax.block_until_ready(result.final_state.transport_state.cell_state)
    duration = time.perf_counter() - started
    reference = realized.final_state.transport_state
    final = result.final_state.transport_state
    return {
        "duration_seconds": duration,
        "cell_defect": float(jnp.max(jnp.abs(final.cell_state - reference.cell_state))),
        "magnetic_defect": float(
            jnp.max(jnp.abs(final.magnetic_flux - reference.magnetic_flux))
        ),
        "all_intervals_accepted": bool(jnp.all(result.accepted)),
        "retained_cell_values": int(result.retained_states.size),
        "retained_auxiliary_values": int(result.retained_transport_auxiliary.size),
    }


def main() -> None:
    transport, runtime, initial, adaptive = _problem()
    started = time.perf_counter()
    realized = adaptive.rollout(initial)
    jax.block_until_ready(realized.final_state.transport_state.cell_state)
    adaptive_duration = time.perf_counter() - started
    final_transport = realized.final_state.transport_state
    constraint = transport.integrator.spatial.magnetic_constraint(
        final_transport.magnetic_flux
    )
    report = {
        "adaptive": {
            "duration_seconds": adaptive_duration,
            "completed": bool(realized.completed),
            "attempt_count": int(realized.journal.attempt_count),
            "accepted_count": int(realized.journal.accepted_count),
            "final_time": float(final_transport.time),
            "magnetic_constraint_maximum": float(
                jnp.max(jnp.abs(constraint), initial=0.0)
            ),
        },
        "replay": {
            "full": _replay(runtime, initial, realized, "full", None),
            "step": _replay(runtime, initial, realized, "step", None),
            "block": _replay(runtime, initial, realized, "block", 2),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
