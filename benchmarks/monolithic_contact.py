#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Bounded ABD-wall capacity campaign; timings include explicit synchronization.

Run: JAX_ENABLE_X64=1 python -m benchmarks.monolithic_contact --contended
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from examples.monolithic_contact import make_case
from phydrax.applications.contact import (
    ContactSolvePolicy,
    initialize_monolithic_contact_state,
    MonolithicContactPlan,
    MonolithicContactState,
    prepare_monolithic_contact_step,
    solve_monolithic_contact_step,
)
from phydrax.discretization.contact import collision_free_step_limit
from phydrax.nonlinear import NonlinearTermination

from ._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)


def prepare_benchmark_case(
    bodies: int, scale: int, /
) -> tuple[MonolithicContactPlan, MonolithicContactState]:
    original, initial = make_case(bodies, scale, friction=True)
    # Millisecond inertia plus the 0.1 mm/s sticking regularization amplify
    # position errors into force residuals. Use a reference-length-appropriate
    # step threshold, not looser force/lag/safety acceptance tolerances.
    policy = ContactSolvePolicy(
        merit="residual",
        termination=NonlinearTermination(absolute_step=1.0e-16, relative_step=1.0e-13),
        maximum_lag_iterations=12,
        lag_tolerance=1.0e-10,
    )
    plan = MonolithicContactPlan(
        original.participants,
        original.contact_scene,
        original.contact,
        original.search,
        original.ccd,
        geometry_owner_indices=original.geometry_owner_indices,
        fixed_geometry_states=original.fixed_geometry_states,
        friction=original.friction,
        policy=policy,
    )
    return plan, initialize_monolithic_contact_state(
        plan, initial.coordinates, initial.histories
    )


def _phase(operation: Callable[[], Any], repeats: int, /) -> dict[str, object]:
    """Host-prepared phases have no claimed compiler-only timing."""
    _, first = measure_synchronized(operation)
    _, steady = measure_repeated(operation, warmup=1, repeats=repeats)
    return {"first_seconds": first, "steady": steady.to_seconds_dict()}


def _compiled_phase(
    action: Callable[[Array], Any], value: Array, repeats: int, /
) -> dict[str, object]:
    lowered = jax.jit(action)
    executable, compilation = measure_lower_and_compile(
        lambda: lowered.lower(value), lambda stage: stage.compile()
    )
    _, first = measure_synchronized(lambda: executable(value))
    _, steady = measure_repeated(lambda: executable(value), warmup=1, repeats=repeats)
    compiler = compiler_evidence(
        executable.cost_analysis(), executable.memory_analysis(), source="xla"
    )
    return {
        "compilation": asdict(compilation),
        "first_seconds": first,
        "steady": steady.to_seconds_dict(),
        "compiler": asdict(compiler),
    }


def _case(bodies: int, scale: int, repeats: int, /) -> dict[str, object]:
    (plan, accepted), construction = measure_synchronized(
        lambda: prepare_benchmark_case(bodies, scale)
    )
    prepared, preparation = measure_synchronized(
        lambda: prepare_monolithic_contact_step(plan, accepted, 0.001)
    )
    participant = plan.participants[0]
    state = accepted.coordinates[0]
    fixed = plan.fixed_geometry_states[1]
    if fixed is None:
        raise ValueError("Benchmark wall must have an explicit fixed state.")
    history = accepted.histories[0]
    scene = plan.contact_scene
    states = (state, fixed)
    positions = scene.positions(states)
    direction = jnp.full_like(state, 1.0e-4)
    end = scene.positions((state + direction, fixed))
    friction = plan.friction
    if friction is None:
        raise ValueError("Benchmark requires a prepared friction owner.")
    lag = friction.build_state(positions, prepared.candidate_epoch)

    def participant_action(value: Array) -> Any:
        return participant.evaluate(
            value, history, state, accepted.time, prepared.step_size, prepared.args
        )

    def contact_action(value: Array) -> Any:
        current_states = (value, fixed)
        return plan.contact.evaluate(
            scene.positions(current_states),
            prepared.candidate_epoch,
            participant_states=current_states,
        )

    def global_action(value: Array) -> Any:
        return prepared.problem.residual((value,), prepared.args)

    def jvp_action(value: Array) -> Any:
        return prepared.residual_jvp((value,), (direction,), prepared.args)

    def bounds_action(value: Array) -> Any:
        current = scene.positions((value, fixed))
        return (
            collision_free_step_limit(
                plan.ccd, scene, prepared.candidate_epoch, current, end
            ),
            participant.step_limit(
                value,
                direction,
                state,
                accepted.time,
                prepared.step_size,
                prepared.args,
            ),
        )

    def refresh_action(value: Array) -> Any:
        refreshed = friction.build_state(
            scene.positions((value, fixed)), prepared.candidate_epoch
        )
        return refreshed, friction.lag_residual(lag, refreshed)

    phases = {
        "prepare_search": _phase(
            lambda: prepare_monolithic_contact_step(plan, accepted, 0.001), repeats
        ),
        "participant_residual": _compiled_phase(participant_action, state, repeats),
        "contact_pullback": _compiled_phase(contact_action, state, repeats),
        "global_residual": _compiled_phase(global_action, state, repeats),
        "ccd_domain_bounds": _compiled_phase(bounds_action, state, repeats),
        "global_jvp": _compiled_phase(jvp_action, state, repeats),
        "native_newton_gmres_line_search": _phase(
            lambda: plan.policy.method.solve(
                prepared.problem,
                accepted.coordinates,
                termination=plan.policy.termination,
                args=prepared.args,
            ),
            repeats,
        ),
        "friction_refresh": _compiled_phase(refresh_action, state, repeats),
        "friction_evaluation": _phase(
            lambda: friction.evaluate(
                scene.velocities(states, (direction, jnp.zeros_like(fixed))),
                lag,
                participant_states=states,
            ),
            repeats,
        ),
        "prepared_solve_transaction": _phase(
            lambda: solve_monolithic_contact_step(prepared), repeats
        ),
        "full_step_prepare_solve_commit": _phase(
            lambda: solve_monolithic_contact_step(
                prepare_monolithic_contact_step(plan, accepted, 0.001)
            ).commit(),
            repeats,
        ),
    }
    result = solve_monolithic_contact_step(prepared)
    if not bool(result.accepted):
        raise RuntimeError(
            f"Accepted-step benchmark refused: reasons={int(result.rejection_reasons)}, "
            f"status={int(result.nonlinear.status)}, "
            f"residual={float(result.diagnostics.final_residual_norm)}."
        )
    # Separate blocking overhead from kernel+blocking wall times; this is not a
    # subtraction-based estimate of asynchronous device execution.
    _, synchronization = measure_synchronized(lambda: positions)
    return {
        "bodies": bodies,
        "dofs": 12 * bodies,
        "capacity_scale": scale,
        "capacities": {
            "edge_vertex": plan.search.edge_vertex_capacity,
            "edge_edge": plan.search.edge_edge_capacity,
            "face_vertex": plan.search.face_vertex_capacity,
        },
        "solve_policy": {
            "merit": plan.policy.merit,
            "absolute_residual": plan.policy.termination.absolute_residual,
            "relative_residual": plan.policy.termination.relative_residual,
            "absolute_step": plan.policy.termination.absolute_step,
            "relative_step": plan.policy.termination.relative_step,
            "maximum_lag_iterations": plan.policy.maximum_lag_iterations,
            "lag_tolerance": plan.policy.lag_tolerance,
            "step_size": float(prepared.step_size),
        },
        "construction_seconds": construction,
        "initial_prepare_search_seconds": preparation,
        "synchronization_only_seconds": synchronization,
        "phases": phases,
        "capacity_memory": {
            "plan_logical_array_bytes": logical_array_bytes(plan),
            "epoch_logical_array_bytes": logical_array_bytes(prepared.candidate_epoch),
            "accepted_logical_array_bytes": logical_array_bytes(accepted),
            "result_logical_array_bytes": logical_array_bytes(result),
            "peak_device_memory": None,
            "peak_device_memory_reason": "Logical payload is not a peak allocation measurement.",
        },
        "accepted": bool(result.accepted),
        "rejection_reasons": int(result.rejection_reasons),
        "candidate_count": int(result.diagnostics.candidate_count),
        "overflow_count": int(result.diagnostics.overflow_count),
        "newton_iterations": int(result.diagnostics.iterations),
        "linear_iterations": int(result.diagnostics.linear_iterations),
        "jvp_actions": int(result.diagnostics.jvp_actions),
        "line_search_evaluations": int(result.diagnostics.line_search_evaluations),
        "lag_iterations": int(result.diagnostics.lag_iterations),
        "lag_residual": float(result.diagnostics.lag_residual),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bodies", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--capacity-scales", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--contended", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/monolithic_contact.json")
    )
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("Repeats must be positive.")
    cases = [_case(b, s, args.repeats) for b in args.bodies for s in args.capacity_scales]
    identity = capture_benchmark_identity(
        Path(__file__).resolve().parent.parent, Path(__file__).resolve(), cases[0].keys()
    )
    report = {
        "kind": "monolithic-contact-capacity-campaign",
        "identity": identity.to_dict(),
        "environment": capture_environment().to_dict(),
        "contended": args.contended,
        "cases": cases,
        "scope": "Bounded ABD/fixed-wall timings; no qualification or performance threshold claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
