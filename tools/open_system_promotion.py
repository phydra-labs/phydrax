#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from time import perf_counter

import jax
import jax.numpy as jnp

import phydrax as phx


def run_promotion(*, output: str | None = None):
    started = perf_counter()
    gaussian_problem = phx.solver.damped_thermal_oscillator(0.4, 1.0)
    gaussian = phx.solver.solve_gaussian_lindblad(
        gaussian_problem, step_size=0.02, steps=5
    )
    jump_problem = phx.solver.amplitude_damping_trajectory_problem(
        1.0, jnp.asarray([0.0j, 1.0 + 0.0j])
    )
    trajectory = phx.solver.solve_event_driven_quantum_jump(
        jump_problem,
        jax.random.PRNGKey(17),
        step_size=0.05,
        steps=10,
        maximum_events=8,
    )
    density = jnp.asarray([[0.6 + 0j, 0j], [0j, 0.4 + 0j]])
    comparison = phx.solver.spin_boson_dephasing_comparison(
        density, heom_depth=1, step_size=0.002, steps=2
    )
    runtime = perf_counter() - started
    spin_bures = comparison.bures_distance[-1]
    root_residual = jnp.max(
        jnp.where(
            trajectory.events.active,
            trajectory.events.root_residuals,
            0.0,
        )
    )
    approximation = phx.operators.quantum.OpenSystemApproximationEvidence(
        "promotion-suite",
        (
            phx.operators.quantum.ApproximationAxis(
                "trajectory-step", 0.05, units="time"
            ),
            phx.operators.quantum.ApproximationAxis("heom-depth", 1),
            phx.operators.quantum.ApproximationAxis("memory-step", 0.002, units="time"),
        ),
        (
            phx.operators.quantum.ApproximationQuantity(
                "maximum-event-root-residual",
                root_residual,
                1e-5,
                units="dimensionless",
                norm_id="maximum",
                estimate_kind="bound",
            ),
            phx.operators.quantum.ApproximationQuantity(
                "spin-boson-final-bures",
                spin_bures,
                1.0,
                units="distance-squared",
                norm_id="bures",
                estimate_kind="estimate",
            ),
        ),
        execution_valid=gaussian.valid & trajectory.valid & comparison.valid,
    )
    physicality = phx.operators.quantum.OpenSystemPhysicalityEvidence(
        positivity_margin=jnp.min(gaussian.uncertainty_margins),
        status="unknown",
    )
    policy = phx.operators.quantum.OpenSystemPromotionPolicy(
        ("trajectory-step", "heom-depth", "memory-step"),
        ("maximum-event-root-residual", "spin-boson-final-bures"),
        require_physicality=True,
        policy_id="open-system-promotion-v1",
    )
    decision = phx.operators.quantum.evaluate_open_system_promotion(
        policy,
        approximation,
        physicality,
        execution_success=gaussian.valid & trajectory.valid & comparison.valid,
        capacity_exhausted=jnp.sum(trajectory.events.active) >= 8,
        archive_verified=output is not None,
    )
    status = "success" if bool(decision.promoted) else "not-promoted"
    summary = {
        "status": status,
        "runtime_seconds": runtime,
        "gaussian_minimum_uncertainty": float(jnp.min(gaussian.uncertainty_margins)),
        "trajectory_event_count": int(jnp.sum(trajectory.events.active)),
        "maximum_event_root_residual": float(root_residual),
        "spin_boson_final_bures": float(spin_bures),
    }
    if output is not None:
        phx.solver.write_open_system_artifact(
            output,
            campaign_id="open-system-promotion-v1",
            representation_id="gaussian+trajectory+heom+memory",
            problem_id="promotion-suite",
            plan_id="promotion-suite-v1",
            precision=str(gaussian.means.dtype),
            backend=jax.default_backend(),
            status=status,
            thresholds={
                "minimum_uncertainty": -1e-8,
                "maximum_spin_boson_bures": 1.0,
            },
            approximation_axes={
                "trajectory_step": 0.05,
                "heom_depth": 1,
                "memory_step": 0.002,
            },
            semantic_rng_schema={"trajectory_seed": 17},
            arrays={
                "gaussian_means": gaussian.means,
                "trajectory_states": trajectory.states,
                "spin_boson_heom": comparison.heom_states,
                "spin_boson_memory": comparison.memory_states,
            },
            extra_manifest={
                "summary": summary,
                "runner_id": "tools.open_system_promotion:v2",
                "runtime_seconds": runtime,
                "static_shapes": {
                    "gaussian_means": list(gaussian.means.shape),
                    "trajectory_states": list(trajectory.states.shape),
                    "spin_boson_heom": list(comparison.heom_states.shape),
                },
                "capacity": {"maximum_events": 8},
            },
        )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str)
    arguments = parser.parse_args()
    summary = run_promotion(output=arguments.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
