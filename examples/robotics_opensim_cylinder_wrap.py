#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json

import jax.numpy as jnp

from phydrax.applications.robotics import OpenSimCylinderRouteWrapPlan


def main() -> None:
    prepared = OpenSimCylinderRouteWrapPlan(32).prepare(
        jnp.zeros(3), jnp.asarray((0.0, 0.0, 1.0)), 1.0, 8.0
    )
    endpoints = jnp.asarray(((-2.0, 0.35, -0.8), (2.1, 0.65, 1.2)))
    velocity = jnp.asarray(((0.12, -0.04, 0.17), (-0.03, 0.08, -0.11)))
    source = prepared.initial_state()
    candidate = prepared.propose(source, endpoints)
    accepted = prepared.commit(candidate, source)
    fixed = prepared.evaluate_fixed_branch(accepted, endpoints)
    # Prescribed positive tension stands in for one native musculotendon owner;
    # it is not a provider raw force and the geometry generates no extra force.
    loads, power = prepared.tensile_force_pullback(
        accepted, endpoints, velocity, 120.0, force_owner="native-tension"
    )
    mirrored = endpoints.at[:, 1].multiply(-1.0)
    switch = prepared.propose(accepted, mirrored)
    transition_loads, transition_power = prepared.tensile_force_pullback(
        accepted, mirrored, velocity, 120.0, force_owner="native-tension"
    )
    payload = {
        "source_revision": fixed.evidence.source_revision,
        "source_sha256": fixed.evidence.source_sha256,
        "numerical_realization": "exact-unrolled-common-axial-slope",
        "source_executable_parity_claimed": False,
        "prepared_id": prepared.prepared_id,
        "successful": bool(candidate.successful & power.successful),
        "accepted_branch": int(accepted.branch),
        "accepted_steps": int(accepted.accepted_steps),
        "length_m": float(fixed.total_length_m),
        "surface_length_m": float(fixed.surface_length_m),
        "tangent_points_m": fixed.tangent_points_m.tolist(),
        "candidate_lengths_m": fixed.evidence.candidate_lengths_m.tolist(),
        "candidate_feasible": fixed.evidence.candidate_feasible.tolist(),
        "shortest_lateral_gap_m": float(fixed.evidence.shortest_lateral_gap_m),
        "tangent_direction_residual": float(fixed.evidence.tangent_direction_residual),
        "fixed_branch_gradient_supported": bool(
            fixed.evidence.fixed_branch_gradient_supported
        ),
        "endpoint_loads_N": loads.tolist(),
        "power_residual_W": float(power.power_residual_W),
        "proposed_switch_successful": bool(switch.successful),
        "proposed_mode_changed": bool(switch.evaluation.evidence.mode_changed),
        "uncommitted_switch_loads_N": transition_loads.tolist(),
        "uncommitted_switch_force_admitted": bool(transition_power.successful),
        "shortest_path_scope": "two-lateral-branches-not-finite-cap-or-rim",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["successful"] or payload["uncommitted_switch_force_admitted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
