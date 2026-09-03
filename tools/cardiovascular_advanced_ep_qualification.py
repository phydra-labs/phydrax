"""Physics qualification for cardiovascular conduction, PMJ, and bidomain routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp

from benchmarks._runtime import capture_environment
from phydrax.applications.cardiovascular.electrophysiology._bidomain import (
    BidomainFEMPlan,
    HeartOnlyBidomainRoute,
    HeartTorsoBidomainRoute,
    initialize_bidomain_state,
    step_bidomain,
    step_proportional_monodomain_limit,
    zero_bidomain_inputs,
)
from phydrax.applications.cardiovascular.electrophysiology._conduction_network import (
    initialize_purkinje_state,
    make_purkinje_stimulus_batch,
    propagate_purkinje,
    PurkinjeEventKind,
    PurkinjeNetworkPlan,
)
from phydrax.applications.cardiovascular.electrophysiology._eikonal import (
    AnisotropicEikonalPlan,
    FiniteElementEikonalRoute,
    GraphEikonalRoute,
    solve_anisotropic_eikonal,
)
from phydrax.applications.cardiovascular.electrophysiology._pacing import (
    evaluate_pmj_exchange,
    PMJExchangePlan,
    schedule_pmj_activations,
)


def _prepared_heart_only():
    return BidomainFEMPlan(
        HeartOnlyBidomainRoute(),
        jnp.asarray((10, 20, 30)),
        jnp.asarray((100, 101)),
        jnp.asarray(((0.0,), (1.0,), (2.0,))),
        jnp.asarray(((0, 1), (1, 2))),
        jnp.asarray(((1.0,),)),
        jnp.asarray(((2.0,),)),
        dt_ms=0.1,
        membrane_capacitance_uF_per_mm3=1.0,
        residual_tolerance=2.0e-5,
        gauge_tolerance_mV=2.0e-5,
        source_compatibility_tolerance_uA=2.0e-5,
    ).prepare()


def qualification():
    graph = GraphEikonalRoute(
        jnp.asarray((10, 20, 30)),
        jnp.asarray((100, 101)),
        jnp.asarray(((0, 1), (1, 2))),
    )
    eikonal = solve_anisotropic_eikonal(
        AnisotropicEikonalPlan(
            graph,
            jnp.asarray(((0.0, 0.0), (2.0, 0.0), (5.0, 0.0))),
            jnp.asarray(((4.0, 0.0), (0.0, 1.0))),
        ).prepare(),
        jnp.asarray((0,)),
        jnp.asarray((0.0,)),
    )
    analytic_error = float(
        jnp.max(jnp.abs(eikonal.arrival_time_ms - jnp.asarray((0.0, 1.0, 2.5))))
    )
    fem_route = FiniteElementEikonalRoute(
        jnp.asarray((10, 20, 30)),
        jnp.asarray((200,)),
        jnp.asarray(((0, 1, 2),)),
    )
    fem_eikonal = solve_anisotropic_eikonal(
        AnisotropicEikonalPlan(
            fem_route,
            jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.5, 1.0))),
            jnp.eye(2),
        ).prepare(),
        jnp.asarray((0, 1)),
        jnp.asarray((0.0, 0.0)),
    )
    fem_analytic_error = float(
        jnp.max(jnp.abs(fem_eikonal.arrival_time_ms - jnp.asarray((0.0, 0.0, 1.0))))
    )

    network = PurkinjeNetworkPlan(
        jnp.asarray((10, 20)),
        jnp.asarray((100,)),
        jnp.asarray(((0, 1),)),
        jnp.asarray((2.0,)),
        10.0,
        event_capacity=8,
        stimulus_capacity=2,
    )
    collision = propagate_purkinje(
        network,
        initialize_purkinje_state(network),
        make_purkinje_stimulus_batch(network, (1, 2), (0, 1), (0.0, 0.0)),
    )
    collision_mask = collision.events.active & (
        collision.events.kind == int(PurkinjeEventKind.WAVE_COLLISION)
    )
    collision_time_error = float(
        jnp.max(
            jnp.where(
                collision_mask,
                jnp.abs(collision.events.time_ms - 1.0),
                0.0,
            )
        )
    )

    pmj = PMJExchangePlan(
        jnp.asarray((900,)),
        jnp.asarray((1,)),
        jnp.asarray((0,)),
        jnp.asarray((0.5,)),
        jnp.asarray((0.2,)),
        purkinje_plan=network,
        tissue_node_count=1,
        event_capacity=4,
    )
    exchange = evaluate_pmj_exchange(
        pmj,
        jnp.asarray((0.0, 10.0)),
        jnp.asarray((-10.0,)),
    )
    scheduled = schedule_pmj_activations(pmj, collision, jnp.asarray((-jnp.inf,)))
    scheduled_mask = scheduled.activations.active & scheduled.activations.accepted
    pmj_timing_error = float(
        jnp.max(
            jnp.where(
                scheduled_mask,
                jnp.abs(scheduled.activations.activation_time_ms - 0.5),
                0.0,
            )
        )
    )

    prepared = _prepared_heart_only()
    state = initialize_bidomain_state(prepared, jnp.asarray((0.0, 1.0, 0.0)))
    inputs = zero_bidomain_inputs(prepared)
    bidomain = step_bidomain(prepared, state, inputs)
    monodomain = step_proportional_monodomain_limit(prepared, state, inputs, 2.0)
    limit_error = float(
        jnp.max(
            jnp.abs(
                bidomain.state.transmembrane_voltage_mV
                - monodomain.transmembrane_voltage_mV
            )
        )
    )

    torso_route = HeartTorsoBidomainRoute(
        jnp.asarray((1000, 1001)),
        jnp.asarray((1100,)),
        jnp.asarray(((1.0,), (2.0,))),
        jnp.asarray(((0, 1),)),
        jnp.asarray(((0.5,),)),
        jnp.asarray((1200,)),
        jnp.asarray(((1, 0),)),
        jnp.asarray((3.0,)),
    )
    torso = BidomainFEMPlan(
        torso_route,
        jnp.asarray((10, 20)),
        jnp.asarray((100,)),
        jnp.asarray(((0.0,), (1.0,))),
        jnp.asarray(((0, 1),)),
        jnp.asarray(((1.0,),)),
        jnp.asarray(((2.0,),)),
        dt_ms=0.1,
        membrane_capacitance_uF_per_mm3=1.0,
        residual_tolerance=2.0e-5,
        gauge_tolerance_mV=2.0e-5,
        source_compatibility_tolerance_uA=2.0e-5,
    ).prepare()
    torso_step = step_bidomain(
        torso,
        initialize_bidomain_state(torso, jnp.asarray((0.0, 1.0))),
        zero_bidomain_inputs(torso),
    )

    cases = {
        "anisotropic_eikonal_analytic_travel": {
            "maximum_arrival_error_ms": analytic_error,
            "bellman_residual_ms": float(eikonal.evidence.maximum_bellman_residual_ms),
            "passed": bool(eikonal.evidence.successful) and analytic_error < 1.0e-6,
        },
        "finite_element_eikonal_affine_simplex": {
            "maximum_arrival_error_ms": fem_analytic_error,
            "update_residual_ms": float(fem_eikonal.evidence.maximum_bellman_residual_ms),
            "passed": bool(fem_eikonal.evidence.successful)
            and fem_analytic_error < 1.0e-6,
        },
        "purkinje_antiparallel_collision": {
            "collision_count": int(collision.evidence.collision_count),
            "collision_time_error_ms": collision_time_error,
            "deterministic_order": bool(collision.evidence.deterministic_order),
            "passed": bool(collision.evidence.successful)
            and int(collision.evidence.collision_count) == 1
            and collision_time_error < 1.0e-6,
        },
        "pmj_current_conservation": {
            "net_exchange_current_uA": float(exchange.evidence.net_exchange_current_uA),
            "scheduled_event_count": int(scheduled.evidence.event_count),
            "activation_timing_error_ms": pmj_timing_error,
            "passed": bool(exchange.evidence.successful)
            and bool(scheduled.evidence.successful)
            and int(scheduled.evidence.accepted_count) == 1
            and pmj_timing_error < 1.0e-7
            and abs(float(exchange.evidence.net_exchange_current_uA)) < 1.0e-7,
        },
        "bidomain_monodomain_limit": {
            "maximum_vm_error_mV": limit_error,
            "gauge_residual": float(bidomain.evidence.gauge.constraint_residual),
            "ungauged_nullspace_residual": float(
                bidomain.evidence.gauge.ungauged_nullspace_residual
            ),
            "passed": bool(bidomain.evidence.successful)
            and bool(monodomain.evidence.successful)
            and limit_error < 2.0e-5,
        },
        "heart_torso_interface": {
            "interface_current_norm_uA": float(
                torso_step.evidence.interface.interface_current_norm_uA
            ),
            "interface_flux_balance_error_uA": float(
                torso_step.evidence.interface.flux_balance_error_uA
            ),
            "gauge_residual": float(torso_step.evidence.gauge.constraint_residual),
            "passed": bool(torso_step.evidence.successful)
            and bool(torso_step.evidence.interface.supported)
            and float(torso_step.evidence.interface.flux_balance_error_uA) < 1.0e-7,
        },
    }
    passed = all(bool(case["passed"]) for case in cases.values())
    return {
        "environment": capture_environment().to_dict(),
        "evidence_levels": {
            "invariant_complete": bool(
                cases["pmj_current_conservation"]["passed"]
                and cases["heart_torso_interface"]["passed"]
            ),
            "physics_qualified": bool(
                cases["anisotropic_eikonal_analytic_travel"]["passed"]
                and cases["finite_element_eikonal_affine_simplex"]["passed"]
                and cases["bidomain_monodomain_limit"]["passed"]
            ),
            "differentiation_qualified": bool(
                eikonal.evidence.fixed_topology_derivative_valid
            ),
            "execution_qualified": passed,
            "deployment_qualified": False,
        },
        "cases": cases,
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/cardiovascular_advanced_ep_qualification.json"),
    )
    arguments = parser.parse_args()
    payload = qualification()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
