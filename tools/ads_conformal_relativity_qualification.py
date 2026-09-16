#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact AdS zero quantities, gauge, boundary, scalar modes, and observables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from phydrax.applications import numerical_relativity as nr


def run_qualification() -> dict[str, object]:
    system = nr.ConformalEinsteinSystem(
        -3.0,
        scalar_curvature_gauge=-12.0,
        residual_tolerance=1e-12,
    )
    state, derivatives = nr.exact_ads_conformal_reference(system, (2, 2, 2))
    zero = nr.evaluate_conformal_einstein_zero_quantities(system, state, derivatives)

    gauge_plan = nr.GeneralizedWaveGaugePlan(
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        transition_time=1.0,
        transition_time_width=0.2,
        transition_radius=0.5,
        transition_radius_width=0.1,
        damping=0.1,
        scalar_curvature_gauge=-12.0,
    )
    radius = jnp.linspace(0.0, 1.0, 16)
    source = gauge_plan.source(2.0, radius)
    gauge = nr.evaluate_generalized_wave_gauge(
        gauge_plan,
        -source,
        -12.0 * jnp.ones_like(radius),
        2.0,
        radius,
        tolerance=1e-12,
    )

    scalar_plan = nr.ConformalAdSScalarPlan(
        65,
        time_step=0.002,
        maximum_steps=200,
    )
    scalar_initial, frequency = nr.conformal_scalar_normal_mode(scalar_plan, 1)
    scalar_run = nr.run_conformal_ads_scalar(
        scalar_plan,
        scalar_initial,
        steps=100,
        energy_tolerance=2e-3,
    )
    exact_field = scalar_initial.field * jnp.cos(frequency * scalar_run.final_state.time)
    mode_error = jnp.linalg.norm(
        scalar_run.final_state.field - exact_field
    ) / jnp.maximum(1.0, jnp.linalg.norm(exact_field))

    defining = jnp.cos(scalar_plan.radial_points)
    boundary_field = 2.0 * defining + 3.0 * defining**2
    boundary_state = nr.ConformalAdSScalarState(
        scalar_plan,
        boundary_field.at[-1].set(0.0),
        jnp.zeros_like(boundary_field),
    )
    scalar_observable = nr.extract_ads_boundary_scalar(
        nr.AdSBoundaryScalarObservablePlan(
            1.0,
            2.0,
            fit_points=10,
            residual_tolerance=1e-10,
        ),
        scalar_plan,
        boundary_state,
    )
    boundary_metric = jnp.diag(jnp.asarray((-1.0, 1.0, 1.0)))
    stress = nr.evaluate_holographic_stress_tensor(
        nr.HolographicStressTensorPlan(
            3,
            1.0,
            "declared-fefferman-graham-control",
        ),
        jnp.diag(jnp.asarray((2.0, 1.0, 1.0))),
        jnp.zeros((3, 3)),
        boundary_metric,
        jnp.zeros((3,)),
    )
    successful = bool(
        zero.accepted
        and gauge.accepted
        and scalar_run.evidence.accepted
        and scalar_observable.accepted
        and stress.accepted
    )
    return {
        "kind": "ads-conformal-relativity-candidate-qualification",
        "profiles": [
            profile.to_record() for profile in nr.ads_conformal_candidate_profiles()
        ],
        "case": {
            "system_id": system.system_id,
            "state_id": state.state_id,
            "gauge_id": gauge_plan.gauge_id,
            "scalar_plan_id": scalar_plan.plan_id,
            "scalar_observable_plan_id": scalar_observable.plan_id,
            "stress_plan_id": stress.plan_id,
        },
        "raw": {
            "zero_quantity_component_maxima": np.asarray(zero.component_maxima).tolist(),
            "scalar_initial_field": np.asarray(scalar_initial.field).tolist(),
            "scalar_final_field": np.asarray(scalar_run.final_state.field).tolist(),
            "scalar_exact_field": np.asarray(exact_field).tolist(),
        },
        "criteria": {
            "maximum_zero_quantity": float(zero.maximum_residual),
            "maximum_wave_gauge_constraint": float(gauge.maximum_wave_constraint),
            "scalar_normal_mode_relative_error": float(mode_error),
            "scalar_relative_energy_drift": float(
                scalar_run.evidence.relative_energy_drift
            ),
            "scalar_boundary_residual": float(
                scalar_run.evidence.maximum_boundary_residual
            ),
            "fitted_source": float(scalar_observable.source_coefficient),
            "fitted_response": float(scalar_observable.response_coefficient),
            "fit_residual": float(scalar_observable.fit_residual),
            "stress_trace_residual": float(stress.trace_residual),
            "stress_divergence_residual": float(stress.divergence_residual),
        },
        "successful": successful,
        "claim": (
            "finite-exact-ads-and-fixed-background-scalar-reference-"
            "no-dynamical-gravity-or-holographic-renormalization-claim"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    encoded = json.dumps(run_qualification(), indent=2, sort_keys=True)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
