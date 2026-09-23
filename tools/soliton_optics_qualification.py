#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Stationary kink and scalar-envelope fundamental-soliton evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from phydrax.applications.phase_field import (
    DoubleWellKinkPlan,
    solve_double_well_kink,
    stationary_soliton_candidate_profiles,
)
from phydrax.discretization import FourierAxisSpec, TensorGridPlan
from phydrax.equations import DoubleWellFreeEnergy
from phydrax.geometry import RigidFrame
from phydrax.optics.wave import (
    AdaptiveEnvelopePolicy,
    envelope_propagation_candidate_profiles,
    EnvelopeNonlinearResponsePlan,
    EnvelopePropagationPlan,
    PlaneFieldSpace,
    prepare_envelope_nonlinear_response,
    prepare_envelope_propagation,
    propagate_envelope,
    propagate_envelope_adaptive,
    PulseEnvelopeField,
    PulseTimeSpace,
)


def _optical_control():
    plane_grid = TensorGridPlan(
        (FourierAxisSpec(2), FourierAxisSpec(2)), axis_names=("u", "v")
    ).prepare(jnp.asarray([[-1.0, -1.0], [1.0, 1.0]]))
    plane = PlaneFieldSpace(plane_grid, RigidFrame.identity(3), "periodic-cell")
    time_grid = TensorGridPlan((FourierAxisSpec(256),), axis_names=("time",)).prepare(
        jnp.asarray([[-20.0], [20.0]])
    )
    time = PulseTimeSpace(time_grid, topology="periodic-cell")
    values = 1.0 / jnp.cosh(time.coordinates)
    field = PulseEnvelopeField(
        plane,
        time,
        jnp.broadcast_to(values.astype(jnp.complex128), plane.shape + time.shape),
        100.0,
        0.0,
        polarization="scalar",
    )
    response = prepare_envelope_nonlinear_response(
        EnvelopeNonlinearResponsePlan(
            1.0,
            source_id="normalized-fundamental-soliton-control",
        ),
        time,
    )
    prepared = prepare_envelope_propagation(
        EnvelopePropagationPlan(
            time,
            {2: -1.0},
            step_count=32,
            maximum_spectral_edge_fraction=1e-7,
            maximum_refinement_error=2e-5,
        ),
        response,
    )
    fixed = propagate_envelope(prepared, field, 0.2)
    adaptive = propagate_envelope_adaptive(
        prepared,
        field,
        0.2,
        AdaptiveEnvelopePolicy(
            relative_tolerance=2e-6,
            absolute_tolerance=1e-8,
            initial_step=0.02,
            minimum_step=1e-5,
            maximum_step=0.05,
            maximum_attempts=100,
        ),
    )
    return field, fixed, adaptive


def run_qualification() -> dict[str, object]:
    kink_plan = DoubleWellKinkPlan(
        jnp.linspace(-8.0, 8.0, 129),
        DoubleWellFreeEnergy(1.0),
        gradient_coefficient=1.0,
        maximum_newton_steps=24,
        residual_tolerance=1e-10,
    )
    kink = solve_double_well_kink(kink_plan)
    field, fixed, adaptive = _optical_control()
    fixed_intensity_error = jnp.max(
        jnp.abs(jnp.abs(fixed.field.values) ** 2 - jnp.abs(field.values) ** 2)
    )
    adaptive_intensity_error = jnp.max(
        jnp.abs(jnp.abs(adaptive.field.values) ** 2 - jnp.abs(field.values) ** 2)
    )
    profiles = (
        *stationary_soliton_candidate_profiles(),
        *envelope_propagation_candidate_profiles(),
    )
    successful = bool(
        kink.evidence.accepted
        and fixed.evidence.accepted
        and adaptive.adaptive.successful
    )
    return {
        "kind": "soliton-optics-candidate-qualification",
        "profiles": [profile.to_record() for profile in profiles],
        "case": {
            "kink_plan_id": kink_plan.plan_id,
            "envelope_prepared_id": fixed.prepared_id,
            "adaptive_policy_id": adaptive.adaptive.policy_id,
        },
        "raw": {
            "kink_coordinates": np.asarray(kink_plan.coordinates).tolist(),
            "kink_field": np.asarray(kink.field).tolist(),
            "kink_residual": np.asarray(kink.evidence.euler_lagrange_residual).tolist(),
            "fixed_final_intensity": np.asarray(
                jnp.abs(fixed.field.values[0, 0]) ** 2
            ).tolist(),
            "adaptive_final_intensity": np.asarray(
                jnp.abs(adaptive.field.values[0, 0]) ** 2
            ).tolist(),
        },
        "criteria": {
            "kink_maximum_residual": float(kink.evidence.maximum_residual),
            "kink_energy": float(kink.evidence.energy),
            "kink_topological_sector": float(kink.evidence.topological_sector),
            "kink_negative_modes": int(kink.evidence.negative_mode_count),
            "fixed_soliton_intensity_error": float(fixed_intensity_error),
            "fixed_refinement_error": float(fixed.evidence.fixed_step_refinement_error),
            "adaptive_soliton_intensity_error": float(adaptive_intensity_error),
            "adaptive_attempted_steps": int(adaptive.adaptive.attempted_steps),
            "adaptive_rejected_steps": int(adaptive.adaptive.rejected_steps),
            "adaptive_final_distance": float(adaptive.adaptive.final_distance),
        },
        "successful": successful,
        "claim": "finite-stationary-kink-and-envelope-soliton-controls-no-universal-soliton-or-full-maxwell-claim",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_qualification()
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
