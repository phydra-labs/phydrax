#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Held-out elliptic metric evidence plus moduli and Chern–Weil controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


jax.config.update("jax_enable_x64", True)


def run_qualification() -> dict[str, object]:
    training = phx.solver.prepare_elliptic_curve(jax.random.key(701), line_count=2)
    heldout = phx.solver.prepare_elliptic_curve(jax.random.key(702), line_count=2)
    result = phx.solver.solve_calabi_yau_metric(
        training.problem,
        policy=phx.solver.CalabiYauSolvePolicy(
            iterations=1,
            learning_rate=1e-4,
            maximum_backtracks=1,
        ),
    )
    evidence_plan = phx.solver.CalabiYauMetricEvidencePlan(
        training.problem.samples,
        heldout.problem.samples,
        batch_count=2,
        residual_rms_tolerance=1e6,
        positivity_floor=0.0,
        minimum_valid_fraction=0.5,
    )
    metric_evidence = phx.solver.evaluate_calabi_yau_metric_evidence(
        result,
        training.hypersurface,
        evidence_plan,
    )
    artifact = phx.solver.freeze_calabi_yau_result(
        result,
        training.hypersurface,
        evidence=metric_evidence,
    )

    measure = phx.integration.projective_measure_target(
        training.hypersurface,
        heldout.problem.samples,
        measure_kind="canonical",
    )
    sample_count = measure.normalized_weights.shape[0]
    representatives = jnp.ones((sample_count, 1, 1), dtype=jnp.complex128)
    yukawa_density = jnp.ones((sample_count, 1, 1, 1), dtype=jnp.complex128)
    moduli_plan = phx.integration.CalabiYauModuliObservablePlan(
        ("u",),
        representative_kind="algebraic",
        representative_source_id="qualification-normal-deformation-proxy",
        batch_count=2,
    )
    moduli = phx.integration.evaluate_calabi_yau_moduli_observables(
        phx.integration.PreparedCalabiYauModuliSamples(
            moduli_plan,
            measure,
            representatives,
            yukawa_density,
        )
    )

    curvature = jnp.asarray(
        [
            [[[-2.0j * np.pi]]],
            [[[-4.0j * np.pi]]],
        ]
    )
    chern = phx.metrix.chern_character_form(
        curvature,
        1,
        2,
        source_id="u1-curvature-normalization-control",
    )
    chern_number = phx.metrix.integrate_top_characteristic_form(
        chern,
        jnp.asarray((0.25, 0.75)),
        2.0,
        measure_id="two-point-control",
    )
    successful = bool(
        metric_evidence.accepted
        and moduli.accepted
        and chern_number.accepted
        and not moduli.authoritative
    )
    return {
        "kind": "calabi-yau-candidate-qualification",
        "profiles": [
            profile.to_record() for profile in phx.solver.calabi_yau_candidate_profiles()
        ],
        "case": {
            "hypersurface_id": training.hypersurface.hypersurface_id,
            "metric_evidence_id": metric_evidence.evidence_id,
            "artifact_metadata": artifact.metadata(),
            "moduli_prepared_id": moduli.prepared_id,
            "chern_form_id": chern.form_id,
        },
        "raw": {
            "heldout_residuals": np.asarray(metric_evidence.residuals).tolist(),
            "heldout_positivity_margins": np.asarray(
                metric_evidence.positivity_margins
            ).tolist(),
            "heldout_valid": np.asarray(metric_evidence.sample_valid).tolist(),
            "heldout_weights": np.asarray(metric_evidence.normalized_weights).tolist(),
            "batch_rms": np.asarray(metric_evidence.batch_rms_residuals).tolist(),
            "weil_petersson_proxy_real": np.real(
                np.asarray(moduli.weil_petersson_metric)
            ).tolist(),
            "yukawa_proxy_real": np.real(np.asarray(moduli.yukawa_couplings)).tolist(),
            "chern_coefficients_real": np.real(np.asarray(chern.coefficients)).tolist(),
        },
        "criteria": {
            "heldout_weighted_rms": float(metric_evidence.weighted_rms_residual),
            "heldout_effective_samples": float(metric_evidence.effective_sample_size),
            "heldout_valid_fraction": float(metric_evidence.valid_fraction),
            "minimum_positivity_margin": float(metric_evidence.minimum_positivity_margin),
            "moduli_hermiticity_residual": float(moduli.hermiticity_residual),
            "moduli_yukawa_symmetry_residual": float(moduli.yukawa_symmetry_residual),
            "moduli_authoritative": bool(moduli.authoritative),
            "chern_number_real": float(jnp.real(chern_number.normalized_value)),
            "chern_number_imaginary_residual": float(chern_number.imaginary_residual),
        },
        "successful": successful,
        "claim": "finite-heldout-and-sampled-observable-evidence-no-exact-ricci-flat-or-topology-claim",
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
