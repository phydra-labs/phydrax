#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json

import jax

import phydrax as phx
from benchmarks._runtime import capture_environment, logical_array_bytes, measure_host


def benchmark_case(line_count: int):
    training, training_seconds = measure_host(
        lambda: phx.solver.prepare_elliptic_curve(
            jax.random.key(811), line_count=line_count
        )
    )
    heldout, heldout_seconds = measure_host(
        lambda: phx.solver.prepare_elliptic_curve(
            jax.random.key(812), line_count=line_count
        )
    )
    result, solve_seconds = measure_host(
        lambda: phx.solver.solve_calabi_yau_metric(
            training.problem,
            policy=phx.solver.CalabiYauSolvePolicy(
                iterations=1,
                learning_rate=1e-4,
                maximum_backtracks=1,
            ),
        )
    )
    plan, plan_seconds = measure_host(
        lambda: phx.solver.CalabiYauMetricEvidencePlan(
            training.problem.samples,
            heldout.problem.samples,
            batch_count=min(2, heldout.problem.samples.homogeneous_points.shape[0]),
            residual_rms_tolerance=1e6,
            positivity_floor=0.0,
            minimum_valid_fraction=0.5,
        )
    )
    evidence, evidence_seconds = measure_host(
        lambda: phx.solver.evaluate_calabi_yau_metric_evidence(
            result, training.hypersurface, plan
        )
    )
    artifact, freeze_seconds = measure_host(
        lambda: phx.solver.freeze_calabi_yau_result(
            result, training.hypersurface, evidence=evidence
        )
    )
    return {
        "axes": {
            "line_count": line_count,
            "training_samples": training.problem.samples.valid.shape[0],
            "heldout_samples": heldout.problem.samples.valid.shape[0],
            "batch_count": plan.batch_count,
        },
        "ids": {
            "hypersurface": training.hypersurface.hypersurface_id,
            "training_samples": plan.training_sample_id,
            "heldout_samples": plan.heldout_sample_id,
            "evidence": evidence.evidence_id,
        },
        "host_seconds": {
            "training_sampling": training_seconds,
            "heldout_sampling": heldout_seconds,
            "metric_solve": solve_seconds,
            "evidence_plan": plan_seconds,
            "heldout_evaluation": evidence_seconds,
            "artifact_freeze": freeze_seconds,
        },
        "logical_bytes": {
            "result": logical_array_bytes(result),
            "evidence": logical_array_bytes(evidence),
            "artifact": logical_array_bytes(artifact),
        },
        "scientific_residuals": {
            "weighted_rms": float(evidence.weighted_rms_residual),
            "maximum_absolute": float(evidence.maximum_absolute_residual),
            "minimum_positivity": float(evidence.minimum_positivity_margin),
            "batch_standard_deviation": float(evidence.batch_standard_deviation),
        },
        "effective_sample_size": float(evidence.effective_sample_size),
        "valid_fraction": float(evidence.valid_fraction),
        "successful": bool(evidence.accepted),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--line-counts", nargs="+", type=int, default=(2, 4))
    parser.add_argument("--output", type=str)
    arguments = parser.parse_args()
    if any(value < 1 for value in arguments.line_counts):
        raise ValueError("line counts must be positive.")
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": [benchmark_case(value) for value in arguments.line_counts],
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
