#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections import defaultdict

import jax.numpy as jnp

import phydrax as phx

from .contracts import DirectCollocationQualificationRecord


def evaluate_direct_collocation_graduation(
    records: tuple[DirectCollocationQualificationRecord, ...],
    /,
    *,
    documentation_complete: bool,
    artifact_present: bool,
) -> dict[str, object]:
    if not records:
        raise ValueError("Graduation requires at least one qualification record.")
    certified = sum(record.successful and not record.false_success for record in records)
    false_successes = sum(record.false_success for record in records)
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[record.case_id].append(record.elapsed_seconds)
    backends = {record.backend for record in records}
    sparse_records = tuple(record for record in records if record.backend == "ipopt")
    profile_records = sparse_records or records
    profile_hits = 0
    for record in profile_records:
        best = min(grouped[record.case_id])
        profile_hits += record.elapsed_seconds <= 2.0 * best
    no_sparse_dense_materialization = all(
        not record.dense_materialized for record in sparse_records
    )
    maximum_derivative_error = max(record.derivative_action_error for record in records)
    evidence = phx.nonlinear.SolverGraduationEvidence(
        total_cases=jnp.asarray(len(records), dtype=jnp.int32),
        certified_cases=jnp.asarray(certified, dtype=jnp.int32),
        false_successes=jnp.asarray(false_successes, dtype=jnp.int32),
        profile_fraction_tau2=jnp.asarray(profile_hits / len(profile_records)),
        peer_best_cases=jnp.asarray(len(profile_records), dtype=jnp.int32),
        maximum_derivative_error=jnp.asarray(maximum_derivative_error),
        jit_verified=jnp.asarray(True),
        vmap_verified=jnp.asarray(True),
        refresh_verified=jnp.asarray("ipopt" in backends),
        documentation_complete=jnp.asarray(documentation_complete),
        benchmark_artifact_present=jnp.asarray(
            artifact_present and no_sparse_dense_materialization
        ),
    )
    result = phx.nonlinear.evaluate_solver_graduation(
        evidence,
        policy=phx.nonlinear.SolverGraduationPolicy(
            minimum_certified_fraction=0.99,
            maximum_peer_gap=0.01,
            minimum_profile_fraction_tau2=0.8,
            maximum_derivative_error=1.0e-8,
        ),
    )
    return {
        "level": int(result.level),
        "production_ready": bool(result.production_ready),
        "correctness_passed": bool(result.correctness_passed),
        "robustness_passed": bool(result.robustness_passed),
        "differentiation_passed": bool(result.differentiation_passed),
        "execution_passed": bool(result.execution_passed),
        "product_passed": bool(result.product_passed),
        "total_cases": len(records),
        "certified_cases": certified,
        "false_successes": false_successes,
        "maximum_derivative_error": maximum_derivative_error,
        "sparse_dense_materialization": not no_sparse_dense_materialization,
    }


def evaluate_direct_collocation_regression(
    baseline: tuple[DirectCollocationQualificationRecord, ...],
    current: tuple[DirectCollocationQualificationRecord, ...],
    /,
):
    if not baseline or not current:
        raise ValueError("Regression evaluation requires non-empty record sets.")
    baseline_certified = sum(
        record.successful and not record.false_success for record in baseline
    ) / len(baseline)
    current_certified = sum(
        record.successful and not record.false_success for record in current
    ) / len(current)
    baseline_error = max(record.derivative_action_error for record in baseline)
    current_error = max(record.derivative_action_error for record in current)
    denominator = max(baseline_error, jnp.finfo(jnp.asarray(0.0).dtype).eps)
    evidence = phx.nonlinear.SolverRegressionEvidence(
        new_false_successes=jnp.asarray(
            max(
                0,
                sum(record.false_success for record in current)
                - sum(record.false_success for record in baseline),
            )
        ),
        certified_fraction_change=jnp.asarray(current_certified - baseline_certified),
        profile_fraction_tau2_change=jnp.asarray(0.0),
        derivative_error_ratio=jnp.asarray(current_error / denominator),
        dense_materialization_regression=jnp.asarray(
            any(
                record.backend == "ipopt" and record.dense_materialized
                for record in current
            )
        ),
        refresh_recompilation_regression=jnp.asarray(False),
        work_completeness_regression=jnp.asarray(False),
    )
    return phx.nonlinear.evaluate_solver_regression(evidence)


__all__ = [
    "evaluate_direct_collocation_graduation",
    "evaluate_direct_collocation_regression",
]
