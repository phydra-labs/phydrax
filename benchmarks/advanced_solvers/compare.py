#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmarks._comparison import compare_performance, PerformancePolicy
from benchmarks._runtime import DurationDistribution

from .schema import row_identity, validate_report


class IncomparableReportsError(ValueError):
    """Reports do not describe the same complete measured configurations."""


def compare_reports(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    /,
    *,
    require_same_environment: bool = True,
    performance_policy: PerformancePolicy | None = None,
) -> dict[str, Any]:
    """Compare complete reports without silently dropping missing or invalid rows."""
    validate_report(reference)
    validate_report(candidate)
    if reference["provenance"] != candidate["provenance"]:
        raise IncomparableReportsError(
            "benchmark harness, certificate evaluator, or case source differs"
        )
    protocol_fields = (
        "seed",
        "warmup",
        "repeats",
        "selected_adapters",
        "selected_cases",
    )
    if any(
        reference["campaign"][field] != candidate["campaign"][field]
        for field in protocol_fields
    ):
        raise IncomparableReportsError(
            "campaign protocols differ in seed, warmup/repeats, selected members, "
            "or selected ordering"
        )
    same_environment = (
        reference["environment"]["fingerprint"] == candidate["environment"]["fingerprint"]
    )
    if require_same_environment and not same_environment:
        raise IncomparableReportsError(
            "environment fingerprints differ; pass require_same_environment=False "
            "only for an intentional cross-environment comparison"
        )
    reference_rows = {row_identity(row): row for row in reference["rows"]}
    candidate_rows = {row_identity(row): row for row in candidate["rows"]}
    missing_from_candidate = sorted(reference_rows.keys() - candidate_rows.keys())
    missing_from_reference = sorted(candidate_rows.keys() - reference_rows.keys())
    if missing_from_candidate or missing_from_reference:
        raise IncomparableReportsError(
            "row sets differ: "
            f"missing from candidate={missing_from_candidate}, "
            f"missing from reference={missing_from_reference}"
        )

    comparisons: list[dict[str, Any]] = []
    for baseline, contender in zip(
        reference["rows"],
        candidate["rows"],
        strict=True,
    ):
        identity = row_identity(baseline)
        if row_identity(contender) != identity:
            raise IncomparableReportsError("validated row ordering differs")
        _require_matched_contract(baseline, contender, identity)
        baseline_status = baseline["outcome"]["status"]
        contender_status = contender["outcome"]["status"]
        if baseline_status == "skipped" or contender_status == "skipped":
            if baseline_status != contender_status:
                raise IncomparableReportsError(
                    f"row {identity} changed between skipped and measured status"
                )
            comparisons.append(
                {
                    "row_identity": identity,
                    "problem": baseline["problem"],
                    "implementation": baseline["implementation"],
                    "status": "skipped",
                    "skip_reason": {
                        "reference": baseline["outcome"]["skip_reason"],
                        "candidate": contender["outcome"]["skip_reason"],
                    },
                    "metrics": None,
                }
            )
            continue
        if baseline_status == "failed" or contender_status == "failed":
            comparisons.append(
                {
                    "row_identity": identity,
                    "problem": baseline["problem"],
                    "implementation": baseline["implementation"],
                    "status": {
                        "reference": baseline_status,
                        "candidate": contender_status,
                    },
                    "skip_reason": None,
                    "metrics": None,
                }
            )
            continue
        baseline_residual = baseline["certificate"]["relative_residual"]
        candidate_residual = contender["certificate"]["relative_residual"]
        baseline_backward = baseline["certificate"]["backward_error"]
        candidate_backward = contender["certificate"]["backward_error"]
        baseline_solve = baseline["timing"]["solve"]["median_ms"]
        candidate_solve = contender["timing"]["solve"]["median_ms"]
        comparisons.append(
            {
                "row_identity": identity,
                "problem": baseline["problem"],
                "implementation": baseline["implementation"],
                "status": {
                    "reference": baseline_status,
                    "candidate": contender_status,
                },
                "skip_reason": None,
                "metrics": {
                    "solve_median_ms": {
                        "reference": baseline_solve,
                        "candidate": candidate_solve,
                        "candidate_over_reference": _ratio(
                            candidate_solve,
                            baseline_solve,
                        ),
                        "speedup_reference_over_candidate": _ratio(
                            baseline_solve,
                            candidate_solve,
                        ),
                    },
                    "solve_performance": _solve_performance(
                        baseline,
                        contender,
                        policy=performance_policy,
                        comparison_id=identity,
                        same_environment=same_environment,
                    ),
                    "relative_residual": {
                        "reference": baseline_residual,
                        "candidate": candidate_residual,
                        "candidate_over_reference": _ratio(
                            candidate_residual,
                            baseline_residual,
                        ),
                    },
                    "backward_error": {
                        "reference": baseline_backward,
                        "candidate": candidate_backward,
                        "candidate_over_reference": _ratio(
                            candidate_backward,
                            baseline_backward,
                        ),
                    },
                },
            }
        )
    regressed = any(
        row["metrics"] is not None
        and row["metrics"]["solve_performance"] is not None
        and row["metrics"]["solve_performance"]["comparison"] is not None
        and row["metrics"]["solve_performance"]["comparison"]["regressed"]
        for row in comparisons
    )
    candidate_mathematically_successful = all(
        row["status"] == "skipped"
        or row["status"]["candidate"] == "success"
        for row in comparisons
    )
    return {
        "reference_environment": reference["environment"]["fingerprint"],
        "candidate_environment": candidate["environment"]["fingerprint"],
        "same_environment": same_environment,
        "passed": candidate_mathematically_successful and not regressed,
        "regressed": regressed,
        "rows": comparisons,
    }


def _solve_performance(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    /,
    *,
    policy: PerformancePolicy | None,
    comparison_id: str,
    same_environment: bool,
) -> dict[str, Any] | None:
    if policy is None:
        return None
    unsuccessful = [
        label
        for label, row in (("reference", reference), ("candidate", candidate))
        if row["outcome"]["status"] != "success"
        or (
            row["refresh"]["applicable"]
            and row["refresh"]["certificate_converged"] is not True
        )
    ]
    if unsuccessful:
        return {
            "eligible": False,
            "comparison": None,
            "reason": (
                "performance requires mathematical success for both rows; failed "
                + ", ".join(unsuccessful)
            ),
        }
    if not same_environment:
        return {
            "eligible": False,
            "comparison": None,
            "reason": "runtime environment fingerprints differ",
        }
    baseline = DurationDistribution(
        tuple(
            float(value) / 1_000.0 for value in reference["timing"]["solve"]["samples_ms"]
        )
    )
    contender = DurationDistribution(
        tuple(
            float(value) / 1_000.0 for value in candidate["timing"]["solve"]["samples_ms"]
        )
    )
    return {
        "eligible": True,
        "comparison": compare_performance(
            baseline,
            contender,
            policy,
            comparison_id=comparison_id,
        ).to_dict(),
        "reason": None,
    }


def _require_matched_contract(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    identity: str,
) -> None:
    for field in ("problem", "sizes", "tolerances"):
        if reference[field] != candidate[field]:
            raise IncomparableReportsError(
                f"row {identity} differs in required comparison field {field!r}"
            )
    implementation_fields = (
        "adapter",
        "backend",
        "method",
        "preconditioner",
        "versions",
    )
    if any(
        reference["implementation"][field] != candidate["implementation"][field]
        for field in implementation_fields
    ):
        raise IncomparableReportsError(
            f"row {identity} differs in its implementation configuration"
        )
    availability_fields = ("capability", "dependency", "dependency_version")
    if any(
        reference["availability"][field] != candidate["availability"][field]
        for field in availability_fields
    ):
        raise IncomparableReportsError(
            f"row {identity} differs in provider identity"
        )
    transfer_fields = (
        "input_origin",
        "host_to_device_bytes",
        "host_to_device_timing_phase",
        "device_to_host_bytes",
        "device_to_host_timing_phase",
    )
    if any(
        reference["transfers"][field] != candidate["transfers"][field]
        for field in transfer_fields
    ):
        raise IncomparableReportsError(f"row {identity} differs in its transfer contract")
    certificate_fields = (
        "kind",
        "capability",
        "problem_fingerprint",
        "evaluator_fingerprint",
    )
    if any(
        reference["certificate"][field] != candidate["certificate"][field]
        for field in certificate_fields
    ):
        raise IncomparableReportsError(
            f"row {identity} uses different certificate identity"
        )
    refresh_contract_fields = (
        "applicable",
        "symbolic_reused",
        "numeric_refreshed",
        "symbolic_refresh_count",
        "numeric_refresh_count",
        "certificate_problem_fingerprint",
        "certificate_kind",
    )
    if any(
        reference["refresh"][field] != candidate["refresh"][field]
        for field in refresh_contract_fields
    ):
        raise IncomparableReportsError(
            f"row {identity} differs in refresh lifecycle identity"
        )
    if (
        reference["memory"]["evidence"] != candidate["memory"]["evidence"]
        or (
            (reference["timing"]["differentiation"]["count"] > 0)
            != (candidate["timing"]["differentiation"]["count"] > 0)
        )
    ):
        raise IncomparableReportsError(
            f"row {identity} differs in memory or differentiation scope"
        )


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


__all__ = ["IncomparableReportsError", "compare_reports"]
