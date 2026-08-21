#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import copy

import pytest

from benchmarks.advanced_solvers.compare import (
    compare_reports,
    IncomparableReportsError,
)
from benchmarks.advanced_solvers.schema import (
    empty_distribution,
    SCHEMA_VERSION,
    SchemaError,
    TIMING_PHASES,
    validate_report,
)


def test_schema_requires_independent_finite_residual_and_backward_error():
    report = _report([_measured_row()])
    validate_report(report)

    for field in ("residual_norm", "relative_residual", "backward_error"):
        invalid = copy.deepcopy(report)
        invalid["rows"][0]["certificate"][field] = None
        with pytest.raises(SchemaError, match=field):
            validate_report(invalid)

    invalid = copy.deepcopy(report)
    invalid["rows"][0]["certificate"]["independently_computed"] = False
    with pytest.raises(SchemaError, match="independently computed"):
        validate_report(invalid)


def test_schema_requires_report_repeat_count_to_match_solve_samples():
    report = _report([_measured_row()])
    report["campaign"]["repeats"] = 2

    with pytest.raises(SchemaError, match="must equal report.campaign.repeats"):
        validate_report(report)


def test_schema_requires_precise_dependency_skip_and_no_fabricated_measurement():
    row = _measured_row()
    row["outcome"] = {
        "status": "skipped",
        "converged": None,
        "message": "not executed",
        "skip_reason": "required module 'slepc4py' is not installed for adapter 'slepc'",
    }
    row["availability"] = {
        "available": False,
        "capability": "eigen.general",
        "dependency": "slepc4py+SLEPc+petsc4py+PETSc",
        "dependency_version": None,
        "reason": "required module 'slepc4py' is not installed for adapter 'slepc'",
    }
    row["implementation"].update(
        adapter="slepc",
        backend="slepc-comm-self",
        method="slepc-eps-nhep-largest-magnitude",
    )
    row["certificate"].update(
        residual_norm=None,
        relative_residual=None,
        backward_error=None,
    )
    row["transfers"].update(
        host_to_device_bytes=None,
        host_to_device_timing_phase=None,
        device_to_host_bytes=None,
        device_to_host_timing_phase=None,
        evidence="not measured because the row was skipped",
    )
    row["timing"] = {phase: empty_distribution() for phase in TIMING_PHASES}
    validate_report(_report([row]))

    missing_reason = copy.deepcopy(row)
    missing_reason["outcome"]["skip_reason"] = ""
    with pytest.raises(SchemaError, match="skip_reason"):
        validate_report(_report([missing_reason]))

    fabricated = copy.deepcopy(row)
    fabricated["certificate"]["relative_residual"] = 0.0
    with pytest.raises(SchemaError, match="must be null for a skip"):
        validate_report(_report([fabricated]))


def test_comparison_rejects_protocol_changes_before_pairing_rows():
    reference = _report([_measured_row()])
    candidate = copy.deepcopy(reference)
    candidate["campaign"]["warmup"] = 2

    with pytest.raises(IncomparableReportsError, match="campaign protocols differ"):
        compare_reports(reference, candidate)


def test_comparison_rejects_incomparable_certificate_relation():
    reference = _report([_measured_row()])
    candidate = copy.deepcopy(reference)
    candidate["rows"][0]["certificate"]["kind"] = "solver-internal-estimate"

    with pytest.raises(IncomparableReportsError, match="different certificate relations"):
        compare_reports(reference, candidate)


def test_comparison_rejects_changed_transfer_contract():
    reference = _report([_measured_row()])
    candidate = copy.deepcopy(reference)
    reference["rows"][0]["transfers"].update(
        host_to_device_bytes=8,
        host_to_device_timing_phase="setup",
    )
    candidate["rows"][0]["transfers"].update(
        host_to_device_bytes=16,
        host_to_device_timing_phase="setup",
    )

    with pytest.raises(IncomparableReportsError, match="transfer contract"):
        compare_reports(reference, candidate)


def test_schema_recomputes_timing_summaries_from_samples():
    report = _report([_measured_row()])
    report["rows"][0]["timing"]["solve"]["median_ms"] = 2.0

    with pytest.raises(SchemaError, match="does not match samples_ms"):
        validate_report(report)


def test_schema_requires_exact_selected_cross_product_and_order():
    report = _report([_measured_row()])
    report["campaign"]["selected_adapters"] = ["fake", "missing"]

    with pytest.raises(SchemaError, match="exact selected case×adapter cross-product"):
        validate_report(report)


def test_schema_requires_explicit_transfer_phase_evidence():
    report = _report([_measured_row()])
    report["rows"][0]["transfers"]["host_to_device_bytes"] = 8

    with pytest.raises(SchemaError, match="host_to_device_timing_phase"):
        validate_report(report)


def test_schema_accepts_exact_transfers_spanning_multiple_measured_phases():
    report = _report([_measured_row()])
    transfers = report["rows"][0]["transfers"]
    transfers["host_to_device_bytes"] = 24
    transfers["host_to_device_timing_phase"] = "setup+preparation+solve"
    transfers["device_to_host_bytes"] = 8
    transfers["device_to_host_timing_phase"] = "solve+verification"

    validate_report(report)

    invalid = copy.deepcopy(report)
    invalid["rows"][0]["transfers"]["host_to_device_timing_phase"] = "setup+refresh"
    with pytest.raises(SchemaError, match="unmeasured timing phases"):
        validate_report(invalid)


def test_schema_requires_paired_differentiation_compilation_and_execution():
    report = _report([_measured_row()])
    report["rows"][0]["timing"]["differentiation_compilation"] = copy.deepcopy(
        report["rows"][0]["timing"]["setup"]
    )

    with pytest.raises(SchemaError, match="compilation and execution counts must match"):
        validate_report(report)


def test_schema_requires_differentiation_samples_to_match_campaign_repeats():
    report = _report([_measured_row()])
    timing = report["rows"][0]["timing"]
    timing["differentiation_compilation"] = copy.deepcopy(timing["setup"])
    timing["differentiation"] = {
        "count": 2,
        "samples_ms": [1.0, 1.0],
        "min_ms": 1.0,
        "median_ms": 1.0,
        "mean_ms": 1.0,
        "std_ms": 0.0,
        "max_ms": 1.0,
    }

    with pytest.raises(
        SchemaError,
        match="differentiation.count must equal report.campaign.repeats",
    ):
        validate_report(report)


def test_schema_requires_independent_refreshed_problem_certificate():
    row = _measured_row()
    row["refresh"].update(
        applicable=True,
        symbolic_reused=True,
        numeric_refreshed=True,
        numeric_refresh_count=1,
        evidence="numeric refresh was independently certified",
        certificate_problem_fingerprint="b" * 64,
        certificate_kind="linear-system",
        certificate_relative_residual=2e-13,
        certificate_backward_error=2e-14,
        certificate_converged=True,
        independently_certified=True,
    )
    for phase in ("refresh", "refreshed_solve", "refreshed_verification"):
        row["timing"][phase] = copy.deepcopy(row["timing"]["setup"])
    report = _report([row])
    validate_report(report)

    invalid = copy.deepcopy(report)
    invalid["rows"][0]["refresh"]["independently_certified"] = False
    with pytest.raises(SchemaError, match="independently_certified must be true"):
        validate_report(invalid)


def test_schema_rejects_optimization_rows_without_stationarity_evidence():
    report = _report([_measured_row()])
    certificate = report["rows"][0]["certificate"]
    certificate["kind"] = "optimization-stationarity"
    certificate["details"] = {
        "objective": 0.0,
        "objective_gap": 0.0,
        "distance_to_reference": 0.0,
    }

    with pytest.raises(SchemaError, match="gradient_norm"):
        validate_report(report)


def test_schema_rejects_continuation_success_without_fold_evidence():
    report = _report([_measured_row()])
    certificate = report["rows"][0]["certificate"]
    certificate["kind"] = "continuation-branch-residual"
    certificate["details"] = {
        "branch_successful": True,
        "finite_branch": True,
        "residuals_satisfied": True,
        "state_sign_change": False,
        "tangent_coordinate_sign_change": False,
        "fold_bracket": False,
        "successful_fold_traversal": False,
    }

    with pytest.raises(SchemaError, match="certified fold traversal"):
        validate_report(report)


def _report(rows):
    return {
        "schema_version": SCHEMA_VERSION,
        "environment": _environment(),
        "campaign": {
            "seed": 7,
            "warmup": 1,
            "repeats": 1,
            "selected_adapters": list(
                dict.fromkeys(row["implementation"]["adapter"] for row in rows)
            ),
            "selected_cases": list(dict.fromkeys(row["case_id"] for row in rows)),
        },
        "rows": rows,
    }


def _measured_row():
    sample = {
        "count": 1,
        "samples_ms": [1.0],
        "min_ms": 1.0,
        "median_ms": 1.0,
        "mean_ms": 1.0,
        "std_ms": 0.0,
        "max_ms": 1.0,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": "linear-scalar",
        "environment": _environment(),
        "problem": {
            "family": "linear",
            "name": "poisson-1d",
            "variant": "scalar-sparse-spd",
            "seed": 7,
            "dtype": "float64",
            "fingerprint": "a" * 64,
            "parameters": {"symmetric": True},
        },
        "implementation": {
            "adapter": "fake",
            "backend": "cpu",
            "method": "direct",
            "preconditioner": "none",
            "versions": {"fake": "1"},
        },
        "sizes": {
            "dimension": 4,
            "rows": 4,
            "columns": 4,
            "nnz": 10,
            "block_size": 1,
            "right_hand_sides": 1,
        },
        "tolerances": {"relative": 1e-8, "absolute": 1e-10, "max_steps": 20},
        "outcome": {
            "status": "success",
            "converged": True,
            "message": "converged",
            "skip_reason": None,
        },
        "certificate": {
            "kind": "linear-system",
            "residual_norm": 1e-12,
            "relative_residual": 1e-13,
            "backward_error": 1e-14,
            "independently_computed": True,
            "evaluator": "benchmarks.advanced_solvers.certificates",
            "details": {},
        },
        "operations": {
            "iterations": 1,
            "matvecs": 1,
            "preconditioner_applications": 0,
            "linear_solves": 1,
            "nonlinear_evaluations": 0,
            "jacobian_evaluations": 0,
        },
        "refresh": {
            "applicable": False,
            "symbolic_reused": None,
            "numeric_refreshed": None,
            "symbolic_refresh_count": 0,
            "numeric_refresh_count": 0,
            "evidence": "not applicable",
            "certificate_problem_fingerprint": None,
            "certificate_kind": None,
            "certificate_relative_residual": None,
            "certificate_backward_error": None,
            "certificate_converged": None,
            "independently_certified": None,
        },
        "memory": {
            "matrix_bytes": 128,
            "setup_bytes": 64,
            "peak_estimate_bytes": 256,
            "evidence": "exact arrays plus documented workspace estimate",
        },
        "transfers": {
            "input_origin": "numpy-host",
            "host_to_device_bytes": 0,
            "host_to_device_timing_phase": None,
            "device_to_host_bytes": 0,
            "device_to_host_timing_phase": None,
            "evidence": "all arrays remained host-resident",
        },
        "timing": {
            "setup": sample,
            "compilation": empty_distribution(),
            "preparation": sample,
            "solve": sample,
            "differentiation_compilation": empty_distribution(),
            "differentiation": empty_distribution(),
            "refresh": empty_distribution(),
            "refreshed_solve": empty_distribution(),
            "refreshed_verification": empty_distribution(),
            "verification": sample,
        },
        "availability": {
            "available": True,
            "capability": "linear.scalar",
            "dependency": "fake",
            "dependency_version": "1",
            "reason": None,
        },
    }


def _environment():
    return {
        "fingerprint": "environment",
        "python_version": "3.12.0",
        "platform": "test-platform",
        "machine": "test-machine",
        "processor": "test-processor",
        "logical_cpus": 1,
        "numpy_version": "2.0.0",
        "jax": {"version": "0.0", "backend": "cpu", "devices": []},
        "thread_environment": {},
    }
