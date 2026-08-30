from dataclasses import replace

import pytest

from tools.direct_collocation_qualification.cases import qualification_setups
from tools.direct_collocation_qualification.contracts import (
    DirectCollocationQualificationArtifact,
    DirectCollocationQualificationRecord,
)
from tools.direct_collocation_qualification.graduation import (
    evaluate_direct_collocation_graduation,
    evaluate_direct_collocation_regression,
)
from tools.direct_collocation_qualification.runner import run_qualification_case


def _record(case_id, backend="native", **overrides):
    values = {
        "case_id": case_id,
        "backend": backend,
        "method_id": f"qualification:{backend}",
        "successful": True,
        "backend_status": 0,
        "public_status": 0,
        "false_success": False,
        "false_failure": False,
        "objective": 1.0,
        "reference_error": 0.0,
        "maximum_defect": 1.0e-10,
        "maximum_constraint_violation": 1.0e-10,
        "maximum_off_grid_defect": 1.0e-5,
        "replay_error": 0.0,
        "derivative_action_error": 1.0e-12,
        "variables": 10,
        "constraints": 8,
        "jacobian_nonzeros": 24,
        "dense_materialized": backend == "native",
        "elapsed_seconds": 1.0,
    }
    values.update(overrides)
    return DirectCollocationQualificationRecord.create(**values)


def test_qualification_case_corpus_is_complete_and_unique():
    setups = qualification_setups()
    identifiers = tuple(setup.case.case_id for setup in setups)
    assert len(setups) == 8
    assert len(set(identifiers)) == len(identifiers)
    assert {setup.case.family for setup in setups} == {
        "analytic",
        "variable-duration",
        "controlled-dae",
        "active-inequality",
        "shared-parameter",
        "stiff-dae",
        "unstable",
        "nonholonomic",
    }


def test_qualification_artifact_fingerprint_and_coverage_are_independent():
    setups = qualification_setups()
    cases = tuple(setup.case for setup in setups)
    records = tuple(_record(case.case_id) for case in cases)
    graduation = evaluate_direct_collocation_graduation(
        records,
        documentation_complete=True,
        artifact_present=True,
    )
    artifact = DirectCollocationQualificationArtifact.create(
        metadata={"source_id": "test", "package_fingerprint": "packages"},
        cases=cases,
        records=records,
        graduation=graduation,
    )
    required = tuple(case.case_id for case in cases)
    artifact.verify(required_case_ids=required)
    round_trip = DirectCollocationQualificationArtifact.from_dict(artifact.to_dict())
    assert round_trip.artifact_id == artifact.artifact_id
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        replace(artifact, artifact_id="tampered").verify(required_case_ids=required)
    with pytest.raises(ValueError, match="coverage is incomplete"):
        artifact.verify(required_case_ids=required[:-1])
    duplicated = replace(artifact, cases=cases + (cases[0],))
    with pytest.raises(ValueError, match="duplicate cases"):
        duplicated.verify(required_case_ids=required)


def test_qualification_record_rejects_nonfinite_and_tampered_metrics():
    record = _record("case")
    record.verify()
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        replace(record, objective=2.0).verify()
    with pytest.raises(ValueError, match="nonfinite"):
        _record("nonfinite", objective=float("nan"))


def test_graduation_requires_sparse_refresh_evidence_for_production():
    cases = tuple(setup.case.case_id for setup in qualification_setups())
    native = tuple(_record(case_id) for case_id in cases)
    validated = evaluate_direct_collocation_graduation(
        native,
        documentation_complete=True,
        artifact_present=True,
    )
    assert validated["level"] == 1
    assert not validated["production_ready"]
    combined = native + tuple(_record(case_id, "ipopt") for case_id in cases)
    production = evaluate_direct_collocation_graduation(
        combined,
        documentation_complete=True,
        artifact_present=True,
    )
    assert production["level"] == 2
    assert production["production_ready"]


def test_regression_detects_new_false_success_and_dense_sparse_path():
    baseline = (_record("case", "ipopt"),)
    current = (
        _record(
            "case",
            "ipopt",
            successful=False,
            false_success=True,
            dense_materialized=True,
            derivative_action_error=1.0e-6,
        ),
    )
    regression = evaluate_direct_collocation_regression(baseline, current)
    assert not bool(regression.passed)
    assert not bool(regression.correctness_passed)
    assert not bool(regression.derivative_passed)
    assert not bool(regression.execution_passed)


def test_native_analytic_qualification_case_produces_certified_record():
    setup = qualification_setups()[0]
    record = run_qualification_case(setup, "native")
    assert record.successful
    assert not record.false_success
    assert not record.false_failure
    assert record.reference_error <= 1.0e-6
    assert record.derivative_action_error <= 1.0e-10
    assert record.record_id
