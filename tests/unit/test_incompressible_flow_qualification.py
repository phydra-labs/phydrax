#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from dataclasses import fields

import pytest

import phydrax as phx
from tools.incompressible_flow_qualification import (
    assemble_candidate_profile,
    canonical_json,
    content_address,
    external_reference_input,
    gate_outcome,
    make_qualification_artifact,
    numeric_gate,
    REQUIRED_CASES,
    verify_qualification_artifact,
)
from tools.incompressible_spectral_benchmarks import (
    IncompressibleSpectralBenchmarkRecord,
)
from tools.structured_flow_benchmarks import StructuredFlowBenchmarkRecord


def _artifact(
    route="periodic-spectral",
    *,
    metric_value=1.0e-12,
    tolerance=1.0e-9,
):
    support = phx.qualification.SupportTuple(
        "incompressible-flow",
        {"route": route, "method": "native-test-route"},
    )
    metrics = {name: {"error": metric_value} for name in REQUIRED_CASES[route]}
    return make_qualification_artifact(
        route=route,
        support_tuple=support,
        inputs={"case": "deterministic-input"},
        reference={"source": "native-analytic"},
        configuration={"error_tolerance": tolerance, "timing_gate": False},
        metrics=metrics,
        gates=(
            numeric_gate(
                "solution",
                f"{REQUIRED_CASES[route][0]}.error",
                metric_value,
                tolerance,
            ),
        ),
    )


def test_content_address_and_serialization_are_deterministic_and_content_derived():
    left = _artifact()
    reordered = _artifact()
    changed_metric = _artifact(metric_value=2.0e-12)
    changed_configuration = _artifact(tolerance=2.0e-9)

    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})
    assert content_address({"a": 1}) != content_address({"a": 2})
    assert left == reordered
    assert left["artifact_id"] == reordered["artifact_id"]
    assert left["artifact_id"] != changed_metric["artifact_id"]
    assert left["artifact_id"] != changed_configuration["artifact_id"]
    assert (
        left["configuration"]["configuration_id"]
        != changed_configuration["configuration"]["configuration_id"]
    )
    verify_qualification_artifact(left)


def test_artifact_is_route_exact_unreleased_and_has_no_schema_or_version_field():
    artifact = _artifact("mac")

    assert artifact["route"] == "mac"
    assert artifact["support_tuple"]["attributes"]["route"] == "mac"
    assert artifact["release_ready"] is False
    assert "schema" not in artifact
    assert "schema_version" not in artifact
    assert "version" not in artifact
    assert artifact["configuration"]["timing_gate"] is False


def test_missing_required_route_evidence_is_explicitly_inconclusive():
    support = phx.qualification.SupportTuple(
        "incompressible-flow",
        {"route": "spectral-channel", "method": "native-test-route"},
    )
    artifact = make_qualification_artifact(
        route="spectral-channel",
        support_tuple=support,
        inputs={"case": "missing-required-evidence"},
        reference={"source": "native-analytic"},
        configuration={"timing_gate": False},
        metrics={"couette": {"error": 0.0}},
        gates=(
            gate_outcome(
                "couette",
                "passed",
                "The supplied Couette metric satisfied its criterion.",
            ),
        ),
    )

    assert artifact["status"] == "inconclusive"
    assert artifact["failed_reasons"] == []
    assert len(artifact["inconclusive_reasons"]) == 2
    assert any(
        gate["gate"] == "required-case:poiseuille" and gate["outcome"] == "inconclusive"
        for gate in artifact["gates"]
    )


def test_missing_or_failed_numeric_evidence_fails_with_reason():
    gate = numeric_gate("required-error", "case.error", None, 1.0e-9)
    support = phx.qualification.SupportTuple(
        "incompressible-flow",
        {"route": "periodic-spectral", "method": "native-test-route"},
    )
    artifact = make_qualification_artifact(
        route="periodic-spectral",
        support_tuple=support,
        inputs={"case": "non-finite"},
        reference={"source": "native-analytic"},
        configuration={"error_tolerance": 1.0e-9, "timing_gate": False},
        metrics={name: {"error": None} for name in REQUIRED_CASES["periodic-spectral"]},
        gates=(gate,),
    )

    assert gate["outcome"] == "failed"
    assert gate["reason"]
    assert artifact["status"] == "failed"
    assert artifact["failed_reasons"] == [gate["reason"]]


def test_external_reference_hook_is_all_or_none_and_never_executes_data():
    with pytest.raises(ValueError, match="all-or-none"):
        external_reference_input(path="external.csv", checksum="sha256:abc")

    reference = external_reference_input(
        path="external.csv",
        checksum="sha256:0123456789abcdef",
        nondimensionalization={"length": "channel-half-height"},
        uncertainty={"velocity_standard_uncertainty": 0.01},
    )

    assert reference is not None
    assert reference["path"] == "external.csv"
    assert reference["checksum_verified"] is False
    assert reference["executed"] is False
    assert reference["reference_input_id"]


def test_assembly_references_verified_artifacts_but_cannot_release_or_sign():
    artifact = _artifact()
    candidate = assemble_candidate_profile((artifact,))

    assert candidate["qualification_artifact_ids"] == [artifact["artifact_id"]]
    assert candidate["release_ready"] is False
    assert candidate["signed"] is False
    assert "signature" not in candidate
    assert candidate["profile"]["released"] is False
    assert candidate["profile"]["release_evidence"] == []
    assert candidate["candidate_id"]

    tampered = dict(artifact)
    tampered["release_ready"] = True
    with pytest.raises(ValueError, match="unreleased"):
        assemble_candidate_profile((tampered,))


def test_benchmark_records_are_raw_smoke_evidence_not_qualification_schemas():
    spectral_fields = {
        field.name for field in fields(IncompressibleSpectralBenchmarkRecord)
    }
    structured_fields = {field.name for field in fields(StructuredFlowBenchmarkRecord)}

    for names in (spectral_fields, structured_fields):
        assert "schema_version" not in names
        assert "passed" not in names
        assert "smoke_successful" in names
