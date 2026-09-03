#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json

import pytest

from phydrax.lifecycle._resolved_run import ResolvedRunSpec
from phydrax.qualification._evidence import (
    ForecastResourceRecord,
    ObservedResourceRecord,
    QualificationEvidence,
    SupportDependency,
)
from phydrax.qualification._reference import ReferenceArtifactManifest
from phydrax.qualification._registry import SupportTuple
from tools import (
    compressible_dns_qualification as compressible,
    distributed_execution_qualification as distributed,
    incompressible_commercial_qualification as incompressible,
    reacting_flow_qualification as reacting,
)
from tools._commercial_qualification import (
    canonical_json,
    GATE_CATEGORIES,
    GateDefinition,
    verify_candidate_artifact,
)


def _request(capability, definition):
    support = SupportTuple(
        capability,
        {"route": definition.route, "method": f"{definition.route}-test-method"},
    )
    dependency = SupportDependency(
        f"{capability}.qualified-provider", support.support_tuple_id
    )
    scientific = (dependency,) if definition.dependency_scope == "scientific" else ()
    deployment = (dependency,) if definition.dependency_scope == "deployment" else ()
    run_spec = ResolvedRunSpec(
        scientific,
        deployment,
        release_index_id="release-index-test",
        profile_ids=(dependency.profile_id,),
        trust_policy_id="trust-policy-test",
        valid_at=20,
        valid_from=10,
        valid_until=30,
        prepared_configuration_id=f"prepared-{definition.route}",
        precision_policy_id="precision-f64",
        resource_policy_id="resource-test",
        checkpoint_policy_id="checkpoint-test",
        output_policy_id="output-test",
        repository_id="repository-test",
        scheduler_id="scheduler-test",
        auth_policy_id="auth-test",
    )
    return {
        "support_tuple": support,
        "support_dependency": dependency,
        "resolved_run_spec": run_spec,
        "evidence_context": {
            "build_id": "build-test",
            "environment_id": "environment-test",
            "backend": "cpu-test",
            "topology": "single-process-test",
            "precision": "float64",
            "reduction": "deterministic-tree",
            "replay_id": "replay-test",
            "reviewer_id": "reviewer-test",
            "issued_at": 12,
            "expires_at": 28,
        },
        "observations": {gate.name: True for gate in definition.gates},
    }


def _serialized_request(request):
    result = dict(request)
    result["support_tuple"] = request["support_tuple"].to_record()
    result["support_dependency"] = request["support_dependency"].to_record()
    result["resolved_run_spec"] = request["resolved_run_spec"].to_record()
    return result


def _gate(artifact, category, name):
    return next(value for value in artifact["gates"][category] if value["name"] == name)


def test_route_inventories_cover_each_commercial_qualification_surface():
    assert set(incompressible.ROUTES) == {
        "weighted-pressure",
        "open-pressure",
        "mapped-pressure",
        "ale-pressure",
        "immersed-pressure",
        "mac-controller",
        "ou-fluid",
    }
    assert set(distributed.ROUTES) == {
        "slab",
        "pencil",
        "padded",
        "channel",
        "global-reductions",
        "line-local",
        "partitioned-thomas",
        "spike",
        "pcr",
        "topology-restart",
        "multiblock-extruded",
        "scale-resource",
    }
    assert set(compressible.ROUTES) == {
        "smooth-dgsem",
        "smooth-fv",
        "forcing",
        "budgets",
        "favre-statistics",
        "boundaries",
        "sponge",
        "imex",
        "all-speed",
        "boundary-layer",
        "slow-growth",
        "shock",
        "material",
    }
    assert set(reacting.ROUTES) == {
        "thermodynamics",
        "transport",
        "mechanism",
        "state",
        "strang",
        "imex",
        "cantera-boundary",
        "low-mach",
        "statistics",
    }


def test_serialization_and_metric_identity_are_deterministic_and_content_derived(
    tmp_path,
):
    definition = incompressible.ROUTES["weighted-pressure"]
    request = _request(incompressible.CAPABILITY, definition)
    request["observations"]["weighted-residual"] = {
        "value": 1.0e-12,
        "comparison": "less-than-or-equal",
        "target": 1.0e-9,
        "quantity": "weighted-pressure-residual",
    }

    left = incompressible.produce_candidate("weighted-pressure", request)
    same = incompressible.produce_candidate("weighted-pressure", request)
    changed_request = dict(request)
    changed_request["observations"] = dict(request["observations"])
    changed_request["observations"]["weighted-residual"] = {
        "value": 2.0e-12,
        "comparison": "less-than-or-equal",
        "target": 1.0e-9,
        "quantity": "weighted-pressure-residual",
    }
    changed = incompressible.produce_candidate("weighted-pressure", changed_request)

    assert left == same
    assert canonical_json(left) == canonical_json(same)
    assert left["artifact_id"] == same["artifact_id"]
    assert left["artifact_id"] != changed["artifact_id"]
    assert (
        left["metrics"]["weighted-residual"]["metric_id"]
        != changed["metrics"]["weighted-residual"]["metric_id"]
    )

    input_path = tmp_path / "request.json"
    output_path = tmp_path / "candidate.json"
    input_path.write_text(json.dumps(_serialized_request(request)))
    incompressible.main(
        [
            "weighted-pressure",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
    )
    assert json.loads(output_path.read_text()) == left
    assert output_path.read_text().endswith("\n")


def test_candidate_binds_exact_tuple_dependency_run_and_separate_evidence():
    definition = incompressible.ROUTES["mac-controller"]
    request = _request(incompressible.CAPABILITY, definition)
    artifact = incompressible.produce_candidate("mac-controller", request)

    verify_candidate_artifact(artifact)
    assert tuple(artifact["gates"]) == GATE_CATEGORIES
    assert tuple(artifact["qualification_evidence"]) == GATE_CATEGORIES
    dependency = SupportDependency.from_record(artifact["support_dependency"])
    run_spec = ResolvedRunSpec.from_record(artifact["resolved_run_spec"])
    for category in GATE_CATEGORIES:
        evidence = QualificationEvidence.from_record(
            artifact["qualification_evidence"][category]
        )
        assert dependency.dependency_id in evidence.subject_ids
        assert run_spec.spec_id in evidence.subject_ids
        assert evidence.evidence_kind == category

    other_support = SupportTuple(
        incompressible.CAPABILITY,
        {"route": "mac-controller", "method": "different-method"},
    )
    bad_request = dict(request)
    bad_request["support_dependency"] = SupportDependency(
        request["support_dependency"].profile_id,
        other_support.support_tuple_id,
    )
    with pytest.raises(ValueError, match="exact SupportTuple"):
        incompressible.produce_candidate("mac-controller", bad_request)


def test_observed_failure_and_unavailable_evidence_remain_distinct():
    definition = incompressible.ROUTES["open-pressure"]
    failed_request = _request(incompressible.CAPABILITY, definition)
    failed_request["observations"]["open-pressure-residual"] = False
    failed = incompressible.produce_candidate("open-pressure", failed_request)

    inconclusive_request = _request(incompressible.CAPABILITY, definition)
    del inconclusive_request["observations"]["open-pressure-residual"]
    inconclusive = incompressible.produce_candidate("open-pressure", inconclusive_request)

    assert failed["status"] == "failed"
    assert _gate(failed, "scientific", "open-pressure-residual")["outcome"] == "failed"
    assert inconclusive["status"] == "inconclusive"
    assert (
        _gate(inconclusive, "scientific", "open-pressure-residual")["outcome"]
        == "inconclusive"
    )
    assert not inconclusive["failed_reasons"]


def test_timing_cannot_be_scientific_evidence():
    with pytest.raises(ValueError, match="Timing measurements"):
        GateDefinition(
            "wall-clock-seconds",
            "scientific",
            "Wall-clock timing is not scientific validation.",
        )

    definition = incompressible.ROUTES["weighted-pressure"]
    request = _request(incompressible.CAPABILITY, definition)
    request["observations"]["weighted-residual"] = {
        "value": 0.001,
        "comparison": "less-than-or-equal",
        "target": 1.0,
        "quantity": "elapsed-seconds",
    }
    with pytest.raises(ValueError, match="Timing measurements"):
        incompressible.produce_candidate("weighted-pressure", request)


def test_compressible_candidate_never_inherits_or_claims_dns_support():
    definition = compressible.ROUTES["smooth-dgsem"]
    request = _request(compressible.CAPABILITY, definition)
    artifact = compressible.produce_candidate("smooth-dgsem", request)

    verify_candidate_artifact(artifact)
    application = artifact["extra"]["application_route_evidence"]
    assert artifact["dns_claimed"] is False
    assert artifact["inherits_dns"] is False
    assert artifact["extra"]["dns_support_inherited"] is False
    assert application["dns_claimed"] is False
    assert application["signed"] is False
    assert application["released"] is False

    claimed = dict(request)
    claimed["dns_claimed"] = True
    with pytest.raises(ValueError, match="cannot claim or inherit DNS"):
        compressible.produce_candidate("smooth-dgsem", claimed)


def test_external_reference_refuses_missing_commercial_rights():
    payload = b"governed-reference"
    manifest = ReferenceArtifactManifest(
        "restricted-reference.bin",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="restricted-test-license",
        commercial_use_permitted=False,
        redistribution_permitted=False,
        training_use_permitted=False,
        export_permitted=False,
        export_classification="restricted",
        nondimensionalization={"length": 1.0},
        uncertainty={"state": 0.01},
        lineage_ids=("reference-lineage-test",),
    )
    definition = compressible.ROUTES["material"]
    request = _request(compressible.CAPABILITY, definition)

    with pytest.raises(PermissionError, match="commercial-use-not-permitted"):
        compressible.produce_candidate(
            "material",
            request,
            reference_manifest=manifest,
            reference_payload=payload,
        )


def test_absent_or_simulated_multidevice_and_provider_are_inconclusive():
    slab_definition = distributed.ROUTES["slab"]
    absent_request = _request(distributed.CAPABILITY, slab_definition)
    absent = distributed.produce_candidate("slab", absent_request)
    assert absent["status"] == "inconclusive"
    assert (
        _gate(absent, "operational", "multi-device-execution")["outcome"]
        == "inconclusive"
    )

    simulated_request = _request(distributed.CAPABILITY, slab_definition)
    simulated_request["availability"] = {
        "observed": True,
        "provider_available": True,
        "device_count": 8,
        "simulated": True,
    }
    simulated = distributed.produce_candidate("slab", simulated_request)
    assert simulated["status"] == "inconclusive"
    assert (
        "simulation-is-not-qualification"
        in _gate(simulated, "operational", "multi-device-execution")["reason"]
    )

    cantera_definition = reacting.ROUTES["cantera-boundary"]
    cantera_request = _request(reacting.CAPABILITY, cantera_definition)
    cantera = reacting.produce_candidate("cantera-boundary", cantera_request)
    assert cantera["status"] == "inconclusive"
    assert _gate(cantera, "operational", "cantera-provider")["outcome"] == "inconclusive"


def test_scale_candidate_binds_observed_and_forecast_resource_records():
    definition = distributed.ROUTES["scale-resource"]
    request = _request(distributed.CAPABILITY, definition)
    context = request["evidence_context"]
    observed = ObservedResourceRecord(
        "distributed-scale-test",
        context["build_id"],
        context["environment_id"],
        backend=context["backend"],
        topology=context["topology"],
        measurements={"device_bytes": 1024.0, "global_cells": 4096.0},
        observed_at=12,
        raw_artifact_ids=("raw-resource-test",),
    )
    forecast = ForecastResourceRecord(
        "distributed-scale-test",
        context["build_id"],
        context["environment_id"],
        backend=context["backend"],
        topology=context["topology"],
        estimates={"device_bytes": 2048.0},
        uncertainty_bounds={"device_bytes": (1536.0, 2560.0)},
        forecast_model_id="forecast-model-test",
        source_record_ids=(observed.record_id,),
        issued_at=12,
        expires_at=28,
    )
    request["observed_resource_records"] = (observed,)
    request["forecast_resource_records"] = (forecast,)
    request["availability"] = {
        "observed": True,
        "provider_available": True,
        "device_count": 2,
        "simulated": False,
    }

    artifact = distributed.produce_candidate("scale-resource", request)
    performance = QualificationEvidence.from_record(
        artifact["qualification_evidence"]["performance"]
    )
    assert artifact["status"] == "passed"
    assert performance.observed_resource_record_ids == (observed.record_id,)
    assert performance.forecast_resource_record_ids == (forecast.record_id,)


def test_profile_is_content_addressed_but_remains_unsigned_and_unreleased():
    definition = reacting.ROUTES["statistics"]
    request = _request(reacting.CAPABILITY, definition)
    artifact = reacting.produce_candidate("statistics", request)
    candidate = reacting.assemble_profile((artifact,))

    assert candidate["qualification_artifact_ids"] == [artifact["artifact_id"]]
    assert candidate["signed"] is False
    assert candidate["release_ready"] is False
    assert "signature" not in candidate
    assert candidate["profile"]["released"] is False
    assert candidate["profile"]["release_evidence"] == []
    assert candidate["candidate_id"]
    assert "schema" not in artifact
    assert "schema_version" not in artifact
    assert "version" not in artifact
