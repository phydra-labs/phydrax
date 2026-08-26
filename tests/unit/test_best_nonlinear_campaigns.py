#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json
from dataclasses import asdict
from pathlib import Path

import jax.numpy as jnp
import pytest

from benchmarks.advanced_solvers.best_nonlinear_campaigns import (
    _global_cases,
    _root_cases,
    _run_root,
    CampaignObservation,
    main,
    performance_profile,
    superiority_audit,
)
from benchmarks.advanced_solvers.nonlinear_peer_runners import (
    make_runner_request,
    PeerSpec,
    stable_fingerprint,
    validate_peer_response,
)


def _observation(case_id, implementation, certified, work, *, backend=True):
    initial = stable_fingerprint({"case": case_id, "initial": [0.0]})
    result = stable_fingerprint({"case": case_id, "result": [work]})
    return CampaignObservation(
        family="root",
        case_id=case_id,
        implementation=implementation,
        available=True,
        availability_reason="available",
        availability_detail=None,
        expected_identity="runtime==1",
        observed_identity="runtime==1",
        source_revision="0" * 40,
        initial_fingerprint=initial,
        result_fingerprint=result,
        backend_success=backend,
        backend_scope="equation",
        backend_status="backend-success" if backend else "backend-failure",
        certified=certified,
        certificate_kind="scaled-root-residual",
        certificate_scope="equation",
        certificate_value=0.0 if certified else 1.0,
        certificate_tolerance=1e-8,
        certificate_components={"relative_residual": 0.0 if certified else 1.0},
        work=work,
        work_unit="residual-evaluations",
        work_counts={"residual_evaluations": work},
        cold_seconds=2.0 * work,
        warmup_seconds=(1.5 * work,),
        steady_seconds=(work, 1.1 * work),
    )


def test_peer_manifest_freezes_revisions_and_runtime_identity_without_schema_metadata():
    manifest_path = (
        Path(__file__).parents[2]
        / "benchmarks"
        / "advanced_solvers"
        / "nonlinear_peer_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())

    assert "schema_version" not in manifest
    assert "protocol_version" not in manifest
    assert "runner_protocol" not in manifest["rules"]
    assert manifest["rules"]["false_successes_allowed"] == 0
    assert manifest["rules"]["certificate_source"] == "independent-physical"
    assert manifest["rules"]["initial_fingerprint"] == "exact-match"
    assert all(len(peer["revision"]) == 40 for peer in manifest["peers"])
    assert all(peer["runner"]["expected_identity"] for peer in manifest["peers"])
    assert {peer["id"] for peer in manifest["peers"]} >= {
        "nonlinearsolve-jl",
        "optimistix",
        "scipy",
        "ceres",
        "ipopt",
        "nlopt",
        "theseus",
        "gtsam",
    }


def test_family_profile_penalizes_failed_certificates_without_mixing_work_units():
    observations = [
        _observation(case_id, implementation, certified, work)
        for case_id, implementation, certified, work in (
            ("a", "fast", True, 1.0),
            ("a", "slow", True, 2.0),
            ("b", "fast", False, 0.5),
            ("b", "slow", True, 1.0),
        )
    ]
    profile = performance_profile(observations, metric="primary-work")
    slow_tau2 = next(
        value for value in profile if value.implementation == "slow" and value.tau == 2.0
    )
    fast_tau2 = next(
        value for value in profile if value.implementation == "fast" and value.tau == 2.0
    )

    assert slow_tau2.family == "root"
    assert slow_tau2.work_unit == "residual-evaluations"
    assert slow_tau2.eligible_cases == 2
    assert slow_tau2.fraction == 1.0
    assert fast_tau2.fraction == 0.5


def test_backend_claims_and_independent_certificates_remain_separate():
    false_success = _observation("a", "backend", False, 1.0, backend=True)
    false_failure = _observation("b", "backend", True, 1.0, backend=False)
    audit = superiority_audit([false_success, false_failure])

    assert audit["false_successes"] == [
        {
            "family": "root",
            "case_id": "a",
            "implementation": "backend",
            "backend_status": "backend-success",
            "backend_scope": "equation",
            "certificate_kind": "scaled-root-residual",
            "certificate_value": 1.0,
        }
    ]
    assert audit["backend_false_negatives"] == [
        {
            "family": "root",
            "case_id": "b",
            "implementation": "backend",
            "backend_status": "backend-failure",
            "backend_scope": "equation",
        }
    ]


def test_campaign_output_is_flat_json_without_schema_records(tmp_path):
    output = tmp_path / "campaign.json"
    assert (
        main(
            [
                "differentiation",
                "--warmup",
                "0",
                "--repeats",
                "1",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text())
    row = payload["observations"][0]

    assert "schema_version" not in payload
    assert "schema_version" not in row
    assert "availability" not in row
    assert "backend" not in row
    assert "certificate" not in row
    assert "timing" not in row
    assert "work" in row
    assert row["available"] is True
    assert row["certified"] is True
    assert "Infinity" not in output.read_text()


def test_runner_messages_reject_revision_and_initial_fingerprint_mismatches():
    spec = PeerSpec(
        "peer",
        "a" * 40,
        "external-process",
        "peer==1",
        None,
        "PEER_RUNNER",
    )
    initial = stable_fingerprint({"initial": [0.0]})
    request = make_runner_request(
        spec,
        "root",
        "case",
        "peer-root",
        initial,
        {},
    )
    response = {
        "request_id": request["request_id"],
        "runner_id": "peer-root",
        "initial_fingerprint": initial,
        "observed_identity": "peer==1",
        "source_revision": "a" * 40,
    }
    validate_peer_response(request, response)

    wrong_initial = {**response, "initial_fingerprint": "0" * 64}
    with pytest.raises(ValueError, match="initial fingerprint"):
        validate_peer_response(request, wrong_initial)

    wrong_revision = {**response, "source_revision": "b" * 40}
    with pytest.raises(ValueError, match="revision"):
        validate_peer_response(request, wrong_revision)


def test_unavailable_observation_serializes_with_nulls():
    available = _observation("case", "implementation", True, 1.0)
    unavailable = CampaignObservation(
        **{
            **asdict(available),
            "available": False,
            "availability_reason": "runtime-missing",
            "result_fingerprint": None,
            "backend_success": None,
            "backend_scope": "unavailable",
            "backend_status": None,
            "certified": None,
            "certificate_kind": "unavailable",
            "certificate_scope": "unavailable",
            "certificate_value": None,
            "certificate_tolerance": None,
            "certificate_components": {},
            "work": None,
            "work_unit": None,
            "work_counts": {},
            "cold_seconds": None,
            "warmup_seconds": (),
            "steady_seconds": (),
        }
    )
    payload = json.dumps(asdict(unavailable), allow_nan=False)
    assert '"certificate_value": null' in payload
    assert "Infinity" not in payload


def test_global_rastrigin_uses_dimension_scaled_known_zero_target():
    rastrigin = _global_cases()["rastrigin"]
    assert float(rastrigin(jnp.zeros(4))) == 0.0
    assert float(rastrigin(jnp.ones(4))) > 0.0


def test_lagged_root_campaign_uses_declared_quasilinear_models():
    function, initial, previous = _root_cases()["quasilinear-diffusion"]
    assert jnp.linalg.norm(function(initial, previous)) > 0.0

    observation = _run_root("quasilinear-diffusion", "phydrax-lagged")

    assert observation.available
    assert observation.backend_claimed_success
    assert observation.certified
    assert observation.certificate <= 1e-8
    assert observation.work_counts["residual_evaluations"] > 0
    assert observation.work_counts["jvp_evaluations"] > 0
    assert observation.work_counts["linear_iterations"] > 0


def test_lagged_root_campaign_retains_unsupported_case_rows():
    observation = _run_root("trigonometric", "phydrax-lagged")

    assert not observation.available
    assert observation.availability_reason == "unsupported-mathematics"
    assert observation.certified is None
