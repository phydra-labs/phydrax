#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json
from dataclasses import asdict, replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import benchmarks.advanced_solvers.best_nonlinear_campaigns as nonlinear_campaigns
from benchmarks.advanced_solvers.best_nonlinear_campaigns import (
    _external_raw_observation,
    _global_cases,
    _independent_root_certificate,
    _root_cases,
    _root_raw_observation,
    _run_root,
    _runner_payload,
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
        certificate_kind="physical-root-residual",
        certificate_scope="equation",
        certificate_value=0.0 if certified else 1.0,
        certificate_tolerance=1e-8,
        certificate_components={"physical_residual_norm": 0.0 if certified else 1.0},
        work=work,
        work_comparable=True,
        work_incomparability_reason=None,
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
    assert set(manifest["corpora"]["root"]) == {
        case.corpus_id for case in _root_cases().values()
    }
    assert set(_root_cases()) == {
        "diagonal-polynomial",
        "brown-almost-linear",
        "domain-restricted",
        "quasilinear-diffusion",
        "singular-start-rational",
        "tiny-column-underflow",
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
            "certificate_kind": "physical-root-residual",
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
            "work_comparable": False,
            "work_incomparability_reason": "implementation was unavailable",
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
    case = _root_cases()["quasilinear-diffusion"]
    initial = jnp.asarray(case.initial)
    previous = jnp.asarray(case.args)
    assert jnp.linalg.norm(case.jax_residual(initial, previous)) > 0.0

    observation = _run_root("quasilinear-diffusion", "phydrax-lagged")

    assert observation.available
    assert observation.backend_claimed_success
    assert observation.certified
    assert observation.certificate <= 1e-8
    assert observation.work_counts["residual_evaluations"] > 0
    assert observation.work_counts["jvp_evaluations"] > 0
    assert observation.work_counts["linear_iterations"] > 0


def test_lagged_root_campaign_retains_unsupported_case_rows():
    observation = _run_root("brown-almost-linear", "phydrax-lagged")

    assert not observation.available
    assert observation.availability_reason == "unsupported-mathematics"
    assert observation.certified is None


def test_root_descriptor_fingerprint_covers_case_content():
    cases = _root_cases()
    domain = cases["domain-restricted"]
    tiny = cases["tiny-column-underflow"]
    pairs = (
        (
            domain,
            replace(domain, relation_id="different-positive-logarithm-relation"),
        ),
        (
            domain,
            replace(domain, relation_parameters=(("logarithm_base", 10.0),)),
        ),
        (
            domain,
            replace(domain, args=(*domain.args[:-1], domain.args[-1] + 0.25)),
        ),
        (
            domain,
            replace(
                domain,
                domain_parameters=(("lower_bound", -0.25), ("strict", True)),
            ),
        ),
        (
            tiny,
            replace(tiny, residual_scale=(1.0, 2e-200)),
        ),
        (
            domain,
            replace(
                domain,
                termination=type(domain.termination)(
                    absolute_residual=2e-8,
                    relative_residual=0.0,
                    absolute_step=0.0,
                    relative_step=0.0,
                    maximum_steps=200,
                    maximum_evaluations=4000,
                    maximum_linear_iterations=20000,
                ),
            ),
        ),
    )

    assert all(
        changed.content_fingerprint != original.content_fingerprint
        for original, changed in pairs
    )
    payload = _runner_payload("root", "tiny-column-underflow")
    assert payload["relation_parameters"]["second_equation_scale"] == 1e-200
    singular_payload = _runner_payload("root", "singular-start-rational")
    assert singular_payload["domain"]["parameters"]["lower_bound"] == -0.1
    assert payload["relation_id"] == tiny.relation_id
    assert payload["parameters"] is None
    assert payload["scales"]["residual"]["values"] == [1.0, 1e-200]
    assert payload["domain"]["validity_id"] is None
    assert payload["termination"]["absolute_residual"] == 1e-8
    assert payload["case_fingerprint"] == tiny.content_fingerprint
    assert payload["initial_fingerprint"] == tiny.content_fingerprint
    json.dumps(payload, allow_nan=False)


def test_native_and_external_root_rows_share_the_physical_certificate():
    solution = np.asarray([0.0, 0.0])
    native = _root_raw_observation(
        "singular-start-rational",
        "phydrax-newton",
        "successful",
        1.0,
        0.0,
        True,
        solution,
        {"residual_evaluations": 1.0},
    )
    external = _external_raw_observation(
        "root",
        "singular-start-rational",
        "nonlinearsolve-jl",
        {
            "available": True,
            "solution": solution.tolist(),
            "work_counts": {"residual_evaluations": 1.0},
            "backend": {"status_code": "successful", "claimed_success": True},
            "observed_identity": "external-runtime",
            "source_revision": "0" * 40,
        },
        0.0,
    )

    assert native.certified is True
    assert external.certified is True
    assert native.certificate == external.certificate
    assert native.certificate_components == external.certificate_components


def test_physical_root_certificate_rejects_initial_norm_false_positive():
    case = _root_cases()["diagonal-polynomial"]
    parameters = np.asarray(case.args)
    candidate = np.sqrt(parameters)
    candidate[0] = np.sqrt(parameters[0] + 3e-8)

    certified, physical_norm, components = _independent_root_certificate(
        case,
        candidate,
    )
    old_normalized_value = physical_norm / (
        1.0 + np.linalg.norm(np.asarray(case.initial))
    )

    assert old_normalized_value <= case.termination.absolute_residual
    assert physical_norm > components["physical_residual_threshold"]
    assert certified is False


def test_ineligible_root_row_skips_without_residual_or_solver_work(monkeypatch):
    cases = _root_cases()
    tiny = cases["tiny-column-underflow"]
    assert [
        implementation for implementation, eligible, _, _ in tiny.eligibility if eligible
    ] == ["phydrax-scaled-newton"]

    def unexpected_execution(*args, **kwargs):
        raise AssertionError("ineligible case executed")

    blocked = replace(
        tiny,
        jax_residual=unexpected_execution,
        numpy_residual=unexpected_execution,
    )
    monkeypatch.setattr(
        nonlinear_campaigns,
        "_root_cases",
        lambda: {**cases, "tiny-column-underflow": blocked},
    )
    observation = nonlinear_campaigns._timed_observation(
        "root",
        "tiny-column-underflow",
        "phydrax-newton",
        _run_root,
        warmup=1,
        repeats=1,
    )

    assert observation.available is False
    assert observation.certified is None
    assert observation.work is None
    assert observation.work_counts == {}
    assert observation.cold_seconds is None
    assert observation.warmup_seconds == ()
    assert observation.steady_seconds == ()


def test_scaled_newton_counts_scale_preparation_as_work():
    observation = _run_root("tiny-column-underflow", "phydrax-scaled-newton")

    assert observation.available
    assert observation.certified
    assert observation.work_counts["scale_preparation_residual_evaluations"] == 1.0
    assert observation.work == observation.work_counts["residual_evaluations"]
    assert observation.work > 1.0


def test_tiny_column_wrong_root_fails_known_root_gate():
    case = _root_cases()["tiny-column-underflow"]
    certified, physical_norm, components = _independent_root_certificate(
        case,
        np.asarray([2.0, 0.0]),
    )

    assert physical_norm <= case.termination.absolute_residual
    assert components["solver_scaled_residual_norm"] == pytest.approx(3.0)
    assert components["known_root_error"] == pytest.approx(3.0)
    assert certified is False
