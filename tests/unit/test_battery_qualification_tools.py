#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import pytest

from benchmarks import battery_performance
from phydrax._fingerprint import canonical_fingerprint
from phydrax.applications.battery._qualification import (
    THERMAL_ECM_CANDIDATE,
    THERMAL_ECM_SUPPORT,
)
from phydrax.applications.battery._release_contracts import RESOURCE_METRICS
from phydrax.applications.battery._results import BatteryRunStatus, BatterySelectedOutputs
from phydrax.artifacts import ArtifactManifest
from phydrax.lifecycle import ResolvedRunSpec
from phydrax.qualification import (
    CampaignObservationRecord,
    CampaignStartRecord,
    QualificationCriterion,
    QualificationEvidence,
    SupportDependency,
    validate_qualification_causality,
)
from tools import battery_qualification
from tools.battery_campaign_registry import (
    get_campaign_entry,
    metric_key,
    trajectory_observation,
)
from tools.battery_campaign_resources import BatteryResourcePlan


def _schedule_record() -> dict[str, object]:
    now = time.time_ns()
    content: dict[str, object] = {
        "kind": "battery-campaign-schedule",
        "sample_times_s": [0.0, 0.5, 1.0],
        "not_before": now - 1_000_000_000,
        "deadline": now + 600_000_000_000,
        "evidence_validity_duration": 60_000_000_000,
    }
    return {**content, "schedule_id": canonical_fingerprint(content)}


def _criteria_record(
    support_tuple_id: str,
    /,
    *,
    issued_at: int,
    valid_until: int,
    target: float = 1.0e-8,
    comparison: str = "less-than-or-equal",
) -> dict[str, object]:
    criterion = QualificationCriterion(
        support_tuple_id=support_tuple_id,
        metric="maximum-normalized-analytic-residual",
        unit="1",
        comparison=comparison,
        target=target,
        aggregation="maximum",
        uncertainty="deterministic",
        applicability="ecm-analytic",
        approval_id="approval:test",
        issued_at=issued_at,
        valid_until=valid_until,
    )
    extra = [
        QualificationCriterion(
            support_tuple_id=support_tuple_id,
            metric=name,
            unit="1",
            comparison="equal",
            target=1.0,
            aggregation="all",
            uncertainty="deterministic",
            applicability="ecm-analytic",
            approval_id="approval:test",
            issued_at=issued_at,
            valid_until=valid_until,
        )
        for name in ("application-status-success", "model-ledger-success")
    ]
    content: dict[str, object] = {
        "kind": "battery-qualification-criteria-set",
        "criteria": [
            item.to_record()
            for item in sorted((criterion, *extra), key=lambda item: item.criterion_id)
        ],
    }
    return {**content, "criteria_set_id": canonical_fingerprint(content)}


def _artifact_record(manifest: ArtifactManifest) -> dict[str, object]:
    return {
        "kind": "artifact-manifest",
        "artifact_id": manifest.artifact_id,
        "producer": manifest.producer,
        "version": manifest.version,
        "sha256": manifest.sha256,
        "byte_size": manifest.byte_size,
        "source_uri": manifest.source_uri,
        "license_id": manifest.license_id,
        "model": manifest.model,
        "coverage": manifest.coverage,
        "manifest_id": manifest.manifest_id,
    }


def _external_payload(
    path: Path,
    expected_content: bytes,
    /,
    *,
    byte_size: int | None = None,
    sha256: str | None = None,
) -> dict[str, object]:
    manifest = ArtifactManifest(
        artifact_id=f"external:{path.name}",
        producer="unit-test",
        version="fixture",
        sha256=(
            hashlib.sha256(expected_content).hexdigest() if sha256 is None else sha256
        ),
        byte_size=len(expected_content) if byte_size is None else byte_size,
        source_uri=str(path),
        license_id="test-only",
        model="battery-ecm",
        coverage="external-fixture",
    )
    return {
        "path": str(path),
        "manifest": _artifact_record(manifest),
        "required_rights": {
            "commercial_use": False,
            "redistribution": False,
            "training_use": False,
            "export": False,
        },
    }


def _synthetic_source_root(directory: Path) -> Path:
    root = directory / "source-root"
    (root / "phydrax").mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "phydrax/__init__.py").write_text("name = 'synthetic'\n")
    (root / "phydrax/transitive.py").write_text("value = 1\n")
    (root / "tools/battery_qualification.py").write_text("runner = True\n")
    for name in (
        "battery_campaign_registry.py",
        "_battery_ecm_campaign.py",
        "battery_campaign_resources.py",
    ):
        (root / "tools" / name).write_text("contract = True\n")
    (root / "benchmarks").mkdir()
    for name in ("battery_performance.py", "_runtime.py", "_comparison.py"):
        (root / "benchmarks" / name).write_text("harness = True\n")
    (root / "pyproject.toml").write_text("[project]\nname='synthetic'\n")
    (root / "uv.lock").write_text("lock = 'initial'\n")
    return root


def _synthetic_runtime_lock(directory: Path) -> tuple[Path, dict[str, str]]:
    versions = {
        "coordax": "1.0.0",
        "diffrax": "1.0.1",
        "equinox": "1.0.2",
        "jax": "1.0.3",
        "jaxlib": "1.0.4",
        "jaxtyping": "1.0.5",
        "lineax": "1.0.6",
        "numpy": "1.0.7",
        "optimistix": "1.0.8",
    }
    root = directory / "runtime-lock-root"
    root.mkdir()
    records = "\n".join(
        f'[[package]]\nname = "{name}"\nversion = "{version}"\n'
        for name, version in versions.items()
    )
    (root / "uv.lock").write_text(f"version = 1\n{records}")
    return root, versions


def _synthetic_harness_root(directory: Path) -> Path:
    root = directory / "harness-root"
    (root / "benchmarks").mkdir(parents=True)
    (root / "benchmarks/battery_performance.py").write_text("runner = 1\n")
    (root / "benchmarks/_runtime.py").write_text("timer = 1\n")
    (root / "benchmarks/_comparison.py").write_text("comparison = 1\n")
    (root / "tools").mkdir()
    for name in (
        "battery_qualification.py",
        "battery_campaign_registry.py",
        "battery_campaign_resources.py",
        "_battery_ecm_campaign.py",
    ):
        (root / "tools" / name).write_text("contract = True\n")
    return root


def _replay_id(record: dict[str, object]) -> str:
    profile = record["candidate_profile"]
    support = record["candidate_support"]
    run_spec = record["resolved_run_spec"]
    schedule = record["planned_schedule"]
    payloads = record["payloads"]
    assert isinstance(profile, dict)
    assert isinstance(support, dict)
    assert isinstance(run_spec, dict)
    assert isinstance(schedule, dict)
    assert isinstance(payloads, list)
    content = {
        "kind": "battery-campaign-replay",
        "campaign_kind": record["campaign_kind"],
        "registry_entry_id": record["registry_entry_id"],
        "candidate_profile_id": profile["profile_id"],
        "candidate_support_tuple_id": support["support_tuple_id"],
        "resolved_run_spec_id": run_spec["spec_id"],
        "criteria_set_id": record["criteria_set_id"],
        "build_id": record["build_id"],
        "environment_id": record["environment_id"],
        "backend_id": record["backend_id"],
        "device_id": record["device_id"],
        "precision_id": record["precision_id"],
        "topology_id": record["topology_id"],
        "discretization_id": record["discretization_id"],
        "parameter_id": record["parameter_id"],
        "payload_manifest_ids": sorted(
            payload["manifest"]["manifest_id"] for payload in payloads
        ),
        "split_id": record["split_id"],
        "preprocessing_id": record["preprocessing_id"],
        "noise_model_id": record["noise_model_id"],
        "model_selection_id": record["model_selection_id"],
        "rng_id": record["rng_id"],
        "schedule_id": schedule["schedule_id"],
    }
    return canonical_fingerprint(content)


def _seal_campaign(record: dict[str, object]) -> None:
    record["replay_id"] = _replay_id(record)
    _seal_campaign_spec_only(record)


def _seal_campaign_spec_only(record: dict[str, object]) -> None:
    content = dict(record)
    content.pop("campaign_spec_id", None)
    record["campaign_spec_id"] = canonical_fingerprint(content)


def _write_documents(
    directory: Path,
    campaign: dict[str, object],
    criteria: dict[str, object],
) -> tuple[Path, Path]:
    campaign_path = directory / "campaign.json"
    criteria_path = directory / "criteria.json"
    campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
    criteria_path.write_text(json.dumps(criteria), encoding="utf-8")
    return campaign_path, criteria_path


def _documents(directory: Path) -> tuple[dict[str, object], dict[str, object]]:
    project_root = Path(battery_qualification.__file__).resolve().parents[1]
    build_id = battery_qualification.execution_source_build_id(project_root)
    support = THERMAL_ECM_SUPPORT
    profile = THERMAL_ECM_CANDIDATE
    schedule = _schedule_record()
    builtin = battery_qualification.prepare_builtin_campaign(
        "ecm-analytic", schedule["sample_times_s"]
    )
    dependency = SupportDependency(profile.profile_id, support.support_tuple_id)
    run_spec = ResolvedRunSpec(
        (dependency,),
        (),
        release_index_id="release-index:test",
        profile_ids=(profile.profile_id,),
        trust_policy_id="trust-policy:test",
        valid_at=schedule["not_before"],
        valid_from=schedule["not_before"],
        valid_until=schedule["deadline"],
        prepared_configuration_id=builtin.prepared_configuration_id(
            schedule["schedule_id"]
        ),
        precision_policy_id="precision-policy:test",
        resource_policy_id="resource-policy:test",
        checkpoint_policy_id="checkpoint-policy:test",
        output_policy_id="output-policy:test",
        repository_id="repository:test",
        scheduler_id="scheduler:test",
        auth_policy_id="auth-policy:test",
    )
    criteria = _criteria_record(
        support.support_tuple_id,
        issued_at=schedule["not_before"] - 1,
        valid_until=schedule["deadline"] + schedule["evidence_validity_duration"],
    )
    runtime = battery_qualification.capture_runtime_identity()
    campaign: dict[str, object] = {
        "kind": "battery-qualification-campaign",
        "campaign_kind": "ecm-analytic",
        "registry_entry_id": get_campaign_entry("ecm-analytic").entry_id,
        "candidate_profile": profile.to_record(),
        "candidate_support": support.to_record(),
        "resolved_run_spec": run_spec.to_record(),
        "criteria_set_id": criteria["criteria_set_id"],
        "build_id": build_id,
        "environment_id": runtime.environment_id,
        "backend_id": runtime.backend_id,
        "device_id": runtime.device_id,
        "precision_id": runtime.precision_id,
        "topology_id": runtime.topology_id,
        "discretization_id": builtin.discretization_id,
        "parameter_id": builtin.parameter_id,
        "replay_id": "pending",
        "split_id": "split:not-applicable",
        "preprocessing_id": "preprocessing:none",
        "noise_model_id": "noise:none",
        "model_selection_id": builtin.model_selection_id,
        "rng_id": "rng:deterministic-no-randomness",
        "payloads": [],
        "reviewer_id": "reviewer:test",
        "planned_schedule": schedule,
    }
    _seal_campaign(campaign)
    return campaign, criteria


def _replace_criterion(
    campaign: dict[str, object],
    criteria: dict[str, object],
    criterion: dict[str, object],
) -> None:
    criteria["criteria"] = sorted(
        [item for item in criteria["criteria"] if item["metric"] != criterion["metric"]]
        + [criterion],
        key=lambda item: item["criterion_id"],
    )
    criteria_content = dict(criteria)
    criteria_content.pop("criteria_set_id", None)
    criteria["criteria_set_id"] = canonical_fingerprint(criteria_content)
    campaign["criteria_set_id"] = criteria["criteria_set_id"]
    _seal_campaign(campaign)


def _run(directory, campaign, criteria, *, timestamps=None):
    campaign_path, criteria_path = _write_documents(directory, campaign, criteria)
    options = {}
    if timestamps is not None:
        options = {
            "utc_timestamp_source": iter(timestamps).__next__,
            "monotonic_timestamp_source": iter((10, 20, 30)).__next__,
        }
    return battery_qualification.run_qualification(
        battery_qualification.load_campaign_spec(campaign_path),
        battery_qualification.load_criteria_set(criteria_path),
        campaign_directory=directory,
        **options,
    )


def test_native_campaign_persists_one_causal_outcome_per_complete_criterion(tmp_path):
    campaign, criteria = _documents(tmp_path)
    beginning = campaign["planned_schedule"]["not_before"]
    result = _run(
        tmp_path,
        campaign,
        criteria,
        timestamps=(beginning + 1, beginning + 2, beginning + 3),
    )
    assert result["outcome"] == "passed"
    parsed = battery_qualification.QualificationCriteriaSet.from_record(
        result["criteria_set"]
    )
    records = result["evidence_records"]
    assert len(records) == len(parsed.criteria) == 3
    for criterion, start_record, observation_record, evidence_record in zip(
        parsed.criteria,
        result["campaign_start_records"],
        result["campaign_observation_records"],
        records,
        strict=True,
    ):
        start = CampaignStartRecord.from_record(start_record)
        observation = CampaignObservationRecord.from_record(observation_record)
        evidence = QualificationEvidence.from_record(evidence_record)
        assert evidence.criteria_ids == (criterion.criterion_id,)
        assert (
            validate_qualification_causality(criterion, start, observation, evidence)
            == evidence.evidence_id
        )
        assert (
            json.loads(
                (
                    tmp_path / "qualification-artifacts" / f"{evidence.evidence_id}.json"
                ).read_text()
            )
            == evidence_record
        )
    raw = result["raw_output"]
    assert raw["execution"]["application_status"] == int(BatteryRunStatus.SUCCESS)
    assert raw["ledger"]["successful"]
    current_index = raw["outputs"]["names"].index("current_a")
    assert [row[current_index] for row in raw["outputs"]["values"]] == [2.0, 0.0, 0.0]
    again = _run(
        tmp_path,
        campaign,
        criteria,
        timestamps=(beginning + 1, beginning + 2, beginning + 3),
    )
    assert again == result


def test_observed_threshold_failure_does_not_discard_passing_criteria(tmp_path):
    campaign, criteria = _documents(tmp_path)
    criterion_record = next(
        item
        for item in criteria["criteria"]
        if item["metric"] == "maximum-normalized-analytic-residual"
    )
    content = {
        key: value for key, value in criterion_record.items() if key != "criterion_id"
    }
    content["target"] = 0.0
    replacement = QualificationCriterion.from_record(content).to_record()
    _replace_criterion(campaign, criteria, replacement)
    result = _run(tmp_path, campaign, criteria)
    assert result["outcome"] == "failed"
    outcomes = {
        item["criteria_ids"][0]: item["outcome"] for item in result["evidence_records"]
    }
    assert outcomes[replacement["criterion_id"]] == "failed"
    assert list(outcomes.values()).count("passed") == 2
    assert (
        result["raw_output"]["metrics"][
            metric_key("ecm-analytic", replacement["metric"])
        ]["value"]
        > 0.0
    )


@pytest.mark.parametrize(
    "case",
    (
        "unknown-campaign",
        "incomplete-matrix",
        "forged-identity",
        "extra-input",
        "unexpected-payload",
        "build-mismatch",
    ),
)
def test_cli_preflight_refusal_is_an_immutable_audit_not_an_observation(tmp_path, case):
    campaign, criteria = _documents(tmp_path)
    if case == "unknown-campaign":
        campaign["campaign_kind"] = "dfn-unimplemented-campaign"
        _seal_campaign(campaign)
    elif case == "incomplete-matrix":
        criteria["criteria"] = criteria["criteria"][:1]
        content = {
            key: value for key, value in criteria.items() if key != "criteria_set_id"
        }
        criteria["criteria_set_id"] = canonical_fingerprint(content)
        campaign["criteria_set_id"] = criteria["criteria_set_id"]
        _seal_campaign(campaign)
    elif case == "forged-identity":
        campaign["campaign_spec_id"] = "forged"
    elif case == "unexpected-payload":
        path = tmp_path / "unused-input.bin"
        path.write_bytes(b"unused-reference")
        campaign["payloads"] = [_external_payload(path, b"unused-reference")]
        _seal_campaign(campaign)
    elif case == "build-mismatch":
        campaign["build_id"] = "not-the-live-build"
        _seal_campaign(campaign)
    else:
        campaign["undeclared-input"] = 1
    paths = _write_documents(tmp_path, campaign, criteria)
    output = tmp_path / "refusal.json"
    assert (
        battery_qualification.main(
            [
                "--campaign-spec",
                str(paths[0]),
                "--criteria",
                str(paths[1]),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    result = json.loads(output.read_text())
    assert result["outcome"] == "preflight-refused"
    assert "campaign_start_records" not in result
    assert "campaign_observation_records" not in result
    assert "evidence_records" not in result
    with pytest.raises(FileExistsError):
        battery_qualification.write_json_immutable(output, {"different-attempt": True})
    assert json.loads(output.read_text()) == result


def test_post_start_clock_infrastructure_failure_retains_native_observations(tmp_path):
    campaign, criteria = _documents(tmp_path)
    schedule = campaign["planned_schedule"]
    result = _run(
        tmp_path,
        campaign,
        criteria,
        timestamps=(
            schedule["not_before"] + 1,
            schedule["deadline"] + 1,
            schedule["deadline"] + 2,
        ),
    )
    assert result["outcome"] == "inconclusive"
    assert all(
        record["outcome"] == "inconclusive" for record in result["evidence_records"]
    )
    assert result["raw_output"]["execution"]["application_status"] == int(
        BatteryRunStatus.SUCCESS
    )
    assert result["raw_output"]["infrastructure_failures"]
    assert (
        len(result["campaign_start_records"])
        == len(result["campaign_observation_records"])
        == 3
    )


def test_expired_criterion_is_historical_inconclusive_not_dropped(tmp_path):
    campaign, criteria = _documents(tmp_path)
    start = campaign["planned_schedule"]["not_before"] + 10
    selected = criteria["criteria"][0]
    content = {key: value for key, value in selected.items() if key != "criterion_id"}
    content["valid_until"] = start + 5
    replacement = QualificationCriterion.from_record(content).to_record()
    _replace_criterion(campaign, criteria, replacement)
    result = _run(
        tmp_path, campaign, criteria, timestamps=(start, start + 10, start + 20)
    )
    expired = next(
        item
        for item in result["evidence_records"]
        if item["criteria_ids"] == [replacement["criterion_id"]]
    )
    assert expired["outcome"] == "inconclusive"
    assert sum(item["outcome"] == "passed" for item in result["evidence_records"]) == 2


def test_native_observations_encode_absent_samples_without_nonfinite_json():
    outputs = BatterySelectedOutputs(
        (0.0, 1.0, float("inf")),
        ((0.0,), (3.8,), (float("nan"),)),
        (True, True, False),
        names=("voltage_v",),
        units=("V",),
    )
    encoded = json.dumps(
        trajectory_observation("terminal", "observed-run", outputs),
        allow_nan=False,
    )
    decoded = json.loads(encoded)
    assert decoded["outputs"] == [[0.0], [3.8], [None]]
    assert decoded["times_s"] == [0.0, 1.0, None]
    assert decoded["valid"] == [True, True, False]


def test_registry_rejects_raw_unavailable_without_reason_and_relabelled_units(tmp_path):
    campaign, criteria = _documents(tmp_path)
    entry = get_campaign_entry("ecm-analytic")
    selected = criteria["criteria"][0]
    content = {key: value for key, value in selected.items() if key != "criterion_id"}
    content["unit"] = "A"
    _replace_criterion(
        campaign, criteria, QualificationCriterion.from_record(content).to_record()
    )
    result = _run(tmp_path, campaign, criteria)
    assert result["outcome"] == "preflight-refused"
    metric = next(iter(entry.metrics()))
    criterion = QualificationCriterion.from_record(selected)
    assert (
        entry.classify(
            criterion, {"value": None, "unavailable_reason": "missing-reference"}
        )[0]
        == "inconclusive"
    )
    with pytest.raises(ValueError):
        entry.validate_raw(
            {"metrics": {metric: {"value": None, "unavailable_reason": None}}}
        )


def _resource_plan(spec):
    criteria = tuple(
        QualificationCriterion(
            support_tuple_id=spec.candidate_support.support_tuple_id,
            metric=name,
            unit=unit,
            comparison="less-than-or-equal",
            target=0.0
            if name in ("global-dense-arrays", "unsuccessful-executions")
            else 1.0e12,
            aggregation=aggregation,
            uncertainty="observed-single-environment",
            applicability="ecm-complete-cpu-workload",
            approval_id="resource-approval:test",
            issued_at=spec.planned_schedule.not_before - 1,
            valid_until=spec.planned_schedule.deadline,
        )
        for name, unit, aggregation, _ in RESOURCE_METRICS
    )
    return BatteryResourcePlan(
        "ecm-complete-cpu-workload", spec.workload_id(), 5, criteria
    )


def test_absolute_performance_preserves_unavailable_and_measures_complete_output(
    tmp_path,
):
    campaign, criteria = _documents(tmp_path)
    campaign_path, _ = _write_documents(tmp_path, campaign, criteria)
    spec = battery_qualification.load_campaign_spec(campaign_path)
    plan = _resource_plan(spec)
    result = battery_performance.run_performance(
        spec, campaign_directory=tmp_path, sample_count=5, resource_plan=plan
    )
    raw = result["performance"]
    assert result["outcome"] == "inconclusive"
    assert not raw["infrastructure_failures"]
    metrics = {
        name: raw["metrics"][metric_key(plan.case_id, name)]
        for name, _, _, _ in RESOURCE_METRICS
    }
    assert metrics["solver-iterations"]["value"] > 0
    assert metrics["output-bytes"]["value"] > 3 * 8
    assert metrics["checkpoint-bytes"]["value"] is None
    assert metrics["checkpoint-bytes"]["unavailable_reason"]
    assert len(raw["timing"]["warm_samples_seconds"]) == plan.sample_count
    for record in result["evidence_records"]:
        criterion = next(
            item
            for item in plan.criteria
            if item.criterion_id == record["criteria_ids"][0]
        )
        if metrics[criterion.metric]["value"] is None:
            assert record["outcome"] == "inconclusive"
    comparison = battery_performance.compare_performance(raw, raw)
    assert comparison["statistics"]["regressed"] is False
    changed = copy.deepcopy(raw)
    changed["environment_id"] = "different-environment"
    changed["raw_artifact_id"] = canonical_fingerprint(
        {key: value for key, value in changed.items() if key != "raw_artifact_id"}
    )
    with pytest.raises(ValueError):
        battery_performance.compare_performance(raw, changed)


def test_absolute_resource_plan_refuses_missing_or_unbounded_targets(tmp_path):
    campaign, criteria = _documents(tmp_path)
    campaign_path, _ = _write_documents(tmp_path, campaign, criteria)
    plan = _resource_plan(battery_qualification.load_campaign_spec(campaign_path))
    with pytest.raises(ValueError):
        BatteryResourcePlan(
            plan.case_id, plan.workload_id, plan.sample_count, plan.criteria[:-1]
        )
    record = plan.to_record()
    record["criteria"][0]["target"] = float("inf")
    with pytest.raises(ValueError):
        BatteryResourcePlan.from_record(record)


def test_no_dense_audit_checks_closed_constants_and_nested_computations():
    import jax
    import jax.numpy as jnp

    matrix = jnp.eye(65)
    closed = jax.make_jaxpr(lambda vector: jax.jit(lambda item: matrix @ item)(vector))(
        jnp.ones(65)
    )
    assert battery_performance._dense_array_count(closed) > 0
    oracle = jax.make_jaxpr(lambda matrix: matrix @ matrix)(jnp.eye(64))
    assert battery_performance._dense_array_count(oracle) == 0


def test_live_execution_closure_covers_unlisted_sources_and_lockfile(tmp_path):
    source_root = _synthetic_source_root(tmp_path)
    original = battery_qualification.execution_source_build_id(source_root)
    transitive = source_root / "phydrax/transitive.py"
    transitive.write_text("value = 2\n")
    assert battery_qualification.execution_source_build_id(source_root) != original
    transitive.write_text("value = 1\n")

    added = source_root / "phydrax/unlisted.py"
    added.write_text("added = True\n")
    assert battery_qualification.execution_source_build_id(source_root) != original
    added.unlink()
    assert battery_qualification.execution_source_build_id(source_root) == original

    transitive.unlink()
    assert battery_qualification.execution_source_build_id(source_root) != original
    transitive.write_text("value = 1\n")
    assert battery_qualification.execution_source_build_id(source_root) == original

    lockfile = source_root / "uv.lock"
    lockfile.write_text("lock = 'changed'\n")
    assert battery_qualification.execution_source_build_id(source_root) != original
    lockfile.write_text("lock = 'initial'\n")

    registry = source_root / "tools/battery_campaign_registry.py"
    registry.write_text("changed = True\n")
    assert battery_qualification.execution_source_build_id(source_root) != original


def test_runtime_identity_requires_installed_versions_to_match_lock(tmp_path):
    source_root, versions = _synthetic_runtime_lock(tmp_path)
    identity = battery_qualification.capture_runtime_identity(
        source_root=source_root,
        installed_versions=versions,
    )
    lock_payload = (source_root / "uv.lock").read_bytes()
    assert identity.environment_record["runtime_dependencies"] == versions
    assert (
        identity.environment_record["lock_sha256"]
        == hashlib.sha256(lock_payload).hexdigest()
    )

    mismatched = dict(versions)
    mismatched["jax"] = "different"
    with pytest.raises(ValueError, match="does not match uv.lock"):
        battery_qualification.capture_runtime_identity(
            source_root=source_root,
            installed_versions=mismatched,
        )

    missing = dict(versions)
    del missing["diffrax"]
    with pytest.raises(ValueError, match="diffrax.*not installed"):
        battery_qualification.capture_runtime_identity(
            source_root=source_root,
            installed_versions=missing,
        )


def test_atomic_output_does_not_follow_predictable_temporary_symlink(tmp_path):
    output = tmp_path / "result.json"
    victim = tmp_path / "victim.txt"
    victim.write_text("unchanged", encoding="utf-8")
    (tmp_path / ".result.json.tmp").symlink_to(victim)
    battery_qualification.write_json_atomic(output, {"kind": "synthetic"})
    assert victim.read_text(encoding="utf-8") == "unchanged"
    assert json.loads(output.read_text(encoding="utf-8")) == {"kind": "synthetic"}


def test_performance_harness_identity_tracks_both_source_files(tmp_path):
    root = _synthetic_harness_root(tmp_path)
    original = battery_performance.performance_harness_source_id(root)
    runner = root / "benchmarks/battery_performance.py"
    runner.write_text("runner = 2\n")
    assert battery_performance.performance_harness_source_id(root) != original
    runner.write_text("runner = 1\n")
    assert battery_performance.performance_harness_source_id(root) == original
    runtime = root / "benchmarks/_runtime.py"
    runtime.write_text("timer = 2\n")
    assert battery_performance.performance_harness_source_id(root) != original
