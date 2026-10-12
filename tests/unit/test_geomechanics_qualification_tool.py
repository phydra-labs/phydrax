#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from phydrax.qualification import (
    CampaignObservationRecord,
    CampaignStartRecord,
    QualificationCriterion,
    QualificationEvidence,
    validate_qualification_causality,
)
from tools import geomechanics_qualification as tool


pytestmark = pytest.mark.strict_jax


def _raw(
    *,
    successful: bool = True,
    inconclusive: bool = False,
    metric: float | None = 1e-9,
    overflow: bool = False,
) -> dict[str, Any]:
    return {
        "resolutions": [2, 4],
        "levels": [],
        "metrics": {"pressure-gradient-relative-error": metric},
        "successful": successful,
        "inconclusive": inconclusive,
        "overflow": overflow,
        "reason": "Injected consumer outcome for evidence-contract test.",
        "resources": {"evaluations": 2},
    }


def test_selector_refuses_unknown_and_empty_or_duplicate_cases() -> None:
    with pytest.raises(ValueError, match="case"):
        tool.qualify(("invented",))  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="distinct"):
        tool.qualify(())
    with pytest.raises(ValueError, match="distinct"):
        tool.qualify(("ergun-packed-bed", "ergun-packed-bed"))


@pytest.mark.parametrize(
    "arguments",
    [[], ["--case", "invented"], ["--all", "--case", "ergun-packed-bed"]],
    ids=["missing", "unknown", "conflicting"],
)
def test_cli_refuses_invalid_selectors(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        tool.main(arguments)
    assert error.value.code == 2


def test_start_checkpoint_precedes_execution_and_causal_records_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    ticks = iter(range(100, 1000))

    def checkpoint(report: dict[str, Any]) -> None:
        events.append(report["cases"][-1]["status"])

    def execute(name: tool.CaseName) -> dict[str, Any]:
        assert name == "ergun-packed-bed"
        assert events == ["started"]
        events.append("execution")
        return _raw()

    monkeypatch.setattr(tool, "run_case", execute)
    report = tool.qualify(
        ("ergun-packed-bed",), checkpoint=checkpoint, clock=lambda: next(ticks)
    )
    assert events == ["started", "execution", "observed"]
    entry = report["cases"][0]
    criterion = QualificationCriterion.from_record(entry["criteria"][0])
    start = CampaignStartRecord.from_record(entry["starts"][0])
    observation = CampaignObservationRecord.from_record(entry["observations"][0])
    evidence = QualificationEvidence.from_record(entry["evidence"])
    assert (
        criterion.issued_at
        < start.started_at
        <= observation.observed_at
        <= evidence.issued_at
    )
    assert (
        validate_qualification_causality(criterion, start, observation, evidence)
        == evidence.evidence_id
    )
    assert criterion.support_tuple_id == entry["support"]["support_tuple_id"]
    assert observation.raw_artifact_ids == (entry["raw-artifact-id"],)
    assert "NOT-approved" in criterion.approval_id
    assert entry["criterion-status"] == "proposed-candidate-not-approved"


def test_criterion_must_predate_start_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(name: tool.CaseName) -> dict[str, Any]:
        pytest.fail(f"Execution must not start: {name}")

    monkeypatch.setattr(tool, "run_case", forbidden)
    with pytest.raises(ValueError, match="strictly before"):
        tool.qualify(("ergun-packed-bed",), clock=lambda: 100)


@pytest.mark.parametrize(
    ("successful", "inconclusive", "metric", "overflow", "expected"),
    [
        (False, False, 0.0, False, "failed"),
        (True, True, 0.0, False, "inconclusive"),
        (True, False, None, False, "inconclusive"),
        (True, False, 0.0, True, "failed"),
        (True, False, 0.1, False, "failed"),
    ],
    ids=["solver-failed", "incomplete", "missing-metric", "overflow", "criterion-failed"],
)
def test_failed_and_inconclusive_observations_never_pass(
    monkeypatch: pytest.MonkeyPatch,
    successful: bool,
    inconclusive: bool,
    metric: float | None,
    overflow: bool,
    expected: str,
) -> None:
    def execute(_: tool.CaseName) -> dict[str, Any]:
        return _raw(
            successful=successful,
            inconclusive=inconclusive,
            metric=metric,
            overflow=overflow,
        )

    monkeypatch.setattr(tool, "run_case", execute)
    report = tool.qualify(("ergun-packed-bed",))
    assert report["outcome"] == expected
    assert report["cases"][0]["outcome"] == expected
    assert report["cases"][0]["evidence"]["outcome"] == expected


def test_observation_cannot_borrow_a_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tool, "run_case", lambda _: _raw())
    entry = tool.qualify(("ergun-packed-bed",))["cases"][0]
    criterion = QualificationCriterion.from_record(entry["criteria"][0])
    start = CampaignStartRecord.from_record(entry["starts"][0])
    evidence = QualificationEvidence.from_record(entry["evidence"])
    record = dict(entry["observations"][0])
    record.pop("observation_record_id")
    record["start_record_id"] = "borrowed-start"
    borrowed = CampaignObservationRecord.from_record(record)
    with pytest.raises(ValueError, match="borrowed"):
        validate_qualification_causality(criterion, start, borrowed, evidence)


def test_report_refuses_excess_cases() -> None:
    with pytest.raises(ValueError, match="case bound"):
        tool.encode_report({"cases": [{}] * (tool.MAX_CASES + 1)})


def test_report_refuses_excess_bytes() -> None:
    with pytest.raises(ValueError, match="metadata bound"):
        tool.encode_report({"cases": [], "payload": "x" * tool.MAX_REPORT_BYTES})


def test_report_refuses_nonfinite_json() -> None:
    with pytest.raises(ValueError, match="JSON compliant"):
        tool.encode_report({"cases": [], "metric": float("nan")})


def test_cli_persists_starts_before_execution_and_failure_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "report.json"

    def execute(_: tool.CaseName) -> dict[str, Any]:
        checkpoint = json.loads(output.read_text())
        assert checkpoint["cases"][0]["status"] == "started"
        assert checkpoint["cases"][0]["starts"]
        return _raw(successful=False)

    monkeypatch.setattr(tool, "run_case", execute)
    with pytest.raises(SystemExit) as error:
        tool.main(("--case", "ergun-packed-bed", "--output", str(output)))
    assert error.value.code == 1
    report = json.loads(output.read_text())
    assert report["outcome"] == "failed"
    assert len(output.read_bytes()) < tool.MAX_REPORT_BYTES


def test_optional_oblique_case_is_omitted_not_passed() -> None:
    report = tool.qualify(("wu-thornton-li-oblique-impact",))
    assert report["outcome"] == "inconclusive"
    raw = report["cases"][0]["raw"]
    assert raw["omitted"]
    assert raw["metrics"]["velocity-relative-error"] is None
    assert raw["resources"]["steps"] == 0


def test_multi_metric_case_has_one_causal_chain_per_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def execute(_: tool.CaseName) -> dict[str, Any]:
        raw = _raw()
        raw["metrics"] = {
            "terminal-velocity-relative-error": 1e-9,
            "force-relative-residual": 1e-10,
        }
        return raw

    monkeypatch.setattr(tool, "run_case", execute)
    entry = tool.qualify(("schiller-naumann-terminal-velocity",))["cases"][0]
    assert entry["outcome"] == "passed"
    assert (
        len(entry["criteria"]) == len(entry["starts"]) == len(entry["observations"]) == 2
    )
    assert len(set(entry["evidence"]["criteria_ids"])) == 2
    assert len(set(entry["evidence"]["campaign_start_record_ids"])) == 2
    assert len(set(entry["evidence"]["campaign_observation_record_ids"])) == 2


def test_observation_before_start_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter((100, 101, 100, 102))
    monkeypatch.setattr(tool, "run_case", lambda _: _raw())
    with pytest.raises(ValueError, match="causality"):
        tool.qualify(("ergun-packed-bed",), clock=lambda: next(ticks))


def test_numpy_scalar_observations_keep_causal_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def execute(_: tool.CaseName) -> dict[str, Any]:
        raw = _raw()
        raw["successful"] = np.bool_(True)
        raw["inconclusive"] = np.bool_(False)
        raw["resources"]["evaluations"] = np.int64(2)
        raw["metrics"]["pressure-gradient-relative-error"] = np.float64(1e-9)
        return raw

    monkeypatch.setattr(tool, "run_case", execute)
    report = tool.qualify(("ergun-packed-bed",))
    decoded = json.loads(tool.encode_report(report))
    entry = decoded["cases"][0]
    assert entry["outcome"] == "passed"
    assert entry["raw"]["successful"] is True
    assert entry["raw"]["resources"]["evaluations"] == 2
    expected_id = hashlib.sha256(
        json.dumps(
            entry["raw"], sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    assert entry["raw-artifact-id"] == expected_id


def test_refined_footing_report_distinguishes_full_and_half_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def execute(_: tool.CaseName) -> dict[str, Any]:
        raw = _raw(inconclusive=True)
        raw["metrics"] = {"bearing-capacity-relative-error": 0.04}
        return raw

    monkeypatch.setattr(tool, "run_case", execute)
    entry = tool.qualify(("prandtl-footing-tresca",))["cases"][0]
    settings = entry["settings"]
    assert settings["resolutions"] == [8, 16]
    assert settings["half_width_cell_counts"] == [4, 8]
    assert settings["maximum_particles"] == 9216
    assert settings["total_step_bound"] == 24002
    assert settings["rigidity_index_G_over_c"] == 5000.0
    assert settings["poisson_ratio"] == pytest.approx(0.45161290322580644)
    assert settings["indentation"] < 0.2 * min(
        settings["half_cell_platen_support_limits"]
    )
    assert settings["initial_elastic_cfl"] < 0.4
    assert "NOT independently measured" in settings["Nq_status"]


def test_footing_settings_refuse_unbudgeted_resolution() -> None:
    with pytest.raises(ValueError, match="\\[4,16\\]"):
        tool.mpm_case_settings("prandtl-footing-frictional", footing_resolutions=(8, 32))


def test_law_only_agreement_cannot_pass_failed_bagnold_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def execute(_: tool.CaseName) -> dict[str, Any]:
        raw = _raw(successful=False)
        raw["metrics"] = {"profile-relative-l2-error": 0.9}
        raw["law_only_diagnostic"] = {
            "primary_qualification": False,
            "profile_error": 0.0,
        }
        return raw

    monkeypatch.setattr(tool, "run_case", execute)
    entry = tool.qualify(("bagnold-inclined-plane",))["cases"][0]
    assert entry["outcome"] == "failed"
    assert entry["evidence"]["outcome"] == "failed"
    assert entry["support"]["attributes"]["route"] == "public-implicit-mu-i-mpm"
    assert entry["raw"]["law_only_diagnostic"]["primary_qualification"] is False


def test_declared_model_range_is_inconclusive_without_repairing_solver_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def execute(_: tool.CaseName) -> dict[str, Any]:
        raw = _raw(successful=False, inconclusive=True)
        raw["metrics"] = {"profile-relative-l2-error": 0.3}
        raw["model_range_limited"] = True
        return raw

    monkeypatch.setattr(tool, "run_case", execute)
    entry = tool.qualify(("bagnold-inclined-plane",))["cases"][0]
    assert entry["outcome"] == "inconclusive"
    assert entry["evidence"]["outcome"] == "inconclusive"
    assert entry["raw"]["successful"] is False
    assert entry["raw"]["model_range_limited"] is True
