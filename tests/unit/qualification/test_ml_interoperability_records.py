#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import pytest

from phydrax._fingerprint import canonical_fingerprint
from phydrax.qualification import (
    CampaignObservationRecord,
    CampaignStartRecord,
    QualificationCriterion,
    QualificationEvidence,
    SupportTuple,
    validate_qualification_causality,
)
from tools import ml_interoperability_qualification as runner


_SUITE = runner.SUITE_PATH
_ALL_GATES = tuple(gate.gate_id for gate in runner.GATES)
_ENVIRONMENT = {
    "python": "3.12.0",
    "system": "Darwin",
    "machine": "arm64",
    "packages": {"jax": "0.0.0"},
}


def _outcome(
    gate_id: str, name: str, outcome: str = "passed", message: str = ""
) -> runner.ScenarioOutcome:
    return runner.ScenarioOutcome(
        f"{_SUITE}::test_g{gate_id[1:]}_{name}", outcome, message
    )


def _passing(gate_ids: Any = _ALL_GATES) -> list[runner.ScenarioOutcome]:
    return [
        _outcome(gate_id, name) for gate_id in gate_ids for name in ("first", "second")
    ]


def _report(
    scenarios: Any, gate_ids: Any = None, *, collection_failed: Any = False
) -> Any:
    return runner.qualification_report(
        runner.ScenarioRun(tuple(scenarios), collection_failed),
        gate_ids,
        build_id="build",
        environment=_ENVIRONMENT,
        backend="cpu",
        precision="float64",
    )


def _chain(entry: Any) -> Any:
    """Reconstruct one serialized gate chain, verifying every content address."""
    support = SupportTuple.from_record(entry["support_tuple"])
    criterion = QualificationCriterion.from_record(entry["criterion"])
    start = CampaignStartRecord.from_record(entry["campaign_start"])
    observation = CampaignObservationRecord.from_record(entry["campaign_observation"])
    evidence = QualificationEvidence.from_record(entry["evidence"])
    return support, criterion, start, observation, evidence


def _gate(report: Any, gate_id: Any) -> Any:
    (entry,) = [entry for entry in report["gates"] if entry["gate"] == gate_id]
    return entry


def test_ml_interoperability_records_scenario_1() -> None:
    report = _report(_passing())

    assert report["outcome"] == "passed"
    assert report["passed_gates"] == list(_ALL_GATES)
    assert [entry["gate"] for entry in report["gates"]] == list(_ALL_GATES)
    support_ids = set()
    criterion_ids = set()
    for entry in report["gates"]:
        support, criterion, start, observation, evidence = _chain(entry)
        raw = dict(entry["raw_output"])
        raw_artifact_id = raw.pop("raw_artifact_id")
        assert canonical_fingerprint(raw) == raw_artifact_id
        assert observation.raw_artifact_ids == (raw_artifact_id,)
        assert criterion.support_tuple_id == support.support_tuple_id
        assert evidence.subject_ids == (support.support_tuple_id,)
        assert (
            validate_qualification_causality(criterion, start, observation, evidence)
            == evidence.evidence_id
        )
        assert evidence.passed
        support_ids.add(support.support_tuple_id)
        criterion_ids.add(criterion.criterion_id)
    assert len(support_ids) == len(criterion_ids) == len(_ALL_GATES)
    failed_scenarios = _passing()
    failed_scenarios[4] = _outcome(
        "G3",
        "first",
        "failed",
        "AssertionError: work grew",
    )
    failed = _report(failed_scenarios)
    failed_evidence = _gate(failed, "G3")["evidence"]
    assert (failed_evidence["outcome"], failed_evidence["reason"]) == (
        "failed",
        "unqualified-scenarios",
    )
    assert _gate(failed, "G3")["raw_output"]["unqualified_scenarios"] == 1
    assert failed["failed_gates"] == ["G3"]
    assert failed["outcome"] == "failed"

    skipped_scenarios = _passing()
    skipped_scenarios[0] = _outcome(
        "G1",
        "first",
        "skipped",
        "Skipped: provider missing",
    )
    skipped = _report(skipped_scenarios)
    assert _gate(skipped, "G1")["evidence"]["outcome"] == "failed"
    assert skipped["outcome"] == "failed"

    missing_scenarios = [
        scenario
        for scenario in _passing()
        if runner.scenario_gate(scenario.nodeid) != "G21"
    ]
    missing = _report(missing_scenarios)
    missing_evidence = _gate(missing, "G21")["evidence"]
    assert (missing_evidence["outcome"], missing_evidence["reason"]) == (
        "failed",
        "no-gate-scenario",
    )
    assert missing["failed_gates"] == ["G21"]

    collection_failure = _report([], collection_failed=True)
    assert collection_failure["inconclusive_gates"] == list(_ALL_GATES)
    assert {entry["evidence"]["reason"] for entry in collection_failure["gates"]} == {
        "suite-collection-failed"
    }
    assert collection_failure["outcome"] == "inconclusive"
    scenarios = _passing()
    assert runner.serialize_report(_report(scenarios)) == runner.serialize_report(
        _report(reversed(scenarios))
    )

    baseline = _report(scenarios)
    changed_scenarios = _passing()
    changed_scenarios[10] = _outcome("G6", "first", "failed", "refusal missing")
    changed = _report(changed_scenarios)
    for before, after in zip(baseline["gates"], changed["gates"], strict=True):
        identities = (
            lambda entry: entry["raw_output"]["raw_artifact_id"],
            lambda entry: entry["campaign_observation"]["observation_record_id"],
            lambda entry: entry["evidence"]["evidence_id"],
        )
        for identity in identities:
            assert (identity(before) != identity(after)) == (before["gate"] == "G6")
        for key in ("support_tuple", "criterion", "campaign_start"):
            assert before[key] == after[key]

    _, _, g1_start, g1_observation, _ = _chain(_gate(baseline, "G1"))
    _, g2_criterion, _, _, g2_evidence = _chain(_gate(baseline, "G2"))
    with pytest.raises(ValueError):
        validate_qualification_causality(
            g2_criterion,
            g1_start,
            g1_observation,
            g2_evidence,
        )


def test_ml_interoperability_records_scenario_2() -> None:
    assert (
        runner.scenario_gate(f"{_SUITE}::test_g12_rollback[PHYSICAL_MAY_COMMIT]") == "G12"
    )
    assert runner.scenario_gate(f"{_SUITE}::test_g1_closure") == "G1"
    invalid = (
        f"{_SUITE}::test_helper_builds_a_closure",
        f"{_SUITE}::test_g25_future_gate",
        f"{_SUITE}::test_g0_no_gate",
        "tests/integration/test_other.py::test_g1_closure",
    )
    for nodeid in invalid:
        with pytest.raises(ValueError):
            runner.scenario_gate(nodeid)
    selection = ("G22", "G17")
    report = _report(_passing(selection), selection)
    assert report["passed_gates"] == ["G17", "G22"]
    assert report["missing_gates"] == [
        gate_id for gate_id in _ALL_GATES if gate_id not in selection
    ]
    assert report["failed_gates"] == report["inconclusive_gates"] == []
    assert report["outcome"] == "inconclusive"
    with pytest.raises(ValueError):
        _report(_passing(selection), ("G17",))
