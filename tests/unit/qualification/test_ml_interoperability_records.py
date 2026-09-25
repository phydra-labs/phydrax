#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

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


def _passing(gate_ids=_ALL_GATES) -> list[runner.ScenarioOutcome]:
    return [
        _outcome(gate_id, name) for gate_id in gate_ids for name in ("first", "second")
    ]


def _report(scenarios, gate_ids=None, *, collection_failed=False):
    return runner.qualification_report(
        runner.ScenarioRun(tuple(scenarios), collection_failed),
        gate_ids,
        build_id="build",
        environment=_ENVIRONMENT,
        backend="cpu",
        precision="float64",
    )


def _chain(entry):
    """Reconstruct one serialized gate chain, verifying every content address."""
    support = SupportTuple.from_record(entry["support_tuple"])
    criterion = QualificationCriterion.from_record(entry["criterion"])
    start = CampaignStartRecord.from_record(entry["campaign_start"])
    observation = CampaignObservationRecord.from_record(entry["campaign_observation"])
    evidence = QualificationEvidence.from_record(entry["evidence"])
    return support, criterion, start, observation, evidence


def _gate(report, gate_id):
    (entry,) = [entry for entry in report["gates"] if entry["gate"] == gate_id]
    return entry


def test_all_passing_gates_qualify_with_distinct_revalidated_chains():
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


def test_one_failed_scenario_fails_only_its_gate_and_the_report():
    scenarios = _passing()
    scenarios[4] = _outcome("G3", "first", "failed", "AssertionError: work grew")
    report = _report(scenarios)

    evidence = _gate(report, "G3")["evidence"]
    assert (evidence["outcome"], evidence["reason"]) == (
        "failed",
        "unqualified-scenarios",
    )
    assert _gate(report, "G3")["raw_output"]["unqualified_scenarios"] == 1
    assert report["failed_gates"] == ["G3"]
    assert report["outcome"] == "failed"


def test_skipped_scenario_does_not_qualify_its_gate():
    scenarios = _passing()
    scenarios[0] = _outcome("G1", "first", "skipped", "Skipped: provider missing")
    report = _report(scenarios)

    assert _gate(report, "G1")["evidence"]["outcome"] == "failed"
    assert report["outcome"] == "failed"


def test_selected_gate_without_scenarios_fails():
    scenarios = [s for s in _passing() if runner.scenario_gate(s.nodeid) != "G21"]
    report = _report(scenarios)

    evidence = _gate(report, "G21")["evidence"]
    assert (evidence["outcome"], evidence["reason"]) == ("failed", "no-gate-scenario")
    assert report["failed_gates"] == ["G21"]
    assert report["outcome"] == "failed"


def test_collection_failure_makes_every_gate_inconclusive():
    report = _report([], collection_failed=True)

    assert report["inconclusive_gates"] == list(_ALL_GATES)
    assert {entry["evidence"]["reason"] for entry in report["gates"]} == {
        "suite-collection-failed"
    }
    assert report["outcome"] == "inconclusive"


def test_report_is_independent_of_scenario_observation_order():
    scenarios = _passing()

    assert runner.serialize_report(_report(scenarios)) == runner.serialize_report(
        _report(reversed(scenarios))
    )


def test_changing_one_outcome_changes_only_that_gates_observation_and_evidence():
    baseline = _report(_passing())
    scenarios = _passing()
    scenarios[10] = _outcome("G6", "first", "failed", "refusal missing")
    changed = _report(scenarios)

    for before, after in zip(baseline["gates"], changed["gates"], strict=True):
        ids = (
            lambda entry: entry["raw_output"]["raw_artifact_id"],
            lambda entry: entry["campaign_observation"]["observation_record_id"],
            lambda entry: entry["evidence"]["evidence_id"],
        )
        for identity in ids:
            assert (identity(before) != identity(after)) == (before["gate"] == "G6")
        for key in ("support_tuple", "criterion", "campaign_start"):
            assert before[key] == after[key]


def test_records_of_different_gates_cannot_be_combined():
    report = _report(_passing())
    _, _, g1_start, g1_observation, _ = _chain(_gate(report, "G1"))
    _, g2_criterion, _, _, g2_evidence = _chain(_gate(report, "G2"))

    with pytest.raises(ValueError):
        validate_qualification_causality(
            g2_criterion, g1_start, g1_observation, g2_evidence
        )


def test_scenario_gate_maps_parametrized_nodes_to_their_gate():
    assert runner.scenario_gate(f"{_SUITE}::test_g12_rollback[PHYSICAL_MAY_COMMIT]") == (
        "G12"
    )
    assert runner.scenario_gate(f"{_SUITE}::test_g1_closure") == "G1"


@pytest.mark.parametrize(
    "nodeid",
    [
        f"{_SUITE}::test_helper_builds_a_closure",
        f"{_SUITE}::test_g25_future_gate",
        f"{_SUITE}::test_g0_no_gate",
        "tests/integration/test_other.py::test_g1_closure",
    ],
)
def test_scenario_gate_rejects_nodes_outside_the_gate_contract(nodeid):
    with pytest.raises(ValueError):
        runner.scenario_gate(nodeid)


def test_subset_selection_reports_missing_gates_and_is_inconclusive():
    selection = ("G22", "G17")
    report = _report(_passing(selection), selection)

    assert report["passed_gates"] == ["G17", "G22"]
    assert report["missing_gates"] == [
        gate_id for gate_id in _ALL_GATES if gate_id not in selection
    ]
    assert report["failed_gates"] == report["inconclusive_gates"] == []
    assert report["outcome"] == "inconclusive"


def test_scenario_of_an_unselected_gate_is_refused():
    with pytest.raises(ValueError):
        _report(_passing(("G17", "G22")), ("G17",))
