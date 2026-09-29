#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Consumer contracts of the numerical-interoperability qualification runner.

The registry is checked against an independent ``pytest --collect-only``
listing and the benchmark campaign's row registry; report outcomes are checked
on synthesized observations, so no scenario is executed here.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from phydrax.qualification import (
    CampaignObservationRecord,
    CampaignStartRecord,
    QualificationCriterion,
    QualificationEvidence,
    SupportTuple,
    validate_qualification_causality,
)
from tools import numerical_interoperability_qualification as runner
from tools._pytest_outcomes import (
    NodeOutcome,
    PytestCollection,
    PytestRun,
    run_pytest_subprocess,
)


_ROOT = Path(runner.__file__).resolve().parents[1]
_EXTERNAL = next(scenario for scenario in runner.SCENARIOS if scenario.optional_providers)
_SCALING = next(scenario for scenario in runner.SCENARIOS if scenario.benchmark_rows)
_MULTI_DEVICE = next(
    scenario for scenario in runner.SCENARIOS if scenario.host_devices > 1
)


def _collected(scenarios: Sequence[runner.Scenario]) -> tuple[str, ...]:
    """One collected node per reference; function references get a parametrization."""
    return tuple(
        dict.fromkeys(
            reference if "[" in reference else f"{reference}[case]"
            for scenario in scenarios
            for _, reference in scenario.references
        )
    )


def _observation(
    scenarios: Sequence[runner.Scenario],
    *,
    outcome: Mapping[str, str] | None = None,
    duration: float = 0.25,
    collected: Sequence[str] | None = None,
    failed_collectors: Sequence[str] = (),
    providers: Mapping[str, bool] | None = None,
    rows: frozenset[str] | None = None,
) -> runner.QualificationObservation:
    nodeids = _collected(scenarios) if collected is None else tuple(collected)
    overrides = outcome or {}
    routes: dict[int, dict[str, NodeOutcome]] = {}
    for scenario in scenarios:
        route = routes.setdefault(scenario.host_devices, {})
        route.update(
            (nodeid, NodeOutcome(nodeid, overrides.get(nodeid, "passed"), "", duration))
            for nodeid in nodeids
            if any(
                runner.reference_matches(reference, nodeid)
                for _, reference in scenario.references
            )
        )
    registered = frozenset(
        row for scenario in scenarios for row in scenario.benchmark_rows
    )
    return runner.QualificationObservation(
        PytestCollection(nodeids, tuple(failed_collectors), int(pytest.ExitCode.OK)),
        tuple(
            runner.RouteObservation(devices, PytestRun(tuple(nodes.values()), False), 1.0)
            for devices, nodes in sorted(routes.items())
        ),
        {
            provider.name: True
            for scenario in scenarios
            for provider in scenario.optional_providers
        }
        if providers is None
        else providers,
        registered if rows is None else rows,
    )


def _report(
    scenarios: Sequence[runner.Scenario], observation: runner.QualificationObservation
) -> Any:
    return runner.qualification_report(
        scenarios,
        observation,
        build_id="build",
        provenance={"git_revision": "0" * 40, "git_dirty": False},
        environment={"python": "3.12.0", "backend": "cpu"},
        backend="cpu",
        precision="float64",
    )


def _evidence(report: Any, scenario: runner.Scenario) -> tuple[str, str]:
    (entry,) = [
        entry
        for entry in report["scenarios"]
        if entry["scenario"] == scenario.scenario_id
    ]
    return entry["evidence"]["outcome"], entry["evidence"]["reason"]


def test_every_family_and_scenario_is_selectable_on_its_own() -> None:
    for family in runner.FAMILIES:
        members = runner.select_scenarios((), (family,))
        assert members, family
        assert {scenario.family for scenario in members} == {family}
    for scenario in runner.SCENARIOS:
        assert runner.select_scenarios((scenario.scenario_id,)) == (scenario,)
    union = runner.select_scenarios((_SCALING.scenario_id,), ("external",))
    assert union == (_EXTERNAL, _SCALING)
    with pytest.raises(ValueError):
        runner.select_scenarios(("unknown/scenario",))
    with pytest.raises(ValueError):
        runner.select_scenarios((), ("unknown-family",))


def test_every_referenced_node_is_collected() -> None:
    files = runner.referenced_test_files(runner.SCENARIOS)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            *files,
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
    collected = {line for line in completed.stdout.splitlines() if "::" in line}
    unresolved = [
        reference
        for scenario in runner.SCENARIOS
        for _, reference in scenario.references
        if reference not in collected
        and not any(nodeid.startswith(reference + "[") for nodeid in collected)
    ]
    assert unresolved == []


def test_referenced_benchmark_rows_are_registered_by_the_campaign() -> None:
    registered = runner.registered_benchmark_rows()
    assert registered is not None
    referenced = {row for scenario in runner.SCENARIOS for row in scenario.benchmark_rows}
    assert referenced
    assert referenced <= registered


def test_missing_optional_provider_is_inconclusive_never_passed() -> None:
    providers = {provider.name for provider in _EXTERNAL.optional_providers}
    assert providers
    available = _report((_EXTERNAL,), _observation((_EXTERNAL,)))
    assert _evidence(available, _EXTERNAL) == ("passed", "all-nodes-passed")
    for absent in sorted(providers):
        unavailable = {name: name != absent for name in providers}
        report = _report((_EXTERNAL,), _observation((_EXTERNAL,), providers=unavailable))
        assert _evidence(report, _EXTERNAL) == (
            "inconclusive",
            "optional-provider-unavailable",
        )
        assert report["inconclusive_scenarios"] == [_EXTERNAL.scenario_id]
        assert report["passed_scenarios"] == []
        assert report["outcome"] == "inconclusive"
    skipping = {nodeid: "skipped" for nodeid in _collected((_EXTERNAL,))}
    skipped = _report(
        (_EXTERNAL,),
        _observation(
            (_EXTERNAL,), outcome=skipping, providers=dict.fromkeys(providers, False)
        ),
    )
    assert _evidence(skipped, _EXTERNAL)[0] == "inconclusive"
    one_skip = dict(list(skipping.items())[:1])
    partial = _report((_EXTERNAL,), _observation((_EXTERNAL,), outcome=one_skip))
    assert _evidence(partial, _EXTERNAL) == ("inconclusive", "skipped-nodes")


def test_failed_or_unresolved_evidence_fails_and_missing_evidence_is_inconclusive() -> (
    None
):
    scenario = runner.SCENARIOS[0]
    nodes = _collected((scenario,))
    positive = nodes[0]
    negative = nodes[len(scenario.positive)]
    cases = {
        "positive": (
            _observation((scenario,), outcome={positive: "failed"}),
            ("failed", "failed-positive-evidence"),
        ),
        "negative": (
            _observation((scenario,), outcome={negative: "failed"}),
            ("failed", "failed-negative-boundary"),
        ),
        "unresolved": (
            _observation((scenario,), collected=nodes[1:]),
            ("failed", "unresolved-node-reference"),
        ),
        "uncollectable": (
            _observation(
                (scenario,),
                collected=nodes[1:],
                failed_collectors=(runner.reference_path(scenario.positive[0]),),
            ),
            ("inconclusive", "collection-failed"),
        ),
    }
    for name, (observation, expected) in cases.items():
        report = _report((scenario,), observation)
        assert _evidence(report, scenario) == expected, name
        assert report["outcome"] == expected[0], name
    unobserved = _observation((_MULTI_DEVICE,))
    single_device_only = runner.QualificationObservation(
        unobserved.collection,
        tuple(route for route in unobserved.routes if route.host_devices == 1),
        unobserved.providers,
        unobserved.benchmark_rows,
    )
    assert _evidence(_report((_MULTI_DEVICE,), single_device_only), _MULTI_DEVICE) == (
        "inconclusive",
        "nodes-not-observed",
    )


def test_scaling_rows_must_exist_in_the_benchmark_campaign() -> None:
    registered = _report((_SCALING,), _observation((_SCALING,)))
    assert _evidence(registered, _SCALING) == ("passed", "all-nodes-passed")
    missing = frozenset(_SCALING.benchmark_rows[1:])
    unregistered = _report((_SCALING,), _observation((_SCALING,), rows=missing))
    assert _evidence(unregistered, _SCALING) == ("failed", "unregistered-benchmark-row")
    observed = _observation((_SCALING,))
    unavailable = runner.QualificationObservation(
        observed.collection, observed.routes, observed.providers, None
    )
    assert _evidence(_report((_SCALING,), unavailable), _SCALING) == (
        "inconclusive",
        "benchmark-registry-unavailable",
    )


def test_report_chains_verify_and_durations_are_not_content_addressed() -> None:
    selected = runner.select_scenarios((), ("derivatives", "external"))
    first = _report(selected, _observation(selected, duration=0.25))
    second = _report(selected, _observation(selected, duration=4.0))
    assert first["passed_scenarios"] == [scenario.scenario_id for scenario in selected]
    assert first["unselected_scenarios"] == [
        scenario.scenario_id
        for scenario in runner.SCENARIOS
        if scenario.family not in ("derivatives", "external")
    ]
    assert first["outcome"] == "inconclusive"
    for before, after in zip(first["scenarios"], second["scenarios"], strict=True):
        support = SupportTuple.from_record(before["support_tuple"])
        criterion = QualificationCriterion.from_record(before["criterion"])
        start = CampaignStartRecord.from_record(before["campaign_start"])
        observation = CampaignObservationRecord.from_record(
            before["campaign_observation"]
        )
        evidence = QualificationEvidence.from_record(before["evidence"])
        assert evidence.subject_ids == (support.support_tuple_id,)
        assert (
            validate_qualification_causality(criterion, start, observation, evidence)
            == evidence.evidence_id
        )
        assert before["evidence"] == after["evidence"]
        assert before["timing"]["duration_seconds"] < after["timing"]["duration_seconds"]
    everything = _report(runner.SCENARIOS, _observation(runner.SCENARIOS))
    assert everything["outcome"] == "passed"
    assert everything["unselected_scenarios"] == []


def test_fresh_interpreter_route_reports_skips_and_failures_not_passes(
    tmp_path: Path,
) -> None:
    (tmp_path / "test_specimen.py").write_text(
        "import os\n"
        "import pytest\n\n"
        "def test_passes():\n"
        "    assert os.environ['SPECIMEN_FLAG'] == 'set'\n\n"
        "def test_fails():\n"
        "    assert False, 'boundary broke'\n\n"
        "def test_skips():\n"
        "    pytest.skip('inconclusive: provider absent')\n",
        encoding="utf-8",
    )
    (tmp_path / "test_broken.py").write_text("import missing_module\n", encoding="utf-8")
    run = run_pytest_subprocess(
        ["test_specimen.py"],
        root=tmp_path,
        workers=1,
        environment={"SPECIMEN_FLAG": "set"},
    )
    outcomes = {node.nodeid: (node.outcome, node.message) for node in run.nodes}
    assert outcomes == {
        "test_specimen.py::test_passes": ("passed", ""),
        "test_specimen.py::test_fails": ("failed", "AssertionError: boundary broke"),
        "test_specimen.py::test_skips": (
            "skipped",
            "Skipped: inconclusive: provider absent",
        ),
    }
    assert not run.collection_failed
    broken = run_pytest_subprocess(
        ["test_broken.py"], root=tmp_path, workers=1, environment={}
    )
    assert broken.nodes == ()
    assert broken.collection_failed


_COLLECT = (
    "import json, sys\n"
    "from pathlib import Path\n"
    "from tools._pytest_outcomes import collect_pytest_nodes\n"
    "found = collect_pytest_nodes(sys.argv[2:], root=Path(sys.argv[1]))\n"
    "print(json.dumps([found.nodeids, found.failed_collectors, found.exit_code]))\n"
)


def _collect_specimen(root: Path, *paths: str) -> PytestCollection:
    """``collect_pytest_nodes`` in a fresh interpreter, isolated from this session."""
    completed = subprocess.run(
        [sys.executable, "-c", _COLLECT, str(root), *(str(root / p) for p in paths)],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    nodeids, failed_collectors, exit_code = json.loads(
        completed.stdout.strip().splitlines()[-1]
    )
    return PytestCollection(tuple(nodeids), tuple(failed_collectors), exit_code)


def _specimen_outcome(
    collection: PytestCollection, positive: str, negative: str, observed: Sequence[str]
) -> tuple[str, str]:
    scenario = dataclasses.replace(
        runner.SCENARIOS[0], positive=(positive,), negative=(negative,)
    )
    run = PytestRun(
        tuple(NodeOutcome(node, "passed", "", 0.1) for node in observed), False
    )
    observation = runner.QualificationObservation(
        collection, (runner.RouteObservation(1, run, 1.0),), {}, frozenset()
    )
    outcome, _, _ = runner.classify_scenario(scenario, observation)
    return outcome.outcome, outcome.reason


def test_aborted_or_failed_collection_is_inconclusive_but_missing_nodes_fail(
    tmp_path: Path,
) -> None:
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    passing = "def test_ok():\n    pass\n"
    specimens = tmp_path / "tests"
    (specimens / "aborted").mkdir(parents=True)
    (specimens / "aborted" / "conftest.py").write_text(
        "import missing_module\n", encoding="utf-8"
    )
    (specimens / "aborted" / "test_a.py").write_text(passing, encoding="utf-8")
    (specimens / "partial" / "broken").mkdir(parents=True)
    (specimens / "partial" / "test_ok.py").write_text(passing, encoding="utf-8")
    (specimens / "partial" / "broken" / "conftest.py").write_text(
        "import missing_module\n", encoding="utf-8"
    )
    (specimens / "partial" / "broken" / "test_b.py").write_text(passing, encoding="utf-8")

    # A conftest import failure stops pytest before it reaches any module.
    aborted = _collect_specimen(tmp_path, "tests/aborted/test_a.py")
    assert aborted.exit_code == pytest.ExitCode.USAGE_ERROR
    assert aborted.aborted
    assert aborted.nodeids == ()
    assert _specimen_outcome(
        aborted,
        "tests/aborted/test_a.py::test_ok",
        "tests/aborted/test_a.py::test_boundary",
        (),
    ) == ("inconclusive", "collection-failed")

    # A failed directory collector hides its subtree; its siblings still collect.
    partial = _collect_specimen(tmp_path, "tests/partial")
    assert not partial.aborted
    assert partial.nodeids == ("tests/partial/test_ok.py::test_ok",)
    assert partial.failed_collectors == ("tests/partial/broken",)
    ok = "tests/partial/test_ok.py::test_ok"
    assert _specimen_outcome(
        partial, ok, "tests/partial/broken/test_b.py::test_ok", (ok,)
    ) == ("inconclusive", "collection-failed")
    assert _specimen_outcome(
        partial, ok, "tests/partial/test_ok.py::test_gone", (ok,)
    ) == ("failed", "unresolved-node-reference")


def _process_alive(pid: int, /) -> bool:
    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False
    ).stdout.strip()
    return bool(state) and not state.startswith("Z")


def test_fresh_interpreter_route_past_its_deadline_observes_nothing(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "descendant.pid"
    (tmp_path / "test_hang.py").write_text(
        "import pathlib, subprocess, time\n\n"
        "def test_hangs():\n"
        "    descendant = subprocess.Popen(['sleep', '600'])\n"
        f"    pathlib.Path({str(marker)!r}).write_text(str(descendant.pid))\n"
        "    time.sleep(600)\n",
        encoding="utf-8",
    )
    started = time.monotonic()
    run = run_pytest_subprocess(
        ["test_hang.py"], root=tmp_path, workers=1, environment={}, timeout=15.0
    )
    assert time.monotonic() - started < 120.0
    assert run == PytestRun(nodes=(), collection_failed=True)
    # The deadline stops the child's whole session, not just the child.
    descendant = int(marker.read_text(encoding="utf-8"))
    reaped_by = time.monotonic() + 10.0
    while _process_alive(descendant) and time.monotonic() < reaped_by:
        time.sleep(0.1)
    assert not _process_alive(descendant)
    with pytest.raises(ValueError, match="timeout"):
        run_pytest_subprocess(
            ["test_hang.py"], root=tmp_path, workers=1, environment={}, timeout=0.0
        )
