from __future__ import annotations

import importlib
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.meshing_qualification import cli


_TEST_SCENARIO = "native-dispatch-contract-test"


def _invoke_with_result(
    monkeypatch: pytest.MonkeyPatch, result: dict[str, object]
) -> dict[str, object]:
    monkeypatch.setitem(
        cli.NATIVE_CASE_REGISTRY, _TEST_SCENARIO, (".test_scenario", "run")
    )
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _module, _package: SimpleNamespace(run=lambda: result),
    )
    monkeypatch.setattr(cli, "runtime_identity", lambda: {"engine": "unit-test"})
    return cli._invoke(_TEST_SCENARIO, Namespace())


def test_dispatch_rejects_missing_explicit_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _invoke_with_result(monkeypatch, {"evidence": {"oracle": "checked"}})

    assert record["status"] == "failed"
    assert record["error_type"] == "RuntimeError"
    assert "explicit nonempty status" in str(record["detail"])


def test_dispatch_rejects_noncanonical_scenario_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _invoke_with_result(
        monkeypatch,
        {"status": "passed", "scenario": "legacy-generation-alias"},
    )

    assert record["status"] == "failed"
    assert record["error_type"] == "RuntimeError"
    assert "noncanonical identity" in str(record["detail"])


def test_dispatch_preserves_explicit_status_and_canonical_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _invoke_with_result(
        monkeypatch,
        {"status": "passed", "evidence": {"oracle": "checked"}},
    )

    assert record["status"] == "passed"
    assert record["scenario"] == _TEST_SCENARIO


def test_run_rejects_empty_scenario_selection() -> None:
    with pytest.raises(ValueError, match="at least one explicit scenario"):
        cli.run((), Namespace())


def test_native_case_migration_covers_every_authored_profile() -> None:
    migration_path = (
        Path(__file__).resolve().parents[2]
        / "tools/meshing_qualification/native_case_migration.json"
    )
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    inventory = migration["inventory"]
    contracts = migration["workflow_contracts"]
    profiles = [profile for contract in contracts for profile in contract["profiles"]]

    assert migration["provenance"]["source_branch"] == "feature/native-meshing-20260929"
    assert (
        migration["provenance"]["source_commit"]
        == "c56614a9fe52c4a8f56484874e91fde6498ec9ff"
    )
    assert inventory["family_count"] == len(contracts) == 13
    assert inventory["profile_count"] == len(profiles) == 114
    assert inventory["mapped_profile_count"] == len(set(profiles)) == 114
    assert inventory["unmapped_profile_count"] == 0
    assert migration["provenance"]["expected_admission"] == "positive"
    assert migration["provenance"]["refusal_counts_as_positive_completion"] is False
    assert all(
        contract.get("target_scenario") or contract.get("target_scenarios")
        for contract in contracts
    )
    targets: set[str] = set()
    for contract in contracts:
        target = contract.get("target_scenario")
        if isinstance(target, str):
            targets.add(target)
        targets.update(contract.get("target_scenarios", ()))
        targets.update((contract.get("profile_target_overrides") or {}).values())
    for workflow in migration["additional_mandatory_workflows"]:
        target = workflow.get("target_scenario")
        if isinstance(target, str):
            targets.add(target)
        targets.update(workflow.get("target_scenarios", ()))
    assert targets <= set(cli.NATIVE_SCENARIOS)
    assert all(
        all(
            contract.get(field)
            for field in (
                "source",
                "source_provenance",
                "required_feature_combinations",
                "independent_oracle",
                "tolerances",
                "target_gate_status",
                "uncovered_contracts",
            )
        )
        for contract in contracts
    )
    assert all(
        all(
            workflow.get(field)
            for field in (
                "source",
                "source_provenance",
                "independent_oracle",
                "tolerances",
                "coverage_status",
            )
        )
        for workflow in migration["additional_mandatory_workflows"]
    )
