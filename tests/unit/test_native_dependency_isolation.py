from __future__ import annotations

import runpy
from pathlib import Path

import pytest


_GATE = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "tools/check_native_dependency_isolation.py")
)
NATIVE_SCENARIOS = _GATE["NATIVE_SCENARIOS"]
_check_backend = _GATE["_check_backend"]
_clean_environment = _GATE["_clean_environment"]
_validate_native_results = _GATE["_validate_native_results"]


def _passed_records() -> dict[str, dict[str, object]]:
    return {
        name: {
            "scenario": name,
            "status": "passed",
            "evidence_id": f"evidence:{name}",
        }
        for name in NATIVE_SCENARIOS
    }

def test_native_results_reject_missing_and_extra_scenarios() -> None:
    records = _passed_records()
    missing = NATIVE_SCENARIOS[0]
    records.pop(missing)
    records["unregistered-extra"] = {
        "scenario": "unregistered-extra",
        "status": "passed",
        "evidence_id": "evidence:extra",
    }

    with pytest.raises(RuntimeError) as error:
        _validate_native_results({"status": "passed", "scenarios": records})

    message = str(error.value)
    assert "missing" in message
    assert missing in message
    assert "extra" in message
    assert "unregistered-extra" in message


@pytest.mark.parametrize("status", ["failed", "missing-dependency", "skipped"])
def test_native_results_reject_nonpassing_mandatory_scenarios(status: str) -> None:
    records = _passed_records()
    failed = NATIVE_SCENARIOS[0]
    records[failed]["status"] = status
    records[failed]["detail"] = "qualification did not execute successfully"

    with pytest.raises(RuntimeError, match=status):
        _validate_native_results({"status": "failed", "scenarios": records})


def test_native_results_reject_a_no_op_pass_record() -> None:
    records = _passed_records()
    first = NATIVE_SCENARIOS[0]
    records[first] = {
        "scenario": first,
        "status": "passed",
        "runtime": {"meshcore": "phydrax-meshcore 0.1.0"},
    }

    with pytest.raises(RuntimeError, match="missing-scenario-specific-evidence"):
        _validate_native_results({"status": "passed", "scenarios": records})


def test_native_results_reject_mislabeled_scenario_records() -> None:
    records = _passed_records()
    records[NATIVE_SCENARIOS[0]]["scenario"] = "different-scenario"

    with pytest.raises(RuntimeError, match="scenario-identity-mismatch"):
        _validate_native_results({"status": "passed", "scenarios": records})


def test_backend_check_requires_a_real_native_identity() -> None:
    scenario = NATIVE_SCENARIOS[0]

    with pytest.raises(RuntimeError, match="non-native backend"):
        _check_backend(
            {
                "accepted": {
                    "provider": "gmsh",
                    "actual_version": "phydrax-meshcore 0.1.0",
                }
            },
            scenario,
        )
    with pytest.raises(RuntimeError, match="valid native backend identity"):
        _check_backend({"meshcore_identity": "not-a-meshcore-identity"}, scenario)


def test_backend_check_uses_consumer_evidence_for_any_scenario_name() -> None:
    with pytest.raises(RuntimeError, match="non-native backend"):
        _check_backend(
            {
                "accepted": {
                    "provider": "phydrax-native",
                    "actual_version": "phydrax-meshcore 0.1.0",
                },
                "consumer": {
                    "provider": "gmsh",
                    "actual_version": "phydrax-meshcore 0.1.0",
                },
            },
            "native-cold-fe-fv-continuation",
        )


def test_clean_environment_removes_source_and_library_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHYDRAX_MESHCORE_LIBRARY", "/tmp/override/libmeshcore.dylib")
    monkeypatch.setenv("PHYDRAX_SOURCE_ROOT", "/tmp/source")
    monkeypatch.setenv("PYTHONPATH", "/tmp/source")
    monkeypatch.setenv("PYTHONHOME", "/tmp/python-home")
    monkeypatch.setenv("PYTHONSTARTUP", "/tmp/startup.py")

    runner = Path("/tmp/native-qualification-runner")
    environment = _clean_environment(runner)

    assert "PHYDRAX_MESHCORE_LIBRARY" not in environment
    assert "PHYDRAX_SOURCE_ROOT" not in environment
    assert "PYTHONHOME" not in environment
    assert "PYTHONSTARTUP" not in environment
    assert environment["PYTHONPATH"] == str(runner)
    assert environment["PYTHONNOUSERSITE"] == "1"
