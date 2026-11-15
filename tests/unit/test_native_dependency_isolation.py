from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

import pytest


_GATE = runpy.run_path(
    str(
        Path(__file__).resolve().parents[2] / "tools/check_native_dependency_isolation.py"
    )
)
NATIVE_SCENARIOS = _GATE["NATIVE_SCENARIOS"]
_check_backend = _GATE["_check_backend"]
_clean_environment = _GATE["_clean_environment"]
_validate_native_results = _GATE["_validate_native_results"]
_command = _GATE["_command"]
_SCRIPT = (
    Path(__file__).resolve().parents[2] / "tools/check_native_dependency_isolation.py"
)


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


def test_clean_environment_removes_source_and_library_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_real_child_receipts_survive_later_nonzero_exit(tmp_path: Path) -> None:
    first = [sys.executable, "-c", "print('first receipt')"]
    environment = dict(
        os.environ, OMP_NUM_THREADS="2", JAX_PLATFORMS="cpu", PRIVATE_TOKEN="secret"
    )
    assert _command(first, evidence=tmp_path, env=environment) == "first receipt"
    second = [
        sys.executable,
        "-c",
        "import sys; print('second stdout'); print('second stderr', file=sys.stderr); sys.exit(7)",
    ]
    with pytest.raises(RuntimeError) as caught:
        _command(second, evidence=tmp_path)
    assert "Command failed (7)" in str(caught.value)
    assert "second stdout" in str(caught.value)
    assert "second stderr" in str(caught.value)
    assert (tmp_path / "command-001.stdout").read_text() == "first receipt\n"
    first_receipt = json.loads((tmp_path / "command-001.json").read_text())
    second_receipt = json.loads((tmp_path / "command-002.json").read_text())
    assert first_receipt["arguments"] == first
    assert first_receipt["returncode"] == 0
    assert first_receipt["environment"]["OMP_NUM_THREADS"] == "2"
    assert first_receipt["environment"]["JAX_PLATFORMS"] == "cpu"
    assert "PRIVATE_TOKEN" not in first_receipt["environment"]
    assert second_receipt["returncode"] == 7
    assert (tmp_path / "command-002.stderr").read_text() == "second stderr\n"


def test_stream_survives_terminated_parent(tmp_path: Path) -> None:
    child = "import os,time; print(os.getpid(), flush=True); time.sleep(60)"
    harness = (
        "import runpy,sys; from pathlib import Path; "
        "command=runpy.run_path(sys.argv[1])['_command']; "
        "command([sys.executable, '-c', sys.argv[3]], evidence=Path(sys.argv[2]))"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", harness, str(_SCRIPT), str(tmp_path), child]
    )
    child_pid = None
    try:
        stream = tmp_path / "command-001.stdout"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if stream.exists() and stream.read_text().strip():
                child_pid = int(stream.read_text().strip())
                break
            if parent.poll() is not None:
                break
            time.sleep(0.01)
        assert child_pid is not None
        parent.kill()
        parent.wait(timeout=10)
        assert stream.read_text().strip() == str(child_pid)
        assert (
            json.loads((tmp_path / "command-001.json").read_text())["returncode"] is None
        )
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=10)
        if child_pid is not None:
            try:
                os.kill(child_pid, 9)
            except ProcessLookupError:
                pass


def test_nonempty_evidence_directory_is_refused(tmp_path: Path) -> None:
    marker = tmp_path / "existing"
    marker.write_text("keep")
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--output-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "must be empty" in json.loads(result.stdout)["detail"]
    assert marker.read_text() == "keep"


def test_existing_child_stream_is_not_overwritten(tmp_path: Path) -> None:
    stream = tmp_path / "command-001.stdout"
    stream.write_text("existing")
    with pytest.raises(FileExistsError):
        _command([sys.executable, "-c", "print('unexpected')"], evidence=tmp_path)
    assert stream.read_text() == "existing"


def test_failed_run_persists_final_result(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--output-dir",
            str(evidence),
            "--python",
            "nonexistent-isolation-interpreter",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    record = json.loads(result.stdout)
    assert record["status"] == "failed"
    assert record["evidence_dir"] == str(evidence)
    assert json.loads((evidence / "result.json").read_text()) == record
    assert str(evidence) in result.stderr
