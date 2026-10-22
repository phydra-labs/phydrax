"""Family reporting survives interruption without changing scientific work."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import weakref
from pathlib import Path
from threading import Event
from time import perf_counter
from typing import Any

import numpy as np
import pytest

from phydrax._meshcore import NativeExecutionLimits, NativeExecutionScope
from tools.meshing_qualification import _family_progress as owner
from tools.meshing_qualification._runtime import Deadline, observe_stages, StageLedger


_COHORT = {
    "source_revision": "source-test", "build_id": "build-test",
    "environment_id": "environment-test", "backend": "cpu",
    "meshcore_build": {"binary_digest": "native-test"},
}


def test_progress_records_entries_exits_and_immutable_first_failure(tmp_path: Path) -> None:
    path = tmp_path / "run-progress.json"
    progress = owner.FamilyProgress(path, {"timeout": 60.0})
    record = {"runtime": _COHORT, "status": "running"}
    deadline = Deadline(perf_counter() + 60)
    try:
        with observe_stages(progress):
            ledger = StageLedger(record, deadline=deadline)
            assert ledger.run("accepted", lambda: 42) == 42
            with pytest.raises(ValueError, match="first scientific error"):
                ledger.run("first", lambda: (_ for _ in ()).throw(ValueError("first scientific error")))
            with pytest.raises(RuntimeError, match="later error"):
                ledger.run("later", lambda: (_ for _ in ()).throw(RuntimeError("later error")))
    finally:
        progress.close()
    saved = owner.read_family_progress(path)
    assert saved["cohort"] == _COHORT
    assert [(event["phase"], event["event"]) for event in saved["events"]] == [
        ("accepted", "entry"), ("accepted", "exit"),
        ("first", "entry"), ("first", "failure"), ("first", "exit"),
        ("later", "entry"), ("later", "failure"), ("later", "exit"),
    ]
    assert saved["first_failure"]["message"] == "first scientific error"
    assert "ValueError" in saved["first_failure"]["traceback"]
    expiries = [event["deadline_monotonic_seconds"] for event in saved["events"]]
    assert max(expiries) - min(expiries) < 0.001
    assert not progress.errors


def test_progress_records_live_resource_semantics_without_charging(tmp_path: Path) -> None:
    progress = owner.FamilyProgress(tmp_path / "live.json", {})
    record = {"runtime": _COHORT}
    try:
        with NativeExecutionScope(NativeExecutionLimits(100, 100, 10, 1000000, 60)) as scope:
            scope.charge(work=3, geometry_queries=2)
            before = scope.remaining()
            with observe_stages(progress):
                assert StageLedger(record).run("observe", lambda: 17) == 17
            after = scope.remaining()
            assert before.remaining_work_units == after.remaining_work_units
            assert before.remaining_geometry_queries == after.remaining_geometry_queries
            assert before.remaining_managed_bytes == after.remaining_managed_bytes
    finally:
        progress.close()
    resources = owner.read_family_progress(progress.path)["latest"]["resources"]
    assert resources["cumulative_admitted_or_consumed_work_units"] == 3
    assert resources["cumulative_admitted_or_consumed_geometry_queries"] == 2
    assert resources["simultaneous_admitted_managed_bytes"] == 0
    assert resources["allocator_bookkeeping_bytes"] == 0


def test_progress_does_not_retain_arrays_or_block_science(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered, release = Event(), Event()
    original = owner._atomic_snapshot

    def blocked(path: Path, payload: bytes) -> None:
        entered.set()
        assert release.wait(10)
        original(path, payload)

    monkeypatch.setattr(owner, "_atomic_snapshot", blocked)
    progress = owner.FamilyProgress(tmp_path / "nonblocking.json", {})
    array = np.arange(8)
    reference = weakref.ref(array)
    try:
        with observe_stages(progress):
            returned = StageLedger({"runtime": _COHORT}).run("array", lambda value=array: value)
        assert returned is array
        assert entered.wait(5)
        del returned, array
        assert reference() is None
    finally:
        release.set()
        progress.close()
    assert not progress.errors


def test_atomic_writer_failure_preserves_prefix_and_scientific_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = tmp_path / "report-failure"
    monkeypatch.setenv("FAMILY_COMPACT_PROFILE_PREFIX", str(prefix))
    original = owner.os.replace
    replacements = 0

    def fail_second(source: Path, target: Path) -> None:
        nonlocal replacements
        replacements += 1
        if replacements == 2:
            raise OSError("injected reporting failure")
        original(source, target)

    monkeypatch.setattr(owner.os, "replace", fail_second)
    record: dict[str, Any] = {"runtime": _COHORT, "status": "passed"}

    def science() -> dict[str, Any]:
        assert StageLedger(record).run("science", lambda: 23) == 23
        return record

    assert owner.run_with_family_progress({}, science) is record
    assert record["status"] == "passed"
    assert record["reporting_status"] == "failed"
    assert record["reporting_failures"][0]["message"] == "injected reporting failure"
    saved = owner.read_family_progress(Path(str(prefix) + "-progress.json"))
    assert len(saved["events"]) == 1
    assert saved["latest"]["event"] == "entry"
    assert list(tmp_path.glob("*.tmp")) == []


def test_interrupted_process_has_readable_cohort_bound_stage_prefix(tmp_path: Path) -> None:
    path = tmp_path / "interrupted.json"
    script = """
import os, sys
from pathlib import Path
from tools.meshing_qualification._family_progress import FamilyProgress
from tools.meshing_qualification._runtime import StageLedger, observe_stages
progress = FamilyProgress(Path(sys.argv[1]), {'timeout': 1200})
record = {'runtime': {'source_revision':'s', 'build_id':'b', 'backend':'cpu', 'meshcore_build':{'binary_digest':'n'}}}
def interrupted():
    progress.flush()
    os._exit(73)
with observe_stages(progress):
    StageLedger(record).run('association', interrupted)
"""
    completed = subprocess.run([sys.executable, "-c", script, str(path)], check=False, timeout=30)
    assert completed.returncode == 73
    reader = """
import json, sys
from pathlib import Path
from tools.meshing_qualification._family_progress import read_family_progress
print(json.dumps(read_family_progress(Path(sys.argv[1]))))
"""
    restarted = subprocess.run(
        [sys.executable, "-c", reader, str(path)], check=True, text=True,
        capture_output=True, timeout=30,
    )
    saved = json.loads(restarted.stdout)
    assert saved["latest"]["phase"] == "association"
    assert saved["latest"]["event"] == "entry"
    assert saved["first_failure"] is None  # Interruption is not a fabricated scientific failure.
    assert saved["cohort"]["meshcore_build"]["binary_digest"] == "n"
    assert not saved["scientific_restart_checkpoint"]


def test_cold_writer_uses_separate_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prefix = tmp_path / "cold"
    monkeypatch.setenv("FAMILY_COMPACT_PROFILE_PREFIX", str(prefix))
    record = {"runtime": _COHORT, "status": "passed"}
    assert owner.run_with_family_progress({}, lambda: record, cold=True) is record
    saved = owner.read_family_progress(Path(f"{prefix}.cold-{os.getpid()}-progress.json"))
    assert saved["latest"]["event"] == "scenario_exit"
    assert not Path(str(prefix) + "-progress.json").exists()


def test_reader_rejects_noncontiguous_evidence(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"schema": "family-progress-v1", "events": [{"sequence": 1}]}))
    with pytest.raises(ValueError, match="contiguous"):
        owner.read_family_progress(path)
