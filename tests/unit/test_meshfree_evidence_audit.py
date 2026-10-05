# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Consumer-visible regression coverage for meshfree evidence retention."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

import numpy as np
import pytest

from benchmarks import meshfree_scaling as scaling


@pytest.mark.parametrize(
    "driver_name",
    ["benchmarks/meshfree_closure.py", "tools/meshfree_qualification.py"],
    ids=["closure-benchmark", "qualification-driver"],
)
@pytest.mark.parametrize(
    "source_name",
    [
        "benchmarks/meshfree_closure_adaptive_restart.py",
        "examples/meshfree_adaptive_learning.py",
    ],
    ids=["scientific-helper", "independent-oracle"],
)
def test_helper_and_oracle_source_changes_rebind_benchmark_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_name: str, driver_name: str
) -> None:
    root = scaling.PROJECT_ROOT
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(root / name, tmp_path / name)
    for name in ("benchmarks", "examples"):
        shutil.copytree(
            root / name,
            tmp_path / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.json"),
        )
    (tmp_path / "phydrax").mkdir()
    (tmp_path / "phydrax" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "tools").mkdir()
    shutil.copyfile(
        root / "tools" / "meshfree_qualification.py",
        tmp_path / "tools" / "meshfree_qualification.py",
    )
    monkeypatch.setattr(scaling, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        scaling, "__file__", str(tmp_path / "benchmarks" / "meshfree_scaling.py")
    )
    driver = tmp_path / driver_name
    config = scaling.MeshfreeConfig(sizes=(64,))
    first = scaling.make_record(config, [], driver)
    source = tmp_path / source_name
    source.write_text(
        source.read_text(encoding="utf-8")
        + "\n# Changed scientific producer or oracle.\n",
        encoding="utf-8",
    )
    second = scaling.make_record(config, [], driver)
    assert first["identity"] != second["identity"]


@pytest.mark.parametrize(
    "refusal", [True, False], ids=["capacity-refusal", "provider-failure"]
)
def test_post_execution_failure_preserves_recorder_and_reservation(
    monkeypatch: pytest.MonkeyPatch, refusal: bool
) -> None:
    from phydrax.discretization import meshfree

    config = scaling.MeshfreeConfig(sizes=(64,))
    reservation = config.check_capacity(64)
    relation = SimpleNamespace(
        valid=np.ones((64, 1), dtype=np.bool_), source_indices=np.arange(64)[:, None]
    )
    prepared = SimpleNamespace(neighborhood=SimpleNamespace(relation=relation))

    class Neighborhood:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def prepare(self) -> SimpleNamespace:
            return SimpleNamespace()

    def local_fit(*args: Any, **kwargs: Any) -> SimpleNamespace:
        return prepared

    def execute(
        action: Any,
        values: Any,
        config: scaling.MeshfreeConfig,
        recorder: scaling.PhaseRecorder,
    ) -> NoReturn:
        recorder.run("warm", lambda: np.ones(1, dtype=np.float64))
        recorder.annotate("compilation", compiler={"estimated_device_memory_bytes": 123})
        if refusal:
            raise scaling.DeclaredCapacityRefusal("post-execution sentinel")
        raise RuntimeError("post-execution sentinel")

    monkeypatch.setattr(meshfree, "MeshfreeNeighborhoodPlan", Neighborhood)
    monkeypatch.setattr(meshfree, "prepare_local_stencils", local_fit)
    monkeypatch.setattr(scaling, "execution_evidence", execute)
    row = scaling.admitted_rows(scaling.measure_capacity, config)[0]
    assert row["status"] == ("post-execution-refusal" if refusal else "failed")
    assert row["execution_stage"] == "post-admission"
    assert row["reserved_working_set_bytes"] == reservation
    assert row["phases"]["warm"]["occurrences"]
    assert (
        row["phases"]["compilation"]["compiler"]["estimated_device_memory_bytes"] == 123
    )
    assert row["failure"]["message"] == "post-execution sentinel"
    assert any("execute" in line for line in row["failure"]["traceback"])


def test_later_derivative_failure_retains_completed_occurrences() -> None:
    recorder = scaling.PhaseRecorder()
    recorder.run("jvp", lambda: np.ones(1, dtype=np.float64))
    recorder.annotate("jvp", completed_extra="retained")
    try:
        raise RuntimeError("second repeat")
    except RuntimeError as error:
        recorder._failures["jvp"] = scaling.failure_record(error)
    phase = recorder.record()["jvp"]
    assert phase["status"] == "failed"
    assert len(phase["occurrences"]) == 1
    assert phase["wall_seconds"] == phase["occurrences"][0]["wall_seconds"]
    assert phase["memory"]
    assert phase["completed_extra"] == "retained"
    assert phase["failure"]["message"] == "second repeat"


def test_effective_support_sweeps_refuse_incompatible_baselines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    def admitted(measure: Any, config: scaling.MeshfreeConfig) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(scaling, "admitted_rows", admitted)
    config = scaling.MeshfreeConfig(sizes=(64,))
    first = scaling.run(config, neighbor_capacities=(24,))
    second = scaling.run(config, neighbor_capacities=(48,))
    assert first["workload_id"] != second["workload_id"]
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(first), encoding="utf-8")
    scaling.apply_baseline(second, baseline)
    assert second["performance_before_after"]["status"] == "unavailable"
    reordered = scaling.run(config, neighbor_capacities=(48, 24, 48))
    canonical = scaling.run(config, neighbor_capacities=(24, 48))
    assert reordered["workload_id"] == canonical["workload_id"]


def _qualification_context() -> dict[str, Any]:
    return {
        "build_id": "build",
        "environment_id": "environment",
        "backend": "cpu",
        "topology": "1-process-1-device",
        "precision": "float64",
        "run_spec_id": "run",
        "criteria_at": 100,
        "campaign_spec_id": "campaign",
        "started_at": 101,
        "observed_at": 102,
        "issued_at": 103,
    }


def test_full_originating_failure_is_content_addressed() -> None:
    from tools import meshfree_qualification as qualification

    def originating_a() -> dict[str, Any]:
        try:
            raise RuntimeError("sentinel")
        except RuntimeError as error:
            return qualification.failure_record(error)

    def originating_b() -> dict[str, Any]:
        try:
            raise RuntimeError("sentinel")
        except RuntimeError as error:
            return qualification.failure_record(error)

    failure_a = originating_a()
    failure_b = originating_b()

    row: dict[str, Any] = {
        "support_tuple_id": "a" * 64,
        "capacity": 64,
        "seed": 0,
        "label": "failed-row",
        "failure": failure_a,
        "gates": [qualification._execution_gate("failed", "RuntimeError: sentinel")],
    }
    first = qualification._campaign_evidence("Q1", [row], [], _qualification_context())
    row["failure"] = failure_b
    second = qualification._campaign_evidence("Q1", [row], [], _qualification_context())
    a = first["records"][0]["raw_observation"]
    b = second["records"][0]["raw_observation"]
    assert a["raw_artifact_id"] != b["raw_artifact_id"]
    assert a["observations"][0]["failure"] == failure_a
    assert b["observations"][0]["failure"] == failure_b
    assert failure_a["message"] == failure_b["message"]
    assert failure_a["traceback"] != failure_b["traceback"]


def test_child_runtime_binds_row_and_aggregate_evidence() -> None:
    from phydrax._fingerprint import canonical_fingerprint
    from tools import meshfree_qualification as qualification

    support_id = "a" * 64
    producer = {
        "environment": {"jax": {"backend": "cpu", "devices": [{}, {}, {}, {}]}},
        "identity": {"source_build_fingerprint": "child-source"},
        "driver_dependencies": {"scientific-helper": "child-helper"},
        "backend": "cpu",
        "topology": "1-process-4-device",
        "precision": "mixed",
    }
    item = qualification._execution_gate("failed", "originating child failure")
    aggregate = {**item, "gate": "aggregate", "support_tuple_id": support_id}
    row = {
        "support_tuple_id": support_id,
        "capacity": 64,
        "seed": 0,
        "label": "forced-child",
        "producer": producer,
        "gates": [item],
    }
    result = qualification._campaign_evidence(
        "Q16", [row], [aggregate], _qualification_context()
    )
    assert len(result["records"]) == 2
    for record in result["records"]:
        evidence = record["evidence"]
        assert evidence["topology"] == "1-process-4-device"
        assert evidence["precision"] == "mixed"
        assert evidence["environment_id"] == canonical_fingerprint(
            producer["environment"]
        )
        assert evidence["build_id"] == canonical_fingerprint(
            {
                "identity": producer["identity"],
                "driver_dependencies": producer["driver_dependencies"],
            }
        )
        assert (
            record["raw_observation"]["producer_context"]["topology"]
            == "1-process-4-device"
        )


def test_point_limit_refusal_has_no_execution_evidence() -> None:
    config = scaling.MeshfreeConfig(sizes=(64,), max_points=32)
    row = scaling.admitted_rows(scaling.measure_capacity, config)[0]
    assert row["status"] == "declared-refusal"
    assert row.get("execution_stage", "pre-admission") == "pre-admission"
    assert "phases" not in row
    assert "compiler" not in row


def test_failed_invocation_retains_all_recorders_and_reservation_attempts() -> None:
    config = scaling.MeshfreeConfig(sizes=(64,))

    def measure(capacity: int, seed: int, config: scaling.MeshfreeConfig) -> NoReturn:
        config.check_capacity(capacity)
        first = scaling.PhaseRecorder()
        first.run("warm", lambda: np.ones(1, dtype=np.float64), scope="first")
        first.annotate("warm", scientific_stage="first")
        second = scaling.PhaseRecorder()
        second.run("warm", lambda: np.ones(1, dtype=np.float64), scope="second")
        second.annotate("warm", scientific_stage="second")
        scaling.declare_reservation(
            config.resource_bytes + 1, config, scope="late-reservation"
        )
        raise AssertionError("An over-budget reservation cannot be admitted")

    row = scaling.admitted_rows(measure, config)[0]
    assert row["status"] == "post-execution-refusal"
    assert [item["status"] for item in row["reservations"]] == ["admitted", "refused"]
    warm = row["phases"]["warm"]
    assert [item["scope"] for item in warm["occurrences"]] == ["first", "second"]
    assert [item["scientific_stage"] for item in warm["annotations"]] == [
        "first",
        "second",
    ]
    assert warm["wall_seconds"] == sum(
        item["wall_seconds"] for item in warm["occurrences"]
    )
    assert warm["memory"]


def test_nested_execution_scopes_reset_without_cross_invocation_evidence() -> None:
    config = scaling.MeshfreeConfig(sizes=(64,))

    def failing_inner(
        capacity: int, seed: int, config: scaling.MeshfreeConfig
    ) -> NoReturn:
        recorder = scaling.PhaseRecorder()
        recorder.run("jvp", lambda: np.ones(1, dtype=np.float64), scope="inner")
        raise RuntimeError("inner failure")

    with scaling.BenchmarkExecutionScope() as outer:
        recorder = scaling.PhaseRecorder()
        recorder.run("warm", lambda: np.ones(1, dtype=np.float64), scope="before-inner")
        inner = scaling.admitted_rows(failing_inner, config)[0]
        recorder.run("warm", lambda: np.ones(1, dtype=np.float64), scope="after-inner")
        outer_record = outer.record()
    assert inner["phases"]["jvp"]["status"] == "measured"
    assert outer_record["phases"]["jvp"]["status"] == "unavailable"
    assert [item["scope"] for item in outer_record["phases"]["warm"]["occurrences"]] == [
        "before-inner",
        "after-inner",
    ]
    with scaling.BenchmarkExecutionScope() as fresh:
        assert "phases" not in fresh.record()
        with pytest.raises(RuntimeError, match="invocation"):
            recorder.run("warm", lambda: np.ones(1, dtype=np.float64))
    with pytest.raises(RuntimeError, match="reused"):
        with outer:
            raise AssertionError("A completed scope cannot be reentered")


def test_qualification_post_execution_refusal_retains_raw_resource_evidence() -> None:
    from dataclasses import replace

    from phydrax.qualification import SupportTuple
    from tools import meshfree_qualification as qualification

    config = scaling.MeshfreeConfig(sizes=(64,))

    def fail() -> NoReturn:
        config.check_capacity(64)
        recorder = scaling.PhaseRecorder()
        recorder.run("warm", lambda: np.ones(1, dtype=np.float64))
        scaling.declare_reservation(
            config.resource_bytes + 1, config, scope="late-reservation"
        )
        raise AssertionError("An over-budget reservation cannot be admitted")

    planned = replace(qualification._plan_q16(config)[0], execute=fail)
    support = SupportTuple(planned.capability, planned.support)
    row = qualification._execute(planned, support, None)
    assert row["status"] == "failed"
    assert row["execution_stage"] == "post-admission"
    assert row["phases"]["warm"]["occurrences"]
    result = qualification._campaign_evidence("Q16", [row], [], _qualification_context())
    raw = result["records"][0]["raw_observation"]["observations"][0]
    assert raw["failure"] == row["failure"]
    assert raw["phases"] == row["phases"]
    assert [item["status"] for item in raw["reservations"]] == ["admitted", "refused"]


def test_copied_thread_context_cannot_attach_foreign_evidence() -> None:
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context

    def foreign() -> None:
        recorder = scaling.PhaseRecorder()
        recorder.annotate("warm", foreign_thread=True)

    with scaling.BenchmarkExecutionScope() as scope:
        recorder = scaling.PhaseRecorder()
        recorder.run("warm", lambda: np.ones(1, dtype=np.float64))
        context = copy_context()
        with ThreadPoolExecutor(max_workers=1) as workers:
            workers.submit(context.run, foreign).result()
        evidence = scope.record()
    assert "foreign_thread" not in evidence["phases"]["warm"]
    assert len(evidence["phases"]["warm"]["occurrences"]) == 1
