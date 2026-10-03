#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib.metadata
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks._runtime import (
    benchmark_driver_fingerprint,
    BenchmarkIdentity,
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    CompilerEvidence,
    DurationDistribution,
    installed_package_fingerprint,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
    source_build_fingerprint,
    synchronize,
    validate_benchmark_record,
)
from phydrax._fingerprint import canonical_fingerprint


def test_benchmark_runtime_scenario_1() -> None:
    distribution = DurationDistribution((0.001, 0.003, 0.002))

    assert distribution.count == 3
    assert distribution.minimum_seconds == pytest.approx(0.001)
    assert distribution.median_seconds == pytest.approx(0.002)
    assert distribution.mean_seconds == pytest.approx(0.002)
    assert distribution.population_std_seconds == pytest.approx(
        float(np.std([0.001, 0.003, 0.002]))
    )
    assert distribution.maximum_seconds == pytest.approx(0.003)
    milliseconds = distribution.to_milliseconds_dict()
    assert milliseconds == {
        "count": 3,
        "samples_ms": [1.0, 3.0, 2.0],
        "min_ms": 1.0,
        "median_ms": 2.0,
        "mean_ms": 2.0,
        "std_ms": pytest.approx(float(np.std([1.0, 3.0, 2.0]))),
        "max_ms": 3.0,
    }
    assert DurationDistribution(()).to_seconds_dict() == {
        "count": 0,
        "samples_seconds": [],
        "min_seconds": None,
        "median_seconds": None,
        "mean_seconds": None,
        "std_seconds": None,
        "max_seconds": None,
    }
    for samples in ((-1.0,), (math.nan,), (math.inf,)):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            DurationDistribution(samples)
    with pytest.raises(ValueError, match="Duration unit"):
        DurationDistribution((1.0,)).to_dict(unit="minutes")  # ty: ignore[invalid-argument-type]
    value = _NestedArrays(
        {"array": jnp.ones((2,), dtype=jnp.float32)},
        (np.ones((3,), dtype=np.float64), "host"),
    )

    assert synchronize(value) is value
    assert logical_array_bytes(value) == 2 * 4 + 3 * 8
    evidence = compiler_evidence(
        {"flops": 101.2, "bytes accessed": 202.4},
        # ty: ignore[invalid-argument-type]
        _MemoryAnalysis(),
        source="xla-cost-analysis",
    )
    assert evidence.flops == 101
    assert evidence.bytes_accessed == 202
    assert evidence.estimated_device_memory_bytes == 60

    unavailable = compiler_evidence(
        None,
        None,
        source="xla-cost-analysis",
        unavailable_reason="unsupported backend",
    )
    assert unavailable.estimated_device_memory_bytes is None
    assert unavailable.unavailable_reason == "unsupported backend"

    not_applicable = CompilerEvidence(0, 0, 0, 0, 0, 0, "not-applicable")
    assert not_applicable.estimated_device_memory_bytes == 0
    with pytest.raises(ValueError, match="requires a reason"):
        CompilerEvidence(None, None, None, None, None, None, "xla-cost-analysis")


def test_measure_repeated_synchronizes_warmups_and_retains_every_sample() -> None:
    events = []
    counter = 0

    def operation() -> Any:
        nonlocal counter
        counter += 1
        events.append(("operation", counter))
        return counter

    def synchronizer(value: Any) -> Any:
        events.append(("synchronize", value))
        return value

    result, distribution = measure_repeated(
        operation,
        warmup=2,
        repeats=3,
        synchronizer=synchronizer,
    )

    assert result == 5
    assert distribution.count == 3
    assert events == [
        ("operation", 1),
        ("synchronize", 1),
        ("operation", 2),
        ("synchronize", 2),
        ("operation", 3),
        ("synchronize", 3),
        ("operation", 4),
        ("synchronize", 4),
        ("operation", 5),
        ("synchronize", 5),
    ]
    with pytest.raises(ValueError, match="repeats must be positive"):
        measure_repeated(operation, warmup=0, repeats=0)
    with pytest.raises(ValueError, match="warmup must be nonnegative"):
        measure_repeated(operation, warmup=-1, repeats=1)


def test_host_and_synchronized_measurement_have_distinct_boundaries() -> None:
    events = []

    def operation() -> str:
        events.append("operation")
        return "value"

    value, host_seconds = measure_host(operation)
    assert value == "value"
    assert host_seconds >= 0.0
    assert events == ["operation"]

    value, synchronized_seconds = measure_synchronized(
        operation,
        synchronizer=lambda result: events.append(f"sync:{result}"),
    )
    assert value == "value"
    assert synchronized_seconds >= 0.0
    assert events == ["operation", "operation", "sync:value"]


def test_lowering_and_compilation_are_ordered_independent_host_phases() -> None:
    events = []

    def lower() -> str:
        events.append("lower")
        return "lowered"

    def compile(lowered: Any) -> str:
        events.append(f"compile:{lowered}")
        return "compiled"

    compiled, timing = measure_lower_and_compile(lower, compile)

    assert compiled == "compiled"
    assert events == ["lower", "compile:lowered"]
    assert timing.lowering_seconds >= 0.0
    assert timing.compilation_seconds >= 0.0


@dataclass(frozen=True)
class _NestedArrays:
    first: object
    second: object


@dataclass(frozen=True)
class _MemoryAnalysis:
    argument_size_in_bytes: int = 10
    output_size_in_bytes: int = 20
    temp_size_in_bytes: int = 30
    generated_code_size_in_bytes: int = 40


class _Distribution:
    def __init__(self, name: str, version: str) -> None:
        self.metadata = {"Name": name}
        self.version = version


def test_installed_package_fingerprint_normalizes_order_and_spelling(
    monkeypatch: Any,
) -> None:
    first = (_Distribution("A_Package", "1"), _Distribution("b.package", "2"))
    second = (_Distribution("B-PACKAGE", "2"), _Distribution("a-package", "1"))
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: first)
    first_fingerprint = installed_package_fingerprint()
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: second)
    assert installed_package_fingerprint() == first_fingerprint

    conflicting = (_Distribution("same", "1"), _Distribution("same", "2"))
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: conflicting)
    with pytest.raises(ValueError, match="conflicting versions"):
        installed_package_fingerprint()


def test_captured_environment_fingerprint_covers_serialized_runtime_evidence() -> None:
    environment = capture_environment()
    payload = environment.to_dict()
    observed = payload.pop("fingerprint")

    assert canonical_fingerprint(payload) == observed
    assert payload["jax"]["devices"]
    assert payload["package_fingerprint"]


def test_captured_environment_records_xla_worker_count(monkeypatch: Any) -> None:
    monkeypatch.setenv("NPROC", "3")
    environment = capture_environment()
    assert dict(environment.performance_environment)["NPROC"] == "3"


def _benchmark_source_tree(root: Path) -> Path:
    (root / "phydrax").mkdir()
    (root / "benchmarks").mkdir()
    (root / "phydrax" / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "benchmarks" / "_runtime.py").write_text("RUNTIME = 1\n", encoding="utf-8")
    driver = root / "benchmarks" / "driver.py"
    driver.write_text("DRIVER = 1\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    (root / "uv.lock").write_text("revision = 1\n", encoding="utf-8")
    return driver


def _stored_record(identity: BenchmarkIdentity) -> dict[str, Any]:
    return {
        "identity": identity.to_dict(),
        "cases": [{"last_step_evidence": {"accepted": True, "residual": 0.0}}],
    }


def test_benchmark_identity_detects_source_mutation(tmp_path: Path) -> None:
    driver = _benchmark_source_tree(tmp_path)
    initial = capture_benchmark_identity(tmp_path, driver, ("accepted", "residual"))
    stored = _stored_record(initial)

    source = tmp_path / "phydrax" / "model.py"
    source.write_text("VALUE = 2\n", encoding="utf-8")
    changed = capture_benchmark_identity(tmp_path, driver, ("accepted", "residual"))

    assert source_build_fingerprint(tmp_path) != initial.source_build_fingerprint
    with pytest.raises(ValueError, match="source/build fingerprint"):
        validate_benchmark_record(stored, changed)


def test_benchmark_identity_detects_driver_mutation(tmp_path: Path) -> None:
    driver = _benchmark_source_tree(tmp_path)
    initial = capture_benchmark_identity(tmp_path, driver, ("accepted", "residual"))
    stored = _stored_record(initial)

    driver.write_text("DRIVER = 2\n", encoding="utf-8")
    changed = capture_benchmark_identity(tmp_path, driver, ("accepted", "residual"))

    assert (
        benchmark_driver_fingerprint(tmp_path, driver)
        != initial.benchmark_driver_fingerprint
    )
    with pytest.raises(ValueError, match="driver fingerprint"):
        validate_benchmark_record(stored, changed)


def test_benchmark_record_rejects_evidence_schema_field_set_mismatch(
    tmp_path: Path,
) -> None:
    driver = _benchmark_source_tree(tmp_path)
    identity = capture_benchmark_identity(tmp_path, driver, ("accepted", "residual"))
    stored = _stored_record(identity)
    stored["identity"]["evidence_schema"]["fields"] = ["accepted"]

    with pytest.raises(ValueError, match="evidence field set/signature"):
        validate_benchmark_record(stored, identity)


def test_benchmark_record_rejects_case_evidence_field_set_mismatch(
    tmp_path: Path,
) -> None:
    driver = _benchmark_source_tree(tmp_path)
    identity = capture_benchmark_identity(tmp_path, driver, ("accepted", "residual"))
    stored = _stored_record(identity)
    stored["cases"][0]["last_step_evidence"].pop("residual")

    with pytest.raises(ValueError, match="evidence field set"):
        validate_benchmark_record(stored, identity)


def test_adaptive_sphere_refuses_point_limit_before_preparation() -> None:
    from benchmarks.meshfree_closure_adaptive_restart import (
        measure_adaptive_sphere_zonal_peak,
    )
    from benchmarks.meshfree_scaling import DeclaredCapacityRefusal, MeshfreeConfig

    config = MeshfreeConfig(sizes=(768,), dimension=3, max_points=64)
    with pytest.raises(DeclaredCapacityRefusal, match="max_points=64"):
        measure_adaptive_sphere_zonal_peak(768, 0, config)
