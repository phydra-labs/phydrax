#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib.metadata
import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks._runtime import (
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
    synchronize,
)
from phydrax._fingerprint import canonical_fingerprint


def test_duration_distribution_derives_both_unit_views_from_raw_samples():
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


def test_duration_distribution_empty_and_invalid_contracts():
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
        DurationDistribution((1.0,)).to_dict(unit="minutes")  # type: ignore[arg-type]


def test_measure_repeated_synchronizes_warmups_and_retains_every_sample():
    events = []
    counter = 0

    def operation():
        nonlocal counter
        counter += 1
        events.append(("operation", counter))
        return counter

    def synchronizer(value):
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


def test_host_and_synchronized_measurement_have_distinct_boundaries():
    events = []

    def operation():
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


def test_lowering_and_compilation_are_ordered_independent_host_phases():
    events = []

    def lower():
        events.append("lower")
        return "lowered"

    def compile(lowered):
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


def test_synchronize_and_logical_bytes_cover_nested_pytrees():
    value = _NestedArrays(
        {"array": jnp.ones((2,), dtype=jnp.float32)},
        (np.ones((3,), dtype=np.float64), "host"),
    )

    assert synchronize(value) is value
    assert logical_array_bytes(value) == 2 * 4 + 3 * 8


@dataclass(frozen=True)
class _MemoryAnalysis:
    argument_size_in_bytes: int = 10
    output_size_in_bytes: int = 20
    temp_size_in_bytes: int = 30
    generated_code_size_in_bytes: int = 40


def test_compiler_evidence_distinguishes_values_unavailability_and_not_applicable():
    evidence = compiler_evidence(
        {"flops": 101.2, "bytes accessed": 202.4},
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


class _Distribution:
    def __init__(self, name: str, version: str):
        self.metadata = {"Name": name}
        self.version = version


def test_installed_package_fingerprint_normalizes_order_and_spelling(monkeypatch):
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


def test_captured_environment_fingerprint_covers_serialized_runtime_evidence():
    environment = capture_environment()
    payload = environment.to_dict()
    observed = payload.pop("fingerprint")

    assert canonical_fingerprint(payload) == observed
    assert payload["jax"]["devices"]
    assert payload["package_fingerprint"]
