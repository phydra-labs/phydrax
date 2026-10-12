#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks._memory import (
    evaluate_memory_gate,
    fitted_scaling_exponent,
    host_allocation_peak,
    MemoryGatePolicy,
)
from benchmarks.memory_scaling import build_summary, compare_gate, compare_growth


def test_memory_gate_decides_at_the_largest_capacity() -> None:
    policy = MemoryGatePolicy()
    capacities = (64, 128, 256)

    fused = evaluate_memory_gate(
        capacities, (4_000, 8_000, 16_000), (1_000, 4_000, 9_000), policy
    )
    bounded = evaluate_memory_gate(
        capacities, (4_000, 16_000, 64_000), (4_000, 8_000, 16_000), policy
    )
    exact_fraction = evaluate_memory_gate(capacities, (10, 20, 40), (40, 40, 20), policy)

    assert fused.decision == "retain-baseline"
    assert fused.capacity == 256
    assert bounded.decision == "adopt"
    assert (bounded.baseline_bytes, bounded.candidate_bytes) == (64_000, 16_000)
    assert exact_fraction.decision == "adopt"
    assert (
        evaluate_memory_gate(capacities, (0, 0, 0), (0, 0, 0), policy).decision
        == "retain-baseline"
    )


@pytest.mark.parametrize(
    ("capacities", "baseline", "candidate"),
    [
        pytest.param((64,), (1,), (1,), id="single-capacity"),
        pytest.param((64, 64), (1, 2), (1, 2), id="non-increasing"),
        pytest.param((64, 128), (1, 2), (1,), id="length-mismatch"),
        pytest.param((64, 128), (1, -2), (1, 2), id="negative-bytes"),
    ],
)
def test_memory_gate_refuses_malformed_series(
    capacities: tuple[int, ...],
    baseline: tuple[int, ...],
    candidate: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        evaluate_memory_gate(capacities, baseline, candidate, MemoryGatePolicy())


def test_fitted_scaling_exponent_recovers_power_laws() -> None:
    capacities = (16, 32, 64, 128)

    assert fitted_scaling_exponent(
        capacities, tuple(8 * n for n in capacities)
    ) == pytest.approx(1.0)
    assert fitted_scaling_exponent(
        capacities, tuple(8 * n * n for n in capacities)
    ) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        fitted_scaling_exponent(capacities, (0, 1, 2, 3))


def test_host_allocation_peak_observes_released_scratch() -> None:
    scratch_bytes = 8 * 1024 * 1024

    def operation() -> float:
        scratch = np.ones(scratch_bytes // 8, dtype=np.float64)
        return float(scratch.sum())

    value, peak = host_allocation_peak(operation)

    assert value == scratch_bytes // 8
    assert peak >= scratch_bytes


@pytest.mark.parametrize(
    ("candidate", "decision"),
    [
        pytest.param((40, 40, 19), "adopt", id="below-boundary"),
        pytest.param((40, 40, 20), "adopt", id="at-boundary"),
        pytest.param((0, 0, 21), "retain-baseline", id="above-boundary"),
    ],
)
def test_compare_gate_recomputes_largest_capacity_decision(
    candidate: tuple[int, ...], decision: str
) -> None:
    capacities = (64, 128, 256)
    baseline = (10, 20, 40)

    result = compare_gate(capacities, baseline, candidate, decision)

    assert (
        result
        == evaluate_memory_gate(
            capacities, baseline, candidate, MemoryGatePolicy()
        ).to_dict()
    )


@pytest.mark.parametrize(
    ("candidate", "declared_decision"),
    [
        pytest.param((1, 2, 10), "retain-baseline", id="undeclared-adoption"),
        pytest.param((1, 2, 30), "adopt", id="unsupported-adoption"),
        pytest.param((1, 2, 10), "unknown", id="unknown-decision"),
    ],
)
def test_compare_gate_rejects_declared_disagreement(
    candidate: tuple[int, ...], declared_decision: str
) -> None:
    with pytest.raises(ValueError):
        compare_gate((64, 128, 256), (10, 20, 40), candidate, declared_decision)


@pytest.mark.parametrize(
    "candidate",
    [
        pytest.param((0, 0), id="zero-candidate"),
        pytest.param((1, 2), id="positive-candidate"),
    ],
)
def test_compare_gate_zero_baseline_retains(
    candidate: tuple[int, ...],
) -> None:
    result = compare_gate((64, 128), (0, 0), candidate, "retain-baseline")

    assert result["decision"] == "retain-baseline"
    with pytest.raises(ValueError):
        compare_gate((64, 128), (0, 0), candidate, "adopt")


@pytest.mark.parametrize(
    ("fraction", "candidate", "decision"),
    [
        pytest.param(0.25, (1, 10), "adopt", id="quarter-boundary"),
        pytest.param(0.25, (1, 11), "retain-baseline", id="above-quarter"),
        pytest.param(1.0, (1, 40), "adopt", id="equal-baseline"),
        pytest.param(1.0, (1, 41), "retain-baseline", id="above-baseline"),
    ],
)
def test_compare_gate_honors_declared_ratio_policy(
    fraction: float, candidate: tuple[int, ...], decision: str
) -> None:
    result = compare_gate(
        (64, 128), (20, 40), candidate, decision, maximum_candidate_fraction=fraction
    )

    assert result["decision"] == decision
    assert result["maximum_candidate_fraction"] == fraction


@pytest.mark.parametrize(
    "scale",
    [
        pytest.param(1, id="bytes"),
        pytest.param(1_024, id="kibibytes"),
        pytest.param(1_048_576, id="mebibytes"),
    ],
)
def test_compare_gate_is_independent_of_absolute_byte_scale(scale: int) -> None:
    capacities = (64, 128, 256)
    baseline = tuple(scale * value for value in (10, 20, 40))
    candidate = tuple(scale * value for value in (5, 10, 20))

    result = compare_gate(capacities, baseline, candidate, "adopt")

    assert result["decision"] == "adopt"
    assert result["candidate_bytes"] / result["baseline_bytes"] == 0.5


@pytest.mark.parametrize(
    ("capacities", "baseline", "candidate"),
    [
        pytest.param((), (), (), id="empty"),
        pytest.param((64,), (1,), (1,), id="single-capacity"),
        pytest.param((0, 64), (1, 2), (1, 2), id="zero-capacity"),
        pytest.param((-1, 64), (1, 2), (1, 2), id="negative-capacity"),
        pytest.param((64, 64), (1, 2), (1, 2), id="repeated-capacity"),
        pytest.param((128, 64), (1, 2), (1, 2), id="descending-capacity"),
        pytest.param((64, 128), (1,), (1, 2), id="baseline-length"),
        pytest.param((64, 128), (1, 2), (1,), id="candidate-length"),
        pytest.param((64, 128), (1, -2), (1, 2), id="negative-baseline"),
        pytest.param((64, 128), (1, 2), (1, -2), id="negative-candidate"),
    ],
)
def test_compare_gate_refuses_malformed_series(
    capacities: tuple[int, ...],
    baseline: tuple[int, ...],
    candidate: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        compare_gate(capacities, baseline, candidate, "retain-baseline")


@pytest.mark.parametrize(
    "fraction",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(-0.5, id="negative"),
        pytest.param(1.01, id="above-one"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinity"),
    ],
)
def test_compare_gate_refuses_invalid_ratio_policy(fraction: float) -> None:
    with pytest.raises(ValueError):
        compare_gate((64, 128), (20, 40), (5, 10), "adopt", fraction)


def test_compare_growth_accepts_linear_evidence() -> None:
    capacities = (16, 32, 64, 128)
    values = tuple(8 * capacity for capacity in capacities)

    result = compare_growth(capacities, values, maximum_exponent=1.1)

    assert result["exponent"] == pytest.approx(
        fitted_scaling_exponent(capacities, values)
    )


def test_compare_growth_rejects_quadratic_evidence_under_linear_bound() -> None:
    capacities = (16, 32, 64, 128)
    values = tuple(8 * capacity * capacity for capacity in capacities)

    with pytest.raises(ValueError):
        compare_growth(capacities, values, maximum_exponent=1.1)


@pytest.mark.parametrize(
    "scale",
    [
        pytest.param(1, id="bytes"),
        pytest.param(1_024, id="kibibytes"),
        pytest.param(1_048_576, id="mebibytes"),
    ],
)
def test_compare_growth_is_independent_of_absolute_byte_scale(scale: int) -> None:
    capacities = (16, 32, 64, 128)

    result = compare_growth(
        capacities,
        tuple(scale * capacity for capacity in capacities),
        maximum_exponent=1.1,
    )

    assert result["exponent"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("capacities", "values"),
    [
        pytest.param((), (), id="empty"),
        pytest.param((64,), (1,), id="single-capacity"),
        pytest.param((0, 64), (1, 2), id="zero-capacity"),
        pytest.param((64, 64), (1, 2), id="repeated-capacity"),
        pytest.param((128, 64), (1, 2), id="descending-capacity"),
        pytest.param((64, 128), (1,), id="length-mismatch"),
        pytest.param((64, 128), (0, 1), id="zero-bytes"),
        pytest.param((64, 128), (1, -2), id="negative-bytes"),
    ],
)
def test_compare_growth_refuses_malformed_series(
    capacities: tuple[int, ...], values: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError):
        compare_growth(capacities, values, maximum_exponent=1.1)


def test_compare_growth_accepts_exact_constant_boundary() -> None:
    result = compare_growth((16, 32, 64, 128), (8, 8, 8, 8), maximum_exponent=0.0)

    assert result["exponent"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "maximum_exponent",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
    ],
)
def test_compare_growth_refuses_nonfinite_bound(maximum_exponent: float) -> None:
    with pytest.raises(ValueError):
        compare_growth((16, 32, 64), (16, 32, 64), maximum_exponent)


def _committed_memory_record(filename: str) -> dict[str, Any]:
    benchmark_directory = Path(__file__).resolve().parents[2] / "benchmarks"
    return json.loads((benchmark_directory / filename).read_text(encoding="utf-8"))


def test_memory_summary_is_independent_of_file_and_mapping_order(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    filenames = (
        "memory_transport_attention.json",
        "memory_transport_attention_baseline.json",
        "memory_process_compression.json",
        "memory_process_compression_baseline.json",
    )
    for filename in filenames:
        record = _committed_memory_record(filename)
        (first / filename).write_text(json.dumps(record), encoding="utf-8")
    for filename in reversed(filenames):
        record = _committed_memory_record(filename)
        reversed_record = dict(reversed(tuple(record.items())))
        (second / filename).write_text(json.dumps(reversed_record), encoding="utf-8")

    summary = build_summary(first)
    assert summary == build_summary(second)
    assert [entry["record"] for entry in summary["records"]] == [
        "memory_process_compression.json",
        "memory_transport_attention.json",
    ]


def test_memory_summary_excludes_its_own_output(tmp_path: Path) -> None:
    for filename in (
        "memory_transport_attention.json",
        "memory_transport_attention_baseline.json",
    ):
        record = _committed_memory_record(filename)
        (tmp_path / filename).write_text(json.dumps(record), encoding="utf-8")
    expected = build_summary(tmp_path)
    (tmp_path / "memory_scaling.json").write_text("not an input record", encoding="utf-8")

    assert build_summary(tmp_path) == expected


def _limber_memory_record(
    *, baseline: bool, environment: str = "same-environment", decision: str = "adopt"
) -> dict[str, Any]:
    capacities = (16, 32, 64)
    record: dict[str, Any] = {
        "environment": {"fingerprint": environment},
        "identity": {
            "benchmark_driver_fingerprint": "limber-driver",
            "evidence_schema": {
                "fields": ["compiler", "logical_workspace_bytes", "reference_defect"],
                "signature": "limber-evidence",
            },
        },
        "records": [
            {
                "capacity": capacity,
                "compiler": {
                    "temporary_bytes": 8 * capacity * capacity
                    if baseline
                    else 8 * capacity,
                },
                "logical_workspace_bytes": 8 * capacity * capacity
                if baseline
                else 32 * capacity,
                "input_bytes": 16 * capacity,
                "reference_defect": 0.0,
            }
            for capacity in capacities
        ],
    }
    if not baseline:
        record["memory_gate"] = {
            "capacity": 64,
            "baseline_bytes": 32_768,
            "candidate_bytes": 512,
            "maximum_candidate_fraction": 0.5,
            "decision": decision,
        }
    return record


def test_memory_summary_compares_matching_environment(tmp_path: Path) -> None:
    for filename, record in (
        ("memory_limber.json", _limber_memory_record(baseline=False)),
        ("memory_limber_baseline.json", _limber_memory_record(baseline=True)),
    ):
        (tmp_path / filename).write_text(json.dumps(record), encoding="utf-8")

    summary = build_summary(tmp_path)

    assert len(summary["records"]) == 1
    entry = summary["records"][0]
    assert entry["record"] == "memory_limber.json"
    assert entry["baseline"] == "memory_limber_baseline.json"
    assert entry["disposition"] == "compared"
    assert entry["environment_fingerprint"] == "same-environment"


def test_memory_summary_does_not_compare_mismatched_environments(tmp_path: Path) -> None:
    candidate = _limber_memory_record(
        baseline=False, environment="candidate-environment", decision="retain-baseline"
    )
    baseline = _limber_memory_record(baseline=True, environment="baseline-environment")
    (tmp_path / "memory_limber.json").write_text(json.dumps(candidate), encoding="utf-8")
    (tmp_path / "memory_limber_baseline.json").write_text(
        json.dumps(baseline), encoding="utf-8"
    )

    summary = build_summary(tmp_path)

    assert len(summary["records"]) == 1
    entry = summary["records"][0]
    assert entry["disposition"] == "environment-mismatch"
    assert entry["environment_fingerprint"] == "candidate-environment"


def test_memory_summary_rejects_disagreement_in_matching_environment(
    tmp_path: Path,
) -> None:
    candidate = _limber_memory_record(baseline=False, decision="retain-baseline")
    baseline = _limber_memory_record(baseline=True)
    (tmp_path / "memory_limber.json").write_text(json.dumps(candidate), encoding="utf-8")
    (tmp_path / "memory_limber_baseline.json").write_text(
        json.dumps(baseline), encoding="utf-8"
    )

    with pytest.raises(ValueError):
        build_summary(tmp_path)


def test_memory_summary_refuses_missing_capacity_evidence(tmp_path: Path) -> None:
    candidate = _limber_memory_record(baseline=False)
    candidate["records"] = []
    baseline = _limber_memory_record(baseline=True)
    (tmp_path / "memory_limber.json").write_text(json.dumps(candidate), encoding="utf-8")
    (tmp_path / "memory_limber_baseline.json").write_text(
        json.dumps(baseline), encoding="utf-8"
    )

    with pytest.raises(ValueError):
        build_summary(tmp_path)


def test_memory_summary_refuses_unsupported_record(tmp_path: Path) -> None:
    record = _limber_memory_record(baseline=False)
    (tmp_path / "memory_unsupported.json").write_text(
        json.dumps(record), encoding="utf-8"
    )

    with pytest.raises(ValueError):
        build_summary(tmp_path)
