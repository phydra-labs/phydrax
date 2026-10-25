#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from benchmarks._memory import (
    evaluate_memory_gate,
    fitted_scaling_exponent,
    host_allocation_peak,
    MemoryGatePolicy,
    process_peak_rss_bytes,
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


def test_peak_rss_observation_does_not_initialize_a_backend(tmp_path: Path) -> None:
    source = (
        "import json\n"
        "from phydrax import config\n"
        "from benchmarks._memory import process_peak_rss_bytes\n"
        "assert not config.snapshot().backends_initialized\n"
        "peak=process_peak_rss_bytes()\n"
        "assert not config.snapshot().backends_initialized\n"
        "print(json.dumps({'peak':peak}))\n"
    )
    environment = dict(os.environ)
    environment["PHYDRAX_ACCELERATOR"] = "cpu"
    environment["JAX_PLATFORMS"] = "cpu"
    # Keep every temporary/cache write inside the test's profile-local root.
    environment.update(TMP=str(tmp_path), TEMP=str(tmp_path), TMPDIR=str(tmp_path))
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["peak"] > 0


def test_peak_rss_unavailability_refuses_instead_of_reporting_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax import _execution_sampling as sampling

    def denied() -> ctypes.CDLL:
        raise PermissionError("current-process observation denied")

    monkeypatch.setattr(sampling, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(sampling, "_windows_system", denied)
    with pytest.raises(RuntimeError, match="current-process observation denied"):
        process_peak_rss_bytes()


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


def test_memory_summary_uses_native_mpc_retention_evidence(tmp_path: Path) -> None:
    filename = "memory_control_horizon.json"
    (tmp_path / filename).write_text(
        json.dumps(_committed_memory_record(filename)), encoding="utf-8"
    )
    cases = build_summary(tmp_path)["records"][0]["cases"]
    native = {case["case"]: case for case in cases if case["case"].startswith("S11/")}
    assert set(native) == {"S11/dense", "S11/sparse"}
    assert native["S11/dense"]["disposition"] == "reference-only"
    assert (
        native["S11/sparse"]["metrics"]["retained_bytes"]["disposition"]
        == "bounded-growth"
    )
    assert all(
        "compiler.temporary_bytes" not in case["metrics"] for case in native.values()
    )


def test_memory_summary_groups_action_products_before_scaling_size(
    tmp_path: Path,
) -> None:
    filename = "memory_optim_interior_point.json"
    (tmp_path / filename).write_text(
        json.dumps(_committed_memory_record(filename)), encoding="utf-8"
    )
    cases = build_summary(tmp_path)["records"][0]["cases"]
    actions = [case for case in cases if case["case"].startswith("actions/")]
    assert len(actions) == 2
    assert all(case["capacities"] == [8, 16, 32] for case in actions)
    assert all(case["case"].endswith("/products=16") for case in actions)


@pytest.mark.parametrize(
    ("family", "added_fields"),
    [
        ("fem_assembly", ("compiler_route", "requested_tensor_families")),
        (
            "polynomial_preparation",
            ("compiler_scaling_exponent", "declared_scaling_exponent", "gate"),
        ),
        ("linalg_solves", ("auxiliary",)),
    ],
)
def test_memory_summary_verifies_exact_additive_evidence_fields(
    tmp_path: Path, family: str, added_fields: tuple[str, ...]
) -> None:
    for filename in (f"memory_{family}.json", f"memory_{family}_baseline.json"):
        (tmp_path / filename).write_text(
            json.dumps(_committed_memory_record(filename)), encoding="utf-8"
        )
    entry = build_summary(tmp_path)["records"][0]
    assert entry["evidence_extension_fields"] == sorted(added_fields)
    assert entry["baseline_comparisons"] > 0


@pytest.mark.parametrize("family", ["linalg_spectral", "structured_transfer"])
def test_memory_summary_preserves_missing_baseline_provenance(
    tmp_path: Path, family: str
) -> None:
    for filename in (f"memory_{family}.json", f"memory_{family}_baseline.json"):
        (tmp_path / filename).write_text(
            json.dumps(_committed_memory_record(filename)), encoding="utf-8"
        )
    entry = build_summary(tmp_path)["records"][0]
    assert entry["disposition"] == "baseline-identity-missing"
    assert entry["baseline_comparisons"] == 0
    assert entry["baseline_driver_fingerprint"] is None


def test_memory_summary_checks_reactor_stencil_without_false_power_bound(
    tmp_path: Path,
) -> None:
    filename = "memory_reactor_diffusion.json"
    candidate = _committed_memory_record(filename)
    (tmp_path / filename).write_text(json.dumps(candidate), encoding="utf-8")
    metrics = build_summary(tmp_path)["records"][0]["cases"][0]["metrics"]
    # Affine a*cells-b storage can have a fitted exponent slightly above one.
    assert metrics["retained_bytes"]["exponent"] > 1.0
    assert metrics["retained_bytes"]["disposition"] == "observed-only"
    candidate["records"][0]["sparse_capacity"] += 1
    (tmp_path / filename).write_text(json.dumps(candidate), encoding="utf-8")
    with pytest.raises(ValueError, match="sparse capacity"):
        build_summary(tmp_path)


def _source_memory_summary(
    tmp_path: Path, family: str, record: dict[str, Any]
) -> dict[str, Any]:
    (tmp_path / f"memory_{family}.json").write_text(json.dumps(record), encoding="utf-8")
    return build_summary(tmp_path)["records"][0]


@pytest.mark.parametrize(
    ("family", "case_name", "capacities", "gate_path"),
    [
        ("astrodynamics_pairs", "direct", [1024, 2048, 4096], ("gates", "direct")),
        ("astrodynamics_pairs", "near", [1024, 2048, 4096], ("gates", "near")),
        ("geometry_interface", "main", [8192, 16384], ("gate",)),
        ("graph_queries", "main", [1024, 2048], ("gate",)),
        ("gp_actions", "main", [4096, 8192], ("gate",)),
        ("gp_actions", "variational", [4096, 8192], ("variational_gate",)),
        ("mac_shadowing", "auto-above-budget", [128, 192], ("memory_gate",)),
        ("periodic_kubo", "main", [2048, 4096], ("gate",)),
        ("ringdown", "main", [32, 64, 128, 256, 512], ("gate",)),
        (
            "transport_attention",
            "attention",
            [8192, 16384, 32768, 65536],
            ("gates", "attention", "memory"),
        ),
        (
            "transport_attention",
            "balanced",
            [32768, 65536],
            ("gates", "balanced", "memory"),
        ),
        (
            "transport_attention",
            "unbalanced",
            [32768, 65536],
            ("gates", "unbalanced", "memory"),
        ),
    ],
)
def test_memory_summary_uses_tuned_same_run_above_budget_gates(
    tmp_path: Path,
    family: str,
    case_name: str,
    capacities: list[int],
    gate_path: tuple[str, ...],
) -> None:
    record = _committed_memory_record(f"memory_{family}.json")
    declared = record
    for key in gate_path:
        declared = declared[key]

    entry = _source_memory_summary(tmp_path, family, record)
    case = next(case for case in entry["cases"] if case["case"] == case_name)

    assert case["capacities"] == capacities
    assert case["comparison_scope"] == "same-record-paired"
    assert case["disposition"] == declared["decision"] == "adopt"
    for key in (
        "capacity",
        "baseline_bytes",
        "candidate_bytes",
        "maximum_candidate_fraction",
        "decision",
    ):
        assert case["gate"][key] == declared[key]
    assert case["gate"]["capacity"] == capacities[-1]


@pytest.mark.parametrize(
    "family",
    [
        "astrodynamics_pairs",
        "geometry_interface",
        "graph_queries",
        "mac_shadowing",
        "periodic_kubo",
        "ringdown",
        "transport_attention",
    ],
)
def test_memory_summary_accepts_below_budget_equality_as_evidence_not_adoption(
    tmp_path: Path, family: str
) -> None:
    entry = _source_memory_summary(
        tmp_path, family, _committed_memory_record(f"memory_{family}.json")
    )
    fitting = [case for case in entry["cases"] if case["case"].startswith("fitting/")]

    assert fitting
    assert all(case["scaling_acceptance"]["accepted"] for case in fitting)
    assert all("gate" not in case for case in fitting)
    assert all(case["disposition"] != "adopt" for case in fitting)


@pytest.mark.parametrize(
    ("family", "path"),
    [
        ("astrodynamics_pairs", ("gates", "direct")),
        ("geometry_interface", ("gate",)),
        ("graph_queries", ("gate",)),
        ("gp_actions", ("gate",)),
        ("gp_actions", ("variational_gate",)),
        ("mac_shadowing", ("memory_gate",)),
        ("periodic_kubo", ("gate",)),
        ("ringdown", ("gate",)),
        ("transport_attention", ("gates", "balanced", "memory")),
    ],
)
@pytest.mark.parametrize("field", ["capacity", "baseline_bytes", "candidate_bytes"])
def test_memory_summary_rejects_tuned_gate_byte_and_capacity_mismatches(
    tmp_path: Path, family: str, path: tuple[str, ...], field: str
) -> None:
    record = _committed_memory_record(f"memory_{family}.json")
    declared = record
    for key in path:
        declared = declared[key]
    declared[field] += 1

    with pytest.raises(ValueError):
        _source_memory_summary(tmp_path, family, record)


@pytest.mark.parametrize(
    ("family", "path"),
    [
        (
            "astrodynamics_pairs",
            ("fitting_parallel_comparison_rows", 0, "compiler", "temporary_bytes"),
        ),
        (
            "geometry_interface",
            ("cases", 0, "paired_baseline", "compiler", "temporary_bytes"),
        ),
        ("graph_queries", ("cases", 0, "whole_axis", "compiler", "temporary_bytes")),
        ("mac_shadowing", ("cases", 2, "compiler", "temporary_bytes")),
        ("periodic_kubo", ("cases", 0, "parallel", "compiler", "temporary_bytes")),
        ("ringdown", ("cases", 0, "full_response", "compiler", "temporary_bytes")),
        ("transport_attention", ("cases", 3, "compiler", "temporary_bytes")),
    ],
)
def test_memory_summary_recomputes_below_budget_equality_from_source_bytes(
    tmp_path: Path, family: str, path: tuple[str | int, ...]
) -> None:
    record = _committed_memory_record(f"memory_{family}.json")
    observed: Any = record
    for key in path[:-1]:
        observed = observed[key]
    # Leave any recorded equality flag unchanged: the actual pair must agree.
    observed[path[-1]] += 1

    with pytest.raises(ValueError):
        _source_memory_summary(tmp_path, family, record)


def test_memory_summary_groups_transport_by_requested_default_and_budget(
    tmp_path: Path,
) -> None:
    record = _committed_memory_record("memory_transport_attention.json")
    entry = _source_memory_summary(tmp_path, "transport_attention", record)
    gates = {case["case"]: case for case in entry["cases"] if "gate" in case}

    assert set(gates) == {"attention", "balanced", "unbalanced"}
    assert gates["attention"]["capacities"] == [8192, 16384, 32768, 65536]
    for family in ("balanced", "unbalanced"):
        rows = [
            row
            for row in record["cases"]
            if row["family"] == family
            and row["requested_route"] == "default"
            and row["above_budget"]
        ]
        # The automatic block changes with capacity; it is not a new workload.
        assert [row["block_size"] for row in rows] == [128, 64]
        assert gates[family]["capacities"] == [row["capacity"] for row in rows]
    assert (
        record["configuration"]["attention_workspace_bytes"]
        != (record["configuration"]["sinkhorn_workspace_bytes"])
    )


@pytest.mark.parametrize("action", ["lanczos", "variational"])
def test_memory_summary_checks_gp_nested_default_policy_resolution(
    tmp_path: Path, action: str
) -> None:
    record = _committed_memory_record("memory_gp_actions.json")
    for row in record["records"]:
        n = row["n"]
        expected = min(
            n,
            record["default_workspace_bytes"]
            // (8 * (n + (1 if action == "lanczos" else 4))),
        )
        assert row[action]["default_policy"]["row_batch_size"] == expected
    # Restricted budgets are distinct evidence, not the public default gate.
    record["records"][-1][action]["default_policy"]["row_batch_size"] += 1

    with pytest.raises(ValueError):
        _source_memory_summary(tmp_path, "gp_actions", record)


@pytest.mark.parametrize("control", ["samples", "rank"])
def test_memory_summary_refuses_mac_paired_scientific_control_mismatch(
    tmp_path: Path, control: str
) -> None:
    record = _committed_memory_record("memory_mac_shadowing.json")
    paired = next(
        row for row in record["cases"] if row["gate_case"] and row["route"] == "dense-svd"
    )
    paired[control] += 1

    with pytest.raises(ValueError):
        _source_memory_summary(tmp_path, "mac_shadowing", record)


def test_memory_summary_refuses_transport_paired_query_count_mismatch(
    tmp_path: Path,
) -> None:
    record = _committed_memory_record("memory_transport_attention.json")
    paired = next(
        row
        for row in record["cases"]
        if row["above_budget"]
        and row["family"] == "attention"
        and row["requested_route"] == "dense"
    )
    paired["query_count"] += 1

    with pytest.raises(ValueError):
        _source_memory_summary(tmp_path, "transport_attention", record)


def test_memory_summary_keeps_gp_single_tile_routes_separate_from_compiler_equality(
    tmp_path: Path,
) -> None:
    record = _committed_memory_record("memory_gp_actions.json")
    entry = _source_memory_summary(tmp_path, "gp_actions", record)
    cases = {case["case"]: case for case in entry["cases"]}

    for action, name in (
        ("lanczos", "fitting/main"),
        ("variational", "fitting/variational"),
    ):
        fitting = [
            row
            for row in record["records"]
            if row[action]["default_policy"]["row_batch_size"] == row["n"]
        ]
        acceptance = cases[name]["scaling_acceptance"]
        assert cases[name]["capacities"] == [row["n"] for row in fitting]
        assert acceptance["accepted"]
        assert "gate" not in cases[name]
        assert acceptance["default_temporary_bytes"] == [
            row[action]["default_policy"]["compiler"]["temporary_bytes"]
            for row in fitting
        ]
        assert acceptance["dense_temporary_bytes"] == [
            row[action]["dense"]["compiler"]["temporary_bytes"] for row in fitting
        ]
    assert (
        cases["fitting/main"]["scaling_acceptance"]["comparison_policy"]
        == "single-tile-route"
    )
    for action, name, gate_key in (
        ("lanczos", "main", "gate"),
        ("variational", "variational", "variational_gate"),
    ):
        rows = [
            row
            for row in record["records"]
            if row[action]["default_policy"]["row_batch_size"] < row["n"]
        ]
        expected = compare_gate(
            tuple(row["n"] for row in rows),
            tuple(row[action]["dense"]["compiler"]["temporary_bytes"] for row in rows),
            tuple(
                row[action]["default_policy"]["compiler"]["temporary_bytes"]
                for row in rows
            ),
            record[gate_key]["decision"],
            record[gate_key]["maximum_candidate_fraction"],
        )
        assert cases[name]["gate"] == expected


@pytest.mark.parametrize("equal_bytes", [True, False], ids=["equal", "unequal"])
def test_gp_single_tile_summary_reports_observed_compiler_relationship(
    tmp_path: Path, equal_bytes: bool
) -> None:
    record = _committed_memory_record("memory_gp_actions.json")
    fitting = [
        row
        for row in record["records"]
        if row["lanczos"]["default_policy"]["row_batch_size"] == row["n"]
    ]
    # Synthetic observations exercise both relationships without pinning a
    # historical compiler allocation or changing the above-budget gate.
    for row in fitting:
        default_bytes = row["lanczos"]["default_policy"]["compiler"]["temporary_bytes"]
        row["lanczos"]["dense"]["compiler"]["temporary_bytes"] = (
            default_bytes if equal_bytes else default_bytes + 1
        )
    entry = _source_memory_summary(tmp_path, "gp_actions", record)
    case = next(case for case in entry["cases"] if case["case"] == "fitting/main")
    acceptance = case["scaling_acceptance"]
    assert acceptance["accepted"]
    assert acceptance["comparison_policy"] == "single-tile-route"
    assert "gate" not in case
    assert acceptance["default_temporary_bytes"] == [
        row["lanczos"]["default_policy"]["compiler"]["temporary_bytes"] for row in fitting
    ]
    assert acceptance["dense_temporary_bytes"] == [
        row["lanczos"]["dense"]["compiler"]["temporary_bytes"] for row in fitting
    ]
    assert (
        acceptance["default_temporary_bytes"] == acceptance["dense_temporary_bytes"]
    ) is equal_bytes


@pytest.mark.parametrize("gate_case", [False, True], ids=["below-budget", "above-budget"])
def test_memory_summary_checks_mac_auto_resolved_route(
    tmp_path: Path, gate_case: bool
) -> None:
    record = _committed_memory_record("memory_mac_shadowing.json")
    row = next(
        row
        for row in record["cases"]
        if row["route"] == "auto" and row["gate_case"] == gate_case
    )
    assert row["resolved_route"] == ("block-banded" if gate_case else "dense-svd")
    row["resolved_route"] = "dense-svd" if gate_case else "block-banded"

    with pytest.raises(ValueError):
        _source_memory_summary(tmp_path, "mac_shadowing", record)


@pytest.mark.parametrize(
    "family",
    [
        "astrodynamics_pairs",
        "geometry_interface",
        "graph_queries",
        "gp_actions",
        "mac_shadowing",
        "periodic_kubo",
        "ringdown",
        "transport_attention",
    ],
)
def test_memory_summary_keeps_historical_baselines_informational_for_tuned_gates(
    tmp_path: Path, family: str
) -> None:
    record = _committed_memory_record(f"memory_{family}.json")
    without_baseline = _source_memory_summary(tmp_path, family, record)
    baseline_filename = f"memory_{family}_baseline.json"
    (tmp_path / baseline_filename).write_text(
        json.dumps(_committed_memory_record(baseline_filename)), encoding="utf-8"
    )
    with_baseline = build_summary(tmp_path)["records"][0]
    expected = {
        case["case"]: case["gate"] for case in without_baseline["cases"] if "gate" in case
    }
    compared = {case["case"]: case for case in with_baseline["cases"] if "gate" in case}

    assert expected
    assert {name: case["gate"] for name, case in compared.items()} == expected
    assert all(
        case["comparison_scope"] == "same-record-paired" for case in compared.values()
    )
