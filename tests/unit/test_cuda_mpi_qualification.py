"""Consumer-visible admission and aggregation boundaries of MPI qualification."""

from __future__ import annotations

import argparse
import copy
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
import pytest

from benchmarks.cuda_mpi import (
    _arguments,
    _check_result,
    Operation,
    Precision,
    Route,
)
from tools.qualify_cuda_mpi import (
    _validate_arguments,
    _validate_workload,
    aggregate_rows,
    coverage_errors,
    validate_record,
)


def _suite(rank_count: int) -> ET.Element:
    suite = ET.Element("testsuite")
    for placement in ("distinct-gpus", "shared-gpu"):
        for route in Route:
            for precision in Precision:
                ET.SubElement(
                    suite,
                    "testcase",
                    name=f"test_cuda_mpi_collectives[ranks-{rank_count}-{placement}-{route.value}-{precision.value}]",
                )
    return suite


@pytest.mark.parametrize("rank_count", [2, 4])
@pytest.mark.parametrize("failure", ["skip", "missing", "duplicate", "failure", "error"])
def test_qualification_requires_complete_unskipped_matrix(
    rank_count: int, failure: str
) -> None:
    suite = _suite(rank_count)
    assert coverage_errors(ET.tostring(suite), rank_count) == []
    if failure in ("skip", "failure", "error"):
        state = "skipped" if failure == "skip" else failure
        ET.SubElement(suite[0], state, message="CUDA support unavailable")
    elif failure == "missing":
        suite.remove(suite[0])
    else:
        suite.append(copy.deepcopy(suite[0]))
    assert coverage_errors(ET.tostring(suite), rank_count)


def test_four_rank_qualification_refuses_two_rank_suite() -> None:
    assert coverage_errors(ET.tostring(_suite(2)), 4)


def _measurement(rank_count: int = 2) -> dict[str, Any]:
    return {
        "configuration": {
            "ranks": rank_count,
            "route": "cuda-direct",
            "sizes": [8],
            "precision": [value.value for value in Precision],
            "operation": [value.value for value in Operation],
            "repeats": 3,
            "maximum_buffer_bytes": 8,
            "profile_capture": False,
        },
        "passed": True,
        "scope": {
            "hosts": 1,
            "mpi_ranks": rank_count,
            "shared_memory_ranks": rank_count,
            "distinct_gpus": rank_count,
        },
        "ownership": [
            {"rank": rank, "host": "node", "gpu_uuid": f"GPU-{rank}"}
            for rank in range(rank_count)
        ],
        "ranks": [
            {
                "rank": rank,
                "rows": [
                    {
                        "operation": operation.value,
                        "precision": precision.value,
                        "message_bytes_per_rank": 8,
                        "warm": {
                            "samples_seconds": [1.0, 9.0, 2.0]
                            if rank % 2 == 0
                            else [9.0, 1.0, 2.0]
                        },
                        "passed": True,
                    }
                    for operation in Operation
                    for precision in Precision
                ],
            }
            for rank in range(rank_count)
        ],
    }


@pytest.mark.parametrize("rank_count", [2, 4])
def test_aggregate_uses_corresponding_rank_samples_not_rank_medians(
    rank_count: int,
) -> None:
    record = _measurement(rank_count)
    record["ranks"].reverse()
    record["ownership"].reverse()
    result = aggregate_rows(record, rank_count)
    assert len(result) == len(Operation) * len(Precision)
    assert result[0]["rank_maximum_warm"]["samples_seconds"] == [9.0, 9.0, 2.0]
    assert result[0]["rank_maximum_warm"]["median_seconds"] == 9.0
    assert result[0]["effective_payload_bytes_per_second"] == pytest.approx(8.0 / 9.0)


def test_four_rank_aggregation_includes_slowest_nonzero_pair_rank() -> None:
    record = _measurement(4)
    for row in record["ranks"][3]["rows"]:
        row["warm"]["samples_seconds"] = [10.0, 11.0, 12.0]
    for row in aggregate_rows(record, 4):
        assert row["rank_maximum_warm"]["samples_seconds"] == [10.0, 11.0, 12.0]


@pytest.mark.parametrize("rank_count", [2, 4])
@pytest.mark.parametrize(
    "failure",
    [
        "shared-device",
        "different-shared-domains",
        "different-hosts",
        "different-points",
        "missing-point-all-ranks",
        "duplicate-point",
        "failed-point",
        "missing-sample",
        "nonfinite-sample",
        "negative-sample",
        "profiled",
        "missing-rank",
        "duplicate-rank",
        "missing-owner",
        "duplicate-owner",
        "wrong-configuration",
        "wrong-scope",
        "failed",
    ],
)
def test_aggregate_refuses_incomparable_or_unqualified_records(
    rank_count: int, failure: str
) -> None:
    record = _measurement(rank_count)
    if failure == "shared-device":
        record["ownership"][1]["gpu_uuid"] = "GPU-0"
    elif failure == "different-shared-domains":
        record["scope"]["shared_memory_ranks"] = 1
    elif failure == "different-hosts":
        record["ownership"][1]["host"] = "other-node"
    elif failure == "different-points":
        record["ranks"][1]["rows"][0]["message_bytes_per_rank"] = 16
    elif failure == "missing-point-all-ranks":
        for rank in record["ranks"]:
            rank["rows"].pop()
    elif failure == "duplicate-point":
        record["ranks"][1]["rows"].append(copy.deepcopy(record["ranks"][1]["rows"][0]))
    elif failure == "failed-point":
        record["ranks"][1]["rows"][0]["passed"] = False
    elif failure == "missing-sample":
        record["ranks"][1]["rows"][0]["warm"]["samples_seconds"].pop()
    elif failure in ("nonfinite-sample", "negative-sample"):
        record["ranks"][1]["rows"][0]["warm"]["samples_seconds"][0] = (
            float("nan") if failure == "nonfinite-sample" else -1.0
        )
    elif failure == "profiled":
        record["configuration"]["profile_capture"] = True
    elif failure == "missing-rank":
        record["ranks"].pop()
    elif failure == "duplicate-rank":
        record["ranks"][1]["rank"] = 0
    elif failure == "missing-owner":
        record["ownership"].pop()
    elif failure == "duplicate-owner":
        record["ownership"][1]["rank"] = 0
    elif failure == "wrong-configuration":
        record["configuration"]["ranks"] = rank_count + 1
    elif failure == "wrong-scope":
        record["scope"]["mpi_ranks"] = rank_count + 1
    else:
        record["passed"] = False
    with pytest.raises(ValueError):
        aggregate_rows(record, rank_count)


def test_four_rank_aggregation_refuses_complete_two_rank_measurement() -> None:
    with pytest.raises(ValueError):
        aggregate_rows(_measurement(2), 4)


def test_profile_validation_requires_all_requested_ranks_and_separate_timings() -> None:
    record = _measurement(4)
    record["configuration"]["profile_capture"] = True
    validate_record(record, 4, profile_capture=True)
    with pytest.raises(ValueError):
        aggregate_rows(record, 4)
    record["ranks"].pop()
    with pytest.raises(ValueError):
        validate_record(record, 4, profile_capture=True)


def test_qualification_requires_full_requested_measurement_workload() -> None:
    record = _measurement(4)
    args = argparse.Namespace(ranks=4, sizes=[8], repeats=3, maximum_buffer_bytes=8)
    _validate_workload(record, args, Route.CUDA_DIRECT, profile_capture=False)
    record["configuration"]["operation"] = ["sum"]
    for rank in record["ranks"]:
        rank["rows"] = [row for row in rank["rows"] if row["operation"] == "sum"]
    with pytest.raises(ValueError):
        _validate_workload(record, args, Route.CUDA_DIRECT, profile_capture=False)


@pytest.mark.parametrize("devices", ["0,1", "0,1,2,3", "GPU-a,GPU-b"])
def test_visible_allocation_sets_explicit_rank_count(devices: str) -> None:
    args = argparse.Namespace(
        timeout=30, repeats=3, maximum_buffer_bytes=8, sizes=[8], visible_devices=devices
    )
    _validate_arguments(argparse.ArgumentParser(), args)
    assert args.ranks == len(devices.split(","))


@pytest.mark.parametrize("devices", ["", "0", "0,0", "0,1,", "0,,1", "0, 0"])
def test_visible_allocation_rejects_missing_or_repeated_devices(devices: str) -> None:
    args = argparse.Namespace(
        timeout=30, repeats=3, maximum_buffer_bytes=8, sizes=[8], visible_devices=devices
    )
    with pytest.raises(SystemExit):
        _validate_arguments(argparse.ArgumentParser(), args)


@pytest.mark.parametrize("ranks", [[], ["--ranks", "1"]])
def test_benchmark_refuses_unspecified_or_single_rank(ranks: list[str]) -> None:
    with pytest.raises(SystemExit):
        _arguments(["--route", "cuda-direct", "--output", "unused.json", *ranks])


@pytest.mark.parametrize("rank_count", [2, 4])
@pytest.mark.parametrize("precision", [np.float32, np.float64])
@pytest.mark.parametrize("operation", list(Operation))
@pytest.mark.parametrize("delta", [0, 8])
def test_numerical_reference_includes_requested_ranks(
    rank_count: int, precision: Any, operation: Operation, delta: int
) -> None:
    device = object()

    class Result(np.ndarray):
        def devices(self) -> set[object]:
            return {device}

    base = np.asarray([0, 1, 256], dtype=precision)
    for rank in range(rank_count):
        if operation is Operation.SUM:
            values = (
                rank_count * base
                + rank_count * (rank_count - 1) // 2
                + rank_count * delta
            )
        else:
            values = base + (rank - 1) % rank_count + delta
        result = np.asarray(values, dtype=precision).view(Result)
        _check_result(result, base, operation, rank, rank_count, delta, device)
        with pytest.raises(AssertionError):
            _check_result(
                result + np.asarray(1, dtype=precision),
                base,
                operation,
                rank,
                rank_count,
                delta,
                device,
            )
