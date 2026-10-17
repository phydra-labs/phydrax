"""Consumer-visible admission and aggregation boundaries of MPI qualification."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from typing import Any

import pytest

from tools.qualify_cuda_mpi import aggregate_rows, coverage_errors


def _suite() -> ET.Element:
    suite = ET.Element("testsuite")
    for placement in ("distinct-gpus", "shared-gpu"):
        for route in ("cuda-direct", "host-staged"):
            for precision in ("float32", "float64"):
                ET.SubElement(
                    suite,
                    "testcase",
                    name=f"test_cuda_mpi_collectives[{placement}-{route}-{precision}]",
                )
    return suite


@pytest.mark.parametrize("failure", ["skip", "missing", "duplicate"])
def test_qualification_requires_complete_unskipped_matrix(failure: str) -> None:
    suite = _suite()
    assert coverage_errors(ET.tostring(suite)) == []
    if failure == "skip":
        ET.SubElement(suite[0], "skipped", message="CUDA support unavailable")
    elif failure == "missing":
        suite.remove(suite[0])
    else:
        suite.append(copy.deepcopy(suite[0]))
    assert coverage_errors(ET.tostring(suite))


def _measurement() -> dict[str, Any]:
    return {
        "configuration": {"repeats": 3, "profile_capture": False},
        "passed": True,
        "scope": {"shared_memory_ranks": 2},
        "ownership": [
            {"host": "node", "gpu_uuid": "GPU-a"},
            {"host": "node", "gpu_uuid": "GPU-b"},
        ],
        "ranks": [
            {
                "rank": rank,
                "rows": [
                    {
                        "operation": "sum",
                        "precision": "float64",
                        "message_bytes_per_rank": 8,
                        "warm": {"samples_seconds": samples},
                        "passed": True,
                    }
                ],
            }
            for rank, samples in enumerate(([1.0, 9.0, 2.0], [9.0, 1.0, 2.0]))
        ],
    }


def test_aggregate_uses_corresponding_rank_samples_not_rank_medians() -> None:
    result = aggregate_rows(_measurement())
    assert result[0]["rank_maximum_warm"]["samples_seconds"] == [9.0, 9.0, 2.0]
    assert result[0]["rank_maximum_warm"]["median_seconds"] == 9.0
    assert result[0]["effective_payload_bytes_per_second"] == pytest.approx(8.0 / 9.0)


@pytest.mark.parametrize(
    "failure",
    [
        "shared-device",
        "different-shared-domains",
        "different-points",
        "missing-sample",
        "profiled",
    ],
)
def test_aggregate_refuses_incomparable_or_unqualified_records(failure: str) -> None:
    record = _measurement()
    if failure == "shared-device":
        record["ownership"][1]["gpu_uuid"] = "GPU-a"
    elif failure == "different-shared-domains":
        record["scope"]["shared_memory_ranks"] = 1
    elif failure == "different-points":
        record["ranks"][1]["rows"][0]["message_bytes_per_rank"] = 16
    elif failure == "missing-sample":
        record["ranks"][1]["rows"][0]["warm"]["samples_seconds"].pop()
    else:
        record["configuration"]["profile_capture"] = True
    with pytest.raises(ValueError):
        aggregate_rows(record)
