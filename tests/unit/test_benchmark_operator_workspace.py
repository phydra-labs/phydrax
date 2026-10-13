#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import pytest

from benchmarks._memory import evaluate_memory_gate, MemoryGatePolicy
from benchmarks.memory_operator_workspace import _matching_baseline


def test_operator_gate_matches_capacity_and_width_not_position() -> None:
    baseline = [
        {"capacity": 256, "width": 16384, "temporary_bytes": 1600},
        {"capacity": 64, "width": 1024, "temporary_bytes": 100},
        {"capacity": 128, "width": 4096, "temporary_bytes": 400},
    ]
    candidate = [
        {"capacity": 64, "width": 1024, "temporary_bytes": 30},
        {"capacity": 128, "width": 4096, "temporary_bytes": 80},
    ]
    matched = _matching_baseline(candidate, baseline)
    gate = evaluate_memory_gate(
        [row["capacity"] for row in candidate],
        [row["temporary_bytes"] for row in matched],
        [row["temporary_bytes"] for row in candidate],
        MemoryGatePolicy(),
    )
    assert gate.capacity == 128
    assert gate.baseline_bytes == 400
    assert gate.candidate_bytes == 80
    assert gate.decision == "adopt"
    assert len(baseline) == 3


@pytest.mark.parametrize(
    ("capacity", "width"),
    [pytest.param(32, 1024, id="target-count"), pytest.param(64, 256, id="source-count")],
)
def test_operator_gate_refuses_missing_historical_case(capacity: int, width: int) -> None:
    with pytest.raises(ValueError, match="no operator case"):
        _matching_baseline(
            [{"capacity": capacity, "width": width}],
            [{"capacity": 64, "width": 1024}],
        )


def test_operator_gate_refuses_ambiguous_historical_case() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        _matching_baseline(
            [{"capacity": 64, "width": 1024}],
            [{"capacity": 64, "width": 1024}, {"capacity": 64, "width": 1024}],
        )
