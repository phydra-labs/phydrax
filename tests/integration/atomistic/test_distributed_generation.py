#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Real multi-process distributed generation quotas and restart.

Every case launches coupled local JAX processes through
`launch_local_processes` (finite whole-job deadline, fail-coupled ranks) via
`tools.atomistic_generation_distributed_smoke`. Each process owns forced host
CPU devices with gloo collectives and is one participant of ONE placed
generation stream. These cases prove the global ordinal window, accepted
quota, attempt budget, publication authority, backpressure and durable restart
across real OS processes on one host; they are functional oracles, not
accelerator or multi-host evidence.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tools.atomistic_distributed_smoke import commit_present
from tools.atomistic_generation_distributed_smoke import (
    CHECKPOINT,
    generation_repository,
    run_scenario,
)


pytestmark = [
    pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups"),
    pytest.mark.skipif(
        not sys.platform.startswith("linux"),
        reason="requires Linux boot identities for observed host mapping",
    ),
]

# Whole-job deadline of one coupled launch (JAX bootstrap plus CPU compilation
# of the sampler, exchange and lifecycle programs); a hang can never exceed it.
DEADLINE_SECONDS = 600.0


@pytest.mark.parametrize(
    ("processes", "devices_per_process"),
    [(2, 1), (2, 2), (3, 1)],
    ids=["2x1-devices", "2x2-devices", "3x1-devices"],
)
def test_global_quota_and_budget_span_participants_with_unchanged_identities(
    tmp_path: Path, processes: int, devices_per_process: int
) -> None:
    record = run_scenario(
        "quota",
        tmp_path,
        processes=processes,
        devices_per_process=devices_per_process,
        timeout_seconds=DEADLINE_SECONDS,
    )
    assert record["passed"], record["errors"]
    ranks = record["ranks"]
    assert len({rank["quota"]["published_digest"] for rank in ranks}) == 1
    assert all(rank["quota"]["ran"] for rank in ranks)
    assert ranks[0]["quota"]["refused_publications"] >= 3
    assert ranks[0]["quota"]["out_of_order_completion"]
    assert ranks[0]["budget"]["stop_reason"] == "BUDGET_EXHAUSTED"


def test_killed_participant_restarts_from_the_global_commit_exactly(
    tmp_path: Path,
) -> None:
    killed = run_scenario("checkpoint-kill", tmp_path, timeout_seconds=DEADLINE_SECONDS)
    assert killed["passed"], killed["errors"]
    assert killed["launch"]["error"] is not None
    assert commit_present(generation_repository(tmp_path), CHECKPOINT)

    restarted = run_scenario("restart", tmp_path, timeout_seconds=DEADLINE_SECONDS)
    assert restarted["passed"], restarted["errors"]
    authority = restarted["ranks"][0]["restart"]
    assert authority["active"] > 0
    assert authority["durable_records"] > authority["watermark"]
    assert authority["reconciled_watermark"] > authority["watermark"]
    assert authority["stop_reason"] == "TARGET_REACHED"
