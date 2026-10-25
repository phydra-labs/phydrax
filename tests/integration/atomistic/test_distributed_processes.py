"""Real multi-process distributed atomistic domain execution.

Every case launches two coupled local JAX processes through
`launch_local_processes` (finite whole-job deadline, fail-coupled ranks) via
`tools.atomistic_distributed_smoke`. Each process owns forced host CPU
devices with gloo collectives: these cases prove the collective program,
process-local ingress, stage transport and durable restart across real OS
processes on ONE host. They are functional oracles only; accelerator,
multi-host and pretrained-model fidelity are separate qualification gates
(`tools/atomistic_distributed_qualification.py`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tools.atomistic_distributed_smoke import (
    commit_present,
    compare_digests,
    DOMAIN_CHECKPOINT,
    DOMAIN_CHECKPOINT_LIMITS,
    domain_repository,
    KILL_CHECKPOINT,
    run_scenario,
    TRANSPORT_KILL_CHECKPOINT,
    transport_repository,
)


pytestmark = [
    pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups"),
    pytest.mark.skipif(
        not sys.platform.startswith("linux"),
        reason="requires Linux boot identities for observed host mapping",
    ),
]

# Whole-job deadline of one coupled launch (JAX bootstrap plus CPU compilation
# of a two-layer MACE domain); a hang can never exceed it.
DEADLINE_SECONDS = 600.0


def _require_passed(record: dict) -> None:
    assert record["passed"], record["errors"]


@pytest.mark.parametrize("strategy", ["halo", "graph-partition"])
@pytest.mark.parametrize(
    "devices_per_process", [1, 2], ids=["2x1-devices", "2x2-devices"]
)
def test_process_local_ingress_matches_single_device_mace(
    tmp_path: Path, strategy: str, devices_per_process: int
) -> None:
    record = run_scenario(
        "parity",
        tmp_path,
        devices_per_process=devices_per_process,
        strategy=strategy,  # ty: ignore[invalid-argument-type]
        timeout_seconds=DEADLINE_SECONDS,
    )
    _require_passed(record)
    owners = 2 * devices_per_process
    for index, rank in enumerate(record["ranks"]):
        parity = rank["parity"]
        assert rank["runtime"]["process_index"] == index
        assert rank["runtime"]["process_count"] == 2
        assert rank["runtime"]["observed_host_count"] == 1
        assert parity["owner_count"] == owners
        assert (
            parity["owner_processes"]
            == [0] * devices_per_process + [1] * devices_per_process
        )
        assert parity["strategy"] == strategy
        assert sum(parity["owned_rows_by_process"]) == 6
    # Replicated energy is identical on every process.
    assert len({rank["parity"]["energy"] for rank in record["ranks"]}) == 1


def test_stage_transport_subgroup_probe_commits_and_restores(tmp_path: Path) -> None:
    record = run_scenario(
        "transport", tmp_path, devices_per_process=1, timeout_seconds=DEADLINE_SECONDS
    )
    _require_passed(record)
    producer, consumer = (rank["probe"] for rank in record["ranks"])
    assert producer["rank"] == 0 and consumer["rank"] == 1
    assert len(producer["physics"]) == 3 and len(consumer["received"]) == 3
    assert any(round_["parked"] > 0 for round_ in producer["rounds"])
    assert producer["restored_bitwise"] and consumer["restored_bitwise"]
    assert commit_present(
        transport_repository(tmp_path), "stage-transport-probe-complete"
    )


def test_killed_transport_rank_fails_without_global_commit(tmp_path: Path) -> None:
    record = run_scenario(
        "transport-kill",
        tmp_path,
        devices_per_process=1,
        timeout_seconds=DEADLINE_SECONDS,
    )
    _require_passed(record)
    launch = record["launch"]
    assert launch["error"] is not None and not launch["timed_out"]
    assert launch["elapsed_seconds"] < DEADLINE_SECONDS
    assert not commit_present(transport_repository(tmp_path), TRANSPORT_KILL_CHECKPOINT)


def test_killed_domain_rank_fails_within_deadline_without_partial_commit(
    tmp_path: Path,
) -> None:
    record = run_scenario("kill", tmp_path, timeout_seconds=DEADLINE_SECONDS)
    _require_passed(record)
    launch = record["launch"]
    assert launch["error"] is not None and not launch["timed_out"]
    assert launch["elapsed_seconds"] < DEADLINE_SECONDS
    assert launch["returncodes"][1] == -9
    assert not commit_present(
        domain_repository(tmp_path), KILL_CHECKPOINT, limits=DOMAIN_CHECKPOINT_LIMITS
    )


def test_same_topology_restart_restores_every_leaf_bitwise(tmp_path: Path) -> None:
    published = run_scenario("publish", tmp_path, timeout_seconds=DEADLINE_SECONDS)
    _require_passed(published)
    assert all(rank["migrated_atoms"] >= 1 for rank in published["ranks"])
    assert commit_present(
        domain_repository(tmp_path), DOMAIN_CHECKPOINT, limits=DOMAIN_CHECKPOINT_LIMITS
    )
    restored = run_scenario("restore", tmp_path, timeout_seconds=DEADLINE_SECONDS)
    _require_passed(restored)
    assert compare_digests(published, restored) == []
    assert all(rank["restart_class"] == "bitwise" for rank in restored["ranks"])
    for before, after in zip(published["ranks"], restored["ranks"], strict=True):
        assert after["parity"]["energy"] == before["parity"]["energy"]


def test_changed_topology_continuation_migrates_with_lineage_evidence(
    tmp_path: Path,
) -> None:
    published = run_scenario(
        "publish", tmp_path, devices_per_process=2, timeout_seconds=DEADLINE_SECONDS
    )
    _require_passed(published)
    migrated = run_scenario(
        "migrate", tmp_path, devices_per_process=1, timeout_seconds=DEADLINE_SECONDS
    )
    _require_passed(migrated)
    restored_rows = 0
    for before, after in zip(published["ranks"], migrated["ranks"], strict=True):
        evidence = after["migration"]
        assert after["restart_class"] == "tolerance"
        assert "same_topology_restore_refused" in after
        assert evidence["source_owner_count"] == 4
        assert evidence["target_owner_count"] == 2
        assert evidence["source_process_count"] == evidence["target_process_count"] == 2
        assert evidence["source_execution_id"] == before["execution_id"]
        assert evidence["source_execution_id"] != evidence["target_execution_id"]
        # Scientific state and RNG lineage are preserved, not the layout.
        assert after["digest"]["rng_key"] == before["digest"]["rng_key"]
        assert after["digest"]["step_index"] == before["digest"]["step_index"]
        restored_rows += evidence["restored_rows"]
    assert restored_rows == 6
    assert len({rank["migration"]["rng_lineage_id"] for rank in migrated["ranks"]}) == 1
