"""Portable migration records replay only after topology restart admission."""

from pathlib import Path
from typing import Any

import pytest

from tests._support.forced_devices import run_on_forced_devices


_MODULE = "tests._support.distributed_meshing.collective_migration_restore"


@pytest.fixture(scope="module")
def migration_checkpoint(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, Any]]:
    path = tmp_path_factory.mktemp("migration-record") / "migration.npz"
    record = run_on_forced_devices(
        _MODULE, "save_migration", (str(path),), devices=4, env={}
    )
    return path, record


def test_migration_payload_contains_only_portable_fields(
    migration_checkpoint: tuple[Path, dict[str, Any]],
) -> None:
    _, record = migration_checkpoint
    assert record["device_free"]
    assert record["migration_id"]


def test_standalone_migration_record_validates_content_and_refuses_tampering(
    migration_checkpoint: tuple[Path, dict[str, Any]],
) -> None:
    _, record = migration_checkpoint
    assert record["standalone_validated"]
    assert record["tampered_record_refused"]


def test_migration_replays_in_fresh_same_topology_process(
    migration_checkpoint: tuple[Path, dict[str, Any]],
) -> None:
    path, saved = migration_checkpoint
    restored = run_on_forced_devices(
        _MODULE, "replay_migration", (str(path), saved["content_id"]), devices=4, env={}
    )
    assert restored["replayed"]
    assert restored["device_free"]
    assert restored["consumed_work"] == 1
    assert restored["restore_native_work"] == 0
    assert restored["restore_native_queries"] == 0
    assert restored["restore_external_work"] == 1
    for name in ("result_id", "evidence_id", "migration_id"):
        assert restored[name] == saved[name]


def test_migration_refuses_unadmitted_changed_device_count(
    migration_checkpoint: tuple[Path, dict[str, Any]],
) -> None:
    path, saved = migration_checkpoint
    refused = run_on_forced_devices(
        _MODULE,
        "refuse_topology_change",
        (str(path), saved["content_id"]),
        devices=1,
        env={},
    )
    assert refused["refused"]
    assert "MigrationPlacement.topology" in refused["error"]
