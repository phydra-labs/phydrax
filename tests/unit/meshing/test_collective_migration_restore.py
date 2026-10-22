"""Canonical migration history replays logical owners on admitted physical meshes."""

import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._fingerprint import logical_array_value_collection_digest
from phydrax.meshing._device_adaptation import _migration_placement_arrays
from tests._support.forced_devices import gloo_rank_environment, run_on_forced_devices


_MODULE = "tests._support.distributed_meshing.collective_migration_restore"


@pytest.mark.parametrize("permuted", ("false", "true"))
@pytest.mark.parametrize("empty", ("false", "true"))
def test_mixed_local_global_migration_comparison_refuses_and_retries(
    permuted: str, empty: str,
) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{listener.getsockname()[1]}"

    def launch(rank: int) -> dict[str, Any]:
        return run_on_forced_devices(
            _MODULE, "mixed_migration_comparison", (permuted, empty),
            devices=1, env=gloo_rank_environment(coordinator, rank),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        reports = tuple(executor.map(launch, range(2)))
    assert [report["rank"] for report in reports] == [0, 1]
    assert all(report["retry"] and report["unchanged"] for report in reports)
    assert all(len(report["refused"]) == 4 for report in reports)


@pytest.mark.parametrize("empty", (False, True))
@pytest.mark.parametrize("permuted", (False, True))
def test_migration_placement_names_own_order_and_current_solver_bank(
    empty: bool, permuted: bool,
) -> None:
    old = jnp.asarray((0, 1), dtype=jnp.int32)
    current = jnp.asarray((1, 0), dtype=jnp.int32)
    counters = jnp.asarray(() if empty else (17, 19), dtype=jnp.uint64)
    physics = jnp.asarray(() if empty else (1.25, 2.5), dtype=jnp.float64)
    retained = (
        ("placement/initial_solver_cell_owners", old),
        ("placement/history/counters", counters),
        ("placement/history/physics", physics),
    )
    proposal = (
        ("rebalance/weights/cells", jnp.asarray((2, 3), dtype=jnp.int64)),
        ("rebalance/weights/edges", jnp.asarray((4, 5), dtype=jnp.int64)),
    )
    expected = dict((*retained, *proposal))
    expected["placement/initial_solver_cell_owners"] = current
    result = _migration_placement_arrays(
        tuple(reversed(retained)) if permuted else retained,
        tuple(reversed(proposal)) if permuted else proposal,
        current,
    )
    assert tuple(name for name, _ in result) == tuple(sorted(expected))
    assert logical_array_value_collection_digest(dict(result)) == logical_array_value_collection_digest(expected)
    assert dict(result)["placement/initial_solver_cell_owners"] is current
    assert retained[0][1] is old
    assert dict(result)["placement/history/counters"] is counters
    assert dict(result)["placement/history/physics"] is physics
    np.testing.assert_array_equal(old, (0, 1))
    assert _migration_placement_arrays((), (), current)[0][1] is current


@pytest.mark.parametrize("fault", ("retained", "proposal", "collision", "foreign-replacement", "noncanonical"))
def test_migration_placement_refuses_duplicate_or_foreign_names(fault: str) -> None:
    value = jnp.asarray((1, 2), dtype=jnp.int32)
    other = jnp.asarray((3, 4), dtype=jnp.int32)
    retained = (("placement/history", value),)
    proposal = (("rebalance/weights", other),)
    if fault == "retained":
        retained = (*retained, ("placement/history", other))
    elif fault == "proposal":
        proposal = (*proposal, ("rebalance/weights", value))
    elif fault == "collision":
        retained = (*retained, ("rebalance/weights", value))
    elif fault == "foreign-replacement":
        proposal = (("placement/initial_solver_cell_owners", other),)
    else:
        retained = ((" placement/history", value),)
    with pytest.raises(ValueError, match="canonical|unique|disjoint"):
        _migration_placement_arrays(retained, proposal, value)


@pytest.mark.parametrize("permuted", ("false", "true"))
@pytest.mark.parametrize("empty", ("false", "true"))
def test_two_process_logical_graph_weight_placement(
    permuted: str, empty: str
) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{listener.getsockname()[1]}"

    def launch(rank: int) -> dict[str, Any]:
        return run_on_forced_devices(
            _MODULE,
            "replay_graph_weight_placement",
            (permuted, empty),
            devices=1,
            env=gloo_rank_environment(coordinator, rank),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        reports = tuple(executor.map(launch, range(2)))
    for report in reports:
        assert report["graph_bytes_unchanged"]
        assert report["retry"]


@pytest.mark.parametrize("devices", (4, 2), ids=("same4", "logical4_on_physical2"))
@pytest.mark.parametrize("permuted", (False, True))
@pytest.mark.parametrize("empty", (False, True))
def test_logical_graph_weights_use_current_workset_placement(
    devices: int, permuted: bool, empty: bool
) -> None:
    report = run_on_forced_devices(
        _MODULE,
        "replay_graph_weight_placement",
        (str(permuted).lower(), str(empty).lower()),
        devices=devices,
        env={},
    )
    assert report["graph_bytes_unchanged"]
    assert report["retry"]


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
    assert record["compact_source_width"] == 1
    assert record["source_shape_changes_migration_id"]
    assert record["shape_change_preserves_entity_physics_rng_bytes"]


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
        (f"{path}.historical", saved["historical_content_id"]),
        devices=1,
        env={},
    )
    assert refused["refused"]
    assert "MigrationPlacement.topology" in refused["error"]


@pytest.fixture(scope="module", params=(4, 2), ids=("same4", "logical4_on_physical2"))
def historical_migration_report(
    migration_checkpoint: tuple[Path, dict[str, Any]], request: pytest.FixtureRequest
) -> tuple[dict[str, Any], dict[str, Any], int]:
    path, saved = migration_checkpoint
    devices = request.param
    report = run_on_forced_devices(
        _MODULE,
        "replay_historical_migration",
        (
            f"{path}.historical",
            saved["historical_content_id"],
            saved["historical_packets_content_id"],
        ),
        devices=devices,
        env={},
    )
    return report, saved, devices


def test_canonical_historical_migration_uses_real_physical_mesh_and_saved_logical_owners(
    historical_migration_report: tuple[dict[str, Any], dict[str, Any], int],
) -> None:
    report, saved, devices = historical_migration_report
    assert report["physical_devices"] == devices
    assert report["logical_owners"] == 4
    assert report["migration_id"] == saved["historical_migration_id"]
    assert report["placement_id"] == saved["historical_placement_id"]
    assert report["sent_cells"] > 0


def test_canonical_historical_migration_preserves_every_packet_history_rng_and_clock(
    historical_migration_report: tuple[dict[str, Any], dict[str, Any], int],
) -> None:
    report, _, _ = historical_migration_report
    assert report["bytewise_records"]
    assert report["bytewise_fields_history_rng_clock"]
    assert report["bytewise_transport"]
    assert report["compact_source_width"] == 1
    assert report["complete_source_multisets"]


def test_historical_logical_owners_cannot_publish_as_live_physical_owners(
    historical_migration_report: tuple[dict[str, Any], dict[str, Any], int],
) -> None:
    report, _, devices = historical_migration_report
    refusals = report["live_publication_refusals"]
    if devices == 2:
        assert set(refusals) == {"execute", "commit"}
        assert all("logical" in error.lower() for error in refusals.values())
    else:
        assert refusals == {}


@pytest.fixture(scope="module")
def historical_migration_refusals(
    migration_checkpoint: tuple[Path, dict[str, Any]],
) -> dict[str, Any]:
    path, saved = migration_checkpoint
    return run_on_forced_devices(
        _MODULE,
        "refuse_historical_mutations",
        (f"{path}.historical", saved["historical_content_id"]),
        devices=2,
        env={},
    )


@pytest.mark.parametrize(
    "corruption",
    (
        "fields",
        "owner",
        "cell_history",
        "vertex_history",
        "rng",
        "raw_clock",
        "transport",
        "transport_control",
        "source_missing",
        "source_negative",
        "source_duplicate",
    ),
)
def test_canonical_historical_migration_refuses_changed_packets_and_receipts(
    historical_migration_refusals: dict[str, Any], corruption: str
) -> None:
    assert historical_migration_refusals["physical_devices"] == 2
    error = historical_migration_refusals["refusals"][corruption]
    assert (
        "predecessor" in error.lower()
        if corruption.startswith("source_")
        else "CollectiveMigrationRecord." in error
    )
