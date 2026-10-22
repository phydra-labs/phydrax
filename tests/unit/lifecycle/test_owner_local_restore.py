# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Cold native owner-map regressions plus packet and collective admission checks."""

import json
from pathlib import Path

import jax
import numpy as np
import pytest
from jax.sharding import SingleDeviceSharding

from phydrax.lifecycle import _distributed_checkpoint as checkpoint
from phydrax.lifecycle._restart_topology import TopologyRestartPolicy
from tests._support.distributed_meshing.owner_local_restore import (
    run_owner_local_restore_workers,
    run_packed_owner_workers,
)
from tests._support.distributed_meshing.restart_builders import (
    build_repository,
    checkpoint_state,
    restart_relation,
    scientific_arrays,
)
from tests._support.forced_devices import run_on_forced_devices


def test_packet_restore_assembles_only_addressable_destinations(tmp_path: Path) -> None:
    record = run_on_forced_devices(
        "tests._support.distributed_meshing.owner_local_restore",
        "packet_assembly",
        (str(tmp_path),),
        devices=2,
        env={},
    )
    assert record == {
        "devices": 2,
        "row_values_exact": True,
        "row_shards": [[0, [2, 6]], [2, [2, 6]]],
        "row_buffers": [[2, 6], [2, 6]],
        "replicated_values_exact": True,
        "replicated_shards": [[4, 6], [4, 6]],
        "replicated_buffers": [[4, 6]],
    }


def test_one_rank_restore_refusal_is_collective(tmp_path: Path) -> None:
    records = run_owner_local_restore_workers("collective_admission", str(tmp_path))
    assert [record["rank"] for record in records] == [0, 1]
    assert all(record["process_count"] == 2 for record in records)
    assert all(record["refused"] for record in records)
    assert records[0]["remote_refusal"] and not records[0]["local_refusal"]
    assert records[1]["local_refusal"] and not records[1]["remote_refusal"]
    assert all(record["admitted_after_refusal"] for record in records)


def test_decoded_source_owners_select_archived_owners_including_the_selected_one(
    tmp_path: Path,
) -> None:
    repository = build_repository(tmp_path / "repository")
    checkpoint.publish_process_meshing_checkpoint(
        repository,
        "mesh-accepted",
        "source-execution",
        checkpoint_state(),
        {name: jax.device_put(value) for name, value in scientific_arrays().items()},
        analysis_plan_id="mesh-analysis",
        numeric_revision_id="accepted-revision",
        writer_id="process-writer",
    )
    manifest = checkpoint.assemble_distributed_checkpoint_from_repository(
        repository,
        "mesh-accepted",
        "mesh-analysis",
        "accepted-revision",
        "source-execution",
        expected_process_count=1,
    )
    sharding = SingleDeviceSharding(jax.local_devices()[0])
    names = tuple(sorted(scientific_arrays()))
    layouts = checkpoint.meshing_checkpoint_array_layouts(manifest, names)
    assert layouts == {
        name: (value.shape, value.dtype) for name, value in scientific_arrays().items()
    }

    def restore(owners: tuple[int, ...]) -> checkpoint.MeshingCheckpointRestore:
        return checkpoint.restore_meshing_checkpoint(
            repository,
            manifest,
            {name: sharding for name in names},
            restart_relation(),
            TopologyRestartPolicy(allow_topology_change=True),
            expected_state_id=checkpoint_state().checkpoint_state_id,
            decoded_source_owners=owners,
        )

    restored = restore((0,))
    assert restored.state.to_payload() == checkpoint_state().to_payload()
    for name, expected in scientific_arrays().items():
        np.testing.assert_array_equal(
            np.asarray(restored.arrays[name]).view(np.uint8), expected.view(np.uint8)
        )
    for owners in ((), (1,), (0, 0)):
        with pytest.raises(ValueError, match="decoded_source_owners"):
            restore(owners)
    with pytest.raises(KeyError, match="no array named"):
        checkpoint.meshing_checkpoint_array_layouts(manifest, ("absent",))


@pytest.fixture(scope="module")
def packed_owner_checkpoint(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[str, list[dict]]:
    """Real adaptive four-owner publication, including its accepted rebalance."""
    directory = str(tmp_path_factory.mktemp("cold-packed-owner-checkpoint"))
    records = run_packed_owner_workers("packed_geometry_publish", directory, 4)
    assert [record["rank"] for record in records] == [0, 1, 2, 3]
    assert any(not record["empty"] for record in records)
    return directory, records


@pytest.fixture(scope="module")
def packed_empty_owner_checkpoint(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[str, list[dict]]:
    """Initial native publication retains genuinely empty owners without relabeling."""
    directory = str(tmp_path_factory.mktemp("cold-packed-empty-owner-checkpoint"))
    records = run_packed_owner_workers(
        "packed_geometry_publish", directory, 4, "empty"
    )
    assert [record["rank"] for record in records] == [0, 1, 2, 3]
    assert [record["rank"] for record in records if record["empty"]] == [1, 3]
    return directory, records


@pytest.fixture(scope="module")
def packed_graph_checkpoint(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[str, list[dict]]:
    """Retain actual weighted GRAPH migration, earned repack proof, and packets."""
    directory = str(tmp_path_factory.mktemp("cold-packed-graph-checkpoint"))
    records = run_packed_owner_workers("packed_geometry_publish", directory, 4, "graph")
    assert [record["rank"] for record in records] == [0, 1, 2, 3]
    assert all(record["graph"] and record["transition_id"] for record in records)
    return directory, records


@pytest.fixture(scope="module")
def packed_planar_checkpoint(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[str, list[dict]]:
    directory = str(tmp_path_factory.mktemp("cold-packed-planar-checkpoint"))
    records = run_packed_owner_workers("packed_geometry_publish", directory, 4, "planar")
    assert all(record["planar"] for record in records)
    return directory, records


@pytest.mark.parametrize("processes", (2, 4), ids=("4-to-2", "4-to-4"))
@pytest.mark.parametrize("permutation", ("original", "permuted"))
def test_native_packed_cold_planar_current_support(
    packed_planar_checkpoint: tuple[str, list[dict]],
    processes: int,
    permutation: str,
) -> None:
    directory, published = packed_planar_checkpoint
    records = run_packed_owner_workers(
        "packed_geometry_restore", directory, processes,
        json.dumps(published), permutation,
    )
    assert all(record["science_exact"] and record["bank_exact"] for record in records)


@pytest.mark.parametrize("processes", (2, 4), ids=("4-to-2", "4-to-4"))
@pytest.mark.parametrize("permutation", ("original", "permuted"))
def test_native_packed_cold_restore_uses_current_addressable_owners(
    packed_owner_checkpoint: tuple[str, list[dict]],
    processes: int,
    permutation: str,
) -> None:
    directory, published = packed_owner_checkpoint
    records = run_packed_owner_workers(
        "packed_geometry_restore",
        directory,
        processes,
        json.dumps(published),
        permutation,
    )
    expected = {
        (2, "original"): [[0, 1], [2, 3]],
        (2, "permuted"): [[2, 3], [0, 1]],
        (4, "original"): [[0], [1], [2], [3]],
        (4, "permuted"): [[1], [3], [0], [2]],
    }[processes, permutation]
    assert [record["rank"] for record in records] == list(range(processes))
    assert [record["source_owners"] for record in records] == expected
    assert [record["decoded_owners"] for record in records] == expected
    assert [record["addressed_owners"] for record in records] == expected
    assert all(record["science_exact"] and record["bank_exact"] for record in records)
    assert sorted(owner for record in records for owner in record["empty_owners"]) == [
        record["rank"] for record in published if record["empty"]
    ]


@pytest.mark.parametrize(
    "fault",
    (
        "missing-owner",
        "duplicate-owner",
        "foreign-owner",
        "missing-projection",
        "foreign-layout",
    ),
)
def test_current_owner_map_refuses_malformed_packed_storage_metadata(
    packed_owner_checkpoint: tuple[str, list[dict]], fault: str
) -> None:
    directory, _ = packed_owner_checkpoint
    records = run_packed_owner_workers(
        "packed_geometry_malformed", directory, 2, fault
    )
    assert [record["rank"] for record in records] == [0, 1]
    assert all(record["fault"] == fault and record["refusal"] for record in records)


@pytest.mark.parametrize("failed_rank", (0, 1), ids=("rank0", "rank1"))
@pytest.mark.parametrize(
    "stage",
    ("before", "after", "preflight", "constructor", "storedid", "changed", "unbound", "deleted"),
)
def test_native_packed_projection_failure_is_collective_and_retryable(
    packed_owner_checkpoint: tuple[str, list[dict]], failed_rank: int, stage: str
) -> None:
    directory, published = packed_owner_checkpoint
    records = run_packed_owner_workers(
        "packed_geometry_restore",
        directory,
        2,
        json.dumps(published),
        "original",
        f"{stage}-{failed_rank}",
    )
    assert all(record["collectively_refused"] for record in records)
    assert [record["injected_locally"] for record in records] == [
        rank == failed_rank for rank in range(2)
    ]
    assert [record["source_owners"] for record in records] == [[0, 1], [2, 3]]
    assert all(record["science_exact"] and record["bank_exact"] for record in records)


@pytest.mark.parametrize("processes", (2, 4), ids=("4-to-2", "4-to-4"))
def test_native_packed_cold_restore_preserves_empty_scientific_owners(
    packed_empty_owner_checkpoint: tuple[str, list[dict]], processes: int
) -> None:
    directory, published = packed_empty_owner_checkpoint
    records = run_packed_owner_workers(
        "packed_geometry_restore", directory, processes, json.dumps(published), "original"
    )
    expected = [[0, 1], [2, 3]] if processes == 2 else [[0], [1], [2], [3]]
    assert [record["source_owners"] for record in records] == expected
    assert [record["decoded_owners"] for record in records] == expected
    assert [record["addressed_owners"] for record in records] == expected
    assert sorted(owner for record in records for owner in record["empty_owners"]) == [
        1, 3
    ]
    assert all(record["science_exact"] and record["bank_exact"] for record in records)


def test_native_packed_graph_transition_cold_restore_retains_actual_proof_and_packets(
    packed_graph_checkpoint: tuple[str, list[dict]],
) -> None:
    directory, published = packed_graph_checkpoint
    records = run_packed_owner_workers(
        "packed_geometry_restore", directory, 2, json.dumps(published), "original"
    )
    assert [record["source_owners"] for record in records] == [[0, 1], [2, 3]]
    for record in records:
        for owner in record["decoded_owners"]:
            assert published[owner]["transition_id"] in record["replayed_transitions"]
        assert record["science_exact"] and record["bank_exact"]


@pytest.mark.parametrize("failed_rank", (0, 1), ids=("rank0", "rank1"))
@pytest.mark.parametrize("fault", ("graphowner", "graphproof"))
def test_native_packed_graph_transition_refuses_corruption_collectively(
    packed_graph_checkpoint: tuple[str, list[dict]], failed_rank: int, fault: str
) -> None:
    directory, published = packed_graph_checkpoint
    records = run_packed_owner_workers(
        "packed_geometry_restore", directory, 2, json.dumps(published),
        "original", f"{fault}-{failed_rank}",
    )
    assert all(record["collectively_refused"] for record in records)
    assert [record["injected_locally"] for record in records] == [
        rank == failed_rank for rank in range(2)
    ]
    for record in records:
        for owner in record["decoded_owners"]:
            assert published[owner]["transition_id"] in record["replayed_transitions"]
        assert record["science_exact"] and record["bank_exact"]


@pytest.mark.parametrize(
    "checkpoint_fixture",
    ("packed_owner_checkpoint", "packed_empty_owner_checkpoint", "packed_graph_checkpoint"),
    ids=("adaptive", "initial-empty", "graph"),
)
@pytest.mark.parametrize("processes", (2, 4), ids=("4-to-2", "4-to-4"))
@pytest.mark.parametrize("permutation", ("original", "permuted"))
@pytest.mark.parametrize("failed_rank", (0, 1), ids=("rank0", "rank1"))
@pytest.mark.parametrize("fault", ("operandmissing", "operandforeign"))
def test_native_packed_organization_operand_refusal_precedes_binding_and_retries(
    request: pytest.FixtureRequest,
    checkpoint_fixture: str,
    processes: int,
    permutation: str,
    failed_rank: int,
    fault: str,
) -> None:
    directory, published = request.getfixturevalue(checkpoint_fixture)
    records = run_packed_owner_workers(
        "packed_geometry_restore", directory, processes, json.dumps(published),
        permutation, f"{fault}-{failed_rank}",
    )
    assert all(record["collectively_refused"] for record in records)
    assert [record["injected_locally"] for record in records] == [
        rank == failed_rank for rank in range(processes)
    ]
    assert all(record["science_exact"] and record["bank_exact"] for record in records)


@pytest.mark.parametrize("headroom", (0, -1), ids=("R", "R-minus-one"))
def test_rectangular_checkpoint_packet_exact_storage_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, headroom: int,
) -> None:
    repository = build_repository(tmp_path / "repository")
    reference = np.arange(24, dtype=np.float64).reshape((4, 6))
    sharding = SingleDeviceSharding(jax.local_devices()[0])
    checkpoint.publish_process_checkpoint(
        repository, "packet-threshold", "one-owner",
        {"value": jax.device_put(reference, sharding)},
        analysis_plan_id="packet-analysis", numeric_revision_id="packet-revision",
        writer_id="packet-writer",
    )
    manifest = checkpoint.assemble_distributed_checkpoint_from_repository(
        repository, "packet-threshold", "packet-analysis", "packet-revision",
        "one-owner", expected_process_count=1,
    )
    path = jax.tree_util.keystr((jax.tree_util.DictKey("value"),))
    if headroom == -1:
        def forbidden_allocation(*args: object, **kwargs: object) -> None:
            raise AssertionError("Packet allocation preceded admission")

        monkeypatch.setattr(checkpoint.jax, "device_put", forbidden_allocation)
        with pytest.raises(ValueError, match="over-limit packet"):
            checkpoint.restore_global_array_from_checkpoint(
                repository, manifest, path, sharding,
                maximum_host_packet_bytes=reference.nbytes + headroom,
            )
    else:
        restored = checkpoint.restore_global_array_from_checkpoint(
            repository, manifest, path, sharding,
            maximum_host_packet_bytes=reference.nbytes,
        )
        assert np.asarray(restored).tobytes() == reference.tobytes()
