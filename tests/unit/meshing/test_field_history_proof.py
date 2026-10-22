# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Earned native repack proofs preserve field recovery and refuse tampering."""

from typing import Any

import pytest

from tests._support.forced_devices import run_on_forced_devices


@pytest.fixture(scope="module")
def history_proof() -> dict[str, Any]:
    return run_on_forced_devices(
        "tests._support.distributed_meshing.field_history_proof",
        "earned_history_proof",
        (),
        devices=2,
        env={},
    )


def test_earned_proof_matches_full_validation_bitwise(
    history_proof: dict[str, Any],
) -> None:
    comparison = history_proof["equivalence"]
    assert comparison["raw_bitwise"]
    assert comparison["compact_bitwise"]
    assert comparison["raw_oracle"]
    assert comparison["compact_oracle"], comparison["compact_diagnostics"]
    assert history_proof["proof_content_id"]


def test_graph_failure_evidence_distinguishes_transport_from_inherited_error(
    history_proof: dict[str, Any],
) -> None:
    evidence = history_proof["failure_evidence"]
    integer = evidence["integer"]
    assert integer["dtype"] == "uint64"
    assert integer["max_error"] == 2**64 - 1
    assert integer["first_differing_global_id"] == 7
    assert integer["first_differing_owner"] == 1
    assert integer["first_actual"] == 2**63
    assert integer["first_expected"] == 2**63 + 1
    assert integer["outside_tolerance_entities"] == 2
    assert evidence["transport"]["max_error"] == 0
    assert evidence["transport"]["first_differing_global_id"] is None
    assert evidence["physical"]["max_error"] > 1e-12
    assert evidence["physical"]["outside_tolerance_entities"] == 1
    assert evidence["physical"]["first_differing_global_id"] == 91
    assert evidence["nonfinite"]["max_error"] == float("inf")
    assert evidence["nonfinite"]["outside_tolerance_entities"] == 1


def test_earned_proof_preserves_role_and_index_identities(
    history_proof: dict[str, Any],
) -> None:
    comparison = history_proof["equivalence"]
    assert comparison["role_identity"]
    assert comparison["raw_ids"]
    assert comparison["target_identity"]
    assert comparison["banks"] == ["epoch/vertex_ids", "mesh/vertex_ids"]


@pytest.mark.parametrize(
    "inventory",
    (
        "target-history",
        "saved-history",
        "target-forest",
        "saved-forest",
        "requested-owners",
        "vertex-owners",
        "saved-locations",
        "saved-controls",
        "record-counts",
        "status",
    ),
)
def test_earned_proof_refuses_changed_receipt_arrays(
    history_proof: dict[str, Any],
    inventory: str,
) -> None:
    refusals = history_proof["array_refusals"][inventory]
    assert "different array bindings" in refusals["proof"], inventory
    assert "different array bindings" in refusals["rebind"], inventory
    assert "different array bindings" in refusals["recover"], inventory


@pytest.mark.parametrize("binding", ("source", "layout", "placement"))
def test_earned_proof_refuses_foreign_receipt_bindings(
    history_proof: dict[str, Any],
    binding: str,
) -> None:
    refusals = history_proof["foreign_refusals"][binding]
    assert refusals["proof"], binding
    assert refusals["rebind"], binding
    assert refusals["recover"], binding


@pytest.mark.parametrize("consumer", ("rebind", "recover"))
def test_valid_repack_proof_still_refuses_modified_field_payload(
    history_proof: dict[str, Any],
    consumer: str,
) -> None:
    message = history_proof["payload_refusals"][consumer]
    expected = (
        "must match the saved payload"
        if consumer == "rebind"
        else "must match the retained payload slot"
    )
    assert expected in message


@pytest.mark.parametrize("dtype", ("uint32", "uint64"))
def test_rng_counter_history_moves_exactly_and_preserves_next_draws(
    history_proof: dict[str, Any],
    dtype: str,
) -> None:
    comparison = history_proof["rng"][dtype]
    assert comparison["dtype_preserved"]
    assert comparison["counter_exact"]
    assert comparison["next_draws_exact"]
    assert comparison["foreign_stream_refused"]
    assert comparison["counter_edit_refused"]


def test_accepted_euler_history_moves_all_four_components_exactly(
    history_proof: dict[str, Any],
) -> None:
    assert all(history_proof["euler"].values())


@pytest.fixture(scope="module")
def live_publication_archive(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = tmp_path_factory.mktemp("live-publication-history") / "source.npz"
    saved = run_on_forced_devices(
        "tests._support.distributed_meshing.field_history_proof",
        "save_live_publication_history",
        (str(path),),
        devices=2,
        env={},
    )
    # A second interpreter has neither the construction call nor its host view.
    restored = run_on_forced_devices(
        "tests._support.distributed_meshing.field_history_proof",
        "replay_live_publication_history",
        (str(path), saved["archive_content_id"]),
        devices=2,
        env={},
    )
    return saved, restored


def test_actual_two_device_public_and_cold_sources_still_reprove_raw_state(
    live_publication_archive: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    _, restored = live_publication_archive
    assert restored["public_restore_raw_replays"] == 1
    assert restored["retained_source_raw_replays"] == 1
    assert restored["cold_full_raw_reproof"] == 1
    assert restored["cold_physical_publications"] > 0
    assert restored["all_restored_state_fields_exact"]


def test_actual_live_view_is_not_restored_or_retained_globally(
    live_publication_archive: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    saved, restored = live_publication_archive
    assert saved["expired_view_refused"]
    assert saved["issued_count"] >= 3
    assert restored["no_restored_or_global_live_view"]
    assert restored["archive_cannot_issue_view"]


def test_deleted_bank_refused_only_after_separate_actual_full_publication(
    live_publication_archive: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    saved, _ = live_publication_archive
    assert saved["deleted_bank_refused_after_full_disposable_publication"]


def test_live_collector_does_not_change_persistent_evidence_fields_or_ids(
    live_publication_archive: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    saved, restored = live_publication_archive
    for name in ("result_id", "evidence_id", "stored_content_id"):
        assert saved[name] == restored[name]
        assert saved[name]
    expected = [
        "partition_checks", "global_checks", "source", "preparation",
        "initial_states", "compiled_states", "raw_source_blocks",
        "publication_source_blocks", "uniform_refinement", "source_exterior",
        "layout", "neighbor_pairs", "axis_name", "logical_arrays", "entity_keys",
        "entity_ids", "entity_owners", "global_entity_counts", "presentation_names",
        "topology_id", "geometry_id", "coordinate_geometry_id", "source_mesh_id",
        "source_audit_id", "source_embedding", "source_premise_id",
        "source_evidence_id", "global_organization_id", "mesh_id", "partition_count",
        "migration_record", "rebalance_proposal", "checks", "evidence_id",
        "stored_content_id",
    ]
    assert saved["persistent_fields"] == restored["persistent_fields"] == expected


@pytest.mark.parametrize("defect", ("missing", "duplicate", "foreign", "missing-field"))
def test_rearchive_refuses_incomplete_role_inventory(defect: str) -> None:
    from types import SimpleNamespace

    from tools.meshing_qualification.scenarios.distributed import (
        _DISTRIBUTED_FIELD_ROLES,
        _DISTRIBUTED_PART,
        _distributed_rearchive_closure,
    )

    names = [name for row in _DISTRIBUTED_FIELD_ROLES for name in row[:2]]
    names.extend(("accepted-epoch", "rng-root-key"))
    values = dict.fromkeys(names)
    if defect == "missing":
        names.pop()
    elif defect == "duplicate":
        names.append(names[0])
    elif defect == "foreign":
        names[-1] = "foreign-field"
    else:
        values.pop(names[0])
    declaration = SimpleNamespace(
        state_roles=tuple(SimpleNamespace(entry_name=name) for name in names)
    )
    restart: Any = SimpleNamespace(
        closure={"field_declarations": {_DISTRIBUTED_PART: declaration}}, fields=values
    )
    with pytest.raises(ValueError, match="restart.rearchive"):
        _distributed_rearchive_closure(restart, 1)


def test_historical_owner_provenance_rearchive(tmp_path: Any) -> None:
    result = run_on_forced_devices(
        "tests._support.distributed_meshing.field_history_proof",
        "historical_owner_rearchive", (str(tmp_path / "owners"),), devices=2, env={},
    )
    assert all(result["refusals"].values())
    assert result["content_id_equal"]
    assert result["current_id_equal"]
    assert result["historical_ids_equal"]


def test_projection_ring_preserves_raw_identity_across_current_placements() -> None:
    matrix = run_on_forced_devices(
        "tests._support.distributed_meshing.field_history_proof",
        "projection_ring_matrix", (), devices=4, env={},
    )
    assert len(matrix) == 4
    assert all(matrix.values())


@pytest.mark.parametrize("count", (2, 4))
def test_native_projection_collective_refusal_and_retry(count: int) -> None:
    import socket
    from concurrent.futures import ThreadPoolExecutor

    from phydrax import config

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{listener.getsockname()[1]}"
    with ThreadPoolExecutor(max_workers=count) as executor:
        futures = [
            executor.submit(
                run_on_forced_devices,
                "tests._support.distributed_meshing.field_history_proof",
                "native_projection_collective", (), devices=1,
                env=config.environment(
                    config.RuntimeConfig(
                        bootstrap=config.RuntimeBootstrap(
                            coordinator_address=coordinator,
                            num_processes=count,
                            process_id=rank,
                            local_device_ids=(0,),
                        ),
                        cpu_collectives_implementation="gloo",
                    ),
                    base={},
                ),
            )
            for rank in range(count)
        ]
        results = [future.result() for future in futures]
    assert all(result["refusal_and_retry"] for result in results)
    assert all(len(result["checked"]) == 23 for result in results)


def test_production_coarsened_projection_prefix(tmp_path: Any) -> None:
    import socket
    from concurrent.futures import ThreadPoolExecutor

    from phydrax import config

    count = 6
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{listener.getsockname()[1]}"
    with ThreadPoolExecutor(max_workers=count) as executor:
        futures = [
            executor.submit(
                run_on_forced_devices,
                "tests._support.distributed_meshing.field_history_proof",
                "production_projection_prefix", (str(tmp_path / "prefix"),),
                devices=1, timeout_seconds=600,
                env=config.environment(
                    config.RuntimeConfig(
                        bootstrap=config.RuntimeBootstrap(
                            coordinator_address=coordinator, num_processes=count,
                            process_id=rank, local_device_ids=(0,),
                        ),
                        cpu_collectives_implementation="gloo",
                    ),
                    base={},
                ),
            )
            for rank in range(count)
        ]
        results = [future.result() for future in futures]
    assert all(result["passed"] for result in results), results
    assert sum(result["retired_vertex_rows"] for result in results) > 0
