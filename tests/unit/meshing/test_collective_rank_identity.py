# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Two-process scientific identities remain independent of local storage anchors."""

import socket
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from tests._support.forced_devices import gloo_rank_environment, run_on_forced_devices


@pytest.fixture(scope="module")
def two_rank_epochs() -> tuple[dict[str, Any], ...]:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{listener.getsockname()[1]}"

    def launch(rank: int) -> dict[str, Any]:
        return run_on_forced_devices(
            "tests._support.distributed_meshing.collective_rank_identity",
            "consecutive_epochs",
            (),
            devices=1,
            env=gloo_rank_environment(coordinator, rank),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        owners = tuple(executor.map(launch, range(2)))
    assert [owner["rank"] for owner in owners] == [0, 1]
    for owner in owners:
        assert [epoch["epoch"] for epoch in owner["epochs"]] == [0, 1]
        assert [epoch["partition"] for epoch in owner["epochs"]] == [owner["rank"]] * 2
    return owners


def test_consecutive_results_have_rank_invariant_scientific_identity(
    two_rank_epochs: tuple[dict[str, Any], ...],
) -> None:
    first, second = two_rank_epochs
    assert first["initial_source_result_id"] == second["initial_source_result_id"]
    for left, right in zip(first["epochs"], second["epochs"], strict=True):
        assert left["result_id"] == right["result_id"], f"epoch {left['epoch']}"
        assert left["global_cell_count"] == right["global_cell_count"]
    for owner in two_rank_epochs:
        initial, continued = owner["epochs"]
        assert initial["result_id"] != owner["initial_source_result_id"]
        assert continued["result_id"] != initial["result_id"]
        assert 2 < initial["global_cell_count"] < continued["global_cell_count"]


def test_owner_local_consumption_preserves_source_and_theorem_identity_across_ranks(
    two_rank_epochs: tuple[dict[str, Any], ...],
) -> None:
    first, second = two_rank_epochs
    for left, right in zip(first["epochs"], second["epochs"], strict=True):
        for field in (
            "source_result_id",
            "witness_source_result_id",
            "source_evidence_id",
            "theorem_evidence_id",
            "predecessor_evidence_id",
        ):
            assert left[field] == right[field], f"epoch {left['epoch']}: {field}"
    for owner in two_rank_epochs:
        initial, continued = owner["epochs"]
        assert not initial["source_is_owner_local"]
        assert continued["source_is_owner_local"]
        assert initial["source_result_id"] == owner["initial_source_result_id"]
        assert initial["witness_source_result_id"] == initial["source_result_id"]
        assert initial["predecessor_evidence_id"] is None
        assert continued["source_result_id"] == initial["result_id"]
        assert continued["witness_source_result_id"] == continued["source_result_id"]
        assert continued["predecessor_evidence_id"] == initial["theorem_evidence_id"]
        assert continued["source_evidence_id"] != initial["source_evidence_id"]
        assert continued["theorem_evidence_id"] != initial["theorem_evidence_id"]


def test_result_storage_anchor_matches_child_on_each_rank(
    two_rank_epochs: tuple[dict[str, Any], ...],
) -> None:
    # Storage digests bind rank-local carriers; they need not agree across ranks.
    for owner in two_rank_epochs:
        for epoch in owner["epochs"]:
            assert epoch["stored_content_id"]
            assert epoch["stored_content_id"] == epoch["child_stored_content_id"], (
                f"rank {owner['rank']}, epoch {epoch['epoch']}"
            )
