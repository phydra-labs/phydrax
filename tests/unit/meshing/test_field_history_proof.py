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
