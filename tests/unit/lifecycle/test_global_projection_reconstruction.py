# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Collective projection restore preserves admitted owner-local bytes."""

from tests._support.distributed_meshing.global_projection_reconstruction import (
    run_global_projection_workers,
)
from tests._support.forced_devices import run_on_forced_devices


def test_global_projection_reconstruction_authenticates_constructor_values() -> None:
    records = run_global_projection_workers("projection_reconstruction")
    assert [record["rank"] for record in records] == [0, 1]
    for record in records:
        assert record["process_count"] == 2
        assert record["local_device_count"] == 1
        assert record["nonaddressable_arrays"] > 0
        assert record["exact_local_shards"] > 0
        assert record["global_cell_count"] > 2
        assert record["unbound_original_refused"]
        assert record["admitted_mutation_refused"]
        assert record["owner_mutation_refused"]
    assert records[0]["source_content_id"] == records[1]["source_content_id"]
    assert records[0]["coordinate_geometry_id"] == records[1]["coordinate_geometry_id"]
    assert records[0]["global_cell_count"] == records[1]["global_cell_count"]


def test_global_array_comparison_preserves_signed_zero_and_nan_payload_bits() -> None:
    records = run_global_projection_workers("exact_payload_comparison")
    assert [record["rank"] for record in records] == [0, 1]
    assert [record["comparison_rounds"] for record in records] == [1, 2]
    for record in records:
        assert record["process_count"] == 2
        assert record["local_device_count"] == 1
        assert record["nonaddressable"]
        assert record["equal_copy_accepted"]
        assert record["signed_zero_refused"]
        assert record["nan_payload_refused"]
        assert record["unbound_original_refused"]
        assert record["maximum_chunk_bytes"] == 8
    assert records[0]["digest"] == records[1]["digest"]


def test_global_array_comparison_retains_default_bound_across_local_layouts() -> None:
    record = run_on_forced_devices(
        "tests._support.distributed_meshing.global_projection_reconstruction",
        "local_layout_comparison",
        (),
        devices=2,
        env={},
    )
    assert record == {
        "equal_copy_accepted": True,
        "signed_zero_refused": True,
        "nan_payload_refused": True,
        "default_chunk_limit": True,
    }
