# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

import json
import socket
from concurrent.futures import ThreadPoolExecutor

from tests._support.forced_devices import gloo_rank_environment, run_on_forced_devices


# Fixed empirical residual bounds from test_overset_owner_traffic.py
# at a6827a9e5; each comparison retains its strictness.
# Two-process sparse donor action, explicit transpose and primal/dual pairing versus
# their reference records.
_SPARSE_OWNER_ACTION_RESIDUAL = 1e-11


def test_actual_two_process_sparse_donor_fringe_action_and_transpose() -> None:

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    coordinator = f"127.0.0.1:{port}"
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                run_on_forced_devices,
                "tests._support.distributed_meshing.overset_traffic",
                "sparse_donor_fringe_traffic",
                (),
                devices=1,
                env=gloo_rank_environment(coordinator, rank),
            )
            for rank in range(2)
        ]
        records = [future.result() for future in futures]
    assert [record["rank"] for record in records] == [0, 1]
    assert all(
        record["processes"] == 2 and record["collective_permute"] for record in records
    )
    assert all(not record["all_gather"] for record in records)
    assert all(record["nonaddressable_routes"] for record in records)
    for profile, capacity in (
        ("fe", 6),
        ("fe_vector", 12),
        ("fv", 1),
        ("fe_image_vector", 12),
        ("fe_image_invariant", 12),
        ("fe_affine_vector", 12),
    ):
        assert all(record[f"{profile}_packet_capacity"] == capacity for record in records)
    assert records[0]["fe_served"] == records[1]["fe_requested"] == 6
    assert records[0]["fe_vector_served"] == records[1]["fe_vector_requested"] == 12
    assert (
        records[0]["fe_image_vector_served"]
        == records[1]["fe_image_vector_requested"]
        == 12
    )
    assert (
        records[0]["fe_image_invariant_served"]
        == records[1]["fe_image_invariant_requested"]
        == 12
    )
    assert (
        records[0]["fe_affine_vector_served"]
        == records[1]["fe_affine_vector_requested"]
        == 12
    )
    assert records[0]["fv_served"] == records[1]["fv_requested"] == 1
    for record in records:
        assert record["max_action_error"] < _SPARSE_OWNER_ACTION_RESIDUAL
        assert record["max_transpose_error"] < _SPARSE_OWNER_ACTION_RESIDUAL
        assert record["max_duality_error"] < _SPARSE_OWNER_ACTION_RESIDUAL
        assert record["bounded_refusal"] and record["missing_owner_refusal"]
        assert record["nonrigid_polar_refusal"]
    print(json.dumps(records, sort_keys=True))
