# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Forest routing and record ownership reuse warmed collective programs."""

from __future__ import annotations

import pytest

from tests._support.forced_devices import run_on_forced_devices


@pytest.mark.parametrize("devices", (2, 3))
@pytest.mark.parametrize("operation", ("owners", "routes"))
def test_forest_numerics_reuse_identical_and_changed_owner_data(
    devices: int, operation: str
) -> None:
    receipt = run_on_forced_devices(
        "tests._support.distributed_meshing.forest_compile_reuse",
        "compile_reuse",
        (operation,),
        devices=devices,
        env={},
    )
    assert receipt["cold_count"] > 0
    assert receipt["warm_count"] == 0, receipt["warm_events"]
    assert receipt["changed_count"] == 0, receipt["changed_events"]
    assert receipt["first"] == receipt["repeated"]
    assert receipt["first"] != receipt["changed"]


@pytest.mark.parametrize("kind", ("shape", "asymmetric"))
def test_forest_routes_retain_route_mask_refusals(kind: str) -> None:
    receipt = run_on_forced_devices(
        "tests._support.distributed_meshing.forest_compile_reuse",
        "route_refusal",
        (kind,),
        devices=2,
        env={},
    )
    assert receipt["refused"] is True


def test_vertex_record_owners_require_a_bound_collective_axis() -> None:
    receipt = run_on_forced_devices(
        "tests._support.distributed_meshing.forest_compile_reuse",
        "owner_refusal",
        (),
        devices=2,
        env={},
    )
    assert receipt["refused"] is True
