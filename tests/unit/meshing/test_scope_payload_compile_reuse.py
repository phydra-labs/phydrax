# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Scope predicates and immutable history payloads retain dynamic numerical inputs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec, SingleDeviceSharding

from phydrax.lifecycle._meshing_field_records import _history_payload_rows
from phydrax.meshing._scope import _scope_packet_values, selected_ids
from tools.compile_counter import CompileCounter


def test_scope_packet_binding_keeps_scientific_ids_dynamic() -> None:
    ids = jnp.asarray([30, 10, 20], dtype=jnp.int64)
    owners = jnp.asarray([1, 0, 1], dtype=jnp.int32)
    members = jnp.asarray([True, False, True])
    packets = jnp.asarray([[10, 20, -1], [30, -1, -1]], dtype=jnp.int64)
    packet_owners = jnp.asarray([[0, 1, -1], [1, -1, -1]], dtype=jnp.int32)
    valid = packets >= 0
    packet_members = jnp.asarray([[False, True, False], [True, False, False]])
    ordered, accepted = _scope_packet_values(
        ids, owners, members, packets, packet_owners, valid, packet_members
    )
    jax.block_until_ready((ordered, accepted))
    np.testing.assert_array_equal(ordered, [10, 20, 30])
    assert bool(accepted)
    changed_ids = ids + 100
    changed_packets = jnp.where(valid, packets + 100, packets)
    bad_members = packet_members.at[0, 2].set(True)
    with CompileCounter() as counter:
        changed, accepted = _scope_packet_values(
            changed_ids,
            owners,
            members,
            changed_packets,
            packet_owners,
            valid,
            packet_members,
        )
        _, refused = _scope_packet_values(
            ids, owners, members, packets, packet_owners, valid, bad_members
        )
        jax.block_until_ready((changed, accepted, refused))
    assert counter.count == 0
    np.testing.assert_array_equal(changed, [110, 120, 130])
    assert bool(accepted)
    assert not bool(refused)


@pytest.mark.parametrize(
    "dtype",
    (
        np.float16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
        np.int32,
        np.uint64,
        np.bool_,
    ),
)
def test_history_payload_row_transport_preserves_all_bytes(dtype: type) -> None:
    source = np.arange(12).astype(dtype).reshape((2, 3, 2))
    if source.dtype.kind in "fc":
        source.flat[0] = -0.0
        source.flat[2] = np.nan
    if source.dtype.kind == "c":
        source.flat[1] = complex(-0.0, -0.0)
    values = jnp.asarray(source)
    rows = np.asarray([3, 0, 4], dtype=np.int64)
    changed_rows = np.asarray([0, 4, 3], dtype=np.int64)
    actual = _history_payload_rows(values, rows)
    jax.block_until_ready(actual)
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint8), source.reshape((6, 2))[rows].view(np.uint8)
    )
    with CompileCounter() as counter:
        changed = _history_payload_rows(values, changed_rows)
        jax.block_until_ready(changed)
    assert counter.count == 0
    np.testing.assert_array_equal(
        np.asarray(changed).view(np.uint8),
        source.reshape((6, 2))[changed_rows].view(np.uint8),
    )


@pytest.mark.parametrize("named_mask", (False, True))
def test_single_cpu_named_scope_selection_compiles_zero(named_mask: bool) -> None:
    device = jax.local_devices()[0]
    mesh = Mesh(np.asarray((device,), dtype=object), ("owner",))
    placement = NamedSharding(mesh, PartitionSpec("owner"))
    mask_placement = placement if named_mask else SingleDeviceSharding(device)
    host_ids = np.asarray((7, 2, 3, 9, -1, 5), dtype=np.int64)
    identifiers = jax.device_put(host_ids, placement)
    host_masks = (
        np.zeros(6, dtype=np.bool_),
        np.asarray((True, False, True, False, False, True), dtype=np.bool_),
        np.ones(6, dtype=np.bool_),
    )
    masks = tuple(jax.device_put(mask, mask_placement) for mask in host_masks)
    jax.clear_caches()
    with CompileCounter() as compiler:
        results = tuple(
            jax.block_until_ready(selected_ids(identifiers, mask)) for mask in masks
        )
    assert compiler.count == 0
    for host_mask, result in zip(host_masks, results, strict=True):
        assert result.sharding == placement
        assert result.dtype == host_ids.dtype
        np.testing.assert_array_equal(np.asarray(result), host_ids[host_mask])
