# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Checkpoint snapshot ownership and persisted scientific identity contracts."""

import jax
import numpy as np
import pytest

from phydrax._host_storage import immutable_host_array, immutable_host_array_layout
from phydrax.lifecycle import MeshingCheckpointState, snapshot_addressable_arrays
from tests._support.distributed_meshing.restart_builders import checkpoint_state


def test_checkpoint_state_retains_baseline_scientific_identity() -> None:
    state = checkpoint_state()
    # SHA-256 of the deterministic metadata payload declared by the pre-remediation
    # checkpoint constructor and the shared checkpoint_state builder.
    expected = "62ff22226f9f091bef3ded26ac12802020b91e50ad677b603df2a851e9a5c7de"
    assert state.checkpoint_state_id == expected
    restored = MeshingCheckpointState.from_payload(state.to_payload())
    assert restored.checkpoint_state_id == expected
    assert restored.to_payload() == state.to_payload()


def test_host_checkpoint_snapshot_copies_writable_coefficients() -> None:
    authored = np.arange(6, dtype=np.float64).reshape(2, 3)
    expected = authored.copy()
    snapshots = snapshot_addressable_arrays({"authored": authored})
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.global_shape == (2, 3)
    assert snapshot.index == ((0, 2, 1), (0, 3, 1))
    assert snapshot.process_index == jax.process_index()
    assert snapshot.replica_id == 0
    assert not snapshot.payload.flags.writeable
    authored[0, 0] = -1.0
    np.testing.assert_array_equal(snapshot.payload, expected)


def test_host_checkpoint_snapshot_copies_readonly_mutable_aliases() -> None:
    owner = np.arange(8, dtype=np.int64).reshape(2, 4)
    authored = owner[:, ::2]
    authored.setflags(write=False)
    expected = authored.copy()
    snapshots = snapshot_addressable_arrays({"authored": authored})
    assert len(snapshots) == 1
    assert snapshots[0].payload is not authored
    owner[:] = -1
    np.testing.assert_array_equal(snapshots[0].payload, expected)
    assert immutable_host_array_layout(snapshots[0].payload) is not None


def test_host_checkpoint_snapshot_reuses_genuinely_immutable_coefficients() -> None:
    authored = immutable_host_array(np.arange(8, dtype=np.int64).reshape(2, 4))
    snapshots = snapshot_addressable_arrays({"authored": authored})
    assert len(snapshots) == 1
    assert snapshots[0].payload is authored
    with pytest.raises(ValueError):
        snapshots[0].payload.setflags(write=True)


def test_mixed_checkpoint_snapshot_preserves_host_then_device_publication() -> None:
    authored = np.asarray([2, 3], dtype=np.int64)
    dynamic = jax.device_put(np.asarray([5, 7], dtype=np.int64))
    snapshots = snapshot_addressable_arrays({"device": dynamic, "host": authored})
    assert tuple(snapshot.array_path for snapshot in snapshots) == (
        "['host']",
        "['device']",
    )
    np.testing.assert_array_equal(snapshots[0].payload, authored)
    np.testing.assert_array_equal(snapshots[1].payload, np.asarray(dynamic))
    assert all(snapshot.index == ((0, 2, 1),) for snapshot in snapshots)
