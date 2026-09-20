#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._fingerprint import canonical_fingerprint
from phydrax.lifecycle._event_graph_repository import (
    EventGraphRepository,
    GlobalWorkItem,
)
from phydrax.lifecycle._repository import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
)
from phydrax.solver._dark_sector_epoch_runtime import (
    admit_dark_sector_work,
    DarkSectorEpochPlan,
    DarkSectorRunCoordinator,
    empty_dark_sector_epoch_state,
    finalize_dark_sector_epoch,
    replace_dark_sector_conservation,
)


class SimulatedCrash(RuntimeError):
    pass


def _digest(name):
    return canonical_fingerprint({"test": name})


def _plan(*, work=1, frontier=2, shards=1, species="species", topology="topology"):
    return DarkSectorEpochPlan(
        packet_capacity=1,
        event_capacity=1,
        product_capacity=1,
        radiation_capacity=1,
        work_capacity=work,
        frontier_capacity=frontier,
        packet_width=4,
        event_width=3,
        product_width=4,
        radiation_width=4,
        work_width=2,
        frontier_width=2,
        species_revision_id=_digest(species),
        topology_revision_id=_digest(topology),
        shard_count=shards,
    )


def _repository(path, *, fail=None):
    profile = HPCFilesystemProfile(
        "dark-sector-runtime-posix",
        "local-posix",
        atomic_rename_same_filesystem=True,
        file_fsync=True,
        directory_fsync=True,
        advisory_locking=True,
        attempt_private_staging=True,
    )
    artifacts = POSIXArtifactRepository(
        path,
        POSIXRepositoryPolicy(
            profile,
            maximum_chunk_bytes=256,
            maximum_metadata_bytes=64 * 1024,
        ),
    )
    return EventGraphRepository(
        artifacts,
        maximum_record_bytes=1024 * 1024,
        failure_injector=fail,
    )


def _work(index, parent=None):
    return GlobalWorkItem(
        "cascade-step",
        (),
        _digest("matrix-element"),
        partition_key=f"partition-{index % 2}",
        priority=index,
        parent_work_id=None if parent is None else parent.work_id,
    )


def test_compile_signature_changes_with_species_topology_capacity_and_sharding():
    base = _plan()
    revisions = (
        _plan(species="species"),
        _plan(topology="topology"),
        _plan(work=2),
        _plan(shards=2),
    )
    assert all(value.plan_id != base.plan_id for value in revisions)
    assert all(
        value.compile_signature_id != base.compile_signature_id for value in revisions
    )


def test_fixed_epoch_spills_without_clipping_and_rolls_back_conservation():
    plan = _plan(work=1, frontier=2)
    initial = empty_dark_sector_epoch_state(plan, epoch_sequence=0)
    work = tuple(_work(index) for index in range(3))
    admission = admit_dark_sector_work(
        initial,
        tuple(item.work_id for item in work),
        jnp.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]),
    )
    assert bool(admission.backpressured)
    assert not bool(admission.refused)
    np.testing.assert_array_equal(admission.state.resident_counts[-2:], (1, 2))

    refused = admit_dark_sector_work(
        admission.state,
        (_work(4).work_id,),
        jnp.asarray([[4.0, 5.0]]),
    )
    assert bool(refused.refused)
    np.testing.assert_array_equal(refused.state.work_ids, admission.state.work_ids)
    np.testing.assert_array_equal(
        refused.state.frontier_ids, admission.state.frontier_ids
    )

    invalid = replace_dark_sector_conservation(
        admission.state,
        jnp.zeros((8,)),
        jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    rolled = finalize_dark_sector_epoch(
        initial, invalid, complete=True, backpressured=True
    )
    assert bool(rolled.rolled_back)
    np.testing.assert_array_equal(rolled.state.work_mask, initial.work_mask)


def test_finalize_is_jax_safe_for_fixed_compiled_segments():
    plan = _plan()
    state = empty_dark_sector_epoch_state(plan, epoch_sequence=0)

    @jax.jit
    def finish(previous, proposed):
        return finalize_dark_sector_epoch(
            previous, proposed, complete=True, backpressured=False
        )

    result = finish(state, state)
    assert bool(result.complete)
    assert bool(result.conservation_ok)
    assert result.state.work_ids.shape == (plan.work_capacity, 8)


def test_coordinator_recovers_after_tip_and_repartitions_changed_plan(tmp_path):
    crashed = False

    def fail(point):
        nonlocal crashed
        if point == "after_tip" and not crashed:
            crashed = True
            raise SimulatedCrash(point)

    repository = _repository(tmp_path / "repository", fail=fail)
    plan = _plan(work=1, frontier=2)
    coordinator = DarkSectorRunCoordinator(
        repository, plan, "unbounded-run", worker_id="worker-a"
    )
    state = coordinator.resume().state
    first = _work(0)
    second = _work(1, first)
    for item in (first, second):
        repository.put_work(item, "worker-a", committed_at=1)
    admission = admit_dark_sector_work(
        state,
        (first.work_id, second.work_id),
        jnp.asarray([[0.0, 1.0], [1.0, 2.0]]),
    )
    result = finalize_dark_sector_epoch(
        state,
        admission.state,
        complete=True,
        backpressured=admission.backpressured,
        evidence_ids=("finite-resident", "durable-frontier"),
    )
    with pytest.raises(SimulatedCrash):
        coordinator.commit_epoch(
            result,
            work_items=(first, second),
            matrix_element_revision_id=_digest("matrix-element"),
            committed_at=2,
        )
    repeated = coordinator.commit_epoch(
        result,
        work_items=(first, second),
        matrix_element_revision_id=_digest("matrix-element"),
        committed_at=2,
    )
    assert repeated.tip.epoch_sequence == 0

    changed_plan = _plan(work=2, frontier=1, shards=2)
    changed = DarkSectorRunCoordinator(
        repository, changed_plan, "unbounded-run", worker_id="worker-a"
    ).resume()
    assert changed.state.epoch_sequence == 1
    assert changed.state.parent_epoch_manifest_id == repeated.manifest.epoch_manifest_id
    assert int(jnp.sum(changed.state.work_mask)) == 1
    assert int(jnp.sum(changed.state.frontier_mask)) == 0
    assert changed.repartitioned
    assert not changed.exact_compile_replay


def test_expired_owner_can_recover_staged_slot_before_tip(tmp_path):
    crashed = False

    def fail(point):
        nonlocal crashed
        if point == "after_epoch_manifest" and not crashed:
            crashed = True
            raise SimulatedCrash(point)

    repository = _repository(tmp_path / "owner-failover", fail=fail)
    plan = _plan()
    original = DarkSectorRunCoordinator(
        repository, plan, "failover-run", worker_id="worker-a"
    )
    state = original.resume().state
    work = _work(0)
    admitted = admit_dark_sector_work(state, (work.work_id,), jnp.asarray([[0.0, 1.0]]))
    result = finalize_dark_sector_epoch(
        state,
        admitted.state,
        complete=True,
        backpressured=False,
    )
    with pytest.raises(SimulatedCrash):
        original.commit_epoch(
            result,
            work_items=(work,),
            matrix_element_revision_id=_digest("matrix-element"),
            committed_at=1,
        )

    replacement = DarkSectorRunCoordinator(
        repository, plan, "failover-run", worker_id="worker-b"
    )
    recovered = replacement.commit_epoch(
        result,
        work_items=(work,),
        matrix_element_revision_id=_digest("matrix-element"),
        committed_at=2,
    )
    assert recovered.tip.epoch_sequence == 0
    assert recovered.manifest.commit_owner_id == "worker-a"
