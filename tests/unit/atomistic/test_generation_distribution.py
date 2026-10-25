#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Single-process contracts of generation placement, quotas and continuation.

These cases run in ONE process: the placed stream is a single-participant
execution group. They establish the scheduler, capacity, authority and
lifecycle-commit contracts; multi-participant behavior is proven only by the
real multi-process smoke (`tests/integration/atomistic/test_distributed_generation.py`).
Independent controls recompute proposals from their semantic addresses.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.random as jr
import numpy as np
import pytest

from phydrax.atomistic._generation import (
    AtomisticGenerationState,
    AtomisticGenerationStopReason,
    AtomisticMemoryGenerationSink,
    PreparedAtomisticGenerationStream,
)
from phydrax.atomistic._generation_distribution import (
    AtomisticGenerationCapacityError,
    AtomisticGenerationCheckpointError,
    AtomisticGenerationPlacement,
)
from phydrax.execution import ExecutionRuntime
from phydrax.lifecycle._repository import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
)
from tests._support.atomistic_generation import expected_positions, stream_plan


ROOT = 11


def _placement(
    round_bytes: int = 1 << 22, replicated: int = 1 << 22
) -> AtomisticGenerationPlacement:
    return AtomisticGenerationPlacement(
        ExecutionRuntime.current().root_group,
        maximum_round_bytes=round_bytes,
        maximum_replicated_bytes=replicated,
    )


def _repository(tmp_path: Path) -> POSIXArtifactRepository:
    profile = HPCFilesystemProfile(
        "posix.generation-test",
        "local-posix",
        atomic_rename_same_filesystem=True,
        file_fsync=True,
        directory_fsync=True,
        advisory_locking=True,
        attempt_private_staging=True,
    )
    return POSIXArtifactRepository(
        tmp_path / "repository",
        POSIXRepositoryPolicy(
            profile, maximum_chunk_bytes=1 << 16, maximum_metadata_bytes=1 << 20
        ),
    )


def _drain(
    stream: PreparedAtomisticGenerationStream, *, target: int, budget: int
) -> tuple[AtomisticMemoryGenerationSink, AtomisticGenerationState]:
    sink = AtomisticMemoryGenerationSink("memory")
    state = stream.start(jr.key(ROOT), target=target, attempt_budget=budget)
    state = stream.run(state, sink, maximum_transitions=400)
    return sink, state


def test_target_reached_retains_in_flight_lanes_and_extension_is_exact() -> None:
    stream = stream_plan(lanes_per_member=3).prepare()
    direct = AtomisticMemoryGenerationSink("direct")
    state = stream.start(jr.key(ROOT), target=9, attempt_budget=40)
    state = stream.run(state, direct, maximum_transitions=400)
    assert state.stop_reason is AtomisticGenerationStopReason.TARGET_REACHED

    sink = AtomisticMemoryGenerationSink("segmented")
    state = stream.start(jr.key(ROOT), target=3, attempt_budget=40)
    state = stream.run(state, sink, maximum_transitions=400)
    assert state.stop_reason is AtomisticGenerationStopReason.TARGET_REACHED
    # The quota stop keeps admitted, unharvested work: nothing drawn is lost.
    in_flight = int(np.count_nonzero(np.asarray(state.lanes[0].active)))
    assert in_flight > 0 and int(state.cursor.active) == in_flight
    admitted = int(state.cursor.next_ordinal)
    assert admitted > int(state.cursor.watermark)
    state = stream.extend(state, target=9, attempt_budget=40)
    state = stream.run(state, sink, maximum_transitions=400)
    assert [item.proposal_id for item in sink.records] == [
        item.proposal_id for item in direct.records
    ]
    for left, right in zip(sink.records, direct.records, strict=True):
        np.testing.assert_array_equal(left.positions, right.positions)
    assert sum(item.accepted for item in sink.records) == 9


def test_single_participant_placement_runs_the_same_scheduler() -> None:
    local = stream_plan(lanes_per_member=2).prepare()
    placed = stream_plan(lanes_per_member=2).prepare(placement=_placement())
    expected, local_state = _drain(local, target=7, budget=40)
    observed, placed_state = _drain(placed, target=7, budget=40)
    assert [item.proposal_id for item in observed.records] == [
        item.proposal_id for item in expected.records
    ]
    for left, right in zip(observed.records, expected.records, strict=True):
        np.testing.assert_array_equal(left.positions, right.positions)
        assert left.status == right.status and left.filter_bits == right.filter_bits
    jax.tree.map(
        np.testing.assert_array_equal,
        jax.device_get(placed_state),
        jax.device_get(local_state),
    )
    reference = expected_positions(ROOT, [item.ordinal for item in observed.records])
    for item, positions in zip(observed.records, reference, strict=True):
        np.testing.assert_allclose(item.positions, positions, rtol=0.0, atol=1e-12)
    assert int(placed_state.cursor.receipts) > 0


def test_placement_charges_exchange_and_replicated_bytes_before_work() -> None:
    plan = stream_plan(lanes_per_member=2)
    admitted = plan.prepare(placement=_placement())
    assert admitted.round_bytes > 0 and admitted.replicated_bytes > 0
    with pytest.raises(AtomisticGenerationCapacityError, match="exchange"):
        plan.prepare(placement=_placement(round_bytes=admitted.round_bytes - 1))
    with pytest.raises(AtomisticGenerationCapacityError, match="replicated"):
        plan.prepare(placement=_placement(replicated=admitted.replicated_bytes - 1))


def test_publication_sink_is_hosted_by_the_authority() -> None:
    for stream in (
        stream_plan(lanes_per_member=2).prepare(),
        stream_plan(lanes_per_member=2).prepare(placement=_placement()),
    ):
        state = stream.start(jr.key(ROOT), target=2, attempt_budget=8)
        with pytest.raises(TypeError, match="authority"):
            stream.run(state, None, maximum_transitions=4)


def test_placed_checkpoint_restores_complete_continuation(tmp_path: Path) -> None:
    placement = _placement()
    stream = stream_plan(lanes_per_member=2).prepare(placement=placement)
    reference, reference_state = _drain(stream, target=8, budget=40)

    sink = AtomisticMemoryGenerationSink("memory")
    state = stream.start(jr.key(ROOT), target=8, attempt_budget=40)
    state = stream.run(state, sink, maximum_transitions=4)
    assert int(state.cursor.active) > 0
    repository = _repository(tmp_path)
    manifest = stream.publish_checkpoint(
        repository, state, checkpoint_id="generation-unit", writer_id="unit"
    )
    restored = stream.restore_checkpoint(repository, "generation-unit")
    jax.tree.map(
        np.testing.assert_array_equal, jax.device_get(restored), jax.device_get(state)
    )
    resumed = stream.run(restored, sink, maximum_transitions=400)
    assert [item.proposal_id for item in sink.records] == [
        item.proposal_id for item in reference.records
    ]
    assert int(resumed.cursor.watermark) == int(reference_state.cursor.watermark)
    assert manifest.analysis_plan_id == stream.stream_id

    other = stream_plan(lanes_per_member=3).prepare(placement=placement)
    with pytest.raises(AtomisticGenerationCheckpointError):
        other.restore_checkpoint(repository, "generation-unit")
    with pytest.raises(ValueError, match="publish_checkpoint"):
        stream.write_checkpoint(tmp_path / "local.npz", state)
