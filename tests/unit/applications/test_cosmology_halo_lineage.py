#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.cosmology._halo_finder import MergerMatchResult
from phydrax.applications.cosmology._halo_lineage import (
    HaloLifecycleState,
    ParticleCoreLineagePlan,
)


def _match(descendants, overlaps, merits=None, *, successful=True):
    descendants = jnp.asarray(descendants)
    overlaps = jnp.asarray(overlaps)
    if merits is None:
        merits = overlaps.astype("float64")
    return MergerMatchResult(
        descendants,
        jnp.asarray(merits),
        overlaps,
        overlaps > 0,
        jnp.asarray(successful),
    )


def _tracks_by_source(snapshot):
    return {
        int(source): int(track)
        for source, track, active in zip(
            snapshot.source_halo_ids,
            snapshot.track_ids,
            snapshot.active_mask,
            strict=True,
        )
        if bool(active)
    }


def test_stable_tracks_survive_row_reordering_and_ties_are_deterministic():
    plan = ParticleCoreLineagePlan(2, 3, 2, 16)
    first = plan.build(
        [[20, 10, -1], [200, 100, -1]],
        [[True, True, False], [True, True, False]],
        (_match([0, 1, -1], [2, 2, 0]),),
    )
    reordered = plan.build(
        [[10, 20, -1], [100, 200, -1]],
        [[True, True, False], [True, True, False]],
        (_match([0, 1, -1], [2, 2, 0]),),
    )

    assert _tracks_by_source(first.snapshots[0]) == _tracks_by_source(
        reordered.snapshots[0]
    )
    assert sorted(_tracks_by_source(first.snapshots[1]).values()) == sorted(
        _tracks_by_source(reordered.snapshots[1]).values()
    )

    contested = plan.build(
        [[20, 10, -1], [100, -1, -1]],
        [[True, True, False], [True, False, False]],
        (_match([0, 0, -1], [3, 3, 0], [0.5, 0.5, 0.0]),),
    )
    accepted = np.asarray(contested.tracer_evidence.accepted_mask[0])
    assert accepted.tolist() == [False, True, False]


def test_missing_core_evidence_never_creates_a_false_descendant():
    plan = ParticleCoreLineagePlan(2, 2, 1, 8)
    malformed_claim = MergerMatchResult(
        jnp.asarray([0, -1]),
        jnp.asarray([1.0, 0.0]),
        jnp.asarray([0, 0]),
        jnp.asarray([True, False]),
        jnp.asarray(True),
    )
    result = plan.build(
        [[1, -1], [2, -1]],
        [[True, False], [True, False]],
        (malformed_claim,),
    )

    assert not bool(result.descendant_mask[0, 0])
    assert not bool(result.tracer_evidence.accepted_mask[0, 0])
    assert int(result.snapshots[1].track_ids[0]) != int(result.snapshots[0].track_ids[0])


def test_sink_descendant_and_reresolution_remain_distinct_representable_states():
    plan = ParticleCoreLineagePlan(3, 2, 1, 16)
    result = plan.build(
        [[5, 9], [9, -1], [5, 9]],
        [[True, True], [True, False], [True, True]],
        (
            _match([-1, 0], [0, 2]),
            _match([1, -1], [2, 0]),
        ),
        sink_target_rows=[[0, -1], [-1, -1]],
        reresolved_track_ids=[[-1, -1], [-1, -1], [0, -1]],
        source_membership_ids=(("s5-0", "s9-0"), ("s9-1", ""), ("s5-2", "s9-2")),
        bound_membership_ids=(("b5-0", "b9-0"), ("b9-1", ""), ("b5-2", "b9-2")),
    )

    assert bool(result.sink_mask[0, 0])
    assert not bool(result.descendant_mask[0, 0])
    assert int(result.snapshots[2].lifecycle_states[0]) == int(
        HaloLifecycleState.RERESOLVED
    )
    assert int(result.snapshots[2].track_ids[0]) == int(result.snapshots[0].track_ids[0])
    assert (
        result.snapshots[2].source_membership_ids[0]
        != result.snapshots[2].bound_membership_ids[0]
    )


def test_duplicate_source_ids_and_event_overflow_are_rejected():
    plan = ParticleCoreLineagePlan(2, 2, 1, 1)
    with pytest.raises(ValueError, match="unique"):
        plan.build(
            [[1, 1], [2, -1]],
            [[True, True], [True, False]],
            (_match([0, -1], [1, 0]),),
        )
    with pytest.raises(ValueError, match="event_capacity"):
        plan.build(
            [[1, -1], [2, -1]],
            [[True, False], [True, False]],
            (_match([0, -1], [1, 0]),),
        )
