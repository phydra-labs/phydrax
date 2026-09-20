#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded longitudinal halo identities and particle-core lineage evidence."""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._halo_finder import MergerMatchResult


class HaloLifecycleState(IntEnum):
    """State of one bounded lineage slot at one output."""

    UNUSED = 0
    RESOLVED = 1
    DISRUPTED = 2
    ORPHAN = 3
    RERESOLVED = 4
    UNKNOWN = 5


class HaloLineageEventKind(IntEnum):
    """Physical and numerical ledger entries; failures are not disruptions."""

    UNUSED = 0
    BIRTH = 1
    DESCENDANT = 2
    DISRUPTION = 3
    ORPHANED = 4
    RERESOLUTION = 5
    SINK = 6
    MATCH_FAILURE = 7


def _array(value: ArrayLike, *, dtype=None) -> Array:
    return jax.lax.stop_gradient(jnp.asarray(value, dtype=dtype))


def _identity_tuple(
    values: Sequence[str] | None, capacity: int, name: str, /
) -> tuple[str, ...]:
    if values is None:
        return ("",) * capacity
    result = tuple(str(value).strip() for value in values)
    if len(result) != capacity:
        raise ValueError(f"{name} must have one entry per fixed-capacity row.")
    return result


def _identity_grid(
    values: Sequence[Sequence[str]] | None,
    snapshots: int,
    capacity: int,
    name: str,
    /,
) -> tuple[tuple[str, ...], ...]:
    if values is None:
        return ((("",) * capacity),) * snapshots
    result = tuple(_identity_tuple(row, capacity, name) for row in values)
    if len(result) != snapshots:
        raise ValueError(f"{name} must have one row per fixed-capacity snapshot.")
    return result


class HaloTrackSnapshot(StrictModule, NonTrainableState):
    """One fixed-capacity mapping from producer rows to persistent track IDs.

    ``source_membership_ids`` and ``bound_membership_ids`` are deliberately
    distinct content identities. An empty identity means that producer did not
    supply that membership, never that the two populations were equal.
    """

    snapshot_index: Array
    source_row_indices: Array
    source_halo_ids: Array
    track_ids: Array
    host_track_ids: Array
    lifecycle_states: Array
    active_mask: Array
    source_membership_known: Array
    bound_membership_known: Array
    source_membership_ids: tuple[str, ...] = eqx.field(static=True)
    bound_membership_ids: tuple[str, ...] = eqx.field(static=True)
    snapshot_id: str = eqx.field(static=True)

    def __init__(
        self,
        snapshot_index: ArrayLike,
        source_row_indices: ArrayLike,
        source_halo_ids: ArrayLike,
        track_ids: ArrayLike,
        host_track_ids: ArrayLike,
        lifecycle_states: ArrayLike,
        active_mask: ArrayLike,
        /,
        *,
        source_membership_ids: Sequence[str] | None = None,
        bound_membership_ids: Sequence[str] | None = None,
    ):
        rows = _array(source_row_indices, dtype=jnp.int64)
        source_ids = _array(source_halo_ids, dtype=jnp.int64)
        tracks = _array(track_ids, dtype=jnp.int64)
        hosts = _array(host_track_ids, dtype=jnp.int64)
        states = _array(lifecycle_states, dtype=jnp.int8)
        active = _array(active_mask, dtype=jnp.bool_)
        snapshot = _array(snapshot_index, dtype=jnp.int32)
        if (
            rows.ndim != 1
            or source_ids.shape != rows.shape
            or tracks.shape != rows.shape
            or hosts.shape != rows.shape
            or states.shape != rows.shape
            or active.shape != rows.shape
            or snapshot.shape != ()
        ):
            raise ValueError("Halo track snapshot arrays have inconsistent shapes.")
        source_memberships = _identity_tuple(
            source_membership_ids, rows.size, "source_membership_ids"
        )
        bound_memberships = _identity_tuple(
            bound_membership_ids, rows.size, "bound_membership_ids"
        )
        host_rows = np.asarray(hosts)
        track_rows = np.asarray(tracks)
        active_rows = np.asarray(active)
        state_rows = np.asarray(states)
        source_rows = np.asarray(rows)
        source_halos = np.asarray(source_ids)
        valid_states = {int(value) for value in HaloLifecycleState}
        if any(int(value) not in valid_states for value in state_rows):
            raise ValueError("Unknown halo lifecycle state.")
        if np.any(active_rows & (state_rows == int(HaloLifecycleState.UNUSED))) or np.any(
            (~active_rows) & (state_rows != int(HaloLifecycleState.UNUSED))
        ):
            raise ValueError(
                "Halo lifecycle UNUSED state must exactly match inactive rows."
            )
        if np.any(active_rows & (track_rows < 0)):
            raise ValueError("Active lineage rows require non-negative track IDs.")
        if len(set(track_rows[active_rows].tolist())) != int(np.sum(active_rows)):
            raise ValueError("Active track IDs must be unique within a snapshot.")
        resolved = active_rows & np.isin(
            state_rows,
            (int(HaloLifecycleState.RESOLVED), int(HaloLifecycleState.RERESOLVED)),
        )
        if np.any(resolved & ((source_rows < 0) | (source_halos < 0))):
            raise ValueError("Resolved lineage rows require a producer row and halo ID.")
        if len(set(source_halos[resolved].tolist())) != int(np.sum(resolved)):
            raise ValueError("Resolved source halo IDs must be unique within a snapshot.")
        present_tracks = set(track_rows[active_rows].tolist())
        if any(
            value >= 0 and value not in present_tracks for value in host_rows[active_rows]
        ):
            raise ValueError(
                "Host track IDs must refer to an active row in the same snapshot."
            )
        parents = {
            int(track): int(host)
            for track, host, is_active in zip(
                track_rows, host_rows, active_rows, strict=True
            )
            if is_active and host >= 0
        }
        for track in parents:
            visited: set[int] = set()
            cursor = track
            while cursor in parents:
                if cursor in visited:
                    raise ValueError("Host track hierarchy must be acyclic.")
                visited.add(cursor)
                cursor = parents[cursor]
        if np.any((~active_rows) & ((track_rows >= 0) | (host_rows >= 0))):
            raise ValueError("Unused lineage rows cannot carry track or host identities.")
        source_known = np.asarray(tuple(bool(value) for value in source_memberships))
        bound_known = np.asarray(tuple(bool(value) for value in bound_memberships))
        if np.any((~active_rows) & (source_known | bound_known)):
            raise ValueError("Unused rows cannot carry membership identities.")
        self.snapshot_index = snapshot
        self.source_row_indices = rows
        self.source_halo_ids = source_ids
        self.track_ids = tracks
        self.host_track_ids = hosts
        self.lifecycle_states = states
        self.active_mask = active
        self.source_membership_known = _array(source_known, dtype=jnp.bool_)
        self.bound_membership_known = _array(bound_known, dtype=jnp.bool_)
        self.source_membership_ids = source_memberships
        self.bound_membership_ids = bound_memberships
        self.snapshot_id = canonical_fingerprint(
            {
                "kind": "halo-track-snapshot",
                "snapshot_index": int(np.asarray(snapshot)),
                "arrays": array_tree_fingerprint(
                    (rows, source_ids, tracks, hosts, states, active)
                ),
                "source_memberships": list(source_memberships),
                "bound_memberships": list(bound_memberships),
            }
        )


class HaloLineageEventLedger(StrictModule, NonTrainableState):
    """Fixed-capacity lifecycle events with explicit related-track semantics."""

    event_kinds: Array
    snapshot_indices: Array
    track_ids: Array
    related_track_ids: Array
    active_mask: Array
    successful: Array
    ledger_id: str = eqx.field(static=True)

    def __init__(
        self,
        event_kinds: ArrayLike,
        snapshot_indices: ArrayLike,
        track_ids: ArrayLike,
        related_track_ids: ArrayLike,
        active_mask: ArrayLike,
        successful: ArrayLike,
        /,
    ):
        kinds = _array(event_kinds, dtype=jnp.int8)
        snapshots = _array(snapshot_indices, dtype=jnp.int32)
        tracks = _array(track_ids, dtype=jnp.int64)
        related = _array(related_track_ids, dtype=jnp.int64)
        active = _array(active_mask, dtype=jnp.bool_)
        success = _array(successful, dtype=jnp.bool_)
        if not (
            kinds.ndim == 1
            and snapshots.shape == kinds.shape
            and tracks.shape == kinds.shape
            and related.shape == kinds.shape
            and active.shape == kinds.shape
            and success.shape == kinds.shape
        ):
            raise ValueError("Halo event ledger arrays have inconsistent capacities.")
        kinds_host = np.asarray(kinds)
        active_host = np.asarray(active)
        if any(
            int(value) not in {int(item) for item in HaloLineageEventKind}
            for value in kinds_host
        ):
            raise ValueError("Unknown halo lineage event kind.")
        if np.any(active_host & (np.asarray(tracks) < 0)):
            raise ValueError("Active lineage events require a track ID.")
        if np.any((~active_host) & (kinds_host != int(HaloLineageEventKind.UNUSED))):
            raise ValueError("Inactive ledger entries must use the UNUSED event kind.")
        self.event_kinds = kinds
        self.snapshot_indices = snapshots
        self.track_ids = tracks
        self.related_track_ids = related
        self.active_mask = active
        self.successful = success
        self.ledger_id = canonical_fingerprint(
            {
                "kind": "halo-lineage-event-ledger",
                "arrays": array_tree_fingerprint(
                    (kinds, snapshots, tracks, related, active, success)
                ),
            }
        )


class HaloTracerEvidence(StrictModule, NonTrainableState):
    """Ranked particle-core evidence and deterministic acceptance decisions."""

    particle_ids: Array
    binding_ranks: Array
    present_mask: Array
    candidate_target_rows: Array
    accepted_target_rows: Array
    overlap_counts: Array
    merits: Array
    matched_mask: Array
    accepted_mask: Array
    successful: Array
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        particle_ids: ArrayLike,
        binding_ranks: ArrayLike,
        present_mask: ArrayLike,
        candidate_target_rows: ArrayLike,
        accepted_target_rows: ArrayLike,
        overlap_counts: ArrayLike,
        merits: ArrayLike,
        matched_mask: ArrayLike,
        accepted_mask: ArrayLike,
        successful: ArrayLike,
        /,
    ):
        particles = _array(particle_ids, dtype=jnp.int64)
        ranks = _array(binding_ranks, dtype=jnp.int32)
        present = _array(present_mask, dtype=jnp.bool_)
        candidates = _array(candidate_target_rows, dtype=jnp.int32)
        accepted_rows = _array(accepted_target_rows, dtype=jnp.int32)
        overlaps = _array(overlap_counts, dtype=jnp.int32)
        merit = _array(merits)
        matched = _array(matched_mask, dtype=jnp.bool_)
        accepted = _array(accepted_mask, dtype=jnp.bool_)
        success = _array(successful, dtype=jnp.bool_)
        if (
            particles.ndim != 3
            or ranks.shape != particles.shape
            or present.shape != particles.shape
            or candidates.shape != (max(particles.shape[0] - 1, 0), particles.shape[1])
            or accepted_rows.shape != candidates.shape
            or overlaps.shape != candidates.shape
            or merit.shape != candidates.shape
            or matched.shape != candidates.shape
            or accepted.shape != candidates.shape
            or success.shape != (max(particles.shape[0] - 1, 0),)
        ):
            raise ValueError("Halo tracer evidence arrays have inconsistent shapes.")
        if np.any(np.asarray(present) & (np.asarray(particles) < 0)):
            raise ValueError("Present tracers require non-negative particle IDs.")
        if np.any(np.asarray(accepted) & (np.asarray(overlaps) <= 0)):
            raise ValueError("A descendant cannot be accepted without core overlap.")
        self.particle_ids = particles
        self.binding_ranks = ranks
        self.present_mask = present
        self.candidate_target_rows = candidates
        self.accepted_target_rows = accepted_rows
        self.overlap_counts = overlaps
        self.merits = merit
        self.matched_mask = matched
        self.accepted_mask = accepted
        self.successful = success
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "halo-tracer-evidence",
                "arrays": array_tree_fingerprint(
                    (
                        particles,
                        ranks,
                        present,
                        candidates,
                        accepted_rows,
                        overlaps,
                        merit,
                        matched,
                        accepted,
                        success,
                    )
                ),
            }
        )


class HaloLineageProduct(StrictModule, NonTrainableState):
    """Longitudinal tracks with separate descendant and physical-sink edges."""

    snapshots: tuple[HaloTrackSnapshot, ...]
    events: HaloLineageEventLedger
    tracer_evidence: HaloTracerEvidence
    descendant_track_ids: Array
    descendant_mask: Array
    sink_track_ids: Array
    sink_mask: Array
    successful: Array
    lineage_id: str = eqx.field(static=True)

    def __init__(
        self,
        snapshots: Sequence[HaloTrackSnapshot],
        events: HaloLineageEventLedger,
        tracer_evidence: HaloTracerEvidence,
        descendant_track_ids: ArrayLike,
        descendant_mask: ArrayLike,
        sink_track_ids: ArrayLike,
        sink_mask: ArrayLike,
        successful: ArrayLike,
        /,
    ):
        snapshots_ = tuple(snapshots)
        if not snapshots_ or not all(
            isinstance(item, HaloTrackSnapshot) for item in snapshots_
        ):
            raise ValueError("A halo lineage requires one or more track snapshots.")
        capacity = snapshots_[0].track_ids.size
        if any(item.track_ids.size != capacity for item in snapshots_):
            raise ValueError("All halo snapshots must use the same fixed capacity.")
        descendants = _array(descendant_track_ids, dtype=jnp.int64)
        descendants_present = _array(descendant_mask, dtype=jnp.bool_)
        sinks = _array(sink_track_ids, dtype=jnp.int64)
        sinks_present = _array(sink_mask, dtype=jnp.bool_)
        success = _array(successful, dtype=jnp.bool_)
        expected = (len(snapshots_), capacity)
        if (
            descendants.shape != expected
            or descendants_present.shape != expected
            or sinks.shape != expected
            or sinks_present.shape != expected
            or success.shape != ()
            or tracer_evidence.particle_ids.shape[:2] != (len(snapshots_), capacity)
            or tracer_evidence.candidate_target_rows.shape
            != (max(len(snapshots_) - 1, 0), capacity)
        ):
            raise ValueError("Halo lineage edge/evidence capacities are inconsistent.")
        descendant_host = np.asarray(descendants)
        sink_host = np.asarray(sinks)
        descendant_present_host = np.asarray(descendants_present)
        sink_present_host = np.asarray(sinks_present)
        if np.any(descendant_present_host & (descendant_host < 0)) or np.any(
            sink_present_host & (sink_host < 0)
        ):
            raise ValueError("Present halo edges require non-negative target track IDs.")
        self.snapshots = snapshots_
        self.events = events
        self.tracer_evidence = tracer_evidence
        self.descendant_track_ids = descendants
        self.descendant_mask = descendants_present
        self.sink_track_ids = sinks
        self.sink_mask = sinks_present
        self.successful = success
        self.lineage_id = canonical_fingerprint(
            {
                "kind": "halo-lineage-product",
                "snapshots": [item.snapshot_id for item in snapshots_],
                "events": events.ledger_id,
                "tracers": tracer_evidence.evidence_id,
                "edges": array_tree_fingerprint(
                    (descendants, descendants_present, sinks, sinks_present)
                ),
            }
        )


class ParticleCoreLineagePlan(StrictModule, NonTrainableState):
    """Bounded deterministic assembly from adjacent particle-core matches."""

    snapshot_capacity: int = eqx.field(static=True)
    halo_capacity: int = eqx.field(static=True)
    tracer_capacity: int = eqx.field(static=True)
    event_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        snapshot_capacity: int,
        halo_capacity: int,
        tracer_capacity: int,
        event_capacity: int,
        /,
    ):
        values = tuple(
            (
                snapshot_capacity,
                halo_capacity,
                tracer_capacity,
                event_capacity,
            )
        )
        if any(value <= 0 for value in values) or values[0] < 2:
            raise ValueError(
                "Lineage capacities must be positive with at least two snapshots."
            )
        (
            self.snapshot_capacity,
            self.halo_capacity,
            self.tracer_capacity,
            self.event_capacity,
        ) = values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "particle-core-lineage-plan",
                "snapshot_capacity": values[0],
                "halo_capacity": values[1],
                "tracer_capacity": values[2],
                "event_capacity": values[3],
                "tie_policy": "overlap-desc-merit-desc-track-asc",
            }
        )

    def build(
        self,
        source_halo_ids: ArrayLike,
        active_mask: ArrayLike,
        matches: Sequence[MergerMatchResult],
        /,
        *,
        host_rows: ArrayLike | None = None,
        sink_target_rows: ArrayLike | None = None,
        tracer_particle_ids: ArrayLike | None = None,
        tracer_binding_ranks: ArrayLike | None = None,
        tracer_present_mask: ArrayLike | None = None,
        reresolved_track_ids: ArrayLike | None = None,
        source_membership_ids: Sequence[Sequence[str]] | None = None,
        bound_membership_ids: Sequence[Sequence[str]] | None = None,
    ) -> HaloLineageProduct:
        ids = np.asarray(source_halo_ids, dtype=np.int64)
        active = np.asarray(active_mask, dtype=np.bool_)
        shape = (self.snapshot_capacity, self.halo_capacity)
        transition_shape = (self.snapshot_capacity - 1, self.halo_capacity)
        if ids.shape != shape or active.shape != shape:
            raise ValueError("Source halo IDs and activity must match plan capacities.")
        if len(matches) != self.snapshot_capacity - 1 or not all(
            isinstance(item, MergerMatchResult) for item in matches
        ):
            raise ValueError("Exactly one MergerMatchResult is required per transition.")
        for snapshot in range(self.snapshot_capacity):
            active_ids = ids[snapshot, active[snapshot]]
            if np.any(active_ids < 0) or len(set(active_ids.tolist())) != active_ids.size:
                raise ValueError(
                    "Active producer halo IDs must be non-negative and unique."
                )
        source_memberships = _identity_grid(
            source_membership_ids,
            self.snapshot_capacity,
            self.halo_capacity,
            "source_membership_ids",
        )
        bound_memberships = _identity_grid(
            bound_membership_ids,
            self.snapshot_capacity,
            self.halo_capacity,
            "bound_membership_ids",
        )
        hosts = (
            np.full(shape, -1, dtype=np.int64)
            if host_rows is None
            else np.asarray(host_rows, dtype=np.int64)
        )
        sinks = (
            np.full(transition_shape, -1, dtype=np.int64)
            if sink_target_rows is None
            else np.asarray(sink_target_rows, dtype=np.int64)
        )
        reresolved = (
            np.full(shape, -1, dtype=np.int64)
            if reresolved_track_ids is None
            else np.asarray(reresolved_track_ids, dtype=np.int64)
        )
        tracer_shape = (*shape, self.tracer_capacity)
        tracer_ids = (
            np.full(tracer_shape, -1, dtype=np.int64)
            if tracer_particle_ids is None
            else np.asarray(tracer_particle_ids, dtype=np.int64)
        )
        tracer_ranks = (
            np.full(tracer_shape, -1, dtype=np.int32)
            if tracer_binding_ranks is None
            else np.asarray(tracer_binding_ranks, dtype=np.int32)
        )
        tracer_present = (
            tracer_ids >= 0
            if tracer_present_mask is None
            else np.asarray(tracer_present_mask, dtype=np.bool_)
        )
        if (
            hosts.shape != shape
            or sinks.shape != transition_shape
            or reresolved.shape != shape
            or tracer_ids.shape != tracer_shape
            or tracer_ranks.shape != tracer_shape
            or tracer_present.shape != tracer_shape
        ):
            raise ValueError("Optional lineage inputs must match their fixed capacities.")
        if np.any(tracer_present & (tracer_ids < 0)):
            raise ValueError("Present core tracers require non-negative particle IDs.")

        track_ids = np.full(shape, -1, dtype=np.int64)
        states = np.zeros(shape, dtype=np.int8)
        descendants = np.full(shape, -1, dtype=np.int64)
        descendant_mask = np.zeros(shape, dtype=np.bool_)
        sink_tracks = np.full(shape, -1, dtype=np.int64)
        sink_mask = np.zeros(shape, dtype=np.bool_)
        candidate_rows = np.full(transition_shape, -1, dtype=np.int32)
        accepted_rows = np.full(transition_shape, -1, dtype=np.int32)
        overlaps = np.zeros(transition_shape, dtype=np.int32)
        merits = np.zeros(transition_shape, dtype=np.float64)
        matched_mask = np.zeros(transition_shape, dtype=np.bool_)
        accepted_mask = np.zeros(transition_shape, dtype=np.bool_)
        transition_success = np.ones(self.snapshot_capacity - 1, dtype=np.bool_)
        events: list[tuple[int, int, int, int, bool]] = []

        first_rows = np.flatnonzero(active[0])
        for track, row in enumerate(
            first_rows[np.argsort(ids[0, first_rows], kind="stable")]
        ):
            track_ids[0, row] = track
            states[0, row] = int(HaloLifecycleState.RESOLVED)
            events.append((int(HaloLineageEventKind.BIRTH), 0, track, -1, True))
        next_track = first_rows.size

        for transition, match in enumerate(matches):
            raw_descendant = np.asarray(match.descendant_indices, dtype=np.int64)
            raw_merits = np.asarray(match.merits, dtype=np.float64)
            raw_overlaps = np.asarray(match.overlap_counts, dtype=np.int32)
            raw_matched = np.asarray(match.matched, dtype=np.bool_)
            successful = bool(np.asarray(match.successful))
            if not (
                raw_descendant.shape == (self.halo_capacity,)
                and raw_merits.shape == raw_descendant.shape
                and raw_overlaps.shape == raw_descendant.shape
                and raw_matched.shape == raw_descendant.shape
            ):
                raise ValueError("Merger match arrays must match halo_capacity.")
            candidate_rows[transition] = raw_descendant
            overlaps[transition] = raw_overlaps
            merits[transition] = raw_merits
            matched_mask[transition] = raw_matched
            transition_success[transition] = successful and np.all(
                np.isfinite(raw_merits)
            )
            claims: dict[int, list[int]] = {}
            if transition_success[transition]:
                for source_row in np.flatnonzero(active[transition]):
                    target_row = int(raw_descendant[source_row])
                    credible = (
                        raw_matched[source_row]
                        and raw_overlaps[source_row] > 0
                        and 0 <= target_row < self.halo_capacity
                        and active[transition + 1, target_row]
                    )
                    if credible:
                        claims.setdefault(target_row, []).append(int(source_row))
            winners: dict[int, int] = {}
            for target_row, source_rows in claims.items():
                winners[target_row] = min(
                    source_rows,
                    key=lambda row: (
                        -int(raw_overlaps[row]),
                        -float(raw_merits[row]),
                        int(track_ids[transition, row]),
                    ),
                )
            for target_row, source_row in winners.items():
                track = int(track_ids[transition, source_row])
                track_ids[transition + 1, target_row] = track
                states[transition + 1, target_row] = int(HaloLifecycleState.RESOLVED)
                descendants[transition, source_row] = track
                descendant_mask[transition, source_row] = True
                accepted_rows[transition, source_row] = target_row
                accepted_mask[transition, source_row] = True
                events.append(
                    (
                        int(HaloLineageEventKind.DESCENDANT),
                        transition + 1,
                        track,
                        track,
                        True,
                    )
                )

            known_prior = set(
                track_ids[: transition + 1][track_ids[: transition + 1] >= 0].tolist()
            )
            for target_row in np.flatnonzero(active[transition + 1]):
                if track_ids[transition + 1, target_row] >= 0:
                    continue
                requested_track = int(reresolved[transition + 1, target_row])
                if requested_track >= 0:
                    if requested_track not in known_prior or requested_track in set(
                        track_ids[transition + 1][track_ids[transition + 1] >= 0].tolist()
                    ):
                        raise ValueError(
                            "Re-resolution must identify one absent prior track."
                        )
                    track_ids[transition + 1, target_row] = requested_track
                    states[transition + 1, target_row] = int(
                        HaloLifecycleState.RERESOLVED
                    )
                    events.append(
                        (
                            int(HaloLineageEventKind.RERESOLUTION),
                            transition + 1,
                            requested_track,
                            -1,
                            True,
                        )
                    )
            new_rows = np.asarray(
                [
                    row
                    for row in np.flatnonzero(active[transition + 1])
                    if track_ids[transition + 1, row] < 0
                ],
                dtype=np.int64,
            )
            if new_rows.size:
                for target_row in new_rows[
                    np.argsort(ids[transition + 1, new_rows], kind="stable")
                ]:
                    track_ids[transition + 1, target_row] = next_track
                    states[transition + 1, target_row] = int(HaloLifecycleState.RESOLVED)
                    if transition_success[transition]:
                        events.append(
                            (
                                int(HaloLineageEventKind.BIRTH),
                                transition + 1,
                                next_track,
                                -1,
                                True,
                            )
                        )
                    next_track += 1

            for source_row in np.flatnonzero(active[transition]):
                source_track = int(track_ids[transition, source_row])
                sink_row = int(sinks[transition, source_row])
                if sink_row >= 0:
                    if (
                        sink_row >= self.halo_capacity
                        or not active[transition + 1, sink_row]
                    ):
                        raise ValueError(
                            "Sink target rows must refer to an active target halo."
                        )
                    sink_track = int(track_ids[transition + 1, sink_row])
                    sink_tracks[transition, source_row] = sink_track
                    sink_mask[transition, source_row] = True
                    events.append(
                        (
                            int(HaloLineageEventKind.SINK),
                            transition + 1,
                            source_track,
                            sink_track,
                            True,
                        )
                    )
                if not descendant_mask[transition, source_row]:
                    kind = (
                        HaloLineageEventKind.DISRUPTION
                        if transition_success[transition]
                        else HaloLineageEventKind.MATCH_FAILURE
                    )
                    events.append(
                        (
                            int(kind),
                            transition + 1,
                            source_track,
                            -1,
                            transition_success[transition],
                        )
                    )

        host_tracks = np.full(shape, -1, dtype=np.int64)
        for snapshot in range(self.snapshot_capacity):
            for row in np.flatnonzero(active[snapshot]):
                host_row = int(hosts[snapshot, row])
                if host_row >= 0:
                    if host_row >= self.halo_capacity or not active[snapshot, host_row]:
                        raise ValueError(
                            "Host rows must refer to an active row in the same snapshot."
                        )
                    host_tracks[snapshot, row] = track_ids[snapshot, host_row]

        if len(events) > self.event_capacity:
            raise ValueError(
                f"Lineage requires {len(events)} events, exceeding event_capacity={self.event_capacity}."
            )
        event_kinds = np.zeros(self.event_capacity, dtype=np.int8)
        event_snapshots = np.full(self.event_capacity, -1, dtype=np.int32)
        event_tracks = np.full(self.event_capacity, -1, dtype=np.int64)
        event_related = np.full(self.event_capacity, -1, dtype=np.int64)
        event_active = np.zeros(self.event_capacity, dtype=np.bool_)
        event_success = np.zeros(self.event_capacity, dtype=np.bool_)
        for index, (kind, snapshot, track, related, success) in enumerate(events):
            event_kinds[index] = kind
            event_snapshots[index] = snapshot
            event_tracks[index] = track
            event_related[index] = related
            event_active[index] = True
            event_success[index] = success

        snapshots: list[HaloTrackSnapshot] = []
        for snapshot in range(self.snapshot_capacity):
            source_membership_row = source_memberships[snapshot]
            bound_membership_row = bound_memberships[snapshot]
            snapshots.append(
                HaloTrackSnapshot(
                    snapshot,
                    np.where(active[snapshot], np.arange(self.halo_capacity), -1),
                    np.where(active[snapshot], ids[snapshot], -1),
                    track_ids[snapshot],
                    host_tracks[snapshot],
                    states[snapshot],
                    active[snapshot],
                    source_membership_ids=source_membership_row,
                    bound_membership_ids=bound_membership_row,
                )
            )
        ledger = HaloLineageEventLedger(
            event_kinds,
            event_snapshots,
            event_tracks,
            event_related,
            event_active,
            event_success,
        )
        evidence = HaloTracerEvidence(
            tracer_ids,
            tracer_ranks,
            tracer_present,
            candidate_rows,
            accepted_rows,
            overlaps,
            merits,
            matched_mask,
            accepted_mask,
            transition_success,
        )
        return HaloLineageProduct(
            snapshots,
            ledger,
            evidence,
            descendants,
            descendant_mask,
            sink_tracks,
            sink_mask,
            np.all(transition_success),
        )


__all__ = [
    "HaloLifecycleState",
    "HaloLineageEventKind",
    "HaloLineageEventLedger",
    "HaloLineageProduct",
    "HaloTracerEvidence",
    "HaloTrackSnapshot",
    "ParticleCoreLineagePlan",
]
