#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Immutable QCD ensemble schedules, segments, and merge evidence."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


MergeStatus: TypeAlias = Literal["complete", "incomplete"]


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


class MeasurementSchedule(StrictModule, NonTrainableState):
    """Prespecified half-open trajectory schedule with exact burn-in exclusion."""

    trajectory_indices: tuple[int, ...] = eqx.field(static=True)
    total_trajectories: int = eqx.field(static=True)
    thermalization_trajectories: int = eqx.field(static=True)
    measurement_interval: int = eqx.field(static=True)
    sources_per_configuration: int = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        total_trajectories: int,
        /,
        *,
        thermalization_trajectories: int,
        measurement_interval: int,
        sources_per_configuration: int = 1,
        maximum_measurements: int = 1 << 22,
    ):
        total = int(total_trajectories)
        thermalization = int(thermalization_trajectories)
        interval = int(measurement_interval)
        sources = int(sources_per_configuration)
        maximum = int(maximum_measurements)
        if total <= 0:
            raise ValueError("total_trajectories must be positive.")
        if thermalization < 0 or thermalization >= total:
            raise ValueError(
                "thermalization_trajectories must lie in [0, total_trajectories)."
            )
        if interval <= 0 or sources <= 0:
            raise ValueError(
                "measurement_interval and sources_per_configuration must be positive."
            )
        indices = tuple(range(thermalization, total, interval))
        if maximum <= 0 or len(indices) * sources > maximum:
            raise ValueError("Measurement schedule exceeds maximum_measurements.")
        self.trajectory_indices = indices
        self.total_trajectories = total
        self.thermalization_trajectories = thermalization
        self.measurement_interval = interval
        self.sources_per_configuration = sources
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "qcd-measurement-schedule",
                "trajectory_domain": [0, total],
                "thermalization_excluded": [0, thermalization],
                "trajectory_indices": indices,
                "measurement_interval": interval,
                "sources_per_configuration": sources,
            }
        )

    @property
    def measurement_count(self) -> int:
        return len(self.trajectory_indices) * self.sources_per_configuration


class EnsembleManifest(StrictModule, NonTrainableState):
    """Identity binding for one Markov chain and independent update/measurement RNGs."""

    schedule: MeasurementSchedule
    ensemble_id: str = eqx.field(static=True)
    recipe_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    field_space_id: str = eqx.field(static=True)
    chain_id: str = eqx.field(static=True)
    update_randomness_id: str = eqx.field(static=True)
    measurement_randomness_id: str = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)

    def __init__(
        self,
        schedule: MeasurementSchedule,
        /,
        *,
        ensemble_id: str,
        recipe_id: str,
        topology_id: str,
        field_space_id: str,
        chain_id: str,
        update_randomness_id: str,
        measurement_randomness_id: str,
    ):
        if not isinstance(schedule, MeasurementSchedule):
            raise TypeError("schedule must be MeasurementSchedule.")
        identifiers = tuple(
            _identifier(value, name)
            for value, name in (
                (ensemble_id, "ensemble_id"),
                (recipe_id, "recipe_id"),
                (topology_id, "topology_id"),
                (field_space_id, "field_space_id"),
                (chain_id, "chain_id"),
                (update_randomness_id, "update_randomness_id"),
                (measurement_randomness_id, "measurement_randomness_id"),
            )
        )
        if identifiers[5] == identifiers[6]:
            raise ValueError(
                "Update and measurement randomness identities must be distinct."
            )
        self.schedule = schedule
        (
            self.ensemble_id,
            self.recipe_id,
            self.topology_id,
            self.field_space_id,
            self.chain_id,
            self.update_randomness_id,
            self.measurement_randomness_id,
        ) = identifiers
        self.manifest_id = canonical_fingerprint(
            {
                "kind": "qcd-ensemble-manifest",
                "ensemble": identifiers[0],
                "recipe": identifiers[1],
                "topology": identifiers[2],
                "field_space": identifiers[3],
                "chain": identifiers[4],
                "schedule": schedule.schedule_id,
                "update_randomness": identifiers[5],
                "measurement_randomness": identifiers[6],
            }
        )


class MeasurementWorkItem(StrictModule, NonTrainableState):
    """One configuration/source address with disjoint Markov and measurement RNG IDs."""

    trajectory_index: int = eqx.field(static=True)
    source_index: int = eqx.field(static=True)
    configuration_id: str = eqx.field(static=True)
    update_randomness_id: str = eqx.field(static=True)
    source_randomness_id: str = eqx.field(static=True)
    measurement_id: str = eqx.field(static=True)


def measurement_work_items(
    manifest: EnsembleManifest,
    /,
) -> tuple[MeasurementWorkItem, ...]:
    """Materialize the fixed semantic workset without consuming PRNG keys."""
    if not isinstance(manifest, EnsembleManifest):
        raise TypeError("manifest must be EnsembleManifest.")
    items: list[MeasurementWorkItem] = []
    for trajectory in manifest.schedule.trajectory_indices:
        configuration_id = canonical_fingerprint(
            {
                "kind": "qcd-configuration-address",
                "manifest": manifest.manifest_id,
                "trajectory": trajectory,
            }
        )
        update_id = canonical_fingerprint(
            {
                "kind": "qcd-update-randomness-address",
                "root": manifest.update_randomness_id,
                "chain": manifest.chain_id,
                "trajectory": trajectory,
            }
        )
        for source in range(manifest.schedule.sources_per_configuration):
            source_id = canonical_fingerprint(
                {
                    "kind": "qcd-measurement-randomness-address",
                    "root": manifest.measurement_randomness_id,
                    "chain": manifest.chain_id,
                    "trajectory": trajectory,
                    "source": source,
                }
            )
            if source_id == update_id:
                raise ValueError("Measurement and update randomness addresses collide.")
            measurement_id = canonical_fingerprint(
                {
                    "kind": "qcd-measurement-work-item",
                    "manifest": manifest.manifest_id,
                    "configuration": configuration_id,
                    "trajectory": trajectory,
                    "source": source,
                    "source_randomness": source_id,
                }
            )
            items.append(
                MeasurementWorkItem(
                    trajectory_index=trajectory,
                    source_index=source,
                    configuration_id=configuration_id,
                    update_randomness_id=update_id,
                    source_randomness_id=source_id,
                    measurement_id=measurement_id,
                )
            )
    return tuple(items)


class EnsembleSegment(StrictModule, NonTrainableState):
    """One half-open trajectory segment and its content-addressed configurations."""

    trajectory_indices: Array
    configuration_ids: tuple[str, ...] = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)
    start_trajectory: int = eqx.field(static=True)
    stop_trajectory: int = eqx.field(static=True)
    initial_checkpoint_id: str = eqx.field(static=True)
    terminal_checkpoint_id: str = eqx.field(static=True)
    segment_id: str = eqx.field(static=True)

    def __init__(
        self,
        manifest_id: str,
        start_trajectory: int,
        stop_trajectory: int,
        trajectory_indices: ArrayLike,
        configuration_ids: Sequence[str],
        /,
        *,
        initial_checkpoint_id: str,
        terminal_checkpoint_id: str,
    ):
        manifest = _identifier(manifest_id, "manifest_id")
        initial = _identifier(initial_checkpoint_id, "initial_checkpoint_id")
        terminal = _identifier(terminal_checkpoint_id, "terminal_checkpoint_id")
        start, stop = int(start_trajectory), int(stop_trajectory)
        indices = np.asarray(trajectory_indices, dtype=np.int64).reshape((-1,))
        identifiers = tuple(
            _identifier(value, "configuration_id") for value in configuration_ids
        )
        if start < 0 or stop <= start:
            raise ValueError(
                "Segment trajectory bounds must be nonnegative and increasing."
            )
        if indices.size != len(identifiers):
            raise ValueError(
                "trajectory_indices and configuration_ids must have equal length."
            )
        if indices.size and (
            np.any(indices < start)
            or np.any(indices >= stop)
            or np.any(np.diff(indices) <= 0)
        ):
            raise ValueError(
                "Segment configuration trajectories must be unique, increasing, and in range."
            )
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Segment configuration IDs must be unique.")
        if initial == terminal:
            raise ValueError("A nonempty trajectory segment must advance its checkpoint.")
        segment_id = canonical_fingerprint(
            {
                "kind": "qcd-ensemble-segment",
                "manifest": manifest,
                "trajectory_range": [start, stop],
                "trajectory_indices": indices,
                "configuration_ids": identifiers,
                "initial_checkpoint": initial,
                "terminal_checkpoint": terminal,
            }
        )
        self.trajectory_indices = jnp.asarray(indices)
        self.configuration_ids = identifiers
        self.manifest_id = manifest
        self.start_trajectory = start
        self.stop_trajectory = stop
        self.initial_checkpoint_id = initial
        self.terminal_checkpoint_id = terminal
        self.segment_id = segment_id


class EnsembleMergeEvidence(StrictModule, NonTrainableState):
    """Auditable range, coverage, burn-in, and checkpoint-chain evidence."""

    covered_trajectories: Array
    missing_measurement_trajectories: tuple[int, ...] = eqx.field(static=True)
    segment_ids: tuple[str, ...] = eqx.field(static=True)
    non_overlapping: bool = eqx.field(static=True)
    checkpoint_chain_continuous: bool = eqx.field(static=True)
    thermalization_excluded: bool = eqx.field(static=True)
    complete_measurement_coverage: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class MergedEnsemble(StrictModule, NonTrainableState):
    trajectory_indices: Array
    configuration_ids: tuple[str, ...] = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)
    merge_status: MergeStatus = eqx.field(static=True)
    evidence: EnsembleMergeEvidence
    merge_id: str = eqx.field(static=True)


def merge_ensemble_segments(
    manifest: EnsembleManifest,
    segments: Sequence[EnsembleSegment],
    /,
    *,
    require_complete: bool = True,
    require_checkpoint_chain: bool = True,
) -> MergedEnsemble:
    """Merge nonoverlapping segments and select only scheduled post-burn-in states."""
    if not isinstance(manifest, EnsembleManifest):
        raise TypeError("manifest must be EnsembleManifest.")
    ordered = tuple(sorted(tuple(segments), key=lambda value: value.start_trajectory))
    if not ordered or any(not isinstance(value, EnsembleSegment) for value in ordered):
        raise TypeError("segments must contain at least one EnsembleSegment.")
    if any(value.manifest_id != manifest.manifest_id for value in ordered):
        raise ValueError("Every segment must belong to the supplied ensemble manifest.")
    if any(
        left.stop_trajectory > right.start_trajectory
        for left, right in zip(ordered[:-1], ordered[1:], strict=True)
    ):
        raise ValueError("Ensemble segment trajectory ranges must not overlap.")
    checkpoint_continuous = all(
        left.terminal_checkpoint_id == right.initial_checkpoint_id
        for left, right in zip(ordered[:-1], ordered[1:], strict=True)
    )
    if require_checkpoint_chain and not checkpoint_continuous:
        raise ValueError("Ensemble segment checkpoints do not form one continuous chain.")
    scheduled = frozenset(manifest.schedule.trajectory_indices)
    by_trajectory: dict[int, str] = {}
    for segment in ordered:
        segment_indices = tuple(np.asarray(segment.trajectory_indices))
        for trajectory, configuration_id in zip(
            segment_indices, segment.configuration_ids, strict=True
        ):
            if trajectory in by_trajectory:
                raise ValueError(
                    "A configuration trajectory occurs in multiple segments."
                )
            by_trajectory[trajectory] = configuration_id
    selected_indices = tuple(
        trajectory
        for trajectory in manifest.schedule.trajectory_indices
        if trajectory in by_trajectory
        and any(
            segment.start_trajectory <= trajectory < segment.stop_trajectory
            for segment in ordered
        )
    )
    missing = tuple(
        trajectory
        for trajectory in manifest.schedule.trajectory_indices
        if trajectory not in selected_indices
    )
    complete = not missing
    if require_complete and not complete:
        raise ValueError(
            "Ensemble segments do not cover every scheduled measurement trajectory."
        )
    if any(trajectory not in scheduled for trajectory in selected_indices):
        raise RuntimeError("Merge selected a trajectory outside the schedule.")
    thermalization_excluded = all(
        trajectory >= manifest.schedule.thermalization_trajectories
        for trajectory in selected_indices
    )
    if not thermalization_excluded:
        raise RuntimeError("Merge retained a thermalization trajectory.")
    selected_ids = tuple(by_trajectory[index] for index in selected_indices)
    covered = np.asarray(
        tuple((segment.start_trajectory, segment.stop_trajectory) for segment in ordered),
        dtype=np.int64,
    )
    segment_ids = tuple(segment.segment_id for segment in ordered)
    evidence_id = canonical_fingerprint(
        {
            "kind": "qcd-ensemble-merge-evidence",
            "manifest": manifest.manifest_id,
            "segments": segment_ids,
            "covered_ranges": covered,
            "missing_measurements": missing,
            "checkpoint_chain_continuous": checkpoint_continuous,
            "thermalization_excluded": thermalization_excluded,
        }
    )
    evidence = EnsembleMergeEvidence(
        covered_trajectories=jnp.asarray(covered),
        missing_measurement_trajectories=missing,
        segment_ids=segment_ids,
        non_overlapping=True,
        checkpoint_chain_continuous=checkpoint_continuous,
        thermalization_excluded=thermalization_excluded,
        complete_measurement_coverage=complete,
        evidence_id=evidence_id,
    )
    status: MergeStatus = "complete" if complete else "incomplete"
    return MergedEnsemble(
        trajectory_indices=jnp.asarray(selected_indices, dtype=jnp.int64),
        configuration_ids=selected_ids,
        manifest_id=manifest.manifest_id,
        merge_status=status,
        evidence=evidence,
        merge_id=canonical_fingerprint(
            {
                "kind": "merged-qcd-ensemble",
                "manifest": manifest.manifest_id,
                "evidence": evidence_id,
                "trajectory_indices": selected_indices,
                "configuration_ids": selected_ids,
            }
        ),
    )


def ensemble_segment_from_configurations(
    manifest: EnsembleManifest,
    start_trajectory: int,
    stop_trajectory: int,
    configurations: ArrayLike,
    trajectory_indices: ArrayLike,
    /,
    *,
    initial_checkpoint_id: str,
    terminal_checkpoint_id: str,
) -> EnsembleSegment:
    """Content-address configurations before discarding their in-memory payloads."""
    if not isinstance(manifest, EnsembleManifest):
        raise TypeError("manifest must be EnsembleManifest.")
    values = np.asarray(configurations)
    indices = np.asarray(trajectory_indices, dtype=np.int64).reshape((-1,))
    if values.ndim < 1 or values.shape[0] != indices.size:
        raise ValueError("configurations require one leading entry per trajectory index.")
    configuration_ids = tuple(
        canonical_fingerprint(
            {
                "kind": "qcd-configuration",
                "manifest": manifest.manifest_id,
                "trajectory": int(trajectory),
                "content": array_tree_fingerprint(values[index]),
            }
        )
        for index, trajectory in enumerate(indices)
    )
    return EnsembleSegment(
        manifest.manifest_id,
        start_trajectory,
        stop_trajectory,
        indices,
        configuration_ids,
        initial_checkpoint_id=initial_checkpoint_id,
        terminal_checkpoint_id=terminal_checkpoint_id,
    )


__all__ = [
    "EnsembleManifest",
    "EnsembleMergeEvidence",
    "EnsembleSegment",
    "MeasurementSchedule",
    "MeasurementWorkItem",
    "MergeStatus",
    "MergedEnsemble",
    "ensemble_segment_from_configurations",
    "measurement_work_items",
    "merge_ensemble_segments",
]
