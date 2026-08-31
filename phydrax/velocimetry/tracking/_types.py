#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
from jaxtyping import Array

from ..._strict import StrictModule


class DetectionStatus(IntEnum):
    """Per-detection refinement status."""

    SUCCESS = 0
    BORDER = 1
    CROWDED = 2


class AssociationStatus(IntEnum):
    """Association and tuple-packing status."""

    SUCCESS = 0
    INFEASIBLE = 1
    CANDIDATE_OVERFLOW = 2
    HEURISTIC_NOT_CERTIFIED = 3
    NONFINITE_INPUT = 4


class ReconstructionStatus(IntEnum):
    """Per-tuple reconstruction status."""

    SUCCESS = 0
    NOT_SELECTED = 1
    INSUFFICIENT_VIEWS = 2
    DEGENERATE = 3
    NONFINITE_INPUT = 4


class TrackStatus(IntEnum):
    """Per-step tracking status."""

    SUCCESS = 0
    CAPACITY_EXHAUSTED = 1
    ASSIGNMENT_FAILED = 2
    NONFINITE_INPUT = 3


def detection_status_name(value: int, /) -> str:
    return DetectionStatus(int(value)).name.lower()


def association_status_name(value: int, /) -> str:
    return AssociationStatus(int(value)).name.lower()


def reconstruction_status_name(value: int, /) -> str:
    return ReconstructionStatus(int(value)).name.lower()


def track_status_name(value: int, /) -> str:
    return TrackStatus(int(value)).name.lower()


class ParticleDetections(StrictModule):
    """Fixed-capacity point-particle detections in ``(row, column)`` order."""

    positions_rc: Array
    covariance_rc: Array
    intensity: Array
    radius: Array
    valid: Array
    status: Array
    overflow_count: Array
    frame_id: str = eqx.field(static=True)
    detection_id: str = eqx.field(static=True)


class AssociationEvidence(StrictModule):
    """Gating, dummy-match, ambiguity, and certification evidence."""

    gated_pair_count: Array
    matched_count: Array
    unmatched_a_count: Array
    unmatched_b_count: Array
    ambiguous_a_count: Array
    assignment_status: Array
    optimality_proven: Array
    plan_id: str = eqx.field(static=True)


class TuplePackingEvidence(StrictModule):
    """Candidate truncation, conflicts, objective, and solver evidence."""

    candidate_count: Array
    overflow_count: Array
    conflict_count: Array
    selected_count: Array
    objective_value: Array
    lower_bound: Array
    upper_bound: Array
    absolute_gap: Array
    optimality_proven: Array
    plan_id: str = eqx.field(static=True)


class ReconstructionEvidence(StrictModule):
    """View support and triangulation degeneracy counts."""

    selected_count: Array
    reconstructed_count: Array
    insufficient_view_count: Array
    degenerate_count: Array
    nonfinite_count: Array
    candidate_overflow_count: Array
    reconstruction_id: str = eqx.field(static=True)


class TrackStepEvidence(StrictModule):
    """One streaming step's association, lifecycle, and capacity counts."""

    matched_count: Array
    missed_count: Array
    birth_count: Array
    death_count: Array
    unmatched_observation_count: Array
    overflow_count: Array
    ambiguous_match_count: Array
    assignment_status: Array
    plan_id: str = eqx.field(static=True)


class TrackRuntimeState(StrictModule):
    """Fixed-capacity streaming constant-velocity track state."""

    track_ids: Array
    active: Array
    age: Array
    missed: Array
    states: Array
    covariance: Array
    last_observation_time: Array
    time: Array
    initialized: Array
    next_track_id: Array
    step_index: Array
    capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class TrackStepResult(StrictModule):
    """Audited output of one predict, associate, update, and lifecycle step."""

    state: TrackRuntimeState
    observation_track_ids: Array
    observation_track_slots: Array
    track_observation_indices: Array
    matched_tracks: Array
    births: Array
    deaths: Array
    unmatched_observations: Array
    ambiguous_tracks: Array
    observations: Array
    observation_covariance: Array
    observation_valid: Array
    status: Array
    evidence: TrackStepEvidence


class TrackResult(StrictModule):
    """Fixed-slot identity-bearing tracking history with explicit gaps and resets."""

    times: Array
    states: Array
    covariances: Array
    active: Array
    observed: Array
    track_ids: Array
    observation_indices: Array
    observations: Array
    observation_covariances: Array
    births: Array
    deaths: Array
    resets: Array
    observation_track_ids: Array
    observation_valid: Array
    step_status: Array
    overflow_count: Array
    final_state: TrackRuntimeState
    track_capacity: int = eqx.field(static=True)
    observation_capacity: int = eqx.field(static=True)
    result_id: str = eqx.field(static=True)
    source_ids: tuple[str, ...] = eqx.field(static=True)


class TrackSmoothingResult(StrictModule):
    """Frozen-association Gaussian smoothing result in fixed track slots."""

    states: Array
    covariances: Array
    valid: Array
    innovations: Array
    innovation_covariances: Array
    segment_status: Array
    filter_results: tuple
    smoothing_id: str = eqx.field(static=True)
    track_result_id: str = eqx.field(static=True)


__all__ = [
    "AssociationEvidence",
    "AssociationStatus",
    "DetectionStatus",
    "ParticleDetections",
    "ReconstructionEvidence",
    "ReconstructionStatus",
    "TrackResult",
    "TrackRuntimeState",
    "TrackSmoothingResult",
    "TrackStatus",
    "TrackStepEvidence",
    "TrackStepResult",
    "TuplePackingEvidence",
    "association_status_name",
    "detection_status_name",
    "reconstruction_status_name",
    "track_status_name",
]
