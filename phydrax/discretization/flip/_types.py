#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum, IntFlag
from typing import Any

import equinox as eqx
from jaxtyping import Array

from ..._strict import StrictModule
from ..finite_volume import FaceVelocity
from ..splatting import ParticleGridSplatState


class FLIPRunStatus(IntEnum):
    SUCCESS = 0
    INVALID_STATE = 1
    TRANSFER_FAILED = 2
    PROJECTION_FAILED = 3
    EXTRAPOLATION_FAILED = 4
    STABILITY_LIMIT_EXCEEDED = 5
    BOUNDARY_EXIT = 6
    NONFINITE_STATE = 7
    GEOMETRY_FAILED = 8
    COLLISION_FAILED = 9


class FLIPRejectionReason(IntFlag):
    NONE = 0
    TRANSFER = 1
    CLASSIFICATION = 2
    EXTRAPOLATION = 4
    PROJECTION = 8
    STABILITY = 16
    BOUNDARY = 32
    NONFINITE = 64
    GEOMETRY = 128
    COLLISION = 256


class FLIPParticleState(StrictModule):
    position: Array
    velocity: Array


class FLIPRuntimeState(StrictModule):
    particles: FLIPParticleState
    pressure: Array
    time: Array
    accepted_step: Array
    status: Array
    geometry_epoch: Array
    geometry_id: str = eqx.field(static=True)


class FLIPTransferState(StrictModule):
    cell: ParticleGridSplatState
    faces: tuple[ParticleGridSplatState, ...]
    transfer_id: str = eqx.field(static=True)


class FLIPParticleToGridResult(StrictModule):
    particle_volume_content: Array
    liquid_fraction: Array
    face_mass: FaceVelocity
    face_momentum: FaceVelocity
    velocity: FaceVelocity
    face_support: tuple[Array, ...]
    mass_balance_defect: Array
    momentum_balance_defect: Array
    finite: Array
    successful: Array
    geometry_id: str = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)


class FLIPGridToParticleResult(StrictModule):
    pic_velocity: Array
    flip_increment: Array
    support: Array
    finite: Array
    successful: Array
    geometry_id: str = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)


class FLIPDiagnostics(StrictModule):
    liquid_count: Array
    air_count: Array
    classification_margin: Array
    mass_balance_defect: Array
    momentum_balance_defect: Array
    extrapolation_holes: Array
    projection_residual: Array
    divergence_norm: Array
    maximum_displacement_fraction: Array
    energy_before: Array
    energy_after: Array
    successful: Array
    geometry_accepted: Array
    collision_count: Array
    wall_work: Array
    maximum_penetration: Array
    rejection_reason: Array
    details: Any


class FLIPStepResult(StrictModule):
    candidate_state: FLIPRuntimeState
    accepted_state: FLIPRuntimeState
    pre_grid_velocity: FaceVelocity
    post_grid_velocity: FaceVelocity
    liquid_fraction: Array
    diagnostics: FLIPDiagnostics
    successful: Array
    solid: Any
    geometry_id: str = eqx.field(static=True)


__all__ = [
    "FLIPDiagnostics",
    "FLIPGridToParticleResult",
    "FLIPParticleState",
    "FLIPParticleToGridResult",
    "FLIPRejectionReason",
    "FLIPRunStatus",
    "FLIPRuntimeState",
    "FLIPStepResult",
    "FLIPTransferState",
]
