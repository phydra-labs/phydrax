#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum, IntFlag

import equinox as eqx
from jaxtyping import Array

from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class MPMRunStatus(IntEnum):
    SUCCESS = 0
    INVALID_INITIAL_STATE = 1
    ROUTE_REJECTED = 2
    DOMAIN_REJECTED = 3
    STABILITY_LIMIT_EXCEEDED = 4
    APIC_MOMENT_FAILED = 5
    MATERIAL_REJECTED = 6
    NONFINITE_STATE = 7
    PRESCRIBED_STEP_REJECTED = 8


class MPMRejectionReason(IntFlag):
    NONE = 0
    ROUTE = 1 << 0
    DOMAIN = 1 << 1
    STABILITY = 1 << 2
    APIC_MOMENT = 1 << 3
    MATERIAL = 1 << 4
    JACOBIAN = 1 << 5
    NONFINITE = 1 << 6


class MPMParticleState(StrictModule):
    position: Array
    velocity: Array
    deformation_gradient: Array
    affine_velocity: Array
    reference_volume: Array
    first_piola: Array
    reference_energy_density: Array
    maximum_wave_speed: Array
    material_state: Array


class MPMRuntimeState(StrictModule):
    particles: MPMParticleState
    time: Array
    accepted_step: Array
    last_status: Array


class MPMGridState(StrictModule):
    mass: Array
    momentum: Array
    velocity_before: Array
    internal_force: Array
    external_force: Array
    velocity_after: Array
    active: Array


class MPMStepRestriction(StrictModule):
    acoustic: Array
    advective: Array
    force: Array
    selected: Array


class MPMTransferEvidence(StrictModule):
    particle_mass: Array
    grid_mass: Array
    relative_mass_defect: Array
    particle_momentum: Array
    grid_momentum: Array
    relative_momentum_defect: Array
    particle_angular_momentum: Array
    grid_angular_momentum: Array
    angular_momentum_valid: Array
    relative_angular_momentum_defect: Array
    net_internal_force: Array
    maximum_partition_defect: Array
    maximum_gradient_sum_defect: Array
    maximum_first_moment_defect: Array
    maximum_apic_condition: Array
    active_grid_nodes: Array
    valid_routes: Array
    route_digest: Array
    successful: Array


class MPMEnergyLedger(StrictModule):
    particle_kinetic_before: Array
    grid_kinetic_before: Array
    grid_kinetic_after: Array
    particle_kinetic_after: Array
    material_energy_before: Array
    material_energy_after: Array
    external_work: Array
    boundary_work: Array
    balance_defect: Array


class MPMDiagnostics(StrictModule):
    transfer: MPMTransferEvidence
    energy: MPMEnergyLedger
    minimum_jacobian: Array
    maximum_jacobian: Array
    material_admissible: Array
    finite: Array


class MPMStepResult(StrictModule):
    candidate_state: MPMRuntimeState
    accepted_state: MPMRuntimeState
    grid: MPMGridState
    restriction: MPMStepRestriction
    diagnostics: MPMDiagnostics
    successful: Array
    rejection_reasons: Array
    requested_step_size: Array
    stability_margin: Array


class MPMPreparationEvidence(StrictModule, NonTrainableState):
    particle_count: int = eqx.field(static=True)
    grid_node_count: int = eqx.field(static=True)
    route_count: int = eqx.field(static=True)
    route_payload_width: int = eqx.field(static=True)
    step_workspace_bytes: int = eqx.field(static=True)
    state_bytes: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


__all__ = [
    "MPMDiagnostics",
    "MPMEnergyLedger",
    "MPMGridState",
    "MPMParticleState",
    "MPMPreparationEvidence",
    "MPMRejectionReason",
    "MPMRunStatus",
    "MPMRuntimeState",
    "MPMStepRestriction",
    "MPMStepResult",
    "MPMTransferEvidence",
]
