#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum, IntFlag

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

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
    CONTACT_REJECTED = 9
    LOCAL_ROOT_FAILED = 10
    NONLINEAR_FAILED = 11
    CAPACITY_EXCEEDED = 12
    TOPOLOGY_REJECTED = 13


class MPMRejectionReason(IntFlag):
    NONE = 0
    ROUTE = 1 << 0
    DOMAIN = 1 << 1
    STABILITY = 1 << 2
    APIC_MOMENT = 1 << 3
    MATERIAL = 1 << 4
    JACOBIAN = 1 << 5
    NONFINITE = 1 << 6
    CONTACT = 1 << 7
    LOCAL_ROOT = 1 << 8
    CAPACITY = 1 << 9
    NONLINEAR = 1 << 10
    TOPOLOGY = 1 << 11
    FRACTURE = 1 << 12
    SPARSE = 1 << 13


class MPMLimitingProcess(IntEnum):
    NONE = 0
    ACOUSTIC = 1
    ADVECTIVE = 2
    FORCE = 3
    CONTACT = 4
    MATERIAL = 5
    SOURCE_DOMAIN = 6
    NONLINEAR = 7


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
    topology_generation: Array
    assignment_input: object
    material_slots: Array
    body_ids: Array
    velocity_field_slots: Array
    storage_state: object

    def __init__(
        self,
        particles: MPMParticleState,
        time: ArrayLike,
        accepted_step: ArrayLike,
        last_status: ArrayLike,
        topology_generation: ArrayLike = 0,
        assignment_input: object = None,
        material_slots: ArrayLike | None = None,
        body_ids: ArrayLike | None = None,
        velocity_field_slots: ArrayLike | None = None,
        storage_state: object = None,
        /,
    ):
        self.particles = particles
        self.time = jnp.asarray(time)
        self.accepted_step = jnp.asarray(accepted_step, dtype=jnp.int32)
        self.last_status = jnp.asarray(last_status, dtype=jnp.int32)
        self.topology_generation = jnp.asarray(topology_generation, dtype=jnp.int32)
        self.assignment_input = assignment_input
        count = int(particles.position.shape[0])
        self.material_slots = (
            jnp.zeros((count,), dtype=jnp.int32)
            if material_slots is None
            else jnp.asarray(material_slots, dtype=jnp.int32)
        )
        self.body_ids = (
            jnp.zeros((count,), dtype=jnp.int32)
            if body_ids is None
            else jnp.asarray(body_ids, dtype=jnp.int32)
        )
        self.velocity_field_slots = (
            jnp.zeros((count,), dtype=jnp.int32)
            if velocity_field_slots is None
            else jnp.asarray(velocity_field_slots, dtype=jnp.int32)
        )
        if any(
            value.shape != (count,)
            for value in (
                self.material_slots,
                self.body_ids,
                self.velocity_field_slots,
            )
        ):
            raise ValueError("MPM particle ownership slots must have capacity shape.")
        self.storage_state = storage_state


class MPMGridState(StrictModule):
    """Nodal fields with leading material/velocity-field axis K."""

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
    contact: Array
    material: Array
    source_domain_motion: Array
    nonlinear: Array
    selected: Array
    limiting_process: Array
    suggested_step: Array


class MPMScheduleEvidence(StrictModule):
    schedule_code: Array
    stress_updated_first: Array
    second_momentum_extrapolation: Array
    second_transfer_mass_defect: Array
    second_transfer_momentum_defect: Array
    second_constraint_work: Array
    phase_digest: Array
    successful: Array


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
    field_action_reaction_defect: Array
    field_contact_successful: Array
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
    contact_work: Array
    contact_dissipation: Array
    plastic_dissipation: Array
    fracture_dissipation: Array
    balance_defect: Array


class MPMDiagnostics(StrictModule):
    transfer: MPMTransferEvidence
    schedule: MPMScheduleEvidence
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
    retry_requested: Array
    suggested_step: Array


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
    "MPMLimitingProcess",
    "MPMParticleState",
    "MPMPreparationEvidence",
    "MPMRejectionReason",
    "MPMRunStatus",
    "MPMRuntimeState",
    "MPMScheduleEvidence",
    "MPMStepRestriction",
    "MPMStepResult",
    "MPMTransferEvidence",
]
