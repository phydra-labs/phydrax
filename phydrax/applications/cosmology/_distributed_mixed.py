#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Executable owner-computes distribution for fixed-grid mixed cosmology."""

from __future__ import annotations

from math import gcd, prod
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec
from jaxtyping import Array

from ..._execution_plan import ExecutionPlan, PlacementKind
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.particle._distributed_runtime import (
    DistributedParticleMigrationResult,
    DistributedParticleRuntimePlan,
    DistributedParticleState,
    PreparedDistributedParticleRuntime,
)
from ...discretization.spectral._distributed import DistributedSpectralExecutionPlan
from ...lifecycle._distributed_checkpoint import restore_global_array_from_checkpoint
from ...lifecycle._models import CheckpointManifest
from ...solver._particle_gravity import DistributedParticleLayout
from ._coupled import ComovingEulerState
from ._force_scalability import DistributedPMFeasibilityEvidence
from ._mixed_matter import (
    PreparedWaveParticleCosmology,
    PreparedWaveParticleGasCosmology,
    WaveParticleCosmologyState,
    WaveParticleGasCosmologyState,
)
from ._particles import CosmologicalParticleState
from ._wave_dark_matter import WaveDarkMatterState


DistributedMixedPreparationStatus: TypeAlias = Literal[
    "prepared",
    "infeasible-resources",
    "layout-mismatch",
    "placement-incomplete",
    "checkpoint-incomplete",
    "missing-distributed-primitives",
]


class DistributedMixedPlacementEvidence(StrictModule, NonTrainableState):
    """Hardware-exact topology and named value-placement agreement."""

    spectral_device_count: int = eqx.field(static=True)
    particle_device_count: int = eqx.field(static=True)
    execution_device_count: int = eqx.field(static=True)
    execution_mesh_matches: bool = eqx.field(static=True)
    device_keys_match: bool = eqx.field(static=True)
    topology_available: bool = eqx.field(static=True)
    required_values: tuple[str, ...] = eqx.field(static=True)
    partitioned_values: tuple[str, ...] = eqx.field(static=True)
    missing_values: tuple[str, ...] = eqx.field(static=True)
    host_gather: bool = eqx.field(static=True)
    successful: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class DistributedMixedCollectiveEvidence(StrictModule, NonTrainableState):
    """Executable collectives bound to the exact prepared mesh."""

    spectral_collective_count: int = eqx.field(static=True)
    particle_send_capacity: int = eqx.field(static=True)
    particle_receive_capacity: int = eqx.field(static=True)
    particle_ghost_capacity: int = eqx.field(static=True)
    particle_exchange_bytes_per_device: int = eqx.field(static=True)
    distributed_fft_executable: bool = eqx.field(static=True)
    particle_migration_executable: bool = eqx.field(static=True)
    distributed_deposit_gather_executable: bool = eqx.field(static=True)
    distributed_poisson_executable: bool = eqx.field(static=True)
    distributed_gas_executable: bool = eqx.field(static=True)
    host_gather_fallback: bool = eqx.field(static=True)
    successful: bool = eqx.field(static=True)
    missing_primitives: tuple[str, ...] = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class DistributedMixedCheckpointEvidence(StrictModule, NonTrainableState):
    """Exact shard coverage and topology-neutral restore binding."""

    required_bytes: int = eqx.field(static=True)
    unpadded_bytes_per_checkpoint: int = eqx.field(static=True)
    payload_bytes_per_checkpoint: int = eqx.field(static=True)
    padding_bytes_per_checkpoint: int = eqx.field(static=True)
    payload_alignment: int = eqx.field(static=True)
    spectral_reserved_bytes: int = eqx.field(static=True)
    maximum_bytes: int = eqx.field(static=True)
    capacity_sufficient: bool = eqx.field(static=True)
    distributed_checkpoint_executable: bool = eqx.field(static=True)
    restart_identity_executable: bool = eqx.field(static=True)
    changed_sharding_restore_executable: bool = eqx.field(static=True)
    covered_values: tuple[str, ...] = eqx.field(static=True)
    shard_plan_id: str = eqx.field(static=True)
    successful: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class DistributedMixedPreparationEvidence(StrictModule, NonTrainableState):
    """Truthful distributed admission result; false is a preparation refusal."""

    placement: DistributedMixedPlacementEvidence
    collective: DistributedMixedCollectiveEvidence
    checkpoint: DistributedMixedCheckpointEvidence
    feasibility: DistributedPMFeasibilityEvidence
    spectral_resource_accepted: bool = eqx.field(static=True)
    executable: bool = eqx.field(static=True)
    status: DistributedMixedPreparationStatus = eqx.field(static=True)
    reasons: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class DistributedMixedPreparationResult(StrictModule, NonTrainableState):
    """Preparation evidence and the separately bound executable, when admitted."""

    evidence: DistributedMixedPreparationEvidence
    successful: Array
    executable_plan_id: str | None = eqx.field(static=True)
    executable: Any


class DistributedMixedDensity(StrictModule):
    wave_density: Array
    particle_density: Array
    gas_density: Array
    total_density: Array
    component_mass: Array
    total_mass: Array
    particle_source_mass: Array
    particle_deposited_mass: Array
    particle_routes: Any
    finite: Array
    successful: Array
    execution_id: str = eqx.field(static=True)


class DistributedMixedGravityResult(StrictModule):
    density: DistributedMixedDensity
    potential: Array
    cell_acceleration: Array
    particle_acceleration: Array
    owner_particle_acceleration: Array
    mean_density: Array
    source_integral: Array
    poisson_relative_residual: Array
    gauge_defect: Array
    component_force: Array
    total_force: Array
    finite: Array
    successful: Array
    execution_id: str = eqx.field(static=True)


class DistributedMixedState(StrictModule):
    wave: WaveDarkMatterState
    particles: DistributedParticleState
    gas: ComovingEulerState | None
    execution_id: str = eqx.field(static=True)


class DistributedMixedStepResult(StrictModule):
    state: DistributedMixedState
    gravity: DistributedMixedGravityResult
    migration: DistributedParticleMigrationResult
    initial_mass: Array
    final_mass: Array
    mass_balance_defect: Array
    kinetic_phase: Array
    potential_phase: Array
    phase_resolved: Array
    homogeneous_gas_successful: Array
    successful: Array
    execution_id: str = eqx.field(static=True)


class DistributedMixedEvolutionResult(StrictModule):
    state: DistributedMixedState
    accepted_steps: Array
    initial_mass: Array
    final_mass: Array
    maximum_mass_balance_defect: Array
    successful: Array
    execution_id: str = eqx.field(static=True)


class DistributedMixedExecutionPlan(StrictModule):
    """Prepare fixed-capacity mixed execution on one exact named device mesh."""

    mixed: PreparedWaveParticleCosmology | PreparedWaveParticleGasCosmology
    spectral: DistributedSpectralExecutionPlan
    execution: ExecutionPlan = eqx.field(static=True)
    particles: DistributedParticleLayout
    feasibility: DistributedPMFeasibilityEvidence
    checkpoint_count: int = eqx.field(static=True)
    maximum_checkpoint_bytes: int = eqx.field(static=True)
    required_checkpoint_bytes: int = eqx.field(static=True)
    checkpoint_unpadded_bytes: int = eqx.field(static=True)
    checkpoint_payload_bytes: int = eqx.field(static=True)
    checkpoint_payload_alignment: int = eqx.field(static=True)
    particle_send_capacity: int = eqx.field(static=True)
    particle_receive_capacity: int = eqx.field(static=True)
    particle_ghost_capacity: int = eqx.field(static=True)
    particle_ghost_width: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mixed: PreparedWaveParticleCosmology | PreparedWaveParticleGasCosmology,
        spectral: DistributedSpectralExecutionPlan,
        execution: ExecutionPlan,
        particles: DistributedParticleLayout,
        feasibility: DistributedPMFeasibilityEvidence,
        /,
        *,
        checkpoint_count: int = 1,
        maximum_checkpoint_bytes: int,
        particle_send_capacity: int | None = None,
        particle_receive_capacity: int | None = None,
        particle_ghost_capacity: int | None = None,
        particle_ghost_width: float = 0.0,
    ):
        if not isinstance(
            mixed, (PreparedWaveParticleCosmology, PreparedWaveParticleGasCosmology)
        ):
            raise TypeError("mixed must be a prepared fixed-grid mixed cosmology plan.")
        if not isinstance(spectral, DistributedSpectralExecutionPlan):
            raise TypeError("spectral must be DistributedSpectralExecutionPlan.")
        if not isinstance(execution, ExecutionPlan):
            raise TypeError("execution must be ExecutionPlan.")
        if not isinstance(particles, DistributedParticleLayout):
            raise TypeError("particles must be DistributedParticleLayout.")
        if not isinstance(feasibility, DistributedPMFeasibilityEvidence):
            raise TypeError("feasibility must be DistributedPMFeasibilityEvidence.")
        checkpoints = int(checkpoint_count)
        maximum = int(maximum_checkpoint_bytes)
        send = (
            particles.capacity_per_device
            if particle_send_capacity is None
            else int(particle_send_capacity)
        )
        receive = (
            particles.capacity_per_device
            if particle_receive_capacity is None
            else int(particle_receive_capacity)
        )
        ghosts = (
            particles.capacity_per_device
            if particle_ghost_capacity is None
            else int(particle_ghost_capacity)
        )
        ghost_width = float(particle_ghost_width)
        if checkpoints < 1 or maximum <= 0:
            raise ValueError(
                "Distributed mixed execution requires positive checkpoint capacity."
            )
        particle_boundaries = np.asarray(particles.key_boundaries, dtype=np.uint64)
        minimum_key_width = int(np.min(np.diff(particle_boundaries)))
        maximum_ghost_width = (
            mixed.plan.particles.box_size[0]
            * minimum_key_width
            / float(np.iinfo(np.uint32).max)
        )
        if (
            send <= 0
            or receive != particles.capacity_per_device
            or ghosts <= 0
            or ghosts > particles.capacity_per_device
            or not np.isfinite(ghost_width)
            or ghost_width < 0.0
            or minimum_key_width <= 0
            or ghost_width > maximum_ghost_width
        ):
            raise ValueError("Distributed particle exchange capacities are invalid.")
        wave = mixed.plan.wave
        real_dtype = np.dtype(
            jnp.empty(
                (), dtype=jnp.dtype(wave.discretization.plan.precision.coefficient_dtype)
            ).real.dtype
        )
        complex_dtype = np.dtype(wave.discretization.plan.precision.coefficient_dtype)
        wave_bytes = prod(wave.discretization.physical_shape) * complex_dtype.itemsize
        particle_support = mixed.plan.particles.particles
        particle_count = particle_support.capacity
        dimension = particle_support.ambient_dimension
        particle_dtype = np.dtype(particle_support.plan.coordinate_dtype)
        mass_dtype = np.dtype(particle_support.masses.dtype)
        particle_bytes = particle_count * (
            2 * dimension * particle_dtype.itemsize
            + mass_dtype.itemsize
            + np.dtype(np.int64).itemsize
            + 2 * np.dtype(np.int32).itemsize
            + np.dtype(np.bool_).itemsize
            + np.dtype(np.uint64).itemsize
        )
        gas_bytes = 0
        scale_bytes = real_dtype.itemsize + particle_dtype.itemsize
        if isinstance(mixed, PreparedWaveParticleGasCosmology):
            gas_shape = mixed.plan.gas.dynamics.discretization.cell_shape
            gas_components = mixed.plan.gas.dynamics.discretization.component_count
            gas_dtype = np.dtype(
                mixed.plan.gas.dynamics.discretization.cell_volumes.dtype
            )
            gas_bytes = prod(gas_shape) * gas_components * gas_dtype.itemsize
            scale_bytes += gas_dtype.itemsize
        checkpoint_unpadded = wave_bytes + particle_bytes + gas_bytes + scale_bytes
        physical_shape = tuple(wave.discretization.physical_shape)
        checkpoint_alignment = (
            gcd(physical_shape[0], physical_shape[1]) if len(physical_shape) >= 2 else 1
        )
        checkpoint_padding = (-checkpoint_unpadded) % checkpoint_alignment
        checkpoint_payload = checkpoint_unpadded + checkpoint_padding
        required = checkpoints * checkpoint_payload
        self.mixed = mixed
        self.spectral = spectral
        self.execution = execution
        self.particles = particles
        self.feasibility = feasibility
        self.checkpoint_count = checkpoints
        self.maximum_checkpoint_bytes = maximum
        self.required_checkpoint_bytes = required
        self.checkpoint_unpadded_bytes = checkpoint_unpadded
        self.checkpoint_payload_bytes = checkpoint_payload
        self.checkpoint_payload_alignment = checkpoint_alignment
        self.particle_send_capacity = send
        self.particle_receive_capacity = receive
        self.particle_ghost_capacity = ghosts
        self.particle_ghost_width = ghost_width
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-mixed-execution-plan",
                "mixed": mixed.prepared_id,
                "spectral": spectral.plan_id,
                "execution": execution.plan_fingerprint,
                "particles": particles.layout_id,
                "feasibility": {
                    "mesh": list(feasibility.mesh_shape),
                    "device_mesh": list(feasibility.device_mesh_shape),
                    "capacity": feasibility.particle_capacity_per_device,
                    "feasible": feasibility.feasible,
                },
                "checkpoint_count": checkpoints,
                "maximum_checkpoint_bytes": maximum,
                "required_checkpoint_bytes": required,
                "checkpoint_unpadded_bytes": checkpoint_unpadded,
                "checkpoint_payload_bytes": checkpoint_payload,
                "checkpoint_payload_alignment": checkpoint_alignment,
                "particle_exchange": {
                    "send": send,
                    "receive": receive,
                    "ghost": ghosts,
                    "ghost_width": ghost_width,
                },
            }
        )

    def prepare(self, /) -> DistributedMixedPreparationResult:
        topology = self.spectral.topology
        group = self.execution.group
        execution_device_count = 0 if group is None else group.device_count
        execution_mesh_matches = self.execution.device_mesh_id == topology.topology_id
        device_keys_match = (
            tuple(group.device_keys) == tuple(topology.device_keys)
            if group is not None
            else False
        )
        try:
            topology.require_available()
            topology_available = True
        except RuntimeError:
            topology_available = False
        required_values = (
            "wave",
            "particle_positions",
            "particle_momenta",
            "potential",
        )
        if isinstance(self.mixed, PreparedWaveParticleGasCosmology):
            required_values = required_values + ("gas",)
        partitioned = tuple(
            placement.name
            for placement in self.execution.value_placements
            if placement.kind is PlacementKind.PARTITIONED
        )
        missing = tuple(name for name in required_values if name not in partitioned)
        placement_successful = (
            topology.device_count == self.particles.device_count
            and topology.device_count == execution_device_count
            and execution_mesh_matches
            and device_keys_match
            and topology_available
            and not missing
            and not self.spectral.report.host_gather
        )
        placement_id = canonical_fingerprint(
            {
                "kind": "distributed-mixed-placement-evidence",
                "spectral_devices": topology.device_count,
                "particle_devices": self.particles.device_count,
                "execution_devices": execution_device_count,
                "execution_mesh_matches": execution_mesh_matches,
                "device_keys_match": device_keys_match,
                "topology_available": topology_available,
                "required_values": list(required_values),
                "partitioned_values": list(partitioned),
                "missing_values": list(missing),
                "host_gather": self.spectral.report.host_gather,
                "successful": placement_successful,
            }
        )
        placement = DistributedMixedPlacementEvidence(
            topology.device_count,
            self.particles.device_count,
            execution_device_count,
            execution_mesh_matches,
            device_keys_match,
            topology_available,
            required_values,
            partitioned,
            missing,
            self.spectral.report.host_gather,
            placement_successful,
            placement_id,
        )

        wave = self.mixed.plan.wave
        support_capacity = self.mixed.plan.particles.particles.capacity
        topology_supported = (
            len(topology.mesh_axis_names) == 1 and self.spectral.schedule == "slab"
        )
        capacity_matches = (
            support_capacity
            == self.particles.device_count * self.particles.capacity_per_device
        )
        shape_matches = tuple(self.feasibility.mesh_shape) == tuple(
            self.spectral.spatial_shape
        ) and tuple(self.spectral.spatial_shape) == tuple(wave.discretization.modal_shape)
        wave_coefficient_dtype = np.dtype(
            jax.dtypes.canonicalize_dtype(
                np.dtype(wave.discretization.plan.precision.coefficient_dtype)
            )
        )
        wave_accumulation_dtype = np.dtype(
            jax.dtypes.canonicalize_dtype(
                np.dtype(wave.discretization.plan.precision.reduction_dtype)
            )
        )
        wave_real_dtype = np.dtype(
            jnp.empty((), dtype=jnp.dtype(wave_coefficient_dtype)).real.dtype
        )
        particle_support = self.mixed.plan.particles.particles
        particle_dtype = np.dtype(
            jax.dtypes.canonicalize_dtype(
                np.dtype(particle_support.plan.coordinate_dtype)
            )
        )
        particle_mass_dtype = np.dtype(
            jax.dtypes.canonicalize_dtype(np.dtype(particle_support.masses.dtype))
        )
        gas_dtype = wave_real_dtype
        if isinstance(self.mixed, PreparedWaveParticleGasCosmology):
            gas_dtype = np.dtype(
                jax.dtypes.canonicalize_dtype(
                    np.dtype(
                        self.mixed.plan.gas.dynamics.discretization.cell_volumes.dtype
                    )
                )
            )
        precision_matches = (
            particle_dtype == wave_real_dtype
            and particle_mass_dtype == wave_real_dtype
            and gas_dtype == wave_real_dtype
        )
        spectral_abi = (
            self.spectral.state_shape == ()
            and np.dtype(self.spectral.coefficient_dtype) == wave_coefficient_dtype
            and np.dtype(self.spectral.accumulation_dtype) == wave_accumulation_dtype
            and precision_matches
        )
        distributed_fft = topology_supported and spectral_abi
        particle_migration = topology_supported and capacity_matches
        deposit_gather = particle_migration
        poisson = distributed_fft and shape_matches
        gas = (
            not isinstance(self.mixed, PreparedWaveParticleGasCosmology)
            or topology_supported
        )
        primitive_flags = {
            "spectral-abi": spectral_abi,
            "distributed-fft": distributed_fft,
            "particle-migration": particle_migration,
            "particle-deposit-gather": deposit_gather,
            "shared-mixed-poisson": poisson,
            "comoving-gas-update": gas,
        }
        missing_primitives = tuple(
            name for name, available in primitive_flags.items() if not available
        )
        collective_successful = (
            not missing_primitives and not self.spectral.report.host_gather
        )
        dimension = self.mixed.plan.particles.particles.ambient_dimension
        real_itemsize = (
            np.dtype(
                self.mixed.plan.wave.discretization.plan.precision.coefficient_dtype
            ).itemsize
            // 2
        )
        particle_packet_bytes = (
            (2 * dimension + 1) * real_itemsize
            + np.dtype(np.int64).itemsize
            + np.dtype(np.int32).itemsize
            + np.dtype(np.uint64).itemsize
            + np.dtype(np.bool_).itemsize
        )
        exchange_bytes = (
            2
            * topology.device_count
            * self.particle_send_capacity
            * particle_packet_bytes
            + 2 * self.particle_ghost_capacity * particle_packet_bytes
            + 2
            * topology.device_count
            * self.particles.capacity_per_device
            * (np.dtype(np.int64).itemsize + np.dtype(np.bool_).itemsize)
        )
        collective_id = canonical_fingerprint(
            {
                "kind": "distributed-mixed-collective-evidence",
                "spectral_collective_count": self.spectral.report.collective_count,
                "particle_send_capacity": self.particle_send_capacity,
                "particle_receive_capacity": self.particle_receive_capacity,
                "particle_ghost_capacity": self.particle_ghost_capacity,
                "particle_exchange_bytes_per_device": exchange_bytes,
                "flags": primitive_flags,
                "host_gather_fallback": False,
                "missing": list(missing_primitives),
            }
        )
        collective = DistributedMixedCollectiveEvidence(
            self.spectral.report.collective_count,
            self.particle_send_capacity,
            self.particle_receive_capacity,
            self.particle_ghost_capacity,
            exchange_bytes,
            distributed_fft,
            particle_migration,
            deposit_gather,
            poisson,
            gas,
            False,
            collective_successful,
            missing_primitives,
            collective_id,
        )

        spectral_reserved = self.spectral.report.resource.checkpoint_bytes
        capacity_sufficient = (
            self.required_checkpoint_bytes <= self.maximum_checkpoint_bytes
            and spectral_reserved > 0
        )
        covered_values = (
            "wave",
            "wave_scale_factor",
            "particle_positions",
            "particle_momenta",
            "particle_masses",
            "particle_stable_ids",
            "particle_logical_slots",
            "particle_active_mask",
            "particle_rng_counters",
            "particle_scale_factor",
            "particle_owner",
        )
        if isinstance(self.mixed, PreparedWaveParticleGasCosmology):
            covered_values = covered_values + ("gas", "gas_scale_factor")
        shard_plan_id = canonical_fingerprint(
            {
                "kind": "distributed-mixed-checkpoint-shard-plan",
                "execution": self.execution.plan_fingerprint,
                "topology": topology.topology_id,
                "coverage": list(covered_values),
                "topology_epoch": self.execution.topology_epoch,
            }
        )
        distributed_checkpoint = topology_supported
        restart_identity = topology_supported
        changed_restore = topology_supported
        checkpoint_successful = (
            capacity_sufficient
            and distributed_checkpoint
            and restart_identity
            and changed_restore
        )
        checkpoint_id = canonical_fingerprint(
            {
                "kind": "distributed-mixed-checkpoint-evidence",
                "required_bytes": self.required_checkpoint_bytes,
                "unpadded_bytes_per_checkpoint": self.checkpoint_unpadded_bytes,
                "payload_bytes_per_checkpoint": self.checkpoint_payload_bytes,
                "padding_bytes_per_checkpoint": (
                    self.checkpoint_payload_bytes - self.checkpoint_unpadded_bytes
                ),
                "payload_alignment": self.checkpoint_payload_alignment,
                "spectral_reserved_bytes": spectral_reserved,
                "maximum_bytes": self.maximum_checkpoint_bytes,
                "capacity_sufficient": capacity_sufficient,
                "distributed_checkpoint": distributed_checkpoint,
                "restart_identity": restart_identity,
                "changed_sharding_restore": changed_restore,
                "coverage": list(covered_values),
                "shard_plan": shard_plan_id,
            }
        )
        checkpoint = DistributedMixedCheckpointEvidence(
            self.required_checkpoint_bytes,
            self.checkpoint_unpadded_bytes,
            self.checkpoint_payload_bytes,
            self.checkpoint_payload_bytes - self.checkpoint_unpadded_bytes,
            self.checkpoint_payload_alignment,
            spectral_reserved,
            self.maximum_checkpoint_bytes,
            capacity_sufficient,
            distributed_checkpoint,
            restart_identity,
            changed_restore,
            covered_values,
            shard_plan_id,
            checkpoint_successful,
            checkpoint_id,
        )

        feasibility_matches = (
            shape_matches
            and np.allclose(
                np.asarray(self.spectral.domain_lengths),
                np.asarray(
                    tuple(float(axis.length) for axis in wave.discretization.axes)
                ),
                rtol=0.0,
                atol=2.0e-13,
            )
            and prod(self.feasibility.device_mesh_shape) == topology.device_count
            and self.feasibility.particle_capacity_per_device
            == self.particles.capacity_per_device
            and topology.device_count == self.particles.device_count
        )
        resources_ok = (
            self.feasibility.feasible
            and feasibility_matches
            and self.spectral.report.resource.accepted
        )
        executable = (
            resources_ok
            and placement.successful
            and collective.successful
            and checkpoint.successful
        )
        reasons: list[str] = []
        if not resources_ok:
            reasons.append("distributed mesh/resource feasibility is not established")
        if not placement.successful:
            reasons.append("distributed value placement is incomplete or inconsistent")
        if not topology_available:
            reasons.append("bound execution topology devices are unavailable")
        if not topology_supported:
            reasons.append("mixed execution currently requires a one-axis slab mesh")
        if not spectral_abi:
            reasons.append(
                "spectral state shape or coefficient/accumulation/component precision "
                "does not match the mixed physics ABI"
            )
        if not capacity_matches:
            reasons.append(
                "particle support does not exactly fill fixed distributed capacity"
            )
        if not collective.successful:
            reasons.append("required executable distributed collectives are unavailable")
        if not checkpoint.successful:
            reasons.append("distributed checkpoint coverage or capacity is incomplete")
        if executable:
            status: DistributedMixedPreparationStatus = "prepared"
        elif not resources_ok:
            status = "infeasible-resources"
        elif not placement.successful:
            status = "placement-incomplete"
        elif not checkpoint.capacity_sufficient:
            status = "checkpoint-incomplete"
        elif not topology_supported or not capacity_matches or not shape_matches:
            status = "layout-mismatch"
        else:
            status = "missing-distributed-primitives"
        evidence_id = canonical_fingerprint(
            {
                "kind": "distributed-mixed-preparation-evidence",
                "plan": self.plan_id,
                "placement": placement.evidence_id,
                "collective": collective.evidence_id,
                "checkpoint": checkpoint.evidence_id,
                "resources_ok": resources_ok,
                "executable": executable,
                "status": status,
                "reasons": reasons,
            }
        )
        evidence = DistributedMixedPreparationEvidence(
            placement,
            collective,
            checkpoint,
            self.feasibility,
            self.spectral.report.resource.accepted,
            executable,
            status,
            tuple(reasons),
            self.plan_id,
            evidence_id,
        )
        prepared = (
            PreparedDistributedMixedExecution(self, evidence) if executable else None
        )
        return DistributedMixedPreparationResult(
            evidence,
            jnp.asarray(executable),
            None if prepared is None else prepared.execution_id,
            prepared,
        )


class PreparedDistributedMixedExecution(StrictModule):
    """Executable sharded mixed evolution bound to one mesh and physics identity."""

    plan: DistributedMixedExecutionPlan
    evidence: DistributedMixedPreparationEvidence
    particle_runtime: PreparedDistributedParticleRuntime
    field_sharding: NamedSharding = eqx.field(static=True)
    modal_sharding: NamedSharding = eqx.field(static=True)
    gas_sharding: NamedSharding | None = eqx.field(static=True)
    replicated_sharding: NamedSharding = eqx.field(static=True)
    checkpoint_physics_id: str = eqx.field(static=True)
    checkpoint_schema_id: str = eqx.field(static=True)
    checkpoint_numeric_id: str = eqx.field(static=True)
    checkpoint_execution_id: str = eqx.field(static=True)
    checkpoint_unpadded_bytes: int = eqx.field(static=True)
    checkpoint_payload_bytes: int = eqx.field(static=True)
    checkpoint_payload_alignment: int = eqx.field(static=True)
    execution_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: DistributedMixedExecutionPlan,
        evidence: DistributedMixedPreparationEvidence,
        /,
    ):
        if not evidence.executable or evidence.plan_id != plan.plan_id:
            raise ValueError(
                "Prepared distributed mixed execution requires admitted evidence."
            )
        topology = plan.spectral.topology
        axis = topology.mesh_axis_names[0]
        particle_plan = DistributedParticleRuntimePlan(
            plan.particles,
            plan.mixed.plan.particles.box_size,
            send_capacity=plan.particle_send_capacity,
            receive_capacity=plan.particle_receive_capacity,
            ghost_capacity=plan.particle_ghost_capacity,
            ghost_width=plan.particle_ghost_width,
        )
        particle_runtime = particle_plan.prepare(topology.mesh, axis)
        physical = plan.spectral.physical_layout.sharding(topology)
        modal = plan.spectral.modal_layout.sharding(topology)
        gas_sharding = None
        if isinstance(plan.mixed, PreparedWaveParticleGasCosmology):
            gas_sharding = NamedSharding(
                topology.mesh,
                PartitionSpec(*plan.spectral.physical_layout.partition, None),
            )
        self.plan = plan
        self.evidence = evidence
        self.particle_runtime = particle_runtime
        self.field_sharding = physical
        self.modal_sharding = modal
        self.gas_sharding = gas_sharding
        self.replicated_sharding = NamedSharding(topology.mesh, PartitionSpec())
        wave = plan.mixed.plan.wave
        support = plan.mixed.plan.particles.particles
        gas_schema = None
        if isinstance(plan.mixed, PreparedWaveParticleGasCosmology):
            gas_schema = {
                "shape": list(
                    plan.mixed.plan.gas.dynamics.discretization.cell_shape
                    + (plan.mixed.plan.gas.dynamics.discretization.component_count,)
                ),
                "dtype": np.dtype(
                    plan.mixed.plan.gas.dynamics.discretization.cell_volumes.dtype
                ).str,
            }
        checkpoint_physics_id = plan.mixed.prepared_id
        checkpoint_schema_id = canonical_fingerprint(
            {
                "kind": "distributed-mixed-checkpoint-schema",
                "wave_shape": list(wave.discretization.physical_shape),
                "particle_capacity": support.capacity,
                "particle_dimension": support.ambient_dimension,
                "gas": gas_schema,
                "payload_order": [
                    "wave",
                    "wave_scale_factor",
                    "particle_positions",
                    "particle_momenta",
                    "particle_masses",
                    "particle_stable_ids",
                    "particle_logical_slots",
                    "particle_active_mask",
                    "particle_rng_counters",
                    "particle_scale_factor",
                    "particle_owner",
                    *(("gas", "gas_scale_factor") if gas_schema is not None else ()),
                ],
            }
        )
        checkpoint_numeric_id = canonical_fingerprint(
            {
                "kind": "distributed-mixed-checkpoint-numeric",
                "wave_coefficient_dtype": np.dtype(plan.spectral.coefficient_dtype).str,
                "wave_accumulation_dtype": np.dtype(plan.spectral.accumulation_dtype).str,
                "particle_dtype": np.dtype(support.plan.coordinate_dtype).str,
                "particle_mass_dtype": np.dtype(support.masses.dtype).str,
                "gas_dtype": None if gas_schema is None else gas_schema["dtype"],
            }
        )
        checkpoint_execution_id = canonical_fingerprint(
            {
                "kind": "distributed-mixed-topology-neutral-checkpoint",
                "physics": checkpoint_physics_id,
                "schema": checkpoint_schema_id,
                "numeric": checkpoint_numeric_id,
            }
        )
        checkpoint_unpadded_bytes = plan.checkpoint_unpadded_bytes
        self.checkpoint_physics_id = checkpoint_physics_id
        self.checkpoint_schema_id = checkpoint_schema_id
        self.checkpoint_numeric_id = checkpoint_numeric_id
        self.checkpoint_execution_id = checkpoint_execution_id
        self.checkpoint_unpadded_bytes = checkpoint_unpadded_bytes
        self.checkpoint_payload_bytes = plan.checkpoint_payload_bytes
        self.checkpoint_payload_alignment = plan.checkpoint_payload_alignment
        self.execution_id = canonical_fingerprint(
            {
                "kind": "prepared-distributed-mixed-execution",
                "plan": plan.plan_id,
                "evidence": evidence.evidence_id,
                "particle_runtime": particle_runtime.runtime_id,
                "physics": plan.mixed.prepared_id,
                "topology": topology.topology_id,
                "checkpoint_physics": checkpoint_physics_id,
                "checkpoint_schema": checkpoint_schema_id,
                "checkpoint_numeric": checkpoint_numeric_id,
                "checkpoint_execution": checkpoint_execution_id,
                "checkpoint_shard_plan": evidence.checkpoint.shard_plan_id,
            }
        )

    @property
    def axis_name(self) -> str:
        return self.plan.spectral.topology.mesh_axis_names[0]

    @property
    def spatial_rank(self) -> int:
        return len(self.plan.spectral.spatial_shape)

    def _field_sharding_for(self, ndim: int, /) -> NamedSharding:
        partition = self.plan.spectral.physical_layout.partition
        return NamedSharding(
            self.plan.spectral.topology.mesh,
            PartitionSpec(*partition, *(None for _ in range(ndim - len(partition)))),
        )

    def _global_reduce(self, values: Array, /) -> Array:
        array = jax.device_put(jnp.asarray(values), self._field_sharding_for(values.ndim))
        axes = tuple(range(self.spatial_rank))
        local = lambda value: jax.lax.psum(jnp.sum(value, axis=axes), self.axis_name)
        return jax.shard_map(
            local,
            mesh=self.plan.spectral.topology.mesh,
            in_specs=self._field_sharding_for(values.ndim).spec,
            out_specs=PartitionSpec(),
            check_vma=False,
        )(array)

    def _global_max(self, values: Array, sharding: NamedSharding, /) -> Array:
        placed = jax.device_put(jnp.asarray(values), sharding)
        return jax.shard_map(
            lambda local: jax.lax.pmax(jnp.max(local), self.axis_name),
            mesh=self.plan.spectral.topology.mesh,
            in_specs=sharding.spec,
            out_specs=PartitionSpec(),
            check_vma=False,
        )(placed)

    def _global_success(self, value: Array, /) -> Array:
        placed = jax.device_put(
            jnp.asarray(value, dtype=jnp.int32), self.replicated_sharding
        )
        return (
            jax.shard_map(
                lambda local: jax.lax.pmin(local, self.axis_name),
                mesh=self.plan.spectral.topology.mesh,
                in_specs=PartitionSpec(),
                out_specs=PartitionSpec(),
                check_vma=False,
            )(placed)
            == 1
        )

    def initialize(
        self,
        state: WaveParticleCosmologyState | WaveParticleGasCosmologyState,
        /,
        *,
        rng_counters: Array | None = None,
    ) -> DistributedMixedState:
        include_gas = isinstance(self.plan.mixed, PreparedWaveParticleGasCosmology)
        if include_gas and not isinstance(state, WaveParticleGasCosmologyState):
            raise TypeError(
                "This distributed execution requires a wave-particle-gas state."
            )
        if not include_gas and not isinstance(state, WaveParticleCosmologyState):
            raise TypeError("This distributed execution requires a wave-particle state.")
        wave_owner = self.plan.mixed.plan.wave
        coefficient_dtype = np.dtype(self.plan.spectral.coefficient_dtype)
        real_dtype = np.dtype(
            jnp.empty((), dtype=jnp.dtype(coefficient_dtype)).real.dtype
        )
        component_dtypes = (
            np.dtype(state.wave.psi.dtype),
            np.dtype(state.particles.positions.dtype),
            np.dtype(state.particles.canonical_momenta.dtype),
        )
        if component_dtypes != (coefficient_dtype, real_dtype, real_dtype):
            raise TypeError(
                "Distributed mixed state dtypes must exactly match the prepared wave/particle precision ABI."
            )
        scales = [
            jnp.asarray(state.wave.scale_factor, dtype=jnp.dtype(real_dtype)),
            jnp.asarray(state.particles.scale_factor, dtype=jnp.dtype(real_dtype)),
        ]
        if include_gas:
            assert isinstance(state, WaveParticleGasCosmologyState)
            if np.dtype(state.gas.cell_average.dtype) != real_dtype:
                raise TypeError(
                    "Distributed gas dtype must exactly match the prepared real precision."
                )
            scales.append(
                jnp.asarray(state.gas.scale_factor, dtype=jnp.dtype(real_dtype))
            )
        if any(scale.shape != () for scale in scales):
            raise ValueError("Distributed mixed component scale factors must be scalar.")
        expected_scale = wave_owner.scale_factors[0].astype(jnp.dtype(real_dtype))
        tolerance = (
            32.0
            * jnp.finfo(expected_scale.dtype).eps
            * jnp.maximum(jnp.abs(expected_scale), 1.0)
        )
        scale_invalid = jnp.asarray(False)
        for scale in scales:
            scale_invalid = (
                scale_invalid
                | ~jnp.isfinite(scale)
                | (scale <= 0.0)
                | (jnp.abs(scale - expected_scale) > tolerance)
            )
        expected_scale = eqx.error_if(
            expected_scale,
            scale_invalid,
            "Distributed mixed components must be finite, positive, aligned, and at the first scheduled scale factor.",
        )
        support = self.plan.mixed.plan.particles.particles
        particles = self.particle_runtime.initialize(
            state.particles.positions,
            state.particles.canonical_momenta,
            support.masses,
            support.particle_ids,
            support.active_mask,
            expected_scale,
            rng_counters=rng_counters,
        )
        wave = WaveDarkMatterState(
            jax.device_put(state.wave.psi, self.field_sharding),
            jax.device_put(expected_scale, self.replicated_sharding),
        )
        gas = None
        if include_gas:
            assert isinstance(state, WaveParticleGasCosmologyState)
            assert self.gas_sharding is not None
            gas = ComovingEulerState(
                jax.device_put(state.gas.cell_average, self.gas_sharding),
                jax.device_put(expected_scale, self.replicated_sharding),
            )
        return DistributedMixedState(wave, particles, gas, self.execution_id)

    def _require_state(self, state: DistributedMixedState, /) -> None:
        if not isinstance(state, DistributedMixedState):
            raise TypeError("state must be DistributedMixedState.")
        if state.execution_id != self.execution_id:
            raise ValueError("Distributed mixed state belongs to a different execution.")
        self.particle_runtime._require_state(state.particles)
        if isinstance(self.plan.mixed, PreparedWaveParticleGasCosmology) != (
            state.gas is not None
        ):
            raise ValueError("Distributed mixed state component set changed.")

    def assemble_density(
        self, state: DistributedMixedState, /
    ) -> DistributedMixedDensity:
        """Deposit each stable particle once and add all sharded density owners."""

        self._require_state(state)
        wave_owner = self.plan.mixed.plan.wave
        logical = self.particle_runtime.logical_arrays(state.particles)
        routes = self.plan.mixed.density.particle_gravity.transfer.build(
            logical["positions"], active_mask=logical["active_mask"]
        )
        deposited = self.plan.mixed.density.particle_gravity.transfer.deposit_content(
            routes, logical["masses"]
        )
        particle_density = jax.device_put(deposited.density, self.field_sharding)
        wave_density = jax.device_put(
            wave_owner.boson_mass * jnp.abs(state.wave.psi) ** 2,
            self.field_sharding,
        )
        gas_density = (
            jnp.zeros_like(wave_density)
            if state.gas is None
            else jax.device_put(state.gas.cell_average[..., 0], self.field_sharding)
        )
        total = jax.device_put(
            wave_density + particle_density + gas_density, self.field_sharding
        )
        volumes = jax.device_put(
            self.plan.mixed.density.cell_volumes.astype(total.dtype), self.field_sharding
        )
        wave_mass = self._global_reduce(wave_density * volumes)
        particle_mass = self._global_reduce(particle_density * volumes)
        gas_mass = self._global_reduce(gas_density * volumes)
        component_mass = jnp.stack((wave_mass, particle_mass, gas_mass))
        source_mass = jnp.sum(jnp.where(logical["active_mask"], logical["masses"], 0.0))
        ids = jnp.where(
            logical["active_mask"],
            logical["stable_ids"],
            jnp.iinfo(logical["stable_ids"].dtype).max,
        )
        sorted_ids = jnp.sort(ids)
        ids_unique = ~jnp.any(
            (sorted_ids[1:] == sorted_ids[:-1])
            & (sorted_ids[1:] != jnp.iinfo(sorted_ids.dtype).max)
        )
        support = self.plan.mixed.plan.particles.particles
        expected_active = support.active_mask
        expected_ids = support.particle_ids
        expected_masses = support.masses.astype(logical["masses"].dtype)
        identity_continuity = (
            jnp.all(logical["active_mask"] == expected_active)
            & jnp.all(
                jnp.where(
                    expected_active,
                    logical["stable_ids"] == expected_ids,
                    True,
                )
            )
            & jnp.all(
                jnp.where(
                    expected_active,
                    logical["masses"] == expected_masses,
                    True,
                )
            )
        )
        finite = (
            jnp.all(jnp.isfinite(total))
            & jnp.all(total >= 0.0)
            & jnp.all(jnp.isfinite(component_mass))
        )
        tolerance = (
            256.0 * jnp.finfo(total.dtype).eps * jnp.maximum(jnp.abs(source_mass), 1.0)
        )
        successful = self._global_success(
            deposited.successful
            & routes.successful
            & ids_unique
            & identity_continuity
            & finite
            & (jnp.abs(particle_mass - source_mass) <= tolerance)
        )
        return DistributedMixedDensity(
            wave_density,
            particle_density,
            gas_density,
            total,
            component_mass,
            jnp.sum(component_mass),
            source_mass,
            particle_mass,
            routes,
            finite,
            successful,
            self.execution_id,
        )

    def solve_gravity(
        self, state: DistributedMixedState, /
    ) -> DistributedMixedGravityResult:
        """Apply one distributed discrete-periodic Poisson owner to total density."""

        density = self.assemble_density(state)
        volumes = jax.device_put(
            self.plan.mixed.density.cell_volumes.astype(density.total_density.dtype),
            self.field_sharding,
        )
        volume = self._global_reduce(volumes)
        mean_density = self._global_reduce(density.total_density * volumes) / volume
        contrast = density.total_density - mean_density
        coupling = jnp.asarray(
            self.plan.mixed.gravity.gravitational_constant,
            dtype=density.total_density.dtype,
        )
        source = 4.0 * jnp.pi * coupling * contrast
        source_coefficients = self.plan.spectral.to_modal(
            source.astype(jnp.dtype(self.plan.spectral.coefficient_dtype))
        )
        eigenvalues = self.plan.mixed.gravity.particle_gravity.gravity.poisson.diagonalization.modal_values
        eigenvalues = jax.device_put(eigenvalues, self.modal_sharding)
        tolerance = (
            64.0
            * jnp.finfo(eigenvalues.dtype).eps
            * jnp.maximum(jnp.max(jnp.abs(eigenvalues)), 1.0)
        )
        potential_coefficients = jnp.where(
            jnp.abs(eigenvalues) > tolerance,
            source_coefficients / eigenvalues,
            jnp.zeros_like(source_coefficients),
        )
        potential = jax.device_put(
            jnp.real(self.plan.spectral.to_physical(potential_coefficients)),
            self.field_sharding,
        )
        gravity_owner = self.plan.mixed.gravity.particle_gravity.gravity
        face_gradients = gravity_owner.diffusion.fluxes(potential, 1.0, None)
        cell_components = tuple(
            -0.5 * (gradient + jnp.roll(gradient, -1, axis=axis))
            for axis, gradient in enumerate(face_gradients)
        )
        cell_acceleration = jax.device_put(
            jnp.stack(cell_components, axis=-1),
            self._field_sharding_for(self.spatial_rank + 1),
        )
        gathered = self.plan.mixed.density.particle_gravity.transfer.gather(
            density.particle_routes, cell_acceleration
        )
        logical = self.particle_runtime.logical_arrays(state.particles)
        particle_acceleration = jnp.where(
            logical["active_mask"][:, None], gathered.values, 0.0
        )
        owner_acceleration = self.particle_runtime.owner_order(
            particle_acceleration, state.particles
        )
        laplacian = gravity_owner.diffusion.mv(potential)
        residual = laplacian - source
        residual_norm = jnp.sqrt(
            jnp.maximum(self._global_reduce(residual * residual * volumes), 0.0)
        )
        source_norm = jnp.sqrt(
            jnp.maximum(self._global_reduce(source * source * volumes), 0.0)
        )
        relative_residual = jnp.where(
            source_norm > 0.0,
            residual_norm / jnp.where(source_norm > 0.0, source_norm, 1.0),
            residual_norm,
        )
        source_integral = self._global_reduce(contrast * volumes)
        gauge_defect = jnp.abs(self._global_reduce(potential * volumes) / volume)
        wave_force = self._global_reduce(
            density.wave_density[..., None] * volumes[..., None] * cell_acceleration
        )
        particle_force = jnp.sum(
            logical["masses"][:, None] * particle_acceleration, axis=0
        )
        gas_force = self._global_reduce(
            density.gas_density[..., None] * volumes[..., None] * cell_acceleration
        )
        component_force = jnp.stack((wave_force, particle_force, gas_force))
        total_force = jnp.sum(component_force, axis=0)
        finite = (
            jnp.all(jnp.isfinite(potential))
            & jnp.all(jnp.isfinite(cell_acceleration))
            & jnp.all(jnp.isfinite(particle_acceleration))
            & jnp.isfinite(relative_residual)
            & jnp.isfinite(gauge_defect)
            & jnp.all(jnp.isfinite(component_force))
        )
        policy = self.plan.mixed.plan.wave.step_policy
        successful = self._global_success(
            density.successful
            & jnp.all(gathered.support | ~logical["active_mask"])
            & finite
            & (relative_residual <= policy.poisson_relative_tolerance)
            & (gauge_defect <= policy.zero_mode_absolute_tolerance)
        )
        return DistributedMixedGravityResult(
            density,
            potential,
            cell_acceleration,
            particle_acceleration,
            owner_acceleration,
            mean_density,
            source_integral,
            relative_residual,
            gauge_defect,
            component_force,
            total_force,
            finite,
            successful,
            self.execution_id,
        )

    def _wave_kick(
        self,
        state: WaveDarkMatterState,
        potential: Array,
        start: Array,
        end: Array,
        fraction: float,
        /,
    ) -> WaveDarkMatterState:
        wave = self.plan.mixed.plan.wave
        kick = wave.background.kick_factor(start, end).astype(state.psi.real.dtype)
        phase = (
            fraction * wave.boson_mass * kick * potential / wave.reduced_planck_constant
        )
        return WaveDarkMatterState(
            jax.device_put(state.psi * jnp.exp(-1j * phase), self.field_sharding),
            state.scale_factor,
        )

    def _wave_drift(
        self, state: WaveDarkMatterState, start: Array, end: Array, /
    ) -> WaveDarkMatterState:
        wave = self.plan.mixed.plan.wave
        drift = wave.background.drift_factor(start, end).astype(state.psi.real.dtype)
        coefficients = self.plan.spectral.to_modal(state.psi)
        wavenumber_squared = jax.device_put(wave.wavenumber_squared, self.modal_sharding)
        coefficient = 0.5 * wave.reduced_planck_constant * drift / wave.boson_mass
        updated = self.plan.spectral.to_physical(
            coefficients * jnp.exp(-1j * coefficient * wavenumber_squared)
        )
        return WaveDarkMatterState(
            jax.device_put(updated, self.field_sharding),
            jax.device_put(end, self.replicated_sharding),
        )

    def advance(
        self,
        state: DistributedMixedState,
        end_scale_factor: Array,
        args: Any = None,
        /,
    ) -> DistributedMixedStepResult:
        """Advance one KDK interval and atomically commit all components."""

        self._require_state(state)
        wave = self.plan.mixed.plan.wave
        particles = self.plan.mixed.plan.particles
        start = state.wave.scale_factor
        end = jnp.asarray(end_scale_factor, dtype=start.dtype)
        gravity_0 = self.solve_gravity(state)
        first_wave = self._wave_kick(state.wave, gravity_0.potential, start, end, 0.5)
        drifted_wave = self._wave_drift(first_wave, start, end)
        logical_0 = self.particle_runtime.logical_arrays(state.particles)
        logical_state = CosmologicalParticleState(
            logical_0["positions"], logical_0["momenta"], state.particles.scale_factor
        )
        proposal = particles.propose(
            wave.background,
            logical_state,
            end,
            gravity_0.particle_acceleration,
        )
        owner_positions = self.particle_runtime.owner_order(
            proposal.positions, state.particles
        )
        owner_half_momenta = self.particle_runtime.owner_order(
            proposal.half_momenta, state.particles
        )
        next_rng = state.particles.rng_counters + state.particles.active_mask.astype(
            jnp.uint64
        )
        migration = self.particle_runtime.migrate(
            state.particles,
            owner_positions,
            owner_half_momenta,
            proposed_rng_counters=next_rng,
            end_scale_factor=end,
        )
        predicted_gas = state.gas
        predicted_gas_success = jnp.asarray(True)
        corrected_gas_success = jnp.asarray(True)
        homogeneous_gas_success = jnp.asarray(True)
        if isinstance(self.plan.mixed, PreparedWaveParticleGasCosmology):
            assert state.gas is not None
            predicted_gas, predicted_evidence = self.plan.mixed.plan.gas.advance(
                wave.background,
                state.gas,
                end,
                gravity_0.cell_acceleration,
                gravity_0.cell_acceleration,
                args,
            )
            assert self.gas_sharding is not None
            predicted_gas = ComovingEulerState(
                jax.device_put(predicted_gas.cell_average, self.gas_sharding),
                jax.device_put(predicted_gas.scale_factor, self.replicated_sharding),
            )
            predicted_gas_success = predicted_evidence.successful
        predicted = DistributedMixedState(
            drifted_wave, migration.state, predicted_gas, self.execution_id
        )
        predicted_gravity = self.solve_gravity(predicted)
        endpoint_gas = predicted_gas
        if isinstance(self.plan.mixed, PreparedWaveParticleGasCosmology):
            assert state.gas is not None
            endpoint_gas, corrected_evidence = self.plan.mixed.plan.gas.advance(
                wave.background,
                state.gas,
                end,
                gravity_0.cell_acceleration,
                predicted_gravity.cell_acceleration,
                args,
            )
            assert self.gas_sharding is not None
            endpoint_gas = ComovingEulerState(
                jax.device_put(endpoint_gas.cell_average, self.gas_sharding),
                jax.device_put(endpoint_gas.scale_factor, self.replicated_sharding),
            )
            corrected_gas_success = corrected_evidence.successful
            zero_gravity = jnp.zeros_like(gravity_0.cell_acceleration)
            _, homogeneous_evidence = self.plan.mixed.plan.gas.advance(
                wave.background,
                state.gas,
                end,
                zero_gravity,
                zero_gravity,
                args,
            )
            homogeneous_gas_success = homogeneous_evidence.successful
            endpoint = DistributedMixedState(
                drifted_wave, migration.state, endpoint_gas, self.execution_id
            )
            gravity_1 = self.solve_gravity(endpoint)
        else:
            gravity_1 = predicted_gravity
        final_wave = self._wave_kick(drifted_wave, gravity_1.potential, start, end, 0.5)
        logical_acceleration_1 = gravity_1.particle_acceleration
        final_logical_particles, particle_evidence = particles.complete(
            logical_state, proposal, logical_acceleration_1
        )
        final_owner_momenta = self.particle_runtime.owner_order(
            final_logical_particles.canonical_momenta, migration.state
        )
        final_particles = DistributedParticleState(
            migration.state.positions,
            final_owner_momenta,
            migration.state.masses,
            migration.state.stable_ids,
            migration.state.logical_slots,
            migration.state.active_mask,
            migration.state.rng_counters,
            migration.state.scale_factor,
            migration.state.owner,
            migration.state.runtime_id,
        )
        candidate = DistributedMixedState(
            final_wave, final_particles, endpoint_gas, self.execution_id
        )
        final_gravity = gravity_1
        mass_defect = final_gravity.density.total_mass - gravity_0.density.total_mass
        mass_factor = (
            512.0 * self.plan.mixed.plan.gas.substeps
            if isinstance(self.plan.mixed, PreparedWaveParticleGasCosmology)
            else 256.0
        )
        tolerance = (
            mass_factor
            * jnp.finfo(mass_defect.dtype).eps
            * jnp.maximum(jnp.abs(gravity_0.density.total_mass), 1.0)
        )
        weights = jax.device_put(
            wave.discretization.quadrature_weights.astype(state.wave.psi.real.dtype),
            self.field_sharding,
        )
        norm_0 = self._global_reduce(jnp.abs(state.wave.psi) ** 2 * weights)
        norm_1 = self._global_reduce(jnp.abs(final_wave.psi) ** 2 * weights)
        norm_error = jnp.abs(norm_1 - norm_0) / jnp.maximum(norm_0, 1.0e-30)
        first_coefficients = self.plan.spectral.to_modal(first_wave.psi)
        first_power = jnp.abs(first_coefficients) ** 2
        maximum_power = self._global_max(first_power, self.modal_sharding)
        occupied = (
            first_power >= wave.step_policy.relative_amplitude_floor * maximum_power
        )
        drift_factor = wave.background.drift_factor(start, end).astype(first_power.dtype)
        modal_wavenumber_squared = jax.device_put(
            wave.wavenumber_squared, self.modal_sharding
        )
        kinetic_phase_field = (
            0.5
            * wave.reduced_planck_constant
            * modal_wavenumber_squared
            * drift_factor
            / wave.boson_mass
        )
        kinetic_phase = self._global_max(
            jnp.where(occupied, jnp.abs(kinetic_phase_field), 0.0),
            self.modal_sharding,
        )
        kick_factor = jnp.abs(
            wave.background.kick_factor(start, end).astype(first_power.dtype)
        )
        maximum_potential = jnp.maximum(
            self._global_max(jnp.abs(gravity_0.potential), self.field_sharding),
            self._global_max(jnp.abs(gravity_1.potential), self.field_sharding),
        )
        potential_phase = (
            0.5
            * wave.boson_mass
            * kick_factor
            * maximum_potential
            / wave.reduced_planck_constant
        )
        phase_resolved = (kinetic_phase <= wave.step_policy.maximum_phase_radians) & (
            potential_phase <= wave.step_policy.maximum_phase_radians
        )
        scale_tolerance = 32.0 * jnp.finfo(end.dtype).eps * jnp.maximum(jnp.abs(end), 1.0)
        time_consistent = (
            (jnp.abs(final_wave.scale_factor - end) <= scale_tolerance)
            & (jnp.abs(final_particles.scale_factor - end) <= scale_tolerance)
            & (
                jnp.asarray(True)
                if endpoint_gas is None
                else jnp.abs(endpoint_gas.scale_factor - end) <= scale_tolerance
            )
        )
        successful = self._global_success(
            gravity_0.successful
            & proposal.successful
            & migration.evidence.successful
            & predicted_gas_success
            & corrected_gas_success
            & homogeneous_gas_success
            & predicted_gravity.successful
            & gravity_1.successful
            & particle_evidence.successful
            & final_gravity.successful
            & (norm_error <= wave.step_policy.norm_relative_tolerance)
            & phase_resolved
            & time_consistent
            & (jnp.abs(mass_defect) <= tolerance)
        )

        def select(candidate_value, previous_value):
            return jnp.where(successful, candidate_value, previous_value)

        accepted_gas = None
        if state.gas is not None:
            assert candidate.gas is not None
            accepted_gas = ComovingEulerState(
                select(candidate.gas.cell_average, state.gas.cell_average),
                select(candidate.gas.scale_factor, state.gas.scale_factor),
            )
        accepted_particles = DistributedParticleState(
            select(candidate.particles.positions, state.particles.positions),
            select(candidate.particles.momenta, state.particles.momenta),
            select(candidate.particles.masses, state.particles.masses),
            select(candidate.particles.stable_ids, state.particles.stable_ids),
            select(candidate.particles.logical_slots, state.particles.logical_slots),
            select(candidate.particles.active_mask, state.particles.active_mask),
            select(candidate.particles.rng_counters, state.particles.rng_counters),
            select(candidate.particles.scale_factor, state.particles.scale_factor),
            select(candidate.particles.owner, state.particles.owner),
            state.particles.runtime_id,
        )
        accepted = DistributedMixedState(
            WaveDarkMatterState(
                select(candidate.wave.psi, state.wave.psi),
                select(candidate.wave.scale_factor, state.wave.scale_factor),
            ),
            accepted_particles,
            accepted_gas,
            self.execution_id,
        )
        return DistributedMixedStepResult(
            accepted,
            final_gravity,
            migration,
            gravity_0.density.total_mass,
            final_gravity.density.total_mass,
            mass_defect,
            kinetic_phase,
            potential_phase,
            phase_resolved,
            homogeneous_gas_success,
            successful,
            self.execution_id,
        )

    def rollout(
        self, state: DistributedMixedState, args: Any = None, /
    ) -> DistributedMixedEvolutionResult:
        """Execute the bound wave schedule; the first failed step stops all mutation."""

        self._require_state(state)
        scale_factors = self.plan.mixed.plan.wave.scale_factors
        current = state
        active = jnp.asarray(True)
        accepted = jnp.asarray(0, dtype=jnp.int32)
        maximum_defect = jnp.asarray(0.0, dtype=state.wave.psi.real.dtype)
        initial_mass = self.solve_gravity(state).density.total_mass
        final_mass = initial_mass
        for index in range(1, scale_factors.size):
            result = self.advance(current, scale_factors[index], args)
            step_success = active & result.successful
            current = jax.tree.map(
                lambda new, old: (
                    jnp.where(step_success, new, old) if eqx.is_array(new) else new
                ),
                result.state,
                current,
            )
            active = step_success
            accepted = accepted + step_success.astype(jnp.int32)
            maximum_defect = jnp.maximum(
                maximum_defect, jnp.abs(result.mass_balance_defect)
            )
            final_mass = jnp.where(step_success, result.final_mass, final_mass)
        successful = active & (accepted == scale_factors.size - 1)
        return DistributedMixedEvolutionResult(
            current,
            accepted,
            initial_mass,
            final_mass,
            maximum_defect,
            successful,
            self.execution_id,
        )

    def checkpoint_tree(self, state: DistributedMixedState, /) -> dict[str, Array]:
        """Pack the complete sharded state into one bounded byte-addressed relation."""

        self._require_state(state)
        arrays = [
            state.wave.psi,
            state.wave.scale_factor,
            state.particles.positions,
            state.particles.momenta,
            state.particles.masses,
            state.particles.stable_ids,
            state.particles.logical_slots,
            state.particles.active_mask,
            state.particles.rng_counters,
            state.particles.scale_factor,
            state.particles.owner,
        ]
        if state.gas is not None:
            arrays.extend((state.gas.cell_average, state.gas.scale_factor))

        def encode(value):
            return jnp.asarray(value).reshape((-1,)).view(jnp.uint8)

        payload = jnp.concatenate(tuple(encode(value) for value in arrays), axis=0)
        if payload.size != self.checkpoint_unpadded_bytes:
            raise RuntimeError(
                "Distributed mixed checkpoint schema byte accounting changed."
            )
        padding = self.checkpoint_payload_bytes - self.checkpoint_unpadded_bytes
        if padding:
            payload = jnp.concatenate(
                (payload, jnp.zeros((padding,), dtype=jnp.uint8)), axis=0
            )
        if (
            payload.size != self.checkpoint_payload_bytes
            or payload.size % self.plan.spectral.topology.device_count
        ):
            raise RuntimeError(
                "Topology-neutral checkpoint alignment does not admit this mesh."
            )
        payload_sharding = NamedSharding(
            self.plan.spectral.topology.mesh, PartitionSpec(self.axis_name)
        )
        return {"payload": jax.device_put(payload, payload_sharding)}

    def restore_checkpoint(
        self,
        repository: Any,
        manifest: Any,
        prototype: DistributedMixedState,
        /,
    ) -> DistributedMixedState:
        """Restore only a complete, topology-neutral, identity-exact checkpoint."""

        if not isinstance(manifest, CheckpointManifest):
            raise TypeError("manifest must be CheckpointManifest.")
        if (
            not manifest.complete
            or manifest.analysis_plan_id != self.checkpoint_schema_id
            or manifest.numeric_revision_id != self.checkpoint_numeric_id
            or manifest.execution_plan_id != self.checkpoint_execution_id
            or manifest.diagnostic_ids != (self.checkpoint_physics_id,)
        ):
            raise ValueError(
                "Distributed mixed checkpoint is incomplete or its physics/schema/numeric identity does not match."
            )
        self._require_state(prototype)
        payload_sharding = NamedSharding(
            self.plan.spectral.topology.mesh, PartitionSpec(self.axis_name)
        )
        payload = restore_global_array_from_checkpoint(
            repository,
            manifest,
            "['payload']",
            payload_sharding,
        )
        if payload.size != self.checkpoint_payload_bytes:
            raise ValueError(
                "Distributed mixed checkpoint payload has the wrong topology-neutral archive length."
            )
        payload = eqx.error_if(
            payload,
            jnp.any(payload[self.checkpoint_unpadded_bytes :] != 0),
            "Distributed mixed checkpoint padding is nonzero.",
        )
        offset = 0

        def take(template):
            nonlocal offset
            shape = tuple(template.shape)
            dtype = np.dtype(template.dtype)
            count = prod(shape) if shape else 1
            byte_count = count * dtype.itemsize
            chunk = payload[offset : offset + byte_count]
            offset += byte_count
            decoded = chunk.view(jnp.dtype(dtype)).reshape(shape)
            return jax.device_put(decoded, template.sharding)

        wave_values = take(prototype.wave.psi)
        wave_scale = take(prototype.wave.scale_factor)
        restored_owner_order = DistributedParticleState(
            take(prototype.particles.positions),
            take(prototype.particles.momenta),
            take(prototype.particles.masses),
            take(prototype.particles.stable_ids),
            take(prototype.particles.logical_slots),
            take(prototype.particles.active_mask),
            take(prototype.particles.rng_counters),
            take(prototype.particles.scale_factor),
            take(prototype.particles.owner),
            self.particle_runtime.runtime_id,
        )
        logical = self.particle_runtime.logical_arrays(restored_owner_order)
        particles = self.particle_runtime.initialize(
            logical["positions"],
            logical["momenta"],
            logical["masses"],
            logical["stable_ids"],
            logical["active_mask"],
            restored_owner_order.scale_factor,
            rng_counters=logical["rng_counters"],
        )
        gas = None
        if prototype.gas is not None:
            gas = ComovingEulerState(
                take(prototype.gas.cell_average),
                take(prototype.gas.scale_factor),
            )
        if offset != self.checkpoint_unpadded_bytes:
            raise ValueError("Distributed mixed checkpoint payload coverage is inexact.")
        return DistributedMixedState(
            WaveDarkMatterState(wave_values, wave_scale),
            particles,
            gas,
            self.execution_id,
        )


__all__ = [
    "DistributedMixedCheckpointEvidence",
    "DistributedMixedCollectiveEvidence",
    "DistributedMixedDensity",
    "DistributedMixedEvolutionResult",
    "DistributedMixedExecutionPlan",
    "DistributedMixedGravityResult",
    "DistributedMixedPlacementEvidence",
    "DistributedMixedPreparationEvidence",
    "DistributedMixedPreparationResult",
    "DistributedMixedPreparationStatus",
    "DistributedMixedState",
    "DistributedMixedStepResult",
    "PreparedDistributedMixedExecution",
]
