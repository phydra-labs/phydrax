#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""AMR ownership, conservative stress transfer, and restart for Einstein--Vlasov."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.amr import AMREntityTransferPlan
from ...discretization.particle._relativistic_stress_transfer import (
    RelativisticParticleState,
)
from ...lifecycle import CheckpointManifest, ProcessCheckpointPublication
from ...lifecycle._repository import ArtifactRepository
from ...metrix._adm_exchange import ADMGridGeometry, StressEnergyProjection
from ._checkpoint import (
    assemble_distributed_numerical_relativity_checkpoint,
    DistributedNumericalRelativityRestart,
    NumericalRelativityCheckpoint,
    NumericalRelativityCheckpointPlan,
    NumericalRelativityRestartPolicy,
    NumericalRelativityRestartState,
    publish_distributed_numerical_relativity_checkpoint,
    read_numerical_relativity_checkpoint,
    restore_distributed_numerical_relativity_checkpoint,
    write_numerical_relativity_checkpoint,
)
from ._distributed import PreparedNumericalRelativityAMRDistribution
from ._einstein_vlasov import EinsteinVlasovMatterState
from ._temporal import Z4cRuntimeState


StressTransferDirection: TypeAlias = Literal["prolong", "restrict"]


class EinsteinVlasovAMRStressEvidence(StrictModule):
    source_integrals: Array
    target_integrals: Array
    conservation_residual: Array
    symmetry_defect: Array
    finite: Array
    conservation_valid: Array
    source_valid: Array
    qualified: Array
    derivative_valid: Array
    transfer_id: str = eqx.field(static=True)


class EinsteinVlasovAMRStressTransferResult(StrictModule):
    projection: StressEnergyProjection
    evidence: EinsteinVlasovAMRStressEvidence


def _proper_volume(value: ArrayLike, shape: tuple[int, ...], dtype, role: str) -> Array:
    volume = jnp.asarray(value, dtype=dtype)
    if volume.shape == ():
        volume = jnp.broadcast_to(volume, shape)
    if volume.shape != shape:
        raise ValueError(f"{role} must be scalar or have shape {shape}.")
    return eqx.error_if(
        volume,
        jnp.any(~jnp.isfinite(volume) | (volume <= 0.0)),
        f"{role} must be finite and strictly positive.",
    )


def _stress_channels(projection: StressEnergyProjection, /) -> Array:
    return jnp.concatenate(
        (
            projection.energy_density[..., None],
            projection.momentum_covector,
            projection.stress_covariant.reshape(projection.leading_shape + (9,)),
        ),
        axis=-1,
    )


def _projection_from_channels(
    channels: Array,
    geometry: ADMGridGeometry,
    valid: Array,
    projection_defect: Array,
    conservation_defect: Array,
    projection_id: str,
    /,
) -> StressEnergyProjection:
    shape = geometry.leading_shape
    return StressEnergyProjection(
        channels[..., 0],
        channels[..., 1:4],
        channels[..., 4:13].reshape(shape + (3, 3)),
        geometry.active,
        valid,
        projection_defect,
        conservation_defect,
        snapshot_token=geometry.snapshot_token,
        geometry_lineage_id=geometry.geometry_lineage_id,
        convention_id=geometry.convention_id,
        scale_id=geometry.scale_id,
        topology_id=geometry.topology_id,
        projection_id=projection_id,
    )


class EinsteinVlasovAMRStressTransferPlan(StrictModule, NonTrainableState):
    """Conservative transfer of the existing 13-component ADM source projection."""

    transfer: AMREntityTransferPlan
    conservation_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        refinement_ratio: int = 2,
        /,
        *,
        conservation_tolerance: float = 1.0e-10,
    ):
        tolerance = float(conservation_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("Stress-transfer tolerance must be finite and nonnegative.")
        transfer = AMREntityTransferPlan.cells(3, refinement_ratio)
        self.transfer = transfer
        self.conservation_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "einstein-vlasov-amr-stress-transfer",
                "transfer": transfer.transfer_id,
                "conservation_tolerance": tolerance,
            }
        )

    def _evidence(
        self,
        source: StressEnergyProjection,
        target: StressEnergyProjection,
        source_volume: Array,
        target_volume: Array,
        /,
    ) -> EinsteinVlasovAMRStressEvidence:
        source_channels = _stress_channels(source)
        target_channels = _stress_channels(target)
        source_integrals = jnp.sum(
            jnp.where(source.active[..., None], source_channels, 0.0)
            * source_volume[..., None],
            axis=(0, 1, 2),
        )
        target_integrals = jnp.sum(
            jnp.where(target.active[..., None], target_channels, 0.0)
            * target_volume[..., None],
            axis=(0, 1, 2),
        )
        residual = target_integrals - source_integrals
        scale = jnp.maximum(
            jnp.maximum(jnp.abs(source_integrals), jnp.abs(target_integrals)), 1.0
        )
        conservation = jnp.all(jnp.abs(residual) <= self.conservation_tolerance * scale)
        symmetry = jnp.max(target.stress_symmetry_defect, initial=0.0)
        finite = (
            source.all_active_valid
            & target.all_active_valid
            & jnp.all(jnp.isfinite(residual))
        )
        source_valid = source.all_active_valid
        qualified = finite & conservation & source_valid
        return EinsteinVlasovAMRStressEvidence(
            source_integrals,
            target_integrals,
            residual,
            symmetry,
            finite,
            conservation,
            source_valid,
            qualified,
            jnp.asarray(False),
            self.plan_id,
        )

    def transfer_projection(
        self,
        projection: StressEnergyProjection,
        target_geometry: ADMGridGeometry,
        source_proper_volume: ArrayLike,
        /,
        *,
        direction: StressTransferDirection,
        target_proper_volume: ArrayLike | None = None,
    ) -> EinsteinVlasovAMRStressTransferResult:
        if not isinstance(projection, StressEnergyProjection):
            raise TypeError("projection must be StressEnergyProjection.")
        if not isinstance(target_geometry, ADMGridGeometry):
            raise TypeError("target_geometry must be ADMGridGeometry.")
        if len(projection.leading_shape) != 3 or len(target_geometry.leading_shape) != 3:
            raise ValueError("Einstein-Vlasov AMR stress transfer is three-dimensional.")
        if direction not in ("prolong", "restrict"):
            raise ValueError("direction must be 'prolong' or 'restrict'.")
        source_channels = jnp.where(
            projection.active[..., None], _stress_channels(projection), 0.0
        )
        source_volume = _proper_volume(
            source_proper_volume,
            projection.leading_shape,
            source_channels.dtype,
            "source_proper_volume",
        )
        ratio_volume = self.transfer.refinement_ratio**3
        if direction == "prolong":
            expected_shape = self.transfer.fine_shape(projection.leading_shape)
            if target_geometry.leading_shape != expected_shape:
                raise ValueError(
                    "Target geometry does not have the refined source shape."
                )
            channels = self.transfer.prolong(source_channels)
            derived_volume = self.transfer.prolong(source_volume) / ratio_volume
        else:
            expected_shape = tuple(
                size // self.transfer.refinement_ratio
                for size in projection.leading_shape
            )
            if (
                any(
                    size % self.transfer.refinement_ratio
                    for size in projection.leading_shape
                )
                or target_geometry.leading_shape != expected_shape
            ):
                raise ValueError(
                    "Target geometry does not have the restricted source shape."
                )
            content = (
                self.transfer.restrict(source_channels * source_volume[..., None])
                * ratio_volume
            )
            derived_volume = self.transfer.restrict(source_volume) * ratio_volume
            target_volume_pre = _proper_volume(
                derived_volume if target_proper_volume is None else target_proper_volume,
                expected_shape,
                source_channels.dtype,
                "target_proper_volume",
            )
            channels = content / target_volume_pre[..., None]
        channels = jnp.where(target_geometry.active[..., None], channels, 0.0)
        target_volume = _proper_volume(
            derived_volume if target_proper_volume is None else target_proper_volume,
            target_geometry.leading_shape,
            source_channels.dtype,
            "target_proper_volume",
        )
        residual_estimate = jnp.max(
            jnp.abs(
                jnp.sum(channels * target_volume[..., None], axis=(0, 1, 2))
                - jnp.sum(source_channels * source_volume[..., None], axis=(0, 1, 2))
            ),
            initial=0.0,
        )
        finite = jnp.all(jnp.isfinite(channels), axis=-1)
        valid = target_geometry.physically_valid & finite
        projection_id = canonical_fingerprint(
            {
                "kind": "einstein-vlasov-amr-stress-projection",
                "plan": self.plan_id,
                "source": projection.projection_id,
                "target": target_geometry.geometry_lineage_id,
                "direction": direction,
            }
        )
        target = _projection_from_channels(
            channels,
            target_geometry,
            valid,
            jnp.broadcast_to(
                jnp.max(projection.projection_defect, initial=0.0),
                target_geometry.leading_shape,
            ),
            jnp.broadcast_to(residual_estimate, target_geometry.leading_shape),
            projection_id,
        )
        evidence = self._evidence(projection, target, source_volume, target_volume)
        return EinsteinVlasovAMRStressTransferResult(target, evidence)


class EinsteinVlasovParticleRoute(StrictModule):
    """Exactly-once owner/local-slot route in stable global-particle order."""

    owner: Array
    local_slot: Array
    level: Array
    block_slot: Array
    stable_order: Array
    inverse_order: Array
    per_owner_count: Array
    active: Array
    migrated: Array
    migration_count: Array
    finite: Array
    ids_unique: Array
    capacity_valid: Array
    topology_valid: Array
    successful: Array
    topology_id: str = eqx.field(static=True)
    ownership_id: str = eqx.field(static=True)


class EinsteinVlasovParticleMigrationResult(StrictModule):
    particles: RelativisticParticleState
    route: EinsteinVlasovParticleRoute
    predecessor: EinsteinVlasovParticleRoute | None
    committed: Array
    derivative_valid: Array


class EinsteinVlasovParticleMigrationPlan(StrictModule, NonTrainableState):
    """Particle routing over the authoritative NR AMR block ownership."""

    distribution: PreparedNumericalRelativityAMRDistribution
    particle_capacity_per_owner: int = eqx.field(static=True)
    owner_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        distribution: PreparedNumericalRelativityAMRDistribution,
        particle_capacity_per_owner: int,
        /,
    ):
        if not isinstance(distribution, PreparedNumericalRelativityAMRDistribution):
            raise TypeError(
                "distribution must be PreparedNumericalRelativityAMRDistribution."
            )
        capacity = int(particle_capacity_per_owner)
        owners = int(distribution.plan.partition.part_count)
        if capacity < 1 or owners < 1:
            raise ValueError("Particle owner count/capacity must be positive.")
        self.distribution = distribution
        self.particle_capacity_per_owner = capacity
        self.owner_count = owners
        self.plan_id = canonical_fingerprint(
            {
                "kind": "einstein-vlasov-amr-particle-migration",
                "distribution": distribution.prepared_id,
                "particle_capacity_per_owner": capacity,
            }
        )

    @property
    def topology_id(self) -> str:
        return self.distribution.distribution.topology.epoch.epoch_id

    def route(
        self,
        particles: RelativisticParticleState,
        level: ArrayLike,
        block_slot: ArrayLike,
        /,
        *,
        predecessor: EinsteinVlasovParticleRoute | None = None,
    ) -> EinsteinVlasovParticleMigrationResult:
        if not isinstance(particles, RelativisticParticleState):
            raise TypeError("particles must be RelativisticParticleState.")
        levels = jnp.asarray(level, dtype=jnp.int32)
        blocks = jnp.asarray(block_slot, dtype=jnp.int32)
        active = particles.active_mask
        capacity = active.shape[0]
        if levels.shape != (capacity,) or blocks.shape != (capacity,):
            raise ValueError("AMR particle level/block arrays must have capacity shape.")
        owner = jnp.full((capacity,), -1, dtype=jnp.int32)
        topology_valid = ~active
        for index, ownership in enumerate(self.distribution.ownership):
            safe = jnp.clip(blocks, 0, ownership.owner_indices.shape[0] - 1)
            selected = ownership.owner_indices[safe]
            valid = (
                (levels == index)
                & (blocks >= 0)
                & (blocks < ownership.owner_indices.shape[0])
                & ownership.active[safe]
                & (selected >= 0)
            )
            owner = jnp.where(active & valid, selected, owner)
            topology_valid = topology_valid | (active & valid)
        ids = particles.particle_ids
        stable_order = jnp.lexsort((ids, owner, ~active)).astype(jnp.int32)
        inverse_order = (
            jnp.zeros_like(stable_order)
            .at[stable_order]
            .set(jnp.arange(capacity, dtype=jnp.int32))
        )
        sorted_owner = jnp.where(active[stable_order], owner[stable_order], 0)
        sorted_active = active[stable_order]
        one_hot = (
            jax.nn.one_hot(sorted_owner, self.owner_count, dtype=jnp.int32)
            * sorted_active[:, None]
        )
        rank_sorted = jnp.sum(
            (jnp.cumsum(one_hot, axis=0) - 1) * one_hot,
            axis=-1,
            dtype=jnp.int32,
        )
        local_slot = (
            jnp.full((capacity,), -1, dtype=jnp.int32)
            .at[stable_order]
            .set(jnp.where(sorted_active, rank_sorted, -1))
        )
        per_owner = jnp.sum(one_hot, axis=0, dtype=jnp.int32)
        capacity_valid = jnp.all(per_owner <= self.particle_capacity_per_owner)
        sorted_ids = ids[stable_order]
        sorted_mask = active[stable_order]
        ids_unique = ~jnp.any(
            sorted_mask[1:] & sorted_mask[:-1] & (sorted_ids[1:] == sorted_ids[:-1])
        )
        finite = jnp.all(jnp.isfinite(particles.positions) | ~active[:, None]) & jnp.all(
            jnp.isfinite(particles.covariant_momenta) | ~active[:, None]
        )
        if predecessor is None:
            migrated = active
            previous = None
        else:
            if predecessor.owner.shape != owner.shape:
                raise ValueError("Predecessor particle route capacity differs.")
            migrated = active & (
                (owner != predecessor.owner)
                | (levels != predecessor.level)
                | (blocks != predecessor.block_slot)
            )
            previous = predecessor
        successful = (
            finite
            & ids_unique
            & capacity_valid
            & jnp.all(topology_valid)
            & jnp.all(~active | (owner >= 0))
        )
        accepted_owner = jnp.where(successful, owner, -1)
        accepted_local = jnp.where(successful, local_slot, -1)
        route = EinsteinVlasovParticleRoute(
            accepted_owner,
            accepted_local,
            levels,
            blocks,
            stable_order,
            inverse_order,
            per_owner,
            active,
            migrated,
            jnp.sum(migrated, dtype=jnp.int32),
            finite,
            ids_unique,
            capacity_valid,
            jnp.all(topology_valid),
            successful,
            self.topology_id,
            self.plan_id,
        )
        return EinsteinVlasovParticleMigrationResult(
            particles,
            route,
            previous,
            successful,
            jnp.asarray(False),
        )

    def owner_packed(
        self, route: EinsteinVlasovParticleRoute, values: ArrayLike, /
    ) -> Array:
        """Pack global stable-ID values into fixed owner/local slots exactly once."""

        if not isinstance(route, EinsteinVlasovParticleRoute):
            raise TypeError("route must be EinsteinVlasovParticleRoute.")
        if route.ownership_id != self.plan_id:
            raise ValueError("Particle route belongs to another migration plan.")
        value = jnp.asarray(values)
        if value.shape[0] != route.owner.shape[0]:
            raise ValueError("Owner-packed values must begin with particle capacity.")
        output_shape = (
            self.owner_count,
            self.particle_capacity_per_owner,
        ) + value.shape[1:]
        output = jnp.zeros(output_shape, dtype=value.dtype)
        safe_owner = jnp.where(route.active, route.owner, 0)
        safe_slot = jnp.where(route.active, route.local_slot, 0)
        mask = route.active.reshape(route.active.shape + (1,) * (value.ndim - 1))
        return output.at[safe_owner, safe_slot].add(jnp.where(mask, value, 0))


class EinsteinVlasovCheckpointPayload(StrictModule):
    """Sharded particle bytes plus bounded replicated synchronization controls."""

    particle_content: Array
    particle_capacity: int = eqx.field(static=True)
    owner_count: int = eqx.field(static=True)
    real_dtype: str = eqx.field(static=True)
    integer_dtype: str = eqx.field(static=True)
    particle_topology_id: str = eqx.field(static=True)
    particle_frame_id: str = eqx.field(static=True)
    frame_lineage_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    ownership_id: str = eqx.field(static=True)
    stress_plan_id: str = eqx.field(static=True)
    frame_provider_id: str = eqx.field(static=True)
    placement_id: str = eqx.field(static=True)


class EinsteinVlasovCheckpointRestore(StrictModule):
    state: EinsteinVlasovMatterState
    route: EinsteinVlasovParticleRoute
    numerical_relativity: (
        DistributedNumericalRelativityRestart | NumericalRelativityCheckpoint
    )
    exact: Array
    placement_changed: Array
    derivative_valid: Array
    checkpoint_id: str = eqx.field(static=True)


class EinsteinVlasovCheckpointPlan(StrictModule, NonTrainableState):
    """Typed particle payload layered on the canonical NR checkpoint owner."""

    numerical_relativity: NumericalRelativityCheckpointPlan
    migration: EinsteinVlasovParticleMigrationPlan
    stress_plan_id: str = eqx.field(static=True)
    frame_provider_id: str = eqx.field(static=True)
    placement_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        template: EinsteinVlasovMatterState,
        route_template: EinsteinVlasovParticleRoute,
        migration: EinsteinVlasovParticleMigrationPlan,
        /,
        *,
        topology_epoch: int,
        analysis_plan_id: str,
        numeric_revision_id: str,
        execution_plan_id: str,
        stress_plan_id: str,
        frame_provider_id: str,
        placement_id: str,
        restart: NumericalRelativityRestartPolicy | None = None,
    ):
        if not isinstance(template, EinsteinVlasovMatterState):
            raise TypeError("template must be EinsteinVlasovMatterState.")
        if not isinstance(route_template, EinsteinVlasovParticleRoute):
            raise TypeError("route_template must be EinsteinVlasovParticleRoute.")
        if not isinstance(migration, EinsteinVlasovParticleMigrationPlan):
            raise TypeError("migration must be EinsteinVlasovParticleMigrationPlan.")
        identities = tuple(
            str(value).strip()
            for value in (stress_plan_id, frame_provider_id, placement_id)
        )
        if any(not value for value in identities):
            raise ValueError("Einstein-Vlasov checkpoint identities must be non-empty.")
        runtime = Z4cRuntimeState(
            template.z4c,
            template.time,
            template.accepted_steps,
            template.runtime_id,
        )
        restart_state = NumericalRelativityRestartState.from_z4c(
            runtime,
            topology_id=route_template.topology_id,
            topology_epoch=topology_epoch,
        )
        numerical = NumericalRelativityCheckpointPlan(
            "z4c",
            template.runtime_id,
            template.z4c.grid_id,
            route_template.topology_id,
            analysis_plan_id=analysis_plan_id,
            numeric_revision_id=numeric_revision_id,
            execution_plan_id=execution_plan_id,
            topology_epoch=topology_epoch,
            state_template=restart_state,
            restart=restart,
        )
        self.numerical_relativity = numerical
        self.migration = migration
        self.stress_plan_id, self.frame_provider_id, self.placement_id = identities
        self.plan_id = canonical_fingerprint(
            {
                "kind": "einstein-vlasov-checkpoint-plan",
                "numerical_relativity": numerical.plan_id,
                "migration": migration.plan_id,
                "stress": identities[0],
                "frame_provider": identities[1],
                "placement": identities[2],
            }
        )
        self._encoded(template, route_template)

    @property
    def checkpoint_id(self) -> str:
        return self.numerical_relativity.checkpoint_id

    def _encoded(
        self,
        state: EinsteinVlasovMatterState,
        route: EinsteinVlasovParticleRoute,
        /,
    ) -> tuple[NumericalRelativityRestartState, EinsteinVlasovCheckpointPayload]:
        if state.runtime_id != self.numerical_relativity.runtime_id:
            raise ValueError("Einstein-Vlasov state runtime identity changed.")
        if route.ownership_id != self.migration.plan_id:
            raise ValueError("Particle route ownership identity changed.")
        runtime = Z4cRuntimeState(
            state.z4c, state.time, state.accepted_steps, state.runtime_id
        )
        restart_state = NumericalRelativityRestartState.from_z4c(
            runtime,
            topology_id=route.topology_id,
            topology_epoch=self.numerical_relativity.topology_epoch,
        )
        particles = state.particles
        if not bool(
            route.successful
            & jnp.array_equal(route.active, particles.active_mask)
            & (state.time == particles.time)
            & (route.topology_id == self.numerical_relativity.topology_id)
        ):
            raise ValueError(
                "Checkpoint requires one successful owner route at an accepted co-temporal synchronization point."
            )
        particle_reals = jnp.concatenate(
            (
                particles.weights[:, None],
                particles.positions,
                particles.local_momenta,
                particles.covariant_momenta,
            ),
            axis=-1,
        )
        particle_integers = jnp.stack(
            (
                particles.particle_ids,
                particles.species_ids.astype(jnp.int64),
                particles.active_mask.astype(jnp.int64),
                particles.incarnations.astype(jnp.int64),
                particles.lineage_ids,
            ),
            axis=-1,
        )
        runtime_counters = jnp.stack(
            (
                state.rejected_steps,
                state.consecutive_failures,
                state.terminal.astype(jnp.int32),
            )
        ).astype(jnp.int32)
        route_integers = jnp.stack(
            (
                route.owner,
                route.local_slot,
                route.level,
                route.block_slot,
                route.stable_order,
                route.inverse_order,
                route.active.astype(jnp.int32),
            ),
            axis=-1,
        )

        def byte_rows(values):
            return jax.lax.bitcast_convert_type(values, jnp.uint8).reshape(
                (particles.capacity, -1)
            )

        particle_content = jnp.concatenate(
            (
                byte_rows(particle_reals),
                byte_rows(particle_integers),
                byte_rows(route_integers),
            ),
            axis=-1,
        )
        control = jnp.concatenate(
            (
                particles.frame_token.reshape((1,)).astype(jnp.int32),
                runtime_counters,
                route.per_owner_count.astype(jnp.int32),
            )
        )
        header = jnp.concatenate(
            (
                jax.lax.bitcast_convert_type(particles.scale_factor, jnp.uint8).reshape(
                    (-1,)
                ),
                jax.lax.bitcast_convert_type(control, jnp.uint8).reshape((-1,)),
            )
        )
        header_rows = (
            jnp.zeros((particles.capacity, header.shape[0]), dtype=jnp.uint8)
            .at[0]
            .set(header)
        )
        particle_content = jnp.concatenate((particle_content, header_rows), axis=-1)
        payload = EinsteinVlasovCheckpointPayload(
            particle_content,
            particles.capacity,
            self.migration.owner_count,
            np.dtype(particle_reals.dtype).str,
            np.dtype(particle_integers.dtype).str,
            particles.topology_id,
            particles.frame_id,
            particles.frame_lineage_id,
            route.topology_id,
            route.ownership_id,
            self.stress_plan_id,
            self.frame_provider_id,
            self.placement_id,
        )
        self.numerical_relativity.validate_state(restart_state)
        return restart_state, payload

    def write_local(
        self,
        path: str | Path,
        state: EinsteinVlasovMatterState,
        route: EinsteinVlasovParticleRoute,
        /,
    ) -> NumericalRelativityCheckpoint:
        restart_state, payload = self._encoded(state, route)
        return write_numerical_relativity_checkpoint(
            path,
            self.numerical_relativity,
            restart_state,
            runtime_args=payload,
        )

    def read_local(
        self,
        path: str | Path,
        template: EinsteinVlasovMatterState,
        route_template: EinsteinVlasovParticleRoute,
        /,
        *,
        target_placement_id: str | None = None,
    ) -> EinsteinVlasovCheckpointRestore:
        placement = (
            self.placement_id
            if target_placement_id is None
            else str(target_placement_id).strip()
        )
        if not placement:
            raise ValueError("target_placement_id must be non-empty.")
        restart_template, payload_template = self._encoded(template, route_template)
        restored = read_numerical_relativity_checkpoint(
            path,
            self.numerical_relativity,
            restart_template,
            runtime_args_template=payload_template,
        )
        return self._decode(
            restored,
            template,
            route_template,
            target_placement_id=placement,
        )

    def publish(
        self,
        repository: ArtifactRepository,
        state: EinsteinVlasovMatterState,
        route: EinsteinVlasovParticleRoute,
        /,
        *,
        writer_id: str,
        attempt_id: str | None = None,
    ) -> ProcessCheckpointPublication:
        restart_state, payload = self._encoded(state, route)
        return publish_distributed_numerical_relativity_checkpoint(
            repository,
            self.numerical_relativity,
            restart_state,
            writer_id=writer_id,
            runtime_args=payload,
            attempt_id=attempt_id,
        )

    def assemble(
        self,
        repository: ArtifactRepository,
        publications: Sequence[ProcessCheckpointPublication],
        /,
        *,
        expected_process_count: int,
        parent_checkpoint_id: str | None = None,
        diagnostic_ids: Sequence[str] = (),
    ) -> CheckpointManifest:
        return assemble_distributed_numerical_relativity_checkpoint(
            repository,
            self.numerical_relativity,
            publications,
            expected_process_count=expected_process_count,
            parent_checkpoint_id=parent_checkpoint_id,
            diagnostic_ids=diagnostic_ids,
        )

    def restore(
        self,
        repository: ArtifactRepository,
        manifest: CheckpointManifest,
        template: EinsteinVlasovMatterState,
        route_template: EinsteinVlasovParticleRoute,
        /,
        *,
        target_placement_id: str | None = None,
    ) -> EinsteinVlasovCheckpointRestore:
        placement = (
            self.placement_id
            if target_placement_id is None
            else str(target_placement_id).strip()
        )
        if not placement:
            raise ValueError("target_placement_id must be non-empty.")
        restart_template, payload_template = self._encoded(template, route_template)
        restored = restore_distributed_numerical_relativity_checkpoint(
            repository,
            manifest,
            self.numerical_relativity,
            restart_template,
            runtime_args_template=payload_template,
        )
        return self._decode(
            restored,
            template,
            route_template,
            target_placement_id=placement,
        )

    def _decode(
        self,
        restored: DistributedNumericalRelativityRestart | NumericalRelativityCheckpoint,
        state_template: EinsteinVlasovMatterState,
        route_template: EinsteinVlasovParticleRoute,
        /,
        *,
        target_placement_id: str,
    ) -> EinsteinVlasovCheckpointRestore:
        checkpoint = (
            restored.checkpoint
            if isinstance(restored, DistributedNumericalRelativityRestart)
            else restored
        )
        payload = checkpoint.runtime_args
        if not isinstance(payload, EinsteinVlasovCheckpointPayload):
            raise TypeError("Checkpoint lacks the typed Einstein-Vlasov payload.")
        if (
            payload.topology_id != route_template.topology_id
            or payload.ownership_id != self.migration.plan_id
            or payload.stress_plan_id != self.stress_plan_id
            or payload.frame_provider_id != self.frame_provider_id
        ):
            raise ValueError("Einstein-Vlasov checkpoint identity changed.")
        runtime = checkpoint.state.field("z4c_runtime_state")
        if not isinstance(runtime, Z4cRuntimeState):
            raise TypeError("Checkpoint did not restore the canonical Z4c runtime.")
        capacity = payload.particle_capacity
        owner_count = payload.owner_count
        real_dtype = np.dtype(payload.real_dtype)
        integer_dtype = np.dtype(payload.integer_dtype)
        real_bytes = 10 * real_dtype.itemsize
        integer_bytes = 5 * integer_dtype.itemsize
        route_bytes = 7 * np.dtype(np.int32).itemsize
        particle_width = real_bytes + integer_bytes + route_bytes
        header_width = (
            real_dtype.itemsize + (4 + owner_count) * np.dtype(np.int32).itemsize
        )
        expected_width = particle_width + header_width
        if payload.particle_content.shape != (capacity, expected_width):
            raise ValueError("Einstein-Vlasov checkpoint payload shape changed.")
        particle_reals = jax.lax.bitcast_convert_type(
            payload.particle_content[:, :real_bytes].reshape(
                (capacity, 10, real_dtype.itemsize)
            ),
            real_dtype,
        )
        particle_integers = jax.lax.bitcast_convert_type(
            payload.particle_content[:, real_bytes : real_bytes + integer_bytes].reshape(
                (capacity, 5, integer_dtype.itemsize)
            ),
            integer_dtype,
        )
        route_integers = jax.lax.bitcast_convert_type(
            payload.particle_content[
                :, real_bytes + integer_bytes : particle_width
            ].reshape((capacity, 7, np.dtype(np.int32).itemsize)),
            np.int32,
        )
        header = payload.particle_content[0, particle_width:]
        scale_factor = jax.lax.bitcast_convert_type(
            header[: real_dtype.itemsize].reshape((1, real_dtype.itemsize)),
            real_dtype,
        ).reshape(())
        control = jax.lax.bitcast_convert_type(
            header[real_dtype.itemsize :].reshape(
                (4 + owner_count, np.dtype(np.int32).itemsize)
            ),
            np.int32,
        )
        frame_token = control[0]
        runtime_counters = control[1:4]
        per_owner_count = control[4:]
        template_particles = state_template.particles

        def placed(value, template_value):
            return jax.device_put(value, template_value.sharding)

        particles = RelativisticParticleState(
            placed(particle_integers[:, 0], template_particles.particle_ids),
            placed(particle_reals[:, 0], template_particles.weights),
            placed(particle_reals[:, 1:4], template_particles.positions),
            placed(particle_reals[:, 4:7], template_particles.local_momenta),
            placed(particle_reals[:, 7:10], template_particles.covariant_momenta),
            placed(
                particle_integers[:, 1].astype(jnp.int32),
                template_particles.species_ids,
            ),
            placed(
                particle_integers[:, 2].astype("bool"),
                template_particles.active_mask,
            ),
            placed(
                particle_integers[:, 3].astype(jnp.int32),
                template_particles.incarnations,
            ),
            placed(particle_integers[:, 4], template_particles.lineage_ids),
            placed(runtime.time, template_particles.time),
            placed(scale_factor, template_particles.scale_factor),
            placed(frame_token, template_particles.frame_token),
            frame_id=payload.particle_frame_id,
            topology_id=payload.particle_topology_id,
            frame_lineage_id=payload.frame_lineage_id,
        )
        state = EinsteinVlasovMatterState(
            runtime.state,
            particles,
            runtime.time,
            runtime.step_index,
            runtime_counters[0],
            runtime_counters[1],
            runtime_counters[2].astype("bool"),
            runtime_id=runtime.runtime_id,
        )
        route_active = route_integers[:, 6].astype("bool")
        expected = self.migration.route(
            particles,
            route_integers[:, 2],
            route_integers[:, 3],
        ).route
        route_matches = (
            jnp.array_equal(route_integers[:, 0], expected.owner)
            & jnp.array_equal(route_integers[:, 1], expected.local_slot)
            & jnp.array_equal(route_integers[:, 4], expected.stable_order)
            & jnp.array_equal(route_integers[:, 5], expected.inverse_order)
            & jnp.array_equal(per_owner_count, expected.per_owner_count)
            & jnp.array_equal(route_active, expected.active)
            & expected.successful
        )
        if not bool(route_matches):
            raise ValueError(
                "Checkpoint particle route is not the authoritative owner-computes route."
            )
        route = EinsteinVlasovParticleRoute(
            placed(route_integers[:, 0], route_template.owner),
            placed(route_integers[:, 1], route_template.local_slot),
            placed(route_integers[:, 2], route_template.level),
            placed(route_integers[:, 3], route_template.block_slot),
            placed(route_integers[:, 4], route_template.stable_order),
            placed(route_integers[:, 5], route_template.inverse_order),
            placed(per_owner_count, route_template.per_owner_count),
            placed(route_active, route_template.active),
            placed(jnp.zeros_like(route_active), route_template.migrated),
            placed(jnp.asarray(0, dtype=jnp.int32), route_template.migration_count),
            expected.finite,
            expected.ids_unique,
            expected.capacity_valid,
            expected.topology_valid,
            expected.successful,
            payload.topology_id,
            payload.ownership_id,
        )
        exact = (
            restored.evidence.exact
            if isinstance(restored, DistributedNumericalRelativityRestart)
            else jnp.asarray(True)
        )
        return EinsteinVlasovCheckpointRestore(
            state,
            route,
            restored,
            exact,
            jnp.asarray(target_placement_id != payload.placement_id),
            jnp.asarray(False),
            checkpoint.checkpoint_id,
        )


__all__ = [
    "EinsteinVlasovAMRStressEvidence",
    "EinsteinVlasovAMRStressTransferPlan",
    "EinsteinVlasovAMRStressTransferResult",
    "EinsteinVlasovCheckpointPayload",
    "EinsteinVlasovCheckpointPlan",
    "EinsteinVlasovCheckpointRestore",
    "EinsteinVlasovParticleMigrationPlan",
    "EinsteinVlasovParticleMigrationResult",
    "EinsteinVlasovParticleRoute",
    "StressTransferDirection",
]
