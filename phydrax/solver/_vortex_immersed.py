#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import LagrangianMarkerKinematics
from ..discretization.vortex._mac_transfer import (
    MACVortexGridState,
    MACVortexParticleTransferPlan,
    MACVortexTransferEvidence,
)
from ..discretization.vortex._source import VortexSourceState
from ._mac_immersed_boundary import (
    MACImmersedBoundaryProjectionPlan,
    MACImmersedBoundaryProjectionResult,
)


class VortexImmersedStepResult(StrictModule):
    source: VortexSourceState
    grid: MACVortexGridState
    projection: MACImmersedBoundaryProjectionResult
    marker_reaction: Array
    work_residual: Array
    successful: Array
    method_id: str = eqx.field(static=True)


class VortexImmersedHybridPlan(StrictModule, NonTrainableState):
    transfer: MACVortexParticleTransferPlan
    projection: MACImmersedBoundaryProjectionPlan
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer: MACVortexParticleTransferPlan,
        projection: MACImmersedBoundaryProjectionPlan,
        /,
    ):
        if not isinstance(transfer, MACVortexParticleTransferPlan) or not isinstance(
            projection, MACImmersedBoundaryProjectionPlan
        ):
            raise TypeError(
                "Immersed vortex hybrid requires transfer and projection plans."
            )
        if (
            projection.operators.prepared_id
            != transfer.dynamics.momentum.operators.prepared_id
        ):
            raise ValueError(
                "Immersed projection and vortex transfer must share MAC operators."
            )
        self.transfer, self.projection = transfer, projection
        self.method_id = canonical_fingerprint(
            {
                "kind": "vortex-immersed-hybrid",
                "transfer": transfer.transfer_id,
                "projection": projection.plan_id,
            }
        )

    def step(
        self,
        source: VortexSourceState,
        marker_kinematics: LagrangianMarkerKinematics,
        inverse_momentum_coefficient: ArrayLike,
        /,
        *,
        pressure: ArrayLike | None = None,
        marker_force_density: ArrayLike | None = None,
    ) -> VortexImmersedStepResult:
        if not isinstance(source, VortexSourceState) or not isinstance(
            marker_kinematics, LagrangianMarkerKinematics
        ):
            raise TypeError("Immersed step requires vortex source and marker kinematics.")
        vorticity, transfer_state = self.transfer.deposit(source)
        velocity_state = self.transfer.vorticity_to_velocity(vorticity)
        velocity = self.transfer.dynamics.unpack_velocity(velocity_state)
        projection = self.projection.project(
            velocity,
            inverse_momentum_coefficient,
            marker_kinematics,
            pressure=pressure,
            marker_force_density=marker_force_density,
        )
        projected_state = self.transfer.dynamics.pack_velocity(projection.velocity)
        projected_vorticity = self.transfer.velocity_to_vorticity(projected_state)
        candidate_source = self.transfer.gather(
            transfer_state, projected_vorticity, source
        )
        deposited_total = jnp.sum(source.safe_strength(), axis=0)
        recovered_total = jnp.sum(candidate_source.safe_strength(), axis=0)
        residual = recovered_total - deposited_total
        before_moment = jnp.sum(
            source.safe_strength().reshape((source.capacity, -1))
            * source.safe_positions()[:, :1],
            axis=0,
        )
        after_moment = jnp.sum(
            candidate_source.safe_strength().reshape((source.capacity, -1))
            * candidate_source.safe_positions()[:, :1],
            axis=0,
        )
        divergence = jnp.linalg.norm(projection.divergence_after)
        transfer_successful = projection.successful & jnp.all(
            jnp.isfinite(projected_vorticity)
        )
        evidence = MACVortexTransferEvidence(
            deposited_total,
            recovered_total,
            residual,
            after_moment - before_moment,
            divergence,
            transfer_successful,
            jnp.all(jnp.isfinite(projected_state)),
        )
        grid = MACVortexGridState(
            projected_vorticity, projected_state, evidence, self.transfer.transfer_id
        )
        marker_reaction = -projection.marker_force_density
        marker_work = jnp.sum(
            projection.marker_force_density * projection.marker_velocity_after
        )
        measures = self.projection.operators.face_dual_measures
        energy_before = 0.5 * sum(
            jnp.sum(component * component * measure)
            for component, measure in zip(
                velocity,
                measures,
                strict=True,
            )
        )
        energy_after = 0.5 * sum(
            jnp.sum(component * component * measure)
            for component, measure in zip(
                projection.velocity,
                measures,
                strict=True,
            )
        )
        fluid_work = energy_after - energy_before
        work_residual = jnp.abs(marker_work + fluid_work)
        successful = (
            projection.successful
            & transfer_successful
            & jnp.all(jnp.isfinite(marker_reaction))
        )
        accepted_source = VortexSourceState(
            source.positions,
            jnp.where(successful, candidate_source.strength, source.strength),
            core_radius=source.core_radius,
            volume=source.volume,
            active_mask=source.active_mask,
            source_kind=source.source_kind,
            source_id=source.source_id,
        )
        return VortexImmersedStepResult(
            accepted_source,
            grid,
            projection,
            marker_reaction,
            work_residual,
            successful,
            self.method_id,
        )


__all__ = [
    "MACVortexParticleTransferPlan",
    "VortexImmersedHybridPlan",
    "VortexImmersedStepResult",
]
