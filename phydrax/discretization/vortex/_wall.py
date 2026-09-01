#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


if TYPE_CHECKING:
    from ...operators.integral.vortex._panels2d import FlowPanelGeometry2D


class WallVortexPoolState(StrictModule):
    position: Array
    circulation: Array
    core_radius: Array
    active: Array
    next_event_id: Array


class BoundarySheetParticleTransferResult(StrictModule):
    candidate: WallVortexPoolState
    accepted: WallVortexPoolState
    emitted_circulation: Array
    circulation_residual: Array
    overflow_count: Array
    clearance_minimum: Array
    successful: Array
    transfer_id: str = eqx.field(static=True)


class BoundarySheetParticleTransferPlan2D(StrictModule, NonTrainableState):
    """Atomic conversion of a resolved boundary sheet to vortex carriers."""

    particle_capacity: int = eqx.field(static=True)
    core_radius: float = eqx.field(static=True)
    normal_offset: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self, particle_capacity: int, core_radius: float, normal_offset: float, /
    ):
        capacity = int(particle_capacity)
        core = float(core_radius)
        offset = float(normal_offset)
        if capacity <= 0 or core <= 0.0 or offset <= 0.0:
            raise ValueError("Wall-vortex capacity/core/offset must be positive.")
        self.particle_capacity = capacity
        self.core_radius = core
        self.normal_offset = offset
        self.plan_id = canonical_fingerprint(
            {
                "kind": "wall-vorticity-transfer-2d",
                "capacity": capacity,
                "core_radius": core,
                "normal_offset": offset,
            }
        )

    def initialize(self, /, *, dtype=float) -> WallVortexPoolState:
        return WallVortexPoolState(
            jnp.zeros((self.particle_capacity, 2), dtype=dtype),
            jnp.zeros((self.particle_capacity,), dtype=dtype),
            jnp.full((self.particle_capacity,), self.core_radius, dtype=dtype),
            jnp.zeros((self.particle_capacity,), dtype=bool),
            jnp.asarray(0, dtype=jnp.int64),
        )

    def transfer(
        self,
        state: WallVortexPoolState,
        geometry: FlowPanelGeometry2D,
        sheet_strength: ArrayLike,
        /,
    ) -> BoundarySheetParticleTransferResult:
        strength = jnp.asarray(sheet_strength, dtype=state.circulation.dtype)
        panel_count = int(geometry.length.size)
        if strength.shape != (panel_count,):
            raise ValueError("sheet_strength must have one value per panel.")
        emission = strength * geometry.length
        positions = geometry.control + self.normal_offset * geometry.normal
        selected = jnp.nonzero(
            ~state.active,
            size=panel_count,
            fill_value=-1,
        )[0]
        valid = selected >= 0
        overflow = jnp.sum(~valid, dtype=jnp.int32)
        safe = jnp.where(valid, selected, 0)
        candidate_position = state.position.at[safe].set(
            jnp.where(valid[:, None], positions, state.position[safe])
        )
        candidate_circulation = state.circulation.at[safe].set(
            jnp.where(valid, emission, state.circulation[safe])
        )
        candidate_core = state.core_radius.at[safe].set(
            jnp.where(valid, self.core_radius, state.core_radius[safe])
        )
        candidate_active = state.active.at[safe].set(
            jnp.where(valid, True, state.active[safe])
        )
        candidate = WallVortexPoolState(
            candidate_position,
            candidate_circulation,
            candidate_core,
            candidate_active,
            state.next_event_id + 1,
        )
        expected = jnp.sum(emission)
        actual = jnp.sum(jnp.where(valid, emission, 0.0))
        residual = actual - expected
        clearance = jnp.min(
            jnp.sum((positions - geometry.control) * geometry.normal, axis=-1)
        )
        finite = jnp.all(jnp.isfinite(positions)) & jnp.all(jnp.isfinite(emission))
        tolerance = (
            64
            * jnp.finfo(strength.dtype).eps
            * jnp.maximum(jnp.sum(jnp.abs(emission)), 1.0)
        )
        successful = (
            (overflow == 0)
            & finite
            & (clearance > 0.0)
            & (jnp.abs(residual) <= tolerance)
        )
        accepted = WallVortexPoolState(
            jnp.where(successful, candidate.position, state.position),
            jnp.where(successful, candidate.circulation, state.circulation),
            jnp.where(successful, candidate.core_radius, state.core_radius),
            jnp.where(successful, candidate.active, state.active),
            jnp.where(successful, candidate.next_event_id, state.next_event_id),
        )
        return BoundarySheetParticleTransferResult(
            candidate,
            accepted,
            actual,
            residual,
            overflow,
            clearance,
            successful,
            self.plan_id,
        )


__all__ = [
    "BoundarySheetParticleTransferPlan2D",
    "BoundarySheetParticleTransferResult",
    "WallVortexPoolState",
]
