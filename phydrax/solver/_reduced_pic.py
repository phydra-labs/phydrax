#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.particle import ParticlePopulationState
from ..discretization.pic import (
    PICParticleState,
    ReducedPICTransferPlan,
    RelativisticBorisPlan,
)
from ._maxwell_reduced import (
    CompatibleMaxwell1DPlan,
    CompatibleMaxwell2DPlan,
    ReducedMaxwellDiagnostics,
)


class ReducedElectromagneticPICState(StrictModule):
    particles: PICParticleState
    population: ParticlePopulationState
    field: Any
    time: Array
    accepted_step: Array


class ReducedElectromagneticPICResult(StrictModule):
    candidate_state: ReducedElectromagneticPICState
    accepted_state: ReducedElectromagneticPICState
    field_diagnostics: ReducedMaxwellDiagnostics
    continuity_defect: Array
    maximum_displacement_fraction: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ReducedElectromagneticPICPlan(StrictModule, NonTrainableState):
    field: CompatibleMaxwell1DPlan | CompatibleMaxwell2DPlan
    transfer: ReducedPICTransferPlan
    pusher: RelativisticBorisPlan
    specific_charge: float = eqx.field(static=True)
    maximum_displacement_fraction: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        field: CompatibleMaxwell1DPlan | CompatibleMaxwell2DPlan,
        transfer: ReducedPICTransferPlan,
        specific_charge: float,
        /,
        *,
        pusher: RelativisticBorisPlan | None = None,
        maximum_displacement_fraction: float = 0.5,
    ):
        if not isinstance(field, (CompatibleMaxwell1DPlan, CompatibleMaxwell2DPlan)):
            raise TypeError("field must be a compatible reduced Maxwell plan.")
        if not isinstance(transfer, ReducedPICTransferPlan):
            raise TypeError("transfer must be ReducedPICTransferPlan.")
        if transfer.grid.prepared_id != field.grid.prepared_id:
            raise ValueError("Reduced PIC field and transfer grids differ.")
        charge = float(specific_charge)
        maximum = float(maximum_displacement_fraction)
        if charge == 0.0 or not jnp.isfinite(charge) or maximum <= 0.0:
            raise ValueError("Reduced PIC charge/displacement policy is invalid.")
        self.field = field
        self.transfer = transfer
        self.pusher = RelativisticBorisPlan() if pusher is None else pusher
        self.specific_charge = charge
        self.maximum_displacement_fraction = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "reduced-electromagnetic-pic",
                "field": field.plan_id,
                "transfer": transfer.plan_id,
                "pusher": self.pusher.plan_id,
                "specific_charge": charge,
                "maximum_displacement_fraction": maximum,
            }
        )

    def step(
        self,
        state: ReducedElectromagneticPICState,
        step_size: ArrayLike,
        /,
    ) -> ReducedElectromagneticPICResult:
        dt = jnp.asarray(step_size, dtype=state.time.dtype).reshape(())
        electric = self.transfer.gather(
            state.particles.position,
            state.field.electric,
            state.population.active,
        )
        magnetic = self.transfer.gather(
            state.particles.position,
            state.field.magnetic,
            state.population.active,
        )
        specific = jnp.where(state.population.active, self.specific_charge, 0.0)
        pushed = self.pusher.push(
            state.particles.proper_velocity,
            electric,
            magnetic,
            specific,
            state.population.active,
            dt,
        )
        displacement = dt * pushed.velocity[:, : self.transfer.dimension]
        position = state.particles.position + displacement
        macrocharge = state.population.mass * specific
        current = self.transfer.current(
            state.particles.position,
            position,
            macrocharge,
            pushed.velocity,
            state.population.active,
            dt,
        )
        field_state, field_diagnostics = self.field.step(state.field, current.current, dt)
        widths = jnp.asarray(self.transfer.spacing, dtype=dt.dtype)
        fraction = jnp.max(
            jnp.where(
                state.population.active[:, None],
                jnp.abs(displacement) / widths,
                0.0,
            ),
            initial=0.0,
        )
        finite = (
            jnp.all(jnp.isfinite(position))
            & jnp.all(jnp.isfinite(pushed.proper_velocity))
            & jnp.isfinite(fraction)
        )
        successful = (
            pushed.successful
            & current.successful
            & field_diagnostics.successful
            & finite
            & (fraction <= self.maximum_displacement_fraction)
        )
        candidate = ReducedElectromagneticPICState(
            PICParticleState(position, pushed.proper_velocity),
            state.population,
            field_state,
            state.time + dt,
            state.accepted_step + 1,
        )
        accepted = ReducedElectromagneticPICState(
            PICParticleState(
                jnp.where(
                    successful, candidate.particles.position, state.particles.position
                ),
                jnp.where(
                    successful,
                    candidate.particles.proper_velocity,
                    state.particles.proper_velocity,
                ),
            ),
            state.population,
            jax_tree_where(successful, field_state, state.field),
            jnp.where(successful, candidate.time, state.time),
            jnp.where(successful, candidate.accepted_step, state.accepted_step),
        )
        return ReducedElectromagneticPICResult(
            candidate,
            accepted,
            field_diagnostics,
            current.maximum_continuity_defect,
            fraction,
            finite,
            successful,
            self.plan_id,
        )


def jax_tree_where(predicate: Array, candidate: Any, current: Any, /):
    import jax

    return jax.tree.map(
        lambda proposed, old: jnp.where(predicate, proposed, old), candidate, current
    )


__all__ = [
    "ReducedElectromagneticPICPlan",
    "ReducedElectromagneticPICResult",
    "ReducedElectromagneticPICState",
]
