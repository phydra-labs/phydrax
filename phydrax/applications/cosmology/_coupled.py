#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...solver import AdaptiveBalanceLawRolloutPlan, BalanceLawRuntimeState
from ._particles import CosmologicalKDKPlan, CosmologicalParticleState


class CosmologicalBaryonParticleState(StrictModule):
    baryons: BalanceLawRuntimeState
    particles: CosmologicalParticleState


class CosmologicalBaryonParticleDiagnostics(StrictModule):
    baryon_completed: Array
    particle_successful: Array
    scale_factor_defect: Array
    successful: Array


class CosmologicalBaryonParticlePlan(StrictModule, NonTrainableState):
    baryon_rollout: AdaptiveBalanceLawRolloutPlan
    particle_plan: CosmologicalKDKPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        baryon_rollout: AdaptiveBalanceLawRolloutPlan,
        particle_plan: CosmologicalKDKPlan,
        /,
    ):
        if not isinstance(
            baryon_rollout, AdaptiveBalanceLawRolloutPlan
        ) or not isinstance(particle_plan, CosmologicalKDKPlan):
            raise TypeError("Coupled cosmology requires baryon and particle plans.")
        self.baryon_rollout = baryon_rollout
        self.particle_plan = particle_plan
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cosmological-baryon-particle-plan",
                "baryon_rollout": baryon_rollout.plan_id,
                "particle_plan": particle_plan.plan_id,
            }
        )

    def advance(
        self,
        state: CosmologicalBaryonParticleState,
        acceleration_start: Array,
        acceleration_end: Array,
        args: Any = None,
        realization=None,
        /,
    ):
        baryons = self.baryon_rollout.rollout(state.baryons, args, realization)
        end_scale = jnp.asarray(
            self.baryon_rollout.final_time,
            dtype=state.particles.scale_factor.dtype,
        )
        particles, particle_evidence = self.particle_plan.advance(
            state.particles,
            end_scale,
            acceleration_start,
            acceleration_end,
        )
        defect = jnp.abs(baryons.final_state.time - particles.scale_factor)
        successful = baryons.completed & particle_evidence.successful & (defect <= 1e-12)
        return (
            CosmologicalBaryonParticleState(baryons.final_state, particles),
            CosmologicalBaryonParticleDiagnostics(
                baryon_completed=baryons.completed,
                particle_successful=particle_evidence.successful,
                scale_factor_defect=defect,
                successful=successful,
            ),
        )


__all__ = [
    "CosmologicalBaryonParticleDiagnostics",
    "CosmologicalBaryonParticlePlan",
    "CosmologicalBaryonParticleState",
]
