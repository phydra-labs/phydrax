#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...solver import (
    AdaptiveBalanceLawRolloutPlan,
    AdaptiveBalanceLawRolloutResult,
    BalanceLawRuntimeState,
)


class AstrophysicalApplicationResult(StrictModule):
    rollout: AdaptiveBalanceLawRolloutResult
    observations: Array
    mass: Array
    total_energy: Array
    successful: Array


class AstrophysicalMultiphysicsApplicationPlan(StrictModule, NonTrainableState):
    rollout: AdaptiveBalanceLawRolloutPlan
    observation: Callable = eqx.field(static=True)
    density_index: int = eqx.field(static=True)
    energy_index: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        rollout: AdaptiveBalanceLawRolloutPlan,
        observation: Callable,
        /,
        *,
        observation_id: str,
    ):
        if (
            not isinstance(rollout, AdaptiveBalanceLawRolloutPlan)
            or not callable(observation)
            or not observation_id
        ):
            raise ValueError("Astrophysical application plan is invalid.")
        names = rollout.runtime.transport.component_names
        self.rollout = rollout
        self.observation = observation
        self.density_index = names.index("density")
        self.energy_index = names.index("total_energy")
        self.plan_id = canonical_fingerprint(
            {
                "kind": "astrophysical-multiphysics-application",
                "rollout": rollout.plan_id,
                "observation_id": observation_id,
            }
        )

    def run(
        self,
        initial_state: BalanceLawRuntimeState,
        args: Any = None,
        realization=None,
        /,
    ) -> AstrophysicalApplicationResult:
        result = self.rollout.rollout(initial_state, args, realization)
        view = self.rollout.runtime.transport.source_view(
            result.final_state.transport_state
        )
        observations = jnp.asarray(self.observation(result.final_state, args))
        mass = jnp.sum(view.cell_volumes * view.cell_average[..., self.density_index])
        energy = jnp.sum(view.cell_volumes * view.cell_average[..., self.energy_index])
        successful = result.completed & jnp.all(jnp.isfinite(observations))
        return AstrophysicalApplicationResult(
            rollout=result,
            observations=observations,
            mass=mass,
            total_energy=energy,
            successful=successful,
        )


__all__ = [
    "AstrophysicalApplicationResult",
    "AstrophysicalMultiphysicsApplicationPlan",
]
