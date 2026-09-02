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


class WellBalancedEquilibriumPlan(StrictModule, NonTrainableState):
    state_provider: Any = eqx.field(static=True)
    time_derivative_provider: Any = eqx.field(static=True)
    entropy_supply_provider: Any = eqx.field(static=True)
    equilibrium_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        state_provider,
        /,
        *,
        equilibrium_id: str,
        time_derivative_provider=None,
        entropy_supply_provider=None,
    ):
        identifier = str(equilibrium_id)
        if not callable(state_provider) or not identifier:
            raise ValueError("Equilibrium plans require state provider and ID.")
        derivative = (
            (lambda time, state, coordinates, args: jnp.zeros_like(state))
            if time_derivative_provider is None
            else time_derivative_provider
        )
        entropy_supply = (
            (lambda time, state, coordinates, args: jnp.zeros(state.shape[:-1]))
            if entropy_supply_provider is None
            else entropy_supply_provider
        )
        if not callable(derivative) or not callable(entropy_supply):
            raise TypeError("Equilibrium derivative and entropy supply must be callable.")
        self.state_provider = state_provider
        self.time_derivative_provider = derivative
        self.entropy_supply_provider = entropy_supply
        self.equilibrium_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "well-balanced-equilibrium-plan",
                "equilibrium": identifier,
            }
        )

    def state(
        self, time: Array, coordinates: Array, args: Any, component_count: int, /
    ) -> Array:
        value = jnp.asarray(self.state_provider(time, coordinates, args))
        expected = coordinates.shape[:-1] + (int(component_count),)
        if value.shape != expected:
            raise ValueError("Equilibrium state shape is incompatible with FE DOFs.")
        return value

    def time_derivative(
        self, time: Array, state: Array, coordinates: Array, args: Any, /
    ) -> Array:
        value = jnp.asarray(self.time_derivative_provider(time, state, coordinates, args))
        if value.shape != state.shape:
            raise ValueError("Equilibrium time derivative changed state shape.")
        return value

    def entropy_supply(
        self, time: Array, state: Array, coordinates: Array, args: Any, /
    ) -> Array:
        value = jnp.asarray(self.entropy_supply_provider(time, state, coordinates, args))
        if value.shape != state.shape[:-1]:
            raise ValueError("Equilibrium entropy supply has incompatible shape.")
        return value


__all__ = ["WellBalancedEquilibriumPlan"]
