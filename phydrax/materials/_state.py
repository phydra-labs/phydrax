#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
from jaxtyping import Array

from .._strict import StrictModule
from .._trainable import NonTrainableState


class MaterialState(StrictModule, NonTrainableState):
    temperature_k: Array
    pressure_pa: Array
    phase_fractions: Array
    internal_variables: Array

    def __init__(
        self, temperature_k, pressure_pa, phase_fractions, internal_variables=()
    ):
        fractions = jnp.asarray(phase_fractions)
        if fractions.ndim < 1:
            raise ValueError("Phase fractions need a trailing phase axis.")
        self.temperature_k = jnp.asarray(temperature_k)
        self.pressure_pa = jnp.asarray(pressure_pa)
        self.phase_fractions = fractions
        self.internal_variables = jnp.asarray(internal_variables)

    @property
    def admissible(self):
        return (
            jnp.all(jnp.isfinite(self.temperature_k))
            & jnp.all(self.temperature_k > 0)
            & jnp.all(jnp.isfinite(self.pressure_pa))
            & jnp.all(self.phase_fractions >= 0)
            & jnp.allclose(jnp.sum(self.phase_fractions, axis=-1), 1)
        )


__all__ = ["MaterialState"]
