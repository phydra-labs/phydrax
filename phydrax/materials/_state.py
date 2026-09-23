#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import equinox as eqx
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
        if fractions.ndim < 1 or fractions.shape[-1] == 0:
            raise ValueError("Phase fractions need a nonempty trailing phase axis.")
        temperature = jnp.asarray(temperature_k)
        pressure = jnp.asarray(pressure_pa)
        internal = jnp.asarray(internal_variables)
        fractions = eqx.error_if(
            fractions,
            jnp.any(~jnp.isfinite(fractions) | (fractions < 0))
            | jnp.any(~jnp.isfinite(temperature) | (temperature <= 0))
            | jnp.any(~jnp.isfinite(pressure) | (pressure <= 0))
            | jnp.any(~jnp.isfinite(internal))
            | jnp.any(~jnp.isclose(jnp.sum(fractions, axis=-1), 1.0)),
            "Material state temperature/pressure/composition/internal variables are inadmissible.",
        )
        self.temperature_k = temperature
        self.pressure_pa = pressure
        self.phase_fractions = fractions
        self.internal_variables = internal

    @property
    def admissible(self):
        return (
            jnp.all(jnp.isfinite(self.temperature_k))
            & jnp.all(self.temperature_k > 0)
            & jnp.all(jnp.isfinite(self.pressure_pa))
            & jnp.all(self.pressure_pa > 0)
            & jnp.all(jnp.isfinite(self.phase_fractions))
            & jnp.all(jnp.isfinite(self.internal_variables))
            & jnp.all(self.phase_fractions >= 0)
            & jnp.allclose(jnp.sum(self.phase_fractions, axis=-1), 1)
        )


__all__ = ["MaterialState"]
