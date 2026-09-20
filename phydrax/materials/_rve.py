#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


@dataclass(frozen=True, slots=True)
class RepresentativeVolumeElement:
    measure_weights: Array
    phase_id: Array

    @classmethod
    def create(cls, weights: ArrayLike, phase_id: ArrayLike):
        value = cls(jnp.asarray(weights), jnp.asarray(phase_id, dtype=jnp.int32))
        if value.measure_weights.shape != value.phase_id.shape or not bool(
            jnp.all(value.measure_weights > 0)
        ):
            raise ValueError("RVE weights/phases must align and be positive.")
        return value

    def average(self, field: ArrayLike):
        values = jnp.asarray(field)
        weights = self.measure_weights / jnp.sum(self.measure_weights)
        return jnp.sum(weights[(...,) + (None,) * (values.ndim - 1)] * values, axis=0)


__all__ = ["RepresentativeVolumeElement"]
