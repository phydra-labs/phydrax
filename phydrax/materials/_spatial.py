#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Spatial material state and conservative field transfer."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..ein import contract
from ._state import MaterialState


@dataclass(frozen=True, slots=True)
class ConservativeMaterialTransfer:
    """Linear transfer preserving constants and weighted material integrals."""

    matrix: Array
    source_weights: Array
    target_weights: Array

    @classmethod
    def create(
        cls,
        matrix: ArrayLike,
        source_weights: ArrayLike,
        target_weights: ArrayLike,
        /,
        *,
        tolerance: float = 1e-10,
    ) -> ConservativeMaterialTransfer:
        interpolation = np.asarray(matrix, dtype=np.float64)
        source = np.asarray(source_weights, dtype=np.float64)
        target = np.asarray(target_weights, dtype=np.float64)
        if interpolation.ndim != 2:
            raise ValueError("Material transfer matrix must be two-dimensional.")
        if interpolation.shape != (target.size, source.size):
            raise ValueError("Material transfer weights do not match matrix dimensions.")
        if source.ndim != 1 or target.ndim != 1:
            raise ValueError("Material transfer measures must be vectors.")
        if np.any(source <= 0) or np.any(target <= 0):
            raise ValueError("Material transfer measures must be positive.")
        if np.any(interpolation < -tolerance):
            raise ValueError("Material transfer may not create negative phase content.")
        if not np.allclose(interpolation.sum(axis=1), 1.0, atol=tolerance, rtol=0):
            raise ValueError("Material transfer must preserve constant fields.")
        if not np.allclose(
            interpolation.T @ target, source, atol=tolerance, rtol=tolerance
        ):
            raise ValueError("Material transfer must preserve weighted integrals.")
        return cls(jnp.asarray(interpolation), jnp.asarray(source), jnp.asarray(target))

    def apply(self, field: ArrayLike, /) -> Array:
        values = jnp.asarray(field)
        if values.ndim < 1 or values.shape[0] != self.matrix.shape[1]:
            raise ValueError("Source material field does not match transfer topology.")
        return contract("ts,s...->t...", self.matrix, values)

    def conservation_residual(self, source: ArrayLike, target: ArrayLike, /) -> Array:
        source_values = jnp.asarray(source)
        target_values = jnp.asarray(target)
        source_integral = contract("s,s...->...", self.source_weights, source_values)
        target_integral = contract("t,t...->...", self.target_weights, target_values)
        return target_integral - source_integral


@dataclass(frozen=True, slots=True)
class SpatialMaterialField:
    """Material state sampled on weighted spatial control volumes."""

    coordinates_m: Array
    measure_weights: Array
    state: MaterialState

    @classmethod
    def create(
        cls,
        coordinates_m: ArrayLike,
        measure_weights: ArrayLike,
        state: MaterialState,
        /,
    ) -> SpatialMaterialField:
        coordinates = np.asarray(coordinates_m, dtype=np.float64)
        weights = np.asarray(measure_weights, dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[0] == 0:
            raise ValueError("Material coordinates require shape (point, dimension).")
        if weights.shape != (coordinates.shape[0],) or np.any(weights <= 0):
            raise ValueError("Material point measures must align and be positive.")
        point_count = coordinates.shape[0]
        if (
            np.shape(state.temperature_k) != (point_count,)
            or np.shape(state.pressure_pa) != (point_count,)
            or np.shape(state.phase_fractions)[0] != point_count
        ):
            raise ValueError("Material state fields must align with spatial points.")
        if not bool(state.admissible):
            raise ValueError("Spatial material state is inadmissible.")
        return cls(jnp.asarray(coordinates), jnp.asarray(weights), state)

    def integral(self, field: ArrayLike, /) -> Array:
        values = jnp.asarray(field)
        if values.ndim < 1 or values.shape[0] != self.measure_weights.size:
            raise ValueError("Integrated field must have one value per material point.")
        return contract("q,q...->...", self.measure_weights, values)

    def average(self, field: ArrayLike, /) -> Array:
        return self.integral(field) / jnp.sum(self.measure_weights)

    def transfer(
        self,
        target_coordinates_m: ArrayLike,
        transfer: ConservativeMaterialTransfer,
        /,
    ) -> SpatialMaterialField:
        coordinates = jnp.asarray(target_coordinates_m)
        if coordinates.shape[0] != transfer.target_weights.size:
            raise ValueError("Target coordinates do not match transfer topology.")
        if self.measure_weights.shape != transfer.source_weights.shape or not bool(
            jnp.allclose(self.measure_weights, transfer.source_weights)
        ):
            raise ValueError("Transfer source measures do not match the material field.")
        fractions = transfer.apply(self.state.phase_fractions)
        fractions = fractions / jnp.sum(fractions, axis=-1, keepdims=True)
        internal = self.state.internal_variables
        transferred_internal = (
            transfer.apply(internal)
            if internal.ndim > 0 and internal.shape[0] == self.measure_weights.size
            else internal
        )
        state = MaterialState(
            transfer.apply(self.state.temperature_k),
            transfer.apply(self.state.pressure_pa),
            fractions,
            transferred_internal,
        )
        return SpatialMaterialField.create(coordinates, transfer.target_weights, state)


__all__ = ["ConservativeMaterialTransfer", "SpatialMaterialField"]
