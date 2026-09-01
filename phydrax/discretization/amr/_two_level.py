#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class TwoLevelAMRState(StrictModule):
    coarse_cell_average: Array
    fine_cell_average: Array
    refined_parent_mask: Array
    scale_factor: Array


class TwoLevelAMRPlan(StrictModule, NonTrainableState):
    coarse_shape: tuple[int, ...] = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    refinement_ratio: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, coarse_shape: tuple[int, ...], component_count: int, /):
        shape = tuple(int(value) for value in coarse_shape)
        components = int(component_count)
        if (
            not shape
            or len(shape) not in (1, 2, 3)
            or any(value <= 0 for value in shape)
            or components <= 0
        ):
            raise ValueError("Two-level AMR shape/components are invalid.")
        self.coarse_shape = shape
        self.component_count = components
        self.refinement_ratio = 2
        self.plan_id = canonical_fingerprint(
            {
                "kind": "two-level-ratio-two-amr",
                "coarse_shape": list(shape),
                "component_count": components,
            }
        )

    @property
    def fine_shape(self) -> tuple[int, ...]:
        return tuple(2 * value for value in self.coarse_shape)

    def initialize(
        self,
        coarse_cell_average: ArrayLike,
        refined_parent_mask: ArrayLike,
        scale_factor: ArrayLike,
        /,
    ) -> TwoLevelAMRState:
        coarse = jnp.asarray(coarse_cell_average)
        mask = jnp.asarray(refined_parent_mask, dtype=bool)
        if (
            coarse.shape != self.coarse_shape + (self.component_count,)
            or mask.shape != self.coarse_shape
        ):
            raise ValueError("Two-level AMR coarse state/mask shapes are invalid.")
        fine = self.prolong(coarse)
        return TwoLevelAMRState(
            coarse, fine, mask, jnp.asarray(scale_factor, dtype=coarse.dtype)
        )

    def prolong(self, coarse: ArrayLike, /) -> Array:
        values = jnp.asarray(coarse)
        result = values
        for axis in range(len(self.coarse_shape)):
            result = jnp.repeat(result, 2, axis=axis)
        return result

    def restrict(self, fine: ArrayLike, /) -> Array:
        values = jnp.asarray(fine)
        expected = self.fine_shape + (self.component_count,)
        if values.shape != expected:
            raise ValueError(f"Fine AMR values must have shape {expected}.")
        reshaped = values
        for axis in reversed(range(len(self.coarse_shape))):
            shape = reshaped.shape
            reshaped = reshaped.reshape(
                shape[:axis] + (self.coarse_shape[axis], 2) + shape[axis + 1 :]
            )
            reshaped = jnp.mean(reshaped, axis=axis + 1)
        return reshaped

    def average_down(self, state: TwoLevelAMRState, /) -> TwoLevelAMRState:
        restricted = self.restrict(state.fine_cell_average)
        coarse = jnp.where(
            state.refined_parent_mask[..., None],
            restricted,
            state.coarse_cell_average,
        )
        return TwoLevelAMRState(
            coarse, state.fine_cell_average, state.refined_parent_mask, state.scale_factor
        )


class CoarseFineFluxRegister(StrictModule):
    coarse_flux_integral: Array
    fine_flux_integral: Array
    mismatch: Array
    finite: Array

    def __init__(self, coarse_flux_integral: ArrayLike, fine_flux_integral: ArrayLike, /):
        coarse = jnp.asarray(coarse_flux_integral)
        fine = jnp.asarray(fine_flux_integral, dtype=coarse.dtype)
        if coarse.shape != fine.shape:
            raise ValueError("Coarse and fine flux integrals must share a face layout.")
        mismatch = coarse - fine
        self.coarse_flux_integral = coarse
        self.fine_flux_integral = fine
        self.mismatch = mismatch
        self.finite = jnp.all(jnp.isfinite(mismatch))

    def reflux(
        self,
        coarse_cell_average: ArrayLike,
        left_cell_indices: ArrayLike,
        right_cell_indices: ArrayLike,
        cell_volumes: ArrayLike,
        /,
    ) -> Array:
        values = jnp.asarray(coarse_cell_average)
        left = jnp.asarray(left_cell_indices, dtype=jnp.int32)
        right = jnp.asarray(right_cell_indices, dtype=jnp.int32)
        volume = jnp.asarray(cell_volumes, dtype=values.dtype)
        flat = values.reshape((-1, values.shape[-1]))
        flat = flat.at[left].add(-self.mismatch / volume[left, None])
        flat = flat.at[right].add(self.mismatch / volume[right, None])
        return flat.reshape(values.shape)


__all__ = [
    "CoarseFineFluxRegister",
    "TwoLevelAMRPlan",
    "TwoLevelAMRState",
]
