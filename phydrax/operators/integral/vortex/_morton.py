#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization.vortex._source import VortexSourceState


class MortonHierarchyState(StrictModule):
    codes: Array
    permutation: Array
    inverse_permutation: Array
    cell_occupancy: Array
    maximum_occupancy: Array
    overflow_count: Array
    in_bounds: Array
    hierarchy_id: str = eqx.field(static=True)


class MortonHierarchyTransition(StrictModule):
    candidate: MortonHierarchyState
    accepted: MortonHierarchyState
    rebuilt: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class VortexMortonHierarchyPlan(StrictModule, NonTrainableState):
    lower: Array
    upper: Array
    bits_per_axis: int = eqx.field(static=True)
    maximum_cell_occupancy: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        lower: ArrayLike,
        upper: ArrayLike,
        /,
        *,
        bits_per_axis: int = 10,
        maximum_cell_occupancy: int = 64,
    ):
        lower_, upper_ = jnp.asarray(lower, dtype=float), jnp.asarray(upper, dtype=float)
        dimension, bits, maximum = (
            int(lower_.size),
            int(bits_per_axis),
            int(maximum_cell_occupancy),
        )
        if (
            lower_.shape != upper_.shape
            or dimension not in (2, 3)
            or jnp.any(upper_ <= lower_)
            or bits <= 0
            or bits > 20
            or maximum <= 0
        ):
            raise ValueError("Morton hierarchy bounds/bits/capacity are invalid.")
        (
            self.lower,
            self.upper,
            self.bits_per_axis,
            self.maximum_cell_occupancy,
            self.dimension,
        ) = lower_, upper_, bits, maximum, dimension
        self.plan_id = canonical_fingerprint(
            {
                "kind": "vortex-morton-hierarchy",
                "lower": tuple(float(value) for value in lower_),
                "upper": tuple(float(value) for value in upper_),
                "bits_per_axis": bits,
                "maximum_cell_occupancy": maximum,
            }
        )

    def _spread_bits(self, value: Array, /) -> Array:
        result = jnp.zeros_like(value, dtype=jnp.uint64)
        for bit in range(self.bits_per_axis):
            result = result | (
                ((value.astype(jnp.uint64) >> bit) & 1) << (self.dimension * bit)
            )
        return result

    def build(self, source: VortexSourceState, /) -> MortonHierarchyState:
        if source.dimension != self.dimension:
            raise ValueError("Morton hierarchy source dimension is incompatible.")
        normalized = (source.safe_positions() - self.lower) / (self.upper - self.lower)
        in_bounds = source.active_mask & jnp.all(
            (normalized >= 0.0) & (normalized < 1.0), axis=-1
        )
        scale = 2**self.bits_per_axis
        integer = jnp.clip(jnp.floor(normalized * scale).astype(jnp.uint32), 0, scale - 1)
        codes = jnp.zeros((source.capacity,), dtype=jnp.uint64)
        for axis in range(self.dimension):
            codes = codes | (self._spread_bits(integer[:, axis]) << axis)
        codes = jnp.where(source.active_mask, codes, jnp.iinfo(jnp.uint64).max)
        permutation = jnp.lexsort((jnp.arange(source.capacity), codes))
        inverse = (
            jnp.zeros_like(permutation)
            .at[permutation]
            .set(jnp.arange(source.capacity, dtype=permutation.dtype))
        )
        sorted_codes = codes[permutation]
        new_cell = jnp.concatenate(
            (jnp.asarray((True,)), sorted_codes[1:] != sorted_codes[:-1])
        )
        cell_id = jnp.cumsum(new_cell.astype(jnp.int32)) - 1
        occupancy = (
            jnp.zeros((source.capacity,), dtype=jnp.int32)
            .at[cell_id]
            .add((sorted_codes != jnp.iinfo(jnp.uint64).max).astype(jnp.int32))
        )
        maximum_occupancy = jnp.max(occupancy)
        overflow = jnp.sum(jnp.maximum(occupancy - self.maximum_cell_occupancy, 0))
        successful = jnp.all(~source.active_mask | in_bounds) & (overflow == 0)
        identifier = canonical_fingerprint(
            {
                "kind": "morton-hierarchy-state",
                "plan": self.plan_id,
                "capacity": source.capacity,
            }
        )
        return MortonHierarchyState(
            codes,
            permutation,
            inverse,
            occupancy,
            maximum_occupancy,
            overflow,
            successful,
            identifier,
        )

    def rebuild(
        self, previous: MortonHierarchyState, source: VortexSourceState, /
    ) -> MortonHierarchyTransition:
        candidate = self.build(source)
        accepted = MortonHierarchyState(
            jnp.where(candidate.in_bounds, candidate.codes, previous.codes),
            jnp.where(candidate.in_bounds, candidate.permutation, previous.permutation),
            jnp.where(
                candidate.in_bounds,
                candidate.inverse_permutation,
                previous.inverse_permutation,
            ),
            jnp.where(
                candidate.in_bounds, candidate.cell_occupancy, previous.cell_occupancy
            ),
            jnp.where(
                candidate.in_bounds,
                candidate.maximum_occupancy,
                previous.maximum_occupancy,
            ),
            jnp.where(
                candidate.in_bounds, candidate.overflow_count, previous.overflow_count
            ),
            candidate.in_bounds,
            candidate.hierarchy_id,
        )
        return MortonHierarchyTransition(
            candidate, accepted, candidate.in_bounds, candidate.in_bounds, self.plan_id
        )


__all__ = [
    "MortonHierarchyState",
    "MortonHierarchyTransition",
    "VortexMortonHierarchyPlan",
]
