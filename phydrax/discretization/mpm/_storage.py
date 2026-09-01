#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from itertools import product
from math import prod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState
from ..splatting import ParticleGridSplatState


class MPMActiveBlockState(StrictModule):
    active_block_ids: Array
    active_block_count: Array
    active_block_mask: Array
    current_previous_union: Array
    active_node_mask: Array
    overflow: Array
    successful: Array


class MPMActiveBlockPlan(StrictModule, NonTrainableState):
    grid_shape: tuple[int, ...] = eqx.field(static=True)
    block_shape: tuple[int, ...] = eqx.field(static=True)
    block_grid_shape: tuple[int, ...] = eqx.field(static=True)
    maximum_active_blocks: int = eqx.field(static=True)
    halo_blocks: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid_shape,
        block_shape,
        maximum_active_blocks: int,
        /,
        *,
        halo_blocks: int = 1,
    ):
        grid = tuple(int(value) for value in grid_shape)
        block = tuple(int(value) for value in block_shape)
        maximum = int(maximum_active_blocks)
        halo = int(halo_blocks)
        if (
            not grid
            or len(grid) != len(block)
            or any(value <= 0 for value in grid + block)
            or any(g % b != 0 for g, b in zip(grid, block, strict=True))
            or maximum <= 0
            or halo < 0
        ):
            raise ValueError("MPM active-block plan is invalid.")
        block_grid = tuple(g // b for g, b in zip(grid, block, strict=True))
        if maximum > prod(block_grid):
            raise ValueError("maximum_active_blocks exceeds the logical block grid.")
        self.grid_shape = grid
        self.block_shape = block
        self.block_grid_shape = block_grid
        self.maximum_active_blocks = maximum
        self.halo_blocks = halo
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mpm-active-block-plan",
                "grid_shape": grid,
                "block_shape": block,
                "maximum_active_blocks": maximum,
                "halo_blocks": halo,
            }
        )

    @property
    def block_count(self) -> int:
        return prod(self.block_grid_shape)

    @property
    def nodes_per_block(self) -> int:
        return prod(self.block_shape)

    def _block_ids(self, logical_indices):
        coordinates = jnp.stack(
            jnp.unravel_index(logical_indices, self.grid_shape), axis=-1
        )
        block_coordinates = coordinates // jnp.asarray(self.block_shape)
        block_id = block_coordinates[..., 0]
        for axis in range(1, len(self.grid_shape)):
            block_id = (
                block_id * self.block_grid_shape[axis] + block_coordinates[..., axis]
            )
        return block_id.astype(jnp.int32), block_coordinates

    def build(
        self,
        routes: ParticleGridSplatState,
        previous: MPMActiveBlockState | None = None,
        /,
    ) -> MPMActiveBlockState:
        logical = routes.stencil.indices.reshape((-1,))
        valid = routes.stencil.valid.reshape((-1,))
        block_ids, _ = self._block_ids(logical)
        mask = (
            jnp.zeros((self.block_count,), dtype=bool)
            .at[jnp.where(valid, block_ids, 0)]
            .set(valid)
        )
        block_coordinates = jnp.stack(
            jnp.unravel_index(
                jnp.arange(self.block_count, dtype=jnp.int32), self.block_grid_shape
            ),
            axis=-1,
        )
        dilated = mask
        for offset in product(
            range(-self.halo_blocks, self.halo_blocks + 1),
            repeat=len(self.grid_shape),
        ):
            shifted = block_coordinates + jnp.asarray(offset)
            inside = jnp.all(
                (shifted >= 0) & (shifted < jnp.asarray(self.block_grid_shape)),
                axis=-1,
            )
            flat = shifted[..., 0]
            for axis in range(1, len(self.grid_shape)):
                flat = flat * self.block_grid_shape[axis] + shifted[..., axis]
            source = jnp.where(inside, jnp.clip(flat, 0, self.block_count - 1), 0)
            dilated = dilated | (inside & mask[source])
        count = jnp.sum(dilated, dtype=jnp.int32)
        overflow = count > self.maximum_active_blocks
        ids = jnp.nonzero(
            dilated,
            size=self.maximum_active_blocks,
            fill_value=0,
        )[0].astype(jnp.int32)
        union = dilated if previous is None else (dilated | previous.active_block_mask)
        node_ids = jnp.arange(prod(self.grid_shape), dtype=jnp.int32)
        node_blocks, _ = self._block_ids(node_ids)
        node_mask = dilated[node_blocks].reshape(self.grid_shape)
        return MPMActiveBlockState(
            ids,
            count,
            dilated,
            union,
            node_mask,
            overflow,
            routes.successful & ~overflow,
        )


class AbstractMPMNodalStoragePlan(StrictModule, NonTrainableState):
    storage_id: AbstractAttribute[str]

    @abc.abstractmethod
    def pack(self, dense: Array, active: MPMActiveBlockState, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def unpack(self, compact: Array, active: MPMActiveBlockState, /) -> Array:
        raise NotImplementedError


class DenseMPMNodalStoragePlan(AbstractMPMNodalStoragePlan):
    grid_shape: tuple[int, ...] = eqx.field(static=True)
    storage_id: str = eqx.field(static=True)

    def __init__(self, grid_shape, /):
        self.grid_shape = tuple(int(value) for value in grid_shape)
        self.storage_id = canonical_fingerprint(
            {"kind": "dense-mpm-nodal-storage", "grid_shape": self.grid_shape}
        )

    def pack(self, dense, active, /):
        del active
        return jnp.asarray(dense)

    def unpack(self, compact, active, /):
        del active
        return jnp.asarray(compact)


class BlockSparseMPMNodalStoragePlan(AbstractMPMNodalStoragePlan):
    blocks: MPMActiveBlockPlan
    storage_id: str = eqx.field(static=True)

    def __init__(self, blocks: MPMActiveBlockPlan, /):
        if not isinstance(blocks, MPMActiveBlockPlan):
            raise TypeError("blocks must be MPMActiveBlockPlan.")
        self.blocks = blocks
        self.storage_id = canonical_fingerprint(
            {"kind": "block-sparse-mpm-storage", "blocks": blocks.plan_id}
        )

    def _block_coordinates(self, ids):
        return jnp.stack(jnp.unravel_index(ids, self.blocks.block_grid_shape), axis=-1)

    def _logical_indices(self, block_ids):
        dimension = len(self.blocks.grid_shape)
        block_coordinates = self._block_coordinates(block_ids)
        local = jnp.stack(
            jnp.unravel_index(
                jnp.arange(self.blocks.nodes_per_block), self.blocks.block_shape
            ),
            axis=-1,
        )
        coordinates = (
            block_coordinates[:, None, :] * jnp.asarray(self.blocks.block_shape)
            + local[None, :, :]
        )
        flat = coordinates[..., 0]
        for axis in range(1, dimension):
            flat = flat * self.blocks.grid_shape[axis] + coordinates[..., axis]
        return flat.astype(jnp.int32)

    def pack(self, dense: ArrayLike, active: MPMActiveBlockState, /) -> Array:
        value = jnp.asarray(dense)
        if value.shape[: len(self.blocks.grid_shape)] != self.blocks.grid_shape:
            raise ValueError("Dense MPM storage has the wrong grid prefix.")
        flat = value.reshape(
            (prod(self.blocks.grid_shape),) + value.shape[len(self.blocks.grid_shape) :]
        )
        logical = self._logical_indices(active.active_block_ids)
        packed = flat[logical]
        valid_blocks = (
            jnp.arange(self.blocks.maximum_active_blocks, dtype=jnp.int32)
            < active.active_block_count
        )
        return jnp.where(
            valid_blocks.reshape(valid_blocks.shape + (1,) * (packed.ndim - 1)),
            packed,
            0.0,
        )

    def unpack(self, compact: ArrayLike, active: MPMActiveBlockState, /) -> Array:
        value = jnp.asarray(compact)
        expected = (
            self.blocks.maximum_active_blocks,
            self.blocks.nodes_per_block,
        )
        if value.shape[:2] != expected:
            raise ValueError("Compact MPM storage has the wrong block prefix.")
        logical = self._logical_indices(active.active_block_ids).reshape((-1,))
        flat_values = value.reshape((logical.size,) + value.shape[2:])
        valid_blocks = (
            jnp.arange(self.blocks.maximum_active_blocks, dtype=jnp.int32)
            < active.active_block_count
        )
        valid = jnp.repeat(valid_blocks, self.blocks.nodes_per_block)
        flat = (
            jnp.zeros(
                (prod(self.blocks.grid_shape),) + value.shape[2:], dtype=value.dtype
            )
            .at[jnp.where(valid, logical, 0)]
            .add(
                jnp.where(
                    valid.reshape(valid.shape + (1,) * (flat_values.ndim - 1)),
                    flat_values,
                    0.0,
                )
            )
        )
        return flat.reshape(self.blocks.grid_shape + value.shape[2:])

    def map_route_indices(
        self, routes: ParticleGridSplatState, active: MPMActiveBlockState, /
    ) -> tuple[Array, Array]:
        block_ids, coordinates = self.blocks._block_ids(routes.stencil.indices)
        matches = block_ids[..., None] == active.active_block_ids
        block_slot = jnp.argmax(matches, axis=-1).astype(jnp.int32)
        block_found = jnp.any(matches, axis=-1)
        local_coordinates = coordinates % jnp.asarray(self.blocks.block_shape)
        local = local_coordinates[..., 0]
        for axis in range(1, len(self.blocks.grid_shape)):
            local = local * self.blocks.block_shape[axis] + local_coordinates[..., axis]
        compact = block_slot * self.blocks.nodes_per_block + local
        valid = routes.stencil.valid & block_found & ~active.overflow
        return compact.astype(jnp.int32), valid


__all__ = [
    "AbstractMPMNodalStoragePlan",
    "BlockSparseMPMNodalStoragePlan",
    "DenseMPMNodalStoragePlan",
    "MPMActiveBlockPlan",
    "MPMActiveBlockState",
]
