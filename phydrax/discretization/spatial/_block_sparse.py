#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from math import prod
from typing import Literal, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...sparse import (
    align_key_groups,
    KeyGroupPlan,
    KeyGroupState,
    KeyGroupTransition,
)
from ._morton import morton_decode_integer, morton_encode_integer


if TYPE_CHECKING:
    from .._tensor_index import PreparedTensorIndexSpace, TensorIndexLayout

BlockKeyOrdering = Literal["row-major", "morton"]


class SparseBlockBuildEvidence(NonTrainableState, StrictModule):
    """Complete capacity and support evidence for one block candidate."""

    requested_sites: Array
    valid_sites: Array
    invalid_sites: Array
    raw_required_blocks: Array
    required_blocks: Array
    block_capacity: Array
    closure_complete: Array
    overflow: Array
    successful: Array


class SparseBlockLookup(NonTrainableState, StrictModule):
    """Map logical sites to compact block and node storage slots."""

    block_slots: Array
    local_slots: Array
    storage_slots: Array
    in_domain: Array
    supported: Array


class SparseBlockTopologyPlan(StrictModule, NonTrainableState):
    """Fixed-capacity sparse outer blocks with dense local tensor sites."""

    index_space: PreparedTensorIndexSpace
    layout: TensorIndexLayout
    closure_offsets: Array
    block_shape: tuple[int, ...] = eqx.field(static=True)
    block_grid_shape: tuple[int, ...] = eqx.field(static=True)
    block_strides: tuple[int, ...] = eqx.field(static=True)
    local_strides: tuple[int, ...] = eqx.field(static=True)
    periodic_axes: tuple[bool, ...] = eqx.field(static=True)
    block_capacity: int = eqx.field(static=True)
    key_ordering: BlockKeyOrdering = eqx.field(static=True)
    morton_depth: int | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        index_space: PreparedTensorIndexSpace,
        block_shape: Sequence[int],
        block_capacity: int,
        /,
        *,
        layout: TensorIndexLayout | None = None,
        closure_offsets: Sequence[Sequence[int]] | None = None,
        key_ordering: BlockKeyOrdering = "row-major",
    ) -> None:
        from .._tensor_index import PreparedTensorIndexSpace, TensorIndexLayout

        if not isinstance(index_space, PreparedTensorIndexSpace):
            raise TypeError("index_space must be PreparedTensorIndexSpace.")
        layout_ = index_space.primary_entity_layout if layout is None else layout
        if not isinstance(layout_, TensorIndexLayout) or not any(
            candidate.layout_id == layout_.layout_id
            for candidate in index_space.entity_layouts
        ):
            raise ValueError("layout must belong to index_space.")
        blocks = tuple(int(size) for size in block_shape)
        if len(blocks) != len(layout_.shape) or any(size <= 0 for size in blocks):
            raise ValueError("block_shape must contain one positive size per axis.")
        if any(
            size % block != 0 for size, block in zip(layout_.shape, blocks, strict=True)
        ):
            raise ValueError("Every logical axis must be divisible by its block size.")
        capacity = int(block_capacity)
        if capacity <= 0:
            raise ValueError("block_capacity must be positive.")
        block_grid = tuple(
            size // block for size, block in zip(layout_.shape, blocks, strict=True)
        )
        if capacity > prod(block_grid):
            raise ValueError("block_capacity cannot exceed the logical block count.")
        if key_ordering not in ("row-major", "morton"):
            raise ValueError("key_ordering must be 'row-major' or 'morton'.")
        morton_depth: int | None = None
        if key_ordering == "morton":
            if len(block_grid) not in (1, 2, 3) or len(set(block_grid)) != 1:
                raise ValueError(
                    "Morton block ordering requires one equal grid size per axis."
                )
            resolution = block_grid[0]
            if resolution <= 1 or resolution & (resolution - 1):
                raise ValueError(
                    "Morton block ordering requires power-of-two resolution."
                )
            morton_depth = int(np.log2(resolution))
            if morton_depth * len(block_grid) > 63:
                raise ValueError("Morton block keys exceed the uint64 code budget.")
        offsets = (
            ((0,) * len(blocks),)
            if closure_offsets is None
            else tuple(
                tuple(int(value) for value in offset) for offset in closure_offsets
            )
        )
        if not offsets or any(len(offset) != len(blocks) for offset in offsets):
            raise ValueError(
                "closure_offsets must contain complete block-coordinate offsets."
            )
        offsets = tuple(sorted(set(offsets)))
        if (0,) * len(blocks) not in offsets:
            raise ValueError("closure_offsets must include the zero offset.")
        self.index_space = index_space
        self.layout = layout_
        self.closure_offsets = jnp.asarray(offsets, dtype=jnp.int32)
        self.block_shape = blocks
        self.block_grid_shape = block_grid
        self.block_strides = tuple(
            prod(block_grid[axis + 1 :]) for axis in range(len(block_grid))
        )
        self.local_strides = tuple(
            prod(blocks[axis + 1 :]) for axis in range(len(blocks))
        )
        self.periodic_axes = index_space.periodic_axes
        self.block_capacity = capacity
        self.key_ordering = key_ordering
        self.morton_depth = morton_depth
        self.plan_id = canonical_fingerprint(
            {
                "kind": "sparse-block-topology-plan",
                "index_space": index_space.prepared_id,
                "layout": layout_.layout_id,
                "block_shape": list(blocks),
                "block_capacity": capacity,
                "block_grid_shape": list(block_grid),
                "periodic_axes": list(self.periodic_axes),
                "closure_offsets": [list(offset) for offset in offsets],
                "key_ordering": key_ordering,
            }
        )

    @property
    def dimension(self) -> int:
        return len(self.block_shape)

    @property
    def logical_block_count(self) -> int:
        return prod(self.block_grid_shape)

    @property
    def sites_per_block(self) -> int:
        return prod(self.block_shape)

    @property
    def storage_capacity(self) -> int:
        return self.block_capacity * self.sites_per_block

    @property
    def key_upper_bound(self) -> int:
        return self.logical_block_count - 1

    def _encode_block_coordinates(self, coordinates: Array, /) -> Array:
        if self.key_ordering == "morton":
            assert self.morton_depth is not None
            return morton_encode_integer(coordinates, self.morton_depth)
        return jnp.sum(
            coordinates.astype(jnp.int64)
            * jnp.asarray(self.block_strides, dtype=jnp.int64),
            axis=-1,
        )

    def _decode_block_keys(self, keys: Array, /) -> Array:
        if self.key_ordering == "morton":
            assert self.morton_depth is not None
            return morton_decode_integer(keys, self.dimension, self.morton_depth).astype(
                jnp.int32
            )
        values = jnp.asarray(keys, dtype=jnp.int64)
        return jnp.stack(
            tuple(
                ((values // stride) % size).astype(jnp.int32)
                for stride, size in zip(
                    self.block_strides, self.block_grid_shape, strict=True
                )
            ),
            axis=-1,
        )

    def _site_parts(self, logical_indices: Array, /) -> tuple[Array, Array, Array, Array]:
        integer, in_domain = self.layout.integer_coordinates(logical_indices)
        block_coordinates = integer // jnp.asarray(self.block_shape, dtype=jnp.int32)
        local_coordinates = integer % jnp.asarray(self.block_shape, dtype=jnp.int32)
        keys = self._encode_block_coordinates(block_coordinates)
        local_slots = jnp.sum(
            local_coordinates * jnp.asarray(self.local_strides, dtype=jnp.int32),
            axis=-1,
        ).astype(jnp.int32)
        return keys, local_slots, block_coordinates, in_domain

    def build(
        self,
        logical_indices: ArrayLike,
        valid: ArrayLike | None = None,
        /,
        *,
        stable_site_ids: ArrayLike | None = None,
        generation: ArrayLike = 0,
    ) -> SparseBlockTopologyState:
        """Build one complete candidate from logical tensor-site IDs."""
        indices = jnp.asarray(logical_indices)
        if indices.ndim != 1 or not jnp.issubdtype(indices.dtype, jnp.integer):
            raise TypeError("logical_indices must be a rank-1 integer array.")
        requested = (
            jnp.ones(indices.shape, dtype=bool)
            if valid is None
            else jnp.asarray(valid, dtype=bool)
        )
        if requested.shape != indices.shape:
            raise ValueError("valid must match logical_indices.")
        keys, _, _, in_domain = self._site_parts(indices)
        site_valid = requested & in_domain
        raw_groups = KeyGroupPlan(
            indices.size,
            self.block_capacity,
            self.key_upper_bound,
        ).build(keys, site_valid, stable_ids=stable_site_ids)
        safe_raw_keys = jnp.where(raw_groups.group_active, raw_groups.group_keys, 0)
        raw_coordinates = self._decode_block_keys(safe_raw_keys)
        expanded = raw_coordinates[:, None, :] + self.closure_offsets[None, :, :]
        expanded_valid = jnp.broadcast_to(
            raw_groups.group_active[:, None], expanded.shape[:-1]
        )
        resolved_axes = []
        for axis, size in enumerate(self.block_grid_shape):
            coordinate = expanded[..., axis]
            if self.periodic_axes[axis]:
                coordinate = jnp.mod(coordinate, size)
            else:
                expanded_valid = expanded_valid & (coordinate >= 0) & (coordinate < size)
                coordinate = jnp.clip(coordinate, 0, size - 1)
            resolved_axes.append(coordinate)
        expanded_coordinates = jnp.stack(resolved_axes, axis=-1)
        expanded_keys = self._encode_block_coordinates(expanded_coordinates).reshape(
            (-1,)
        )
        expanded_valid = expanded_valid.reshape((-1,))
        closure_groups = KeyGroupPlan(
            expanded_keys.size,
            self.block_capacity,
            self.key_upper_bound,
        ).build(expanded_keys, expanded_valid)
        raw_success = raw_groups.evidence.successful
        closure_success = closure_groups.evidence.successful
        domain_success = jnp.all(~requested | in_domain)
        successful = raw_success & closure_success & domain_success
        safe_block_keys = jnp.where(
            closure_groups.group_active & successful,
            closure_groups.group_keys,
            0,
        )
        block_coordinates = self._decode_block_keys(safe_block_keys)
        local_coordinates = jnp.asarray(
            tuple(product(*(range(size) for size in self.block_shape))),
            dtype=jnp.int32,
        )
        logical_coordinates = (
            block_coordinates[:, None, :]
            * jnp.asarray(self.block_shape, dtype=jnp.int32)[None, None, :]
            + local_coordinates[None, :, :]
        )
        logical_node_ids, node_in_domain = self.layout.flat_indices(logical_coordinates)
        node_valid = closure_groups.group_active[:, None] & node_in_domain & successful
        logical_node_ids = jnp.where(node_valid, logical_node_ids, 0)
        evidence = SparseBlockBuildEvidence(
            requested_sites=jnp.sum(requested, dtype=jnp.int32),
            valid_sites=jnp.sum(site_valid, dtype=jnp.int32),
            invalid_sites=jnp.sum(requested & ~in_domain, dtype=jnp.int32),
            raw_required_blocks=raw_groups.evidence.required_groups,
            required_blocks=closure_groups.evidence.required_groups,
            block_capacity=jnp.asarray(self.block_capacity, dtype=jnp.int32),
            closure_complete=raw_success & closure_success,
            overflow=(
                raw_groups.evidence.group_overflow
                | closure_groups.evidence.group_overflow
            ),
            successful=successful,
        )
        return SparseBlockTopologyState(
            plan=self,
            groups=closure_groups,
            block_coordinates=block_coordinates,
            logical_node_ids=logical_node_ids,
            node_valid=node_valid,
            generation=jnp.asarray(generation, dtype=jnp.int32),
            evidence=evidence,
        )

    def refresh(
        self,
        previous: SparseBlockTopologyState,
        logical_indices: ArrayLike,
        valid: ArrayLike | None = None,
        /,
        *,
        stable_site_ids: ArrayLike | None = None,
    ) -> SparseBlockTransition:
        """Build and align a candidate without committing it."""
        if not isinstance(previous, SparseBlockTopologyState):
            raise TypeError("previous must be SparseBlockTopologyState.")
        if previous.plan.plan_id != self.plan_id:
            raise ValueError("previous topology belongs to another plan.")
        candidate = self.build(
            logical_indices,
            valid,
            stable_site_ids=stable_site_ids,
            generation=previous.generation,
        )
        key_transition = align_key_groups(previous.groups, candidate.groups)
        next_generation = previous.generation + (
            key_transition.topology_changed & candidate.evidence.successful
        ).astype(jnp.int32)
        candidate = eqx.tree_at(
            lambda topology: topology.generation,
            candidate,
            next_generation,
        )
        return SparseBlockTransition(
            previous=previous,
            candidate=candidate,
            key_transition=key_transition,
            successful=candidate.evidence.successful,
        )


class SparseBlockTopologyState(NonTrainableState, StrictModule):
    """Fixed-shape sparse blocks and their dense local logical sites."""

    plan: SparseBlockTopologyPlan
    groups: KeyGroupState
    block_coordinates: Array
    logical_node_ids: Array
    node_valid: Array
    generation: Array
    evidence: SparseBlockBuildEvidence

    def lookup(
        self,
        logical_indices: ArrayLike,
        valid: ArrayLike | None = None,
        /,
    ) -> SparseBlockLookup:
        indices = jnp.asarray(logical_indices)
        requested = (
            jnp.ones(indices.shape, dtype=bool)
            if valid is None
            else jnp.asarray(valid, dtype=bool)
        )
        if requested.shape != indices.shape:
            raise ValueError("valid must match logical_indices.")
        keys, local_slots, _, in_domain = self.plan._site_parts(indices)
        group_lookup = self.groups.lookup(keys, valid=requested & in_domain)
        supported = (
            group_lookup.supported & requested & in_domain & self.evidence.successful
        )
        storage_slots = (
            group_lookup.group_slots * self.plan.sites_per_block + local_slots
        ).astype(jnp.int32)
        return SparseBlockLookup(
            block_slots=jnp.where(supported, group_lookup.group_slots, 0),
            local_slots=jnp.where(supported, local_slots, 0),
            storage_slots=jnp.where(supported, storage_slots, 0),
            in_domain=in_domain,
            supported=supported,
        )

    def materialize_support(self, /) -> Array:
        flat = jnp.zeros((self.plan.layout.size,), dtype=bool)
        ids = self.logical_node_ids.reshape((-1,))
        valid = self.node_valid.reshape((-1,))
        flat = flat.at[jnp.where(valid, ids, 0)].max(valid)
        return flat.reshape(self.plan.layout.shape)


class SparseBlockTransition(NonTrainableState, StrictModule):
    """Uncommitted logical-key transition for persistent block fields."""

    previous: SparseBlockTopologyState
    candidate: SparseBlockTopologyState
    key_transition: KeyGroupTransition
    successful: Array


__all__ = [
    "BlockKeyOrdering",
    "SparseBlockBuildEvidence",
    "SparseBlockLookup",
    "SparseBlockTopologyPlan",
    "SparseBlockTopologyState",
    "SparseBlockTransition",
]
