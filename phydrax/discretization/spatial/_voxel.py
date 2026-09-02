#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import product

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState

from ._morton import morton_decode_integer, morton_encode_integer, MortonAddressPlan


_UINT64_MAX = np.iinfo(np.uint64).max


class SparseVoxelBuildEvidence(NonTrainableState, StrictModule):
    """Canonical sparse-voxel preparation evidence."""

    active_voxels: jax.Array
    active_bricks: jax.Array
    duplicate_voxels: jax.Array
    brick_capacity: jax.Array
    successful: jax.Array


class SparseVoxelLookup(NonTrainableState, StrictModule):
    """Storage locations and support evidence for voxel coordinates."""

    brick_slots: jax.Array
    local_slots: jax.Array
    supported: jax.Array
    in_domain: jax.Array


class SparseVoxelQueryResult(StrictModule):
    """Sparse voxel samples with their complete interpolation stencil."""

    values: jax.Array
    supported: jax.Array
    stencil_complete: jax.Array
    brick_slots: jax.Array
    local_slots: jax.Array
    weights: jax.Array


class SparseVoxelDepositResult(StrictModule):
    """Deposited brick values and source support evidence."""

    values: jax.Array
    supported: jax.Array
    weight_sum: jax.Array


class PreparedSparseVoxelGrid(NonTrainableState, StrictModule):
    """Fixed-capacity canonical topology for sparse, fixed-resolution voxels."""

    address_plan: MortonAddressPlan
    brick_codes: jax.Array
    brick_active: jax.Array
    voxel_active: jax.Array
    evidence: SparseVoxelBuildEvidence
    brick_size: int = eqx.field(static=True)
    brick_depth: int = eqx.field(static=True)
    voxels_per_brick: int = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)

    @property
    def dimension(self) -> int:
        return self.address_plan.dimension

    @property
    def brick_capacity(self) -> int:
        return int(self.brick_codes.shape[0])

    def lookup_integer(self, integer_coordinates: jax.Array) -> SparseVoxelLookup:
        coordinates = jnp.asarray(integer_coordinates, dtype=jnp.int64)
        if coordinates.ndim < 1 or coordinates.shape[-1] != self.dimension:
            raise ValueError(
                f"integer_coordinates must have trailing dimension {self.dimension}."
            )
        periodic = jnp.asarray(self.address_plan.periodic_axes, dtype=bool)
        resolution = self.address_plan.resolution
        in_bounds_components = periodic | (
            (coordinates >= 0) & (coordinates < resolution)
        )
        in_domain = jnp.all(in_bounds_components, axis=-1)
        wrapped = jnp.where(periodic, jnp.mod(coordinates, resolution), coordinates)
        safe_coordinates = jnp.where(in_domain[..., None], wrapped, 0)
        brick_coordinates = safe_coordinates // self.brick_size
        if self.brick_depth == 0:
            brick_codes = jnp.zeros(coordinates.shape[:-1], dtype=jnp.uint64)
        else:
            brick_codes = morton_encode_integer(brick_coordinates, self.brick_depth)
        brick_slots = jnp.searchsorted(self.brick_codes, brick_codes, side="left").astype(
            jnp.int32
        )
        safe_bricks = jnp.minimum(brick_slots, self.brick_capacity - 1)
        brick_found = (
            (brick_slots < self.brick_capacity)
            & self.brick_active[safe_bricks]
            & (self.brick_codes[safe_bricks] == brick_codes)
        )
        local_coordinates = jnp.mod(safe_coordinates, self.brick_size)
        local_slots = jnp.zeros(coordinates.shape[:-1], dtype=jnp.int32)
        for axis in range(self.dimension):
            stride = self.brick_size ** (self.dimension - axis - 1)
            local_slots = local_slots + local_coordinates[..., axis].astype(
                jnp.int32
            ) * jnp.int32(stride)
        supported = in_domain & brick_found & self.voxel_active[safe_bricks, local_slots]
        return SparseVoxelLookup(
            brick_slots=jnp.where(supported, brick_slots, -1),
            local_slots=jnp.where(supported, local_slots, -1),
            supported=supported,
            in_domain=in_domain,
        )

    def voxel_centers(self) -> jax.Array:
        """Return physical centers for all brick slots and local voxel slots."""
        if self.brick_depth == 0:
            brick_coordinates = jnp.zeros(
                (self.brick_capacity, self.dimension), dtype=jnp.int64
            )
        else:
            brick_coordinates = morton_decode_integer(
                self.brick_codes,
                self.dimension,
                self.brick_depth,
            )
        local_coordinates = jnp.asarray(
            tuple(product(range(self.brick_size), repeat=self.dimension)),
            dtype=jnp.int64,
        )
        integer = (
            brick_coordinates[:, None, :] * self.brick_size
            + local_coordinates[None, :, :]
        )
        lower = jnp.asarray(self.address_plan.lower)
        upper = jnp.asarray(self.address_plan.upper)
        extent = upper - lower
        return (
            lower
            + extent * (integer.astype(extent.dtype) + 0.5) / self.address_plan.resolution
        )

    def interpolation_stencil(self, points: jax.Array):
        values = jnp.asarray(points)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(f"points must have shape (count, {self.dimension}).")
        encoding = self.address_plan.encode(values)
        lower = jnp.asarray(self.address_plan.lower, dtype=values.dtype)
        upper = jnp.asarray(self.address_plan.upper, dtype=values.dtype)
        index_coordinate = (encoding.coordinates - lower) / (
            upper - lower
        ) * self.address_plan.resolution - 0.5
        base = jnp.floor(index_coordinate).astype(jnp.int64)
        fraction = index_coordinate - base.astype(index_coordinate.dtype)
        corners = jnp.asarray(
            tuple(product((0, 1), repeat=self.dimension)), dtype=jnp.int64
        )
        integer = base[:, None, :] + corners[None, :, :]
        lookup = self.lookup_integer(integer)
        weights = jnp.ones((values.shape[0], corners.shape[0]), dtype=values.dtype)
        for axis in range(self.dimension):
            axis_weight = jnp.where(
                corners[None, :, axis] == 1,
                fraction[:, None, axis],
                1.0 - fraction[:, None, axis],
            )
            weights = weights * axis_weight
        in_domain = encoding.in_domain
        return lookup, weights, in_domain

    def deposit_multilinear(
        self,
        points: jax.Array,
        amounts: jax.Array,
        *,
        deterministic: bool = False,
    ) -> SparseVoxelDepositResult:
        point_values = jnp.asarray(points)
        amount_values = jnp.asarray(amounts)
        if amount_values.shape[0] != point_values.shape[0]:
            raise ValueError("amounts must have one leading entry per point.")
        lookup, weights, in_domain = self.interpolation_stencil(point_values)
        stencil_complete = in_domain & jnp.all(lookup.supported, axis=1)
        safe_bricks = jnp.maximum(lookup.brick_slots, 0)
        safe_local = jnp.maximum(lookup.local_slots, 0)
        flat_slots = safe_bricks * self.voxels_per_brick + safe_local
        trailing_shape = amount_values.shape[1:]
        contribution_weight = weights.reshape(weights.shape + (1,) * len(trailing_shape))
        contributions = amount_values[:, None, ...] * contribution_weight
        contributions = jnp.where(
            stencil_complete.reshape(
                stencil_complete.shape + (1,) * (contributions.ndim - 1)
            ),
            contributions,
            0.0,
        )
        flat_capacity = self.brick_capacity * self.voxels_per_brick
        flat_values = jnp.zeros(
            (flat_capacity,) + trailing_shape,
            dtype=amount_values.dtype,
        )
        flat_indices = flat_slots.reshape((-1,))
        flat_contributions = contributions.reshape(
            (flat_indices.shape[0],) + trailing_shape
        )
        if deterministic:

            def add_one(index, current):
                return current.at[flat_indices[index]].add(flat_contributions[index])

            flat_values = jax.lax.fori_loop(
                0, flat_indices.shape[0], add_one, flat_values
            )
        else:
            flat_values = flat_values.at[flat_indices].add(flat_contributions)
        return SparseVoxelDepositResult(
            values=flat_values.reshape(
                (self.brick_capacity, self.voxels_per_brick) + trailing_shape
            ),
            supported=stencil_complete,
            weight_sum=jnp.sum(
                jnp.where(stencil_complete[:, None], weights, 0.0), axis=1
            ),
        )


class SparseVoxelField(StrictModule):
    """Trainable values on a nontrainable sparse voxel topology."""

    grid: PreparedSparseVoxelGrid
    values: jax.Array
    background_value: jax.Array
    background_mode: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedSparseVoxelGrid,
        values: ArrayLike,
        /,
        *,
        background_mode: str = "unsupported",
        background_value: ArrayLike = 0.0,
    ) -> None:
        if not isinstance(grid, PreparedSparseVoxelGrid):
            raise TypeError("grid must be PreparedSparseVoxelGrid.")
        value_array = jnp.asarray(values)
        expected_prefix = (grid.brick_capacity, grid.voxels_per_brick)
        if value_array.shape[:2] != expected_prefix:
            raise ValueError("values must start with (brick_capacity, voxels_per_brick).")
        mode = str(background_mode)
        if mode not in {"unsupported", "constant"}:
            raise ValueError("background_mode must be 'unsupported' or 'constant'.")
        background = jnp.asarray(background_value, dtype=value_array.dtype)
        if background.shape not in ((), value_array.shape[2:]):
            raise ValueError("background_value must be scalar or match field shape.")
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "values", value_array)
        object.__setattr__(self, "background_value", background)
        object.__setattr__(self, "background_mode", mode)

    def sample_nearest(self, points: jax.Array) -> SparseVoxelQueryResult:
        encoding = self.grid.address_plan.encode(points)
        lookup = self.grid.lookup_integer(encoding.integer_coordinates)
        safe_bricks = jnp.maximum(lookup.brick_slots, 0)
        safe_local = jnp.maximum(lookup.local_slots, 0)
        gathered = self.values[safe_bricks, safe_local]
        fallback = jnp.broadcast_to(self.background_value, gathered.shape)
        sampled = jnp.where(
            lookup.supported.reshape(
                lookup.supported.shape + (1,) * (gathered.ndim - lookup.supported.ndim)
            ),
            gathered,
            fallback,
        )
        supported = (
            encoding.finite if self.background_mode == "constant" else lookup.supported
        )
        return SparseVoxelQueryResult(
            values=sampled,
            supported=supported,
            stencil_complete=lookup.supported,
            brick_slots=lookup.brick_slots[..., None],
            local_slots=lookup.local_slots[..., None],
            weights=jnp.ones(lookup.supported.shape + (1,), dtype=gathered.dtype),
        )

    def sample_multilinear(self, points: jax.Array) -> SparseVoxelQueryResult:
        lookup, weights, in_domain = self.grid.interpolation_stencil(points)
        safe_bricks = jnp.maximum(lookup.brick_slots, 0)
        safe_local = jnp.maximum(lookup.local_slots, 0)
        gathered = self.values[safe_bricks, safe_local]
        fallback = jnp.broadcast_to(self.background_value, gathered.shape)
        corner_values = jnp.where(
            lookup.supported.reshape(
                lookup.supported.shape + (1,) * (gathered.ndim - lookup.supported.ndim)
            ),
            gathered,
            fallback,
        )
        weight_values = weights.reshape(
            weights.shape + (1,) * (corner_values.ndim - weights.ndim)
        )
        sampled = jnp.sum(weight_values * corner_values, axis=1)
        stencil_complete = in_domain & jnp.all(lookup.supported, axis=1)
        supported = (
            jnp.all(jnp.isfinite(jnp.asarray(points)), axis=-1)
            if self.background_mode == "constant"
            else stencil_complete
        )
        return SparseVoxelQueryResult(
            values=sampled,
            supported=supported,
            stencil_complete=stencil_complete,
            brick_slots=lookup.brick_slots,
            local_slots=lookup.local_slots,
            weights=weights,
        )


class SparseVoxelGridPlan(StrictModule):
    """Prepare a fixed-resolution sparse voxel topology in aligned bricks."""

    address_plan: MortonAddressPlan
    brick_size: int = eqx.field(static=True)
    brick_depth: int = eqx.field(static=True)
    brick_capacity: int = eqx.field(static=True)
    voxels_per_brick: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        address_plan: MortonAddressPlan,
        *,
        brick_size: int,
        brick_capacity: int,
    ) -> None:
        size = int(brick_size)
        capacity = int(brick_capacity)
        if size < 1 or size & (size - 1):
            raise ValueError("brick_size must be a positive power of two.")
        brick_levels = int(np.log2(size))
        if brick_levels > address_plan.maximum_depth:
            raise ValueError("brick_size exceeds the Morton grid resolution.")
        if capacity < 1:
            raise ValueError("brick_capacity must be positive.")
        object.__setattr__(self, "address_plan", address_plan)
        object.__setattr__(self, "brick_size", size)
        object.__setattr__(self, "brick_depth", address_plan.maximum_depth - brick_levels)
        object.__setattr__(self, "brick_capacity", capacity)
        object.__setattr__(self, "voxels_per_brick", size**address_plan.dimension)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "sparse-voxel-grid-plan",
                    "address_plan_id": address_plan.plan_id,
                    "brick_size": size,
                    "brick_capacity": capacity,
                }
            ),
        )

    def prepare(self, active_coordinates: ArrayLike) -> PreparedSparseVoxelGrid:
        coordinates = np.asarray(active_coordinates, dtype=np.int64)
        if coordinates.ndim != 2 or coordinates.shape[1] != self.address_plan.dimension:
            raise ValueError(
                "active_coordinates must have shape (count, address dimension)."
            )
        if np.any(coordinates < 0) or np.any(coordinates >= self.address_plan.resolution):
            raise ValueError("active voxel coordinates lie outside the Morton grid.")
        unique_coordinates = np.unique(coordinates, axis=0)
        duplicate_count = int(coordinates.shape[0] - unique_coordinates.shape[0])
        brick_coordinates = unique_coordinates // self.brick_size
        if self.brick_depth == 0:
            voxel_brick_codes = np.zeros((unique_coordinates.shape[0],), dtype=np.uint64)
        else:
            voxel_brick_codes = np.asarray(
                morton_encode_integer(jnp.asarray(brick_coordinates), self.brick_depth)
            )
        unique_brick_codes, inverse = np.unique(voxel_brick_codes, return_inverse=True)
        required_bricks = int(unique_brick_codes.size)
        if required_bricks > self.brick_capacity:
            raise ValueError(
                f"Sparse voxel topology requires {required_bricks} bricks but "
                f"capacity is {self.brick_capacity}."
            )
        brick_codes = np.full((self.brick_capacity,), _UINT64_MAX, dtype=np.uint64)
        brick_codes[:required_bricks] = unique_brick_codes
        brick_active = np.zeros((self.brick_capacity,), dtype=bool)
        brick_active[:required_bricks] = True
        voxel_active = np.zeros((self.brick_capacity, self.voxels_per_brick), dtype=bool)
        local = unique_coordinates % self.brick_size
        strides = np.asarray(
            [
                self.brick_size ** (self.address_plan.dimension - axis - 1)
                for axis in range(self.address_plan.dimension)
            ],
            dtype=np.int64,
        )
        local_slots = np.sum(local * strides[None, :], axis=1)
        voxel_active[inverse, local_slots] = True
        grid_id = canonical_fingerprint(
            {
                "kind": "prepared-sparse-voxel-grid",
                "plan": self.plan_id,
                "coordinates": unique_coordinates.tolist(),
            }
        )
        evidence = SparseVoxelBuildEvidence(
            active_voxels=jnp.asarray(unique_coordinates.shape[0], dtype=jnp.int32),
            active_bricks=jnp.asarray(required_bricks, dtype=jnp.int32),
            duplicate_voxels=jnp.asarray(duplicate_count, dtype=jnp.int32),
            brick_capacity=jnp.asarray(self.brick_capacity, dtype=jnp.int32),
            successful=jnp.asarray(True),
        )
        return PreparedSparseVoxelGrid(
            address_plan=self.address_plan,
            brick_codes=jnp.asarray(brick_codes),
            brick_active=jnp.asarray(brick_active),
            voxel_active=jnp.asarray(voxel_active),
            evidence=evidence,
            brick_size=self.brick_size,
            brick_depth=self.brick_depth,
            voxels_per_brick=self.voxels_per_brick,
            grid_id=grid_id,
        )


__all__ = [
    "PreparedSparseVoxelGrid",
    "SparseVoxelBuildEvidence",
    "SparseVoxelDepositResult",
    "SparseVoxelField",
    "SparseVoxelGridPlan",
    "SparseVoxelLookup",
    "SparseVoxelQueryResult",
]
