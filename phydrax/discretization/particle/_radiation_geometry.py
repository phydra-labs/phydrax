#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class VoxelRadiationLocation(StrictModule):
    material_index: Array
    voxel_index: Array
    inside: Array
    finite: Array
    successful: Array
    geometry_id: str = eqx.field(static=True)


class VoxelRadiationGeometryPlan(StrictModule, NonTrainableState):
    """Axis-aligned dense voxel material universe for delta tracking."""

    lower: Array
    upper: Array
    material_indices: Array
    voxel_width: Array
    material_count: int = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        lower: ArrayLike,
        upper: ArrayLike,
        material_indices: ArrayLike,
        /,
        *,
        material_count: int,
    ):
        lower_ = np.asarray(lower, dtype=float)
        upper_ = np.asarray(upper, dtype=float)
        materials = np.asarray(material_indices)
        count = int(material_count)
        if (
            lower_.shape != (3,)
            or upper_.shape != (3,)
            or np.any(~np.isfinite(lower_))
            or np.any(~np.isfinite(upper_))
            or np.any(upper_ <= lower_)
            or materials.ndim != 3
            or any(size < 1 for size in materials.shape)
            or not np.issubdtype(materials.dtype, np.integer)
            or count < 1
            or np.any(materials < 0)
            or np.any(materials >= count)
        ):
            raise ValueError("Voxel radiation bounds, materials, or count are invalid.")
        materials = materials.astype(np.int32, copy=False)
        width = (upper_ - lower_) / np.asarray(materials.shape)
        self.lower = jnp.asarray(lower_)
        self.upper = jnp.asarray(upper_)
        self.material_indices = jnp.asarray(materials)
        self.voxel_width = jnp.asarray(width)
        self.material_count = count
        self.geometry_id = canonical_fingerprint(
            {
                "kind": "dense-voxel-radiation-geometry",
                "lower": array_tree_fingerprint(lower_),
                "upper": array_tree_fingerprint(upper_),
                "materials": array_tree_fingerprint(materials),
                "material_count": count,
            }
        )

    def locate(self, position: ArrayLike, /) -> VoxelRadiationLocation:
        point = jnp.asarray(position, dtype=self.lower.dtype)
        if point.shape[-1:] != (3,):
            raise ValueError("Radiation positions must end in a three-vector.")
        finite = jnp.all(jnp.isfinite(point), axis=-1)
        inside = (
            finite
            & jnp.all(point >= self.lower, axis=-1)
            & jnp.all(point < self.upper, axis=-1)
        )
        index = jnp.floor((point - self.lower) / self.voxel_width).astype(jnp.int32)
        shape = jnp.asarray(self.material_indices.shape, dtype=jnp.int32)
        safe = jnp.clip(index, 0, shape - 1)
        material = self.material_indices[safe[..., 0], safe[..., 1], safe[..., 2]]
        return VoxelRadiationLocation(
            jnp.where(inside, material, -1),
            safe,
            inside,
            finite,
            finite,
            self.geometry_id,
        )

    def distance_to_exit(self, position: ArrayLike, direction: ArrayLike, /) -> Array:
        point = jnp.asarray(position, dtype=self.lower.dtype)
        ray = jnp.asarray(direction, dtype=self.lower.dtype)
        if point.shape != ray.shape or point.shape[-1:] != (3,):
            raise ValueError("Radiation position/direction shapes must match.")
        positive = ray > 0.0
        negative = ray < 0.0
        distance = jnp.where(
            positive,
            (self.upper - point) / ray,
            jnp.where(negative, (self.lower - point) / ray, jnp.inf),
        )
        forward = jnp.where(distance >= 0.0, distance, jnp.inf)
        return jnp.min(forward, axis=-1)

    def distance_to_voxel_boundary(
        self, position: ArrayLike, direction: ArrayLike, /
    ) -> Array:
        point = jnp.asarray(position, dtype=self.lower.dtype)
        ray = jnp.asarray(direction, dtype=self.lower.dtype)
        if point.shape != ray.shape or point.shape[-1:] != (3,):
            raise ValueError("Radiation position/direction shapes must match.")
        location = self.locate(point)
        voxel_lower = self.lower + location.voxel_index * self.voxel_width
        voxel_upper = voxel_lower + self.voxel_width
        distance = jnp.where(
            ray > 0.0,
            (voxel_upper - point) / ray,
            jnp.where(ray < 0.0, (voxel_lower - point) / ray, jnp.inf),
        )
        positive = jnp.where(distance > 1.0e-12, distance, jnp.inf)
        return jnp.min(positive, axis=-1)


__all__ = ["VoxelRadiationGeometryPlan", "VoxelRadiationLocation"]
