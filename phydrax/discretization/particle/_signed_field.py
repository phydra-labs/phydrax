#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Posed signed-field values with explicit spatial enclosure evidence.

For a supported query, every point within distance ``r`` of the query has a
field value in ``value ± (lipschitz_bound * r + evaluation_error)``. This is a
field certificate, not a claim that every signed field is a Euclidean SDF.
In particular, a sphere clump uses the minimum of its component sphere fields.
Unsupported lanes carry NaN values and infinite bounds/errors, never a usable
surrogate certificate. ``successful`` admits the complete query batch.
"""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from ..._strict import StrictModule
from ._rigid_body import quaternion_rotation_matrix, RigidBodyKinematics


@final
class PosedSignedFieldQueryResult(StrictModule):
    value: Array
    lipschitz_bound: Array
    evaluation_error: Array
    supported: Array
    successful: Array
    shape_id: str = eqx.field(static=True)


def posed_signed_field_coordinates(
    kinematics: RigidBodyKinematics,
    body_indices: Array,
    world_points: Array,
    active_mask: Array,
    /,
    *,
    shape_scale: Array | None = None,
    rotate: bool = True,
) -> tuple[Array, Array, Array, Array]:
    """Validate route structure and map queries to COM-centered body coordinates."""
    capacity, dimension = kinematics.position.shape
    if body_indices.ndim != 1 or not jnp.issubdtype(body_indices.dtype, jnp.integer):
        raise TypeError("body_indices must be a rank-1 integer array.")
    if world_points.shape != (body_indices.shape[0], dimension):
        raise ValueError("world_points must have shape (queries, ambient_dimension).")
    orientation_width = 1 if dimension == 2 else 4
    if dimension not in (2, 3) or kinematics.orientation.shape != (
        capacity,
        orientation_width,
    ):
        raise ValueError("Signed-field queries require a matching 2-D or 3-D pose.")
    if active_mask.shape != (capacity,):
        raise ValueError("active_mask must match body capacity.")
    scale = (
        jnp.ones((capacity,), dtype=kinematics.position.dtype)
        if shape_scale is None
        else shape_scale
    )
    if scale.shape != (capacity,):
        raise ValueError("shape_scale must match body capacity.")
    indices = jnp.clip(body_indices, 0, capacity - 1)
    route_scale = scale[indices]
    orientation = kinematics.orientation[indices]
    finite_orientation = jnp.all(jnp.isfinite(orientation), axis=-1)
    rotation_valid = finite_orientation
    if dimension == 3:
        norm_squared = jnp.sum(orientation * orientation, axis=-1)
        rotation_valid = rotation_valid & (
            jnp.abs(norm_squared - 1.0) <= 64.0 * jnp.finfo(orientation.dtype).eps
        )
    supported = (
        (body_indices >= 0)
        & (body_indices < capacity)
        & active_mask[indices]
        & jnp.isfinite(route_scale)
        & (route_scale > 0.0)
        & rotation_valid
        & jnp.all(jnp.isfinite(world_points), axis=-1)
        & jnp.all(jnp.isfinite(kinematics.position[indices]), axis=-1)
    )
    safe_scale = jnp.where(supported, route_scale, 1.0)
    displacement = (world_points - kinematics.position[indices]) / safe_scale[:, None]
    if not rotate:
        local = displacement
    elif dimension == 2:
        angle = orientation[:, 0]
        cosine, sine = jnp.cos(angle), jnp.sin(angle)
        local = jnp.stack(
            (
                cosine * displacement[:, 0] + sine * displacement[:, 1],
                -sine * displacement[:, 0] + cosine * displacement[:, 1],
            ),
            axis=-1,
        )
    else:
        rotation = quaternion_rotation_matrix(orientation)
        local = (jnp.swapaxes(rotation, -1, -2) @ displacement[:, :, None])[:, :, 0]
    return local, indices, safe_scale, supported


def exact_signed_field_result(
    value: Array, supported: Array, shape_id: str, /
) -> PosedSignedFieldQueryResult:
    """Publish exact field evaluations with their global unit Lipschitz bound."""
    supported = supported & jnp.isfinite(value)
    return PosedSignedFieldQueryResult(
        jnp.where(supported, value, jnp.nan),
        jnp.where(supported, jnp.ones_like(value), jnp.inf),
        jnp.where(supported, jnp.zeros_like(value), jnp.inf),
        supported,
        jnp.all(supported),
        shape_id,
    )


__all__ = ["PosedSignedFieldQueryResult"]
