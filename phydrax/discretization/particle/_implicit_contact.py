#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._rigid_body import quaternion_rotation_matrix
from ._rigid_contact import RigidContactGeometry


class ImplicitRigidShapePlan(StrictModule, NonTrainableState):
    signed_distance_function: Callable[[Array], Array]
    normal_function: Callable[[Array], Array]
    lower_bound: Array
    upper_bound: Array
    distance_error_bound: float = eqx.field(static=True)
    lipschitz_bound: float = eqx.field(static=True)
    material_id: int = eqx.field(static=True)
    shape_id: str = eqx.field(static=True)

    def __init__(
        self,
        signed_distance_function: Callable[[Array], Array],
        normal_function: Callable[[Array], Array],
        lower_bound: Array,
        upper_bound: Array,
        material_id: int,
        /,
        *,
        distance_error_bound: float = 0.0,
        lipschitz_bound: float = 1.0,
        shape_id: str,
    ):
        if not callable(signed_distance_function) or not callable(normal_function):
            raise TypeError(
                "Implicit shape distance and normal functions must be callable."
            )
        lower = np.asarray(lower_bound)
        upper = np.asarray(upper_bound)
        error = float(distance_error_bound)
        lipschitz = float(lipschitz_bound)
        material = int(material_id)
        identifier = str(shape_id)
        if lower.ndim != 1 or lower.shape != upper.shape or lower.size not in (2, 3):
            raise ValueError("Implicit shape bounds must be matching 2-D/3-D vectors.")
        if (
            np.any(~np.isfinite(lower))
            or np.any(~np.isfinite(upper))
            or np.any(upper <= lower)
            or not np.isfinite(error)
            or error < 0.0
            or not np.isfinite(lipschitz)
            or lipschitz < 1.0
            or material < 0
            or not identifier
        ):
            raise ValueError(
                "Implicit shape certificate, bounds, material, or ID is invalid."
            )
        self.signed_distance_function = signed_distance_function
        self.normal_function = normal_function
        self.lower_bound = jnp.asarray(lower)
        self.upper_bound = jnp.asarray(upper)
        self.distance_error_bound = error
        self.lipschitz_bound = lipschitz
        self.material_id = material
        self.shape_id = canonical_fingerprint(
            {
                "kind": "implicit-rigid-shape-plan",
                "user_id": identifier,
                "bounds": [lower.tolist(), upper.tolist()],
                "distance_error_bound": error,
                "lipschitz_bound": lipschitz,
                "material_id": material,
            }
        )

    def signed_distance(self, local_points: Array, /) -> Array:
        value = self.signed_distance_function(local_points)
        value = jnp.asarray(value, dtype=local_points.dtype)
        if value.shape != local_points.shape[:-1]:
            raise ValueError("Implicit signed distance returned an invalid shape.")
        return value

    def normal(self, local_points: Array, /) -> Array:
        value = jnp.asarray(self.normal_function(local_points), dtype=local_points.dtype)
        if value.shape != local_points.shape:
            raise ValueError("Implicit normal returned an invalid shape.")
        norm = jnp.linalg.norm(value, axis=-1, keepdims=True)
        return value / jnp.where(norm > 0.0, norm, 1.0)


class ImplicitContactResult(StrictModule):
    geometry: RigidContactGeometry
    certified_distance_error: Array
    lipschitz_bound: Array
    successful: Array


def sphere_implicit_contact(
    sphere_position: Array,
    sphere_velocity: Array,
    sphere_angular_velocity: Array,
    sphere_radius: Array,
    implicit_shape: ImplicitRigidShapePlan,
    implicit_position: Array,
    implicit_velocity: Array,
    implicit_orientation: Array,
    implicit_angular_velocity: Array,
    contact_key: Array,
    /,
    *,
    certificate_tolerance: float = 1.0e-10,
) -> ImplicitContactResult:
    dimension = int(sphere_position.shape[0])
    if dimension not in (2, 3) or implicit_shape.lower_bound.shape != (dimension,):
        raise ValueError("Sphere/implicit contact dimension is invalid.")
    if dimension == 2:
        angle = implicit_orientation[0]
        rotation = jnp.asarray(
            (
                (jnp.cos(angle), -jnp.sin(angle)),
                (jnp.sin(angle), jnp.cos(angle)),
            )
        )
    else:
        rotation = quaternion_rotation_matrix(implicit_orientation[None, :])[0]
    local_center = contract("ji,j->i", rotation, sphere_position - implicit_position)
    local_center_batch = local_center[None, :]
    signed_distance = implicit_shape.signed_distance(local_center_batch)[0]
    local_normal = implicit_shape.normal(local_center_batch)[0]
    world_normal = contract("ij,j->i", rotation, local_normal)
    closest_point = sphere_position - signed_distance * world_normal
    gap = signed_distance - sphere_radius
    overlap = jnp.maximum(-gap, 0.0)
    sphere_arm = closest_point - sphere_position
    implicit_arm = closest_point - implicit_position
    if dimension == 2:
        sphere_spin = sphere_angular_velocity[0] * jnp.asarray(
            (-sphere_arm[1], sphere_arm[0])
        )
        implicit_spin = implicit_angular_velocity[0] * jnp.asarray(
            (-implicit_arm[1], implicit_arm[0])
        )
    else:
        sphere_spin = jnp.cross(sphere_angular_velocity, sphere_arm)
        implicit_spin = jnp.cross(implicit_angular_velocity, implicit_arm)
    relative = sphere_velocity + sphere_spin - implicit_velocity - implicit_spin
    normal_velocity = jnp.dot(relative, world_normal)
    tangential = relative - normal_velocity * world_normal
    error = jnp.asarray(implicit_shape.distance_error_bound, dtype=gap.dtype)
    certificate_margin = jnp.abs(gap) - error
    inside_bounds = jnp.all(
        (local_center >= implicit_shape.lower_bound)
        & (local_center <= implicit_shape.upper_bound)
    )
    normal_valid = jnp.isfinite(jnp.linalg.norm(world_normal)) & (
        jnp.linalg.norm(world_normal) > certificate_tolerance
    )
    valid = (
        inside_bounds
        & normal_valid
        & jnp.isfinite(signed_distance)
        & (certificate_margin > certificate_tolerance)
    )
    geometry = RigidContactGeometry(
        world_normal[None, :],
        gap[None],
        overlap[None],
        jnp.zeros((1,), dtype=gap.dtype),
        closest_point[None, :],
        sphere_arm[None, :],
        implicit_arm[None, :],
        sphere_arm[None, :],
        implicit_arm[None, :],
        relative[None, :],
        normal_velocity[None],
        tangential[None, :],
        sphere_angular_velocity[None, :],
        implicit_angular_velocity[None, :],
        jnp.asarray([contact_key], dtype=jnp.int64),
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.zeros((1,), dtype=jnp.int32),
        valid[None],
        jnp.where(valid, 0, 1).astype(jnp.int32)[None],
        certificate_margin[None],
        valid,
        "rigid-contact:sphere-implicit",
    )
    return ImplicitContactResult(
        geometry,
        error,
        jnp.asarray(implicit_shape.lipschitz_bound, dtype=gap.dtype),
        valid,
    )


__all__ = [
    "ImplicitContactResult",
    "ImplicitRigidShapePlan",
    "sphere_implicit_contact",
]
