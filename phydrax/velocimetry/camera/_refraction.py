#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class RefractionStatus(IntEnum):
    """Portable status for a ray traversing a planar refractive stack."""

    SUCCESS = 0
    NONFINITE_INPUT = 1
    INVALID_DIRECTION = 2
    PARALLEL_INTERFACE = 3
    INTERFACE_BEHIND_RAY = 4
    TOTAL_INTERNAL_REFLECTION = 5
    NONCONVERGENCE = 6


class RefractiveLayerStack(StrictModule, NonTrainableState):
    """Fixed-capacity ordered planar interfaces between homogeneous media.

    Interface normals point from medium ``i`` toward medium ``i + 1``. Active
    interfaces form a prefix and are traversed in array order away from the camera.
    """

    interface_points: Array
    interface_normals: Array
    refractive_indices: Array
    interface_valid: Array
    capacity: int = eqx.field(static=True)
    active_count: int = eqx.field(static=True)
    stack_id: str = eqx.field(static=True)

    def __init__(
        self,
        interface_points: ArrayLike,
        interface_normals: ArrayLike,
        refractive_indices: ArrayLike,
        *,
        interface_valid: ArrayLike | None = None,
    ):
        points_host = np.asarray(interface_points, dtype=float)
        normals_host = np.asarray(interface_normals, dtype=float)
        indices_host = np.asarray(refractive_indices, dtype=float)
        if points_host.ndim != 2 or points_host.shape[1:] != (3,):
            raise ValueError("interface_points must have shape (capacity, 3).")
        capacity = int(points_host.shape[0])
        if normals_host.shape != (capacity, 3):
            raise ValueError("interface_normals must match interface_points.")
        if indices_host.shape != (capacity + 1,):
            raise ValueError("refractive_indices must have shape (capacity + 1,).")
        if not (
            np.all(np.isfinite(points_host))
            and np.all(np.isfinite(normals_host))
            and np.all(np.isfinite(indices_host))
        ):
            raise ValueError("Refractive layer data must be finite.")
        normal_norms = np.linalg.norm(normals_host, axis=-1)
        if capacity > 0 and np.any(normal_norms <= 0.0):
            raise ValueError("Interface normals must have non-zero length.")
        if np.any(indices_host <= 0.0):
            raise ValueError("Refractive indices must be positive.")
        if interface_valid is None:
            valid_host = np.ones((capacity,), dtype=bool)
        else:
            valid_host = np.asarray(interface_valid, dtype=bool)
            if valid_host.shape != (capacity,):
                raise ValueError("interface_valid must have shape (capacity,).")
        active_count = int(np.sum(valid_host))
        if not np.array_equal(
            valid_host,
            np.arange(capacity, dtype=int) < active_count,
        ):
            raise ValueError("Active interfaces must form a prefix of the stack.")
        normalized = (
            normals_host / normal_norms[:, None] if capacity > 0 else normals_host
        )
        self.interface_points = jnp.asarray(points_host)
        self.interface_normals = jnp.asarray(normalized)
        self.refractive_indices = jnp.asarray(indices_host)
        self.interface_valid = jnp.asarray(valid_host)
        self.capacity = capacity
        self.active_count = active_count
        self.stack_id = canonical_fingerprint(
            {
                "kind": "planar-refractive-layer-stack",
                "capacity": capacity,
                "active_count": active_count,
                "data": array_tree_fingerprint(
                    (points_host, normalized, indices_host, valid_host)
                ),
            }
        )


class RefractionResult(StrictModule, NonTrainableState):
    """Final ray state and per-ray failure evidence from a layer stack."""

    origins: Array
    directions: Array
    valid: Array
    status: Array
    traversed_interfaces: Array
    minimum_discriminant: Array


def _trace_refracted_arrays(
    stack: RefractiveLayerStack,
    origins: Array,
    directions: Array,
    *,
    parallel_tolerance: float,
    intersection_tolerance: float,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    dtype = jnp.result_type(origins, directions, stack.interface_points, 0.0)
    origin = origins.astype(dtype)
    raw_direction = directions.astype(dtype)
    norm = jnp.sqrt(jnp.sum(raw_direction * raw_direction, axis=-1))
    finite_input = jnp.all(jnp.isfinite(origin), axis=-1) & jnp.all(
        jnp.isfinite(raw_direction), axis=-1
    )
    direction_ok = norm > jnp.asarray(parallel_tolerance, dtype=dtype)
    safe_norm = jnp.where(direction_ok, norm, 1.0)
    direction = raw_direction / safe_norm[..., None]
    valid = finite_input & direction_ok
    status = jnp.where(
        finite_input,
        jnp.where(
            direction_ok,
            int(RefractionStatus.SUCCESS),
            int(RefractionStatus.INVALID_DIRECTION),
        ),
        int(RefractionStatus.NONFINITE_INPUT),
    ).astype(jnp.int32)
    traversed = jnp.zeros(origin.shape[:-1], dtype=jnp.int32)
    minimum_discriminant = jnp.full(origin.shape[:-1], jnp.inf, dtype=dtype)

    for index in range(stack.capacity):
        active = valid & stack.interface_valid[index]
        point = stack.interface_points[index].astype(dtype)
        normal = stack.interface_normals[index].astype(dtype)
        denominator = jnp.sum(direction * normal, axis=-1)
        parallel = denominator <= parallel_tolerance
        safe_denominator = jnp.where(parallel, 1.0, denominator)
        distance = jnp.sum((point - origin) * normal, axis=-1) / safe_denominator
        behind = distance < -intersection_tolerance
        intersection = origin + distance[..., None] * direction

        normal_component = jnp.sum(direction * normal, axis=-1)
        tangent = direction - normal_component[..., None] * normal
        ratio = (
            stack.refractive_indices[index] / stack.refractive_indices[index + 1]
        ).astype(dtype)
        transmitted_tangent = ratio * tangent
        discriminant = 1.0 - jnp.sum(
            transmitted_tangent * transmitted_tangent,
            axis=-1,
        )
        total_internal_reflection = discriminant < 0.0
        transmitted = (
            transmitted_tangent
            + jnp.sqrt(jnp.maximum(discriminant, 0.0))[..., None] * normal
        )
        transmitted_norm = jnp.sqrt(jnp.sum(transmitted * transmitted, axis=-1))
        transmitted = (
            transmitted
            / jnp.where(
                transmitted_norm > 0.0,
                transmitted_norm,
                1.0,
            )[..., None]
        )

        failure = jnp.where(
            parallel,
            int(RefractionStatus.PARALLEL_INTERFACE),
            jnp.where(
                behind,
                int(RefractionStatus.INTERFACE_BEHIND_RAY),
                jnp.where(
                    total_internal_reflection,
                    int(RefractionStatus.TOTAL_INTERNAL_REFLECTION),
                    int(RefractionStatus.SUCCESS),
                ),
            ),
        ).astype(jnp.int32)
        step_success = active & (failure == int(RefractionStatus.SUCCESS))
        step_failure = active & ~step_success
        status = jnp.where(step_failure, failure, status)
        origin = jnp.where(step_success[..., None], intersection, origin)
        direction = jnp.where(step_success[..., None], transmitted, direction)
        traversed = traversed + step_success.astype(jnp.int32)
        minimum_discriminant = jnp.where(
            active,
            jnp.minimum(minimum_discriminant, discriminant),
            minimum_discriminant,
        )
        valid = valid & ~step_failure

    return origin, direction, valid, status, traversed, minimum_discriminant


def trace_refracted_rays(
    stack: RefractiveLayerStack,
    origins: ArrayLike,
    directions: ArrayLike,
    /,
    *,
    parallel_tolerance: float = 1e-10,
    intersection_tolerance: float = 1e-9,
) -> RefractionResult:
    """Trace rays through all active interfaces using vector Snell refraction."""

    if not isinstance(stack, RefractiveLayerStack):
        raise TypeError("stack must be a RefractiveLayerStack.")
    if not math.isfinite(parallel_tolerance) or parallel_tolerance <= 0.0:
        raise ValueError("parallel_tolerance must be finite and positive.")
    if not math.isfinite(intersection_tolerance) or intersection_tolerance < 0.0:
        raise ValueError("intersection_tolerance must be finite and non-negative.")
    origins_ = jnp.asarray(origins)
    directions_ = jnp.asarray(directions)
    if origins_.shape != directions_.shape or origins_.shape[-1:] != (3,):
        raise ValueError("origins and directions must have the same shape (..., 3).")
    if jnp.issubdtype(origins_.dtype, jnp.complexfloating) or jnp.issubdtype(
        directions_.dtype,
        jnp.complexfloating,
    ):
        raise TypeError("Ray origins and directions must be real-valued.")
    values = _trace_refracted_arrays(
        stack,
        origins_,
        directions_,
        parallel_tolerance=float(parallel_tolerance),
        intersection_tolerance=float(intersection_tolerance),
    )
    return RefractionResult(*values)


__all__ = [
    "RefractiveLayerStack",
    "RefractionResult",
    "RefractionStatus",
    "trace_refracted_rays",
]
