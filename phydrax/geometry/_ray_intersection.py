#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from .._trainable import NonTrainableState


class RayIntersectionStatus(IntEnum):
    """Status of a normalized forward ray-plane intersection."""

    SUCCESS = 0
    NONFINITE_INPUT = 1
    DEGENERATE_DIRECTION = 2
    DEGENERATE_NORMAL = 3
    COPLANAR = 4
    PARALLEL = 5
    BEHIND_RAY = 6


class RayIntersectionResult(StrictModule, NonTrainableState):
    """Fixed-shape ray-plane hit facts.

    ``distances`` are physical signed distances along normalized input rays.
    Numerical values in invalid lanes are safe placeholders and must be
    interpreted through ``valid`` and ``status``.
    """

    points: Array
    distances: Array
    parallel_margin: Array
    valid: Array
    status: Array


def intersect_ray_plane(
    origins: ArrayLike,
    directions: ArrayLike,
    plane_point: ArrayLike,
    plane_normal: ArrayLike,
    /,
    *,
    parallel_tolerance: float = 1e-10,
    forward_tolerance: float = 1e-9,
) -> RayIntersectionResult:
    """Intersect normalized rays with one plane or a matching batch of planes.

    ``origins`` and ``directions`` have the same shape ``B + (3,)``. Plane
    vectors may have shape ``(3,)`` or exactly ``B + (3,)``. Plane-normal sign
    does not affect geometric hit validity.
    """

    if not math.isfinite(parallel_tolerance) or parallel_tolerance <= 0.0:
        raise ValueError("parallel_tolerance must be finite and positive.")
    if not math.isfinite(forward_tolerance) or forward_tolerance < 0.0:
        raise ValueError("forward_tolerance must be finite and non-negative.")

    origins_ = jnp.asarray(origins)
    directions_ = jnp.asarray(directions)
    point_ = jnp.asarray(plane_point)
    normal_ = jnp.asarray(plane_normal)
    if origins_.shape != directions_.shape or origins_.shape[-1:] != (3,):
        raise ValueError("origins and directions must have the same shape B + (3,).")
    batch_shape = origins_.shape[:-1]
    expected_shape = batch_shape + (3,)
    if point_.shape not in ((3,), expected_shape):
        raise ValueError("plane_point must have shape (3,) or B + (3,).")
    if normal_.shape not in ((3,), expected_shape):
        raise ValueError("plane_normal must have shape (3,) or B + (3,).")
    if any(
        jnp.issubdtype(value.dtype, jnp.complexfloating)
        for value in (origins_, directions_, point_, normal_)
    ):
        raise TypeError("Ray and plane coordinates must be real-valued.")

    dtype = jnp.result_type(origins_, directions_, point_, normal_, 0.0)
    origins_ = origins_.astype(dtype)
    directions_ = directions_.astype(dtype)
    point_ = jnp.broadcast_to(point_.astype(dtype), expected_shape)
    normal_ = jnp.broadcast_to(normal_.astype(dtype), expected_shape)

    finite = (
        jnp.all(jnp.isfinite(origins_), axis=-1)
        & jnp.all(jnp.isfinite(directions_), axis=-1)
        & jnp.all(jnp.isfinite(point_), axis=-1)
        & jnp.all(jnp.isfinite(normal_), axis=-1)
    )
    safe_directions = jnp.where(finite[..., None], directions_, 0.0)
    safe_normals = jnp.where(finite[..., None], normal_, 0.0)
    direction_norm = jnp.sqrt(jnp.sum(safe_directions * safe_directions, axis=-1))
    normal_norm = jnp.sqrt(jnp.sum(safe_normals * safe_normals, axis=-1))
    direction_ok = direction_norm > 0.0
    normal_ok = normal_norm > 0.0
    unit_direction = (
        safe_directions / jnp.where(direction_ok, direction_norm, 1.0)[..., None]
    )
    unit_normal = safe_normals / jnp.where(normal_ok, normal_norm, 1.0)[..., None]

    denominator = jnp.sum(unit_direction * unit_normal, axis=-1)
    signed_offset = jnp.sum((point_ - origins_) * unit_normal, axis=-1)
    parallel = jnp.abs(denominator) <= parallel_tolerance
    coplanar = parallel & (jnp.abs(signed_offset) <= forward_tolerance)
    safe_denominator = jnp.where(parallel, 1.0, denominator)
    candidate_distance = signed_offset / safe_denominator
    behind = candidate_distance < -forward_tolerance

    status = jnp.where(
        ~finite,
        int(RayIntersectionStatus.NONFINITE_INPUT),
        jnp.where(
            ~direction_ok,
            int(RayIntersectionStatus.DEGENERATE_DIRECTION),
            jnp.where(
                ~normal_ok,
                int(RayIntersectionStatus.DEGENERATE_NORMAL),
                jnp.where(
                    coplanar,
                    int(RayIntersectionStatus.COPLANAR),
                    jnp.where(
                        parallel,
                        int(RayIntersectionStatus.PARALLEL),
                        jnp.where(
                            behind,
                            int(RayIntersectionStatus.BEHIND_RAY),
                            int(RayIntersectionStatus.SUCCESS),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    valid = status == int(RayIntersectionStatus.SUCCESS)
    distances = jnp.where(valid, candidate_distance, 0.0)
    candidate_point = origins_ + candidate_distance[..., None] * unit_direction
    points = jnp.where(valid[..., None], candidate_point, origins_)
    parallel_margin = jnp.where(
        finite & direction_ok & normal_ok,
        jnp.abs(denominator) - parallel_tolerance,
        0.0,
    )
    return RayIntersectionResult(points, distances, parallel_margin, valid, status)


__all__ = ["RayIntersectionResult", "RayIntersectionStatus", "intersect_ray_plane"]
