#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry._ray_intersection import (
    intersect_ray_plane,
    RayIntersectionStatus,
)
from ._interface import (
    evaluate_refractive_interface,
    OpticalRayState,
    RefractiveInterfaceStatus,
)
from ._sequential import SequentialOpticsResult, SequentialOpticsStatus


class PlanarRefractiveStack(StrictModule, NonTrainableState):
    """Fixed-capacity ordered planar interfaces between isotropic media.

    Normals point from medium ``i`` toward medium ``i + 1``. Active interfaces
    form a prefix and are traversed in array order. Capacity zero is a valid
    identity stack.
    """

    interface_points: Array
    interface_normals: Array
    refractive_indices: Array
    interface_active: Array
    capacity: int = eqx.field(static=True)
    active_count: int = eqx.field(static=True)
    stack_id: str = eqx.field(static=True)

    def __init__(
        self,
        interface_points: ArrayLike,
        interface_normals: ArrayLike,
        refractive_indices: ArrayLike,
        *,
        interface_active: ArrayLike | None = None,
    ):
        points_host = np.asarray(interface_points, dtype=float)
        normals_host = np.asarray(interface_normals, dtype=float)
        indices_host = np.asarray(refractive_indices, dtype=float)
        if points_host.ndim != 2 or points_host.shape[1:] != (3,):
            raise ValueError("interface_points must have shape (capacity, 3).")
        capacity = int(points_host.shape[0])
        if normals_host.shape != (capacity, 3):
            raise ValueError("interface_normals must have shape (capacity, 3).")
        if indices_host.shape != (capacity + 1,):
            raise ValueError("refractive_indices must have shape (capacity + 1,).")
        if not (
            np.all(np.isfinite(points_host))
            and np.all(np.isfinite(normals_host))
            and np.all(np.isfinite(indices_host))
        ):
            raise ValueError("Planar refractive stack data must be finite.")
        normal_norms = np.sqrt(np.sum(normals_host * normals_host, axis=-1))
        if capacity and np.any(normal_norms <= 0.0):
            raise ValueError("Interface normals must have nonzero length.")
        if np.any(indices_host <= 0.0):
            raise ValueError("Refractive indices must be positive.")
        if interface_active is None:
            active_host = np.ones((capacity,), dtype=bool)
        else:
            active_host = np.asarray(interface_active, dtype=bool)
            if active_host.shape != (capacity,):
                raise ValueError("interface_active must have shape (capacity,).")
        active_count = int(np.sum(active_host))
        if not np.array_equal(active_host, np.arange(capacity, dtype=int) < active_count):
            raise ValueError("Active interfaces must form a prefix of the stack.")
        normalized = normals_host / normal_norms[:, None] if capacity else normals_host
        self.interface_points = jnp.asarray(points_host)
        self.interface_normals = jnp.asarray(normalized)
        self.refractive_indices = jnp.asarray(indices_host)
        self.interface_active = jnp.asarray(active_host)
        self.capacity = capacity
        self.active_count = active_count
        self.stack_id = canonical_fingerprint(
            {
                "kind": "planar-refractive-stack",
                "capacity": capacity,
                "active_count": active_count,
                "data": array_tree_fingerprint(
                    (points_host, normalized, indices_host, active_host)
                ),
            }
        )


def _intersection_status(status: Array, /) -> Array:
    mapped = jnp.full(
        jnp.shape(status),
        int(SequentialOpticsStatus.NUMERICAL_FAILURE),
        dtype=jnp.int32,
    )
    mappings = (
        (RayIntersectionStatus.SUCCESS, SequentialOpticsStatus.SUCCESS),
        (
            RayIntersectionStatus.NONFINITE_INPUT,
            SequentialOpticsStatus.NONFINITE_INPUT,
        ),
        (
            RayIntersectionStatus.DEGENERATE_DIRECTION,
            SequentialOpticsStatus.INVALID_DIRECTION,
        ),
        (
            RayIntersectionStatus.DEGENERATE_NORMAL,
            SequentialOpticsStatus.INVALID_NORMAL,
        ),
        (RayIntersectionStatus.COPLANAR, SequentialOpticsStatus.COPLANAR),
        (RayIntersectionStatus.PARALLEL, SequentialOpticsStatus.PARALLEL),
        (RayIntersectionStatus.BEHIND_RAY, SequentialOpticsStatus.BEHIND_RAY),
    )
    for source, target in mappings:
        mapped = jnp.where(status == int(source), int(target), mapped)
    return mapped


def _interface_status(status: Array, /) -> Array:
    mapped = jnp.full(
        jnp.shape(status),
        int(SequentialOpticsStatus.NUMERICAL_FAILURE),
        dtype=jnp.int32,
    )
    mappings = (
        (RefractiveInterfaceStatus.SUCCESS, SequentialOpticsStatus.SUCCESS),
        (
            RefractiveInterfaceStatus.NONFINITE_INPUT,
            SequentialOpticsStatus.NONFINITE_INPUT,
        ),
        (
            RefractiveInterfaceStatus.INVALID_DIRECTION,
            SequentialOpticsStatus.INVALID_DIRECTION,
        ),
        (
            RefractiveInterfaceStatus.INVALID_NORMAL,
            SequentialOpticsStatus.INVALID_NORMAL,
        ),
        (
            RefractiveInterfaceStatus.WRONG_SIDE_INCIDENCE,
            SequentialOpticsStatus.WRONG_SIDE_INCIDENCE,
        ),
        (
            RefractiveInterfaceStatus.GRAZING_INCIDENCE,
            SequentialOpticsStatus.PARALLEL,
        ),
        (
            RefractiveInterfaceStatus.TOTAL_INTERNAL_REFLECTION,
            SequentialOpticsStatus.TOTAL_INTERNAL_REFLECTION,
        ),
        (
            RefractiveInterfaceStatus.NUMERICAL_FAILURE,
            SequentialOpticsStatus.NUMERICAL_FAILURE,
        ),
    )
    for source, target in mappings:
        mapped = jnp.where(status == int(source), int(target), mapped)
    return mapped


def trace_planar_refractive_stack(
    stack: PlanarRefractiveStack,
    origins: ArrayLike,
    directions: ArrayLike,
    /,
    *,
    parallel_tolerance: float = 1e-10,
    forward_tolerance: float = 1e-9,
    incidence_tolerance: float = 1e-10,
) -> SequentialOpticsResult:
    """Trace the transmitted-only route through a planar refractive stack."""

    if not isinstance(stack, PlanarRefractiveStack):
        raise TypeError("stack must be a PlanarRefractiveStack.")
    if not math.isfinite(parallel_tolerance) or parallel_tolerance <= 0.0:
        raise ValueError("parallel_tolerance must be finite and positive.")
    if not math.isfinite(forward_tolerance) or forward_tolerance < 0.0:
        raise ValueError("forward_tolerance must be finite and non-negative.")
    if not math.isfinite(incidence_tolerance) or incidence_tolerance < 0.0:
        raise ValueError("incidence_tolerance must be finite and non-negative.")
    origins_ = jnp.asarray(origins)
    directions_ = jnp.asarray(directions)
    if origins_.shape != directions_.shape or origins_.shape[-1:] != (3,):
        raise ValueError("origins and directions must have the same shape B + (3,).")
    if jnp.issubdtype(origins_.dtype, jnp.complexfloating) or jnp.issubdtype(
        directions_.dtype, jnp.complexfloating
    ):
        raise TypeError("Ray origins and directions must be real-valued.")

    dtype = jnp.result_type(origins_, directions_, stack.interface_points, 0.0)
    origin = origins_.astype(dtype)
    raw_direction = directions_.astype(dtype)
    finite_input = jnp.all(jnp.isfinite(origin), axis=-1) & jnp.all(
        jnp.isfinite(raw_direction), axis=-1
    )
    safe_direction = jnp.where(finite_input[..., None], raw_direction, 0.0)
    direction_norm = jnp.sqrt(jnp.sum(safe_direction * safe_direction, axis=-1))
    direction_ok = direction_norm > 0.0
    direction = safe_direction / jnp.where(direction_ok, direction_norm, 1.0)[..., None]
    valid = finite_input & direction_ok
    status = jnp.where(
        finite_input,
        jnp.where(
            direction_ok,
            int(SequentialOpticsStatus.SUCCESS),
            int(SequentialOpticsStatus.INVALID_DIRECTION),
        ),
        int(SequentialOpticsStatus.NONFINITE_INPUT),
    ).astype(jnp.int32)
    refractive_index = jnp.broadcast_to(
        stack.refractive_indices[0].astype(dtype), origin.shape[:-1]
    )
    geometric_path = jnp.zeros(origin.shape[:-1], dtype=dtype)
    optical_path = jnp.zeros(origin.shape[:-1], dtype=dtype)
    traversed = jnp.zeros(origin.shape[:-1], dtype=jnp.int32)
    minimum_discriminant = jnp.full(origin.shape[:-1], jnp.inf, dtype=dtype)

    for index in range(stack.capacity):
        active = valid & stack.interface_active[index]
        hit = intersect_ray_plane(
            origin,
            direction,
            stack.interface_points[index].astype(dtype),
            stack.interface_normals[index].astype(dtype),
            parallel_tolerance=parallel_tolerance,
            forward_tolerance=forward_tolerance,
        )
        hit_success = active & hit.valid
        interface = evaluate_refractive_interface(
            direction,
            stack.interface_normals[index].astype(dtype),
            stack.refractive_indices[index].astype(dtype),
            stack.refractive_indices[index + 1].astype(dtype),
            incidence_tolerance=incidence_tolerance,
        )
        step_success = hit_success & interface.transmission_valid
        hit_failure = active & ~hit.valid
        interface_failure = hit_success & ~interface.transmission_valid
        step_failure = hit_failure | interface_failure
        step_status = jnp.where(
            hit_failure,
            _intersection_status(hit.status),
            _interface_status(interface.status),
        )
        status = jnp.where(step_failure, step_status, status)

        segment_length = jnp.maximum(hit.distances, 0.0)
        geometric_path = geometric_path + jnp.where(step_success, segment_length, 0.0)
        optical_path = optical_path + jnp.where(
            step_success,
            segment_length * stack.refractive_indices[index].astype(dtype),
            0.0,
        )
        origin = jnp.where(step_success[..., None], hit.points, origin)
        direction = jnp.where(
            step_success[..., None], interface.transmitted_directions, direction
        )
        refractive_index = jnp.where(
            step_success,
            stack.refractive_indices[index + 1].astype(dtype),
            refractive_index,
        )
        traversed = traversed + step_success.astype(jnp.int32)
        minimum_discriminant = jnp.where(
            hit_success,
            jnp.minimum(minimum_discriminant, interface.snell_discriminant),
            minimum_discriminant,
        )
        valid = valid & ~step_failure

    finite = finite_input & (status != int(SequentialOpticsStatus.NUMERICAL_FAILURE))
    successful = valid & (status == int(SequentialOpticsStatus.SUCCESS))
    rays = OpticalRayState(
        origin,
        direction,
        refractive_index,
        geometric_path,
        optical_path,
    )
    return SequentialOpticsResult(
        rays=rays,
        valid=valid,
        status=status,
        traversed_surfaces=traversed,
        minimum_snell_discriminant=minimum_discriminant,
        minimum_aperture_margin=jnp.full(origin.shape[:-1], jnp.inf, dtype=dtype),
        maximum_intersection_residual=jnp.zeros(origin.shape[:-1], dtype=dtype),
        finite=finite,
        successful=successful,
        producer_id=stack.stack_id,
    )


__all__ = ["PlanarRefractiveStack", "trace_planar_refractive_stack"]
