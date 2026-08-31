#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import ParticleDiscretization
from ._rigid_body import quaternion_rotation_matrix, RigidBodySetPlan


class SuperquadricSetPlan(StrictModule, NonTrainableState):
    semi_axes: Array
    first_blockiness: Array
    second_blockiness: Array
    material_ids: Array
    fixed_mask: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        semi_axes: ArrayLike,
        first_blockiness: ArrayLike,
        second_blockiness: ArrayLike,
        material_ids: ArrayLike,
        /,
        *,
        fixed_mask: ArrayLike | None = None,
        plan_id: str | None = None,
    ):
        axes = np.asarray(semi_axes, dtype=float)
        first = np.asarray(first_blockiness, dtype=float)
        second = np.asarray(second_blockiness, dtype=float)
        material = np.asarray(material_ids)
        if axes.ndim != 2 or axes.shape[1] != 3 or axes.shape[0] == 0:
            raise ValueError("semi_axes must have nonempty shape (body,3).")
        count = axes.shape[0]
        if (
            first.shape != (count,)
            or second.shape != (count,)
            or material.shape != (count,)
        ):
            raise ValueError("Superquadric body properties must share body capacity.")
        if (
            np.any(~np.isfinite(axes))
            or np.any(axes <= 0.0)
            or np.any(~np.isfinite(first))
            or np.any(~np.isfinite(second))
            or np.any(first < 2.0)
            or np.any(second < 2.0)
            or not np.issubdtype(material.dtype, np.integer)
            or np.any(material < 0)
        ):
            raise ValueError("Superquadric shape/material parameters are invalid.")
        fixed = (
            np.zeros((count,), dtype=bool)
            if fixed_mask is None
            else np.asarray(fixed_mask, dtype=bool)
        )
        if fixed.shape != (count,):
            raise ValueError("fixed_mask must have body-capacity shape.")
        generated = canonical_fingerprint(
            {
                "kind": "superquadric-set-plan",
                "values": array_tree_fingerprint(
                    {
                        "semi_axes": axes,
                        "first_blockiness": first,
                        "second_blockiness": second,
                        "material_ids": material,
                        "fixed_mask": fixed,
                    }
                ),
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be nonempty.")
        self.semi_axes = jnp.asarray(axes)
        self.first_blockiness = jnp.asarray(first)
        self.second_blockiness = jnp.asarray(second)
        self.material_ids = jnp.asarray(material, dtype=jnp.int32)
        self.fixed_mask = jnp.asarray(fixed)
        self.plan_id = identifier

    def prepare(self, particles: ParticleDiscretization, /):
        return PreparedSuperquadricSet(self, particles)

    def rigid_body_plan(self, particles: ParticleDiscretization, /) -> RigidBodySetPlan:
        if (
            particles.ambient_dimension != 3
            or particles.capacity != self.semi_axes.shape[0]
        ):
            raise ValueError(
                "Superquadrics require a matching three-dimensional population."
            )
        inertia = _superquadric_inertia(
            np.asarray(particles.safe_masses),
            np.asarray(self.semi_axes),
            np.asarray(self.first_blockiness),
            np.asarray(self.second_blockiness),
        )
        return RigidBodySetPlan(
            self.material_ids,
            inertia,
            fixed_mask=self.fixed_mask,
            name="superquadric-rigid-bodies",
        )


class PreparedSuperquadricSet(StrictModule, NonTrainableState):
    plan: SuperquadricSetPlan
    particles: ParticleDiscretization
    semi_axes: Array
    first_blockiness: Array
    second_blockiness: Array
    bounding_radii: Array
    material_ids: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: SuperquadricSetPlan, particles: ParticleDiscretization, /):
        if not isinstance(plan, SuperquadricSetPlan):
            raise TypeError("plan must be a SuperquadricSetPlan.")
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be a ParticleDiscretization.")
        if particles.ambient_dimension != 3 or plan.semi_axes.shape != (
            particles.capacity,
            3,
        ):
            raise ValueError("Superquadric plan requires matching 3D particle capacity.")
        active = particles.active_mask
        self.plan = plan
        self.particles = particles
        self.semi_axes = jnp.where(active[:, None], plan.semi_axes, 1.0)
        self.first_blockiness = jnp.where(active, plan.first_blockiness, 2.0)
        self.second_blockiness = jnp.where(active, plan.second_blockiness, 2.0)
        self.bounding_radii = jnp.where(
            active, jnp.linalg.norm(plan.semi_axes, axis=-1), 0.0
        )
        self.material_ids = jnp.where(active, plan.material_ids, 0)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-superquadric-set",
                "plan": plan.plan_id,
                "particles": particles.prepared_id,
            }
        )


class SuperquadricContactPlan(StrictModule, NonTrainableState):
    iterations: int = eqx.field(static=True)
    relaxation: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        iterations: int = 24,
        relaxation: float = 0.5,
        residual_tolerance: float = 1.0e-6,
        plan_id: str | None = None,
    ):
        count = int(iterations)
        relax = float(relaxation)
        tolerance = float(residual_tolerance)
        if (
            count <= 0
            or not 0.0 < relax <= 1.0
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError("Superquadric contact iteration controls are invalid.")
        generated = canonical_fingerprint(
            {
                "kind": "superquadric-contact-plan",
                "iterations": count,
                "relaxation": relax,
                "residual_tolerance": tolerance,
            }
        )
        self.iterations = count
        self.relaxation = relax
        self.residual_tolerance = tolerance
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")


class SuperquadricContactResult(StrictModule):
    normal: Array
    gap: Array
    contact_point: Array
    left_arm: Array
    right_arm: Array
    effective_radius: Array
    left_principal_curvature: Array
    right_principal_curvature: Array
    residual: Array
    regularity_margin: Array
    valid: Array
    plan_id: str = eqx.field(static=True)


def superquadric_pair_contact(
    plan: SuperquadricContactPlan,
    shapes: PreparedSuperquadricSet,
    position: Array,
    orientation: Array,
    left_indices: Array,
    right_indices: Array,
    /,
) -> SuperquadricContactResult:
    if not isinstance(plan, SuperquadricContactPlan):
        raise TypeError("plan must be a SuperquadricContactPlan.")
    if not isinstance(shapes, PreparedSuperquadricSet):
        raise TypeError("shapes must be a PreparedSuperquadricSet.")
    positions = jnp.asarray(position)
    orientations = jnp.asarray(orientation)
    left = jnp.asarray(left_indices, dtype=jnp.int32)
    right = jnp.asarray(right_indices, dtype=jnp.int32)
    if positions.shape != (shapes.particles.capacity, 3) or orientations.shape != (
        shapes.particles.capacity,
        4,
    ):
        raise ValueError("Superquadric pose arrays do not match body capacity.")
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Superquadric pair indices must be equal rank-1 arrays.")
    left_position = positions[left]
    right_position = positions[right]
    left_rotation = quaternion_rotation_matrix(orientations[left])
    right_rotation = quaternion_rotation_matrix(orientations[right])
    left_axes = shapes.semi_axes[left]
    right_axes = shapes.semi_axes[right]
    left_first = shapes.first_blockiness[left]
    right_first = shapes.first_blockiness[right]
    left_second = shapes.second_blockiness[left]
    right_second = shapes.second_blockiness[right]
    displacement = right_position - left_position
    distance = jnp.linalg.norm(displacement, axis=-1)
    fallback = jnp.broadcast_to(jnp.asarray([1.0, 0.0, 0.0]), displacement.shape)
    direction = jnp.where(
        (distance > 0.0)[:, None],
        displacement / jnp.maximum(distance[:, None], 1.0e-30),
        fallback,
    )

    def support(rotation, axes, first, second, world_direction):
        local_direction = contract("...ji,...j->...i", rotation, world_direction)
        local = jax.vmap(_support_local)(local_direction, axes, first, second)
        return contract("...ij,...j->...i", rotation, local)

    def iteration(_, axis):
        left_support = left_position + support(
            left_rotation, left_axes, left_first, left_second, axis
        )
        right_support = right_position + support(
            right_rotation, right_axes, right_first, right_second, -axis
        )
        separation = right_support - left_support
        axial = jnp.sum(separation * axis, axis=-1, keepdims=True)
        tangent = separation - axial * axis
        scale = jnp.maximum(
            jnp.linalg.norm(separation, axis=-1, keepdims=True),
            jnp.minimum(jnp.min(left_axes, axis=-1), jnp.min(right_axes, axis=-1))[
                :, None
            ],
        )
        candidate = axis + plan.relaxation * tangent / jnp.maximum(scale, 1.0e-30)
        return candidate / jnp.maximum(
            jnp.linalg.norm(candidate, axis=-1, keepdims=True), 1.0e-30
        )

    direction = jax.lax.fori_loop(0, plan.iterations, iteration, direction)
    left_support = left_position + support(
        left_rotation, left_axes, left_first, left_second, direction
    )
    right_support = right_position + support(
        right_rotation, right_axes, right_first, right_second, -direction
    )
    separation = right_support - left_support
    gap = jnp.sum(separation * direction, axis=-1)
    tangent = separation - gap[:, None] * direction
    scale = jnp.maximum(
        jnp.minimum(jnp.min(left_axes, axis=-1), jnp.min(right_axes, axis=-1)),
        1.0e-30,
    )
    residual = jnp.linalg.norm(tangent, axis=-1) / scale
    left_local = contract("...ji,...j->...i", left_rotation, left_support - left_position)
    right_local = contract(
        "...ji,...j->...i", right_rotation, right_support - right_position
    )
    left_curvature, left_valid, left_margin = jax.vmap(_principal_curvature)(
        left_local, left_axes, left_first, left_second
    )
    right_curvature, right_valid, right_margin = jax.vmap(_principal_curvature)(
        right_local, right_axes, right_first, right_second
    )
    curvature_sum = 0.5 * jnp.sum(left_curvature + right_curvature, axis=-1)
    effective_radius = jnp.where(curvature_sum > 0.0, 1.0 / curvature_sum, 0.0)
    active_shape = (
        shapes.particles.active_mask[left] & shapes.particles.active_mask[right]
    )
    finite = (
        jnp.all(jnp.isfinite(direction), axis=-1)
        & jnp.isfinite(gap)
        & jnp.isfinite(residual)
        & jnp.isfinite(effective_radius)
    )
    valid = (
        active_shape
        & (left != right)
        & (distance > 0.0)
        & (residual <= plan.residual_tolerance)
        & left_valid
        & right_valid
        & finite
    )
    normal = -direction
    contact_point = 0.5 * (left_support + right_support)
    margin = jnp.minimum(
        jnp.minimum(left_margin, right_margin),
        plan.residual_tolerance - residual,
    )
    return SuperquadricContactResult(
        normal,
        gap,
        contact_point,
        contact_point - left_position,
        contact_point - right_position,
        effective_radius,
        left_curvature,
        right_curvature,
        residual,
        margin,
        valid,
        plan.plan_id,
    )


def _support_local(direction, axes, first, second):
    def dual_norm(value):
        first_dual = first / (first - 1.0)
        second_dual = second / (second - 1.0)
        scaled = axes * value
        planar = (
            jnp.abs(scaled[0]) ** first_dual + jnp.abs(scaled[1]) ** first_dual
        ) ** (second_dual / first_dual)
        return (planar + jnp.abs(scaled[2]) ** second_dual) ** (1.0 / second_dual)

    return jax.grad(dual_norm)(direction)


def _principal_curvature(point, axes, first, second):
    scale = jnp.min(axes)

    def field(value):
        normalized = value / axes
        planar = (normalized[0] ** 2) ** (0.5 * first) + (normalized[1] ** 2) ** (
            0.5 * first
        )
        norm = (planar ** (second / first) + (normalized[2] ** 2) ** (0.5 * second)) ** (
            1.0 / second
        )
        return scale * (norm - 1.0)

    gradient = jax.grad(field)(point)
    step = jnp.sqrt(jnp.finfo(point.dtype).eps) * jnp.minimum(scale, 1.0)
    offsets = step * jnp.eye(3, dtype=point.dtype)
    plus = jax.vmap(jax.grad(field))(point[None, :] + offsets)
    minus = jax.vmap(jax.grad(field))(point[None, :] - offsets)
    hessian = ((plus - minus) / (2.0 * step)).T
    hessian = 0.5 * (hessian + hessian.T)
    gradient_norm = jnp.linalg.norm(gradient)
    normal = gradient / jnp.maximum(gradient_norm, 1.0e-30)
    projector = jnp.eye(3, dtype=point.dtype) - normal[:, None] * normal[None, :]
    shape = projector @ hessian @ projector / jnp.maximum(gradient_norm, 1.0e-30)
    eigenvalues = jnp.linalg.eigvalsh(shape)
    principal = eigenvalues[1:]
    valid = jnp.all(jnp.isfinite(principal)) & (gradient_norm > 1.0e-10)
    return principal, valid, jnp.minimum(gradient_norm, step)


def _superquadric_inertia(mass, axes, first, second):
    result = np.zeros((mass.shape[0], 3, 3), dtype=np.result_type(mass, axes))
    for index in range(mass.shape[0]):
        p = float(first[index])
        q = float(second[index])
        area = 4.0 * math.exp(
            2.0 * math.lgamma(1.0 + 1.0 / p) - math.lgamma(1.0 + 2.0 / p)
        )
        planar_second = (4.0 / (p * p)) * math.exp(
            math.lgamma(3.0 / p) + math.lgamma(1.0 / p) - math.lgamma(1.0 + 4.0 / p)
        )
        i0 = _beta_integral(q, 0.0, 2.0)
        i4 = _beta_integral(q, 0.0, 4.0)
        iz = _beta_integral(q, 2.0, 2.0)
        mean_x2 = axes[index, 0] ** 2 * planar_second * i4 / (area * i0)
        mean_y2 = axes[index, 1] ** 2 * planar_second * i4 / (area * i0)
        mean_z2 = axes[index, 2] ** 2 * iz / i0
        diagonal = mass[index] * np.asarray(
            (
                mean_y2 + mean_z2,
                mean_x2 + mean_z2,
                mean_x2 + mean_y2,
            )
        )
        result[index] = np.diag(diagonal)
    return result


def _beta_integral(exponent, moment_power, radial_power):
    return math.exp(
        math.lgamma((moment_power + 1.0) / exponent)
        + math.lgamma(1.0 + radial_power / exponent)
        - math.lgamma(1.0 + (moment_power + radial_power + 1.0) / exponent)
        - math.log(exponent)
    )


__all__ = [
    "PreparedSuperquadricSet",
    "SuperquadricContactPlan",
    "SuperquadricContactResult",
    "SuperquadricSetPlan",
    "superquadric_pair_contact",
]
