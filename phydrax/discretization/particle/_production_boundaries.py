#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._free_surface import FreeSurfaceState
from ._wall import PreparedWallParticles


class BoundaryFeatureKind(StrEnum):
    SMOOTH_FACE = "smooth-face"
    CONVEX_EDGE = "convex-edge"
    CONCAVE_EDGE = "concave-edge"
    CORNER = "corner"
    THIN_GAP = "thin-gap"
    NON_MANIFOLD = "non-manifold"


class BoundaryFeatureState(StrictModule):
    kind_code: Array
    normal_variation: Array
    minimum_separation: Array
    ambiguous: Array


def classify_boundary_features(
    wall: PreparedWallParticles,
    /,
    *,
    edge_angle: float = 0.35,
    corner_angle: float = 0.8,
) -> BoundaryFeatureState:
    displacement = wall.positions[:, None, :] - wall.positions[None, :, :]
    distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
    neighbor = (distance > 0.0) & (distance < 2.5 * wall.quality.minimum_spacing)
    normal_dot = jnp.sum(wall.normals[:, None, :] * wall.normals[None, :, :], axis=-1)
    angle = jnp.arccos(jnp.clip(normal_dot, -1.0, 1.0))
    variation = jnp.max(jnp.where(neighbor, angle, 0.0), axis=1)
    minimum = jnp.min(jnp.where(distance > 0.0, distance, jnp.inf), axis=1)
    kind = jnp.where(
        variation >= corner_angle,
        3,
        jnp.where(variation >= edge_angle, 1, 0),
    ).astype(jnp.int32)
    ambiguous = ~jnp.isfinite(minimum) | (minimum < 0.25 * wall.quality.minimum_spacing)
    return BoundaryFeatureState(kind, variation, minimum, ambiguous)


class WallRelaxationPlan(StrictModule, NonTrainableState):
    iterations: int = eqx.field(static=True)
    step_fraction: float = eqx.field(static=True)
    target_spacing: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        target_spacing: float,
        /,
        *,
        iterations: int = 20,
        step_fraction: float = 0.1,
    ):
        if target_spacing <= 0.0 or iterations <= 0 or not 0.0 < step_fraction <= 0.5:
            raise ValueError("Wall relaxation parameters are invalid.")
        self.iterations = int(iterations)
        self.step_fraction = float(step_fraction)
        self.target_spacing = float(target_spacing)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "wall-relaxation-plan",
                "iterations": iterations,
                "step_fraction": step_fraction,
                "target_spacing": target_spacing,
            }
        )

    def relax(self, geometry, positions: ArrayLike, /) -> Array:
        initial = jnp.asarray(positions)

        def body(_, current):
            displacement = current[:, None, :] - current[None, :, :]
            distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
            mask = (distance > 0.0) & (distance < 1.5 * self.target_spacing)
            direction = displacement / jnp.where(distance > 0.0, distance, 1.0)[..., None]
            strength = jnp.where(
                mask,
                (self.target_spacing - distance) / self.target_spacing,
                0.0,
            )
            update = (
                self.step_fraction
                * self.target_spacing
                * jnp.sum(strength[..., None] * direction, axis=1)
            )
            normal = geometry.boundary_normal(current)
            tangential = update - jnp.sum(update * normal, axis=-1)[:, None] * normal
            candidate = current + tangential
            field = geometry.signed_distance(candidate)
            return candidate - field[:, None] * geometry.boundary_normal(candidate)

        return jax.lax.fori_loop(0, self.iterations, body, initial)


class WallMomentCertification(StrictModule):
    zeroth_moment_error: Array
    first_moment_error: Array
    volume_coefficient_of_variation: Array
    normal_defect: Array
    successful: Array


def certify_wall_moments(
    wall: PreparedWallParticles,
    kernel,
    smoothing_length: float,
    /,
    *,
    zeroth_tolerance: float = 0.2,
    first_tolerance: float = 0.2,
) -> WallMomentCertification:
    displacement = wall.positions[:, None, :] - wall.positions[None, :, :]
    distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
    weight = kernel.value(distance, smoothing_length)
    zeroth = jnp.sum(wall.volumes[None, :] * weight, axis=1)
    first = jnp.sum(
        wall.volumes[None, :, None] * displacement * weight[..., None], axis=1
    )
    zeroth_error = jnp.max(jnp.abs(zeroth - 1.0))
    first_error = jnp.max(jnp.sqrt(jnp.sum(first * first, axis=-1))) / smoothing_length
    volume_cv = jnp.std(wall.volumes) / jnp.maximum(jnp.mean(wall.volumes), 1e-14)
    normal_norm = jnp.sqrt(jnp.sum(wall.normals * wall.normals, axis=-1))
    normal_defect = jnp.max(jnp.abs(normal_norm - 1.0))
    successful = (
        (zeroth_error <= zeroth_tolerance)
        & (first_error <= first_tolerance)
        & jnp.all(jnp.isfinite(wall.volumes))
    )
    return WallMomentCertification(
        zeroth_error, first_error, volume_cv, normal_defect, successful
    )


class FreeSurfaceReconstructionPlan(StrictModule, NonTrainableState):
    maximum_fit_residual: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, maximum_fit_residual: float = 0.25, /):
        if maximum_fit_residual <= 0.0:
            raise ValueError("maximum_fit_residual must be positive.")
        self.maximum_fit_residual = float(maximum_fit_residual)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "free-surface-reconstruction",
                "maximum_fit_residual": maximum_fit_residual,
            }
        )


class FreeSurfaceGeometryState(StrictModule):
    surface_point: Array
    normal: Array
    curvature: Array
    signed_distance: Array
    kernel_volume_fraction: Array
    fit_residual: Array
    confidence: Array
    successful: Array


def reconstruct_free_surface(
    plan: FreeSurfaceReconstructionPlan,
    position: ArrayLike,
    surface: FreeSurfaceState,
    smoothing_length: float,
    /,
) -> FreeSurfaceGeometryState:
    position_ = jnp.asarray(position)
    normal = surface.normal
    signed_distance = smoothing_length * (surface.completeness - 0.5)
    point = position_ - signed_distance[:, None] * normal
    curvature = jnp.zeros((position_.shape[0],), dtype=position_.dtype)
    residual = jnp.abs(surface.completeness - 0.5)
    confidence = surface.smooth_weight * jnp.exp(-residual)
    successful = surface.hard_mask & (residual <= plan.maximum_fit_residual)
    return FreeSurfaceGeometryState(
        point,
        normal,
        curvature,
        signed_distance,
        jnp.clip(surface.completeness, 0.0, 1.0),
        residual,
        confidence,
        successful,
    )


class TruncatedKernelMomentState(StrictModule):
    zeroth_fraction: Array
    first_normal_moment: Array
    correction: Array
    successful: Array


def truncated_kernel_moments(
    geometry: FreeSurfaceGeometryState,
    /,
    *,
    minimum_fraction: float = 0.1,
) -> TruncatedKernelMomentState:
    fraction = jnp.clip(geometry.kernel_volume_fraction, minimum_fraction, 1.0)
    first = (1.0 - fraction)[:, None] * geometry.normal
    correction = 1.0 / fraction
    return TruncatedKernelMomentState(
        fraction, first, correction, geometry.successful & jnp.isfinite(correction)
    )


class ContactAnglePlan(StrictModule, NonTrainableState):
    angle: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, angle: float, /):
        angle_ = float(angle)
        if not 0.0 < angle_ < np.pi:
            raise ValueError("Contact angle must be in (0, pi).")
        self.angle = angle_
        self.plan_id = canonical_fingerprint(
            {"kind": "contact-angle-plan", "angle": angle_}
        )

    def apply(self, interface_normal: ArrayLike, wall_normal: ArrayLike, /) -> Array:
        interface = jnp.asarray(interface_normal)
        wall = jnp.asarray(wall_normal)
        tangent = interface - jnp.sum(interface * wall, axis=-1)[:, None] * wall
        tangent = (
            tangent
            / jnp.where(
                jnp.sqrt(jnp.sum(tangent * tangent, axis=-1)) > 0.0,
                jnp.sqrt(jnp.sum(tangent * tangent, axis=-1)),
                1.0,
            )[:, None]
        )
        return jnp.cos(self.angle) * wall + jnp.sin(self.angle) * tangent


__all__ = [
    "BoundaryFeatureKind",
    "BoundaryFeatureState",
    "ContactAnglePlan",
    "FreeSurfaceGeometryState",
    "FreeSurfaceReconstructionPlan",
    "TruncatedKernelMomentState",
    "WallMomentCertification",
    "WallRelaxationPlan",
    "certify_wall_moments",
    "classify_boundary_features",
    "reconstruct_free_surface",
    "truncated_kernel_moments",
]
