#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry import RigidFrame
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from ._refraction import (
    _trace_refracted_arrays,
    RefractionStatus,
    RefractiveLayerStack,
)


class ProjectionStatus(IntEnum):
    SUCCESS = 0
    NONFINITE_INPUT = 1
    BEHIND_CAMERA = 2
    OUTSIDE_IMAGE = 3
    REFRACTION_FAILED = 4
    REFRACTION_NONCONVERGENCE = 5


class RayStatus(IntEnum):
    SUCCESS = 0
    NONFINITE_INPUT = 1
    INVALID_DIRECTION = 2
    PARALLEL_INTERFACE = 3
    INTERFACE_BEHIND_RAY = 4
    TOTAL_INTERNAL_REFLECTION = 5
    DISTORTION_NONCONVERGENCE = 6
    OUTSIDE_IMAGE = 7


class CameraIntrinsics(StrictModule):
    """Pinhole intrinsics in explicit ``(row, column)`` component order."""

    focal_length: Array
    principal_point: Array
    skew: Array
    image_shape: tuple[int, int] | None = eqx.field(static=True)

    def __init__(
        self,
        focal_length: ArrayLike,
        principal_point: ArrayLike,
        *,
        image_shape: tuple[int, int] | None = None,
        skew: Any = 0.0,
    ):
        focal_host = np.asarray(focal_length, dtype=float)
        principal_host = np.asarray(principal_point, dtype=float)
        skew_host = np.asarray(skew, dtype=float)
        if focal_host.shape != (2,) or principal_host.shape != (2,):
            raise ValueError("focal_length and principal_point must have shape (2,).")
        if skew_host.shape != ():
            raise ValueError("skew must be scalar.")
        if not (
            np.all(np.isfinite(focal_host))
            and np.all(np.isfinite(principal_host))
            and np.isfinite(skew_host)
        ):
            raise ValueError("Camera intrinsics must be finite.")
        if np.any(focal_host <= 0.0):
            raise ValueError("Focal lengths must be positive.")
        if image_shape is None:
            resolved_shape = None
        else:
            resolved_shape = tuple(int(value) for value in image_shape)
            if len(resolved_shape) != 2 or any(value <= 0 for value in resolved_shape):
                raise ValueError("image_shape must contain two positive dimensions.")
        self.focal_length = jnp.asarray(focal_host)
        self.principal_point = jnp.asarray(principal_host)
        self.skew = jnp.asarray(skew_host)
        self.image_shape = resolved_shape


class BrownConradyDistortion(StrictModule):
    """Three radial and two tangential Brown--Conrady coefficients."""

    radial: Array
    tangential: Array

    def __init__(
        self,
        radial: ArrayLike = (0.0, 0.0, 0.0),
        tangential: ArrayLike = (0.0, 0.0),
    ):
        radial_host = np.asarray(radial, dtype=float)
        tangential_host = np.asarray(tangential, dtype=float)
        if radial_host.shape != (3,) or tangential_host.shape != (2,):
            raise ValueError("radial and tangential must have shapes (3,) and (2,).")
        if not np.all(np.isfinite(radial_host)) or not np.all(
            np.isfinite(tangential_host)
        ):
            raise ValueError("Distortion coefficients must be finite.")
        self.radial = jnp.asarray(radial_host)
        self.tangential = jnp.asarray(tangential_host)


class CameraPose(StrictModule):
    """Camera-to-world pose backed by the canonical geometry ``RigidFrame``."""

    frame: RigidFrame

    def __init__(self, frame: RigidFrame):
        if not isinstance(frame, RigidFrame) or frame.dimension != 3:
            raise TypeError("frame must be a three-dimensional RigidFrame.")
        self.frame = frame


class CameraModel(StrictModule):
    intrinsics: CameraIntrinsics
    pose: CameraPose
    distortion: BrownConradyDistortion
    refraction: RefractiveLayerStack | None

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        *,
        pose: CameraPose | None = None,
        distortion: BrownConradyDistortion | None = None,
        refraction: RefractiveLayerStack | None = None,
    ):
        if not isinstance(intrinsics, CameraIntrinsics):
            raise TypeError("intrinsics must be CameraIntrinsics.")
        pose_ = CameraPose(RigidFrame.identity(3)) if pose is None else pose
        distortion_ = BrownConradyDistortion() if distortion is None else distortion
        if not isinstance(pose_, CameraPose):
            raise TypeError("pose must be CameraPose or None.")
        if not isinstance(distortion_, BrownConradyDistortion):
            raise TypeError("distortion must be BrownConradyDistortion or None.")
        if refraction is not None and not isinstance(refraction, RefractiveLayerStack):
            raise TypeError("refraction must be RefractiveLayerStack or None.")
        self.intrinsics = intrinsics
        self.pose = pose_
        self.distortion = distortion_
        self.refraction = refraction


class ProjectionResult(StrictModule, NonTrainableState):
    pixels: Array
    depth: Array
    camera_points: Array
    valid: Array
    status: Array


class RayResult(StrictModule, NonTrainableState):
    origins: Array
    directions: Array
    valid: Array
    status: Array
    iterations: Array
    residual_norm: Array


def _distort_normalized(
    distortion: BrownConradyDistortion,
    normalized: Array,
) -> Array:
    x = normalized[..., 0]
    y = normalized[..., 1]
    radial = distortion.radial.astype(normalized.dtype)
    tangential = distortion.tangential.astype(normalized.dtype)
    radius_squared = x * x + y * y
    radial_scale = 1.0 + radius_squared * (
        radial[0] + radius_squared * (radial[1] + radius_squared * radial[2])
    )
    p1, p2 = tangential[0], tangential[1]
    distorted_x = (
        x * radial_scale + 2.0 * p1 * x * y + p2 * (radius_squared + 2.0 * x * x)
    )
    distorted_y = (
        y * radial_scale + p1 * (radius_squared + 2.0 * y * y) + 2.0 * p2 * x * y
    )
    return jnp.stack((distorted_x, distorted_y), axis=-1)


def _distortion_jacobian(
    distortion: BrownConradyDistortion,
    normalized: Array,
) -> Array:
    x = normalized[..., 0]
    y = normalized[..., 1]
    k1, k2, k3 = distortion.radial.astype(normalized.dtype)
    p1, p2 = distortion.tangential.astype(normalized.dtype)
    radius_squared = x * x + y * y
    radial_scale = 1.0 + radius_squared * (
        k1 + radius_squared * (k2 + radius_squared * k3)
    )
    radial_derivative_scale = (
        k1 + 2.0 * k2 * radius_squared + 3.0 * k3 * radius_squared**2
    )
    radial_x = 2.0 * x * radial_derivative_scale
    radial_y = 2.0 * y * radial_derivative_scale
    dxdx = radial_scale + x * radial_x + 2.0 * p1 * y + 6.0 * p2 * x
    dxdy = x * radial_y + 2.0 * p1 * x + 2.0 * p2 * y
    dydx = y * radial_x + 2.0 * p1 * x + 2.0 * p2 * y
    dydy = radial_scale + y * radial_y + 6.0 * p1 * y + 2.0 * p2 * x
    return jnp.stack(
        (
            jnp.stack((dxdx, dxdy), axis=-1),
            jnp.stack((dydx, dydy), axis=-1),
        ),
        axis=-2,
    )


def _normalized_to_pixels(intrinsics: CameraIntrinsics, normalized: Array) -> Array:
    x = normalized[..., 0]
    y = normalized[..., 1]
    focal = intrinsics.focal_length.astype(normalized.dtype)
    principal = intrinsics.principal_point.astype(normalized.dtype)
    row = focal[0] * y + principal[0]
    column = focal[1] * x + intrinsics.skew.astype(normalized.dtype) * y + principal[1]
    return jnp.stack((row, column), axis=-1)


def _pixels_to_distorted_normalized(intrinsics: CameraIntrinsics, pixels: Array) -> Array:
    focal = intrinsics.focal_length.astype(pixels.dtype)
    principal = intrinsics.principal_point.astype(pixels.dtype)
    distorted_y = (pixels[..., 0] - principal[0]) / focal[0]
    distorted_x = (
        pixels[..., 1] - principal[1] - intrinsics.skew.astype(pixels.dtype) * distorted_y
    ) / focal[1]
    return jnp.stack((distorted_x, distorted_y), axis=-1)


def _inside_image(intrinsics: CameraIntrinsics, pixels: Array) -> Array:
    finite = jnp.all(jnp.isfinite(pixels), axis=-1)
    if intrinsics.image_shape is None:
        return finite
    rows, columns = intrinsics.image_shape
    return (
        finite
        & (pixels[..., 0] >= 0.0)
        & (pixels[..., 0] <= rows - 1)
        & (pixels[..., 1] >= 0.0)
        & (pixels[..., 1] <= columns - 1)
    )


def _project_refracted_points(
    camera: CameraModel,
    points: Array,
    initial_normalized: Array,
    *,
    maximum_iterations: int,
    tolerance: float,
) -> tuple[Array, Array, Array]:
    stack = camera.refraction
    assert stack is not None
    origin = camera.pose.frame.translation.astype(points.dtype)
    rotation = camera.pose.frame.rotation.astype(points.dtype)
    solve_plan = SmallLinearSolvePlan(
        2,
        singular_tolerance=1e-12,
        maximum_condition=1e10,
        refinement_iterations=1,
    )

    def solve_one(point, initial):
        def path(candidate):
            direction_camera = jnp.stack(
                (candidate[0], candidate[1], jnp.ones((), dtype=candidate.dtype))
            )
            direction_camera = direction_camera / jnp.sqrt(
                jnp.sum(direction_camera * direction_camera)
            )
            direction_world = contract("i,ji->j", direction_camera, rotation)
            traced = _trace_refracted_arrays(
                stack,
                origin,
                direction_world,
                parallel_tolerance=1e-10,
                intersection_tolerance=1e-9,
            )
            final_origin, final_direction, path_valid = traced[:3]
            target_offset = point - final_origin
            path_valid = path_valid & (
                jnp.sum(final_direction * target_offset) >= -tolerance
            )
            residual = jnp.cross(final_direction, target_offset)
            return residual, path_valid

        candidate = initial
        linear_valid = jnp.asarray(True)
        for _ in range(maximum_iterations):
            residual, path_valid = path(candidate)
            jacobian = jax.jacfwd(lambda value: path(value)[0])(candidate)
            normal = contract("ki,kj->ij", jacobian, jacobian)
            right = contract("ki,k->i", jacobian, residual)
            linear = solve_small_linear(solve_plan, normal, right)
            residual_norm = jnp.sqrt(jnp.sum(residual * residual))
            converged = residual_norm <= tolerance * (
                1.0 + jnp.sqrt(jnp.sum((point - origin) ** 2))
            )
            update = path_valid & linear.successful & linear_valid & ~converged
            candidate = candidate - jnp.where(update, linear.value, 0.0)
            linear_valid = linear_valid & (linear.successful | converged)
        residual, path_valid = path(candidate)
        residual_norm = jnp.sqrt(jnp.sum(residual * residual))
        converged = residual_norm <= tolerance * (
            1.0 + jnp.sqrt(jnp.sum((point - origin) ** 2))
        )
        valid = path_valid & linear_valid & converged
        status = jnp.where(
            ~path_valid,
            int(ProjectionStatus.REFRACTION_FAILED),
            jnp.where(
                valid,
                int(ProjectionStatus.SUCCESS),
                int(ProjectionStatus.REFRACTION_NONCONVERGENCE),
            ),
        ).astype(jnp.int32)
        return candidate, valid, status

    flat_points = points.reshape((-1, 3))
    flat_initial = initial_normalized.reshape((-1, 2))
    normalized, valid, status = jax.vmap(solve_one)(flat_points, flat_initial)
    return (
        normalized.reshape(points.shape[:-1] + (2,)),
        valid.reshape(points.shape[:-1]),
        status.reshape(points.shape[:-1]),
    )


def project_points(
    camera: CameraModel,
    points: ArrayLike,
    *,
    minimum_depth: float = 1e-8,
    refraction_maximum_iterations: int = 12,
    refraction_tolerance: float = 1e-9,
) -> ProjectionResult:
    """Project right-handed world ``(x, y, z)`` points to ``(row, column)`` pixels."""

    if not isinstance(camera, CameraModel):
        raise TypeError("camera must be a CameraModel.")
    if not math.isfinite(minimum_depth) or minimum_depth <= 0.0:
        raise ValueError("minimum_depth must be finite and positive.")
    if refraction_maximum_iterations < 0:
        raise ValueError("refraction_maximum_iterations must be non-negative.")
    if not math.isfinite(refraction_tolerance) or refraction_tolerance <= 0.0:
        raise ValueError("refraction_tolerance must be finite and positive.")
    points_ = jnp.asarray(points)
    if points_.shape[-1:] != (3,):
        raise ValueError("points must have shape (..., 3).")
    if jnp.issubdtype(points_.dtype, jnp.complexfloating):
        raise TypeError("Physical points must be real-valued.")
    if not jnp.issubdtype(points_.dtype, jnp.inexact):
        points_ = points_.astype(float)
    camera_points = camera.pose.frame.inverse_apply(points_)
    depth = camera_points[..., 2]
    finite = jnp.all(jnp.isfinite(points_), axis=-1) & jnp.all(
        jnp.isfinite(camera_points), axis=-1
    )
    in_front = depth > minimum_depth
    safe_depth = jnp.where(
        jnp.abs(depth) > minimum_depth,
        depth,
        jnp.where(depth < 0.0, -minimum_depth, minimum_depth),
    )
    normalized = jnp.stack(
        (camera_points[..., 0] / safe_depth, camera_points[..., 1] / safe_depth),
        axis=-1,
    )
    refraction_valid = jnp.ones(depth.shape, dtype=bool)
    refraction_status = jnp.full(
        depth.shape,
        int(ProjectionStatus.SUCCESS),
        dtype=jnp.int32,
    )
    if camera.refraction is not None:
        normalized, refraction_valid, refraction_status = _project_refracted_points(
            camera,
            points_,
            normalized,
            maximum_iterations=int(refraction_maximum_iterations),
            tolerance=float(refraction_tolerance),
        )
    distorted = _distort_normalized(camera.distortion, normalized)
    pixels = _normalized_to_pixels(camera.intrinsics, distorted)
    inside = _inside_image(camera.intrinsics, pixels)
    valid = finite & in_front & refraction_valid & inside
    status = jnp.where(
        ~finite,
        int(ProjectionStatus.NONFINITE_INPUT),
        jnp.where(
            ~in_front,
            int(ProjectionStatus.BEHIND_CAMERA),
            jnp.where(
                ~refraction_valid,
                refraction_status,
                jnp.where(
                    ~inside,
                    int(ProjectionStatus.OUTSIDE_IMAGE),
                    int(ProjectionStatus.SUCCESS),
                ),
            ),
        ),
    ).astype(jnp.int32)
    return ProjectionResult(pixels, depth, camera_points, valid, status)


def pixels_to_rays(
    camera: CameraModel,
    pixels: ArrayLike,
    *,
    maximum_iterations: int = 12,
    tolerance: float = 1e-9,
) -> RayResult:
    """Unproject ``(row, column)`` pixels to normalized world-space rays."""

    if not isinstance(camera, CameraModel):
        raise TypeError("camera must be a CameraModel.")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive.")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    pixels_ = jnp.asarray(pixels)
    if pixels_.shape[-1:] != (2,):
        raise ValueError("pixels must have shape (..., 2).")
    if jnp.issubdtype(pixels_.dtype, jnp.complexfloating):
        raise TypeError("Pixel coordinates must be real-valued.")
    if not jnp.issubdtype(pixels_.dtype, jnp.inexact):
        pixels_ = pixels_.astype(float)
    finite = jnp.all(jnp.isfinite(pixels_), axis=-1)
    inside = _inside_image(camera.intrinsics, pixels_)
    target = _pixels_to_distorted_normalized(camera.intrinsics, pixels_)
    normalized = target
    iterations = jnp.zeros(target.shape[:-1], dtype=jnp.int32)
    linear_valid = jnp.ones(target.shape[:-1], dtype=bool)
    plan = SmallLinearSolvePlan(
        2,
        singular_tolerance=1e-12,
        maximum_condition=1e10,
        refinement_iterations=1,
    )
    for _ in range(int(maximum_iterations)):
        residual = _distort_normalized(camera.distortion, normalized) - target
        residual_norm = jnp.sqrt(jnp.sum(residual * residual, axis=-1))
        jacobian = _distortion_jacobian(camera.distortion, normalized)
        linear = solve_small_linear(plan, jacobian, residual)
        converged = residual_norm <= tolerance
        update = finite & linear_valid & linear.successful & ~converged
        normalized = normalized - jnp.where(update[..., None], linear.value, 0.0)
        iterations = iterations + update.astype(jnp.int32)
        linear_valid = linear_valid & (linear.successful | converged)
    residual = _distort_normalized(camera.distortion, normalized) - target
    residual_norm = jnp.sqrt(jnp.sum(residual * residual, axis=-1))
    distortion_valid = linear_valid & (residual_norm <= tolerance)

    direction_camera = jnp.concatenate(
        (normalized, jnp.ones(normalized.shape[:-1] + (1,), dtype=normalized.dtype)),
        axis=-1,
    )
    direction_camera = direction_camera / jnp.sqrt(
        jnp.sum(direction_camera * direction_camera, axis=-1, keepdims=True)
    )
    rotation = camera.pose.frame.rotation.astype(direction_camera.dtype)
    directions = contract("...i,ji->...j", direction_camera, rotation)
    origins = jnp.broadcast_to(
        camera.pose.frame.translation.astype(direction_camera.dtype),
        directions.shape,
    )
    refraction_valid = jnp.ones(finite.shape, dtype=bool)
    refraction_status = jnp.full(
        finite.shape,
        int(RefractionStatus.SUCCESS),
        dtype=jnp.int32,
    )
    if camera.refraction is not None:
        traced = _trace_refracted_arrays(
            camera.refraction,
            origins,
            directions,
            parallel_tolerance=1e-10,
            intersection_tolerance=1e-9,
        )
        origins, directions, refraction_valid, refraction_status = traced[:4]
    valid = finite & inside & distortion_valid & refraction_valid
    status = jnp.where(
        ~finite,
        int(RayStatus.NONFINITE_INPUT),
        jnp.where(
            ~inside,
            int(RayStatus.OUTSIDE_IMAGE),
            jnp.where(
                ~distortion_valid,
                int(RayStatus.DISTORTION_NONCONVERGENCE),
                refraction_status,
            ),
        ),
    ).astype(jnp.int32)
    return RayResult(origins, directions, valid, status, iterations, residual_norm)


__all__ = [
    "BrownConradyDistortion",
    "CameraIntrinsics",
    "CameraModel",
    "CameraPose",
    "ProjectionResult",
    "ProjectionStatus",
    "RayResult",
    "RayStatus",
    "pixels_to_rays",
    "project_points",
]
