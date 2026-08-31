#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike
from opt_einsum import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..imaging import ImageGeometry2D
from ._types import DetectionStatus, ParticleDetections


class ParticleDetectionPlan(StrictModule, NonTrainableState):
    """Fixed-capacity difference-of-Gaussians particle detector."""

    small_sigma: float = eqx.field(static=True)
    large_sigma: float = eqx.field(static=True)
    threshold: float = eqx.field(static=True)
    local_maximum_radius: int = eqx.field(static=True)
    centroid_radius: int = eqx.field(static=True)
    border_width: int = eqx.field(static=True)
    crowding_distance: float = eqx.field(static=True)
    covariance_floor: float = eqx.field(static=True)
    maximum_detections: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        small_sigma: float = 0.8,
        large_sigma: float = 1.6,
        threshold: float = 0.0,
        local_maximum_radius: int = 1,
        centroid_radius: int = 2,
        border_width: int = 2,
        crowding_distance: float = 3.0,
        covariance_floor: float = 1e-4,
        maximum_detections: int = 1024,
    ):
        if not (0.0 < float(small_sigma) < float(large_sigma)):
            raise ValueError("Require 0 < small_sigma < large_sigma.")
        if not jnp.isfinite(threshold):
            raise ValueError("threshold must be finite.")
        for name, value in (
            ("local_maximum_radius", local_maximum_radius),
            ("centroid_radius", centroid_radius),
            ("border_width", border_width),
            ("maximum_detections", maximum_detections),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer.")
        if local_maximum_radius < 1 or centroid_radius < 1 or border_width < 0:
            raise ValueError("Detection window radii are invalid.")
        if maximum_detections <= 0:
            raise ValueError("maximum_detections must be positive.")
        if crowding_distance <= 0.0 or covariance_floor <= 0.0:
            raise ValueError("crowding_distance and covariance_floor must be positive.")
        self.small_sigma = float(small_sigma)
        self.large_sigma = float(large_sigma)
        self.threshold = float(threshold)
        self.local_maximum_radius = int(local_maximum_radius)
        self.centroid_radius = int(centroid_radius)
        self.border_width = int(border_width)
        self.crowding_distance = float(crowding_distance)
        self.covariance_floor = float(covariance_floor)
        self.maximum_detections = int(maximum_detections)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "point-particle-dog-detector",
                "small_sigma": self.small_sigma,
                "large_sigma": self.large_sigma,
                "threshold": self.threshold,
                "local_maximum_radius": self.local_maximum_radius,
                "centroid_radius": self.centroid_radius,
                "border_width": self.border_width,
                "crowding_distance": self.crowding_distance,
                "covariance_floor": self.covariance_floor,
                "maximum_detections": self.maximum_detections,
            }
        )


def _gaussian_kernel(sigma: float, dtype, /):
    radius = max(1, int(3.0 * sigma + 0.5))
    coordinates = jnp.arange(-radius, radius + 1, dtype=dtype)
    kernel = jnp.exp(-0.5 * (coordinates / sigma) ** 2)
    return kernel / jnp.sum(kernel)


def _separable_blur(image, sigma: float, /):
    kernel = _gaussian_kernel(sigma, image.dtype)
    values = image[None, None, :, :]
    vertical = kernel[:, None][None, None, :, :]
    horizontal = kernel[None, :][None, None, :, :]
    blurred = jax.lax.conv_general_dilated(
        values,
        vertical,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    blurred = jax.lax.conv_general_dilated(
        blurred,
        horizontal,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    return blurred[0, 0]


def detect_particles(
    image: ArrayLike,
    geometry: ImageGeometry2D,
    plan: ParticleDetectionPlan,
    /,
    *,
    valid_mask: ArrayLike | None = None,
    frame_id: str | None = None,
) -> ParticleDetections:
    """Detect and moment-refine bright point particles without dynamic allocation."""
    if not isinstance(geometry, ImageGeometry2D):
        raise TypeError("geometry must be an ImageGeometry2D.")
    if not isinstance(plan, ParticleDetectionPlan):
        raise TypeError("plan must be a ParticleDetectionPlan.")
    values = jnp.asarray(image)
    if values.ndim != 2 or tuple(values.shape) != geometry.image_shape:
        raise ValueError("image shape must equal geometry.image_shape.")
    if values.size < plan.maximum_detections:
        raise ValueError("maximum_detections cannot exceed the image pixel count.")
    if not jnp.issubdtype(values.dtype, jnp.inexact):
        values = values.astype(float)
    finite = jnp.isfinite(values)
    support = (
        finite if valid_mask is None else finite & jnp.asarray(valid_mask, dtype=bool)
    )
    if support.shape != values.shape:
        raise ValueError("valid_mask must have the image shape.")
    safe_image = jnp.where(support, values, 0.0)
    narrow = _separable_blur(safe_image, plan.small_sigma)
    broad = _separable_blur(safe_image, plan.large_sigma)
    response_image = narrow - broad
    window = 2 * plan.local_maximum_radius + 1
    local_maximum = jax.lax.reduce_window(
        response_image,
        -jnp.inf,
        jax.lax.max,
        (window, window),
        (1, 1),
        "SAME",
    )
    candidates = (
        support & (response_image >= plan.threshold) & (response_image == local_maximum)
    )
    candidate_count = jnp.sum(candidates, dtype=jnp.int32)
    flat_scores = jnp.where(
        candidates.reshape((-1,)), response_image.reshape((-1,)), -jnp.inf
    )
    selected_response, flat_index = jax.lax.top_k(flat_scores, plan.maximum_detections)
    selected_valid = jnp.isfinite(selected_response)
    row_count, column_count = values.shape
    centers = jnp.stack(
        (flat_index // column_count, flat_index % column_count), axis=-1
    ).astype(values.dtype)

    radius = plan.centroid_radius
    offsets_1d = jnp.arange(-radius, radius + 1, dtype=jnp.int32)
    offset_rows, offset_columns = jnp.meshgrid(offsets_1d, offsets_1d, indexing="ij")
    offsets = jnp.stack(
        (offset_rows.reshape((-1,)), offset_columns.reshape((-1,))), axis=-1
    )
    integer_centers = centers.astype(jnp.int32)
    locations = integer_centers[:, None, :] + offsets[None, :, :]
    in_bounds = (
        (locations[..., 0] >= 0)
        & (locations[..., 0] < row_count)
        & (locations[..., 1] >= 0)
        & (locations[..., 1] < column_count)
    )
    clipped_rows = jnp.clip(locations[..., 0], 0, row_count - 1)
    clipped_columns = jnp.clip(locations[..., 1], 0, column_count - 1)
    patch_support = in_bounds & support[clipped_rows, clipped_columns]
    patch = jnp.where(
        patch_support,
        safe_image[clipped_rows, clipped_columns],
        0.0,
    )
    patch_minimum = jnp.min(jnp.where(patch_support, patch, jnp.inf), axis=-1)
    patch_minimum = jnp.where(jnp.isfinite(patch_minimum), patch_minimum, 0.0)
    weights = jnp.where(
        patch_support, jnp.maximum(patch - patch_minimum[:, None], 0.0), 0.0
    )
    total_weight = jnp.sum(weights, axis=-1)
    usable_moment = selected_valid & (total_weight > 0.0)
    safe_weight = jnp.where(total_weight > 0.0, total_weight, 1.0)
    displacement = (
        contract("dn,ni->di", weights, offsets.astype(values.dtype))
        / safe_weight[:, None]
    )
    row_column = centers + displacement
    centered_offsets = offsets.astype(values.dtype)[None, :, :] - displacement[:, None, :]
    covariance = (
        contract("dn,dni,dnj->dij", weights, centered_offsets, centered_offsets)
        / safe_weight[:, None, None]
    )
    covariance = covariance + plan.covariance_floor * jnp.eye(2, dtype=values.dtype)
    intensity = jnp.sum(jnp.where(patch_support, patch, 0.0), axis=-1)
    particle_radius = jnp.sqrt(
        jnp.maximum(jnp.trace(covariance, axis1=-2, axis2=-1), 0.0)
    )

    border = usable_moment & (
        (row_column[:, 0] < plan.border_width)
        | (row_column[:, 0] > row_count - 1 - plan.border_width)
        | (row_column[:, 1] < plan.border_width)
        | (row_column[:, 1] > column_count - 1 - plan.border_width)
    )
    crowding_radius = int(plan.crowding_distance) + 1
    crowding_offsets = jnp.arange(
        -crowding_radius,
        crowding_radius + 1,
        dtype=values.dtype,
    )
    crowding_rows, crowding_columns = jnp.meshgrid(
        crowding_offsets,
        crowding_offsets,
        indexing="ij",
    )
    crowding_kernel = (
        crowding_rows * crowding_rows + crowding_columns * crowding_columns
        < plan.crowding_distance**2
    ).astype(jnp.int32)
    candidate_neighbors = jax.lax.conv_general_dilated(
        candidates[None, None].astype(jnp.int32),
        crowding_kernel[None, None],
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )[0, 0]
    crowded = usable_moment & (
        candidate_neighbors[integer_centers[:, 0], integer_centers[:, 1]] > 1
    )
    status = jnp.where(
        crowded,
        int(DetectionStatus.CROWDED),
        jnp.where(border, int(DetectionStatus.BORDER), int(DetectionStatus.SUCCESS)),
    ).astype(jnp.int32)
    valid = usable_moment
    row_column = jnp.where(valid[:, None], row_column, 0.0)
    covariance = jnp.where(valid[:, None, None], covariance, 0.0)
    intensity = jnp.where(valid, intensity, 0.0)
    particle_radius = jnp.where(valid, particle_radius, 0.0)
    overflow_count = jnp.maximum(candidate_count - plan.maximum_detections, 0)
    resolved_frame_id = (
        f"frame:{array_tree_fingerprint(values)}" if frame_id is None else str(frame_id)
    )
    if not resolved_frame_id:
        raise ValueError("frame_id must be non-empty.")
    detection_id = "detections:" + canonical_fingerprint(
        {
            "plan": plan.plan_id,
            "frame": resolved_frame_id,
            "geometry": geometry.geometry_id,
        }
    )
    return ParticleDetections(
        positions_rc=row_column,
        covariance_rc=covariance,
        intensity=intensity,
        radius=particle_radius,
        valid=valid,
        status=status,
        overflow_count=overflow_count,
        frame_id=resolved_frame_id,
        detection_id=detection_id,
    )


__all__ = ["ParticleDetectionPlan", "detect_particles"]
