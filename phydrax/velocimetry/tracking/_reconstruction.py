#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import SmallLinearSolvePlan
from ...optim import AbstractRobustLoss
from ..camera import CameraRig, pixels_to_rays, triangulate_weighted_rays
from ._association import MultiViewAssociationResult
from ._types import ParticleDetections


class TriangulationPlan(StrictModule, NonTrainableState):
    """Camera-core triangulation policy for frozen camera tuples."""

    robust_loss: AbstractRobustLoss | None
    small_solve_plan: SmallLinearSolvePlan
    minimum_views: int = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    convergence_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        minimum_views: int = 2,
        maximum_iterations: int = 8,
        convergence_tolerance: float = 1e-7,
        robust_loss: AbstractRobustLoss | None = None,
        small_solve_plan: SmallLinearSolvePlan | None = None,
    ):
        for name, value in (
            ("minimum_views", minimum_views),
            ("maximum_iterations", maximum_iterations),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer.")
        if minimum_views < 2 or maximum_iterations <= 0:
            raise ValueError("Triangulation view and iteration limits are invalid.")
        if convergence_tolerance <= 0.0 or not jnp.isfinite(convergence_tolerance):
            raise ValueError("convergence_tolerance must be finite and positive.")
        if robust_loss is not None and not isinstance(robust_loss, AbstractRobustLoss):
            raise TypeError("robust_loss must be an AbstractRobustLoss or None.")
        resolved_small_solve = (
            SmallLinearSolvePlan(3) if small_solve_plan is None else small_solve_plan
        )
        if not isinstance(resolved_small_solve, SmallLinearSolvePlan):
            raise TypeError("small_solve_plan must be a SmallLinearSolvePlan or None.")
        if resolved_small_solve.dimension != 3:
            raise ValueError("small_solve_plan must solve dimension three.")
        self.robust_loss = robust_loss
        self.small_solve_plan = resolved_small_solve
        self.minimum_views = int(minimum_views)
        self.maximum_iterations = int(maximum_iterations)
        self.convergence_tolerance = float(convergence_tolerance)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "particle-ray-triangulation",
                "minimum_views": self.minimum_views,
                "maximum_iterations": self.maximum_iterations,
                "convergence_tolerance": self.convergence_tolerance,
                "robust_loss": None if robust_loss is None else robust_loss.loss_id,
                "small_solve_plan": resolved_small_solve.plan_id,
            }
        )


class ParticleReconstructionResult(StrictModule):
    """Fixed-capacity physical particle reconstruction in ``(x, y, z)`` order."""

    positions_xyz: Array
    covariance_xyz: Array
    intensity: Array
    valid: Array
    status: Array
    detection_indices: Array
    reprojection_residual: Array
    reconstruction_id: str = eqx.field(static=True)


def reconstruct_particles(
    detections_by_camera: tuple[ParticleDetections, ...],
    rig: CameraRig,
    association: MultiViewAssociationResult,
    triangulation_plan: TriangulationPlan,
    /,
) -> ParticleReconstructionResult:
    """Triangulate selected multiview detections through the camera-core solver."""
    if not isinstance(rig, CameraRig):
        raise TypeError("rig must be a CameraRig.")
    if not isinstance(association, MultiViewAssociationResult):
        raise TypeError("association must be a MultiViewAssociationResult.")
    if not isinstance(triangulation_plan, TriangulationPlan):
        raise TypeError("triangulation_plan must be a TriangulationPlan.")
    camera_count = rig.capacity
    if len(detections_by_camera) != camera_count:
        raise ValueError("detections_by_camera and rig capacities must match.")
    indices = jnp.asarray(association.detection_indices, dtype=jnp.int32)
    if indices.ndim != 2 or indices.shape[1] != camera_count:
        raise ValueError(
            "association.detection_indices must have shape (particle, camera)."
        )
    origins = []
    directions = []
    ray_valid = []
    ray_weights = []
    intensities = []
    source_ids = []
    for camera_index, camera in enumerate(rig.cameras):
        detections = detections_by_camera[camera_index]
        if not isinstance(detections, ParticleDetections):
            raise TypeError(
                "detections_by_camera must contain ParticleDetections values."
            )
        detection_capacity = int(detections.positions_rc.shape[0])
        if detections.positions_rc.shape != (detection_capacity, 2):
            raise ValueError(
                "ParticleDetections.positions_rc must have shape (capacity, 2)."
            )
        camera_indices = indices[:, camera_index]
        safe_indices = jnp.clip(camera_indices, 0, detection_capacity - 1)
        rays = pixels_to_rays(camera, detections.positions_rc[safe_indices])
        selected = association.valid & (camera_indices >= 0)
        valid = (
            selected
            & jnp.asarray(detections.valid, dtype=bool)[safe_indices]
            & rig.camera_valid[camera_index]
            & rays.valid
        )
        covariance_trace = jnp.trace(
            detections.covariance_rc[safe_indices], axis1=-2, axis2=-1
        )
        intensity = detections.intensity[safe_indices]
        precision = intensity / jnp.maximum(covariance_trace, 1e-12)
        origins.append(rays.origins)
        directions.append(rays.directions)
        ray_valid.append(valid)
        ray_weights.append(jnp.where(valid, precision, 0.0))
        intensities.append(jnp.where(valid, intensity, 0.0))
        source_ids.append(detections.detection_id)
    stacked_origins = jnp.stack(origins, axis=1)
    stacked_directions = jnp.stack(directions, axis=1)
    stacked_valid = jnp.stack(ray_valid, axis=1)
    stacked_weights = jnp.stack(ray_weights, axis=1)
    view_count = jnp.sum(stacked_valid, axis=-1, dtype=jnp.int32)
    usable = association.valid & (view_count >= triangulation_plan.minimum_views)
    stacked_valid = stacked_valid & usable[:, None]
    triangulation = triangulate_weighted_rays(
        stacked_origins,
        stacked_directions,
        stacked_valid,
        stacked_weights,
        robust_loss=triangulation_plan.robust_loss,
        maximum_iterations=triangulation_plan.maximum_iterations,
        convergence_tolerance=triangulation_plan.convergence_tolerance,
        small_solve_plan=triangulation_plan.small_solve_plan,
    )
    valid = usable & triangulation.valid
    intensity_values = jnp.stack(intensities, axis=1)
    intensity_weight = stacked_valid.astype(intensity_values.dtype)
    reconstructed_intensity = jnp.sum(intensity_values, axis=-1) / jnp.maximum(
        jnp.sum(intensity_weight, axis=-1), 1.0
    )
    reconstruction_id = "reconstruction:" + canonical_fingerprint(
        {
            "detections": tuple(source_ids),
            "association_shape": tuple(int(size) for size in indices.shape),
            "plan": triangulation_plan.plan_id,
        }
    )
    return ParticleReconstructionResult(
        positions_xyz=jnp.where(valid[:, None], triangulation.point, 0.0),
        covariance_xyz=jnp.where(valid[:, None, None], triangulation.covariance, 0.0),
        intensity=jnp.where(valid, reconstructed_intensity, 0.0),
        valid=valid,
        status=triangulation.status,
        detection_indices=jnp.where(valid[:, None], indices, -1),
        reprojection_residual=triangulation.residuals,
        reconstruction_id=reconstruction_id,
    )


__all__ = [
    "ParticleReconstructionResult",
    "TriangulationPlan",
    "reconstruct_particles",
]
