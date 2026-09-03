#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from ._geometry import SurfelGeometryState


class SurfelFootprintEvaluation(StrictModule):
    signed_normal_distance: Array
    tangent_coordinates: Array
    projected_points: Array
    normalized_radius_squared: Array
    kernel_weight: Array
    inside: Array
    active: Array
    solve_successful: Array
    finite: Array
    successful: Array


class SurfelFootprintPlan(NonTrainableState, StrictModule):
    """Evaluate compact anisotropic surfel footprints for paired queries."""

    ambient_dimension: int = eqx.field(static=True)
    solve_plan: SmallLinearSolvePlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        ambient_dimension: int,
        /,
        *,
        singular_tolerance: float = 1.0e-12,
        maximum_condition: float = 1.0e8,
    ) -> None:
        dimension = int(ambient_dimension)
        if dimension not in (2, 3):
            raise ValueError("Surfel footprints require ambient dimension two or three.")
        self.ambient_dimension = dimension
        self.solve_plan = SmallLinearSolvePlan(
            dimension - 1,
            singular_tolerance=singular_tolerance,
            maximum_condition=maximum_condition,
            refinement_iterations=1,
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "surfel-footprint-plan",
                "ambient_dimension": dimension,
                "solve_plan": self.solve_plan.plan_id,
            }
        )

    def evaluate(
        self,
        geometry: SurfelGeometryState,
        query_points: ArrayLike,
        surfel_slots: ArrayLike,
        /,
    ) -> SurfelFootprintEvaluation:
        if not isinstance(geometry, SurfelGeometryState):
            raise TypeError("geometry must be SurfelGeometryState.")
        if geometry.ambient_dimension != self.ambient_dimension:
            raise ValueError("Surfel geometry and footprint dimensions disagree.")
        points = jnp.asarray(query_points, dtype=geometry.position.dtype)
        slots = jnp.asarray(surfel_slots)
        if points.ndim < 1 or points.shape[-1] != self.ambient_dimension:
            raise ValueError(
                "query_points must have the surfel ambient trailing dimension."
            )
        if slots.shape != points.shape[:-1] or not jnp.issubdtype(
            slots.dtype, jnp.integer
        ):
            raise ValueError(
                "surfel_slots must be an integer array matching query leading axes."
            )
        valid_slot = (slots >= 0) & (slots < geometry.capacity)
        safe_slot = jnp.clip(slots, 0, geometry.capacity - 1).astype(jnp.int32)
        position = geometry.position[safe_slot]
        normal = geometry.normal[safe_slot]
        axes = geometry.tangent_axes[safe_slot]
        gram = geometry.tangent_gram[safe_slot]
        displacement = points - position
        normal_distance = jnp.sum(normal * displacement, axis=-1)
        right = contract("...ik,...i->...k", axes, displacement)
        solve = solve_small_linear(self.solve_plan, gram, right)
        tangent_coordinates = solve.value
        tangent_displacement = contract("...ik,...k->...i", axes, tangent_coordinates)
        projected = position + tangent_displacement
        radius_squared = jnp.sum(tangent_coordinates**2, axis=-1)
        radius = jnp.sqrt(jnp.maximum(radius_squared, 0.0))
        compact_radius = jnp.maximum(1.0 - radius, 0.0)
        kernel_weight = compact_radius**4 * (4.0 * radius + 1.0)
        active = valid_slot & geometry.active_mask[safe_slot]
        finite = (
            jnp.all(jnp.isfinite(points), axis=-1)
            & jnp.isfinite(normal_distance)
            & jnp.all(jnp.isfinite(tangent_coordinates), axis=-1)
            & jnp.isfinite(radius_squared)
        )
        inside = active & solve.successful & finite & (radius_squared <= 1.0)
        successful = active & solve.successful & finite & geometry.evidence.successful
        return SurfelFootprintEvaluation(
            signed_normal_distance=jnp.where(active, normal_distance, 0.0),
            tangent_coordinates=jnp.where(active[..., None], tangent_coordinates, 0.0),
            projected_points=jnp.where(active[..., None], projected, 0.0),
            normalized_radius_squared=jnp.where(active, radius_squared, jnp.inf),
            kernel_weight=jnp.where(inside, kernel_weight, 0.0),
            inside=inside,
            active=active,
            solve_successful=active & solve.successful,
            finite=finite,
            successful=successful,
        )


__all__ = [
    "SurfelFootprintEvaluation",
    "SurfelFootprintPlan",
]
