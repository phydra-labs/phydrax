#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..spatial import PreparedSparseVoxelGrid
from ._footprint import SurfelFootprintPlan
from ._geometry import SurfelGeometryState, SurfelOrientationScope


class SurfelVoxelRouteEvidence(NonTrainableState, StrictModule):
    required_routes: Array
    route_capacity: Array
    active_routes: Array
    maximum_candidates_per_surfel: Array
    surfel_candidate_capacity: Array
    candidate_overflow: Array
    route_overflow: Array
    finite: Array
    successful: Array


class SurfelVoxelProjectionEvidence(StrictModule):
    active_routes: Array
    supported_voxels: Array
    conflicting_voxels: Array
    stale_surfels: Array
    invalid_confidence_surfels: Array
    finite: Array
    successful: Array


class SurfelVoxelProjectionResult(StrictModule):
    implicit_value: Array
    denominator: Array
    surface_weight: Array
    normal: Array
    normal_coherence: Array
    contributor_count: Array
    minimum_signed_distance: Array
    maximum_signed_distance: Array
    attributes: Array | None
    supported: Array
    conflicting: Array
    evidence: SurfelVoxelProjectionEvidence
    successful: Array


class PreparedSurfelVoxelProjection(StrictModule):
    plan: SurfelVoxelProjectionPlan
    surfel_slots: Array
    brick_slots: Array
    local_slots: Array
    voxel_points: Array
    route_active: Array
    envelope_lower: Array
    envelope_upper: Array
    reference_epoch: Array
    evidence: SurfelVoxelRouteEvidence
    prepared_id: str = eqx.field(static=True)

    def project(
        self,
        geometry: SurfelGeometryState,
        /,
        *,
        confidence: ArrayLike | None = None,
        attributes: ArrayLike | None = None,
    ) -> SurfelVoxelProjectionResult:
        if not isinstance(geometry, SurfelGeometryState):
            raise TypeError("geometry must be SurfelGeometryState.")
        if geometry.discretization.prepared_id != self.plan.geometry_discretization_id:
            raise ValueError("Surfel geometry uses a different discretization.")
        capacity = geometry.capacity
        confidence_value = (
            jnp.ones((capacity,), dtype=geometry.position.dtype)
            if confidence is None
            else jnp.asarray(confidence, dtype=geometry.position.dtype)
        )
        if confidence_value.shape != (capacity,):
            raise ValueError("confidence must match surfel capacity.")
        if attributes is None:
            attribute_value = None
        else:
            attribute_value = jnp.asarray(attributes, dtype=geometry.position.dtype)
            if attribute_value.shape[0] != capacity:
                raise ValueError("attributes must have one leading value per surfel.")
        expanded_half_width = (
            geometry.footprint_half_width
            + jnp.abs(geometry.normal) * self.plan.normal_distance_support
        )
        current_lower = geometry.position - expanded_half_width
        current_upper = geometry.position + expanded_half_width
        scale = jnp.max(
            jnp.asarray(self.plan.grid.address_plan.upper)
            - jnp.asarray(self.plan.grid.address_plan.lower)
        )
        tolerance = 64.0 * jnp.finfo(geometry.position.dtype).eps * scale
        stale = geometry.active_mask & (
            jnp.any(current_lower < self.envelope_lower - tolerance, axis=-1)
            | jnp.any(current_upper > self.envelope_upper + tolerance, axis=-1)
        )
        safe_surfel = jnp.maximum(self.surfel_slots, 0)
        footprint = self.plan.footprint_plan.evaluate(
            geometry,
            self.voxel_points,
            safe_surfel,
        )
        normal_radius = jnp.abs(footprint.signed_normal_distance) / (
            self.plan.normal_distance_support
        )
        normal_compact = jnp.maximum(1.0 - normal_radius, 0.0)
        normal_weight = normal_compact**4 * (4.0 * normal_radius + 1.0)
        valid_confidence = jnp.isfinite(confidence_value) & (confidence_value >= 0.0)
        active_route = (
            self.route_active
            & geometry.active_mask[safe_surfel]
            & ~stale[safe_surfel]
            & footprint.inside
            & (normal_radius <= 1.0)
            & valid_confidence[safe_surfel]
        )
        route_weight = (
            footprint.kernel_weight
            * normal_weight
            * geometry.physical_surface_weight[safe_surfel]
            * confidence_value[safe_surfel]
        )
        route_weight = jnp.where(active_route, route_weight, 0.0)
        flat_capacity = self.plan.grid.brick_capacity * self.plan.grid.voxels_per_brick
        flat_slot = jnp.maximum(
            self.brick_slots, 0
        ) * self.plan.grid.voxels_per_brick + jnp.maximum(self.local_slots, 0)
        denominator = (
            jnp.zeros((flat_capacity,), dtype=geometry.position.dtype)
            .at[flat_slot]
            .add(route_weight)
        )
        signed_numerator = (
            jnp.zeros_like(denominator)
            .at[flat_slot]
            .add(route_weight * footprint.signed_normal_distance)
        )
        normal_numerator = (
            jnp.zeros(
                (flat_capacity, geometry.ambient_dimension),
                dtype=geometry.position.dtype,
            )
            .at[flat_slot]
            .add(route_weight[:, None] * geometry.normal[safe_surfel])
        )
        contributor_count = (
            jnp.zeros((flat_capacity,), dtype=jnp.int32)
            .at[flat_slot]
            .add(active_route.astype(jnp.int32))
        )
        minimum_distance = (
            jnp.full((flat_capacity,), jnp.inf, dtype=geometry.position.dtype)
            .at[flat_slot]
            .min(
                jnp.where(
                    active_route,
                    footprint.signed_normal_distance,
                    jnp.inf,
                )
            )
        )
        maximum_distance = (
            jnp.full((flat_capacity,), -jnp.inf, dtype=geometry.position.dtype)
            .at[flat_slot]
            .max(
                jnp.where(
                    active_route,
                    footprint.signed_normal_distance,
                    -jnp.inf,
                )
            )
        )
        safe_denominator = jnp.where(denominator > 0.0, denominator, 1.0)
        normal_norm = jnp.sqrt(jnp.sum(normal_numerator**2, axis=-1))
        normalized_normal = (
            normal_numerator / jnp.where(normal_norm > 0.0, normal_norm, 1.0)[:, None]
        )
        coherence = normal_norm / safe_denominator
        oriented = (
            geometry.certificate.orientation_scope
            is not SurfelOrientationScope.UNORIENTED
        )
        supported = (
            (denominator >= self.plan.minimum_denominator)
            & (contributor_count > 0)
            & oriented
        )
        conflicting = supported & (coherence < self.plan.minimum_normal_coherence)
        supported = supported & ~conflicting
        implicit_value = signed_numerator / safe_denominator
        attribute_result = None
        if attribute_value is not None:
            trailing_shape = attribute_value.shape[1:]
            attribute_weight = route_weight.reshape(
                route_weight.shape + (1,) * len(trailing_shape)
            )
            attribute_numerator = (
                jnp.zeros(
                    (flat_capacity,) + trailing_shape,
                    dtype=attribute_value.dtype,
                )
                .at[flat_slot]
                .add(attribute_weight * attribute_value[safe_surfel])
            )
            attribute_result = attribute_numerator / safe_denominator.reshape(
                safe_denominator.shape + (1,) * len(trailing_shape)
            )
        output_shape = (
            self.plan.grid.brick_capacity,
            self.plan.grid.voxels_per_brick,
        )
        confidence_valid = jnp.all(~geometry.active_mask | valid_confidence)
        finite = (
            jnp.all(jnp.isfinite(denominator))
            & jnp.all(jnp.isfinite(signed_numerator))
            & jnp.all(jnp.isfinite(normal_numerator))
            & confidence_valid
            & (
                jnp.asarray(True)
                if attribute_result is None
                else jnp.all(jnp.isfinite(attribute_result))
            )
        )
        stale_count = jnp.sum(stale, dtype=jnp.int32)
        successful = (
            self.evidence.successful
            & geometry.evidence.successful
            & finite
            & (stale_count == 0)
        )
        projection_evidence = SurfelVoxelProjectionEvidence(
            active_routes=jnp.sum(active_route, dtype=jnp.int32),
            supported_voxels=jnp.sum(supported, dtype=jnp.int32),
            conflicting_voxels=jnp.sum(conflicting, dtype=jnp.int32),
            stale_surfels=stale_count,
            invalid_confidence_surfels=jnp.sum(
                geometry.active_mask & ~valid_confidence,
                dtype=jnp.int32,
            ),
            finite=finite,
            successful=successful,
        )
        return SurfelVoxelProjectionResult(
            implicit_value=jnp.where(supported, implicit_value, 0.0).reshape(
                output_shape
            ),
            denominator=denominator.reshape(output_shape),
            surface_weight=denominator.reshape(output_shape),
            normal=jnp.where(supported[:, None], normalized_normal, 0.0).reshape(
                output_shape + (geometry.ambient_dimension,)
            ),
            normal_coherence=coherence.reshape(output_shape),
            contributor_count=contributor_count.reshape(output_shape),
            minimum_signed_distance=jnp.where(
                contributor_count > 0, minimum_distance, jnp.inf
            ).reshape(output_shape),
            maximum_signed_distance=jnp.where(
                contributor_count > 0, maximum_distance, -jnp.inf
            ).reshape(output_shape),
            attributes=(
                None
                if attribute_result is None
                else attribute_result.reshape(output_shape + attribute_value.shape[1:])
            ),
            supported=supported.reshape(output_shape),
            conflicting=conflicting.reshape(output_shape),
            evidence=projection_evidence,
            successful=successful,
        )


class SurfelVoxelProjectionPlan(NonTrainableState, StrictModule):
    """Build bounded fixed-topology routes from surfel footprints to sparse voxels."""

    grid: PreparedSparseVoxelGrid
    footprint_plan: SurfelFootprintPlan
    maximum_voxels_per_surfel: int = eqx.field(static=True)
    route_capacity: int = eqx.field(static=True)
    normal_distance_support: float = eqx.field(static=True)
    route_padding: float = eqx.field(static=True)
    minimum_denominator: float = eqx.field(static=True)
    minimum_normal_coherence: float = eqx.field(static=True)
    geometry_discretization_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedSparseVoxelGrid,
        geometry: SurfelGeometryState,
        /,
        *,
        maximum_voxels_per_surfel: int,
        route_capacity: int,
        normal_distance_support: float,
        route_padding: float = 0.0,
        minimum_denominator: float = 1.0e-12,
        minimum_normal_coherence: float = 0.25,
    ) -> None:
        if not isinstance(grid, PreparedSparseVoxelGrid):
            raise TypeError("grid must be PreparedSparseVoxelGrid.")
        if not isinstance(geometry, SurfelGeometryState):
            raise TypeError("geometry must be SurfelGeometryState.")
        if not bool(grid.evidence.successful) or not bool(geometry.evidence.successful):
            raise ValueError(
                "Surfel voxel projection requires successful grid and geometry."
            )
        candidates = int(maximum_voxels_per_surfel)
        routes = int(route_capacity)
        support = float(normal_distance_support)
        padding = float(route_padding)
        denominator = float(minimum_denominator)
        coherence = float(minimum_normal_coherence)
        if any(grid.address_plan.periodic_axes):
            raise ValueError(
                "Initial surfel voxel projection requires nonperiodic voxel axes."
            )
        if grid.dimension != geometry.ambient_dimension:
            raise ValueError("Voxel and surfel dimensions disagree.")
        if (
            candidates < 1
            or routes < 1
            or not np.isfinite(support)
            or support <= 0.0
            or not np.isfinite(padding)
            or padding < 0.0
            or not np.isfinite(denominator)
            or denominator <= 0.0
            or not np.isfinite(coherence)
            or coherence < 0.0
            or coherence > 1.0
        ):
            raise ValueError("Surfel voxel projection controls are invalid.")
        self.grid = grid
        self.footprint_plan = SurfelFootprintPlan(geometry.ambient_dimension)
        self.maximum_voxels_per_surfel = candidates
        self.route_capacity = routes
        self.normal_distance_support = support
        self.route_padding = padding
        self.minimum_denominator = denominator
        self.minimum_normal_coherence = coherence
        self.geometry_discretization_id = geometry.discretization.prepared_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "surfel-voxel-projection-plan",
                "grid": grid.grid_id,
                "geometry_discretization": self.geometry_discretization_id,
                "maximum_voxels_per_surfel": candidates,
                "route_capacity": routes,
                "normal_distance_support": support,
                "route_padding": padding,
                "minimum_denominator": denominator,
                "minimum_normal_coherence": coherence,
            }
        )

    def prepare(self, geometry: SurfelGeometryState, /) -> PreparedSurfelVoxelProjection:
        if geometry.discretization.prepared_id != self.geometry_discretization_id:
            raise ValueError("Surfel geometry uses a different discretization.")
        lower = jnp.asarray(self.grid.address_plan.lower, dtype=geometry.position.dtype)
        upper = jnp.asarray(self.grid.address_plan.upper, dtype=geometry.position.dtype)
        resolution = self.grid.address_plan.resolution
        voxel_width = (upper - lower) / resolution
        expanded_half_width = (
            geometry.footprint_half_width
            + jnp.abs(geometry.normal) * self.normal_distance_support
            + self.route_padding
        )
        envelope_lower = geometry.position - expanded_half_width
        envelope_upper = geometry.position + expanded_half_width
        minimum_coordinate = jnp.ceil(
            (envelope_lower - lower) / voxel_width - 0.5
        ).astype(jnp.int64)
        maximum_coordinate = jnp.floor(
            (envelope_upper - lower) / voxel_width - 0.5
        ).astype(jnp.int64)
        minimum_coordinate = jnp.clip(minimum_coordinate, 0, resolution - 1)
        maximum_coordinate = jnp.clip(maximum_coordinate, 0, resolution - 1)
        counts = jnp.maximum(maximum_coordinate - minimum_coordinate + 1, 0)
        candidate_count = jnp.prod(counts, axis=-1).astype(jnp.int32)
        candidate_slots = jnp.arange(self.maximum_voxels_per_surfel, dtype=jnp.int64)
        remainder = jnp.broadcast_to(
            candidate_slots[None, :],
            (geometry.capacity, self.maximum_voxels_per_surfel),
        )
        candidate_coordinates = jnp.zeros(
            (
                geometry.capacity,
                self.maximum_voxels_per_surfel,
                geometry.ambient_dimension,
            ),
            dtype=jnp.int64,
        )
        for axis in range(geometry.ambient_dimension - 1, -1, -1):
            safe_count = jnp.maximum(counts[:, axis], 1)
            offset = jnp.mod(remainder, safe_count[:, None])
            candidate_coordinates = candidate_coordinates.at[:, :, axis].set(
                minimum_coordinate[:, axis, None] + offset
            )
            remainder = remainder // safe_count[:, None]
        within_candidate_count = candidate_slots[None, :] < candidate_count[:, None]
        lookup = self.grid.lookup_integer(candidate_coordinates)
        candidate_active = (
            geometry.active_mask[:, None] & within_candidate_count & lookup.supported
        )
        flat_active = candidate_active.reshape((-1,))
        required_routes = jnp.sum(flat_active, dtype=jnp.int32)
        selected = jnp.nonzero(
            flat_active,
            size=self.route_capacity,
            fill_value=flat_active.size,
        )[0].astype(jnp.int32)
        route_slots = jnp.arange(self.route_capacity, dtype=jnp.int32)
        route_active = route_slots < jnp.minimum(required_routes, self.route_capacity)
        safe_selected = jnp.minimum(selected, flat_active.size - 1)
        surfel_slots = safe_selected // self.maximum_voxels_per_surfel
        candidate_index = safe_selected % self.maximum_voxels_per_surfel
        brick_slots = lookup.brick_slots[surfel_slots, candidate_index]
        local_slots = lookup.local_slots[surfel_slots, candidate_index]
        integer_coordinate = candidate_coordinates[surfel_slots, candidate_index]
        voxel_points = lower + voxel_width * (
            integer_coordinate.astype(geometry.position.dtype) + 0.5
        )
        candidate_overflow = jnp.any(
            geometry.active_mask & (candidate_count > self.maximum_voxels_per_surfel)
        )
        route_overflow = required_routes > self.route_capacity
        finite = (
            jnp.all(
                jnp.where(
                    geometry.active_mask[:, None],
                    jnp.isfinite(envelope_lower),
                    True,
                )
            )
            & jnp.all(
                jnp.where(
                    geometry.active_mask[:, None],
                    jnp.isfinite(envelope_upper),
                    True,
                )
            )
            & jnp.all(
                jnp.where(
                    route_active[:, None],
                    jnp.isfinite(voxel_points),
                    True,
                )
            )
        )
        successful = (
            geometry.evidence.successful
            & self.grid.evidence.successful
            & ~candidate_overflow
            & ~route_overflow
            & finite
        )
        evidence = SurfelVoxelRouteEvidence(
            required_routes=required_routes,
            route_capacity=jnp.asarray(self.route_capacity, dtype=jnp.int32),
            active_routes=jnp.sum(route_active, dtype=jnp.int32),
            maximum_candidates_per_surfel=jnp.max(candidate_count, initial=0),
            surfel_candidate_capacity=jnp.asarray(
                self.maximum_voxels_per_surfel, dtype=jnp.int32
            ),
            candidate_overflow=candidate_overflow,
            route_overflow=route_overflow,
            finite=finite,
            successful=successful,
        )
        return PreparedSurfelVoxelProjection(
            plan=self,
            surfel_slots=jnp.where(route_active, surfel_slots, -1),
            brick_slots=jnp.where(route_active, brick_slots, -1),
            local_slots=jnp.where(route_active, local_slots, -1),
            voxel_points=jnp.where(route_active[:, None], voxel_points, 0.0),
            route_active=route_active,
            envelope_lower=envelope_lower,
            envelope_upper=envelope_upper,
            reference_epoch=geometry.epoch,
            evidence=evidence,
            prepared_id=canonical_fingerprint(
                {
                    "kind": "prepared-surfel-voxel-projection",
                    "plan": self.plan_id,
                }
            ),
        )


__all__ = [
    "PreparedSurfelVoxelProjection",
    "SurfelVoxelProjectionEvidence",
    "SurfelVoxelProjectionPlan",
    "SurfelVoxelProjectionResult",
    "SurfelVoxelRouteEvidence",
]
