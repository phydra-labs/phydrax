#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact nearest-surface LiDAR range prediction."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import PolygonalConnectivity
from ..geometry import (
    intersect_triangle_rays,
    prepare_triangle_ray_query,
    PreparedTriangleRayQuery,
    refit_triangle_ray_geometry,
    TriangleRayIntersectionStatus,
    TriangleRayQueryPlan,
)
from ..geometry.surface import SurfaceRealization
from ..measurement import (
    PreparedQuantityField,
    QuantitySpec,
    RaySampleSupport,
    SamplingSemantics,
    ValueKind,
    ValueLayout,
)
from ..units import conversion_factor, LENGTH
from ._result import RenderEvidence


class LidarRenderResult(StrictModule, NonTrainableState):
    prediction: PreparedQuantityField
    points: Array
    normals: Array
    primitive_ids: Array
    entity_ids: Array
    incidence_cosine: Array
    uniqueness_margin: Array
    evidence: RenderEvidence


class LidarSurfacePlan(StrictModule, NonTrainableState):
    realization: SurfaceRealization
    rays: RaySampleSupport = eqx.field(static=True)
    quantity: QuantitySpec = eqx.field(static=True)
    sampling: SamplingSemantics = eqx.field(static=True)
    leaf_size: int = eqx.field(static=True)
    traversal_stack_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        realization: SurfaceRealization,
        rays: RaySampleSupport,
        quantity: QuantitySpec,
        sampling: SamplingSemantics,
        /,
        *,
        leaf_size: int = 8,
        traversal_stack_capacity: int = 64,
    ):
        if not isinstance(realization, SurfaceRealization):
            raise TypeError("realization must be SurfaceRealization.")
        if not isinstance(rays, RaySampleSupport):
            raise TypeError("rays must be RaySampleSupport.")
        if not isinstance(quantity, QuantitySpec) or quantity.unit.dimension != LENGTH:
            raise ValueError("quantity must be a length QuantitySpec.")
        if not isinstance(sampling, SamplingSemantics):
            raise TypeError("sampling must be SamplingSemantics.")
        leaf_size_ = int(leaf_size)
        stack_capacity = int(traversal_stack_capacity)
        if leaf_size_ < 1 or stack_capacity < 1:
            raise ValueError("leaf_size and traversal_stack_capacity must be positive.")
        self.realization = realization
        self.rays = rays
        self.quantity = quantity
        self.sampling = sampling
        self.leaf_size = leaf_size_
        self.traversal_stack_capacity = stack_capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lidar-surface-plan",
                "surface": realization.realization_id,
                "rays": rays.support_id,
                "quantity": quantity.quantity_id,
                "sampling": sampling.sampling_id,
                "leaf_size": leaf_size_,
                "stack_capacity": stack_capacity,
            }
        )

    def prepare(self) -> PreparedLidarSurface:
        mesh = self.realization.mesh
        connectivity = mesh.connectivity
        if not isinstance(connectivity, PolygonalConnectivity):
            raise TypeError("LidarSurfacePlan requires polygonal surface connectivity.")
        faces = np.asarray(connectivity.cell_vertices, dtype=np.int32)
        kinds = np.asarray(connectivity.cell_kinds, dtype=np.int32)
        if faces.ndim != 2 or kinds.shape != (faces.shape[0],) or np.any(kinds != 3):
            raise ValueError("LidarSurfacePlan requires affine triangle cells.")
        triangles = faces[:, :3]
        entity_ids = np.asarray(mesh.entity_set(2).entity_ids)
        if np.any(entity_ids > np.iinfo(np.int32).max):
            raise ValueError("Surface entity IDs exceed the triangle-query range.")
        query = prepare_triangle_ray_query(
            TriangleRayQueryPlan(
                mesh.coordinates,
                triangles,
                entity_ids=entity_ids,
                leaf_size=self.leaf_size,
                traversal_stack_capacity=self.traversal_stack_capacity,
            )
        )
        layout = ValueLayout(ValueKind.REAL_SCALAR)
        factor = float(
            conversion_factor(
                self.rays.coordinate_contract.length_unit,
                self.quantity.unit,
            )
        )
        return PreparedLidarSurface(
            query,
            jnp.asarray(self.rays.origins),
            jnp.asarray(self.rays.directions),
            jnp.asarray(self.rays.active_mask),
            jnp.asarray(self.rays.near),
            jnp.asarray(self.rays.far),
            factor,
            self.quantity.quantity_id,
            self.quantity.compatibility_id,
            layout.layout_id,
            self.rays.support_id,
            self.sampling.sampling_id,
            self.quantity.unit.unit_id,
            canonical_fingerprint(
                {
                    "kind": "prepared-lidar-surface",
                    "plan": self.plan_id,
                    "query": query.prepared_id,
                }
            ),
        )


class PreparedLidarSurface(StrictModule):
    triangle_query: PreparedTriangleRayQuery
    origins: Array
    directions: Array
    ray_active: Array
    near: Array
    far: Array
    range_factor: float = eqx.field(static=True)
    quantity_id: str = eqx.field(static=True)
    compatibility_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    sampling_id: str = eqx.field(static=True)
    unit_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def predict(
        self,
        vertices: ArrayLike,
        /,
        *,
        geometry_id: str,
    ) -> LidarRenderResult:
        geometry = refit_triangle_ray_geometry(
            self.triangle_query,
            vertices,
            geometry_id=geometry_id,
        )
        hit = intersect_triangle_rays(
            self.triangle_query,
            self.origins,
            self.directions,
            geometry=geometry,
        )
        distance = hit.intersection.distances
        within = (distance >= self.near) & (distance <= self.far)
        valid = self.ray_active & hit.successful & within
        ranges = jnp.where(valid, distance * self.range_factor, 0.0)
        prediction = PreparedQuantityField(
            ranges,
            valid,
            standard_uncertainty=None,
            quantity_id=self.quantity_id,
            compatibility_id=self.compatibility_id,
            layout_id=self.layout_id,
            support_id=self.support_id,
            sampling_id=self.sampling_id,
            unit_id=self.unit_id,
            field_id=f"{self.prepared_id}:range",
        )
        incidence = jnp.where(
            valid,
            jnp.abs(jnp.sum(self.directions * hit.oriented_normals, axis=-1)),
            0.0,
        )
        capacity_sufficient = ~jnp.any(
            hit.status == int(TriangleRayIntersectionStatus.TRAVERSAL_CAPACITY_EXHAUSTED)
        )
        finite = geometry.finite & jnp.all(jnp.isfinite(ranges))
        route_stable = jnp.all(
            (~hit.successful)
            | (hit.uniqueness_margin > self.triangle_query.tie_tolerance)
        )
        status = jnp.where(
            ~geometry.successful,
            1,
            jnp.where(~capacity_sufficient, 2, jnp.where(~finite, 3, 0)),
        ).astype(jnp.int32)
        evidence = RenderEvidence(
            finite,
            jnp.asarray(True),
            capacity_sufficient,
            geometry.successful,
            route_stable,
            status,
            approximation="exact-primary-ray",
            plan_id=self.prepared_id,
            support_id=self.support_id,
            geometry_id=geometry.geometry_id,
        )
        return LidarRenderResult(
            prediction,
            hit.intersection.points,
            hit.oriented_normals,
            hit.triangle_indices,
            hit.entity_ids,
            incidence,
            hit.uniqueness_margin,
            evidence,
        )


__all__ = [
    "LidarRenderResult",
    "LidarSurfacePlan",
    "PreparedLidarSurface",
]
