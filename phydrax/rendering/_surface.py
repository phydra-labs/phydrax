#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact primary-ray scientific images of audited triangle surfaces."""

from __future__ import annotations

from typing import Literal

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
from ..imaging import image_coordinates, ImagePlaneSupport
from ..imaging.camera import CameraModel, pixels_to_rays
from ..measurement import (
    PreparedQuantityField,
    QuantitySpec,
    SamplingSemantics,
    ValueLayout,
)
from ._result import ImageRenderResult, RenderEvidence


FieldAssociation = Literal["vertex", "face"]


class SurfaceImagePlan(StrictModule, NonTrainableState):
    """Audited surface, camera, image support, and physical field semantics."""

    realization: SurfaceRealization
    support: ImagePlaneSupport
    camera: CameraModel
    quantity: QuantitySpec = eqx.field(static=True)
    layout: ValueLayout = eqx.field(static=True)
    sampling: SamplingSemantics = eqx.field(static=True)
    field_association: FieldAssociation = eqx.field(static=True)
    leaf_size: int = eqx.field(static=True)
    traversal_stack_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        realization: SurfaceRealization,
        support: ImagePlaneSupport,
        camera: CameraModel,
        quantity: QuantitySpec,
        layout: ValueLayout,
        sampling: SamplingSemantics,
        /,
        *,
        field_association: FieldAssociation = "vertex",
        leaf_size: int = 8,
        traversal_stack_capacity: int = 64,
    ):
        if not isinstance(realization, SurfaceRealization):
            raise TypeError("realization must be SurfaceRealization.")
        if not isinstance(support, ImagePlaneSupport):
            raise TypeError("support must be ImagePlaneSupport.")
        if not isinstance(camera, CameraModel):
            raise TypeError("camera must be CameraModel.")
        if not isinstance(quantity, QuantitySpec) or not isinstance(layout, ValueLayout):
            raise TypeError("quantity and layout must be measurement contracts.")
        if not isinstance(sampling, SamplingSemantics):
            raise TypeError("sampling must be SamplingSemantics.")
        if field_association not in ("vertex", "face"):
            raise ValueError("field_association must be 'vertex' or 'face'.")
        if camera.intrinsics.image_shape is not None and (
            camera.intrinsics.image_shape != support.image_shape
        ):
            raise ValueError("Camera image_shape and image support differ.")
        leaf_size_ = int(leaf_size)
        stack_capacity = int(traversal_stack_capacity)
        if leaf_size_ < 1 or stack_capacity < 1:
            raise ValueError("leaf_size and traversal_stack_capacity must be positive.")
        self.realization = realization
        self.support = support
        self.camera = camera
        self.quantity = quantity
        self.layout = layout
        self.sampling = sampling
        self.field_association = field_association
        self.leaf_size = leaf_size_
        self.traversal_stack_capacity = stack_capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "surface-image-plan",
                "surface": realization.realization_id,
                "support": support.support_id,
                "quantity": quantity.quantity_id,
                "layout": layout.layout_id,
                "sampling": sampling.sampling_id,
                "association": field_association,
                "leaf_size": leaf_size_,
                "stack_capacity": stack_capacity,
            }
        )

    def prepare(self) -> PreparedSurfaceImage:
        return prepare_surface_image(self)


class PreparedSurfaceImage(StrictModule):
    plan: SurfaceImagePlan
    triangle_query: PreparedTriangleRayQuery
    pixel_coordinates: Array
    vertex_count: int = eqx.field(static=True)
    face_count: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def render(
        self,
        vertices: ArrayLike,
        field_values: ArrayLike,
        /,
        *,
        geometry_id: str,
    ) -> ImageRenderResult:
        geometry = refit_triangle_ray_geometry(
            self.triangle_query,
            vertices,
            geometry_id=geometry_id,
        )
        rays = pixels_to_rays(self.plan.camera, self.pixel_coordinates)
        hits = intersect_triangle_rays(
            self.triangle_query,
            rays.origins,
            rays.directions,
            geometry=geometry,
        )
        values = jnp.asarray(field_values)
        expected_count = (
            self.vertex_count
            if self.plan.field_association == "vertex"
            else self.face_count
        )
        expected_shape = (expected_count,) + self.plan.layout.component_shape
        if values.shape != expected_shape:
            raise ValueError(f"field_values must have shape {expected_shape}.")
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            finite_values = jnp.isfinite(values.real) & jnp.isfinite(values.imag)
        else:
            finite_values = jnp.isfinite(values)
        safe_triangle = jnp.maximum(hits.triangle_indices, 0)
        if self.plan.field_association == "vertex":
            triangle_vertices = self.triangle_query.triangles[safe_triangle]
            selected = values[triangle_vertices]
            payload_rank = len(self.plan.layout.component_shape)
            barycentric = hits.barycentric_coordinates.reshape(
                hits.barycentric_coordinates.shape + (1,) * payload_rank
            )
            rendered = jnp.sum(barycentric * selected, axis=2)
            selected_finite = jnp.all(finite_values[triangle_vertices], axis=2)
        else:
            rendered = values[safe_triangle]
            selected_finite = finite_values[safe_triangle]
        if self.plan.layout.component_shape:
            selected_finite = jnp.all(
                selected_finite,
                axis=tuple(range(-len(self.plan.layout.component_shape), 0)),
            )
        valid = rays.valid & hits.successful & selected_finite
        expanded_valid = valid.reshape(
            valid.shape + (1,) * len(self.plan.layout.component_shape)
        )
        rendered = jnp.where(expanded_valid, rendered, 0.0)
        prediction = PreparedQuantityField(
            rendered,
            valid,
            standard_uncertainty=None,
            quantity_id=self.plan.quantity.quantity_id,
            compatibility_id=self.plan.quantity.compatibility_id,
            layout_id=self.plan.layout.layout_id,
            support_id=self.plan.support.support_id,
            sampling_id=self.plan.sampling.sampling_id,
            unit_id=self.plan.quantity.unit.unit_id,
            field_id=f"{self.prepared_id}:prediction",
        )
        capacity_sufficient = ~jnp.any(
            hits.status == int(TriangleRayIntersectionStatus.TRAVERSAL_CAPACITY_EXHAUSTED)
        )
        finite = geometry.finite & jnp.all(
            jnp.where(expanded_valid, jnp.isfinite(rendered), True)
        )
        coverage = jnp.all(rays.valid)
        route_stable = jnp.all(
            (~hits.successful)
            | (hits.uniqueness_margin > self.triangle_query.tie_tolerance)
        )
        status = jnp.where(
            ~geometry.successful,
            1,
            jnp.where(~capacity_sufficient, 2, jnp.where(~finite, 3, 0)),
        ).astype(jnp.int32)
        evidence = RenderEvidence(
            finite,
            coverage,
            capacity_sufficient,
            geometry.successful,
            route_stable,
            status,
            approximation="exact-primary-ray",
            plan_id=self.prepared_id,
            support_id=self.plan.support.support_id,
            geometry_id=geometry.geometry_id,
        )
        return ImageRenderResult(
            prediction,
            hits.intersection.distances,
            hits.successful,
            hits.triangle_indices,
            hits.entity_ids,
            hits.barycentric_coordinates,
            hits.intersection.points,
            hits.oriented_normals,
            hits.front_facing,
            hits.uniqueness_margin,
            evidence,
        )


def prepare_surface_image(plan: SurfaceImagePlan, /) -> PreparedSurfaceImage:
    if not isinstance(plan, SurfaceImagePlan):
        raise TypeError("plan must be SurfaceImagePlan.")
    mesh = plan.realization.mesh
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolygonalConnectivity):
        raise TypeError("SurfaceImagePlan requires polygonal surface connectivity.")
    faces = np.asarray(connectivity.cell_vertices, dtype=np.int32)
    kinds = np.asarray(connectivity.cell_kinds, dtype=np.int32)
    if faces.ndim != 2 or kinds.shape != (faces.shape[0],) or np.any(kinds != 3):
        raise ValueError("SurfaceImagePlan requires affine triangle cells.")
    triangles = faces[:, :3]
    entity_ids = np.asarray(mesh.entity_set(2).entity_ids)
    if np.any(entity_ids > np.iinfo(np.int32).max):
        raise ValueError("Surface entity IDs exceed the triangle-query integer range.")
    query = prepare_triangle_ray_query(
        TriangleRayQueryPlan(
            mesh.coordinates,
            triangles,
            entity_ids=entity_ids,
            leaf_size=plan.leaf_size,
            traversal_stack_capacity=plan.traversal_stack_capacity,
        )
    )
    indices = image_coordinates(plan.support)
    pixels = plan.support.pixel_origin_rc + plan.support.pixel_spacing_rc * indices
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-surface-image",
            "plan": plan.plan_id,
            "triangle_query": query.prepared_id,
        }
    )
    return PreparedSurfaceImage(
        plan,
        query,
        pixels,
        int(mesh.coordinates.shape[0]),
        int(triangles.shape[0]),
        prepared_id,
    )


__all__ = [
    "FieldAssociation",
    "PreparedSurfaceImage",
    "SurfaceImagePlan",
    "prepare_surface_image",
]
