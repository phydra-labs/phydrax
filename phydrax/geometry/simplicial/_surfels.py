#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.surfel import (
    PreparedSurfelDiscretization,
    SurfelAccuracy,
    SurfelCoverageScope,
    SurfelFootprintMeaning,
    SurfelGeometryCertificate,
    SurfelGeometryPlan,
    SurfelGeometryState,
    SurfelOrientationScope,
    SurfelSetPlan,
)
from ._regions import TriangleSurface


class PreparedSimplicialSurfels(StrictModule):
    surface: TriangleSurface
    discretization: PreparedSurfelDiscretization
    geometry: SurfelGeometryState
    prepared_id: str = eqx.field(static=True)


class SimplicialSurfelPlan(StrictModule, NonTrainableState):
    """Prepare one oriented quadrature surfel per triangular surface face."""

    surface: TriangleSurface
    footprint_area_ratio: float = eqx.field(static=True)
    name: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        surface: TriangleSurface,
        /,
        *,
        footprint_area_ratio: float = 1.0,
        name: str = "simplicial-surfels",
    ) -> None:
        if not isinstance(surface, TriangleSurface):
            raise TypeError("surface must be TriangleSurface.")
        ratio = float(footprint_area_ratio)
        name_value = str(name).strip()
        if not np.isfinite(ratio) or ratio <= 0.0:
            raise ValueError("footprint_area_ratio must be finite and positive.")
        if not name_value:
            raise ValueError("name must be nonempty.")
        self.surface = surface
        self.footprint_area_ratio = ratio
        self.name = name_value
        self.plan_id = canonical_fingerprint(
            {
                "kind": "simplicial-surfel-plan",
                "surface": surface.mesh.source_id,
                "footprint_area_ratio": ratio,
                "name": name_value,
            }
        )

    def prepare(self, /, *, numeric_version: str = "0") -> PreparedSimplicialSurfels:
        mesh = self.surface.mesh
        triangles = mesh.triangles
        positions = jnp.mean(triangles, axis=1)
        normals = mesh.face_normals
        areas = mesh.face_areas
        first_axis = triangles[:, 1] - triangles[:, 0]
        first_axis = first_axis / jnp.sqrt(jnp.sum(first_axis**2, axis=-1, keepdims=True))
        second_axis = jnp.cross(normals, first_axis)
        radius = jnp.sqrt(self.footprint_area_ratio * areas / jnp.pi)
        tangent_axes = (
            jnp.stack((first_axis, second_axis), axis=-1) * radius[:, None, None]
        )
        face_ids = jnp.arange(triangles.shape[0], dtype=jnp.int64)
        discretization = SurfelSetPlan(
            face_ids,
            positions,
            areas,
            source_entity_ids=face_ids,
            name=self.name,
            plan_id=canonical_fingerprint(
                {
                    "kind": "simplicial-surfel-set",
                    "plan": self.plan_id,
                }
            ),
        ).prepare(numeric_version=numeric_version)
        certificate = SurfelGeometryCertificate(
            source_geometry_id=mesh.source_id,
            source_kind="triangle_surface",
            position_accuracy=SurfelAccuracy.EXACT,
            normal_accuracy=SurfelAccuracy.EXACT,
            orientation_scope=SurfelOrientationScope.COMPONENT,
            coverage_scope=SurfelCoverageScope.SAMPLED,
            footprint_meaning=SurfelFootprintMeaning.QUADRATURE_PATCH,
            one_sided=True,
            provenance=("triangle_surface", "face_centroid_quadrature"),
        )
        geometry = SurfelGeometryPlan(discretization).materialize(
            positions,
            normals,
            tangent_axes,
            physical_surface_weights=areas,
            certificate=certificate,
        )
        return PreparedSimplicialSurfels(
            surface=self.surface,
            discretization=discretization,
            geometry=geometry,
            prepared_id=canonical_fingerprint(
                {
                    "kind": "prepared-simplicial-surfels",
                    "plan": self.plan_id,
                    "discretization": discretization.prepared_id,
                }
            ),
        )


__all__ = ["PreparedSimplicialSurfels", "SimplicialSurfelPlan"]
