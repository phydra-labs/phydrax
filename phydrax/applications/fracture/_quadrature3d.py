#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._geometry3d import CrackSurfaceGeometry3D


class SharpCrackQuadrature3D(StrictModule, NonTrainableState):
    """One-point face/front rules preserving represented area and length exactly."""

    surface_points: Array
    surface_weights: Array
    surface_normals: Array
    front_points: Array
    front_weights: Array
    front_tangents: Array
    quadrature_id: str = eqx.field(static=True)

    def __init__(self, geometry: CrackSurfaceGeometry3D, /):
        if not isinstance(geometry, CrackSurfaceGeometry3D):
            raise TypeError("geometry must be CrackSurfaceGeometry3D.")
        points = jnp.mean(geometry.vertices[geometry.triangles], axis=1)
        self.surface_points = points
        self.surface_weights = geometry.triangle_areas
        self.surface_normals = geometry.triangle_normals
        self.front_points = geometry.front_midpoints
        self.front_weights = geometry.front_lengths
        self.front_tangents = geometry.front_tangents
        self.quadrature_id = canonical_fingerprint(
            {
                "kind": "sharp-crack-quadrature-3d",
                "surface": geometry.surface_id,
                "surface_rule": "triangle-centroid",
                "front_rule": "edge-midpoint",
            }
        )

    @property
    def represented_surface_area(self) -> Array:
        return jnp.sum(self.surface_weights)

    @property
    def represented_front_length(self) -> Array:
        return jnp.sum(self.front_weights)


__all__ = ["SharpCrackQuadrature3D"]
