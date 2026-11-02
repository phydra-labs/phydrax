#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bind one B-Rep spline face to the existing atlas manifold geometry consumer.

This is a differential evaluation of the carrier, not an embedding, continuity,
trim-membership, or whole-face regularity certificate. Coordinates are the
normalized B-Rep face chart coordinates, not the native spline parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ...geometry._atlas import BoundaryAtlas
from ...geometry.brep._model import BRepBoundaryMap, BRepEntityId, BRepModel
from ...geometry.brep._patches import BSplineSurfacePatch
from ._manifold import (
    AtlasManifold,
    ManifoldBasisRealization,
    ManifoldPointGeometry,
    ParametricBasisPayload,
    realize_manifold_basis,
)


@dataclass(frozen=True, slots=True, init=False)
class BRepSplineManifoldBinding:
    """A revision-scoped spline face and its normalized-chart manifold.

    Query points must lie inside the closed chart and away from every interior
    spline knot. Spline-domain endpoints use the active one-sided boundary span.
    The returned geometry is numerical and only establishes rank at query points.
    Trim membership is not checked; callers must select points on the face.
    Construction and evaluation are host-side, not JIT-compatible.
    """

    model: BRepModel
    face_index: int
    face_id: BRepEntityId
    geometry_id: str
    binding_id: str
    manifold: AtlasManifold

    def __init__(
        self, model: BRepModel, face_index: int, /, *, rank_tolerance: float = 1.0e-10
    ) -> None:
        if not isinstance(model, BRepModel):
            raise TypeError("model must be a BRepModel.")
        if isinstance(face_index, bool) or not isinstance(face_index, int):
            raise TypeError("face_index must be an integer.")
        model.require_current()
        if not 0 <= face_index < len(model.patches):
            raise IndexError("face_index is outside the B-Rep face table.")
        patch = model.patches[face_index]
        if not isinstance(patch, BSplineSurfacePatch):
            raise TypeError("The selected B-Rep face must use a BSplineSurfacePatch.")
        bounds = np.asarray(model.parameter_bounds[face_index])
        domain = np.asarray(patch.parameter_domain)
        if np.any(bounds[0] < domain[0]) or np.any(bounds[1] > domain[1]):
            raise ValueError("Face chart bounds must lie inside the spline domain.")
        tolerance = float(rank_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("rank_tolerance must be finite and positive.")
        face_id = model.face_ids[face_index]
        binding_id = canonical_fingerprint(
            {
                "kind": "brep-spline-manifold-binding",
                "source_id": model.source_id,
                "source_revision": model.source_revision,
                "geometry_id": model.geometry.geometry_id,
                "face_index": face_index,
                "rank_tolerance": tolerance,
            }
        )
        atlas = BoundaryAtlas(
            BRepBoundaryMap(
                (patch,), model.parameter_bounds[face_index : face_index + 1]
            ),
            source_entity_ids=jnp.asarray([face_index], dtype=jnp.int32),
            source_id=binding_id,
            physical_tags=(model.physical_tags[face_index],),
            orientation=model.orientation[face_index : face_index + 1],
            trim_domains=(model.trim_domains[face_index],),
        )
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "face_index", face_index)
        object.__setattr__(self, "face_id", face_id)
        object.__setattr__(self, "geometry_id", model.geometry.geometry_id)
        object.__setattr__(self, "binding_id", binding_id)
        object.__setattr__(
            self, "manifold", AtlasManifold(atlas, rank_tolerance=tolerance)
        )

    def evaluate(self, reference: ArrayLike, /) -> ManifoldPointGeometry:
        """Evaluate first and second chart derivatives and manifold operators."""
        self.model.require_current()
        points = np.asarray(reference, dtype=np.float64)
        if points.ndim < 1 or points.shape[-1] != 2:
            raise ValueError("reference must have trailing dimension 2.")
        if not np.all(np.isfinite(points)) or np.any(points < 0) or np.any(points > 1):
            raise ValueError("Reference points must lie inside the closed unit chart.")
        patch = self.model.patches[self.face_index]
        if not isinstance(patch, BSplineSurfacePatch):
            raise RuntimeError(
                "The bound B-Rep face no longer contains a B-spline patch."
            )
        bounds = np.asarray(self.model.parameter_bounds[self.face_index])
        parameters = bounds[0] + points * (bounds[1] - bounds[0])
        for axis, knots, degree, count in (
            (0, np.asarray(patch.u_knots), patch.u_degree, patch.control_points.shape[0]),
            (1, np.asarray(patch.v_knots), patch.v_degree, patch.control_points.shape[1]),
        ):
            interior = knots[degree + 1 : count]
            if np.any(parameters[..., axis, None] == interior):
                raise ValueError(
                    "Second-geometry queries at interior spline knots are unsupported."
                )
        indices = jnp.zeros(points.shape[:-1], dtype=jnp.int32)
        return self.manifold.evaluate(indices, jnp.asarray(points))

    def realize_basis(
        self, reference: ArrayLike, payload: ParametricBasisPayload, /
    ) -> ManifoldBasisRealization:
        """Transform a supplied basis through this face's second geometry jet."""
        return realize_manifold_basis(payload, self.evaluate(reference))
