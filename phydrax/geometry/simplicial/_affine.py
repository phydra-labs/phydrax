#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import factorial

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.linalg as la
from phydrax import ein

from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class AffineSimplexEvidence(StrictModule):
    """Condition, measure, and degeneracy evidence for affine simplices."""

    orientation_determinant: Array
    gram_determinant: Array
    measure: Array
    condition_estimate: Array
    finite: Array
    nondegenerate: Array
    successful: Array


class AffineSimplexMap(StrictModule, NonTrainableState):
    """Prepared full-dimensional or embedded affine simplex maps."""

    vertices: Array
    origin: Array
    jacobian: Array
    dual: Array
    barycentric_gradients: Array
    evidence: AffineSimplexEvidence
    intrinsic_dimension: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)

    def __init__(self, vertices: ArrayLike, /):
        points = jnp.asarray(vertices)
        if not jnp.issubdtype(points.dtype, jnp.inexact):
            points = points.astype(float)
        if points.ndim < 2:
            raise ValueError(
                "vertices must end in (simplex_vertices, ambient_dimension)."
            )
        vertex_count = int(points.shape[-2])
        ambient_dimension = int(points.shape[-1])
        intrinsic_dimension = vertex_count - 1
        if not (
            1 <= intrinsic_dimension <= ambient_dimension <= 3
            and vertex_count == intrinsic_dimension + 1
        ):
            raise ValueError(
                "AffineSimplexMap supports intrinsic dimensions one through three "
                "embedded in ambient dimensions up to three."
            )

        origin = points[..., 0, :]
        edges = points[..., 1:, :] - origin[..., None, :]
        jacobian = jnp.swapaxes(edges, -1, -2)
        if intrinsic_dimension == ambient_dimension:
            inverse = la.inverse_small_linear(
                la.SmallLinearSolvePlan(intrinsic_dimension),
                jacobian,
            )
            dual = inverse.value
            orientation = inverse.determinant
            gram_determinant = jnp.real(orientation * jnp.conj(orientation))
            condition = inverse.condition_estimate
            algebra_successful = inverse.successful
        else:
            gram = ein.contract("...ai,...aj->...ij", jacobian, jacobian)
            inverse = la.inverse_small_linear(
                la.SmallLinearSolvePlan(intrinsic_dimension),
                gram,
            )
            dual = ein.contract("...ij,...aj->...ia", inverse.value, jacobian)
            orientation = jnp.full(
                gram.shape[:-2],
                jnp.nan,
                dtype=jnp.real(points).dtype,
            )
            gram_determinant = jnp.real(inverse.determinant)
            condition = jnp.sqrt(inverse.condition_estimate)
            algebra_successful = inverse.successful

        measure = jnp.sqrt(jnp.maximum(gram_determinant, 0.0)) / float(
            factorial(intrinsic_dimension)
        )
        scale = jnp.max(jnp.abs(jacobian), axis=(-2, -1))
        tolerance = (
            64.0
            * intrinsic_dimension
            * jnp.finfo(jnp.real(points).dtype).eps
            * jnp.maximum(scale, 1.0) ** intrinsic_dimension
        )
        finite = (
            jnp.all(jnp.isfinite(points), axis=(-2, -1))
            & jnp.all(jnp.isfinite(dual), axis=(-2, -1))
            & jnp.isfinite(measure)
            & jnp.isfinite(condition)
        )
        nondegenerate = measure > tolerance
        successful = algebra_successful & finite & nondegenerate
        first_gradient = -jnp.sum(dual, axis=-2, keepdims=True)

        self.vertices = points
        self.origin = origin
        self.jacobian = jacobian
        self.dual = dual
        self.barycentric_gradients = jnp.concatenate(
            (first_gradient, dual),
            axis=-2,
        )
        self.evidence = AffineSimplexEvidence(
            orientation,
            gram_determinant,
            measure,
            condition,
            finite,
            nondegenerate,
            successful,
        )
        self.intrinsic_dimension = intrinsic_dimension
        self.ambient_dimension = ambient_dimension

    def barycentric(self, points: ArrayLike, /) -> Array:
        """Map physical points to barycentric coordinates."""
        value = jnp.asarray(points, dtype=self.vertices.dtype)
        if value.shape[-1:] != (self.ambient_dimension,):
            raise ValueError(
                f"points must end in ambient dimension {self.ambient_dimension}."
            )
        relative = value - self.origin
        local = ein.contract("...ia,...a->...i", self.dual, relative)
        return jnp.concatenate(
            (1.0 - jnp.sum(local, axis=-1, keepdims=True), local),
            axis=-1,
        )

    @property
    def simplex_count(self) -> int:
        """Number of simplices on the indexed collection axis."""
        if self.vertices.ndim < 3:
            raise ValueError("Indexed simplex operations require a simplex collection.")
        return int(self.vertices.shape[-3])

    def _indices(self, simplex_indices: ArrayLike, /) -> Array:
        indices = jnp.asarray(simplex_indices)
        if not jnp.issubdtype(indices.dtype, jnp.integer):
            raise TypeError("simplex_indices must have an integer dtype.")
        indices = indices.astype(jnp.int32)
        return eqx.error_if(
            indices,
            jnp.any((indices < 0) | (indices >= self.simplex_count)),
            "simplex_indices contain an out-of-range simplex.",
        )

    def barycentric_at(
        self,
        points: ArrayLike,
        simplex_indices: ArrayLike,
        /,
    ) -> Array:
        """Map points through selected simplices without dense broadcasting."""
        value = jnp.asarray(points, dtype=self.vertices.dtype)
        if value.shape[-1:] != (self.ambient_dimension,):
            raise ValueError(
                f"points must end in ambient dimension {self.ambient_dimension}."
            )
        indices = self._indices(simplex_indices)
        origin = jnp.take(self.origin, indices, axis=-2)
        dual = jnp.take(self.dual, indices, axis=-3)
        local = ein.contract("...ia,...a->...i", dual, value - origin)
        return jnp.concatenate(
            (1.0 - jnp.sum(local, axis=-1, keepdims=True), local),
            axis=-1,
        )

    def contains_at(
        self,
        points: ArrayLike,
        simplex_indices: ArrayLike,
        /,
        *,
        tolerance: ArrayLike | None = None,
    ) -> Array:
        """Return containment for selected point-simplex pairs."""
        indices = self._indices(simplex_indices)
        coordinates = self.barycentric_at(points, indices)
        resolved = (
            64.0 * jnp.finfo(jnp.real(coordinates).dtype).eps
            if tolerance is None
            else jnp.asarray(tolerance, dtype=jnp.real(coordinates).dtype)
        )
        if jnp.asarray(resolved).shape != ():
            raise ValueError("tolerance must be scalar or None.")
        successful = jnp.take(self.evidence.successful, indices, axis=-1)
        return (
            jnp.all(jnp.isfinite(coordinates), axis=-1)
            & jnp.all(coordinates >= -resolved, axis=-1)
            & jnp.all(coordinates <= 1.0 + resolved, axis=-1)
            & successful
        )

    def contains(
        self,
        points: ArrayLike,
        /,
        *,
        tolerance: ArrayLike | None = None,
    ) -> Array:
        """Return closed-simplex containment with an explicit tolerance."""
        coordinates = self.barycentric(points)
        resolved = (
            64.0 * jnp.finfo(jnp.real(coordinates).dtype).eps
            if tolerance is None
            else jnp.asarray(tolerance, dtype=jnp.real(coordinates).dtype)
        )
        if jnp.asarray(resolved).shape != ():
            raise ValueError("tolerance must be scalar or None.")
        return (
            jnp.all(jnp.isfinite(coordinates), axis=-1)
            & jnp.all(coordinates >= -resolved, axis=-1)
            & jnp.all(coordinates <= 1.0 + resolved, axis=-1)
            & self.evidence.successful
        )

    def reference_to_physical(self, barycentric: ArrayLike, /) -> Array:
        """Map barycentric coordinates to physical points."""
        coordinates = jnp.asarray(barycentric, dtype=self.vertices.dtype)
        if coordinates.shape[-1:] != (self.intrinsic_dimension + 1,):
            raise ValueError("barycentric coordinates have the wrong trailing size.")
        return ein.contract("...v,...va->...a", coordinates, self.vertices)

    def physical_gradient(self, nodal_values: ArrayLike, /) -> Array:
        """Return the physical gradient of one scalar affine nodal field."""
        values = jnp.asarray(nodal_values)
        if values.shape[-1:] != (self.intrinsic_dimension + 1,):
            raise ValueError("nodal_values must end in the simplex vertex count.")
        return ein.contract("...v,...va->...a", values, self.barycentric_gradients)

    def physical_gradient_at(
        self,
        nodal_values: ArrayLike,
        simplex_indices: ArrayLike,
        /,
    ) -> Array:
        """Return gradients for fields on selected simplices."""
        values = jnp.asarray(nodal_values)
        if values.shape[-1:] != (self.intrinsic_dimension + 1,):
            raise ValueError("nodal_values must end in the simplex vertex count.")
        indices = self._indices(simplex_indices)
        gradients = jnp.take(self.barycentric_gradients, indices, axis=-3)
        return ein.contract("...v,...va->...a", values, gradients)


__all__ = ["AffineSimplexEvidence", "AffineSimplexMap"]
