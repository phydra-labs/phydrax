#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, Key

from .._doc import DOC_KEY0
from ..geometry import (
    BoundaryAtlas,
    bounded_rejection_sample,
    CompiledGeometry,
    CubatureAtlas,
    CubatureComponent,
    DistanceSemantics,
    GeometryCapability,
    GeometryKind,
    ReconstructionReport,
    ReconstructionReportProvider,
    RejectionSamplingPlan,
    SamplingResult,
)
from ..geometry._sampling import require_complete
from ._base import _make_compact_boundary_factor, AbstractGeometry


class GeometryDomain(AbstractGeometry):
    """Domain adapter around one JAX-safe compiled geometry."""

    geometry: CompiledGeometry
    _label: str
    adf: Callable[[Array], Array]

    def __init__(self, geometry: CompiledGeometry, *, label: str = "x"):
        if not isinstance(geometry, CompiledGeometry):
            raise TypeError("GeometryDomain requires a CompiledGeometry.")
        if geometry.kind is not GeometryKind.REGION:
            raise ValueError("GeometryDomain currently adapts region kernels only.")
        if not isinstance(label, str) or not label:
            raise ValueError("GeometryDomain label must be a non-empty string.")
        self.geometry = geometry
        self._label = label

    @property
    def compiled(self) -> CompiledGeometry:
        return self.geometry

    @property
    def reconstruction_report(self) -> ReconstructionReport:
        kernel = self.geometry.kernel
        if not isinstance(kernel, ReconstructionReportProvider):
            raise AttributeError("This geometry was not produced by reconstruction.")
        return kernel.report

    @property
    def spatial_dim(self) -> int:
        return self.geometry.ambient_dimension

    @property
    def bounds(self) -> Array:
        return self.geometry.bounds

    @property
    def volume(self) -> Array:
        return self.geometry.measure

    @property
    def measure(self) -> Array:
        return self.geometry.measure

    @property
    def area(self) -> Array:
        if self.spatial_dim != 2:
            raise AttributeError("area is defined only for a two-dimensional region.")
        return self.geometry.measure

    @property
    def boundary_measure_value(self) -> Array:
        return self.geometry.boundary_measure

    @property
    def boundary_measure(self) -> Array:
        return self.geometry.boundary_measure

    @property
    def boundary_length_value(self) -> Array:
        if self.spatial_dim != 2:
            raise AttributeError(
                "boundary_length_value is defined only for a two-dimensional region."
            )
        return self.geometry.boundary_measure

    @property
    def surface_area_value(self) -> Array:
        if self.spatial_dim != 3:
            raise AttributeError(
                "surface_area_value is defined only for a three-dimensional region."
            )
        return self.geometry.boundary_measure

    @property
    def volume_proportion(self) -> Array:
        bounds = jnp.asarray(self.bounds, dtype=float)
        bounding_measure = jnp.prod(bounds[1] - bounds[0])
        return self.volume / bounding_measure

    @property
    def boundary_atlas(self) -> BoundaryAtlas:
        return self.geometry.boundary_atlas

    def cubature_atlas(self, component: CubatureComponent, /) -> CubatureAtlas:
        return self.geometry.cubature_atlas(component)

    def adf(self, points: Array, /) -> Array:
        """Evaluate the certified negative-inside boundary field."""
        points_ = jnp.asarray(points, dtype=float)
        if points_.ndim == 0:
            points_ = jnp.repeat(points_[None], self.spatial_dim)
        return self.geometry.boundary_field(points_)

    @property
    def boundary_ansatz_factor(self) -> Callable[[Array], Array]:
        """Return a compact dimensional factor with outward unit boundary jet."""
        source: Callable[[Array], Array]
        if (
            self.geometry.field_certificate.distance_semantics
            is not DistanceSemantics.LEVEL_SET
        ):
            source = self.adf

        def normalized_level_set(points: Array) -> Array:
            points_ = jnp.asarray(points, dtype=float)
            leading = points_.shape[:-1]
            flat = points_.reshape((-1, self.spatial_dim))

            def normalize(point):
                value, gradient = jax.value_and_grad(self.adf)(point)
                magnitude = jnp.linalg.norm(gradient)
                denominator = jnp.where(
                    magnitude > jnp.sqrt(jnp.finfo(point.dtype).eps),
                    magnitude,
                    jnp.asarray(1.0, dtype=point.dtype),
                )
                return value / denominator

            return jax.vmap(normalize)(flat).reshape(leading)

        if (
            self.geometry.field_certificate.distance_semantics
            is DistanceSemantics.LEVEL_SET
        ):
            source = normalized_level_set

        return _make_compact_boundary_factor(
            source,
            scale=self.enforcement_characteristic_length,
            saturation_fraction=0.5,
            linear_fraction=0.5,
        )

    def _same_factor_support(self, other: object, /) -> bool:
        return isinstance(other, GeometryDomain) and self.geometry.equivalent(
            other.geometry
        )

    def sample_interior_result(
        self,
        num_points: int,
        *,
        where: Callable | None = None,
        key: Key[Array, ""] = DOC_KEY0,
        plan: RejectionSamplingPlan | None = None,
    ) -> SamplingResult:
        count = int(num_points)
        if count < 0:
            raise ValueError("num_points must be non-negative.")
        if where is None:
            return self.geometry.sample_interior(count, key=key, plan=plan)

        bounds = jnp.asarray(self.bounds, dtype=float)
        plan_ = RejectionSamplingPlan() if plan is None else plan

        def proposal(proposal_key, proposal_count):
            return jr.uniform(
                proposal_key,
                shape=(proposal_count, self.spatial_dim),
                minval=bounds[0],
                maxval=bounds[1],
                dtype=bounds.dtype,
            )

        def accept(points):
            return self._contains(points) & jax.vmap(where)(points)

        return bounded_rejection_sample(
            proposal,
            accept,
            num_points=count,
            point_dimension=self.spatial_dim,
            key=key,
            plan=plan_,
            dtype=bounds.dtype,
        )

    def sample_boundary_result(
        self,
        num_points: int,
        *,
        where: Callable | None = None,
        key: Key[Array, ""] = DOC_KEY0,
        plan: RejectionSamplingPlan | None = None,
    ) -> SamplingResult:
        count = int(num_points)
        if count < 0:
            raise ValueError("num_points must be non-negative.")
        if where is None:
            return self.geometry.sample_boundary(count, key=key)
        plan_ = RejectionSamplingPlan() if plan is None else plan

        def proposal(proposal_key, proposal_count):
            return self.geometry.sample_boundary(
                proposal_count,
                key=proposal_key,
            ).points

        def accept(points):
            return jax.vmap(where)(points)

        return bounded_rejection_sample(
            proposal,
            accept,
            num_points=count,
            point_dimension=self.spatial_dim,
            key=key,
            plan=plan_,
            dtype=jnp.asarray(self.bounds).dtype,
        )

    def sample_interior(
        self,
        num_points: int,
        *,
        where: Callable | None = None,
        sampler: str = "latin_hypercube",
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        del sampler
        result = self.sample_interior_result(num_points, where=where, key=key)
        return require_complete(result, context="Geometry interior sampling")

    def sample_boundary(
        self,
        num_points: int,
        *,
        where: Callable | None = None,
        sampler: str = "latin_hypercube",
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        del sampler
        result = self.sample_boundary_result(num_points, where=where, key=key)
        return require_complete(result, context="Geometry boundary sampling")

    def estimate_boundary_subset_measure(
        self,
        where: Callable[[Array], Bool[Array, ""]],
        *,
        num_samples: int = 4096,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        points = self.sample_boundary(int(num_samples), key=key)
        selected = jnp.asarray(jax.vmap(where)(points), dtype=float)
        return self.boundary_measure_value * jnp.mean(selected)

    def _sample_interior_separable(
        self,
        num_points: int | Sequence[int],
        *,
        sampler: str = "latin_hypercube",
        where: Callable | None = None,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> tuple[tuple[Array, ...], Array]:
        if isinstance(num_points, int):
            counts = (int(num_points),) * self.spatial_dim
        else:
            counts = tuple(int(count) for count in num_points)
        if len(counts) != self.spatial_dim:
            raise ValueError(
                f"num_points must contain {self.spatial_dim} coordinate counts."
            )
        if any(count < 0 for count in counts):
            raise ValueError("Coordinate sample counts must be non-negative.")

        bounds = jnp.asarray(self.bounds, dtype=float)
        keys = jr.split(key, self.spatial_dim)
        coordinates = []
        for axis, (count, axis_key) in enumerate(zip(counts, keys, strict=True)):
            if sampler == "latin_hypercube":
                unit = (
                    jr.permutation(axis_key, jnp.arange(count, dtype=float)) + 0.5
                ) / jnp.maximum(count, 1)
            else:
                unit = jr.uniform(axis_key, shape=(count,), dtype=bounds.dtype)
            coordinates.append(
                bounds[0, axis] + unit * (bounds[1, axis] - bounds[0, axis])
            )
        coordinates_ = tuple(coordinates)
        mesh = jnp.meshgrid(*coordinates_, indexing="ij")
        grid = jnp.stack(mesh, axis=-1)
        flat = grid.reshape((-1, self.spatial_dim))
        mask = self._contains(flat)
        if where is not None:
            mask = mask & jax.vmap(where)(flat)
        return coordinates_, mask.reshape(grid.shape[:-1])

    def _contains(self, points: Array) -> Array:
        return jnp.asarray(self.geometry.contains(points), dtype=bool)

    def _on_boundary(self, points: Array) -> Array:
        bounds = jnp.asarray(self.bounds, dtype=float)
        scale = jnp.max(bounds[1] - bounds[0])
        tolerance = self.geometry.tolerance.threshold(scale)
        return jnp.abs(self.geometry.boundary_field(points)) <= tolerance

    def _boundary_normals(self, points: Array) -> Array:
        return self.geometry.boundary_normal(points)

    @property
    def field_certificate(self):
        return self.geometry.field_certificate

    def has_geometry_capability(self, capability: GeometryCapability, /) -> bool:
        return self.geometry.has_capability(capability)


__all__ = ["GeometryDomain"]
