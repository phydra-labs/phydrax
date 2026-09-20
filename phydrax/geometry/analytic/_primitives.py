#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any
from uuid import uuid4

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Key

from ..._polynomial._cubature import CubatureReference
from .._atlas import (
    BoundaryAtlas,
    box_boundary_atlas,
    circle_boundary_atlas,
    sphere_boundary_atlas,
)
from .._capabilities import GeometryCapability
from .._certificate import exact_signed_distance_certificate, FieldCertificate
from .._closest_point import box_closest_point, radial_closest_point
from .._contracts import (
    ContactCurvatureResult,
    GeometryKernel,
    GeometryKind,
    GeometrySource,
)
from .._cubature import AbstractCubatureMap, CubatureAtlas, CubatureComponent
from .._sampling import (
    complete_sampling_result,
    RejectionSamplingPlan,
    SamplingResult,
)
from ..design._schema import (
    _ParameterCollector,
    DesignState,
    ParameterBinding,
    ParameterId,
)


_REGION_CAPABILITIES = frozenset(
    {
        GeometryCapability.REGION_QUERY,
        GeometryCapability.SIGNED_DISTANCE,
        GeometryCapability.CLOSEST_POINT,
        GeometryCapability.BOUNDARY_NORMAL,
        GeometryCapability.INTERIOR_MEASURE,
        GeometryCapability.BOUNDARY_MEASURE,
        GeometryCapability.INTERIOR_SAMPLING,
        GeometryCapability.BOUNDARY_SAMPLING,
    }
)
_ANALYTIC_CAPABILITIES = _REGION_CAPABILITIES | frozenset(
    {GeometryCapability.BOUNDARY_ATLAS}
)
_RADIAL_CAPABILITIES = _ANALYTIC_CAPABILITIES | frozenset(
    {
        GeometryCapability.CONTACT_CURVATURE,
        GeometryCapability.CUBATURE_ATLAS,
    }
)


def _feature_id(value: str | None, prefix: str) -> str:
    if value is None:
        return f"{prefix}-{uuid4().hex}"
    if not value:
        raise ValueError("feature_id must be non-empty.")
    return value


def _validate_vector(value: Any, dimension: int, *, name: str) -> Array:
    host = np.asarray(value, dtype=np.float64)
    if host.shape != (dimension,):
        raise ValueError(f"{name} must have shape ({dimension},), got {host.shape}.")
    if not np.all(np.isfinite(host)):
        raise ValueError(f"{name} must contain only finite values.")
    return jnp.asarray(host, dtype=jnp.float64)


def _validate_nonempty_vector(value: Any, *, name: str) -> Array:
    host = np.asarray(value, dtype=np.float64)
    if host.ndim != 1 or host.size == 0:
        raise ValueError(f"{name} must be a non-empty vector, got shape {host.shape}.")
    if not np.all(np.isfinite(host)):
        raise ValueError(f"{name} must contain only finite values.")
    return jnp.asarray(host, dtype=jnp.float64)


def _validate_positive_scalar(value: Any, *, name: str) -> Array:
    host = np.asarray(value, dtype=np.float64)
    if host.shape != ():
        raise ValueError(f"{name} must be scalar, got shape {host.shape}.")
    scalar = float(host)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return jnp.asarray(scalar, dtype=jnp.float64)


def _check_points(points: Array, dimension: int) -> Array:
    points_ = jnp.asarray(points, dtype=jnp.float64)
    if points_.ndim == 0 or points_.shape[-1] != dimension:
        raise ValueError(f"points must have trailing dimension {dimension}.")
    return points_


class _RadialCubatureMap(AbstractCubatureMap):
    center: Array
    radius: Array
    reference: str = eqx.field(static=True)

    def __init__(
        self,
        center: Array,
        radius: Array,
        reference: CubatureReference,
    ):
        self.center = jnp.asarray(center, dtype=jnp.float64)
        self.radius = jnp.asarray(radius, dtype=jnp.float64).reshape(())
        self.reference = reference

    @property
    def num_charts(self) -> int:
        return 1

    @property
    def reference_domain(self) -> CubatureReference:
        return self.reference

    @property
    def ambient_dimension(self) -> int:
        return self.center.shape[0]

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        del chart_indices
        return self.center + self.radius * reference

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        del chart_indices
        intrinsic_dimension = {
            "circle": 1,
            "disk": 2,
            "sphere": 2,
            "ball": 3,
        }[self.reference]
        return jnp.broadcast_to(
            self.radius**intrinsic_dimension,
            reference.shape[:-1],
        )

    def reference_mask(self, chart_indices: Array, reference: Array, /) -> Array:
        del reference
        return jnp.ones(jnp.asarray(chart_indices).shape, dtype=jnp.bool_)

    def evaluate(
        self,
        chart_indices: Array,
        reference: Array,
        /,
    ):
        return super().evaluate(chart_indices, reference)


@jax.custom_jvp
def _finite_norm(value: Array) -> Array:
    """Euclidean norm with an exact primal and finite zero pseudoderivatives."""
    return jnp.sqrt(jnp.sum(value * value, axis=-1))


@_finite_norm.defjvp
def _finite_norm_jvp(primals, tangents):
    (value,) = primals
    (tangent,) = tangents
    norm = _finite_norm(value)
    nonzero = norm > 0.0
    denominator = jnp.where(nonzero, norm, jnp.ones_like(norm))
    directional = jnp.sum(value * tangent, axis=-1) / denominator
    return norm, jnp.where(nonzero, directional, jnp.zeros_like(directional))


def _compile_ball(
    context: _ParameterCollector,
    center_value: Array,
    radius_value: Array,
    *,
    source_id: str,
) -> GeometryKernel:
    center = context.bind(
        ParameterId(source_id, "center"),
        center_value,
        role="position",
    )
    radius = context.bind(
        ParameterId(source_id, "radius"),
        radius_value,
        role="length",
        physical_scale=float(radius_value),
        bounds=(0.0, None),
    )
    return _BallKernel(
        center,
        radius,
        dimension=center_value.shape[0],
        source_id=source_id,
    )


class Ball(GeometrySource):
    """Analytic solid ball in arbitrary positive dimension."""

    center: Array
    radius: Array
    feature_id: str = eqx.field(static=True)

    def __init__(
        self,
        center: Any,
        radius: Any,
        *,
        feature_id: str | None = None,
    ):
        self.center = _validate_nonempty_vector(center, name="center")
        self.radius = _validate_positive_scalar(radius, name="radius")
        self.feature_id = _feature_id(feature_id, "ball")

    def _compile(self, context: _ParameterCollector, /) -> GeometryKernel:
        return _compile_ball(
            context,
            self.center,
            self.radius,
            source_id=self.feature_id,
        )


class Circle(GeometrySource):
    """Analytic filled circle source."""

    center: Array
    radius: Array
    feature_id: str = eqx.field(static=True)

    def __init__(
        self,
        center: Any,
        radius: Any,
        *,
        feature_id: str | None = None,
    ):
        self.center = _validate_vector(center, 2, name="center")
        self.radius = _validate_positive_scalar(radius, name="radius")
        self.feature_id = _feature_id(feature_id, "circle")

    def _compile(self, context: _ParameterCollector, /) -> GeometryKernel:
        return _compile_ball(
            context,
            self.center,
            self.radius,
            source_id=self.feature_id,
        )


class Sphere(GeometrySource):
    """Analytic solid sphere source."""

    center: Array
    radius: Array
    feature_id: str = eqx.field(static=True)

    def __init__(
        self,
        center: Any,
        radius: Any,
        *,
        feature_id: str | None = None,
    ):
        self.center = _validate_vector(center, 3, name="center")
        self.radius = _validate_positive_scalar(radius, name="radius")
        self.feature_id = _feature_id(feature_id, "sphere")

    def _compile(self, context: _ParameterCollector, /) -> GeometryKernel:
        return _compile_ball(
            context,
            self.center,
            self.radius,
            source_id=self.feature_id,
        )


class _BallKernel(GeometryKernel):
    center: ParameterBinding = eqx.field(static=True)
    radius: ParameterBinding = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        center: ParameterBinding,
        radius: ParameterBinding,
        *,
        dimension: int,
        source_id: str,
    ):
        if dimension <= 0:
            raise ValueError("Ball dimension must be positive.")
        self.center = center
        self.radius = radius
        self.dimension = dimension
        self.source_id = source_id

    @property
    def ambient_dimension(self) -> int:
        return self.dimension

    @property
    def intrinsic_dimension(self) -> int:
        return self.dimension

    @property
    def kind(self) -> GeometryKind:
        return GeometryKind.REGION

    @property
    def capabilities(self) -> frozenset[GeometryCapability]:
        capabilities = set(_REGION_CAPABILITIES)
        if self.dimension in (2, 3):
            capabilities.update(
                {
                    GeometryCapability.CONTACT_CURVATURE,
                    GeometryCapability.BOUNDARY_ATLAS,
                    GeometryCapability.CUBATURE_ATLAS,
                }
            )
        return frozenset(capabilities)

    @property
    def field_certificate(self) -> FieldCertificate:
        return exact_signed_distance_certificate(smooth=False)

    def _parameters(self, state: DesignState) -> tuple[Array, Array]:
        return self.center.read(state), self.radius.read(state)

    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        points_ = _check_points(points, self.dimension)
        center, radius = self._parameters(state)
        return _finite_norm(points_ - center) - radius

    def contains(self, state: DesignState, points: Array, /) -> Array:
        return self.boundary_field(state, points) <= 0.0

    def boundary_normal(self, state: DesignState, points: Array, /) -> Array:
        points_ = _check_points(points, self.dimension)
        center, _ = self._parameters(state)
        direction = points_ - center
        norm = jnp.linalg.norm(direction, axis=-1, keepdims=True)
        return direction / jnp.maximum(norm, jnp.finfo(points_.dtype).eps)

    def closest_point(self, state: DesignState, points: Array, /):
        points_ = _check_points(points, self.dimension)
        center, radius = self._parameters(state)
        return radial_closest_point(
            points_,
            center,
            radius,
            represented_geometry_id=self.source_id,
        )

    def contact_curvature(
        self, state: DesignState, points: Array, /
    ) -> ContactCurvatureResult:
        if self.dimension not in (2, 3):
            raise NotImplementedError(
                "Contact curvature is available only for two- and three-dimensional balls."
            )
        points_ = _check_points(points, self.dimension)
        _, radius = self._parameters(state)
        count = points_.shape[0]
        curvature = jnp.broadcast_to(
            (1.0 / radius)[None, None],
            (count, self.dimension - 1),
        )
        valid = jnp.broadcast_to(jnp.isfinite(radius) & (radius > 0.0), (count,))
        margin = jnp.broadcast_to(radius, (count,))
        return ContactCurvatureResult(
            curvature,
            valid,
            margin,
            ambient_dimension=self.dimension,
        )

    def bounds(self, state: DesignState, /) -> Array:
        center, radius = self._parameters(state)
        return jnp.stack((center - radius, center + radius))

    def _unit_measure(self, dtype) -> Array:
        half_dimension = jnp.asarray(0.5 * self.dimension, dtype=dtype)
        return jnp.exp(
            half_dimension * jnp.log(jnp.asarray(jnp.pi, dtype=dtype))
            - jax.scipy.special.gammaln(half_dimension + 1.0)
        )

    def measure(self, state: DesignState, /) -> Array:
        _, radius = self._parameters(state)
        return self._unit_measure(radius.dtype) * radius**self.dimension

    def boundary_measure(self, state: DesignState, /) -> Array:
        _, radius = self._parameters(state)
        return (
            jnp.asarray(self.dimension, dtype=radius.dtype)
            * self._unit_measure(radius.dtype)
            * radius ** (self.dimension - 1)
        )

    def _directions(
        self,
        count: int,
        key: Key[Array, ""],
        *,
        dtype: jnp.dtype,
    ) -> Array:
        vectors = jr.normal(key, shape=(count, self.dimension), dtype=dtype)
        norms = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
        return vectors / jnp.maximum(norms, jnp.finfo(dtype).eps)

    def sample_interior(
        self,
        state: DesignState,
        num_points: int,
        /,
        *,
        key: Key[Array, ""],
        plan: RejectionSamplingPlan | None = None,
    ) -> SamplingResult:
        del plan
        center, radius = self._parameters(state)
        direction_key, radial_key = jr.split(key)
        count = int(num_points)
        directions = self._directions(count, direction_key, dtype=center.dtype)
        radial = radius * jr.uniform(
            radial_key,
            shape=(count,),
            dtype=center.dtype,
        ) ** (1.0 / self.dimension)
        return complete_sampling_result(center + radial[:, None] * directions)

    def sample_boundary(
        self,
        state: DesignState,
        num_points: int,
        /,
        *,
        key: Key[Array, ""],
    ) -> SamplingResult:
        center, radius = self._parameters(state)
        directions = self._directions(int(num_points), key, dtype=center.dtype)
        return complete_sampling_result(center + radius * directions)

    def boundary_atlas(self, state: DesignState, /) -> BoundaryAtlas:
        center, radius = self._parameters(state)
        match self.dimension:
            case 2:
                return circle_boundary_atlas(center, radius, source_id=self.source_id)
            case 3:
                return sphere_boundary_atlas(center, radius, source_id=self.source_id)
            case _:
                raise NotImplementedError(
                    "Boundary atlases are not provided for this ball dimension."
                )

    def cubature_atlas(
        self, state: DesignState, component: CubatureComponent, /
    ) -> CubatureAtlas:
        center, radius = self._parameters(state)
        match self.dimension, component:
            case 2, "interior":
                reference: CubatureReference = "disk"
            case 2, "boundary":
                reference = "circle"
            case 3, "interior":
                reference = "ball"
            case 3, "boundary":
                reference = "sphere"
            case _:
                raise NotImplementedError(
                    "Native cubature atlases are available only for two- and three-dimensional balls."
                )
        return CubatureAtlas(
            _RadialCubatureMap(center, radius, reference),
            source_entity_ids=jnp.asarray([0], dtype=jnp.int32),
            source_id=self.source_id,
            physical_tags=(component,),
        )


def _compile_orthotope(
    context: _ParameterCollector,
    center_value: Array,
    size_value: Array,
    *,
    source_id: str,
) -> GeometryKernel:
    center = context.bind(
        ParameterId(source_id, "center"),
        center_value,
        role="position",
    )
    size = context.bind(
        ParameterId(source_id, "size"),
        size_value,
        role="length",
        physical_scale=float(jnp.min(size_value)),
        bounds=(0.0, None),
    )
    return _OrthotopeKernel(
        center,
        size,
        dimension=center_value.shape[0],
        source_id=source_id,
    )


class Orthotope(GeometrySource):
    """Analytic axis-aligned region in arbitrary positive dimension."""

    center: Array
    size: Array
    feature_id: str = eqx.field(static=True)

    def __init__(
        self,
        center: Any,
        size: Any,
        *,
        feature_id: str | None = None,
    ):
        center_ = _validate_nonempty_vector(center, name="center")
        size_ = _validate_nonempty_vector(size, name="size")
        if size_.shape != center_.shape:
            raise ValueError("center and size must have the same shape.")
        if np.any(np.asarray(size_) <= 0.0):
            raise ValueError("size entries must be positive.")
        self.center = center_
        self.size = size_
        self.feature_id = _feature_id(feature_id, "orthotope")

    def _compile(self, context: _ParameterCollector, /) -> GeometryKernel:
        return _compile_orthotope(
            context,
            self.center,
            self.size,
            source_id=self.feature_id,
        )


class Box(GeometrySource):
    """Analytic axis-aligned solid box source."""

    center: Array
    size: Array
    feature_id: str = eqx.field(static=True)

    def __init__(
        self,
        center: Any,
        size: Any,
        *,
        feature_id: str | None = None,
    ):
        center_ = _validate_vector(center, 3, name="center")
        size_ = _validate_vector(size, 3, name="size")
        if np.any(np.asarray(size_) <= 0.0):
            raise ValueError("size entries must be positive.")
        self.center = center_
        self.size = size_
        self.feature_id = _feature_id(feature_id, "box")

    def _compile(self, context: _ParameterCollector, /) -> GeometryKernel:
        return _compile_orthotope(
            context,
            self.center,
            self.size,
            source_id=self.feature_id,
        )


class _OrthotopeKernel(GeometryKernel):
    center: ParameterBinding = eqx.field(static=True)
    size: ParameterBinding = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        center: ParameterBinding,
        size: ParameterBinding,
        *,
        dimension: int,
        source_id: str,
    ):
        if dimension <= 0:
            raise ValueError("Orthotope dimension must be positive.")
        self.center = center
        self.size = size
        self.dimension = dimension
        self.source_id = source_id

    @property
    def ambient_dimension(self) -> int:
        return self.dimension

    @property
    def intrinsic_dimension(self) -> int:
        return self.dimension

    @property
    def kind(self) -> GeometryKind:
        return GeometryKind.REGION

    @property
    def capabilities(self) -> frozenset[GeometryCapability]:
        capabilities = set(_REGION_CAPABILITIES)
        if self.dimension in (2, 3):
            capabilities.add(GeometryCapability.BOUNDARY_ATLAS)
        return frozenset(capabilities)

    @property
    def field_certificate(self) -> FieldCertificate:
        return exact_signed_distance_certificate(smooth=False)

    def _parameters(self, state: DesignState) -> tuple[Array, Array]:
        return self.center.read(state), self.size.read(state)

    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        points_ = _check_points(points, self.dimension)
        center, size = self._parameters(state)
        offset = jnp.abs(points_ - center) - 0.5 * size
        maximum = jnp.max(offset, axis=-1)
        outside = _finite_norm(jnp.maximum(offset, 0.0))
        return jnp.where(maximum <= 0.0, maximum, outside)

    def contains(self, state: DesignState, points: Array, /) -> Array:
        points_ = _check_points(points, self.dimension)
        center, size = self._parameters(state)
        return jnp.all(jnp.abs(points_ - center) <= 0.5 * size, axis=-1)

    def boundary_normal(self, state: DesignState, points: Array, /) -> Array:
        points_ = _check_points(points, self.dimension)
        center, size = self._parameters(state)
        relative = points_ - center
        half = 0.5 * size
        face_gap = jnp.abs(jnp.abs(relative) - half)
        minimum_gap = jnp.min(face_gap, axis=-1, keepdims=True)
        scale = jnp.max(size)
        tolerance = 32.0 * jnp.finfo(points_.dtype).eps * jnp.maximum(scale, 1.0)
        active = face_gap <= minimum_gap + tolerance
        normal = jnp.sign(relative) * active.astype(points_.dtype)
        norm = jnp.linalg.norm(normal, axis=-1, keepdims=True)
        return normal / jnp.maximum(norm, jnp.finfo(points_.dtype).eps)

    def closest_point(self, state: DesignState, points: Array, /):
        points_ = _check_points(points, self.dimension)
        center, size = self._parameters(state)
        return box_closest_point(
            points_,
            center,
            size,
            represented_geometry_id=self.source_id,
        )

    def bounds(self, state: DesignState, /) -> Array:
        center, size = self._parameters(state)
        half = 0.5 * size
        return jnp.stack((center - half, center + half))

    def measure(self, state: DesignState, /) -> Array:
        _, size = self._parameters(state)
        return jnp.prod(size)

    def boundary_measure(self, state: DesignState, /) -> Array:
        _, size = self._parameters(state)
        volume = jnp.prod(size)
        return 2.0 * jnp.sum(volume / size)

    def sample_interior(
        self,
        state: DesignState,
        num_points: int,
        /,
        *,
        key: Key[Array, ""],
        plan: RejectionSamplingPlan | None = None,
    ) -> SamplingResult:
        del plan
        bounds = self.bounds(state)
        points = jr.uniform(
            key,
            shape=(int(num_points), self.dimension),
            minval=bounds[0],
            maxval=bounds[1],
            dtype=bounds.dtype,
        )
        return complete_sampling_result(points)

    def sample_boundary(
        self,
        state: DesignState,
        num_points: int,
        /,
        *,
        key: Key[Array, ""],
    ) -> SamplingResult:
        count = int(num_points)
        face_key, coordinate_key = jr.split(key)
        center, size = self._parameters(state)
        half = 0.5 * size
        volume = jnp.prod(size)
        face_measures = jnp.repeat(volume / size, 2)
        face = jr.choice(
            face_key,
            2 * self.dimension,
            shape=(count,),
            p=face_measures / jnp.sum(face_measures),
        )
        points = jr.uniform(
            coordinate_key,
            shape=(count, self.dimension),
            minval=center - half,
            maxval=center + half,
            dtype=center.dtype,
        )
        axis = face // 2
        side = jnp.where(face % 2 == 0, -1.0, 1.0)
        boundary_coordinate = center[axis] + side * half[axis]
        points = points.at[jnp.arange(count), axis].set(boundary_coordinate)
        return complete_sampling_result(points)

    def boundary_atlas(self, state: DesignState, /) -> BoundaryAtlas:
        center, size = self._parameters(state)
        match self.dimension:
            case 2:
                from ._extended import _RectangleBoundaryMap

                return BoundaryAtlas(
                    _RectangleBoundaryMap(center, size),
                    physical_tags=("y_min", "x_max", "y_max", "x_min"),
                    source_entity_ids=jnp.arange(4, dtype=jnp.int32),
                    source_id=self.source_id,
                )
            case 3:
                return box_boundary_atlas(center, size, source_id=self.source_id)
            case _:
                raise NotImplementedError(
                    "Boundary atlases are not provided for this orthotope dimension."
                )


def Cube(
    center: Any,
    side: Any,
    *,
    feature_id: str | None = None,
) -> Box:
    """Construct a box with equal side lengths."""
    side_ = _validate_positive_scalar(side, name="side")
    return Box(center, jnp.repeat(side_[None], 3), feature_id=feature_id)


__all__ = ["Ball", "Box", "Circle", "Cube", "Orthotope", "Sphere"]
