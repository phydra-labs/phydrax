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
from .._contracts import GeometryKernel, GeometryKind, GeometrySource
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


_ANALYTIC_CAPABILITIES = frozenset(
    {
        GeometryCapability.REGION_QUERY,
        GeometryCapability.SIGNED_DISTANCE,
        GeometryCapability.BOUNDARY_NORMAL,
        GeometryCapability.MEASURE,
        GeometryCapability.INTERIOR_SAMPLING,
        GeometryCapability.BOUNDARY_SAMPLING,
        GeometryCapability.BOUNDARY_ATLAS,
    }
)
_RADIAL_CAPABILITIES = _ANALYTIC_CAPABILITIES | frozenset(
    {GeometryCapability.CUBATURE_ATLAS}
)


def _feature_id(value: str | None, prefix: str) -> str:
    if value is None:
        return f"{prefix}-{uuid4().hex}"
    if not value:
        raise ValueError("feature_id must be non-empty.")
    return value


def _validate_vector(value: Any, dimension: int, *, name: str) -> Array:
    host = np.asarray(value, dtype=float)
    if host.shape != (dimension,):
        raise ValueError(f"{name} must have shape ({dimension},), got {host.shape}.")
    if not np.all(np.isfinite(host)):
        raise ValueError(f"{name} must contain only finite values.")
    return jnp.asarray(host, dtype=float)


def _validate_positive_scalar(value: Any, *, name: str) -> Array:
    host = np.asarray(value, dtype=float)
    if host.shape != ():
        raise ValueError(f"{name} must be scalar, got shape {host.shape}.")
    scalar = float(host)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return jnp.asarray(scalar, dtype=float)


def _check_points(points: Array, dimension: int) -> Array:
    points_ = jnp.asarray(points, dtype=float)
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
        self.center = jnp.asarray(center, dtype=float)
        self.radius = jnp.asarray(radius, dtype=float).reshape(())
        self.reference = reference

    @property
    def num_charts(self) -> int:
        return 1

    @property
    def reference_domain(self) -> CubatureReference:
        return self.reference

    @property
    def ambient_dimension(self) -> int:
        return int(self.center.shape[0])

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
        return jnp.ones(jnp.asarray(chart_indices).shape, dtype=bool)


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
        center = context.bind(
            ParameterId(self.feature_id, "center"),
            self.center,
            role="position",
        )
        radius = context.bind(
            ParameterId(self.feature_id, "radius"),
            self.radius,
            role="length",
            physical_scale=float(self.radius),
            bounds=(0.0, None),
        )
        return _CircleKernel(center, radius, source_id=self.feature_id)


class _CircleKernel(GeometryKernel):
    center: ParameterBinding = eqx.field(static=True)
    radius: ParameterBinding = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        center: ParameterBinding,
        radius: ParameterBinding,
        *,
        source_id: str,
    ):
        self.center = center
        self.radius = radius
        self.source_id = source_id

    @property
    def ambient_dimension(self) -> int:
        return 2

    @property
    def intrinsic_dimension(self) -> int:
        return 2

    @property
    def kind(self) -> GeometryKind:
        return GeometryKind.REGION

    @property
    def capabilities(self) -> frozenset[GeometryCapability]:
        return _RADIAL_CAPABILITIES

    @property
    def field_certificate(self) -> FieldCertificate:
        return exact_signed_distance_certificate(smooth=False)

    def _parameters(self, state: DesignState) -> tuple[Array, Array]:
        return self.center.read(state), self.radius.read(state)

    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        points_ = _check_points(points, 2)
        center, radius = self._parameters(state)
        return _finite_norm(points_ - center) - radius

    def contains(self, state: DesignState, points: Array, /) -> Array:
        return self.boundary_field(state, points) <= 0.0

    def boundary_normal(self, state: DesignState, points: Array, /) -> Array:
        points_ = _check_points(points, 2)
        center, _ = self._parameters(state)
        direction = points_ - center
        norm = jnp.linalg.norm(direction, axis=-1, keepdims=True)
        return direction / jnp.maximum(norm, jnp.finfo(points_.dtype).eps)

    def bounds(self, state: DesignState, /) -> Array:
        center, radius = self._parameters(state)
        return jnp.stack((center - radius, center + radius))

    def measure(self, state: DesignState, /) -> Array:
        _, radius = self._parameters(state)
        return jnp.pi * radius**2

    def boundary_measure(self, state: DesignState, /) -> Array:
        _, radius = self._parameters(state)
        return 2.0 * jnp.pi * radius

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
        radial_key, angular_key = jr.split(key)
        radial = radius * jnp.sqrt(
            jr.uniform(radial_key, shape=(int(num_points),), dtype=center.dtype)
        )
        angle = (
            2.0
            * jnp.pi
            * jr.uniform(
                angular_key,
                shape=(int(num_points),),
                dtype=center.dtype,
            )
        )
        points = center + radial[:, None] * jnp.stack(
            (jnp.cos(angle), jnp.sin(angle)), axis=-1
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
        center, radius = self._parameters(state)
        angle = (
            2.0
            * jnp.pi
            * jr.uniform(
                key,
                shape=(int(num_points),),
                dtype=center.dtype,
            )
        )
        points = center + radius * jnp.stack((jnp.cos(angle), jnp.sin(angle)), axis=-1)
        return complete_sampling_result(points)

    def boundary_atlas(self, state: DesignState, /) -> BoundaryAtlas:
        center, radius = self._parameters(state)
        return circle_boundary_atlas(center, radius, source_id=self.source_id)

    def cubature_atlas(
        self, state: DesignState, component: CubatureComponent, /
    ) -> CubatureAtlas:
        center, radius = self._parameters(state)
        reference: CubatureReference = "disk" if component == "interior" else "circle"
        return CubatureAtlas(
            _RadialCubatureMap(center, radius, reference),
            source_entity_ids=jnp.asarray([0], dtype=jnp.int32),
            source_id=self.source_id,
            physical_tags=(component,),
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
        center = context.bind(
            ParameterId(self.feature_id, "center"),
            self.center,
            role="position",
        )
        radius = context.bind(
            ParameterId(self.feature_id, "radius"),
            self.radius,
            role="length",
            physical_scale=float(self.radius),
            bounds=(0.0, None),
        )
        return _SphereKernel(center, radius, source_id=self.feature_id)


class _SphereKernel(GeometryKernel):
    center: ParameterBinding = eqx.field(static=True)
    radius: ParameterBinding = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        center: ParameterBinding,
        radius: ParameterBinding,
        *,
        source_id: str,
    ):
        self.center = center
        self.radius = radius
        self.source_id = source_id

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def intrinsic_dimension(self) -> int:
        return 3

    @property
    def kind(self) -> GeometryKind:
        return GeometryKind.REGION

    @property
    def capabilities(self) -> frozenset[GeometryCapability]:
        return _RADIAL_CAPABILITIES

    @property
    def field_certificate(self) -> FieldCertificate:
        return exact_signed_distance_certificate(smooth=False)

    def _parameters(self, state: DesignState) -> tuple[Array, Array]:
        return self.center.read(state), self.radius.read(state)

    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        points_ = _check_points(points, 3)
        center, radius = self._parameters(state)
        return _finite_norm(points_ - center) - radius

    def contains(self, state: DesignState, points: Array, /) -> Array:
        return self.boundary_field(state, points) <= 0.0

    def boundary_normal(self, state: DesignState, points: Array, /) -> Array:
        points_ = _check_points(points, 3)
        center, _ = self._parameters(state)
        direction = points_ - center
        norm = jnp.linalg.norm(direction, axis=-1, keepdims=True)
        return direction / jnp.maximum(norm, jnp.finfo(points_.dtype).eps)

    def bounds(self, state: DesignState, /) -> Array:
        center, radius = self._parameters(state)
        return jnp.stack((center - radius, center + radius))

    def measure(self, state: DesignState, /) -> Array:
        _, radius = self._parameters(state)
        return (4.0 / 3.0) * jnp.pi * radius**3

    def boundary_measure(self, state: DesignState, /) -> Array:
        _, radius = self._parameters(state)
        return 4.0 * jnp.pi * radius**2

    def _directions(
        self,
        count: int,
        key: Key[Array, ""],
        *,
        dtype: jnp.dtype,
    ) -> Array:
        vectors = jr.normal(key, shape=(count, 3), dtype=dtype)
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
        ) ** (1.0 / 3.0)
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
        return sphere_boundary_atlas(center, radius, source_id=self.source_id)

    def cubature_atlas(
        self, state: DesignState, component: CubatureComponent, /
    ) -> CubatureAtlas:
        center, radius = self._parameters(state)
        reference: CubatureReference = "ball" if component == "interior" else "sphere"
        return CubatureAtlas(
            _RadialCubatureMap(center, radius, reference),
            source_entity_ids=jnp.asarray([0], dtype=jnp.int32),
            source_id=self.source_id,
            physical_tags=(component,),
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
        center = context.bind(
            ParameterId(self.feature_id, "center"),
            self.center,
            role="position",
        )
        size = context.bind(
            ParameterId(self.feature_id, "size"),
            self.size,
            role="length",
            physical_scale=float(jnp.min(self.size)),
            bounds=(0.0, None),
        )
        return _BoxKernel(center, size, source_id=self.feature_id)


class _BoxKernel(GeometryKernel):
    center: ParameterBinding = eqx.field(static=True)
    size: ParameterBinding = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        center: ParameterBinding,
        size: ParameterBinding,
        *,
        source_id: str,
    ):
        self.center = center
        self.size = size
        self.source_id = source_id

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def intrinsic_dimension(self) -> int:
        return 3

    @property
    def kind(self) -> GeometryKind:
        return GeometryKind.REGION

    @property
    def capabilities(self) -> frozenset[GeometryCapability]:
        return _ANALYTIC_CAPABILITIES

    @property
    def field_certificate(self) -> FieldCertificate:
        return exact_signed_distance_certificate(smooth=False)

    def _parameters(self, state: DesignState) -> tuple[Array, Array]:
        return self.center.read(state), self.size.read(state)

    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        points_ = _check_points(points, 3)
        center, size = self._parameters(state)
        offset = jnp.abs(points_ - center) - 0.5 * size
        maximum = jnp.max(offset, axis=-1)
        outside = _finite_norm(jnp.maximum(offset, 0.0))
        return jnp.where(maximum <= 0.0, maximum, outside)

    def contains(self, state: DesignState, points: Array, /) -> Array:
        points_ = _check_points(points, 3)
        center, size = self._parameters(state)
        return jnp.all(jnp.abs(points_ - center) <= 0.5 * size, axis=-1)

    def boundary_normal(self, state: DesignState, points: Array, /) -> Array:
        points_ = _check_points(points, 3)
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

    def bounds(self, state: DesignState, /) -> Array:
        center, size = self._parameters(state)
        half = 0.5 * size
        return jnp.stack((center - half, center + half))

    def measure(self, state: DesignState, /) -> Array:
        _, size = self._parameters(state)
        return jnp.prod(size)

    def boundary_measure(self, state: DesignState, /) -> Array:
        _, size = self._parameters(state)
        x, y, z = size
        return 2.0 * (x * y + x * z + y * z)

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
            shape=(int(num_points), 3),
            minval=bounds[0],
            maxval=bounds[1],
            dtype=bounds.dtype,
        )
        return complete_sampling_result(points)

    def _face_data(self, state: DesignState) -> tuple[Array, Array, Array, Array]:
        center, size = self._parameters(state)
        half = 0.5 * size
        hx, hy, hz = half
        dx, dy, dz = size
        centers = center + jnp.asarray(
            [
                [-hx, 0.0, 0.0],
                [hx, 0.0, 0.0],
                [0.0, -hy, 0.0],
                [0.0, hy, 0.0],
                [0.0, 0.0, -hz],
                [0.0, 0.0, hz],
            ]
        )
        first = jnp.asarray(
            [
                [0.0, hy, 0.0],
                [0.0, hy, 0.0],
                [hx, 0.0, 0.0],
                [hx, 0.0, 0.0],
                [hx, 0.0, 0.0],
                [hx, 0.0, 0.0],
            ]
        )
        second = jnp.asarray(
            [
                [0.0, 0.0, hz],
                [0.0, 0.0, hz],
                [0.0, 0.0, hz],
                [0.0, 0.0, hz],
                [0.0, hy, 0.0],
                [0.0, hy, 0.0],
            ]
        )
        areas = jnp.asarray([dy * dz, dy * dz, dx * dz, dx * dz, dx * dy, dx * dy])
        return centers, first, second, areas

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
        centers, first, second, areas = self._face_data(state)
        face = jr.choice(face_key, 6, shape=(count,), p=areas / jnp.sum(areas))
        coordinates = jr.uniform(
            coordinate_key,
            shape=(count, 2),
            minval=-1.0,
            maxval=1.0,
            dtype=centers.dtype,
        )
        points = (
            centers[face]
            + coordinates[:, :1] * first[face]
            + coordinates[:, 1:] * second[face]
        )
        return complete_sampling_result(points)

    def boundary_atlas(self, state: DesignState, /) -> BoundaryAtlas:
        center, size = self._parameters(state)
        return box_boundary_atlas(center, size, source_id=self.source_id)


def Cube(
    center: Any,
    side: Any,
    *,
    feature_id: str | None = None,
) -> Box:
    """Construct a box with equal side lengths."""
    side_ = _validate_positive_scalar(side, name="side")
    return Box(center, jnp.repeat(side_[None], 3), feature_id=feature_id)


__all__ = ["Box", "Circle", "Cube", "Sphere"]
