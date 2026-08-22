#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Key
from shapely.geometry import Polygon as ShapelyPolygon

from ..._numerics._quadrature_rules import gauss_legendre_data
from .._atlas import AbstractBoundaryMap, BoundaryAtlas
from .._capabilities import GeometryCapability
from .._certificate import (
    DistanceSemantics,
    exact_signed_distance_certificate,
    FieldCertificate,
    FieldRegularity,
    SignReliability,
    ZeroSetAccuracy,
)
from .._contracts import GeometryKernel, GeometryKind, GeometrySource
from .._sampling import (
    bounded_rejection_sample,
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
from ._primitives import (
    _check_points,
    _feature_id,
    _finite_norm,
    _validate_positive_scalar,
    _validate_vector,
)


_REGION_CAPABILITIES = frozenset(
    {
        GeometryCapability.REGION_QUERY,
        GeometryCapability.BOUNDARY_NORMAL,
        GeometryCapability.MEASURE,
        GeometryCapability.INTERIOR_SAMPLING,
        GeometryCapability.BOUNDARY_SAMPLING,
        GeometryCapability.BOUNDARY_ATLAS,
    }
)
_EXACT_REGION_CAPABILITIES = _REGION_CAPABILITIES | {GeometryCapability.SIGNED_DISTANCE}
_LEVEL_SET_CERTIFICATE = FieldCertificate(
    zero_set_accuracy=ZeroSetAccuracy.EXACT,
    sign_reliability=SignReliability.RELIABLE,
    distance_semantics=DistanceSemantics.LEVEL_SET,
    regularity=FieldRegularity.PIECEWISE_SMOOTH,
    safe_step_factor=None,
    validity_region="all_space",
    parameter_differentiable=True,
    provenance=("analytic_level_set",),
)
_TWO_PI = 2.0 * jnp.pi
_GL_RULE = gauss_legendre_data(48)
_GL_NODES = jnp.asarray(_GL_RULE.nodes, dtype=float)
_GL_WEIGHTS = jnp.asarray(_GL_RULE.weights, dtype=float)


def _validate_positive_vector(value: Any, dimension: int, *, name: str) -> Array:
    result = _validate_vector(value, dimension, name=name)
    if np.any(np.asarray(result) <= 0.0):
        raise ValueError(f"{name} entries must be positive.")
    return result


def _validate_angle(value: Any, *, name: str = "angle") -> Array:
    host = np.asarray(value, dtype=float)
    if host.shape != () or not np.isfinite(host):
        raise ValueError(f"{name} must be a finite scalar.")
    scalar = float(host)
    if scalar <= 0.0 or scalar > 2.0 * math.pi:
        raise ValueError(f"{name} must be in (0, 2π].")
    return jnp.asarray(scalar, dtype=float)


def _normal_from_field(
    kernel: GeometryKernel, state: DesignState, points: Array
) -> Array:
    points_ = jnp.asarray(points, dtype=float)
    shape = points_.shape
    flat = points_.reshape((-1, shape[-1]))
    gradient = jax.vmap(jax.grad(lambda point: kernel.boundary_field(state, point)))(flat)
    norm = jnp.linalg.norm(gradient, axis=-1, keepdims=True)
    unit = gradient / jnp.maximum(norm, jnp.finfo(points_.dtype).eps)
    return unit.reshape(shape)


def _uniform_in_bounds(
    kernel: GeometryKernel,
    state: DesignState,
    count: int,
    key: Key[Array, ""],
    plan: RejectionSamplingPlan | None,
) -> SamplingResult:
    bounds = kernel.bounds(state)
    plan_ = RejectionSamplingPlan() if plan is None else plan

    def proposal(proposal_key, proposal_count):
        return jr.uniform(
            proposal_key,
            shape=(proposal_count, kernel.ambient_dimension),
            minval=bounds[0],
            maxval=bounds[1],
            dtype=bounds.dtype,
        )

    return bounded_rejection_sample(
        proposal,
        lambda points: kernel.contains(state, points),
        num_points=count,
        point_dimension=kernel.ambient_dimension,
        key=key,
        plan=plan_,
        dtype=bounds.dtype,
    )


def _axis_frame(axis: Array) -> tuple[Array, Array, Array, Array]:
    height = _finite_norm(axis)
    direction = axis / jnp.maximum(height, jnp.finfo(axis.dtype).eps)
    helper = jnp.where(
        jnp.abs(direction[0]) < 0.875,
        jnp.asarray([1.0, 0.0, 0.0], dtype=axis.dtype),
        jnp.asarray([0.0, 1.0, 0.0], dtype=axis.dtype),
    )
    first = jnp.cross(direction, helper)
    first = first / jnp.maximum(jnp.linalg.norm(first), jnp.finfo(axis.dtype).eps)
    second = jnp.cross(direction, first)
    return first, second, direction, height


def _axial_coordinates(
    points: Array, base: Array, axis: Array
) -> tuple[Array, Array, Array, Array, Array]:
    first, second, direction, height = _axis_frame(axis)
    relative = points - base
    x = jnp.sum(relative * first, axis=-1)
    y = jnp.sum(relative * second, axis=-1)
    z = jnp.sum(relative * direction, axis=-1)
    return x, y, z, height, jnp.stack((first, second, direction))


def _angle_membership(x: Array, y: Array, angle: Array) -> tuple[Array, Array]:
    theta = jnp.mod(jnp.arctan2(y, x), _TWO_PI)
    inside = theta <= angle
    radial = jnp.sqrt(x * x + y * y)
    distance0 = jnp.abs(y)
    distance1 = jnp.abs(x * jnp.sin(angle) - y * jnp.cos(angle))
    edge_distance = jnp.minimum(distance0, distance1)
    signed = jnp.where(inside, -edge_distance, edge_distance)
    return inside, jnp.where(radial > 0.0, signed, -jnp.zeros_like(signed))


class _EllipseBoundaryMap(AbstractBoundaryMap):
    center: Array
    radii: Array

    def __init__(self, center: Array, radii: Array):
        self.center = center
        self.radii = radii

    @property
    def num_charts(self) -> int:
        return 4

    @property
    def reference_dimension(self) -> int:
        return 1

    @property
    def ambient_dimension(self) -> int:
        return 2

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        angle = 0.5 * jnp.pi * (chart_indices.astype(reference.dtype) + reference[..., 0])
        return self.center + self.radii * jnp.stack(
            (jnp.cos(angle), jnp.sin(angle)), axis=-1
        )

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        angle = 0.5 * jnp.pi * (chart_indices.astype(reference.dtype) + reference[..., 0])
        speed = jnp.sqrt(
            (self.radii[0] * jnp.sin(angle)) ** 2 + (self.radii[1] * jnp.cos(angle)) ** 2
        )
        return 0.5 * jnp.pi * speed


class Ellipse(GeometrySource):
    """Axis-aligned filled ellipse with an exact signed zero set."""

    center: Array
    radii: Array
    feature_id: str = eqx.field(static=True)

    def __init__(
        self,
        center: Any,
        radii: Any,
        *,
        feature_id: str | None = None,
    ):
        self.center = _validate_vector(center, 2, name="center")
        self.radii = _validate_positive_vector(radii, 2, name="radii")
        self.feature_id = _feature_id(feature_id, "ellipse")

    def _compile(self, context: _ParameterCollector, /) -> GeometryKernel:
        center = context.bind(
            ParameterId(self.feature_id, "center"), self.center, role="position"
        )
        radii = context.bind(
            ParameterId(self.feature_id, "radii"),
            self.radii,
            role="length",
            physical_scale=float(jnp.min(self.radii)),
            bounds=(0.0, None),
        )
        return _EllipseKernel(center, radii, source_id=self.feature_id)


class _EllipseKernel(GeometryKernel):
    center: ParameterBinding = eqx.field(static=True)
    radii: ParameterBinding = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self, center: ParameterBinding, radii: ParameterBinding, *, source_id: str
    ):
        self.center = center
        self.radii = radii
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
        return _REGION_CAPABILITIES

    @property
    def field_certificate(self) -> FieldCertificate:
        return _LEVEL_SET_CERTIFICATE

    def _parameters(self, state: DesignState) -> tuple[Array, Array]:
        return self.center.read(state), self.radii.read(state)

    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        points_ = _check_points(points, 2)
        center, radii = self._parameters(state)
        return (_finite_norm((points_ - center) / radii) - 1.0) * jnp.min(radii)

    def contains(self, state: DesignState, points: Array, /) -> Array:
        return self.boundary_field(state, points) <= 0.0

    def boundary_normal(self, state: DesignState, points: Array, /) -> Array:
        return _normal_from_field(self, state, points)

    def bounds(self, state: DesignState, /) -> Array:
        center, radii = self._parameters(state)
        return jnp.stack((center - radii, center + radii))

    def measure(self, state: DesignState, /) -> Array:
        _, radii = self._parameters(state)
        return jnp.pi * jnp.prod(radii)

    def boundary_measure(self, state: DesignState, /) -> Array:
        _, radii = self._parameters(state)
        theta = jnp.pi * (_GL_NODES + 1.0)
        speed = jnp.sqrt(
            (radii[0] * jnp.sin(theta)) ** 2 + (radii[1] * jnp.cos(theta)) ** 2
        )
        return jnp.pi * jnp.sum(_GL_WEIGHTS * speed)

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
        center, radii = self._parameters(state)
        radial_key, angle_key = jr.split(key)
        count = int(num_points)
        radial = jnp.sqrt(jr.uniform(radial_key, (count,), dtype=center.dtype))
        angle = _TWO_PI * jr.uniform(angle_key, (count,), dtype=center.dtype)
        direction = jnp.stack((jnp.cos(angle), jnp.sin(angle)), axis=-1)
        return complete_sampling_result(center + radial[:, None] * radii * direction)

    def sample_boundary(
        self,
        state: DesignState,
        num_points: int,
        /,
        *,
        key: Key[Array, ""],
    ) -> SamplingResult:
        center, radii = self._parameters(state)
        grid = jnp.linspace(0.0, _TWO_PI, 1025, dtype=center.dtype)
        speed = jnp.sqrt(
            (radii[0] * jnp.sin(grid)) ** 2 + (radii[1] * jnp.cos(grid)) ** 2
        )
        increments = 0.5 * (speed[1:] + speed[:-1]) * (grid[1] - grid[0])
        cumulative = jnp.concatenate(
            (jnp.zeros((1,), dtype=center.dtype), jnp.cumsum(increments))
        )
        targets = jr.uniform(key, (int(num_points),), dtype=center.dtype) * cumulative[-1]
        angle = jnp.interp(targets, cumulative, grid)
        points = center + radii * jnp.stack((jnp.cos(angle), jnp.sin(angle)), axis=-1)
        return complete_sampling_result(points)

    def boundary_atlas(self, state: DesignState, /) -> BoundaryAtlas:
        center, radii = self._parameters(state)
        return BoundaryAtlas(
            _EllipseBoundaryMap(center, radii),
            source_entity_ids=jnp.zeros((4,), dtype=jnp.int32),
            source_id=self.source_id,
        )


class _RectangleBoundaryMap(AbstractBoundaryMap):
    origins: Array
    directions: Array
    lengths: Array

    def __init__(self, center: Array, size: Array):
        half = 0.5 * size
        self.origins = center + jnp.stack(
            (
                jnp.asarray([-half[0], -half[1]]),
                jnp.asarray([half[0], -half[1]]),
                jnp.asarray([half[0], half[1]]),
                jnp.asarray([-half[0], half[1]]),
            )
        )
        self.directions = jnp.stack(
            (
                jnp.asarray([size[0], 0.0]),
                jnp.asarray([0.0, size[1]]),
                jnp.asarray([-size[0], 0.0]),
                jnp.asarray([0.0, -size[1]]),
            )
        )
        self.lengths = jnp.asarray([size[0], size[1], size[0], size[1]])

    @property
    def num_charts(self) -> int:
        return 4

    @property
    def reference_dimension(self) -> int:
        return 1

    @property
    def ambient_dimension(self) -> int:
        return 2

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        return self.origins[chart_indices] + reference * self.directions[chart_indices]

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        del reference
        return self.lengths[chart_indices]


class Rectangle(GeometrySource):
    """Axis-aligned filled rectangle."""

    center: Array
    size: Array
    feature_id: str = eqx.field(static=True)

    def __init__(self, center: Any, size: Any, *, feature_id: str | None = None):
        self.center = _validate_vector(center, 2, name="center")
        self.size = _validate_positive_vector(size, 2, name="size")
        self.feature_id = _feature_id(feature_id, "rectangle")

    def _compile(self, context: _ParameterCollector, /) -> GeometryKernel:
        center = context.bind(
            ParameterId(self.feature_id, "center"), self.center, role="position"
        )
        size = context.bind(
            ParameterId(self.feature_id, "size"),
            self.size,
            role="length",
            physical_scale=float(jnp.min(self.size)),
            bounds=(0.0, None),
        )
        return _RectangleKernel(center, size, source_id=self.feature_id)


class _RectangleKernel(GeometryKernel):
    center: ParameterBinding = eqx.field(static=True)
    size: ParameterBinding = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self, center: ParameterBinding, size: ParameterBinding, *, source_id: str
    ):
        self.center = center
        self.size = size
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
        return _EXACT_REGION_CAPABILITIES

    @property
    def field_certificate(self) -> FieldCertificate:
        return exact_signed_distance_certificate(smooth=False)

    def _parameters(self, state: DesignState) -> tuple[Array, Array]:
        return self.center.read(state), self.size.read(state)

    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        points_ = _check_points(points, 2)
        center, size = self._parameters(state)
        offset = jnp.abs(points_ - center) - 0.5 * size
        maximum = jnp.max(offset, axis=-1)
        outside = _finite_norm(jnp.maximum(offset, 0.0))
        return jnp.where(maximum <= 0.0, maximum, outside)

    def contains(self, state: DesignState, points: Array, /) -> Array:
        return self.boundary_field(state, points) <= 0.0

    def boundary_normal(self, state: DesignState, points: Array, /) -> Array:
        return _normal_from_field(self, state, points)

    def bounds(self, state: DesignState, /) -> Array:
        center, size = self._parameters(state)
        return jnp.stack((center - 0.5 * size, center + 0.5 * size))

    def measure(self, state: DesignState, /) -> Array:
        _, size = self._parameters(state)
        return jnp.prod(size)

    def boundary_measure(self, state: DesignState, /) -> Array:
        _, size = self._parameters(state)
        return 2.0 * jnp.sum(size)

    def sample_interior(self, state, num_points, /, *, key, plan=None) -> SamplingResult:
        del plan
        bounds = self.bounds(state)
        return complete_sampling_result(
            jr.uniform(
                key,
                (int(num_points), 2),
                minval=bounds[0],
                maxval=bounds[1],
                dtype=bounds.dtype,
            )
        )

    def sample_boundary(self, state, num_points, /, *, key) -> SamplingResult:
        center, size = self._parameters(state)
        count = int(num_points)
        edge_key, coordinate_key = jr.split(key)
        lengths = jnp.asarray([size[0], size[1], size[0], size[1]])
        edge = jr.choice(edge_key, 4, (count,), p=lengths / jnp.sum(lengths))
        coordinate = jr.uniform(coordinate_key, (count,), dtype=center.dtype)
        atlas = _RectangleBoundaryMap(center, size)
        return complete_sampling_result(atlas.map(edge, coordinate[:, None]))

    def boundary_atlas(self, state: DesignState, /) -> BoundaryAtlas:
        center, size = self._parameters(state)
        return BoundaryAtlas(
            _RectangleBoundaryMap(center, size),
            physical_tags=("y_min", "x_max", "y_max", "x_min"),
            source_entity_ids=jnp.arange(4, dtype=jnp.int32),
            source_id=self.source_id,
        )


def Square(center: Any, side: Any, *, feature_id: str | None = None) -> Rectangle:
    side_ = _validate_positive_scalar(side, name="side")
    return Rectangle(center, jnp.repeat(side_[None], 2), feature_id=feature_id)


class _PolygonBoundaryMap(AbstractBoundaryMap):
    vertices: Array

    def __init__(self, vertices: Array):
        self.vertices = vertices

    @property
    def num_charts(self) -> int:
        return self.vertices.shape[0]

    @property
    def reference_dimension(self) -> int:
        return 1

    @property
    def ambient_dimension(self) -> int:
        return 2

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        start = self.vertices[chart_indices]
        end = self.vertices[(chart_indices + 1) % self.vertices.shape[0]]
        return start + reference * (end - start)

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        del reference
        start = self.vertices[chart_indices]
        end = self.vertices[(chart_indices + 1) % self.vertices.shape[0]]
        return jnp.linalg.norm(end - start, axis=-1)


class Polygon(GeometrySource):
    """Simple counter-clockwise polygonal region."""

    vertices: Array
    feature_id: str = eqx.field(static=True)

    def __init__(self, vertices: Any, *, feature_id: str | None = None):
        host = np.asarray(vertices, dtype=float)
        if host.ndim != 2 or host.shape[1] != 2 or host.shape[0] < 3:
            raise ValueError("vertices must have shape (num_vertices >= 3, 2).")
        if not np.all(np.isfinite(host)):
            raise ValueError("vertices must contain only finite values.")
        if np.unique(host, axis=0).shape[0] != host.shape[0]:
            raise ValueError("Non-unique vertices are not allowed.")
        polygon = ShapelyPolygon(host)
        if not polygon.is_valid or polygon.area <= 0.0:
            raise ValueError("Self-intersection or zero-area polygon detected.")
        if not polygon.exterior.is_ccw:
            host = host[::-1].copy()
        self.vertices = jnp.asarray(host, dtype=float)
        self.feature_id = _feature_id(feature_id, "polygon")

    def _compile(self, context: _ParameterCollector, /) -> GeometryKernel:
        vertices = context.bind(
            ParameterId(self.feature_id, "vertices"),
            self.vertices,
            role="position",
            physical_scale=float(np.max(np.ptp(np.asarray(self.vertices), axis=0))),
        )
        return _PolygonKernel(vertices, source_id=self.feature_id)


class _PolygonKernel(GeometryKernel):
    vertices: ParameterBinding = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(self, vertices: ParameterBinding, *, source_id: str):
        self.vertices = vertices
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
        return _EXACT_REGION_CAPABILITIES

    @property
    def field_certificate(self) -> FieldCertificate:
        return exact_signed_distance_certificate(smooth=False)

    def _vertices(self, state: DesignState) -> Array:
        return self.vertices.read(state)

    def _unsigned_distance(self, state: DesignState, points: Array) -> Array:
        vertices = self._vertices(state)
        start = vertices
        edge = jnp.roll(vertices, -1, axis=0) - vertices
        points_ = _check_points(points, 2)
        relative = points_[..., None, :] - start
        denominator = jnp.sum(edge * edge, axis=-1)
        coordinate = jnp.clip(jnp.sum(relative * edge, axis=-1) / denominator, 0.0, 1.0)
        closest = start + coordinate[..., None] * edge
        return jnp.min(_finite_norm(points_[..., None, :] - closest), axis=-1)

    def contains(self, state: DesignState, points: Array, /) -> Array:
        vertices = self._vertices(state)
        points_ = _check_points(points, 2)
        start = vertices
        end = jnp.roll(vertices, -1, axis=0)
        px = points_[..., 0, None]
        py = points_[..., 1, None]
        crosses = (start[:, 1] > py) != (end[:, 1] > py)
        x_intersection = (end[:, 0] - start[:, 0]) * (py - start[:, 1]) / jnp.where(
            end[:, 1] != start[:, 1], end[:, 1] - start[:, 1], 1.0
        ) + start[:, 0]
        inside = jnp.sum(crosses & (px < x_intersection), axis=-1) % 2 == 1
        return inside | (
            self._unsigned_distance(state, points_) <= 16.0 * jnp.finfo(points_.dtype).eps
        )

    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        distance = self._unsigned_distance(state, points)
        return jnp.where(self.contains(state, points), -distance, distance)

    def boundary_normal(self, state: DesignState, points: Array, /) -> Array:
        vertices = self._vertices(state)
        points_ = _check_points(points, 2)
        leading = points_.shape[:-1]
        flat = points_.reshape((-1, 2))
        edge = jnp.roll(vertices, -1, axis=0) - vertices
        relative = flat[:, None, :] - vertices
        coordinate = jnp.clip(
            jnp.sum(relative * edge, axis=-1) / jnp.sum(edge * edge, axis=-1),
            0.0,
            1.0,
        )
        closest = vertices + coordinate[..., None] * edge
        index = jnp.argmin(_finite_norm(flat[:, None, :] - closest), axis=-1)
        selected = edge[index]
        normal = jnp.stack((selected[:, 1], -selected[:, 0]), axis=-1)
        normal = normal / jnp.linalg.norm(normal, axis=-1, keepdims=True)
        return normal.reshape((*leading, 2))

    def bounds(self, state: DesignState, /) -> Array:
        vertices = self._vertices(state)
        return jnp.stack((jnp.min(vertices, axis=0), jnp.max(vertices, axis=0)))

    def measure(self, state: DesignState, /) -> Array:
        vertices = self._vertices(state)
        successor = jnp.roll(vertices, -1, axis=0)
        return 0.5 * jnp.sum(
            vertices[:, 0] * successor[:, 1] - successor[:, 0] * vertices[:, 1]
        )

    def boundary_measure(self, state: DesignState, /) -> Array:
        vertices = self._vertices(state)
        return jnp.sum(
            jnp.linalg.norm(jnp.roll(vertices, -1, axis=0) - vertices, axis=-1)
        )

    def sample_interior(self, state, num_points, /, *, key, plan=None) -> SamplingResult:
        return _uniform_in_bounds(self, state, int(num_points), key, plan)

    def sample_boundary(self, state, num_points, /, *, key) -> SamplingResult:
        vertices = self._vertices(state)
        edges = jnp.roll(vertices, -1, axis=0) - vertices
        lengths = jnp.linalg.norm(edges, axis=-1)
        edge_key, coordinate_key = jr.split(key)
        count = int(num_points)
        indices = jr.choice(
            edge_key, vertices.shape[0], (count,), p=lengths / jnp.sum(lengths)
        )
        coordinate = jr.uniform(coordinate_key, (count, 1), dtype=vertices.dtype)
        return complete_sampling_result(vertices[indices] + coordinate * edges[indices])

    def boundary_atlas(self, state: DesignState, /) -> BoundaryAtlas:
        vertices = self._vertices(state)
        return BoundaryAtlas(
            _PolygonBoundaryMap(vertices),
            source_entity_ids=jnp.arange(vertices.shape[0], dtype=jnp.int32),
            source_id=self.source_id,
        )


def Triangle(
    vertices: Sequence[tuple[float, float]], *, feature_id: str | None = None
) -> Polygon:
    if len(vertices) != 3:
        raise ValueError("Triangle must have exactly 3 vertices.")
    return Polygon(vertices, feature_id=feature_id)


class _EllipsoidBoundaryMap(AbstractBoundaryMap):
    center: Array
    radii: Array

    def __init__(self, center: Array, radii: Array):
        self.center = center
        self.radii = radii

    @property
    def num_charts(self) -> int:
        return 1

    @property
    def reference_dimension(self) -> int:
        return 2

    @property
    def ambient_dimension(self) -> int:
        return 3

    def _direction(self, reference: Array) -> Array:
        azimuth = _TWO_PI * reference[..., 0]
        vertical = 1.0 - 2.0 * reference[..., 1]
        radial = jnp.sqrt(jnp.maximum(1.0 - vertical * vertical, 0.0))
        return jnp.stack(
            (radial * jnp.cos(azimuth), radial * jnp.sin(azimuth), vertical),
            axis=-1,
        )

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        del chart_indices
        return self.center + self.radii * self._direction(reference)

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        del chart_indices
        direction = self._direction(reference)
        return (
            4.0
            * jnp.pi
            * jnp.prod(self.radii)
            * jnp.linalg.norm(direction / self.radii, axis=-1)
        )


class Ellipsoid(GeometrySource):
    """Axis-aligned solid ellipsoid with certified level-set semantics."""

    center: Array
    radii: Array
    feature_id: str = eqx.field(static=True)

    def __init__(self, center: Any, radii: Any, *, feature_id: str | None = None):
        self.center = _validate_vector(center, 3, name="center")
        self.radii = _validate_positive_vector(radii, 3, name="radii")
        self.feature_id = _feature_id(feature_id, "ellipsoid")

    def _compile(self, context: _ParameterCollector, /) -> GeometryKernel:
        center = context.bind(
            ParameterId(self.feature_id, "center"), self.center, role="position"
        )
        radii = context.bind(
            ParameterId(self.feature_id, "radii"),
            self.radii,
            role="length",
            physical_scale=float(jnp.min(self.radii)),
            bounds=(0.0, None),
        )
        return _EllipsoidKernel(center, radii, source_id=self.feature_id)


class _EllipsoidKernel(GeometryKernel):
    center: ParameterBinding = eqx.field(static=True)
    radii: ParameterBinding = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(self, center, radii, *, source_id):
        self.center = center
        self.radii = radii
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
        return _REGION_CAPABILITIES

    @property
    def field_certificate(self) -> FieldCertificate:
        return _LEVEL_SET_CERTIFICATE

    def _parameters(self, state):
        return self.center.read(state), self.radii.read(state)

    def boundary_field(self, state, points, /):
        points_ = _check_points(points, 3)
        center, radii = self._parameters(state)
        return (_finite_norm((points_ - center) / radii) - 1.0) * jnp.min(radii)

    def contains(self, state, points, /):
        return self.boundary_field(state, points) <= 0.0

    def boundary_normal(self, state, points, /):
        return _normal_from_field(self, state, points)

    def bounds(self, state, /):
        center, radii = self._parameters(state)
        return jnp.stack((center - radii, center + radii))

    def measure(self, state, /):
        _, radii = self._parameters(state)
        return (4.0 / 3.0) * jnp.pi * jnp.prod(radii)

    def boundary_measure(self, state, /):
        _, radii = self._parameters(state)
        z = _GL_NODES
        phi = jnp.pi * (_GL_NODES + 1.0)
        radial = jnp.sqrt(jnp.maximum(1.0 - z[:, None] ** 2, 0.0))
        direction = jnp.stack(
            (
                radial * jnp.cos(phi)[None, :],
                radial * jnp.sin(phi)[None, :],
                jnp.broadcast_to(z[:, None], (z.shape[0], phi.shape[0])),
            ),
            axis=-1,
        )
        density = jnp.prod(radii) * jnp.linalg.norm(direction / radii, axis=-1)
        return jnp.pi * jnp.sum(_GL_WEIGHTS[:, None] * _GL_WEIGHTS[None, :] * density)

    def sample_interior(self, state, num_points, /, *, key, plan=None):
        del plan
        center, radii = self._parameters(state)
        direction_key, radial_key = jr.split(key)
        count = int(num_points)
        directions = jr.normal(direction_key, (count, 3), dtype=center.dtype)
        directions = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)
        radial = jr.uniform(radial_key, (count, 1), dtype=center.dtype) ** (1.0 / 3.0)
        return complete_sampling_result(center + radii * radial * directions)

    def sample_boundary(self, state, num_points, /, *, key):
        center, radii = self._parameters(state)
        proposal_count = max(8 * int(num_points), 64)
        direction_key, choice_key = jr.split(key)
        directions = jr.normal(direction_key, (proposal_count, 3), dtype=center.dtype)
        directions = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)
        density = jnp.prod(radii) * jnp.linalg.norm(directions / radii, axis=-1)
        indices = jr.choice(
            choice_key,
            proposal_count,
            (int(num_points),),
            replace=True,
            p=density / jnp.sum(density),
        )
        return complete_sampling_result(center + radii * directions[indices])

    def boundary_atlas(self, state, /):
        center, radii = self._parameters(state)
        return BoundaryAtlas(
            _EllipsoidBoundaryMap(center, radii),
            orientation=-jnp.ones((1,), dtype=float),
            source_entity_ids=jnp.asarray([0], dtype=jnp.int32),
            source_id=self.source_id,
        )


class _CylinderBoundaryMap(AbstractBoundaryMap):
    base: Array
    axis: Array
    radius: Array
    angle: Array
    full: bool = eqx.field(static=True)

    def __init__(
        self, base: Array, axis: Array, radius: Array, angle: Array, *, full: bool
    ):
        self.base = base
        self.axis = axis
        self.radius = radius
        self.angle = angle
        self.full = full

    @property
    def num_charts(self) -> int:
        return 3 if self.full else 5

    @property
    def reference_dimension(self) -> int:
        return 2

    @property
    def ambient_dimension(self) -> int:
        return 3

    def map(self, chart_indices, reference, /):
        first, second, direction, height = _axis_frame(self.axis)
        theta = self.angle * reference[..., 0]
        radial_direction = (
            jnp.cos(theta)[..., None] * first + jnp.sin(theta)[..., None] * second
        )
        side = (
            self.base + reference[..., 1:2] * self.axis + self.radius * radial_direction
        )
        rho = self.radius * jnp.sqrt(reference[..., 1])
        disk = rho[..., None] * radial_direction
        bottom = self.base + disk
        top = self.base + self.axis + disk
        radial0 = (
            self.base
            + reference[..., :1] * self.axis
            + reference[..., 1:2] * self.radius * first
        )
        end_direction = jnp.cos(self.angle) * first + jnp.sin(self.angle) * second
        radial1 = (
            self.base
            + reference[..., :1] * self.axis
            + reference[..., 1:2] * self.radius * end_direction
        )
        result = jnp.where((chart_indices == 0)[..., None], side, bottom)
        result = jnp.where((chart_indices == 2)[..., None], top, result)
        if not self.full:
            result = jnp.where((chart_indices == 3)[..., None], radial0, result)
            result = jnp.where((chart_indices == 4)[..., None], radial1, result)
        del direction, height
        return result

    def jacobian(self, chart_indices, reference, /):
        del reference
        height = _finite_norm(self.axis)
        side = self.angle * self.radius * height
        cap = 0.5 * self.angle * self.radius**2
        radial = self.radius * height
        value = jnp.where(chart_indices == 0, side, cap)
        if not self.full:
            value = jnp.where(chart_indices >= 3, radial, value)
        return value


class Cylinder(GeometrySource):
    """Finite oriented cylinder or cylindrical sector."""

    base_center: Array
    axis: Array
    radius: Array
    angle: Array
    full: bool = eqx.field(static=True)
    feature_id: str = eqx.field(static=True)

    def __init__(
        self,
        base_center: Any,
        axis: Any,
        radius: Any,
        angle: Any = 2.0 * math.pi,
        *,
        feature_id: str | None = None,
    ):
        self.base_center = _validate_vector(base_center, 3, name="base_center")
        self.axis = _validate_vector(axis, 3, name="axis")
        if float(np.linalg.norm(np.asarray(self.axis))) <= 0.0:
            raise ValueError("axis must have non-zero length.")
        self.radius = _validate_positive_scalar(radius, name="radius")
        self.angle = _validate_angle(angle)
        self.full = math.isclose(
            float(self.angle), 2.0 * math.pi, rel_tol=0.0, abs_tol=1e-12
        )
        self.feature_id = _feature_id(feature_id, "cylinder")

    def _compile(self, context):
        base = context.bind(
            ParameterId(self.feature_id, "base_center"), self.base_center, role="position"
        )
        axis = context.bind(
            ParameterId(self.feature_id, "axis"), self.axis, role="direction_length"
        )
        radius = context.bind(
            ParameterId(self.feature_id, "radius"),
            self.radius,
            role="length",
            bounds=(0.0, None),
        )
        angle = context.bind(
            ParameterId(self.feature_id, "angle"),
            self.angle,
            role="angle",
            bounds=(0.0, 2.0 * math.pi),
        )
        return _CylinderKernel(
            base, axis, radius, angle, full=self.full, source_id=self.feature_id
        )


class _CylinderKernel(GeometryKernel):
    base: ParameterBinding = eqx.field(static=True)
    axis: ParameterBinding = eqx.field(static=True)
    radius: ParameterBinding = eqx.field(static=True)
    angle: ParameterBinding = eqx.field(static=True)
    full: bool = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(self, base, axis, radius, angle, *, full, source_id):
        self.base, self.axis, self.radius, self.angle = base, axis, radius, angle
        self.full, self.source_id = full, source_id

    @property
    def ambient_dimension(self):
        return 3

    @property
    def intrinsic_dimension(self):
        return 3

    @property
    def kind(self):
        return GeometryKind.REGION

    @property
    def capabilities(self):
        return _EXACT_REGION_CAPABILITIES if self.full else _REGION_CAPABILITIES

    @property
    def field_certificate(self):
        return (
            exact_signed_distance_certificate(smooth=False)
            if self.full
            else _LEVEL_SET_CERTIFICATE
        )

    def _parameters(self, state):
        return (
            self.base.read(state),
            self.axis.read(state),
            self.radius.read(state),
            self.angle.read(state),
        )

    def boundary_field(self, state, points, /):
        points_ = _check_points(points, 3)
        base, axis, radius, angle = self._parameters(state)
        x, y, z, height, _ = _axial_coordinates(points_, base, axis)
        q = jnp.stack(
            (jnp.sqrt(x * x + y * y) - radius, jnp.abs(z - 0.5 * height) - 0.5 * height),
            axis=-1,
        )
        field = _finite_norm(jnp.maximum(q, 0.0)) + jnp.minimum(jnp.max(q, axis=-1), 0.0)
        if not self.full:
            _, angular = _angle_membership(x, y, angle)
            field = jnp.maximum(field, angular)
        return field

    def contains(self, state, points, /):
        return self.boundary_field(state, points) <= 0.0

    def boundary_normal(self, state, points, /):
        return _normal_from_field(self, state, points)

    def bounds(self, state, /):
        base, axis, radius, _ = self._parameters(state)
        direction = axis / _finite_norm(axis)
        radial_extent = radius * jnp.sqrt(jnp.maximum(1.0 - direction * direction, 0.0))
        return jnp.stack(
            (
                jnp.minimum(base, base + axis) - radial_extent,
                jnp.maximum(base, base + axis) + radial_extent,
            )
        )

    def measure(self, state, /):
        _, axis, radius, angle = self._parameters(state)
        return 0.5 * angle * radius**2 * _finite_norm(axis)

    def boundary_measure(self, state, /):
        _, axis, radius, angle = self._parameters(state)
        height = _finite_norm(axis)
        area = angle * radius * height + angle * radius**2
        return area if self.full else area + 2.0 * radius * height

    def sample_interior(self, state, num_points, /, *, key, plan=None):
        del plan
        base, axis, radius, angle = self._parameters(state)
        first, second, _, _ = _axis_frame(axis)
        radial_key, angle_key, height_key = jr.split(key, 3)
        count = int(num_points)
        rho = radius * jnp.sqrt(jr.uniform(radial_key, (count,), dtype=base.dtype))
        theta = angle * jr.uniform(angle_key, (count,), dtype=base.dtype)
        axial = jr.uniform(height_key, (count, 1), dtype=base.dtype)
        points = (
            base
            + axial * axis
            + rho[:, None]
            * (jnp.cos(theta)[:, None] * first + jnp.sin(theta)[:, None] * second)
        )
        return complete_sampling_result(points)

    def sample_boundary(self, state, num_points, /, *, key):
        atlas = self.boundary_atlas(state)
        reference_key, chart_key = jr.split(key)
        count = int(num_points)
        reference = jr.uniform(reference_key, (count, 2))
        probes = jnp.full((atlas.num_charts, 2), 0.5)
        areas = atlas.jacobian(jnp.arange(atlas.num_charts), probes)
        charts = jr.choice(
            chart_key, atlas.num_charts, (count,), p=areas / jnp.sum(areas)
        )
        return complete_sampling_result(atlas.map(charts, reference))

    def boundary_atlas(self, state, /):
        base, axis, radius, angle = self._parameters(state)
        mapping = _CylinderBoundaryMap(base, axis, radius, angle, full=self.full)
        return BoundaryAtlas(
            mapping,
            physical_tags=(
                ("side", "base", "top")
                if self.full
                else ("side", "base", "top", "cut_start", "cut_end")
            ),
            orientation=(
                jnp.asarray([1.0, 1.0, -1.0])
                if self.full
                else jnp.asarray([1.0, 1.0, -1.0, -1.0, 1.0])
            ),
            source_entity_ids=jnp.arange(mapping.num_charts, dtype=jnp.int32),
            source_id=self.source_id,
        )


class _ConeBoundaryMap(AbstractBoundaryMap):
    base: Array
    axis: Array
    radii: Array
    angle: Array
    full: bool = eqx.field(static=True)

    def __init__(self, base, axis, radii, angle, *, full):
        self.base, self.axis, self.radii, self.angle, self.full = (
            base,
            axis,
            radii,
            angle,
            full,
        )

    @property
    def num_charts(self):
        return 3 if self.full else 5

    @property
    def reference_dimension(self):
        return 2

    @property
    def ambient_dimension(self):
        return 3

    def map(self, chart_indices, reference, /):
        first, second, _, _ = _axis_frame(self.axis)
        theta = self.angle * reference[..., 0]
        direction = jnp.cos(theta)[..., None] * first + jnp.sin(theta)[..., None] * second
        radius = self.radii[0] + reference[..., 1] * (self.radii[1] - self.radii[0])
        side = self.base + reference[..., 1:2] * self.axis + radius[..., None] * direction
        rho0 = self.radii[0] * jnp.sqrt(reference[..., 1])
        rho1 = self.radii[1] * jnp.sqrt(reference[..., 1])
        bottom = self.base + rho0[..., None] * direction
        top = self.base + self.axis + rho1[..., None] * direction
        radial0 = (
            self.base
            + reference[..., :1] * self.axis
            + (self.radii[0] + reference[..., 0] * (self.radii[1] - self.radii[0]))[
                ..., None
            ]
            * reference[..., 1:2]
            * first
        )
        end_direction = jnp.cos(self.angle) * first + jnp.sin(self.angle) * second
        radial1 = (
            self.base
            + reference[..., :1] * self.axis
            + (self.radii[0] + reference[..., 0] * (self.radii[1] - self.radii[0]))[
                ..., None
            ]
            * reference[..., 1:2]
            * end_direction
        )
        result = jnp.where((chart_indices == 0)[..., None], side, bottom)
        result = jnp.where((chart_indices == 2)[..., None], top, result)
        if not self.full:
            result = jnp.where((chart_indices == 3)[..., None], radial0, result)
            result = jnp.where((chart_indices == 4)[..., None], radial1, result)
        return result

    def jacobian(self, chart_indices, reference, /):
        height = _finite_norm(self.axis)
        slant = jnp.sqrt(height**2 + (self.radii[1] - self.radii[0]) ** 2)
        radius = self.radii[0] + reference[..., 1] * (self.radii[1] - self.radii[0])
        side = self.angle * radius * slant
        cap0 = 0.5 * self.angle * self.radii[0] ** 2
        cap1 = 0.5 * self.angle * self.radii[1] ** 2
        radial = 0.5 * (self.radii[0] + self.radii[1]) * height
        value = jnp.where(chart_indices == 0, side, cap0)
        value = jnp.where(chart_indices == 2, cap1, value)
        if not self.full:
            value = jnp.where(chart_indices >= 3, radial, value)
        return value


class Cone(GeometrySource):
    """Finite oriented cone or conical frustum."""

    base_center: Array
    axis: Array
    radii: Array
    angle: Array
    full: bool = eqx.field(static=True)
    feature_id: str = eqx.field(static=True)

    def __init__(
        self,
        base_center,
        axis,
        radius0,
        radius1=0.0,
        angle=2.0 * math.pi,
        *,
        feature_id=None,
    ):
        self.base_center = _validate_vector(base_center, 3, name="base_center")
        self.axis = _validate_vector(axis, 3, name="axis")
        if float(np.linalg.norm(np.asarray(self.axis))) <= 0.0:
            raise ValueError("axis must have non-zero length.")
        r0 = np.asarray(radius0, dtype=float)
        r1 = np.asarray(radius1, dtype=float)
        if (
            r0.shape != ()
            or r1.shape != ()
            or not np.isfinite(r0)
            or not np.isfinite(r1)
            or float(r0) <= 0.0
            or float(r1) < 0.0
        ):
            raise ValueError("radius0 must be positive and radius1 must be non-negative.")
        self.radii = jnp.asarray([r0, r1], dtype=float)
        self.angle = _validate_angle(angle)
        self.full = math.isclose(
            float(self.angle), 2.0 * math.pi, rel_tol=0.0, abs_tol=1e-12
        )
        self.feature_id = _feature_id(feature_id, "cone")

    def _compile(self, context):
        base = context.bind(
            ParameterId(self.feature_id, "base_center"), self.base_center, role="position"
        )
        axis = context.bind(
            ParameterId(self.feature_id, "axis"), self.axis, role="direction_length"
        )
        radii = context.bind(
            ParameterId(self.feature_id, "radii"),
            self.radii,
            role="length",
            bounds=(0.0, None),
        )
        angle = context.bind(
            ParameterId(self.feature_id, "angle"),
            self.angle,
            role="angle",
            bounds=(0.0, 2.0 * math.pi),
        )
        return _ConeKernel(
            base, axis, radii, angle, full=self.full, source_id=self.feature_id
        )


class _ConeKernel(GeometryKernel):
    base: ParameterBinding = eqx.field(static=True)
    axis: ParameterBinding = eqx.field(static=True)
    radii: ParameterBinding = eqx.field(static=True)
    angle: ParameterBinding = eqx.field(static=True)
    full: bool = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(self, base, axis, radii, angle, *, full, source_id):
        self.base, self.axis, self.radii, self.angle = base, axis, radii, angle
        self.full, self.source_id = full, source_id

    @property
    def ambient_dimension(self):
        return 3

    @property
    def intrinsic_dimension(self):
        return 3

    @property
    def kind(self):
        return GeometryKind.REGION

    @property
    def capabilities(self):
        return _REGION_CAPABILITIES

    @property
    def field_certificate(self):
        return _LEVEL_SET_CERTIFICATE

    def _parameters(self, state):
        return (
            self.base.read(state),
            self.axis.read(state),
            self.radii.read(state),
            self.angle.read(state),
        )

    def boundary_field(self, state, points, /):
        points_ = _check_points(points, 3)
        base, axis, radii, angle = self._parameters(state)
        x, y, z, height, _ = _axial_coordinates(points_, base, axis)
        radial = jnp.sqrt(x * x + y * y)
        t = jnp.clip(z / height, 0.0, 1.0)
        radius = radii[0] + t * (radii[1] - radii[0])
        side = (
            (radial - radius) * height / jnp.sqrt(height**2 + (radii[1] - radii[0]) ** 2)
        )
        field = jnp.maximum(side, jnp.maximum(-z, z - height))
        if not self.full:
            _, angular = _angle_membership(x, y, angle)
            field = jnp.maximum(field, angular)
        return field

    def contains(self, state, points, /):
        return self.boundary_field(state, points) <= 0.0

    def boundary_normal(self, state, points, /):
        return _normal_from_field(self, state, points)

    def bounds(self, state, /):
        base, axis, radii, _ = self._parameters(state)
        direction = axis / _finite_norm(axis)
        radial_extent = jnp.max(radii) * jnp.sqrt(
            jnp.maximum(1.0 - direction * direction, 0.0)
        )
        return jnp.stack(
            (
                jnp.minimum(base, base + axis) - radial_extent,
                jnp.maximum(base, base + axis) + radial_extent,
            )
        )

    def measure(self, state, /):
        _, axis, radii, angle = self._parameters(state)
        return (
            angle
            * _finite_norm(axis)
            * (radii[0] ** 2 + radii[0] * radii[1] + radii[1] ** 2)
            / 6.0
        )

    def boundary_measure(self, state, /):
        _, axis, radii, angle = self._parameters(state)
        height = _finite_norm(axis)
        slant = jnp.sqrt(height**2 + (radii[1] - radii[0]) ** 2)
        area = 0.5 * angle * (radii[0] + radii[1]) * slant + 0.5 * angle * jnp.sum(
            radii**2
        )
        return area if self.full else area + (radii[0] + radii[1]) * height

    def sample_interior(self, state, num_points, /, *, key, plan=None):
        return _uniform_in_bounds(self, state, int(num_points), key, plan)

    def sample_boundary(self, state, num_points, /, *, key):
        atlas = self.boundary_atlas(state)
        reference_key, chart_key = jr.split(key)
        count = int(num_points)
        reference = jr.uniform(reference_key, (count, 2))
        probes = jnp.full((atlas.num_charts, 2), 0.5)
        areas = atlas.jacobian(jnp.arange(atlas.num_charts), probes)
        charts = jr.choice(
            chart_key, atlas.num_charts, (count,), p=areas / jnp.sum(areas)
        )
        return complete_sampling_result(atlas.map(charts, reference))

    def boundary_atlas(self, state, /):
        base, axis, radii, angle = self._parameters(state)
        mapping = _ConeBoundaryMap(base, axis, radii, angle, full=self.full)
        return BoundaryAtlas(
            mapping,
            physical_tags=(
                ("side", "base", "top")
                if self.full
                else ("side", "base", "top", "cut_start", "cut_end")
            ),
            orientation=(
                jnp.asarray([1.0, 1.0, -1.0])
                if self.full
                else jnp.asarray([1.0, 1.0, -1.0, -1.0, 1.0])
            ),
            source_entity_ids=jnp.arange(mapping.num_charts, dtype=jnp.int32),
            source_id=self.source_id,
        )


class _TorusBoundaryMap(AbstractBoundaryMap):
    center: Array
    major: Array
    minor: Array
    angle: Array
    full: bool = eqx.field(static=True)

    def __init__(self, center, major, minor, angle, *, full):
        self.center, self.major, self.minor, self.angle, self.full = (
            center,
            major,
            minor,
            angle,
            full,
        )

    @property
    def num_charts(self):
        return 1 if self.full else 3

    @property
    def reference_dimension(self):
        return 2

    @property
    def ambient_dimension(self):
        return 3

    def map(self, chart_indices, reference, /):
        sweep = self.angle * reference[..., 0]
        tube = _TWO_PI * reference[..., 1]
        ring = self.major + self.minor * jnp.cos(tube)
        surface = self.center + jnp.stack(
            (ring * jnp.cos(sweep), ring * jnp.sin(sweep), self.minor * jnp.sin(tube)),
            axis=-1,
        )
        rho = self.minor * jnp.sqrt(reference[..., 1])
        disk0 = self.center + jnp.stack(
            (
                self.major + rho * jnp.cos(_TWO_PI * reference[..., 0]),
                jnp.zeros_like(rho),
                rho * jnp.sin(_TWO_PI * reference[..., 0]),
            ),
            axis=-1,
        )
        end_center = self.center + jnp.asarray(
            [self.major * jnp.cos(self.angle), self.major * jnp.sin(self.angle), 0.0]
        )
        radial = jnp.stack(
            (jnp.cos(self.angle), jnp.sin(self.angle), jnp.zeros_like(self.angle))
        )
        vertical = jnp.asarray([0.0, 0.0, 1.0])
        disk1 = end_center + rho[..., None] * (
            jnp.cos(_TWO_PI * reference[..., 0])[..., None] * radial
            + jnp.sin(_TWO_PI * reference[..., 0])[..., None] * vertical
        )
        result = surface
        if not self.full:
            result = jnp.where((chart_indices == 1)[..., None], disk0, result)
            result = jnp.where((chart_indices == 2)[..., None], disk1, result)
        return result

    def jacobian(self, chart_indices, reference, /):
        tube = _TWO_PI * reference[..., 1]
        surface = (
            self.angle * _TWO_PI * self.minor * (self.major + self.minor * jnp.cos(tube))
        )
        cap = jnp.pi * self.minor**2
        return jnp.where(chart_indices == 0, surface, cap)


class Torus(GeometrySource):
    """Solid torus or toroidal sector around the global z axis."""

    center: Array
    major_radius: Array
    minor_radius: Array
    angle: Array
    full: bool = eqx.field(static=True)
    feature_id: str = eqx.field(static=True)

    def __init__(
        self, center, inner_radius, outer_radius, angle=2.0 * math.pi, *, feature_id=None
    ):
        self.center = _validate_vector(center, 3, name="center")
        inner = float(np.asarray(inner_radius))
        outer = float(np.asarray(outer_radius))
        if (
            not np.isfinite(inner)
            or not np.isfinite(outer)
            or inner < 0.0
            or outer <= inner
        ):
            raise ValueError("Torus radii require 0 <= inner_radius < outer_radius.")
        self.major_radius = jnp.asarray(0.5 * (inner + outer), dtype=float)
        self.minor_radius = jnp.asarray(0.5 * (outer - inner), dtype=float)
        self.angle = _validate_angle(angle)
        self.full = math.isclose(
            float(self.angle), 2.0 * math.pi, rel_tol=0.0, abs_tol=1e-12
        )
        self.feature_id = _feature_id(feature_id, "torus")

    def _compile(self, context):
        center = context.bind(
            ParameterId(self.feature_id, "center"), self.center, role="position"
        )
        major = context.bind(
            ParameterId(self.feature_id, "major_radius"),
            self.major_radius,
            role="length",
            bounds=(0.0, None),
        )
        minor = context.bind(
            ParameterId(self.feature_id, "minor_radius"),
            self.minor_radius,
            role="length",
            bounds=(0.0, None),
        )
        angle = context.bind(
            ParameterId(self.feature_id, "angle"),
            self.angle,
            role="angle",
            bounds=(0.0, 2.0 * math.pi),
        )
        return _TorusKernel(
            center, major, minor, angle, full=self.full, source_id=self.feature_id
        )


class _TorusKernel(GeometryKernel):
    center: ParameterBinding = eqx.field(static=True)
    major: ParameterBinding = eqx.field(static=True)
    minor: ParameterBinding = eqx.field(static=True)
    angle: ParameterBinding = eqx.field(static=True)
    full: bool = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(self, center, major, minor, angle, *, full, source_id):
        self.center, self.major, self.minor, self.angle = center, major, minor, angle
        self.full, self.source_id = full, source_id

    @property
    def ambient_dimension(self):
        return 3

    @property
    def intrinsic_dimension(self):
        return 3

    @property
    def kind(self):
        return GeometryKind.REGION

    @property
    def capabilities(self):
        return _EXACT_REGION_CAPABILITIES if self.full else _REGION_CAPABILITIES

    @property
    def field_certificate(self):
        return (
            exact_signed_distance_certificate(smooth=False)
            if self.full
            else _LEVEL_SET_CERTIFICATE
        )

    def _parameters(self, state):
        return (
            self.center.read(state),
            self.major.read(state),
            self.minor.read(state),
            self.angle.read(state),
        )

    def boundary_field(self, state, points, /):
        points_ = _check_points(points, 3)
        center, major, minor, angle = self._parameters(state)
        relative = points_ - center
        radial = jnp.sqrt(relative[..., 0] ** 2 + relative[..., 1] ** 2)
        field = (
            _finite_norm(jnp.stack((radial - major, relative[..., 2]), axis=-1)) - minor
        )
        if not self.full:
            _, angular = _angle_membership(relative[..., 0], relative[..., 1], angle)
            field = jnp.maximum(field, angular)
        return field

    def contains(self, state, points, /):
        return self.boundary_field(state, points) <= 0.0

    def boundary_normal(self, state, points, /):
        return _normal_from_field(self, state, points)

    def bounds(self, state, /):
        center, major, minor, _ = self._parameters(state)
        extent = jnp.asarray([major + minor, major + minor, minor])
        return jnp.stack((center - extent, center + extent))

    def measure(self, state, /):
        _, major, minor, angle = self._parameters(state)
        return angle * jnp.pi * major * minor**2

    def boundary_measure(self, state, /):
        _, major, minor, angle = self._parameters(state)
        area = angle * _TWO_PI * major * minor
        return area if self.full else area + 2.0 * jnp.pi * minor**2

    def sample_interior(self, state, num_points, /, *, key, plan=None):
        return _uniform_in_bounds(self, state, int(num_points), key, plan)

    def sample_boundary(self, state, num_points, /, *, key):
        atlas = self.boundary_atlas(state)
        return _sample_boundary_atlas(atlas, int(num_points), key)

    def boundary_atlas(self, state, /):
        center, major, minor, angle = self._parameters(state)
        mapping = _TorusBoundaryMap(center, major, minor, angle, full=self.full)
        return BoundaryAtlas(
            mapping,
            physical_tags=(
                ("surface",) if self.full else ("surface", "cut_start", "cut_end")
            ),
            orientation=(
                jnp.asarray([1.0]) if self.full else jnp.asarray([1.0, -1.0, 1.0])
            ),
            source_entity_ids=jnp.arange(mapping.num_charts, dtype=jnp.int32),
            source_id=self.source_id,
        )


def _sample_boundary_atlas(
    atlas: BoundaryAtlas, count: int, key: Key[Array, ""]
) -> SamplingResult:
    candidate_count = max(8 * count, 64)
    chart_key, reference_key, choice_key = jr.split(key, 3)
    charts = jr.randint(chart_key, (candidate_count,), 0, atlas.num_charts)
    reference = jr.uniform(reference_key, (candidate_count, atlas.reference_dimension))
    weights = atlas.jacobian(charts, reference)
    selected = jr.choice(
        choice_key, candidate_count, (count,), replace=True, p=weights / jnp.sum(weights)
    )
    return complete_sampling_result(atlas.map(charts[selected], reference[selected]))


class _TriangleBoundaryMap(AbstractBoundaryMap):
    triangles: Array

    def __init__(self, triangles: Array):
        self.triangles = triangles

    @property
    def num_charts(self):
        return self.triangles.shape[0]

    @property
    def reference_dimension(self):
        return 2

    @property
    def ambient_dimension(self):
        return 3

    def map(self, chart_indices, reference, /):
        triangle = self.triangles[chart_indices]
        u = reference[..., 0]
        v = reference[..., 1]
        first = 1.0 - jnp.sqrt(jnp.maximum(u, 0.0))
        second = jnp.sqrt(jnp.maximum(u, 0.0)) * (1.0 - v)
        third = jnp.sqrt(jnp.maximum(u, 0.0)) * v
        return (
            first[..., None] * triangle[..., 0, :]
            + second[..., None] * triangle[..., 1, :]
            + third[..., None] * triangle[..., 2, :]
        )

    def jacobian(self, chart_indices, reference, /):
        del reference
        triangle = self.triangles[chart_indices]
        return 0.5 * jnp.linalg.norm(
            jnp.cross(
                triangle[..., 1, :] - triangle[..., 0, :],
                triangle[..., 2, :] - triangle[..., 0, :],
            ),
            axis=-1,
        )


class Wedge(GeometrySource):
    """Convex right wedge obtained by extruding a quadrilateral ramp."""

    corner: Array
    extents: Array
    top_extent: Array
    feature_id: str = eqx.field(static=True)

    def __init__(self, corner, extents, top_extent, *, feature_id=None):
        self.corner = _validate_vector(corner, 3, name="corner")
        self.extents = _validate_positive_vector(extents, 3, name="extents")
        top = float(np.asarray(top_extent))
        if not np.isfinite(top) or top < 0.0 or top > float(self.extents[0]):
            raise ValueError("top_extent must lie in [0, extents[0]].")
        self.top_extent = jnp.asarray(top, dtype=float)
        self.feature_id = _feature_id(feature_id, "wedge")

    def _compile(self, context):
        corner = context.bind(
            ParameterId(self.feature_id, "corner"), self.corner, role="position"
        )
        extents = context.bind(
            ParameterId(self.feature_id, "extents"),
            self.extents,
            role="length",
            bounds=(0.0, None),
        )
        top = context.bind(
            ParameterId(self.feature_id, "top_extent"),
            self.top_extent,
            role="length",
            bounds=(0.0, float(self.extents[0])),
        )
        return _WedgeKernel(corner, extents, top, source_id=self.feature_id)


class _WedgeKernel(GeometryKernel):
    corner: ParameterBinding = eqx.field(static=True)
    extents: ParameterBinding = eqx.field(static=True)
    top_extent: ParameterBinding = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    faces: Array

    def __init__(self, corner, extents, top_extent, *, source_id):
        self.corner, self.extents, self.top_extent, self.source_id = (
            corner,
            extents,
            top_extent,
            source_id,
        )
        self.faces = jnp.asarray(
            [
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 0, 4],
                [3, 4, 7],
            ],
            dtype=jnp.int32,
        )

    @property
    def ambient_dimension(self):
        return 3

    @property
    def intrinsic_dimension(self):
        return 3

    @property
    def kind(self):
        return GeometryKind.REGION

    @property
    def capabilities(self):
        return _REGION_CAPABILITIES

    @property
    def field_certificate(self):
        return _LEVEL_SET_CERTIFICATE

    def _parameters(self, state):
        return (
            self.corner.read(state),
            self.extents.read(state),
            self.top_extent.read(state),
        )

    def _vertices(self, state):
        corner, extents, top = self._parameters(state)
        x, y, z = extents
        local = jnp.asarray(
            [
                [0, 0, 0],
                [x, 0, 0],
                [top, 0, z],
                [0, 0, z],
                [0, y, 0],
                [x, y, 0],
                [top, y, z],
                [0, y, z],
            ],
            dtype=corner.dtype,
        )
        return corner + local

    def _planes(self, state):
        vertices = self._vertices(state)
        triangles = vertices[self.faces]
        normals = jnp.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        normals = normals / jnp.linalg.norm(normals, axis=-1, keepdims=True)
        centroid = jnp.mean(vertices, axis=0)
        flip = jnp.sum(normals * (centroid - triangles[:, 0]), axis=-1) > 0.0
        normals = jnp.where(flip[:, None], -normals, normals)
        offsets = -jnp.sum(normals * triangles[:, 0], axis=-1)
        return normals, offsets

    def boundary_field(self, state, points, /):
        points_ = _check_points(points, 3)
        normals, offsets = self._planes(state)
        return jnp.max(
            jnp.sum(points_[..., None, :] * normals, axis=-1) + offsets, axis=-1
        )

    def contains(self, state, points, /):
        return self.boundary_field(state, points) <= 0.0

    def boundary_normal(self, state, points, /):
        return _normal_from_field(self, state, points)

    def bounds(self, state, /):
        vertices = self._vertices(state)
        return jnp.stack((jnp.min(vertices, axis=0), jnp.max(vertices, axis=0)))

    def measure(self, state, /):
        _, extents, top = self._parameters(state)
        return extents[1] * extents[2] * 0.5 * (extents[0] + top)

    def boundary_measure(self, state, /):
        triangles = self._vertices(state)[self.faces]
        return jnp.sum(
            0.5
            * jnp.linalg.norm(
                jnp.cross(
                    triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
                ),
                axis=-1,
            )
        )

    def sample_interior(self, state, num_points, /, *, key, plan=None):
        return _uniform_in_bounds(self, state, int(num_points), key, plan)

    def sample_boundary(self, state, num_points, /, *, key):
        return _sample_boundary_atlas(self.boundary_atlas(state), int(num_points), key)

    def boundary_atlas(self, state, /):
        mapping = _TriangleBoundaryMap(self._vertices(state)[self.faces])
        return BoundaryAtlas(
            mapping,
            source_entity_ids=jnp.arange(mapping.num_charts, dtype=jnp.int32),
            source_id=self.source_id,
        )


__all__ = [
    "Cone",
    "Cylinder",
    "Ellipse",
    "Ellipsoid",
    "Polygon",
    "Rectangle",
    "Square",
    "Torus",
    "Triangle",
    "Wedge",
]
