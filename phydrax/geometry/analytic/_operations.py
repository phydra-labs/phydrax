#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any, Literal
from uuid import uuid4

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import opt_einsum as oe
from jax.scipy.special import logsumexp
from jaxtyping import Array

from ..._numerics._quadrature_rules import gauss_legendre_data
from .._atlas import AbstractBoundaryMap, BoundaryAtlas
from .._capabilities import ContactCurvatureProvider, GeometryCapability
from .._certificate import (
    DistanceSemantics,
    FieldCertificate,
    FieldRegularity,
    SignReliability,
    ZeroSetAccuracy,
)
from .._contracts import (
    ContactCurvatureResult,
    GeometryKernel,
    GeometryKind,
    GeometrySource,
)
from .._cubature import AbstractCubatureMap, CubatureAtlas, CubatureComponent
from .._sampling import (
    bounded_rejection_sample,
    RejectionSamplingPlan,
    SamplingResult,
)
from ..design._schema import (
    _ParameterCollector,
    ParameterBinding,
    ParameterId,
)


class RigidFrame(eqx.Module):
    """Validated right-handed rigid frame represented by rotation and translation."""

    rotation: Array
    translation: Array

    def __init__(self, rotation: Any, translation: Any):
        rotation_host = np.asarray(rotation, dtype=float)
        translation_host = np.asarray(translation, dtype=float)
        if rotation_host.ndim != 2 or rotation_host.shape[0] != rotation_host.shape[1]:
            raise ValueError("rotation must be a square matrix.")
        dimension = rotation_host.shape[0]
        if translation_host.shape != (dimension,):
            raise ValueError(f"translation must have shape ({dimension},).")
        if not np.all(np.isfinite(rotation_host)) or not np.all(
            np.isfinite(translation_host)
        ):
            raise ValueError("RigidFrame values must be finite.")
        identity = np.eye(dimension)
        if not np.allclose(
            rotation_host.T @ rotation_host, identity, rtol=1e-10, atol=1e-12
        ):
            raise ValueError("rotation must be orthogonal.")
        if not np.isclose(np.linalg.det(rotation_host), 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("rotation must be right-handed with determinant one.")
        self.rotation = jnp.asarray(rotation_host, dtype=float)
        self.translation = jnp.asarray(translation_host, dtype=float)

    @property
    def dimension(self) -> int:
        return self.translation.shape[0]

    @classmethod
    def identity(cls, dimension: int) -> RigidFrame:
        if dimension <= 0:
            raise ValueError("dimension must be positive.")
        return cls(np.eye(dimension), np.zeros((dimension,)))

    @classmethod
    def from_axis_angle(
        cls,
        axis: Any,
        angle: float,
        *,
        translation: Any = (0.0, 0.0, 0.0),
    ) -> RigidFrame:
        axis_host = np.asarray(axis, dtype=float)
        if axis_host.shape != (3,) or not np.all(np.isfinite(axis_host)):
            raise ValueError("axis must be a finite three-vector.")
        norm = np.linalg.norm(axis_host)
        if norm <= 0.0:
            raise ValueError("axis must have non-zero length.")
        angle_ = float(angle)
        if not math.isfinite(angle_):
            raise ValueError("angle must be finite.")
        x, y, z = axis_host / norm
        skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        rotation = (
            np.eye(3) + math.sin(angle_) * skew + (1.0 - math.cos(angle_)) * (skew @ skew)
        )
        return cls(rotation, translation)

    def apply(self, points: Array, /) -> Array:
        return jnp.asarray(points) @ self.rotation.T + self.translation

    def inverse_apply(self, points: Array, /) -> Array:
        return (jnp.asarray(points) - self.translation) @ self.rotation

    def inverse(self) -> RigidFrame:
        inverse_rotation = self.rotation.T
        return RigidFrame(inverse_rotation, -inverse_rotation @ self.translation)


class _AffineBoundaryMap(AbstractBoundaryMap):
    base: AbstractBoundaryMap
    linear: Array
    offset: Array

    def __init__(self, base: AbstractBoundaryMap, linear: Array, offset: Array):
        self.base = base
        self.linear = jnp.asarray(linear, dtype=float)
        self.offset = jnp.asarray(offset, dtype=float)

    @property
    def num_charts(self) -> int:
        return self.base.num_charts

    @property
    def reference_dimension(self) -> int:
        return self.base.reference_dimension

    @property
    def ambient_dimension(self) -> int:
        return self.base.ambient_dimension

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        return self.base.map(chart_indices, reference) @ self.linear.T + self.offset

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        indices = jnp.asarray(chart_indices, dtype=jnp.int32)
        reference_ = jnp.asarray(reference, dtype=float)
        leading = reference_.shape[:-1]
        flat_indices = indices.reshape((-1,))
        flat_reference = reference_.reshape((-1, self.reference_dimension))

        def differential(index, coordinate):
            return jax.jacfwd(lambda value: self.base.map(index, value))(coordinate)

        base_differential = jax.vmap(differential)(flat_indices, flat_reference)
        transformed = oe.contract("ij,njk->nik", self.linear, base_differential)
        gram = jnp.swapaxes(transformed, -1, -2) @ transformed
        measure = jnp.sqrt(jnp.maximum(jnp.linalg.det(gram), 0.0))
        return measure.reshape(leading)


class _AffineCubatureMap(AbstractCubatureMap):
    base: AbstractCubatureMap
    linear: Array
    offset: Array
    measure_scale: Array

    def __init__(
        self,
        base: AbstractCubatureMap,
        linear: Array,
        offset: Array,
        measure_scale: Array,
    ):
        self.base = base
        self.linear = jnp.asarray(linear, dtype=float)
        self.offset = jnp.asarray(offset, dtype=float)
        self.measure_scale = jnp.asarray(measure_scale, dtype=float).reshape(())

    @property
    def num_charts(self) -> int:
        return self.base.num_charts

    @property
    def reference_domain(self):
        return self.base.reference_domain

    @property
    def ambient_dimension(self) -> int:
        return self.base.ambient_dimension

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        return self.base.map(chart_indices, reference) @ self.linear.T + self.offset

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        return self.base.jacobian(chart_indices, reference) * self.measure_scale

    def reference_mask(self, chart_indices: Array, reference: Array, /) -> Array:
        return self.base.reference_mask(chart_indices, reference)


class RigidTransform(GeometrySource):
    """Rigid transform expression with trainable frame values."""

    child: GeometrySource
    frame: RigidFrame
    feature_id: str = eqx.field(static=True)

    def __init__(
        self, child: GeometrySource, frame: RigidFrame, *, feature_id: str | None = None
    ):
        if not isinstance(child, GeometrySource):
            raise TypeError("child must be a GeometrySource.")
        if not isinstance(frame, RigidFrame):
            raise TypeError("frame must be a RigidFrame.")
        self.child = child
        self.frame = frame
        self.feature_id = feature_id or f"rigid-transform-{uuid4().hex}"
        if not self.feature_id:
            raise ValueError("feature_id must be non-empty.")

    def _compile(self, context: _ParameterCollector, /) -> GeometryKernel:
        child = self.child._compile(context)
        if child.ambient_dimension != self.frame.dimension:
            raise ValueError("Rigid frame dimension must match child ambient dimension.")
        rotation = context.bind(
            ParameterId(self.feature_id, "rotation"),
            self.frame.rotation,
            role="rotation_matrix",
        )
        translation = context.bind(
            ParameterId(self.feature_id, "translation"),
            self.frame.translation,
            role="position_offset",
        )
        return _RigidTransformKernel(child, rotation, translation)


class _RigidTransformKernel(GeometryKernel):
    child: GeometryKernel
    rotation: ParameterBinding = eqx.field(static=True)
    translation: ParameterBinding = eqx.field(static=True)

    def __init__(self, child, rotation, translation):
        self.child, self.rotation, self.translation = child, rotation, translation

    @property
    def ambient_dimension(self):
        return self.child.ambient_dimension

    @property
    def intrinsic_dimension(self):
        return self.child.intrinsic_dimension

    @property
    def kind(self):
        return self.child.kind

    @property
    def capabilities(self):
        return self.child.capabilities

    @property
    def field_certificate(self):
        certificate = self.child.field_certificate
        return FieldCertificate(
            certificate.zero_set_accuracy,
            certificate.sign_reliability,
            certificate.distance_semantics,
            certificate.regularity,
            certificate.safe_step_factor,
            certificate.validity_region,
            certificate.parameter_differentiable,
            (*certificate.provenance, "rigid_transform"),
        )

    def _parameters(self, state):
        return self.rotation.read(state), self.translation.read(state)

    def _local(self, state, points):
        rotation, translation = self._parameters(state)
        return (jnp.asarray(points) - translation) @ rotation

    def boundary_field(self, state, points, /):
        return self.child.boundary_field(state, self._local(state, points))

    def contains(self, state, points, /):
        return self.child.contains(state, self._local(state, points))

    def boundary_normal(self, state, points, /):
        rotation, _ = self._parameters(state)
        return self.child.boundary_normal(state, self._local(state, points)) @ rotation.T

    def contact_curvature(self, state, points, /):
        if not isinstance(self.child, ContactCurvatureProvider):
            raise TypeError("Transformed child lacks contact-curvature provider.")
        result = self.child.contact_curvature(state, self._local(state, points))
        if not isinstance(result, ContactCurvatureResult):
            raise TypeError("Child curvature query returned an invalid result.")
        return result

    def bounds(self, state, /):
        rotation, translation = self._parameters(state)
        bounds = self.child.bounds(state)
        dimension = self.ambient_dimension
        corners = jnp.stack(
            tuple(
                jnp.where(
                    jnp.asarray(
                        [(index >> axis) & 1 for axis in range(dimension)], dtype=bool
                    ),
                    bounds[1],
                    bounds[0],
                )
                for index in range(1 << dimension)
            )
        )
        transformed = corners @ rotation.T + translation
        return jnp.stack((jnp.min(transformed, axis=0), jnp.max(transformed, axis=0)))

    def measure(self, state, /):
        return self.child.measure(state)

    def boundary_measure(self, state, /):
        return self.child.boundary_measure(state)

    def sample_interior(self, state, num_points, /, *, key, plan=None):
        rotation, translation = self._parameters(state)
        result = self.child.sample_interior(state, num_points, key=key, plan=plan)
        return SamplingResult(
            result.points @ rotation.T + translation,
            result.valid,
            result.report,
            weights=result.weights,
            strata=result.strata,
        )

    def sample_boundary(self, state, num_points, /, *, key):
        rotation, translation = self._parameters(state)
        result = self.child.sample_boundary(state, num_points, key=key)
        return SamplingResult(
            result.points @ rotation.T + translation,
            result.valid,
            result.report,
            weights=result.weights,
            strata=result.strata,
        )

    def boundary_atlas(self, state, /):
        rotation, translation = self._parameters(state)
        atlas = self.child.boundary_atlas(state)
        return BoundaryAtlas(
            _AffineBoundaryMap(atlas.mapping, rotation, translation),
            source_entity_ids=atlas.source_entity_ids,
            source_id=atlas.source_id,
            physical_tags=atlas.physical_tags,
            orientation=atlas.orientation,
            seam_owner=atlas.seam_owner,
            trim_domains=atlas.trim_domains,
        )

    def cubature_atlas(self, state, component: CubatureComponent, /) -> CubatureAtlas:
        rotation, translation = self._parameters(state)
        atlas = self.child.cubature_atlas(state, component)
        return CubatureAtlas(
            _AffineCubatureMap(
                atlas.mapping,
                rotation,
                translation,
                jnp.asarray(1.0),
            ),
            source_entity_ids=atlas.source_entity_ids,
            source_id=atlas.source_id,
            physical_tags=atlas.physical_tags,
        )


class Scaling(GeometrySource):
    """Positive affine scaling about a fixed or trainable center."""

    child: GeometrySource
    scale: Array
    center: Array
    uniform: bool = eqx.field(static=True)
    feature_id: str = eqx.field(static=True)

    def __init__(
        self,
        child: GeometrySource,
        scale: Any,
        *,
        center: Any | None = None,
        feature_id: str | None = None,
    ):
        if not isinstance(child, GeometrySource):
            raise TypeError("child must be a GeometrySource.")
        scale_host = np.asarray(scale, dtype=float)
        if scale_host.ndim == 0:
            if not np.isfinite(scale_host) or float(scale_host) <= 0.0:
                raise ValueError("scale must be finite and positive.")
            scale_host = np.asarray([float(scale_host)])
            uniform = True
            scalar_scale = True
        elif scale_host.ndim == 1 and scale_host.size > 0:
            if not np.all(np.isfinite(scale_host)) or np.any(scale_host <= 0.0):
                raise ValueError("scale entries must be finite and positive.")
            uniform = bool(np.allclose(scale_host, scale_host[0]))
            scalar_scale = False
        else:
            raise ValueError("scale must be a positive scalar or vector.")
        if center is None:
            center_host = np.zeros((1 if scalar_scale else scale_host.size,), dtype=float)
        else:
            center_host = np.asarray(center, dtype=float)
        if (
            center_host.ndim != 1
            or center_host.size == 0
            or not np.all(np.isfinite(center_host))
        ):
            raise ValueError("center must be a finite vector.")
        self.child = child
        self.scale = jnp.asarray(scale_host, dtype=float)
        self.center = jnp.asarray(center_host, dtype=float)
        self.uniform = uniform
        self.feature_id = feature_id or f"scaling-{uuid4().hex}"

    def _compile(self, context):
        child = self.child._compile(context)
        scale = self.scale
        center = self.center
        if scale.shape == (1,):
            scale = jnp.repeat(scale, child.ambient_dimension)
            center = (
                jnp.zeros((child.ambient_dimension,), dtype=scale.dtype)
                if center.shape == (1,)
                else center
            )
        if scale.shape != (child.ambient_dimension,) or center.shape != (
            child.ambient_dimension,
        ):
            raise ValueError(
                "scale and center dimensions must match child ambient dimension."
            )
        scale_binding = context.bind(
            ParameterId(self.feature_id, "scale"), scale, role="scale", bounds=(0.0, None)
        )
        center_binding = context.bind(
            ParameterId(self.feature_id, "center"), center, role="position"
        )
        return _ScalingKernel(child, scale_binding, center_binding, uniform=self.uniform)


class _ScalingKernel(GeometryKernel):
    child: GeometryKernel
    scale: ParameterBinding = eqx.field(static=True)
    center: ParameterBinding = eqx.field(static=True)
    uniform: bool = eqx.field(static=True)

    def __init__(self, child, scale, center, *, uniform):
        self.child, self.scale, self.center, self.uniform = child, scale, center, uniform

    @property
    def ambient_dimension(self):
        return self.child.ambient_dimension

    @property
    def intrinsic_dimension(self):
        return self.child.intrinsic_dimension

    @property
    def kind(self):
        return self.child.kind

    @property
    def capabilities(self):
        capabilities = set(self.child.capabilities)
        if not self.uniform:
            capabilities.discard(GeometryCapability.SIGNED_DISTANCE)
            capabilities.discard(GeometryCapability.CUBATURE_ATLAS)
        return frozenset(capabilities)

    @property
    def field_certificate(self):
        certificate = self.child.field_certificate
        return FieldCertificate(
            certificate.zero_set_accuracy,
            certificate.sign_reliability,
            certificate.distance_semantics
            if self.uniform
            else DistanceSemantics.LEVEL_SET,
            certificate.regularity,
            certificate.safe_step_factor if self.uniform else None,
            certificate.validity_region,
            certificate.parameter_differentiable,
            (
                *certificate.provenance,
                "uniform_scale" if self.uniform else "nonuniform_scale",
            ),
        )

    def _parameters(self, state):
        return self.scale.read(state), self.center.read(state)

    def _local(self, state, points):
        scale, center = self._parameters(state)
        return center + (jnp.asarray(points) - center) / scale

    def boundary_field(self, state, points, /):
        scale, _ = self._parameters(state)
        return self.child.boundary_field(state, self._local(state, points)) * jnp.min(
            scale
        )

    def contains(self, state, points, /):
        return self.child.contains(state, self._local(state, points))

    def boundary_normal(self, state, points, /):
        scale, _ = self._parameters(state)
        normal = self.child.boundary_normal(state, self._local(state, points)) / scale
        return normal / jnp.linalg.norm(normal, axis=-1, keepdims=True)

    def bounds(self, state, /):
        scale, center = self._parameters(state)
        bounds = self.child.bounds(state)
        return center + (bounds - center) * scale

    def measure(self, state, /):
        scale, _ = self._parameters(state)
        return self.child.measure(state) * jnp.prod(scale)

    def boundary_measure(self, state, /):
        scale, _ = self._parameters(state)
        if self.uniform:
            return self.child.boundary_measure(state) * scale[0] ** (
                self.intrinsic_dimension - 1
            )
        atlas = self.boundary_atlas(state)
        rule = gauss_legendre_data(24)
        reference_axis = jnp.asarray(0.5 * (rule.nodes + 1.0))
        weights_axis = jnp.asarray(0.5 * rule.weights)
        if atlas.reference_dimension == 1:
            charts = jnp.repeat(jnp.arange(atlas.num_charts), reference_axis.size)
            reference = jnp.tile(reference_axis, atlas.num_charts)[:, None]
            quadrature = jnp.tile(weights_axis, atlas.num_charts)
        else:
            u, v = jnp.meshgrid(reference_axis, reference_axis, indexing="ij")
            wu, wv = jnp.meshgrid(weights_axis, weights_axis, indexing="ij")
            cell_reference = jnp.stack((u.reshape(-1), v.reshape(-1)), axis=-1)
            cell_weights = (wu * wv).reshape(-1)
            charts = jnp.repeat(jnp.arange(atlas.num_charts), cell_reference.shape[0])
            reference = jnp.tile(cell_reference, (atlas.num_charts, 1))
            quadrature = jnp.tile(cell_weights, atlas.num_charts)
        return jnp.sum(atlas.jacobian(charts, reference) * quadrature)

    def sample_interior(self, state, num_points, /, *, key, plan=None):
        scale, center = self._parameters(state)
        result = self.child.sample_interior(state, num_points, key=key, plan=plan)
        return SamplingResult(
            center + (result.points - center) * scale,
            result.valid,
            result.report,
            weights=result.weights,
            strata=result.strata,
        )

    def sample_boundary(self, state, num_points, /, *, key):
        atlas = self.boundary_atlas(state)
        candidate_count = max(8 * int(num_points), 64)
        chart_key, reference_key, choice_key = jr.split(key, 3)
        charts = jr.randint(chart_key, (candidate_count,), 0, atlas.num_charts)
        reference = jr.uniform(
            reference_key, (candidate_count, atlas.reference_dimension)
        )
        weights = atlas.jacobian(charts, reference)
        selected = jr.choice(
            choice_key,
            candidate_count,
            (int(num_points),),
            p=weights / jnp.sum(weights),
            replace=True,
        )
        points = atlas.map(charts[selected], reference[selected])
        from .._sampling import complete_sampling_result

        return complete_sampling_result(points)

    def boundary_atlas(self, state, /):
        scale, center = self._parameters(state)
        atlas = self.child.boundary_atlas(state)
        offset = center - scale * center
        return BoundaryAtlas(
            _AffineBoundaryMap(atlas.mapping, jnp.diag(scale), offset),
            source_entity_ids=atlas.source_entity_ids,
            source_id=atlas.source_id,
            physical_tags=atlas.physical_tags,
            orientation=atlas.orientation,
            seam_owner=atlas.seam_owner,
            trim_domains=atlas.trim_domains,
        )

    def cubature_atlas(self, state, component: CubatureComponent, /) -> CubatureAtlas:
        if not self.uniform:
            raise NotImplementedError(
                "Native cubature does not support nonuniform geometry scaling."
            )
        scale, center = self._parameters(state)
        atlas = self.child.cubature_atlas(state, component)
        offset = center - scale * center
        measure_dimension = (
            self.intrinsic_dimension
            if component == "interior"
            else self.intrinsic_dimension - 1
        )
        return CubatureAtlas(
            _AffineCubatureMap(
                atlas.mapping,
                jnp.diag(scale),
                offset,
                jnp.abs(scale[0]) ** measure_dimension,
            ),
            source_entity_ids=atlas.source_entity_ids,
            source_id=atlas.source_id,
            physical_tags=atlas.physical_tags,
        )


_CSGOperation = Literal["intersection", "difference"]


class SharpCSG(GeometrySource):
    """Sharp intersection or difference with exact set semantics."""

    children: tuple[GeometrySource, ...]
    operation: _CSGOperation = eqx.field(static=True)

    def __init__(self, children: tuple[GeometrySource, ...], operation: _CSGOperation):
        children_ = tuple(children)
        minimum = 2
        if len(children_) < minimum or not all(
            isinstance(child, GeometrySource) for child in children_
        ):
            raise ValueError("Sharp CSG requires at least two geometry sources.")
        if operation not in ("intersection", "difference"):
            raise ValueError("operation must be 'intersection' or 'difference'.")
        self.children = children_
        self.operation = operation

    def _compile(self, context):
        children = tuple(child._compile(context) for child in self.children)
        dimension = children[0].ambient_dimension
        if any(
            child.kind is not GeometryKind.REGION or child.ambient_dimension != dimension
            for child in children
        ):
            raise ValueError("CSG children must be regions in one ambient dimension.")
        return _SharpCSGKernel(children, operation=self.operation)


class _SharpCSGKernel(GeometryKernel):
    children: tuple[GeometryKernel, ...]
    operation: _CSGOperation = eqx.field(static=True)

    def __init__(self, children, *, operation):
        self.children, self.operation = children, operation

    @property
    def ambient_dimension(self):
        return self.children[0].ambient_dimension

    @property
    def intrinsic_dimension(self):
        return self.children[0].intrinsic_dimension

    @property
    def kind(self):
        return GeometryKind.REGION

    @property
    def capabilities(self):
        shared = set(self.children[0].capabilities)
        for child in self.children[1:]:
            shared.intersection_update(child.capabilities)
        shared.difference_update(
            {
                GeometryCapability.SIGNED_DISTANCE,
                GeometryCapability.MEASURE,
                GeometryCapability.BOUNDARY_SAMPLING,
                GeometryCapability.BOUNDARY_ATLAS,
            }
        )
        shared.update(
            {
                GeometryCapability.REGION_QUERY,
                GeometryCapability.BOUNDARY_NORMAL,
                GeometryCapability.INTERIOR_SAMPLING,
            }
        )
        return frozenset(shared)

    @property
    def field_certificate(self):
        certificates = tuple(child.field_certificate for child in self.children)
        return FieldCertificate(
            max(
                (item.zero_set_accuracy for item in certificates),
                key=lambda item: list(ZeroSetAccuracy).index(item),
            ),
            max(
                (item.sign_reliability for item in certificates),
                key=lambda item: list(SignReliability).index(item),
            ),
            DistanceSemantics.LEVEL_SET,
            FieldRegularity.NONSMOOTH,
            None,
            "all_space",
            all(item.parameter_differentiable for item in certificates),
            (
                *(entry for item in certificates for entry in item.provenance),
                f"sharp_{self.operation}",
            ),
        )

    def _fields(self, state, points):
        return jnp.stack(
            tuple(child.boundary_field(state, points) for child in self.children), axis=-1
        )

    def boundary_field(self, state, points, /):
        fields = self._fields(state, points)
        if self.operation == "intersection":
            return jnp.max(fields, axis=-1)
        return jnp.maximum(fields[..., 0], jnp.max(-fields[..., 1:], axis=-1))

    def contains(self, state, points, /):
        values = jnp.stack(
            tuple(child.contains(state, points) for child in self.children), axis=-1
        )
        if self.operation == "intersection":
            return jnp.all(values, axis=-1)
        return values[..., 0] & ~jnp.any(values[..., 1:], axis=-1)

    def boundary_normal(self, state, points, /):
        fields = self._fields(state, points)
        normals = jnp.stack(
            tuple(child.boundary_normal(state, points) for child in self.children),
            axis=-2,
        )
        if self.operation == "intersection":
            active = jnp.argmax(fields, axis=-1)
            signed_normals = normals
        else:
            transformed = jnp.concatenate((fields[..., :1], -fields[..., 1:]), axis=-1)
            active = jnp.argmax(transformed, axis=-1)
            signed_normals = normals.at[..., 1:, :].multiply(-1.0)
        return jnp.take_along_axis(signed_normals, active[..., None, None], axis=-2)[
            ..., 0, :
        ]

    def bounds(self, state, /):
        bounds = jnp.stack(tuple(child.bounds(state) for child in self.children))
        if self.operation == "intersection":
            return jnp.stack(
                (jnp.max(bounds[:, 0], axis=0), jnp.min(bounds[:, 1], axis=0))
            )
        return bounds[0]

    def measure(self, state, /):
        del state
        raise NotImplementedError("Sharp CSG measure requires a realization.")

    def boundary_measure(self, state, /):
        del state
        raise NotImplementedError("Sharp CSG boundary measure requires a realization.")

    def sample_interior(self, state, num_points, /, *, key, plan=None):
        bounds = self.bounds(state)
        plan_ = RejectionSamplingPlan() if plan is None else plan
        return bounded_rejection_sample(
            lambda proposal_key, count: jr.uniform(
                proposal_key,
                (count, self.ambient_dimension),
                minval=bounds[0],
                maxval=bounds[1],
                dtype=bounds.dtype,
            ),
            lambda points: self.contains(state, points),
            num_points=num_points,
            point_dimension=self.ambient_dimension,
            key=key,
            plan=plan_,
            dtype=bounds.dtype,
        )

    def sample_boundary(self, state, num_points, /, *, key):
        del state, num_points, key
        raise NotImplementedError("Sharp CSG boundary sampling requires realization.")

    def boundary_atlas(self, state, /):
        del state
        raise NotImplementedError("Sharp CSG boundary atlas requires realization.")


class BlendCSG(GeometrySource):
    """Smooth CSG expression with deliberately approximate zero-set semantics."""

    children: tuple[GeometrySource, ...]
    width: Array
    operation: Literal["union", "intersection", "difference"] = eqx.field(static=True)
    feature_id: str = eqx.field(static=True)

    def __init__(self, children, width, operation="union", *, feature_id=None):
        children_ = tuple(children)
        if len(children_) < 2 or not all(
            isinstance(child, GeometrySource) for child in children_
        ):
            raise ValueError("Blend CSG requires at least two geometry sources.")
        width_ = float(np.asarray(width))
        if not math.isfinite(width_) or width_ <= 0.0:
            raise ValueError("blend width must be finite and positive.")
        if operation not in ("union", "intersection", "difference"):
            raise ValueError("Unsupported blend operation.")
        self.children = children_
        self.width = jnp.asarray(width_, dtype=float)
        self.operation = operation
        self.feature_id = feature_id or f"blend-{uuid4().hex}"

    def _compile(self, context):
        children = tuple(child._compile(context) for child in self.children)
        dimension = children[0].ambient_dimension
        if any(
            child.kind is not GeometryKind.REGION or child.ambient_dimension != dimension
            for child in children
        ):
            raise ValueError("Blend children must be regions in one ambient dimension.")
        width = context.bind(
            ParameterId(self.feature_id, "width"),
            self.width,
            role="blend_width",
            bounds=(0.0, None),
        )
        return _BlendCSGKernel(children, width, operation=self.operation)


class _BlendCSGKernel(GeometryKernel):
    children: tuple[GeometryKernel, ...]
    width: ParameterBinding = eqx.field(static=True)
    operation: Literal["union", "intersection", "difference"] = eqx.field(static=True)

    def __init__(self, children, width, *, operation):
        self.children, self.width, self.operation = children, width, operation

    @property
    def ambient_dimension(self):
        return self.children[0].ambient_dimension

    @property
    def intrinsic_dimension(self):
        return self.children[0].intrinsic_dimension

    @property
    def kind(self):
        return GeometryKind.REGION

    @property
    def capabilities(self):
        return frozenset(
            {
                GeometryCapability.REGION_QUERY,
                GeometryCapability.BOUNDARY_NORMAL,
                GeometryCapability.INTERIOR_SAMPLING,
            }
        )

    @property
    def field_certificate(self):
        return FieldCertificate(
            ZeroSetAccuracy.APPROXIMATE,
            SignReliability.RELIABLE,
            DistanceSemantics.LEVEL_SET,
            FieldRegularity.SMOOTH,
            None,
            "all_space",
            True,
            (f"blend_{self.operation}",),
        )

    def _fields(self, state, points):
        return jnp.stack(
            tuple(child.boundary_field(state, points) for child in self.children), axis=-1
        )

    def boundary_field(self, state, points, /):
        fields = self._fields(state, points)
        width = self.width.read(state)
        if self.operation == "union":
            return -width * logsumexp(-fields / width, axis=-1)
        if self.operation == "intersection":
            return width * logsumexp(fields / width, axis=-1)
        transformed = jnp.concatenate((fields[..., :1], -fields[..., 1:]), axis=-1)
        return width * logsumexp(transformed / width, axis=-1)

    def contains(self, state, points, /):
        return self.boundary_field(state, points) <= 0.0

    def boundary_normal(self, state, points, /):
        points_ = jnp.asarray(points)
        flat = points_.reshape((-1, self.ambient_dimension))
        gradient = jax.vmap(jax.grad(lambda point: self.boundary_field(state, point)))(
            flat
        )
        gradient = gradient / jnp.maximum(
            jnp.linalg.norm(gradient, axis=-1, keepdims=True),
            jnp.finfo(points_.dtype).eps,
        )
        return gradient.reshape(points_.shape)

    def bounds(self, state, /):
        bounds = jnp.stack(tuple(child.bounds(state) for child in self.children))
        if self.operation == "union":
            return jnp.stack(
                (jnp.min(bounds[:, 0], axis=0), jnp.max(bounds[:, 1], axis=0))
            )
        if self.operation == "intersection":
            return jnp.stack(
                (jnp.max(bounds[:, 0], axis=0), jnp.min(bounds[:, 1], axis=0))
            )
        return bounds[0]

    def measure(self, state, /):
        del state
        raise NotImplementedError("Blend measure is estimator-only.")

    def boundary_measure(self, state, /):
        del state
        raise NotImplementedError("Blend boundary measure is estimator-only.")

    def sample_interior(self, state, num_points, /, *, key, plan=None):
        bounds = self.bounds(state)
        plan_ = RejectionSamplingPlan() if plan is None else plan
        return bounded_rejection_sample(
            lambda proposal_key, count: jr.uniform(
                proposal_key,
                (count, self.ambient_dimension),
                minval=bounds[0],
                maxval=bounds[1],
                dtype=bounds.dtype,
            ),
            lambda points: self.contains(state, points),
            num_points=num_points,
            point_dimension=self.ambient_dimension,
            key=key,
            plan=plan_,
            dtype=bounds.dtype,
        )

    def sample_boundary(self, state, num_points, /, *, key):
        del state, num_points, key
        raise NotImplementedError("Blend boundary sampling requires realization.")

    def boundary_atlas(self, state, /):
        del state
        raise NotImplementedError("Blend boundary atlas requires realization.")


def Intersection(*children: GeometrySource) -> SharpCSG:
    return SharpCSG(tuple(children), "intersection")


def Difference(base: GeometrySource, *subtract: GeometrySource) -> SharpCSG:
    return SharpCSG((base, *subtract), "difference")


def BlendUnion(*children: GeometrySource, width: float) -> BlendCSG:
    return BlendCSG(tuple(children), width, "union")


def BlendIntersection(*children: GeometrySource, width: float) -> BlendCSG:
    return BlendCSG(tuple(children), width, "intersection")


def BlendDifference(
    base: GeometrySource, *subtract: GeometrySource, width: float
) -> BlendCSG:
    return BlendCSG((base, *subtract), width, "difference")


__all__ = [
    "BlendCSG",
    "BlendDifference",
    "BlendIntersection",
    "BlendUnion",
    "Difference",
    "Intersection",
    "RigidFrame",
    "RigidTransform",
    "Scaling",
    "SharpCSG",
]
