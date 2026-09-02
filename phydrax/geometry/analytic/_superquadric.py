#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array

from phydrax.ein import contract

from .._atlas import BoundaryAtlas
from .._capabilities import GeometryCapability
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
from .._sampling import (
    bounded_rejection_sample,
    complete_sampling_result,
    RejectionSamplingPlan,
)
from .._validity import GeometryValidityEvidence
from ..design._schema import (
    _ParameterCollector,
    DesignState,
    ParameterBinding,
    ParameterId,
)
from ._primitives import _check_points, _feature_id


_SUPERQUADRIC_CAPABILITIES = frozenset(
    {
        GeometryCapability.REGION_QUERY,
        GeometryCapability.BOUNDARY_NORMAL,
        GeometryCapability.CONTACT_CURVATURE,
        GeometryCapability.SUPPORT_MAP,
        GeometryCapability.MEASURE,
        GeometryCapability.INTERIOR_SAMPLING,
        GeometryCapability.BOUNDARY_SAMPLING,
    }
)
_SUPERQUADRIC_CERTIFICATE = FieldCertificate(
    zero_set_accuracy=ZeroSetAccuracy.EXACT,
    sign_reliability=SignReliability.RELIABLE,
    distance_semantics=DistanceSemantics.LEVEL_SET,
    regularity=FieldRegularity.SMOOTH,
    safe_step_factor=None,
    validity_region="convex blockiness exponents at least two",
    parameter_differentiable=True,
    provenance=("analytic_superquadric_level_set",),
)


class Superquadric(GeometrySource):
    """Smooth convex superquadric with certified level-set contact geometry."""

    center: Array
    semi_axes: Array
    orientation: Array
    first_blockiness: Array
    second_blockiness: Array
    feature_id: str = eqx.field(static=True)

    def __init__(
        self,
        center: Any,
        semi_axes: Any,
        /,
        *,
        orientation: Any = (1.0, 0.0, 0.0, 0.0),
        first_blockiness: Any = 2.0,
        second_blockiness: Any = 2.0,
        feature_id: str | None = None,
    ):
        center_ = np.asarray(center, dtype=float)
        axes = np.asarray(semi_axes, dtype=float)
        quaternion = np.asarray(orientation, dtype=float)
        first = np.asarray(first_blockiness, dtype=float)
        second = np.asarray(second_blockiness, dtype=float)
        if center_.shape != (3,) or np.any(~np.isfinite(center_)):
            raise ValueError("center must be a finite three-vector.")
        if axes.shape != (3,) or np.any(~np.isfinite(axes)) or np.any(axes <= 0.0):
            raise ValueError("semi_axes must be a finite positive three-vector.")
        if quaternion.shape != (4,) or np.any(~np.isfinite(quaternion)):
            raise ValueError("orientation must be a finite quaternion.")
        norm = float(np.linalg.norm(quaternion))
        if norm <= 0.0:
            raise ValueError("orientation quaternion must be nonzero.")
        quaternion = quaternion / norm
        if first.shape != () or second.shape != ():
            raise ValueError("blockiness values must be scalars.")
        if (
            not np.isfinite(float(first))
            or not np.isfinite(float(second))
            or float(first) < 2.0
            or float(second) < 2.0
        ):
            raise ValueError(
                "Convex smooth superquadric blockiness must be at least two."
            )
        self.center = jnp.asarray(center_)
        self.semi_axes = jnp.asarray(axes)
        self.orientation = jnp.asarray(quaternion)
        self.first_blockiness = jnp.asarray(first)
        self.second_blockiness = jnp.asarray(second)
        self.feature_id = _feature_id(feature_id, "superquadric")

    def _compile(self, context: _ParameterCollector, /) -> GeometryKernel:
        center = context.bind(
            ParameterId(self.feature_id, "center"), self.center, role="position"
        )
        axes = context.bind(
            ParameterId(self.feature_id, "semi_axes"),
            self.semi_axes,
            role="length",
            physical_scale=float(jnp.min(self.semi_axes)),
            bounds=(0.0, None),
        )
        orientation = context.bind(
            ParameterId(self.feature_id, "orientation"),
            self.orientation,
            role="orientation",
        )
        first = context.bind(
            ParameterId(self.feature_id, "first_blockiness"),
            self.first_blockiness,
            role="shape",
            bounds=(2.0, None),
        )
        second = context.bind(
            ParameterId(self.feature_id, "second_blockiness"),
            self.second_blockiness,
            role="shape",
            bounds=(2.0, None),
        )
        return _SuperquadricKernel(
            center,
            axes,
            orientation,
            first,
            second,
            source_id=self.feature_id,
        )


class _SuperquadricKernel(GeometryKernel):
    center: ParameterBinding = eqx.field(static=True)
    semi_axes: ParameterBinding = eqx.field(static=True)
    orientation: ParameterBinding = eqx.field(static=True)
    first_blockiness: ParameterBinding = eqx.field(static=True)
    second_blockiness: ParameterBinding = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        center,
        semi_axes,
        orientation,
        first_blockiness,
        second_blockiness,
        *,
        source_id,
    ):
        self.center = center
        self.semi_axes = semi_axes
        self.orientation = orientation
        self.first_blockiness = first_blockiness
        self.second_blockiness = second_blockiness
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
        return _SUPERQUADRIC_CAPABILITIES

    @property
    def field_certificate(self) -> FieldCertificate:
        return _SUPERQUADRIC_CERTIFICATE

    def _parameters(self, state):
        orientation = self.orientation.read(state)
        orientation = orientation / jnp.maximum(
            jnp.linalg.norm(orientation), jnp.finfo(orientation.dtype).eps
        )
        return (
            self.center.read(state),
            self.semi_axes.read(state),
            orientation,
            self.first_blockiness.read(state),
            self.second_blockiness.read(state),
        )

    def geometry_validity(self, state, /) -> GeometryValidityEvidence:
        axes = self.semi_axes.read(state)
        orientation = self.orientation.read(state)
        first = self.first_blockiness.read(state)
        second = self.second_blockiness.read(state)
        orientation_norm = jnp.linalg.norm(orientation)
        margins = jnp.stack(
            (
                jnp.min(axes),
                orientation_norm - jnp.finfo(orientation.dtype).eps,
                first - 2.0,
                second - 2.0,
            )
        )
        finite = jnp.all(jnp.isfinite(margins))
        return GeometryValidityEvidence(
            finite=finite,
            conditions_satisfied=jnp.all(margins >= 0.0),
            resolved=True,
            margins=margins,
            margin_names=(
                "semi_axes_minimum",
                "orientation_norm",
                "first_blockiness",
                "second_blockiness",
            ),
            contract_id="analytic_superquadric",
        )

    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        points_ = _check_points(points, 3)
        center, axes, orientation, first, second = self._parameters(state)
        rotation = quaternion_rotation_matrix(orientation)
        local = (points_ - center) @ rotation
        value = superquadric_norm(local, axes, first, second) - 1.0
        return value * jnp.min(axes)

    def contains(self, state, points, /):
        return self.boundary_field(state, points) <= 0.0

    def boundary_normal(self, state, points, /):
        points_ = _check_points(points, 3)

        def field(point):
            return self.boundary_field(state, point[None, :])[0]

        gradient = jax.vmap(jax.grad(field))(points_)
        norm = jnp.linalg.norm(gradient, axis=-1, keepdims=True)
        return gradient / jnp.maximum(norm, jnp.finfo(points_.dtype).eps)

    def contact_curvature(self, state, points, /):
        points_ = _check_points(points, 3)
        _, axes, _, _, _ = self._parameters(state)

        def field(point):
            return self.boundary_field(state, point[None, :])[0]

        gradient_function = jax.grad(field)
        gradient = jax.vmap(gradient_function)(points_)
        step = jnp.sqrt(jnp.finfo(points_.dtype).eps) * jnp.minimum(jnp.min(axes), 1.0)
        offsets = step * jnp.eye(3, dtype=points_.dtype)

        def point_hessian(point):
            plus = jax.vmap(gradient_function)(point[None, :] + offsets)
            minus = jax.vmap(gradient_function)(point[None, :] - offsets)
            value = ((plus - minus) / (2.0 * step)).T
            return 0.5 * (value + value.T)

        hessian = jax.vmap(point_hessian)(points_)
        gradient_norm = jnp.linalg.norm(gradient, axis=-1)
        normal = gradient / jnp.maximum(gradient_norm[:, None], 1.0e-30)
        identity = jnp.eye(3, dtype=points_.dtype)
        projector = identity[None, :, :] - normal[:, :, None] * normal[:, None, :]
        shape = contract("nij,njk,nkl->nil", projector, hessian, projector)
        shape = shape / jnp.maximum(gradient_norm[:, None, None], 1.0e-30)
        eigenvalues = jnp.linalg.eigvalsh(shape)
        principal = eigenvalues[:, 1:]
        finite = jnp.all(jnp.isfinite(principal), axis=-1)
        valid = finite & (gradient_norm > 1.0e-10) & (step > 0.0)
        margin = jnp.minimum(gradient_norm, jnp.broadcast_to(step, gradient_norm.shape))
        return ContactCurvatureResult(principal, valid, margin)

    def support_map(self, state, directions: Array, /) -> Array:
        direction = _check_points(directions, 3)
        center, axes, orientation, first, second = self._parameters(state)
        rotation = quaternion_rotation_matrix(orientation)
        local_direction = direction @ rotation

        def dual_norm(value):
            first_dual = first / (first - 1.0)
            second_dual = second / (second - 1.0)
            scaled = axes * value
            planar = (
                jnp.abs(scaled[0]) ** first_dual + jnp.abs(scaled[1]) ** first_dual
            ) ** (second_dual / first_dual)
            return (planar + jnp.abs(scaled[2]) ** second_dual) ** (1.0 / second_dual)

        local_support = jax.vmap(jax.grad(dual_norm))(local_direction)
        return center + local_support @ rotation.T

    def bounds(self, state, /):
        center, axes, orientation, _, _ = self._parameters(state)
        rotation = quaternion_rotation_matrix(orientation)
        extent = jnp.abs(rotation) @ axes
        return jnp.stack((center - extent, center + extent))

    def measure(self, state, /):
        _, axes, _, first, second = self._parameters(state)
        log_area_factor = (
            jnp.log(4.0)
            + 2.0 * jsp.special.gammaln(1.0 + 1.0 / first)
            - jsp.special.gammaln(1.0 + 2.0 / first)
        )
        log_integral = (
            jsp.special.gammaln(1.0 + 1.0 / second)
            + jsp.special.gammaln(1.0 + 2.0 / second)
            - jsp.special.gammaln(1.0 + 3.0 / second)
        )
        return 2.0 * jnp.prod(axes) * jnp.exp(log_area_factor + log_integral)

    def boundary_measure(self, state, /):
        del state
        raise NotImplementedError(
            "Superquadric boundary measure requires an explicit cubature plan."
        )

    def sample_interior(self, state, num_points, /, *, key, plan=None):
        bounds = self.bounds(state)
        selected = RejectionSamplingPlan() if plan is None else plan

        def proposal(proposal_key, count):
            return jr.uniform(
                proposal_key,
                shape=(count, 3),
                minval=bounds[0],
                maxval=bounds[1],
                dtype=bounds.dtype,
            )

        return bounded_rejection_sample(
            proposal,
            lambda points: self.contains(state, points),
            num_points=int(num_points),
            point_dimension=3,
            key=key,
            plan=selected,
            dtype=bounds.dtype,
        )

    def sample_boundary(self, state, num_points, /, *, key):
        center, axes, orientation, first, second = self._parameters(state)
        count = int(num_points)
        direction = jr.normal(key, (count, 3), dtype=center.dtype)
        direction = direction / jnp.maximum(
            jnp.linalg.norm(direction, axis=-1, keepdims=True),
            jnp.finfo(center.dtype).eps,
        )
        scale = 1.0 / superquadric_norm(direction, axes, first, second)
        rotation = quaternion_rotation_matrix(orientation)
        return complete_sampling_result(
            center + (scale[:, None] * direction) @ rotation.T
        )

    def boundary_atlas(self, state, /) -> BoundaryAtlas:
        del state
        raise NotImplementedError(
            "Superquadric boundary atlas requires a dedicated chart plan."
        )


def superquadric_norm(local, semi_axes, first_blockiness, second_blockiness):
    scaled = local / semi_axes
    planar = (
        _even_power(scaled[..., 0], first_blockiness)
        + _even_power(scaled[..., 1], first_blockiness)
    ) ** (second_blockiness / first_blockiness)
    return (planar + _even_power(scaled[..., 2], second_blockiness)) ** (
        1.0 / second_blockiness
    )


def _even_power(value, exponent):
    return jnp.square(value) ** (0.5 * exponent)


def quaternion_rotation_matrix(quaternion):
    q = quaternion / jnp.maximum(jnp.linalg.norm(quaternion), 1.0e-30)
    w, x, y, z = q
    return jnp.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=q.dtype,
    )


__all__ = ["Superquadric", "quaternion_rotation_matrix", "superquadric_norm"]
