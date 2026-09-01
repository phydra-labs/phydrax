#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Key

from .._atlas import BoundaryAtlas
from .._capabilities import GeometryCapability
from .._certificate import (
    DistanceSemantics,
    FieldCertificate,
    FieldRegularity,
)
from .._contracts import GeometryKernel, GeometryKind, GeometrySource
from .._sampling import RejectionSamplingPlan, SamplingReport, SamplingResult
from .._validity import (
    combine_validity,
    GeometryValidityEvidence,
    representation_validity,
)
from ..design._schema import (
    _ParameterCollector,
    DesignState,
    ParameterBinding,
    ParameterId,
)
from ._primitives import _check_points, _feature_id, _finite_norm


def _sweep_regularity(certificate: FieldCertificate) -> FieldRegularity:
    if certificate.regularity is FieldRegularity.NONSMOOTH:
        return FieldRegularity.NONSMOOTH
    return FieldRegularity.PIECEWISE_SMOOTH


def _extrusion_certificate(certificate: FieldCertificate) -> FieldCertificate:
    exact = certificate.distance_semantics is DistanceSemantics.EXACT
    return FieldCertificate(
        zero_set_accuracy=certificate.zero_set_accuracy,
        sign_reliability=certificate.sign_reliability,
        distance_semantics=(
            DistanceSemantics.EXACT if exact else DistanceSemantics.LEVEL_SET
        ),
        regularity=_sweep_regularity(certificate),
        safe_step_factor=1.0 if exact else None,
        validity_region=certificate.validity_region,
        parameter_differentiable=certificate.parameter_differentiable,
        provenance=(*certificate.provenance, "straight_extrusion"),
    )


def _revolution_certificate(certificate: FieldCertificate) -> FieldCertificate:
    exact = certificate.distance_semantics is DistanceSemantics.EXACT
    return FieldCertificate(
        zero_set_accuracy=certificate.zero_set_accuracy,
        sign_reliability=certificate.sign_reliability,
        distance_semantics=(
            DistanceSemantics.EXACT if exact else DistanceSemantics.LEVEL_SET
        ),
        regularity=_sweep_regularity(certificate),
        safe_step_factor=1.0 if exact else None,
        validity_region="profile remains in the non-negative radial half-plane",
        parameter_differentiable=certificate.parameter_differentiable,
        provenance=(*certificate.provenance, "axisymmetric_revolution"),
    )


class Extrusion(GeometrySource):
    """Centered straight extrusion of a two-dimensional region along local z."""

    profile: GeometrySource
    height: Array
    feature_id: str = eqx.field(static=True)

    def __init__(
        self,
        profile: GeometrySource,
        height: Any,
        *,
        feature_id: str | None = None,
    ):
        if not isinstance(profile, GeometrySource):
            raise TypeError("Extrusion.profile must be a GeometrySource.")
        height_host = np.asarray(height, dtype=float)
        if height_host.shape != () or not np.isfinite(height_host):
            raise ValueError("Extrusion.height must be a finite scalar.")
        if float(height_host) <= 0.0:
            raise ValueError("Extrusion.height must be positive.")
        self.profile = profile
        self.height = jnp.asarray(height_host, dtype=float)
        self.feature_id = _feature_id(feature_id, "extrusion")

    def _compile(self, context: _ParameterCollector, /) -> GeometryKernel:
        profile = self.profile._compile(context)
        if (
            profile.ambient_dimension != 2
            or profile.intrinsic_dimension != 2
            or profile.kind is not GeometryKind.REGION
        ):
            raise ValueError("Extrusion requires a two-dimensional region profile.")
        height = context.bind(
            ParameterId(self.feature_id, "height"),
            self.height,
            role="extrusion_height",
            bounds=(0.0, None),
        )
        return _ExtrusionKernel(profile, height)


class _ExtrusionKernel(GeometryKernel):
    profile: GeometryKernel
    height: ParameterBinding = eqx.field(static=True)

    def __init__(self, profile: GeometryKernel, height: ParameterBinding):
        self.profile = profile
        self.height = height

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
        capabilities = {
            GeometryCapability.REGION_QUERY,
            GeometryCapability.BOUNDARY_NORMAL,
        }
        if GeometryCapability.SIGNED_DISTANCE in self.profile.capabilities:
            capabilities.add(GeometryCapability.SIGNED_DISTANCE)
        if GeometryCapability.MEASURE in self.profile.capabilities:
            capabilities.add(GeometryCapability.MEASURE)
        if GeometryCapability.INTERIOR_SAMPLING in self.profile.capabilities:
            capabilities.add(GeometryCapability.INTERIOR_SAMPLING)
        boundary_requirements = {
            GeometryCapability.MEASURE,
            GeometryCapability.INTERIOR_SAMPLING,
            GeometryCapability.BOUNDARY_SAMPLING,
        }
        if boundary_requirements.issubset(self.profile.capabilities):
            capabilities.add(GeometryCapability.BOUNDARY_SAMPLING)
        return frozenset(capabilities)

    @property
    def field_certificate(self) -> FieldCertificate:
        return _extrusion_certificate(self.profile.field_certificate)

    def geometry_validity(self, state, /):
        height = self.height.read(state)
        local = GeometryValidityEvidence(
            finite=jnp.isfinite(height),
            conditions_satisfied=height > 0.0,
            resolved=True,
            margins=jnp.reshape(height, (1,)),
            margin_names=("height",),
            contract_id="straight_extrusion_height",
        )
        return combine_validity(
            (representation_validity(self.profile, state), local),
            contract_id="straight_extrusion",
        )

    def _height(self, state: DesignState) -> Array:
        return self.height.read(state)

    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        points_ = _check_points(points, 3)
        radial = self.profile.boundary_field(state, points_[..., :2])
        axial = jnp.abs(points_[..., 2]) - 0.5 * self._height(state)
        pair = jnp.stack((radial, axial), axis=-1)
        inside = jnp.minimum(jnp.maximum(radial, axial), 0.0)
        outside = _finite_norm(jnp.maximum(pair, 0.0))
        return inside + outside

    def contains(self, state: DesignState, points: Array, /) -> Array:
        points_ = _check_points(points, 3)
        return self.profile.contains(state, points_[..., :2]) & (
            jnp.abs(points_[..., 2]) <= 0.5 * self._height(state)
        )

    def boundary_normal(self, state: DesignState, points: Array, /) -> Array:
        points_ = _check_points(points, 3)
        leading = points_.shape[:-1]
        flat = points_.reshape((-1, 3))

        def field(point):
            return self.boundary_field(state, point[None, :])[0]

        gradient = jax.vmap(jax.grad(field))(flat)
        norm = _finite_norm(gradient).reshape((-1, 1))
        normal = gradient / jnp.maximum(norm, jnp.finfo(gradient.dtype).eps)
        return normal.reshape((*leading, 3))

    def bounds(self, state: DesignState, /) -> Array:
        profile_bounds = self.profile.bounds(state)
        half = 0.5 * self._height(state)
        return jnp.concatenate(
            (
                jnp.concatenate((profile_bounds[0], -half[None]))[None, :],
                jnp.concatenate((profile_bounds[1], half[None]))[None, :],
            ),
            axis=0,
        )

    def measure(self, state: DesignState, /) -> Array:
        return self.profile.measure(state) * self._height(state)

    def boundary_measure(self, state: DesignState, /) -> Array:
        return 2.0 * self.profile.measure(state) + self.profile.boundary_measure(
            state
        ) * self._height(state)

    def sample_interior(
        self,
        state: DesignState,
        num_points: int,
        /,
        *,
        key: Key[Array, ""],
        plan: RejectionSamplingPlan | None = None,
    ) -> SamplingResult:
        profile_key, axial_key = jr.split(key)
        result = self.profile.sample_interior(
            state,
            num_points,
            key=profile_key,
            plan=plan,
        )
        half = 0.5 * self._height(state)
        axial = jr.uniform(
            axial_key,
            (int(num_points), 1),
            minval=-half,
            maxval=half,
            dtype=result.points.dtype,
        )
        return SamplingResult(
            jnp.concatenate((result.points, axial), axis=-1),
            result.valid,
            result.report,
            weights=result.weights,
            strata=result.strata,
        )

    def sample_boundary(
        self,
        state: DesignState,
        num_points: int,
        /,
        *,
        key: Key[Array, ""],
    ) -> SamplingResult:
        side_key, cap_key, axial_key, choose_key, sign_key = jr.split(key, 5)
        count = int(num_points)
        side = self.profile.sample_boundary(state, count, key=side_key)
        cap = self.profile.sample_interior(state, count, key=cap_key)
        half = 0.5 * self._height(state)
        side_axial = jr.uniform(
            axial_key,
            (count, 1),
            minval=-half,
            maxval=half,
            dtype=side.points.dtype,
        )
        signs = jnp.where(jr.bernoulli(sign_key, shape=(count,)), 1.0, -1.0)
        cap_axial = (half * signs)[:, None]
        side_points = jnp.concatenate((side.points, side_axial), axis=-1)
        cap_points = jnp.concatenate((cap.points, cap_axial), axis=-1)
        side_area = self.profile.boundary_measure(state) * self._height(state)
        cap_area = 2.0 * self.profile.measure(state)
        choose_side = jr.bernoulli(
            choose_key,
            side_area / jnp.maximum(side_area + cap_area, jnp.finfo(float).eps),
            shape=(count,),
        )
        points = jnp.where(choose_side[:, None], side_points, cap_points)
        valid = jnp.where(choose_side, side.valid, cap.valid)
        report = SamplingReport(
            requested=count,
            proposed=side.report.proposed + cap.report.proposed,
            accepted=jnp.sum(valid, dtype=jnp.int32),
            rounds=jnp.maximum(side.report.rounds, cap.report.rounds),
        )
        return SamplingResult(points, valid, report)

    def boundary_atlas(self, state: DesignState, /) -> BoundaryAtlas:
        del state
        raise NotImplementedError(
            "Extrusion boundary atlas requires explicit cap and side chart maps."
        )


class Revolution(GeometrySource):
    """Axisymmetric revolution of a radial/axial profile around local z."""

    profile: GeometrySource
    feature_id: str = eqx.field(static=True)

    def __init__(
        self,
        profile: GeometrySource,
        *,
        feature_id: str | None = None,
    ):
        if not isinstance(profile, GeometrySource):
            raise TypeError("Revolution.profile must be a GeometrySource.")
        self.profile = profile
        self.feature_id = _feature_id(feature_id, "revolution")

    def _compile(self, context: _ParameterCollector, /) -> GeometryKernel:
        profile = self.profile._compile(context)
        if (
            profile.ambient_dimension != 2
            or profile.intrinsic_dimension != 2
            or profile.kind is not GeometryKind.REGION
        ):
            raise ValueError("Revolution requires a two-dimensional region profile.")
        return _RevolutionKernel(profile)


class _RevolutionKernel(GeometryKernel):
    profile: GeometryKernel

    def __init__(self, profile: GeometryKernel):
        self.profile = profile

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
        capabilities = {
            GeometryCapability.REGION_QUERY,
            GeometryCapability.BOUNDARY_NORMAL,
        }
        if GeometryCapability.SIGNED_DISTANCE in self.profile.capabilities:
            capabilities.add(GeometryCapability.SIGNED_DISTANCE)
        return frozenset(capabilities)

    @property
    def field_certificate(self) -> FieldCertificate:
        return _revolution_certificate(self.profile.field_certificate)

    def geometry_validity(self, state, /):
        bounds = self.profile.bounds(state)
        radial_margin = bounds[0, 0]
        local = GeometryValidityEvidence(
            finite=jnp.all(jnp.isfinite(bounds)),
            conditions_satisfied=radial_margin >= 0.0,
            resolved=True,
            margins=jnp.reshape(radial_margin, (1,)),
            margin_names=("minimum_profile_radius",),
            contract_id="axisymmetric_profile_half_plane",
        )
        return combine_validity(
            (representation_validity(self.profile, state), local),
            contract_id="axisymmetric_revolution",
        )

    def _profile_points(self, points: Array) -> Array:
        points_ = _check_points(points, 3)
        radius = _finite_norm(points_[..., :2])
        return jnp.stack((radius, points_[..., 2]), axis=-1)

    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        return self.profile.boundary_field(state, self._profile_points(points))

    def contains(self, state: DesignState, points: Array, /) -> Array:
        return self.profile.contains(state, self._profile_points(points))

    def boundary_normal(self, state: DesignState, points: Array, /) -> Array:
        points_ = _check_points(points, 3)
        leading = points_.shape[:-1]
        flat = points_.reshape((-1, 3))

        def field(point):
            return self.boundary_field(state, point[None, :])[0]

        gradient = jax.vmap(jax.grad(field))(flat)
        norm = _finite_norm(gradient).reshape((-1, 1))
        normal = gradient / jnp.maximum(norm, jnp.finfo(gradient.dtype).eps)
        return normal.reshape((*leading, 3))

    def bounds(self, state: DesignState, /) -> Array:
        profile_bounds = self.profile.bounds(state)
        radial_maximum = jnp.maximum(profile_bounds[1, 0], 0.0)
        lower = jnp.asarray(
            (-radial_maximum, -radial_maximum, profile_bounds[0, 1]),
            dtype=profile_bounds.dtype,
        )
        upper = jnp.asarray(
            (radial_maximum, radial_maximum, profile_bounds[1, 1]),
            dtype=profile_bounds.dtype,
        )
        return jnp.stack((lower, upper))

    def measure(self, state: DesignState, /) -> Array:
        del state
        raise NotImplementedError(
            "Revolution measure requires radial-moment cubature of the profile."
        )

    def boundary_measure(self, state: DesignState, /) -> Array:
        del state
        raise NotImplementedError(
            "Revolution boundary measure requires radial-moment boundary cubature."
        )

    def sample_interior(
        self,
        state: DesignState,
        num_points: int,
        /,
        *,
        key: Key[Array, ""],
        plan: RejectionSamplingPlan | None = None,
    ) -> SamplingResult:
        del state, num_points, key, plan
        raise NotImplementedError(
            "Uniform revolution sampling requires radial Jacobian weighting."
        )

    def sample_boundary(
        self,
        state: DesignState,
        num_points: int,
        /,
        *,
        key: Key[Array, ""],
    ) -> SamplingResult:
        del state, num_points, key
        raise NotImplementedError(
            "Uniform revolution boundary sampling requires radial weighting."
        )

    def boundary_atlas(self, state: DesignState, /) -> BoundaryAtlas:
        del state
        raise NotImplementedError(
            "Revolution boundary atlas requires explicit axis-aware charts."
        )


__all__ = ["Extrusion", "Revolution"]
