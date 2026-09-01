#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any
from uuid import uuid4

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Key

from .._atlas import BoundaryAtlas
from .._capabilities import GeometryCapability
from .._certificate import FieldCertificate, sharp_union_certificate
from .._contracts import GeometryKernel, GeometryKind, GeometrySource
from .._cubature import CubatureAtlas, CubatureComponent
from .._sampling import (
    bounded_rejection_sample,
    RejectionSamplingPlan,
    SamplingResult,
)
from .._validity import combine_validity, representation_validity
from ..design._schema import (
    _ParameterCollector,
    DesignState,
    ParameterBinding,
    ParameterId,
)


class Translation(GeometrySource):
    """Immutable translation expression preserving child geometry semantics."""

    child: GeometrySource
    offset: Array
    feature_id: str = eqx.field(static=True)

    def __init__(
        self,
        child: GeometrySource,
        offset: Any,
        *,
        feature_id: str | None = None,
    ):
        if not isinstance(child, GeometrySource):
            raise TypeError("Translation.child must be a GeometrySource.")
        offset_ = np.asarray(offset, dtype=float)
        if offset_.ndim != 1 or offset_.size == 0:
            raise ValueError("Translation.offset must be a non-empty vector.")
        if not np.all(np.isfinite(offset_)):
            raise ValueError("Translation.offset must contain only finite values.")
        if feature_id is not None and not feature_id:
            raise ValueError("feature_id must be non-empty.")
        self.child = child
        self.offset = jnp.asarray(offset_, dtype=float)
        self.feature_id = feature_id or f"translation-{uuid4().hex}"

    def _compile(self, context: _ParameterCollector, /) -> GeometryKernel:
        child = self.child._compile(context)
        if self.offset.shape != (child.ambient_dimension,):
            raise ValueError(
                f"Translation offset must have shape ({child.ambient_dimension},)."
            )
        offset = context.bind(
            ParameterId(self.feature_id, "offset"),
            self.offset,
            role="position_offset",
        )
        return _TranslationKernel(child, offset)


class _TranslationKernel(GeometryKernel):
    child: GeometryKernel
    offset: ParameterBinding = eqx.field(static=True)

    def __init__(self, child: GeometryKernel, offset: ParameterBinding):
        self.child = child
        self.offset = offset

    @property
    def ambient_dimension(self) -> int:
        return self.child.ambient_dimension

    @property
    def intrinsic_dimension(self) -> int:
        return self.child.intrinsic_dimension

    @property
    def kind(self) -> GeometryKind:
        return self.child.kind

    @property
    def capabilities(self) -> frozenset[GeometryCapability]:
        return self.child.capabilities

    @property
    def field_certificate(self) -> FieldCertificate:
        return self.child.field_certificate.translated()

    def geometry_validity(self, state, /):
        return representation_validity(self.child, state)

    def _offset(self, state: DesignState) -> Array:
        return self.offset.read(state)

    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        return self.child.boundary_field(state, jnp.asarray(points) - self._offset(state))

    def contains(self, state: DesignState, points: Array, /) -> Array:
        return self.child.contains(state, jnp.asarray(points) - self._offset(state))

    def boundary_normal(self, state: DesignState, points: Array, /) -> Array:
        return self.child.boundary_normal(
            state,
            jnp.asarray(points) - self._offset(state),
        )

    def bounds(self, state: DesignState, /) -> Array:
        return self.child.bounds(state) + self._offset(state)

    def measure(self, state: DesignState, /) -> Array:
        return self.child.measure(state)

    def boundary_measure(self, state: DesignState, /) -> Array:
        return self.child.boundary_measure(state)

    def sample_interior(
        self,
        state: DesignState,
        num_points: int,
        /,
        *,
        key: Key[Array, ""],
        plan: RejectionSamplingPlan | None = None,
    ) -> SamplingResult:
        result = self.child.sample_interior(state, num_points, key=key, plan=plan)
        return SamplingResult(
            result.points + self._offset(state),
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
        result = self.child.sample_boundary(state, num_points, key=key)
        return SamplingResult(
            result.points + self._offset(state),
            result.valid,
            result.report,
            weights=result.weights,
            strata=result.strata,
        )

    def boundary_atlas(self, state: DesignState, /) -> BoundaryAtlas:
        return self.child.boundary_atlas(state).translated(self._offset(state))

    def cubature_atlas(
        self, state: DesignState, component: CubatureComponent, /
    ) -> CubatureAtlas:
        return self.child.cubature_atlas(state, component).translated(self._offset(state))


class Union(GeometrySource):
    """Sharp set-theoretic union of region sources."""

    children: tuple[GeometrySource, ...]
    feature_id: str = eqx.field(static=True)

    def __init__(
        self,
        children: tuple[GeometrySource, ...],
        *,
        feature_id: str | None = None,
    ):
        children_ = tuple(children)
        if len(children_) < 2:
            raise ValueError("Union requires at least two child sources.")
        if not all(isinstance(child, GeometrySource) for child in children_):
            raise TypeError("Every Union child must be a GeometrySource.")
        if feature_id is not None and not feature_id:
            raise ValueError("feature_id must be non-empty.")
        self.children = children_
        self.feature_id = feature_id or f"union-{uuid4().hex}"

    def _compile(self, context: _ParameterCollector, /) -> GeometryKernel:
        children = tuple(child._compile(context) for child in self.children)
        dimension = children[0].ambient_dimension
        intrinsic_dimension = children[0].intrinsic_dimension
        if any(child.ambient_dimension != dimension for child in children[1:]):
            raise ValueError("Union children must share an ambient dimension.")
        if any(
            child.intrinsic_dimension != intrinsic_dimension for child in children[1:]
        ):
            raise ValueError("Union children must share an intrinsic dimension.")
        if any(child.kind is not GeometryKind.REGION for child in children):
            raise ValueError("Sharp union is defined only for region kernels.")
        return _UnionKernel(children)


class _UnionKernel(GeometryKernel):
    children: tuple[GeometryKernel, ...]

    def __init__(self, children: tuple[GeometryKernel, ...]):
        self.children = children

    @property
    def ambient_dimension(self) -> int:
        return self.children[0].ambient_dimension

    @property
    def intrinsic_dimension(self) -> int:
        return self.children[0].intrinsic_dimension

    @property
    def kind(self) -> GeometryKind:
        return GeometryKind.REGION

    @property
    def capabilities(self) -> frozenset[GeometryCapability]:
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
    def field_certificate(self) -> FieldCertificate:
        return sharp_union_certificate(
            tuple(child.field_certificate for child in self.children)
        )

    def geometry_validity(self, state, /):
        return combine_validity(
            tuple(representation_validity(child, state) for child in self.children),
            contract_id="sharp_union",
        )

    def _fields(self, state: DesignState, points: Array) -> Array:
        return jnp.stack(
            tuple(child.boundary_field(state, points) for child in self.children),
            axis=-1,
        )

    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        return jnp.min(self._fields(state, points), axis=-1)

    def contains(self, state: DesignState, points: Array, /) -> Array:
        return jnp.any(
            jnp.stack(
                tuple(child.contains(state, points) for child in self.children),
                axis=-1,
            ),
            axis=-1,
        )

    def boundary_normal(self, state: DesignState, points: Array, /) -> Array:
        fields = self._fields(state, points)
        normals = jnp.stack(
            tuple(child.boundary_normal(state, points) for child in self.children),
            axis=-2,
        )
        active = jnp.argmin(fields, axis=-1)
        selected = jnp.take_along_axis(
            normals,
            active[..., None, None],
            axis=-2,
        )
        return selected[..., 0, :]

    def bounds(self, state: DesignState, /) -> Array:
        bounds = jnp.stack(tuple(child.bounds(state) for child in self.children))
        return jnp.stack((jnp.min(bounds[:, 0], axis=0), jnp.max(bounds[:, 1], axis=0)))

    def measure(self, state: DesignState, /) -> Array:
        del state
        raise NotImplementedError("A general sharp union has no closed-form measure.")

    def boundary_measure(self, state: DesignState, /) -> Array:
        del state
        raise NotImplementedError(
            "A general sharp union has no closed-form boundary measure."
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
        bounds = self.bounds(state)
        plan_ = RejectionSamplingPlan() if plan is None else plan

        def proposal(proposal_key, count):
            return jr.uniform(
                proposal_key,
                shape=(count, self.ambient_dimension),
                minval=bounds[0],
                maxval=bounds[1],
                dtype=bounds.dtype,
            )

        def accept(points):
            return self.contains(state, points)

        return bounded_rejection_sample(
            proposal,
            accept,
            num_points=num_points,
            point_dimension=self.ambient_dimension,
            key=key,
            plan=plan_,
            dtype=bounds.dtype,
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
            "Sharp-union boundary sampling requires an explicit boundary realization."
        )

    def boundary_atlas(self, state: DesignState, /) -> BoundaryAtlas:
        del state
        raise NotImplementedError(
            "Sharp-union integration requires an explicit boundary realization."
        )


__all__ = ["Translation", "Union"]
