#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Key

from .._strict import StrictModule
from ._capabilities import ContactCurvatureProvider, GeometryCapability
from ._certificate import FieldCertificate
from .design._schema import (
    _ParameterCollector,
    DesignState,
    ParameterId,
    ParameterSchema,
)


if TYPE_CHECKING:
    from ._atlas import BoundaryAtlas
    from ._cubature import CubatureAtlas, CubatureComponent
    from ._sampling import RejectionSamplingPlan, SamplingResult


class GeometryKind(str, Enum):
    """Measure-theoretic role of a geometry representation."""

    REGION = "region"
    MANIFOLD = "manifold"
    POINT_SET = "point_set"


@dataclass(frozen=True, slots=True)
class GeometryTolerance:
    """Scale-aware tolerance used only for approximate boundary predicates."""

    absolute: float = 1e-10
    relative: float = 1e-8

    def __post_init__(self):
        if not np.isfinite(self.absolute) or self.absolute < 0.0:
            raise ValueError(
                "GeometryTolerance.absolute must be finite and non-negative."
            )
        if not np.isfinite(self.relative) or self.relative < 0.0:
            raise ValueError(
                "GeometryTolerance.relative must be finite and non-negative."
            )

    def threshold(self, scale: Array, /) -> Array:
        scale_ = jnp.asarray(scale, dtype=float)
        return self.absolute + self.relative * jnp.maximum(scale_, 1.0)


class ContactCurvatureResult(StrictModule):
    principal_curvatures: Array
    valid: Array
    regularity_margin: Array

    def __init__(
        self,
        principal_curvatures: Array,
        valid: Array,
        regularity_margin: Array,
        /,
    ):
        curvature = jnp.asarray(principal_curvatures)
        valid_ = jnp.asarray(valid, dtype=bool)
        margin = jnp.asarray(regularity_margin, dtype=curvature.dtype)
        if curvature.ndim != 2 or curvature.shape[1] not in (1, 2):
            raise ValueError(
                "Principal curvatures must have shape (points, dimension-1)."
            )
        if valid_.shape != curvature.shape[:1] or margin.shape != valid_.shape:
            raise ValueError("Curvature validity and margins must have point shape.")
        self.principal_curvatures = curvature
        self.valid = valid_
        self.regularity_margin = margin


class AbstractGeometryKernel(StrictModule):
    """Pure JAX geometry program evaluated against a dynamic design state."""

    @property
    @abstractmethod
    def ambient_dimension(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def intrinsic_dimension(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def kind(self) -> GeometryKind:
        raise NotImplementedError

    @property
    @abstractmethod
    def capabilities(self) -> frozenset[GeometryCapability]:
        raise NotImplementedError

    @property
    @abstractmethod
    def field_certificate(self) -> FieldCertificate:
        raise NotImplementedError

    @abstractmethod
    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def contains(self, state: DesignState, points: Array, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def boundary_normal(self, state: DesignState, points: Array, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def bounds(self, state: DesignState, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def measure(self, state: DesignState, /) -> Array:
        del state
        raise NotImplementedError(f"{type(self).__name__} does not provide measure.")

    @abstractmethod
    def boundary_measure(self, state: DesignState, /) -> Array:
        del state
        raise NotImplementedError(
            f"{type(self).__name__} does not provide boundary measure."
        )

    @abstractmethod
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
            f"{type(self).__name__} does not provide interior sampling."
        )

    @abstractmethod
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
            f"{type(self).__name__} does not provide boundary sampling."
        )

    @abstractmethod
    def boundary_atlas(self, state: DesignState, /) -> BoundaryAtlas:
        del state
        raise NotImplementedError(
            f"{type(self).__name__} does not provide a boundary atlas."
        )


class CompiledGeometry(StrictModule):
    """JAX-safe kernel, dynamic state, schema, and tolerance bundle."""

    kernel: GeometryKernel
    state: DesignState
    schema: ParameterSchema = eqx.field(static=True)
    tolerance: GeometryTolerance = eqx.field(static=True)

    def __init__(
        self,
        kernel: GeometryKernel,
        state: DesignState,
        *,
        tolerance: GeometryTolerance = GeometryTolerance(),
    ):
        self.kernel = kernel
        self.state = state
        self.schema = state.schema
        self.tolerance = tolerance

    @property
    def ambient_dimension(self) -> int:
        return self.kernel.ambient_dimension

    @property
    def intrinsic_dimension(self) -> int:
        return self.kernel.intrinsic_dimension

    @property
    def kind(self) -> GeometryKind:
        return self.kernel.kind

    @property
    def capabilities(self) -> frozenset[GeometryCapability]:
        return self.kernel.capabilities

    @property
    def field_certificate(self) -> FieldCertificate:
        return self.kernel.field_certificate

    def has_capability(self, capability: GeometryCapability, /) -> bool:
        return capability in self.capabilities

    def require(self, capability: GeometryCapability, /) -> GeometryKernel:
        if not self.has_capability(capability):
            raise NotImplementedError(
                f"Geometry kernel {type(self.kernel).__name__} does not provide "
                f"{capability.value}."
            )
        return self.kernel

    def boundary_field(self, points: Array, /) -> Array:
        return self.kernel.boundary_field(self.state, points)

    def signed_distance(self, points: Array, /) -> Array:
        self.require(GeometryCapability.SIGNED_DISTANCE)
        return self.kernel.boundary_field(self.state, points)

    def contains(self, points: Array, /) -> Array:
        self.require(GeometryCapability.REGION_QUERY)
        return self.kernel.contains(self.state, points)

    def boundary_normal(self, points: Array, /) -> Array:
        self.require(GeometryCapability.BOUNDARY_NORMAL)
        return self.kernel.boundary_normal(self.state, points)

    def contact_curvature(self, points: Array, /) -> ContactCurvatureResult:
        kernel = self.require(GeometryCapability.CONTACT_CURVATURE)
        if not isinstance(kernel, ContactCurvatureProvider):
            raise TypeError("Geometry advertises contact curvature without a provider.")
        result = kernel.contact_curvature(self.state, points)
        if not isinstance(result, ContactCurvatureResult):
            raise TypeError(
                "Contact-curvature provider must return ContactCurvatureResult."
            )
        return result

    @property
    def bounds(self) -> Array:
        return self.kernel.bounds(self.state)

    @property
    def measure(self) -> Array:
        self.require(GeometryCapability.MEASURE)
        return self.kernel.measure(self.state)

    @property
    def boundary_measure(self) -> Array:
        self.require(GeometryCapability.MEASURE)
        return self.kernel.boundary_measure(self.state)

    @property
    def boundary_atlas(self) -> BoundaryAtlas:
        self.require(GeometryCapability.BOUNDARY_ATLAS)
        return self.kernel.boundary_atlas(self.state)

    def cubature_atlas(self, component: CubatureComponent, /) -> CubatureAtlas:
        self.require(GeometryCapability.CUBATURE_ATLAS)
        return self.kernel.cubature_atlas(self.state, component)

    def sample_interior(
        self,
        num_points: int,
        /,
        *,
        key: Key[Array, ""],
        plan: RejectionSamplingPlan | None = None,
    ) -> SamplingResult:
        self.require(GeometryCapability.INTERIOR_SAMPLING)
        return self.kernel.sample_interior(
            self.state,
            num_points,
            key=key,
            plan=plan,
        )

    def sample_boundary(
        self,
        num_points: int,
        /,
        *,
        key: Key[Array, ""],
    ) -> SamplingResult:
        self.require(GeometryCapability.BOUNDARY_SAMPLING)
        return self.kernel.sample_boundary(self.state, num_points, key=key)

    def with_state(self, state: DesignState, /) -> CompiledGeometry:
        if state.schema != self.schema:
            raise ValueError("Replacement DesignState must use the compiled schema.")
        return CompiledGeometry(self.kernel, state, tolerance=self.tolerance)

    def with_parameters(
        self,
        updates: Mapping[ParameterId, Any],
        /,
    ) -> CompiledGeometry:
        return self.with_state(self.state.updated(updates))

    def equivalent(self, other: object, /) -> bool:
        if not isinstance(other, CompiledGeometry):
            return False
        if self.schema != other.schema or self.tolerance != other.tolerance:
            return False
        equal = eqx.tree_equal(self.kernel, other.kernel)
        if not isinstance(equal, bool):
            equal = bool(np.asarray(equal))
        if not equal:
            return False
        return all(
            np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(
                self.state.values,
                other.state.values,
                strict=True,
            )
        )


class AbstractGeometrySource(StrictModule):
    """Authoritative declarative or host-side source of compiled geometry."""

    def compile(
        self,
        *,
        tolerance: GeometryTolerance = GeometryTolerance(),
    ) -> CompiledGeometry:
        context = _ParameterCollector()
        kernel = self._compile(context)
        _, state = context.finish()
        return CompiledGeometry(kernel, state, tolerance=tolerance)

    @abstractmethod
    def _compile(self, context: _ParameterCollector, /) -> GeometryKernel:
        raise NotImplementedError

    def translated(self, offset: Any, /) -> GeometrySource:
        from .analytic._expressions import Translation

        return Translation(self, offset)

    def transformed(self, frame: Any, /) -> GeometrySource:
        from .analytic._operations import RigidTransform

        return RigidTransform(self, frame)

    def rotated(
        self,
        axis: Any,
        angle: float,
        /,
        *,
        center: Any = (0.0, 0.0, 0.0),
    ) -> GeometrySource:
        from .analytic._operations import RigidFrame, RigidTransform

        frame = RigidFrame.from_axis_angle(axis, angle, translation=center)
        if np.any(np.asarray(center, dtype=float) != 0.0):
            rotation = np.asarray(frame.rotation)
            center_ = np.asarray(center, dtype=float)
            frame = RigidFrame(rotation, center_ - rotation @ center_)
        return RigidTransform(self, frame)

    def scaled(
        self,
        scale: Any,
        /,
        *,
        center: Any | None = None,
    ) -> GeometrySource:
        from .analytic._operations import Scaling

        return Scaling(self, scale, center=center)

    def __and__(self, other: GeometrySource, /) -> GeometrySource:
        from .analytic._operations import Intersection

        if not isinstance(other, GeometrySource):
            return NotImplemented
        return Intersection(self, other)

    def __sub__(self, other: GeometrySource, /) -> GeometrySource:
        from .analytic._operations import Difference

        if not isinstance(other, GeometrySource):
            return NotImplemented
        return Difference(self, other)

    def __or__(self, other: GeometrySource, /) -> GeometrySource:
        from .analytic._expressions import Union

        if not isinstance(other, GeometrySource):
            return NotImplemented
        return Union((self, other))


GeometryKernel = AbstractGeometryKernel
GeometrySource = AbstractGeometrySource


__all__ = [
    "ContactCurvatureResult",
    "CompiledGeometry",
    "GeometryKernel",
    "GeometryKind",
    "GeometrySource",
    "GeometryTolerance",
]
