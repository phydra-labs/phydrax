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
from ._capabilities import (
    ClosestPointProvider,
    ContactCurvatureProvider,
    GeometryCapability,
    SupportMapProvider,
)
from ._certificate import FieldCertificate
from ._validity import (
    GeometryValidityEvidence,
    parameter_validity,
    representation_validity,
)
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


class ClosestPointResult(StrictModule):
    """Closest-point data with pointwise tubular-neighborhood evidence.

    Exactness is scoped independently to the represented geometry and to its
    physical source.  In particular, a mesh/BVH query may be exact for the
    piecewise-linear representation without being exact for the source CAD.
    """

    closest_point: Array
    normal_coordinate: Array
    oriented_normal: Array
    source_entity_id: Array
    unique: Array
    regular: Array
    margin: Array
    normal_coordinate_valid: Array
    represented_geometry_id: str = eqx.field(static=True)
    physical_geometry_id: str | None = eqx.field(static=True)
    exact_to_physical: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        closest_point: Array,
        normal_coordinate: Array,
        oriented_normal: Array,
        source_entity_id: Array,
        unique: Array,
        regular: Array,
        margin: Array,
        represented_geometry_id: str,
        physical_geometry_id: str | None = None,
        exact_to_physical: bool = False,
        normal_coordinate_valid: Array | None = None,
    ):
        point = jnp.asarray(closest_point, dtype=float)
        normal = jnp.asarray(oriented_normal, dtype=point.dtype)
        coordinate = jnp.asarray(normal_coordinate, dtype=point.dtype)
        entity = jnp.asarray(source_entity_id, dtype=jnp.int32)
        unique_ = jnp.asarray(unique, dtype=bool)
        regular_ = jnp.asarray(regular, dtype=bool)
        margin_ = jnp.asarray(margin, dtype=point.dtype)
        if point.ndim == 0:
            raise ValueError("closest_point must have a trailing coordinate axis.")
        leading = point.shape[:-1]
        if normal.shape != point.shape:
            raise ValueError("oriented_normal must match closest_point shape.")
        if any(
            value.shape != leading
            for value in (coordinate, entity, unique_, regular_, margin_)
        ):
            raise ValueError(
                "Closest-point coordinates, entities, masks, and margins must "
                "match the point leading shape."
            )
        coordinate_valid = (
            jnp.isfinite(coordinate)
            if normal_coordinate_valid is None
            else jnp.asarray(normal_coordinate_valid, dtype=bool)
        )
        if coordinate_valid.shape != leading:
            raise ValueError(
                "normal_coordinate_valid must match the point leading shape."
            )
        represented = str(represented_geometry_id)
        if not represented:
            raise ValueError("represented_geometry_id must be non-empty.")
        physical = None if physical_geometry_id is None else str(physical_geometry_id)
        if physical is not None and not physical:
            raise ValueError("physical_geometry_id must be non-empty when provided.")
        exact_physical = bool(exact_to_physical)
        if exact_physical and physical is None:
            raise ValueError(
                "exact_to_physical requires an explicit physical_geometry_id."
            )
        self.closest_point = point
        self.normal_coordinate = coordinate
        self.oriented_normal = normal
        self.source_entity_id = entity
        self.unique = unique_
        self.regular = regular_
        self.margin = margin_
        self.normal_coordinate_valid = coordinate_valid
        self.represented_geometry_id = represented
        self.physical_geometry_id = physical
        self.exact_to_physical = exact_physical

    @property
    def signed_coordinate(self) -> Array:
        return self.normal_coordinate

    @property
    def normal(self) -> Array:
        return self.oriented_normal

    @property
    def uniqueness_mask(self) -> Array:
        return self.unique

    @property
    def regularity_mask(self) -> Array:
        return self.regular

    @property
    def reach_margin(self) -> Array:
        return self.margin


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

    def validity(self, state: DesignState | None = None, /) -> GeometryValidityEvidence:
        selected = self.state if state is None else state
        common = parameter_validity(self.schema, selected)
        representation = representation_validity(self.kernel, selected)
        return common.combined_with(
            representation,
            contract_id=f"compiled:{representation.contract_id}",
        )

    def require_valid(
        self,
        state: DesignState | None = None,
        /,
    ) -> GeometryValidityEvidence:
        evidence = self.validity(state)
        checked = eqx.error_if(
            evidence.conditions_satisfied,
            ~evidence.accepted,
            "Geometry state is invalid or its validity is inconclusive.",
        )
        return eqx.tree_at(
            lambda item: item.conditions_satisfied,
            evidence,
            checked,
        )

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

    def closest_point(self, points: Array, /) -> ClosestPointResult:
        kernel = self.require(GeometryCapability.CLOSEST_POINT)
        if not isinstance(kernel, ClosestPointProvider):
            raise TypeError("Geometry advertises closest points without a provider.")
        result = kernel.closest_point(self.state, points)
        if not isinstance(result, ClosestPointResult):
            raise TypeError("Closest-point provider must return ClosestPointResult.")
        return result

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

    def support_map(self, directions: Array, /) -> Array:
        kernel = self.require(GeometryCapability.SUPPORT_MAP)
        if not isinstance(kernel, SupportMapProvider):
            raise TypeError("Geometry advertises support mapping without a provider.")
        return kernel.support_map(self.state, directions)

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

    def extruded(
        self,
        height: Any,
        /,
        *,
        feature_id: str | None = None,
    ) -> GeometrySource:
        from .analytic._sweeps import Extrusion

        return Extrusion(self, height, feature_id=feature_id)

    def revolved(
        self,
        *,
        feature_id: str | None = None,
    ) -> GeometrySource:
        from .analytic._sweeps import Revolution

        return Revolution(self, feature_id=feature_id)

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
    "ClosestPointResult",
    "ContactCurvatureResult",
    "CompiledGeometry",
    "GeometryKernel",
    "GeometryKind",
    "GeometrySource",
    "GeometryTolerance",
]
