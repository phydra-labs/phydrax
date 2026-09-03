#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import PreparedSurfelDiscretization


class SurfelAccuracy(str, Enum):
    EXACT = "exact"
    APPROXIMATE = "approximate"
    UNKNOWN = "unknown"


class SurfelOrientationScope(str, Enum):
    GLOBAL = "global"
    COMPONENT = "component"
    LOCAL = "local"
    UNORIENTED = "unoriented"


class SurfelCoverageScope(str, Enum):
    CERTIFIED = "certified"
    SAMPLED = "sampled"
    UNKNOWN = "unknown"


class SurfelFootprintMeaning(str, Enum):
    QUADRATURE_PATCH = "quadrature_patch"
    RECONSTRUCTION_FILTER = "reconstruction_filter"
    ACQUISITION_FOOTPRINT = "acquisition_footprint"


@dataclass(frozen=True)
class SurfelGeometryCertificate:
    """Static authority for one surfel realization family."""

    source_geometry_id: str
    source_kind: str
    position_accuracy: SurfelAccuracy = SurfelAccuracy.UNKNOWN
    normal_accuracy: SurfelAccuracy = SurfelAccuracy.UNKNOWN
    orientation_scope: SurfelOrientationScope = SurfelOrientationScope.UNORIENTED
    coverage_scope: SurfelCoverageScope = SurfelCoverageScope.UNKNOWN
    footprint_meaning: SurfelFootprintMeaning = (
        SurfelFootprintMeaning.RECONSTRUCTION_FILTER
    )
    maximum_position_error: float | None = None
    maximum_normal_angle: float | None = None
    curvature_upper_bound: float | None = None
    one_sided: bool = False
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.source_geometry_id):
            raise ValueError("source_geometry_id must be nonempty.")
        if not str(self.source_kind):
            raise ValueError("source_kind must be nonempty.")
        for name, value in (
            ("maximum_position_error", self.maximum_position_error),
            ("maximum_normal_angle", self.maximum_normal_angle),
            ("curvature_upper_bound", self.curvature_upper_bound),
        ):
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and nonnegative when supplied.")
        if self.one_sided and self.orientation_scope is SurfelOrientationScope.UNORIENTED:
            raise ValueError("One-sided surfels require an oriented certificate.")

    @property
    def globally_oriented(self) -> bool:
        return self.orientation_scope is SurfelOrientationScope.GLOBAL


class SurfelGeometryEvidence(NonTrainableState, StrictModule):
    active_surfels: Array
    finite_surfels: Array
    minimum_physical_weight: Array
    minimum_footprint_measure: Array
    maximum_normal_norm_defect: Array
    maximum_tangency_defect: Array
    minimum_tangent_eigenvalue: Array
    maximum_tangent_condition: Array
    minimum_orientation_cosine: Array
    finite: Array
    successful: Array


class SurfelGeometryState(StrictModule):
    """One validated fixed-topology surfel geometry realization."""

    discretization: PreparedSurfelDiscretization
    position: Array
    normal: Array
    tangent_axes: Array
    physical_surface_weight: Array
    tangent_gram: Array
    footprint_measure: Array
    footprint_half_width: Array
    active_mask: Array
    evidence: SurfelGeometryEvidence
    epoch: Array
    certificate: SurfelGeometryCertificate = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    @property
    def capacity(self) -> int:
        return self.discretization.capacity

    @property
    def ambient_dimension(self) -> int:
        return self.discretization.ambient_dimension


class SurfelGeometryPlan(NonTrainableState, StrictModule):
    """Validate and canonicalize dynamic geometry on a prepared surfel support."""

    discretization: PreparedSurfelDiscretization
    normal_tolerance: float = eqx.field(static=True)
    tangency_tolerance: float = eqx.field(static=True)
    minimum_axis_scale: float = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: PreparedSurfelDiscretization,
        /,
        *,
        normal_tolerance: float = 1.0e-8,
        tangency_tolerance: float = 1.0e-8,
        minimum_axis_scale: float = 1.0e-12,
        maximum_condition: float = 1.0e8,
    ) -> None:
        if not isinstance(discretization, PreparedSurfelDiscretization):
            raise TypeError("discretization must be PreparedSurfelDiscretization.")
        normal_tol = float(normal_tolerance)
        tangent_tol = float(tangency_tolerance)
        axis_scale = float(minimum_axis_scale)
        condition = float(maximum_condition)
        if (
            not np.isfinite(normal_tol)
            or normal_tol <= 0.0
            or not np.isfinite(tangent_tol)
            or tangent_tol <= 0.0
            or not np.isfinite(axis_scale)
            or axis_scale <= 0.0
            or not np.isfinite(condition)
            or condition <= 1.0
        ):
            raise ValueError("Surfel geometry tolerances are invalid.")
        self.discretization = discretization
        self.normal_tolerance = normal_tol
        self.tangency_tolerance = tangent_tol
        self.minimum_axis_scale = axis_scale
        self.maximum_condition = condition
        self.plan_id = canonical_fingerprint(
            {
                "kind": "surfel-geometry-plan",
                "discretization": discretization.prepared_id,
                "normal_tolerance": normal_tol,
                "tangency_tolerance": tangent_tol,
                "minimum_axis_scale": axis_scale,
                "maximum_condition": condition,
            }
        )

    def materialize(
        self,
        positions: ArrayLike,
        normals: ArrayLike,
        tangent_axes: ArrayLike,
        /,
        *,
        physical_surface_weights: ArrayLike | None = None,
        certificate: SurfelGeometryCertificate | None = None,
        epoch: int | Array = 0,
    ) -> SurfelGeometryState:
        position = jnp.asarray(positions, dtype=self.discretization.plan.coordinate_dtype)
        normal_input = jnp.asarray(
            normals, dtype=self.discretization.plan.coordinate_dtype
        )
        axes_input = jnp.asarray(
            tangent_axes, dtype=self.discretization.plan.coordinate_dtype
        )
        capacity = self.discretization.capacity
        dimension = self.discretization.ambient_dimension
        tangent_dimension = dimension - 1
        if position.shape != (capacity, dimension):
            raise ValueError("positions must have shape (surfel_capacity,dimension).")
        if normal_input.shape != position.shape:
            raise ValueError("normals must match positions.")
        if axes_input.shape != (capacity, dimension, tangent_dimension):
            raise ValueError(
                "tangent_axes must have shape (surfel_capacity,dimension,dimension-1)."
            )
        weights = (
            self.discretization.reference_surface_weight
            if physical_surface_weights is None
            else jnp.asarray(
                physical_surface_weights,
                dtype=self.discretization.plan.coordinate_dtype,
            )
        )
        if weights.shape != (capacity,):
            raise ValueError("physical_surface_weights must match surfel capacity.")
        active = self.discretization.active_mask
        finite_components = (
            jnp.all(jnp.isfinite(position), axis=-1)
            & jnp.all(jnp.isfinite(normal_input), axis=-1)
            & jnp.all(jnp.isfinite(axes_input), axis=(-2, -1))
            & jnp.isfinite(weights)
        )
        safe_position = jnp.where(
            finite_components[:, None],
            position,
            self.discretization.reference_position,
        )
        default_normal = jnp.zeros((dimension,), dtype=normal_input.dtype).at[-1].set(1.0)
        finite_normal = jnp.all(jnp.isfinite(normal_input), axis=-1)
        safe_normal_input = jnp.where(
            finite_normal[:, None], normal_input, default_normal
        )
        normal_norm = jnp.sqrt(jnp.sum(safe_normal_input**2, axis=-1))
        safe_normal = jnp.where(
            (normal_norm > self.minimum_axis_scale)[:, None],
            safe_normal_input / jnp.where(normal_norm > 0.0, normal_norm, 1.0)[:, None],
            default_normal,
        )
        safe_axes_input = jnp.where(finite_components[:, None, None], axes_input, 0.0)
        raw_axis_norm = jnp.sqrt(jnp.sum(safe_axes_input**2, axis=-2))
        raw_normal_projection = contract("ni,nik->nk", safe_normal, safe_axes_input)
        tangency_defect = jnp.max(
            jnp.abs(raw_normal_projection)
            / jnp.maximum(raw_axis_norm, self.minimum_axis_scale),
            axis=-1,
        )
        projected_axes = (
            safe_axes_input - safe_normal[:, :, None] * raw_normal_projection[:, None, :]
        )
        if dimension == 3:
            orientation = jnp.sum(
                jnp.cross(projected_axes[:, :, 0], projected_axes[:, :, 1]) * safe_normal,
                axis=-1,
            )
            projected_axes = projected_axes.at[:, :, 1].set(
                jnp.where(
                    (orientation < 0.0)[:, None],
                    -projected_axes[:, :, 1],
                    projected_axes[:, :, 1],
                )
            )
        tangent_gram = contract("nik,nil->nkl", projected_axes, projected_axes)
        if tangent_dimension == 1:
            minimum_eigenvalue = tangent_gram[:, 0, 0]
            maximum_eigenvalue = minimum_eigenvalue
            orientation_cosine = jnp.ones((capacity,), dtype=position.dtype)
        else:
            first = tangent_gram[:, 0, 0]
            mixed = tangent_gram[:, 0, 1]
            second = tangent_gram[:, 1, 1]
            discriminant = jnp.sqrt(
                jnp.maximum((first - second) ** 2 + 4.0 * mixed**2, 0.0)
            )
            minimum_eigenvalue = 0.5 * (first + second - discriminant)
            maximum_eigenvalue = 0.5 * (first + second + discriminant)
            cross_axis = jnp.cross(projected_axes[:, :, 0], projected_axes[:, :, 1])
            cross_norm = jnp.sqrt(jnp.sum(cross_axis**2, axis=-1))
            orientation_cosine = jnp.sum(cross_axis * safe_normal, axis=-1) / jnp.maximum(
                cross_norm, self.minimum_axis_scale
            )
        condition = maximum_eigenvalue / jnp.maximum(
            minimum_eigenvalue, self.minimum_axis_scale**2
        )
        gram_determinant = (
            tangent_gram[:, 0, 0]
            if tangent_dimension == 1
            else tangent_gram[:, 0, 0] * tangent_gram[:, 1, 1]
            - tangent_gram[:, 0, 1] * tangent_gram[:, 1, 0]
        )
        footprint_measure = (
            2.0 * jnp.sqrt(jnp.maximum(gram_determinant, 0.0))
            if tangent_dimension == 1
            else jnp.pi * jnp.sqrt(jnp.maximum(gram_determinant, 0.0))
        )
        half_width = jnp.sqrt(jnp.sum(projected_axes**2, axis=-1))
        normal_defect = jnp.where(finite_normal, jnp.abs(normal_norm - 1.0), jnp.inf)
        active_finite = active & finite_components
        minimum_weight = jnp.min(
            jnp.where(
                active,
                jnp.where(jnp.isfinite(weights), weights, -jnp.inf),
                jnp.inf,
            )
        )
        minimum_measure = jnp.min(jnp.where(active, footprint_measure, jnp.inf))
        minimum_eigen = jnp.min(jnp.where(active, minimum_eigenvalue, jnp.inf))
        maximum_condition_value = jnp.max(jnp.where(active, condition, 0.0), initial=0.0)
        maximum_normal_defect = jnp.max(
            jnp.where(active, normal_defect, 0.0), initial=0.0
        )
        maximum_tangency_defect = jnp.max(
            jnp.where(active, tangency_defect, 0.0), initial=0.0
        )
        minimum_orientation = jnp.min(jnp.where(active, orientation_cosine, 1.0))
        finite = jnp.all(~active | finite_components)
        successful = (
            finite
            & (minimum_weight > 0.0)
            & (minimum_measure > 0.0)
            & (minimum_eigen > self.minimum_axis_scale**2)
            & (maximum_condition_value <= self.maximum_condition)
            & (maximum_normal_defect <= self.normal_tolerance)
            & (maximum_tangency_defect <= self.tangency_tolerance)
            & (minimum_orientation > 0.0)
        )
        certificate_value = (
            SurfelGeometryCertificate(
                source_geometry_id=self.discretization.support.embedding_id,
                source_kind="explicit",
                provenance=("explicit_surfel_geometry",),
            )
            if certificate is None
            else certificate
        )
        geometry_id = canonical_fingerprint(
            {
                "kind": "surfel-geometry-state",
                "plan": self.plan_id,
                "source_geometry_id": certificate_value.source_geometry_id,
                "source_kind": certificate_value.source_kind,
            }
        )
        evidence = SurfelGeometryEvidence(
            active_surfels=jnp.sum(active, dtype=jnp.int32),
            finite_surfels=jnp.sum(active_finite, dtype=jnp.int32),
            minimum_physical_weight=minimum_weight,
            minimum_footprint_measure=minimum_measure,
            maximum_normal_norm_defect=maximum_normal_defect,
            maximum_tangency_defect=maximum_tangency_defect,
            minimum_tangent_eigenvalue=minimum_eigen,
            maximum_tangent_condition=maximum_condition_value,
            minimum_orientation_cosine=minimum_orientation,
            finite=finite,
            successful=successful,
        )
        return SurfelGeometryState(
            discretization=self.discretization,
            position=jnp.where(active[:, None], safe_position, 0.0),
            normal=jnp.where(active[:, None], safe_normal, 0.0),
            tangent_axes=jnp.where(active[:, None, None], projected_axes, 0.0),
            physical_surface_weight=jnp.where(
                active & jnp.isfinite(weights) & (weights > 0.0),
                weights,
                0.0,
            ),
            tangent_gram=jnp.where(active[:, None, None], tangent_gram, 0.0),
            footprint_measure=jnp.where(active, footprint_measure, 0.0),
            footprint_half_width=jnp.where(active[:, None], half_width, 0.0),
            active_mask=active,
            evidence=evidence,
            epoch=jnp.asarray(epoch, dtype=jnp.int32),
            certificate=certificate_value,
            geometry_id=geometry_id,
        )


__all__ = [
    "SurfelAccuracy",
    "SurfelCoverageScope",
    "SurfelFootprintMeaning",
    "SurfelGeometryCertificate",
    "SurfelGeometryEvidence",
    "SurfelGeometryPlan",
    "SurfelGeometryState",
    "SurfelOrientationScope",
]
