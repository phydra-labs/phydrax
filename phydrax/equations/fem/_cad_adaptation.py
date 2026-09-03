#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.fem._generic import FiniteElementDiscretization
from ._moving_conservation import (
    FiniteElementGeometrySnapshot,
    GeometryRecoveryResult,
    recover_geometry_snapshot,
)


class CADProjectionEvidence(StrictModule, NonTrainableState):
    distances: Array
    normals: Array
    maximum_distance: Array
    converged: Array
    selector_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class CADProjectionPlan(StrictModule, NonTrainableState):
    projector: Callable = eqx.field(static=True)
    normal_provider: Callable = eqx.field(static=True)
    selector_id: str = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        projector: Callable,
        normal_provider: Callable,
        /,
        *,
        selector_id: str,
        tolerance: float = 1.0e-10,
    ):
        tolerance_ = float(tolerance)
        identifier = str(selector_id)
        if (
            not callable(projector)
            or not callable(normal_provider)
            or not identifier
            or not math.isfinite(tolerance_)
            or tolerance_ < 0.0
        ):
            raise ValueError("CAD projection plan is invalid.")
        self.projector = projector
        self.normal_provider = normal_provider
        self.selector_id = identifier
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cad-projection-plan",
                "selector": identifier,
                "tolerance": tolerance_,
            }
        )

    def project(self, points: ArrayLike, /) -> tuple[Array, CADProjectionEvidence]:
        values = jnp.asarray(points)
        projected = jnp.asarray(self.projector(values))
        normals = jnp.asarray(self.normal_provider(projected))
        if projected.shape != values.shape or normals.shape != values.shape:
            raise ValueError("CAD projection or normal provider changed point shape.")
        normal_norm = jnp.sqrt(
            ein.contract("...d,...d->...", normals, normals, backend="jax")
        )
        normals = normals / normal_norm[..., None]
        distances = jnp.sqrt(jnp.sum((projected - values) ** 2, axis=-1))
        maximum = jnp.max(distances)
        finite = jnp.all(jnp.isfinite(projected)) & jnp.all(jnp.isfinite(normals))
        evidence_id = canonical_fingerprint(
            {
                "kind": "cad-projection-evidence",
                "plan": self.plan_id,
                "shape": tuple(values.shape),
            }
        )
        return projected, CADProjectionEvidence(
            distances,
            normals,
            maximum,
            finite & jnp.all(jnp.abs(normal_norm - 1.0) <= self.tolerance + 1.0e-8),
            self.selector_id,
            evidence_id,
        )


class CurvatureAdaptationPlan(StrictModule, NonTrainableState):
    target_error: float = eqx.field(static=True)
    minimum_degree: int = eqx.field(static=True)
    maximum_degree: int = eqx.field(static=True)
    maximum_displacement_fraction: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        target_error: float,
        minimum_degree: int = 1,
        maximum_degree: int = 8,
        maximum_displacement_fraction: float = 0.25,
    ):
        error = float(target_error)
        minimum = int(minimum_degree)
        maximum = int(maximum_degree)
        fraction = float(maximum_displacement_fraction)
        if error <= 0.0 or minimum < 1 or maximum < minimum or not 0.0 < fraction <= 1.0:
            raise ValueError("Curvature adaptation controls are invalid.")
        self.target_error = error
        self.minimum_degree = minimum
        self.maximum_degree = maximum
        self.maximum_displacement_fraction = fraction
        self.plan_id = canonical_fingerprint(
            {
                "kind": "curvature-adaptation-plan",
                "target_error": error,
                "minimum_degree": minimum,
                "maximum_degree": maximum,
                "maximum_displacement_fraction": fraction,
            }
        )

    def requested_degrees(self, curvature: ArrayLike, cell_length: ArrayLike, /) -> Array:
        indicator = jnp.abs(jnp.asarray(curvature)) * jnp.asarray(cell_length) ** 2
        increments = jnp.ceil(
            jnp.log2(jnp.maximum(indicator / self.target_error, 1.0))
        ).astype(jnp.int32)
        return jnp.clip(
            self.minimum_degree + increments,
            self.minimum_degree,
            self.maximum_degree,
        )

    def limited_displacement(
        self,
        displacement: ArrayLike,
        local_length: ArrayLike,
        /,
    ) -> Array:
        value = jnp.asarray(displacement)
        maximum = self.maximum_displacement_fraction * jnp.asarray(local_length)
        norm = jnp.sqrt(ein.contract("...d,...d->...", value, value, backend="jax"))
        factor = jnp.minimum(1.0, maximum / jnp.maximum(norm, 1.0e-30))
        return value * factor[..., None]


class CADAdaptationResult(StrictModule, NonTrainableState):
    snapshot: FiniteElementGeometrySnapshot
    projection: CADProjectionEvidence
    recovery: GeometryRecoveryResult
    requested_degrees: Array
    accepted: Array
    result_id: str = eqx.field(static=True)


def project_and_recover_cad_geometry(
    discretization: FiniteElementDiscretization,
    accepted: FiniteElementGeometrySnapshot,
    boundary_dofs: ArrayLike,
    projection_plan: CADProjectionPlan,
    curvature_plan: CurvatureAdaptationPlan,
    curvature: ArrayLike,
    local_length: ArrayLike,
    /,
) -> CADAdaptationResult:
    indices = jnp.asarray(boundary_dofs, dtype=jnp.int32)
    if indices.ndim != 1:
        raise ValueError("CAD boundary DOFs must be one-dimensional.")
    projected, evidence = projection_plan.project(accepted.coordinates[indices])
    displacement = projected - accepted.coordinates[indices]
    limited = curvature_plan.limited_displacement(displacement, local_length)
    coordinates = accepted.coordinates.at[indices].add(limited)
    candidate = FiniteElementGeometrySnapshot(
        coordinates,
        accepted.coordinate_velocity,
        accepted.time,
        topology_id=accepted.topology_id,
        geometry_layout_id=accepted.geometry_layout_id,
    )
    recovery = recover_geometry_snapshot(discretization, accepted, candidate)
    degrees = curvature_plan.requested_degrees(curvature, local_length)
    result_id = canonical_fingerprint(
        {
            "kind": "cad-adaptation-result",
            "projection": evidence.evidence_id,
            "recovered_snapshot": recovery.snapshot.snapshot_id,
            "curvature_plan": curvature_plan.plan_id,
        }
    )
    return CADAdaptationResult(
        recovery.snapshot,
        evidence,
        recovery,
        degrees,
        recovery.accepted,
        result_id,
    )


__all__ = [
    "CADAdaptationResult",
    "CADProjectionEvidence",
    "CADProjectionPlan",
    "CurvatureAdaptationPlan",
    "project_and_recover_cad_geometry",
]
