#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._lagrangian_marker import (
    LagrangianMarkerDiscretization,
    LagrangianMarkerKinematics,
    LagrangianMarkerSetPlan,
)
from ._atlas import BoundaryAtlas


MarkerVelocityProvider = Callable[[Array, Array], Array]


class ImmersedMarkerMaterialization(StrictModule):
    """Differentiable physical marker geometry from an atlas quadrature."""

    position: Array
    velocity: Array
    physical_quadrature_weight: Array
    surface_jacobian: Array
    source_entity_id: Array
    finite: Array
    materialization_id: str = eqx.field(static=True)

    def kinematics(
        self, markers: LagrangianMarkerDiscretization, /
    ) -> LagrangianMarkerKinematics:
        return markers.kinematics(self.position, self.velocity)


class ImmersedMarkerQuadraturePlan(StrictModule, NonTrainableState):
    """Stable marker IDs and reference quadrature on boundary-atlas charts."""

    marker_ids: Array
    chart_indices: Array
    reference_coordinates: Array
    reference_weights: Array
    active_mask: Array
    active_indices: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        marker_ids: ArrayLike,
        chart_indices: ArrayLike,
        reference_coordinates: ArrayLike,
        reference_weights: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
    ):
        ids = np.asarray(marker_ids)
        charts = np.asarray(chart_indices)
        reference = np.asarray(reference_coordinates)
        weights = np.asarray(reference_weights)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("marker_ids must be a rank-one integer array.")
        if ids.size == 0 or np.unique(ids).size != ids.size:
            raise ValueError("marker_ids must be nonempty and unique.")
        if charts.shape != ids.shape or not np.issubdtype(charts.dtype, np.integer):
            raise ValueError("chart_indices must match marker_ids.")
        if reference.ndim != 2 or reference.shape[0] != ids.size:
            raise ValueError("reference_coordinates must have one row per marker.")
        if (
            weights.shape != ids.shape
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ValueError("reference_weights must be positive and finite.")
        active = (
            np.ones(ids.shape, dtype=bool)
            if active_mask is None
            else np.asarray(active_mask, dtype=bool)
        )
        if active.shape != ids.shape or not np.any(active):
            raise ValueError("active_mask must activate at least one marker.")
        self.marker_ids = jnp.asarray(ids, dtype=jnp.int64)
        self.chart_indices = jnp.asarray(charts, dtype=jnp.int32)
        self.reference_coordinates = jnp.asarray(reference)
        self.reference_weights = jnp.asarray(weights)
        self.active_mask = jnp.asarray(active)
        self.active_indices = jnp.asarray(np.flatnonzero(active), dtype=jnp.int32)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "immersed-marker-quadrature",
                "arrays": array_tree_fingerprint(
                    (ids, charts, reference, weights, active)
                ),
            }
        )

    def materialize(
        self,
        atlas: BoundaryAtlas,
        time: ArrayLike,
        /,
        *,
        velocity: MarkerVelocityProvider | ArrayLike | None = None,
    ) -> ImmersedMarkerMaterialization:
        if not isinstance(atlas, BoundaryAtlas):
            raise TypeError("atlas must be a BoundaryAtlas.")
        if atlas.reference_dimension != self.reference_coordinates.shape[1]:
            raise ValueError("Atlas and marker reference dimensions differ.")
        positions = atlas.map(self.chart_indices, self.reference_coordinates)
        jacobian = atlas.jacobian(self.chart_indices, self.reference_coordinates)
        physical_weights = self.reference_weights * jacobian
        if velocity is None:
            velocities = jnp.zeros_like(positions)
        elif callable(velocity):
            velocities = velocity(jnp.asarray(time), positions)
        else:
            velocities = jnp.asarray(velocity, dtype=positions.dtype)
        if velocities.shape != positions.shape:
            raise ValueError("Marker velocities must have the position shape.")
        source_entity = atlas.source_entity_ids[self.chart_indices]
        finite = (
            jnp.all(jnp.isfinite(positions))
            & jnp.all(jnp.isfinite(velocities))
            & jnp.all(jnp.isfinite(physical_weights))
            & jnp.all(physical_weights[self.active_indices] > 0.0)
        )
        return ImmersedMarkerMaterialization(
            positions,
            velocities,
            physical_weights,
            jacobian,
            source_entity,
            finite,
            self.plan_id,
        )

    def marker_plan(
        self,
        materialization: ImmersedMarkerMaterialization,
        /,
        *,
        name: str = "immersed-atlas-markers",
    ) -> LagrangianMarkerSetPlan:
        """Freeze one materialized reference measure for a prepared solve epoch."""
        return LagrangianMarkerSetPlan(
            self.marker_ids,
            materialization.position,
            materialization.physical_quadrature_weight,
            active_mask=self.active_mask,
            name=name,
            plan_id=canonical_fingerprint(
                {
                    "kind": "materialized-immersed-marker-plan",
                    "quadrature": self.plan_id,
                    "atlas": materialization.materialization_id,
                }
            ),
        )


__all__ = [
    "ImmersedMarkerMaterialization",
    "ImmersedMarkerQuadraturePlan",
    "MarkerVelocityProvider",
]
