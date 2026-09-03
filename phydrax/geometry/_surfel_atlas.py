#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._lagrangian_marker import (
    LagrangianMarkerDiscretization,
    LagrangianMarkerKinematics,
)
from ..discretization.surfel import (
    PreparedSurfelDiscretization,
    SurfelAccuracy,
    SurfelCoverageScope,
    SurfelFootprintMeaning,
    SurfelGeometryCertificate,
    SurfelGeometryPlan,
    SurfelGeometryState,
    SurfelOrientationScope,
    SurfelSetPlan,
)
from ._atlas import BoundaryAtlas
from ._immersed_markers import (
    ImmersedMarkerMaterialization,
    ImmersedMarkerQuadraturePlan,
    MarkerVelocityProvider,
)


class BoundaryAtlasSurfelMaterialization(StrictModule):
    geometry: SurfelGeometryState
    velocity: Array
    marker_materialization: ImmersedMarkerMaterialization
    finite: Array
    successful: Array
    materialization_id: str = eqx.field(static=True)

    def marker_kinematics(
        self, markers: LagrangianMarkerDiscretization, /
    ) -> LagrangianMarkerKinematics:
        if not isinstance(markers, LagrangianMarkerDiscretization):
            raise TypeError("markers must be LagrangianMarkerDiscretization.")
        if not np.array_equal(
            np.asarray(markers.marker_ids),
            np.asarray(self.geometry.discretization.surfel_ids),
        ):
            raise ValueError("Marker and surfel stable identities disagree.")
        return self.marker_materialization.kinematics(markers)


class PreparedBoundaryAtlasSurfels(StrictModule, NonTrainableState):
    plan: BoundaryAtlasSurfelPlan
    discretization: PreparedSurfelDiscretization
    prepared_id: str = eqx.field(static=True)

    def materialize(
        self,
        atlas: BoundaryAtlas,
        time: ArrayLike,
        /,
        *,
        velocity: MarkerVelocityProvider | ArrayLike | None = None,
        epoch: int | Array = 0,
    ) -> BoundaryAtlasSurfelMaterialization:
        if not isinstance(atlas, BoundaryAtlas):
            raise TypeError("atlas must be a BoundaryAtlas.")
        quadrature = self.plan.quadrature
        marker = quadrature.materialize(atlas, time, velocity=velocity)
        frame = atlas.frame(
            quadrature.chart_indices,
            quadrature.reference_coordinates,
        )
        tangent_dimension = atlas.reference_dimension
        footprint_measure = (
            self.plan.footprint_area_ratio * marker.physical_quadrature_weight
        )
        if tangent_dimension == 1:
            radius = 0.5 * footprint_measure
        else:
            radius = jnp.sqrt(footprint_measure / jnp.pi)
        tangent_axes = jnp.swapaxes(frame.tangents, -1, -2) * radius[:, None, None]
        certificate = SurfelGeometryCertificate(
            source_geometry_id=atlas.source_id,
            source_kind="boundary_atlas",
            position_accuracy=SurfelAccuracy.EXACT,
            normal_accuracy=SurfelAccuracy.EXACT,
            orientation_scope=SurfelOrientationScope.COMPONENT,
            coverage_scope=SurfelCoverageScope.SAMPLED,
            footprint_meaning=SurfelFootprintMeaning.QUADRATURE_PATCH,
            one_sided=True,
            provenance=("boundary_atlas", "immersed_marker_quadrature"),
        )
        geometry = SurfelGeometryPlan(self.discretization).materialize(
            marker.position,
            frame.normal,
            tangent_axes,
            physical_surface_weights=marker.physical_quadrature_weight,
            certificate=certificate,
            epoch=epoch,
        )
        finite = marker.finite & geometry.evidence.finite
        successful = finite & geometry.evidence.successful
        return BoundaryAtlasSurfelMaterialization(
            geometry=geometry,
            velocity=marker.velocity,
            marker_materialization=marker,
            finite=finite,
            successful=successful,
            materialization_id=canonical_fingerprint(
                {
                    "kind": "boundary-atlas-surfel-materialization",
                    "prepared": self.prepared_id,
                    "atlas": atlas.source_id,
                }
            ),
        )


class BoundaryAtlasSurfelPlan(StrictModule, NonTrainableState):
    """Materialize isotropic surfel footprints from an atlas quadrature."""

    quadrature: ImmersedMarkerQuadraturePlan
    footprint_area_ratio: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        quadrature: ImmersedMarkerQuadraturePlan,
        /,
        *,
        footprint_area_ratio: float = 1.0,
    ) -> None:
        if not isinstance(quadrature, ImmersedMarkerQuadraturePlan):
            raise TypeError("quadrature must be ImmersedMarkerQuadraturePlan.")
        ratio = float(footprint_area_ratio)
        if not np.isfinite(ratio) or ratio <= 0.0:
            raise ValueError("footprint_area_ratio must be finite and positive.")
        self.quadrature = quadrature
        self.footprint_area_ratio = ratio
        self.plan_id = canonical_fingerprint(
            {
                "kind": "boundary-atlas-surfel-plan",
                "quadrature": quadrature.plan_id,
                "footprint_area_ratio": ratio,
            }
        )

    def prepare(
        self,
        atlas: BoundaryAtlas,
        time: ArrayLike,
        /,
        *,
        velocity: MarkerVelocityProvider | ArrayLike | None = None,
        name: str = "boundary-atlas-surfels",
        numeric_version: str = "0",
    ) -> PreparedBoundaryAtlasSurfels:
        marker = self.quadrature.materialize(atlas, time, velocity=velocity)
        if not bool(marker.finite):
            raise ValueError("Reference atlas surfel materialization is not finite.")
        discretization = SurfelSetPlan(
            self.quadrature.marker_ids,
            marker.position,
            marker.physical_quadrature_weight,
            active_mask=self.quadrature.active_mask,
            source_entity_ids=marker.source_entity_id,
            name=name,
            plan_id=canonical_fingerprint(
                {
                    "kind": "boundary-atlas-surfel-set",
                    "plan": self.plan_id,
                    "atlas": atlas.source_id,
                }
            ),
        ).prepare(numeric_version=numeric_version)
        return PreparedBoundaryAtlasSurfels(
            plan=self,
            discretization=discretization,
            prepared_id=canonical_fingerprint(
                {
                    "kind": "prepared-boundary-atlas-surfels",
                    "plan": self.plan_id,
                    "discretization": discretization.prepared_id,
                }
            ),
        )


__all__ = [
    "BoundaryAtlasSurfelMaterialization",
    "BoundaryAtlasSurfelPlan",
    "PreparedBoundaryAtlasSurfels",
]
