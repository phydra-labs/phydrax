#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from jaxtyping import ArrayLike

from ..._flow_matching_metric import ManifoldFlowMatchingMetric
from ..._geometry_precision import GeometryPrecisionPolicy
from ..._strict import StrictModule
from ...metrix import AbstractGeodesicManifold, GeodesicManifoldStateGeometry
from ._geodesic_interpolant import GeodesicEndpointInterpolant


class ManifoldTransportGeometry(StrictModule):
    """Shared geometry identity for endpoint paths, losses, and sample evolution."""

    manifold: AbstractGeodesicManifold
    interpolant: GeodesicEndpointInterpolant
    metric: ManifoldFlowMatchingMetric
    state_geometry: GeodesicManifoldStateGeometry

    def __init__(
        self,
        manifold: AbstractGeodesicManifold,
        /,
        *,
        source_coordinate: ArrayLike = 0.0,
        target_coordinate: ArrayLike = 1.0,
        precision: GeometryPrecisionPolicy | None = None,
    ):
        if not isinstance(manifold, AbstractGeodesicManifold):
            raise TypeError("manifold must be an AbstractGeodesicManifold.")
        self.manifold = manifold
        self.interpolant = GeodesicEndpointInterpolant(
            manifold,
            source_coordinate=source_coordinate,
            target_coordinate=target_coordinate,
        )
        self.metric = ManifoldFlowMatchingMetric(
            manifold,
            precision=precision,
        )
        self.state_geometry = GeodesicManifoldStateGeometry(manifold)

    @property
    def geometry_id(self) -> str:
        return self.state_geometry.geometry_id


__all__ = ["ManifoldTransportGeometry"]
