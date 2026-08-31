#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, runtime_checkable

from jaxtyping import Array

from ._atlas import BoundaryAtlas
from ._cubature import CubatureAtlasProvider


class GeometryCapability(str, Enum):
    """Typed computational capabilities exposed by a compiled geometry."""

    REGION_QUERY = "region_query"
    SIGNED_DISTANCE = "signed_distance"
    BOUNDARY_NORMAL = "boundary_normal"
    CONTACT_CURVATURE = "contact_curvature"
    SUPPORT_MAP = "support_map"
    MEASURE = "measure"
    INTERIOR_SAMPLING = "interior_sampling"
    BOUNDARY_SAMPLING = "boundary_sampling"
    BOUNDARY_ATLAS = "boundary_atlas"
    CUBATURE_ATLAS = "cubature_atlas"
    SEAM_DIAGNOSTICS = "seam_diagnostics"


@runtime_checkable
class ContactCurvatureProvider(Protocol):
    """Provider of certified principal boundary curvatures."""

    def contact_curvature(self, state: Any, points: Array, /) -> Any: ...


@runtime_checkable
class SupportMapProvider(Protocol):
    """Provider of support points for convex contact geometry."""

    def support_map(self, state: Any, directions: Array, /) -> Array: ...


@runtime_checkable
class BoundaryAtlasProvider(Protocol):
    """Structural provider used by representation-independent integration."""

    @property
    def boundary_atlas(self) -> BoundaryAtlas:
        """Return a boundary atlas for the provider's current state."""
        ...


@runtime_checkable
class SeamDiagnosticsProvider(Protocol):
    """Structural provider of differentiable fixed-topology seam diagnostics."""

    def seam_residual(self, state: Any, /) -> Array: ...


__all__ = [
    "BoundaryAtlasProvider",
    "ContactCurvatureProvider",
    "CubatureAtlasProvider",
    "GeometryCapability",
    "SupportMapProvider",
    "SeamDiagnosticsProvider",
]
