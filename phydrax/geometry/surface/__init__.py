#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Authoritative surface geometry, audits, selections, and intersections."""

from ._contracts import (
    InterfaceSide,
    SurfaceAuditPolicy,
    SurfaceAuditReport,
    SurfaceChartMappingEvidence,
    SurfaceInterface,
    SurfaceMetadata,
    SurfaceOrientationRepair,
    SurfacePreparationError,
    SurfacePreparationStatus,
    SurfaceSelection,
    SurfaceValidityCertificate,
)
from ._high_order import *  # noqa: F403
from ._high_order import __all__ as _high_order_all
from ._interop import *  # noqa: F403
from ._interop import __all__ as _interop_all
from ._intersection import (
    intersect_plane_surface,
    PlaneSectionEvidence,
    PlaneSectionLoop,
    PlaneSectionStatus,
    PlaneSurfaceSection,
)
from ._model import SurfaceModel, SurfaceRealization


__all__ = [
    "InterfaceSide",
    "PlaneSectionEvidence",
    "PlaneSectionLoop",
    "PlaneSectionStatus",
    "PlaneSurfaceSection",
    "SurfaceAuditPolicy",
    "SurfaceAuditReport",
    "SurfaceChartMappingEvidence",
    "SurfaceInterface",
    "SurfaceMetadata",
    "SurfaceModel",
    "SurfaceOrientationRepair",
    "SurfacePreparationError",
    "SurfacePreparationStatus",
    "SurfaceRealization",
    "SurfaceSelection",
    "SurfaceValidityCertificate",
    "intersect_plane_surface",
]
__all__ += [name for name in (*_high_order_all, *_interop_all) if name not in __all__]
