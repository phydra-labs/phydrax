#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Weighted oriented surface elements over stable point ownership."""

from ._core import PreparedSurfelDiscretization, SurfelSetPlan
from ._footprint import SurfelFootprintEvaluation, SurfelFootprintPlan
from ._geometry import (
    SurfelAccuracy,
    SurfelCoverageScope,
    SurfelFootprintMeaning,
    SurfelGeometryCertificate,
    SurfelGeometryEvidence,
    SurfelGeometryPlan,
    SurfelGeometryState,
    SurfelOrientationScope,
)
from ._quadrature import SurfelQuadraturePlan, SurfelQuadratureResult
from ._query import (
    SurfelRayQueryEvidence,
    SurfelRayQueryPlan,
    SurfelRayQueryResult,
)
from ._voxel import (
    PreparedSurfelVoxelProjection,
    SurfelVoxelProjectionEvidence,
    SurfelVoxelProjectionPlan,
    SurfelVoxelProjectionResult,
    SurfelVoxelRouteEvidence,
)


__all__ = [
    "PreparedSurfelDiscretization",
    "PreparedSurfelVoxelProjection",
    "SurfelAccuracy",
    "SurfelCoverageScope",
    "SurfelFootprintEvaluation",
    "SurfelFootprintMeaning",
    "SurfelFootprintPlan",
    "SurfelGeometryCertificate",
    "SurfelGeometryEvidence",
    "SurfelGeometryPlan",
    "SurfelGeometryState",
    "SurfelOrientationScope",
    "SurfelQuadraturePlan",
    "SurfelQuadratureResult",
    "SurfelRayQueryEvidence",
    "SurfelRayQueryPlan",
    "SurfelRayQueryResult",
    "SurfelSetPlan",
    "SurfelVoxelProjectionEvidence",
    "SurfelVoxelProjectionPlan",
    "SurfelVoxelProjectionResult",
    "SurfelVoxelRouteEvidence",
]
