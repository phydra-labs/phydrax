from typing import TYPE_CHECKING

from ..._deferred_exports import deferred_exports


#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


# Re-exports load on first use, so importing one owner never imports the
# whole package and its import cycles.
_DEFERRED_EXPORTS = {
    "reconstruct_dem_region": "._core",
    "reconstruct_lidar_region": "._core",
    "reconstruct_planar_region": "._core",
    "reconstruct_point_region": "._core",
    "reconstruct_surface_region": "._core",
    "reconstruct_trimmed_surface": "._core",
    "ReconstructedGeometrySource": "._core",
    "ReconstructionFailure": "._core",
    "ReconstructionReport": "._core",
    "ReconstructionReportProvider": "._core",
    "TrimmedSurfaceReconstruction": "._core",
    "estimate_point_normals": "._normals",
    "NormalEstimationEvidence": "._normals",
    "NormalOrientation": "._normals",
    "PointNormals": "._normals",
    "PoissonDiscretization": "._poisson",
    "PoissonSolveEvidence": "._poisson",
    "SampledSurfaceDeviation": "._poisson",
    "ComponentFitEvidence": "._robustness",
    "IncompleteSamplingPolicy": "._robustness",
    "OutlierRemovalEvidence": "._robustness",
    "ReconstructionRobustness": "._robustness",
    "SamplingCoverageEvidence": "._robustness",
    "ThinFeatureEvidence": "._robustness",
}


__getattr__, __dir__ = deferred_exports(__name__, globals(), _DEFERRED_EXPORTS)


if TYPE_CHECKING:
    from ._core import (
        reconstruct_dem_region,
        reconstruct_lidar_region,
        reconstruct_planar_region,
        reconstruct_point_region,
        reconstruct_surface_region,
        reconstruct_trimmed_surface,
        ReconstructedGeometrySource,
        ReconstructionFailure,
        ReconstructionReport,
        ReconstructionReportProvider,
        TrimmedSurfaceReconstruction,
    )
    from ._normals import (
        estimate_point_normals,
        NormalEstimationEvidence,
        NormalOrientation,
        PointNormals,
    )
    from ._poisson import (
        PoissonDiscretization,
        PoissonSolveEvidence,
        SampledSurfaceDeviation,
    )
    from ._robustness import (
        ComponentFitEvidence,
        IncompleteSamplingPolicy,
        OutlierRemovalEvidence,
        ReconstructionRobustness,
        SamplingCoverageEvidence,
        ThinFeatureEvidence,
    )


__all__ = [
    "ComponentFitEvidence",
    "estimate_point_normals",
    "IncompleteSamplingPolicy",
    "NormalEstimationEvidence",
    "NormalOrientation",
    "OutlierRemovalEvidence",
    "PointNormals",
    "PoissonDiscretization",
    "PoissonSolveEvidence",
    "reconstruct_dem_region",
    "reconstruct_lidar_region",
    "reconstruct_planar_region",
    "reconstruct_point_region",
    "reconstruct_surface_region",
    "reconstruct_trimmed_surface",
    "ReconstructedGeometrySource",
    "ReconstructionFailure",
    "ReconstructionReport",
    "ReconstructionReportProvider",
    "ReconstructionRobustness",
    "SampledSurfaceDeviation",
    "SamplingCoverageEvidence",
    "ThinFeatureEvidence",
    "TrimmedSurfaceReconstruction",
]
