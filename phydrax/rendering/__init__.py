#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Differentiable state-to-measurement rendering operators."""

from ._lidar import LidarRenderResult, LidarSurfacePlan, PreparedLidarSurface
from ._lidar_waveform import (
    AtmosphericLidarPlan,
    HardSurfaceLidarWaveformPlan,
    LidarReturnExtractionPlan,
    LidarReturnExtractionResult,
    LidarWaveformEvidence,
    LidarWaveformResult,
    SpecularLidarMultipathPlan,
    TimeResolvedMultipleScatteringPlan,
)
from ._point import (
    GaussianRasterEvidence,
    GaussianRasterizer,
    GaussianRasterResult,
    RASTER_CLIPPED,
    RASTER_COMPLETE,
    RASTER_INACTIVE,
    RASTER_INVALID,
    RASTER_SUPPORT_OVERFLOW,
    rasterize_gaussians,
)
from ._result import ImageRenderResult, RenderEvidence
from ._sensor import (
    apply_photometry,
    CameraStackRenderResult,
    ParticleImageFormation,
    PhotometricResponse,
    PhotometryEvidence,
    PhotometryResult,
    render_camera_stack,
)
from ._surface import (
    prepare_surface_image,
    PreparedSurfaceImage,
    SurfaceImagePlan,
)


__all__ = [
    "AtmosphericLidarPlan",
    "CameraStackRenderResult",
    "GaussianRasterEvidence",
    "GaussianRasterResult",
    "GaussianRasterizer",
    "ImageRenderResult",
    "HardSurfaceLidarWaveformPlan",
    "PreparedSurfaceImage",
    "LidarRenderResult",
    "LidarSurfacePlan",
    "LidarReturnExtractionPlan",
    "LidarReturnExtractionResult",
    "LidarWaveformEvidence",
    "LidarWaveformResult",
    "ParticleImageFormation",
    "PhotometricResponse",
    "PhotometryEvidence",
    "PhotometryResult",
    "RASTER_CLIPPED",
    "RASTER_COMPLETE",
    "PreparedLidarSurface",
    "RASTER_INACTIVE",
    "RASTER_INVALID",
    "RASTER_SUPPORT_OVERFLOW",
    "RenderEvidence",
    "SpecularLidarMultipathPlan",
    "SurfaceImagePlan",
    "TimeResolvedMultipleScatteringPlan",
    "apply_photometry",
    "rasterize_gaussians",
    "render_camera_stack",
    "prepare_surface_image",
]
