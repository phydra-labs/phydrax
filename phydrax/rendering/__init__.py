#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Differentiable state-to-measurement rendering operators."""

from ._lidar import LidarRenderResult, LidarSurfacePlan, PreparedLidarSurface
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
    "CameraStackRenderResult",
    "GaussianRasterEvidence",
    "GaussianRasterResult",
    "GaussianRasterizer",
    "ImageRenderResult",
    "PreparedSurfaceImage",
    "LidarRenderResult",
    "LidarSurfacePlan",
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
    "SurfaceImagePlan",
    "apply_photometry",
    "rasterize_gaussians",
    "render_camera_stack",
    "prepare_surface_image",
]
