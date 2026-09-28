#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Differentiable state-to-measurement rendering operators."""

from ._colorimetry import (
    AbstractColorMatchingFunctions,
    AnalyticColorMatchingFunctions,
    COLOR_MATCHING_TABLE_MODEL,
    ColorMatchingFitError,
    ColorMatchingObserver,
    encode_srgb,
    GamutMapping,
    read_spectral_illuminant,
    spectral_to_xyz,
    SpectralColorimetryEvidence,
    SpectralColorimetryPlan,
    SpectralColorimetryResult,
    SpectralColorimetryStatus,
    SpectralIlluminant,
    TabulatedColorMatchingFunctions,
    xyz_to_linear_srgb,
)
from ._kinetic_video import KineticVideoPlan
from ._kinetic_volume import (
    KineticVolumeProjection,
    KineticVolumeRenderEvidence,
    KineticVolumeRenderPlan,
    KineticVolumeRenderResult,
)
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
    GaussianRasterAccumulation,
    GaussianRasterEvidence,
    GaussianRasterExecutionPlan,
    GaussianRasterizer,
    GaussianRasterKind,
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
from ._thin_film_appearance import (
    ThinFilmAppearanceEvidence,
    ThinFilmAppearancePlan,
    ThinFilmAppearanceResult,
    ThinFilmAppearanceStatus,
)
from ._thin_film_qualification import thin_film_appearance_candidate_profiles
from ._thin_film_surface import (
    thin_film_surface_colors,
    ThinFilmSurfaceColorResult,
    ThinFilmSurfaceColorStatus,
)


__all__ = [
    "AbstractColorMatchingFunctions",
    "AnalyticColorMatchingFunctions",
    "AtmosphericLidarPlan",
    "CameraStackRenderResult",
    "COLOR_MATCHING_TABLE_MODEL",
    "ColorMatchingFitError",
    "ColorMatchingObserver",
    "GamutMapping",
    "GaussianRasterAccumulation",
    "GaussianRasterEvidence",
    "GaussianRasterExecutionPlan",
    "GaussianRasterKind",
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
    "KineticVideoPlan",
    "KineticVolumeProjection",
    "KineticVolumeRenderEvidence",
    "KineticVolumeRenderPlan",
    "KineticVolumeRenderResult",
    "RenderEvidence",
    "SpectralColorimetryEvidence",
    "SpectralColorimetryPlan",
    "SpectralColorimetryResult",
    "SpectralColorimetryStatus",
    "SpectralIlluminant",
    "SpecularLidarMultipathPlan",
    "SurfaceImagePlan",
    "TabulatedColorMatchingFunctions",
    "ThinFilmAppearanceEvidence",
    "ThinFilmAppearancePlan",
    "ThinFilmAppearanceResult",
    "ThinFilmAppearanceStatus",
    "ThinFilmSurfaceColorResult",
    "ThinFilmSurfaceColorStatus",
    "TimeResolvedMultipleScatteringPlan",
    "apply_photometry",
    "encode_srgb",
    "rasterize_gaussians",
    "render_camera_stack",
    "prepare_surface_image",
    "read_spectral_illuminant",
    "spectral_to_xyz",
    "thin_film_appearance_candidate_profiles",
    "thin_film_surface_colors",
    "xyz_to_linear_srgb",
]
