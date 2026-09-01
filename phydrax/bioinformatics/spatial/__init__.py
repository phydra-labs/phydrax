#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Spatial assay, imaging, neighborhood, registration, and statistics."""

from ._assay import (
    SpatialAssay,
    SpatialAssayData,
    SpatialSampleRecord,
)
from ._frame import (
    AffineSpatialTransform,
    MICROMETRE,
    MILLIMETRE,
    NANOMETRE,
    PIXEL,
    SpatialCoordinates,
    SpatialFrame,
    SpatialUnit,
    unit_conversion_transform,
)
from ._imaging import (
    BoundedImagePatch,
    BoundedImagePatchPlan,
    extract_bounded_image_patch,
    image_tile_metadata,
    ImageAxis,
    ImagePatchEvidence,
    ImagePatchStatus,
    ImagePyramidLevel,
    ImagePyramidMetadata,
    ImageTileMetadata,
    pyramid_level_tile_bounds,
)
from ._morphology import (
    MorphologyEvidence,
    MorphologyPlan,
    MorphologyStatus,
    MorphologySummary,
    PersistenceTopologyEvidence,
    PersistenceTopologySummary,
    summarize_label_morphology,
    summarize_persistence_topology,
)
from ._neighbors import (
    assay_neighbor_graph,
    build_spatial_neighbor_graph,
    NeighborGraphEvidence,
    NeighborGraphStatus,
    SpatialNeighborGraph,
    SpatialNeighborPlan,
)
from ._registration import (
    register_spatial_points,
    RegistrationEvidence,
    RegistrationStatus,
    RegistrationUncertainty,
    SpatialRegistrationPlan,
    SpatialRegistrationResult,
)
from ._statistics import (
    assay_autocorrelation_test,
    spatial_autocorrelation_test,
    SpatialAutocorrelationResult,
    SpatialStatisticEvidence,
    SpatialStatisticStatus,
)


__all__ = [
    "AffineSpatialTransform",
    "assay_autocorrelation_test",
    "assay_neighbor_graph",
    "BoundedImagePatch",
    "BoundedImagePatchPlan",
    "build_spatial_neighbor_graph",
    "extract_bounded_image_patch",
    "image_tile_metadata",
    "ImageAxis",
    "ImagePatchEvidence",
    "ImagePatchStatus",
    "ImagePyramidLevel",
    "ImagePyramidMetadata",
    "ImageTileMetadata",
    "MICROMETRE",
    "MILLIMETRE",
    "MorphologyEvidence",
    "MorphologyPlan",
    "MorphologyStatus",
    "MorphologySummary",
    "NANOMETRE",
    "NeighborGraphEvidence",
    "NeighborGraphStatus",
    "PersistenceTopologyEvidence",
    "PersistenceTopologySummary",
    "PIXEL",
    "pyramid_level_tile_bounds",
    "register_spatial_points",
    "RegistrationEvidence",
    "RegistrationStatus",
    "RegistrationUncertainty",
    "spatial_autocorrelation_test",
    "SpatialAssay",
    "SpatialAssayData",
    "SpatialAutocorrelationResult",
    "SpatialCoordinates",
    "SpatialFrame",
    "SpatialNeighborGraph",
    "SpatialNeighborPlan",
    "SpatialRegistrationPlan",
    "SpatialRegistrationResult",
    "SpatialSampleRecord",
    "SpatialStatisticEvidence",
    "SpatialStatisticStatus",
    "SpatialUnit",
    "summarize_label_morphology",
    "summarize_persistence_topology",
    "unit_conversion_transform",
]
