#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical native persistence and explicit-loss velocimetry adapters."""

from ._adapters import (
    PhysicalPIVValue,
    piv_to_observation_sequence,
    piv_to_tensor_grid,
    tracks_to_observation_sequence,
    tracks_to_trajectory_data,
)
from ._archive import (
    read_velocimetry_archive,
    VelocimetryArchive,
    write_velocimetry_archive,
)
from ._images import ImageLoader, LazyImageSequence2D
from ._learned_artifact import (
    LearnedPIVArtifact,
    LearnedPIVArtifactManifest,
    load_learned_piv_model,
    read_learned_piv_artifact,
    register_learned_piv_model,
    save_learned_piv_artifact,
)
from ._openpiv import (
    OpenPIVCoordinateConvention,
    OpenPIVValueKind,
    read_openpiv_text,
    write_openpiv_text,
)
from ._openptv import (
    OpenPTVReconstructionRecords,
    OpenPTVTargetRecords,
    OpenPTVTrackRecords,
    read_openptv_reconstruction,
    read_openptv_targets,
    read_openptv_tracks,
    write_openptv_reconstruction,
    write_openptv_targets,
    write_openptv_tracks,
)
from ._pivlab import PIVlabStage, PIVlabYAxis, read_pivlab, write_pivlab
from ._xarray import (
    from_pivpy,
    from_xarray,
    is_pivpy_available,
    is_xarray_available,
    PivpyYAxis,
    require_pivpy,
    require_xarray,
    to_pivpy,
    to_xarray,
)


__all__ = [
    "ImageLoader",
    "LazyImageSequence2D",
    "LearnedPIVArtifact",
    "LearnedPIVArtifactManifest",
    "OpenPIVCoordinateConvention",
    "OpenPIVValueKind",
    "OpenPTVReconstructionRecords",
    "OpenPTVTargetRecords",
    "OpenPTVTrackRecords",
    "PIVlabStage",
    "PIVlabYAxis",
    "PhysicalPIVValue",
    "PivpyYAxis",
    "VelocimetryArchive",
    "from_pivpy",
    "from_xarray",
    "is_pivpy_available",
    "is_xarray_available",
    "load_learned_piv_model",
    "piv_to_observation_sequence",
    "piv_to_tensor_grid",
    "read_learned_piv_artifact",
    "read_openpiv_text",
    "read_openptv_reconstruction",
    "read_openptv_targets",
    "read_openptv_tracks",
    "read_pivlab",
    "read_velocimetry_archive",
    "register_learned_piv_model",
    "require_pivpy",
    "require_xarray",
    "save_learned_piv_artifact",
    "to_pivpy",
    "to_xarray",
    "tracks_to_observation_sequence",
    "tracks_to_trajectory_data",
    "write_openpiv_text",
    "write_openptv_reconstruction",
    "write_openptv_targets",
    "write_openptv_tracks",
    "write_pivlab",
    "write_velocimetry_archive",
]
