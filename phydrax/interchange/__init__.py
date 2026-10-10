#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Generic external-representation interchange contracts."""

from importlib import import_module
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from ._cad_occt import (
        import_occt_brep,
        model_from_occt_shape,
        persist_occt_shape,
        read_occt_shape,
    )
    from ._cad_occt_projection import prepare_occt_projection, PreparedOcctProjection
from .. import _external_runtime as external_runtime
from .._document_resource import (
    decode_json_resource,
    decode_text_resource,
    decode_xml_resource,
    DecodedJSON,
    DecodedText,
    DecodedXML,
)
from .._external_resource import (
    account_bounded_resource,
    bounded_resource_from_bytes,
    BoundedResource,
    open_bounded_resource,
    OpenedResource,
    read_bounded_resource,
    ResourceLimits,
    ResourceManifest,
    ResourceReadError,
)
from .._external_runtime import __all__ as _external_runtime_all
from .._hdf5_resource import (
    HDF5DatasetManifest,
    HDF5Limits,
    HDF5Manifest,
    inspect_hdf5_resource,
)
from .._numpy_resource import (
    decode_npy_resource,
    decode_npz_resource,
    DecodedNumpyArchive,
    DecodedNumpyArray,
    NumpyFormatLimits,
)
from .._publication import (
    PublicationMode,
    PublicationReceipt,
    publish_bytes,
    publish_file,
    publish_resource_set,
    PublishedMemberReceipt,
    ResourceSetPublicationReceipt,
)
from .._resource_archive import (
    admit_zip_resource,
    ArchiveLimits,
    ArchiveMemberManifest,
    BoundedArchive,
    read_zip_members,
)
from .._resource_set import (
    BoundedResourceSet,
    read_bounded_resource_set,
    ResourceMemberManifest,
    ResourceSetLimits,
    ResourceSetManifest,
    ResourceSetReadError,
)
from . import (
    dafoam,
    fmi,
    geant4_detector_design,
    helics,
    hfss_design,
    opticstudio,
    xfoil,
)
from ._bilby_result import ImportedBilbyResult, read_bilby_result_json
from ._black_hole import (
    BlackHoleArtifactKind,
    BlackHoleArtifactRights,
    BlackHoleArtifactSchema,
    BlackHoleArtifactUsePolicy,
    map_black_hole_artifact,
    map_field_artifact,
    map_image_artifact,
    map_numeric_model_artifact,
    map_visibility_artifact,
    map_waveform_artifact,
    NeutralBlackHoleArtifact,
)
from ._borehole import BoreholeInterval, BoreholeTrajectory, PreparedBoreholeSampling
from ._cad import (
    approximate_intersection_curve,
    CadCoverage,
    CadCurveApproximation,
    CadCurveFitPolicy,
    CadExportPolicy,
    CadExportResult,
    CadImportPolicy,
    CadImportResult,
    CadInterchangeError,
    CadRefusal,
    CadRefusalReason,
)
from ._cad_archive import BRepArchiveReceipt, load_brep_archive, save_brep_archive
from ._cad_brep_text import (
    BRepTextVersion,
    decode_brep_text_bytes,
    decode_brep_text_resource,
    read_brep_text,
    write_brep_text,
)
from ._cad_io import read_cad
from ._catalog import (
    CarrierKind,
    format_capabilities,
    FormatCapability,
    FormatDirection,
)
from ._compressible_kinetic import (
    CompressibleKineticCaseIR,
    import_packed_kinetic_boundary,
    LegacyKineticBoundaryImport,
    read_compressible_kinetic_yaml,
)
from ._coordinate_transform import (
    CoordinateTransformPlan,
    CoordinateTransformResult,
    execute_coordinate_transform,
    GeodeticDependencyError,
)
from ._electrical_formats import ElectricalTabularSurvey, read_electrical_survey_csv
from ._em_formats import MTImpedanceData, read_edi_impedance, read_emtf_xml_impedance
from ._geodetic_formats import (
    GeodeticFormatDependencyError,
    read_rinex_observations,
    read_sinex_positions,
    RINEXObservationData,
    SINEXPositionSolution,
)
from ._geospatial import (
    GeospatialContract,
    GeospatialTransform,
    QualifiedGeospatialGrid,
)
from ._iges import (
    decode_iges_bytes,
    decode_iges_resource,
    read_iges,
    write_iges,
)
from ._inspection import (
    HostInspectionConversion,
    HostInspectionField,
    HostInspectionFrame,
)
from ._layout import (
    decode_layout_bytes,
    decode_layout_resource,
    LayoutAdapterError,
    LayoutFormat,
    LayoutImportPolicy,
    LayoutImportResult,
    LayoutLayerKey,
    LayoutModel,
    LayoutRegion,
    read_layout,
)
from ._mesh_arrays import (
    MeshArrayArtifact,
    MeshArrayAssociation,
    MeshArrayBlock,
    MeshArrayField,
    MeshArraySelection,
)
from ._openpmd_laser import (
    OpenPMDLaserEnvelopeError,
    OpenPMDLaserEnvelopeExportResult,
    OpenPMDLaserEnvelopeImportPolicy,
    OpenPMDLaserEnvelopeImportResult,
    OpenPMDLaserEnvelopeProfile,
    read_openpmd_laser_envelope_hdf5,
    write_openpmd_laser_envelope_hdf5,
)
from ._openpmd_mesh import (
    convert_openpmd_hdf5_to_adios2,
    OpenPMDADIOS2Artifact,
    OpenPMDADIOS2Provider,
    OpenPMDFieldRecordName,
    OpenPMDMeshError,
    OpenPMDMeshExportResult,
    OpenPMDMeshGeometry,
    OpenPMDMeshImportPolicy,
    OpenPMDMeshImportResult,
    OpenPMDMeshIteration,
    OpenPMDMeshRecord,
    read_openpmd_adios2,
    read_openpmd_meshes_hdf5,
    write_openpmd_meshes_hdf5,
)
from ._openpmd_particles import (
    OpenPMDParticleTrackError,
    OpenPMDParticleTrackExportResult,
    OpenPMDParticleTrackImportPolicy,
    OpenPMDParticleTrackImportResult,
    OpenPMDParticleTrackSelection,
    read_openpmd_particle_tracks_hdf5,
    write_openpmd_particle_tracks_hdf5,
)
from ._openpmd_pic import (
    OpenPMDPICExportResult,
    OpenPMDPICImportResult,
    OpenPMDPICLayout,
    OpenPMDPICStreamReceipt,
    OpenPMDPICStreamWriter,
    read_openpmd_pic_state,
    write_openpmd_pic_state,
)
from ._potential_formats import (
    GeomagneticHarmonicModel,
    ICGEMGravityModel,
    read_geomagnetic_coefficients,
    read_icgem_gfc,
)
from ._pygmt import (
    export_geospatial_grid,
    GeospatialDependencyError,
    PyGMTRenderResult,
    render_geospatial_grid,
    XarrayGridExport,
)
from ._raster_formats import (
    GeospatialFormatDependencyError,
    read_cf_netcdf_grid,
    read_consolidated_zip_zarr_grid,
    read_geotiff_grid,
)
from ._reference_body import PlanetaryCoordinateContract, ReferenceBodyContract
from ._report import (
    AdapterCapability,
    AdapterError,
    AdapterFormatProfile,
    AdapterLoss,
    AdapterNegotiationResult,
    AdapterReport,
    AdapterRequirement,
    AdapterStatus,
    AdapterWaiver,
    compose_adapter_reports,
    negotiate_adapter,
    require_lossless,
)
from ._segy import (
    decode_segy_bytes,
    decode_segy_resource,
    DecodedSEGY,
    read_segy,
    SEGYDecodeError,
    SEGYRev1IEEEProfile,
)
from ._segy_rev2 import (
    decode_segy_rev2_bytes,
    decode_segy_rev2_resource,
    read_segy_rev2,
    SEGYRev2IEEEProfile,
)
from ._step import (
    decode_step_bytes,
    decode_step_resource,
    read_step,
    STEP_ENTITY_COVERAGE,
    StepSchema,
    write_step,
)
from ._time_reference import (
    convert_time,
    LeapSecondTable,
    TimeReferenceContract,
    TimeScale,
    TimeTransform,
)
from ._waveform_formats import (
    QualifiedWaveformCollection,
    QualifiedWaveformTrace,
    read_miniseed3,
    read_sac,
    read_stationxml,
    SeismicFormatDependencyError,
    StationChannelMetadata,
    StationXMLMetadata,
)
from ._well_formats import QualifiedWellLog, read_las_curve, WellFormatDependencyError


def __getattr__(name: str) -> Any:
    if name == "hep":
        value = import_module(".hep", __name__)
    elif name in _external_runtime_all:
        value = external_runtime.__dict__[name]
    elif name in (
        "import_occt_brep",
        "model_from_occt_shape",
        "persist_occt_shape",
        "read_occt_shape",
    ):
        value = import_module("._cad_occt", __name__).__dict__[name]
    elif name in ("PreparedOcctProjection", "prepare_occt_projection"):
        value = import_module("._cad_occt_projection", __name__).__dict__[name]
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "dafoam",
    "geant4_detector_design",
    "hep",
    "hfss_design",
    "xfoil",
    "external_runtime",
    "fmi",
    "helics",
    "opticstudio",
    "CoordinateTransformPlan",
    "BoreholeInterval",
    "BoreholeTrajectory",
    "PreparedBoreholeSampling",
    "CoordinateTransformResult",
    "execute_coordinate_transform",
    "GeodeticDependencyError",
    "GeospatialContract",
    "GeospatialTransform",
    "QualifiedGeospatialGrid",
    "ImportedBilbyResult",
    "HostInspectionConversion",
    "HostInspectionField",
    "HostInspectionFrame",
    "decode_layout_bytes",
    "decode_layout_resource",
    "LayoutAdapterError",
    "LayoutFormat",
    "LayoutImportPolicy",
    "LayoutImportResult",
    "LayoutLayerKey",
    "LayoutModel",
    "LayoutRegion",
    "read_layout",
    "MeshArrayArtifact",
    "MeshArrayAssociation",
    "MeshArrayBlock",
    "MeshArrayField",
    "MeshArraySelection",
    "export_geospatial_grid",
    "GeospatialDependencyError",
    "PyGMTRenderResult",
    "render_geospatial_grid",
    "XarrayGridExport",
    "AdapterCapability",
    "AdapterError",
    "AdapterFormatProfile",
    "AdapterLoss",
    "AdapterNegotiationResult",
    "AdapterReport",
    "AdapterStatus",
    "AdapterRequirement",
    "require_lossless",
    "AdapterWaiver",
    "BoundedResource",
    "ResourceLimits",
    "ResourceManifest",
    "ResourceReadError",
    "account_bounded_resource",
    "bounded_resource_from_bytes",
    "read_bounded_resource",
    "OpenedResource",
    "open_bounded_resource",
    "ArchiveLimits",
    "ArchiveMemberManifest",
    "BoundedArchive",
    "admit_zip_resource",
    "read_zip_members",
    "BoundedResourceSet",
    "ResourceMemberManifest",
    "ResourceSetLimits",
    "ResourceSetManifest",
    "ResourceSetReadError",
    "read_bounded_resource_set",
    "PublicationMode",
    "PublicationReceipt",
    "PublishedMemberReceipt",
    "ResourceSetPublicationReceipt",
    "publish_bytes",
    "publish_file",
    "publish_resource_set",
    "CarrierKind",
    "FormatCapability",
    "FormatDirection",
    "format_capabilities",
    "DecodedJSON",
    "DecodedText",
    "DecodedXML",
    "decode_json_resource",
    "decode_text_resource",
    "decode_xml_resource",
    "DecodedNumpyArchive",
    "DecodedNumpyArray",
    "NumpyFormatLimits",
    "decode_npy_resource",
    "decode_npz_resource",
    "HDF5DatasetManifest",
    "HDF5Limits",
    "HDF5Manifest",
    "inspect_hdf5_resource",
    "compose_adapter_reports",
    "negotiate_adapter",
    "decode_segy_bytes",
    "decode_segy_resource",
    "DecodedSEGY",
    "read_segy",
    "SEGYDecodeError",
    "SEGYRev1IEEEProfile",
    "convert_time",
    "LeapSecondTable",
    "TimeReferenceContract",
    "TimeScale",
    "TimeTransform",
    "GeospatialFormatDependencyError",
    "read_cf_netcdf_grid",
    "read_consolidated_zip_zarr_grid",
    "read_geotiff_grid",
    "decode_segy_rev2_bytes",
    "decode_segy_rev2_resource",
    "read_segy_rev2",
    "SEGYRev2IEEEProfile",
    "QualifiedWaveformCollection",
    "QualifiedWaveformTrace",
    "read_miniseed3",
    "read_sac",
    "read_bilby_result_json",
    "read_stationxml",
    "SeismicFormatDependencyError",
    "StationChannelMetadata",
    "StationXMLMetadata",
    "ElectricalTabularSurvey",
    "read_electrical_survey_csv",
    "MTImpedanceData",
    "read_edi_impedance",
    "read_emtf_xml_impedance",
    "GeodeticFormatDependencyError",
    "read_rinex_observations",
    "read_sinex_positions",
    "RINEXObservationData",
    "SINEXPositionSolution",
    "GeomagneticHarmonicModel",
    "ICGEMGravityModel",
    "read_geomagnetic_coefficients",
    "read_icgem_gfc",
    "QualifiedWellLog",
    "read_las_curve",
    "WellFormatDependencyError",
    "OpenPMDLaserEnvelopeError",
    "OpenPMDLaserEnvelopeExportResult",
    "OpenPMDLaserEnvelopeImportPolicy",
    "OpenPMDLaserEnvelopeImportResult",
    "OpenPMDLaserEnvelopeProfile",
    "read_openpmd_laser_envelope_hdf5",
    "write_openpmd_laser_envelope_hdf5",
    "OpenPMDParticleTrackError",
    "OpenPMDParticleTrackExportResult",
    "OpenPMDParticleTrackImportPolicy",
    "OpenPMDParticleTrackImportResult",
    "OpenPMDParticleTrackSelection",
    "read_openpmd_particle_tracks_hdf5",
    "write_openpmd_particle_tracks_hdf5",
    "OpenPMDADIOS2Artifact",
    "OpenPMDADIOS2Provider",
    "OpenPMDFieldRecordName",
    "OpenPMDMeshError",
    "OpenPMDMeshExportResult",
    "OpenPMDMeshGeometry",
    "OpenPMDMeshImportPolicy",
    "OpenPMDMeshImportResult",
    "OpenPMDMeshIteration",
    "OpenPMDMeshRecord",
    "convert_openpmd_hdf5_to_adios2",
    "read_openpmd_adios2",
    "read_openpmd_meshes_hdf5",
    "write_openpmd_meshes_hdf5",
    "OpenPMDPICExportResult",
    "OpenPMDPICImportResult",
    "OpenPMDPICLayout",
    "OpenPMDPICStreamReceipt",
    "OpenPMDPICStreamWriter",
    "read_openpmd_pic_state",
    "write_openpmd_pic_state",
    "PlanetaryCoordinateContract",
    "ReferenceBodyContract",
    "BlackHoleArtifactKind",
    "BlackHoleArtifactRights",
    "BlackHoleArtifactSchema",
    "BlackHoleArtifactUsePolicy",
    "map_black_hole_artifact",
    "map_field_artifact",
    "map_image_artifact",
    "map_numeric_model_artifact",
    "map_visibility_artifact",
    "map_waveform_artifact",
    "NeutralBlackHoleArtifact",
    "CompressibleKineticCaseIR",
    "LegacyKineticBoundaryImport",
    "import_packed_kinetic_boundary",
    "read_compressible_kinetic_yaml",
    "BRepArchiveReceipt",
    "CadCoverage",
    "CadCurveApproximation",
    "CadCurveFitPolicy",
    "CadExportPolicy",
    "CadExportResult",
    "CadImportPolicy",
    "CadImportResult",
    "CadInterchangeError",
    "CadRefusal",
    "CadRefusalReason",
    "STEP_ENTITY_COVERAGE",
    "StepSchema",
    "approximate_intersection_curve",
    "decode_step_bytes",
    "decode_step_resource",
    "load_brep_archive",
    "read_step",
    "save_brep_archive",
    "write_step",
    "read_cad",
    "decode_brep_text_bytes",
    "decode_brep_text_resource",
    "decode_iges_bytes",
    "decode_iges_resource",
    "import_occt_brep",
    "model_from_occt_shape",
    "persist_occt_shape",
    "read_brep_text",
    "read_iges",
    "read_occt_shape",
    "write_brep_text",
    "write_iges",
    "PreparedOcctProjection",
    "prepare_occt_projection",
    "BRepTextVersion",
]
__all__ += [name for name in _external_runtime_all if name not in __all__]
