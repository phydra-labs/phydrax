#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Generic external-representation interchange contracts."""

from . import (
    dafoam,
    energy_runtime,
    fmi,
    geant4_detector_design,
    helics,
    hfss_design,
    opticstudio,
    xfoil,
)
from ._borehole import BoreholeInterval, BoreholeTrajectory, PreparedBoreholeSampling
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
from ._inspection import (
    HostInspectionConversion,
    HostInspectionField,
    HostInspectionFrame,
)
from ._mesh_arrays import (
    MeshArrayArtifact,
    MeshArrayAssociation,
    MeshArrayBlock,
    MeshArrayField,
    MeshArraySelection,
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
from ._resource import (
    account_bounded_resource,
    bounded_resource_from_bytes,
    BoundedResource,
    read_bounded_resource,
    ResourceLimits,
    ResourceManifest,
    ResourceReadError,
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


__all__ = [
    "dafoam",
    "geant4_detector_design",
    "hfss_design",
    "xfoil",
    "energy_runtime",
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
    "HostInspectionConversion",
    "HostInspectionField",
    "HostInspectionFrame",
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
    "PlanetaryCoordinateContract",
    "ReferenceBodyContract",
]
