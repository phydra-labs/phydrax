#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic introspection catalog for explicit file-format capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from .._mesh_file_profiles import mesh_file_profiles
from ._report import AdapterFormatProfile


FormatDirection = Literal["read", "write", "append"]


class CarrierKind(StrEnum):
    """Physical carrier required by one exact file-format profile."""

    RESIDENT_FILE = "resident-file"
    SEEKABLE_FILE = "seekable-file"
    ARCHIVE = "archive"
    DIRECTORY_STORE = "directory-store"
    RESOURCE_SET = "resource-set"
    APPENDABLE_STREAM = "appendable-stream"


@dataclass(frozen=True, slots=True)
class FormatCapability:
    """Static declaration of one implemented format profile and direction set."""

    profile: AdapterFormatProfile
    domain: str
    extensions: tuple[str, ...]
    carrier: CarrierKind
    directions: tuple[FormatDirection, ...]
    optional_dependency: str | None
    semantic_capabilities: tuple[str, ...]
    known_losses: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.domain.strip():
            raise ValueError("Format capability domain must be non-empty.")
        if not self.directions or len(self.directions) != len(set(self.directions)):
            raise ValueError("Format capability directions must be non-empty and unique.")
        if any(
            direction not in ("read", "write", "append") for direction in self.directions
        ):
            raise ValueError("Format capability direction is invalid.")
        if len(self.extensions) != len(set(self.extensions)) or any(
            not extension.startswith(".") or extension != extension.lower()
            for extension in self.extensions
        ):
            raise ValueError(
                "Format capability extensions must be unique lowercase suffixes."
            )
        if len(self.semantic_capabilities) != len(set(self.semantic_capabilities)):
            raise ValueError("Format semantic capabilities must be unique.")
        if len(self.known_losses) != len(set(self.known_losses)):
            raise ValueError("Format known losses must be unique.")


def _capability(
    format_name: str,
    domain: str,
    extensions: tuple[str, ...],
    carrier: CarrierKind,
    directions: tuple[FormatDirection, ...],
    /,
    *,
    revision: str,
    dependency: str | None = None,
    capabilities: tuple[str, ...] = (),
    losses: tuple[str, ...] = (),
) -> FormatCapability:
    return FormatCapability(
        AdapterFormatProfile(format_name, qualifiers={"profile": revision}),
        domain,
        extensions,
        carrier,
        directions,
        dependency,
        capabilities,
        losses,
    )


_BASE_CAPABILITIES = tuple(
    sorted(
        (
            _capability(
                "phydrax-array-archive",
                "native",
                (".phx",),
                CarrierKind.ARCHIVE,
                ("read", "write"),
                revision="canonical",
                capabilities=("finite-json", "typed-arrays", "checksums"),
            ),
            _capability(
                "phydrax-ml-artifact",
                "ml",
                (".phxml",),
                CarrierKind.ARCHIVE,
                ("read", "write"),
                revision="canonical",
                capabilities=("model-recipe", "typed-leaves", "provenance"),
            ),
            _capability(
                "onnx",
                "ml",
                (".onnx",),
                CarrierKind.RESIDENT_FILE,
                ("write",),
                revision="explicit-opset",
                dependency="onnx-export",
                capabilities=("inference-graph",),
            ),
            _capability(
                "iree-vmfb-bundle",
                "ml",
                (".phxiree",),
                CarrierKind.RESOURCE_SET,
                ("read", "write"),
                revision="canonical",
                dependency="iree",
                capabilities=("compiled-module", "entrypoint-manifest"),
            ),
            _capability(
                "csv",
                "tabular",
                (".csv",),
                CarrierKind.RESIDENT_FILE,
                ("read", "write"),
                revision="explicit-schema",
                capabilities=("typed-columns",),
                losses=("domain-semantics-require-profile",),
            ),
            _capability(
                "json",
                "document",
                (".json",),
                CarrierKind.RESIDENT_FILE,
                ("read", "write"),
                revision="rfc8259-finite",
                capabilities=("duplicate-free-objects",),
            ),
            _capability(
                "yaml-chemical-mechanism",
                "chemistry",
                (".yaml", ".yml"),
                CarrierKind.RESIDENT_FILE,
                ("read",),
                revision="safe-explicit-schema",
                dependency="chemistry",
                capabilities=("species", "reactions"),
            ),
            _capability(
                "npy",
                "array",
                (".npy",),
                CarrierKind.RESIDENT_FILE,
                ("read",),
                revision="pickle-free",
                capabilities=("dense-array",),
            ),
            _capability(
                "npz",
                "array",
                (".npz",),
                CarrierKind.ARCHIVE,
                ("read",),
                revision="pickle-free",
                capabilities=("named-dense-arrays",),
            ),
            _capability(
                "hdf5-profiled",
                "array",
                (".h5", ".hdf5"),
                CarrierKind.SEEKABLE_FILE,
                ("read", "write"),
                revision="profile-required",
                dependency="h5py",
                capabilities=("hierarchical-arrays", "chunking"),
                losses=("no-generic-semantic-interpretation",),
            ),
            _capability(
                "cf-netcdf",
                "geospatial",
                (".nc", ".nc4"),
                CarrierKind.SEEKABLE_FILE,
                ("read",),
                revision="cf-profile",
                dependency="geophysical-data",
                capabilities=("named-dimensions", "coordinates", "units"),
            ),
            _capability(
                "zarr",
                "array",
                (".zarr",),
                CarrierKind.DIRECTORY_STORE,
                ("read",),
                revision="explicit-codec-profile",
                dependency="geophysical-data",
                capabilities=("chunked-arrays",),
            ),
            _capability(
                "grib",
                "geophysics",
                (".grib", ".grib2"),
                CarrierKind.SEEKABLE_FILE,
                ("read",),
                revision="provider-profile",
                dependency="geophysical-data",
                capabilities=("forecast-fields",),
            ),
            _capability(
                "geotiff",
                "geospatial",
                (".tif", ".tiff"),
                CarrierKind.SEEKABLE_FILE,
                ("read",),
                revision="explicit-raster-profile",
                dependency="geophysical-data",
                capabilities=("raster", "crs", "nodata"),
            ),
            _capability(
                "segy-rev1-ieee",
                "geophysics",
                (".segy", ".sgy"),
                CarrierKind.SEEKABLE_FILE,
                ("read",),
                revision="rev1-ieee",
                capabilities=("trace-headers", "samples"),
            ),
            _capability(
                "segy-rev2-ieee",
                "geophysics",
                (".segy", ".sgy"),
                CarrierKind.SEEKABLE_FILE,
                ("read",),
                revision="rev2-ieee",
                capabilities=("trace-headers", "samples"),
            ),
            _capability(
                "miniseed3",
                "seismology",
                (".mseed",),
                CarrierKind.SEEKABLE_FILE,
                ("read",),
                revision="3",
                dependency="geophysical-data",
                capabilities=("waveforms",),
            ),
            _capability(
                "sac",
                "seismology",
                (".sac",),
                CarrierKind.SEEKABLE_FILE,
                ("read",),
                revision="provider-profile",
                dependency="geophysical-data",
                capabilities=("waveforms",),
            ),
            _capability(
                "stationxml",
                "seismology",
                (".xml",),
                CarrierKind.RESIDENT_FILE,
                ("read",),
                revision="fdsn-stationxml",
                dependency="geophysical-data",
                capabilities=("station-metadata",),
            ),
            _capability(
                "sinex",
                "geodesy",
                (".snx",),
                CarrierKind.RESIDENT_FILE,
                ("read",),
                revision="explicit-subset",
                capabilities=("stations", "solutions"),
            ),
            _capability(
                "rinex",
                "geodesy",
                (".rnx", ".obs"),
                CarrierKind.SEEKABLE_FILE,
                ("read",),
                revision="provider-profile",
                dependency="geophysical-data",
                capabilities=("observations",),
            ),
            _capability(
                "las-well-log",
                "geophysics",
                (".las",),
                CarrierKind.RESIDENT_FILE,
                ("read",),
                revision="explicit-subset",
                dependency="geophysical-data",
                capabilities=("curves", "depth"),
            ),
            _capability(
                "edi",
                "electromagnetics",
                (".edi",),
                CarrierKind.RESIDENT_FILE,
                ("read",),
                revision="explicit-subset",
                capabilities=("impedance",),
            ),
            _capability(
                "emtf-xml",
                "electromagnetics",
                (".xml",),
                CarrierKind.RESIDENT_FILE,
                ("read",),
                revision="explicit-subset",
                capabilities=("impedance",),
            ),
            _capability(
                "icgem-gfc",
                "geophysics",
                (".gfc",),
                CarrierKind.RESIDENT_FILE,
                ("read",),
                revision="explicit-subset",
                capabilities=("spherical-harmonics",),
            ),
            _capability(
                "cf-radial",
                "measurement",
                (".nc",),
                CarrierKind.SEEKABLE_FILE,
                ("read",),
                revision="explicit-profile",
                dependency="radar-cfradial",
                capabilities=("radar-sweeps",),
            ),
            _capability(
                "las-point-cloud",
                "measurement",
                (".las", ".laz"),
                CarrierKind.SEEKABLE_FILE,
                ("read",),
                revision="explicit-profile",
                dependency="lidar-las",
                capabilities=("point-cloud",),
            ),
            _capability(
                "e57",
                "measurement",
                (".e57",),
                CarrierKind.SEEKABLE_FILE,
                ("read",),
                revision="explicit-profile",
                dependency="e57",
                capabilities=("point-cloud",),
            ),
            _capability(
                "rosbag",
                "measurement",
                (".bag", ".db3"),
                CarrierKind.RESOURCE_SET,
                ("read",),
                revision="explicit-profile",
                dependency="rosbag",
                capabilities=("messages", "timestamps"),
            ),
            _capability(
                "xtf",
                "measurement",
                (".xtf",),
                CarrierKind.SEEKABLE_FILE,
                ("read",),
                revision="explicit-profile",
                dependency="sonar-xtf",
                capabilities=("sonar-pings",),
            ),
            _capability(
                "dicom",
                "imaging",
                (".dcm",),
                CarrierKind.RESIDENT_FILE,
                ("read",),
                revision="explicit-sop-profiles",
                dependency="imaging-dicom",
                capabilities=("pixels", "patient-space", "radiotherapy"),
            ),
            _capability(
                "nifti",
                "imaging",
                (".nii", ".nii.gz"),
                CarrierKind.SEEKABLE_FILE,
                ("read", "write"),
                revision="nifti-1",
                dependency="imaging-nifti",
                capabilities=("volume", "affine", "units"),
            ),
            _capability(
                "mgz",
                "imaging",
                (".mgz",),
                CarrierKind.SEEKABLE_FILE,
                ("read",),
                revision="provider-profile",
                dependency="imaging-nifti",
                capabilities=("volume", "affine"),
            ),
            _capability(
                "openpmd-laser-envelope",
                "plasma",
                (".h5",),
                CarrierKind.SEEKABLE_FILE,
                ("read", "write"),
                revision="laser-envelope-subset",
                dependency="h5py",
                capabilities=("complex-field", "axes", "units"),
            ),
            _capability(
                "gmsh",
                "mesh",
                (".msh",),
                CarrierKind.SEEKABLE_FILE,
                ("read", "write"),
                revision="meshio-profile",
                dependency="meshing-gmsh",
                capabilities=("unstructured-cells", "fields"),
            ),
            _capability(
                "vtk-legacy",
                "mesh",
                (".vtk",),
                CarrierKind.SEEKABLE_FILE,
                ("read", "write"),
                revision="meshio-profile",
                dependency="meshio",
                capabilities=("unstructured-cells", "fields"),
                losses=("limited-metadata",),
            ),
            _capability(
                "vtu",
                "mesh",
                (".vtu",),
                CarrierKind.SEEKABLE_FILE,
                ("read", "write"),
                revision="vtk-xml-unstructured",
                dependency="meshio",
                capabilities=("unstructured-cells", "fields", "global-ids"),
                losses=("domain-semantics-require-report",),
            ),
            _capability(
                "stl",
                "surface",
                (".stl",),
                CarrierKind.SEEKABLE_FILE,
                ("read", "write"),
                revision="meshio-profile",
                dependency="meshio",
                capabilities=("triangles",),
                losses=("field-and-unit-metadata",),
            ),
            _capability(
                "obj",
                "surface",
                (".obj",),
                CarrierKind.SEEKABLE_FILE,
                ("read", "write"),
                revision="meshio-profile",
                dependency="meshio",
                capabilities=("surface-mesh",),
                losses=("scientific-field-metadata",),
            ),
            _capability(
                "xdmf",
                "mesh",
                (".xdmf", ".xmf"),
                CarrierKind.RESOURCE_SET,
                ("read", "write", "append"),
                revision="meshio-profile",
                dependency="meshio",
                capabilities=("time-series", "hdf5-arrays"),
            ),
            _capability(
                "step",
                "cad",
                (".step", ".stp"),
                CarrierKind.SEEKABLE_FILE,
                ("read",),
                revision="occt-profile",
                dependency="cad",
                capabilities=("brep-topology",),
            ),
            _capability(
                "iges",
                "cad",
                (".iges", ".igs"),
                CarrierKind.SEEKABLE_FILE,
                ("read",),
                revision="occt-profile",
                dependency="cad",
                capabilities=("brep-topology",),
            ),
            _capability(
                "touchstone",
                "circuit",
                (".s1p", ".s2p", ".s3p", ".s4p", ".snp"),
                CarrierKind.RESIDENT_FILE,
                ("read", "write"),
                revision="1.x-2.x-explicit",
                capabilities=("network-parameters", "reference-impedance"),
            ),
            _capability(
                "spice-netlist",
                "circuit",
                (".cir", ".sp", ".spice"),
                CarrierKind.RESIDENT_FILE,
                ("read",),
                revision="passive-linear-subset",
                capabilities=("linear-circuit",),
            ),
            _capability(
                "gdsii",
                "layout",
                (".gds", ".gdsii"),
                CarrierKind.SEEKABLE_FILE,
                ("read",),
                revision="gdstk-profile",
                dependency="layout-interop",
                capabilities=("polygons", "hierarchy"),
            ),
            _capability(
                "oasis",
                "layout",
                (".oas", ".oasis"),
                CarrierKind.SEEKABLE_FILE,
                ("read",),
                revision="gdstk-profile",
                dependency="layout-interop",
                capabilities=("polygons", "hierarchy"),
            ),
            _capability(
                "urdf",
                "robotics",
                (".urdf",),
                CarrierKind.RESIDENT_FILE,
                ("read",),
                revision="serial-tree-subset",
                capabilities=("links", "joints", "inertia"),
            ),
            _capability(
                "fmu",
                "fmi",
                (".fmu",),
                CarrierKind.ARCHIVE,
                ("read",),
                revision="fmi-2-model-exchange",
                dependency="fmi",
                capabilities=("model-description", "binary-provider"),
            ),
            _capability(
                "epw",
                "building-energy",
                (".epw",),
                CarrierKind.RESIDENT_FILE,
                ("read",),
                revision="energyplus-weather",
                capabilities=("weather-series",),
            ),
            _capability(
                "h5md",
                "atomistic",
                (".h5",),
                CarrierKind.SEEKABLE_FILE,
                ("read", "write", "append"),
                revision="explicit-profile",
                dependency="atomistic-interop",
                capabilities=("trajectory", "units"),
            ),
            _capability(
                "extended-xyz",
                "atomistic",
                (".xyz", ".extxyz"),
                CarrierKind.APPENDABLE_STREAM,
                ("read", "write", "append"),
                revision="explicit-profile",
                capabilities=("trajectory", "per-atom-fields"),
            ),
            _capability(
                "pdb-records",
                "atomistic",
                (".pdb",),
                CarrierKind.RESIDENT_FILE,
                ("read",),
                revision="atom-record-subset",
                capabilities=("atom-identity", "coordinates"),
            ),
            _capability(
                "wannier90-hr",
                "chemistry",
                (".dat",),
                CarrierKind.RESIDENT_FILE,
                ("read", "write"),
                revision="hr-explicit",
                capabilities=("hamiltonian", "lattice-translations"),
            ),
            _capability(
                "wannier90-mmn",
                "chemistry",
                (".mmn",),
                CarrierKind.RESIDENT_FILE,
                ("read", "write"),
                revision="mmn-explicit",
                capabilities=("overlaps", "k-neighbors"),
            ),
            _capability(
                "lhef",
                "hep",
                (".lhe", ".lhef"),
                CarrierKind.RESIDENT_FILE,
                ("read", "write"),
                revision="explicit-subset",
                capabilities=("events", "weights"),
            ),
            _capability(
                "hepmc3-ascii",
                "hep",
                (".hepmc",),
                CarrierKind.APPENDABLE_STREAM,
                ("read", "write"),
                revision="explicit-subset",
                capabilities=("event-graph",),
            ),
            _capability(
                "slha",
                "hep",
                (".slha",),
                CarrierKind.RESIDENT_FILE,
                ("read", "write"),
                revision="1-2-explicit",
                capabilities=("blocks", "decays"),
            ),
            _capability(
                "root-tree",
                "hep",
                (".root",),
                CarrierKind.SEEKABLE_FILE,
                ("read",),
                revision="explicit-tree-profile",
                dependency="hep-columnar",
                capabilities=("columnar-events",),
            ),
            _capability(
                "openmc-statepoint",
                "nuclear",
                (".h5",),
                CarrierKind.SEEKABLE_FILE,
                ("read",),
                revision="explicit-profile",
                dependency="openmc",
                capabilities=("tallies", "run-metadata"),
            ),
            _capability(
                "dagmc",
                "nuclear",
                (".h5m",),
                CarrierKind.SEEKABLE_FILE,
                ("read",),
                revision="explicit-profile",
                dependency="dagmc",
                capabilities=("geometry", "material-tags"),
            ),
            _capability(
                "fpml",
                "finance",
                (".xml",),
                CarrierKind.RESIDENT_FILE,
                ("read", "write"),
                revision="narrow-explicit-profile",
                capabilities=("trades", "terms"),
            ),
            _capability(
                "tle",
                "astrodynamics",
                (".tle",),
                CarrierKind.RESIDENT_FILE,
                ("read",),
                revision="two-line-element",
                capabilities=("orbit-elements",),
            ),
            _capability(
                "ccsds-kvn",
                "astrodynamics",
                (".oem", ".opm", ".aem", ".apm", ".tdm", ".cdm"),
                CarrierKind.RESIDENT_FILE,
                ("read",),
                revision="explicit-kvn-profiles",
                capabilities=("spacecraft-messages",),
            ),
            _capability(
                "xgboost-json-ubj",
                "ml",
                (".json", ".ubj"),
                CarrierKind.RESIDENT_FILE,
                ("read",),
                revision="schema-only",
                capabilities=("tree-ensemble",),
            ),
        ),
        key=lambda capability: (capability.profile.format, capability.profile.profile_id),
    )
)

_ADDITIONAL_CAPABILITIES = (
    _capability(
        "bilby-json",
        "inference",
        (".json",),
        CarrierKind.RESIDENT_FILE,
        ("read",),
        revision="current-plain-json",
        capabilities=("posterior-samples", "weights", "evidence"),
    ),
    _capability(
        "phydrax-velocimetry-archive",
        "velocimetry",
        (".phxv",),
        CarrierKind.ARCHIVE,
        ("read", "write"),
        revision="canonical",
        capabilities=("piv-fields", "calibration", "provenance"),
    ),
    _capability(
        "openpiv-text",
        "velocimetry",
        (".txt", ".dat"),
        CarrierKind.RESIDENT_FILE,
        ("read", "write"),
        revision="named-column-profile",
        capabilities=("planar-vector-field", "validity-flags"),
    ),
    _capability(
        "openptv-counted-text",
        "velocimetry",
        (),
        CarrierKind.RESOURCE_SET,
        ("read", "write"),
        revision="targets-rt-is-ptv-is",
        capabilities=("targets", "reconstructions", "tracks"),
    ),
    _capability(
        "pivlab-mat",
        "velocimetry",
        (".mat",),
        CarrierKind.SEEKABLE_FILE,
        ("read", "write"),
        revision="field-variable-subset",
        dependency="scipy",
        capabilities=("piv-fields", "processing-stage", "validity"),
    ),
    _capability(
        "pivlab-hdf5",
        "velocimetry",
        (".h5", ".hdf5"),
        CarrierKind.SEEKABLE_FILE,
        ("read", "write"),
        revision="field-variable-subset",
        dependency="h5py",
        capabilities=("piv-fields", "processing-stage", "validity"),
    ),
    _capability(
        "rmd17-npz",
        "atomistic",
        (".npz",),
        CarrierKind.ARCHIVE,
        ("read",),
        revision="documented-layout-aliases",
        capabilities=("coordinates", "energies", "forces", "sample-ids"),
    ),
    _capability(
        "energyplus-epw",
        "building-energy",
        (".epw",),
        CarrierKind.RESIDENT_FILE,
        ("read",),
        revision="standard-35-field-record",
        capabilities=("weather-series", "missingness", "calendar"),
    ),
    _capability(
        "energyplus-csv",
        "building-energy",
        (".csv",),
        CarrierKind.RESIDENT_FILE,
        ("read",),
        revision="explicit-variable-profile",
        capabilities=("named-time-series",),
    ),
    _capability(
        "radiance-matrix",
        "building-energy",
        (".mtx", ".dat"),
        CarrierKind.RESIDENT_FILE,
        ("read",),
        revision="explicit-matrix-profile",
        capabilities=("radiation-transfer",),
    ),
    _capability(
        "sonata",
        "electrophysiology",
        (".h5", ".json"),
        CarrierKind.RESOURCE_SET,
        ("read", "write"),
        revision="explicit-node-edge-spike-profile",
        capabilities=("nodes", "edges", "spikes", "configuration"),
    ),
    _capability(
        "swc",
        "electrophysiology",
        (".swc",),
        CarrierKind.RESIDENT_FILE,
        ("read",),
        revision="validated-morphology-profile",
        capabilities=("morphology", "stable-node-mapping"),
    ),
    _capability(
        "megascale-archive",
        "protein-biophysics",
        (".zip",),
        CarrierKind.ARCHIVE,
        ("read",),
        revision="pinned-zenodo-source",
        capabilities=("stability-tables", "rights-lineage"),
    ),
    _capability(
        "megascale-table",
        "protein-biophysics",
        (".csv",),
        CarrierKind.RESIDENT_FILE,
        ("read",),
        revision="processed-and-figure-profiles",
        capabilities=("stability-records", "censoring", "lineage"),
    ),
    _capability(
        "strand-displacement-xlsx",
        "nucleic-acid-biophysics",
        (".xlsx",),
        CarrierKind.ARCHIVE,
        ("read",),
        revision="scalar-ooxml-subset",
        capabilities=("fluorescence-traces", "plate-layout"),
    ),
    _capability(
        "strand-displacement-csv",
        "nucleic-acid-biophysics",
        (".csv",),
        CarrierKind.RESIDENT_FILE,
        ("read",),
        revision="prepared-trace-profile",
        capabilities=("fluorescence-traces", "raw-source-lineage"),
    ),
    _capability(
        "shapemapper-parsed-mut",
        "nucleic-acid-biophysics",
        (".mut",),
        CarrierKind.RESIDENT_FILE,
        ("read",),
        revision="dance-map-profile",
        capabilities=("mutation-counts", "mapped-depth", "preparation-identity"),
    ),
    _capability(
        "lattice-qcd-nersc",
        "lattice-field",
        (".nersc",),
        CarrierKind.SEEKABLE_FILE,
        ("read", "write"),
        revision="explicit-envelope",
        capabilities=("gauge-links", "lattice-layout"),
    ),
    _capability(
        "lattice-qcd-milc",
        "lattice-field",
        (".milc",),
        CarrierKind.SEEKABLE_FILE,
        ("read", "write"),
        revision="explicit-envelope",
        capabilities=("gauge-links", "lattice-layout"),
    ),
    _capability(
        "lattice-qcd-ildg-like",
        "lattice-field",
        (".ildg",),
        CarrierKind.SEEKABLE_FILE,
        ("read", "write"),
        revision="nonconformant-explicit-envelope",
        capabilities=("gauge-links", "lattice-layout"),
        losses=("not-full-ildg-conformance",),
    ),
    _capability(
        "moqui-npz",
        "radiation-biophysics",
        (".npz",),
        CarrierKind.ARCHIVE,
        ("read",),
        revision="score-affine-profile",
        capabilities=("dose-score", "affine", "uncertainty"),
    ),
    _capability(
        "metaimage-mha",
        "imaging",
        (".mha", ".mhd"),
        CarrierKind.SEEKABLE_FILE,
        ("read",),
        revision="explicit-profile",
        capabilities=("volume", "affine"),
    ),
    _capability(
        "openxraymc-hdf5",
        "radiation-biophysics",
        (".h5", ".hdf5"),
        CarrierKind.SEEKABLE_FILE,
        ("read",),
        revision="pinned-producer-profile",
        capabilities=("radiation-score", "run-identity"),
    ),
    _capability(
        "dna-damage-root",
        "radiation-biophysics",
        (".root",),
        CarrierKind.SEEKABLE_FILE,
        ("read",),
        revision="explicit-tree-profile",
        dependency="radiation-interop",
        capabilities=("damage-events",),
    ),
    _capability(
        "mcgpu-artifacts",
        "radiation-biophysics",
        (".raw", ".txt"),
        CarrierKind.RESOURCE_SET,
        ("read",),
        revision="configuration-and-score-profile",
        capabilities=("configuration", "voxel-scores"),
    ),
    _capability(
        "brep",
        "cad",
        (".brep", ".brp"),
        CarrierKind.SEEKABLE_FILE,
        ("read", "write"),
        revision="occt-native",
        dependency="cad",
        capabilities=("exact-topology", "surface-patches"),
    ),
    _capability(
        "imageio-raster",
        "imaging",
        (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
        CarrierKind.RESIDENT_FILE,
        ("read",),
        revision="provider-profile-required",
        dependency="imageio",
        capabilities=("scalar-image",),
        losses=("channel-selection-requires-explicit-loader",),
    ),
    _capability(
        "pygmt-render",
        "geospatial",
        (".png", ".pdf", ".eps"),
        CarrierKind.RESIDENT_FILE,
        ("write",),
        revision="signature-validated-render",
        dependency="geophysical-data",
        capabilities=("rendered-grid",),
    ),
)

_EXISTING_MESH_FORMATS = {
    capability.profile.format
    for capability in (*_BASE_CAPABILITIES, *_ADDITIONAL_CAPABILITIES)
}
_MESH_CAPABILITIES = tuple(
    _capability(
        profile.profile_id,
        "mesh",
        profile.extensions,
        (
            CarrierKind.RESOURCE_SET
            if profile.carrier == "resource-set"
            else CarrierKind.SEEKABLE_FILE
        ),
        tuple(
            direction
            for direction, available in (
                ("read", profile.readable),
                ("write", profile.writable),
            )
            if available
        ),
        revision="meshio-provider-profile",
        dependency="meshio",
        capabilities=("meshio-numeric-arrays",),
        losses=("profile-specific-semantics-require-adapter-report",),
    )
    for profile in mesh_file_profiles()
    if profile.profile_id not in _EXISTING_MESH_FORMATS
)
_CAPABILITIES = tuple(
    sorted(
        (*_BASE_CAPABILITIES, *_ADDITIONAL_CAPABILITIES, *_MESH_CAPABILITIES),
        key=lambda capability: (
            capability.profile.format,
            capability.profile.profile_id,
        ),
    )
)


def format_capabilities() -> tuple[FormatCapability, ...]:
    """Return the deterministic immutable format capability catalog."""

    return _CAPABILITIES


__all__ = [
    "CarrierKind",
    "FormatCapability",
    "FormatDirection",
    "format_capabilities",
]
