#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded HDF5 adapter for openPMD 1.1.0 electromagnetic mesh records.

The profile covers the ``E``, ``B``, ``J``, and ``rho`` mesh records in
``cartesian`` and ``thetaMode`` geometry. Stored values are converted through
the component ``unitSI`` (grid axes through ``gridUnitSI``) to SI and from SI
to the units of the bound ``ElectromagneticScaleContract``; every record's
``unitDimension`` must equal the scale's openPMD dimension of its quantity.
Writers publish one iteration per file of a ``fileBased`` series.

ADIOS2 BP4 is an optional provider route: a pinned openPMD-api
``openpmd-pipe`` converts between one bounded HDF5 image and one BP4
directory, so the ADIOS2 layout stays owned by that pinned release.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from io import BytesIO
from numbers import Integral
from pathlib import Path
from typing import assert_never, get_args, Literal, TypeAlias

import h5py
import numpy as np

from .._external_resource import (
    account_bounded_resource,
    bounded_resource_from_bytes,
    BoundedResource,
    read_bounded_resource,
    ResourceLimits,
    ResourceReadError,
)
from .._external_runtime import (
    PinnedExecutable,
    PinnedFileArtifact,
    PinnedFileOutputs,
    PinnedFileRequest,
    run_pinned_command,
)
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import ElectromagneticScaleContract
from .._publication import publish_bytes
from ..typing import parse
from ._openpmd_base import (
    attribute_text,
    BoundedHDF5Buffer,
    component_shape,
    component_values,
    LENGTH_DIMENSION,
    numeric_attribute,
    OPENPMD_MESH_REVISION,
    OpenPMDDimension,
    OpenPMDIterationTime,
    OpenPMDSeriesRoot,
    OpenPMDUnit,
    OpenPMDUnsupportedError,
    preflight_hdf5,
    read_grid_unit_si,
    read_iteration,
    read_record_unit,
    read_series_root,
    required_text,
    scalar_attribute,
    write_component,
    write_iteration,
    write_record_unit,
    write_series_root,
)
from ._report import AdapterReport, AdapterStatus


OpenPMDMeshGeometry: TypeAlias = Literal["cartesian", "thetaMode"]
OpenPMDFieldRecordName: TypeAlias = Literal["E", "B", "J", "rho"]
_DataOrder: TypeAlias = Literal["C", "F"]

_REVISION = OPENPMD_MESH_REVISION
MESHES_PATH = "meshes/"
_ASSUMPTIONS = (
    "openPMD 1.1.0 mesh records E, B, J, and rho",
    "thetaMode geometryParameters m=<modes>;imag=+ with 2*modes-1 mode planes",
)
_NAME = re.compile(r"^\w+$", re.ASCII)
_RECORD_ORDER: tuple[OpenPMDFieldRecordName, ...] = get_args(OpenPMDFieldRecordName)
_RECORD_QUANTITIES: dict[OpenPMDFieldRecordName, str] = {
    "E": "electric_field",
    "B": "magnetic_field",
    "J": "current_density",
    "rho": "charge_density",
}
_VECTOR_RECORDS: tuple[OpenPMDFieldRecordName, ...] = ("E", "B", "J")
CARTESIAN_AXES = ("x", "y", "z")
_THETA_AXES = ("r", "z")
_THETA_COMPONENTS = ("r", "t", "z")
# Geometries named by openPMD 1.1.0 that this profile does not interpret.
_UNINTERPRETED_GEOMETRIES = ("cylindrical", "spherical", "other")
# Decoding holds the stored float64 copy and its converted copy per value.
_DECODE_BYTES = 2 * 8
# openPMD-api writes exactly these BP4 files with engine profiling disabled.
_ADIOS2_FILES = ("data.0", "md.0", "md.idx")
_ADIOS2_CONFIG = json.dumps(
    {"adios2": {"engine": {"type": "bp4", "parameters": {"Profile": "Off"}}}},
    sort_keys=True,
)
_ADIOS2_FORMAT = "openPMD-1.1.0-ADIOS2-BP4"
_ADIOS2_MAXIMUM_FILES = 64


def series_name(value: str, /) -> str:
    """Validate one openPMD record, species, or series name (regex ``\\w+``)."""
    if not isinstance(value, str):
        raise TypeError("openPMD names must be strings.")
    if _NAME.fullmatch(value) is None:
        raise ValueError(
            f"openPMD name {value!r} must contain only letters, digits, and underscores."
        )
    return value


def iteration_index(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    index = int(value)
    if index < 0 or index >= 2**64:
        raise ValueError(f"{name} must be an unsigned 64-bit integer.")
    return index


def _finite_vector(values: object, count: int, name: str, /) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (count,) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must hold {count} finite values.")
    return array


def _component_names(
    name: OpenPMDFieldRecordName, geometry: OpenPMDMeshGeometry, /
) -> tuple[str, ...]:
    if name not in _VECTOR_RECORDS:
        return ()
    match geometry:
        case "cartesian":
            return CARTESIAN_AXES
        case "thetaMode":
            return _THETA_COMPONENTS
        case unreachable:
            assert_never(unreachable)


def scale_unit(
    units: dict[str, tuple[float, tuple[float, ...]]], quantity: str, /
) -> OpenPMDUnit:
    """The openPMD unit of one quantity in the units of a bound scale."""
    factor, powers = units[quantity]
    if len(powers) != 7:
        raise ValueError("Scale unit dimensions must hold seven openPMD powers.")
    dimension: OpenPMDDimension = (
        powers[0],
        powers[1],
        powers[2],
        powers[3],
        powers[4],
        powers[5],
        powers[6],
    )
    return OpenPMDUnit(factor, dimension)


@dataclass(frozen=True, slots=True, eq=False)
class OpenPMDMeshRecord:
    """One ``E``, ``B``, ``J``, or ``rho`` mesh record in a bound scale's units.

    ``components`` hold ``x, y, z`` (cartesian) or ``r, t, z`` (thetaMode)
    for the vector records and one array for ``rho``; every component of one
    record shares one shape. Cartesian arrays are indexed in ``axis_labels``
    order. ThetaMode arrays are ``[2M - 1, N_r, N_z]`` in openPMD mode order:
    mode 0, then the real and imaginary parts of modes ``1 … M - 1`` under the
    ``imag=+`` convention, so the field at azimuth ``θ`` is
    ``F_0 + Σ_m (Re F_m cos mθ + Im F_m sin mθ)``. ``grid_spacing`` and
    ``grid_global_offset`` (the start of the first cell) are in the scale
    length unit, ``time_offset`` in the scale time unit, and ``positions[c]``
    is the in-cell position of component ``c`` in ``[0, 1)`` per axis.
    """

    name: OpenPMDFieldRecordName
    geometry: OpenPMDMeshGeometry
    axis_labels: tuple[str, ...]
    grid_spacing: tuple[float, ...]
    grid_global_offset: tuple[float, ...]
    components: tuple[np.ndarray, ...]
    positions: tuple[tuple[float, ...], ...]
    time_offset: float = 0.0
    record_id: str = field(init=False)

    def __post_init__(self) -> None:
        name = parse(self.name, OpenPMDFieldRecordName, "name")
        geometry = parse(self.geometry, OpenPMDMeshGeometry, "geometry")
        labels = tuple(self.axis_labels)
        match geometry:
            case "cartesian":
                if (
                    not labels
                    or len(set(labels)) != len(labels)
                    or any(label not in CARTESIAN_AXES for label in labels)
                ):
                    raise ValueError(
                        "Cartesian axis_labels must be distinct labels among x, y, z."
                    )
                rank = len(labels)
            case "thetaMode":
                if labels != _THETA_AXES:
                    raise ValueError("thetaMode axis_labels must be ('r', 'z').")
                rank = 3
            case unreachable:
                assert_never(unreachable)
        count = len(labels)
        spacing = _finite_vector(self.grid_spacing, count, "grid_spacing")
        if np.any(spacing <= 0.0):
            raise ValueError("grid_spacing must be positive.")
        offset = _finite_vector(self.grid_global_offset, count, "grid_global_offset")
        time_offset = float(self.time_offset)
        if not np.isfinite(time_offset):
            raise ValueError("time_offset must be finite.")
        expected = max(len(_component_names(name, geometry)), 1)
        arrays = tuple(
            np.array(value, dtype=np.float64, copy=True) for value in self.components
        )
        if len(arrays) != expected:
            raise ValueError(f"Mesh record {name} requires {expected} components.")
        shape = arrays[0].shape
        if any(value.shape != shape for value in arrays):
            raise ValueError("Components of one mesh record must share one shape.")
        if len(shape) != rank or any(extent < 1 for extent in shape):
            raise ValueError(
                f"Mesh record {name} components must be nonempty rank-{rank} arrays."
            )
        if geometry == "thetaMode" and shape[0] % 2 == 0:
            raise ValueError("thetaMode records hold 2M - 1 mode planes.")
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError(f"Mesh record {name} values must be finite.")
        positions = tuple(
            tuple(float(value) for value in _finite_vector(item, count, "positions"))
            for item in self.positions
        )
        if len(positions) != expected or any(
            not 0.0 <= value < 1.0 for item in positions for value in item
        ):
            raise ValueError(
                "positions must give one in-cell position in [0, 1) per component."
            )
        for value in arrays:
            value.flags.writeable = False
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "axis_labels", labels)
        object.__setattr__(self, "grid_spacing", tuple(float(v) for v in spacing))
        object.__setattr__(self, "grid_global_offset", tuple(float(v) for v in offset))
        object.__setattr__(self, "components", arrays)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "time_offset", time_offset)
        object.__setattr__(
            self,
            "record_id",
            canonical_fingerprint(
                {
                    "kind": "openpmd-mesh-record",
                    "name": name,
                    "geometry": geometry,
                    "axis_labels": list(labels),
                    "grid_spacing": list(self.grid_spacing),
                    "grid_global_offset": list(self.grid_global_offset),
                    "positions": [list(item) for item in positions],
                    "time_offset": time_offset,
                    "components": array_tree_fingerprint(arrays),
                }
            ),
        )

    @property
    def component_names(self) -> tuple[str, ...]:
        """Component names of a vector record; empty for the scalar ``rho``."""
        return _component_names(self.name, self.geometry)

    @property
    def mode_count(self) -> int | None:
        """Azimuthal mode count ``M`` of a thetaMode record, else ``None``."""
        match self.geometry:
            case "cartesian":
                return None
            case "thetaMode":
                return (self.components[0].shape[0] + 1) // 2
            case unreachable:
                assert_never(unreachable)


@dataclass(frozen=True, slots=True, eq=False)
class OpenPMDMeshIteration:
    """Mesh records of one iteration; ``time`` and ``dt`` in the scale time unit."""

    iteration: int
    time: float
    dt: float
    records: tuple[OpenPMDMeshRecord, ...]

    def __post_init__(self) -> None:
        index = iteration_index(self.iteration, "iteration")
        time, step = float(self.time), float(self.dt)
        if not np.isfinite(time) or not np.isfinite(step):
            raise ValueError("Iteration time and dt must be finite.")
        records = tuple(self.records)
        if not records or any(
            not isinstance(value, OpenPMDMeshRecord) for value in records
        ):
            raise TypeError("records must be a nonempty sequence of OpenPMDMeshRecord.")
        names = [value.name for value in records]
        if len(set(names)) != len(names):
            raise ValueError("Mesh record names must be unique within an iteration.")
        object.__setattr__(self, "iteration", index)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "dt", step)
        object.__setattr__(
            self,
            "records",
            tuple(sorted(records, key=lambda value: _RECORD_ORDER.index(value.name))),
        )

    def record(self, name: OpenPMDFieldRecordName, /) -> OpenPMDMeshRecord:
        selected = parse(name, OpenPMDFieldRecordName, "name")
        for value in self.records:
            if value.name == selected:
                return value
        raise ValueError(f"Mesh record {selected} is absent from the iteration.")


@dataclass(frozen=True, slots=True)
class OpenPMDMeshImportPolicy:
    """Iteration, required records, decoded-payload budget, and metadata tolerance."""

    iteration: int
    records: Sequence[OpenPMDFieldRecordName] = _RECORD_ORDER
    maximum_decoded_bytes: int = 256 * 1024 * 1024
    metadata_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        index = iteration_index(self.iteration, "iteration")
        if isinstance(self.records, (str, bytes)) or not isinstance(
            self.records, Sequence
        ):
            raise TypeError("records must be a sequence of mesh record names.")
        names = tuple(
            parse(value, OpenPMDFieldRecordName, "records") for value in self.records
        )
        if not names or len(set(names)) != len(names):
            raise ValueError("records must be nonempty and unique.")
        if isinstance(self.maximum_decoded_bytes, bool) or not isinstance(
            self.maximum_decoded_bytes, Integral
        ):
            raise TypeError("maximum_decoded_bytes must be an integer.")
        maximum = int(self.maximum_decoded_bytes)
        tolerance = float(self.metadata_tolerance)
        if maximum <= 0:
            raise ValueError("maximum_decoded_bytes must be positive.")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("metadata_tolerance must be finite and nonnegative.")
        object.__setattr__(self, "iteration", index)
        object.__setattr__(self, "records", tuple(sorted(names, key=_RECORD_ORDER.index)))
        object.__setattr__(self, "maximum_decoded_bytes", maximum)
        object.__setattr__(self, "metadata_tolerance", tolerance)


class OpenPMDMeshError(ValueError):
    """Rejected openPMD mesh or PIC-state conversion with an auditable report."""

    status: AdapterStatus
    report: AdapterReport

    def __init__(self, message: str, report: AdapterReport, /) -> None:
        self.status = report.status
        self.report = report
        super().__init__(str(message))


@dataclass(frozen=True, slots=True, eq=False)
class OpenPMDMeshImportResult:
    iteration: OpenPMDMeshIteration
    resource: BoundedResource
    report: AdapterReport
    policy: OpenPMDMeshImportPolicy


@dataclass(frozen=True, slots=True)
class OpenPMDMeshExportResult:
    path: Path
    resource: BoundedResource
    report: AdapterReport


def refusal(
    status: AdapterStatus,
    source_id: str,
    message: str,
    target_format: str,
    assumptions: tuple[str, ...],
    /,
) -> OpenPMDMeshError:
    """Invalid report and exception for one refused openPMD import."""
    failure_id = canonical_fingerprint(
        {
            "kind": "openpmd-mesh-failure",
            "source": source_id,
            "target": target_format,
            "status": int(status),
            "message": str(message),
        }
    )
    report = AdapterReport(
        status,
        _REVISION.format_id,
        target_format,
        source_id=source_id,
        target_id=failure_id,
        assumptions=assumptions,
    )
    return OpenPMDMeshError(message, report)


# Structural scan ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Component:
    item: h5py.Dataset | h5py.Group
    shape: tuple[int, ...]
    unit: OpenPMDUnit
    position: np.ndarray


@dataclass(frozen=True, slots=True)
class ScannedMeshRecord:
    """One mesh record whose structure, units, and extents passed preflight.

    ``labels``, ``spacing_si``, ``offset_si``, and component positions follow
    the stored (C-ordered) array axes; ``time_offset`` is in iteration time units.
    """

    name: OpenPMDFieldRecordName
    geometry: OpenPMDMeshGeometry
    labels: tuple[str, ...]
    spacing_si: np.ndarray
    offset_si: np.ndarray
    time_offset: float
    components: tuple[_Component, ...]

    @property
    def element_count(self) -> int:
        return sum(int(np.prod(value.shape)) for value in self.components)


def _geometry(
    attributes: h5py.AttributeManager, /
) -> tuple[OpenPMDMeshGeometry, int | None]:
    text = required_text(attributes, "geometry")
    if text in _UNINTERPRETED_GEOMETRIES:
        raise OpenPMDUnsupportedError(f"openPMD {text} mesh geometry is unsupported.")
    geometry = parse(text, OpenPMDMeshGeometry, "geometry")
    match geometry:
        case "cartesian":
            return geometry, None
        case "thetaMode":
            entries: dict[str, str] = {}
            for token in required_text(attributes, "geometryParameters").split(";"):
                key, separator, value = token.strip().partition("=")
                if not separator or not key or key in entries:
                    raise ValueError(
                        "thetaMode geometryParameters must be distinct key=value pairs."
                    )
                entries[key] = value.strip()
            if "m" not in entries or "imag" not in entries:
                raise ValueError("thetaMode geometryParameters require m and imag.")
            if entries["imag"] != "+":
                raise OpenPMDUnsupportedError(
                    "Only the imag=+ thetaMode convention is supported."
                )
            modes = entries["m"]
            if not modes.isdecimal() or int(modes) < 1:
                raise ValueError("thetaMode m must be a positive mode count.")
            return geometry, int(modes)
        case unreachable:
            assert_never(unreachable)


def _axis_labels(attributes: h5py.AttributeManager, /) -> tuple[str, ...]:
    if "axisLabels" not in attributes:
        raise ValueError("Missing required openPMD attribute axisLabels.")
    values = np.asarray(attributes["axisLabels"])
    # openPMD-api stores one-element attribute arrays as scalars.
    if values.ndim == 0:
        return (attribute_text(values, "axisLabels"),)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("axisLabels must be a nonempty one-dimensional array.")
    return tuple(attribute_text(value, "axisLabels") for value in values)


def _axis_attribute(
    attributes: h5py.AttributeManager, name: str, rank: int, /
) -> np.ndarray:
    """One value per axis; a one-axis vector may be stored as a scalar."""
    if rank == 1 and name in attributes and np.asarray(attributes[name]).shape == ():
        return numeric_attribute(attributes, name, ()).reshape(1)
    return numeric_attribute(attributes, name, (rank,))


def _data_order(attributes: h5py.AttributeManager, rank: int, /) -> _DataOrder:
    if "dataOrder" not in attributes:
        # openPMD 1.1.0 lets one-dimensional records omit dataOrder.
        if rank == 1:
            return "C"
        raise ValueError("Missing required openPMD attribute dataOrder.")
    return parse(required_text(attributes, "dataOrder"), _DataOrder, "dataOrder")


def scan_mesh_record(
    meshes: h5py.Group,
    name: OpenPMDFieldRecordName,
    dimension: OpenPMDDimension,
    tolerance: float,
    /,
) -> ScannedMeshRecord:
    """Validate one record's structure and units without reading its payload."""
    if name not in meshes:
        raise ValueError(f"Mesh record {name} is absent from the iteration.")
    item = meshes[name]
    path = f"{meshes.name}/{name}"
    if name in _VECTOR_RECORDS and (
        not isinstance(item, h5py.Group) or "value" in item.attrs
    ):
        raise ValueError(f"Mesh record {name} must be a vector record group.")
    attributes = item.attrs
    geometry, modes = _geometry(attributes)
    labels = _axis_labels(attributes)
    rank = len(labels)
    order = _data_order(attributes, rank)
    spacing = _axis_attribute(attributes, "gridSpacing", rank)
    offset = _axis_attribute(attributes, "gridGlobalOffset", rank)
    if np.any(spacing <= 0.0):
        raise ValueError(f"gridSpacing of mesh record {name} must be positive.")
    grid_unit = read_grid_unit_si(
        attributes, _REVISION, (LENGTH_DIMENSION,) * rank, tolerance
    )
    time_offset = scalar_attribute(attributes, "timeOffset")
    names = _component_names(name, geometry)
    if isinstance(item, h5py.Group) and names:
        missing = [value for value in names if value not in item]
        if missing:
            raise OpenPMDUnsupportedError(
                f"Mesh record {name} lacks components {missing}."
            )
        items = tuple((item[value], f"{path}/{value}") for value in names)
    else:
        items = ((item, path),)
    components: list[_Component] = []
    for component, component_path in items:
        shape = component_shape(component, component_path)
        position = _axis_attribute(component.attrs, "position", rank)
        if np.any(position < 0.0) or np.any(position >= 1.0):
            raise ValueError(f"Component {component_path} position must lie in [0, 1).")
        unit = read_record_unit(attributes, component.attrs, dimension, tolerance)
        components.append(_Component(component, shape, unit, position))
    shapes = sorted({value.shape for value in components})
    if len(shapes) != 1:
        raise ValueError(
            f"Mesh record {name} is truncated: component extents {shapes} disagree."
        )
    shape = shapes[0]
    if any(extent < 1 for extent in shape):
        raise ValueError(f"Mesh record {name} must be nonempty.")
    match geometry:
        case "cartesian":
            if len(set(labels)) != rank or any(
                label not in CARTESIAN_AXES for label in labels
            ):
                raise OpenPMDUnsupportedError(
                    "Cartesian axisLabels must be distinct labels among x, y, z."
                )
            if len(shape) != rank:
                raise ValueError(f"Mesh record {name} rank differs from its axisLabels.")
            if order == "F":
                # Fortran-ordered attributes list the stored axes in reverse.
                labels = labels[::-1]
                spacing, offset, grid_unit = spacing[::-1], offset[::-1], grid_unit[::-1]
                components = [
                    _Component(value.item, value.shape, value.unit, value.position[::-1])
                    for value in components
                ]
        case "thetaMode":
            if order != "C":
                raise OpenPMDUnsupportedError(
                    "Only C-ordered thetaMode records are supported."
                )
            if labels not in (_THETA_AXES, _THETA_AXES[::-1]):
                raise OpenPMDUnsupportedError(
                    "thetaMode axisLabels must be ('r', 'z') or ('z', 'r')."
                )
            if modes is None or len(shape) != 3 or shape[0] != 2 * modes - 1:
                raise ValueError(
                    f"thetaMode record {name} must hold 2m - 1 mode planes of [r, z]."
                )
        case unreachable:
            assert_never(unreachable)
    return ScannedMeshRecord(
        name,
        geometry,
        labels,
        spacing * grid_unit,
        offset * grid_unit,
        time_offset,
        tuple(components),
    )


def decode_mesh_record(
    scanned: ScannedMeshRecord,
    units: dict[str, tuple[float, tuple[float, ...]]],
    time_unit_si: float,
    /,
) -> OpenPMDMeshRecord:
    """Read one preflighted record into scale units, Cartesian axes in x, y, z order."""
    factor = np.float64(units[_RECORD_QUANTITIES[scanned.name]][0])
    arrays = []
    for component in scanned.components:
        values = (
            component.unit.to_si(component_values(component.item, component.shape))
            / factor
        )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Mesh record {scanned.name} payload must be finite.")
        arrays.append(values)
    labels = scanned.labels
    positions = [value.position for value in scanned.components]
    length = np.float64(units["length"][0])
    spacing, offset = scanned.spacing_si / length, scanned.offset_si / length
    if scanned.geometry == "thetaMode" and labels != _THETA_AXES:
        # Stored [mode, z, r] planes (e.g. WarpX) are returned as [mode, r, z].
        arrays = [np.transpose(value, (0, 2, 1)) for value in arrays]
        positions = [value[::-1] for value in positions]
        labels = _THETA_AXES
        spacing, offset = spacing[::-1], offset[::-1]
    if scanned.geometry == "cartesian":
        order = tuple(
            sorted(
                range(len(labels)), key=lambda axis: CARTESIAN_AXES.index(labels[axis])
            )
        )
        arrays = [np.transpose(value, order) for value in arrays]
        positions = [value[list(order)] for value in positions]
        labels = tuple(labels[axis] for axis in order)
        spacing, offset = spacing[list(order)], offset[list(order)]
    return OpenPMDMeshRecord(
        scanned.name,
        scanned.geometry,
        labels,
        tuple(spacing.tolist()),
        tuple(offset.tolist()),
        tuple(arrays),
        tuple(tuple(value.tolist()) for value in positions),
        scanned.time_offset * time_unit_si / units["time"][0],
    )


def record_dimension(
    units: dict[str, tuple[float, tuple[float, ...]]], name: OpenPMDFieldRecordName, /
) -> OpenPMDDimension:
    return scale_unit(units, _RECORD_QUANTITIES[name]).unit_dimension


def open_iteration(
    handle: h5py.File, iteration: int, /
) -> tuple[OpenPMDSeriesRoot, h5py.Group, OpenPMDIterationTime]:
    """Validated root, iteration group, and time of one openPMD 1.1.0 file."""
    root = read_series_root(handle, _REVISION)
    group, time = read_iteration(handle, iteration)
    return root, group, time


def records_group(group: h5py.Group, path: str | None, name: str, /) -> h5py.Group:
    if path is None:
        raise ValueError(f"Missing required openPMD attribute {name}.")
    relative = path.rstrip("/")
    if relative not in group or not isinstance(group[relative], h5py.Group):
        raise ValueError(f"The openPMD {name} group is absent from the iteration.")
    return group[relative]


def _decode_meshes(
    resource: BoundedResource,
    policy: OpenPMDMeshImportPolicy,
    units: dict[str, tuple[float, tuple[float, ...]]],
    /,
) -> tuple[OpenPMDMeshIteration, int, int, int]:
    tolerance = policy.metadata_tolerance
    with h5py.File(BytesIO(resource.data), "r") as handle:
        inventory = preflight_hdf5(handle, resource.manifest.limits)
        root, group, time = open_iteration(handle, policy.iteration)
        meshes = records_group(group, root.meshes_path, "meshesPath")
        scanned = tuple(
            scan_mesh_record(meshes, name, record_dimension(units, name), tolerance)
            for name in policy.records
        )
        elements = sum(value.element_count for value in scanned)
        inventory.require_budget(
            resource.manifest.limits,
            policy.maximum_decoded_bytes,
            elements * _DECODE_BYTES,
        )
        records = tuple(
            decode_mesh_record(value, units, time.time_unit_si) for value in scanned
        )
    time_unit = time.time_unit_si / units["time"][0]
    return (
        OpenPMDMeshIteration(
            policy.iteration, time.time * time_unit, time.dt * time_unit, records
        ),
        inventory.maximum_depth,
        inventory.object_count + elements,
        inventory.attribute_count,
    )


def read_openpmd_meshes_hdf5(
    resource: BoundedResource,
    policy: OpenPMDMeshImportPolicy,
    /,
    *,
    scale: ElectromagneticScaleContract,
) -> OpenPMDMeshImportResult:
    """Read E/B/J/rho mesh records of one iteration from a bounded HDF5 image.

    The complete HDF5 tree, every selected record's structure, units, and
    extents, and the decoded-byte budget are validated before any mesh payload
    is read. Refusals raise ``OpenPMDMeshError`` with an invalid report:
    unsupported geometry, geometry parameters, data order, labels, or standard
    versions are ``UNSUPPORTED_REQUIRED_SEMANTIC``; missing or malformed
    attributes, truncated components, unit-dimension mismatches, and damaged
    HDF5 images are ``MALFORMED_SOURCE``; resource-limit overflow is
    ``INCONSISTENT_SOURCE``.
    """
    if not isinstance(resource, BoundedResource):
        raise TypeError("resource must be a BoundedResource.")
    if not isinstance(policy, OpenPMDMeshImportPolicy):
        raise TypeError("policy must be an OpenPMDMeshImportPolicy.")
    if not isinstance(scale, ElectromagneticScaleContract):
        raise TypeError("scale must be an ElectromagneticScaleContract.")
    units = scale.unit_si_map()
    source_id = resource.manifest.manifest_id
    try:
        iteration, depth, nodes, attributes = _decode_meshes(resource, policy, units)
    except ResourceReadError as error:
        raise refusal(
            AdapterStatus.INCONSISTENT_SOURCE,
            source_id,
            str(error),
            "OpenPMDMeshIteration",
            _ASSUMPTIONS,
        ) from error
    except OpenPMDUnsupportedError as error:
        raise refusal(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            source_id,
            str(error),
            "OpenPMDMeshIteration",
            _ASSUMPTIONS,
        ) from error
    except (OSError, KeyError, UnicodeError, TypeError, ValueError) as error:
        raise refusal(
            AdapterStatus.MALFORMED_SOURCE,
            source_id,
            str(error),
            "OpenPMDMeshIteration",
            _ASSUMPTIONS,
        ) from error
    accounted = account_bounded_resource(
        resource, depth=depth, nodes=nodes, attributes=attributes, losses=0
    )
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        _REVISION.format_id,
        "OpenPMDMeshIteration",
        source_id=accounted.manifest.manifest_id,
        target_id=canonical_fingerprint(
            {
                "kind": "openpmd-mesh-import",
                "scale": scale.scale_id,
                "iteration": iteration.iteration,
                "time": iteration.time,
                "dt": iteration.dt,
                "records": [value.record_id for value in iteration.records],
            }
        ),
        coordinate_mapping=(
            "component values * unitSI -> SI -> scale units of the record quantity",
            "gridSpacing/gridGlobalOffset * gridUnitSI -> scale length",
            "(time, dt, timeOffset) * timeUnitSI -> scale time",
            "Cartesian axes reordered to x, y, z; Fortran dataOrder reversed",
        ),
        preserved_fields=(
            "E, B, J, rho components",
            "geometry and azimuthal mode planes",
            "grid spacing and global offset",
            "component staggering positions",
            "record time offsets",
        ),
        assumptions=_ASSUMPTIONS,
    )
    return OpenPMDMeshImportResult(iteration, accounted, report, policy)


# Export -----------------------------------------------------------------------------


def write_mesh_record(
    meshes: h5py.Group,
    record: OpenPMDMeshRecord,
    units: dict[str, tuple[float, tuple[float, ...]]],
    /,
) -> None:
    """Write one record in scale units with its ``unitSI``/``unitDimension``."""
    unit = scale_unit(units, _RECORD_QUANTITIES[record.name])
    names = record.component_names
    if names:
        group = meshes.create_group(record.name)
        components = tuple(
            write_component(group, name, values)
            for name, values in zip(names, record.components, strict=True)
        )
        attributes = group.attrs
    else:
        # A scalar record is its own component.
        scalar = write_component(meshes, record.name, record.components[0])
        components = (scalar,)
        attributes = scalar.attrs
    attributes["geometry"] = np.bytes_(record.geometry)
    if record.mode_count is not None:
        attributes["geometryParameters"] = np.bytes_(f"m={record.mode_count};imag=+")
    attributes["dataOrder"] = np.bytes_("C")
    attributes["axisLabels"] = np.asarray(
        [label.encode("ascii") for label in record.axis_labels]
    )
    attributes["gridSpacing"] = np.asarray(record.grid_spacing, dtype=np.float64)
    attributes["gridGlobalOffset"] = np.asarray(
        record.grid_global_offset, dtype=np.float64
    )
    # Grid values stay in the scale length unit; gridUnitSI converts them.
    attributes["gridUnitSI"] = np.float64(units["length"][0])
    attributes["timeOffset"] = np.float64(record.time_offset)
    for component, position in zip(components, record.positions, strict=True):
        component.attrs["position"] = np.asarray(position, dtype=np.float64)
        write_record_unit(attributes, component.attrs, unit)


def require_encoding_budget(element_count: int, limits: ResourceLimits, /) -> None:
    if element_count * 8 > limits.max_bytes or element_count > limits.max_nodes:
        raise ResourceReadError("limit", "openPMD output exceeds its resource limits.")


def write_openpmd_meshes_hdf5(
    directory: str | Path,
    iteration: OpenPMDMeshIteration,
    /,
    *,
    scale: ElectromagneticScaleContract,
    limits: ResourceLimits,
    series: str = "fields",
) -> OpenPMDMeshExportResult:
    """Encode and exclusively publish one iteration as ``<series>_<iteration>.h5``.

    The file is one member of the ``fileBased`` series ``<series>_%T.h5``.
    Values, grids, and times are stored in the scale's units with ``unitSI``,
    ``gridUnitSI``, ``timeUnitSI``, and ``unitDimension`` from
    ``scale.unit_si_map()``; arrays are written C-ordered in ``axis_labels``
    order, and records whose values all agree become constant components.
    """
    if not isinstance(iteration, OpenPMDMeshIteration):
        raise TypeError("iteration must be an OpenPMDMeshIteration.")
    if not isinstance(scale, ElectromagneticScaleContract):
        raise TypeError("scale must be an ElectromagneticScaleContract.")
    if not isinstance(limits, ResourceLimits):
        raise TypeError("limits must be ResourceLimits.")
    name = series_name(series)
    require_encoding_budget(
        sum(value.size for record in iteration.records for value in record.components),
        limits,
    )
    units = scale.unit_si_map()
    file_format = f"{name}_%T.h5"
    buffer = BoundedHDF5Buffer(limits.max_bytes)
    with h5py.File(buffer, "w") as handle:
        write_series_root(
            handle,
            _REVISION,
            meshes_path=MESHES_PATH,
            particles_path=None,
            file_format=file_format,
        )
        group = write_iteration(
            handle,
            iteration.iteration,
            OpenPMDIterationTime(iteration.time, iteration.dt, units["time"][0]),
        )
        meshes = group.create_group(MESHES_PATH.rstrip("/"))
        for record in iteration.records:
            write_mesh_record(meshes, record, units)
    data = buffer.getvalue()
    destination = Path(directory) / file_format.replace("%T", str(iteration.iteration))
    resource = bounded_resource_from_bytes(
        data, limits=limits, source_path=str(destination)
    )
    publish_bytes(destination, data, maximum_bytes=limits.max_bytes, mode="exclusive")
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        "OpenPMDMeshIteration",
        _REVISION.format_id,
        source_id=canonical_fingerprint(
            {
                "kind": "openpmd-mesh-export",
                "scale": scale.scale_id,
                "iteration": iteration.iteration,
                "time": iteration.time,
                "dt": iteration.dt,
                "records": [value.record_id for value in iteration.records],
            }
        ),
        target_id=resource.manifest.manifest_id,
        coordinate_mapping=(
            "scale units -> stored values with unitSI of the record quantity",
            "scale length grid -> gridSpacing/gridGlobalOffset with gridUnitSI",
            "scale time -> time, dt, and timeOffset with timeUnitSI",
        ),
        preserved_fields=(
            "E, B, J, rho components",
            "geometry and azimuthal mode planes",
            "grid spacing and global offset",
            "component staggering positions",
            "record time offsets",
        ),
        assumptions=_ASSUMPTIONS,
    )
    return OpenPMDMeshExportResult(destination, resource, report)


# ADIOS2 provider route -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OpenPMDADIOS2Provider:
    """Pinned openPMD-api ``openpmd-pipe`` converting HDF5 and ADIOS2 BP4.

    The conversion is performed entirely by the pinned openPMD-api release;
    phydrax stages bounded bytes in and admits only declared artifacts out.
    """

    executable: PinnedExecutable
    timeout: float = 600.0

    def __post_init__(self) -> None:
        if not isinstance(self.executable, PinnedExecutable):
            raise TypeError("executable must be a PinnedExecutable.")
        timeout = float(self.timeout)
        if not np.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout must be finite and positive.")
        object.__setattr__(self, "timeout", timeout)

    @property
    def provider_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "openpmd-adios2-provider",
                "executable": self.executable.sha256,
                "version": self.executable.version,
                "configuration": _ADIOS2_CONFIG,
            }
        )


@dataclass(frozen=True, slots=True)
class OpenPMDADIOS2Artifact:
    """One published BP4 directory holding a single openPMD iteration."""

    path: Path
    iteration: int
    files: tuple[PinnedFileArtifact, ...]
    report: AdapterReport

    @property
    def size_bytes(self) -> int:
        return sum(value.size_bytes for value in self.files)


def _file_based_member(resource: BoundedResource, /) -> tuple[str, int]:
    """The ``%T`` stem (without ``.h5``) and iteration of one fileBased member."""
    with h5py.File(BytesIO(resource.data), "r") as handle:
        preflight_hdf5(handle, resource.manifest.limits)
        root = read_series_root(handle, _REVISION)
        if "data" not in handle or not isinstance(handle["data"], h5py.Group):
            raise ValueError("The openPMD file holds no iterations.")
        names = tuple(handle["data"])
    if (
        root.iteration_encoding != "fileBased"
        or not root.iteration_format.endswith(".h5")
        or len(names) != 1
        or not names[0].isdecimal()
    ):
        raise OpenPMDUnsupportedError(
            "The ADIOS2 route converts single-iteration fileBased HDF5 series members."
        )
    return root.iteration_format[: -len(".h5")], int(names[0])


def convert_openpmd_hdf5_to_adios2(
    provider: OpenPMDADIOS2Provider,
    resource: BoundedResource,
    directory: str | Path,
    /,
    *,
    limits: ResourceLimits,
) -> OpenPMDADIOS2Artifact:
    """Convert one fileBased HDF5 member into ``<member>.bp`` beneath ``directory``.

    The BP4 files are admitted against ``limits.max_bytes`` each and in total
    and are published all together or not at all.
    """
    if not isinstance(provider, OpenPMDADIOS2Provider):
        raise TypeError("provider must be an OpenPMDADIOS2Provider.")
    if not isinstance(resource, BoundedResource):
        raise TypeError("resource must be a BoundedResource.")
    if not isinstance(limits, ResourceLimits):
        raise TypeError("limits must be ResourceLimits.")
    stem, iteration = _file_based_member(resource)
    member = stem.replace("%T", str(iteration))
    bp = f"{member}.bp"
    destination = Path(directory)
    result = run_pinned_command(
        provider.executable,
        (
            "--infile",
            f"{stem}.h5",
            "--outfile",
            f"{stem}.bp",
            "--outconfig",
            _ADIOS2_CONFIG,
        ),
        inputs={f"{member}.h5": resource.data},
        timeout=provider.timeout,
        max_output_bytes=limits.max_bytes,
        artifacts=PinnedFileOutputs(
            str(destination),
            tuple(
                PinnedFileRequest(f"{bp}/{name}", limits.max_bytes)
                for name in _ADIOS2_FILES
            ),
            limits.max_bytes,
        ),
    )
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        _REVISION.format_id,
        _ADIOS2_FORMAT,
        source_id=resource.manifest.manifest_id,
        target_id=canonical_fingerprint(
            {
                "kind": "openpmd-adios2-export",
                "provider": provider.provider_id,
                "files": [[value.path, value.sha256] for value in result.file_artifacts],
            }
        ),
        coordinate_mapping=("openpmd-pipe HDF5 -> ADIOS2 BP4 of the same series",),
        preserved_fields=("every openPMD record, attribute, and iteration",),
        assumptions=(
            f"openPMD-api openpmd-pipe {provider.executable.version} owns the "
            "ADIOS2 layout",
        ),
    )
    return OpenPMDADIOS2Artifact(
        destination / bp, iteration, result.file_artifacts, report
    )


def read_openpmd_adios2(
    provider: OpenPMDADIOS2Provider,
    path: str | Path,
    /,
    *,
    limits: ResourceLimits,
) -> BoundedResource:
    """Convert one bounded BP4 directory into an in-memory HDF5 openPMD image.

    Every regular file of the directory is read through the bounded resource
    substrate (at most ``limits.max_bytes`` in total), staged for the pinned
    ``openpmd-pipe``, and the resulting group-based HDF5 image is returned for
    the bounded HDF5 readers of this package.
    """
    if not isinstance(provider, OpenPMDADIOS2Provider):
        raise TypeError("provider must be an OpenPMDADIOS2Provider.")
    if not isinstance(limits, ResourceLimits):
        raise TypeError("limits must be ResourceLimits.")
    source = Path(path)
    if source.suffix != ".bp" or source.is_symlink() or not source.is_dir():
        raise ValueError("path must be an ADIOS2 .bp directory.")
    names = sorted(os.listdir(source))
    if not names or len(names) > _ADIOS2_MAXIMUM_FILES:
        raise ResourceReadError("limit", "ADIOS2 directory file count is out of bounds.")
    inputs: dict[str, bytes] = {}
    total = 0
    for name in names:
        member = read_bounded_resource(
            f"{source.name}/{name}", trusted_root=source.parent, limits=limits
        )
        total += len(member.data)
        if total > limits.max_bytes:
            raise ResourceReadError("limit", "ADIOS2 directory exceeds max_bytes.")
        inputs[f"{source.name}/{name}"] = member.data
    result = run_pinned_command(
        provider.executable,
        ("--infile", source.name, "--outfile", "converted.h5"),
        inputs=inputs,
        outputs=("converted.h5",),
        timeout=provider.timeout,
        max_output_bytes=limits.max_bytes,
    )
    return bounded_resource_from_bytes(
        result.output("converted.h5"), limits=limits, source_path=str(source)
    )


__all__ = [
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
]
