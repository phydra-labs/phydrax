#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Shared bounded HDF5 substrate of every openPMD profile adapter.

This owner holds the per-profile standard revisions, series-root and iteration
metadata, record units (``unitSI`` and ``unitDimension``), grid-axis units, and
the structural preflight that must finish before any dataset payload is read.
Profile adapters own record selection and scientific interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from numbers import Integral
from typing import assert_never, Literal, TYPE_CHECKING, TypeAlias

import h5py
import h5py.h5o as h5o
import numpy as np

from .._external_resource import ResourceLimits, ResourceReadError
from ..typing import parse


if TYPE_CHECKING:
    from _typeshed import ReadableBuffer


OpenPMDProfile: TypeAlias = Literal["meshes", "particles", "laser-envelope"]
OpenPMDExtensionEncoding: TypeAlias = Literal["bitmask", "names"]
OpenPMDGridUnitLayout: TypeAlias = Literal["shared-scalar", "per-axis"]
OpenPMDIterationEncoding: TypeAlias = Literal["groupBased", "fileBased"]

# Fixed by openPMD 1.1.0 and the 2.0 draft alike.
OPENPMD_BASE_PATH = "/data/%T/"
# openPMD 1.x registers exactly one extension identifier in its bitmask.
_BITMASK_EXTENSIONS = {1: "ED-PIC"}
# HDF5 filter identifiers: deflate, shuffle, Fletcher32.
_ADMITTED_FILTERS = (1, 2, 3)


class OpenPMDUnsupportedError(ValueError):
    """Well-formed openPMD metadata outside the pinned profile's semantics."""


@dataclass(frozen=True, slots=True)
class OpenPMDStandardRevision:
    """One pinned openPMD standard revision for one record profile.

    ``version`` is the exact root ``openPMD`` attribute. ``source`` is the
    release tag or the pinned openPMD-standard commit. ``extension`` names the
    extension the profile requires (empty for the base standard).
    """

    profile: OpenPMDProfile
    version: str
    source: str
    extension: str
    extension_encoding: OpenPMDExtensionEncoding
    grid_unit_layout: OpenPMDGridUnitLayout
    format_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "profile", parse(self.profile, OpenPMDProfile, "profile")
        )
        object.__setattr__(
            self,
            "extension_encoding",
            parse(
                self.extension_encoding, OpenPMDExtensionEncoding, "extension_encoding"
            ),
        )
        object.__setattr__(
            self,
            "grid_unit_layout",
            parse(self.grid_unit_layout, OpenPMDGridUnitLayout, "grid_unit_layout"),
        )
        if not self.version or not self.source or not self.format_id:
            raise ValueError("openPMD revisions require version, source, and format_id.")


OPENPMD_MESH_REVISION = OpenPMDStandardRevision(
    "meshes",
    "1.1.0",
    "1.1.0",
    "",
    "bitmask",
    "shared-scalar",
    "openPMD-1.1.0-meshes-HDF5",
)
OPENPMD_PARTICLE_REVISION = OpenPMDStandardRevision(
    "particles",
    "1.1.0",
    "1.1.0",
    "",
    "bitmask",
    "shared-scalar",
    "openPMD-1.1.0-particles-HDF5",
)
OPENPMD_LASER_ENVELOPE_REVISION = OpenPMDStandardRevision(
    "laser-envelope",
    "2.0.0",
    "095799703baf69111e8383a6fda05ce029e65bb2",
    "LaserEnvelope",
    "names",
    "per-axis",
    "openPMD-upcoming-2.0.0-LaserEnvelope-HDF5@0957997",
)


# Powers of the SI base quantities in openPMD order (L, M, T, I, theta, N, J).
type OpenPMDDimension = tuple[float, float, float, float, float, float, float]

LENGTH_DIMENSION: OpenPMDDimension = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
TIME_DIMENSION: OpenPMDDimension = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
ELECTRIC_FIELD_DIMENSION: OpenPMDDimension = (1.0, 1.0, -3.0, -1.0, 0.0, 0.0, 0.0)


def _dimension(values: object, name: str, /) -> OpenPMDDimension:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (7,) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be seven finite SI base-quantity powers.")
    return (
        float(array[0]),
        float(array[1]),
        float(array[2]),
        float(array[3]),
        float(array[4]),
        float(array[5]),
        float(array[6]),
    )


@dataclass(frozen=True, slots=True)
class OpenPMDUnit:
    """One record unit: stored values times ``unit_si`` are SI values."""

    unit_si: float
    unit_dimension: OpenPMDDimension

    def __post_init__(self) -> None:
        unit_si = float(self.unit_si)
        if not np.isfinite(unit_si) or unit_si <= 0.0:
            raise ValueError("unitSI must be finite and positive.")
        object.__setattr__(self, "unit_si", unit_si)
        object.__setattr__(
            self, "unit_dimension", _dimension(self.unit_dimension, "unitDimension")
        )

    def to_si(self, values: np.ndarray, /) -> np.ndarray:
        """Convert stored record values to SI."""
        return values * np.asarray(self.unit_si, dtype=np.float64)

    def from_si(self, values: np.ndarray, /) -> np.ndarray:
        """Convert SI values to this record's stored unit."""
        return values / np.asarray(self.unit_si, dtype=np.float64)


# Attribute readers ---------------------------------------------------------------


def attribute_text(value: object, name: str, /) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must be a scalar string attribute.")
    scalar = array.reshape(()).item()
    if isinstance(scalar, bytes):
        result = scalar.decode("ascii", errors="strict")
    elif isinstance(scalar, str):
        result = scalar
    else:
        raise TypeError(f"{name} must be a string attribute.")
    result = result.rstrip("\x00").strip()
    if not result:
        raise ValueError(f"{name} must be nonempty.")
    return result


def required_text(attributes: h5py.AttributeManager, name: str, /) -> str:
    if name not in attributes:
        raise ValueError(f"Missing required openPMD attribute {name}.")
    return attribute_text(attributes[name], name)


def numeric_attribute(
    attributes: h5py.AttributeManager, name: str, shape: tuple[int, ...], /
) -> np.ndarray:
    if name not in attributes:
        raise ValueError(f"Missing required openPMD attribute {name}.")
    supplied = np.asarray(attributes[name])
    if supplied.shape != shape or not np.issubdtype(supplied.dtype, np.number):
        raise ValueError(f"openPMD attribute {name} must have shape {shape}.")
    result = np.asarray(supplied, dtype=np.float64)
    if np.any(~np.isfinite(result)):
        raise ValueError(f"openPMD attribute {name} must be finite.")
    return result


def scalar_attribute(attributes: h5py.AttributeManager, name: str, /) -> float:
    return float(numeric_attribute(attributes, name, ()).reshape(()))


# Series root and iterations --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OpenPMDSeriesRoot:
    """Validated root metadata of one openPMD HDF5 file."""

    revision: OpenPMDStandardRevision
    iteration_encoding: OpenPMDIterationEncoding
    iteration_format: str
    extensions: tuple[str, ...]
    meshes_path: str | None
    particles_path: str | None


def _declared_extensions(
    attributes: h5py.AttributeManager, revision: OpenPMDStandardRevision, /
) -> tuple[str, ...]:
    match revision.extension_encoding:
        case "names":
            names = required_text(attributes, "openPMDextension").split(";")
            return tuple(sorted({name.strip() for name in names if name.strip()}))
        case "bitmask":
            if "openPMDextension" not in attributes:
                raise ValueError("Missing required openPMD attribute openPMDextension.")
            supplied = np.asarray(attributes["openPMDextension"])
            if supplied.shape != () or not np.issubdtype(supplied.dtype, np.integer):
                raise ValueError("openPMDextension must be an unsigned integer bitmask.")
            mask = int(supplied.reshape(()))
            if mask < 0 or mask >= 2**32:
                raise ValueError("openPMDextension must be an unsigned integer bitmask.")
            unknown = mask & ~sum(_BITMASK_EXTENSIONS)
            if unknown:
                raise OpenPMDUnsupportedError(
                    f"openPMD extension bits {unknown:#x} are unsupported."
                )
            return tuple(
                sorted(name for bit, name in _BITMASK_EXTENSIONS.items() if mask & bit)
            )
        case unreachable:
            assert_never(unreachable)


def _records_path(attributes: h5py.AttributeManager, name: str, /) -> str | None:
    if name not in attributes:
        return None
    path = required_text(attributes, name)
    if path.startswith("/") or not path.endswith("/") or ".." in path.split("/"):
        raise ValueError(f"{name} must be a relative group path ending in '/'.")
    return path


def read_series_root(
    handle: h5py.File, revision: OpenPMDStandardRevision, /
) -> OpenPMDSeriesRoot:
    """Validate root metadata against one pinned standard revision."""
    attributes = handle.attrs
    version = required_text(attributes, "openPMD")
    if version != revision.version:
        raise OpenPMDUnsupportedError(
            f"openPMD {version} is unsupported; the {revision.profile} profile pins "
            f"openPMD {revision.version} ({revision.source})."
        )
    if required_text(attributes, "basePath") != OPENPMD_BASE_PATH:
        raise ValueError(f"openPMD basePath must be {OPENPMD_BASE_PATH!r}.")
    encoding = parse(
        required_text(attributes, "iterationEncoding"),
        OpenPMDIterationEncoding,
        "iterationEncoding",
    )
    iteration_format = required_text(attributes, "iterationFormat")
    match encoding:
        case "groupBased":
            if iteration_format != OPENPMD_BASE_PATH:
                raise ValueError(
                    "iterationFormat must match the canonical group base path."
                )
        case "fileBased":
            if "%T" not in iteration_format or "/" in iteration_format:
                raise ValueError(
                    "fileBased iterationFormat must be one file name containing %T."
                )
        case unreachable:
            assert_never(unreachable)
    extensions = _declared_extensions(attributes, revision)
    if revision.extension and revision.extension not in extensions:
        raise ValueError(f"The {revision.extension} extension is not declared.")
    return OpenPMDSeriesRoot(
        revision,
        encoding,
        iteration_format,
        extensions,
        _records_path(attributes, "meshesPath"),
        _records_path(attributes, "particlesPath"),
    )


@dataclass(frozen=True, slots=True)
class OpenPMDIterationTime:
    """Iteration ``time``/``dt`` in stored units and their SI factor."""

    time: float
    dt: float
    time_unit_si: float

    def __post_init__(self) -> None:
        values = (float(self.time), float(self.dt), float(self.time_unit_si))
        if not all(np.isfinite(value) for value in values) or values[2] <= 0.0:
            raise ValueError("Iteration time and dt must be finite; timeUnitSI positive.")
        object.__setattr__(self, "time", values[0])
        object.__setattr__(self, "dt", values[1])
        object.__setattr__(self, "time_unit_si", values[2])

    @property
    def time_seconds(self) -> float:
        return self.time * self.time_unit_si

    @property
    def dt_seconds(self) -> float:
        return self.dt * self.time_unit_si


def _iteration_index(iteration: int, /) -> int:
    if isinstance(iteration, bool) or not isinstance(iteration, Integral):
        raise TypeError("iteration must be an integer.")
    index = int(iteration)
    if index < 0 or index >= 2**64:
        raise ValueError("iteration must be an unsigned 64-bit integer.")
    return index


def read_iteration(
    handle: h5py.File, iteration: int, /
) -> tuple[h5py.Group, OpenPMDIterationTime]:
    """Return one iteration group beneath the fixed basePath and its time."""
    path = f"data/{_iteration_index(iteration)}"
    if path not in handle or not isinstance(handle[path], h5py.Group):
        raise ValueError("The selected openPMD iteration is absent.")
    group = handle[path]
    return group, OpenPMDIterationTime(
        scalar_attribute(group.attrs, "time"),
        scalar_attribute(group.attrs, "dt"),
        scalar_attribute(group.attrs, "timeUnitSI"),
    )


def write_series_root(
    handle: h5py.File,
    revision: OpenPMDStandardRevision,
    /,
    *,
    meshes_path: str | None,
    particles_path: str | None,
) -> None:
    """Write group-based root metadata of one pinned revision."""
    attributes = handle.attrs
    attributes["openPMD"] = np.bytes_(revision.version)
    attributes["basePath"] = np.bytes_(OPENPMD_BASE_PATH)
    if meshes_path is not None:
        attributes["meshesPath"] = np.bytes_(meshes_path)
    if particles_path is not None:
        attributes["particlesPath"] = np.bytes_(particles_path)
    attributes["iterationEncoding"] = np.bytes_("groupBased")
    attributes["iterationFormat"] = np.bytes_(OPENPMD_BASE_PATH)
    match revision.extension_encoding:
        case "names":
            attributes["openPMDextension"] = np.bytes_(revision.extension)
        case "bitmask":
            # Base-standard profiles declare no extension.
            attributes["openPMDextension"] = np.uint32(0)
        case unreachable:
            assert_never(unreachable)
    attributes["software"] = np.bytes_("phydrax")


def write_iteration(
    handle: h5py.File, iteration: int, time: OpenPMDIterationTime, /
) -> h5py.Group:
    group = handle.require_group(f"data/{_iteration_index(iteration)}")
    group.attrs["time"] = np.float64(time.time)
    group.attrs["dt"] = np.float64(time.dt)
    group.attrs["timeUnitSI"] = np.float64(time.time_unit_si)
    return group


# Units ------------------------------------------------------------------------------


def read_record_unit(
    record_attributes: h5py.AttributeManager,
    component_attributes: h5py.AttributeManager,
    expected_dimension: OpenPMDDimension,
    tolerance: float,
    /,
) -> OpenPMDUnit:
    """Read ``unitDimension`` (record) and ``unitSI`` (component) of one record.

    A scalar record is its own component, so both mappings may be the same.
    """
    dimension = numeric_attribute(record_attributes, "unitDimension", (7,))
    expected = np.asarray(expected_dimension, dtype=np.float64)
    if not np.allclose(dimension, expected, rtol=0.0, atol=tolerance):
        raise ValueError(
            f"openPMD unitDimension {tuple(dimension.tolist())} does not match the "
            f"expected {tuple(expected.tolist())}."
        )
    return OpenPMDUnit(
        scalar_attribute(component_attributes, "unitSI"), expected_dimension
    )


def write_record_unit(
    record_attributes: h5py.AttributeManager,
    component_attributes: h5py.AttributeManager,
    unit: OpenPMDUnit,
    /,
) -> None:
    record_attributes["unitDimension"] = np.asarray(unit.unit_dimension, dtype=np.float64)
    component_attributes["unitSI"] = np.float64(unit.unit_si)


def read_grid_unit_si(
    attributes: h5py.AttributeManager,
    revision: OpenPMDStandardRevision,
    axis_dimensions: tuple[OpenPMDDimension, ...],
    tolerance: float,
    /,
) -> np.ndarray:
    """Return per-axis SI factors of ``gridSpacing`` and ``gridGlobalOffset``."""
    count = len(axis_dimensions)
    expected = np.asarray(axis_dimensions, dtype=np.float64).reshape((count, 7))
    match revision.grid_unit_layout:
        case "shared-scalar":
            if not np.array_equal(expected, np.tile(LENGTH_DIMENSION, (count, 1))):
                raise OpenPMDUnsupportedError(
                    f"openPMD {revision.version} grids carry only spatial axes; "
                    "non-length grid axes are unsupported."
                )
            unit_si = np.full(
                count, scalar_attribute(attributes, "gridUnitSI"), dtype=np.float64
            )
        case "per-axis":
            unit_si = numeric_attribute(attributes, "gridUnitSI", (count,))
            supplied = numeric_attribute(
                attributes, "gridUnitDimension", (7 * count,)
            ).reshape((count, 7))
            if not np.allclose(supplied, expected, rtol=0.0, atol=tolerance):
                raise OpenPMDUnsupportedError(
                    "openPMD grid-axis unit dimensions are unsupported by this profile."
                )
        case unreachable:
            assert_never(unreachable)
    if np.any(unit_si <= 0.0):
        raise ValueError("openPMD gridUnitSI factors must be positive.")
    return unit_si


def write_grid_units(
    attributes: h5py.AttributeManager,
    revision: OpenPMDStandardRevision,
    axis_dimensions: tuple[OpenPMDDimension, ...],
    /,
) -> None:
    """Write SI grid units (factor one) for axes stored in SI."""
    count = len(axis_dimensions)
    match revision.grid_unit_layout:
        case "shared-scalar":
            if any(dimension != LENGTH_DIMENSION for dimension in axis_dimensions):
                raise OpenPMDUnsupportedError(
                    f"openPMD {revision.version} grids carry only spatial axes."
                )
            attributes["gridUnitSI"] = np.float64(1.0)
        case "per-axis":
            attributes["gridUnitSI"] = np.ones(count, dtype=np.float64)
            attributes["gridUnitDimension"] = np.asarray(
                axis_dimensions, dtype=np.float64
            ).reshape(7 * count)
        case unreachable:
            assert_never(unreachable)


# Bounded structural preflight ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OpenPMDHDF5Inventory:
    """Complete bounded structure of one HDF5 image before any payload read."""

    datasets: dict[str, h5py.Dataset]
    maximum_depth: int
    object_count: int
    attribute_count: int
    decoded_bytes: int
    decoded_elements: int

    def require_budget(
        self,
        limits: ResourceLimits,
        maximum_decoded_bytes: int,
        canonical_bytes: int,
        /,
    ) -> None:
        """Refuse unless every stored payload plus canonical copies fits the budget."""
        if self.decoded_bytes + canonical_bytes > maximum_decoded_bytes:
            raise ResourceReadError(
                "limit", "openPMD decoded datasets exceed maximum_decoded_bytes."
            )
        if self.object_count + self.decoded_elements > limits.max_nodes:
            raise ResourceReadError(
                "limit", "openPMD decoded element count exceeds its limit."
            )


def _admit_dataset(item: h5py.Dataset, /) -> None:
    if item.shape is None or item.is_virtual or item.external:
        raise ValueError("openPMD datasets must be resident fixed dataspaces.")
    dtype = item.dtype
    if dtype.hasobject or h5py.check_dtype(ref=dtype) is not None:
        raise TypeError("openPMD datasets cannot use object or reference data.")
    creation = item.id.get_create_plist()
    for index in range(creation.get_nfilters()):
        if creation.get_filter(index)[0] not in _ADMITTED_FILTERS:
            raise OpenPMDUnsupportedError(
                "Only HDF5 deflate, shuffle, and Fletcher32 filters are supported."
            )


def preflight_hdf5(handle: h5py.File, limits: ResourceLimits, /) -> OpenPMDHDF5Inventory:
    """Traverse the complete HDF5 tree within resource bounds, reading no payload."""
    stack: list[tuple[str, h5py.Group | h5py.Dataset, int]] = [("", handle, 0)]
    addresses: set[int] = set()
    datasets: dict[str, h5py.Dataset] = {}
    object_count = 0
    attribute_count = 0
    maximum_depth = 0
    decoded_bytes = 0
    decoded_elements = 0
    while stack:
        name, item, depth = stack.pop()
        object_count += 1
        maximum_depth = max(maximum_depth, depth)
        if maximum_depth > limits.max_depth or object_count > limits.max_nodes:
            raise ResourceReadError("limit", "openPMD HDF5 structure exceeds its bounds.")
        address = h5o.get_info(item.id).addr
        if address in addresses:
            raise OpenPMDUnsupportedError(
                "HDF5 hard-link aliases and cycles are unsupported."
            )
        addresses.add(address)
        attribute_count += len(item.attrs)
        if attribute_count > limits.max_attributes:
            raise ResourceReadError(
                "limit", "openPMD HDF5 attributes exceed their limit."
            )
        if isinstance(item, h5py.Group):
            for child_name in item:
                link = item.get(child_name, getlink=True)
                if not isinstance(link, h5py.HardLink):
                    raise ValueError("HDF5 soft and external links are forbidden.")
                stack.append(
                    (f"{name}/{child_name}".lstrip("/"), item[child_name], depth + 1)
                )
        elif isinstance(item, h5py.Dataset):
            _admit_dataset(item)
            decoded_bytes += item.size * item.dtype.itemsize
            decoded_elements += item.size
            datasets[name] = item
        else:
            raise OpenPMDUnsupportedError("HDF5 named datatypes are unsupported.")
    return OpenPMDHDF5Inventory(
        datasets,
        maximum_depth,
        object_count,
        attribute_count,
        decoded_bytes,
        decoded_elements,
    )


class BoundedHDF5Buffer(BytesIO):
    """In-memory HDF5 image that refuses growth beyond its byte limit."""

    def __init__(self, maximum_bytes: int) -> None:
        super().__init__()
        self.maximum_bytes = int(maximum_bytes)

    def write(self, data: ReadableBuffer, /) -> int:
        projected = max(self.tell() + memoryview(data).nbytes, len(self.getbuffer()))
        if projected > self.maximum_bytes:
            raise ResourceReadError("limit", "Encoded HDF5 output exceeds max_bytes.")
        return super().write(data)

    def truncate(self, size: int | None = None, /) -> int:
        resolved = self.tell() if size is None else int(size)
        if resolved > self.maximum_bytes:
            raise ResourceReadError("limit", "Encoded HDF5 output exceeds max_bytes.")
        return super().truncate(resolved)


__all__ = [
    "BoundedHDF5Buffer",
    "ELECTRIC_FIELD_DIMENSION",
    "LENGTH_DIMENSION",
    "OPENPMD_BASE_PATH",
    "OPENPMD_LASER_ENVELOPE_REVISION",
    "OPENPMD_MESH_REVISION",
    "OPENPMD_PARTICLE_REVISION",
    "OpenPMDDimension",
    "OpenPMDExtensionEncoding",
    "OpenPMDGridUnitLayout",
    "OpenPMDHDF5Inventory",
    "OpenPMDIterationEncoding",
    "OpenPMDIterationTime",
    "OpenPMDProfile",
    "OpenPMDSeriesRoot",
    "OpenPMDStandardRevision",
    "OpenPMDUnit",
    "OpenPMDUnsupportedError",
    "TIME_DIMENSION",
    "attribute_text",
    "numeric_attribute",
    "preflight_hdf5",
    "read_grid_unit_si",
    "read_iteration",
    "read_record_unit",
    "read_series_root",
    "required_text",
    "scalar_attribute",
    "write_grid_units",
    "write_iteration",
    "write_record_unit",
    "write_series_root",
]
