#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import base64
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._contracts import SurfaceMetadata
from ._model import SurfaceModel


class SurfaceFileFormat(str, Enum):
    """Surface interchange formats with concrete providers."""

    GMSH = "gmsh"
    VTK = "vtk"
    STL = "stl"
    OBJ = "obj"
    STEP = "step"
    IGES = "iges"


class SurfaceFieldAssociation(str, Enum):
    """Portable association of numeric data with authoritative entities."""

    POINT = "point"
    CELL = "cell"


class SurfaceOrientationPolicy(str, Enum):
    """Import handling for inconsistent triangle orientation."""

    REJECT = "reject"
    REPAIR_WITH_EVIDENCE = "repair_with_evidence"


class SurfaceInteropError(ValueError):
    """Base class for fail-closed surface interchange failures."""


class SurfaceProviderUnavailableError(ImportError):
    """Raised before file IO when a requested real provider is unavailable."""


class SurfaceResourceLimitError(SurfaceInteropError):
    """Raised when host-side preflight exceeds an explicit capacity."""


class SurfaceDataCorruptionError(SurfaceInteropError):
    """Raised when portable surface data is malformed or contradictory."""


class SurfaceLossyOperationError(SurfaceInteropError):
    """Raised when explicit permission for a lossy operation is absent."""


class SurfaceUnsupportedFormatError(SurfaceInteropError):
    """Raised when no existing real provider implements an operation."""


_UNIT_TO_METERS = {
    "m": 1.0,
    "mm": 1.0e-3,
    "cm": 1.0e-2,
    "um": 1.0e-6,
    "km": 1.0e3,
    "in": 0.0254,
    "ft": 0.3048,
}
_EXTENSION_FORMAT = {
    ".msh": SurfaceFileFormat.GMSH,
    ".vtk": SurfaceFileFormat.VTK,
    ".vtu": SurfaceFileFormat.VTK,
    ".stl": SurfaceFileFormat.STL,
    ".obj": SurfaceFileFormat.OBJ,
    ".step": SurfaceFileFormat.STEP,
    ".stp": SurfaceFileFormat.STEP,
    ".iges": SurfaceFileFormat.IGES,
    ".igs": SurfaceFileFormat.IGES,
}
_VERTEX_ID = "phydrax_vertex_global_id"
_CELL_ID = "phydrax_cell_global_id"
_TAG_PREFIX = "phydrax_cell_tag__"
_META_SOURCE = "phydrax_meta_source__"
_META_REVISION = "phydrax_meta_revision__"
_META_COORDINATES = "phydrax_meta_coordinates__"
_META_PROVENANCE = "phydrax_meta_provenance_"
_UNIT_PREFIX = "phydrax_length_unit__"
_RESERVED_EXACT = frozenset((_VERTEX_ID, _CELL_ID))
_RESERVED_PREFIXES = (
    _TAG_PREFIX,
    _META_SOURCE,
    _META_REVISION,
    _META_COORDINATES,
    _META_PROVENANCE,
    _UNIT_PREFIX,
)


def _canonical_unit(value: str, /) -> str:
    unit = str(value).strip().lower()
    aliases = {
        "meter": "m",
        "meters": "m",
        "metre": "m",
        "metres": "m",
        "millimeter": "mm",
        "millimeters": "mm",
        "millimetre": "mm",
        "millimetres": "mm",
        "centimeter": "cm",
        "centimeters": "cm",
        "centimetre": "cm",
        "centimetres": "cm",
        "micrometer": "um",
        "micrometers": "um",
        "micrometre": "um",
        "micrometres": "um",
        "kilometer": "km",
        "kilometers": "km",
        "kilometre": "km",
        "kilometres": "km",
        "inch": "in",
        "inches": "in",
        "foot": "ft",
        "feet": "ft",
    }
    unit = aliases.get(unit, unit)
    if unit not in _UNIT_TO_METERS:
        supported = ", ".join(_UNIT_TO_METERS)
        raise ValueError(f"Unsupported length unit {value!r}; choose one of {supported}.")
    return unit


def _positive_capacity(name: str, value: int, /) -> int:
    capacity = int(value)
    if capacity <= 0:
        raise ValueError(f"{name} must be positive.")
    return capacity


@dataclass(frozen=True, slots=True)
class SurfaceImportPolicy:
    """Immutable, explicitly unitful and resource-bounded import policy."""

    source_length_unit: str
    orientation: SurfaceOrientationPolicy = SurfaceOrientationPolicy.REJECT
    allow_lossy: bool = False
    maximum_file_bytes: int = 256 * 1024 * 1024
    maximum_data_bytes: int = 512 * 1024 * 1024
    maximum_vertices: int = 10_000_000
    maximum_cells: int = 20_000_000
    maximum_fields: int = 256
    cad_linear_deflection_in_source_units: float = 1.0e-3
    cad_angular_deflection: float = 0.1
    cad_trim_samples_per_edge: int = 33

    def __post_init__(self):
        object.__setattr__(
            self, "source_length_unit", _canonical_unit(self.source_length_unit)
        )
        if not isinstance(self.orientation, SurfaceOrientationPolicy):
            raise TypeError("orientation must be SurfaceOrientationPolicy.")
        for name in (
            "maximum_file_bytes",
            "maximum_data_bytes",
            "maximum_vertices",
            "maximum_cells",
            "maximum_fields",
            "cad_trim_samples_per_edge",
        ):
            object.__setattr__(self, name, _positive_capacity(name, getattr(self, name)))
        linear = float(self.cad_linear_deflection_in_source_units)
        angular = float(self.cad_angular_deflection)
        if not np.isfinite(linear) or linear <= 0.0:
            raise ValueError("cad_linear_deflection_in_source_units must be positive.")
        if not np.isfinite(angular) or angular <= 0.0:
            raise ValueError("cad_angular_deflection must be positive.")
        object.__setattr__(self, "cad_linear_deflection_in_source_units", linear)
        object.__setattr__(self, "cad_angular_deflection", angular)


@dataclass(frozen=True, slots=True)
class SurfaceExportPolicy:
    """Immutable export units, loss permission, and host capacities."""

    target_length_unit: str
    allow_lossy: bool = False
    binary: bool = False
    maximum_data_bytes: int = 512 * 1024 * 1024
    maximum_vertices: int = 10_000_000
    maximum_cells: int = 20_000_000
    maximum_fields: int = 256

    def __post_init__(self):
        object.__setattr__(
            self, "target_length_unit", _canonical_unit(self.target_length_unit)
        )
        for name in (
            "maximum_data_bytes",
            "maximum_vertices",
            "maximum_cells",
            "maximum_fields",
        ):
            object.__setattr__(self, name, _positive_capacity(name, getattr(self, name)))


@dataclass(frozen=True, slots=True)
class PortableSurfaceField:
    """Immutable numeric field with an explicit portable entity association."""

    name: str
    association: SurfaceFieldAssociation
    values: np.ndarray

    def __init__(
        self,
        name: str,
        association: SurfaceFieldAssociation,
        values: ArrayLike,
    ):
        name_ = str(name)
        if not name_:
            raise ValueError("Portable surface field names must be non-empty.")
        if name_ in _RESERVED_EXACT or name_.startswith(_RESERVED_PREFIXES):
            raise ValueError(f"Portable surface field name {name_!r} is reserved.")
        if not isinstance(association, SurfaceFieldAssociation):
            raise TypeError("association must be SurfaceFieldAssociation.")
        array = np.array(values, copy=True)
        if array.ndim == 0 or array.shape[0] == 0:
            raise ValueError("Portable field values require one non-empty entity axis.")
        if array.dtype.kind not in "biuf":
            raise TypeError(
                "Portable fields must contain real numeric or boolean values."
            )
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise ValueError("Portable floating-point fields must be finite.")
        array.setflags(write=False)
        object.__setattr__(self, "name", name_)
        object.__setattr__(self, "association", association)
        object.__setattr__(self, "values", array)


@dataclass(frozen=True, slots=True)
class SurfaceFieldRecord:
    """Portable field association and shape evidence retained in a report."""

    name: str
    association: SurfaceFieldAssociation
    component_shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True, slots=True)
class SurfaceInteropReport:
    """Immutable units, fidelity, identity, and capacity evidence for one operation."""

    operation: str
    file_format: SurfaceFileFormat
    provider: str
    source_length_unit: str
    target_length_unit: str
    coordinate_scale: float
    source_id: str
    source_revision: str
    artifact_digest: str
    vertex_count: int
    cell_count: int
    source_ids_preserved: bool
    tags_preserved: bool
    source_metadata_preserved: bool
    orientation_changed: bool
    lossy: bool
    losses: tuple[str, ...]
    fields: tuple[SurfaceFieldRecord, ...]


@dataclass(frozen=True, slots=True)
class SurfaceImportResult:
    """Authoritative SI SurfaceModel plus retained portable interchange evidence."""

    model: SurfaceModel
    fields: tuple[PortableSurfaceField, ...]
    report: SurfaceInteropReport
    cad_model: Any | None = None


@dataclass(frozen=True, slots=True)
class SurfaceExportResult:
    """Completed surface artifact and immutable fidelity report."""

    path: Path
    report: SurfaceInteropReport


def _encode_text(value: str, /) -> str:
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")


def _decode_text(value: str, /) -> str:
    padding = "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(value + padding).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as error:
        raise SurfaceDataCorruptionError(
            "Invalid encoded PHYDRAX metadata field."
        ) from error


def _resolve_format(
    path: Path, requested: SurfaceFileFormat | None, /
) -> SurfaceFileFormat:
    if requested is not None:
        if not isinstance(requested, SurfaceFileFormat):
            raise TypeError("file_format must be SurfaceFileFormat or None.")
        return requested
    suffix = path.suffix.lower()
    if suffix not in _EXTENSION_FORMAT:
        raise SurfaceUnsupportedFormatError(
            f"Cannot infer a supported surface format from extension {suffix!r}."
        )
    return _EXTENSION_FORMAT[suffix]


def _require_module(module: str, purpose: str, /):
    if find_spec(module) is None:
        raise SurfaceProviderUnavailableError(
            f"{purpose} requires installed provider module {module!r}."
        )
    return import_module(module)


def _preflight_file(path: Path, maximum_file_bytes: int, /) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size > maximum_file_bytes:
        raise SurfaceResourceLimitError(
            f"Surface file has {size} bytes, exceeding limit {maximum_file_bytes}."
        )
    return size


def _artifact_digest(path: Path, /) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _meshio_file_format(file_format: SurfaceFileFormat, path: Path, /) -> str:
    if file_format is SurfaceFileFormat.GMSH:
        return "gmsh"
    if file_format is SurfaceFileFormat.VTK:
        return "vtu" if path.suffix.lower() == ".vtu" else "vtk"
    return file_format.value


def _array_payload_bytes(mesh: Any, /) -> int:
    total = np.asarray(mesh.points).nbytes
    total += sum(np.asarray(block.data).nbytes for block in mesh.cells)
    total += sum(np.asarray(value).nbytes for value in mesh.point_data.values())
    total += sum(
        np.asarray(value).nbytes for values in mesh.cell_data.values() for value in values
    )
    return int(total)


def _triangle_blocks(mesh: Any, /) -> tuple[tuple[int, ...], np.ndarray]:
    unsupported = tuple(
        block.type
        for block in mesh.cells
        if int(block.dim) == 2 and block.type != "triangle"
    )
    if unsupported:
        names = ", ".join(unsupported)
        raise SurfaceDataCorruptionError(
            "Surface import refuses implicit linearization or triangulation of "
            f"two-dimensional cell types: {names}."
        )
    indices = tuple(
        index for index, block in enumerate(mesh.cells) if block.type == "triangle"
    )
    if not indices:
        raise SurfaceDataCorruptionError(
            "Surface artifact contains no linear triangle cells."
        )
    faces = np.concatenate(tuple(np.asarray(mesh.cells[index].data) for index in indices))
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise SurfaceDataCorruptionError(
            "Triangle connectivity must have shape (cells, 3)."
        )
    return indices, faces


def _integer_ids(value: ArrayLike, count: int, name: str, /) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (count,) or raw.dtype.kind not in "iuf":
        raise SurfaceDataCorruptionError(f"{name} must be one numeric scalar per entity.")
    if raw.dtype.kind == "f" and (
        not np.all(np.isfinite(raw)) or not np.array_equal(raw, np.floor(raw))
    ):
        raise SurfaceDataCorruptionError(f"{name} contains non-integral values.")
    ids = raw.astype(np.int64)
    if np.any(ids < 0) or np.unique(ids).size != ids.size:
        raise SurfaceDataCorruptionError(f"{name} must contain unique non-negative IDs.")
    return ids


def _cell_data(
    mesh: Any, name: str, block_indices: Sequence[int], /
) -> np.ndarray | None:
    if name not in mesh.cell_data:
        return None
    values = mesh.cell_data[name]
    if len(values) != len(mesh.cells):
        raise SurfaceDataCorruptionError(f"Cell field {name!r} is not block-aligned.")
    return np.concatenate(tuple(np.asarray(values[index]) for index in block_indices))


def _marker_values_valid(value: ArrayLike, count: int, /) -> bool:
    array = np.asarray(value)
    return array.shape == (count,) and np.all(array == 0)


def _decode_single_marker(
    point_data: dict[str, Any], prefix: str, point_count: int, /
) -> str | None:
    matches = tuple(name for name in point_data if name.startswith(prefix))
    if not matches:
        return None
    if len(matches) != 1 or not _marker_values_valid(point_data[matches[0]], point_count):
        raise SurfaceDataCorruptionError(
            f"Contradictory metadata markers for {prefix!r}."
        )
    return _decode_text(matches[0][len(prefix) :])


def _decode_provenance(
    point_data: dict[str, Any], point_count: int, /
) -> tuple[str, ...]:
    matches = sorted(name for name in point_data if name.startswith(_META_PROVENANCE))
    result = []
    for name in matches:
        if not _marker_values_valid(point_data[name], point_count):
            raise SurfaceDataCorruptionError("Malformed provenance marker values.")
        suffix = name[len(_META_PROVENANCE) :]
        separator = suffix.find("__")
        if separator <= 0 or not suffix[:separator].isdigit():
            raise SurfaceDataCorruptionError("Malformed provenance marker name.")
        result.append(_decode_text(suffix[separator + 2 :]))
    return tuple(result)


def _decode_tags(
    mesh: Any, block_indices: Sequence[int], cell_count: int, /
) -> tuple[tuple[str, ...], bool]:
    markers = tuple(name for name in mesh.cell_data if name.startswith(_TAG_PREFIX))
    if markers:
        active = np.zeros((len(markers), cell_count), dtype=bool)
        decoded = []
        for row, name in enumerate(markers):
            values = _cell_data(mesh, name, block_indices)
            if values is None or values.shape != (cell_count,):
                raise SurfaceDataCorruptionError("Cell tag marker is not cell-aligned.")
            if not np.all((values == 0) | (values == 1)):
                raise SurfaceDataCorruptionError("Cell tag markers must be zero or one.")
            active[row] = values.astype(bool)
            decoded.append(_decode_text(name[len(_TAG_PREFIX) :]))
        if not np.all(np.sum(active, axis=0) == 1):
            raise SurfaceDataCorruptionError(
                "Each cell requires exactly one encoded tag."
            )
        tags = tuple(decoded[int(index)] for index in np.argmax(active, axis=0))
        return tags, True
    physical = _cell_data(mesh, "gmsh:physical", block_indices)
    if physical is not None:
        reverse = {}
        for name, value in mesh.field_data.items():
            entry = np.asarray(value).reshape((-1,))
            if entry.size >= 2 and int(entry[1]) == 2:
                reverse[int(entry[0])] = str(name)
        tags = tuple(
            reverse.get(int(value), f"physical-{int(value)}") for value in physical
        )
        return tags, True
    return tuple("surface" for _ in range(cell_count)), False


def _field_records(
    fields: Sequence[PortableSurfaceField], /
) -> tuple[SurfaceFieldRecord, ...]:
    return tuple(
        SurfaceFieldRecord(
            field.name,
            field.association,
            tuple(int(value) for value in field.values.shape[1:]),
            field.values.dtype.str,
        )
        for field in fields
    )


def _portable_fields(
    mesh: Any,
    block_indices: Sequence[int],
    point_count: int,
    cell_count: int,
    maximum_fields: int,
    /,
) -> tuple[PortableSurfaceField, ...]:
    fields = []
    for name, value in mesh.point_data.items():
        if name in _RESERVED_EXACT or name.startswith(_RESERVED_PREFIXES):
            continue
        array = np.asarray(value)
        if array.shape[0] != point_count:
            raise SurfaceDataCorruptionError(
                f"Point field {name!r} is not point-aligned."
            )
        fields.append(PortableSurfaceField(name, SurfaceFieldAssociation.POINT, array))
    for name in mesh.cell_data:
        if name in _RESERVED_EXACT or name.startswith(_RESERVED_PREFIXES):
            continue
        array = _cell_data(mesh, name, block_indices)
        if array is None or array.shape[0] != cell_count:
            raise SurfaceDataCorruptionError(f"Cell field {name!r} is not cell-aligned.")
        fields.append(PortableSurfaceField(name, SurfaceFieldAssociation.CELL, array))
    if len(fields) > maximum_fields:
        raise SurfaceResourceLimitError(
            f"Surface artifact contains {len(fields)} fields, exceeding limit {maximum_fields}."
        )
    if len({(field.association, field.name) for field in fields}) != len(fields):
        raise SurfaceDataCorruptionError(
            "Portable field association/name pairs must be unique."
        )
    return tuple(fields)


def _loss_permission(losses: Sequence[str], allowed: bool, /) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(loss) for loss in losses))
    if normalized and not allowed:
        details = ", ".join(normalized)
        raise SurfaceLossyOperationError(
            f"Surface operation would be lossy ({details}); set allow_lossy=True explicitly."
        )
    return normalized


def _import_meshio_surface(
    source: Path,
    file_format: SurfaceFileFormat,
    policy: SurfaceImportPolicy,
    artifact_digest: str,
    /,
) -> SurfaceImportResult:
    meshio = _require_module("meshio", f"{file_format.value} surface import")
    try:
        mesh = meshio.read(source, file_format=_meshio_file_format(file_format, source))
    except meshio.ReadError as error:
        raise SurfaceDataCorruptionError(
            f"Provider could not parse {file_format.value} surface artifact {source}."
        ) from error
    points = np.asarray(mesh.points)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise SurfaceDataCorruptionError("Surface points must have shape (vertices, 3).")
    block_indices, faces = _triangle_blocks(mesh)
    point_count = int(points.shape[0])
    cell_count = int(faces.shape[0])
    if point_count > policy.maximum_vertices or cell_count > policy.maximum_cells:
        raise SurfaceResourceLimitError(
            "Surface entity counts exceed the import policy capacities."
        )
    payload_bytes = _array_payload_bytes(mesh)
    if payload_bytes > policy.maximum_data_bytes:
        raise SurfaceResourceLimitError(
            f"Decoded surface payload has {payload_bytes} bytes, exceeding limit "
            f"{policy.maximum_data_bytes}."
        )

    embedded_unit = _decode_single_marker(mesh.point_data, _UNIT_PREFIX, point_count)
    if (
        embedded_unit is not None
        and _canonical_unit(embedded_unit) != policy.source_length_unit
    ):
        raise SurfaceDataCorruptionError(
            "Explicit source_length_unit contradicts the artifact unit marker."
        )
    source_marker = _decode_single_marker(mesh.point_data, _META_SOURCE, point_count)
    revision_marker = _decode_single_marker(mesh.point_data, _META_REVISION, point_count)
    coordinates_marker = _decode_single_marker(
        mesh.point_data, _META_COORDINATES, point_count
    )
    provenance_marker = _decode_provenance(mesh.point_data, point_count)
    metadata_preserved = (
        source_marker is not None
        and revision_marker is not None
        and coordinates_marker is not None
        and bool(provenance_marker)
    )

    losses = []
    vertex_data = mesh.point_data.get(_VERTEX_ID)
    if vertex_data is None:
        vertex_ids = np.arange(point_count, dtype=np.int64)
        losses.append("generated_vertex_global_ids")
    else:
        vertex_ids = _integer_ids(vertex_data, point_count, _VERTEX_ID)
    cell_data = _cell_data(mesh, _CELL_ID, block_indices)
    if cell_data is None:
        cell_ids = np.arange(cell_count, dtype=np.int64)
        losses.append("generated_cell_global_ids")
    else:
        cell_ids = _integer_ids(cell_data, cell_count, _CELL_ID)
    tags, tags_preserved = _decode_tags(mesh, block_indices, cell_count)
    if not tags_preserved:
        losses.append("generated_cell_tags")
    if not metadata_preserved:
        losses.append("generated_source_metadata")
    if file_format in (SurfaceFileFormat.STL, SurfaceFileFormat.OBJ):
        losses.append(f"{file_format.value}_cannot_retain_ids_tags_or_fields")
    losses_ = _loss_permission(losses, policy.allow_lossy)

    fields = _portable_fields(
        mesh,
        block_indices,
        point_count,
        cell_count,
        policy.maximum_fields,
    )
    source_id = str(source.resolve()) if source_marker is None else source_marker
    source_revision = artifact_digest if revision_marker is None else revision_marker
    coordinate_system = "cartesian" if coordinates_marker is None else coordinates_marker
    scale = _UNIT_TO_METERS[policy.source_length_unit]
    provenance = (
        provenance_marker
        if provenance_marker
        else (f"source-artifact-sha256:{artifact_digest}",)
    )
    metadata = SurfaceMetadata(
        source_id=source_id,
        source_revision=source_revision,
        length_unit="m",
        coordinate_system=coordinate_system,
        provenance=(
            *provenance,
            f"surface-import:{file_format.value}",
            f"coordinate-scale-to-si:{scale:.17g}",
        ),
        cell_tags=tags,
    )
    model = SurfaceModel.from_triangles(
        np.asarray(points, dtype=float) * scale,
        faces,
        metadata,
        vertex_global_ids=vertex_ids,
        cell_global_ids=cell_ids,
        numeric_version=f"import-{artifact_digest[:16]}",
        repair_orientation=(
            policy.orientation is SurfaceOrientationPolicy.REPAIR_WITH_EVIDENCE
        ),
    )
    orientation_changed = model.orientation_repair is not None and bool(
        np.any(np.asarray(model.orientation_repair.flipped))
    )
    report = SurfaceInteropReport(
        operation="import",
        file_format=file_format,
        provider="meshio",
        source_length_unit=policy.source_length_unit,
        target_length_unit="m",
        coordinate_scale=scale,
        source_id=source_id,
        source_revision=source_revision,
        artifact_digest=artifact_digest,
        vertex_count=point_count,
        cell_count=cell_count,
        source_ids_preserved=vertex_data is not None and cell_data is not None,
        tags_preserved=tags_preserved,
        source_metadata_preserved=metadata_preserved,
        orientation_changed=orientation_changed,
        lossy=bool(losses_),
        losses=losses_,
        fields=_field_records(fields),
    )
    return SurfaceImportResult(model, fields, report)


def _import_cad_surface(
    source: Path,
    file_format: SurfaceFileFormat,
    policy: SurfaceImportPolicy,
    artifact_digest: str,
    /,
) -> SurfaceImportResult:
    _require_module("OCP", f"{file_format.value} direct BRep import")
    from ..brep._occt import import_brep

    cad_model = import_brep(
        source,
        linear_deflection=policy.cad_linear_deflection_in_source_units,
        angular_deflection=policy.cad_angular_deflection,
        trim_samples_per_edge=policy.cad_trim_samples_per_edge,
    )
    points = np.asarray(cad_model.mesh_vertices)
    faces = np.asarray(cad_model.mesh_faces)
    if points.shape[0] > policy.maximum_vertices or faces.shape[0] > policy.maximum_cells:
        raise SurfaceResourceLimitError(
            "CAD tessellation exceeds surface entity capacities."
        )
    payload_bytes = points.nbytes + faces.nbytes
    if payload_bytes > policy.maximum_data_bytes:
        raise SurfaceResourceLimitError("CAD tessellation exceeds decoded data capacity.")
    losses = _loss_permission(
        (
            "cad_exact_geometry_tessellated_for_authoritative_cell_mesh",
            "generated_tessellation_global_ids",
        ),
        policy.allow_lossy,
    )
    face_ids = np.asarray(cad_model.triangle_face_ids, dtype=np.int32)
    tags = tuple(cad_model.physical_tags[int(index)] for index in face_ids)
    scale = _UNIT_TO_METERS[policy.source_length_unit]
    metadata = SurfaceMetadata(
        source_id=cad_model.source_id,
        source_revision=cad_model.source_revision,
        length_unit="m",
        provenance=(
            f"direct-brep-import:{file_format.value}",
            f"source-artifact-sha256:{artifact_digest}",
            f"coordinate-scale-to-si:{scale:.17g}",
            "authoritative-cellmesh-from-reported-occt-tessellation",
        ),
        cell_tags=tags,
    )
    model = SurfaceModel.from_triangles(
        points * scale,
        faces,
        metadata,
        numeric_version=f"cad-{artifact_digest[:16]}",
        repair_orientation=(
            policy.orientation is SurfaceOrientationPolicy.REPAIR_WITH_EVIDENCE
        ),
    )
    orientation_changed = model.orientation_repair is not None and bool(
        np.any(np.asarray(model.orientation_repair.flipped))
    )
    report = SurfaceInteropReport(
        operation="import",
        file_format=file_format,
        provider="OCP",
        source_length_unit=policy.source_length_unit,
        target_length_unit="m",
        coordinate_scale=scale,
        source_id=cad_model.source_id,
        source_revision=cad_model.source_revision,
        artifact_digest=artifact_digest,
        vertex_count=int(points.shape[0]),
        cell_count=int(faces.shape[0]),
        source_ids_preserved=False,
        tags_preserved=True,
        source_metadata_preserved=True,
        orientation_changed=orientation_changed,
        lossy=True,
        losses=losses,
        fields=(),
    )
    return SurfaceImportResult(model, (), report, cad_model)


def import_surface(
    path: str | Path,
    policy: SurfaceImportPolicy,
    /,
    *,
    file_format: SurfaceFileFormat | None = None,
) -> SurfaceImportResult:
    """Import a surface to authoritative SI coordinates without healing or welding."""

    if not isinstance(policy, SurfaceImportPolicy):
        raise TypeError("policy must be SurfaceImportPolicy.")
    source = Path(path).expanduser().resolve()
    format_ = _resolve_format(source, file_format)
    if format_ in (SurfaceFileFormat.STEP, SurfaceFileFormat.IGES):
        _require_module("OCP", f"{format_.value} direct BRep import")
    else:
        _require_module("meshio", f"{format_.value} surface import")
    _preflight_file(source, policy.maximum_file_bytes)
    digest = _artifact_digest(source)
    if format_ in (SurfaceFileFormat.STEP, SurfaceFileFormat.IGES):
        return _import_cad_surface(source, format_, policy, digest)
    return _import_meshio_surface(source, format_, policy, digest)


def _validate_export_fields(
    fields: Sequence[PortableSurfaceField],
    point_count: int,
    cell_count: int,
    maximum_fields: int,
    /,
) -> tuple[PortableSurfaceField, ...]:
    values = tuple(fields)
    if len(values) > maximum_fields:
        raise SurfaceResourceLimitError("Portable field count exceeds export capacity.")
    if not all(isinstance(field, PortableSurfaceField) for field in values):
        raise TypeError("fields must contain PortableSurfaceField values.")
    if len({(field.association, field.name) for field in values}) != len(values):
        raise ValueError("Portable field association/name pairs must be unique.")
    for field in values:
        expected = (
            point_count
            if field.association is SurfaceFieldAssociation.POINT
            else cell_count
        )
        if field.values.shape[0] != expected:
            raise ValueError(
                f"Field {field.name!r} has {field.values.shape[0]} entities; expected {expected}."
            )
    return values


def _metadata_point_data(model: SurfaceModel, point_count: int, unit: str, /):
    marker = np.zeros((point_count,), dtype=np.uint8)
    data = {
        _META_SOURCE + _encode_text(model.metadata.source_id): marker,
        _META_REVISION + _encode_text(model.metadata.source_revision): marker,
        _META_COORDINATES + _encode_text(model.metadata.coordinate_system): marker,
        _UNIT_PREFIX + _encode_text(unit): marker,
    }
    for index, entry in enumerate(model.metadata.provenance):
        data[f"{_META_PROVENANCE}{index:08d}__{_encode_text(entry)}"] = marker
    return data


def _meshio_export_mesh(
    meshio: Any,
    model: SurfaceModel,
    fields: Sequence[PortableSurfaceField],
    target_unit: str,
    include_metadata: bool,
    /,
):
    points = (
        np.asarray(model.mesh.coordinates, dtype=float) / _UNIT_TO_METERS[target_unit]
    )
    faces = np.asarray(model.mesh.connectivity.cell_vertices, dtype=np.int32)[:, :3]
    point_data = {}
    cell_data = {}
    field_data = {}
    if include_metadata:
        point_data.update(_metadata_point_data(model, points.shape[0], target_unit))
        point_data[_VERTEX_ID] = np.asarray(model.mesh.vertex_global_ids, dtype=np.int64)
        cell_data[_CELL_ID] = [
            np.asarray(model.mesh.entity_set(2).entity_ids, dtype=np.int64)
        ]
        unique_tags = tuple(dict.fromkeys(model.metadata.cell_tags))
        for tag in unique_tags:
            cell_data[_TAG_PREFIX + _encode_text(tag)] = [
                np.asarray(
                    [value == tag for value in model.metadata.cell_tags], dtype=np.uint8
                )
            ]
        if unique_tags:
            codes = {tag: index + 1 for index, tag in enumerate(unique_tags)}
            physical = np.asarray(
                [codes[tag] for tag in model.metadata.cell_tags], dtype=np.int32
            )
            cell_data["gmsh:physical"] = [physical]
            cell_data["gmsh:geometrical"] = [physical]
            field_data = {
                tag: np.asarray((code, 2), dtype=np.int32) for tag, code in codes.items()
            }
        for field in fields:
            if field.association is SurfaceFieldAssociation.POINT:
                point_data[field.name] = np.asarray(field.values)
            else:
                cell_data[field.name] = [np.asarray(field.values)]
    return meshio.Mesh(
        points,
        [("triangle", faces)],
        point_data=point_data,
        cell_data=cell_data,
        field_data=field_data,
    )


def export_surface(
    path: str | Path,
    model: SurfaceModel,
    policy: SurfaceExportPolicy,
    /,
    *,
    file_format: SurfaceFileFormat | None = None,
    fields: Sequence[PortableSurfaceField] = (),
) -> SurfaceExportResult:
    """Export an authoritative SI SurfaceModel without implicit geometry changes."""

    if not isinstance(model, SurfaceModel):
        raise TypeError("model must be SurfaceModel.")
    if not isinstance(policy, SurfaceExportPolicy):
        raise TypeError("policy must be SurfaceExportPolicy.")
    destination = Path(path).expanduser().resolve()
    format_ = _resolve_format(destination, file_format)
    if format_ in (SurfaceFileFormat.STEP, SurfaceFileFormat.IGES):
        _require_module("OCP", f"{format_.value} CAD export")
        raise SurfaceUnsupportedFormatError(
            "The existing OCCT/BRep substrate provides direct STEP/IGES import but no "
            "real SurfaceModel STEP/IGES writer; no fallback mesh export is permitted."
        )
    meshio = _require_module("meshio", f"{format_.value} surface export")
    if _canonical_unit(model.metadata.length_unit) != "m":
        raise SurfaceInteropError(
            "Surface export requires authoritative coordinates declared in SI meters."
        )
    point_count = int(model.mesh.coordinates.shape[0])
    cell_count = int(model.mesh.connectivity.cell_count)
    if point_count > policy.maximum_vertices or cell_count > policy.maximum_cells:
        raise SurfaceResourceLimitError("Surface entity counts exceed export capacities.")
    fields_ = _validate_export_fields(
        fields,
        point_count,
        cell_count,
        policy.maximum_fields,
    )
    loss_candidates = []
    include_metadata = format_ in (SurfaceFileFormat.GMSH, SurfaceFileFormat.VTK)
    if not include_metadata:
        loss_candidates.extend(
            (
                f"{format_.value}_cannot_retain_global_ids",
                f"{format_.value}_cannot_retain_cell_tags",
                f"{format_.value}_cannot_retain_units_or_provenance",
            )
        )
        if fields_:
            loss_candidates.append(f"{format_.value}_cannot_retain_portable_fields")
    losses = _loss_permission(loss_candidates, policy.allow_lossy)
    mesh = _meshio_export_mesh(
        meshio,
        model,
        fields_,
        policy.target_length_unit,
        include_metadata,
    )
    payload_bytes = _array_payload_bytes(mesh)
    if payload_bytes > policy.maximum_data_bytes:
        raise SurfaceResourceLimitError(
            f"Export payload has {payload_bytes} bytes, exceeding limit "
            f"{policy.maximum_data_bytes}."
        )
    write_format = _meshio_file_format(format_, destination)
    try:
        if format_ is SurfaceFileFormat.GMSH:
            meshio.write(destination, mesh, file_format="gmsh22", binary=policy.binary)
        elif format_ in (SurfaceFileFormat.VTK, SurfaceFileFormat.STL):
            meshio.write(
                destination, mesh, file_format=write_format, binary=policy.binary
            )
        else:
            meshio.write(destination, mesh, file_format=write_format)
    except meshio.WriteError as error:
        raise SurfaceInteropError(
            f"Provider could not write {format_.value} surface artifact {destination}."
        ) from error
    digest = _artifact_digest(destination)
    report = SurfaceInteropReport(
        operation="export",
        file_format=format_,
        provider="meshio",
        source_length_unit="m",
        target_length_unit=policy.target_length_unit,
        coordinate_scale=1.0 / _UNIT_TO_METERS[policy.target_length_unit],
        source_id=model.metadata.source_id,
        source_revision=model.metadata.source_revision,
        artifact_digest=digest,
        vertex_count=point_count,
        cell_count=cell_count,
        source_ids_preserved=include_metadata,
        tags_preserved=include_metadata,
        source_metadata_preserved=include_metadata,
        orientation_changed=False,
        lossy=bool(losses),
        losses=losses,
        fields=_field_records(fields_) if include_metadata else (),
    )
    return SurfaceExportResult(destination, report)


__all__ = [
    "PortableSurfaceField",
    "SurfaceDataCorruptionError",
    "SurfaceExportPolicy",
    "SurfaceExportResult",
    "SurfaceFieldAssociation",
    "SurfaceFieldRecord",
    "SurfaceFileFormat",
    "SurfaceImportPolicy",
    "SurfaceImportResult",
    "SurfaceInteropError",
    "SurfaceInteropReport",
    "SurfaceLossyOperationError",
    "SurfaceOrientationPolicy",
    "SurfaceProviderUnavailableError",
    "SurfaceResourceLimitError",
    "SurfaceUnsupportedFormatError",
    "export_surface",
    "import_surface",
]
