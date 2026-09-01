#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import numpy as np

from ...interchange import (
    AdapterError,
    AdapterLoss,
    AdapterReport,
    AdapterStatus,
    require_lossless,
)
from ..imaging import DenseDisplacementField2D
from ..piv import PhysicalPIVResult2D
from ._piv_field import field_columns, field_from_columns


OpenPIVCoordinateConvention = Literal["physical", "image"]
OpenPIVValueKind = Literal["pixel-displacement", "physical-velocity"]


def read_openpiv_text(
    path: str | Path,
    /,
    *,
    value_kind: OpenPIVValueKind,
    geometry_id: str | None = None,
    spatial_unit: str | None = None,
    time_unit: str | None = None,
    coordinate_convention: OpenPIVCoordinateConvention = "physical",
    pixels_per_unit: float = 1.0,
    delta_t: float = 1.0,
    delimiter: str | None = None,
) -> tuple[DenseDisplacementField2D | PhysicalPIVResult2D, AdapterReport]:
    """Read OpenPIV text only under an explicit pixel or physical interpretation."""
    source = Path(path)
    scale, time = _scales(pixels_per_unit, delta_t)
    if value_kind not in ("pixel-displacement", "physical-velocity"):
        raise ValueError(
            "value_kind must be 'pixel-displacement' or 'physical-velocity'."
        )
    if coordinate_convention not in ("physical", "image"):
        raise ValueError("coordinate_convention must be 'physical' or 'image'.")
    with source.open("r", encoding="utf-8") as stream:
        header = stream.readline().lstrip("#").strip()
    columns = (
        tuple(header.split(delimiter)) if delimiter is not None else tuple(header.split())
    )
    columns = tuple(column.strip() for column in columns)
    if len(columns) != len(set(columns)) or not {"x", "y", "u", "v"}.issubset(columns):
        raise AdapterError(
            AdapterStatus.MALFORMED_SOURCE,
            "OpenPIV text requires unique named x, y, u, and v columns.",
        )
    table = np.atleast_1d(
        np.genfromtxt(
            source,
            names=True,
            delimiter=delimiter,
            dtype=None,
            encoding=None,
        )
    )
    if table.size == 0 or table.dtype.names != columns:
        raise AdapterError(
            AdapterStatus.MALFORMED_SOURCE,
            "OpenPIV text data do not match the declared header.",
        )
    x = _numeric_column(table, "x")
    y = _numeric_column(table, "y")
    u = _numeric_column(table, "u")
    v = _numeric_column(table, "v")
    flags = (
        _integer_column(table, "flags")
        if "flags" in columns
        else np.zeros(table.size, dtype=int)
    )
    mask = (
        _integer_column(table, "mask")
        if "mask" in columns
        else np.zeros(table.size, dtype=int)
    )
    if np.any(flags < 0) or np.any(mask < 0):
        raise AdapterError(
            AdapterStatus.INCONSISTENT_SOURCE,
            "OpenPIV flags and masks must be non-negative integers.",
        )
    valid = (flags == 0) & (mask == 0)
    source_id = _file_id(source, "openpiv-text")
    if value_kind == "physical-velocity":
        if coordinate_convention != "physical":
            raise AdapterError(
                AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                "Physical OpenPIV velocity requires the right-handed physical coordinate convention.",
            )
        spatial = "" if spatial_unit is None else str(spatial_unit).strip()
        temporal = "" if time_unit is None else str(time_unit).strip()
        if not spatial or not temporal:
            raise AdapterError(
                AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                "Physical OpenPIV velocity requires explicit spatial_unit and time_unit.",
            )
        positions, velocity, validity = _physical_grid(x, y, u, v, valid)
        displacement = velocity * time
        field = PhysicalPIVResult2D(
            positions,
            displacement,
            velocity,
            validity,
            source_id,
            f"openpiv-physical:{source_id}",
            spatial,
            temporal,
        )
        mapping = (
            "OpenPIV x -> physical x",
            "OpenPIV y -> physical y",
            "OpenPIV u -> velocity x",
            "OpenPIV v -> velocity y",
        )
        target_format = "PhysicalPIVResult2D"
        target_id = source_id
        preserved = ("positions_xy", "displacement_xy", "velocity_xy", "valid")
    else:
        if geometry_id is None or not str(geometry_id).strip():
            raise AdapterError(
                AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                "Pixel-displacement OpenPIV import requires geometry_id.",
            )
        if coordinate_convention == "physical":
            row = (np.min(y) + np.max(y) - y) * scale
            displacement_row = -v * time * scale
            mapping = (
                "OpenPIV x -> column_right",
                "OpenPIV physical y -> reflected row_down",
                "OpenPIV u * delta_t -> delta_column_right",
                "-OpenPIV v * delta_t -> delta_row_down",
            )
        else:
            row = y * scale
            displacement_row = v * time * scale
            mapping = (
                "OpenPIV x -> column_right",
                "OpenPIV image y -> row_down",
                "OpenPIV u * delta_t -> delta_column_right",
                "OpenPIV v * delta_t -> delta_row_down",
            )
        field = field_from_columns(
            row,
            x * scale,
            displacement_row,
            u * time * scale,
            valid,
            geometry_id=str(geometry_id),
            source_id=source_id,
        )
        target_format = "DenseDisplacementField2D"
        target_id = field.field_id
        preserved = ("positions_rc", "displacement_rc", "valid")
    losses = [
        AdapterLoss(
            "vector_status",
            "import",
            "dropped",
            "Native validity is preserved, but the OpenPIV numeric flag category has no field-level status slot.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "measurement_provenance",
            "import",
            "unsupported",
            "OpenPIV text does not encode interrogation, validation, calibration, or uncertainty provenance.",
            changes_interpretation=True,
        ),
    ]
    if "flags" not in columns:
        losses.append(
            AdapterLoss(
                "flags",
                "import",
                "synthesized",
                "The source omitted flags; all unmasked finite vectors were treated as unflagged.",
                changes_interpretation=True,
            )
        )
    if "mask" not in columns:
        losses.append(
            AdapterLoss(
                "mask",
                "import",
                "synthesized",
                "The source omitted its image-region mask.",
                changes_interpretation=True,
            )
        )
    extras = tuple(
        name for name in columns if name not in {"x", "y", "u", "v", "flags", "mask"}
    )
    if extras:
        losses.append(
            AdapterLoss(
                "columns." + ",".join(extras),
                "import",
                "dropped",
                "Extra OpenPIV table columns have no selected native field semantic.",
                changes_interpretation=False,
            )
        )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        "OpenPIV-text",
        target_format,
        source_id=source_id,
        target_id=target_id,
        coordinate_mapping=mapping,
        preserved_fields=preserved,
        assumptions=(
            f"value_kind={value_kind}",
            f"pixels_per_unit={scale}",
            f"delta_t={time}",
            f"coordinate_convention={coordinate_convention}",
        ),
        losses=losses,
    )
    return field, report


def write_openpiv_text(
    path: str | Path,
    field: DenseDisplacementField2D | PhysicalPIVResult2D,
    /,
    *,
    coordinate_convention: OpenPIVCoordinateConvention = "physical",
    pixels_per_unit: float = 1.0,
    delta_t: float = 1.0,
    delimiter: str = "\t",
    fmt: str = "%.9g",
    lossless: bool = False,
) -> AdapterReport:
    """Write OpenPIV text only after declaring its unavoidable semantic losses."""
    scale, time = _scales(pixels_per_unit, delta_t)
    if coordinate_convention not in ("physical", "image"):
        raise ValueError("coordinate_convention must be 'physical' or 'image'.")
    if isinstance(field, PhysicalPIVResult2D):
        if coordinate_convention != "physical":
            raise AdapterError(
                AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                "PhysicalPIVResult2D can only be exported under the physical OpenPIV convention.",
            )
        positions = np.asarray(field.positions_xy)
        velocity = np.asarray(field.velocity_xy)
        valid = np.asarray(field.valid, dtype=bool)
        if (
            positions.ndim != 3
            or positions.shape[-1] != 2
            or velocity.shape != positions.shape
            or valid.shape != positions.shape[:-1]
        ):
            raise AdapterError(
                AdapterStatus.INCONSISTENT_SOURCE,
                "PhysicalPIVResult2D arrays have inconsistent shapes.",
            )
        x, y = positions[..., 0], positions[..., 1]
        u, v = velocity[..., 0], velocity[..., 1]
        mapping = (
            "physical x -> OpenPIV x",
            "physical y -> OpenPIV y",
            "velocity x -> OpenPIV u",
            "velocity y -> OpenPIV v",
        )
        source_id = field.source_field_id
        source_format = "PhysicalPIVResult2D"
        preserved = ("positions_xy", "velocity_xy", "valid")
    elif isinstance(field, DenseDisplacementField2D):
        row, column, dr, dc, valid = field_columns(field)
        x = column / scale
        u = dc / (time * scale)
        if coordinate_convention == "physical":
            y = (np.min(row) + np.max(row) - row) / scale
            v = -dr / (time * scale)
            mapping = (
                "column_right -> OpenPIV x",
                "reflected row_down -> OpenPIV physical y",
                "delta_column_right / delta_t -> OpenPIV u",
                "-delta_row_down / delta_t -> OpenPIV v",
            )
        else:
            y = row / scale
            v = dr / (time * scale)
            mapping = (
                "column_right -> OpenPIV x",
                "row_down -> OpenPIV image y",
                "delta_column_right / delta_t -> OpenPIV u",
                "delta_row_down / delta_t -> OpenPIV v",
            )
        source_id = field.field_id
        source_format = "DenseDisplacementField2D"
        preserved = ("positions_rc", "displacement_rc", "valid")
    else:
        raise TypeError("field must be DenseDisplacementField2D or PhysicalPIVResult2D.")
    flags = (~valid).astype(np.int32)
    mask = np.zeros(valid.shape, dtype=np.int32)
    values = np.column_stack(
        tuple(array.reshape((-1,)) for array in (x, y, u, v, flags, mask))
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        destination,
        values,
        delimiter=delimiter,
        fmt=fmt,
        header=delimiter.join(("x", "y", "u", "v", "flags", "mask")),
    )
    target_id = _file_id(destination, "openpiv-text")
    losses = (
        AdapterLoss(
            "identity_and_provenance",
            "export",
            "dropped",
            "OpenPIV text has no native identity or provenance fields.",
            changes_interpretation=True,
        ),
        AdapterLoss(
            "invalid_reason",
            "export",
            "synthesized",
            "All invalid native samples were encoded as flag=1 and mask=0 because their cause is unavailable.",
            changes_interpretation=False,
        ),
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        source_format,
        "OpenPIV-text",
        source_id=source_id,
        target_id=target_id,
        coordinate_mapping=mapping,
        preserved_fields=preserved,
        assumptions=(f"pixels_per_unit={scale}", f"delta_t={time}"),
        losses=losses,
    )
    if lossless:
        require_lossless(report)
    return report


def _physical_grid(
    x,
    y,
    u,
    v,
    valid,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_ = np.asarray(x, dtype=float).reshape((-1,))
    y_ = np.asarray(y, dtype=float).reshape((-1,))
    u_ = np.asarray(u, dtype=float).reshape((-1,))
    v_ = np.asarray(v, dtype=float).reshape((-1,))
    valid_ = np.asarray(valid, dtype=bool).reshape((-1,))
    if np.any(valid_ & (~np.isfinite(u_) | ~np.isfinite(v_))):
        raise AdapterError(
            AdapterStatus.INCONSISTENT_SOURCE,
            "Every valid physical OpenPIV velocity must be finite.",
        )
    x_axis = np.unique(x_)
    y_axis = np.unique(y_)
    if x_axis.size * y_axis.size != x_.size:
        raise AdapterError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "Physical OpenPIV positions must define a complete rectilinear grid.",
        )
    x_index = np.searchsorted(x_axis, x_)
    y_index = np.searchsorted(y_axis, y_)
    flat = y_index * x_axis.size + x_index
    if np.unique(flat).size != x_.size:
        raise AdapterError(
            AdapterStatus.INCONSISTENT_SOURCE,
            "Physical OpenPIV positions contain duplicates.",
        )
    shape = (y_axis.size, x_axis.size)
    positions = np.empty(shape + (2,), dtype=float)
    positions[..., 0] = x_axis[None, :]
    positions[..., 1] = y_axis[:, None]
    velocity = np.zeros(shape + (2,), dtype=np.result_type(u_, v_, float))
    validity = np.zeros(shape, dtype=bool)
    velocity[y_index, x_index, 0] = np.where(valid_, u_, 0.0)
    velocity[y_index, x_index, 1] = np.where(valid_, v_, 0.0)
    validity[y_index, x_index] = valid_
    return positions, velocity, validity


def _scales(pixels_per_unit: float, delta_t: float, /) -> tuple[float, float]:
    scale = float(pixels_per_unit)
    time = float(delta_t)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("pixels_per_unit must be finite and positive.")
    if not np.isfinite(time) or time <= 0.0:
        raise ValueError("delta_t must be finite and positive.")
    return scale, time


def _numeric_column(table: np.ndarray, name: str, /) -> np.ndarray:
    value = np.asarray(table[name])
    if value.dtype.hasobject or not np.issubdtype(value.dtype, np.number):
        raise AdapterError(
            AdapterStatus.MALFORMED_SOURCE,
            f"OpenPIV column {name!r} must be numeric.",
        )
    return value.astype(float, copy=False)


def _integer_column(table: np.ndarray, name: str, /) -> np.ndarray:
    value = _numeric_column(table, name)
    if np.any(value != np.floor(value)):
        raise AdapterError(
            AdapterStatus.INCONSISTENT_SOURCE,
            f"OpenPIV column {name!r} must contain integers.",
        )
    return value.astype(np.int64)


def _file_id(path: Path, format_name: str, /) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"{format_name}:sha256:{digest}"


__all__ = [
    "OpenPIVCoordinateConvention",
    "OpenPIVValueKind",
    "read_openpiv_text",
    "write_openpiv_text",
]
