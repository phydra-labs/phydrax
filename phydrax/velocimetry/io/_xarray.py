#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from ...interchange import AdapterError, AdapterLoss, AdapterReport, AdapterStatus
from ..imaging import DenseDisplacementField2D
from ._piv_field import field_columns, field_from_columns


PivpyYAxis = Literal["down", "up"]


def is_xarray_available() -> bool:
    """Return whether xarray can be imported, without importing it."""
    return importlib.util.find_spec("xarray") is not None


def is_pivpy_available() -> bool:
    """Return whether pivpy can be imported, without importing it."""
    return importlib.util.find_spec("pivpy") is not None


def require_xarray():
    """Import xarray only at the interoperability call boundary."""
    if not is_xarray_available():
        raise AdapterError(
            AdapterStatus.OPTIONAL_DEPENDENCY_UNAVAILABLE,
            "xarray conversion requires optional dependency 'xarray'.",
        )
    return importlib.import_module("xarray")


def require_pivpy():
    """Import pivpy only at the interoperability call boundary."""
    if not is_pivpy_available():
        raise AdapterError(
            AdapterStatus.OPTIONAL_DEPENDENCY_UNAVAILABLE,
            "pivpy conversion requires optional dependency 'pivpy'.",
        )
    return importlib.import_module("pivpy")


def to_xarray(field: DenseDisplacementField2D, /) -> tuple[Any, AdapterReport]:
    """Convert a native field to an explicitly labeled xarray Dataset."""
    xr = require_xarray()
    row, column, dr, dc, valid = field_columns(field)
    dataset = xr.Dataset(
        data_vars={
            "position_row_down": (("row", "column"), row),
            "position_column_right": (("row", "column"), column),
            "displacement_row_down": (("row", "column"), dr),
            "displacement_column_right": (("row", "column"), dc),
            "valid": (("row", "column"), valid),
        },
        coords={"row": row[:, 0], "column": column[0, :]},
        attrs={
            "coordinate_convention": "row-down-column-right",
            "geometry_id": field.geometry_id,
            "field_id": field.field_id,
            "provenance": list(field.provenance),
        },
    )
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        "DenseDisplacementField2D",
        "xarray.Dataset",
        source_id=field.field_id,
        target_id=field.field_id,
        coordinate_mapping=("row -> row_down", "column -> column_right"),
        preserved_fields=(
            "positions_rc",
            "displacement_rc",
            "valid",
            "geometry_id",
            "field_id",
            "provenance",
        ),
    )
    return dataset, report


def from_xarray(
    dataset: Any,
    /,
    *,
    materialize_lazy: bool = False,
) -> tuple[DenseDisplacementField2D, AdapterReport]:
    """Restore the exact labeled native Dataset layout, rejecting ambiguous arrays."""
    xr = require_xarray()
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("dataset must be an xarray.Dataset.")
    required = {
        "position_row_down",
        "position_column_right",
        "displacement_row_down",
        "displacement_column_right",
        "valid",
    }
    if set(dataset.data_vars) != required:
        raise AdapterError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "xarray Dataset must contain exactly the native labeled field variables.",
        )
    for name in required:
        if tuple(dataset[name].dims) != ("row", "column"):
            raise AdapterError(
                AdapterStatus.INCONSISTENT_SOURCE,
                f"xarray variable {name!r} must have dimensions ('row', 'column').",
            )
    arrays = {
        name: _array_data(dataset[name].data, materialize_lazy=materialize_lazy)
        for name in required
    }
    attrs = dict(dataset.attrs)
    if set(attrs) != {"coordinate_convention", "geometry_id", "field_id", "provenance"}:
        raise AdapterError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "xarray Dataset native metadata fields are incomplete or unexpected.",
        )
    if attrs["coordinate_convention"] != "row-down-column-right":
        raise AdapterError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "xarray Dataset has an incompatible coordinate convention.",
        )
    provenance = attrs["provenance"]
    if not isinstance(provenance, (tuple, list)) or not all(
        isinstance(item, str) and item for item in provenance
    ):
        raise AdapterError(
            AdapterStatus.INCONSISTENT_SOURCE,
            "xarray Dataset provenance must be a sequence of non-empty strings.",
        )
    validated = field_from_columns(
        arrays["position_row_down"],
        arrays["position_column_right"],
        arrays["displacement_row_down"],
        arrays["displacement_column_right"],
        arrays["valid"],
        geometry_id=str(attrs["geometry_id"]),
        source_id=str(attrs["field_id"]),
    )
    field = DenseDisplacementField2D(
        validated.positions_rc,
        validated.displacement_rc,
        validated.valid,
        geometry_id=str(attrs["geometry_id"]),
        field_id=str(attrs["field_id"]),
        provenance=tuple(provenance),
    )
    lazy = any(_is_lazy(dataset[name].data) for name in required)
    losses = (
        (
            AdapterLoss(
                "array_execution",
                "import",
                "transformed",
                "Lazy external arrays were explicitly materialized on the host.",
                changes_interpretation=False,
            ),
        )
        if lazy
        else ()
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS if lazy else AdapterStatus.LOSSLESS,
        "xarray.Dataset",
        "DenseDisplacementField2D",
        source_id=field.field_id,
        target_id=field.field_id,
        coordinate_mapping=("row -> row_down", "column -> column_right"),
        preserved_fields=(
            "positions_rc",
            "displacement_rc",
            "valid",
            "geometry_id",
            "field_id",
            "provenance",
        ),
        losses=losses,
    )
    return field, report


def to_pivpy(
    fields: DenseDisplacementField2D | Sequence[DenseDisplacementField2D],
    /,
    *,
    times: Sequence[float] | None = None,
    y_axis: PivpyYAxis = "up",
) -> tuple[Any, AdapterReport]:
    """Create pivpy's y/x/t Eulerian layout without importing it at package import."""
    require_pivpy()
    xr = require_xarray()
    fields_ = (fields,) if isinstance(fields, DenseDisplacementField2D) else tuple(fields)
    if not fields_:
        raise ValueError("pivpy conversion requires at least one field.")
    columns = [field_columns(field) for field in fields_]
    reference_row, reference_column = columns[0][0], columns[0][1]
    if any(
        not (
            np.array_equal(item[0], reference_row)
            and np.array_equal(item[1], reference_column)
        )
        for item in columns[1:]
    ):
        raise AdapterError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "pivpy conversion requires one grid shared by every time sample.",
        )
    times_ = (
        np.arange(len(fields_), dtype=float)
        if times is None
        else np.asarray(tuple(times), dtype=float)
    )
    if times_.shape != (len(fields_),) or not np.all(np.isfinite(times_)):
        raise ValueError("times must contain one finite value per field.")
    if y_axis not in ("down", "up"):
        raise ValueError("y_axis must be 'down' or 'up'.")
    row_axis = reference_row[:, 0]
    x = reference_column[0, :]
    y = row_axis if y_axis == "down" else np.min(row_axis) + np.max(row_axis) - row_axis
    u = np.stack([item[3] for item in columns], axis=-1)
    v_native = np.stack([item[2] for item in columns], axis=-1)
    v = v_native if y_axis == "down" else -v_native
    chc = np.stack([item[4] for item in columns], axis=-1).astype(np.int8)
    dataset = xr.Dataset(
        data_vars={
            "u": (("y", "x", "t"), u),
            "v": (("y", "x", "t"), v),
            "chc": (("y", "x", "t"), chc),
        },
        coords={"y": y, "x": x, "t": times_},
        attrs={
            "coordinate_convention": "y-up-x-right"
            if y_axis == "up"
            else "y-down-x-right",
            "geometry_id": fields_[0].geometry_id,
            "field_ids": [field.field_id for field in fields_],
            "provenance": [list(field.provenance) for field in fields_],
        },
    )
    losses = (
        AdapterLoss(
            "validity",
            "export",
            "transformed",
            "Native Boolean validity was encoded as pivpy chc=1/0 without collapsing "
            "invalid vectors to valid zero vectors.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "pivpy_schema_version",
            "export",
            "unsupported",
            "No pivpy schema-version attribute is emitted; the current typed conversion is canonical.",
            changes_interpretation=False,
        ),
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        "DenseDisplacementField2D-sequence",
        "pivpy-xarray.Dataset",
        source_id=fields_[0].field_id,
        target_id="pivpy:" + ":".join(field.field_id for field in fields_),
        coordinate_mapping=(
            "column_right -> x",
            "row_down -> y" if y_axis == "down" else "reflected row_down -> y_up",
            "delta_column_right -> u",
            "delta_row_down -> v" if y_axis == "down" else "-delta_row_down -> v_up",
        ),
        preserved_fields=("positions_rc", "displacement_rc", "valid", "times"),
        losses=losses,
    )
    return dataset, report


def from_pivpy(
    dataset: Any,
    /,
    *,
    geometry_id: str,
    y_axis: PivpyYAxis = "up",
    materialize_lazy: bool = False,
) -> tuple[tuple[DenseDisplacementField2D, ...], AdapterReport]:
    """Import a strict pivpy y/x/t field sequence while preserving invalid zeros."""
    require_pivpy()
    xr = require_xarray()
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("dataset must be an xarray.Dataset.")
    if not {"u", "v", "chc"}.issubset(dataset.data_vars):
        raise AdapterError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "pivpy Dataset requires u, v, and chc variables.",
        )
    if any(tuple(dataset[name].dims) != ("y", "x", "t") for name in ("u", "v", "chc")):
        raise AdapterError(
            AdapterStatus.INCONSISTENT_SOURCE,
            "pivpy variables must have exact dimension order ('y', 'x', 't').",
        )
    if y_axis not in ("down", "up"):
        raise ValueError("y_axis must be 'down' or 'up'.")
    y = _array_data(dataset.coords["y"].data, materialize_lazy=materialize_lazy).reshape(
        (-1,)
    )
    x = _array_data(dataset.coords["x"].data, materialize_lazy=materialize_lazy).reshape(
        (-1,)
    )
    if (
        y.size == 0
        or x.size == 0
        or not np.all(np.isfinite(y))
        or not np.all(np.isfinite(x))
        or not (np.all(np.diff(y) > 0.0) or np.all(np.diff(y) < 0.0))
        or not np.all(np.diff(x) > 0.0)
    ):
        raise AdapterError(
            AdapterStatus.INCONSISTENT_SOURCE,
            "pivpy x/y coordinates must be finite, unique, and monotone.",
        )
    u = _array_data(dataset["u"].data, materialize_lazy=materialize_lazy)
    v = _array_data(dataset["v"].data, materialize_lazy=materialize_lazy)
    chc = _array_data(dataset["chc"].data, materialize_lazy=materialize_lazy)
    if (
        u.shape != (y.size, x.size, dataset.sizes["t"])
        or v.shape != u.shape
        or chc.shape != u.shape
    ):
        raise AdapterError(
            AdapterStatus.INCONSISTENT_SOURCE,
            "pivpy coordinate and variable shapes are inconsistent.",
        )
    row_axis = y if y_axis == "down" else np.min(y) + np.max(y) - y
    row_grid = np.broadcast_to(row_axis[:, None], (y.size, x.size))
    column_grid = np.broadcast_to(x[None, :], (y.size, x.size))
    fields: list[DenseDisplacementField2D] = []
    for index in range(u.shape[-1]):
        valid = (
            (chc[..., index] > 0)
            & np.isfinite(u[..., index])
            & np.isfinite(v[..., index])
        )
        dr = v[..., index] if y_axis == "down" else -v[..., index]
        fields.append(
            field_from_columns(
                row_grid,
                column_grid,
                dr,
                u[..., index],
                valid,
                geometry_id=geometry_id,
                source_id=f"pivpy:frame:{index}",
            )
        )
    losses = (
        AdapterLoss(
            "chc",
            "import",
            "transformed",
            "pivpy's numeric chc values were reduced to authoritative Boolean validity without changing vector zeros.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "attrs",
            "import",
            "dropped",
            "Unrecognized mutable pivpy attributes and accessor state are not part of the native field contract.",
            changes_interpretation=False,
        ),
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        "pivpy-xarray.Dataset",
        "DenseDisplacementField2D-sequence",
        source_id="pivpy-dataset",
        target_id="native:" + ":".join(field.field_id for field in fields),
        coordinate_mapping=(
            "x -> column_right",
            "y -> row_down" if y_axis == "down" else "reflected y_up -> row_down",
            "u -> delta_column_right",
            "v -> delta_row_down" if y_axis == "down" else "-v_up -> delta_row_down",
        ),
        preserved_fields=("x", "y", "u", "v", "chc", "t"),
        losses=losses,
    )
    return tuple(fields), report


def _is_lazy(data: Any, /) -> bool:
    return type(data).__module__.split(".", 1)[0] == "dask"


def _array_data(data: Any, /, *, materialize_lazy: bool) -> np.ndarray:
    if _is_lazy(data) and not materialize_lazy:
        raise AdapterError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "Lazy external arrays require materialize_lazy=True; implicit host computation is forbidden.",
        )
    value = np.asarray(data)
    if value.dtype.hasobject:
        raise AdapterError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "Object-dtype labeled arrays are unsupported.",
        )
    return value


__all__ = [
    "PivpyYAxis",
    "from_pivpy",
    "from_xarray",
    "is_pivpy_available",
    "is_xarray_available",
    "require_pivpy",
    "require_xarray",
    "to_pivpy",
    "to_xarray",
]
