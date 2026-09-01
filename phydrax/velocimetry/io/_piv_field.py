#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...interchange import AdapterError, AdapterStatus
from ..imaging import DenseDisplacementField2D


def field_from_columns(
    row: Any,
    column: Any,
    displacement_row: Any,
    displacement_column: Any,
    valid: Any,
    /,
    *,
    geometry_id: str,
    source_id: str,
) -> DenseDisplacementField2D:
    """Build one exact rectilinear native field from unordered table columns."""
    row_ = np.asarray(row, dtype=float).reshape((-1,))
    column_ = np.asarray(column, dtype=float).reshape((-1,))
    dr = np.asarray(displacement_row, dtype=float).reshape((-1,))
    dc = np.asarray(displacement_column, dtype=float).reshape((-1,))
    valid_ = np.asarray(valid, dtype=bool).reshape((-1,))
    size = row_.size
    if size == 0 or any(array.size != size for array in (column_, dr, dc, valid_)):
        raise AdapterError(
            AdapterStatus.MALFORMED_SOURCE,
            "PIV table columns must be non-empty and have equal length.",
        )
    if not np.all(np.isfinite(row_)) or not np.all(np.isfinite(column_)):
        raise AdapterError(
            AdapterStatus.INCONSISTENT_SOURCE,
            "PIV grid coordinates must be finite.",
        )
    if np.any(valid_ & (~np.isfinite(dr) | ~np.isfinite(dc))):
        raise AdapterError(
            AdapterStatus.INCONSISTENT_SOURCE,
            "Every valid PIV vector must be finite.",
        )
    rows = np.unique(row_)
    columns = np.unique(column_)
    if rows.size * columns.size != size:
        raise AdapterError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "PIV table coordinates do not define a complete rectilinear grid.",
        )
    row_index = np.searchsorted(rows, row_)
    column_index = np.searchsorted(columns, column_)
    flat_index = row_index * columns.size + column_index
    if np.unique(flat_index).size != size:
        raise AdapterError(
            AdapterStatus.INCONSISTENT_SOURCE,
            "PIV table contains duplicate grid coordinates.",
        )
    shape = (rows.size, columns.size)
    positions = np.empty(shape + (2,), dtype=float)
    displacement = np.zeros(shape + (2,), dtype=np.result_type(dr, dc, float))
    validity = np.zeros(shape, dtype=bool)
    positions[..., 0] = rows[:, None]
    positions[..., 1] = columns[None, :]
    displacement[row_index, column_index, 0] = np.where(valid_, dr, 0.0)
    displacement[row_index, column_index, 1] = np.where(valid_, dc, 0.0)
    validity[row_index, column_index] = valid_
    field_id = canonical_fingerprint(
        {
            "kind": "imported-dense-displacement-field-2d",
            "geometry_id": str(geometry_id),
            "source_id": str(source_id),
            "content": array_tree_fingerprint(
                {
                    "positions_rc": positions,
                    "displacement_rc": displacement,
                    "valid": validity,
                }
            ),
        }
    )
    return DenseDisplacementField2D(
        positions,
        displacement,
        validity,
        geometry_id=str(geometry_id),
        field_id=field_id,
        provenance=(str(source_id),),
    )


def field_columns(
    field: DenseDisplacementField2D,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return native row/column/vector columns after strict rectilinear validation."""
    if not isinstance(field, DenseDisplacementField2D):
        raise TypeError("field must be DenseDisplacementField2D.")
    positions = np.asarray(field.positions_rc)
    displacement = np.asarray(field.displacement_rc)
    valid = np.asarray(field.valid, dtype=bool)
    if (
        positions.ndim != 3
        or positions.shape[-1] != 2
        or displacement.shape != positions.shape
        or valid.shape != positions.shape[:-1]
    ):
        raise AdapterError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "External PIV adapters require one two-dimensional rectilinear field.",
        )
    rows = positions[:, 0, 0]
    columns = positions[0, :, 1]
    expected_row = np.broadcast_to(rows[:, None], valid.shape)
    expected_column = np.broadcast_to(columns[None, :], valid.shape)
    if not (
        np.allclose(positions[..., 0], expected_row, rtol=0.0, atol=0.0)
        and np.allclose(positions[..., 1], expected_column, rtol=0.0, atol=0.0)
        and np.all(np.diff(rows) > 0.0)
        and np.all(np.diff(columns) > 0.0)
    ):
        raise AdapterError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "External PIV adapters require strictly increasing rectilinear row and column coordinates.",
        )
    if np.any(valid & ~np.all(np.isfinite(displacement), axis=-1)):
        raise AdapterError(
            AdapterStatus.INCONSISTENT_SOURCE,
            "Every valid native PIV vector must be finite.",
        )
    return (
        positions[..., 0],
        positions[..., 1],
        displacement[..., 0],
        displacement[..., 1],
        valid,
    )


__all__: list[str] = []
