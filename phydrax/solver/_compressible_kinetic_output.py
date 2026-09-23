#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from xml.sax.saxutils import escape

import numpy as np


@dataclass(frozen=True, slots=True)
class CompressibleKineticVTKResult:
    path: Path
    sha256: str
    byte_size: int
    field_names: tuple[str, ...]


def write_compressible_kinetic_vti(
    path: str | Path,
    fields: Mapping[str, np.ndarray],
    /,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> CompressibleKineticVTKResult:
    """Write accepted scalar/vector fields as deterministic ASCII VTK ImageData."""
    destination = Path(path).expanduser().absolute()
    if destination.suffix.lower() != ".vti":
        raise ValueError("Compressible kinetic VTK output requires a .vti path.")
    items = tuple(
        sorted((str(name), np.asarray(value)) for name, value in fields.items())
    )
    if not items or any(not name for name, _ in items):
        raise ValueError("fields must contain named arrays.")
    first = items[0][1]
    if first.ndim not in (3, 4):
        raise ValueError("VTK ImageData fields must be scalar or vector 3D arrays.")
    shape = first.shape[:3]
    if any(size < 1 for size in shape):
        raise ValueError("VTK ImageData extents must be positive.")
    arrays = []
    for name, value in items:
        if value.shape[:3] != shape or value.ndim not in (3, 4):
            raise ValueError("All VTK fields must share one 3D spatial shape.")
        components = 1 if value.ndim == 3 else value.shape[-1]
        if components < 1 or not np.all(np.isfinite(value)):
            raise ValueError("VTK fields must contain finite scalar/vector values.")
        flattened = (
            value.reshape((-1, components)) if components > 1 else value.reshape((-1, 1))
        )
        text = " ".join(
            format(float(entry), ".17g") for entry in flattened.ravel(order="C")
        )
        arrays.append(
            f'<DataArray type="Float64" Name="{escape(name)}" '
            f'NumberOfComponents="{components}" format="ascii">{text}</DataArray>'
        )
    extent = f"0 {shape[0] - 1} 0 {shape[1] - 1} 0 {shape[2] - 1}"
    spacing_text = " ".join(format(float(value), ".17g") for value in spacing)
    origin_text = " ".join(format(float(value), ".17g") for value in origin)
    document = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">\n'
        f'  <ImageData WholeExtent="{extent}" Origin="{origin_text}" Spacing="{spacing_text}">\n'
        f'    <Piece Extent="{extent}">\n'
        "      <PointData>\n        "
        + "\n        ".join(arrays)
        + "\n      </PointData>\n      <CellData/>\n"
        "    </Piece>\n  </ImageData>\n</VTKFile>\n"
    ).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(document)
    temporary.replace(destination)
    return CompressibleKineticVTKResult(
        destination,
        hashlib.sha256(document).hexdigest(),
        len(document),
        tuple(name for name, _ in items),
    )


__all__ = ["CompressibleKineticVTKResult", "write_compressible_kinetic_vti"]
