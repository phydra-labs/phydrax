#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Lazy, host-only xarray/PyGMT export and real local rendering.

Optional modules are imported only inside the requested operation. Native grids
remain independent immutable arrays; external plotting objects are never solver
state. No dataset lookup, network fetch, reprojection of native coordinates, or
substitute renderer is provided.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import equinox as eqx
import numpy as np

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint, canonical_json
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..units import conversion_factor, DEGREE
from ._geospatial import QualifiedGeospatialGrid
from ._report import AdapterLoss, AdapterReport, AdapterStatus
from ._resource import read_bounded_resource, ResourceLimits, ResourceManifest


class GeospatialDependencyError(ImportError):
    """An absent optional Python package, with an explicit unsuccessful report."""

    report: AdapterReport

    def __init__(self, package: str, source_id: str):
        self.report = AdapterReport(
            AdapterStatus.OPTIONAL_DEPENDENCY_UNAVAILABLE,
            "qualified-geospatial-grid",
            package,
            source_id=source_id,
            target_id=f"unavailable:{package}",
            losses=(
                AdapterLoss(
                    "runtime",
                    "export",
                    "unsupported",
                    f"The optional {package} package is not installed.",
                    changes_interpretation=False,
                ),
            ),
        )
        super().__init__(f"This operation requires the optional {package} package.")


def _optional_module(package: str, source_id: str):
    try:
        return importlib.import_module(package)
    except ModuleNotFoundError as error:
        if error.name != package:
            raise
        raise GeospatialDependencyError(package, source_id) from error


class XarrayGridExport(StrictModule, NonTrainableState):
    """Host-only external DataArray and auditable conversion; not numerical state."""

    dataarray: Any = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)
    export_id: str = eqx.field(static=True)
    region: tuple[float, float, float, float] = eqx.field(static=True)
    report: AdapterReport = eqx.field(static=True)

    def __init__(self, dataarray, grid_id, export_id, region, report):
        self.dataarray = dataarray
        self.grid_id = grid_id
        self.export_id = export_id
        self.region = region
        self.report = report


class PyGMTRenderResult(StrictModule, NonTrainableState):
    """Actual rendered local resource, figure, and complete rendering provenance.

    The returned figure is an external mutable host object. ``manifest`` hashes
    the actual saved output; failures propagate rather than returning this result.
    """

    figure: Any = eqx.field(static=True)
    manifest: ResourceManifest = eqx.field(static=True)
    report: AdapterReport = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)
    render_id: str = eqx.field(static=True)
    provenance_json: str = eqx.field(static=True)

    def __init__(self, figure, manifest, report, grid_id, render_id, provenance_json):
        self.figure = figure
        self.manifest = manifest
        self.report = report
        self.grid_id = grid_id
        self.render_id = render_id
        self.provenance_json = provenance_json

    @property
    def provenance(self) -> dict[str, Any]:
        return json.loads(self.provenance_json)


def export_geospatial_grid(
    grid: QualifiedGeospatialGrid,
    /,
    *,
    for_pygmt: bool = False,
) -> XarrayGridExport:
    """Export exact native samples to xarray, or explicitly adapt them to GMT.

    Plain xarray export preserves both coordinate orientations, values (including
    masked payloads), and an explicit ``valid`` coordinate. GMT export reverses
    descending axes with their data, converts angular coordinates to degrees,
    encodes invalid samples as NaN, and explicitly sets both GMT registration and
    coordinate type on the final DataArray. GMT's single-precision grid conversion
    is performed here and any rounding is recorded instead of hidden downstream.
    No interpolation or change to the native artifact occurs.
    """
    if not isinstance(grid, QualifiedGeospatialGrid):
        raise TypeError("grid must be QualifiedGeospatialGrid.")
    if not isinstance(for_pygmt, bool):
        raise TypeError("for_pygmt must be bool.")
    grid.contract.require_grid()
    xr = _optional_module("xarray", grid.grid_id)
    if for_pygmt:
        _optional_module("pygmt", grid.grid_id)
    x, y = np.array(grid.x), np.array(grid.y)
    values, valid = np.array(grid.values), np.array(grid.valid)
    region = grid.region
    losses = []
    mapping = ["DataArray dimensions are (y,x); valid[j,i] qualifies values[j,i]"]
    horizontal_units = grid.contract.horizontal_units
    x_unit, y_unit = horizontal_units
    if x_unit is None or y_unit is None:
        raise ValueError("Qualified grid export requires explicit horizontal units.")
    if for_pygmt:
        for axis, direction in enumerate(grid.orientation):
            if direction == "decreasing":
                if axis == 0:
                    x, values, valid = x[::-1], values[:, ::-1], valid[:, ::-1]
                else:
                    y, values, valid = y[::-1], values[::-1, :], valid[::-1, :]
                name = "x" if axis == 0 else "y"
                mapping.append(
                    f"{name} coordinates and associated values/mask reversed to increasing order"
                )
                losses.append(
                    AdapterLoss(
                        f"coordinates.{name}.orientation",
                        "export",
                        "transformed",
                        "GMT receives increasing coordinates with the corresponding data permutation.",
                        changes_interpretation=False,
                    )
                )
        if grid.contract.horizontal_kind == "geographic":
            factor = float(conversion_factor(x_unit, DEGREE))
            if factor != 1:
                x, y = x * factor, y * factor
                region = tuple(value * factor for value in region)
                mapping.append(
                    f"Angular coordinates converted to degrees with factor {factor!r}"
                )
                losses.append(
                    AdapterLoss(
                        "coordinates.units",
                        "export",
                        "transformed",
                        "GMT requires degrees; exact declared angular units are converted at this host boundary.",
                        changes_interpretation=False,
                    )
                )
            x_unit, y_unit = DEGREE, DEGREE
        with np.errstate(over="ignore", invalid="ignore"):
            float_values = values.astype(np.float32)
        if not np.all(np.isfinite(float_values[valid])):
            raise ValueError("Valid grid samples exceed GMT single-precision range.")
        if np.any(
            float_values[valid].astype(np.longdouble)
            != values[valid].astype(np.longdouble)
        ):
            losses.append(
                AdapterLoss(
                    "values.precision",
                    "export",
                    "transformed",
                    "GMT grids store float32 values; rounded samples are for visualization, not numerical composition.",
                    changes_interpretation=False,
                )
            )
        if not np.all(valid):
            float_values[~valid] = np.nan
            losses.append(
                AdapterLoss(
                    "values.masked_payload",
                    "export",
                    "dropped",
                    "Invalid sample payloads are replaced by GMT NaN; the explicit "
                    "Boolean validity coordinate is retained.",
                    changes_interpretation=False,
                )
            )
        values = float_values
    descriptor = {
        "kind": "geospatial-xarray-export",
        "grid_id": grid.grid_id,
        "for_pygmt": for_pygmt,
        "xarray_version": xr.__version__,
        "arrays": array_tree_fingerprint((x, y, values, valid)),
        "region": region,
    }
    export_id = canonical_fingerprint(descriptor)
    dataarray = xr.DataArray(
        values,
        dims=("y", "x"),
        coords={"x": x, "y": y, "valid": (("y", "x"), valid)},
        name=grid.name,
        attrs={
            "units": grid.value_unit.symbol,
            "phydrax_value_unit": canonical_json(grid.value_unit.to_dict()),
            "phydrax_grid_id": grid.grid_id,
            "phydrax_export_id": export_id,
            "phydrax_geospatial_contract": canonical_json(grid.contract.to_dict()),
            "phydrax_source_manifest": canonical_json(asdict(grid.manifest)),
            "phydrax_resources": canonical_json(
                [asdict(item) for item in grid.resources]
            ),
            "phydrax_value_role": grid.value_role,
            "registration": grid.contract.registration,
            "node_offset": int(grid.contract.registration == "pixel"),
            "region": list(region),
        },
    )
    dataarray.coords["x"].attrs = {"units": x_unit.symbol, "axis": "X"}
    dataarray.coords["y"].attrs = {"units": y_unit.symbol, "axis": "Y"}
    dataarray.coords["valid"].attrs = {"meaning": "True marks a valid sample"}
    if grid.value_role == "vertical":
        dataarray.attrs["positive"] = grid.contract.vertical_positive
        dataarray.attrs["vertical_datum"] = grid.contract.vertical_datum
    if for_pygmt:
        # Set these last: xarray operations create a new accessor with defaults.
        dataarray.gmt.registration = int(grid.contract.registration == "pixel")
        dataarray.gmt.gtype = int(grid.contract.horizontal_kind == "geographic")
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS if losses else AdapterStatus.LOSSLESS,
        "qualified-geospatial-grid",
        "pygmt-dataarray" if for_pygmt else "xarray-dataarray",
        source_id=grid.grid_id,
        target_id=export_id,
        coordinate_mapping=tuple(mapping),
        preserved_fields=(
            "coordinates",
            "registration",
            "vertical_reference",
            "longitude_seam",
            "mask",
            "metadata",
            "resources",
        ),
        losses=tuple(losses),
        stages=(grid.report,),
    )
    dataarray.attrs["phydrax_adapter_report_id"] = report.report_id
    return XarrayGridExport(dataarray, grid.grid_id, export_id, region, report)


def render_geospatial_grid(
    grid: QualifiedGeospatialGrid,
    output_path: str | os.PathLike[str],
    /,
    *,
    projection: str = "X12c",
    cmap: str = "viridis",
    frame: bool | str = True,
    dpi: int = 150,
    max_output_bytes: int = 64 * 1024 * 1024,
) -> PyGMTRenderResult:
    """Render the supplied grid with real PyGMT/GMT, then hash the saved resource.

    PNG, PDF, and EPS outputs are supported. Cartesian grids require a linear
    ``X``/``x`` display projection; geographic grids may select an explicit GMT map
    projection. A display projection never changes native coordinate metadata.
    Only built-in GMT color palettes are accepted; remote-resource spellings are
    rejected before the local-array render. Runtime/Ghostscript errors are not hidden.
    The output is atomically installed only after rendering and bounded reading
    succeed. No plot, report, or file is fabricated when a dependency is absent.
    """
    if not isinstance(grid, QualifiedGeospatialGrid):
        raise TypeError("grid must be QualifiedGeospatialGrid.")
    if (
        not isinstance(projection, str)
        or not projection
        or any(char in projection for char in ("@", "\n", "\r"))
    ):
        raise ValueError("An explicit local GMT display projection is required.")
    if grid.contract.horizontal_kind != "geographic" and projection[0] not in "Xx":
        raise ValueError(
            "Cartesian grids require a linear X/x display projection, not map reprojection."
        )
    if (
        not isinstance(cmap, str)
        or re.fullmatch(r"(?:gmt/)?[A-Za-z][A-Za-z0-9_]*", cmap) is None
    ):
        raise ValueError(
            "cmap must name a built-in GMT palette, not a local or remote resource."
        )
    if not isinstance(frame, (bool, str)) or (
        isinstance(frame, str) and any(char in frame for char in ("@", "\n", "\r"))
    ):
        raise ValueError("frame must be bool or a local GMT frame specification.")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi must be a positive integer.")
    if (
        isinstance(max_output_bytes, bool)
        or not isinstance(max_output_bytes, int)
        or max_output_bytes <= 0
    ):
        raise ValueError("max_output_bytes must be a positive integer.")
    path = Path(output_path).expanduser().absolute()
    if path.suffix.lower() not in (".png", ".pdf", ".eps"):
        raise ValueError("Rendering output must be a PNG, PDF, or EPS file.")
    parent = path.parent.resolve(strict=True)
    path = parent / path.name
    if path.is_symlink() or path.is_dir():
        raise ValueError("Rendering output cannot be a symlink or directory.")
    exported = export_geospatial_grid(grid, for_pygmt=True)
    pygmt = _optional_module("pygmt", grid.grid_id)
    limits = ResourceLimits(max_output_bytes, 16, 128, 128, 32)
    with pygmt.clib.Session() as session:
        gmt_version = session.info["version"]
    figure = pygmt.Figure()
    figure.grdimage(
        grid=exported.dataarray,
        region=list(exported.region),
        projection=projection,
        cmap=cmap,
        frame=frame,
        interpolation="n",
        nan_transparent=True,
    )
    with tempfile.TemporaryDirectory(prefix=".phydrax-render-", dir=parent) as temporary:
        rendered = Path(temporary) / ("grid" + path.suffix.lower())
        figure.savefig(str(rendered), dpi=dpi)
        resource = read_bounded_resource(rendered, trusted_root=temporary, limits=limits)
        _require_rendered_format(resource.data, path.suffix.lower())
        os.replace(rendered, path)
    manifest = read_bounded_resource(path, trusted_root=parent, limits=limits).manifest
    provenance = {
        "kind": "pygmt-geospatial-render",
        "grid_id": grid.grid_id,
        "export_id": exported.export_id,
        "contract": grid.contract.to_dict(),
        "source_manifest": asdict(grid.manifest),
        "resources": [asdict(item) for item in grid.resources],
        "pygmt_version": pygmt.__version__,
        "gmt_version": gmt_version,
        "projection": projection,
        "cmap": cmap,
        "frame": frame,
        "dpi": dpi,
        "interpolation": "nearest",
        "region": exported.region,
        "remote_download": False,
        "output_manifest": asdict(manifest),
    }
    render_id = canonical_fingerprint(provenance)
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        "qualified-geospatial-grid",
        "pygmt-rendered-figure",
        source_id=grid.grid_id,
        target_id=render_id,
        coordinate_mapping=(
            f"Display projection {projection}; native coordinates remain unchanged",
        ),
        preserved_fields=(
            "geospatial_provenance",
            "source_identity",
            "render_parameters",
            "output_identity",
        ),
        assumptions=(
            "Only the supplied in-memory grid is rendered; GMT_AUTO_DOWNLOAD=off",
        ),
        losses=(
            AdapterLoss(
                "representation",
                "export",
                "transformed",
                "Colors and display pixels are a visualization, not recoverable numerical grid values.",
                changes_interpretation=False,
            ),
        ),
        stages=(exported.report,),
    )
    return PyGMTRenderResult(
        figure, manifest, report, grid.grid_id, render_id, canonical_json(provenance)
    )


def _require_rendered_format(data: bytes, suffix: str) -> None:
    signature = {".png": b"\x89PNG\r\n\x1a\n", ".pdf": b"%PDF-", ".eps": b"%!PS-Adobe"}[
        suffix
    ]
    if not data.startswith(signature):
        raise ValueError("GMT did not produce the requested rendered file format.")


__all__ = [
    "GeospatialDependencyError",
    "PyGMTRenderResult",
    "XarrayGridExport",
    "export_geospatial_grid",
    "render_geospatial_grid",
]
