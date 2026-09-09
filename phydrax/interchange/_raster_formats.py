#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from importlib import import_module
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast, Literal

import numpy as np

from ..units import UnitDefinition
from ._geospatial import GeospatialContract, QualifiedGeospatialGrid
from ._resource import read_bounded_resource, ResourceLimits


class GeospatialFormatDependencyError(RuntimeError):
    """An optional bounded raster/grid decoder is unavailable."""


def _rasterio():
    try:
        return cast(Any, import_module("rasterio"))
    except ImportError as error:
        raise GeospatialFormatDependencyError(
            "GeoTIFF decoding requires the optional rasterio runtime."
        ) from error


def _xarray():
    try:
        return cast(Any, import_module("xarray"))
    except ImportError as error:
        raise GeospatialFormatDependencyError(
            "CF-NetCDF and consolidated ZIP-Zarr decoding require xarray."
        ) from error


def read_geotiff_grid(
    path: str | Path,
    /,
    *,
    trusted_root: str | Path,
    limits: ResourceLimits,
    contract: GeospatialContract,
    value_unit: UnitDefinition,
    band: int = 1,
    expected_crs: str,
    name: str = "z",
    value_role: Literal["scalar", "vertical"] = "scalar",
) -> QualifiedGeospatialGrid:
    if contract.registration != "pixel":
        raise ValueError("GeoTIFF raster samples require a pixel-registered contract.")
    resource = read_bounded_resource(path, trusted_root=trusted_root, limits=limits)
    rasterio = _rasterio()
    with rasterio.io.MemoryFile(resource.data) as memory:
        with memory.open() as dataset:
            if dataset.count < band or band < 1:
                raise ValueError("Requested GeoTIFF band is absent.")
            if dataset.crs is None:
                raise ValueError(
                    "GeoTIFF has no CRS and cannot be geospatially qualified."
                )
            source_crs = dataset.crs.to_string()
            if source_crs != expected_crs or contract.horizontal_crs != expected_crs:
                raise ValueError(
                    "GeoTIFF CRS does not exactly match the declared contract."
                )
            transform = dataset.transform
            if (
                transform.b != 0
                or transform.d != 0
                or transform.a == 0
                or transform.e == 0
            ):
                raise ValueError(
                    "Qualified GeoTIFF grid requires axis-aligned nondegenerate pixels."
                )
            decoded_nodes = int(
                dataset.height * dataset.width + dataset.height + dataset.width
            )
            attribute_count = len(dataset.tags()) + len(dataset.tags(band))
            if (
                decoded_nodes > limits.max_nodes
                or attribute_count > limits.max_attributes
            ):
                raise ValueError("GeoTIFF decoded grid exceeds its resource limits.")
            values = dataset.read(band, masked=True)
            data = np.asarray(values.filled(0))
            valid = ~np.ma.getmaskarray(values)
            x = transform.c + (np.arange(dataset.width) + 0.5) * transform.a
            y = transform.f + (np.arange(dataset.height) + 0.5) * transform.e
    return QualifiedGeospatialGrid(
        x,
        y,
        data,
        contract,
        value_unit=value_unit,
        name=name,
        valid=valid,
        value_role=value_role,
        resources=(resource.manifest,),
    )


def _grid_from_dataset(
    dataset,
    *,
    resource,
    contract,
    value_unit,
    variable,
    x_name,
    y_name,
    name,
    value_role,
    limits,
):
    if (
        variable not in dataset
        or x_name not in dataset.coords
        or y_name not in dataset.coords
    ):
        raise ValueError("Declared grid variable or coordinate is absent.")
    values = dataset[variable]
    selected = (values, dataset.coords[x_name], dataset.coords[y_name])
    decoded_nodes = sum(int(item.size) for item in selected)
    attribute_count = len(dataset.attrs) + sum(len(item.attrs) for item in selected)
    if (
        decoded_nodes > limits.max_nodes
        or attribute_count > limits.max_attributes
        or not np.issubdtype(values.dtype, np.number)
        or np.issubdtype(values.dtype, np.complexfloating)
    ):
        raise ValueError("Selected grid exceeds resource limits or is not real numeric.")
    if tuple(values.dims) != (y_name, x_name):
        raise ValueError(
            "Grid variable dimensions must be exactly declared (y, x) order."
        )
    raw = np.asarray(values.values)
    valid = np.isfinite(raw)
    fill = values.attrs.get("_FillValue", values.encoding.get("_FillValue"))
    if fill is not None:
        valid &= raw != fill
    data = np.where(valid, raw, 0)
    return QualifiedGeospatialGrid(
        np.asarray(dataset.coords[x_name].values),
        np.asarray(dataset.coords[y_name].values),
        data,
        contract,
        value_unit=value_unit,
        name=name,
        valid=valid,
        value_role=value_role,
        resources=(resource.manifest,),
    )


def read_cf_netcdf_grid(
    path: str | Path,
    /,
    *,
    trusted_root: str | Path,
    limits: ResourceLimits,
    contract: GeospatialContract,
    value_unit: UnitDefinition,
    variable: str,
    x_name: str,
    y_name: str,
    name: str = "z",
    value_role: Literal["scalar", "vertical"] = "scalar",
) -> QualifiedGeospatialGrid:
    resource = read_bounded_resource(path, trusted_root=trusted_root, limits=limits)
    xarray = _xarray()
    with xarray.open_dataset(BytesIO(resource.data)) as dataset:
        return _grid_from_dataset(
            dataset,
            resource=resource,
            contract=contract,
            value_unit=value_unit,
            variable=variable,
            x_name=x_name,
            y_name=y_name,
            name=name,
            value_role=value_role,
            limits=limits,
        )


def read_consolidated_zip_zarr_grid(
    path: str | Path,
    /,
    *,
    trusted_root: str | Path,
    limits: ResourceLimits,
    contract: GeospatialContract,
    value_unit: UnitDefinition,
    variable: str,
    x_name: str,
    y_name: str,
    name: str = "z",
    value_role: Literal["scalar", "vertical"] = "scalar",
) -> QualifiedGeospatialGrid:
    resource = read_bounded_resource(path, trusted_root=trusted_root, limits=limits)
    xarray = _xarray()
    try:
        zarr = cast(Any, import_module("zarr"))
    except ImportError as error:
        raise GeospatialFormatDependencyError(
            "Consolidated ZIP-Zarr decoding requires the optional zarr runtime."
        ) from error
    with TemporaryDirectory(prefix="phydrax-zarr-") as temporary:
        archive = Path(temporary) / "grid.zarr.zip"
        archive.write_bytes(resource.data)
        store = zarr.storage.ZipStore(archive, mode="r")
        try:
            with xarray.open_zarr(store, consolidated=True) as dataset:
                return _grid_from_dataset(
                    dataset,
                    resource=resource,
                    contract=contract,
                    value_unit=value_unit,
                    variable=variable,
                    x_name=x_name,
                    y_name=y_name,
                    name=name,
                    value_role=value_role,
                    limits=limits,
                )
        finally:
            store.close()


__all__ = [
    "GeospatialFormatDependencyError",
    "read_cf_netcdf_grid",
    "read_consolidated_zip_zarr_grid",
    "read_geotiff_grid",
]
