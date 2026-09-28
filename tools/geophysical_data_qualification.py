# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Run with: python tools/geophysical_data_qualification.py [--format zarr]."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from phydrax.applications.geophysics import GeophysicalFieldBinding, GeophysicalQuantity
from phydrax.applications.geophysics._archive import (
    read_geophysical_archive,
    write_geophysical_archive,
)
from phydrax.applications.geophysics._data import from_cf_dataset, read_cf, write_cf
from phydrax.dynamics import StateLayout
from phydrax.units import derived_unit, KELVIN, KILOGRAM, METER


def qualify(directory: Path, format: str) -> dict[str, object]:
    import xarray as xr  # Optional host boundary, never imported by compiled models.

    layout = StateLayout((2,), component_names=("temperature", "rain_amount"))
    bindings = {
        "temperature": GeophysicalFieldBinding(
            GeophysicalQuantity(
                "temperature", "temperature", KELVIN, axes=("time", "latitude")
            ),
            state_layout=layout,
            components=("temperature",),
        ),
        "rain": GeophysicalFieldBinding(
            GeophysicalQuantity(
                "rain_amount",
                "precipitation_amount",
                derived_unit("kg/m2", ((KILOGRAM, 1), (METER, -2))),
                axes=("time", "latitude"),
            ),
            state_layout=layout,
            components=("rain_amount",),
        ),
    }
    dataset = xr.Dataset(
        {
            "temperature": (
                ("time", "latitude"),
                np.array([[0, 100], [200, -999]], dtype=np.int16),
                {
                    "standard_name": "air_temperature",
                    "units": "degree_Celsius",
                    "scale_factor": 0.1,
                    "_FillValue": np.int16(-999),
                },
            ),
            "rain": (
                ("time", "latitude"),
                [[2.0, 4.0], [3.0, 5.0]],
                {
                    "standard_name": "precipitation_amount",
                    "units": "kg m-2",
                    "cell_methods": "time: sum",
                },
            ),
            "time_bounds": (("time", "bounds"), [[0.0, 1.0], [1.0, 2.0]]),
        },
        coords={
            "time": (
                "time",
                [1.0, 2.0],
                {
                    "standard_name": "time",
                    "units": "days since 2001-02-29",
                    "calendar": "360_day",
                    "bounds": "time_bounds",
                },
            ),
            "latitude": (
                "latitude",
                [-45.0, 45.0],
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
        },
        attrs={"Conventions": "CF-1.11", "source": "analytic qualification"},
    )
    data, report = from_cf_dataset(dataset, bindings=bindings, purpose="sampled-result")
    archive_path = directory / "native.zip"
    archive = write_geophysical_archive(
        archive_path,
        data,
        run_id="geophysical-data-qualification",
        evidence_ids=(report.report_id,),
    )
    restored = read_geophysical_archive(archive_path, bindings=bindings)
    assert restored.data_id == data.data_id
    destination = directory / ("weather.zarr" if format == "zarr" else "weather.nc")
    write_cf(restored, destination, format=format)
    reimported, _ = read_cf(destination, bindings=bindings, format=format)
    mask = data.valid("temperature")
    assert np.array_equal(reimported.valid("temperature"), mask)
    temperature_error = float(
        np.max(
            np.abs(
                reimported.values("temperature")[mask]
                - np.array([273.15, 283.15, 293.15])
            )
        )
    )
    rain_error = float(np.max(np.abs(reimported.values("rain") - dataset.rain.values)))
    assert temperature_error < 1.0e-12 and rain_error == 0.0
    assert reimported.descriptor["time"] == data.descriptor["time"]
    assert reimported.descriptor["fields"]["rain"]["temporal"]["kind"] == "accumulation"
    return {
        "format": format,
        "temperature_error_K": temperature_error,
        "accumulation_error_kg_m2": rain_error,
        "invalid_temperature_count": int(np.count_nonzero(~mask)),
        "calendar": data.descriptor["time"]["time"]["calendar"],
        "native_data_id": data.data_id,
        "archive_id": archive.archive_id,
        "archive_bytes": archive_path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qualify bounded CF -> native -> lifecycle archive -> CF scientific interchange."
    )
    parser.add_argument("--format", choices=("netcdf", "zarr"), default="netcdf")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        with TemporaryDirectory(prefix="phydrax-geophysical-data-") as temporary:
            result = qualify(Path(temporary), args.format)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        result = qualify(args.output, args.format)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
