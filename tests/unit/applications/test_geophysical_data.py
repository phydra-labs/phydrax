# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import copy

import numpy as np
import pytest

from phydrax._array_archive import read_array_archive, write_array_archive
from phydrax.applications.geophysics import (
    GeophysicalFieldBinding,
    GeophysicalQuantity,
    HybridPressureCoordinate,
    TemporalSupport,
)
from phydrax.applications.geophysics._archive import (
    read_geophysical_archive,
    write_geophysical_archive,
)
from phydrax.applications.geophysics._data import (
    _grib_to_cf,
    from_cf_dataset,
    GeophysicalData,
    read_cf,
    to_cf_dataset,
    write_cf,
)
from phydrax.dynamics import StateLayout
from phydrax.interchange import AdapterError, AdapterStatus
from phydrax.interchange._resource import ResourceLimits, ResourceReadError
from phydrax.lifecycle import query
from phydrax.units import (
    derived_unit,
    KELVIN,
    KILOGRAM,
    KILOPASCAL,
    METER,
    SECOND,
    UnitDefinition,
)


def _binding(
    kind="temperature",
    unit=KELVIN,
    *,
    axes=(),
    temporal=None,
    vertical=None,
    reference="absolute",
):
    layout = StateLayout((1,), component_names=("native_field",))
    quantity = GeophysicalQuantity(
        "native_field", kind, unit, axes=axes, reference_configuration=reference
    )
    return GeophysicalFieldBinding(
        quantity,
        state_layout=layout,
        components=("native_field",),
        temporal=temporal,
        vertical=vertical,
    )


def _packed_dataset():
    xr = pytest.importorskip("xarray")
    return xr.Dataset(
        {
            "tas": (
                ("time", "latitude"),
                np.array([[0, -999], [200, 300]], dtype=np.int16),
                {
                    "standard_name": "air_temperature",
                    "units": "degree_Celsius",
                    "scale_factor": 0.1,
                    "_FillValue": np.int16(-999),
                },
            ),
            "time_bounds": (("time", "bounds"), [[0.0, 1.0], [1.0, 2.0]]),
            "latitude_bounds": (("latitude", "bounds"), [[-90.0, 0.0], [0.0, 90.0]]),
        },
        coords={
            "time": (
                "time",
                [0.5, 1.5],
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
                {
                    "standard_name": "latitude",
                    "units": "degrees_north",
                    "bounds": "latitude_bounds",
                },
            ),
        },
        attrs={"Conventions": "CF-1.11", "source": "controlled analytic field"},
    )


def test_packed_masks_affine_units_calendar_and_native_archive_round_trip(tmp_path):
    binding = _binding(axes=("latitude", "time"))
    data, report = from_cf_dataset(_packed_dataset(), bindings={"tas": binding})
    assert report.status == AdapterStatus.DECLARED_LOSS
    np.testing.assert_array_equal(data.valid("tas"), [[True, True], [False, True]])
    np.testing.assert_allclose(
        data.values("tas"), [[273.15, 293.15], [np.nan, 303.15]], equal_nan=True
    )
    path = tmp_path / "native.zip"
    archive = write_geophysical_archive(path, data, run_id="analytic-source")
    restored = read_geophysical_archive(path, bindings={"tas": binding})
    assert restored.data_id == data.data_id
    assert restored.descriptor["purpose"] == "initialization"
    assert restored.descriptor["time"]["time"]["calendar"] == "360_day"
    np.testing.assert_array_equal(query(archive).fields[0].values, data.values("tas"))
    exported, _ = to_cf_dataset(restored)
    assert exported.tas.dims == ("latitude", "time")
    assert exported.tas.attrs["units"] == "K"
    assert exported.time.attrs["bounds"] == "time_bounds"
    np.testing.assert_array_equal(exported.time_bounds, [[0.0, 1.0], [1.0, 2.0]])
    assert np.isnan(exported.tas.values[1, 0])
    with pytest.raises(ValueError):
        restored.values("tas")[0, 0] = 1.0


def test_accumulations_keep_amounts_and_require_exact_interval_support():
    xr = pytest.importorskip("xarray")
    unit = derived_unit("kg/m2", ((KILOGRAM, 1), (METER, -2)))
    binding = _binding(
        "precipitation_amount",
        unit,
        temporal=TemporalSupport("accumulation", ((0.0, 6.0), (6.0, 12.0)), "end"),
    )
    dataset = xr.Dataset(
        {
            "rain": (
                "time",
                [6.0, 18.0],
                {
                    "standard_name": "precipitation_amount",
                    "units": "kg m-2",
                    "cell_methods": "time: sum",
                },
            ),
            "time_bounds": (("time", "bounds"), [[0.0, 6.0], [6.0, 12.0]]),
        },
        coords={
            "time": (
                "time",
                [6.0, 12.0],
                {
                    "standard_name": "time",
                    "units": "hours since 2000-02-28",
                    "calendar": "noleap",
                    "bounds": "time_bounds",
                },
            )
        },
    )
    data, _ = from_cf_dataset(dataset, bindings={"rain": binding})
    np.testing.assert_array_equal(data.values("rain"), [6.0, 18.0])
    exported, _ = to_cf_dataset(data)
    assert exported.rain.attrs["cell_methods"] == "time: sum"
    assert data.descriptor["fields"]["rain"]["temporal"]["position"] == "end"
    dataset["time"].attrs.pop("bounds")
    with pytest.raises(AdapterError):
        from_cf_dataset(dataset, bindings={"rain": binding})


def _hybrid_dataset():
    xr = pytest.importorskip("xarray")
    dataset = xr.Dataset(
        {
            "temperature": (
                ("latitude", "lev"),
                [[280.0, 290.0], [281.0, 291.0]],
                {"standard_name": "air_temperature", "units": "K"},
            ),
            "a": ("lev", [0.09375, 0.03125], {"units": "1", "bounds": "a_bounds"}),
            "b": ("lev", [0.25, 0.75], {"units": "1", "bounds": "b_bounds"}),
            "a_bounds": (
                ("lev", "bounds"),
                [[0.125, 0.0625], [0.0625, 0.0]],
                {"units": "1"},
            ),
            "b_bounds": (("lev", "bounds"), [[0.0, 0.5], [0.5, 1.0]], {"units": "1"}),
            "lev_bounds": (("lev", "bounds"), [[0.0, 0.5], [0.5, 1.0]]),
            "p0": ((), 100000.0, {"units": "Pa"}),
            "ps": (
                "latitude",
                [90000.0, 100000.0],
                {"standard_name": "surface_air_pressure", "units": "Pa"},
            ),
        },
        coords={
            "latitude": (
                "latitude",
                [-45.0, 45.0],
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "lev": (
                "lev",
                [0.25, 0.75],
                {
                    "standard_name": "atmosphere_hybrid_sigma_pressure_coordinate",
                    "units": "1",
                    "bounds": "lev_bounds",
                    "formula_terms": "a: a b: b ps: ps p0: p0",
                },
            ),
        },
    )
    coordinate = HybridPressureCoordinate([0.125, 0.0625, 0.0], [0.0, 0.5, 1.0])
    return dataset, _binding(axes=("latitude", "lev"), vertical=coordinate)


def test_hybrid_interfaces_are_derived_from_bounds_not_midpoint_coefficients(tmp_path):
    dataset, binding = _hybrid_dataset()
    data, _ = from_cf_dataset(dataset, bindings={"temperature": binding})
    record = data.descriptor["vertical"]["lev"]
    assert record["coordinate_id"] == binding.vertical_id
    np.testing.assert_array_equal(record["a"], [0.125, 0.0625, 0.0])
    path = tmp_path / "hybrid.zip"
    write_geophysical_archive(path, data, run_id="hybrid-column")
    assert read_geophysical_archive(path).data_id == data.data_id
    dataset["ps"].values[0] = 1000.0
    with pytest.raises(AdapterError):
        from_cf_dataset(dataset, bindings={"temperature": binding})
    dataset["ps"].values[0] = 90000.0
    dataset["a"].attrs.pop("bounds")
    with pytest.raises(AdapterError):
        from_cf_dataset(dataset, bindings={"temperature": binding})


def test_native_archive_rejects_semantic_and_payload_tampering(tmp_path):
    data, _ = from_cf_dataset(_packed_dataset(), bindings={"tas": _binding()})
    path = tmp_path / "native.zip"
    write_geophysical_archive(path, data, run_id="source")
    container, arrays = read_array_archive(path)
    changed = dict(arrays)
    changed["values/tas"] = changed["values/tas"] + 1.0
    write_array_archive(
        path,
        manifest={key: value for key, value in container.items() if key != "arrays"},
        arrays=changed,
    )
    _, tampered_arrays = read_array_archive(path)
    np.testing.assert_array_equal(tampered_arrays["values/tas"], changed["values/tas"])
    with pytest.raises(ValueError):
        read_geophysical_archive(path)
    descriptor = copy.deepcopy(data.descriptor)
    descriptor["time"]["time"]["calendar"] = "noleap"
    with pytest.raises(ValueError):
        GeophysicalData(descriptor, data.arrays)
    descriptor = copy.deepcopy(data.descriptor)
    descriptor["fields"]["tas"]["binding"]["role"] = "forcing"
    with pytest.raises(ValueError):
        GeophysicalData(descriptor, data.arrays)


@pytest.mark.parametrize("scalar", [False, True])
def test_time_support_follows_scalar_or_auxiliary_coordinate(scalar):
    xr = pytest.importorskip("xarray")
    unit = derived_unit("kg/m2", ((KILOGRAM, 1), (METER, -2)))
    dataset = xr.Dataset(
        {
            "rain": (
                "t",
                [6.0, 12.0],
                {
                    "standard_name": "precipitation_amount",
                    "units": "kg m-2",
                    "cell_methods": "time: sum",
                },
            ),
            "time_bounds": (("t", "bounds"), [[0.0, 6.0], [6.0, 12.0]]),
        },
        coords={
            "time": (
                "t",
                [6.0, 12.0],
                {
                    "standard_name": "time",
                    "units": "hours since 2000-01-01",
                    "bounds": "time_bounds",
                },
            )
        },
    )
    if scalar:
        dataset = dataset.isel(t=0)
    data, _ = from_cf_dataset(
        dataset, bindings={"rain": _binding("precipitation_amount", unit)}
    )
    support = data.descriptor["fields"]["rain"]["temporal"]
    assert support["kind"] == "accumulation"
    assert support["bounds"] == ([[0.0, 6.0]] if scalar else [[0.0, 6.0], [6.0, 12.0]])
    exported, _ = to_cf_dataset(data)
    np.testing.assert_array_equal(exported.rain.values, dataset.rain.values)


def test_native_custom_temperature_unit_exports_as_cf_reference_unit():
    milli = UnitDefinition(
        "application_temperature_tick",
        KELVIN.dimension,
        KELVIN.reference_system_id,
        (1, 1000),
    )
    bindings = {"tas": _binding(unit=milli)}
    data, _ = from_cf_dataset(_packed_dataset(), bindings=bindings)
    exported, report = to_cf_dataset(data)
    assert exported.tas.attrs["units"] == "K"
    assert report.status == AdapterStatus.DECLARED_LOSS
    np.testing.assert_allclose(
        exported.tas.values, [[273.15, np.nan], [293.15, 303.15]], equal_nan=True
    )
    restored, _ = from_cf_dataset(exported, bindings=bindings)
    np.testing.assert_allclose(restored.values("tas"), data.values("tas"), equal_nan=True)


def test_lwe_precipitation_rate_remains_volume_flux_not_mass_flux():
    xr = pytest.importorskip("xarray")
    unit = derived_unit("m/s", ((METER, 1), (SECOND, -1)))
    dataset = xr.Dataset(
        {
            "rain": (
                "x",
                [1.0e-5],
                {"standard_name": "lwe_precipitation_rate", "units": "m s-1"},
            )
        }
    )
    data, _ = from_cf_dataset(
        dataset, bindings={"rain": _binding("water_volume_flux", unit)}
    )
    np.testing.assert_array_equal(data.values("rain"), [1.0e-5])
    mass_unit = derived_unit("kg/m2/s", ((KILOGRAM, 1), (METER, -2), (SECOND, -1)))
    with pytest.raises(AdapterError):
        from_cf_dataset(
            dataset, bindings={"rain": _binding("precipitation_rate", mass_unit)}
        )


def test_bound_pressure_coordinate_converts_its_bounds_with_its_values():
    xr = pytest.importorskip("xarray")
    dataset = xr.Dataset(
        {"pressure_bounds": (("pressure", "bounds"), [[100.0, 300.0], [300.0, 900.0]])},
        coords={
            "pressure": (
                "pressure",
                [200.0, 600.0],
                {
                    "standard_name": "air_pressure",
                    "units": "hPa",
                    "bounds": "pressure_bounds",
                    "positive": "down",
                },
            )
        },
    )
    data, _ = from_cf_dataset(
        dataset, bindings={"pressure": _binding("pressure", KILOPASCAL)}
    )
    np.testing.assert_array_equal(data.values("pressure"), [20.0, 60.0])
    np.testing.assert_array_equal(
        data.values("pressure_bounds"), [[10.0, 30.0], [30.0, 90.0]]
    )
    exported, _ = to_cf_dataset(data)
    assert exported.pressure.attrs["units"] == "Pa"
    np.testing.assert_array_equal(
        exported.pressure_bounds, [[10000.0, 30000.0], [30000.0, 90000.0]]
    )


def test_directional_quantity_cannot_silently_reinterpret_wind_component():
    xr = pytest.importorskip("xarray")
    unit = derived_unit("m/s", ((METER, 1), (SECOND, -1)))
    dataset = xr.Dataset(
        {"wind": ("x", [4.0], {"standard_name": "eastward_wind", "units": "m s-1"})}
    )
    with pytest.raises(AdapterError):
        from_cf_dataset(
            dataset,
            bindings={"wind": _binding("velocity", unit, reference="northward_wind")},
        )
    data, _ = from_cf_dataset(
        dataset, bindings={"wind": _binding("velocity", unit, reference="eastward_wind")}
    )
    np.testing.assert_array_equal(data.values("wind"), [4.0])


def test_unknown_required_semantics_and_eager_limits_fail_closed():
    dataset = _packed_dataset()
    bindings = {"tas": _binding()}
    with pytest.raises(ResourceReadError):
        from_cf_dataset(
            dataset, bindings=bindings, limits=ResourceLimits(1, 32, 100, 1000, 100)
        )
    with pytest.raises(AdapterError) as raised:
        from_cf_dataset(
            dataset, bindings=bindings, required_semantics=("uninterpreted_projection",)
        )
    assert raised.value.status == AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
    dataset.tas.attrs["cell_methods"] = "area: mean time: sum"
    with pytest.raises(AdapterError):
        from_cf_dataset(dataset, bindings=bindings)
    with pytest.raises(AdapterError):
        from_cf_dataset(dataset, bindings=bindings, purpose="exact-restart")


@pytest.mark.parametrize(
    ("format", "dependency", "suffix"),
    [("netcdf", "scipy", ".nc"), ("zarr", "zarr", ".zarr")],
)
def test_actual_optional_cf_storage_round_trip(tmp_path, format, dependency, suffix):
    pytest.importorskip(dependency)
    bindings = {"tas": _binding()}
    data, _ = from_cf_dataset(_packed_dataset(), bindings=bindings)
    path = tmp_path / ("weather" + suffix)
    write_cf(data, path, format=format)
    restored, _ = read_cf(path, format=format, bindings=bindings)
    np.testing.assert_array_equal(restored.valid("tas"), data.valid("tas"))
    np.testing.assert_allclose(restored.values("tas"), data.values("tas"), equal_nan=True)
    assert restored.descriptor["time"] == data.descriptor["time"]
    assert restored.descriptor["provenance"]["resources"][0]["content_sha256"]


def test_actual_optional_grib_decoder(tmp_path):
    codes = pytest.importorskip("eccodes")
    pytest.importorskip("cfgrib")
    pytest.importorskip("xarray")
    handle = codes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    try:
        for key, value in {
            "Ni": 2,
            "Nj": 2,
            "latitudeOfFirstGridPointInDegrees": 1.0,
            "latitudeOfLastGridPointInDegrees": 0.0,
            "longitudeOfFirstGridPointInDegrees": 0.0,
            "longitudeOfLastGridPointInDegrees": 1.0,
            "iDirectionIncrementInDegrees": 1.0,
            "jDirectionIncrementInDegrees": 1.0,
            "shortName": "t",
            "dataDate": 20000101,
            "dataTime": 0,
            "step": 6,
        }.items():
            codes.codes_set(handle, key, value)
        codes.codes_set_values(handle, np.array([280.0, 281.0, 282.0, 283.0]))
        path = tmp_path / "temperature.grib"
        with path.open("wb") as stream:
            codes.codes_write(handle, stream)
    finally:
        codes.codes_release(handle)
    data, _ = read_cf(path, format="grib", bindings={"t": _binding()})
    np.testing.assert_allclose(data.values("t").reshape(-1), [280.0, 281.0, 282.0, 283.0])
    assert (
        data.descriptor["time"]["time"]["calendar"]
        == data.descriptor["time"]["forecast_reference_time"]["calendar"]
    )
    np.testing.assert_array_equal(
        data.values("time") - data.values("forecast_reference_time"), [21600.0]
    )
    np.testing.assert_array_equal(data.values("forecast_period"), 6.0)
    assert data.descriptor["fields"]["t"]["temporal"]["kind"] == "instantaneous"
    exported, _ = to_cf_dataset(data)
    assert exported.forecast_reference_time.dims == ()
    assert exported.forecast_period.dims == ()
    references = set(exported.t.attrs["coordinates"].split())
    assert {"time", "forecast_reference_time", "forecast_period"}.issubset(references)
    assert references.issubset(exported.variables)


def test_grib_temporal_dependencies_follow_semantics_and_reject_inconsistency():
    xr = pytest.importorskip("xarray")
    dataset = xr.Dataset(
        {
            "temperature": (
                "x",
                [280.0, 281.0],
                {
                    "standard_name": "air_temperature",
                    "units": "K",
                    "coordinates": "initial lead endpoint",
                    "GRIB_stepType": "avg",
                    "GRIB_startStep": 0,
                    "GRIB_endStep": 6,
                    "GRIB_stepUnits": 1,
                },
            ),
            "initial": (
                (),
                0.0,
                {
                    "standard_name": "forecast_reference_time",
                    "units": "hours since 2000-01-01",
                },
            ),
            "lead": (
                (),
                6.0,
                {"standard_name": "forecast_period", "units": "hours"},
            ),
            "endpoint": (
                (),
                21600.0,
                {"standard_name": "time", "units": "seconds since 2000-01-01"},
            ),
        }
    )
    bindings = {"temperature": _binding()}
    data, _ = from_cf_dataset(_grib_to_cf(dataset, bindings), bindings=bindings)
    np.testing.assert_array_equal(data.values("temperature"), [[280.0, 281.0]])
    np.testing.assert_array_equal(data.values("time"), [6.0])
    support = data.descriptor["fields"]["temperature"]["temporal"]
    assert support["kind"] == "mean"
    assert support["bounds"] == [[0.0, 6.0]]
    assert support["position"] == "end"
    exported, _ = to_cf_dataset(data)
    assert exported.forecast_reference_time.dims == ()
    assert exported.forecast_period.dims == ()
    assert set(exported.temperature.attrs["coordinates"].split()) == {
        "forecast_reference_time",
        "forecast_period",
        "time",
    }
    dataset["endpoint"].values[...] += 3600.0
    with pytest.raises(AdapterError) as raised:
        _grib_to_cf(dataset, bindings)
    assert raised.value.status == AdapterStatus.INCONSISTENT_SOURCE
    dataset["endpoint"].values[...] = 21600.0
    dataset.temperature.attrs["coordinates"] += " missing_coordinate"
    with pytest.raises(AdapterError) as raised:
        from_cf_dataset(_grib_to_cf(dataset, bindings), bindings=bindings)
    assert raised.value.status == AdapterStatus.INCONSISTENT_SOURCE
