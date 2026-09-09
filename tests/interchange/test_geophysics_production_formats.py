#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import hashlib
import struct
from pathlib import Path

import numpy as np
import pyproj
import pytest

import phydrax as phx
from phydrax.interchange import (
    BoreholeTrajectory,
    bounded_resource_from_bytes,
    CoordinateTransformPlan,
    decode_segy_rev2_bytes,
    ElectricalTabularSurvey,
    GeospatialContract,
    LeapSecondTable,
    read_cf_netcdf_grid,
    read_edi_impedance,
    read_electrical_survey_csv,
    read_emtf_xml_impedance,
    read_geotiff_grid,
    read_icgem_gfc,
    read_las_curve,
    read_miniseed3,
    read_sac,
    read_sinex_positions,
    read_stationxml,
    ResourceLimits,
    SEGYRev2IEEEProfile,
    TimeReferenceContract,
)
from phydrax.units import DEGREE, METER, PASCAL


def _limits(max_bytes=10_000_000):
    return ResourceLimits(max_bytes, 8, 100_000, 100_000, 100)


def _local(registration="pixel"):
    return GeospatialContract.local_cartesian(
        phx.SpatialCoordinateContract(METER, reference_frame="survey"),
        vertical_datum="local",
        registration=registration,
        mask_semantics="valid_true",
    )


def _geographic():
    return GeospatialContract(
        phx.SpatialCoordinateContract(METER, reference_frame="earth"),
        horizontal_crs="EPSG:4326",
        horizontal_kind="geographic",
        horizontal_axes=("longitude", "latitude"),
        horizontal_units=(DEGREE, DEGREE),
        horizontal_datum="WGS84",
        epoch_required=False,
        vertical_kind="ellipsoidal_height",
        vertical_positive="up",
        vertical_datum="WGS84-ellipsoid",
        vertical_unit=METER,
        registration="gridline",
        longitude_seam="none",
        longitude_domain=(-180, 180),
        mask_semantics="valid_true",
    )


def _segy_rev2_bytes():
    text = bytearray(b" " * 3200)
    binary = bytearray(400)
    struct.pack_into(">H", binary, 16, 2000)
    struct.pack_into(">H", binary, 20, 0)
    struct.pack_into(">H", binary, 24, 5)
    struct.pack_into(">H", binary, 52, 1)
    struct.pack_into(">H", binary, 54, 1)
    struct.pack_into(">H", binary, 56, 2)
    struct.pack_into(">H", binary, 300, 0x0200)
    struct.pack_into(">H", binary, 302, 0)
    struct.pack_into(">h", binary, 304, 0)
    records = []
    for row, values in enumerate(((1.0, 2.0), (3.0, 4.0, 5.0))):
        header = bytearray(240)
        struct.pack_into(">i", header, 4, row + 1)
        struct.pack_into(">h", header, 28, 1)
        struct.pack_into(">i", header, 40, 100)
        struct.pack_into(">i", header, 44, 120)
        struct.pack_into(">i", header, 48, 20)
        struct.pack_into(">h", header, 68, 1)
        struct.pack_into(">h", header, 70, 1)
        struct.pack_into(">i", header, 72, 1000)
        struct.pack_into(">i", header, 76, 2000)
        struct.pack_into(">i", header, 80, 1100)
        struct.pack_into(">i", header, 84, 2100)
        struct.pack_into(">h", header, 88, 1)
        struct.pack_into(">HH", header, 114, len(values), 2000 + row * 1000)
        struct.pack_into(">h", header, 202, 1)
        struct.pack_into(">h", header, 214, 1)
        records.append(bytes(header) + struct.pack(f">{len(values)}f", *values))
    return bytes(text + binary) + b"".join(records)


def test_explicit_proj_pipeline_and_variable_length_segy_rev2_are_qualified():
    source = _geographic()
    target = GeospatialContract(
        phx.SpatialCoordinateContract(METER, reference_frame="earth-fixed"),
        horizontal_crs="WGS84-geocentric",
        horizontal_kind="geocentric",
        horizontal_axes=("x", "y"),
        horizontal_units=(METER, METER),
        horizontal_datum="WGS84",
        epoch_required=False,
        vertical_kind="cartesian_z",
        vertical_positive="up",
        vertical_datum="earth-center",
        vertical_unit=METER,
        registration="unknown",
        longitude_seam="none",
        mask_semantics="none",
    )
    proj_database = Path(pyproj.datadir.get_data_dir()) / "proj.db"
    proj_database_sha256 = hashlib.sha256(proj_database.read_bytes()).hexdigest()
    transform = CoordinateTransformPlan(
        source,
        target,
        "+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad "
        "+step +proj=cart +ellps=WGS84",
        expected_pyproj_version=pyproj.__version__,
        expected_proj_version=pyproj.proj_version_str,
        expected_resource_sha256={"proj.db": proj_database_sha256},
        source_bounds=((-180, 180), (-90, 90), (-1000, 100000)),
        maximum_resource_bytes=100_000_000,
    ).execute([[0.0, 0.0, 0.0]])
    np.testing.assert_allclose(transform.coordinates[0], [6378137.0, 0.0, 0.0], atol=1e-6)
    assert not transform.transform.parameters["network_enabled"]
    assert transform.transform.resources[0].content_sha256 == proj_database_sha256
    with pytest.raises(ValueError, match="pinned SHA-256"):
        CoordinateTransformPlan(
            source,
            target,
            "+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad "
            "+step +proj=cart +ellps=WGS84",
            expected_pyproj_version=pyproj.__version__,
            expected_proj_version=pyproj.proj_version_str,
            expected_resource_sha256={"proj.db": "0" * 64},
            source_bounds=((-180, 180), (-90, 90), (-1000, 100000)),
            maximum_resource_bytes=100_000_000,
        ).execute([[0.0, 0.0, 0.0]])

    decoded = decode_segy_rev2_bytes(
        _segy_rev2_bytes(),
        profile=SEGYRev2IEEEProfile(
            byte_order="big",
            text_encoding="ascii",
            trace_length="variable",
            pressure_polarity="positive",
        ),
        limits=_limits(),
    )
    assert decoded.series.values.shape == (2, 3)
    np.testing.assert_array_equal(
        decoded.series.sample_valid, [[True, True, False], [True] * 3]
    )
    np.testing.assert_allclose(decoded.sample_intervals, [0.002, 0.003])


def test_electrical_mt_gravity_and_geodetic_text_profiles(tmp_path):
    electrical = tmp_path / "survey.csv"
    electrical.write_text(
        "source_a_x,source_a_y,source_a_z,source_b_x,source_b_y,source_b_z,"
        "receiver_m_x,receiver_m_y,receiver_m_z,receiver_n_x,receiver_n_y,receiver_n_z,"
        "current,voltage,standard_deviation\n"
        "0,0,0,1,0,0,0,1,0,1,1,0,1,0.2,0.01\n"
    )
    survey = read_electrical_survey_csv(
        electrical.name,
        trusted_root=tmp_path,
        limits=_limits(),
        coordinates=_local(registration="unknown"),
    )
    assert isinstance(survey, ElectricalTabularSurvey)
    np.testing.assert_allclose(survey.voltage_V, [0.2])

    edi = tmp_path / "site.edi"
    edi.write_text(
        ">HEAD\nLAT=45:0:0\nLONG=-120:0:0\nELEV=100\n"
        ">FREQ // 2\n1 10\n>ZROT // 1\n0\n"
        ">ZXXR // 2\n1 2\n>ZXXI // 2\n-1 -2\n>ZXX.VAR // 2\n0.01 0.04\n"
        ">ZXYR // 2\n3 4\n>ZXYI // 2\n-3 -4\n>ZXY.VAR // 2\n0.01 0.04\n"
        ">ZYXR // 2\n-3 -4\n>ZYXI // 2\n3 4\n>ZYX.VAR // 2\n0.01 0.04\n"
        ">ZYYR // 2\n1 2\n>ZYYI // 2\n-1 -2\n>ZYY.VAR // 2\n0.01 0.04\n"
    )
    mt = read_edi_impedance(
        edi.name,
        trusted_root=tmp_path,
        limits=_limits(),
        coordinates=_geographic(),
        impedance_scale_ohm=1.0,
    )
    assert mt.impedance_ohm.shape == (2, 2, 2)

    xml = tmp_path / "site.xml"
    xml.write_text(
        "<EM_TF><Site><Location><Latitude>45</Latitude><Longitude>-120</Longitude>"
        "<Elevation>100</Elevation></Location></Site><Data><Period value='1'>"
        + "".join(
            f"<Z output='{out}' input='{inp}'><Real>1</Real><Imaginary>-1</Imaginary>"
            f"<Error>0.1</Error></Z>"
            for out, inp in (("Ex", "Hx"), ("Ex", "Hy"), ("Ey", "Hx"), ("Ey", "Hy"))
        )
        + "</Period></Data></EM_TF>"
    )
    mt_xml = read_emtf_xml_impedance(
        xml.name,
        trusted_root=tmp_path,
        limits=_limits(),
        coordinates=_geographic(),
    )
    np.testing.assert_allclose(mt_xml.frequencies_Hz, [1.0])

    gfc = tmp_path / "model.gfc"
    gfc.write_text(
        "modelname TEST\nearth_gravity_constant 4e14\nradius 6400000\nmax_degree 1\n"
        "norm fully_normalized\ntide_system tide_free\nend_of_head\n"
        "gfc 0 0 1 0 0 0\ngfc 1 0 0 0 0 0\ngfc 1 1 0 0 0 0\n"
    )
    model = read_icgem_gfc(gfc.name, trusted_root=tmp_path, limits=_limits())
    assert model.maximum_degree == 1
    np.testing.assert_allclose(model.cosine[0, 0], 1.0)

    sinex = tmp_path / "solution.snx"
    sinex.write_text(
        "%=SNX 2.02 TEST 24:001:00000 TEST 24:001:00000 24:001:00000 P 00001 1 S\n"
        "+SOLUTION/ESTIMATE\n"
        " 1 STAX ABCD A 0001 24:001:00000 m 2 1.0 0.1\n"
        " 2 STAY ABCD A 0001 24:001:00000 m 2 2.0 0.1\n"
        " 3 STAZ ABCD A 0001 24:001:00000 m 2 3.0 0.1\n"
        "-SOLUTION/ESTIMATE\n"
    )
    positions = read_sinex_positions(
        sinex.name,
        trusted_root=tmp_path,
        limits=_limits(),
        frame_id="ITRF-test",
    )
    np.testing.assert_allclose(positions.ecef_m, [[1.0, 2.0, 3.0]])


def test_geotiff_netcdf_sac_stationxml_and_las_profiles_execute(tmp_path):
    import obspy
    import rasterio
    import xarray as xr
    from obspy.core.inventory import Channel, Inventory, Network, Site, Station
    from rasterio.transform import from_origin

    local = _local()
    geotiff = tmp_path / "grid.tif"
    with rasterio.open(
        geotiff,
        "w",
        driver="GTiff",
        height=2,
        width=3,
        count=1,
        dtype="float64",
        crs="EPSG:32610",
        transform=from_origin(0, 2, 1, 1),
    ) as dataset:
        dataset.write(np.asarray([[1, 2, 3], [4, 5, 6]], dtype=float), 1)
    projected = GeospatialContract(
        local.spatial,
        horizontal_crs="EPSG:32610",
        horizontal_kind="projected",
        horizontal_axes=("east", "north"),
        horizontal_units=(METER, METER),
        horizontal_datum="WGS84",
        epoch_required=False,
        vertical_kind="elevation",
        vertical_positive="up",
        vertical_datum="local",
        vertical_unit=METER,
        registration="pixel",
        longitude_seam="none",
        mask_semantics="valid_true",
    )
    raster = read_geotiff_grid(
        geotiff.name,
        trusted_root=tmp_path,
        limits=_limits(),
        contract=projected,
        value_unit=METER,
        expected_crs="EPSG:32610",
    )
    np.testing.assert_allclose(raster.values, [[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="resource limits"):
        read_geotiff_grid(
            geotiff.name,
            trusted_root=tmp_path,
            limits=ResourceLimits(10_000_000, 8, 5, 100_000, 100),
            contract=projected,
            value_unit=METER,
            expected_crs="EPSG:32610",
        )

    netcdf = tmp_path / "grid.nc"
    xr.Dataset(
        {"z": (("y", "x"), np.asarray([[1.0, 2.0], [3.0, 4.0]]))},
        coords={"x": [0.5, 1.5], "y": [1.5, 0.5]},
    ).to_netcdf(netcdf)
    grid = read_cf_netcdf_grid(
        netcdf.name,
        trusted_root=tmp_path,
        limits=_limits(),
        contract=local,
        value_unit=METER,
        variable="z",
        x_name="x",
        y_name="y",
    )
    np.testing.assert_allclose(grid.values, [[1, 2], [3, 4]])

    leap_resource = bounded_resource_from_bytes(
        b"leap", limits=ResourceLimits(100, 1, 1, 1, 0)
    ).manifest
    utc = TimeReferenceContract(
        "utc",
        "unix",
        10.0,
        epoch_nominal_seconds=0.0,
        leap_seconds=LeapSecondTable([], [], 10.0, leap_resource),
    )
    trace = obspy.Trace(np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    trace.stats.starttime = obspy.UTCDateTime(0)
    trace.stats.delta = 0.1
    trace.stats.network, trace.stats.station, trace.stats.channel = "XX", "AAA", "BHZ"
    sac = tmp_path / "trace.sac"
    trace.write(str(sac), format="SAC")
    waveform = read_sac(
        sac.name,
        trusted_root=tmp_path,
        limits=_limits(),
        utc_time=utc,
        sample_unit=PASCAL,
    )
    np.testing.assert_allclose(waveform.traces[0].series.values, [1, 2, 3])
    with pytest.raises(ValueError, match="decoded-sample limit"):
        read_sac(
            sac.name,
            trusted_root=tmp_path,
            limits=ResourceLimits(10_000_000, 8, 2, 100_000, 100),
            utc_time=utc,
            sample_unit=PASCAL,
        )
    import pymseed

    mini = pymseed.MS3TraceList()
    mini.add_data(
        "FDSN:XX_AAA__B_H_Z",
        np.asarray([4.0, 5.0, 6.0], dtype=np.float32),
        "f",
        10.0,
        starttime_seconds=0.0,
    )
    miniseed = tmp_path / "trace.mseed3"
    mini.to_file(
        miniseed,
        overwrite=True,
        encoding=pymseed.DataEncoding.FLOAT32,
        format_version=3,
    )
    mini.close()
    waveform3 = read_miniseed3(
        miniseed.name,
        trusted_root=tmp_path,
        limits=_limits(),
        utc_time=utc,
        sample_unit=PASCAL,
    )
    assert waveform3.format == "miniseed3"
    assert waveform3.traces[0].channel == "BHZ"
    np.testing.assert_allclose(waveform3.traces[0].series.values, [4, 5, 6])

    channel = Channel(
        code="BHZ",
        location_code="",
        latitude=45,
        longitude=-120,
        elevation=100,
        depth=1,
        azimuth=0,
        dip=-90,
        sample_rate=10,
    )
    station = Station(
        code="AAA",
        latitude=45,
        longitude=-120,
        elevation=100,
        creation_date=obspy.UTCDateTime(0),
        site=Site(name="site"),
        channels=[channel],
    )
    inventory = Inventory([Network(code="XX", stations=[station])], source="test")
    stationxml = tmp_path / "stations.xml"
    inventory.write(stationxml, format="STATIONXML")
    metadata = read_stationxml(
        stationxml.name,
        trusted_root=tmp_path,
        limits=_limits(),
        coordinates=_geographic(),
    )
    np.testing.assert_allclose(metadata.channels[0].orientation, [0, 0, 1], atol=1e-12)

    trajectory = BoreholeTrajectory(
        "well", [0.0, 10.0], [[0, 0, 0], [0, 0, -10]], _local(registration="unknown")
    )
    las = tmp_path / "well.las"
    las.write_text(
        "~Version\nVERS. 2.0\nWRAP. NO\n~Well\nSTRT.M 0\nSTOP.M 10\nSTEP.M 5\nNULL. -999.25\n"
        "~Curve\nDEPT.M : Depth\nRHOB.KG/M3 : Density\n~ASCII\n0 2000\n5 2100\n10 2200\n"
    )
    log = read_las_curve(
        las.name,
        trusted_root=tmp_path,
        limits=_limits(),
        trajectory=trajectory,
        mnemonic="RHOB",
        measured_depth_unit=METER,
        expected_index_unit_label="M",
    )
    np.testing.assert_allclose(log.values, [2000, 2100, 2200])
