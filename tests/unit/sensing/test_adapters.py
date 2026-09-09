from hashlib import sha256

import numpy as np
import pytest

import phydrax as phx


def _manifest():
    return phx.qualification.ReferenceArtifactManifest(
        "sensor-source",
        checksum_algorithm="sha256",
        checksum="7" * 64,
        size_bytes=1,
        license_id="synthetic",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="public",
        nondimensionalization={"signal": 1.0},
        uncertainty={"signal": 0.1},
        lineage_ids=("synthetic",),
    )


def test_normalized_ros_records_lower_to_collection_and_frame_graph():
    profiles = (
        phx.sensing.RosTopicProfile(
            "/scan", "LaserScan", phx.sensing.RosMessageKind.LASER_SCAN, "sensor"
        ),
        phx.sensing.RosTopicProfile(
            "/points",
            "PointCloud2",
            phx.sensing.RosMessageKind.POINT_CLOUD,
            "sensor",
        ),
        phx.sensing.RosTopicProfile(
            "/tf", "Transform", phx.sensing.RosMessageKind.TRANSFORM, "world"
        ),
    )
    records = (
        phx.sensing.RosMessageRecord(
            "/scan",
            phx.sensing.RosMessageKind.LASER_SCAN,
            0.0,
            0.0,
            0,
            {
                "ranges": np.asarray((1.0, 2.0)),
                "angles": np.asarray((0.0, 1.0)),
                "range_min": 0.1,
                "range_max": 10.0,
            },
        ),
        phx.sensing.RosMessageRecord(
            "/points",
            phx.sensing.RosMessageKind.POINT_CLOUD,
            0.0,
            0.05,
            1,
            {"points": np.asarray(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))},
        ),
        phx.sensing.RosMessageRecord(
            "/tf",
            phx.sensing.RosMessageKind.TRANSFORM,
            0.0,
            0.1,
            2,
            {
                "source_frame": "sensor",
                "target_frame": "world",
                "rotation": np.eye(3),
                "translation": np.zeros(3),
            },
        ),
        phx.sensing.RosMessageRecord(
            "/tf",
            phx.sensing.RosMessageKind.TRANSFORM,
            1.0,
            0.2,
            3,
            {
                "source_frame": "sensor",
                "target_frame": "world",
                "rotation": np.eye(3),
                "translation": np.asarray((1.0, 0.0, 0.0)),
            },
        ),
    )
    result = phx.sensing.RosbagImportPlan(profiles).lower(
        records, _manifest(), campaign_id="robot-run"
    )
    assert len(result.collection.assets) == 2
    _, translation, evidence = result.frame_graph.prepare_route(
        "sensor", "world"
    ).evaluate(0.5)
    np.testing.assert_allclose(translation, (0.5, 0.0, 0.0))
    assert bool(evidence.successful)


def test_normalized_ros_image_camera_imu_odometry_and_joint_profiles():
    kinds = (
        ("/image", phx.sensing.RosMessageKind.IMAGE),
        ("/camera", phx.sensing.RosMessageKind.CAMERA_INFO),
        ("/imu", phx.sensing.RosMessageKind.IMU),
        ("/odom", phx.sensing.RosMessageKind.ODOMETRY),
        ("/joints", phx.sensing.RosMessageKind.JOINT_STATE),
    )
    profiles = tuple(
        phx.sensing.RosTopicProfile(topic, kind.value, kind, "sensor")
        for topic, kind in kinds
    )
    payloads = (
        {"values": np.ones((2, 2)), "encoding": "mono8"},
        {"intrinsic": np.eye(3), "distortion": np.zeros(5)},
        {
            "angular_velocity": np.ones(3),
            "linear_acceleration": np.ones(3),
        },
        {"position": np.ones(3), "quaternion": np.asarray((1.0, 0.0, 0.0, 0.0))},
        {
            "names": ("joint",),
            "positions": np.asarray((1.0,)),
            "velocities": np.asarray((0.5,)),
        },
    )
    records = tuple(
        phx.sensing.RosMessageRecord(
            topic,
            kind,
            float(index),
            float(index),
            index,
            payload,
        )
        for index, ((topic, kind), payload) in enumerate(
            zip(kinds, payloads, strict=True)
        )
    )
    result = phx.sensing.RosbagImportPlan(profiles).lower(
        records,
        _manifest(),
        campaign_id="multisensor-ros",
    )
    names = {asset.field.quantity.name for asset in result.collection.assets}
    assert {
        "values",
        "intrinsic",
        "distortion",
        "angular-velocity",
        "linear-acceleration",
        "position",
        "quaternion",
        "positions",
        "velocities",
    } <= names


def test_e57_collection_preserves_scan_pose_invalidity_and_image_links():
    reference = _manifest()
    contract = phx.SpatialCoordinateContract(
        phx.units.METER,
        coordinate_system="cartesian",
        reference_frame="scanner",
    )
    support = phx.measurement.PointSampleSupport(
        np.asarray(((1.0, 2.0, 3.0), (0.0, 0.0, 0.0))),
        ("point-0", "point-1"),
        contract,
        active_mask=np.asarray((True, False)),
    )
    derivation = phx.measurement.DerivationRecord(
        phx.measurement.DataOrigin.EXTERNAL,
        phx.measurement.DataStage.DERIVED,
        transformation_id="e57-lowering",
    )
    product = phx.measurement.LidarPointProduct(
        support,
        (),
        derivation,
        "e57-scan",
        (reference,),
    )
    anchor_field = phx.measurement.QuantityField(
        "scan-image-link",
        phx.measurement.QuantitySpec(
            "e57",
            "embedded-image-link",
            "embedded-image-link",
            phx.units.ONE,
            "e57.image-link",
        ),
        phx.measurement.ValueLayout(phx.measurement.ValueKind.COUNT),
        support,
        phx.measurement.SamplingSemantics(phx.measurement.SpatialSamplingKind.POINT),
        np.asarray((1, 0)),
        np.asarray((True, False)),
    )
    anchor = phx.measurement.MeasurementAsset.from_single_reference(
        "embedded-image",
        anchor_field,
        reference,
        derivation,
    )
    collection = phx.measurement.MeasurementCollection(
        "e57-campaign",
        "e57-campaign",
        (anchor,),
        (
            phx.measurement.MeasurementRoleAssignment(
                anchor.asset_id, phx.measurement.MeasurementRole.REFERENCE
            ),
        ),
    )
    report = phx.interchange.AdapterReport(
        phx.interchange.AdapterStatus.LOSSLESS,
        "E57",
        "phydrax-e57",
        source_id=reference.manifest_id,
        target_id=collection.content_id,
    )
    scan = phx.sensing.E57ScanRecord(
        "scan-guid",
        product,
        phx.geometry.RigidFrame(np.eye(3), np.asarray((1.0, 0.0, 0.0))),
        np.asarray((0, 0)),
        np.asarray((0, 1)),
        (anchor.asset_id,),
    )
    result = phx.sensing.E57ScanCollection(collection, (scan,), report)
    assert result.scans[0].image_asset_ids == (anchor.asset_id,)
    assert not result.scans[0].points.support.active_mask[1]


def test_actual_e57_and_cfradial_adapters_are_bounded(tmp_path):
    pye57 = pytest.importorskip("pye57")
    source = tmp_path / "scan.e57"
    writer = pye57.E57(str(source), mode="w")
    writer.write_scan_raw(
        {
            "cartesianX": np.asarray((1.0, 2.0)),
            "cartesianY": np.asarray((3.0, 4.0)),
            "cartesianZ": np.asarray((5.0, 6.0)),
            "intensity": np.asarray((0.2, 0.4)),
        },
        name="scan",
        rotation=np.asarray((1.0, 0.0, 0.0, 0.0)),
        translation=np.asarray((1.0, 0.0, 0.0)),
    )
    writer.close()
    reference = phx.qualification.ReferenceArtifactManifest(
        "actual-e57",
        checksum_algorithm="sha256",
        checksum=sha256(source.read_bytes()).hexdigest(),
        size_bytes=source.stat().st_size,
        license_id="synthetic",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="public",
        nondimensionalization={"length": 1.0},
        uncertainty={"position": 0.0},
        lineage_ids=("synthetic",),
    )
    contract = phx.SpatialCoordinateContract(
        phx.units.METER, coordinate_system="cartesian", reference_frame="scanner"
    )
    imported = phx.sensing.E57Provider().read(
        source,
        reference,
        contract,
        campaign_id="actual-e57",
        maximum_points_per_scan=4,
    )
    assert len(imported.scans) == 1
    np.testing.assert_allclose(
        imported.scans[0].sensor_to_collection.translation,
        (1.0, 0.0, 0.0),
    )

    xarray = pytest.importorskip("xarray")
    radar_source = tmp_path / "radar.nc"
    dataset = xarray.Dataset(
        data_vars={
            "reflectivity": (("ray", "range"), np.asarray(((1.0, 2.0), (3.0, 4.0)))),
        },
        coords={
            "azimuth": (("ray",), np.asarray((0.0, 1.0))),
            "elevation": (("ray",), np.asarray((0.1, 0.1))),
            "range": (("range",), np.asarray((100.0, 200.0))),
        },
        attrs={"instrument_name": "synthetic-radar"},
    )
    dataset.to_netcdf(radar_source)
    dataset.close()
    radar_reference = phx.qualification.ReferenceArtifactManifest(
        "actual-cfradial",
        checksum_algorithm="sha256",
        checksum=sha256(radar_source.read_bytes()).hexdigest(),
        size_bytes=radar_source.stat().st_size,
        license_id="synthetic",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="public",
        nondimensionalization={"signal": 1.0},
        uncertainty={"reflectivity": 0.0},
        lineage_ids=("synthetic",),
    )
    radar = phx.sensing.CfRadialProvider().read(
        radar_source,
        radar_reference,
        campaign_id="actual-radar",
        quantity_units={"reflectivity": phx.units.ONE},
        maximum_gates=8,
    )
    assert radar.assets[0].field.values.shape == (2, 2)


def test_actual_rosbag_laser_scan_admission(tmp_path):
    rosbags = pytest.importorskip("rosbags")
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import get_typestore, Stores

    del rosbags
    typestore = get_typestore(Stores.ROS2_JAZZY)
    time_type = typestore.types["builtin_interfaces/msg/Time"]
    header_type = typestore.types["std_msgs/msg/Header"]
    scan_type = typestore.types["sensor_msgs/msg/LaserScan"]
    message_type = "sensor_msgs/msg/LaserScan"
    message = scan_type(
        header_type(time_type(0, 0), "sensor"),
        0.0,
        0.5,
        0.5,
        0.0,
        0.1,
        0.1,
        10.0,
        np.asarray((1.0, 2.0), dtype=np.float32),
        np.asarray((), dtype=np.float32),
    )
    source = tmp_path / "bag"
    with Writer(source, version=9) as writer:
        connection = writer.add_connection("/scan", message_type, typestore=typestore)
        writer.write(
            connection,
            0,
            typestore.serialize_cdr(message, message_type),
        )
    members = tuple(sorted(value for value in source.rglob("*") if value.is_file()))
    digest = sha256()
    for member in members:
        relative = str(member.relative_to(source)).encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(member.read_bytes())
    reference = phx.qualification.ReferenceArtifactManifest(
        "actual-rosbag",
        checksum_algorithm="sha256",
        checksum=digest.hexdigest(),
        size_bytes=sum(value.stat().st_size for value in members),
        license_id="synthetic",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="public",
        nondimensionalization={"range": 1.0},
        uncertainty={"range": 0.0},
        lineage_ids=("synthetic",),
    )
    profile = phx.sensing.RosTopicProfile(
        "/scan",
        message_type,
        phx.sensing.RosMessageKind.LASER_SCAN,
        "sensor",
    )
    result = phx.sensing.RosbagImportPlan((profile,)).read(
        source, reference, campaign_id="actual-ros"
    )
    np.testing.assert_allclose(result.collection.assets[0].field.values, (1.0, 2.0))


def test_actual_xtf_side_scan_admission(tmp_path):
    import ctypes

    pyxtf = pytest.importorskip("pyxtf")
    header = pyxtf.XTFFileHeader()
    header.NumberOfSonarChannels = 1
    header.ChanInfo[0].TypeOfChannel = pyxtf.XTFChannelType.port.value
    header.ChanInfo[0].BytesPerSample = 2
    header.ChanInfo[0].SampleFormat = 3
    ping = pyxtf.XTFPingHeader()
    ping.NumChansToFollow = 1
    channel = pyxtf.XTFPingChanHeader()
    channel.ChannelNumber = 0
    channel.NumSamples = 4
    samples = np.asarray((1, 2, 3, 4), dtype=np.uint16)
    ping.ping_chan_headers = [channel]
    ping.data = [samples]
    ping.NumBytesThisRecord = (
        ctypes.sizeof(ping) + ctypes.sizeof(channel) + samples.nbytes
    )
    source = tmp_path / "side-scan.xtf"
    source.write_bytes(header.to_bytes() + ping.to_bytes())
    reference = phx.qualification.ReferenceArtifactManifest(
        "actual-xtf",
        checksum_algorithm="sha256",
        checksum=sha256(source.read_bytes()).hexdigest(),
        size_bytes=source.stat().st_size,
        license_id="synthetic",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="public",
        nondimensionalization={"pressure": 1.0},
        uncertainty={"pressure": 0.0},
        lineage_ids=("synthetic",),
    )
    axis = phx.measurement.SampleTimeAxis(
        "xtf-time", np.linspace(0.0, 0.003, 4), phx.units.SECOND
    )
    acquisition = phx.sensing.SonarAcquisition(
        np.zeros(3),
        np.zeros((1, 3)),
        1500.0,
        axis,
        "sonar",
        "xtf-acquisition",
    )
    result = phx.sensing.XtfSideScanProvider().read(
        source,
        reference,
        acquisition,
        asset_id="xtf-side-scan",
        pressure_scale_pascals_per_count=1.0,
        maximum_samples=8,
    )
    np.testing.assert_allclose(result.measurement.field.values, ((1, 2, 3, 4),))


def test_fmcw_and_sonar_signal_profiles_preserve_acquisition_axes():
    acquisition = phx.sensing.FMCWAcquisition(
        77e9, 1e12, 1e-6, 1e-3, np.zeros((2, 3)), "radar"
    )
    radar = phx.sensing.FMCWTransformPlan(
        acquisition, 8, 4, propagation_speed=3e8
    ).evaluate(np.ones((4, 8, 2), dtype=np.complex64))
    assert radar.range_doppler_channels.shape == (4, 4, 2)
    assert bool(radar.successful)

    velocity_unit = phx.units.derived_unit(
        "m/s", ((phx.units.METER, 1), (phx.units.SECOND, -1))
    )
    radar_profile = phx.sensing.AutomotiveRadarProfile(
        "radar-frame", phx.units.METER, velocity_unit
    ).lower(
        np.asarray(((10.0, 0.1, -2.0, 3.0),)),
        _manifest(),
        asset_id="radar-detections",
    )
    assert len(radar_profile.assets) == 4
    assert {asset.field.quantity.name for asset in radar_profile.assets} == {
        "range",
        "azimuth",
        "radial-velocity",
        "radar-cross-section",
    }

    axis = phx.measurement.SampleTimeAxis(
        "sonar-time", np.linspace(0.0, 1.0, 8), phx.units.SECOND
    )
    sonar = phx.sensing.SonarAcquisition(
        np.zeros(3), np.asarray(((0, 0, 0), (0.1, 0, 0))), 1.0, axis, "tank", "sonar"
    )
    plan = phx.sensing.DelayAndSumBeamformingPlan(sonar, np.asarray(((0.5, 0.0, 0.0),)))
    waveforms = np.zeros((2, 8))
    waveforms[:, 4] = 1.0
    image = plan.evaluate(waveforms)
    assert image.image.shape == (1,)
    assert bool(image.successful)
