from hashlib import sha256

import numpy as np
import pytest

import phydrax as phx


def _manifest():
    return phx.qualification.ReferenceArtifactManifest(
        "synthetic-lidar",
        checksum_algorithm="sha256",
        checksum="3" * 64,
        size_bytes=1,
        license_id="synthetic",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="public",
        nondimensionalization={"length": 1.0},
        uncertainty={"range": 0.01},
        lineage_ids=("synthetic",),
    )


def test_lidar_ranges_cartesianize_without_turning_no_returns_into_points():
    contract = phx.SpatialCoordinateContract(
        phx.units.METER,
        coordinate_system="cartesian-world",
        reference_frame="world",
    )
    rays = phx.measurement.RaySampleSupport(
        np.zeros((3, 3)),
        np.asarray(((0.0, 0.0, 2.0), (0.0, 3.0, 0.0), (1.0, 0.0, 0.0))),
        ("pulse-0", "pulse-1", "pulse-2"),
        contract,
        active_mask=np.asarray((True, True, True)),
        far=np.asarray((10.0, 10.0, 10.0)),
    )
    quantity = phx.measurement.QuantitySpec(
        "lidar", "range", "range", phx.units.METER, "physical.range"
    )
    field = phx.measurement.QuantityField(
        "ranges",
        quantity,
        phx.measurement.ValueLayout.scalar(),
        rays,
        phx.measurement.SamplingSemantics(phx.measurement.SpatialSamplingKind.EVENT),
        np.asarray((2.0, 3.0, np.nan)),
        np.asarray((True, True, False)),
        phx.measurement.IndependentStandardUncertainty(
            np.asarray((0.1, 0.1, 0.1)), phx.units.METER
        ),
    )
    asset = phx.measurement.MeasurementAsset.from_single_reference(
        "scan-ranges",
        field,
        _manifest(),
        phx.measurement.DerivationRecord(
            phx.measurement.DataOrigin.EXTERNAL,
            phx.measurement.DataStage.CALIBRATED,
        ),
    )
    product = phx.measurement.cartesianize_lidar_scan(phx.measurement.LidarScan(asset))
    np.testing.assert_allclose(
        product.support.points[:2], ((0.0, 0.0, 2.0), (0.0, 3.0, 0.0))
    )
    assert not product.support.active_mask[2]
    np.testing.assert_allclose(product.support.points[2], (0.0, 0.0, 0.0))
    assert product.references[0].manifest_id == asset.references[0].manifest_id


def test_las_provider_preserves_derived_point_attributes(tmp_path):
    laspy = pytest.importorskip("laspy")
    header = laspy.LasHeader(point_format=3, version="1.2")
    data = laspy.LasData(header)
    data.x = np.asarray((1.0, 2.0))
    data.y = np.asarray((3.0, 4.0))
    data.z = np.asarray((5.0, 6.0))
    data.intensity = np.asarray((7, 8), dtype=np.uint16)
    data.return_number = np.asarray((1, 1), dtype=np.uint8)
    data.number_of_returns = np.asarray((1, 1), dtype=np.uint8)
    source = tmp_path / "points.las"
    data.write(source)
    reference = phx.qualification.ReferenceArtifactManifest(
        "bounded-las",
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
        uncertainty={"position": 0.01},
        lineage_ids=("synthetic",),
    )
    contract = phx.SpatialCoordinateContract(
        phx.units.METER,
        coordinate_system="cartesian-world",
        reference_frame="world",
    )
    product, report = phx.measurement.LasPointProvider().read(
        source,
        reference,
        contract,
        product_id="las-points",
        maximum_points=10,
    )
    np.testing.assert_allclose(product.support.points, ((1.0, 3.0, 5.0), (2.0, 4.0, 6.0)))
    assert {field.quantity.name for field in product.attributes} >= {
        "return-intensity",
        "return-number",
        "return-count",
    }
    assert report.valid
