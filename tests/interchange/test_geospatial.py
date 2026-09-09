#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json

import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.interchange._geospatial import (
    GeospatialContract,
    GeospatialTransform,
    QualifiedGeospatialGrid,
)
from phydrax.units import DEGREE, METER, RADIAN, UnitDefinition


def _local(*, registration="pixel", mask_semantics="valid_true"):
    return GeospatialContract.local_cartesian(
        SpatialCoordinateContract(METER, reference_frame="survey-site"),
        vertical_datum="survey-benchmark-A",
        registration=registration,
        mask_semantics=mask_semantics,
    )


def _geographic(*, registration="gridline", seam="none", units=DEGREE):
    return GeospatialContract(
        SpatialCoordinateContract(METER, reference_frame="earth"),
        horizontal_crs="EPSG:4326",
        horizontal_kind="geographic",
        horizontal_axes=("longitude", "latitude"),
        horizontal_units=(units, units),
        horizontal_datum="WGS84",
        epoch_required=False,
        vertical_kind="ellipsoidal_height",
        vertical_positive="up",
        vertical_datum="WGS84-ellipsoid",
        vertical_unit=METER,
        registration=registration,
        longitude_seam=seam,
        longitude_domain=(-180, 180) if units == DEGREE else (-np.pi, np.pi),
        mask_semantics="valid_true",
    )


def test_geospatial_metadata_cannot_change_native_spatial_serialization():
    spatial = SpatialCoordinateContract(METER, reference_frame="survey-site")
    original = spatial.to_dict()
    extension = {"unrecognized_vendor_reference": {"offsets": [1, 2], "authority": None}}
    contract = GeospatialContract(spatial, metadata=extension)
    extension["unrecognized_vendor_reference"]["offsets"][0] = 99
    exported = contract.to_dict()
    assert exported["metadata"]["unrecognized_vendor_reference"]["offsets"] == [1, 2]
    assert exported["spatial"] == original == spatial.to_dict()
    assert (
        SpatialCoordinateContract.from_dict(exported["spatial"]).spatial_id
        == spatial.spatial_id
    )
    exported["metadata"]["unrecognized_vendor_reference"]["offsets"].append(3)
    assert contract.metadata["unrecognized_vendor_reference"]["offsets"] == [1, 2]
    with pytest.raises(ValueError):
        contract.require_cartesian()


def test_cartesian_qualification_rejects_angular_swapped_and_mismatched_coordinates():
    local = _local()
    assert local.require_cartesian().spatial_id == local.spatial.spatial_id
    with pytest.raises(ValueError):
        _geographic().require_cartesian()
    reversed_axes = GeospatialContract(
        local.spatial,
        horizontal_crs=local.horizontal_crs,
        horizontal_kind="local_cartesian",
        horizontal_axes=("north", "east"),
        horizontal_units=(METER, METER),
        horizontal_datum=local.horizontal_datum,
        epoch_required=False,
        longitude_seam="none",
    )
    with pytest.raises(ValueError):
        reversed_axes.require_cartesian(dimensions=2)
    with pytest.raises(ValueError):
        local.require_cartesian(
            SpatialCoordinateContract(METER, reference_frame="other-site")
        )
    kilometre = UnitDefinition("km", METER.dimension, METER.reference_system_id, 1000)
    with pytest.raises(ValueError):
        local.require_cartesian(
            SpatialCoordinateContract(kilometre, reference_frame="survey-site")
        )


def test_dynamic_epoch_and_vertical_datum_are_required_for_composition():
    spatial = SpatialCoordinateContract(METER, reference_frame="survey-site")
    common = dict(
        horizontal_crs="qualified-projection",
        horizontal_kind="projected",
        horizontal_axes=("east", "north"),
        horizontal_units=(METER, METER),
        horizontal_datum="dynamic-frame",
        epoch_required=True,
        longitude_seam="none",
        vertical_kind="orthometric_height",
        vertical_positive="up",
        vertical_datum="geoid-A",
        vertical_unit=METER,
    )
    with pytest.raises(ValueError):
        GeospatialContract(spatial, **common).require_cartesian()
    qualified = GeospatialContract(spatial, **common, coordinate_epoch=2020.0)
    assert qualified.require_cartesian().spatial_id == spatial.spatial_id
    other_epoch = GeospatialContract(spatial, **common, coordinate_epoch=2021.0)
    with pytest.raises(ValueError):
        qualified.require_compatible(other_epoch)
    for change in (
        {"vertical_datum": "geoid-B"},
        {"vertical_kind": "ellipsoidal_height"},
    ):
        other = GeospatialContract(spatial, **(common | change), coordinate_epoch=2020.0)
        with pytest.raises(ValueError):
            qualified.require_compatible(other)
        qualified.require_compatible(other, dimensions=2)
    depth = GeospatialContract(
        spatial,
        **(common | {"vertical_kind": "depth", "vertical_positive": "down"}),
        coordinate_epoch=2020.0,
    )
    with pytest.raises(ValueError):
        depth.require_cartesian()


def test_pixel_and_gridline_bounds_do_not_silently_shift_samples():
    x, y = np.array([10.0, 12.0, 14.0]), np.array([8.0, 5.0])
    values = np.array([[11.0, 12.0, 19.0], [31.0, 38.0, 45.0]])
    pixel = QualifiedGeospatialGrid(x, y, values, _local(), value_unit=METER)
    nodes = QualifiedGeospatialGrid(
        x, y, values, _local(registration="gridline"), value_unit=METER
    )
    assert pixel.region == (9.0, 15.0, 3.5, 9.5)
    assert nodes.region == (10.0, 14.0, 5.0, 8.0)
    assert pixel.orientation == ("increasing", "decreasing")
    np.testing.assert_array_equal(pixel.values, values)
    np.testing.assert_array_equal(pixel.x, nodes.x)
    np.testing.assert_array_equal(pixel.y, nodes.y)
    with pytest.raises(ValueError):
        pixel.contract.require_compatible(nodes.contract, require_grid=True)
    with pytest.raises(ValueError):
        pixel.values.setflags(write=True)
    with pytest.raises(ValueError):
        QualifiedGeospatialGrid([10, 12, 14.2], y, values, _local(), value_unit=METER)
    with pytest.raises(ValueError):
        QualifiedGeospatialGrid(x, [5, 5], values, _local(), value_unit=METER)


def test_unknown_registration_and_missing_data_are_not_inferred():
    with pytest.raises(ValueError):
        QualifiedGeospatialGrid(
            [0, 1],
            [0, 1],
            np.ones((2, 2)),
            _local(registration="unknown"),
            value_unit=METER,
        )
    values = np.array([[1.0, np.nan], [3.0, 7.0]])
    with pytest.raises(ValueError):
        QualifiedGeospatialGrid([0, 1], [0, 1], values, _local(), value_unit=METER)
    valid = np.array([[True, False], [True, True]])
    grid = QualifiedGeospatialGrid(
        [0, 1], [0, 1], values, _local(), value_unit=METER, valid=valid
    )
    np.testing.assert_array_equal(grid.valid, valid)
    with pytest.raises(ValueError):
        QualifiedGeospatialGrid(
            [0, 1],
            [0, 1],
            values,
            _local(mask_semantics="none"),
            value_unit=METER,
            valid=valid,
        )


def test_periodic_seams_require_matching_endpoint_values_and_masks():
    contract = _geographic(seam="periodic")
    values = np.array([[1, 2, 3, 4, 1], [5, 6, 7, 8, 5]], dtype=float)
    grid = QualifiedGeospatialGrid(
        [-180, -90, 0, 90, 180], [-20, 20], values, contract, value_unit=METER
    )
    assert grid.region == (-180.0, 180.0, -20.0, 20.0)
    broken = values.copy()
    broken[0, -1] = 99
    with pytest.raises(ValueError):
        QualifiedGeospatialGrid(grid.x, grid.y, broken, contract, value_unit=METER)
    valid = np.ones(values.shape, dtype=bool)
    valid[0, -1] = False
    with pytest.raises(ValueError):
        QualifiedGeospatialGrid(
            grid.x, grid.y, values, contract, value_unit=METER, valid=valid
        )
    with pytest.raises(ValueError):
        QualifiedGeospatialGrid(grid.x, grid.y, values, _geographic(), value_unit=METER)
    with pytest.raises(ValueError):
        QualifiedGeospatialGrid(
            [170, 180, -170], [-20, 20], np.ones((2, 3)), _geographic(), value_unit=METER
        )
    with pytest.raises(ValueError):
        QualifiedGeospatialGrid(
            [170, 180, 190], [-20, 20], np.ones((2, 3)), _geographic(), value_unit=METER
        )


def test_pixel_periodic_seam_and_radian_support_are_qualified_without_resampling():
    contract = _geographic(registration="pixel", seam="periodic", units=RADIAN)
    x = np.deg2rad([-135, -45, 45, 135])
    y = np.deg2rad([-45, 45])
    values = np.array([[1, 2, 3, 4], [8, 7, 6, 5]], dtype=float)
    grid = QualifiedGeospatialGrid(x, y, values, contract, value_unit=METER)
    np.testing.assert_array_equal(grid.x, x)
    np.testing.assert_allclose(grid.region, [-np.pi, np.pi, -np.pi / 2, np.pi / 2])
    with pytest.raises(ValueError):
        QualifiedGeospatialGrid(x[1:], y, values[:, 1:], contract, value_unit=METER)


def test_vertical_grid_preserves_depth_and_does_not_relabel_it_as_height():
    local = _local()
    args = dict(
        horizontal_crs=local.horizontal_crs,
        horizontal_kind=local.horizontal_kind,
        horizontal_axes=local.horizontal_axes,
        horizontal_units=local.horizontal_units,
        horizontal_datum=local.horizontal_datum,
        epoch_required=False,
        longitude_seam="none",
        registration="pixel",
        mask_semantics="none",
        vertical_kind="depth",
        vertical_positive="down",
        vertical_datum="survey-benchmark-A",
        vertical_unit=METER,
    )
    depth = GeospatialContract(local.spatial, **args)
    values = np.array([[1.0, 2.0], [4.0, 8.0]])
    grid = QualifiedGeospatialGrid(
        [0, 1], [0, 1], values, depth, value_unit=METER, value_role="vertical"
    )
    np.testing.assert_array_equal(grid.values, values)
    assert grid.contract.vertical_positive == "down"
    with pytest.raises(ValueError):
        grid.contract.require_cartesian()
    unknown = GeospatialContract(local.spatial, **(args | {"vertical_datum": None}))
    with pytest.raises(ValueError):
        QualifiedGeospatialGrid(
            [0, 1], [0, 1], values, unknown, value_unit=METER, value_role="vertical"
        )


def test_transformation_provenance_must_end_at_the_qualified_coordinate_identity():
    source, target = _geographic(), _local()
    operation = GeospatialTransform(
        "caller-qualified-local-projection",
        source.coordinate_id,
        target.coordinate_id,
        parameters={"method": "external-survey", "epoch": 2020.0},
    )
    arguments = target.to_dict()
    for name in ("spatial", "coordinate_id", "geospatial_id"):
        arguments.pop(name)
    arguments["horizontal_units"] = target.horizontal_units
    arguments["vertical_unit"] = target.vertical_unit
    arguments["transformations"] = (operation,)
    transformed = GeospatialContract(target.spatial, **arguments)
    transformed.require_compatible(target)
    assert transformed.geospatial_id != target.geospatial_id
    assert (
        json.loads(json.dumps(transformed.to_dict()))["transformations"][0]["parameters"][
            "method"
        ]
        == "external-survey"
    )
    invalid = GeospatialTransform(
        "wrong-endpoint", target.coordinate_id, source.coordinate_id
    )
    with pytest.raises(ValueError):
        GeospatialContract(
            target.spatial, **(arguments | {"transformations": (invalid,)})
        )
