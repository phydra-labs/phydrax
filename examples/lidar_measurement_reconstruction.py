#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Admit calibrated LiDAR ranges and retain lineage through Cartesianization."""

import json

import numpy as np

import phydrax as phx


contract = phx.SpatialCoordinateContract(
    phx.units.METER,
    coordinate_system="cartesian-world",
    reference_frame="laboratory",
)
rays = phx.measurement.RaySampleSupport(
    np.zeros((4, 3)),
    np.asarray(((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 3.0), (1.0, 1.0, 0.0))),
    tuple(f"pulse-{index}" for index in range(4)),
    contract,
    active_mask=np.asarray((True, True, True, True)),
    far=np.full((4,), 10.0),
)
quantity = phx.measurement.QuantitySpec(
    "lidar", "range", "range", phx.units.METER, "physical.range"
)
field = phx.measurement.QuantityField(
    "laboratory-ranges",
    quantity,
    phx.measurement.ValueLayout.scalar(),
    rays,
    phx.measurement.SamplingSemantics(phx.measurement.SpatialSamplingKind.EVENT),
    np.asarray((1.0, 2.0, 3.0, np.nan)),
    np.asarray((True, True, True, False)),
)
reference = phx.qualification.ReferenceArtifactManifest(
    "synthetic-lidar-ranges",
    checksum_algorithm="sha256",
    checksum="5" * 64,
    size_bytes=1,
    license_id="synthetic",
    commercial_use_permitted=True,
    redistribution_permitted=True,
    training_use_permitted=True,
    export_permitted=True,
    export_classification="public",
    nondimensionalization={"length": 1.0},
    uncertainty={"range": 0.0},
    lineage_ids=("synthetic",),
)
asset = phx.measurement.MeasurementAsset.from_single_reference(
    "laboratory-scan",
    field,
    reference,
    phx.measurement.DerivationRecord(
        phx.measurement.DataOrigin.EXTERNAL,
        phx.measurement.DataStage.CALIBRATED,
    ),
)
points = phx.measurement.cartesianize_lidar_scan(phx.measurement.LidarScan(asset))
if int(np.sum(points.support.active_mask)) != 3:
    raise RuntimeError("No-return samples were not excluded from the point product.")
print(
    json.dumps(
        {
            "valid_points": int(np.sum(points.support.active_mask)),
            "point_product_id": points.point_product_id,
            "source_manifest_id": points.references[0].manifest_id,
            "points_m": np.asarray(points.support.points)[
                points.support.active_mask
            ].tolist(),
        },
        indent=2,
    )
)
