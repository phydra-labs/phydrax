#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Render an exact camera-space physical field from a moving surface."""

import json

import numpy as np

import phydrax as phx


vertices = np.asarray(
    ((-1.0, -1.0, 2.0), (1.0, -1.0, 2.0), (0.0, 1.0, 2.0), (0.0, 0.0, 4.0))
)
triangles = np.asarray(((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)))
contract = phx.SpatialCoordinateContract(
    phx.units.METER,
    coordinate_system="cartesian-world",
    reference_frame="world",
)
metadata = phx.geometry.surface.SurfaceMetadata(
    source_id="moving-tetrahedron",
    source_revision="1",
    coordinate_contract=contract,
    provenance=("synthetic",),
)
realization = phx.geometry.surface.SurfaceModel.from_triangles(
    vertices,
    triangles,
    metadata,
    repair_orientation=True,
    orient_closed_outward=True,
).prepare()
support = phx.imaging.ImagePlaneSupport((5, 5), detector_frame_id="camera")
camera = phx.imaging.camera.CameraModel(
    phx.imaging.camera.CameraIntrinsics(
        (2.0, 2.0), (2.0, 2.0), image_shape=support.image_shape
    )
)
quantity = phx.measurement.QuantitySpec(
    "example",
    "surface-temperature",
    "temperature",
    phx.units.KELVIN,
    "physical.temperature",
)
prepared = phx.rendering.SurfaceImagePlan(
    realization,
    support,
    camera,
    quantity,
    phx.measurement.ValueLayout.scalar(),
    phx.measurement.SamplingSemantics(phx.measurement.SpatialSamplingKind.POINT),
).prepare()
result = prepared.render(
    vertices + np.asarray((0.0, 0.0, 0.5)),
    np.asarray((300.0, 300.0, 300.0, 330.0)),
    geometry_id="translated-surface",
)
if not bool(result.evidence.successful):
    raise RuntimeError("Surface image evidence rejected the rendered result.")
print(
    json.dumps(
        {
            "center_depth_m": float(result.depth[2, 2]),
            "center_temperature_k": float(result.prediction.values[2, 2]),
            "hit_count": int(np.sum(np.asarray(result.hit))),
            "route_stable": bool(result.evidence.route_stable),
        },
        indent=2,
    )
)
