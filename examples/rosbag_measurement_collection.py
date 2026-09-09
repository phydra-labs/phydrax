"""Lower normalized ROS records without introducing implicit synchronization."""

import numpy as np

import phydrax as phx


reference = phx.qualification.ReferenceArtifactManifest(
    "ros-example",
    checksum_algorithm="sha256",
    checksum="9" * 64,
    size_bytes=1,
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
profiles = (
    phx.sensing.RosTopicProfile(
        "/scan", "LaserScan", phx.sensing.RosMessageKind.LASER_SCAN, "sensor"
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
            "angles": np.asarray((0.0, 0.5)),
            "range_min": 0.1,
            "range_max": 10.0,
        },
    ),
    phx.sensing.RosMessageRecord(
        "/tf",
        phx.sensing.RosMessageKind.TRANSFORM,
        0.0,
        0.1,
        1,
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
        2,
        {
            "source_frame": "sensor",
            "target_frame": "world",
            "rotation": np.eye(3),
            "translation": np.asarray((1.0, 0.0, 0.0)),
        },
    ),
)
result = phx.sensing.RosbagImportPlan(profiles).lower(
    records, reference, campaign_id="ros-example"
)
_, translation, evidence = result.frame_graph.prepare_route("sensor", "world").evaluate(
    0.5
)
if not bool(evidence.successful):
    raise RuntimeError("ROS transform route failed.")
print(
    {
        "assets": len(result.collection.assets),
        "translation": np.asarray(translation).tolist(),
    }
)
