#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


geometry = phx.velocimetry.imaging.ImageGeometry2D((33, 33))
intrinsics = phx.velocimetry.camera.CameraIntrinsics(
    (20.0, 20.0),
    (16.0, 16.0),
    image_shape=geometry.image_shape,
)
left = phx.velocimetry.camera.CameraModel(
    intrinsics,
    pose=phx.velocimetry.camera.CameraPose(
        phx.geometry.RigidFrame(jnp.eye(3), jnp.asarray((-0.5, 0.0, 0.0)))
    ),
)
right = phx.velocimetry.camera.CameraModel(
    intrinsics,
    pose=phx.velocimetry.camera.CameraPose(
        phx.geometry.RigidFrame(jnp.eye(3), jnp.asarray((0.5, 0.0, 0.0)))
    ),
)
rig = phx.velocimetry.camera.CameraRig((left, right))
detection_plan = phx.velocimetry.tracking.ParticleDetectionPlan(
    threshold=0.02,
    maximum_detections=4,
)
association_plan = phx.velocimetry.tracking.MultiViewAssociationPlan(
    2,
    8,
    2,
    maximum_ray_distance=0.05,
)


def gaussian_image(position_rc):
    row, column = jnp.meshgrid(jnp.arange(33.0), jnp.arange(33.0), indexing="ij")
    delta = jnp.stack((row - position_rc[0], column - position_rc[1]), axis=-1)
    return 10.0 * jnp.exp(-0.5 * jnp.sum(delta * delta, axis=-1) / 0.7**2)


def reconstruct(point_xyz, frame_index):
    pixels = tuple(
        phx.velocimetry.camera.project_points(camera, point_xyz[None]).pixels[0]
        for camera in rig.cameras
    )
    detections = tuple(
        phx.velocimetry.tracking.detect_particles(
            gaussian_image(pixel),
            geometry,
            detection_plan,
            frame_id=f"frame-{frame_index}-camera-{camera_index}",
        )
        for camera_index, pixel in enumerate(pixels)
    )
    association = phx.velocimetry.tracking.associate_multiview(
        detections,
        rig,
        association_plan,
    )
    return phx.velocimetry.tracking.reconstruct_particles(
        detections,
        rig,
        association,
        phx.velocimetry.tracking.TriangulationPlan(),
    )


truth = (jnp.asarray((0.1, -0.1, 5.0)), jnp.asarray((0.2, -0.1, 5.0)))
reconstructions = tuple(reconstruct(point, index) for index, point in enumerate(truth))
tracks = phx.velocimetry.tracking.link_tracks(
    reconstructions,
    jnp.asarray((0.0, 1.0)),
    phx.velocimetry.tracking.TrackLinkPlan(2, maximum_missed=1),
)
trajectory = phx.velocimetry.tracking.to_trajectory_data(tracks)

print(
    "reconstructed positions", reconstructions[0].positions_xyz[reconstructions[0].valid]
)
print("active track ids", tracks.track_ids[tracks.observed])
print("trajectory samples", int(jnp.sum(trajectory.sample_valid)))
print("successful", bool(jnp.all(reconstructions[0].valid[:1])))
