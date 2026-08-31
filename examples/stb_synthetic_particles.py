#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


geometry = phx.velocimetry.imaging.ImageGeometry2D((32, 32))
intrinsics = phx.velocimetry.camera.CameraIntrinsics(
    (24.0, 24.0),
    (15.5, 15.5),
    image_shape=geometry.image_shape,
)
rig = phx.velocimetry.camera.CameraRig(
    (
        phx.velocimetry.camera.CameraModel(
            intrinsics,
            pose=phx.velocimetry.camera.CameraPose(
                phx.geometry.RigidFrame(jnp.eye(3), jnp.asarray((-0.5, 0.0, 0.0)))
            ),
        ),
        phx.velocimetry.camera.CameraModel(
            intrinsics,
            pose=phx.velocimetry.camera.CameraPose(
                phx.geometry.RigidFrame(jnp.eye(3), jnp.asarray((0.5, 0.0, 0.0)))
            ),
        ),
    )
)
formation = phx.velocimetry.imaging.ParticleImageFormation(
    phx.velocimetry.imaging.GaussianRasterizer(4, cutoff=3.0),
    phx.velocimetry.imaging.PhotometricResponse(),
)
detection = phx.velocimetry.tracking.ParticleDetectionPlan(
    threshold=0.01,
    maximum_detections=2,
    crowding_distance=1.0,
)
association = phx.velocimetry.tracking.MultiViewAssociationPlan(
    2,
    4,
    1,
    maximum_ray_distance=0.04,
)
ipr = phx.velocimetry.tracking.IPRPlan(
    detection,
    association,
    phx.velocimetry.tracking.TriangulationPlan(),
    particle_capacity=1,
    iterations=1,
    duplicate_distance=0.1,
    minimum_candidate_intensity=0.01,
)
plan = phx.velocimetry.tracking.STBPlan(
    ipr,
    phx.velocimetry.tracking.ShakePlan(
        iterations=1,
        position_step=0.05,
        amplitude_step=0.05,
    ),
    phx.velocimetry.tracking.TrackLinkPlan(1, maximum_missed=0),
    promotion_steps=1,
)
prepared = phx.velocimetry.tracking.prepare_stb(
    plan,
    formation,
    rig,
    geometry,
    jnp.ones((1,)),
)
state = phx.velocimetry.tracking.initialize_stb(
    prepared,
    jnp.zeros((1, 3)),
    jnp.zeros((1,)),
    jnp.zeros((1,), dtype=bool),
    first_track_id=100,
)
truth_position = jnp.asarray([[0.1, 0.0, 6.0]])
observed = phx.velocimetry.imaging.render_camera_stack(
    formation,
    rig,
    geometry,
    truth_position,
    jnp.asarray([18.0]),
    jnp.ones((1,)),
    jnp.asarray([True]),
).images
result = phx.velocimetry.tracking.stb_step(prepared, state, observed, 1.0)

print("reconstructed position", result.state.positions_xyz[result.state.active])
print("residual energy", float(jnp.sum(result.residual * result.residual)))
print("promoted track id", result.state.track_ids[result.state.active])
print("successful", bool(result.successful))
