import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.geometry import RigidFrame
from phydrax.velocimetry.camera import (
    CameraIntrinsics,
    CameraModel,
    CameraPose,
    CameraRig,
    project_points,
)
from phydrax.velocimetry.imaging import ImageGeometry2D
from phydrax.velocimetry.tracking import (
    associate_multiview,
    associate_two_view,
    detect_particles,
    initialize_tracks,
    link_tracks,
    link_tracks_step,
    MultiViewAssociationPlan,
    OfflineTrackRefinementPlan,
    ParticleDetectionPlan,
    ParticleDetections,
    ParticleReconstructionResult,
    reconstruct_particles,
    refine_tracks_min_cost_flow,
    smooth_tracks,
    to_trajectory_data,
    TrackLinkPlan,
    TrackSmoothingPlan,
    TriangulationPlan,
    TwoViewAssociationPlan,
)


def _detections(positions, *, valid=None, intensities=None, name="detections"):
    positions = jnp.asarray(positions, dtype=float)
    capacity = positions.shape[0]
    valid = jnp.ones((capacity,), dtype=bool) if valid is None else jnp.asarray(valid)
    intensities = (
        jnp.ones((capacity,), dtype=float)
        if intensities is None
        else jnp.asarray(intensities, dtype=float)
    )
    return ParticleDetections(
        positions,
        jnp.broadcast_to(0.02 * jnp.eye(2), (capacity, 2, 2)),
        intensities,
        jnp.ones((capacity,)),
        valid,
        jnp.zeros((capacity,), dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        f"frame:{name}",
        name,
    )


def _reconstruction(positions, valid, name):
    positions = jnp.asarray(positions, dtype=float)
    valid = jnp.asarray(valid, dtype=bool)
    capacity = positions.shape[0]
    return ParticleReconstructionResult(
        jnp.where(valid[:, None], positions, 0.0),
        jnp.where(
            valid[:, None, None],
            jnp.broadcast_to(0.01 * jnp.eye(3), (capacity, 3, 3)),
            0.0,
        ),
        valid.astype(float),
        valid,
        jnp.zeros((capacity,), dtype=jnp.int32),
        jnp.full((capacity, 2), -1, dtype=jnp.int32),
        jnp.zeros((capacity, 2)),
        name,
    )


def _camera_rays(origins, points):
    origins = jnp.asarray(origins, dtype=float)
    points = jnp.asarray(points, dtype=float)
    directions = points - origins
    return directions / jnp.sqrt(jnp.sum(directions * directions, axis=-1, keepdims=True))


def _stereo_rig():
    intrinsics = CameraIntrinsics((20.0, 20.0), (16.0, 16.0), image_shape=(33, 33))
    left = CameraModel(
        intrinsics,
        pose=CameraPose(RigidFrame(jnp.eye(3), jnp.asarray((-0.5, 0.0, 0.0)))),
    )
    right = CameraModel(
        intrinsics,
        pose=CameraPose(RigidFrame(jnp.eye(3), jnp.asarray((0.5, 0.0, 0.0)))),
    )
    return CameraRig((left, right))


def _gaussian_image(position_rc, *, amplitude=10.0):
    row, column = jnp.meshgrid(jnp.arange(33.0), jnp.arange(33.0), indexing="ij")
    delta = jnp.stack((row - position_rc[0], column - position_rc[1]), axis=-1)
    return amplitude * jnp.exp(-0.5 * jnp.sum(delta * delta, axis=-1) / 0.7**2)


def test_detector_reports_border_crowding_and_capacity_overflow():
    row, column = jnp.meshgrid(jnp.arange(17.0), jnp.arange(17.0), indexing="ij")
    image = jnp.zeros((17, 17))
    for center, amplitude in (
        ((1.0, 1.0), 12.0),
        ((7.0, 7.0), 10.0),
        ((7.0, 10.0), 9.0),
        ((14.0, 14.0), 8.0),
    ):
        squared = (row - center[0]) ** 2 + (column - center[1]) ** 2
        image = image + amplitude * jnp.exp(-0.5 * squared / 0.6**2)
    result = detect_particles(
        image,
        ImageGeometry2D((17, 17)),
        ParticleDetectionPlan(
            threshold=0.05,
            centroid_radius=1,
            border_width=2,
            crowding_distance=4.0,
            maximum_detections=2,
        ),
        frame_id="crowded-border",
    )

    assert result.positions_rc.shape == (2, 2)
    assert int(result.overflow_count) >= 1
    assert jnp.all(jnp.isfinite(result.covariance_rc[result.valid]))
    assert jnp.any(result.status[result.valid] == 1)
    assert jnp.any(result.status[result.valid] == 2)


def test_two_view_hungarian_uses_unique_resources_and_explicit_dummies():
    world_a = jnp.asarray([[-0.4, 0.0, 4.0], [0.5, 0.2, 5.0], [0.0, 0.0, 1.0]])
    world_b = jnp.asarray([[0.5, 0.2, 5.0], [-0.4, 0.0, 4.0], [1.5, 0.0, 3.0]])
    origin_a = jnp.broadcast_to(jnp.asarray((-0.5, 0.0, 0.0)), world_a.shape)
    origin_b = jnp.broadcast_to(jnp.asarray((0.5, 0.0, 0.0)), world_b.shape)
    detections_a = _detections(
        ((4.0, 4.0), (8.0, 8.0), (0.0, 0.0)), valid=(1, 1, 0), name="a"
    )
    detections_b = _detections(((8.0, 8.0), (4.0, 4.0), (12.0, 12.0)), name="b")
    result = associate_two_view(
        detections_a,
        detections_b,
        origin_a,
        _camera_rays(origin_a, world_a),
        origin_b,
        _camera_rays(origin_b, world_b),
        TwoViewAssociationPlan(maximum_ray_distance=0.05, unmatched_cost=4.0),
    )

    assert result.valid
    assert jnp.array_equal(result.matches_a_to_b[:2], jnp.asarray((1, 0)))
    assert len(np.unique(np.asarray(result.matches_a_to_b[result.matched_a]))) == 2
    assert jnp.array_equal(result.unmatched_b, jnp.asarray((False, False, True)))
    assert result.evidence.optimality_proven


def test_public_detection_association_reconstruction_workflow_is_physical():
    rig = _stereo_rig()
    point = jnp.asarray([[0.1, -0.1, 5.0]])
    projected = tuple(project_points(camera, point).pixels[0] for camera in rig.cameras)
    geometry = ImageGeometry2D((33, 33))
    detection_plan = ParticleDetectionPlan(threshold=0.02, maximum_detections=4)
    detections = tuple(
        detect_particles(
            _gaussian_image(pixel),
            geometry,
            detection_plan,
            frame_id=f"camera-{index}",
        )
        for index, pixel in enumerate(projected)
    )
    association = associate_multiview(
        detections,
        rig,
        MultiViewAssociationPlan(
            2,
            8,
            2,
            maximum_ray_distance=0.05,
            exact_candidate_limit=8,
        ),
    )
    reconstruction = reconstruct_particles(
        detections,
        rig,
        association,
        TriangulationPlan(),
    )

    assert jnp.sum(association.valid) == 1
    assert reconstruction.valid[0]
    assert jnp.allclose(reconstruction.positions_xyz[0], point[0], atol=0.08)
    used = association.detection_indices[association.valid]
    for camera in range(2):
        assert len(np.unique(np.asarray(used[:, camera]))) == used.shape[0]
    assert jnp.isfinite(reconstruction.reprojection_residual[0]).all()


def test_streaming_tracks_keep_ids_through_crossing_and_one_miss():
    reconstructions = (
        _reconstruction(((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), (1, 1), "r0"),
        _reconstruction(((-0.3, 0.0, 0.0), (0.3, 0.0, 0.0)), (1, 1), "r1"),
        _reconstruction(((0.4, 0.0, 0.0), (0.0, 0.0, 0.0)), (1, 0), "r2"),
        _reconstruction(((-0.4, 0.0, 0.0), (1.1, 0.0, 0.0)), (1, 1), "r3"),
    )
    result = link_tracks(
        reconstructions,
        jnp.arange(4.0),
        TrackLinkPlan(2, maximum_missed=1, mahalanobis_gate=25.0),
    )

    assert set(np.asarray(result.track_ids[result.observed], dtype=int)) == {0, 1}
    assert jnp.any(~result.observed[:, 2] & result.active[:, 2])
    missed_slot = int(jnp.argmax((~result.observed[:, 2]) & result.active[:, 2]))
    assert result.track_ids[missed_slot, 2] == result.track_ids[missed_slot, 3]
    assert jnp.all(result.overflow_count == 0)


def test_birth_death_capacity_monotone_time_and_trajectory_reset_semantics():
    plan = TrackLinkPlan(1, maximum_missed=0, mahalanobis_gate=4.0)
    state = initialize_tracks(plan)
    covariance = jnp.broadcast_to(0.01 * jnp.eye(3), (2, 3, 3))
    first = link_tracks_step(
        state,
        jnp.asarray(((0.0, 0.0, 0.0), (3.0, 0.0, 0.0))),
        covariance,
        jnp.asarray((True, True)),
        0.0,
        plan,
    )
    assert first.births[0]
    assert first.evidence.overflow_count == 1
    with pytest.raises(
        (eqx.EquinoxRuntimeError, RuntimeError, ValueError),
        match="strictly increasing",
    ):
        link_tracks_step(
            first.state,
            jnp.zeros((2, 3)),
            covariance,
            jnp.asarray((False, False)),
            0.0,
            plan,
        )

    sequence = (
        _reconstruction(((0.0, 0.0, 0.0),), (1,), "one"),
        _reconstruction(((5.0, 0.0, 0.0),), (1,), "two"),
        _reconstruction(((5.2, 0.0, 0.0),), (1,), "three"),
    )
    result = link_tracks(sequence, jnp.arange(3.0), plan)
    trajectory = to_trajectory_data(result)
    assert result.deaths[0, 1] and result.births[0, 1]
    assert result.track_ids[0, 0] != result.track_ids[0, 1]
    assert trajectory.reset_mask[0, 0]
    assert not trajectory.transition_valid[0, 0]
    assert trajectory.source_id == result.result_id


def test_frozen_association_smoothing_preserves_capacity_and_gaps():
    observations = (
        _reconstruction(((0.1, 0.0, 0.0),), (1,), "s0"),
        _reconstruction(((0.9, 0.0, 0.0),), (1,), "s1"),
        _reconstruction(((0.0, 0.0, 0.0),), (0,), "s2"),
        _reconstruction(((3.1, 0.0, 0.0),), (1,), "s3"),
    )
    tracks = link_tracks(
        observations,
        jnp.arange(4.0),
        TrackLinkPlan(2, maximum_missed=1, mahalanobis_gate=40.0),
    )
    smoothed = smooth_tracks(
        tracks,
        TrackSmoothingPlan(process_acceleration_variance=0.01),
    )

    assert smoothed.states.shape == tracks.states.shape
    assert jnp.all(smoothed.valid <= tracks.active)
    assert jnp.all(jnp.isfinite(smoothed.covariances[smoothed.valid]))
    assert len(smoothed.filter_results) >= 1
    trajectory = to_trajectory_data(tracks)
    assert not trajectory.sample_valid[:, 2].any()
    assert trajectory.reset_mask[:, 1].all()


def test_offline_min_cost_flow_refinement_links_without_resource_reuse():
    reconstructions = (
        _reconstruction(((0.0, 0.0, 0.0), (3.0, 0.0, 0.0)), (1, 1), "f0"),
        _reconstruction(((0.2, 0.0, 0.0), (2.8, 0.0, 0.0)), (1, 1), "f1"),
        _reconstruction(((0.4, 0.0, 0.0), (2.6, 0.0, 0.0)), (1, 1), "f2"),
    )
    refined = refine_tracks_min_cost_flow(
        reconstructions,
        jnp.arange(3.0),
        OfflineTrackRefinementPlan(maximum_displacement=0.5),
    )

    assert refined.valid
    assert jnp.all(refined.selected_observations)
    assert jnp.array_equal(refined.successor[0], jnp.asarray((2, 3)))
    assert jnp.array_equal(refined.predecessor[1], jnp.asarray((0, 1)))
