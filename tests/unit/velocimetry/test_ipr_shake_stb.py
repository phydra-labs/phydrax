#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp

from phydrax.geometry import RigidFrame
from phydrax.velocimetry.camera._model import (
    CameraIntrinsics,
    CameraModel,
    CameraPose,
)
from phydrax.velocimetry.camera._rig import CameraRig
from phydrax.velocimetry.imaging._photometry import (
    ParticleImageFormation,
    PhotometricResponse,
    render_camera_stack,
)
from phydrax.velocimetry.imaging._raster import GaussianRasterizer
from phydrax.velocimetry.imaging._types import ImageGeometry2D
from phydrax.velocimetry.tracking._association import MultiViewAssociationPlan
from phydrax.velocimetry.tracking._detection import ParticleDetectionPlan
from phydrax.velocimetry.tracking._ipr import (
    IPR_CAPACITY_EXHAUSTED,
    IPR_SUBSET_REJECTED,
    IPRPlan,
    iterative_particle_reconstruction,
)
from phydrax.velocimetry.tracking._reconstruction import TriangulationPlan
from phydrax.velocimetry.tracking._shake import shake_particles, ShakePlan
from phydrax.velocimetry.tracking._stb import (
    initialize_stb,
    prepare_stb,
    stb_step,
    STBPlan,
)
from phydrax.velocimetry.tracking._tracks import TrackLinkPlan


def _rig_and_geometry():
    geometry = ImageGeometry2D((48, 48))
    intrinsics = CameraIntrinsics(
        (30.0, 30.0),
        (23.5, 23.5),
        image_shape=geometry.image_shape,
    )
    left = CameraModel(
        intrinsics,
        pose=CameraPose(RigidFrame(jnp.eye(3), jnp.asarray((-0.5, 0.0, 0.0)))),
    )
    right = CameraModel(
        intrinsics,
        pose=CameraPose(RigidFrame(jnp.eye(3), jnp.asarray((0.5, 0.0, 0.0)))),
    )
    return CameraRig((left, right)), geometry


def _formation():
    return ParticleImageFormation(
        GaussianRasterizer(5, cutoff=3.0),
        PhotometricResponse(),
    )


def _ipr_plan(particle_capacity: int, *, selected_capacity: int = 4):
    detection = ParticleDetectionPlan(
        threshold=0.01,
        maximum_detections=4,
        crowding_distance=1.0,
    )
    association = MultiViewAssociationPlan(
        2,
        8,
        selected_capacity,
        maximum_ray_distance=0.03,
    )
    return IPRPlan(
        detection,
        association,
        TriangulationPlan(),
        particle_capacity=particle_capacity,
        iterations=2,
        duplicate_distance=0.12,
        minimum_candidate_intensity=0.01,
    )


def _render_truth(formation, rig, geometry, positions, amplitude, active):
    return render_camera_stack(
        formation,
        rig,
        geometry,
        positions,
        amplitude,
        jnp.ones((positions.shape[0],)),
        active,
    ).images


def test_noiseless_ipr_recovers_missing_particle_and_reduces_residual():
    rig, geometry = _rig_and_geometry()
    formation = _formation()
    truth_position = jnp.asarray([[-0.45, 0.05, 6.0], [0.55, -0.1, 6.4], [0.0, 0.0, 0.0]])
    truth_amplitude = jnp.asarray([18.0, 15.0, 0.0])
    truth_active = jnp.asarray([True, True, False])
    observed = _render_truth(
        formation, rig, geometry, truth_position, truth_amplitude, truth_active
    )
    initial_active = jnp.asarray([True, False, False])

    result = iterative_particle_reconstruction(
        _ipr_plan(3),
        formation,
        rig,
        geometry,
        observed,
        truth_position.at[1].set(0.0),
        truth_amplitude.at[1].set(0.0),
        jnp.ones((3,)),
        initial_active,
    )

    recovered_distance = jnp.sqrt(
        jnp.sum((result.positions_xyz - truth_position[1]) ** 2, axis=-1)
    )
    assert result.accepted_count >= 1
    assert jnp.sum(result.active) == 2
    assert jnp.min(jnp.where(result.active, recovered_distance, jnp.inf)) < 0.2
    assert result.final_loss < result.initial_loss
    assert jnp.sum(result.residual * result.residual) < jnp.sum(observed * observed)


def test_ipr_rejects_duplicates_and_reports_subset_and_capacity():
    rig, geometry = _rig_and_geometry()
    formation = _formation()
    one_position = jnp.asarray([[0.0, 0.0, 6.0], [0.0, 0.0, 0.0]])
    observed = _render_truth(
        formation,
        rig,
        geometry,
        one_position,
        jnp.asarray([20.0, 0.0]),
        jnp.asarray([True, False]),
    )
    duplicate = iterative_particle_reconstruction(
        _ipr_plan(2),
        formation,
        rig,
        geometry,
        observed,
        one_position,
        jnp.asarray([10.0, 0.0]),
        jnp.ones((2,)),
        jnp.asarray([True, False]),
    )
    assert duplicate.duplicate_rejected_count >= 1
    assert jnp.sum(duplicate.active) == 1
    assert duplicate.status == IPR_SUBSET_REJECTED

    two_positions = jnp.asarray([[-0.5, 0.0, 6.0], [0.6, 0.0, 6.0]])
    two_images = _render_truth(
        formation,
        rig,
        geometry,
        two_positions,
        jnp.asarray([16.0, 16.0]),
        jnp.asarray([True, True]),
    )
    capacity = iterative_particle_reconstruction(
        _ipr_plan(1, selected_capacity=2),
        formation,
        rig,
        geometry,
        two_images,
        two_positions[:1],
        jnp.asarray([16.0]),
        jnp.asarray([1.0]),
        jnp.asarray([True]),
    )
    assert capacity.capacity_rejected_count >= 1
    assert capacity.status == IPR_CAPACITY_EXHAUSTED


def test_shake_reduces_robust_residual_and_frozen_topology_is_differentiable():
    rig, geometry = _rig_and_geometry()
    formation = _formation()
    truth = jnp.asarray([[0.2, -0.1, 6.0]])
    observed = _render_truth(
        formation,
        rig,
        geometry,
        truth,
        jnp.asarray([18.0]),
        jnp.asarray([True]),
    )
    plan = ShakePlan(
        iterations=6,
        position_step=0.2,
        amplitude_step=0.5,
        maximum_displacement=0.5,
    )
    initial = truth + jnp.asarray([[0.08, 0.03, 0.0]])
    result = shake_particles(
        plan,
        formation,
        rig,
        geometry,
        observed,
        initial,
        jnp.asarray([16.0]),
        jnp.asarray([1.0]),
        jnp.asarray([True]),
    )

    assert result.accepted_steps >= 1
    assert result.loss_history[-1] < result.loss_history[0]
    assert jnp.sum(result.residual * result.residual) < jnp.sum(
        result.initial_residual * result.initial_residual
    )

    def refined_loss(x_coordinate):
        shifted = initial.at[0, 0].set(x_coordinate)
        refined = shake_particles(
            plan,
            formation,
            rig,
            geometry,
            observed,
            shifted,
            jnp.asarray([16.0]),
            jnp.asarray([1.0]),
            jnp.asarray([True]),
        )
        return refined.loss_history[-1]

    derivative = jax.grad(refined_loss)(initial[0, 0])
    assert jnp.isfinite(derivative)


def test_stb_promotes_distinct_identity_and_terminates_through_tracking_core():
    rig, geometry = _rig_and_geometry()
    formation = _formation()
    capacity = 2
    ipr = _ipr_plan(capacity, selected_capacity=2)
    shake = ShakePlan(iterations=2, position_step=0.1, amplitude_step=0.1)
    tracking = TrackLinkPlan(capacity, maximum_missed=0)
    prepared = prepare_stb(
        STBPlan(ipr, shake, tracking, promotion_steps=2),
        formation,
        rig,
        geometry,
        jnp.ones((capacity,)),
    )
    empty = initialize_stb(
        prepared,
        jnp.zeros((capacity, 3)),
        jnp.zeros((capacity,)),
        jnp.zeros((capacity,), dtype=bool),
        first_track_id=40,
    )
    truth_position = jnp.asarray([[0.1, 0.0, 6.0], [0.0, 0.0, 0.0]])
    observed = _render_truth(
        formation,
        rig,
        geometry,
        truth_position,
        jnp.asarray([18.0, 0.0]),
        jnp.asarray([True, False]),
    )

    first = stb_step(prepared, empty, observed, 1.0)
    born_slot = jnp.argmax(first.state.active.astype(jnp.int32))
    assert first.evidence.born_count == 1
    assert first.state.candidate_track_ids[born_slot] >= 40
    assert first.state.track_ids[born_slot] == -1
    second = stb_step(prepared, first.state, observed, 2.0)
    assert second.evidence.promoted_count == 1
    assert second.state.track_ids[born_slot] >= 40
    assert second.state.track_ids[born_slot] != second.state.slot_ids[born_slot]

    terminating = prepare_stb(
        STBPlan(
            ipr,
            shake,
            tracking,
            promotion_steps=1,
            minimum_active_amplitude=100.0,
        ),
        formation,
        rig,
        geometry,
        jnp.ones((capacity,)),
    )
    seeded = initialize_stb(
        terminating,
        truth_position,
        jnp.asarray([18.0, 0.0]),
        jnp.asarray([True, False]),
        first_track_id=70,
    )
    blank = jnp.zeros_like(observed)
    ended = stb_step(terminating, seeded, blank, 1.0)
    assert ended.evidence.terminated_count == 1
    assert not jnp.any(ended.state.active)
    assert jnp.all(ended.state.track_ids == -1)
