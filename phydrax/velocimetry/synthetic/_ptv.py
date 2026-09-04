#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry import RigidFrame
from ...optics.geometric import PlanarRefractiveStack
from ..camera import CameraIntrinsics, CameraModel, CameraPose, CameraRig
from ..imaging import ImageGeometry2D
from ..imaging._photometry import (
    CameraStackRenderResult,
    ParticleImageFormation,
    PhotometricResponse,
    render_camera_stack,
)
from ..imaging._raster import GaussianRasterizer
from ._common import PTVScenarioKind, SyntheticEvidence


def _finite_vector(
    value: Sequence[float], length: int, /, *, name: str
) -> tuple[float, ...]:
    result = tuple(float(item) for item in value)
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain {length} finite values.")
    return result


class PTVScenarioPlan(StrictModule, NonTrainableState):
    """Finite deterministic multiview trajectory and image-formation recipe."""

    kind: PTVScenarioKind = eqx.field(static=True)
    family_id: str = eqx.field(static=True)
    image_shape: tuple[int, int] = eqx.field(static=True)
    frame_count: int = eqx.field(static=True)
    camera_count: int = eqx.field(static=True)
    particle_capacity: int = eqx.field(static=True)
    particle_count: int = eqx.field(static=True)
    focal_length: float = eqx.field(static=True)
    camera_baseline: float = eqx.field(static=True)
    volume_center_xyz: tuple[float, float, float] = eqx.field(static=True)
    volume_extent_xyz: tuple[float, float, float] = eqx.field(static=True)
    velocity_scale_xyz: tuple[float, float, float] = eqx.field(static=True)
    calibration_perturbation: float = eqx.field(static=True)
    refractive_index: float = eqx.field(static=True)
    refraction_interface_z: float = eqx.field(static=True)
    occlusion_radius: float = eqx.field(static=True)
    observation_dropout_probability: float = eqx.field(static=True)
    particle_diameter: float = eqx.field(static=True)
    particle_intensity: float = eqx.field(static=True)
    read_noise_std: float = eqx.field(static=True)
    shot_noise: bool = eqx.field(static=True)
    seed: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: PTVScenarioKind | str = PTVScenarioKind.BASELINE,
        /,
        *,
        family_id: str | None = None,
        image_shape: Sequence[int] = (64, 64),
        frame_count: int = 6,
        camera_count: int = 3,
        particle_capacity: int = 64,
        particle_count: int | None = None,
        focal_length: float = 70.0,
        camera_baseline: float = 0.7,
        volume_center_xyz: Sequence[float] = (0.0, 0.0, 4.0),
        volume_extent_xyz: Sequence[float] = (0.8, 0.6, 0.8),
        velocity_scale_xyz: Sequence[float] = (0.25, 0.2, 0.15),
        calibration_perturbation: float = 0.01,
        refractive_index: float = 1.33,
        refraction_interface_z: float = 1.0,
        occlusion_radius: float = 1.5,
        observation_dropout_probability: float = 0.0,
        particle_diameter: float = 2.0,
        particle_intensity: float = 1.0,
        read_noise_std: float = 0.0,
        shot_noise: bool = False,
        seed: int = 0,
    ):
        kind_ = PTVScenarioKind(kind)
        family = kind_.value if family_id is None else str(family_id)
        shape = tuple(int(item) for item in image_shape)
        frames = int(frame_count)
        cameras = int(camera_count)
        capacity = int(particle_capacity)
        count = (
            capacity
            if particle_count is None and kind_ is PTVScenarioKind.DENSE
            else min(24, capacity)
            if particle_count is None
            else int(particle_count)
        )
        focal = float(focal_length)
        baseline = float(camera_baseline)
        perturbation = float(calibration_perturbation)
        refractive_index_ = float(refractive_index)
        interface_z = float(refraction_interface_z)
        occlusion = float(occlusion_radius)
        dropout = float(observation_dropout_probability)
        diameter = float(particle_diameter)
        intensity = float(particle_intensity)
        noise = float(read_noise_std)
        center = _finite_vector(volume_center_xyz, 3, name="volume_center_xyz")
        extent = _finite_vector(volume_extent_xyz, 3, name="volume_extent_xyz")
        velocity = _finite_vector(velocity_scale_xyz, 3, name="velocity_scale_xyz")
        scalars = (
            focal,
            baseline,
            perturbation,
            refractive_index_,
            interface_z,
            occlusion,
            dropout,
            diameter,
            intensity,
            noise,
        )
        if not family:
            raise ValueError("family_id must be non-empty.")
        if len(shape) != 2 or any(item < 4 for item in shape):
            raise ValueError(
                "image_shape must contain two dimensions of at least four pixels."
            )
        if frames < 2:
            raise ValueError("frame_count must be at least two.")
        if cameras < 2:
            raise ValueError("camera_count must be at least two.")
        if capacity < 2 or not 1 <= count <= capacity:
            raise ValueError("particle_count must lie in [1, particle_capacity].")
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("PTV scenario scalar parameters must be finite.")
        if focal <= 0.0 or baseline < 0.0:
            raise ValueError(
                "focal_length must be positive and camera_baseline non-negative."
            )
        if any(value <= 0.0 for value in extent):
            raise ValueError("volume_extent_xyz must be positive.")
        if any(value < 0.0 for value in velocity):
            raise ValueError("velocity_scale_xyz must be non-negative.")
        minimum_trajectory_z = center[2] - 0.5 * extent[2] - 0.5 * velocity[2]
        if minimum_trajectory_z <= 0.0:
            raise ValueError("Every particle trajectory must remain in front of the rig.")
        if not 0.0 <= perturbation < 1.0:
            raise ValueError("calibration_perturbation must lie in [0, 1).")
        if refractive_index_ <= 0.0:
            raise ValueError("refractive_index must be positive.")
        if kind_ is PTVScenarioKind.REFRACTION and not (
            0.0 < interface_z < minimum_trajectory_z
        ):
            raise ValueError(
                "The refractive interface must lie between the rig and particle volume."
            )
        if occlusion < 0.0:
            raise ValueError("occlusion_radius must be non-negative.")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("observation_dropout_probability must lie in [0, 1].")
        if diameter <= 0.0 or intensity <= 0.0 or noise < 0.0:
            raise ValueError(
                "Particle diameter/intensity must be positive and noise non-negative."
            )

        self.kind = kind_
        self.family_id = family
        self.image_shape = shape
        self.frame_count = frames
        self.camera_count = cameras
        self.particle_capacity = capacity
        self.particle_count = count
        self.focal_length = focal
        self.camera_baseline = baseline
        self.volume_center_xyz = center
        self.volume_extent_xyz = extent
        self.velocity_scale_xyz = velocity
        self.calibration_perturbation = perturbation
        self.refractive_index = refractive_index_
        self.refraction_interface_z = interface_z
        self.occlusion_radius = occlusion
        self.observation_dropout_probability = dropout
        self.particle_diameter = diameter
        self.particle_intensity = intensity
        self.read_noise_std = noise
        self.shot_noise = bool(shot_noise)
        self.seed = int(seed)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "ptv-synthetic-scenario-plan",
                "scenario_kind": kind_.value,
                "family": family,
                "image_shape": list(shape),
                "frames": frames,
                "cameras": cameras,
                "capacity": capacity,
                "particle_count": count,
                "camera": {
                    "focal_length": focal,
                    "baseline": baseline,
                    "calibration_perturbation": perturbation,
                    "refractive_index": refractive_index_,
                    "refraction_interface_z": interface_z,
                },
                "volume": {"center_xyz": list(center), "extent_xyz": list(extent)},
                "velocity_scale_xyz": list(velocity),
                "observation": {
                    "occlusion_radius": occlusion,
                    "dropout_probability": dropout,
                    "particle_diameter": diameter,
                    "particle_intensity": intensity,
                    "read_noise_std": noise,
                    "shot_noise": bool(shot_noise),
                },
                "seed": int(seed),
                "world_coordinate_convention": "right-handed-x-y-z",
                "image_coordinate_convention": "row-down-column-right",
            }
        )


class PTVSyntheticCase(StrictModule, NonTrainableState):
    """Padded multiview images, camera contracts, and right-handed world truth."""

    geometry: ImageGeometry2D
    true_rig: CameraRig
    nominal_rig: CameraRig
    images: Array
    ideal_images: Array
    projection_pixels_rc: Array
    projection_depth: Array
    projection_valid: Array
    projection_status: Array
    visible: Array
    world_positions_xyz: Array
    particle_active: Array
    trajectory_ids: Array
    renderings: tuple[CameraStackRenderResult, ...]
    evidence: SyntheticEvidence
    family_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    scenario_id: str = eqx.field(static=True)
    world_coordinate_convention: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: ImageGeometry2D,
        true_rig: CameraRig,
        nominal_rig: CameraRig,
        images: Array,
        ideal_images: Array,
        projection_pixels_rc: Array,
        projection_depth: Array,
        projection_valid: Array,
        projection_status: Array,
        visible: Array,
        world_positions_xyz: Array,
        particle_active: Array,
        trajectory_ids: Array,
        renderings: tuple[CameraStackRenderResult, ...],
        evidence: SyntheticEvidence,
        /,
        *,
        family_id: str,
        plan_id: str,
        scenario_id: str,
    ):
        if not isinstance(geometry, ImageGeometry2D):
            raise TypeError("geometry must be ImageGeometry2D.")
        if not isinstance(true_rig, CameraRig) or not isinstance(nominal_rig, CameraRig):
            raise TypeError("true_rig and nominal_rig must be CameraRig values.")
        if not isinstance(evidence, SyntheticEvidence):
            raise TypeError("evidence must be SyntheticEvidence.")
        frames, capacity, components = world_positions_xyz.shape
        camera_count = true_rig.capacity
        if components != 3 or particle_active.shape != (frames, capacity):
            raise ValueError(
                "PTV world truth must have shapes (frames, capacity, 3)/(frames, capacity)."
            )
        if (
            images.shape != (frames, camera_count, *geometry.image_shape)
            or ideal_images.shape != images.shape
        ):
            raise ValueError(
                "PTV image stacks must have shape (frames, cameras, rows, columns)."
            )
        projection_shape = (frames, camera_count, capacity)
        if projection_pixels_rc.shape != projection_shape + (2,):
            raise ValueError(
                "Projection pixels must have shape (frames, cameras, capacity, 2)."
            )
        if any(
            value.shape != projection_shape
            for value in (
                projection_depth,
                projection_valid,
                projection_status,
                visible,
            )
        ):
            raise ValueError(
                "Projection evidence must match (frames, cameras, capacity)."
            )
        if trajectory_ids.shape != (capacity,):
            raise ValueError("trajectory_ids must have shape (capacity,).")
        if nominal_rig.capacity != camera_count:
            raise ValueError("True and nominal camera rigs must have equal capacity.")
        if len(renderings) != frames:
            raise ValueError("Native camera-stack renderings must cover every frame.")
        if any(not isinstance(result, CameraStackRenderResult) for result in renderings):
            raise TypeError("renderings must contain CameraStackRenderResult values.")
        if any(result.geometry_id != geometry.geometry_id for result in renderings):
            raise ValueError("Every native rendering must share the case image geometry.")
        identifiers = (str(family_id), str(plan_id), str(scenario_id))
        if any(not value for value in identifiers):
            raise ValueError("PTV case identifiers must be non-empty.")
        self.geometry = geometry
        self.true_rig = true_rig
        self.nominal_rig = nominal_rig
        self.images = jnp.asarray(images)
        self.ideal_images = jnp.asarray(ideal_images)
        self.projection_pixels_rc = jnp.asarray(projection_pixels_rc)
        self.projection_depth = jnp.asarray(projection_depth)
        self.projection_valid = jnp.asarray(projection_valid, dtype=bool)
        self.projection_status = jnp.asarray(projection_status, dtype=jnp.int32)
        self.visible = jnp.asarray(visible, dtype=bool)
        self.world_positions_xyz = jnp.asarray(world_positions_xyz)
        self.particle_active = jnp.asarray(particle_active, dtype=bool)
        self.trajectory_ids = jnp.asarray(trajectory_ids, dtype=jnp.int32)
        self.renderings = renderings
        self.evidence = evidence
        self.family_id, self.plan_id, self.scenario_id = identifiers
        self.world_coordinate_convention = "right-handed-x-y-z"


def _planar_refractive_stack(plan: PTVScenarioPlan) -> PlanarRefractiveStack:
    return PlanarRefractiveStack(
        jnp.asarray(((0.0, 0.0, plan.refraction_interface_z),)),
        jnp.asarray(((0.0, 0.0, 1.0),)),
        jnp.asarray((1.0, plan.refractive_index)),
    )


def _camera_rigs(plan: PTVScenarioPlan) -> tuple[CameraRig, CameraRig]:
    rows, columns = plan.image_shape
    principal = ((rows - 1.0) / 2.0, (columns - 1.0) / 2.0)
    baseline = (
        0.0 if plan.kind is PTVScenarioKind.DEGENERATE_RAYS else plan.camera_baseline
    )
    offsets = jnp.linspace(-0.5 * baseline, 0.5 * baseline, plan.camera_count)
    refractive_stack = (
        _planar_refractive_stack(plan)
        if plan.kind is PTVScenarioKind.REFRACTION
        else None
    )
    true_cameras: list[CameraModel] = []
    nominal_cameras: list[CameraModel] = []
    for camera_index in range(plan.camera_count):
        translation = (float(offsets[camera_index]), 0.0, 0.0)
        intrinsics = CameraIntrinsics(
            (plan.focal_length, plan.focal_length),
            principal,
            image_shape=plan.image_shape,
        )
        true_camera = CameraModel(
            intrinsics,
            pose=CameraPose(
                RigidFrame.identity(3)
                if baseline == 0.0
                else RigidFrame(jnp.eye(3), translation)
            ),
            refractive_stack=refractive_stack,
        )
        true_cameras.append(true_camera)
        if plan.kind is PTVScenarioKind.CALIBRATION:
            signed = -1.0 if camera_index % 2 else 1.0
            angle = signed * plan.calibration_perturbation
            perturbed_frame = RigidFrame.from_axis_angle(
                (0.0, 1.0, 0.0),
                angle,
                translation=(
                    translation[0]
                    + signed * plan.calibration_perturbation * max(baseline, 1.0),
                    0.0,
                    0.0,
                ),
            )
            perturbed_intrinsics = CameraIntrinsics(
                (
                    plan.focal_length * (1.0 + signed * plan.calibration_perturbation),
                    plan.focal_length * (1.0 - signed * plan.calibration_perturbation),
                ),
                (
                    principal[0] + signed * plan.calibration_perturbation * rows,
                    principal[1] - signed * plan.calibration_perturbation * columns,
                ),
                image_shape=plan.image_shape,
            )
            nominal_cameras.append(
                CameraModel(perturbed_intrinsics, pose=CameraPose(perturbed_frame))
            )
        elif plan.kind is PTVScenarioKind.REFRACTION:
            nominal_cameras.append(CameraModel(intrinsics, pose=true_camera.pose))
        else:
            nominal_cameras.append(true_camera)
    return CameraRig(tuple(true_cameras)), CameraRig(tuple(nominal_cameras))


def _world_trajectories(
    plan: PTVScenarioPlan, position_key: Array, velocity_key: Array
) -> tuple[Array, Array]:
    capacity = plan.particle_capacity
    center = jnp.asarray(plan.volume_center_xyz)
    extent = jnp.asarray(plan.volume_extent_xyz)
    initial = center + (jr.uniform(position_key, (capacity, 3)) - 0.5) * extent
    velocity_scale = jnp.asarray(plan.velocity_scale_xyz)
    velocities = (2.0 * jr.uniform(velocity_key, (capacity, 3)) - 1.0) * velocity_scale
    time = jnp.linspace(-0.5, 0.5, plan.frame_count)
    positions = initial[None, ...] + time[:, None, None] * velocities[None, ...]
    slots = jnp.arange(capacity)
    active = jnp.broadcast_to(
        slots[None, :] < plan.particle_count, (plan.frame_count, capacity)
    )

    if plan.kind is PTVScenarioKind.CROSSINGS and plan.particle_count >= 2:
        crossing_time = jnp.linspace(-1.0, 1.0, plan.frame_count)
        crossing_center = jnp.asarray(plan.volume_center_xyz)
        first = jnp.stack(
            (
                crossing_center[0] + 0.25 * crossing_time,
                jnp.full_like(crossing_time, crossing_center[1]),
                jnp.full_like(crossing_time, crossing_center[2]),
            ),
            axis=-1,
        )
        second = first.at[:, 0].set(crossing_center[0] - 0.25 * crossing_time)
        positions = positions.at[:, 0].set(first).at[:, 1].set(second)
    elif plan.kind is PTVScenarioKind.OCCLUSION and plan.particle_count >= 2:
        foreground = jnp.asarray(plan.volume_center_xyz).at[2].add(-0.2)
        background = jnp.asarray(plan.volume_center_xyz).at[2].add(0.2)
        positions = positions.at[:, 0].set(foreground).at[:, 1].set(background)
    elif plan.kind is PTVScenarioKind.BIRTHS_DEATHS:
        frame = jnp.arange(plan.frame_count)[:, None]
        phase = slots % 3
        birth = 1 + slots % (plan.frame_count - 1)
        death = 1 + (2 * slots + 1) % (plan.frame_count - 1)
        temporal = jnp.where(
            phase[None, :] == 1,
            frame >= birth[None, :],
            jnp.where(phase[None, :] == 2, frame < death[None, :], True),
        )
        active = active & temporal
    return positions, active


def _occlusion_visibility(
    plan: PTVScenarioPlan,
    projection_pixels: Array,
    projection_depth: Array,
    projection_valid: Array,
    active: Array,
    /,
) -> Array:
    camera_visibility: list[Array] = []
    for camera_index in range(projection_pixels.shape[0]):
        visible = active & projection_valid[camera_index]
        if plan.kind is PTVScenarioKind.OCCLUSION:
            pixels = projection_pixels[camera_index]
            depth = projection_depth[camera_index]
            pixel_delta = pixels[:, None, :] - pixels[None, :, :]
            pixel_distance_squared = jnp.sum(pixel_delta * pixel_delta, axis=-1)
            nearer = depth[None, :] < depth[:, None]
            competing = visible[None, :] & nearer
            occluded = jnp.any(
                competing & (pixel_distance_squared <= plan.occlusion_radius**2),
                axis=1,
            )
            visible = visible & ~occluded
        camera_visibility.append(visible)
    return jnp.stack(tuple(camera_visibility), axis=0)


def generate_ptv_case(
    plan: PTVScenarioPlan,
    /,
    *,
    rasterizer: GaussianRasterizer | None = None,
) -> PTVSyntheticCase:
    """Materialize a deterministic multiview PTV/STB scenario and native evidence."""
    if not isinstance(plan, PTVScenarioPlan):
        raise TypeError("plan must be a PTVScenarioPlan.")
    rasterizer_ = GaussianRasterizer() if rasterizer is None else rasterizer
    if not isinstance(rasterizer_, GaussianRasterizer):
        raise TypeError("rasterizer must be GaussianRasterizer or None.")
    geometry = ImageGeometry2D(plan.image_shape)
    true_rig, nominal_rig = _camera_rigs(plan)
    response = PhotometricResponse(
        saturation_level=1.0e12,
        shot_noise=plan.shot_noise,
        read_noise_std=plan.read_noise_std,
    )
    formation = ParticleImageFormation(rasterizer_, response)
    scenario_id = canonical_fingerprint(
        {
            "kind": "ptv-synthetic-case",
            "plan": plan.plan_id,
            "true_rig": true_rig.rig_id,
            "nominal_rig": nominal_rig.rig_id,
            "geometry": geometry.geometry_id,
            "formation": formation.formation_id,
        }
    )
    keys = jr.split(jr.key(plan.seed), 3 + plan.frame_count)
    positions, active = _world_trajectories(plan, keys[0], keys[1])
    amplitudes = jnp.full((plan.particle_capacity,), plan.particle_intensity)
    sigma = plan.particle_diameter / 2.3548200450309493

    image_frames: list[Array] = []
    ideal_frames: list[Array] = []
    projection_pixels: list[Array] = []
    projection_depth: list[Array] = []
    projection_valid: list[Array] = []
    projection_status: list[Array] = []
    visibility_frames: list[Array] = []
    rendering_frames: list[CameraStackRenderResult] = []
    for frame in range(plan.frame_count):
        dropout_keep = (
            jr.uniform(
                jr.fold_in(keys[2], frame),
                (plan.particle_capacity,),
            )
            >= plan.observation_dropout_probability
        )
        render_active = active[frame] & dropout_keep
        rendering = render_camera_stack(
            formation,
            true_rig,
            geometry,
            positions[frame],
            amplitudes,
            sigma,
            render_active,
            key=keys[3 + frame],
        )
        visible = _occlusion_visibility(
            plan,
            rendering.projection_pixels,
            rendering.projection_depth,
            rendering.projection_valid,
            render_active,
        )
        image_frames.append(rendering.images)
        ideal_frames.append(rendering.ideal_images)
        projection_pixels.append(rendering.projection_pixels)
        projection_depth.append(rendering.projection_depth)
        projection_valid.append(rendering.projection_valid)
        projection_status.append(rendering.projection_status)
        visibility_frames.append(visible)
        rendering_frames.append(rendering)

    trajectory_base = int(scenario_id[:8], 16) & 0x3FFFFFFF
    trajectory_ids = jnp.where(
        jnp.arange(plan.particle_capacity) < plan.particle_count,
        trajectory_base + jnp.arange(plan.particle_capacity),
        -1,
    ).astype(jnp.int32)
    images = jnp.stack(tuple(image_frames))
    ideal_images = jnp.stack(tuple(ideal_frames))
    projection_pixels_array = jnp.stack(tuple(projection_pixels))
    projection_depth_array = jnp.stack(tuple(projection_depth))
    projection_valid_array = jnp.stack(tuple(projection_valid))
    projection_status_array = jnp.stack(tuple(projection_status))
    visibility_array = jnp.stack(tuple(visibility_frames))
    finite = bool(
        jnp.all(jnp.isfinite(images))
        & jnp.all(jnp.isfinite(positions))
        & jnp.all(jnp.isfinite(projection_pixels_array))
        & jnp.all(jnp.isfinite(projection_depth_array))
    )
    raster_success = bool(
        jnp.all(jnp.stack(tuple(result.successful for result in rendering_frames)))
    )
    projection_complete = bool(jnp.all(~active[:, None, :] | projection_valid_array))
    status = (
        f"{plan.kind.value}-raster-warning"
        if not raster_success
        else f"{plan.kind.value}-projection-warning"
        if not projection_complete
        else plan.kind.value
    )
    evidence = SyntheticEvidence(
        plan.particle_capacity,
        plan.particle_count,
        finite=finite,
        status=status,
        source_id=scenario_id,
    )
    return PTVSyntheticCase(
        geometry,
        true_rig,
        nominal_rig,
        images,
        ideal_images,
        projection_pixels_array,
        projection_depth_array,
        projection_valid_array,
        projection_status_array,
        visibility_array,
        positions,
        active,
        trajectory_ids,
        tuple(rendering_frames),
        evidence,
        family_id=plan.family_id,
        plan_id=plan.plan_id,
        scenario_id=scenario_id,
    )


__all__ = ["PTVScenarioPlan", "PTVSyntheticCase", "generate_ptv_case"]
