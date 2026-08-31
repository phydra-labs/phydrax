#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


@dataclass(frozen=True)
class PIVQualificationEvidence:
    median_displacement_rc: tuple[float, float]
    displacement_error: float
    valid_fraction: float
    successful: bool


@dataclass(frozen=True)
class CameraQualificationEvidence:
    reconstructed_xyz: tuple[float, float, float]
    reconstruction_error: float
    valid: bool
    successful: bool


@dataclass(frozen=True)
class STBQualificationEvidence:
    reconstructed_xyz: tuple[float, float, float]
    reconstruction_error: float
    initial_residual_energy: float
    final_residual_energy: float
    born_count: int
    successful: bool


@dataclass(frozen=True)
class VelocimetryQualificationReport:
    maturity: str
    piv: PIVQualificationEvidence
    camera: CameraQualificationEvidence
    stb: STBQualificationEvidence

    @property
    def passed(self) -> bool:
        return self.piv.successful and self.camera.successful and self.stb.successful


def _rig(image_shape: tuple[int, int]):
    principal = ((image_shape[0] - 1) / 2, (image_shape[1] - 1) / 2)
    intrinsics = phx.velocimetry.camera.CameraIntrinsics(
        (24.0, 24.0),
        principal,
        image_shape=image_shape,
    )
    cameras = tuple(
        phx.velocimetry.camera.CameraModel(
            intrinsics,
            pose=phx.velocimetry.camera.CameraPose(
                phx.geometry.RigidFrame(jnp.eye(3), jnp.asarray((x, 0.0, 0.0)))
            ),
        )
        for x in (-0.5, 0.5)
    )
    return phx.velocimetry.camera.CameraRig(cameras)


def _piv_qualification(*, smoke: bool):
    size = 32 if smoke else 48
    window = 16
    first = jr.normal(jr.key(12), (size, size))
    second = jnp.zeros_like(first).at[2:, :-1].set(first[:-2, 1:])
    geometry = phx.velocimetry.imaging.ImageGeometry2D(first.shape)
    plan = phx.velocimetry.piv.PIVPlan(
        (phx.velocimetry.piv.PIVPassPlan(window, window // 2, 4),),
        correlation_mode="extended",
        minimum_valid_fraction=0.5,
        minimum_peak_ratio=0.0,
        minimum_correlation=-1.0,
        minimum_neighbors=0,
        replacement_iterations=0,
        chunk_size=4,
    )
    result = phx.velocimetry.piv.piv(first, second, plan, geometry=geometry)
    measured = jnp.median(result.raw.displacement_rc[result.raw.valid], axis=0)
    truth = jnp.asarray((2.0, -1.0))
    error = jnp.sqrt(jnp.sum((measured - truth) ** 2))
    valid_fraction = jnp.mean(result.raw.valid)
    return PIVQualificationEvidence(
        tuple(float(value) for value in measured),
        float(error),
        float(valid_fraction),
        bool(error < 0.25 and valid_fraction > 0.5),
    )


def _camera_qualification(rig):
    truth = jnp.asarray((0.1, -0.1, 5.0))
    pixels = tuple(
        phx.velocimetry.camera.project_points(camera, truth[None]).pixels[0]
        for camera in rig.cameras
    )
    rays = tuple(
        phx.velocimetry.camera.pixels_to_rays(camera, pixel[None])
        for camera, pixel in zip(rig.cameras, pixels, strict=True)
    )
    origins = jnp.stack(tuple(ray.origins[0] for ray in rays))
    directions = jnp.stack(tuple(ray.directions[0] for ray in rays))
    valid = jnp.stack(tuple(ray.valid[0] for ray in rays))
    result = phx.velocimetry.camera.triangulate_weighted_rays(
        origins,
        directions,
        valid,
        jnp.ones((rig.capacity,)),
    )
    error = jnp.sqrt(jnp.sum((result.point - truth) ** 2))
    return CameraQualificationEvidence(
        tuple(float(value) for value in result.point),
        float(error),
        bool(result.valid),
        bool(result.valid & (error < 1e-7)),
    )


def _stb_qualification(rig, geometry):
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
    prepared = phx.velocimetry.tracking.prepare_stb(
        phx.velocimetry.tracking.STBPlan(
            ipr,
            phx.velocimetry.tracking.ShakePlan(
                iterations=1,
                position_step=0.05,
                amplitude_step=0.05,
            ),
            phx.velocimetry.tracking.TrackLinkPlan(1, maximum_missed=0),
            promotion_steps=1,
        ),
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
    )
    truth = jnp.asarray([[0.1, 0.0, 6.0]])
    observed = phx.velocimetry.imaging.render_camera_stack(
        formation,
        rig,
        geometry,
        truth,
        jnp.asarray((18.0,)),
        jnp.ones((1,)),
        jnp.asarray((True,)),
    ).images
    initial_energy = jnp.sum(observed * observed)
    result = phx.velocimetry.tracking.stb_step(prepared, state, observed, 1.0)
    reconstructed = result.state.positions_xyz[0]
    error = jnp.sqrt(jnp.sum((reconstructed - truth[0]) ** 2))
    final_energy = jnp.sum(result.residual * result.residual)
    return STBQualificationEvidence(
        tuple(float(value) for value in reconstructed),
        float(error),
        float(initial_energy),
        float(final_energy),
        int(result.evidence.born_count),
        bool(
            result.successful
            & (result.evidence.born_count == 1)
            & (error < 0.2)
            & (final_energy < initial_energy)
        ),
    )


def run_velocimetry_qualification(*, smoke: bool = False):
    geometry = phx.velocimetry.imaging.ImageGeometry2D((32, 32))
    rig = _rig(geometry.image_shape)
    return VelocimetryQualificationReport(
        maturity="smoke" if smoke else "qualified",
        piv=_piv_qualification(smoke=smoke),
        camera=_camera_qualification(rig),
        stb=_stb_qualification(rig, geometry),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_velocimetry_qualification(smoke=arguments.smoke)
    payload = asdict(report) | {"passed": report.passed}
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n")
    print(rendered)
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
