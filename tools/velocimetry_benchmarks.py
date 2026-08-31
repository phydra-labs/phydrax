#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


@dataclass(frozen=True)
class VelocimetryBenchmarkCase:
    name: str
    problem_size: str
    compile_and_first_ms: float
    steady_ms: float
    successful: bool


@dataclass(frozen=True)
class VelocimetryBenchmarkReport:
    maturity: str
    backend: str
    device: str
    cases: tuple[VelocimetryBenchmarkCase, ...]

    @property
    def passed(self) -> bool:
        return bool(self.cases and all(case.successful for case in self.cases))


def _timed(function, arguments, *, repetitions: int):
    started = perf_counter()
    first = function(*arguments)
    jax.block_until_ready(first)
    first_ms = (perf_counter() - started) * 1e3
    started = perf_counter()
    result = first
    for _ in range(repetitions):
        result = function(*arguments)
    jax.block_until_ready(result)
    steady_ms = (perf_counter() - started) * 1e3 / repetitions
    return result, first_ms, steady_ms


def _rig(image_shape: tuple[int, int]):
    principal = ((image_shape[0] - 1) / 2, (image_shape[1] - 1) / 2)
    intrinsics = phx.velocimetry.camera.CameraIntrinsics(
        (24.0, 24.0),
        principal,
        image_shape=image_shape,
    )
    return phx.velocimetry.camera.CameraRig(
        tuple(
            phx.velocimetry.camera.CameraModel(
                intrinsics,
                pose=phx.velocimetry.camera.CameraPose(
                    phx.geometry.RigidFrame(
                        jnp.eye(3),
                        jnp.asarray((x, 0.0, 0.0)),
                    )
                ),
            )
            for x in (-0.5, 0.5)
        )
    )


def _piv_case(*, smoke: bool, repetitions: int):
    size = 32 if smoke else 64
    first = jr.normal(jr.key(1), (size, size))
    second = jnp.zeros_like(first).at[2:, 1:].set(first[:-2, :-1])
    geometry = phx.velocimetry.imaging.ImageGeometry2D(first.shape)
    pair = phx.velocimetry.imaging.ImagePair2D(first, second, geometry)
    plan = phx.velocimetry.piv.PIVPlan(
        (phx.velocimetry.piv.PIVPassPlan(16, 8, 4),),
        correlation_mode="extended",
        minimum_valid_fraction=0.5,
        minimum_peak_ratio=0.0,
        minimum_correlation=-1.0,
        minimum_neighbors=0,
        replacement_iterations=0,
        chunk_size=4,
    )
    prepared = plan.prepare(geometry)
    execute = jax.jit(lambda value: prepared.run(value).raw.displacement_rc)
    result, first_ms, steady_ms = _timed(execute, (pair,), repetitions=repetitions)
    valid = prepared.run(pair).raw.valid
    measured = jnp.median(result[valid], axis=0)
    successful = bool(jnp.allclose(measured, jnp.asarray((2.0, 1.0)), atol=0.25))
    return VelocimetryBenchmarkCase(
        "classical-piv",
        f"image={size}x{size},windows={prepared.report.window_counts[0]}",
        first_ms,
        steady_ms,
        successful,
    )


def _camera_case(*, repetitions: int):
    rig = _rig((32, 32))
    truth = jnp.asarray((0.1, -0.1, 5.0))
    rays = tuple(
        phx.velocimetry.camera.pixels_to_rays(
            camera,
            phx.velocimetry.camera.project_points(camera, truth[None]).pixels,
        )
        for camera in rig.cameras
    )
    origins = jnp.stack(tuple(ray.origins[0] for ray in rays))
    directions = jnp.stack(tuple(ray.directions[0] for ray in rays))
    valid = jnp.stack(tuple(ray.valid[0] for ray in rays))
    weights = jnp.ones((rig.capacity,))
    execute = jax.jit(
        lambda current_origins, current_directions: (
            phx.velocimetry.camera.triangulate_weighted_rays(
                current_origins,
                current_directions,
                valid,
                weights,
            ).point
        )
    )
    result, first_ms, steady_ms = _timed(
        execute,
        (origins, directions),
        repetitions=repetitions,
    )
    return VelocimetryBenchmarkCase(
        "camera-triangulation",
        f"cameras={rig.capacity},points=1",
        first_ms,
        steady_ms,
        bool(jnp.allclose(result, truth, atol=1e-7)),
    )


def _raster_case(*, smoke: bool, repetitions: int):
    size = 32 if smoke else 64
    capacity = 16 if smoke else 64
    geometry = phx.velocimetry.imaging.ImageGeometry2D((size, size))
    rasterizer = phx.velocimetry.imaging.GaussianRasterizer(4, cutoff=3.0)
    positions = jr.uniform(jr.key(2), (capacity, 2)) * (size - 1)
    amplitudes = jnp.ones((capacity,))
    sigma = jnp.ones((capacity,))
    active = jnp.ones((capacity,), dtype=bool)
    execute = jax.jit(
        lambda current_positions: (
            rasterizer.render(
                geometry,
                current_positions,
                amplitudes,
                sigma,
                active,
            ).image
        )
    )
    result, first_ms, steady_ms = _timed(
        execute,
        (positions,),
        repetitions=repetitions,
    )
    return VelocimetryBenchmarkCase(
        "particle-image-raster",
        f"image={size}x{size},particles={capacity}",
        first_ms,
        steady_ms,
        bool(jnp.all(jnp.isfinite(result)) & (jnp.sum(result) > 0.0)),
    )


def _learned_case(*, smoke: bool, repetitions: int):
    size = 8 if smoke else 16
    level_count = 2
    plan = phx.velocimetry.piv.LearnedDensePIVPlan(
        (size, size),
        level_count=level_count,
        search_radius=1,
        cost_volume_chunk_size=4,
    )
    model = phx.velocimetry.piv.CorrelationPyramidPIV(
        plan,
        feature_channels=3 if smoke else 8,
        refinement_channels=4 if smoke else 12,
        key=jr.key(3),
    )
    first = jr.normal(jr.key(4), (size, size, 1))
    second = jnp.roll(first, 1, axis=1)
    execute = jax.jit(
        lambda first_image, second_image: (
            model(plan.prepare(first_image, second_image)).displacement_rc
        )
    )
    result, first_ms, steady_ms = _timed(
        execute,
        (first, second),
        repetitions=repetitions,
    )
    return VelocimetryBenchmarkCase(
        "learned-dense-piv",
        f"image={size}x{size},levels={level_count},radius=1",
        first_ms,
        steady_ms,
        bool(jnp.all(jnp.isfinite(result))),
    )


def run_velocimetry_benchmarks(*, smoke: bool = False):
    repetitions = 1 if smoke else 5
    cases = (
        _piv_case(smoke=smoke, repetitions=repetitions),
        _camera_case(repetitions=repetitions),
        _raster_case(smoke=smoke, repetitions=repetitions),
        _learned_case(smoke=smoke, repetitions=repetitions),
    )
    device = jax.devices()[0]
    return VelocimetryBenchmarkReport(
        maturity="smoke" if smoke else "benchmark",
        backend=jax.default_backend(),
        device=f"{device.platform}:{device.device_kind}",
        cases=cases,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_velocimetry_benchmarks(smoke=arguments.smoke)
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
