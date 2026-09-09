"""Portable timing evidence for multimodal sensing operators."""

from __future__ import annotations

import argparse
import json

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._runtime import capture_environment, measure_repeated


def benchmark(*, smoke: bool) -> dict[str, object]:
    repeats = 2 if smoke else 10
    contract = phx.SpatialCoordinateContract(
        phx.units.METER, coordinate_system="cartesian", reference_frame="world"
    )

    refractive = phx.optics.geometric.StructuredRefractiveIndexField(
        np.ones((8, 8, 8)), (0, 0, 0), (1, 1, 1), contract, field_id="benchmark-index"
    )
    graded = phx.optics.geometric.GradedIndexRayPlan(
        refractive, 0.02, 16 if smoke else 64
    ).prepare()
    positions = jnp.asarray(((1.0, 1.0, 1.0), (2.0, 2.0, 1.0)))
    directions = jnp.asarray(((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)))
    graded_call = eqx.filter_jit(graded.integrate)
    graded_result, graded_times = measure_repeated(
        lambda: graded_call(positions, directions), warmup=1, repeats=repeats
    )

    rays = phx.measurement.RaySampleSupport(
        np.asarray(((-1.0, 0.5, 0.5),)),
        np.asarray(((1.0, 0.0, 0.0),)),
        ("ray",),
        contract,
        far=np.asarray((5.0,)),
    )
    projection = phx.imaging.tomography.ProjectionSupport(rays, (1,), ("view",))
    ct = phx.imaging.tomography.VoxelXRayTransformPlan(
        projection, (8, 1, 1), (0, 0, 0), (0.25, 1, 1)
    )
    volume = jnp.ones((8, 1, 1))
    ct_call = eqx.filter_jit(ct.forward)
    ct_result, ct_times = measure_repeated(
        lambda: ct_call(volume), warmup=1, repeats=repeats
    )

    coils = phx.imaging.mri.CoilSensitivityField(
        np.ones((2, 16, 16), dtype=np.complex64), ("c0", "c1"), field_id="benchmark-coils"
    )
    mri = phx.imaging.mri.CartesianMRIEncodingPlan(coils)
    image = jnp.ones((16, 16), dtype=jnp.complex64)
    mri_call = eqx.filter_jit(mri.forward)
    mri_result, mri_times = measure_repeated(
        lambda: mri_call(image), warmup=1, repeats=repeats
    )

    radar_acquisition = phx.sensing.FMCWAcquisition(
        77e9, 1e12, 1e-6, 1e-3, np.zeros((2, 3)), "benchmark-radar"
    )
    radar = phx.sensing.FMCWTransformPlan(
        radar_acquisition, 32, 16, propagation_speed=3e8
    )
    adc = jnp.ones((16, 32, 2), dtype=jnp.complex64)
    radar_call = eqx.filter_jit(radar.evaluate)
    radar_result, radar_times = measure_repeated(
        lambda: radar_call(adc), warmup=1, repeats=repeats
    )
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(16),
            phx.discretization.UniformAxisSpec(16),
        ),
        axis_names=("u", "v"),
    ).prepare(jnp.asarray(((-1.0, -1.0), (1.0, 1.0))))
    plane = phx.optics.wave.PlaneFieldSpace(
        grid, phx.geometry.RigidFrame.identity(3), "finite-window"
    )
    phase_screen = phx.imaging.RefractivePhaseScreenPlan(plane, 20.0)
    phase_call = eqx.filter_jit(phase_screen.evaluate)
    phase_result, phase_times = measure_repeated(
        lambda: phase_call(jnp.zeros(plane.shape)), warmup=1, repeats=repeats
    )

    delay = phx.measurement.SampleTimeAxis(
        "benchmark-delay", np.linspace(0.0, 6.0, 121), phx.units.SECOND
    )
    waveform_support = phx.measurement.WaveformSupport(rays, delay)
    pulse = phx.measurement.PulseResponse(
        np.asarray((-0.1, 0.0, 0.1)),
        np.asarray((0.0, 1.0, 0.0)),
        phx.units.SECOND.unit_id,
    )
    atmosphere = phx.rendering.AtmosphericLidarPlan(
        waveform_support,
        pulse,
        np.asarray(((1.0, 2.0, 3.0),)),
        np.ones((1, 3)),
        wave_speed=2.0,
    )
    atmosphere_call = eqx.filter_jit(atmosphere.evaluate)
    lidar_result, lidar_times = measure_repeated(
        lambda: atmosphere_call(jnp.zeros((1, 3)), jnp.ones((1, 3))),
        warmup=1,
        repeats=repeats,
    )

    if not all(
        bool(value)
        for value in (
            graded_result.evidence.successful,
            ct_result.evidence.successful,
            mri_result[1].successful,
            radar_result.successful,
            phase_result[1].successful,
            lidar_result.evidence.successful,
        )
    ):
        raise RuntimeError("A multimodal benchmark result failed its evidence contract.")
    return {
        "environment": capture_environment().to_dict(),
        "repeats": repeats,
        "graded_index": graded_times.to_milliseconds_dict(),
        "ct_projection": ct_times.to_milliseconds_dict(),
        "mri_encoding": mri_times.to_milliseconds_dict(),
        "fmcw_transform": radar_times.to_milliseconds_dict(),
        "wave_phase_screen": phase_times.to_milliseconds_dict(),
        "lidar_atmosphere": lidar_times.to_milliseconds_dict(),
        "successful": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(benchmark(smoke=arguments.smoke), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
