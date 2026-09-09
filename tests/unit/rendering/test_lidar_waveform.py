import equinox as eqx
import jax.random as jr
import numpy as np

import phydrax as phx


def _surface_and_rays():
    vertices = np.asarray(((-1, -1, 2), (1, -1, 2), (0, 1, 2), (0, 0, 4)), dtype=float)
    triangles = np.asarray(((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)))
    contract = phx.SpatialCoordinateContract(
        phx.units.METER, coordinate_system="cartesian", reference_frame="world"
    )
    metadata = phx.geometry.surface.SurfaceMetadata(
        source_id="lidar-target",
        source_revision="1",
        coordinate_contract=contract,
        provenance=("synthetic",),
    )
    surface = phx.geometry.surface.SurfaceModel.from_triangles(
        vertices, triangles, metadata, repair_orientation=True, orient_closed_outward=True
    ).prepare()
    rays = phx.measurement.RaySampleSupport(
        np.zeros((1, 3)),
        np.asarray(((0, 0, 1),)),
        ("pulse",),
        contract,
        far=np.asarray((10.0,)),
    )
    quantity = phx.measurement.QuantitySpec(
        "lidar", "range", "range", phx.units.METER, "physical.range"
    )
    prepared = phx.rendering.LidarSurfacePlan(
        surface,
        rays,
        quantity,
        phx.measurement.SamplingSemantics(phx.measurement.SpatialSamplingKind.EVENT),
    ).prepare()
    return vertices, rays, prepared


def _waveform_support(rays):
    axis = phx.measurement.SampleTimeAxis(
        "delay", np.linspace(0.0, 6.0, 121), phx.units.SECOND
    )
    return phx.measurement.WaveformSupport(rays, axis)


def _pulse():
    return phx.measurement.PulseResponse(
        np.asarray((-0.1, 0.0, 0.1)),
        np.asarray((0.0, 1.0, 0.0)),
        phx.units.SECOND.unit_id,
    )


def test_hard_surface_waveform_and_return_extraction_recover_delay():
    vertices, rays, surface = _surface_and_rays()
    support = _waveform_support(rays)
    pulse = _pulse()
    plan = phx.rendering.HardSurfaceLidarWaveformPlan(
        surface, support, pulse, pulse, wave_speed=2.0
    )
    result = eqx.filter_jit(plan.evaluate)(vertices, geometry_id="target")
    peak = int(np.argmax(np.asarray(result.values[0, :, 0])))
    np.testing.assert_allclose(support.delay_axis.sample_times[peak], 2.0, atol=0.05)
    returns = phx.rendering.LidarReturnExtractionPlan(pulse, 0.01, 2).evaluate(
        result, support
    )
    assert bool(returns.successful)
    np.testing.assert_allclose(returns.delays[0, 0], 2.0, atol=0.05)


def test_atmosphere_multipath_and_multiple_scattering_are_time_resolved():
    _, rays, _ = _surface_and_rays()
    support = _waveform_support(rays)
    pulse = _pulse()
    atmosphere = phx.rendering.AtmosphericLidarPlan(
        support, pulse, np.asarray(((1.0, 2.0, 3.0),)), np.ones((1, 3)), wave_speed=2.0
    )
    clear = eqx.filter_jit(atmosphere.evaluate)(np.zeros((1, 3)), np.ones((1, 3)))
    attenuated = atmosphere.evaluate(np.ones((1, 3)), np.ones((1, 3)))
    assert np.sum(np.asarray(clear.values)) > np.sum(np.asarray(attenuated.values))

    multipath = phx.rendering.SpecularLidarMultipathPlan(
        support, pulse, wave_speed=2.0, path_capacity=2
    ).evaluate(
        np.asarray(((2.0, 4.0),)), np.asarray(((1.0, 0.5),)), np.asarray(((True, True),))
    )
    assert bool(multipath.evidence.successful)
    assert np.count_nonzero(np.asarray(multipath.values)) >= 2

    scattered = phx.rendering.TimeResolvedMultipleScatteringPlan(
        support,
        packet_count=16,
        event_count=2,
        extinction=1.0,
        scattering_albedo=0.8,
        wave_speed=2.0,
    ).evaluate(jr.key(0))
    assert bool(scattered.evidence.successful)
    assert np.sum(np.asarray(scattered.values)) > 0.0
