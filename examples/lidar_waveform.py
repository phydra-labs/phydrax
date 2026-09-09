"""Generate a time-resolved atmospheric LiDAR return."""

import numpy as np

import phydrax as phx


contract = phx.SpatialCoordinateContract(
    phx.units.METER, coordinate_system="cartesian", reference_frame="sensor"
)
rays = phx.measurement.RaySampleSupport(
    np.zeros((1, 3)), np.asarray(((0, 0, 1),)), ("pulse-0",), contract
)
delay = phx.measurement.SampleTimeAxis(
    "delay", np.linspace(0.0, 10.0, 201), phx.units.SECOND
)
support = phx.measurement.WaveformSupport(rays, delay)
pulse = phx.measurement.PulseResponse(
    np.asarray((-0.1, 0.0, 0.1)), np.asarray((0.0, 1.0, 0.0)), phx.units.SECOND.unit_id
)
plan = phx.rendering.AtmosphericLidarPlan(
    support, pulse, np.asarray(((1.0, 2.0, 3.0),)), np.ones((1, 3)), wave_speed=2.0
)
result = plan.evaluate(np.full((1, 3), 0.1), np.ones((1, 3)))
if not bool(result.evidence.successful):
    raise RuntimeError("Atmospheric LiDAR waveform failed.")
print(
    {
        "received_energy": float(result.evidence.received_energy),
        "peak_delay": float(
            delay.sample_times[np.argmax(np.asarray(result.values[0, :, 0]))]
        ),
    }
)
