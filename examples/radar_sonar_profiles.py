"""Evaluate explicit FMCW radar and sonar acquisition profiles."""

import numpy as np

import phydrax as phx


radar_acquisition = phx.sensing.FMCWAcquisition(
    77e9, 1e12, 1e-6, 1e-3, np.zeros((2, 3)), "radar"
)
radar = phx.sensing.FMCWTransformPlan(
    radar_acquisition, 16, 8, propagation_speed=3e8
).evaluate(np.ones((8, 16, 2), dtype=np.complex64))
axis = phx.measurement.SampleTimeAxis(
    "sonar-time", np.linspace(0.0, 1.0, 64), phx.units.SECOND
)
sonar_acquisition = phx.sensing.SonarAcquisition(
    np.zeros(3), np.asarray(((0, 0, 0), (0.1, 0, 0))), 1500.0, axis, "water", "sonar"
)
waveforms = np.zeros((2, 64))
waveforms[:, 20] = 1.0
sonar = phx.sensing.DelayAndSumBeamformingPlan(
    sonar_acquisition, np.asarray(((1.0, 0.0, 0.0),))
).evaluate(waveforms)
if not bool(radar.successful & sonar.successful):
    raise RuntimeError("Radar or sonar profile failed.")
print(
    {
        "radar_shape": radar.range_doppler_channels.shape,
        "sonar_image": np.asarray(sonar.image).tolist(),
    }
)
