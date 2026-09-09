# Radar measurement profiles

`PolarVolumeSupport` retains azimuth, elevation, range gates, and radar frame. `CfRadialProvider` lowers supported weather-radar moments into separate quantity fields; reflectivity, radial velocity, spectrum width, differential reflectivity, correlation ratio, and differential phase are not merged.

`FMCWAcquisition` declares SI carrier frequency, chirp slope, fast/slow timing,
and receiver geometry. `FMCWTransformPlan` requires propagation speed in metres
per second, applies explicit windows, range FFT, Doppler FFT, and returns complex
channel and integrated-power products with physical range and Doppler axes.

`AutomotiveRadarProfile` admits derived range/azimuth/radial-velocity/RCS detections. It does not claim raw ADC or waveform authority.
