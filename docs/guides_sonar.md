# Sonar measurements

`SonarAcquisition` binds source and receiver geometry, sound speed, sample clock, frame, and acquisition identity. `SonarWaveformAsset` requires pressure-valued receiver-by-time data.

`DelayAndSumBeamformingPlan` computes bistatic source–point–receiver delays and nonperiodically samples calibrated pressure channels. Beamformed images remain derived products.

`XtfSideScanProvider` is an optional bounded host adapter for side-scan channels.
It requires an explicit positive pressure calibration in pascals per stored
count, retains channel padding through validity masks, and never infers
navigation, towfish pose, gain, or water-column semantics absent from the
selected backend profile.
