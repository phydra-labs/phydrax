# Measurement collections, clocks, and frames

`MeasurementCollection` groups heterogeneous assets without implying common support, units, frames, clocks, or calibration. Every asset has one explicit role and every cross-asset relation has typed evidence.

`AffineClockMap` and `PiecewiseClockMap` retain source and target clock identities, calibration, validity interval, fit residual, and extrapolation evidence. Extrapolation is refused unless requested explicitly.

`FrameTransformGraph` stores time-dependent `FrameTransformTimeline` edges. Host preparation chooses one unambiguous shortest route; compiled execution only interpolates and composes that route. Translation is linear and rotation uses shortest-arc quaternion interpolation. Disconnected or ambiguous routes fail before execution.

`MeasurementSelectionPlan` selects bounded assets and first-axis sample intervals before allocation. Selected products retain parent collection and asset lineage. It never aligns heterogeneous data implicitly.
