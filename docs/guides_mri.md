# MRI k-space encoding and reconstruction

`KSpaceSupport` stores normalized-radian k-vectors, coil identities, trajectory, frame, and optional sample times. `KSpaceAsset` requires complex measurement storage.

`CartesianMRIEncodingPlan` applies coil sensitivities and orthonormal FFT sampling with an exact adjoint. `NUFFTMRIEncodingPlan` uses matched Type-2 and Type-1 NUFFT actions. Density compensation is not part of the adjoint.

`CoilNoiseCovariance` validates Hermitian positive-definite coil covariance and exposes prewhitening. `CGSensePlan` solves the normal equations; `RegularizedMRIPlan` adds explicit complex soft shrinkage. Complex images remain authoritative; magnitude and phase are derived products.

`OffResonanceMRIEncodingPlan` includes sample-time phase and translational motion. `PhaseContrastMRIPlan`, `QuantitativeMRIPlan`, and `BlochSequencePlan` provide explicit velocity phase, relaxation signal, and fixed-step magnetization dynamics. They do not infer pulse-sequence metadata.
