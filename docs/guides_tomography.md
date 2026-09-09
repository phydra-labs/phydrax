# Computed tomography

`ProjectionSupport` binds source rays, projection shape, view identities, acquisition frame, and timing. Projection detector counts and logarithmic attenuation integrals remain different quantities.

`VoxelXRayTransformPlan` precomputes exact Siddon voxel segment routes. Forward and transpose use the same segment table. `TetrahedralXRayTransformPlan` provides the corresponding piecewise-constant tetrahedral route.

`BeerLambertPlan` maps nonnegative line integrals to transmitted and detector signal with dark level, gain, saturation, and explicit scatter. `PolychromaticBeerLambertPlan` integrates declared spectral channels without pretending a monoenergetic log transform.

`FilteredBackprojectionPlan` is limited to parallel-beam geometry. `IterativeCTPlan` reuses the matched projector/transpose and returns residual history. Cone beam, material decomposition, and high-fidelity scatter require separate explicit plans.
