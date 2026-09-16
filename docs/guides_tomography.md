# Computed tomography

`ProjectionSupport` binds source rays, projection shape, view identities, acquisition frame, and timing. Projection detector counts and logarithmic attenuation integrals remain different quantities.

`VoxelXRayTransformPlan` precomputes exact Siddon voxel segment routes. Forward and transpose use the same segment table. `TetrahedralXRayTransformPlan` provides the corresponding piecewise-constant tetrahedral route.

`BeerLambertPlan` is the monoenergetic mathematical route. The diagnostic
polychromatic route is intentionally more explicit: `MaterialBasisProjectionPlan`
projects density-weighted material fractions to areal masses, while
`PolychromaticDetectorPlan` binds an ordered diagnostic coefficient table to
per-view tube spectrum, filtration, bowtie, AEC, detector response, exposure
basis, and explicitly labelled scatter. Relative acquisition produces only
relative expected signal; absolute counts require absolute exposure calibration.

`FilteredBackprojectionPlan` is limited to parallel-beam geometry. `IterativeCTPlan`
reuses the matched projector/transpose and returns residual history. Cone beam
reconstruction, material decomposition inversion, and native high-fidelity
scatter transport require separate explicit plans.
