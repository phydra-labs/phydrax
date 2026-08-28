# SPH walls and free surfaces

`WallParticleGenerationPlan` samples a signed-distance geometry near its
boundary, projects samples to the surface, creates explicit layers, obtains
normals, and derives boundary volume from the wall--wall kernel sum. The quality
report records spacing, volume range, and normal defect; no empirical volume
multiplier is used.

`AdamiWallBoundaryPlan` extrapolates wall pressure from neighboring fluid
particles, inverts the barotropic material closure for wall density, and supports
no-slip or free-slip ghost velocity. Volume-based pressure and viscosity forces
produce an explicit wall reaction and `ParticleInteractionLedger`. Static and
prescribed wall motion are supported; position clamping and bounce-back are not.

`FreeSurfaceDetectionPlan` combines kernel completeness, color-gradient normal,
and an empty-cone test. It returns hard masks, smooth weights, confidence, and
ambiguity. `FreeSurfacePressurePlan` applies atmospheric pressure through an
explicit hard or smooth policy. `FreeSurfaceOperatorCorrectionPlan` provides
zeroth-order normalization with a denominator floor.

Hard detector and pressure masks have frozen-active-set derivatives. Smooth
weights are differentiable surrogates. Density renormalization remains a
separate accepted-step operation.
