# SIDM interaction and regime profiles

SIDM profiles are explicit model families. Phydrax never selects rare, frequent, fluid,
equal-weight or weighted evolution automatically.

## Microscopic species and differential kernels

`DarkSectorSpeciesPlan` stores microscopic rest mass, internal energy, degeneracy,
conserved charges, units and stable identity. Macro gravitational mass is not a
microscopic collision mass.

`TwoBodyDifferentialKernelPlan` tabulates a bounded differential cross section on
relative speed, `mu = cos(theta)`, and optional azimuth with explicit solid-angle
measure, identical-particle and screening conventions, provenance and support. It
computes total, transfer, viscosity and indistinguishable-particle modified moments and
uses bounded monotone inverse-CDF sampling.

`SmallAngleSplitPlan` creates complementary rare and frequent angular domains. It proves
no angular gap or overlap and reports reconstructed moment residuals. The current
constant-isotropic SIDM plan is a specialization of this kernel family.

## Anisotropic rare scattering

The event probability uses the total cross section. The recoil direction is sampled from
the differential kernel. Stable pair IDs own randomness; per-pair and per-particle
probability, Knudsen, capacity, support, conservation and rollback gates remain active.
An isotropic transfer-cross-section surrogate is not an anisotropic production claim.

## Unequal-weight rare scattering

`WeightedSIDMPacketState` separates microscopic mass, statistical weight, gravitational
macro mass and canonical macro momentum. Runtime PM source mass is part of the state.

For packets i and j, retained subpacket weight is `q = min(w_i, w_j)` and the candidate
rate uses the maximum packet weight. A selected event keeps any residual heavy packet,
allocates deterministic child slots and stable child/parent/event lineage, scatters the
equal-weight retained pieces, and closes mass, canonical momentum and kinetic energy.
Capacity failure rolls back all state.

`WeightedPacketResamplingPlan` is accepted-boundary only. It declares preserved mass,
centroid, momentum, kinetic/covariance moments and all higher-moment loss. It fails if no
bounded nonnegative construction is available.

## Frequent small-angle scattering

`FrequentSmallAngleSIDMPlan` consumes the small-angle kernel moments and applies
pair-owned antisymmetric drag plus paired transverse stochastic diffusion. It reports
positive-semidefinite diffusion, timestep, identity, momentum and energy evidence. A
rare-tail plus frequent-core composition is valid only when both are complementary
parts of the same declared microscopic kernel.

## Gravothermal fluid profile

`GravothermalSIDMPlan` is a separate isolated, spherical, radial density/velocity-
dispersion/conductivity closure. Calibration artifact, boundary conditions, mean-free-
path regime and conservative energy update are explicit. It is not generic 3-D gas
dynamics and is not entered automatically from particles.

## Differentiation

Kernel interpolation and smooth fixed-pair moment evaluation may be differentiable.
Pair selection, child allocation, resampling, random events, regime selection and fluid
projection are not pathwise differentiable unless a separately qualified estimator is
returned.
