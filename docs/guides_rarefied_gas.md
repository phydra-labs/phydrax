# Rarefied gas dynamics

Phydrax distinguishes physical molecular-velocity kinetics from low-order D2V/LBM reference-population methods.

`MolecularVelocityQuadrature` always stores three-dimensional molecular velocities and independent physical integration weights. Its spatial streaming projection may have one, two, or three dimensions. `PopulationUpwindFluxPlan` streams every population using its projected molecular velocity.

`PositiveDiscreteMaxwellianPlan` solves the five-moment entropy-dual problem for a strictly positive discrete equilibrium. It matches mass, three momentum components, and energy on the actual finite velocity quadrature rather than sampling a continuous Maxwellian and accepting quadrature drift.

`MonatomicBGKCollisionPlan` uses relaxation time equal to dynamic viscosity divided by pressure and advances homogeneous relaxation analytically. It reports moment defect, entropy change, positivity, and relaxation time. `ShakhovCollisionPlan` provides the selected non-unit-Prandtl extension; positivity and invariant correction are explicit evidence.

`MaxwellGasSurfaceBoundary` blends exact specular routing and diffuse wall emission. Construction requires a velocity set closed under reflection for the selected wall normal, and diffuse density enforces zero normal mass flux.

`KineticBreakdownPlan` combines a local Knudsen estimate and distribution-to-equilibrium defect. It is the physical eligibility seam for later continuum-kinetic routing; shock sensing alone is not a Knudsen model.

`KineticSyntheticAccelerationPlan` exposes the deterministic micro-macro correction used after a continuum synthetic solve. It preserves a zero-invariant micro distribution, limits only for positivity, and reports the resulting moment defect. It is not DSMC and performs no stochastic reconstruction or dynamic particle repartitioning.

## Continuum gradient-length evidence

`GradientLengthKnudsenPlan` evaluates mean-free-path-scaled gradients of density,
heavy temperature, velocity, every species mass fraction, and every explicit mode
temperature. Enter and leave thresholds are separate, so
`RarefactionHysteresisState` does not chatter in the transition band. The triggering
component and all componentwise estimates remain observable.

This is continuum-breakdown evidence only. `KineticBreakdownPlan` instead measures a
kinetic population's relaxation scale and distribution defect. Neither plan performs
DSMC, reconstructs particles, changes topology, or silently switches the governing
equations.

## Cell-local DSMC

`DSMCProductionPlan` now executes streaming and physical face events before a
cell-local no-time-counter schedule. `DSMCNTCSchedulePlan` retains one fractional
candidate remainder and one majorant per cell. Candidate count depends on local
occupancy, simulator weight, cell volume, step size, and the accepted
cross-section-speed majorant. Occupancy, event, injection, and particle capacities
reject the whole candidate; no list is truncated.

`DSMCVHSCollisionPlan` and `DSMCVSSCollisionPlan` are distinct. VHS is isotropic;
VSS uses its declared angular parameter. Pair data is an explicit symmetric
`DSMCPairCollisionParameters` table. Collision events run sequentially, so a
particle selected more than once observes its latest accepted velocity.
`DSMCInternalReactionPlan` sees only accepted collisions, conserves declared
mass/charge/elements for supported two-to-two channels, performs microcanonical
rotational redistribution, and leaves vibration frozen.

Specular and diffuse Maxwell walls, open deletion, and half-range equilibrium
reservoir injection retain extensive mass, momentum, and energy ledgers.
`DSMCMomentPlan` returns weighted species moments, pressure tensor, temperatures,
block covariance, and unresolved-statistics evidence.

## Conservative continuum coupling

`ContinuumDSMCInterfacePlan` uses measured DSMC particle crossings as the one
common interface flux and applies exact equal-opposite extensive exchanges. Sampling
covariance is diagnostic; it is never blended against an implicit continuum
variance. `ContinuumToDSMCConversionPlan` and
`DSMCToContinuumReductionPlan` preserve their declared extensive moments.
`HybridOwnershipEpochPlan.classify` returns a request only. The host must provide
admitted conversion and reduction evidence before `transition_epoch` changes
ownership.

## First-order continuum walls

`MaxwellSmoluchowskiContinuumWallPlan` supplies Maxwell velocity slip,
Smoluchowski temperature jump, and thermal creep to structured, mapped, and
triangle viscous finite-volume fluxes. Tangential-momentum and thermal
accommodation are independent. The wall uses the equation-owned ideal-mixture
mean-free-path and rejects unsupported gas closures, ALE motion, excessive
Knudsen number, inadequate wall resolution, nonfinite states, and normal wall
motion.

All of these capabilities are candidate support. Synthetic conservation and
analytic-limit checks are not experimental rarefied-flow validation.
