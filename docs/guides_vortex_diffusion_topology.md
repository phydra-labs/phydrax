# Vortex diffusion and topology

## Diffusion

`GaussianParticleStrengthExchangePlan` is the dense authority.
`GaussianPSENeighborhoodPlan` consumes a prepared particle relation and applies
the same antisymmetric integrated-strength flux over fixed pair routes.
`GaussianCoreSpreadingPlan` evolves core radius consistently with Gaussian
variance growth and reports overlap quality and a diffusive step bound.
`GaussianRBFReinitializationPlan` reconstructs strengths through a native
least-squares solve. `VortexRedistributionPlan` matches circulation and selected
first/second moments on candidate points.

`WallCorrectedPSEPlan` provides named mirror and one-sided support policies. Its
ledger separates conservative bulk exchange from prescribed or induced wall
flux.

## Three-dimensional formulations

`ClassicVPMFormulation` applies stretching plus selected diffusion.
`ReformulatedVPMFormulation` evolves strength and core in the differential
state. `VortexLESPlan` exposes constant or dynamic coefficients, filter identity,
energy transfer, dissipation, and optional backscatter. Relaxation occurs only
at the declared accepted-step cadence.

`BaroclinicVortexFormulation` adds density-gradient/pressure-gradient and
buoyancy sources. It remains a variable-density incompressible formulation;
compressible dilatation and shocks require separate coupled state.

## Population events

`VortexPopulationPlan` owns stable IDs, lineage, age, core, volume, and fixed
capacity. Insert, deactivate, merge, split, and prune are atomic candidate/
accepted transitions. `CompleteVortexRemeshPlan` supports 2-D/3-D tensor grids,
degree one through three, periodic or bounded routing, obstacle clearance, and
moment evidence. `VortexCapacityGrowthPlan` migrates exact numeric state into a
new epoch and requests backend rebuild rather than mutating a compiled support.
