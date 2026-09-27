# Vortex field backends

## Authorities

Gaussian and Gaussian-erf direct sums remain the regularized authorities.
Singular and Rosenhead cores provide independent near/far limits.
`PeriodicVortexEwaldPlan` supplies screened real-image plus reciprocal periodic
reference values and rejects incompatible nonzero mean vorticity.
`FreeSpaceVortexFFTPlan` zero-pads every spatial axis and reports boundary
vorticity contamination rather than silently wrapping it.

## Particle mesh and P3M

Periodic VIC retains its assignment/filter identity. It is not declared
Gaussian-equivalent. `CorrectedP3MPlan` applies a screened direct near field and
regularized-core correction to the mesh far field; near/far work, assignment,
spectral, cutoff, and correction defects remain separate evidence.

## Hierarchical FMM

`VortexFMMPlan` supports an adaptive-octree `execution="level_octree"` route and
a fixed-envelope `execution="plane_dual"` route. Adaptive-octree execution
prepares a `phydrax.discretization.spatial.AdaptiveOctree` over the reference
sources, subdividing cells with more than `leaf_capacity` sources down to
`depth`, and locates arbitrary targets in its leaves at evaluation. Its U/V/W/X
interaction lists keep one cell of clearance for sources displaced by up to
`maximum_reference_displacement`: V routes translate monopole and first-moment
payloads to local expansions, X routes convert source points to local
expansions, W routes evaluate multipoles at targets, and U routes remain exact
regularized pairs. Plane execution freezes separate source and target Morton
schedules, aggregates monopole and first-moment vector payloads, and uses
deterministic source-to-target far routes. Both preserve exact
Gaussian/Gaussian-erf near interactions, velocity gradients, vorticity, core
radius handling, explicit self identity, and geometric-tail evidence.

Plane mode requires reference targets for arbitrary-target execution and fails
when source or target motion leaves its padded reference envelope. Expansion
order zero or one remains explicit. Queue, far, near, and node capacities are
accepted at preparation; no truncated route publishes a successful field.

The older fixed-leaf approximation is named `FixedClusterVortexPlan2D`; it is
not an FMM.

## Arbitrary targets and devices

Direct, Ewald, FMM, ring/sheet, and Fourier interpolation routes accept explicit
`VortexTargetState` values. Probes are never inserted into source storage.
`VortexShardingPolicy` distinguishes target, source, grid, and tree-leaf
sharding. Collective accumulation can be fast, deterministic, or compensated;
device and memory preflight occurs before execution.
