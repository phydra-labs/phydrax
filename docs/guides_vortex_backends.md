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

`VortexFMMPlan` builds sparse occupied source/target prefixes through the shared
Morton level-octree substrate. It performs leaf P2M, bottom-up M2M, same-level
M2L, parent-to-child L2L, local linear evaluation, and direct regularized
neighbor interactions without allocating a complete quadtree or octree.
Expansion order zero or one is explicit. Tree bounds, reference displacement,
interaction capacities, interaction counts, and geometric tail bounds fail
closed.

The older fixed-leaf approximation is named `FixedClusterVortexPlan2D`; it is
not an FMM.

## Arbitrary targets and devices

Direct, Ewald, FMM, ring/sheet, and Fourier interpolation routes accept explicit
`VortexTargetState` values. Probes are never inserted into source storage.
`VortexShardingPolicy` distinguishes target, source, grid, and tree-leaf
sharding. Collective accumulation can be fast, deterministic, or compensated;
device and memory preflight occurs before execution.
