# Particle methods

Phydrax represents particles as stable material entities with evolving numeric
state. Particle identity, current position, and proximity are deliberately
separate:

```text
ParticleSetPlan
  -> ParticleDiscretization         stable IDs, support, mass measure
current position                    temporal state
ParticleNeighborhoodPlan
  -> ParticlePairRelation           execution routes for one method
```

## Material support

`ParticleSetPlan` prepares a zero-dimensional `EntitySet`, a `PointTopology`
without neighborhoods, position and velocity `particle_value` spaces, and a
physical material-mass measure. Current coordinates are not part of the plan or
support identity. Moving a particle therefore changes numeric state without
changing the material topology.

Particle IDs are stable physical identities. Logical particle order is distinct
from any future locality-oriented execution permutation. Static `EntitySubset`
values label material regions or observation groups. The active mask belongs to
topological identity; activating, deleting, emitting, splitting, or merging a
particle requires a future explicit topology event rather than an in-trajectory
array mutation.

## Pair relations

`DenseParticleNeighborhoodPlan` prepares every canonical same-set pair under an
explicit `maximum_pairs` budget. `CellListParticleNeighborhoodPlan` instead
prepares fixed cell geometry, neighboring-cell routes, particle-per-cell
capacity, candidate-slot capacity, and pair capacity. At runtime it sorts by
cell and stable particle ID, builds a fixed particle table, and packs canonical
unordered pairs without changing public logical particle order.

`ParticleNeighborhoodState` carries the realized relation, logical/storage
permutations, cell counts and offsets, actual pair count, maximum occupancy,
and independent cell-overflow, pair-overflow, and nonperiodic-domain status.
Overflow is fail-closed: truncated routes remain inspectable but cannot be used
by SPH dynamics.

`ParticleBox` implements branchwise minimum-image displacement on selected
periodic axes. Nonperiodic axes use half-open bounds. Pair geometry is finite at
coincident positions and returns a zero direction there. Conservative pair
exchange evaluates one unordered interaction and scatters equal and opposite
endpoint contributions.

`ParticleExecutionPolicy` provides `dense_pairs` and `cell_edge_list`
realizations with fast, deterministic, or compensated accumulation. Dense
execution remains the correctness authority. `particle_graph_view` converts the
exact existing candidate or physical interaction relation to fixed-capacity
`GraphIR` without a second search.

## Precision

`ParticlePrecisionPolicy` distinguishes geometry, pair evaluation, accumulation,
certification, and output dtypes. Prepared SPH dynamics retain a complete
precision-evidence envelope. Float64 is the scientific default; reduced storage
or evaluation precision must be requested explicitly.

## Differentiation

The implemented contract is the fixed discrete particle program:

- particle positions, pair displacements, kernels, density, energy, and force are
  ordinary differentiable JAX calculations;
- dense candidate indices are static, while cell IDs, sorting, and packed
  cell-list routes are stopped-gradient decisions;
- compact-support membership and periodic minimum-image choices are branchwise;
- particle activation and capacity changes are unsupported inside a differentiated
  trajectory;
- inactive padding is selected to finite values before arithmetic.

No straight-through topology estimator is used.

## Current limits

The substrate supports a fixed population, one particle set, fixed search and
smoothing radii, dense or fixed-capacity cell-list pairs, and periodic or
nonperiodic boxes. Cached Verlet neighborhoods, fused cell-range traversal,
wall particles, particle emission, edge-local contact history, distribution,
and particle–continuum transfer remain future prepared backends or method
families over these contracts.
