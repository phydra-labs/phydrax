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

Particle IDs are stable physical identities. Logical particle order is distinct from any locality-oriented execution permutation. Static `EntitySubset` values label material regions or observation groups. The prepared support fixes every potential slot. Process operations may activate or deactivate preallocated runtime body state, but they never resize or reorder the support.

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
by particle dynamics.

`ParticleBox` implements branchwise minimum-image displacement on selected
periodic axes. Nonperiodic axes use half-open bounds. Pair geometry is finite at
coincident positions and returns a zero direction there. Methods that require a
defined contact normal, including DEM, reject an overlapping coincident pair.
Conservative pair exchange evaluates one unordered interaction and scatters
equal and opposite endpoint contributions.

`ParticleExecutionPolicy` provides `dense_pairs` and `cell_edge_list`
realizations with fast, deterministic, or compensated accumulation. Dense
execution remains the correctness authority. `particle_graph_view` converts the
exact existing candidate or physical interaction relation to fixed-capacity
`GraphIR` without a second search.

`ParticlePairKeySpace` assigns collision-free keys from stable endpoint
identities. `match_particle_pair_keys` remaps edge-local state when a rebuilt
cell-list relation moves a physical pair to another route slot. This is the
persistent-state substrate for frictional contact.

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
- particle activation and fixed-pool topology events are discrete stopped-gradient decisions;
- inactive padding is selected to finite values before arithmetic.

No straight-through topology estimator is used.

## Particle-grid transfer

[`ParticleGridSplatPlan`](guides_particle_splatting.md) prepares multilinear or
degree-one through degree-three tensor B-spline transfer from material particles
to nodal, cell, face, or edge layouts. Extensive deposition, intensive
reconstruction, grid-to-particle gather, route moments, support truncation,
balance evidence, and piecewise routing derivatives share one prepared contract.

## Fluid method families

Conservative barotropic SPH compiles position and canonical momentum to a
separable Hamiltonian problem. [Weakly compressible SPH](guides_wcsph.md) uses a
first-order position/velocity state, explicit summation or continuity density,
optional Morris physical viscosity, and SSPRK integration. Both methods share
the same kernels, pair relations, dense authority, cell-list execution, and
`GraphIR` views.

## Discrete element method

[Soft-contact DEM](guides_discrete_element_method.md) composes rigid-sphere properties, stable pair-state remapping, normal, cohesion, tangential, and rotational contact channels, barriers, multicontact correction, and structured fixed-step state. A separate superquadric route shares neighborhoods and rigid-body integration. DEM reuses the same dense authority, cell-list execution, precision, and reduction policies as SPH; it does not introduce a second particle search API.

## Internal conversion and process operations

[Particle internal transport](guides_particle_internal_transport.md) maps selected owners to homogeneous radial finite-volume batches with extensive energy and species state. [Particle thermochemistry](guides_particle_thermochemistry.md) adds typed phases, elements, reactions, evaporation, shrinking-core conversion, and morphology.

[Reactive CFD–DEM](guides_reactive_cfd_dem.md) composes conversion and mechanics through conservative particle-grid exchange and atomic macro windows. Fixed-pool insertion, removal, residence, mass-flow, fragmentation, and deactivation use explicit process events without changing array shape.

## Current limits

The substrate supports fixed-capacity populations, dense or cell-list pairs, certified cached Verlet neighborhoods, periodic or nonperiodic boxes, fused reductions, persistent same-set pair state, wall/contact history, radial internal conversion, fixed-pool process events, and prepared particle-grid transfer. Distributed ownership, dynamic memory growth, and general adaptive intraparticle meshes remain future method families over these contracts.
