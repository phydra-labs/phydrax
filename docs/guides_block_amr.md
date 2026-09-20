# Block AMR

Phydrax provides one host-prepared, finite-capacity block-AMR lifecycle. Fixed
Cartesian blocks and finite variable-patch buckets lower through one canonical
patch hierarchy and resource preflight. Production geometry adds high-order mapped
cell/face quadrature, explicit nonconforming physical mortars, exact piecewise-linear
three-dimensional multivalued cut complexes, moving topology transactions, and
node/edge/face/cell cochains. Every numerical route belongs to an immutable
`TopologyEpoch`, geometry revision, layout, and partition identity.

The baseline public example is
[`examples/block_amr_cartesian_fv.py`](https://github.com/phydra-labs/phydrax/blob/main/examples/block_amr_cartesian_fv.py).
The advanced geometry/entity example is
[`examples/block_amr_advanced_geometry.py`](https://github.com/phydra-labs/phydrax/blob/main/examples/block_amr_advanced_geometry.py).
Both use only public APIs.

## Hierarchy lifecycle

`BlockHierarchyPlan` derives each level's global cell lattice, spacing, block lattice,
aligned child counts, and stable integer block-ID range from one prepared tensor grid.
`BlockLevelPlan` fixes block shape, halo width, refinement ratio, and maximum blocks.
Level zero always covers the full base grid. Active blocks occupy a compact prefix in
canonical lattice/stable-ID order; inactive metadata uses explicit sentinels.

`FDAMRHierarchyPlan.prepare()` owns the cell-centered lifecycle:

1. `initial_topology()` realizes the complete level-zero topology.
2. `compile_topology(source, tags)` buffers cell tags, checks proper nesting and every
   fixed capacity, and returns `BlockTopologyCompileResult` atomically.
3. `field_transition(source, target, field_name, ...)` constructs a conservative,
   positivity-preserving nested cell-field transition between consecutive epochs.
4. `fill_patch(state, ...)` applies prepared source precedence: interior, same level,
   periodic, coarse-time interpolation, then caller-owned physical boundary values.

Compilation failure retains the source topology and records status/evidence; it never
returns a partially selected hierarchy. Physical boundary cells remain unresolved until
arrays matching each padded level workspace are supplied. `require_complete()` fails
closed if any requested value is missing.

## Block finite volume

`BlockAMRFiniteVolumePlan` binds one prepared hierarchy, conservation system,
`FiniteVolumeMethodPlan`, boundary set, optional identified source, and
`FiniteVolumePrecisionPolicy`. Preparation compiles block-local Cartesian face routes,
physical boundary slots, coarse/fine route pairs, FillPatch plans, cell coordinates,
and active-cell masks for exactly one topology epoch.

`BlockAMRFiniteVolumePlan` remains the optimized fixed-Cartesian, cell-centered
inviscid path. The general production path constructs a
`MultivaluedCutCellComplex` over canonical leaf cells and lowers its connected
fluid components to the ordinary polyhedral `UnstructuredFiniteVolumePlan`.
Consequently stationary mapped and embedded components use the existing
finite-volume compiler, arbitrary-normal Riemann fluxes, boundary policies,
positivity ledger, and `PreparedFiniteVolumeRuntime` rather than a second physics
implementation.

`BlockAMRConservationPlan` consumes accepted integral ledgers. It restricts covered
coarse cells and applies oriented coarse/fine `FluxRegister` corrections in the runtime's
fixed synchronization order. It is bound to one immutable topology; there is no public
pairwise/two-level synchronization facade.

## N-level runtime and schedules

`AMRTimeSchedulePlan` derives a complete N-level SSPRK(3,3) schedule. Its default edge
substep count equals the adjacent spatial refinement ratio; `subcycling=False` assigns
one interval to every edge. `multirate_amr_schedule_plan()` can attach the identity of a
compatible third-order `MultiratePartitionedRK` without transferring advancement
ownership away from the block runtime.

`BlockAMRRuntimePlan.prepare(topology)` creates `PreparedBlockAMRRuntime`. One
`advance(state, step_size, args)` attempt owns:

- FillPatch and all SSPRK stages at every scheduled level interval;
- accepted face-integral ledgers and edge restriction;
- reflux followed by covered-cell restriction;
- optional identified specialist synchronization;
- final finiteness/admissibility checks; and
- an optional post-accept indicator/topology transaction.

`BlockAMRAdvanceResult` reports the first failed `BlockAMRAdvancePhase`, level attempt
order, synchronization order, stage times, accepted ledgers, flux registers,
conservation defect, precision evidence, and any successor runtime. Failure rolls the
whole root interval back; `BlockAMRRuntimeState` remains synchronized and has one
canonical finite-volume topology journal.

## Composite scalar diffusion

`CompositeAMRCellLayout` packs leaf cells from every fixed-capacity level into an
ordinary `PyTreeSpace`. Physical leaf rows use cell-volume pairing weights; covered and
inactive storage rows use positive dummy weights and require a zero right-hand side.
`CompositeAMRDiffusionPlan` prepares a scalar Cartesian diffusion action for one exact
epoch. `PreparedCompositeAMRDiffusion.linear_system()` is consumed by ordinary
`phydrax.linalg.solve`; it does not add AMR state to the linear algebra package.

`composite_amr_multigrid_builder()` connects explicitly supplied AMR operators,
restriction/prolongation maps, and smoother builders to the native multigrid hierarchy.
Direct and Galerkin coarse-operator choices are explicit. This is a fixed-topology
scalar diffusion profile, not a general composite elliptic claim.

`MultivaluedCutCellDiffusionPlan` builds a volume-paired matrix-free graph
Laplacian from open aperture measures and physical center distances. It supports
positive variable coefficients, Dirichlet or Neumann boundary faces, one certified
constant nullspace per disconnected unanchored fluid region, and native projected
PCG. `ViscousFluxPlan.unstructured_*` supplies equation-owned viscous tensors,
least-squares gradients, conservative owner/neighbor scatter, explicit stability
evidence, and prescribed boundary normal fluxes on polygonal or polyhedral cells.
The same component graph feeds three-dimensional conservative small-cell
redistribution.

## Differentiation contracts

`BlockAMRDerivativePolicy` makes the derivative meaning explicit:

- `frozen-history` differentiates numerical values while patch, cut-component,
  route, limiter, and event history remain fixed. Extensive transition JVPs and
  algebraic VJPs use the exact common-refinement routes.
- `event-aware` delegates isolated transverse topology events to the matrix-free
  hybrid-event saltation JVP/VJP. Grazing, simultaneous, or branch-changing events
  poison derivative output instead of returning a plausible zero.
- `relaxed` is an explicitly different smooth multiresolution blend. Its gradient
  is not represented as the derivative of the hard topology selector.

`MappedGeometryDerivativePlan` differentiates traceable mapped quadrature under a
fixed topology and requires a declared topology margin. Stable IDs, capacity
decisions, and hard patch choices remain discrete; no straight-through estimator
is installed.

## Checkpoint and output lifecycle

Block continuation reuses `FiniteVolumeCheckpointPlan`, `FiniteVolumeCheckpoint`,
`write_finite_volume_checkpoint`, and `read_finite_volume_checkpoint`. Construct the
plan with a `PreparedBlockAMRRuntime`; optionally pass its matching
`PreparedDistributedBlockAMRHierarchy`. The same pickle-free finite-volume archive
family records:

- runtime time, root and per-level accepted counts, and status;
- every level value in canonical stable-block order and checkpoint precision;
- active/stable/parent/logical/neighbor metadata and derived coverage/interfaces;
- the complete `TopologyEpoch` and finite-volume topology journal;
- FillPatch, face, edge, conservation, and topology-artifact route identities; and
- optional exact distributed partition compatibility data.

Restore reconstructs and revalidates metadata, topology, coverage, routes, precision,
journal, runtime, and partition identity before returning a state. A changed schedule,
epoch, runtime, capacity, route, dtype, checksum, or supplied partition is incompatible;
restore never silently repartitions or recompiles.

`FiniteVolumeOutputPlan(path, prepared_block_runtime, partition=...)` uses the existing
HDF5/XDMF output owner. Immutable hierarchy metadata, coverage, route identities,
precision, epoch, and optional partition record live beside temporal snapshots. Each
step stores level arrays in canonical fixed-capacity order and records the stable IDs of
its active prefix. XDMF exposes one Cartesian grid per active block. Output is for
inspection, never restart, and requires the optional `h5py` dependency.

For mapped multivalued continuation,
`MultivaluedBlockAMRCheckpointPlan` and the corresponding read/write functions
store canonical patch buckets, exact cut geometry evidence, content, time,
revision, body/map identities, and polyhedral connectivity in the same
pickle-free array-archive substrate. `CutCellRestartRegistry` resolves declared
geometry callables and reconstructs the topology; callers do not preconstruct the
archived layout. The resulting canonical state may then be assigned to a new
`DistributedCutCellPartitionPlan`. `write_multivalued_cut_cell_output()` writes
partition-independent component, face, body, coordinate, and connectivity arrays.

## Distributed execution

`BlockAMRPartitionPlan` retains the optimized fixed-block owner routes.
`DistributedCutCellPartitionPlan` binds multivalued control-volume components to
a live `ExecutionGroup`, places the part axis under `NamedSharding`, prepares
cross-part aperture evidence, and gathers owner/neighbor states through global
JAX routes. Process-global commit decisions use an actual multi-process
collective. Stable semantic component order is independent of process count, so
repartition changes placement rather than topology or state meaning.
`pack_process_local()` constructs the global sharded array through
`jax.make_array_from_process_local_data`; no process needs a full host copy.

## Variable logical patches

`LogicalPatchBox` is the semantic half-open reference-cell box. A
`PatchShapeSignature` declares one compiled envelope, halo, and alignment, while
`PatchBucketPlan` fixes its lane capacity. `VariablePatchTopologyCompiler` buffers
block-local tags sparsely, clusters face-connected components, splits boxes to the
finite catalog, checks full parent support, and accepts or rejects the complete
candidate atomically.

Patch topology identity is based on canonically sorted logical boxes, not bucket lanes.
`VariablePatchLevelMetadata` records bucket/lane placement separately. Envelope cells
outside each actual extent are inactive and cannot become donors, flux owners, geometry
cells, or reductions.

## Canonical entity execution

`VariablePatchEntityComplexPlan` lowers each accepted level into one bounded
`CellComplexTopology`. Nodes, oriented edges, oriented faces, and cells receive stable
reference-coordinate identities. Shared patch entities have one owner; bucket views
gather from and scatter back to the canonical entity state through exact sparse
transpose routes. Periodic point-like coordinates are identified before entity
deduplication.

`CompatibleEntityTransferFamily` prepares distinct primal, algebraic-transpose,
Hilbert-adjoint, and semantic-restriction maps. The family checks route capacity,
constant preservation, supported-region roundtrip, and the discrete de Rham
commutator before admission. `VariablePatchCochainSynchronizationPlan` applies
edge-integrated EMF reflux-curl without changing discrete magnetic divergence.
`CutCellCochainTransferPlan` constructs commuting degree maps, exact algebraic
transposes, and dynamic-Hodge adjoints. A changed topology is admitted only when
its physical cell common refinement extends to the remaining degrees within the
declared commutator tolerance.

## Mapped, mortar, and moving geometry

`PatchCoordinateMapSet` supplies a default traceable map plus stable per-patch
overrides. `CanonicalMappedGeometryPlan` integrates cell volume/centroid and all
face quadrature from map Jacobians at a declared order; it records Jacobian,
face-closure, mesh-volume-rate, and GCL evidence on active bucket cells.

`MappedMortarPlan` is the required seam for nonconforming charts. Owner and
neighbor reference traces are compared to one explicit common physical surface;
quadrature weights and owner-oriented area vectors are generated only when both
trace mismatches and the physical measure pass.
`MappedMortarFluxPlan` evaluates arbitrary-normal numerical fluxes at that common
quadrature and scatters one exactly canceling owner/neighbor content rate.

`MovingMultivaluedCutCellPlan` prepares start, endpoint, and midpoint cut
complexes, constructs an incomplete physical common refinement when a wall sweeps
volume, and accounts separately for overlap, covered, and newly uncovered
measure. Newly uncovered content requires an explicit state provider. The
accepted result binds the successor geometry, content, swept-volume/content
ledger, topology-event count, time, and revision atomically.
`MovingTopologyLocalizationPlan` fingerprints sample-sign topology independently
of near-event polyhedron construction, brackets every crossing visible to its
finite probe envelope, bisects each event, enforces a finite post-event margin,
and advances the requested interval as consecutive atomic substeps.

`CutCellCochainPlan` uses the polyhedral shell's exact incidence and positive
mass-lumped primal/dual metrics. `CutCellCochainSynchronizationPlan` applies
edge-integrated EMF reflux-curl while preserving the face-flux divergence.

## Multivalued embedded boundaries

`EmbeddedLevelSetBodySet` represents stable tagged solid CSG. `union` takes the
minimum signed field, `intersection` takes the maximum, and per-body ±1 signs
represent complements and differences without losing the selected boundary tag.
`MultivaluedCutCellPlan` samples the declared map and level-set fields on a
conforming Freudenthal tetrahedralization. Each tetrahedron is clipped by its
piecewise-linear field, connected fluid fragments are assembled into independent
component control volumes, and shared internal faces cancel before a canonical
polyhedral mesh is created.
`MultivaluedCutCell2DPlan` performs the corresponding triangle clipping and
component-edge assembly in two dimensions. Its component graph supports
disconnected regions and inner boundary loops directly and provides a compiled
conservative SSPRK(3,3) path.

One background cell may therefore own several independent conserved states, and
one background face may own several aperture fragments. Component volumes,
centers, volume fractions, body facets, face routes, and area vectors are
capacity-bounded. `MultivaluedCutCellEvidence` records regular, covered, cut, and
multivalued counts plus predicate margin and independent volume/face closure.
Subcell samples on the predicate tolerance, nonmanifold shells, unresolved patch
map disagreement, or exhausted component/aperture/facet capacity fail before an
execution state exists.

The exactness claim is relative to the declared piecewise-affine map and
piecewise-linear sampled level set. A smooth map or implicit surface is the
corresponding finite geometric approximation; corner-only samples do not receive
a hidden-topology certificate.
`CertifiedImplicitBody` supplies physical interval bounds and a local
piecewise-linear topology certificate. `AdaptiveImplicitSamplingPlan` recursively
subdivides cells whose interval may contain an unresolved zero and fails if the
declared depth cannot certify them; its selected finite subdivision feeds either
the two- or three-dimensional cut plan.

## Dynamic signatures, placement, and restart

`PatchSignaturePolicy` generates aligned finite envelopes for previously unseen
finite patch extents. `PatchExecutableCachePlan` compiles every missing JAX
signature before publishing a new immutable cache generation; cache failure
cannot mutate the active runtime. Unbounded shapes are not created inside XLA.

Variable-patch and multivalued component partition plans remain explicit.
Portable multivalued restart reconstructs semantic topology first and applies a
destination partition afterward, rather than requiring the archived device
layout.

Run all three qualification tiers and the production benchmark explicitly:

```bash
python tools/block_amr_qualification.py --output block-amr-qualification.json
python tools/block_amr_advanced_qualification.py
python tools/block_amr_production_qualification.py
python tools/block_amr_production_benchmarks.py --smoke
```

Production qualification covers resource preflight, high-order mapped GCL,
two- and three-dimensional multivalued cuts, adaptive hidden-topology sampling,
conservative mortar fluxes, public finite-volume advancement, component
diffusion/nullspaces, exact chain identity and cochain transfer, swept-volume
balance, multiple localized topology events, fixed-history derivatives, dynamic
executable installation, process-local distributed sharding, and
topology-reconstructing restart. The report is `inconclusive` rather than `pass`
when the required real multi-host hardware is unavailable.
`BlockAMRReferenceParityPlan` binds independent observable arrays to provider and
revision identities and reports mixed-tolerance defects without using runtime
output as its own reference.

The production contract still refuses mathematically undefined or unbounded
requests: nonmanifold input without declared CSG resolution, arbitrary infinite
device shapes, and ordinary derivatives of hard integer topology choices.
A release claim requires a `pass` production report on the declared hardware tuple.
