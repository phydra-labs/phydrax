# Block AMR

Phydrax provides one host-prepared, fixed-capacity block-AMR lifecycle. The baseline
profile uses fixed Cartesian cell blocks. The advanced profile adds true half-open
logical patch boxes assigned to finite static envelope buckets, bounded canonical
node/edge/face/cell complexes, traceable mapped/ALE metric states, exact two-dimensional
sharp embedded-boundary clipping, and explicit repartition/restart transactions. Every
compiled numeric route belongs to one immutable `TopologyEpoch`.

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

`FDAMRHierarchyPlan.prepare()` owns the cell-centred lifecycle:

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

The current block profile is cell-centred and inviscid. It accepts numerical-flux finite
volume methods whose reconstruction reach fits every declared block halo. Viscous
methods, mapped blocks, embedded boundaries, unstructured cells, and topology selection
inside a device kernel are rejected rather than approximated by another route.
`BlockAMRFiniteVolumeStageResult` returns block-shaped residuals, the authoritative
routed stage ledger, maximum rate, and precision evidence.

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

## Differentiation boundary

Topology selection, stable-ID allocation, capacity rejection, partition selection, and
topology transition are host decisions and are not differentiated. A
`BlockFieldTopologyTransition` reports `differentiable_geometry=False`, and its
`TopologyEpochTransition.require_differentiable_topology()` refuses a topology
gradient. Within one prepared epoch, block finite-volume numeric kernels, FillPatch
array routes, and composite operator actions retain their declared JAX transformations.
A trajectory derivative is valid only while epoch, routes, schedule, and discrete branch
history are frozen.

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

## Distributed execution and qualification limits

`BlockAMRPartitionPlan` computes deterministic Morton-contiguous owner ranges from
canonical logical indices and optional costs. `PreparedDistributedBlockAMRHierarchy`
provides packed owner-computes state, exact transpose routes, distributed FillPatch, and
`DistributedBlockAMRResourceEvidence`. `BlockAMRStableIDMigrationPlan` is an explicit
repartition between equal topologies; checkpoint restore itself never invokes it.

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

## Mapped and moving geometry

`VariablePatchGeometryPlan` owns reference vertices and a declared global coordinate
map. It produces `VariablePatchGeometryState` values containing mapped vertices, cell
centres, positive oriented volumes, mesh-volume rates, and independent GCL defects.
One `SmallLinearSolvePlan` supplies scaled one-to-three-dimensional determinants.
Inactive envelope vertices are mapped from a safe reference point and are never part of
validity evidence.

`VariablePatchALEPlan.prepare_step()` constructs the three SSPRK(3,3) geometry stages,
checks each stage, and certifies the accepted endpoint-volume recurrence. Geometry
commit is all-or-nothing. Topology is fixed throughout an attempted step; only metric
arrays and numeric revision change.

`PreparedCochainTopology`, `CochainMetricPlan`, and `CochainMetricState` separate static
incidence from traceable diagonal-Hodge metric arrays. The runtime state accepts JAX
tracers, checks only active capacity entries, and creates a `CochainDiscretization`
snapshot only on the host.

## Embedded boundaries

`VariablePatchEmbeddedBoundaryPlan` reuses the canonical linear-edge clipping kernel
over accepted mapped patch geometry. The current qualified profile is stationary,
two-dimensional, single-body, single-segment-per-cut-cell geometry. It returns fluid
volumes/centres, background-face open fractions/measures/endpoints, cut-face
centres/normals/measures, body tags, small-cell masks, and explicit volume and oriented
face-closure evidence. Shared reference faces are clipped once through a canonical
face-key cache.

`MovingEmbeddedBoundaryEventPlan` handles a sign-topology crossing only at an accepted
boundary. It checks fixed vertex capacity, sign margin, and a declared
symmetric-difference versus swept-wall-volume budget before committing one consecutive
epoch to the existing finite-volume topology journal. Failure retains the source epoch.

Three-dimensional sharp clipping, disconnected/multivalued cut cells, topology changes
inside a step, and full mapped/EB finite-volume time advancement remain refused rather
than inferred from the two-dimensional geometry contract.

## Variable-patch placement and restart

`VariablePatchPartitionPlan` assigns canonical boxes to fixed local bucket capacities
using deterministic positive costs. `PreparedVariablePatchPartition` packs and unpacks
bucket tensors and emits an explicit partition-only successor epoch.
`VariablePatchCheckpointPlan` writes canonical logical metadata and field arrays with
the shared pickle-free array archive. Restore requires the exact topology/layout and
partition; changed placement is a separate explicit repartition.

Run the candidate qualification and resource benchmark tools explicitly:

```bash
python tools/block_amr_qualification.py --output block-amr-qualification.json
python tools/block_amr_benchmarks.py --output block-amr-benchmarks.json
python tools/block_amr_advanced_qualification.py
python tools/block_amr_advanced_benchmarks.py --smoke
```

Baseline qualification covers fixed-block conservation, topology transition,
composite Poisson, fixed-epoch AD, restart, and distributed determinism. Advanced
qualification covers variable box/bucket admission, bounded entity incidence,
mapped/ALE GCL, two-dimensional embedded volume/face closure, and explicit
partition evidence. Real-device and multi-host gates remain inconclusive when the
required hardware is unavailable.

The advanced profile does not claim three-dimensional or multivalued embedded
clipping, general viscous physics, nonconforming patch-local maps, arbitrary
runtime-created signatures, topology gradients, or external-library parity.
Qualification tools produce candidate evidence; they do not declare a release.
