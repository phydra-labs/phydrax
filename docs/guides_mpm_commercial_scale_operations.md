# Commercial MPM scale and operations

## Execution and capacity

`MPMExecutionPlan` fixes backend, device mesh, precision policy, deterministic mode,
kernel realization, and particle/grid/route/field/block/contact capacities.
`MPMCapacityCertificate` is valid only for the exact execution plan, source,
toolchain, and hardware.

Reference, fused-JAX, and custom-accelerator realizations must preserve one physical
method. Fast kernels never substitute different material/contact equations.

## Distributed ownership

`MPMDistributedPlan` fixes logical blocks, block owners, shard count, particle
capacity, and halo width. Accepted-boundary migration uses stable particle IDs and
returns per-device capacity evidence.

Distributed P2G reductions use deterministic shard ordering and compensated sums.
`distributed_global_transaction` commits a generation only when every shard succeeds.
`MPMShardCheckpointManifest` requires every shard payload under one ownership plan.

A missing worker, failed collective, timeout, or shard-capacity overflow is global
no-commit followed by recovery from the last validated generation.

## Particle lifecycle

`MPMParticleLifecyclePlan` supports accepted-epoch activation, retirement, split,
and merge under fixed capacity. It returns mass, momentum, volume, ID-uniqueness, and
capacity evidence. Parent/child IDs and topology generation are retained.

`MPMCapacityBucketPlan` chooses a separately prepared larger bucket after a terminal
capacity result. It does not resize arrays inside a JIT step.

## Dynamic pages and AMR

`MPMPageTablePlan` is a bounded deterministic page map with typed overflow.
`MPMAMRPlan` currently requires a ratio-two nested hierarchy, fixed level/block
capacities, conservative restriction/prolongation, and deterministic subcycle counts.
Refinement/coarsening belongs to an accepted topology epoch recorded by
`MPMAMRTopologyJournal`.

## Durable operations

- `MPMCheckpointPlan`: atomic accepted-state generations and validated restore.
- `MPMOutputPlan`: HDF5/XDMF time series and VTK snapshots.
- `MPMBoundedOutputBuffer`: explicit output backpressure.
- `MPMRunSupervisor`: host lifecycle, checkpoint/output coordination, recovery,
  quarantine, release, metrics, and append-only event records.

The supervisor never retries a numerical step; adaptive MPM remains the sole numerical
retry authority.
