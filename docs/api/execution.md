# Execution substrate

`phydrax.execution` separates process bootstrap, resource ownership, logical
decomposition, physical placement, and host lifecycle. Numerical values remain
ordinary JAX arrays. The default path remains local; distributed execution is
enabled by an explicit runtime or execution group.

`execution.default_device(device)` scopes placement of uncommitted computations
on a live JAX device and restores the enclosing placement after exit or failure.
A live device is execution state, not a serializable `RuntimeConfig` or
`NumericsConfig` field. This scope does not move already committed arrays.

## Bootstrap before device initialization

Launcher rank settings must exist before `import phydrax`. The lazy root consumes
them and coordinates without device discovery; `runtime_info()` discovers
devices. Explicit bootstrap intent belongs to [configuration](configuration.md).

```python
# PHYDRAX_COORDINATOR_ADDRESS, PHYDRAX_NUM_PROCESSES, and
# PHYDRAX_PROCESS_ID were set by the launcher.
import phydrax as phx

runtime_info = phx.execution.runtime_info()
runtime = phx.execution.ExecutionRuntime.current()
```

The bootstrap accepts explicit `PHYDRAX_COORDINATOR_ADDRESS`,
`PHYDRAX_NUM_PROCESSES`, `PHYDRAX_PROCESS_ID`, `PHYDRAX_LOCAL_DEVICE_IDS`,
`PHYDRAX_COORDINATOR_BIND_ADDRESS`, `PHYDRAX_HEARTBEAT_TIMEOUT_SECONDS`, and
`PHYDRAX_INITIALIZATION_TIMEOUT_SECONDS` settings. It also delegates to JAX
launcher detection under Slurm, Open MPI, PMI, and PMIx. `PHYDRAX_CPU_COLLECTIVES`
selects `gloo` or `mpi` before CPU backend initialization. An attempted
distributed initialization after JAX has initialized fails rather than
silently producing an incomplete process set.

For NVIDIA multi-GPU and multi-node work, use JAX global arrays and NCCL with
`phydrax.backends.JaxCollectiveProvider`. Multi-process `srun` and `mpirun`
launches are coordinated automatically through the startup detection above.
MPI serves as a launcher or an optional interoperability boundary; JAX/NCCL
collectives do not require GPU-direct mpi4jax. See
[Installation](../installation.md#multi-gpu-and-multi-node-execution) for CUDA
extras and [Configuration](configuration.md#phases-and-startup) for startup
settings.

For a local process set, `LocalProcessLaunchPlan` and
`launch_local_processes` start one shell-free argument vector per rank and
inject the same explicit rendezvous identity. Each rank's environment replaces
the base environment's rank settings as one group, clearing local device IDs and
bind addresses it does not set, while keeping timeouts the base already selects.
Use `devices_per_process=1` to assign one distinct local GPU to each worker.
The local substrate benchmark exposes this as `--devices-per-process`, checks
launched workers' actual GPU ownership, and admits its root group against an
explicit declared resource request before numerical work. The local smoke
matrix includes one/two/four/eight-device arrangements, including one process
with eight devices, two workers with four each, four with two each, and eight
with one each. A collected case is not hardware qualification. A single-process invocation uses all visible GPUs;
select a subset with `CUDA_VISIBLE_DEVICES` before launch. Reports include the
explicit `NCCL_NVLS_ENABLE` setting, physical-host ownership, inventory/group
identity, resource admission, weighted mean error, and diagonal PCG
convergence/error. This bounded smoke is not a
solver-scaling qualification or a certified memory envelope.

The launcher enforces an explicit whole-job `timeout_seconds` and bounded
`max_output_bytes`, terminates its owned rank process groups on failure, and
retains rank-ordered output/status evidence. MPI monitor loss can leave a failed
rank's exit status unknown (`None`); it is never replaced by a fabricated signal
code. Successful launches require complete zero exit statuses.

## Supervised site MPI and ordered vendor collectives

The optional cuDSS MGMN and cuDecomp profiles require an actual, explicitly
selected, same-host CUDA-aware Open MPI world. Pass the absolute site executable
and its arguments through `launch_local_processes(..., mpi_launcher=...)`;
the launcher supplies the process count. The admitted profile uses one GPU per
rank. Ordinary JAX/NCCL execution continues to use its existing bootstrap.

Rank monitors and pre-exec bootstrap are standard-library-only: they must not
import Phydrax/JAX and accidentally join an extra distributed process set.
The actual worker obtains `require_local_process_supervision()`. Its live
socket/peer credentials, process ancestry, rank group, and deadline establish
trusted-local supervision; an environment boolean is not equivalent. This is
not containment of a malicious process that deliberately escapes its session.

`create_collective_lease` binds an already initialized `mpi4py.MPI.Comm`,
an explicit JAX-process/MPI-rank map, an execution group, stable
provider/configuration/preparation identifiers, and the live supervisor.
Creation does not initialize another MPI world. The bridge verifies the exact
site MPI library/header/build and the MPI instance behind the communicator,
`MPI_THREAD_MULTIPLE`, runtime CUDA-buffer support, and one shared-memory MPI
domain containing every participating rank. A split-domain communicator refuses
before any vendor provider executes.

The invocation protocol has distinct phases:

1. Prepare vendor descriptors through the live lease's named host preparation
   operation. This phase is collective and bounded.
2. Use `prepare_collective_program` with explicit dynamic example arrays and the
   lease specification. This is a host staging operation: tracing executes the
   supplied Python body once. Captured numerical constants must instead become
   dynamic arguments.
3. The resulting `CollectiveProgram` owns both the admitted transformed schedule
   and the executable lowered from that exact staged graph. Schedule agreement
   precedes coupled compilation. Unknown placement, opaque compiled callbacks,
   unsupported transform schedules, and unmatched activity are refused.
4. `lease.invoke(program, *args)` binds fresh dynamic invocation metadata,
   executes that owned program, waits for values/effects, and closes the ticket.
   It neither discovers a schedule by running an arbitrary callback nor traces
   the user function again.
5. `lease.close()` collectively retires vendor attachments and communicators.
   Stale compiled invocations and closed bindings refuse; retirement is not a
   local LRU destructor or an interpreter-exit collective.

Ordered site programs admit exactly **JAX/JAXlib 0.12.0** at the private dispatch
boundary. Admission verifies the published upstream source and observed code,
descriptor, executable-owner, token-input, token-output, and profiler-wrapper
identities; a modified downstream build with the same version string refuses.
That release's cached AOT C++ call path can omit ordered-effect runtime tokens.
The sole execution owner therefore invokes the same exact compiled program
through JAX's token-preserving Python call path. It does not substitute or
recompile a graph, patch JAX globally, add fake arguments, or remove ordering.
`CollectiveProgramProvenance` records the observed dispatch identity, and staging
and invocation revalidate it before ticket admission. This host artifact is not
a numerical PyTree field or a serialized live handle; MPI and hardware
qualification remain separate.

`lease.binding` is exact-local-device metadata for rank-local CSR execution.
`binding_for(sharding)` explicitly admits replicated global metadata for the
manual spectral route. Invocation preserves each binding's admitted placement.
Live IDs/tickets are dynamic leaves, never scientific fingerprints or portable
checkpoint fields.

Rank-local and global arrays are different contracts. MGMN allows unequal owned
CSR partitions without inventing global shape ownership from coincident
dimensions. All ranks must enter compatible transformed collective schedules;
materialized zero tangents do not justify rank-asymmetric symbolic omission of a
collective. Fatal rank loss uses supervised coupled termination, not another
ordinary blocking status collective.

These site profiles do not claim multi-host, RDMA, arbitrary batching,
rematerialization, or unrestricted asynchronous callback execution. Current
eight-A40 peer integrity is blocked as described in
[installation](../installation.md#eight-a40-host-admission).

::: phydrax.execution.CollectiveLeaseSpec

::: phydrax.execution.CollectiveBinding

::: phydrax.execution.CollectiveProgram

::: phydrax.execution.CollectiveProgramProvenance

::: phydrax.execution.create_collective_lease

::: phydrax.execution.prepare_collective_program

::: phydrax.execution.LocalProcessSupervision

::: phydrax.execution.require_local_process_supervision


::: phydrax.execution.RuntimeInfo

---

::: phydrax.execution.ExecutionRuntime

## Policy, requirements, and resolved plans

`ExecutionPolicy` is user intent. An execution owner supplies complete valid
`ExecutionCandidate` values; the planner admits one candidate against observed
resources and fixes providers before execution begins. There is no mid-run
provider fallback.

```python
policy = phx.execution.ExecutionPolicy.auto(
    determinism=phx.execution.DeterminismScope.LOGICAL,
    recovery=phx.execution.RecoveryPolicy.CHECKPOINT_RESTART,
)
```

`ExecutionPlan` is immutable and serializable. It contains process/device keys,
logical-axis bindings, value placement, provider identities, topology epoch,
and decision evidence. Live meshes, devices, communicators, threads, and
scheduler clients exist only in an `ExecutionRuntime`.

This is a clean ownership cutover: use `phx.execution.ExecutionPlan` rather
than `phx.lifecycle.ExecutionPlan`, and `phx.execution.ResourceRequest` rather
than `phx.service.ResourceRequest`.

::: phydrax.execution.ExecutionPolicy

---

::: phydrax.execution.ExecutionPlan

---

::: phydrax.execution.ExecutionGroup

## Memory measurement and certified envelopes

Memory quantities are distinct measurements, never one number. A
`MemoryMeasurement` records its method, its scope (`baseline` instantaneous or
`peak` over a window), the source counter resolution, or an explicit
`unavailable_reason`. `sample_host_memory` reports process RSS, the kernel RSS
high-water counter, and host physical and available memory (Linux procfs, Darwin
Mach, or Windows working-set and memory-status APIs). Windows observations retain
their method and page resolution; failures are unavailable evidence, not zero.
On Linux, RSS and high-water counters come from one `/proc/self/status`
snapshot (`VmRSS` and `VmHWM`), so the reported current value cannot exceed that
snapshot's high-water value. Both counters use approximate kernel accounting:
`VmHWM` can fall below an earlier RSS observation after pages are freed, and it
does not certify an RSS upper bound. `sample_device_memory` reports the PJRT
allocator limit, pool reservation, live allocations and allocator high-water
mark, plus the bytes of live JAX arrays from each requested device's client;
device IDs are not treated as interchangeable across clients. The CPU provider
exposes no allocator statistics, and the sample says so.
`compiled_memory_estimate` returns XLA's buffer-assignment plan for a compiled
executable, which is an estimate rather than a measurement.

Local Windows file publication uses held directory handles, reparse-point
refusal, owner/DACL checks, and native namespace synchronization on NTFS.
Unsupported filesystems or failed synchronization do not produce a publication
receipt. Successful flush requests are not hardware power-loss qualification.
The separately named `POSIXArtifactRepository` remains POSIX-only and refuses
Windows construction rather than weakening its security or durability contract.

Regular-file `DurableCheckpointStore` generations use an exclusive sibling hard
link, remove the staged name, and synchronize the directory before publishing
the commit pointer. This requires hard-link support, not Linux
`RENAME_NOREPLACE`; collisions cannot overwrite an immutable generation. POSIX
resource-set publication retains ordinary directory-rename semantics rather
than adding a rename-flags requirement: exclusive mode refuses a nonempty
destination, but can replace an empty directory, including one introduced
concurrently after admission. It does not promise atomic no-replace for that
race. Atomic replacement of an existing resource set still requires supported
directory exchange. These contracts do not constitute qualification of a
particular network filesystem.

Windows SQLite service admission uses the same descriptor-relative primitives
and retains its ancestor-handle chain until the connection closes. Admission,
connect, or schema failures release the chain; database restarts do not bypass
path ownership or reparse checks.

```python
with phx.execution.PhaseMemorySampler("assemble", interval_seconds=0.005) as s:
    assemble()
evidence = s.evidence
evidence.sampled_peak_resident       # sampled maximum, never an upper bound
evidence.resident_peak_upper_bound   # trustworthy kernel bound, or unavailable
evidence.interval_peak_resident      # exact only if a trustworthy bound rose
payload = evidence.to_payload()
```

The sampler keeps constant-size state and reports its sample count and largest
observed gap; transients shorter than the gap can be missed. Linux procfs cannot
certify either `resident_peak_upper_bound` or `interval_peak_resident`: both
report explicit unavailability, while the sampled RSS maximum and approximate
process high-water counter remain measured evidence.
The [Linux procfs documentation](https://man7.org/linux/man-pages/man5/proc_pid_status.5.html)
marks RSS and high-water counters as inaccurate; source counter resolution is
their granularity, not an accuracy guarantee.

`discover_resource_inventory` and `ExecutionRuntime.current()` are collective
in an initialized multi-process runtime: every rank must call them in the same
collective sequence. Device allocator capacities are exchanged in canonical
device order, using only each device owner's observation, so all ranks derive
the same inventory and execution-group identities. Multi-process discovery also
exchanges each process's Linux `/proc/sys/kernel/random/boot_id`, an authoritative
kernel boot identity shared by ranks on the same physical host. Canonically
ordered `process_host_ids` use `linux-boot-<uuid>` identities, not hostnames,
launcher rank guesses, or one-host-per-process assumptions. Missing, unreadable,
invalid, or unsupported kernel identity on any rank leaves the entire mapping
empty; multi-process resource admission then explicitly refuses missing host
mapping evidence. Discovery itself can still report devices and local samples.
Multi-process inventory and derived execution-group/plan identities intentionally
change to include this host ownership and change again after a host reboot.
Single-process discovery retains its existing identity and empty host mapping.

Local timestamped host and device memory samples remain evidence excluded from
`inventory_id`; discovery does not infer or fabricate remote memory samples.
Use `sample_host_memory` and `sample_device_memory` for rank-local observation
without coordination.
The allocator limit becomes `DeviceResource.memory_bytes` when exposed. A
`ResourceRequest(memory_envelope="certified")` admits a candidate only when its
`ExecutionResourceEvidence` declares per-host (and, for accelerators,
per-device) peak and reserve bytes with `memory_basis` `declared` or
`compiler_analysis`, every placed host has measured available memory at least
that footprint, and every placed accelerator has measured allocator headroom.
Unknown evidence, a remote host without a sample, or a `sampled` or
`provider_estimate` basis refuses the certified plan instead of assuming free
memory. A finite vendor estimate is not an allocation bound. Preserve
per-device estimates and unknown host/resident storage in the provider's own
evidence; do not divide one group total into invented per-device measurements.
The default `declared` envelope keeps checking ceilings against owner estimates
only.

Streamed sparse relations use the same distinction between declared and
compiled memory. `phydrax.sparse.StreamedRelationResources` declares only the
logical bytes owned by the streamed substrate: schedules, one tile of
workspace, spill and replay carries, outputs, and cotangent accumulators. It
excludes callback internals and compiler temporaries. To certify a forward,
reverse, force-loss, or HVP route, call `compiled_memory_estimate` on that
transformed executable. `MACEAcceleratedCoupling.resource_evidence` converts
this estimate into `ExecutionResourceEvidence` with
`memory_basis="compiler_analysis"`; kernel registers and shared memory remain
outside XLA's analysis. See
[Streamed nonlinear relations](../guides_sparse_spatial_hierarchies.md#streamed-nonlinear-relations).

::: phydrax.execution.PhaseMemorySampler

---

::: phydrax.execution.PhaseMemoryEvidence

---

::: phydrax.execution.MemoryMeasurement

---

::: phydrax.execution.HostMemorySample

---

::: phydrax.execution.DeviceMemorySample

---

::: phydrax.execution.CompiledMemoryEstimate

---

Owners that admit work against a declared byte cap refuse it before execution
with `MemoryBudgetError`. Its `budget` names the cap parameter, and
`required_bytes`/`limit_bytes` carry the owner's admission estimate and the cap.
The statistical periodic-flow owners (increment moments, power-law fields,
initial measures, fixed-step ensembles, coupled-ensemble metrics, and operator
datasets) share this contract.

::: phydrax.execution.MemoryBudgetError

## Native global arrays and rank-local providers

Native JAX global arrays are the default numerical route. Use
`shard_array_axis` for independent arrays, owner-specific placement policies
for structured values, and `shard_map`-based owners for manual halo or
redistribution algorithms. A JAX reduction over a global array already has
global semantics; explicit collectives belong inside rank-local mapped
kernels. `shard_tree_axis` and `replicate_tree` place every array leaf of a
PyTree the same way, independently of array roles: FIXED data is placed like
parameters.

`phydrax.backends.JaxCollectiveProvider` supplies named-axis operations inside
mapped regions, using JAX/NCCL for NVIDIA multi-GPU and multi-node execution.
`Mpi4JaxCollectiveProvider` is an optional rank-local interoperability route with
caller-owned communicator lifetime and operation-specific transformation
support. GPU-direct MPI requires matching user-provided mpi4jax and site MPI
builds; it is not the native distributed JAX route. Availability never implies
qualification.

### Distributed point relations

`phydrax.discretization.spatial.DistributedOwnershipPlan` binds one owner per
device of a one-dimensional `ExecutionGroup` mesh, a per-owner slot capacity,
and the owner-to-process map. `DistributedPointLayout` stores owner-blocked
points with stable global IDs, logical rows, and per-owner epochs; ownership may
be arbitrarily uneven up to the capacity. `from_global` validates the owner
map, capacity, and ID uniqueness on the host and orders each owner's rows by
Morton code; `from_blocks` adopts device rows and audits ID uniqueness with a
bounded all-to-all.

`DistributedNeighborQueryPlan` (k nearest) and `DistributedRadiusQueryPlan`
(inclusive radius rows, optionally pair-once) never replicate targets. Owners
publish source boxes and populations; each target searches its own owner,
derives a certified search radius from that result and the owner populations,
and is shipped only to owners whose periodic boxes intersect that ball.
Answers return through bounded all-to-all packets and merge by
`(distance, stable_id, owner)`. `maximum_remote_owners`, `halo_capacity`, local
candidate capacity, and row width are explicit; exceeding one refuses the
affected targets with a `DistributedRelationStatus` instead of truncating. A
stale owner epoch reports every target as `MISSING_OWNER`. Ownership plans,
query plans, and queries require `jax_enable_x64=True` (uint64 Morton codes,
int64 stable IDs) and raise `ValueError` otherwise; float32 coordinates are
supported with x64 enabled.

`DistributedHaloPlan` binds owner-blocked routes to owned columns plus
deduplicated halo columns. Its forward gather and transpose return every halo
contribution exactly once, accumulating after owned contributions in ascending
requesting-owner order. `DistributedPointLayout.migrate` moves each active row
exactly once with its IDs, logical row, coordinates, payload, and sender epoch;
any invalid destination, packet or receive overflow, or epoch inconsistency
rolls every owner back atomically.

`DistributedMortonNeighborQueryPlan` is the contiguous-logical-shard form of
the same query over an `ExecutionGroup`, returning rows in logical target order
with global logical source indices.

Forced CPU devices (`JAX_NUM_CPU_DEVICES=N`) demonstrate
functional distributed parity only; accelerator and multi-host performance need
real hardware campaigns.

::: phydrax.execution.shard_array_axis

---

::: phydrax.execution.global_weighted_mean

## Deterministic worksets and child groups

Execution worksets group exact homogeneous signatures into deterministic,
fixed-capacity buckets. They own canonical ordering, reversible gather/scatter,
semantic restartable RNG keys, and content-addressed checkpoints.

`evaluate_execution_worksets_grouped` allocates each item the number of devices
declared by its `PoolExecutionSignature`. Child groups are disjoint and
process-symmetric. Multi-process JAX groups execute in one controller-consistent
order; independent scheduler jobs use separate process sets.

Homogeneous Equinox module batches use
`evaluate_execution_worksets_filter_vmap`. The item lane is a caller-declared
`phydrax.LaneLayout` of kind `"item"`: declared leaves carry the leading item
axis, while every other leaf (static configuration and shared arrays) is
broadcast within each signature bucket. Lanes are independent of array roles,
so FIXED data such as per-item normalizers may be mapped while parameters are
shared. Without a layout every array leaf carries the item axis. Large or
heterogeneous modules remain separate worksets rather than being forced through
one unbounded vectorization.

Identity follows how weights reach the executable. Callables compiled into a
pooled method with statically held weights are passed as
`PoolExecutionSignature(static_callables=...)` and enter its
`ExecutableSignature` through `callable_payload`. Dynamic weights passed as
arguments do not change the signature; an `ExecutionWorksetCheckpoint` records
the `NumericRevision`s bound into its items (`numeric_revisions=()` declares
none), and `restore_execution_workset_checkpoint` requires the same bound
revisions.

```python
import jax
import jax.numpy as jnp
import phydrax as phx
from phydrax.execution import (
    ExecutionWorksetPlan,
    PoolExecutionSignature,
    evaluate_execution_worksets_filter_vmap,
)


class Normalized(phx.StrictModule, phx.ParameterOwner):
    weight: jax.Array
    shift: jax.Array = phx.fixed_field()

    def __call__(self, x):
        return self.weight * (x - self.shift)


signature = PoolExecutionSignature(
    topology_id="cells",
    method_id="normalized-map",
    precision_id="float64",
    backend_id="jax-cpu",
)
prepared = ExecutionWorksetPlan(
    ("cell-a", "cell-b", "cell-c"), (signature,) * 3, bucket_capacity=2
).prepare()
items = Normalized(jnp.ones(2), jnp.arange(6.0).reshape(3, 2))
evaluation = evaluate_execution_worksets_filter_vmap(
    prepared,
    lambda signature, item, key, index: item(jnp.zeros(2)),
    items,
    jax.random.key(0),
    jnp.zeros(3, dtype=jnp.uint32),
    layout=phx.LaneLayout("item", (".shift",)),
)
assert evaluation.values.shape == (3, 2)
```

::: phydrax.execution.ExecutionWorksetPlan

---

::: phydrax.execution.PreparedExecutionWorksets

---

::: phydrax.execution.evaluate_execution_worksets_serial

---

::: phydrax.execution.evaluate_execution_worksets_vmap

---

::: phydrax.execution.evaluate_execution_worksets_filter_vmap

---

::: phydrax.execution.evaluate_execution_worksets_grouped

---

::: phydrax.execution.ExecutionWorksetCheckpoint

---

::: phydrax.execution.restore_execution_workset_checkpoint

### Atomistic lane campaigns as a workset consumer

`phydrax.atomistic.PreparedAtomisticCampaign` composes these owners. It adds no
scheduler of its own.

- **Signatures.** Each lane bank's `AtomisticStageSignature` wraps a
  `PoolExecutionSignature` (`.pool`). Its topology and method IDs come from the system,
  resource-bundle, and Hamiltonian layouts, together with the derivative request, the
  method-state layout, and an optional `ExecutionGroupSpec` placement. Task identity,
  material identity, and numeric revisions stay out of the signature, so every material
  and revision of one layout shares an executable.
- **RNG.** At refill, an `ExecutionWorksetPlan` is prepared over the candidates, and
  `PreparedExecutionWorksets.semantic_keys` supplies the admission keys.
  The semantic IDs are `stage/task`, and each task's stage entry count is the counter,
  so the keys are independent of lane, bucket, round, and completion order.
- **Admission.** The pool's internal bounded-frontier allocator admits whole task groups
  and pending outputs with prefix semantics, and its refill router places admitted tasks
  into free lanes. Neither helper is a public API.
- **Observation.** `run_round(..., session=IterationSession)` emits one commit-phase
  `IterationRecord` per round. Its scope is `atomistic-campaign:<campaign_id>` with role
  `atomistic-campaign`, and its metrics are lane utilization, admissions, parked lanes,
  pending and published outputs, backpressure, shared evaluations, and capacity events.

Outputs and the continuation envelope are written as atomically published lifecycle
array archives. A fresh-process `resume` restores the last durable commit. In a lane
campaign, each lane advances a whole system or a whole path band, and no task is
partitioned across child groups. Distributed stage x domain campaigns run each stage's
domain dynamics on its own child `ExecutionGroup` through one fixed root-group stage
transport schedule; see
[stage x domain campaigns](../guides_atomistic_distributed_execution.md#stage-x-domain-campaigns).
The campaign API is documented under
[staged campaigns](atomistic.md#staged-campaigns).

## Process-local ingress and distributed checkpointing

`DistributedIndexEpochPlan` preserves one global ordering while returning
fixed-capacity process-local indices and validity masks. Padding duplicates a
valid logical ID only as storage; the mask gives it zero scientific mass.
`make_global_array_from_process_local_data` constructs global arrays without
materializing global host data.

Distributed checkpoints elect one authoritative device for replicated shards,
publish process-local artifacts transactionally, validate exact non-overlapping
coverage, and restore directly into a destination sharding. A failed process
publication cannot replace the previous visible checkpoint.
Replicated scalars and arrays with one locally addressable replica use JAX's
public local-data view rather than its globally sharded shard-view shortcut.
Canonical owner selection, replica IDs, dtypes, shapes, and checkpoint layout
identities are preserved; no global gather or device-to-device copy is introduced.

`phydrax.lifecycle.publish_process_checkpoint(..., writer_id, writer_index=None,
...)` publishes under a checkpoint writer slot: the `process-{slot}` artifact, the
artifact and shard `process_index` metadata, and the slot range
`[0, expected_process_count)` that assembly requires. The slot defaults to
`jax.process_index()` for a whole-job publication; an execution subgroup passes
its group-local rank and is validated before anything is written. Every shard
also records `source_process_index`, the physical process that held the snapshot,
and its layout identity keeps binding the physical device. The writer slot and
the physical source process are therefore distinct metadata.

Each process artifact stores its canonical shard-descriptor inventory in the
authenticated `checkpoint-shard-inventory` chunk stream. Small metadata binds
its digest, shard count, lifecycle identities and parent lineage; inventories
are not embedded in a metadata value. `ArrayArchiveLimits.max_manifest_bytes`,
`max_manifest_nesting` and `max_members` admit inventory bytes, depth and count
before decoding, without enlarging the repository's metadata-value ceiling.
Use the same sufficient limits for publication, assembly, commit, loading and
restore. Array payload bytes, layout, placement and typed-key guarantees remain
unchanged.

The repository's `maximum_metadata_bytes` separately bounds the complete
artifact manifest, including its chunk records. Size that allocation for the
writer's complete inventory, especially when one process owns several execution
groups. A larger explicit artifact-manifest allocation does not change the fixed
per-value metadata ceiling.

This is a clean wire cutover: checkpoints using the retired metadata-embedded
inventory must be republished. Process artifact IDs, global manifest IDs and
republished descendant lineage change; there is no legacy reader or historical
identity substitution.


### Numerical-relativity specialization

`phydrax.applications.numerical_relativity.NumericalRelativityDistributedPlan`
specializes three-dimensional named sharding for the exact Z4c/GRHD/GRMHD formulation
signature. `NumericalRelativityAMRDistributionPlan` specializes the canonical
Morton-contiguous block partition and FillPatch transpose routes. Neither introduces a
second execution planner.

`NumericalRelativityCheckpointPlan` uses the same local/distributed pickle-free
checkpoint owners while binding formulation, runtime, geometry, topology/epoch,
analysis, numeric revision, execution, constrained transport, and an exact typed
state template. Repository-committed shards restore through
`restore_distributed_numerical_relativity_checkpoint` to a complete
`DistributedNumericalRelativityRestart`.

`NumericalRelativityProductionPlan` binds exact scientific/deployment support, a
resolver-produced execution plan and evidence, case/runtime/checkpoint policy, and
mandatory staging/backlog plus finite artifact/output limits. Output manifests require
an acknowledged `NumericalRelativityCommittedOutputReceipt`; cancellation requires a
verified `CheckpointCommitReceipt`. These are technical production controls, not a
released profile or PNPL deployment authorization. See the
[black-hole execution guide](../guides_black_hole_execution.md).

## Host concurrency and failure

Host tasks use bounded inline, thread, process, or scheduler execution.
Cancellation is cooperative or drain-only; PHYDRAX never injects asynchronous
exceptions into JAX, MPI, FFI, external solvers, or repository transactions.
Coupled groups fail together. Independent work items retain isolated attempt
boundaries and stable logical IDs.

## Iteration observation and control

`phydrax.execution` also owns the common lifecycle for solver-controlled
iterations. Pure observers execute as fixed-shape JAX state and return
`IterationEvidence`; host `IterationSession` objects deliver already-materialized
records only at an owner's declared host boundary. No Python callback executes
inside a differentiated or rematerialized numerical loop.

```python
import jax.numpy as jnp
import phydrax as phx

problem = phx.linalg.LinearSystem(
    phx.linalg.DenseLinearOperator(jnp.diag(jnp.array([1.0, 2.0, 4.0])))
)
iteration = phx.execution.IterationPlan(
    granularity="inner-iteration",
    observers=(phx.execution.IterationTraceObserver(16),),
)
result = phx.linalg.solve(
    problem,
    jnp.ones(3),
    policy=phx.linalg.LinearSolvePolicy(
        phx.linalg.PCG(),
        materialization=phx.linalg.MaterializationPolicy(
            max_entries=1,
            max_bytes=1,
        ),
    ),
    iteration=iteration,
)
trace = result.iteration_evidence.observer_outputs[0]
```

The disabled path is `iteration=None`. It carries no observer arrays and returns
no iteration evidence. Trace capacity bounds retained progress records; initial
and terminal records are stored separately, so overflow increments
`dropped_count` without discarding terminal certification.

Host observation and host control are deliberately separate:

```python
events = []
session = phx.execution.IterationSession(
    "training-run",
    sinks=(
        phx.execution.CallableIterationSink(
            lambda event: events.append(event),
            "in-memory-events",
        ),
    ),
    control=phx.execution.CallableIterationHostControl(
        lambda event: event.sequence >= 100,
        "stop-after-event-100",
    ),
)
```

A sink must return `None`; only `IterationHostControl` may request a stop.
`IterationSessionState` persists the ordered cursor and stop request across
training, production, sampling-chunk, continuation, and decomposition restarts.
The session ID and scope ID produce deterministic event IDs.

External experiment tracking composes through the same sink boundary. For example,
`phydrax.service.WandbTrainingSink` consumes typed training records from an
application-owned W&B run; it cannot request a stop. `FunctionalSolver.solve`
accepts `session` and `session_every`, while the session cursor remains part of
checkpointed `TrainingProgress`.

Iteration granularity is capability-checked before execution:

| Owner | Exact granularity |
| --- | --- |
| native scalar/block Krylov | `inner-iteration`, `terminal` |
| direct or opaque linear providers | `terminal` |
| Newton and native scalar optimization | `attempt`, `step`, `terminal` |
| fixed-step rollout and Markov/Hamiltonian scans | `step`, `terminal` |
| Diffrax | requested saved `output`, `terminal` |
| host-batched finite search, production, continuation, decomposition | `segment` or owner-defined `step`, `terminal` |

Opaque backends never synthesize unavailable internal histories. A request for
unsupported granularity raises before numerical work.

Nested scientific events remain separate from iteration records: residuals,
vector fields, hybrid zero crossings, branch monitors, and backend residual or
Jacobian functions are model semantics rather than lifecycle callbacks.

::: phydrax.execution.IterationPlan

---

::: phydrax.execution.IterationTraceObserver

---

::: phydrax.execution.IterationMomentObserver

---

::: phydrax.execution.CallableIterationStopRule

---

::: phydrax.execution.IterationEvidence

---

::: phydrax.execution.IterationSession

---

::: phydrax.execution.CallableIterationSink

---

::: phydrax.execution.CallableIterationHostControl

`python -m tools.iteration_execution_benchmarks --smoke` compares the disabled,
constant-memory count, and bounded-trace paths with synchronized timing and
compiler memory evidence.
