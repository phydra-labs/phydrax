# Execution substrate

`phydrax.execution` separates process bootstrap, resource ownership, logical
decomposition, physical placement, and host lifecycle. Numerical values remain
ordinary JAX arrays. The default path remains local; distributed execution is
enabled by an explicit runtime or execution group.

## Bootstrap before device initialization

Launcher rank settings must exist before `import phydrax`. The package consumes
them before loading any device-owning module, then preserves its dependency-
ordered public imports:

```python
# PHYDRAX_COORDINATOR_ADDRESS, PHYDRAX_NUM_PROCESSES, and
# PHYDRAX_PROCESS_ID were set by the launcher.
import phydrax as phx

runtime_info = phx.execution.runtime_info()
runtime = phx.execution.ExecutionRuntime.current()
```

The bootstrap accepts explicit `PHYDRAX_COORDINATOR_ADDRESS`,
`PHYDRAX_NUM_PROCESSES`, `PHYDRAX_PROCESS_ID`, and
`PHYDRAX_LOCAL_DEVICE_IDS` settings. It also delegates to JAX launcher
detection under Slurm, Open MPI, PMI, and PMIx. `PHYDRAX_CPU_COLLECTIVES`
selects `gloo` or `mpi` before CPU backend initialization. An attempted
distributed initialization after JAX has initialized fails rather than
silently producing an incomplete process set.

For a local process set, `LocalProcessLaunchPlan` and
`launch_local_processes` start one shell-free argument vector per rank and
inject the same explicit rendezvous identity.

::: phydrax.execution.RuntimeBootstrap

---

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

## Native global arrays and rank-local providers

Native JAX global arrays are the default numerical route. Use
`shard_array_axis` for independent arrays, owner-specific placement policies
for structured values, and `shard_map`-based owners for manual halo or
redistribution algorithms. A JAX reduction over a global array already has
global semantics; explicit collectives belong inside rank-local mapped
kernels.

`phydrax.backends.JaxCollectiveProvider` supplies named-axis operations inside
mapped regions. `Mpi4JaxCollectiveProvider` is an optional rank-local route with
caller-owned communicator lifetime and operation-specific transformation
support. Availability never implies qualification.

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

::: phydrax.execution.ExecutionWorksetPlan

---

::: phydrax.execution.PreparedExecutionWorksets

---

::: phydrax.execution.evaluate_execution_worksets_serial

---

::: phydrax.execution.evaluate_execution_worksets_vmap

---

::: phydrax.execution.evaluate_execution_worksets_grouped

---

::: phydrax.execution.ExecutionWorksetCheckpoint

---

::: phydrax.execution.restore_execution_workset_checkpoint

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
