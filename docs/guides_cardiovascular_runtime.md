# Cardiovascular capacity, execution, restart, and replay

The cardiovascular runtime is an orchestration boundary, not a second solver or
archive stack. It binds cardiovascular case and topology identities to the
existing PhydraX lifecycle archive, execution-pool, finite-element partition,
and checkpointed replay substrates. Numerical kernels remain owned by their
native packages.

## Execution invariants

Every run follows the same transaction:

1. Create a `CardiovascularCapacityManifest` with hard host and compiled-shape
   limits.
2. Create one concrete execution route and a
   `CardiovascularExecutionManifest`.
3. Prepare a cohort, distributed contract, checkpoint codec, or multirate
   scheduler.
4. Produce a fixed-shape candidate and evidence record.
5. Cross the explicit commit boundary. A failed candidate returns its original
   rollback state and cannot be checkpointed as committed.

Capacity is an admission decision, not a truncation policy. If any requested
resource exceeds its limit, `admit_cardiovascular_capacity` reports every
exceeded dimension and rejects the complete request. Preparation and
checkpoint publication use the same fail-closed rule.

```python
import jax

from phydrax.applications.cardiovascular import (
    CardiovascularCapacityManifest,
    CardiovascularExecutionManifest,
    CardiovascularSerialExecution,
)

capacity = CardiovascularCapacityManifest(
    maximum_cohort_cases=32,
    maximum_state_values=100_000,
    maximum_checkpoint_arrays=16,
    maximum_checkpoint_bytes=8_000_000,
    maximum_macro_steps=1_000,
    maximum_scheduled_steps=8_000,
    maximum_events=256,
    maximum_partitions=8,
)
execution = CardiovascularExecutionManifest(
    case_manifest_id=case.manifest_id,
    analysis_plan_id="analysis:whole-heart:001",
    numeric_revision_id="revision:calibrated:042",
    topology_id="topology:fixed:017",
    solver_policy_id="solver:coupled-newton:004",
    precision_policy_id="precision:f64:001",
    backend=jax.default_backend(),
    capacity=capacity,
    route=CardiovascularSerialExecution(0),
)
```

The execution identity includes the complete capacity and concrete route.
Changing either produces a different `manifest_id`. Runtime records therefore
cannot silently move between a serial run, cohort pool, local distributed
reference, or collective request.

## Concrete execution routes

Routes are distinct Python types rather than string modes:

- `CardiovascularSerialExecution` selects one visible JAX device.
- `CardiovascularCohortExecution` selects a fixed number of single-device pool
  lanes.
- `CardiovascularDistributedReferenceExecution` executes owned-local FEM phase
  semantics on one host and compares them with the serial definition.
- `CardiovascularDistributedCollectiveExecution` requests an exact JAX
  process/device mesh, named collective axis, and partition count.

`observe_single_device_runtime` records the observed platform, device kind,
device ID, process index, and visible backend-device count. An unavailable
backend or out-of-range device is evidence of ineligibility, not a fallback to
another device.

### Distributed scope

The collective route is a real generic finite-element execution route. It does
not call the cell-sum reference proof. Preparation binds the requested process
count, each selected `(process_index, device_id)` pair, the
`FiniteElementDistributedPhasePlan`, its partition and workset identities, the
mesh axis, and the owned-array transport identity. An eligible one-process mesh
is built from exactly the selected backend devices.

`execute_cardiovascular_distributed_collective` accepts the prepared contract,
the field's `FiniteElementDofMap`, an `AbstractLinearOperator` assembled by the
generic FEM runtime, and a right-hand side. DOF ownership is derived from the
existing cell partition and cell-to-DOF map; shared DOFs have one deterministic
owner and appear as halos on dependent partitions. Values are packed once into
partition-local padded arrays and placed with `NamedSharding` under
`PartitionSpec(axis_name, ...)`. The `shard_map` input specification is that
owned partitioning, never `PartitionSpec()` replication of the complete input.

Inside `shard_map`, each device scatters only its owned values. Named-axis
`psum` reconstructs the global/halo view. `DistributedFiniteElementOperator`
and `JaxCollectiveBackend` apply the actual generic FEM operator to exactly-once
owned contributions; the same route applies its algebraic transpose. A
prepared generic GMRES solve then uses that distributed operator. A one-device
mesh executes this identical owned-shard, reconstruction, operator, transpose,
and solver path, so single-device CI does not qualify a surrogate algorithm.

The returned `CardiovascularDistributedCollectiveEvidence` reports forward,
transpose, halo-reconstruction, Krylov residual, and serial-solver equivalence.
Its `CardiovascularDistributedSolverState` retains only owned solution/RHS
shards and binds the exact execution manifest, distributed contract, FEM phase,
partition, workset, DOF map, finite-element operator, distributed operator,
declared solver policy, prepared solver plan, device mesh, and transport IDs.
Evidence therefore names the precise mesh/operator/partition/transport route
that ran.

If a requested local device mesh is absent, preparation records
`insufficient-local-device-mesh`. A process-spanning request records requested
and observed process counts and fails closed as
`insufficient-process-device-mesh` or
`process-spanning-owned-array-mesh-unavailable`; it never relabels a local
reduction as a multi-host simulation. `require_cardiovascular_distributed_transport`
and collective execution raise `CARDIOVASCULAR_DISTRIBUTED_INELIGIBLE`.
Qualification and benchmark reports expose a support tuple containing the
eligibility bit, exact reason, requested count, and observed count. Actual
multi-device performance remains blocked in that tuple when the required
hardware is not visible.

`prepare_cardiovascular_distributed_execution` also supports the distinct
`CardiovascularDistributedReferenceExecution`. That route executes owned-local
cell contributions for serial semantic comparison and may invoke scheduled
checkpoint/recomputation through `execute_cardiovascular_distributed_replay`.
It does not claim a device transport or collective solver.

## Deterministic cohort pools

`prepare_cardiovascular_cohort` canonicalizes stable case IDs and creates the
native `PoolExecutionSignature`. `execute_cardiovascular_cohort` derives each
random key from the semantic case index with `semantic_task_keys`; lane
placement and refill wave do not affect it. Completed lanes are assigned with
`refill_completed_tasks`.

```python
import jax

from phydrax.applications.cardiovascular import (
    CardiovascularCohortCaseCandidate,
    CardiovascularCohortExecution,
    execute_cardiovascular_cohort,
    prepare_cardiovascular_cohort,
)

cohort_execution = CardiovascularExecutionManifest(
    # same identity fields and capacity as above
    case_manifest_id=execution.case_manifest_id,
    analysis_plan_id=execution.analysis_plan_id,
    numeric_revision_id=execution.numeric_revision_id,
    topology_id=execution.topology_id,
    solver_policy_id=execution.solver_policy_id,
    precision_policy_id=execution.precision_policy_id,
    backend=execution.backend,
    capacity=execution.capacity,
    route=CardiovascularCohortExecution(8),
)
prepared = prepare_cardiovascular_cohort(
    cohort_execution, ("case-c", "case-a", "case-b")
)


def run_case(case_id, semantic_key):
    value = jax.random.normal(semantic_key)
    return CardiovascularCohortCaseCandidate((case_id, value))


result = execute_cardiovascular_cohort(
    prepared, jax.random.key(2026), run_case
)
```

Results are returned in canonical case-ID order. If any case candidate is
rejected, the cohort result is uncommitted and contains no partial value tuple.
The task acceptance mask remains available as evidence.

## Lifecycle checkpoint codec

`CardiovascularLifecycleCheckpointCodec` uses `LifecycleArchive` directly. It
does not invent another file format. Each named state array becomes a
checksum- and byte-count-bound `CheckpointShard`; the lifecycle
`CheckpointManifest` binds the cardiovascular analysis plan, numeric revision,
and exact execution manifest.

```python
from phydrax.applications.cardiovascular import (
    CardiovascularLifecycleCheckpointCodec,
)

codec = CardiovascularLifecycleCheckpointCodec(execution)
record = codec.write(
    "checkpoints/accepted-0042.phx",
    {
        "state/voltage_mV": voltage,
        "state/pressure_kPa": pressure,
        "runtime/accepted_step": accepted_step,
    },
    checkpoint_id="checkpoint:0042",
    parent_checkpoint_id="checkpoint:0041",
    committed=True,
    layout_ids={
        "state/voltage_mV": ("layout:myocardial-nodes",),
        "state/pressure_kPa": ("layout:cavity-pressure",),
    },
)
restored = codec.read("checkpoints/accepted-0042.phx")
```

The underlying archive writes atomically and validates its canonical ZIP
structure, manifest identity, payload inventory, SHA-256 checksums, shapes,
dtypes, and byte counts. Restart additionally checks the cardiovascular
analysis, numerical revision, and execution identities. Arrays are finite,
numeric or boolean, and read-only after opening. An uncommitted state or a
payload above the execution capacity is refused before a destination is
published.

Distributed solver restart uses
`write_cardiovascular_distributed_solver_checkpoint` and
`read_cardiovascular_distributed_solver_checkpoint`. The checkpoint contains
the owned solution/RHS shards, ownership routes, validity mask, solve count,
iteration count, and successful status. Every `CheckpointShard.layout_ids`
entry carries the full contract/operator/solver/mesh/transport binding listed
above. Restore requires those layout IDs and ownership arrays to match exactly,
then places the arrays back on the same `NamedSharding`. Restart feeds those
owned shards directly to the same `shard_map` solve path. A failed solver state
is passed to the lifecycle codec as uncommitted, so publication is refused
before opening the destination and the last accepted checkpoint remains intact.

Do not catch a corruption error and continue from partially decoded state. Open
the last complete checkpoint instead.

## Event-split multirate scheduling

`CardiovascularMultiratePlan` declares stable subsystem IDs, the number of local
substeps per macro step, and the macro step in kernel milliseconds. Preparation
precomputes a fixed owner/time schedule for the complete execution capacity.
Ties between subsystem transitions are ordered by stable subsystem ID.

Each event has:

- a stable source ID;
- a crossing direction (`-1`, `0`, or `1`);
- an integer priority used before source ID for simultaneous events;
- an optional terminal flag; and
- an optional guard-unit and minimum absolute slope-per-ms saltation policy.

```python
import jax.numpy as jnp

from phydrax.applications.cardiovascular import (
    CardiovascularEventSpec,
    CardiovascularMultiratePlan,
    CardiovascularSaltationPolicy,
    CardiovascularStepCandidate,
    commit_cardiovascular_schedule,
    prepare_cardiovascular_scheduler,
    run_cardiovascular_schedule,
)

plan = CardiovascularMultiratePlan(
    ("electrophysiology", "circulation"),
    (8, 2),
    1.0,  # ms
    events=(
        CardiovascularEventSpec(
            "aortic-valve-open",
            direction=1,
            priority=10,
            saltation_policy=CardiovascularSaltationPolicy("kPa", 1.0e-3),
        ),
        CardiovascularEventSpec(
            "aortic-valve-close",
            direction=-1,
            priority=20,
            saltation_policy=CardiovascularSaltationPolicy("kPa", 1.0e-3),
        ),
    ),
    localization_iterations=48,
    localization_tolerance_ms=1.0e-9,
)
prepared = prepare_cardiovascular_scheduler(execution, plan)


def advance(state, subsystem_id, start_ms, end_ms):
    proposed = native_subsystem_step(state, subsystem_id, start_ms, end_ms)
    return CardiovascularStepCandidate(proposed)


def event_values(state, time_ms):
    return jnp.asarray((
        state.aortic_pressure - state.ventricular_pressure,
        state.ventricular_pressure - state.aortic_pressure,
    ))


def reset(state, source_id, time_ms):
    proposed = native_valve_reset(state, source_id, time_ms)
    return CardiovascularStepCandidate(proposed)


candidate = run_cardiovascular_schedule(
    prepared, initial_state, 100, advance, event_values, reset
)
commit = commit_cardiovascular_schedule(candidate)
```

Callbacks return explicit candidates. A rejected step, rejected reset, nonfinite
guard, localization failure, macro-step overflow, state-value overflow, or event
capacity overflow marks the complete schedule candidate unsuccessful. Commit
then returns the original initial state. No partial transition crosses the
boundary.

Event localization replays the owning subsystem transition from the pre-step
state using deterministic bisection. Events localized within the declared time
tolerance are reset in `(priority, source_id)` order. Evidence uses fixed-size
arrays for scheduled owners, active steps, event source indices, event times,
guard values, and saltation eligibility.

Saltation eligibility is evidence, not an automatically applied derivative. A
`CardiovascularSaltationPolicy` declares the guard unit and a positive minimum
absolute guard slope per millisecond. Eligibility requires a finite post-reset
guard and an observed slope above that dimensionally consistent threshold.
Downstream sensitivity code must still supply the guard and reset derivatives
on the accepted fixed event route.

`replay_cardiovascular_schedule` repeats the same prepared schedule and compares
active routes, source indices, localized times, terminal status, and committed
state. Replay mismatch is fail-closed evidence.

## Sanitized diagnostics and qualification

`cardiovascular_runtime_diagnostic` accepts only a finite runtime status,
phase, stable run ID, and stable entity IDs. Messages and remediation text come
from a fixed inventory. Raw exception strings, array payloads, file contents,
patient labels, and injected failure details never enter a diagnostic.

Run the qualification campaign from the repository root:

```text
python tools/cardiovascular_runtime_qualification.py
```

It verifies:

- exact serial checkpoint restart and lineage;
- corruption detection and atomic uncommitted-checkpoint refusal;
- semantic cohort determinism across lane counts and input ordering;
- fixed-capacity event localization, simultaneous-event ordering, saltation
  records, and exact replay;
- injected numerical failure rollback with sanitized diagnostics;
- distributed owned-local reference equality;
- real single-device `shard_map` collective, halo, and transpose semantics; and
- qualified multi-device execution when hardware exists, otherwise explicit
  support-tuple blocking without fallback.

The performance harness is:

```text
python benchmarks/cardiovascular_runtime.py --output runtime.json
```

It reports the captured runtime environment, cohort cases per second,
multirate scheduled steps per second, replay time, lifecycle checkpoint
read/write throughput, and synchronized one-device collective execution with
serial, halo, and transpose residuals. Benchmark results are evidence for a
specific capacity, software environment, and device; they are not portable
performance promises.
