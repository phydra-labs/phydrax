# Structured logging and observability

PhydraX uses Loguru to transport bounded, structured host-side events. Logging is
silent by default because PhydraX is a library: importing the package adds no sink,
removes no application handler, and collects no runtime information.

Logging is not the authoritative store for scientific state:

- Loguru records execution events.
- TensorBoard records scalar histories.
- `Diagnostic` values and result statuses record numerical validity.
- lifecycle archives and provenance records preserve reproducibility evidence.
- service `AuditRecord` values preserve the hash-chained security history.
- `HostTelemetrySnapshot` preserves explicitly collected runtime facts.

Events correlate these substrates by their existing IDs rather than duplicating their
payloads.

## Enable event emission

```python
from pathlib import Path

import phydrax as phx

phx.logging.enable()
handler_id = phx.logging.add_json_sink(Path("run.events.jsonl"))
try:
    with phx.logging.context(run_id="run-42", execution_plan_id="plan-7"):
        phx.logging.emit(
            "INFO",
            "application.run.started",
            "Application run started",
            worker_count=4,
        )
        # Invoke PhydraX operations here.
finally:
    phx.logging.remove_sink(handler_id)
    phx.logging.disable()
```

`add_text_sink()` provides a one-line human view. `add_json_sink()` emits canonical
JSON Lines suitable for local indexing. Both add handlers filtered to canonical
PhydraX events and return the exact Loguru handler ID that owns the sink. Removing
that ID never removes an application-owned handler.

Applications that already configure `loguru.logger` only need to call
`phydrax.logging.enable()`. Canonical fields are stored under
`record["extra"]["phydrax"]`.

## Correlation context

`phydrax.logging.context()` uses context-local state. Nested scopes restore their
parent values, and asynchronous tasks retain independent contexts.

Prefer existing content-addressed identities:

```python
with phx.logging.context(
    run_id=run.run_id,
    analysis_plan_id=analysis.analysis_plan_id,
    execution_plan_id=execution.execution_plan_id,
    environment_snapshot_id=snapshot.snapshot_id,
):
    ...
```

Context does not automatically cross a newly submitted thread or a spawned process.
PhydraX asynchronous publishers explicitly copy it into their worker. Applications
must bind it independently in each process.

## Event contract

Event names use lowercase dotted names: `<subsystem>.<object>.<action>`. State
transitions use consistent actions such as `started`, `completed`, `failed`,
`cancelled`, `selected`, and `committed`.

Field names use lowercase underscores. Unit-bearing values name the unit explicitly,
for example `elapsed_seconds`, `byte_count`, and `iteration_count`.

Levels have fixed meanings:

- `TRACE`: explicitly enabled high-frequency details.
- `DEBUG`: probes, preparation, cache decisions, and bounded I/O summaries.
- `INFO`: run lifecycle, sampled training progress, selected backends, and commits.
- `WARNING`: explicit fallback, degraded results, cancellation, or resource pressure.
- `ERROR`: a terminal operation failure handled at its owning boundary.
- `CRITICAL`: an integrity, audit-chain, privacy, or security invariant failure.

PhydraX does not use Loguru's custom `SUCCESS` level in its canonical event contract.

## Training

Functional solvers emit `training.step.completed` when `log_every` is positive and
logging is enabled. `log_every` defaults to zero. `log_terms=True` adds per-term and
model-loss metrics to the same bounded event.

```python
phx.logging.enable()
handler_id = phx.logging.add_text_sink("training.log")
try:
    trained = solver.solve(num_iter=100, log_every=10)
finally:
    phx.logging.remove_sink(handler_id)
```

File ownership is process-level; functional solvers no longer accept `log_path`.
TensorBoard remains independent:

```python
trained = solver.solve(
    num_iter=100,
    log_every=10,
    tensorboard_log_dir="runs/example",
    tensorboard_every=5,
)
```

When Loguru and TensorBoard are due on the same step, PhydraX materializes the shared
scalar report once and sends it to both sinks. No event is emitted from JIT, `vmap`,
`pmap`, or a signal handler.

Shared training frontends map `TrainingIterationKind` records to the same
default-silent logging events before optional `IterationSession` delivery.
Logging remains observational: only a separately configured
`IterationHostControl` can stop a training loop.

## Runtime environment snapshots

Collection is explicit:

```python
policy = phx.service.HostTelemetryPolicy(
    include_host_identity=False,
    include_jax_runtime=True,
    include_jax_devices=True,
)
snapshot = phx.service.HostTelemetryCollector(policy=policy).collect()
record = snapshot.to_record(phx.service.PrivacyClassification.INTERNAL)

with phx.logging.context(environment_snapshot_id=snapshot.snapshot_id):
    ...
```

Device discovery occurs only when `include_jax_devices=True`. Host names are
sensitive and omitted unless explicitly requested. PhydraX never captures all of
`os.environ`; installed-package state remains build provenance rather than repeated
log data.

## Privacy and failure semantics

Canonical events admit JSON scalars and bounded mappings/sequences. Unsupported
objects, arrays, and oversized values are omitted without calling `repr()` and are
listed in `omitted_fields`. Non-finite floating-point values become `null` and are
listed in `nonfinite_fields`.

Never pass model inputs, arrays, mesh contents, credentials, tokens, raw environment,
command arguments, stdout, stderr, or unrestricted exception text. Provider events
contain IDs, return codes, durations, and byte counts. PhydraX-owned sinks disable
Loguru local-variable diagnosis and backtraces.

Python warnings remain warnings. PhydraX does not replace `warnings.showwarning` or
intercept the standard root logger. Fallback events provide structured correlation
without changing warning-filter semantics.

Sink failures do not define scientific success. Audit records, diagnostics,
checkpoints, lifecycle records, and provider results remain authoritative.

## Processes and multihost execution

Use one JSONL file per independently launched process or JAX process index. A human
console sink should normally be configured only on the primary process. Loguru's
`enqueue=True` can serialize records from locally related processes; it is not a
multihost collector and must not be used to make independent ranks write one file.

Call `remove_sink()` before process shutdown so an enqueued file handler drains.
PhydraX never configures a network sink; any remote sink is application-owned and
outside the qualified local evidence boundary.

## Support bundles

Raw logs are not included automatically. Support bundles retain their explicit field
allowlist, privacy ceiling, and structural redaction. Export only selected structured
fields or reference lifecycle, diagnostic, audit, and environment snapshot IDs.
