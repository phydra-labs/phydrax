# Execution worksets

`phydrax.execution` groups exact homogeneous execution signatures into deterministic,
fixed-capacity worksets. It owns canonical ordering, reversible gather/scatter,
serial/vectorized equivalence, semantic restartable RNG keys, finite/coverage evidence,
and content-addressed checkpoints. It does not expose a distributed path when no real
multi-device qualification exists.

::: phydrax.execution.ExecutionWorksetPlan

---

::: phydrax.execution.PreparedExecutionWorksets

---

::: phydrax.execution.evaluate_execution_worksets_serial

---

::: phydrax.execution.evaluate_execution_worksets_vmap

---

::: phydrax.execution.ExecutionWorksetCheckpoint

---

::: phydrax.execution.restore_execution_workset_checkpoint

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
