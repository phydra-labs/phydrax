# Commercial MPM architecture

Commercial MPM does not infer support from independently available components.
Every claim is keyed by `MPMClaimTuple`, including dimension, kinematics, grid,
assignment, particle domain, transfer, schedule, material, fields/contact,
fracture, integrator, storage/backend, precision, capacity, and derivative mode.

`MPMSupportMatrix` returns one explicit outcome:

```text
SUPPORTED
REJECTED
NOT_APPLICABLE
EXPERIMENTAL
```

A rejected claim fails before compilation. A claimed `MaterialPointProblemIR`
requires the matching `MPMSupportDecision`; its IDs become part of
`CompiledMaterialPointProblem.compilation_id`.

## Intended use and release evidence

`MPMIntendedUse` declares the decision, phenomena, geometry/load/material scope,
target observables, prohibited uses, risk class, and accuracy/UQ objective.

`MPMReleaseEvidenceBundle` requires G0–G7 evidence and an independent approver.
`MPMCommercialProfile` combines the exact support matrix with a standards
traceability matrix. A release assessment is claim-specific; no generic
certification is implied.

## Transaction authority

`PreparedMPMDynamics` remains the numerical kernel. It owns one attempt and
returns candidate/accepted states. `AdaptiveMPMRolloutPlan` is the sole numerical
retry authority.

`MPMRunSupervisor` is host-side operational orchestration. It may write output,
promote an accepted checkpoint generation, recover from a durable checkpoint, or
start a separately prepared capacity/topology epoch. It never invents a second
numerical retry loop.

## Durable state

`MPMCheckpointPlan` inventories every dynamic runtime leaf by path, shape, dtype,
and SHA-256 checksum. Accepted generations are written to a temporary archive,
flushed, atomically renamed, read through a validated manifest, and then promoted
through an atomic `CURRENT` pointer. A failed attempt is never restartable state.

`MPMOutputPlan` stores accepted particle fields in HDF5, emits an XDMF temporal
view, and can write VTK particle snapshots. Output is not a checkpoint.

## Failure semantics

Commercial failures use `MPMCommercialFailure`. Unsupported configuration,
capacity, topology, material, contact, nonlinear, derivative, checkpoint, output,
and validation failures remain distinct. Every accepted commercial result must be
free of unresolved required evidence and integrity failures.
