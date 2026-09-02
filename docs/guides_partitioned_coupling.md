# Partitioned multiphysics coupling

`phydrax.solver.coupling` composes pure fixed-window subsystem maps without
introducing another mesh, nonlinear-solver, or communication stack. It is intended
for native same-process coupling when subsystem reuse is more important than a
monolithic discrete solve.

Use a specialized or monolithic method instead when participants share algebraic
unknowns, stagewise conservation must hold exactly, or a coupled Jacobian and
preconditioner are available. Existing CFD--DEM, compatible Maxwell--PIC, immersed
boundary, and other transactional physics runtimes remain distinct.

## Mathematical contract

For participant `i`, one frozen-window evaluation has the form

```text
(candidate state, outputs) = S_i(window-start state, interface inputs, args)
```

The exchanges define `H(interface)`, and implicit coupling solves the physical
interface equation

```text
R(interface) = interface - H(interface) = 0.
```

Every implicit residual evaluation restarts every participant from the same accepted
window-start state. A final simultaneous evaluation certifies the original physical
exchange residual before any candidate state is committed.

## Participants, ports, and exchanges

A `CouplingPort` declares an exact `AbstractVectorSpace`, direction, semantic ID, and
positive physical `reference_scale`. Field-valued ports additionally retain their
`DiscreteFieldSpace`. No exchange broadcasts, reshapes, casts, or selects a mapping.

Each input port has exactly one driver. Output ports may fan out. Participants and
ports use globally unique IDs.

A direct `CouplingExchange` requires identical source and target vector-space IDs.
A field exchange receives an already prepared `FieldTransfer` and may apply either
its forward operator or declared paired adjoint. `CouplingTransferRequirement`
rejects a supplied transfer that does not already certify requested constant,
conservative, positivity, adjoint, or exactness properties.

For a primal transfer `P` and paired load transfer `P*`, the intended work identity is

```text
inner_target(P u, f) = inner_source(u, P* f).
```

The coupling runtime never manufactures the reverse map independently.

## Pure participant callback

`CallableCouplingSubsystem` adapts a callback with signature

```python
advance(window, start_state, inputs, args) -> CouplingSubsystemResult
```

`inputs` and `outputs` are tuples aligned with the declared ports. The result reports
one candidate state, scalar success/status/residual/iteration/work evidence, and
optional fixed-structure auxiliary data.

Preparation uses shape evaluation to prove that the callback preserves participant
state structure and returns every declared output space. Native participants must be
JIT-capable and fixed-topology. Participants in implicit cycles must additionally
provide deterministic replay.

Random keys, adaptive-controller state, contact history, and every other path-dependent
quantity belong in participant state or runtime arguments. Because iterations reuse
the same window checkpoint, a realization advances only when the coupled window is
accepted.

## Graph preparation

`prepare_coupling`:

1. canonicalizes subsystem and exchange order by semantic ID;
2. validates exact ports, transfers, drivers, and initial exchange values;
3. finds strongly connected components;
4. topologically orders the condensation graph;
5. binds normalized cyclic-interface coordinates;
6. shape-checks all participant callbacks;
7. checks JIT, replay, differentiation, and resource capabilities;
8. emits a stable `PreparedCoupling` and `CouplingPreparationReport`.

Graph identity does not depend on declaration order. Gauss--Seidel ordering is a
separate numerical policy and therefore remains order-sensitive.

`refresh_coupling` accepts only unchanged graph identity. It replaces numeric
participant or transfer leaves and increments `numeric_version`; changed spaces,
topology, transfers, or schedules require preparation of a new plan.

## Explicit coupling

`ExplicitCouplingPolicy` applies exactly one sweep.

- `CouplingSweep("jacobi")`: every participant consumes the incoming accepted
  exchange values.
- `CouplingSweep("gauss-seidel", subsystem_order=(...))`: outgoing values become
  available immediately to later participants in the declared order.

A successful explicit step means the declared finite sweep completed. It does not
mean the interface equation converged: `successful` is true and `converged` remains
false. The result retains the nonzero interface defect.

## Implicit coupling

`ImplicitCouplingPolicy` has two execution paths.

### Fixed-point path

Pass `FixedPointIteration`, an explicit `fixed_point_sweep`, and physical per-port
`CouplingTolerance` values. Existing `AndersonAcceleration` supplies safeguarded
fixed-capacity acceleration.

The fixed-point method may use Jacobi or Gauss--Seidel iteration. Its returned iterate
is always re-evaluated with the simultaneous physical exchange residual before
acceptance. This path intentionally exposes no implicit derivative contract.

### General-root path

Pass an existing `AbstractNonlinearMethod` such as `Broyden`, `NewtonKrylov`, or
`NewtonTrustRegion`. The method solves the simultaneous normalized interface
residual. Physical certification still uses each target port's declared pairing and
physical tolerance.

No nonlinear method is selected or retried automatically.

## Scales and convergence

Nonlinear acceleration acts on coordinates flattened from each cyclic target port and
divided by its explicit `reference_scale`. This makes heterogeneous field magnitudes
intentional without claiming that canonical coordinates are an isometry for a
non-Euclidean pairing.

Physical certification is independent. For exchange `e`,

```text
physical norm = sqrt(real(inner_target(residual_e, residual_e)))
threshold = absolute + relative * reference_scale.
```

Every cyclic target port must have exactly one `CouplingTolerance`. A nonlinear method
reporting success while any physical block fails becomes
`CouplingStatus.CERTIFICATION_FAILURE`; the accepted state remains the window
checkpoint.

## Transactional results

`CouplingWindowResult` separates `candidate_state` from `accepted_state`.

- Participant failure, nonfinite output, nonlinear failure, work exhaustion, or
  physical-certification failure leaves accepted participant states, exchange values,
  time, and window index unchanged.
- Explicit success commits the finite single-sweep candidate.
- Implicit success commits only the freshly certified candidate.

`CouplingWindowDiagnostics` aligns exchange and participant arrays with static IDs and
retains physical residuals, thresholds, participant statuses, final-certification
participant work, exact participant evaluation counts, transfer applications, and
coupling iterations. Complete total-work evidence is claimed for explicit sweeps only;
implicit participant work may vary between interface trials and is not inferred from
the final evaluation.

## Fixed-window rollout

`CouplingProblem` requires an exact integer number of fixed coupling windows and
explicit initial values for every exchange. `CouplingRolloutPlan` provides final,
checkpoint, or trajectory retention and reuses `FixedStepReplayPolicy` for deterministic
reverse recomputation.

After the first failed window, later scan positions perform no participant work. The
accepted state remains fixed and retained validity is a prefix mask.

```python
problem = cpl.CouplingProblem(
    graph,
    initial_participant_states,
    initial_exchange_values,
    policy,
    t0=0.0,
    t1=10.0,
    window_size=0.1,
)
solution = cpl.solve_coupling(
    problem,
    rollout=cpl.CouplingRolloutPlan(retention="trajectory"),
)
```

## Differentiation

`CouplingDifferentiationPolicy` makes derivative ownership explicit.

| Mode | Contract |
|---|---|
| `"none"` | Primal solve only; returned accepted and candidate states are stopped. |
| `"algorithmic"` | Differentiate one finite explicit sweep and fixed-window rollout. |
| `"implicit"` | Differentiate a successful general-root implicit interface solve. |

Implicit mode uses `implicit_root_result` and requires differentiable participants,
fixed topology, deterministic replay, and differentiable field-transfer geometry. A
non-Newton primal method must supply tangent and adjoint linear policies through
`ImplicitRootDerivativePolicy`.

Algorithmic differentiation through a convergence-dependent implicit loop is not
exposed. A failed root remains failed and has no valid implicit derivative.

## Fixed-capacity higher-order waveform coupling

A waveform port declares `CouplingWaveformPlan`: normalized nodes, sample
capacity, polynomial degree zero through three, metric order, and optional finite
adaptation reservoir. `CouplingWaveformGrid` keeps a sorted active prefix with
exact endpoints zero/one. Values always retain capacity rows; inactive rows are
canonical zero and contribute no callback work, residual coordinate, or norm.

`BarycentricCouplingTemporalTransfer` selects deterministic local nonuniform
stencils and never extrapolates. The physical waveform norm integrates the
piecewise polynomial pairing with an exact declared Gauss order; degree-one
all-active data recovers the fixed linear behavior. Adaptive defects activate a
deterministic candidate and restart every participant from the unchanged window
checkpoint. Exhausted capacity returns `CouplingWaveformCapacityRequest`; growth
occurs only in a host-prepared epoch.

`AdaptiveCouplingWindowPolicy` accepts only physically certified windows with
reliable participant `CouplingWindowErrorEstimate` ratios at most one. Its bounded
PI controller retries only declared statuses, saturates at explicit minimum and
maximum sizes, and reaches final time exactly through the canonical fixed-segment
runner. Every rejected attempt restarts the same checkpoint.

`PreparedCouplingEpoch` and `CouplingEpochTransitionPlan` apply topology or
source-owned remesh requests only after an accepted window. Retained values need
an explicit transfer; added and removed participants need an initializer or
finalizer. Missing routes, coverage, or conservation retain the complete old
graph/state epoch. Frozen event replay is required across node, window, and
topology decisions.

## Statuses

| Status | Meaning |
|---|---|
| `SUCCESS` | Explicit sweep completed or implicit root physically certified. |
| `PARTICIPANT_FAILURE` | At least one participant rejected its window evaluation. |
| `NONFINITE_EVALUATION` | Participant, exchange, or residual data became nonfinite. |
| `NONLINEAR_FAILURE` | The selected implicit nonlinear method failed. |
| `WORK_EXHAUSTED` | Nonlinear step, evaluation, or inner-linear budget was exhausted. |
| `CERTIFICATION_FAILURE` | Final physical interface residual exceeded a port tolerance. |

There is no success alias for an exhausted or uncertified implicit solve.

## Unsupported boundaries

The native substrate does not provide process communication, MPI/socket routing,
external mutable participants, XML configuration, automatic mapping selection,
hidden iterate clipping, or fallback solvers. Topology changes are accepted-boundary
host transitions over explicitly prepared graphs and source-owned transfers, never
an in-trace graph mutation. An external participant backend remains a separate
host-only concern and is not a native JIT or differentiation path.

See `examples/partitioned_coupled_oscillators.py` for an implicit differentiable
example and `tools/partitioned_coupling_qualification.py` for executed convergence,
acceleration, residual, and derivative evidence.
